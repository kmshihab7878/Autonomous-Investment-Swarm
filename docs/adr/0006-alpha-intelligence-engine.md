# ADR-0006: Alpha Intelligence Engine

## Status

Accepted

## Date

2026-03-22

## Context

AIS strategy agents (momentum, funding rate) generate signals from price action
and market microstructure. These are systematic strategies with well-understood
edges but limited alpha diversity. A complementary source of alpha is behavioral:
identifying traders who consistently outperform and systematically following
their positions. This is the digital equivalent of "smart money tracking" used
by institutional prime brokers.

Exchanges expose trade data through their APIs, and some provide leaderboards
ranking traders by P&L. This data is noisy -- most leaderboard traders are
profitable by luck, not skill. The system needs a pipeline to: (1) scan for
notable trade activity across connected exchanges, (2) build statistical
profiles that distinguish skilled traders from lucky ones, (3) classify trading
strategies to understand *how* a trader generates returns (and whether their
edge is durable), and (4) generate follow signals that feed into the existing
AIS arbitration and risk pipeline.

The pipeline must operate autonomously within the 60-second trading cycle,
persist its state across restarts, and produce signals that are compatible with
the existing Signal model consumed by the coordinator.

## Decision

The Alpha Intelligence Engine is a four-stage pipeline with a dedicated
persistence layer:

### Stage 1: Scanner (`src/aiswarm/intelligence/scanner.py`)

`TradeScanner` monitors connected exchanges for notable trade activity:

- Polls each exchange provider's `get_my_trades()` for recent trades across
  configured symbols.
- Identifies two categories of notable activity:
  - **Whale trades**: notional value exceeding a configurable threshold
    (default: $50,000 USD).
  - **Volume spikes**: notional value exceeding 3x the rolling average for that
    symbol (computed from the last 100 observed trades).
- Deduplicates using a `_seen_trade_ids` set to avoid reprocessing.
- Outputs `TradeActivity` observations with fields: activity_id, trader_id
  (formatted as `{exchange}:{order_id}`), exchange, symbol, side, quantity,
  price, notional, timestamp, source, optional P&L, optional holding time.

### Stage 2: Profiler (`src/aiswarm/intelligence/profiler.py`)

`TraderProfiler` builds statistical profiles from accumulated trade data:

- Queries the AlphaStore for up to 5,000 historical activities per trader.
- Computes performance metrics: win rate, average return, total return,
  annualized Sharpe ratio, Sortino ratio, maximum drawdown, profit factor
  (gross profit / gross loss).
- Computes behavioral metrics: average holding time, trade frequency (trades
  per day), preferred symbols (top 5 by frequency), preferred side
  (LONG/SHORT/BOTH based on buy/sell ratio), average and maximum position size.
- Classifies the trader into a tier using a composite scoring system:
  - Win rate contribution: up to 30 points.
  - Sharpe contribution: up to 30 points.
  - Trade count contribution: up to 20 points (logarithmic).
  - Drawdown penalty: up to -20 points.
  - **ELITE** (>= 60): sustained edge, high Sharpe.
  - **STRONG** (>= 45): consistently profitable.
  - **NOTABLE** (>= 30): above average.
  - **AVERAGE** (>= 15): median performers.
  - **WEAK** (< 15): below average or insufficient data (< 10 trades).
- Computes a consistency score by splitting trades into 5 chunks, computing
  per-chunk win rates, and measuring variance. Low variance = high consistency.
- Persists the profile via `AlphaStore.upsert_profile()`.

### Stage 3: Classifier (`src/aiswarm/intelligence/strategy_classifier.py`)

`StrategyClassifier` reverse-engineers a trader's strategy from their trade
patterns:

- Requires at least 10 trades with P&L data for meaningful classification.
- Primary style classification based on holding time and return distribution:
  - `SCALPER`: average holding time < 15 minutes.
  - `CONTRARIAN`: win rate < 40% but positive average return with
    winner/loser ratio > 3x.
  - `MEAN_REVERSION`: win rate > 65% with low return standard deviation.
  - `BREAKOUT`: win rate < 50% with winner/loser ratio > 2x.
  - `TREND_FOLLOWING`: average holding > 7 days.
  - `SWING`: average holding > 8 hours but < 7 days.
  - `MOMENTUM`: default for intraday traders not matching other patterns.
- Extracts a `StrategyFingerprint` with entry patterns (timing, distance from
  MA), exit patterns (winner/loser hold times, stop loss detection, take profit
  levels), sizing patterns (scaling in/out detection), and market condition
  preferences (trending vs ranging).
- Confidence scales linearly with sample size, capping at 1.0 at 100 trades.
- Persists the fingerprint via `AlphaStore.save_fingerprint()`.

### Stage 4: Alpha Follower Agent (`src/aiswarm/intelligence/agents/alpha_follower.py`)

`AlphaFollowerAgent` extends the standard `Agent` ABC and integrates with the
existing AIS arbitration pipeline:

- Each cycle, queries the AlphaStore for recent activity from traders at or
  above `NOTABLE` tier (configurable via `min_tier`).
- Filters for activity within the last hour (configurable via
  `max_activity_age`).
- Computes signal confidence from three factors:
  - Tier-based base confidence: ELITE=0.85, STRONG=0.70, NOTABLE=0.55,
    AVERAGE=0.40, WEAK=0.25.
  - Consistency score from the trader's profile.
  - Recency decay of the observed activity.
- Emits the strongest trader signal as a standard AIS `Signal` that flows
  through the coordinator's arbitration, risk validation, and execution
  pipeline like any other agent's signal.

### Persistence: AlphaStore (`src/aiswarm/intelligence/alpha_store.py`)

`AlphaStore` is a dedicated SQLite database (default: `data/ais_alpha.db`,
separate from the EventStore) with four tables:

- `trader_profiles` -- keyed by `trader_id`, stores full profile JSON with
  `ON CONFLICT ... DO UPDATE` upsert semantics.
- `trade_activities` -- append-only with `INSERT OR IGNORE` on `activity_id`
  (idempotent ingestion). Indexed on `trader_id`, `symbol`, `timestamp`.
- `strategy_fingerprints` -- append-only (historical fingerprint versions
  retained). Indexed on `trader_id` and `style`.
- `leaderboard_snapshots` -- append-only snapshots for rank tracking over time.

The store follows the same connection management pattern as EventStore:
context-managed connections with commit/rollback, `Row` factory for dict-like
access.

### Domain Models (`src/aiswarm/intelligence/models.py`)

All models are frozen Pydantic v2 `BaseModel` instances:

- `TradeActivity` -- single observed trade.
- `TraderProfile` -- full statistical profile.
- `StrategyFingerprint` -- extracted strategy characteristics.
- `LeaderboardEntry` -- exchange leaderboard snapshot.
- `TraderSignal` -- internal signal before conversion to AIS Signal.
- Supporting enums: `TradingStyle` (7 styles + UNKNOWN), `TraderTier` (5 tiers),
  `ActivitySource` (4 sources: LEADERBOARD, TRADE_FEED, ON_CHAIN, WHALE_ALERT).

## Consequences

### Positive

- **Autonomous alpha generation**: The pipeline runs within the existing 60-second
  cycle without manual intervention. New traders are automatically discovered,
  profiled, and followed.
- **Statistical rigor**: The tier classification uses multiple metrics (win rate,
  Sharpe, drawdown, trade count) rather than relying on any single measure. The
  consistency score explicitly penalizes traders whose performance is concentrated
  in a few lucky trades.
- **Pipeline composability**: Each stage (scan, profile, classify, follow) is an
  independent class with a single responsibility. Stages can be run independently
  for testing or replaced without affecting others.
- **Native AIS integration**: The AlphaFollowerAgent emits standard AIS Signals,
  meaning all existing risk controls, mandate governance, and execution modes
  apply. There is no separate execution path for follow trades.
- **Persistent learning**: Profiles and fingerprints are persisted in SQLite.
  Across restarts, the system retains its knowledge of trader behavior rather
  than starting from scratch.
- **Separate database**: Using a dedicated `ais_alpha.db` prevents intelligence
  data growth from affecting EventStore query performance or backup size.

### Negative

- **Data latency**: The scanner relies on `get_my_trades()` which returns
  already-executed trades. By the time a trade is observed, profiled, and
  converted to a signal, the opportunity may have passed. The 1-hour activity
  window mitigates stale signals but does not eliminate latency.
- **Survivorship bias risk**: The leaderboard and trade feed data sources
  naturally surface successful traders. Traders who lost money and stopped
  trading are invisible, which can bias the profiler toward overfitting to
  recent winners.
- **Classification accuracy**: The strategy classifier uses heuristic rules
  (holding time thresholds, win rate cutoffs) rather than machine learning.
  Edge cases will be misclassified. The confidence score partially accounts
  for this, but operators should not treat style labels as ground truth.
- **SQLite scaling**: The alpha database will grow as more traders are tracked.
  A year of active scanning could produce millions of activity rows. SQLite
  handles this but query performance may degrade without periodic archival.
- **Single-exchange trader identity**: Traders are identified by
  `{exchange}:{order_id}`, which means the same person trading on two exchanges
  appears as two separate traders. Cross-exchange identity resolution is not
  implemented.

### Neutral

- The Alpha Intelligence Engine is decoupled from the backtesting system.
  Backtests use historical price data, not trader behavior data. Combining
  them (e.g., backtesting a follow strategy against historical leaderboard
  data) would require a separate data pipeline.

## Alternatives Considered

### Copy Trading via Exchange APIs

Use exchange-native copy trading features (e.g., Bybit Copy Trading, Binance
Copy Trading). Rejected because: (1) locks the system into a single exchange's
copy trading infrastructure, (2) no control over risk management -- the
exchange's copy parameters may not match AIS risk budgets, (3) no statistical
profiling -- the exchange decides who is "top" based on its own opaque metrics,
(4) cannot integrate with AIS mandate governance.

### Machine Learning Trader Classification

Use a trained ML model (e.g., gradient boosting, neural network) to classify
trading strategies instead of heuristic rules. Rejected for initial
implementation because: (1) requires labeled training data that does not yet
exist, (2) ML model serving adds operational complexity (model versioning,
inference latency), (3) heuristic rules are interpretable and auditable --
operators can understand why a trader was classified as SCALPER, (4) the
heuristic approach is sufficient to bootstrap the pipeline and generate training
data for a future ML classifier.

### Real-time WebSocket Trade Feeds

Subscribe to exchange WebSocket trade streams for lower-latency activity
detection instead of polling `get_my_trades()`. Rejected for initial
implementation because: (1) WebSocket management adds significant complexity
(reconnection, backpressure, message ordering), (2) the 60-second trading
cycle means sub-second latency is not materially useful, (3) not all target
exchanges expose public trade streams via MCP, (4) polling is simpler to
implement, test, and debug.

### Centralized Intelligence Service

Deploy the Alpha Intelligence pipeline as a separate microservice with its own
API. Rejected because: (1) adds network latency between the trading loop and
the intelligence pipeline, (2) requires additional infrastructure (service
discovery, health checks, deployment), (3) the pipeline is tightly coupled to
the AlphaFollowerAgent which must run in-process with other agents, (4) a
single-process architecture is a stated design constraint (ADR-0003).

## References

- `src/aiswarm/intelligence/scanner.py` -- `TradeScanner`
- `src/aiswarm/intelligence/profiler.py` -- `TraderProfiler`
- `src/aiswarm/intelligence/strategy_classifier.py` -- `StrategyClassifier`
- `src/aiswarm/intelligence/agents/alpha_follower.py` -- `AlphaFollowerAgent`
- `src/aiswarm/intelligence/alpha_store.py` -- `AlphaStore`
- `src/aiswarm/intelligence/models.py` -- all domain models and enums
- ADR-0002: Exchange Provider Abstraction (scanner dependency)
- ADR-0004: Mandate Governance (follower agent mandate requirement)
