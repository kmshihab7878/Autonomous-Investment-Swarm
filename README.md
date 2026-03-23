<p align="center">
  <img src="docs/assets/ais-banner.svg" alt="AIS — Autonomous Investment Swarm" width="600">
</p>

<h1 align="center">Autonomous Investment Swarm</h1>

<p align="center">
  <strong>Risk-gated autonomous trading with multi-agent orchestration</strong><br>
  <em>Every order requires cryptographic risk approval. The system fails closed, not open.</em>
</p>

<p align="center">
  <a href="https://github.com/kmshihab7878/Autonomous-Investment-Swarm/actions/workflows/ci.yml"><img src="https://github.com/kmshihab7878/Autonomous-Investment-Swarm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://kmshihab7878.github.io/Autonomous-Investment-Swarm/"><img src="https://img.shields.io/badge/docs-mkdocs%20material-blue" alt="Docs"></a>
  <a href="https://codecov.io/gh/kmshihab7878/Autonomous-Investment-Swarm"><img src="https://img.shields.io/badge/coverage-89%25-brightgreen" alt="Coverage: 89%"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="Python 3.10 | 3.11 | 3.12"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000" alt="Ruff"></a>
  <a href="http://mypy-lang.org/"><img src="https://img.shields.io/badge/typing-mypy%20strict-blue" alt="mypy strict"></a>
  <a href="https://github.com/kmshihab7878/Autonomous-Investment-Swarm/discussions"><img src="https://img.shields.io/badge/community-discussions-purple" alt="Discussions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

<p align="center">
  <a href="https://kmshihab7878.github.io/Autonomous-Investment-Swarm/">Documentation</a> &middot;
  <a href="https://kmshihab7878.github.io/Autonomous-Investment-Swarm/getting-started/quickstart/">Quick Start</a> &middot;
  <a href="https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/overview/">Architecture</a> &middot;
  <a href="https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/api/">API Reference</a> &middot;
  <a href="ROADMAP.md">Roadmap</a> &middot;
  <a href="https://github.com/kmshihab7878/Autonomous-Investment-Swarm/discussions">Discussions</a>
</p>

---

> **Warning**: This software is experimental and intended for research and educational purposes. Trading involves substantial risk of loss. Never deploy with funds you cannot afford to lose.

## Why AIS?

Most trading bots execute a single strategy with basic stop-losses. AIS is an **autonomous investment operating system** — a governed, multi-agent pipeline where every trade passes through cryptographic risk validation before execution.

<table>
<tr>
<td width="50%">

### Risk-Gated Execution
Every order requires an HMAC-signed approval token from the risk engine. No token, no trade. The system **fails closed**, not open. Token signing supports zero-downtime key rotation.

### Multi-Agent Orchestration
Strategy agents compete to generate signals. Weighted arbitration selects the best signal by confidence, expected return, and liquidity — preventing conflicting positions.

### Mandate Governance
Strategies operate within explicit mandates that cap allocation, restrict instruments, and enforce position limits. Mandates are validated before every trading cycle.

</td>
<td width="50%">

### Three Execution Modes
Paper (simulated), Shadow (read-only), Live (gated). **Same pipeline in all modes** — what you test is what you deploy.

### Multi-Exchange
Unified abstraction across Aster DEX, Binance, Coinbase, Bybit, and Interactive Brokers with config-driven symbol routing via `ExchangeRegistry` and `SymbolRouter`.

### Alpha Intelligence Engine
Scans trades across all exchanges, profiles top performers, reverse-engineers their strategies, and generates signals from their behavior. Automated edge discovery.

### Full Observability
Prometheus metrics, Grafana dashboards, Alertmanager alerts, structured JSON logging, position reconciliation, and append-only SQLite event store for audit trail.

</td>
</tr>
</table>

## At a Glance

| Metric | Value |
|--------|-------|
| Source files | 147 Python modules |
| Lines of code | 18,278 |
| Test suite | 1,273 tests (unit + integration + property-based + benchmarks) |
| Coverage | 89% |
| Strategy agents | 10 built-in + plugin system for custom strategies |
| Exchanges | 5 (Aster, Binance, Coinbase, Bybit, IB) |
| Doc pages | 42 (MkDocs Material + 6 ADRs) |
| CI matrix | Python 3.10, 3.11, 3.12 |
| Type safety | mypy strict + Pydantic v2 frozen models |
| Dependencies | Locked (`uv.lock`) for reproducible builds |
| Deployment | Docker Compose + Kubernetes Helm chart |
| License | Apache 2.0 |

## Architecture

```mermaid
graph TD
    subgraph Data Layer
        D[Market Data Providers]
    end

    subgraph Intelligence Layer
        MI[Market Intelligence Agents]
        ST[Strategy Agents]
        AI[Alpha Intelligence Engine]
    end

    subgraph Orchestration Layer
        ARB[Weighted Arbitration]
        COORD[Coordinator]
        MEM[Shared Memory]
    end

    subgraph Risk Layer
        PA[Portfolio Allocator]
        RE[Risk Engine]
        KS[Kill Switch]
    end

    subgraph Execution Layer
        OMS[Order Management]
        EX[Exchange Providers]
    end

    subgraph Observability
        MON[Prometheus Metrics]
        REC[Position Reconciliation]
        EVT[Event Store]
    end

    D --> MI
    D --> ST
    D --> AI
    MI --> ARB
    ST --> ARB
    AI --> ARB
    ARB --> COORD
    COORD --> PA
    PA --> RE
    RE -->|HMAC Token| OMS
    RE -->|Veto| COORD
    KS -.->|Emergency Stop| OMS
    OMS --> EX
    EX --> MON
    EX --> REC
    COORD --> EVT
    COORD <--> MEM
    RE <--> MEM

    style RE fill:#e8eaf6,stroke:#5e35b1,stroke-width:2px
    style KS fill:#ffebee,stroke:#c62828,stroke-width:2px
```

<details>
<summary><strong>Project Structure</strong></summary>

```
src/aiswarm/
├── agents/         # Strategy agents with dynamic registry (@register_agent)
├── api/            # FastAPI control plane (auth, routes, Prometheus)
├── backtest/       # Backtesting engine, adapters, data loader
├── bootstrap.py    # Config → component graph wiring
├── data/           # EventStore (SQLite), market data providers
├── exchange/       # Multi-exchange abstraction layer
│   └── providers/  # Aster, Binance, Coinbase, Bybit, Interactive Brokers
├── execution/      # Order executor, order store, fill tracker, slippage models
├── intelligence/   # Alpha Intelligence Engine + HMM regime detection
├── integrations/   # TradingView webhooks, portfolio trackers, tax export
├── loop/           # Autonomous trading loop (60s cycle)
├── mandates/       # Governance: mandate registry, validator
├── monitoring/     # Prometheus metrics, alerts, reconciliation
├── orchestration/  # Coordinator, arbitration, shared memory
├── portfolio/      # Allocator, exposure manager
├── quant/          # Kelly criterion, risk metrics, drift detection
├── resilience/     # Circuit breaker, rate limiter, retry with backoff, graceful shutdown
├── risk/           # Risk engine, kill switch, drawdown, leverage checks
├── session/        # Session lifecycle management
├── observability/  # OpenTelemetry tracing (optional)
├── plugins/        # Plugin system (strategy, data source, risk guard, integration)
├── types/          # Pydantic domain models (Signal, Order, Portfolio)
└── utils/          # Secrets provider, logging, time utilities
```

</details>

## Quick Start

```bash
# Install
git clone https://github.com/kmshihab7878/Autonomous-Investment-Swarm.git
cd Autonomous-Investment-Swarm
pip install -e ".[dev]"

# Configure (minimum: set HMAC secret)
cp .env.example .env
export AIS_RISK_HMAC_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Run paper trading
python -m aiswarm --mode paper
```

**Docker (full stack with monitoring):**

```bash
cp .env.example .env
# Edit .env with required values
docker compose up --build
```

| Service | Port | Purpose |
|---------|------|---------|
| API | [localhost:8000](http://localhost:8000) | FastAPI control plane + [Swagger UI](http://localhost:8000/docs) + [ReDoc](http://localhost:8000/redoc) |
| Prometheus | [localhost:9090](http://localhost:9090) | Metrics collection |
| Grafana | [localhost:3000](http://localhost:3000) | Dashboards |
| Alertmanager | [localhost:9093](http://localhost:9093) | Alert routing |

**Kubernetes (production):**

```bash
helm install ais deploy/helm/ais/ \
  --set secrets.hmacSecret=$(python -c "import secrets; print(secrets.token_urlsafe(32))") \
  --set loop.executionMode=paper
```

**GitHub Codespaces (zero setup):** Open in Codespaces via `.devcontainer/` — auto-installs dependencies, pre-commit hooks, and forwards ports.

See the [full quickstart guide](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/getting-started/quickstart/) for detailed walkthrough.

## Supported Exchanges

| Exchange | Spot | Futures | Options | Symbol Format |
|----------|:----:|:-------:|:-------:|---------------|
| Aster DEX | x | x | | `BTCUSDT` |
| Binance | x | x | | `BTCUSDT` |
| Coinbase | x | | | `BTC-USD` |
| Bybit | x | x | x | `BTCUSDT` |
| Interactive Brokers | x | x | x | `AAPL`, `BTCUSD` |

Exchange routing is config-driven via `config/exchanges.yaml`. See [Multi-Exchange Setup](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/multi-exchange/).

## How AIS Compares

<table>
<tr><th>Feature</th><th>AIS</th><th>Freqtrade</th><th>Hummingbot</th><th>Jesse</th><th>Qlib</th></tr>
<tr><td><strong>HMAC risk gating</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Alpha Intelligence</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>Partial</td></tr>
<tr><td><strong>HMM regime detection</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Mandate governance</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Plugin system</strong></td><td>Yes</td><td>-</td><td>Partial</td><td>-</td><td>-</td></tr>
<tr><td><strong>Walk-forward + Monte Carlo</strong></td><td>Yes</td><td>Partial</td><td>-</td><td>Partial</td><td>Yes</td></tr>
<tr><td><strong>Multi-exchange + multi-asset</strong></td><td>5 exchanges</td><td>20+ (CCXT)</td><td>30+ DEX</td><td>3</td><td>-</td></tr>
<tr><td><strong>K8s Helm chart</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>OpenTelemetry</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Web dashboard</strong></td><td>Yes (WebSocket)</td><td>FreqUI</td><td>Yes</td><td>Yes</td><td>-</td></tr>
<tr><td><strong>3 execution modes</strong></td><td>Paper/Shadow/Live</td><td>Paper/Live</td><td>Paper/Live</td><td>Paper/Live</td><td>Research only</td></tr>
<tr><td><strong>Property-based tests</strong></td><td>Yes (Hypothesis)</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Session lifecycle</strong></td><td>Yes</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td><strong>Tax/compliance export</strong></td><td>CSV/Koinly/CoinTracker</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
</table>

AIS combines cryptographic risk gating, AI-powered intelligence, mandate governance, and enterprise operations in a way no single competitor matches.

## Built-in Strategies (10)

| Strategy | Agent | Signal Logic |
|----------|-------|-------------|
| `momentum_ma_crossover` | MomentumAgent | Fast/slow SMA crossover with trend consistency |
| `funding_rate_contrarian` | FundingRateAgent | Contrarian on extreme funding rates |
| `mean_reversion_bollinger` | MeanReversionAgent | Bollinger Bands + RSI oversold/overbought |
| `volatility_breakout` | VolatilityBreakoutAgent | Keltner Channel breaks + ATR expansion |
| `rsi_divergence` | RSIDivergenceAgent | Price/RSI divergence reversal detection |
| `vwap_reversion` | VWAPReversionAgent | Mean-reversion to Volume-Weighted Average Price |
| `grid_trading` | GridAgent | Systematic contrarian at regular price intervals |
| `pairs_stat_arb` | PairsAgent | Z-score of correlated pair spreads |
| `sentiment_contrarian` | SentimentAgent | Contrarian to Fear & Greed extremes |
| `regime_hmm` | RegimeDetectorAgent | HMM market regime classification |

All agents use `@register_agent` for config-driven activation. Add to `config/base.yaml` to enable.

## Example Output

Paper trading loop (structured JSON):

```json
{"event": "session_started", "mode": "paper", "strategies": ["momentum_ma_crossover", "funding_rate_contrarian"]}
{"event": "cycle_start", "cycle": 1, "timestamp": "2025-01-15T10:00:00Z"}
{"event": "signal_generated", "agent": "momentum", "symbol": "BTCUSDT", "direction": 1, "confidence": 0.72}
{"event": "risk_approved", "symbol": "BTCUSDT", "size": 0.001, "token": "hmac:a3f2..."}
{"event": "order_submitted", "symbol": "BTCUSDT", "side": "BUY", "qty": 0.001, "mode": "paper"}
{"event": "cycle_end", "cycle": 1, "duration_ms": 245}
```

## Development

```bash
# All quality checks (lint + typecheck + tests with coverage)
make check

# Individual commands
pytest tests/                                              # All tests (unit + integration)
pytest tests/unit/ --cov=src/aiswarm --cov-fail-under=83   # Unit tests with coverage
pytest tests/integration/                                   # Integration tests
pytest tests/benchmarks/ --benchmark-only                   # Performance benchmarks
HYPOTHESIS_PROFILE=ci pytest tests/unit/test_properties.py  # Property-based tests (200 examples)
ruff check src/ tests/                                      # Lint
ruff format --check src/ tests/                             # Format check
mypy src/aiswarm/ --ignore-missing-imports                  # Type check

# Security
make security                                               # pip-audit + bandit

# Documentation
make docs-serve                                             # Local docs at http://localhost:8000
```

## Documentation

Full documentation at **[kmshihab7878.github.io/Autonomous-Investment-Swarm](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/)**:

- **Getting Started** — [Installation](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/getting-started/installation/), [Quick Start](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/getting-started/quickstart/), [Configuration](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/getting-started/configuration/)
- **Architecture** — [Overview](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/overview/), [Agent System](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/agents/), [Risk Engine](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/risk-engine/), [Alpha Intelligence](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/intelligence/), [Execution](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/execution/), [Exchange Layer](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/exchange-layer/), [Portfolio](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/portfolio/), [Data Model](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/architecture/data-model/)
- **Guides** — [Strategy Development](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/strategy-development/), [Backtesting](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/backtesting/), [Multi-Exchange](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/multi-exchange/), [Deployment](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/deployment/), [Monitoring](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/monitoring/)
- **Reference** — [API](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/api/), [Configuration](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/configuration/), [Metrics](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/metrics/), [Decision Log](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/decision-log/), [Quantitative Tools](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/reference/quant-tools/)
- **Operations** — [Risk Policy](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/operations/risk-policy/), [Operating Model](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/operations/operating-model/), [Sessions](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/operations/sessions/)

## Examples

The [`examples/`](examples/) directory includes:

- `paper_trading.env` — Minimal environment for paper trading
- `mean_reversion_agent.py` — Example custom strategy agent
- `backtest_walkforward.py` — Walk-forward optimization demo
- `backtest_montecarlo.py` — Monte Carlo simulation demo
- `plugins/simple_strategy.py` — Example plugin (SMA crossover)

See the [Strategy Development Guide](https://kmshihab7878.github.io/Autonomous-Investment-Swarm/guides/strategy-development/) for a complete tutorial on building custom agents.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, commit conventions, and PR requirements.

Questions? Start a [discussion](https://github.com/kmshihab7878/Autonomous-Investment-Swarm/discussions).

## Security

If you discover a security vulnerability, please report it responsibly. See [SECURITY.md](SECURITY.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned milestones (v1.3 through v2.0) and the research track.

## License

Apache License 2.0. See [LICENSE](LICENSE).
