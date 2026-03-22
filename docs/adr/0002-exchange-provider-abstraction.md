# ADR-0002: Exchange Provider Abstraction Layer

## Status

Accepted

## Date

2026-03-22

## Context

AIS must support multiple exchanges: Aster DEX, Binance, Coinbase, Bybit, and
Interactive Brokers. Each exchange has a different API surface, different symbol
formats (e.g., `BTCUSDT` on Binance vs `BTC-USD` on Coinbase), different
supported asset classes (Coinbase has spot only; Bybit supports options), and
different MCP tool names (e.g., `mcp__aster__create_order` vs
`mcp__binance__create_order`). Agents and the execution pipeline should not know
or care which exchange an order targets.

The previous design had exchange-specific tool names scattered throughout the
codebase. Adding a new exchange required modifying the coordinator, the
executor, the data pipeline, and multiple agent files. This violated the
open-closed principle and made exchange addition a cross-cutting, error-prone
change.

A secondary concern is testability. The trading loop needs to run in paper mode
with simulated fills, in shadow mode with real market data but no order
submission, and in live mode with real execution. These modes must use the same
code paths to avoid mode-specific bugs.

## Decision

All exchange communication goes through an `ExchangeProvider` abstract base class
defined in `src/aiswarm/exchange/provider.py`. The ABC defines 16 methods across
four categories:

**Symbol normalization (2 abstract methods):**
- `normalize_symbol(canonical) -> str` -- converts canonical format (e.g.,
  `BTC/USDT`) to exchange-native format.
- `to_canonical_symbol(exchange_sym) -> str` -- reverse conversion.

**Market data (4 methods, 3 abstract + 1 default):**
- `get_klines(symbol, interval, limit) -> list[OHLCV]` -- OHLCV candle data.
- `get_ticker(symbol) -> Ticker | None` -- 24h ticker snapshot.
- `get_order_book(symbol) -> OrderBook | None` -- depth snapshot.
- `get_funding_rate(symbol) -> FundingRate | None` -- default returns `None`
  (not all exchanges support funding rates).

**Account (3 methods, 2 abstract + 1 default):**
- `get_balance() -> AccountBalance | None` -- account summary.
- `get_positions() -> list[ExchangePosition]` -- open positions.
- `get_income() -> list[IncomeRecord]` -- default returns empty list.

**Trading (5 methods, 3 abstract + 2 default):**
- `place_order(symbol, side, quantity, order_type, price, venue, **kwargs) -> dict`
- `cancel_order(symbol, order_id, venue) -> dict`
- `cancel_all_orders(symbol, venue) -> dict`
- `get_order_status(symbol, order_id, venue) -> dict` -- default raises
  `NotImplementedError`.
- `get_my_trades(symbol, venue) -> list[TradeRecord]` -- default returns empty
  list.

**Optional leverage/margin (2 default methods):**
- `set_leverage(symbol, leverage) -> dict`
- `set_margin_mode(symbol, mode) -> dict`

Each concrete provider (e.g., `AsterExchangeProvider` in
`src/aiswarm/exchange/providers/aster.py`) encapsulates ALL exchange-specific MCP
tool name strings as module-level constants. No tool name like
`mcp__aster__create_order` appears anywhere outside the provider module.

Providers also declare two properties:
- `exchange_id: str` -- unique identifier (e.g., `"aster"`, `"binance"`).
- `supported_asset_classes: AssetClass` -- a `Flag` enum combining `SPOT`,
  `FUTURES`, `OPTIONS`, `STOCKS`, `FOREX`.

All return types are canonical types defined in `src/aiswarm/exchange/types.py`:
`OHLCV`, `Ticker`, `OrderBook`, `AccountBalance`, `ExchangePosition`,
`FundingRate`, `TradeRecord`, `IncomeRecord`. Providers are responsible for
parsing exchange-specific responses into these canonical types.

**Multi-exchange orchestration** uses two additional components:
- `ExchangeRegistry` (`exchange/registry.py`): manages provider instances by
  exchange ID, supports a default provider, auto-sets the first registered
  provider as default.
- `SymbolRouter` (`exchange/symbols.py`): maps symbols to exchanges based on
  `config/exchanges.yaml`, enabling the coordinator to route orders to the
  correct provider without knowing exchange details.

## Consequences

### Positive

- **Clean separation**: All exchange-specific knowledge is encapsulated in a
  single module per exchange. The rest of the codebase depends only on the ABC
  and canonical types.
- **Easy extension**: Adding a new exchange requires implementing one class with
  well-defined abstract methods. No changes to agents, coordinator, or risk
  engine.
- **No tool name leakage**: MCP tool strings are confined to provider modules,
  eliminating the risk of typos or stale references in consuming code.
- **Testability**: Paper mode injects a `MockMCPGateway` into the same provider
  class, ensuring paper and live modes exercise identical code paths.
- **Gradual capability**: Non-abstract methods with sensible defaults
  (`get_funding_rate` returns `None`, `get_income` returns `[]`) mean
  spot-only exchanges do not need to stub out futures-specific methods.

### Negative

- **Lowest common denominator**: The ABC surface area is limited to operations
  common across exchanges. Exchange-specific features (e.g., Bybit's unified
  trading account, IB's combo orders) must be accessed through `**kwargs` on
  `place_order` or by downcasting, which is not type-safe.
- **Response normalization cost**: Each provider must parse exchange-specific
  JSON into canonical types. Parsing bugs in a provider can be subtle because
  the types are shared but the source formats differ.
- **Dual venue complexity**: The `venue` parameter (`"futures"` / `"spot"`) on
  trading methods adds a dimension that not all exchanges need. Coinbase ignores
  it; Aster uses it to dispatch between separate API endpoints.

### Neutral

- The canonical types in `exchange/types.py` are data containers, not Pydantic
  models. This is intentional to avoid serialization overhead on the hot path
  (thousands of OHLCV records per cycle). They are `@dataclass(frozen=True)`.

## Alternatives Considered

### Direct MCP Tool Calls in Agents

Have each agent call exchange-specific MCP tools directly (e.g.,
`mcp__aster__get_klines`). Rejected because: (1) every agent would need
exchange-specific branches, (2) adding an exchange requires touching every agent
file, (3) tool name strings scattered across the codebase are fragile and
untestable, (4) paper mode would require mocking at the MCP layer rather than
the provider layer.

### Unified REST Client with Exchange Adapters

Use a generic HTTP client (like ccxt) with per-exchange configuration rather
than an ABC. Rejected because: (1) AIS communicates with exchanges through MCP
servers, not direct REST; the adapter layer would be an awkward wrapper around
MCP tool calls, (2) ccxt's type system is stringly-typed dicts, whereas our ABC
enforces typed return values, (3) ccxt does not support MCP gateway protocols.

### Strategy Pattern with Function Dispatch

Use a dictionary mapping `(exchange, operation)` tuples to callable handlers
instead of an ABC class hierarchy. Rejected because: (1) loses the structural
guarantee that all operations are implemented per exchange, (2) no IDE support
for "implement all abstract methods", (3) harder to test in isolation since
handlers share no common interface.

### Protocol (Structural Typing) Instead of ABC

Use `typing.Protocol` for structural subtyping instead of `ABC` for nominal
subtyping. Considered viable but rejected because: (1) `ABC` provides clearer
error messages when a method is not implemented ("Can't instantiate abstract
class"), (2) the providers are explicitly registered in the `ExchangeRegistry`,
so nominal typing is appropriate, (3) `Protocol` provides less discoverability
for implementors.

## References

- `src/aiswarm/exchange/provider.py` -- `ExchangeProvider` ABC, `AssetClass` flag
- `src/aiswarm/exchange/registry.py` -- `ExchangeRegistry`
- `src/aiswarm/exchange/symbols.py` -- `SymbolRouter`
- `src/aiswarm/exchange/types.py` -- canonical data types
- `src/aiswarm/exchange/providers/aster.py` -- reference implementation
- `src/aiswarm/exchange/providers/binance.py` -- Binance implementation
- `src/aiswarm/exchange/providers/bybit.py` -- Bybit implementation
- `config/exchanges.yaml` -- symbol-to-exchange routing configuration
