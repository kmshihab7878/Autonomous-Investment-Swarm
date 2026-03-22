# ADR-0005: Three Execution Modes (Paper / Shadow / Live)

## Status

Accepted

## Date

2026-03-22

## Context

Moving an autonomous trading system from development to production is inherently
dangerous. A single bug in order sizing, symbol mapping, or side selection can
cause immediate financial loss. The gap between "it works in unit tests" and "it
trades real money safely" requires a graduated confidence-building process.

AIS needs a mechanism to run the full trading pipeline -- market data ingestion,
signal generation, arbitration, risk validation, and order management -- while
controlling whether orders actually reach the exchange. Crucially, the code paths
must be identical across modes. Mode-specific branching (e.g.,
`if mode == "paper": skip_risk_check()`) introduces behavioral divergence that
defeats the purpose of staged deployment.

The system also needs to support parallel operation: a paper instance validating
a new strategy while a live instance runs proven strategies. The mode is a
per-instance configuration, not a global toggle.

## Decision

AIS defines three execution modes as an enum in
`src/aiswarm/execution/aster_executor.py`:

```python
class ExecutionMode(str, Enum):
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"
```

### Mode Behavior

**Paper mode** (`AIS_EXECUTION_MODE=paper`):

- The `ExchangeProvider` is initialized with a `MockMCPGateway` that simulates
  exchange responses without any network calls.
- Orders flow through the full pipeline: agent signal generation, coordinator
  arbitration, mandate validation, risk engine approval (including HMAC token
  signing), and executor submission.
- The `LiveOrderExecutor._handle_paper()` method simulates fills immediately:
  the order is tracked in the `OrderStore`, assigned a `paper_{order_id}`
  exchange ID, and recorded as filled with simulated price and quantity.
- Market data can be either simulated (from backtesting data loader) or real
  (fetched via MCP for forward-testing with real prices but simulated execution).
- No external services required beyond the configuration files.

**Shadow mode** (`AIS_EXECUTION_MODE=shadow`):

- The `ExchangeProvider` is initialized with a real `HTTPMCPGateway` pointing
  to the exchange's MCP server.
- Market data, balances, positions, and order books are fetched from the live
  exchange. All read operations are real.
- Orders flow through the full pipeline including risk validation and HMAC
  signing, but `AsterExecutor` blocks the final `place_order` call. The order
  is logged with status `SHADOW_BLOCKED` but never submitted.
- This mode validates that: (a) MCP connectivity works, (b) exchange data
  parsing is correct, (c) risk validation produces sensible results against
  real market conditions, (d) order parameters are well-formed.
- Requires `AIS_MCP_SERVER_URL` and exchange credentials for read access.

**Live mode** (`AIS_EXECUTION_MODE=live`):

- Full real execution. Orders are submitted to the exchange via the
  `ExchangeProvider.place_order()` method through the `HTTPMCPGateway`.
- Requires two explicit safety gates:
  1. `AIS_EXECUTION_MODE=live` (configures the mode).
  2. `AIS_ENABLE_LIVE_TRADING=true` (separate boolean gate).
  Both must be set. If `AIS_ENABLE_LIVE_TRADING` is missing or not `"true"`,
  the executor refuses to submit orders even in live mode.
- Additionally requires: valid `AIS_RISK_HMAC_SECRET`, exchange-specific API
  credentials, and `AIS_API_KEY` for the control plane.
- Every order must carry a valid, non-expired HMAC risk token (ADR-0001).
  The executor calls `verify_risk_token(token, order_id)` before submission.

### Code Path Guarantee

The critical design property is that **all three modes execute the same code
paths** up to the point of exchange submission:

1. Agents generate signals using the same logic regardless of mode.
2. The coordinator arbitrates signals identically.
3. The risk engine validates and signs orders identically.
4. The `LiveOrderExecutor.submit_order()` method branches only at the final
   step: paper mode calls `_handle_paper()` for simulated fills, shadow mode
   logs and blocks, live mode calls `provider.place_order()`.

This means a bug discovered in paper mode (e.g., incorrect position sizing)
will also exist in live mode. Conversely, if paper mode passes all tests, the
only remaining risks in live mode are exchange-specific (connectivity, rate
limits, fill behavior).

### Mode Selection

The mode is determined by the `AIS_EXECUTION_MODE` environment variable
(default: `"paper"`). It can also be set via CLI (`--mode paper`) or the
FastAPI control plane API. The `bootstrap_from_config()` function wires the
appropriate gateway based on mode.

## Consequences

### Positive

- **Graduated confidence**: Operators follow a natural progression: paper (verify
  logic) -> shadow (verify exchange integration) -> live (deploy with real
  capital). Each stage adds confidence without requiring code changes.
- **No mode-specific bugs**: The identical code path guarantee means paper mode
  is a high-fidelity preview of live behavior. Bugs found in paper mode are
  real bugs that would affect live trading.
- **Safe default**: The default mode is `paper`. A misconfigured deployment that
  omits `AIS_EXECUTION_MODE` will simulate trades rather than execute them.
  The double gate for live mode (`AIS_EXECUTION_MODE=live` AND
  `AIS_ENABLE_LIVE_TRADING=true`) makes accidental live trading virtually
  impossible.
- **Parallel operation**: Multiple instances can run different modes
  simultaneously. A shadow instance can validate a new exchange integration
  while a paper instance tests a new strategy and a live instance runs
  production strategies.
- **Incident response**: Switching from live to paper mode is a single
  environment variable change (or API call via the control plane). The system
  continues running, generating signals and logging decisions, but stops
  submitting orders.

### Negative

- **Shadow mode latency**: Shadow mode makes real exchange API calls for market
  data, which adds latency compared to paper mode. This is intentional (the
  point is to test real connectivity) but means shadow mode is slower.
- **Paper mode fill simulation is simplistic**: `_handle_paper()` simulates
  immediate fills at the current price with no slippage, partial fills, or
  order book impact. Paper mode performance will be optimistic compared to
  live. Future work: implement a proper fill simulator with slippage models.
- **Double gate friction**: Requiring both `AIS_EXECUTION_MODE=live` and
  `AIS_ENABLE_LIVE_TRADING=true` is redundant by design. Some operators may
  find the extra configuration annoying. The friction is the feature.
- **No automatic mode promotion**: There is no built-in mechanism to
  automatically graduate from paper to shadow to live based on performance
  metrics. Mode transitions require manual operator intervention.

### Neutral

- The `ExecutionMode` enum inherits from both `str` and `Enum`, allowing it to
  be used directly in string comparisons and serialized naturally in JSON/YAML.
  This matches the convention used by other AIS enums.

## Alternatives Considered

### Two Modes (Test / Live)

A binary toggle between simulated and real execution. Rejected because:
(1) loses the valuable intermediate shadow mode that tests exchange connectivity
without financial risk, (2) the jump from "everything simulated" to "everything
real" is too large for a system managing real capital, (3) shadow mode has caught
real integration bugs (incorrect symbol normalization, unexpected API response
formats) that paper mode cannot detect.

### Feature Flags per Operation

Use per-operation flags (e.g., `allow_read=true, allow_write=false`) instead of
named modes. Rejected because: (1) combinatorial explosion of flag combinations,
most of which are nonsensical, (2) harder to reason about than three well-defined
modes, (3) risk of misconfiguration (e.g., `allow_write=true` with
`allow_read=false` is dangerous), (4) named modes communicate intent clearly
to operators and in logs.

### Separate Codebases for Simulation and Production

Maintain a simulation-only version and a production version of the executor.
Rejected because: (1) behavioral divergence is inevitable when two codebases
evolve independently, (2) doubles the testing burden, (3) bugs fixed in one
must be manually ported to the other, (4) the entire point of this decision is
that one codebase serves all modes.

### Dry-Run Flag on Orders

Add a `dry_run: bool` field to the Order model and let the exchange gateway
honor it. Rejected because: (1) the flag travels with the order, meaning a bug
could flip it, (2) exchanges do not universally support dry-run submission,
(3) mode is a system-level concern, not a per-order concern, (4) this pattern
puts the safety decision at the wrong layer (the order rather than the
executor).

## References

- `src/aiswarm/execution/aster_executor.py` -- `ExecutionMode`, `AsterExecutor`
- `src/aiswarm/execution/live_executor.py` -- `LiveOrderExecutor`, `_handle_paper()`
- `src/aiswarm/bootstrap.py` -- mode-based gateway wiring
- `src/aiswarm/exchange/provider.py` -- `ExchangeProvider` ABC
- ADR-0001: HMAC Risk Tokens (live mode dependency)
- ADR-0002: Exchange Provider Abstraction (gateway injection)
