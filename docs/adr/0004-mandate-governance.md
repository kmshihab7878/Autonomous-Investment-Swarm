# ADR-0004: Mandate-Based Governance System

## Status

Accepted

## Date

2026-03-22

## Context

AIS runs multiple autonomous strategy agents (momentum MA crossover, funding
rate contrarian, alpha follower, etc.) concurrently. Without governance, each
agent has implicit access to the full portfolio capital and can trade any symbol.
This creates several risks: (1) capital overcommitment -- two agents
simultaneously deploying 80% of capital produces 160% exposure, (2) uncontrolled
risk accumulation -- agents with different risk profiles stack positions in the
same direction without awareness of each other, (3) no revocability -- there is
no mechanism to disable a specific strategy without stopping the entire system.

Institutional trading desks solve this with a mandate system: each portfolio
manager receives a written mandate specifying which instruments they may trade,
how much capital they can deploy, and what loss limits apply. AIS needs an
analogous system that is machine-readable, enforceable at the risk engine level,
and auditable.

Additionally, the mandate system must integrate with the existing HMAC risk
token pipeline (ADR-0001). An order must first match an active mandate, then
pass global risk validation, then pass mandate-specific risk validation. The
stricter-of-two limits wins for overlapping checks (e.g., if the global daily
loss limit is 3% but the mandate daily loss limit is 1%, the 1% limit applies).

## Decision

Governance is implemented through three components:

### 1. Mandate Model (`src/aiswarm/mandates/models.py`)

A `Mandate` is a frozen Pydantic v2 model defining:

- `mandate_id: str` -- unique identifier.
- `strategy: str` -- which agent strategy this mandate authorizes (must match
  the agent's registered strategy name, e.g., `"momentum_ma_crossover"`).
- `symbols: tuple[str, ...]` -- the set of symbols the strategy may trade under
  this mandate.
- `risk_budget: MandateRiskBudget` -- the per-mandate risk constraints.
- `status: MandateStatus` -- one of `ACTIVE`, `PAUSED`, `REVOKED`, `EXPIRED`.
- `created_at`, `updated_at`, `created_by`, `notes` -- metadata.

`MandateRiskBudget` defines:

- `max_capital: float` -- maximum notional capital for this mandate.
- `max_daily_loss: float` -- maximum daily loss as a fraction of `max_capital`.
- `max_drawdown: float` -- maximum drawdown as a fraction of `max_capital`.
- `max_open_orders: int` -- maximum concurrent open orders (default: 5).
- `max_position_notional: float` -- per-position notional cap (0 = use
  `max_capital`). Exposed as `effective_position_notional` property.

### 2. Mandate Registry (`src/aiswarm/mandates/registry.py`)

`MandateRegistry` maintains an in-memory dictionary of mandates backed by the
EventStore for durability. It provides:

- `create(mandate_id, strategy, symbols, risk_budget, ...)` -- creates and
  persists a mandate. Raises `ValueError` if the mandate_id already exists.
  Emits a `"mandate"` event with `action: "created"` to the EventStore.
- `get(mandate_id)` -- point lookup.
- `list_active()` -- all mandates with `status == ACTIVE`.
- `list_all()` -- all mandates regardless of status.
- `update_status(mandate_id, new_status)` -- transitions status. Emits a
  `"mandate"` event with `action: "status_changed"`.
- `revoke(mandate_id)` -- convenience for `update_status(id, REVOKED)`.
- `find_mandate_for_order(strategy, symbol)` -- searches active mandates for
  one matching both the strategy name and symbol. Returns the first match or
  `None`.

### 3. Mandate Validator (`src/aiswarm/mandates/validator.py`)

`MandateValidator` is the enforcement layer. It wraps the registry and provides:

- `validate_order(order) -> MandateValidation` -- checks that an active mandate
  exists for the order's strategy and symbol combination. Returns `(ok=True,
  mandate)` or `(ok=False, reason)`. Orders without a matching mandate are
  rejected with a descriptive reason.
- `check_mandate_capital(mandate, current_exposure) -> bool` -- checks remaining
  capital budget.
- `check_mandate_daily_loss(mandate, daily_pnl) -> bool` -- checks daily loss
  against the mandate's limit.

### Integration with Risk Engine

`RiskEngine.validate_with_mandate()` layers mandate-specific checks on top of
global validation:

1. Run all global checks (kill switch, exposure, drawdown, leverage, liquidity).
2. If global checks fail, reject immediately (mandate checks are irrelevant).
3. If global checks pass, apply mandate-specific checks:
   - Mandate daily loss: `abs(daily_pnl) / max_capital >= max_daily_loss`.
   - Mandate drawdown: `drawdown >= max_drawdown`.
   - Mandate capital: `current_exposure + order.notional > max_capital`.
   - Position notional: `order.notional > effective_position_notional`.
4. The stricter-of-two wins: if global drawdown is 5% and mandate drawdown is
   2%, the 2% limit governs.

### Bootstrap Validation

At system startup, `bootstrap_from_config()` validates that every mandate's
`strategy` field matches a registered agent. Mandates referencing unknown
strategies are logged as warnings and not activated, preventing configuration
drift.

## Consequences

### Positive

- **Capital isolation**: Each strategy operates within its allocated capital
  budget. Two strategies cannot collectively exceed total available capital if
  mandates are properly sized.
- **Granular control**: Operators can pause or revoke a single strategy's mandate
  without stopping the entire system. This is critical during incidents.
- **Layered risk**: Mandate limits layer on top of global limits with
  stricter-wins semantics. This prevents a loose global config from undermining
  tight per-strategy controls.
- **Audit trail**: All mandate lifecycle events (creation, status changes) are
  persisted in the EventStore, providing a complete governance history.
- **Strategy-symbol binding**: Mandates explicitly authorize which symbols a
  strategy may trade, preventing a momentum agent from accidentally trading an
  illiquid symbol it was not designed for.

### Negative

- **Configuration overhead**: Every strategy requires a mandate before it can
  trade. Forgetting to create a mandate for a new strategy results in all its
  orders being rejected. This is intentional (fail-safe) but can be confusing
  during initial setup.
- **In-memory index**: The mandate registry is an in-memory dictionary. If the
  process restarts, mandates must be re-created from configuration. There is no
  automatic reload from the EventStore. Mitigated by mandates being defined in
  YAML config and created at bootstrap.
- **No dynamic capital rebalancing**: If Strategy A uses only 30% of its mandate
  capital, the unused 70% is not available to Strategy B. Capital efficiency
  requires manual mandate resizing. This is a deliberate trade-off favoring
  safety over efficiency.
- **Single mandate per strategy-symbol pair**: `find_mandate_for_order` returns
  the first match. If two mandates cover the same strategy-symbol combination
  (e.g., different time horizons), only one is used. Future work may require
  mandate priority or specificity ranking.

### Neutral

- Mandates are frozen Pydantic models. Status changes produce new model
  instances via `model_copy(update=...)`. This matches the immutability
  convention used throughout AIS but means the registry dictionary values
  are replaced, not mutated.

## Alternatives Considered

### Role-Based Access Control (RBAC)

Assign each agent a role with permissions (e.g., `can_trade_futures`,
`max_order_size_10k`). Rejected because: (1) RBAC is coarse-grained -- it
controls what an agent *can* do but not how much capital it should use,
(2) no concept of capital budgets or drawdown limits in standard RBAC models,
(3) mandates are a superset of RBAC for trading use cases.

### Global Risk Limits Only

Use a single set of risk limits shared across all strategies without
per-strategy budgets. Rejected because: (1) a single aggressive strategy can
consume all risk budget and crowd out conservative strategies, (2) no way to
pause one strategy without affecting others, (3) performance attribution is
impossible without per-strategy tracking, (4) institutional best practice
mandates per-strategy risk budgets.

### Smart Contract Governance

Encode mandates as on-chain smart contracts that validate orders before
execution. Rejected because: (1) most target exchanges (Binance, Coinbase, IB)
are centralized and do not support on-chain validation, (2) gas costs and
latency are unacceptable for a 60-second trading cycle, (3) smart contract
upgrades are complex and irreversible.

## References

- `src/aiswarm/mandates/models.py` -- `Mandate`, `MandateRiskBudget`, `MandateStatus`
- `src/aiswarm/mandates/registry.py` -- `MandateRegistry`
- `src/aiswarm/mandates/validator.py` -- `MandateValidator`, `MandateValidation`
- `src/aiswarm/risk/limits.py` -- `RiskEngine.validate_with_mandate()`
- `config/mandates.yaml` -- mandate configuration
- ADR-0001: HMAC Risk Tokens (dependency)
