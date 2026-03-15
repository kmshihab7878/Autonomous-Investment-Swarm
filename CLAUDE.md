# AIS — CLAUDE.md

## Project Identity

- **Name**: Autonomous Investment Swarm
- **Owner**: Khaled Shihab (kmshihab7878)
- **Repository**: `Financial-Intelligence-Department-FID`
- **Production Track**: `src/aiswarm/` (AIS — risk-gated execution on Aster DEX)

## Architecture

```
src/aiswarm/
├── agents/         # Strategy agents (momentum, funding rate)
├── api/            # FastAPI control plane (auth, routes, Prometheus)
├── bootstrap.py    # Config → component graph wiring
├── data/           # EventStore (SQLite), Aster data provider
├── execution/      # AsterExecutor, LiveOrderExecutor, OrderStore, FillTracker
├── loop/           # Autonomous trading loop (60s cycle)
├── mandates/       # Governance: mandate registry, validator
├── monitoring/     # Prometheus metrics, alerts, reconciliation
├── orchestration/  # Coordinator, arbitration, SharedMemory
├── portfolio/      # Allocator, exposure manager
├── quant/          # Kelly criterion, risk metrics
├── resilience/     # Circuit breaker, rate limiter, graceful shutdown
├── risk/           # RiskEngine, kill switch, drawdown, leverage, liquidity
├── session/        # Session lifecycle (schedule → approve → active → end)
└── types/          # Pydantic domain models (Signal, Order, Portfolio)
```

## Critical Invariants

1. **No order executes without HMAC-signed risk approval** — `RiskEngine.validate()` signs; token verified before submission
2. **Fail closed on missing secrets** — `AIS_RISK_HMAC_SECRET` and `AIS_API_KEY` (in live mode) are mandatory
3. **Mandate-strategy alignment** — config strategy names must match agent code (`momentum_ma_crossover`, `funding_rate_contrarian`)
4. **Control state via Redis** — API and loop share state through `ais:control:state` Redis key
5. **Three execution modes** — PAPER (simulated), SHADOW (read-only), LIVE (requires `AIS_ENABLE_LIVE_TRADING=true`)

## Development Commands

```bash
# Tests
pytest tests/unit/ --cov=src/aiswarm --cov-fail-under=60

# Lint
ruff check src/ tests/unit/
ruff format --check src/ tests/unit/

# Type check
mypy src/aiswarm/ --ignore-missing-imports

# Docker
docker compose up --build
```

## Environment Variables (Required)

| Variable | Purpose | Required When |
|----------|---------|---------------|
| `AIS_RISK_HMAC_SECRET` | HMAC key for risk token signing | Always |
| `AIS_API_KEY` | Bearer token for API auth | Live mode |
| `AIS_EXECUTION_MODE` | `paper` / `shadow` / `live` | Always (default: paper) |
| `AIS_ENABLE_LIVE_TRADING` | Safety gate for live orders | Live mode |
| `AIS_MCP_SERVER_URL` | Aster DEX MCP server endpoint | Shadow / Live mode |
| `ASTER_ACCOUNT_ID` | Aster DEX account identifier | Live mode |
| `REDIS_URL` | Redis connection for control state | Always (default: redis://localhost:6379/0) |

## MCP Gateway Modes

- **Paper**: `MockMCPGateway` — simulated fills, no exchange connection
- **Shadow/Live**: `AsterMCPGateway` — real MCP server with circuit breaker + rate limiter
- Gateway auto-selected by execution mode; override via `bootstrap_from_config(gateway=...)`
- CLI: `python -m aiswarm --mode live --mcp-server-url http://mcp:8080`

## Conventions

- Python 3.10+, type hints on all signatures
- Pydantic v2 for domain models (frozen=True)
- structlog for structured logging
- SQLite EventStore for append-only audit trail
- YAML config in `config/` (base, risk, execution, mandates, portfolio, monitoring)
- Tests in `tests/unit/` using pytest, 60% coverage minimum
