# Autonomous Investment Swarm (AIS)

[![Status: Experimental](https://img.shields.io/badge/status-experimental-orange)]()
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An experimental autonomous trading system built around risk-gated multi-agent orchestration. AIS coordinates specialist strategy agents through a governed execution pipeline with mandatory risk validation before any order submission.

> **WARNING**: This software is experimental and intended for research and educational purposes. Trading cryptocurrencies involves substantial risk of loss. Never deploy with funds you cannot afford to lose. The authors accept no liability for financial losses incurred through use of this software.

## Architecture

```
Signal Generation -> Coordinator -> Portfolio Allocator -> Risk Engine -> Order Management
```

AIS enforces a strict control path: every order must pass through `RiskEngine.validate()`, which issues an HMAC-signed approval token. The token is verified again before submission. Orders without valid tokens are rejected.

```
src/aiswarm/
├── agents/         # Strategy agents (momentum, funding rate)
├── api/            # FastAPI control plane (auth, routes, Prometheus)
├── bootstrap.py    # Config -> component graph wiring
├── data/           # EventStore (SQLite), Aster data provider
├── execution/      # Order executor, order store, fill tracker
├── loop/           # Autonomous trading loop (60s cycle)
├── mandates/       # Governance: mandate registry, validator
├── monitoring/     # Prometheus metrics, alerts, reconciliation
├── orchestration/  # Coordinator, arbitration, shared memory
├── portfolio/      # Allocator, exposure manager
├── quant/          # Kelly criterion, risk metrics
├── resilience/     # Circuit breaker, rate limiter, graceful shutdown
├── risk/           # Risk engine, kill switch, drawdown, leverage checks
├── session/        # Session lifecycle management
└── types/          # Pydantic domain models (Signal, Order, Portfolio)
```

## Prerequisites

- Python 3.10+
- Redis (for control state)
- Docker and Docker Compose (optional, for full stack)

## Quick Start

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env — at minimum set AIS_RISK_HMAC_SECRET

# Run API server
uvicorn aiswarm.api.app:app --app-dir src --reload

# Run trading loop (paper mode, default)
python -m aiswarm --config config/
```

### Docker

```bash
# Configure environment first
cp .env.example .env
# Edit .env with required values (see Configuration below)

docker compose up --build
```

## Configuration

AIS uses YAML configuration files in `config/`:

| File | Purpose |
|------|---------|
| `base.yaml` | Core system settings |
| `risk.yaml` | Risk limits, drawdown thresholds, leverage caps |
| `execution.yaml` | Execution mode, order routing |
| `mandates.yaml` | Strategy mandates, allowed assets, allocation limits |
| `portfolio.yaml` | Portfolio constraints, rebalancing rules |
| `monitoring.yaml` | Alerting, metrics, reconciliation |

### Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `AIS_RISK_HMAC_SECRET` | HMAC key for risk token signing (always required) |
| `AIS_API_KEY` | Bearer token for API auth (live mode) |
| `AIS_EXECUTION_MODE` | `paper` / `shadow` / `live` (default: `paper`) |
| `REDIS_URL` | Redis connection (default: `redis://localhost:6379/0`) |

See `.env.example` for the full list.

### Execution Modes

- **Paper** (default): Simulated fills, no exchange connection
- **Shadow**: Read-only connection to exchange, no order submission
- **Live**: Real order submission (requires `AIS_ENABLE_LIVE_TRADING=true`)

## Running Tests

```bash
# Unit tests with coverage
pytest tests/unit/ -v --cov=src/aiswarm --cov-fail-under=60

# Lint
ruff check src/ tests/unit/
ruff format --check src/ tests/unit/

# Type check
mypy src/aiswarm/ --ignore-missing-imports
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR requirements.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

Apache License 2.0. See [LICENSE](LICENSE).
