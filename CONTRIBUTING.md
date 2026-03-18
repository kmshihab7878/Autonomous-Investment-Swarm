# Contributing to AIS

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

## Code Style

- **Linter/formatter**: [Ruff](https://docs.astral.sh/ruff/) (line length 100)
- **Type checker**: [mypy](https://mypy-lang.org/) (strict mode)
- **Type hints**: Required on all function signatures
- **Models**: Pydantic v2 with `frozen=True`
- **Logging**: stdlib `logging` with custom `JsonFormatter` (structured JSON)

## Pull Request Requirements

Before submitting a PR:

1. **Tests pass**: `pytest tests/unit/ -v --cov=src/aiswarm --cov-fail-under=83`
2. **Lint clean**: `ruff check src/ tests/unit/`
3. **Format clean**: `ruff format --check src/ tests/unit/`
4. **Type check passes**: `mypy src/aiswarm/ --ignore-missing-imports`

## Testing

- Tests live in `tests/unit/`
- Use pytest with the AAA pattern (Arrange, Act, Assert)
- Minimum 83% code coverage
- New features require corresponding tests

## Project Structure

All production code lives under `src/aiswarm/`. See [README.md](README.md) for the module layout.

## Security

If you find a security vulnerability, do NOT open a public issue. See [SECURITY.md](SECURITY.md).
