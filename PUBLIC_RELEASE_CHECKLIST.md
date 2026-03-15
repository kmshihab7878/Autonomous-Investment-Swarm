# Public Release Checklist

Pre-release verification for the AIS repository.

## Secrets & Credentials

- [ ] No API keys, tokens, or passwords in source code
- [ ] `.env.example` contains only placeholder values
- [ ] `.gitignore` excludes `.env`, `*.pem`, `*.key`, credentials files
- [ ] No hardcoded connection strings in config files
- [ ] HMAC secrets use environment variables only

## Legacy & Internal References

- [ ] No references to internal project names or legacy codepaths
- [ ] No internal URLs, IPs, or hostnames in source or docs
- [ ] README reflects current project scope only

## License & Legal

- [ ] LICENSE file present (Apache 2.0)
- [ ] Copyright holder correct in LICENSE
- [ ] License header consistent across documentation

## Documentation

- [ ] README.md accurate and up to date
- [ ] CONTRIBUTING.md present with dev setup instructions
- [ ] SECURITY.md present with reporting instructions
- [ ] CODE_OF_CONDUCT.md present
- [ ] ARCHITECTURE.md reflects current module layout

## CI/CD & Infrastructure

- [ ] GitHub Actions workflows present and functional
- [ ] Docker build succeeds: `docker build -t ais:test .`
- [ ] docker-compose.yml uses environment variables (no inline secrets)
- [ ] Pre-commit hooks configured

## Code Quality

- [ ] Linter passes: `ruff check src/ tests/unit/`
- [ ] Formatter passes: `ruff format --check src/ tests/unit/`
- [ ] Type checker passes: `mypy src/aiswarm/ --ignore-missing-imports`
- [ ] Tests pass: `pytest tests/unit/ --cov=src/aiswarm --cov-fail-under=60`
- [ ] No `TODO`/`FIXME` without linked issues
