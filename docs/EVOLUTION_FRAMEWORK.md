# Quantitative Analysis Framework

## Overview

AIS includes a suite of quantitative tools for position sizing, risk measurement, and distribution drift detection. These are domain-agnostic implementations extracted from the AIS research system, located in `src/aiswarm/quant/`.

## Components

### Kelly Criterion (`quant/kelly.py`)

Optimal position sizing based on edge and odds:

- `kelly_fraction(win_prob, payout_ratio)` — full Kelly optimal bet fraction
- `half_kelly(win_prob, payout_ratio)` — conservative half-Kelly variant
- `kelly_position_size(win_prob, payout_ratio, capital, max_position_pct)` — constrained sizing with max position cap
- `expected_value()`, `variance()`, `sharpe_ratio()` — supporting statistics

### Risk Metrics (`quant/risk_metrics.py`)

Comprehensive return distribution analysis:

- `compute_risk_metrics(returns)` — full suite: mean return, volatility, Sharpe, Sortino, VaR (95%), CVaR, max drawdown, skewness, kurtosis
- `parametric_var(mean, std, confidence)` — Gaussian Value at Risk
- `parametric_es(mean, std, confidence)` — Expected Shortfall (CVaR)
- `monte_carlo_var(returns, n_simulations, horizon)` — bootstrap Monte Carlo VaR with percentile distribution

### Drift Detection (`quant/drift.py`)

Statistical tests for detecting distribution shifts in market data:

- `ks_drift_test(reference, current)` — Kolmogorov-Smirnov two-sample test
- `psi_drift_test(reference, current)` — Population Stability Index (PSI < 0.1: stable, > 0.2: significant drift)
- `cusum_test(data, target_mean)` — Cumulative sum control chart for mean shift detection
- `detect_drift(reference, current)` — combined KS + PSI assessment

## Usage in AIS

The risk engine uses these tools during order validation:

1. **Position sizing**: Kelly criterion determines optimal position size given strategy win rate and payout ratio
2. **Risk assessment**: VaR and CVaR estimates feed into drawdown and leverage checks
3. **Regime monitoring**: Drift detection flags when market conditions diverge from strategy assumptions

## Roadmap

A champion/challenger framework for automated strategy evaluation is planned. This would enable:

- Shadow deployment of strategy variants alongside incumbents
- Statistical comparison of risk-adjusted returns over evaluation windows
- Automated promotion/rollback based on significance testing
