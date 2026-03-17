"""Backtesting framework for AIS strategies.

Provides an event-driven backtesting engine that replays historical candle
data through strategy signal generators, simulates order execution with
configurable commission/slippage, and computes comprehensive performance
metrics (Sharpe, Sortino, max drawdown, win rate, profit factor).

Key components:
- BacktestEngine: core simulation loop
- SignalGenerator: protocol for strategy adapters
- MomentumSignalGenerator / FundingRateSignalGenerator: adapters for
  existing AIS agents
- load_candles_from_csv: historical data ingestion
"""

from aiswarm.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestTrade,
    OHLCV,
    SignalGenerator,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "BacktestTrade",
    "OHLCV",
    "SignalGenerator",
]
