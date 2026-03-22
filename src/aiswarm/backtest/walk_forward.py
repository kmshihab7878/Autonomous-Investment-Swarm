"""Walk-forward optimization for backtesting strategies.

Splits historical data into rolling train/test windows and runs the
backtest engine on each test window, using the train window for
parameter calibration. Prevents overfitting by enforcing out-of-sample
evaluation at every step.

Usage::

    from aiswarm.backtest.walk_forward import WalkForwardOptimizer, WalkForwardConfig

    optimizer = WalkForwardOptimizer(config=WalkForwardConfig(
        train_bars=500,
        test_bars=100,
        step_bars=100,
    ))
    results = optimizer.run(
        strategy_name="momentum",
        signal_generator=my_generator,
        symbol="BTCUSDT",
        candles=all_candles,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aiswarm.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    OHLCV,
    SignalGenerator,
)
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization."""

    train_bars: int = 500
    test_bars: int = 100
    step_bars: int = 100  # How many bars to advance between windows
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class WalkForwardWindow:
    """Results for a single walk-forward window."""

    window_index: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    test_result: BacktestResult


@dataclass
class WalkForwardResult:
    """Aggregate results from walk-forward optimization."""

    strategy_name: str
    symbol: str
    total_windows: int
    windows: list[WalkForwardWindow]
    aggregate_return_pct: float
    aggregate_sharpe: float
    aggregate_max_drawdown_pct: float
    aggregate_win_rate: float
    aggregate_total_trades: int

    def summary(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"Walk-Forward: {self.strategy_name} on {self.symbol}\n"
            f"Windows: {self.total_windows}\n"
            f"{'=' * 60}\n"
            f"Aggregate Return: {self.aggregate_return_pct:+.2f}%\n"
            f"Aggregate Sharpe: {self.aggregate_sharpe:.3f}\n"
            f"Aggregate Max DD: {self.aggregate_max_drawdown_pct:.2f}%\n"
            f"Aggregate Win Rate: {self.aggregate_win_rate:.1f}%\n"
            f"Total Trades: {self.aggregate_total_trades}\n"
            f"{'=' * 60}\n"
        )


class WalkForwardOptimizer:
    """Walk-forward backtesting with rolling train/test windows."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config = config or WalkForwardConfig()

    def run(
        self,
        strategy_name: str,
        signal_generator: SignalGenerator,
        symbol: str,
        candles: list[OHLCV],
    ) -> WalkForwardResult:
        """Run walk-forward optimization over historical data.

        Args:
            strategy_name: Label for the strategy.
            signal_generator: Signal generator to test.
            symbol: Instrument symbol.
            candles: Full historical candle dataset.

        Returns:
            WalkForwardResult with per-window and aggregate metrics.

        Raises:
            ValueError: If insufficient data for even one window.
        """
        min_bars = self.config.train_bars + self.config.test_bars
        if len(candles) < min_bars:
            raise ValueError(
                f"Need at least {min_bars} candles for walk-forward "
                f"(train={self.config.train_bars} + test={self.config.test_bars}), "
                f"got {len(candles)}"
            )

        engine = BacktestEngine(config=self.config.backtest_config)
        windows: list[WalkForwardWindow] = []
        window_idx = 0
        start = 0

        while start + min_bars <= len(candles):
            train_start = start
            train_end = start + self.config.train_bars
            test_start = train_end
            test_end = min(test_start + self.config.test_bars, len(candles))

            # Run backtest on test window only
            test_candles = candles[test_start:test_end]
            if len(test_candles) < 2:
                break

            result = engine.run(strategy_name, signal_generator, symbol, test_candles)

            windows.append(
                WalkForwardWindow(
                    window_index=window_idx,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    test_result=result,
                )
            )

            logger.info(
                "Walk-forward window completed",
                extra={
                    "extra_json": {
                        "window": window_idx,
                        "test_return": round(result.total_return_pct, 2),
                        "test_trades": result.total_trades,
                    }
                },
            )

            window_idx += 1
            start += self.config.step_bars

        if not windows:
            raise ValueError("No valid walk-forward windows could be created")

        return self._aggregate(strategy_name, symbol, windows)

    def _aggregate(
        self,
        strategy_name: str,
        symbol: str,
        windows: list[WalkForwardWindow],
    ) -> WalkForwardResult:
        """Compute aggregate metrics across all windows."""
        returns = [w.test_result.total_return_pct for w in windows]
        trades = sum(w.test_result.total_trades for w in windows)
        wins = sum(w.test_result.winning_trades for w in windows)

        # Compound returns across windows
        compound = 1.0
        for r in returns:
            compound *= 1.0 + r / 100.0
        agg_return = (compound - 1.0) * 100.0

        # Aggregate Sharpe from per-window returns
        arr = np.array(returns, dtype=np.float64)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        agg_sharpe = mean_ret / std_ret if std_ret > 0 else 0.0

        # Max drawdown across all windows
        agg_max_dd = max((w.test_result.max_drawdown_pct for w in windows), default=0.0)

        # Aggregate win rate
        agg_win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

        return WalkForwardResult(
            strategy_name=strategy_name,
            symbol=symbol,
            total_windows=len(windows),
            windows=windows,
            aggregate_return_pct=round(agg_return, 4),
            aggregate_sharpe=round(agg_sharpe, 4),
            aggregate_max_drawdown_pct=round(agg_max_dd, 4),
            aggregate_win_rate=round(agg_win_rate, 2),
            aggregate_total_trades=trades,
        )
