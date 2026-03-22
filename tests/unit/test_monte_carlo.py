"""Tests for Monte Carlo backtesting simulation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aiswarm.backtest.engine import BacktestResult, BacktestTrade
from aiswarm.backtest.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloSimulator,
)

BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_backtest_result(
    pnls: list[float],
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Create a BacktestResult with given trade PnLs."""
    trades = []
    for i, pnl in enumerate(pnls):
        trades.append(
            BacktestTrade(
                timestamp=BASE_TS + timedelta(hours=i),
                symbol="BTCUSDT",
                side="SELL" if pnl != 0 else "BUY",
                price=100.0,
                quantity=1.0,
                notional=100.0,
                signal_confidence=0.6,
                pnl=pnl,
            )
        )

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    final_capital = initial_capital + sum(pnls)

    return BacktestResult(
        strategy_name="test",
        symbol="BTCUSDT",
        start_date=BASE_TS,
        end_date=BASE_TS + timedelta(hours=len(pnls)),
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=(final_capital - initial_capital) / initial_capital * 100,
        sharpe_ratio=0.5,
        sortino_ratio=0.3,
        max_drawdown_pct=5.0,
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(pnls) * 100 if pnls else 0,
        avg_win=sum(wins) / len(wins) if wins else 0,
        avg_loss=sum(losses) / len(losses) if losses else 0,
        profit_factor=sum(wins) / abs(sum(losses)) if losses else float("inf"),
        trades=trades,
        equity_curve=[initial_capital + sum(pnls[:i]) for i in range(len(pnls) + 1)],
    )


class TestMonteCarloSimulator:
    def test_produces_result_with_correct_structure(self) -> None:
        bt = _make_backtest_result([50, -20, 30, -10, 40, -15, 25, -5])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=100, seed=42))
        result = sim.run(bt)

        assert isinstance(result, MonteCarloResult)
        assert result.num_simulations == 100
        assert result.num_trades > 0
        assert len(result.simulated_returns) == 100

    def test_percentiles_ordered_correctly(self) -> None:
        bt = _make_backtest_result([50, -20, 30, -10, 40, -15, 25, -5, 60, -30])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=500, seed=42))
        result = sim.run(bt)

        assert result.return_p5 <= result.return_p25
        assert result.return_p25 <= result.return_median
        assert result.return_median <= result.return_p75
        assert result.return_p75 <= result.return_p95

    def test_all_positive_pnl_gives_positive_mean_return(self) -> None:
        bt = _make_backtest_result([50, 30, 40, 20, 60])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=200, seed=42))
        result = sim.run(bt)

        assert result.return_mean > 0
        assert result.win_rate_mean > 90

    def test_all_negative_pnl_gives_negative_mean_return(self) -> None:
        bt = _make_backtest_result([-50, -30, -40, -20, -60])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=200, seed=42))
        result = sim.run(bt)

        assert result.return_mean < 0

    def test_no_trades_raises(self) -> None:
        bt = _make_backtest_result([])
        bt.trades = []
        sim = MonteCarloSimulator()

        with pytest.raises(ValueError, match="no trades"):
            sim.run(bt)

    def test_seed_produces_deterministic_results(self) -> None:
        bt = _make_backtest_result([50, -20, 30, -10])
        sim1 = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, seed=123))
        sim2 = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, seed=123))

        r1 = sim1.run(bt)
        r2 = sim2.run(bt)

        assert r1.return_mean == r2.return_mean
        assert r1.simulated_returns == r2.simulated_returns

    def test_max_drawdown_is_non_negative(self) -> None:
        bt = _make_backtest_result([50, -20, 30, -10, 40])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=100, seed=42))
        result = sim.run(bt)

        assert result.max_dd_mean >= 0
        assert result.max_dd_p95 >= 0

    def test_summary_returns_readable_string(self) -> None:
        bt = _make_backtest_result([50, -20, 30])
        sim = MonteCarloSimulator(MonteCarloConfig(num_simulations=50, seed=42))
        result = sim.run(bt)
        summary = result.summary()

        assert "Monte Carlo" in summary
        assert "Return" in summary
        assert "Sharpe" in summary
