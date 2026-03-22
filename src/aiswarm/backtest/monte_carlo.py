"""Monte Carlo simulation for strategy return analysis.

Bootstraps trade returns from a BacktestResult to generate a distribution
of possible outcomes, providing confidence intervals on key metrics
(total return, max drawdown, Sharpe ratio).

Usage::

    from aiswarm.backtest.monte_carlo import MonteCarloSimulator, MonteCarloConfig

    simulator = MonteCarloSimulator(config=MonteCarloConfig(num_simulations=1000))
    mc_result = simulator.run(backtest_result)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aiswarm.backtest.engine import BacktestResult
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 1000
    seed: int | None = None


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""

    num_simulations: int
    num_trades: int

    # Return distribution
    return_mean: float
    return_median: float
    return_std: float
    return_p5: float  # 5th percentile (worst case)
    return_p25: float
    return_p75: float
    return_p95: float  # 95th percentile (best case)

    # Drawdown distribution
    max_dd_mean: float
    max_dd_median: float
    max_dd_p95: float  # 95th percentile worst drawdown

    # Sharpe distribution
    sharpe_mean: float
    sharpe_median: float
    sharpe_p5: float

    # Win rate distribution
    win_rate_mean: float
    win_rate_std: float

    # Raw simulation data (for custom analysis)
    simulated_returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"\n{'=' * 60}\n"
            f"Monte Carlo Simulation ({self.num_simulations} runs, "
            f"{self.num_trades} trades)\n"
            f"{'=' * 60}\n"
            f"Return: {self.return_mean:+.2f}% mean, "
            f"[{self.return_p5:+.2f}%, {self.return_p95:+.2f}%] 90% CI\n"
            f"Max DD: {self.max_dd_mean:.2f}% mean, "
            f"{self.max_dd_p95:.2f}% worst (95th pct)\n"
            f"Sharpe: {self.sharpe_mean:.3f} mean, "
            f"{self.sharpe_p5:.3f} worst (5th pct)\n"
            f"Win Rate: {self.win_rate_mean:.1f}% +/- {self.win_rate_std:.1f}%\n"
            f"{'=' * 60}\n"
        )


class MonteCarloSimulator:
    """Bootstrap Monte Carlo simulation from backtest trade returns."""

    def __init__(self, config: MonteCarloConfig | None = None) -> None:
        self.config = config or MonteCarloConfig()

    def run(self, backtest_result: BacktestResult) -> MonteCarloResult:
        """Run Monte Carlo simulation from a backtest result.

        Bootstraps trade PnLs and simulates many possible equity paths
        to estimate the distribution of outcomes.

        Args:
            backtest_result: Completed backtest with trade history.

        Returns:
            MonteCarloResult with return/drawdown/Sharpe distributions.

        Raises:
            ValueError: If backtest has no trades.
        """
        trades = backtest_result.trades
        if not trades:
            raise ValueError("Cannot run Monte Carlo with no trades")

        pnls = np.array([t.pnl for t in trades if t.pnl != 0], dtype=np.float64)
        if len(pnls) == 0:
            raise ValueError("No trades with realized PnL")

        rng = np.random.default_rng(self.config.seed)
        initial_capital = backtest_result.initial_capital
        num_trades = len(pnls)

        sim_returns: list[float] = []
        sim_max_dds: list[float] = []
        sim_sharpes: list[float] = []
        sim_win_rates: list[float] = []

        for _ in range(self.config.num_simulations):
            # Bootstrap: sample trades with replacement
            sampled_pnls = rng.choice(pnls, size=num_trades, replace=True)

            # Build equity curve
            equity = initial_capital
            peak = equity
            max_dd = 0.0
            returns_series: list[float] = []
            wins = 0

            for pnl in sampled_pnls:
                prev_equity = equity
                equity += float(pnl)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
                ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
                returns_series.append(ret)
                if pnl > 0:
                    wins += 1

            total_return = (equity - initial_capital) / initial_capital * 100.0
            sim_returns.append(total_return)
            sim_max_dds.append(max_dd * 100.0)
            sim_win_rates.append(wins / num_trades * 100.0 if num_trades > 0 else 0.0)

            # Sharpe from per-trade returns
            arr = np.array(returns_series, dtype=np.float64)
            mean_r = float(np.mean(arr))
            std_r = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            sharpe = mean_r / std_r if std_r > 0 else 0.0
            sim_sharpes.append(sharpe)

        # Compute distribution statistics
        ret_arr = np.array(sim_returns)
        dd_arr = np.array(sim_max_dds)
        sharpe_arr = np.array(sim_sharpes)
        wr_arr = np.array(sim_win_rates)

        result = MonteCarloResult(
            num_simulations=self.config.num_simulations,
            num_trades=num_trades,
            return_mean=round(float(np.mean(ret_arr)), 4),
            return_median=round(float(np.median(ret_arr)), 4),
            return_std=round(float(np.std(ret_arr)), 4),
            return_p5=round(float(np.percentile(ret_arr, 5)), 4),
            return_p25=round(float(np.percentile(ret_arr, 25)), 4),
            return_p75=round(float(np.percentile(ret_arr, 75)), 4),
            return_p95=round(float(np.percentile(ret_arr, 95)), 4),
            max_dd_mean=round(float(np.mean(dd_arr)), 4),
            max_dd_median=round(float(np.median(dd_arr)), 4),
            max_dd_p95=round(float(np.percentile(dd_arr, 95)), 4),
            sharpe_mean=round(float(np.mean(sharpe_arr)), 4),
            sharpe_median=round(float(np.median(sharpe_arr)), 4),
            sharpe_p5=round(float(np.percentile(sharpe_arr, 5)), 4),
            win_rate_mean=round(float(np.mean(wr_arr)), 2),
            win_rate_std=round(float(np.std(wr_arr)), 2),
            simulated_returns=sim_returns,
        )

        logger.info(
            "Monte Carlo simulation completed",
            extra={
                "extra_json": {
                    "simulations": self.config.num_simulations,
                    "trades": num_trades,
                    "mean_return": result.return_mean,
                    "p5_return": result.return_p5,
                    "p95_return": result.return_p95,
                }
            },
        )

        return result
