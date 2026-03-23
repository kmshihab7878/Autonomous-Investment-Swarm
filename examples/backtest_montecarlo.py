"""Example: Monte Carlo simulation from backtest results.

Demonstrates how to run a Monte Carlo simulation to estimate
the distribution of possible outcomes from a trading strategy.

Usage:
    python examples/backtest_montecarlo.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

from aiswarm.backtest.engine import BacktestConfig, BacktestEngine, OHLCV
from aiswarm.backtest.monte_carlo import MonteCarloConfig, MonteCarloSimulator
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now


class AlternatingGenerator:
    """Alternates buy/sell every N bars for demo purposes."""

    def __init__(self, period: int = 10) -> None:
        self.period = period
        self.bar_count = 0

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        self.bar_count += 1
        if self.bar_count % self.period != 0:
            return None

        direction = -1 if current_position else 1
        return Signal(
            signal_id=new_id("sig"),
            agent_id="alternating",
            symbol=symbol,
            strategy="alternating",
            thesis="Periodic signal for MC demo",
            direction=direction,
            confidence=0.5,
            expected_return=0.005,
            horizon_minutes=60,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
            reference_price=candles[-1].close,
        )


if __name__ == "__main__":
    os.environ["AIS_RISK_HMAC_SECRET"] = "example-secret"

    import math

    # Generate synthetic data
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(300):
        price = max(10.0, price + 0.3 * math.sin(i / 15.0) + 0.1)
        candles.append(
            OHLCV(
                timestamp=base + timedelta(hours=i),
                open=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000.0,
            )
        )

    # Run backtest
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000.0))
    result = engine.run("alternating", AlternatingGenerator(), "BTCUSDT", candles)
    print(result.summary())

    # Run Monte Carlo
    simulator = MonteCarloSimulator(MonteCarloConfig(num_simulations=1000, seed=42))
    mc = simulator.run(result)
    print(mc.summary())
