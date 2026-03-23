"""Example: Walk-forward backtesting with AIS.

Demonstrates how to run walk-forward optimization on a strategy
using the AIS backtesting engine.

Usage:
    python examples/backtest_walkforward.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aiswarm.backtest.engine import BacktestConfig, OHLCV
from aiswarm.backtest.walk_forward import WalkForwardConfig, WalkForwardOptimizer
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now


class TrendFollowingGenerator:
    """Simple trend-following signal generator for backtesting."""

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        if len(candles) < 20:
            return None

        # Simple: if price is above 20-bar SMA, go long; below, go short
        closes = [c.close for c in candles[-20:]]
        sma = sum(closes) / len(closes)
        price = candles[-1].close

        if current_position is not None:
            # Close position on reversal
            is_long = current_position["side"] == "LONG"
            if (is_long and price < sma) or (not is_long and price > sma):
                direction = -1 if is_long else 1
            else:
                return None
        else:
            direction = 1 if price > sma else -1

        return Signal(
            signal_id=new_id("sig"),
            agent_id="trend_follower",
            symbol=symbol,
            strategy="trend_following",
            thesis=f"Price {'above' if direction == 1 else 'below'} SMA(20)",
            direction=direction,
            confidence=0.6,
            expected_return=0.01,
            horizon_minutes=60,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
            reference_price=price,
        )


def make_sample_candles(n: int = 500) -> list[OHLCV]:
    """Generate synthetic trending price data."""
    import math

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = 100.0
    for i in range(n):
        # Trending with noise
        trend = 0.02 * math.sin(i / 50.0)
        noise = 0.5 * math.sin(i * 7.3)
        price = max(10.0, price + trend + noise)
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
    return candles


if __name__ == "__main__":
    import os

    os.environ["AIS_RISK_HMAC_SECRET"] = "example-secret"

    candles = make_sample_candles(500)
    generator = TrendFollowingGenerator()

    config = WalkForwardConfig(
        train_bars=100,
        test_bars=50,
        step_bars=50,
        backtest_config=BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.1,
            commission_bps=5.0,
            slippage_bps=2.0,
        ),
    )

    optimizer = WalkForwardOptimizer(config=config)
    result = optimizer.run("trend_following", generator, "BTCUSDT", candles)

    print(result.summary())
    print(
        f"Per-window returns: {[round(w.test_result.total_return_pct, 2) for w in result.windows]}"
    )
