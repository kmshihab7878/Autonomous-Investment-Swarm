"""RSI divergence agent — detects price/RSI divergences for reversal signals.

Strategy:
  - Bullish divergence: price makes lower low but RSI makes higher low → long
  - Bearish divergence: price makes higher high but RSI makes lower high → short
  - Confidence scales with divergence magnitude
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import register_agent
from aiswarm.data.providers.aster import AsterDataProvider, OHLCV
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


def _rsi_series(candles: list[OHLCV], period: int = 14) -> list[float]:
    """Compute RSI for each bar (returns list aligned with candles[period:])."""
    if len(candles) < period + 1:
        return []
    result: list[float] = []
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = candles[i].close - candles[i - 1].close
        if change > 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period + 1, len(candles)):
        change = candles[i].close - candles[i - 1].close
        if change > 0:
            avg_gain = (avg_gain * (period - 1) + change) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - change) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            rs = avg_gain / avg_loss
            result.append(100.0 - 100.0 / (1.0 + rs))
    return result


@register_agent("rsi_divergence")
class RSIDivergenceAgent(Agent):
    """Generates reversal signals from RSI/price divergences."""

    def __init__(
        self,
        agent_id: str = "rsi_divergence_agent",
        cluster: str = "strategy",
        rsi_period: int = 14,
        lookback: int = 20,
        min_candles: int = 50,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.min_candles = min_candles
        self.provider = AsterDataProvider()

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_klines = context.get("klines_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_klines is None:
            return {"signal": None, "reason": "no_klines_data"}

        candles = self.provider.parse_klines(raw_klines, symbol)
        if len(candles) < self.min_candles:
            return {"signal": None, "reason": f"insufficient_data: {len(candles)}"}

        rsi_vals = _rsi_series(candles, self.rsi_period)
        if len(rsi_vals) < self.lookback:
            return {"signal": None, "reason": "insufficient_rsi_data"}

        # Align price and RSI (RSI starts at candles[rsi_period])
        offset = self.rsi_period
        recent_prices = [c.close for c in candles[offset:]]
        recent_rsi = rsi_vals

        n = min(len(recent_prices), len(recent_rsi), self.lookback)
        prices_window = recent_prices[-n:]
        rsi_window = recent_rsi[-n:]

        price_now = prices_window[-1]
        rsi_now = rsi_window[-1]
        price_min_idx = prices_window.index(min(prices_window))
        price_max_idx = prices_window.index(max(prices_window))

        # Bullish divergence: price lower low, RSI higher low
        if price_min_idx > n // 2 and price_now <= prices_window[price_min_idx] * 1.01:
            early_min_rsi = min(rsi_window[: n // 2])
            late_min_rsi = min(rsi_window[n // 2 :])
            if late_min_rsi > early_min_rsi + 3:
                divergence = late_min_rsi - early_min_rsi
                confidence = min(0.80, 0.40 + divergence * 0.02)
                signal = self._build_signal(symbol, 1, confidence, divergence, rsi_now, price_now)
                return {"signal": signal, "divergence": "bullish", "magnitude": divergence}

        # Bearish divergence: price higher high, RSI lower high
        if price_max_idx > n // 2 and price_now >= prices_window[price_max_idx] * 0.99:
            early_max_rsi = max(rsi_window[: n // 2])
            late_max_rsi = max(rsi_window[n // 2 :])
            if late_max_rsi < early_max_rsi - 3:
                divergence = early_max_rsi - late_max_rsi
                confidence = min(0.80, 0.40 + divergence * 0.02)
                signal = self._build_signal(symbol, -1, confidence, divergence, rsi_now, price_now)
                return {"signal": signal, "divergence": "bearish", "magnitude": divergence}

        return {"signal": None, "reason": "no_divergence_detected", "rsi": rsi_now}

    def _build_signal(
        self,
        symbol: str,
        direction: int,
        confidence: float,
        divergence: float,
        rsi: float,
        price: float,
    ) -> Signal:
        direction_str = "long" if direction == 1 else "short"
        return Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="rsi_divergence",
            thesis=(
                f"RSI divergence {direction_str}: "
                f"RSI={rsi:.1f}, divergence={divergence:.1f} pts, "
                f"price={price:.2f}"
            ),
            direction=direction,
            confidence=max(0.35, confidence),
            expected_return=divergence * 0.002,
            horizon_minutes=240,
            liquidity_score=0.75,
            regime=MarketRegime.TRANSITION,
            created_at=utc_now(),
            reference_price=price,
        )

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
