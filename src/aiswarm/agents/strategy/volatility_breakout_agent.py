"""Volatility breakout agent using ATR and Keltner Channels.

Strategy:
  - Price breaks above upper Keltner Channel with expanding ATR → breakout long
  - Price breaks below lower Keltner Channel with expanding ATR → breakout short
  - Confidence scales with ATR expansion ratio and breakout magnitude
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


def _ema(candles: list[OHLCV], period: int) -> float | None:
    if len(candles) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = sum(c.close for c in candles[:period]) / period
    for c in candles[period:]:
        ema = (c.close - ema) * multiplier + ema
    return ema


def _atr(candles: list[OHLCV], period: int = 14) -> float | None:
    """Average True Range."""
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(-period, 0):
        high_low = candles[i].high - candles[i].low
        high_close = abs(candles[i].high - candles[i - 1].close)
        low_close = abs(candles[i].low - candles[i - 1].close)
        trs.append(max(high_low, high_close, low_close))
    return sum(trs) / len(trs)


def _keltner_channels(
    candles: list[OHLCV], ema_period: int = 20, atr_period: int = 14, atr_mult: float = 2.0
) -> tuple[float, float, float] | None:
    """Return (upper, middle, lower) Keltner Channels."""
    middle = _ema(candles, ema_period)
    atr_val = _atr(candles, atr_period)
    if middle is None or atr_val is None:
        return None
    return (middle + atr_mult * atr_val, middle, middle - atr_mult * atr_val)


@register_agent("volatility_breakout")
class VolatilityBreakoutAgent(Agent):
    """Generates breakout signals from Keltner Channel breaks with ATR expansion."""

    def __init__(
        self,
        agent_id: str = "volatility_breakout_agent",
        cluster: str = "strategy",
        ema_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        atr_expansion_threshold: float = 1.2,
        min_candles: int = 30,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.atr_expansion_threshold = atr_expansion_threshold
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

        channels = _keltner_channels(candles, self.ema_period, self.atr_period, self.atr_multiplier)
        current_atr = _atr(candles, self.atr_period)
        # Use older candles for previous ATR
        prev_atr = (
            _atr(candles[:-5], self.atr_period)
            if len(candles) > self.min_candles + 5
            else current_atr
        )
        price = candles[-1].close

        if channels is None or current_atr is None or prev_atr is None:
            return {"signal": None, "reason": "insufficient_data_for_indicators"}

        upper, middle, lower = channels
        atr_expansion = current_atr / prev_atr if prev_atr > 0 else 1.0

        # Check for breakout with volatility expansion
        if price > upper and atr_expansion >= self.atr_expansion_threshold:
            direction = 1
            breakout_magnitude = (price - upper) / current_atr if current_atr > 0 else 0
        elif price < lower and atr_expansion >= self.atr_expansion_threshold:
            direction = -1
            breakout_magnitude = (lower - price) / current_atr if current_atr > 0 else 0
        else:
            return {
                "signal": None,
                "reason": "no_breakout",
                "atr_expansion": round(atr_expansion, 3),
            }

        confidence = min(0.85, 0.40 + breakout_magnitude * 0.15 + (atr_expansion - 1.0) * 0.2)
        confidence = max(0.35, confidence)
        expected_return = breakout_magnitude * current_atr / price if price > 0 else 0

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="volatility_breakout",
            thesis=(
                f"Volatility breakout {direction_str}: price={price:.2f}, "
                f"Keltner=[{lower:.2f},{upper:.2f}], "
                f"ATR expansion={atr_expansion:.2f}x"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=180,
            liquidity_score=0.7,
            regime=MarketRegime.RISK_ON if direction == 1 else MarketRegime.TRANSITION,
            created_at=utc_now(),
            reference_price=price,
        )

        logger.info(
            "Volatility breakout signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "atr_expansion": round(atr_expansion, 3),
                    "breakout_magnitude": round(breakout_magnitude, 4),
                }
            },
        )
        return {"signal": signal, "atr_expansion": atr_expansion, "channels": channels}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
