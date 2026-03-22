"""VWAP reversion agent — trades mean-reversion to Volume-Weighted Average Price.

Strategy:
  - Price significantly below VWAP → long (expect reversion up)
  - Price significantly above VWAP → short (expect reversion down)
  - Confidence scales with deviation from VWAP as percentage of price
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


def _vwap(candles: list[OHLCV]) -> float | None:
    """Compute Volume-Weighted Average Price."""
    if not candles:
        return None
    total_volume = sum(c.volume for c in candles)
    if total_volume == 0:
        return None
    typical_price_volume = sum(((c.high + c.low + c.close) / 3.0) * c.volume for c in candles)
    return typical_price_volume / total_volume


@register_agent("vwap_reversion")
class VWAPReversionAgent(Agent):
    """Generates signals when price deviates significantly from VWAP."""

    def __init__(
        self,
        agent_id: str = "vwap_reversion_agent",
        cluster: str = "strategy",
        deviation_threshold: float = 0.015,
        min_candles: int = 20,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.deviation_threshold = deviation_threshold
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

        vwap = _vwap(candles)
        price = candles[-1].close

        if vwap is None or vwap == 0:
            return {"signal": None, "reason": "cannot_compute_vwap"}

        deviation = (price - vwap) / vwap

        if abs(deviation) < self.deviation_threshold:
            return {
                "signal": None,
                "reason": "deviation_below_threshold",
                "deviation": round(deviation, 6),
            }

        # Contrarian: price above VWAP → short, below → long
        direction = -1 if deviation > 0 else 1
        abs_dev = abs(deviation)

        confidence = min(0.85, 0.40 + abs_dev * 10)
        confidence = max(0.35, confidence)
        expected_return = abs_dev * 0.5

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="vwap_reversion",
            thesis=(
                f"VWAP reversion {direction_str}: "
                f"price={price:.2f}, VWAP={vwap:.2f}, "
                f"deviation={deviation:+.4f}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=60,
            liquidity_score=0.85,
            regime=MarketRegime.RISK_ON if abs_dev < 0.03 else MarketRegime.TRANSITION,
            created_at=utc_now(),
            reference_price=price,
        )

        logger.info(
            "VWAP reversion signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "deviation": round(deviation, 6),
                    "vwap": round(vwap, 2),
                }
            },
        )
        return {"signal": signal, "vwap": vwap, "deviation": deviation}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
