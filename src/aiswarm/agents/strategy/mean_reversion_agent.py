"""Mean reversion agent using Bollinger Bands + RSI confirmation.

Strategy:
  - Price below lower Bollinger Band + RSI < 30 → oversold → long signal
  - Price above upper Bollinger Band + RSI > 70 → overbought → short signal
  - Confidence scales with distance from band and RSI extremity
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


def _sma(candles: list[OHLCV], period: int) -> float | None:
    if len(candles) < period:
        return None
    return float(sum(c.close for c in candles[-period:]) / period)


def _bollinger_bands(
    candles: list[OHLCV], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float] | None:
    """Return (upper, middle, lower) Bollinger Bands."""
    if len(candles) < period:
        return None
    closes = [c.close for c in candles[-period:]]
    middle = sum(closes) / len(closes)
    variance = sum((c - middle) ** 2 for c in closes) / len(closes)
    std = variance**0.5
    return (middle + num_std * std, middle, middle - num_std * std)


def _rsi(candles: list[OHLCV], period: int = 14) -> float | None:
    """Compute Relative Strength Index."""
    if len(candles) < period + 1:
        return None
    changes = [candles[i].close - candles[i - 1].close for i in range(-period, 0)]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


@register_agent("mean_reversion_bollinger")
class MeanReversionAgent(Agent):
    """Generates mean-reversion signals from Bollinger Bands + RSI."""

    def __init__(
        self,
        agent_id: str = "mean_reversion_agent",
        cluster: str = "strategy",
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        min_candles: int = 30,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
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

        bands = _bollinger_bands(candles, self.bb_period, self.bb_std)
        rsi = _rsi(candles, self.rsi_period)
        price = candles[-1].close

        if bands is None or rsi is None:
            return {"signal": None, "reason": "insufficient_data_for_indicators"}

        upper, middle, lower = bands

        # Oversold: price below lower band + RSI confirms
        if price < lower and rsi < self.rsi_oversold:
            direction = 1
            band_distance = (lower - price) / middle if middle > 0 else 0
            rsi_extremity = (self.rsi_oversold - rsi) / self.rsi_oversold
            confidence = min(0.90, 0.45 + band_distance * 5 + rsi_extremity * 0.2)
            confidence = max(0.35, confidence)
            expected_return = band_distance * 0.5
            regime = MarketRegime.RISK_OFF
        # Overbought: price above upper band + RSI confirms
        elif price > upper and rsi > self.rsi_overbought:
            direction = -1
            band_distance = (price - upper) / middle if middle > 0 else 0
            rsi_extremity = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
            confidence = min(0.90, 0.45 + band_distance * 5 + rsi_extremity * 0.2)
            confidence = max(0.35, confidence)
            expected_return = band_distance * 0.5
            regime = MarketRegime.RISK_ON
        else:
            return {"signal": None, "reason": "no_mean_reversion_setup", "rsi": rsi}

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="mean_reversion_bollinger",
            thesis=(
                f"Mean reversion {direction_str}: price={price:.2f}, "
                f"BB({self.bb_period},{self.bb_std})=[{lower:.2f},{upper:.2f}], "
                f"RSI({self.rsi_period})={rsi:.1f}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=120,
            liquidity_score=0.8,
            regime=regime,
            created_at=utc_now(),
            reference_price=price,
        )

        logger.info(
            "Mean reversion signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "rsi": round(rsi, 2),
                    "band_distance": round(band_distance, 6),
                }
            },
        )
        return {"signal": signal, "rsi": rsi, "bands": bands}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
