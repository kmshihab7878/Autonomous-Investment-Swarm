"""Simple momentum agent.

Consumes OHLCV candle data from exchange via ExchangeProvider
and generates momentum signals based on moving average crossovers
and price position relative to moving averages.

Strategy:
  - Price above N-period SMA → bullish momentum → long signal
  - Price below N-period SMA → bearish momentum → short signal
  - Confidence scales with distance from MA and trend consistency
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
    """Compute simple moving average of close prices."""
    if len(candles) < period:
        return None
    closes = [c.close for c in candles[-period:]]
    return float(sum(closes) / len(closes))


def _trend_consistency(candles: list[OHLCV], period: int) -> float:
    """Measure what fraction of recent candles closed in the trend direction.

    Returns a value between 0.0 (no consistency) and 1.0 (perfectly consistent).
    """
    if len(candles) < period:
        return 0.5
    recent = candles[-period:]
    up_closes = sum(1 for c in recent if c.close >= c.open)
    return up_closes / len(recent)


@register_agent("momentum_ma_crossover")
class MomentumAgent(Agent):
    """Generates momentum signals from OHLCV price data."""

    def __init__(
        self,
        agent_id: str = "momentum_agent",
        cluster: str = "strategy",
        fast_period: int = 20,
        slow_period: int = 50,
        min_candles: int = 50,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.min_candles = min_candles
        self.provider = AsterDataProvider()

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze OHLCV data and generate momentum signal.

        Expects context to contain:
            klines_data: list[dict] | dict — raw MCP response from get_klines
            symbol: str — the symbol being analyzed
        """
        raw_klines = context.get("klines_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_klines is None:
            return {"signal": None, "reason": "no_klines_data"}

        candles = self.provider.parse_klines(raw_klines, symbol)
        if len(candles) < self.min_candles:
            return {
                "signal": None,
                "reason": f"insufficient_data: {len(candles)} < {self.min_candles}",
            }

        # Compute moving averages
        fast_ma = _sma(candles, self.fast_period)
        slow_ma = _sma(candles, self.slow_period)
        current_price = candles[-1].close

        if fast_ma is None or slow_ma is None:
            return {"signal": None, "reason": "insufficient_data_for_ma"}

        # Determine direction
        if fast_ma > slow_ma and current_price > fast_ma:
            direction = 1  # Bullish
        elif fast_ma < slow_ma and current_price < fast_ma:
            direction = -1  # Bearish
        else:
            return {
                "signal": None,
                "reason": "no_clear_momentum",
                "fast_ma": fast_ma,
                "slow_ma": slow_ma,
                "price": current_price,
            }

        # Confidence based on:
        # 1. Distance of fast MA from slow MA (stronger trend = higher confidence)
        # 2. Trend consistency (more consistent = higher confidence)
        ma_spread = abs(fast_ma - slow_ma) / slow_ma
        consistency = _trend_consistency(candles, self.fast_period)

        # Base confidence 0.4, scale up with spread and consistency
        confidence = min(0.90, 0.40 + ma_spread * 10 + (consistency - 0.5) * 0.3)
        confidence = max(0.35, confidence)

        # Expected return: proportional to momentum strength
        expected_return = ma_spread * 0.3  # Conservative estimate

        # Determine regime
        if ma_spread > 0.02:
            regime = MarketRegime.RISK_ON
        elif ma_spread < 0.005:
            regime = MarketRegime.TRANSITION
        else:
            regime = MarketRegime.RISK_ON

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="momentum_ma_crossover",
            thesis=(
                f"Momentum {direction_str}: price={current_price:.2f}, "
                f"fast_ma({self.fast_period})={fast_ma:.2f}, "
                f"slow_ma({self.slow_period})={slow_ma:.2f}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=240,  # 4 hours
            liquidity_score=0.8,
            regime=regime,
            created_at=utc_now(),
            reference_price=current_price,
        )

        logger.info(
            "Momentum signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "ma_spread": round(ma_spread, 6),
                    "consistency": round(consistency, 4),
                }
            },
        )

        return {
            "signal": signal,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "price": current_price,
            "ma_spread": ma_spread,
            "consistency": consistency,
        }

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        """Propose action based on analysis."""
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        """Validate that we have sufficient data."""
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
