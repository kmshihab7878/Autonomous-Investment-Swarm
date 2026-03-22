"""Pairs trading (statistical arbitrage) agent.

Strategy:
  - Tracks the spread between two correlated instruments
  - Long spread (buy A, sell B) when z-score < -threshold
  - Short spread (sell A, buy B) when z-score > +threshold
  - Expects mean-reversion of the spread
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import register_agent
from aiswarm.data.providers.aster import AsterDataProvider
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


def _compute_spread_zscore(
    prices_a: list[float], prices_b: list[float], lookback: int
) -> float | None:
    """Compute z-score of the price ratio spread."""
    if len(prices_a) < lookback or len(prices_b) < lookback:
        return None
    ratios: list[float] = [
        a / b for a, b in zip(prices_a[-lookback:], prices_b[-lookback:]) if b != 0
    ]
    if len(ratios) < lookback:
        return None
    mean: float = sum(ratios) / len(ratios)
    variance: float = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    std: float = variance**0.5
    if std == 0:
        return 0.0
    current_ratio: float = ratios[-1]
    return (current_ratio - mean) / std


@register_agent("pairs_stat_arb")
class PairsAgent(Agent):
    """Generates statistical arbitrage signals from correlated pair spreads.

    Requires context to contain klines for BOTH the primary symbol and a
    paired symbol. The pair symbol is configured via ``pair_symbol``.
    """

    def __init__(
        self,
        agent_id: str = "pairs_agent",
        cluster: str = "strategy",
        pair_symbol: str = "ETHUSDT",
        zscore_threshold: float = 2.0,
        lookback: int = 50,
        min_candles: int = 50,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.pair_symbol = pair_symbol
        self.zscore_threshold = zscore_threshold
        self.lookback = lookback
        self.min_candles = min_candles
        self.provider = AsterDataProvider()

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_klines = context.get("klines_data")
        raw_pair_klines = context.get("pair_klines_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_klines is None:
            return {"signal": None, "reason": "no_klines_data"}
        if raw_pair_klines is None:
            return {"signal": None, "reason": "no_pair_klines_data"}

        candles_a = self.provider.parse_klines(raw_klines, symbol)
        candles_b = self.provider.parse_klines(raw_pair_klines, self.pair_symbol)

        if len(candles_a) < self.min_candles or len(candles_b) < self.min_candles:
            return {"signal": None, "reason": "insufficient_data"}

        prices_a = [c.close for c in candles_a]
        prices_b = [c.close for c in candles_b]

        zscore = _compute_spread_zscore(prices_a, prices_b, self.lookback)
        if zscore is None:
            return {"signal": None, "reason": "cannot_compute_zscore"}

        if abs(zscore) < self.zscore_threshold:
            return {
                "signal": None,
                "reason": "zscore_below_threshold",
                "zscore": round(zscore, 3),
            }

        # Long spread: buy A when z-score is very negative (A is cheap relative to B)
        direction = 1 if zscore < 0 else -1
        abs_z = abs(zscore)

        confidence = min(0.85, 0.40 + (abs_z - self.zscore_threshold) * 0.15)
        confidence = max(0.35, confidence)
        expected_return = (abs_z - self.zscore_threshold) * 0.005

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="pairs_stat_arb",
            thesis=(
                f"Pairs {direction_str} {symbol}/{self.pair_symbol}: "
                f"z-score={zscore:+.2f}, threshold={self.zscore_threshold}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=360,
            liquidity_score=0.75,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
            reference_price=candles_a[-1].close,
        )

        logger.info(
            "Pairs trading signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "pair": self.pair_symbol,
                    "direction": direction_str,
                    "zscore": round(zscore, 3),
                    "confidence": round(confidence, 4),
                }
            },
        )
        return {"signal": signal, "zscore": zscore, "pair_symbol": self.pair_symbol}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        return (
            context.get("klines_data") is not None and context.get("pair_klines_data") is not None
        )
