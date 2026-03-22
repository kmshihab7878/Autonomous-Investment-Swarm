"""Sentiment agent — generates signals from aggregate market sentiment data.

Strategy:
  - Contrarian approach to extreme sentiment readings
  - Extreme fear (< 20) → long signal (market is oversold emotionally)
  - Extreme greed (> 80) → short signal (market is overbought emotionally)
  - Confidence scales with sentiment extremity

This agent consumes pre-fetched sentiment data from context rather than
making external API calls directly, keeping the agent pure and testable.
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import register_agent
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

# Sentiment thresholds (0-100 scale, Fear & Greed style)
EXTREME_FEAR = 20
FEAR = 35
GREED = 65
EXTREME_GREED = 80


@register_agent("sentiment_contrarian")
class SentimentAgent(Agent):
    """Generates contrarian signals based on aggregate market sentiment."""

    def __init__(
        self,
        agent_id: str = "sentiment_agent",
        cluster: str = "market_intelligence",
        extreme_fear: float = EXTREME_FEAR,
        extreme_greed: float = EXTREME_GREED,
        fear: float = FEAR,
        greed: float = GREED,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.extreme_fear = extreme_fear
        self.extreme_greed = extreme_greed
        self.fear = fear
        self.greed = greed

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze sentiment data from context.

        Expects context to contain:
            sentiment_score: float (0-100) — aggregate sentiment index
            symbol: str — the symbol being analyzed
        """
        sentiment_score = context.get("sentiment_score")
        symbol = context.get("symbol", "BTCUSDT")

        if sentiment_score is None:
            return {"signal": None, "reason": "no_sentiment_data"}

        score = float(sentiment_score)

        # Determine signal based on extreme sentiment
        if score <= self.extreme_fear:
            direction = 1  # Contrarian: extreme fear → buy
            level = "extreme_fear"
            extremity = (self.extreme_fear - score) / self.extreme_fear
            confidence = min(0.85, 0.55 + extremity * 0.25)
            regime = MarketRegime.RISK_OFF
        elif score <= self.fear:
            direction = 1  # Moderate fear → weak buy
            level = "fear"
            extremity = (self.fear - score) / (self.fear - self.extreme_fear)
            confidence = min(0.65, 0.40 + extremity * 0.15)
            regime = MarketRegime.TRANSITION
        elif score >= self.extreme_greed:
            direction = -1  # Contrarian: extreme greed → sell
            level = "extreme_greed"
            extremity = (score - self.extreme_greed) / (100 - self.extreme_greed)
            confidence = min(0.85, 0.55 + extremity * 0.25)
            regime = MarketRegime.RISK_ON
        elif score >= self.greed:
            direction = -1  # Moderate greed → weak sell
            level = "greed"
            extremity = (score - self.greed) / (self.extreme_greed - self.greed)
            confidence = min(0.65, 0.40 + extremity * 0.15)
            regime = MarketRegime.RISK_ON
        else:
            return {
                "signal": None,
                "reason": "sentiment_neutral",
                "score": score,
            }

        direction_str = "long" if direction == 1 else "short"
        expected_return = confidence * 0.02

        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="sentiment_contrarian",
            thesis=f"Sentiment {level} ({score:.0f}/100), contrarian {direction_str}",
            direction=direction,
            confidence=max(0.35, confidence),
            expected_return=expected_return,
            horizon_minutes=480,
            liquidity_score=0.8,
            regime=regime,
            created_at=utc_now(),
            reference_price=0.0,
        )

        logger.info(
            "Sentiment signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "score": score,
                    "level": level,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                }
            },
        )
        return {"signal": signal, "score": score, "level": level}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        return context.get("sentiment_score") is not None
