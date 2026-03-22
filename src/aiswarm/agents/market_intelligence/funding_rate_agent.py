"""Funding rate contrarian agent.

Consumes funding rate data from exchange via ExchangeProvider
and generates contrarian signals when funding is extreme:
  - Extreme positive funding → market is over-leveraged long → contrarian short signal
  - Extreme negative funding → market is over-leveraged short → contrarian long signal

This is a well-known crypto-native alpha source. Extreme funding rates tend
to mean-revert, making contrarian positions profitable on average.
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

# Funding rate thresholds (annualized 8h rate)
EXTREME_THRESHOLD = 0.001  # 0.1% per 8h = ~13.7% annualized
HIGH_THRESHOLD = 0.0005  # 0.05% per 8h = ~6.8% annualized
CONFIDENCE_EXTREME = 0.75
CONFIDENCE_HIGH = 0.55


@register_agent("funding_rate_contrarian")
class FundingRateAgent(Agent):
    """Generates contrarian signals based on extreme funding rates."""

    def __init__(
        self,
        agent_id: str = "funding_rate_agent",
        cluster: str = "market_intelligence",
        extreme_threshold: float = EXTREME_THRESHOLD,
        high_threshold: float = HIGH_THRESHOLD,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.extreme_threshold = extreme_threshold
        self.high_threshold = high_threshold
        self.provider = AsterDataProvider()

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze funding rate data from context.

        Expects context to contain:
            funding_data: dict — raw MCP response from get_funding_rate
            symbol: str — the symbol being analyzed
        """
        raw_funding = context.get("funding_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_funding is None:
            return {"signal": None, "reason": "no_funding_data"}

        funding = self.provider.parse_funding_response(raw_funding)
        if funding is None:
            return {"signal": None, "reason": "parse_failed"}

        rate = funding.funding_rate
        abs_rate = abs(rate)

        # Determine signal strength
        if abs_rate >= self.extreme_threshold:
            confidence = CONFIDENCE_EXTREME
            level = "extreme"
        elif abs_rate >= self.high_threshold:
            confidence = CONFIDENCE_HIGH
            level = "high"
        else:
            return {
                "signal": None,
                "reason": f"funding_rate_normal: {rate:.6f}",
                "funding": funding,
            }

        # Contrarian direction: positive funding → short, negative → long
        direction = -1 if rate > 0 else 1
        direction_str = "short" if direction == -1 else "long"

        # Expected return: proportional to funding rate deviation
        # Assume mean-reversion captures ~50% of the excess funding
        expected_return = abs_rate * 0.5

        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="funding_rate_contrarian",
            thesis=f"Funding rate {level} ({rate:.6f}), contrarian {direction_str} signal",
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=480,  # 8 hours (one funding period)
            liquidity_score=0.8,  # Funding plays are typically in liquid markets
            regime=MarketRegime.RISK_ON if abs_rate < 0.002 else MarketRegime.STRESSED,
            created_at=utc_now(),
            reference_price=funding.mark_price,
        )

        logger.info(
            "Funding rate signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "rate": rate,
                    "direction": direction_str,
                    "confidence": confidence,
                    "level": level,
                }
            },
        )

        return {
            "signal": signal,
            "funding": funding,
            "level": level,
        }

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        """Propose action based on analysis."""
        analysis = self.analyze(context)
        return analysis

    def validate(self, context: dict[str, Any]) -> bool:
        """Validate that we have sufficient data to generate a signal."""
        return context.get("funding_data") is not None
