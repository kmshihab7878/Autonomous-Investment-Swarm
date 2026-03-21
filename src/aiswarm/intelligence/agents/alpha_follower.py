"""Alpha Follower Agent — generates AIS Signals from top trader behavior.

Monitors the Alpha Intelligence pipeline for recent activity from
high-tier traders and generates signals when they open or add to
positions. Confidence is derived from the trader's track record,
strategy consistency, and recency of the activity.

This agent extends the standard AIS Agent ABC and integrates with
the existing arbitration → risk → execution pipeline.
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import (
    TradeActivity,
    TraderProfile,
    TraderSignal,
    TraderTier,
)
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

# Confidence weights by trader tier
_TIER_CONFIDENCE: dict[TraderTier, float] = {
    TraderTier.ELITE: 0.85,
    TraderTier.STRONG: 0.70,
    TraderTier.NOTABLE: 0.55,
    TraderTier.AVERAGE: 0.40,
    TraderTier.WEAK: 0.25,
}

# Minimum tier to generate a signal
_MIN_TIER = TraderTier.NOTABLE

# Maximum age of activity to consider (seconds)
_MAX_ACTIVITY_AGE_SECONDS = 3600  # 1 hour


class AlphaFollowerAgent(Agent):
    """Generates signals by following top-performing traders.

    Each AIS cycle, this agent:
    1. Queries the AlphaStore for recent activity from top-tier traders
    2. Filters for activity on the target symbol
    3. Computes signal confidence from trader tier + consistency + recency
    4. Emits the strongest trader signal as an AIS Signal

    Parameters:
        store: AlphaStore instance for querying trader intelligence
        min_tier: Minimum trader tier to follow (default: NOTABLE)
        max_activity_age: Maximum age of activity in seconds (default: 3600)
        max_follow_count: Maximum number of trader signals to consider per cycle
    """

    def __init__(
        self,
        store: AlphaStore,
        agent_id: str = "alpha_follower_agent",
        cluster: str = "market_intelligence",
        min_tier: TraderTier = _MIN_TIER,
        max_activity_age: int = _MAX_ACTIVITY_AGE_SECONDS,
        max_follow_count: int = 10,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.store = store
        self.min_tier = min_tier
        self.max_activity_age = max_activity_age
        self.max_follow_count = max_follow_count

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze recent top-trader activity and generate a signal.

        Expects context to contain:
            symbol: str — the symbol to look for trader activity on
            timestamp: datetime (optional) — current time for recency filtering
        """
        symbol = context.get("symbol", "BTCUSDT")
        now = context.get("timestamp", utc_now())

        # Get recent activities for this symbol from top traders
        recent = self.store.get_activities(symbol=symbol, limit=200)
        if not recent:
            return {"signal": None, "reason": "no_recent_activity"}

        # Filter by recency
        cutoff_ts = now.timestamp() - self.max_activity_age
        fresh = [a for a in recent if a.timestamp.timestamp() >= cutoff_ts]
        if not fresh:
            return {"signal": None, "reason": "no_fresh_activity"}

        # Score each activity based on trader profile
        scored_signals: list[TraderSignal] = []
        for activity in fresh[: self.max_follow_count * 3]:
            profile = self.store.get_profile(activity.trader_id)
            if profile is None:
                continue
            if not self._meets_tier_requirement(profile.tier):
                continue

            trader_signal = self._score_activity(activity, profile)
            if trader_signal is not None:
                scored_signals.append(trader_signal)

        if not scored_signals:
            return {"signal": None, "reason": "no_qualifying_traders"}

        # Select the strongest signal
        best = max(scored_signals, key=lambda s: s.trader_confidence)

        # Convert to AIS Signal
        direction = best.direction
        confidence = best.trader_confidence
        direction_str = "long" if direction == 1 else "short"

        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="alpha_follower",
            thesis=(
                f"Following {best.trader_tier.value} trader {best.trader_id[:20]}: "
                f"{direction_str} with {best.strategy_match.value} style. "
                f"{best.reasoning}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=confidence * 0.03,  # Conservative: scale with confidence
            horizon_minutes=240,  # 4 hours default
            liquidity_score=0.7,
            regime=MarketRegime.RISK_ON if direction == 1 else MarketRegime.RISK_OFF,
            created_at=utc_now(),
            reference_price=best.source_activity_id and 0.0 or 0.0,  # Will be overridden
        )

        logger.info(
            "Alpha follower signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "trader_id": best.trader_id[:30],
                    "trader_tier": best.trader_tier.value,
                    "strategy": best.strategy_match.value,
                    "candidates_evaluated": len(scored_signals),
                }
            },
        )

        return {
            "signal": signal,
            "trader_signal": best,
            "candidates_evaluated": len(scored_signals),
        }

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        """Propose action based on analysis."""
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        """Check if we have any recent activity data to work with."""
        symbol = context.get("symbol", "BTCUSDT")
        activities = self.store.get_activities(symbol=symbol, limit=1)
        return len(activities) > 0

    def _meets_tier_requirement(self, tier: TraderTier) -> bool:
        """Check if a trader's tier meets the minimum requirement."""
        tier_order = [
            TraderTier.WEAK,
            TraderTier.AVERAGE,
            TraderTier.NOTABLE,
            TraderTier.STRONG,
            TraderTier.ELITE,
        ]
        return tier_order.index(tier) >= tier_order.index(self.min_tier)

    def _score_activity(
        self,
        activity: TradeActivity,
        profile: TraderProfile,
    ) -> TraderSignal | None:
        """Score a trade activity based on the trader's profile.

        Returns a TraderSignal with computed confidence, or None if
        the activity doesn't warrant a signal.
        """
        # Base confidence from tier
        base_confidence = _TIER_CONFIDENCE.get(profile.tier, 0.3)

        # Consistency bonus (up to +0.10)
        consistency_bonus = profile.consistency_score * 0.10

        # Win rate adjustment (-0.10 to +0.10)
        win_rate_adj = (profile.win_rate - 0.5) * 0.20

        # Sharpe adjustment (0 to +0.05)
        sharpe_adj = min(max(profile.sharpe_ratio, 0) * 0.02, 0.05)

        # Recency decay: more recent = higher confidence
        age_seconds = (utc_now() - activity.timestamp).total_seconds()
        recency_factor = max(0.5, 1.0 - (age_seconds / self.max_activity_age) * 0.5)

        # Final confidence
        confidence = min(
            0.90,
            max(
                0.20,
                (base_confidence + consistency_bonus + win_rate_adj + sharpe_adj) * recency_factor,
            ),
        )

        # Direction: map side to direction
        direction = 1 if activity.side == "BUY" else -1

        # Build reasoning
        reasoning = (
            f"WR={profile.win_rate:.0%}, "
            f"Sharpe={profile.sharpe_ratio:.1f}, "
            f"trades={profile.total_trades}, "
            f"consistency={profile.consistency_score:.0%}"
        )

        return TraderSignal(
            trader_id=activity.trader_id,
            trader_tier=profile.tier,
            symbol=activity.symbol,
            direction=direction,
            trader_confidence=confidence,
            strategy_match=profile.primary_style,
            reasoning=reasoning,
            source_activity_id=activity.activity_id,
            detected_at=utc_now(),
        )
