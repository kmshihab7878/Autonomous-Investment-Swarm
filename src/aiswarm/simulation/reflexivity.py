"""Reflexivity detection — Soros-inspired feedback loop modeling for crypto.

Adapted from ATLAS-GIC's MiroFish reflexivity rules. Identifies five
crypto-specific feedback loops that amplify market moves:

1. Price -> Liquidation Cascade: Large price moves trigger leveraged
   position liquidations, which amplify the move further.
2. P&L -> Forced Behavior: Losses force selling (margin calls, fund
   redemptions), creating supply that drives further losses.
3. Narrative -> Flows: Social media narratives drive retail FOMO/panic,
   which amplifies directional moves.
4. Market -> Policy: Extreme market conditions trigger exchange circuit
   breakers or regulatory response.
5. Reversal Detection: Extended directional moves become reflexive
   extremes, signaling mean-reversion opportunity.

Each detector returns a ReflexivitySignal that the RiskEngine can
consume to adjust position sizing or trigger protective actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

# Thresholds calibrated for crypto market dynamics
LIQUIDATION_CASCADE_THRESHOLD = 0.05  # 5% move triggers cascade risk
FORCED_SELLING_DRAWDOWN = 0.10  # 10% drawdown signals forced behavior
NARRATIVE_STREAK_LENGTH = 5  # 5+ consecutive same-direction candles
POLICY_VOLATILITY_THRESHOLD = 0.08  # 8% daily vol signals policy risk
REVERSAL_STREAK_LENGTH = 7  # 7+ rounds in one direction = reflexive extreme
REVERSAL_MAGNITUDE_THRESHOLD = 0.15  # 15% cumulative move


class FeedbackLoopType(str, Enum):
    LIQUIDATION_CASCADE = "liquidation_cascade"
    FORCED_SELLING = "forced_selling"
    NARRATIVE_FLOWS = "narrative_flows"
    POLICY_RESPONSE = "policy_response"
    REVERSAL_EXTREME = "reversal_extreme"


class ReflexivitySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ReflexivitySignal:
    """A detected feedback loop signal."""

    loop_type: FeedbackLoopType
    severity: ReflexivitySeverity
    description: str
    confidence: float  # 0.0 - 1.0
    suggested_action: str  # "reduce_size", "hedge", "exit", "pause"
    detected_at: datetime
    metadata: dict[str, float]


@dataclass(frozen=True)
class PriceObservation:
    """A single price/market observation for reflexivity analysis."""

    timestamp: datetime
    price: float
    volume: float
    open_interest: float = 0.0  # For futures: total open contracts
    funding_rate: float = 0.0  # Perpetual futures funding rate


class ReflexivityDetector:
    """Detects reflexive feedback loops in crypto market data.

    Maintains a sliding window of price observations and analyzes
    them for patterns consistent with the five feedback loops.
    """

    def __init__(
        self,
        liquidation_threshold: float = LIQUIDATION_CASCADE_THRESHOLD,
        forced_selling_drawdown: float = FORCED_SELLING_DRAWDOWN,
        narrative_streak: int = NARRATIVE_STREAK_LENGTH,
        policy_vol_threshold: float = POLICY_VOLATILITY_THRESHOLD,
        reversal_streak: int = REVERSAL_STREAK_LENGTH,
        reversal_magnitude: float = REVERSAL_MAGNITUDE_THRESHOLD,
        max_window_size: int = 500,
    ) -> None:
        self._observations: list[PriceObservation] = []
        self._liquidation_threshold = liquidation_threshold
        self._forced_selling_drawdown = forced_selling_drawdown
        self._narrative_streak = narrative_streak
        self._policy_vol_threshold = policy_vol_threshold
        self._reversal_streak = reversal_streak
        self._reversal_magnitude = reversal_magnitude
        self._max_window = max_window_size

    def add_observation(self, obs: PriceObservation) -> None:
        """Add a new price observation to the window."""
        self._observations.append(obs)
        if len(self._observations) > self._max_window:
            self._observations = self._observations[-self._max_window :]

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def detect_all(self) -> list[ReflexivitySignal]:
        """Run all five feedback loop detectors. Returns active signals."""
        if len(self._observations) < 3:
            return []

        signals: list[ReflexivitySignal] = []

        liquidation = self._detect_liquidation_cascade()
        if liquidation:
            signals.append(liquidation)

        forced = self._detect_forced_selling()
        if forced:
            signals.append(forced)

        narrative = self._detect_narrative_flows()
        if narrative:
            signals.append(narrative)

        policy = self._detect_policy_response()
        if policy:
            signals.append(policy)

        reversal = self._detect_reversal_extreme()
        if reversal:
            signals.append(reversal)

        if signals:
            logger.info(
                "Reflexivity signals detected",
                extra={
                    "extra_json": {
                        "count": len(signals),
                        "types": [s.loop_type.value for s in signals],
                        "max_severity": max(
                            s.severity.value for s in signals
                        ),
                    }
                },
            )

        return signals

    def _detect_liquidation_cascade(self) -> ReflexivitySignal | None:
        """Detect potential liquidation cascade from rapid price movement.

        Large price moves in leveraged crypto markets trigger cascading
        liquidations. The signal strengthens when open interest is high
        and funding rates are extreme (indicating crowded positioning).
        """
        if len(self._observations) < 2:
            return None

        recent = self._observations[-10:]
        if len(recent) < 2:
            return None

        price_change = (recent[-1].price - recent[0].price) / recent[0].price
        abs_change = abs(price_change)

        if abs_change < self._liquidation_threshold:
            return None

        # Amplify confidence if open interest is available and elevated
        oi_factor = 1.0
        if recent[-1].open_interest > 0 and recent[0].open_interest > 0:
            oi_change = recent[-1].open_interest / recent[0].open_interest
            if oi_change > 1.1:  # OI grew 10%+ → crowded
                oi_factor = 1.3

        # Extreme funding rate indicates crowded positioning
        funding_factor = 1.0
        if abs(recent[-1].funding_rate) > 0.001:  # > 0.1% funding
            funding_factor = 1.2

        raw_confidence = min(1.0, (abs_change / self._liquidation_threshold) * 0.5)
        confidence = min(1.0, raw_confidence * oi_factor * funding_factor)

        severity = self._severity_from_confidence(confidence)
        direction = "downward" if price_change < 0 else "upward"

        return ReflexivitySignal(
            loop_type=FeedbackLoopType.LIQUIDATION_CASCADE,
            severity=severity,
            description=(
                f"Potential {direction} liquidation cascade: "
                f"{abs_change:.1%} move may trigger leveraged position liquidations"
            ),
            confidence=confidence,
            suggested_action="reduce_size" if severity.value in ("low", "medium") else "hedge",
            detected_at=utc_now(),
            metadata={
                "price_change_pct": price_change,
                "open_interest_factor": oi_factor,
                "funding_factor": funding_factor,
            },
        )

    def _detect_forced_selling(self) -> ReflexivitySignal | None:
        """Detect forced selling from sustained drawdown.

        Prolonged drawdowns trigger margin calls, fund redemptions,
        and forced liquidations — creating a self-reinforcing sell cycle.
        """
        if len(self._observations) < 5:
            return None

        prices = np.array([o.price for o in self._observations])
        peak = np.maximum.accumulate(prices)
        drawdown = (peak - prices) / peak
        current_dd = float(drawdown[-1])

        if current_dd < self._forced_selling_drawdown:
            return None

        # Check if drawdown is accelerating (getting worse)
        recent_dd = drawdown[-5:]
        is_accelerating = all(
            recent_dd[i] >= recent_dd[i - 1] for i in range(1, len(recent_dd))
        )

        confidence = min(1.0, current_dd / (self._forced_selling_drawdown * 2))
        if is_accelerating:
            confidence = min(1.0, confidence * 1.3)

        severity = self._severity_from_confidence(confidence)

        return ReflexivitySignal(
            loop_type=FeedbackLoopType.FORCED_SELLING,
            severity=severity,
            description=(
                f"Forced selling pressure: {current_dd:.1%} drawdown from peak, "
                f"{'accelerating' if is_accelerating else 'sustained'}"
            ),
            confidence=confidence,
            suggested_action="reduce_size" if not is_accelerating else "exit",
            detected_at=utc_now(),
            metadata={
                "drawdown_pct": current_dd,
                "is_accelerating": float(is_accelerating),
            },
        )

    def _detect_narrative_flows(self) -> ReflexivitySignal | None:
        """Detect narrative-driven flow patterns from price action streaks.

        In crypto, social media narratives drive retail participation.
        Sustained directional moves indicate narrative capture — FOMO
        on the way up, panic on the way down.
        """
        if len(self._observations) < self._narrative_streak + 1:
            return None

        recent = self._observations[-(self._narrative_streak + 1) :]
        changes = [
            (recent[i + 1].price - recent[i].price) / recent[i].price
            for i in range(len(recent) - 1)
        ]

        # Count consecutive same-direction moves
        positive = all(c > 0 for c in changes)
        negative = all(c < 0 for c in changes)

        if not positive and not negative:
            return None

        cumulative = sum(changes)
        direction = "bullish FOMO" if positive else "bearish panic"

        # Volume trend amplifies confidence
        volumes = [o.volume for o in recent]
        vol_trend = (volumes[-1] / volumes[0]) if volumes[0] > 0 else 1.0

        confidence = min(1.0, 0.5 + abs(cumulative) * 2)
        if vol_trend > 1.5:  # Rising volume confirms narrative
            confidence = min(1.0, confidence * 1.2)

        severity = self._severity_from_confidence(confidence)

        return ReflexivitySignal(
            loop_type=FeedbackLoopType.NARRATIVE_FLOWS,
            severity=severity,
            description=(
                f"Narrative-driven flows: {self._narrative_streak} consecutive "
                f"{'up' if positive else 'down'} periods ({direction})"
            ),
            confidence=confidence,
            suggested_action="reduce_size",
            detected_at=utc_now(),
            metadata={
                "streak_length": float(len(changes)),
                "cumulative_return": cumulative,
                "volume_trend": vol_trend,
                "direction": 1.0 if positive else -1.0,
            },
        )

    def _detect_policy_response(self) -> ReflexivitySignal | None:
        """Detect conditions likely to trigger exchange or regulatory intervention.

        Extreme volatility triggers exchange circuit breakers, and
        prolonged market stress triggers regulatory scrutiny.
        """
        if len(self._observations) < 10:
            return None

        prices = np.array([o.price for o in self._observations[-24:]])
        if len(prices) < 2:
            return None

        returns = np.diff(prices) / prices[:-1]
        daily_vol = float(np.std(returns))

        if daily_vol < self._policy_vol_threshold:
            return None

        confidence = min(1.0, daily_vol / (self._policy_vol_threshold * 2))
        severity = self._severity_from_confidence(confidence)

        return ReflexivitySignal(
            loop_type=FeedbackLoopType.POLICY_RESPONSE,
            severity=severity,
            description=(
                f"Elevated volatility ({daily_vol:.1%}) may trigger "
                f"exchange circuit breakers or regulatory response"
            ),
            confidence=confidence,
            suggested_action="pause" if severity == ReflexivitySeverity.CRITICAL else "reduce_size",
            detected_at=utc_now(),
            metadata={
                "daily_volatility": daily_vol,
                "threshold": self._policy_vol_threshold,
            },
        )

    def _detect_reversal_extreme(self) -> ReflexivitySignal | None:
        """Detect reflexive extremes signaling potential mean reversion.

        Extended directional moves (7+ consecutive periods or 15%+
        cumulative) become self-reinforcing extremes that eventually
        snap back. This is the reflexive reversal signal.
        """
        if len(self._observations) < self._reversal_streak + 1:
            return None

        # Check for extended streak
        recent = self._observations[-(self._reversal_streak + 1) :]
        changes = [
            (recent[i + 1].price - recent[i].price) / recent[i].price
            for i in range(len(recent) - 1)
        ]

        all_positive = all(c > 0 for c in changes)
        all_negative = all(c < 0 for c in changes)

        cumulative = abs(sum(changes))
        has_streak = all_positive or all_negative
        has_magnitude = cumulative >= self._reversal_magnitude

        if not has_streak and not has_magnitude:
            return None

        direction = "upward" if (all_positive or sum(changes) > 0) else "downward"
        reversal_dir = "downward" if direction == "upward" else "upward"

        confidence = 0.4
        if has_streak:
            confidence += 0.3
        if has_magnitude:
            confidence += 0.3
        confidence = min(1.0, confidence)

        severity = self._severity_from_confidence(confidence)

        return ReflexivitySignal(
            loop_type=FeedbackLoopType.REVERSAL_EXTREME,
            severity=severity,
            description=(
                f"Reflexive {direction} extreme: "
                f"{'streak of ' + str(self._reversal_streak) + ' periods' if has_streak else ''}"
                f"{' and ' if has_streak and has_magnitude else ''}"
                f"{f'{cumulative:.1%} cumulative move' if has_magnitude else ''}"
                f" — {reversal_dir} reversal risk elevated"
            ),
            confidence=confidence,
            suggested_action="hedge",
            detected_at=utc_now(),
            metadata={
                "streak_detected": float(has_streak),
                "magnitude_detected": float(has_magnitude),
                "cumulative_return": sum(changes),
                "direction": 1.0 if direction == "upward" else -1.0,
            },
        )

    @staticmethod
    def _severity_from_confidence(confidence: float) -> ReflexivitySeverity:
        """Map confidence to severity level."""
        if confidence >= 0.8:
            return ReflexivitySeverity.CRITICAL
        if confidence >= 0.6:
            return ReflexivitySeverity.HIGH
        if confidence >= 0.4:
            return ReflexivitySeverity.MEDIUM
        return ReflexivitySeverity.LOW

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
