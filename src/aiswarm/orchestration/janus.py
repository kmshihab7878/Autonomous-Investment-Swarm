"""JANUS meta-weighting — multi-cohort blending with regime detection.

Adapted from ATLAS-GIC's JANUS layer. Sits above the standard
WeightedArbitration to blend recommendations from multiple
"cohorts" — different weight configurations trained on different
market conditions.

Key features:
- Multiple cohorts (e.g. "recent" trained on last 60 days, "extended"
  trained on 6 months) each with their own agent weight vectors
- Cohort accuracy tracked via hit rate + normalized Sharpe
- Softmax weight blending with min/max constraints
- Regime detection from cohort weight differential
- Disagreement penalty when cohorts disagree on signal direction
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from aiswarm.types.market import Signal
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

MIN_COHORT_WEIGHT = 0.20
MAX_COHORT_WEIGHT = 0.80
ROLLING_WINDOW = 30
REGIME_THRESHOLD = 0.15
DISAGREEMENT_PENALTY = 0.50  # 50% conviction reduction on disagreement


class JanusRegime(str, Enum):
    NOVEL = "novel"  # Recent cohort dominates → market is in new territory
    HISTORICAL = "historical"  # Extended cohort dominates → familiar patterns
    MIXED = "mixed"  # Cohorts are balanced → no clear regime



@dataclass(frozen=True)
class CohortMetrics:
    """Performance metrics for a single cohort."""

    cohort_id: str
    hit_rate: float
    sharpe: float
    combined_score: float  # 0.5 * hit_rate + 0.5 * normalized_sharpe
    weight: float
    signal_count: int


@dataclass
class ScoredOutcome:
    """A historical signal outcome used for cohort scoring."""

    signal_id: str
    cohort_id: str
    symbol: str
    direction: int  # 1 or -1
    confidence: float
    actual_return: float
    timestamp: datetime


@dataclass(frozen=True)
class BlendedSignal:
    """A signal after JANUS cross-cohort blending."""

    symbol: str
    direction: int
    blended_confidence: float
    is_contested: bool  # Cohorts disagree on direction
    contributing_cohorts: tuple[str, ...]
    regime: JanusRegime


class JanusMetaWeighting:
    """Multi-cohort meta-weighting layer.

    Manages multiple cohorts, tracks their accuracy, and blends
    their signal recommendations with regime-aware weighting.
    """

    def __init__(
        self,
        cohort_ids: list[str],
        min_weight: float = MIN_COHORT_WEIGHT,
        max_weight: float = MAX_COHORT_WEIGHT,
        rolling_window: int = ROLLING_WINDOW,
        regime_threshold: float = REGIME_THRESHOLD,
        disagreement_penalty: float = DISAGREEMENT_PENALTY,
    ) -> None:
        if len(cohort_ids) < 2:
            raise ValueError("JANUS requires at least 2 cohorts")

        self._cohort_ids = list(cohort_ids)
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._rolling_window = rolling_window
        self._regime_threshold = regime_threshold
        self._disagreement_penalty = disagreement_penalty

        # Initialize equal weights
        n = len(cohort_ids)
        self._weights: dict[str, float] = {cid: 1.0 / n for cid in cohort_ids}

        # Outcome history per cohort
        self._outcomes: dict[str, list[ScoredOutcome]] = defaultdict(list)

        # Daily history for charting
        self._history: list[dict[str, object]] = []

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def cohort_ids(self) -> list[str]:
        return list(self._cohort_ids)

    def record_outcome(self, outcome: ScoredOutcome) -> None:
        """Record a signal outcome for a specific cohort."""
        self._outcomes[outcome.cohort_id].append(outcome)

    def _cohort_hit_rate(self, cohort_id: str) -> float:
        """Compute hit rate (fraction of correct direction calls) for a cohort."""
        outcomes = self._outcomes.get(cohort_id, [])
        recent = outcomes[-self._rolling_window :]
        if not recent:
            return 0.5  # Neutral prior
        hits = sum(
            1
            for o in recent
            if (o.direction > 0 and o.actual_return > 0)
            or (o.direction < 0 and o.actual_return < 0)
        )
        return hits / len(recent)

    def _cohort_sharpe(self, cohort_id: str) -> float:
        """Compute conviction-weighted Sharpe for a cohort."""
        outcomes = self._outcomes.get(cohort_id, [])
        recent = outcomes[-self._rolling_window :]
        if len(recent) < 2:
            return 0.0
        returns = np.array(
            [o.actual_return * o.confidence for o in recent], dtype=np.float64
        )
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        if std_ret < 1e-10:
            return float("inf") if mean_ret > 0 else 0.0
        return mean_ret / std_ret

    def _softmax_with_constraints(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """Softmax normalization with min/max weight constraints.

        Ensures no cohort has less than min_weight or more than max_weight.
        """
        values = np.array([scores[cid] for cid in self._cohort_ids], dtype=np.float64)

        # Softmax
        exp_values = np.exp(values - np.max(values))  # Numerically stable
        weights = exp_values / np.sum(exp_values)

        # Clamp
        weights = np.clip(weights, self._min_weight, self._max_weight)

        # Re-normalize to sum to 1.0
        weights = weights / np.sum(weights)

        return {cid: float(w) for cid, w in zip(self._cohort_ids, weights)}

    def update_weights(self) -> dict[str, CohortMetrics]:
        """Recalculate cohort weights from recent performance.

        Combined score = 0.5 * hit_rate + 0.5 * normalized_sharpe
        """
        metrics: dict[str, CohortMetrics] = {}

        # Compute raw scores
        sharpe_values: dict[str, float] = {}
        hit_rates: dict[str, float] = {}
        for cid in self._cohort_ids:
            hit_rates[cid] = self._cohort_hit_rate(cid)
            sharpe_values[cid] = self._cohort_sharpe(cid)

        # Normalize Sharpe to [0, 1] range
        all_sharpes = list(sharpe_values.values())
        finite_sharpes = [s for s in all_sharpes if math.isfinite(s)]
        if finite_sharpes:
            min_s = min(finite_sharpes)
            max_s = max(finite_sharpes)
            range_s = max_s - min_s if max_s > min_s else 1.0
        else:
            min_s = 0.0
            range_s = 1.0

        normalized_sharpe: dict[str, float] = {}
        for cid, s in sharpe_values.items():
            if not math.isfinite(s):
                normalized_sharpe[cid] = 1.0  # Inf Sharpe → max score
            else:
                normalized_sharpe[cid] = (s - min_s) / range_s

        # Combined scores
        combined: dict[str, float] = {}
        for cid in self._cohort_ids:
            combined[cid] = 0.5 * hit_rates[cid] + 0.5 * normalized_sharpe[cid]

        # Update weights via constrained softmax
        self._weights = self._softmax_with_constraints(combined)

        for cid in self._cohort_ids:
            metrics[cid] = CohortMetrics(
                cohort_id=cid,
                hit_rate=hit_rates[cid],
                sharpe=sharpe_values[cid],
                combined_score=combined[cid],
                weight=self._weights[cid],
                signal_count=len(self._outcomes.get(cid, [])),
            )

        # Record history
        self._history.append(
            {
                "timestamp": utc_now().isoformat(),
                "weights": dict(self._weights),
                "regime": self.detect_regime().value,
            }
        )

        logger.info(
            "JANUS weights updated",
            extra={
                "extra_json": {
                    "weights": {k: round(v, 4) for k, v in self._weights.items()},
                    "regime": self.detect_regime().value,
                }
            },
        )

        return metrics

    def detect_regime(self) -> JanusRegime:
        """Detect market regime from cohort weight differential.

        If the weight difference between the first and second cohort
        exceeds the threshold, the dominant cohort's regime applies.
        """
        if len(self._cohort_ids) < 2:
            return JanusRegime.MIXED

        w0 = self._weights[self._cohort_ids[0]]
        w1 = self._weights[self._cohort_ids[1]]
        diff = abs(w0 - w1)

        if diff < self._regime_threshold:
            return JanusRegime.MIXED
        if w0 > w1:
            return JanusRegime.NOVEL  # First cohort (typically "recent") dominates
        return JanusRegime.HISTORICAL  # Second cohort (typically "extended") dominates

    def blend_signals(
        self,
        cohort_signals: dict[str, list[Signal]],
    ) -> list[BlendedSignal]:
        """Blend signals from multiple cohorts with weighted averaging.

        For each symbol, merges direction and confidence across cohorts.
        Applies disagreement penalty when cohorts disagree on direction.
        """
        # Group signals by symbol across cohorts
        symbol_signals: dict[str, dict[str, Signal]] = defaultdict(dict)
        for cohort_id, signals in cohort_signals.items():
            for signal in signals:
                symbol_signals[signal.symbol][cohort_id] = signal

        blended: list[BlendedSignal] = []
        regime = self.detect_regime()

        for symbol, cohort_map in symbol_signals.items():
            if not cohort_map:
                continue

            # Weighted direction and confidence
            weighted_direction = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            directions: set[int] = set()
            contributing: list[str] = []

            for cohort_id, signal in cohort_map.items():
                w = self._weights.get(cohort_id, 0.0)
                weighted_direction += signal.direction * signal.confidence * w
                weighted_confidence += signal.confidence * w
                total_weight += w
                directions.add(signal.direction)
                contributing.append(cohort_id)

            if total_weight == 0:
                continue

            # Normalize
            avg_direction = weighted_direction / total_weight
            avg_confidence = weighted_confidence / total_weight

            # Check for disagreement
            is_contested = len(directions) > 1 and 0 not in directions
            if is_contested:
                avg_confidence *= (1 - self._disagreement_penalty)

            # Final direction: sign of weighted average
            final_direction = 1 if avg_direction > 0 else (-1 if avg_direction < 0 else 0)

            blended.append(
                BlendedSignal(
                    symbol=symbol,
                    direction=final_direction,
                    blended_confidence=max(0.0, min(1.0, avg_confidence)),
                    is_contested=is_contested,
                    contributing_cohorts=tuple(contributing),
                    regime=regime,
                )
            )

        return blended

    def get_history(self, days: int = 30) -> list[dict[str, object]]:
        """Return recent JANUS history entries."""
        return self._history[-days:]

    def to_dict(self) -> dict[str, object]:
        """Serialize state for checkpointing."""
        return {
            "cohort_ids": self._cohort_ids,
            "weights": dict(self._weights),
            "history_length": len(self._history),
            "outcome_counts": {
                cid: len(outcomes) for cid, outcomes in self._outcomes.items()
            },
        }
