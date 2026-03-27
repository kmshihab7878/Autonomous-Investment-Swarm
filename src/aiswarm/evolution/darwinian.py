"""Darwinian agent weighting — performance-based weight evolution.

Adapted from ATLAS-GIC's Darwinian selection system. Each agent's weight
evolves based on rolling Sharpe ratio performance:

  - Top quartile agents: weight *= BOOST_FACTOR (1.05)
  - Bottom quartile agents: weight *= DECAY_FACTOR (0.95)
  - Weights clamped to [MIN_WEIGHT, MAX_WEIGHT]

The weight vector feeds directly into WeightedArbitration, giving
higher-performing agents more influence over signal selection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

MIN_WEIGHT = 0.3
MAX_WEIGHT = 2.5
BOOST_FACTOR = 1.05
DECAY_FACTOR = 0.95
DEFAULT_WEIGHT = 1.0
ROLLING_WINDOW_DAYS = 60
MIN_OBSERVATIONS = 5


@dataclass(frozen=True)
class AgentPerformance:
    """Snapshot of an agent's performance metrics."""

    agent_id: str
    rolling_sharpe: float
    hit_rate: float
    total_trades: int
    weight: float
    quartile: int  # 1 = top, 4 = bottom


@dataclass
class TradeOutcome:
    """Recorded outcome of an agent's signal."""

    agent_id: str
    signal_id: str
    direction: int
    confidence: float
    expected_return: float
    actual_return: float
    timestamp: datetime


class DarwinianWeightManager:
    """Manages agent weights using Darwinian selection pressure.

    Tracks per-agent trade outcomes and adjusts weights based on
    rolling Sharpe ratio. Higher-performing agents gain influence;
    lower-performing agents lose it.
    """

    def __init__(
        self,
        agent_ids: list[str],
        initial_weights: dict[str, float] | None = None,
        rolling_window_days: int = ROLLING_WINDOW_DAYS,
        min_observations: int = MIN_OBSERVATIONS,
        boost_factor: float = BOOST_FACTOR,
        decay_factor: float = DECAY_FACTOR,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ) -> None:
        self._weights: dict[str, float] = {}
        for agent_id in agent_ids:
            if initial_weights and agent_id in initial_weights:
                self._weights[agent_id] = initial_weights[agent_id]
            else:
                self._weights[agent_id] = DEFAULT_WEIGHT

        self._outcomes: dict[str, list[TradeOutcome]] = defaultdict(list)
        self._rolling_window = timedelta(days=rolling_window_days)
        self._min_observations = min_observations
        self._boost_factor = boost_factor
        self._decay_factor = decay_factor
        self._min_weight = min_weight
        self._max_weight = max_weight
        self._update_count = 0

    @property
    def weights(self) -> dict[str, float]:
        """Current agent weight vector."""
        return dict(self._weights)

    @property
    def update_count(self) -> int:
        """Number of weight update cycles performed."""
        return self._update_count

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record a trade outcome for an agent."""
        self._outcomes[outcome.agent_id].append(outcome)

    def _prune_old_outcomes(self, cutoff: datetime) -> None:
        """Remove outcomes older than the rolling window."""
        for agent_id in list(self._outcomes):
            self._outcomes[agent_id] = [
                o for o in self._outcomes[agent_id] if o.timestamp >= cutoff
            ]

    def _conviction_weighted_returns(self, outcomes: list[TradeOutcome]) -> np.ndarray:
        """Compute conviction-weighted returns for a set of outcomes.

        Following ATLAS: weighted_return = actual_return * (confidence)
        Direction is already encoded in actual_return sign.
        """
        if not outcomes:
            return np.array([], dtype=np.float64)
        returns = []
        for o in outcomes:
            weighted = o.actual_return * o.confidence
            returns.append(weighted)
        return np.array(returns, dtype=np.float64)

    def _rolling_sharpe(self, agent_id: str, cutoff: datetime) -> float | None:
        """Compute rolling Sharpe ratio for an agent within the window."""
        outcomes = [o for o in self._outcomes.get(agent_id, []) if o.timestamp >= cutoff]
        if len(outcomes) < self._min_observations:
            return None

        returns = self._conviction_weighted_returns(outcomes)
        if len(returns) < 2:
            return None

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns, ddof=1))
        if std_ret < 1e-10:
            return float("inf") if mean_ret > 0 else 0.0
        return mean_ret / std_ret

    def _hit_rate(self, agent_id: str, cutoff: datetime) -> float:
        """Fraction of trades with positive actual return."""
        outcomes = [o for o in self._outcomes.get(agent_id, []) if o.timestamp >= cutoff]
        if not outcomes:
            return 0.0
        wins = sum(1 for o in outcomes if o.actual_return > 0)
        return wins / len(outcomes)

    def compute_performance(self) -> list[AgentPerformance]:
        """Compute current performance metrics for all agents.

        Returns a sorted list (best to worst by Sharpe).
        """
        cutoff = utc_now() - self._rolling_window
        self._prune_old_outcomes(cutoff)

        performances: list[tuple[str, float]] = []
        for agent_id in self._weights:
            sharpe = self._rolling_sharpe(agent_id, cutoff)
            if sharpe is None:
                sharpe = 0.0  # Agents with insufficient data get neutral Sharpe
            performances.append((agent_id, sharpe))

        # Sort descending by Sharpe
        performances.sort(key=lambda x: x[1], reverse=True)

        n = len(performances)
        q_size = max(1, n // 4)

        results: list[AgentPerformance] = []
        for rank, (agent_id, sharpe) in enumerate(performances):
            if rank < q_size:
                quartile = 1
            elif rank >= n - q_size:
                quartile = 4
            else:
                quartile = 2 if rank < n // 2 else 3

            outcomes = [
                o for o in self._outcomes.get(agent_id, []) if o.timestamp >= cutoff
            ]
            results.append(
                AgentPerformance(
                    agent_id=agent_id,
                    rolling_sharpe=sharpe,
                    hit_rate=self._hit_rate(agent_id, cutoff),
                    total_trades=len(outcomes),
                    weight=self._weights[agent_id],
                    quartile=quartile,
                )
            )

        return results

    def update_weights(self) -> dict[str, float]:
        """Run one Darwinian weight update cycle.

        Top quartile agents get boosted, bottom quartile get decayed.
        All weights are clamped to [min_weight, max_weight].

        Returns the updated weight vector.
        """
        performances = self.compute_performance()

        for perf in performances:
            old_weight = self._weights[perf.agent_id]
            if perf.quartile == 1:
                new_weight = old_weight * self._boost_factor
            elif perf.quartile == 4:
                new_weight = old_weight * self._decay_factor
            else:
                new_weight = old_weight  # Middle quartiles unchanged

            # Clamp
            new_weight = max(self._min_weight, min(self._max_weight, new_weight))
            self._weights[perf.agent_id] = new_weight

            if old_weight != new_weight:
                logger.info(
                    "Darwinian weight update",
                    extra={
                        "extra_json": {
                            "agent_id": perf.agent_id,
                            "old_weight": round(old_weight, 4),
                            "new_weight": round(new_weight, 4),
                            "quartile": perf.quartile,
                            "sharpe": round(perf.rolling_sharpe, 4),
                        }
                    },
                )

        self._update_count += 1
        return dict(self._weights)

    def get_worst_agent(self) -> str | None:
        """Return the agent_id with the worst rolling Sharpe.

        Used by the autoresearch loop to identify modification targets.
        Returns None if no agent has enough data.
        """
        cutoff = utc_now() - self._rolling_window
        worst_id: str | None = None
        worst_sharpe = float("inf")

        for agent_id in self._weights:
            sharpe = self._rolling_sharpe(agent_id, cutoff)
            if sharpe is not None and sharpe < worst_sharpe:
                worst_sharpe = sharpe
                worst_id = agent_id

        return worst_id

    def get_weight(self, agent_id: str) -> float:
        """Get current weight for an agent."""
        return self._weights.get(agent_id, DEFAULT_WEIGHT)

    def set_weight(self, agent_id: str, weight: float) -> None:
        """Manually set an agent's weight (e.g. after autoresearch revert)."""
        self._weights[agent_id] = max(self._min_weight, min(self._max_weight, weight))

    def add_agent(self, agent_id: str, weight: float = DEFAULT_WEIGHT) -> None:
        """Register a new agent for tracking."""
        if agent_id not in self._weights:
            self._weights[agent_id] = weight

    def to_dict(self) -> dict[str, object]:
        """Serialize state for checkpointing."""
        return {
            "weights": dict(self._weights),
            "update_count": self._update_count,
            "outcomes": {
                agent_id: [
                    {
                        "agent_id": o.agent_id,
                        "signal_id": o.signal_id,
                        "direction": o.direction,
                        "confidence": o.confidence,
                        "expected_return": o.expected_return,
                        "actual_return": o.actual_return,
                        "timestamp": o.timestamp.isoformat(),
                    }
                    for o in outcomes
                ]
                for agent_id, outcomes in self._outcomes.items()
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        agent_ids: list[str] | None = None,
    ) -> DarwinianWeightManager:
        """Restore from checkpoint data."""
        weights = dict(data.get("weights", {}))  # type: ignore[arg-type]
        ids = agent_ids or list(weights.keys())
        mgr = cls(agent_ids=ids, initial_weights=weights)  # type: ignore[arg-type]
        mgr._update_count = int(data.get("update_count", 0))  # type: ignore[arg-type]

        raw_outcomes = data.get("outcomes", {})
        for agent_id, outcome_list in raw_outcomes.items():  # type: ignore[union-attr]
            for o in outcome_list:  # type: ignore[union-attr]
                mgr._outcomes[agent_id].append(
                    TradeOutcome(
                        agent_id=o["agent_id"],  # type: ignore[index]
                        signal_id=o["signal_id"],  # type: ignore[index]
                        direction=o["direction"],  # type: ignore[index]
                        confidence=o["confidence"],  # type: ignore[index]
                        expected_return=o["expected_return"],  # type: ignore[index]
                        actual_return=o["actual_return"],  # type: ignore[index]
                        timestamp=datetime.fromisoformat(o["timestamp"]),  # type: ignore[index]
                    )
                )
        return mgr
