"""Autoresearch — self-improving agent parameter evolution.

Adapted from ATLAS-GIC's Karpathy-inspired autoresearch loop. Instead
of modifying LLM prompts (ATLAS uses prompt-based agents), AIS modifies
agent constructor parameters since our agents are code-based.

The loop:
1. Identify the worst-performing agent (by rolling Sharpe)
2. Propose a parameter modification (e.g. change fast_period from 20 to 15)
3. Record the modification and start a trial period
4. After N cycles, compare new Sharpe to baseline Sharpe
5. Keep the modification if improved, revert if not

Constraints:
- Each agent has a cooldown period (cannot be modified twice within N cycles)
- Only ONE agent is modified at a time (sequential experimentation)
- Modifications are bounded to prevent degenerate parameters
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from aiswarm.evolution.darwinian import DarwinianWeightManager
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

TRIAL_CYCLES = 5
COOLDOWN_CYCLES = 10
SHARPE_IMPROVEMENT_THRESHOLD = 0.05  # Must improve by at least 5% relative


class ModificationStatus(str, Enum):
    ACTIVE = "active"  # Currently being trialed
    KEPT = "kept"  # Trial succeeded, modification retained
    REVERTED = "reverted"  # Trial failed, modification rolled back


@dataclass
class ParameterBounds:
    """Defines the valid range for a tunable parameter."""

    name: str
    min_value: float
    max_value: float
    step: float  # Minimum modification increment
    current_value: float


@dataclass
class Modification:
    """A single parameter modification being trialed."""

    modification_id: str
    agent_id: str
    parameter: str
    old_value: float
    new_value: float
    baseline_sharpe: float
    status: ModificationStatus
    created_at: datetime
    resolved_at: datetime | None = None
    final_sharpe: float | None = None
    cycles_elapsed: int = 0


@dataclass
class AgentTuningConfig:
    """Tunable parameters for a single agent."""

    agent_id: str
    parameters: list[ParameterBounds]
    last_modified_cycle: int = -999  # Cycle when last modified


# Default tunable parameters per known strategy
DEFAULT_TUNING: dict[str, list[ParameterBounds]] = {
    "momentum_ma_crossover": [
        ParameterBounds("fast_period", 5, 50, 5, 20),
        ParameterBounds("slow_period", 20, 200, 10, 50),
    ],
    "mean_reversion": [
        ParameterBounds("lookback_period", 10, 100, 5, 20),
        ParameterBounds("z_score_threshold", 1.0, 3.0, 0.25, 2.0),
    ],
    "volatility_breakout": [
        ParameterBounds("atr_period", 5, 30, 1, 14),
        ParameterBounds("breakout_multiplier", 1.0, 3.0, 0.25, 1.5),
    ],
    "rsi_divergence": [
        ParameterBounds("rsi_period", 7, 28, 1, 14),
        ParameterBounds("overbought", 65, 85, 5, 70),
        ParameterBounds("oversold", 15, 35, 5, 30),
    ],
    "funding_rate_contrarian": [
        ParameterBounds("threshold", 0.0005, 0.005, 0.0005, 0.001),
    ],
}


class AutoresearchLoop:
    """Self-improving agent parameter evolution.

    Works with DarwinianWeightManager to identify underperforming
    agents, then proposes and trials parameter modifications.
    """

    def __init__(
        self,
        darwinian: DarwinianWeightManager,
        tuning_configs: dict[str, AgentTuningConfig] | None = None,
        trial_cycles: int = TRIAL_CYCLES,
        cooldown_cycles: int = COOLDOWN_CYCLES,
        improvement_threshold: float = SHARPE_IMPROVEMENT_THRESHOLD,
    ) -> None:
        self._darwinian = darwinian
        self._tuning: dict[str, AgentTuningConfig] = tuning_configs or {}
        self._trial_cycles = trial_cycles
        self._cooldown_cycles = cooldown_cycles
        self._improvement_threshold = improvement_threshold

        self._active_modification: Modification | None = None
        self._history: list[Modification] = []
        self._cycle_count = 0
        self._modification_counter = 0

    @property
    def active_modification(self) -> Modification | None:
        return self._active_modification

    @property
    def history(self) -> list[Modification]:
        return list(self._history)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def keep_rate(self) -> float:
        """Fraction of completed modifications that were kept."""
        completed = [m for m in self._history if m.status != ModificationStatus.ACTIVE]
        if not completed:
            return 0.0
        kept = sum(1 for m in completed if m.status == ModificationStatus.KEPT)
        return kept / len(completed)

    def register_agent(
        self,
        agent_id: str,
        strategy: str,
        current_params: dict[str, float] | None = None,
    ) -> None:
        """Register an agent for autoresearch with its tunable parameters.

        If no custom parameters are provided, uses DEFAULT_TUNING for
        the strategy (if available).
        """
        if agent_id in self._tuning:
            return

        defaults = DEFAULT_TUNING.get(strategy, [])
        params = []
        for bound in defaults:
            current = (current_params or {}).get(bound.name, bound.current_value)
            params.append(
                ParameterBounds(
                    name=bound.name,
                    min_value=bound.min_value,
                    max_value=bound.max_value,
                    step=bound.step,
                    current_value=current,
                )
            )

        self._tuning[agent_id] = AgentTuningConfig(
            agent_id=agent_id,
            parameters=params,
        )

    def step(self) -> Modification | None:
        """Run one autoresearch cycle.

        If a trial is active, checks if it should be resolved.
        If no trial is active, identifies the worst agent and proposes
        a modification.

        Returns the current or new Modification, or None if nothing happened.
        """
        self._cycle_count += 1

        # Check active trial
        if self._active_modification is not None:
            self._active_modification.cycles_elapsed += 1
            if self._active_modification.cycles_elapsed >= self._trial_cycles:
                return self._resolve_trial()
            return self._active_modification

        # No active trial — propose new modification
        return self._propose_modification()

    def _propose_modification(self) -> Modification | None:
        """Identify worst agent and propose a parameter change."""
        worst_id = self._darwinian.get_worst_agent()
        if worst_id is None:
            return None

        config = self._tuning.get(worst_id)
        if config is None:
            logger.info(
                "No tuning config for worst agent",
                extra={"extra_json": {"agent_id": worst_id}},
            )
            return None

        # Check cooldown
        if self._cycle_count - config.last_modified_cycle < self._cooldown_cycles:
            logger.info(
                "Agent in cooldown period",
                extra={
                    "extra_json": {
                        "agent_id": worst_id,
                        "cooldown_remaining": (
                            self._cooldown_cycles - (self._cycle_count - config.last_modified_cycle)
                        ),
                    }
                },
            )
            return None

        if not config.parameters:
            return None

        # Select a random parameter to modify
        param = random.choice(config.parameters)

        # Propose a modification: random step in either direction
        direction = random.choice([-1, 1])
        new_value = param.current_value + direction * param.step
        new_value = max(param.min_value, min(param.max_value, new_value))

        if new_value == param.current_value:
            # Try opposite direction
            new_value = param.current_value - direction * param.step
            new_value = max(param.min_value, min(param.max_value, new_value))

        if new_value == param.current_value:
            return None  # Parameter at boundary, can't modify

        # Get baseline Sharpe
        performances = self._darwinian.compute_performance()
        baseline_sharpe = 0.0
        for perf in performances:
            if perf.agent_id == worst_id:
                baseline_sharpe = perf.rolling_sharpe
                break

        self._modification_counter += 1
        mod = Modification(
            modification_id=f"mod_{self._modification_counter:04d}",
            agent_id=worst_id,
            parameter=param.name,
            old_value=param.current_value,
            new_value=new_value,
            baseline_sharpe=baseline_sharpe,
            status=ModificationStatus.ACTIVE,
            created_at=utc_now(),
        )

        self._active_modification = mod
        config.last_modified_cycle = self._cycle_count

        # Apply the modification to the parameter bounds tracker
        param.current_value = new_value

        logger.info(
            "Autoresearch modification proposed",
            extra={
                "extra_json": {
                    "modification_id": mod.modification_id,
                    "agent_id": worst_id,
                    "parameter": param.name,
                    "old_value": mod.old_value,
                    "new_value": mod.new_value,
                    "baseline_sharpe": round(baseline_sharpe, 4),
                }
            },
        )

        return mod

    def _resolve_trial(self) -> Modification:
        """Resolve an active trial: keep or revert based on Sharpe improvement."""
        mod = self._active_modification
        if mod is None:
            raise RuntimeError("_resolve_trial called with no active modification")

        # Get current Sharpe for the modified agent
        performances = self._darwinian.compute_performance()
        current_sharpe = 0.0
        for perf in performances:
            if perf.agent_id == mod.agent_id:
                current_sharpe = perf.rolling_sharpe
                break

        mod.final_sharpe = current_sharpe

        # Determine if improvement meets threshold
        if mod.baseline_sharpe != 0:
            relative_improvement = (current_sharpe - mod.baseline_sharpe) / abs(mod.baseline_sharpe)
        else:
            relative_improvement = current_sharpe  # Any positive Sharpe is improvement from 0

        if relative_improvement >= self._improvement_threshold:
            mod.status = ModificationStatus.KEPT
            logger.info(
                "Autoresearch modification KEPT",
                extra={
                    "extra_json": {
                        "modification_id": mod.modification_id,
                        "agent_id": mod.agent_id,
                        "improvement": round(relative_improvement, 4),
                        "old_sharpe": round(mod.baseline_sharpe, 4),
                        "new_sharpe": round(current_sharpe, 4),
                    }
                },
            )
        else:
            mod.status = ModificationStatus.REVERTED
            # Revert the parameter
            config = self._tuning.get(mod.agent_id)
            if config:
                for param in config.parameters:
                    if param.name == mod.parameter:
                        param.current_value = mod.old_value
                        break
            logger.info(
                "Autoresearch modification REVERTED",
                extra={
                    "extra_json": {
                        "modification_id": mod.modification_id,
                        "agent_id": mod.agent_id,
                        "improvement": round(relative_improvement, 4),
                        "old_sharpe": round(mod.baseline_sharpe, 4),
                        "new_sharpe": round(current_sharpe, 4),
                    }
                },
            )

        mod.resolved_at = utc_now()
        self._history.append(mod)
        self._active_modification = None
        return mod

    def get_current_params(self, agent_id: str) -> dict[str, float]:
        """Get current parameter values for an agent."""
        config = self._tuning.get(agent_id)
        if config is None:
            return {}
        return {p.name: p.current_value for p in config.parameters}

    def to_dict(self) -> dict[str, object]:
        """Serialize state for checkpointing."""
        return {
            "cycle_count": self._cycle_count,
            "modification_counter": self._modification_counter,
            "keep_rate": self.keep_rate,
            "history": [
                {
                    "modification_id": m.modification_id,
                    "agent_id": m.agent_id,
                    "parameter": m.parameter,
                    "old_value": m.old_value,
                    "new_value": m.new_value,
                    "baseline_sharpe": m.baseline_sharpe,
                    "final_sharpe": m.final_sharpe,
                    "status": m.status.value,
                    "cycles_elapsed": m.cycles_elapsed,
                }
                for m in self._history
            ],
            "tuning": {
                agent_id: {
                    "parameters": {p.name: p.current_value for p in config.parameters},
                    "last_modified_cycle": config.last_modified_cycle,
                }
                for agent_id, config in self._tuning.items()
            },
        }
