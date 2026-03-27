"""Tests for autoresearch self-improvement loop."""

from __future__ import annotations

from datetime import timedelta

import pytest

from aiswarm.evolution.autoresearch import (
    AgentTuningConfig,
    AutoresearchLoop,
    ModificationStatus,
    ParameterBounds,
)
from aiswarm.evolution.darwinian import DarwinianWeightManager, TradeOutcome
from aiswarm.utils.time import utc_now


def _make_outcome(
    agent_id: str,
    actual_return: float,
    ts_offset_hours: int = 0,
) -> TradeOutcome:
    return TradeOutcome(
        agent_id=agent_id,
        signal_id=f"sig_{agent_id}_{ts_offset_hours}",
        direction=1 if actual_return > 0 else -1,
        confidence=0.8,
        expected_return=abs(actual_return),
        actual_return=actual_return,
        timestamp=utc_now() - timedelta(hours=ts_offset_hours),
    )


def _make_darwinian_with_data() -> DarwinianWeightManager:
    agents = ["good_agent", "bad_agent"]
    mgr = DarwinianWeightManager(agents, min_observations=3)

    for i in range(10):
        mgr.record_outcome(_make_outcome("good_agent", 0.05, ts_offset_hours=i))
        mgr.record_outcome(_make_outcome("bad_agent", -0.05, ts_offset_hours=i))

    return mgr


class TestAutoresearchLoop:
    def test_register_agent_default_tuning(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        loop.register_agent("a", "momentum_ma_crossover")

        params = loop.get_current_params("a")
        assert "fast_period" in params
        assert "slow_period" in params

    def test_register_agent_custom_params(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        loop.register_agent(
            "a",
            "momentum_ma_crossover",
            current_params={"fast_period": 15, "slow_period": 40},
        )

        params = loop.get_current_params("a")
        assert params["fast_period"] == 15
        assert params["slow_period"] == 40

    def test_register_agent_idempotent(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        loop.register_agent("a", "momentum_ma_crossover", {"fast_period": 15})
        loop.register_agent("a", "momentum_ma_crossover", {"fast_period": 99})

        # Second registration should be ignored
        params = loop.get_current_params("a")
        assert params["fast_period"] == 15

    def test_step_proposes_modification(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=3, cooldown_cycles=5)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        mod = loop.step()
        assert mod is not None
        assert mod.agent_id == "bad_agent"
        assert mod.status == ModificationStatus.ACTIVE

    def test_step_no_tuning_config(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin)
        # Don't register any agents for tuning

        mod = loop.step()
        assert mod is None

    def test_trial_resolves_after_n_cycles(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=3, cooldown_cycles=1)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        # Step 1: propose
        mod = loop.step()
        assert mod is not None
        assert mod.cycles_elapsed == 0

        # Steps 2-3: trial period
        mod = loop.step()
        assert mod.cycles_elapsed == 1
        mod = loop.step()
        assert mod.cycles_elapsed == 2

        # Step 4: resolve (3 cycles elapsed)
        mod = loop.step()
        assert mod.status in (ModificationStatus.KEPT, ModificationStatus.REVERTED)
        assert mod.resolved_at is not None

    def test_cooldown_prevents_immediate_remodification(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=1, cooldown_cycles=5)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        # Step 1: propose + resolve immediately
        loop.step()  # propose
        loop.step()  # resolve after 1 cycle

        # Step 3: should NOT propose (cooldown)
        mod = loop.step()
        assert mod is None

    def test_keep_rate_tracking(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=1, cooldown_cycles=0)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        # Run a few cycles
        for _ in range(10):
            loop.step()

        # keep_rate should be between 0 and 1
        assert 0.0 <= loop.keep_rate <= 1.0

    def test_history_accumulates(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=1, cooldown_cycles=0)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        loop.step()  # propose
        loop.step()  # resolve

        assert len(loop.history) == 1
        assert loop.history[0].status != ModificationStatus.ACTIVE

    def test_to_dict_serialization(self) -> None:
        darwin = _make_darwinian_with_data()
        loop = AutoresearchLoop(darwin, trial_cycles=1, cooldown_cycles=0)
        loop.register_agent("bad_agent", "momentum_ma_crossover")

        loop.step()
        loop.step()

        data = loop.to_dict()
        assert "cycle_count" in data
        assert "history" in data
        assert "tuning" in data

    def test_no_modification_when_no_data(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        loop.register_agent("a", "momentum_ma_crossover")

        mod = loop.step()
        assert mod is None  # No data → worst_agent is None

    def test_cycle_count_increments(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        assert loop.cycle_count == 0
        loop.step()
        assert loop.cycle_count == 1

    def test_unknown_strategy_gets_empty_tuning(self) -> None:
        darwin = DarwinianWeightManager(["a"])
        loop = AutoresearchLoop(darwin)
        loop.register_agent("a", "unknown_strategy_xyz")

        params = loop.get_current_params("a")
        assert params == {}
