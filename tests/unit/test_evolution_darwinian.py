"""Tests for Darwinian agent weighting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aiswarm.evolution.darwinian import (
    DEFAULT_WEIGHT,
    AgentPerformance,
    DarwinianWeightManager,
    TradeOutcome,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_outcome(
    agent_id: str,
    actual_return: float,
    confidence: float = 0.8,
    ts_offset_hours: int = 0,
) -> TradeOutcome:
    return TradeOutcome(
        agent_id=agent_id,
        signal_id=f"sig_{agent_id}_{ts_offset_hours}",
        direction=1 if actual_return > 0 else -1,
        confidence=confidence,
        expected_return=abs(actual_return),
        actual_return=actual_return,
        timestamp=_now() - timedelta(hours=ts_offset_hours),
    )


class TestDarwinianWeightManager:
    def test_init_default_weights(self) -> None:
        mgr = DarwinianWeightManager(["agent_a", "agent_b", "agent_c"])
        assert mgr.weights == {
            "agent_a": DEFAULT_WEIGHT,
            "agent_b": DEFAULT_WEIGHT,
            "agent_c": DEFAULT_WEIGHT,
        }

    def test_init_custom_weights(self) -> None:
        mgr = DarwinianWeightManager(
            ["a", "b"],
            initial_weights={"a": 1.5, "b": 0.5},
        )
        assert mgr.weights["a"] == 1.5
        assert mgr.weights["b"] == 0.5

    def test_record_and_count(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        mgr.record_outcome(_make_outcome("a", 0.05))
        mgr.record_outcome(_make_outcome("a", -0.02))
        perfs = mgr.compute_performance()
        assert perfs[0].total_trades == 2

    def test_update_weights_top_quartile_boosted(self) -> None:
        agents = ["top", "mid1", "mid2", "bottom"]
        mgr = DarwinianWeightManager(agents, min_observations=2)

        # Top agent: consistent positive returns
        for i in range(10):
            mgr.record_outcome(_make_outcome("top", 0.05, ts_offset_hours=i))
        # Bottom agent: consistent negative returns
        for i in range(10):
            mgr.record_outcome(_make_outcome("bottom", -0.05, ts_offset_hours=i))
        # Mid agents: mixed
        for i in range(10):
            ret = 0.01 if i % 2 == 0 else -0.01
            mgr.record_outcome(_make_outcome("mid1", ret, ts_offset_hours=i))
            mgr.record_outcome(_make_outcome("mid2", ret * 0.5, ts_offset_hours=i))

        weights = mgr.update_weights()

        assert weights["top"] > DEFAULT_WEIGHT  # Boosted
        assert weights["bottom"] < DEFAULT_WEIGHT  # Decayed

    def test_weights_clamped(self) -> None:
        mgr = DarwinianWeightManager(
            ["a", "b", "c", "d"],
            initial_weights={"a": 2.5, "b": 0.3, "c": 1.0, "d": 1.0},
            min_observations=2,
        )
        for i in range(10):
            mgr.record_outcome(_make_outcome("a", 0.1, ts_offset_hours=i))
            mgr.record_outcome(_make_outcome("b", -0.1, ts_offset_hours=i))
            mgr.record_outcome(_make_outcome("c", 0.01, ts_offset_hours=i))
            mgr.record_outcome(_make_outcome("d", -0.01, ts_offset_hours=i))

        weights = mgr.update_weights()
        assert weights["a"] <= 2.5
        assert weights["b"] >= 0.3

    def test_get_worst_agent(self) -> None:
        mgr = DarwinianWeightManager(["good", "bad"], min_observations=3)
        for i in range(5):
            mgr.record_outcome(_make_outcome("good", 0.05, ts_offset_hours=i))
            mgr.record_outcome(_make_outcome("bad", -0.05, ts_offset_hours=i))

        assert mgr.get_worst_agent() == "bad"

    def test_get_worst_agent_no_data(self) -> None:
        mgr = DarwinianWeightManager(["a", "b"])
        assert mgr.get_worst_agent() is None

    def test_add_agent(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        mgr.add_agent("b", weight=1.5)
        assert mgr.get_weight("b") == 1.5

    def test_add_agent_idempotent(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        mgr.add_agent("a", weight=2.0)
        assert mgr.get_weight("a") == DEFAULT_WEIGHT  # Not overwritten

    def test_set_weight(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        mgr.set_weight("a", 1.8)
        assert mgr.get_weight("a") == 1.8

    def test_set_weight_clamped(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        mgr.set_weight("a", 5.0)
        assert mgr.get_weight("a") == 2.5  # Clamped to MAX_WEIGHT

    def test_serialization_roundtrip(self) -> None:
        mgr = DarwinianWeightManager(["a", "b"])
        mgr.record_outcome(_make_outcome("a", 0.05))
        mgr.record_outcome(_make_outcome("b", -0.03))

        data = mgr.to_dict()
        restored = DarwinianWeightManager.from_dict(data)

        assert restored.weights == mgr.weights

    def test_prune_old_outcomes(self) -> None:
        mgr = DarwinianWeightManager(["a"], rolling_window_days=1, min_observations=1)

        # Old outcome (2 days ago)
        old = TradeOutcome(
            agent_id="a",
            signal_id="old",
            direction=1,
            confidence=0.8,
            expected_return=0.05,
            actual_return=0.05,
            timestamp=_now() - timedelta(days=3),
        )
        mgr.record_outcome(old)
        # Recent outcome
        mgr.record_outcome(_make_outcome("a", 0.02))

        perfs = mgr.compute_performance()
        # After pruning, only 1 recent outcome should remain
        assert perfs[0].total_trades == 1

    def test_update_count_increments(self) -> None:
        mgr = DarwinianWeightManager(["a"])
        assert mgr.update_count == 0
        mgr.update_weights()
        assert mgr.update_count == 1
        mgr.update_weights()
        assert mgr.update_count == 2

    def test_compute_performance_returns_all_agents(self) -> None:
        mgr = DarwinianWeightManager(["a", "b", "c"])
        perfs = mgr.compute_performance()
        assert len(perfs) == 3
        ids = {p.agent_id for p in perfs}
        assert ids == {"a", "b", "c"}

    def test_hit_rate_calculation(self) -> None:
        mgr = DarwinianWeightManager(["a"], min_observations=1)
        # 3 wins, 1 loss
        mgr.record_outcome(_make_outcome("a", 0.05, ts_offset_hours=0))
        mgr.record_outcome(_make_outcome("a", 0.03, ts_offset_hours=1))
        mgr.record_outcome(_make_outcome("a", 0.01, ts_offset_hours=2))
        mgr.record_outcome(_make_outcome("a", -0.02, ts_offset_hours=3))

        perfs = mgr.compute_performance()
        assert perfs[0].hit_rate == pytest.approx(0.75)
