"""Tests for JANUS meta-weighting layer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aiswarm.orchestration.janus import (
    BlendedSignal,
    CohortMetrics,
    JanusMetaWeighting,
    JanusRegime,
    ScoredOutcome,
)
from aiswarm.types.market import MarketRegime, Signal


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_signal(
    symbol: str = "BTCUSDT",
    direction: int = 1,
    confidence: float = 0.8,
    agent_id: str = "test_agent",
) -> Signal:
    return Signal(
        signal_id=f"sig_{symbol}_{direction}",
        agent_id=agent_id,
        symbol=symbol,
        strategy="test",
        thesis="Test signal for JANUS blending",
        direction=direction,
        confidence=confidence,
        expected_return=0.05,
        horizon_minutes=240,
        liquidity_score=0.8,
        regime=MarketRegime.RISK_ON,
        created_at=_now(),
    )


def _make_outcome(
    cohort_id: str,
    direction: int = 1,
    actual_return: float = 0.05,
) -> ScoredOutcome:
    return ScoredOutcome(
        signal_id=f"sig_{cohort_id}",
        cohort_id=cohort_id,
        symbol="BTCUSDT",
        direction=direction,
        confidence=0.8,
        actual_return=actual_return,
        timestamp=_now(),
    )


class TestJanusMetaWeighting:
    def test_requires_at_least_two_cohorts(self) -> None:
        with pytest.raises(ValueError, match="at least 2 cohorts"):
            JanusMetaWeighting(["only_one"])

    def test_initial_equal_weights(self) -> None:
        janus = JanusMetaWeighting(["recent", "extended"])
        weights = janus.weights
        assert weights["recent"] == pytest.approx(0.5)
        assert weights["extended"] == pytest.approx(0.5)

    def test_three_cohort_equal_weights(self) -> None:
        janus = JanusMetaWeighting(["a", "b", "c"])
        for w in janus.weights.values():
            assert w == pytest.approx(1.0 / 3.0, abs=0.01)

    def test_update_weights_with_data(self) -> None:
        janus = JanusMetaWeighting(["good", "bad"])

        # Good cohort: correct predictions
        for _ in range(20):
            janus.record_outcome(_make_outcome("good", direction=1, actual_return=0.05))
        # Bad cohort: wrong predictions
        for _ in range(20):
            janus.record_outcome(_make_outcome("bad", direction=1, actual_return=-0.05))

        metrics = janus.update_weights()
        assert metrics["good"].weight > metrics["bad"].weight

    def test_weights_sum_to_one(self) -> None:
        janus = JanusMetaWeighting(["a", "b", "c"])
        for _ in range(15):
            janus.record_outcome(_make_outcome("a", actual_return=0.05))
            janus.record_outcome(_make_outcome("b", actual_return=-0.02))
            janus.record_outcome(_make_outcome("c", actual_return=0.01))

        janus.update_weights()
        assert sum(janus.weights.values()) == pytest.approx(1.0, abs=0.01)

    def test_weight_constraints(self) -> None:
        janus = JanusMetaWeighting(["a", "b"], min_weight=0.2, max_weight=0.8)

        # Heavily skew data
        for _ in range(50):
            janus.record_outcome(_make_outcome("a", actual_return=0.10))
            janus.record_outcome(_make_outcome("b", actual_return=-0.10))

        janus.update_weights()
        assert janus.weights["a"] <= 0.85  # May exceed slightly after normalization
        assert janus.weights["b"] >= 0.15

    def test_regime_detection_mixed(self) -> None:
        janus = JanusMetaWeighting(["a", "b"], regime_threshold=0.15)
        assert janus.detect_regime() == JanusRegime.MIXED

    def test_regime_detection_novel(self) -> None:
        janus = JanusMetaWeighting(["recent", "extended"], regime_threshold=0.10)

        # Make recent much better than extended
        for _ in range(30):
            janus.record_outcome(_make_outcome("recent", actual_return=0.08))
            janus.record_outcome(_make_outcome("extended", actual_return=-0.05))

        janus.update_weights()
        regime = janus.detect_regime()
        # Recent should dominate → NOVEL
        assert regime == JanusRegime.NOVEL

    def test_blend_signals_agreeing_cohorts(self) -> None:
        janus = JanusMetaWeighting(["a", "b"])

        signals = {
            "a": [_make_signal("BTCUSDT", direction=1, confidence=0.8)],
            "b": [_make_signal("BTCUSDT", direction=1, confidence=0.7)],
        }

        blended = janus.blend_signals(signals)
        assert len(blended) == 1
        assert blended[0].direction == 1
        assert not blended[0].is_contested
        assert blended[0].blended_confidence > 0

    def test_blend_signals_disagreeing_cohorts(self) -> None:
        janus = JanusMetaWeighting(["a", "b"], disagreement_penalty=0.5)

        signals = {
            "a": [_make_signal("BTCUSDT", direction=1, confidence=0.8)],
            "b": [_make_signal("BTCUSDT", direction=-1, confidence=0.7)],
        }

        blended = janus.blend_signals(signals)
        assert len(blended) == 1
        assert blended[0].is_contested
        # Confidence should be penalized
        assert blended[0].blended_confidence < 0.8

    def test_blend_multiple_symbols(self) -> None:
        janus = JanusMetaWeighting(["a", "b"])

        signals = {
            "a": [
                _make_signal("BTCUSDT", direction=1),
                _make_signal("ETHUSDT", direction=-1),
            ],
            "b": [
                _make_signal("BTCUSDT", direction=1),
            ],
        }

        blended = janus.blend_signals(signals)
        symbols = {b.symbol for b in blended}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    def test_history_records(self) -> None:
        janus = JanusMetaWeighting(["a", "b"])
        janus.update_weights()
        janus.update_weights()

        history = janus.get_history()
        assert len(history) == 2
        assert "weights" in history[0]
        assert "regime" in history[0]

    def test_to_dict_serialization(self) -> None:
        janus = JanusMetaWeighting(["a", "b"])
        janus.record_outcome(_make_outcome("a"))
        data = janus.to_dict()

        assert data["cohort_ids"] == ["a", "b"]
        assert "weights" in data
        assert data["outcome_counts"]["a"] == 1

    def test_empty_signals_returns_empty(self) -> None:
        janus = JanusMetaWeighting(["a", "b"])
        blended = janus.blend_signals({})
        assert blended == []

    def test_cohort_ids_property(self) -> None:
        janus = JanusMetaWeighting(["x", "y", "z"])
        assert janus.cohort_ids == ["x", "y", "z"]
