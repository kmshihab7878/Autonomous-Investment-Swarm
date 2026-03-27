"""Tests for reflexivity feedback loop detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aiswarm.simulation.reflexivity import (
    FeedbackLoopType,
    PriceObservation,
    ReflexivityDetector,
    ReflexivitySeverity,
    ReflexivitySignal,
)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _obs(price: float, volume: float = 1000.0, oi: float = 0.0, fr: float = 0.0) -> PriceObservation:
    return PriceObservation(
        timestamp=_now(),
        price=price,
        volume=volume,
        open_interest=oi,
        funding_rate=fr,
    )


class TestReflexivityDetector:
    def test_no_signals_with_insufficient_data(self) -> None:
        detector = ReflexivityDetector()
        detector.add_observation(_obs(100.0))
        detector.add_observation(_obs(101.0))
        signals = detector.detect_all()
        assert signals == []

    def test_observation_count(self) -> None:
        detector = ReflexivityDetector()
        assert detector.observation_count == 0
        detector.add_observation(_obs(100.0))
        assert detector.observation_count == 1

    def test_clear(self) -> None:
        detector = ReflexivityDetector()
        detector.add_observation(_obs(100.0))
        detector.clear()
        assert detector.observation_count == 0

    def test_max_window_size(self) -> None:
        detector = ReflexivityDetector(max_window_size=5)
        for i in range(10):
            detector.add_observation(_obs(100.0 + i))
        assert detector.observation_count == 5


class TestLiquidationCascade:
    def test_detects_large_price_drop(self) -> None:
        detector = ReflexivityDetector(liquidation_threshold=0.05)
        # 7% drop across 10 observations
        for i in range(10):
            detector.add_observation(_obs(100.0 - i * 0.8))

        signals = detector.detect_all()
        cascade_signals = [
            s for s in signals if s.loop_type == FeedbackLoopType.LIQUIDATION_CASCADE
        ]
        assert len(cascade_signals) >= 1

    def test_no_cascade_on_small_move(self) -> None:
        detector = ReflexivityDetector(liquidation_threshold=0.05)
        # 1% move
        for i in range(10):
            detector.add_observation(_obs(100.0 + i * 0.1))

        signals = detector.detect_all()
        cascade_signals = [
            s for s in signals if s.loop_type == FeedbackLoopType.LIQUIDATION_CASCADE
        ]
        assert len(cascade_signals) == 0

    def test_open_interest_amplifies_confidence(self) -> None:
        detector = ReflexivityDetector(liquidation_threshold=0.05)
        # Large move with rising OI
        for i in range(10):
            detector.add_observation(
                _obs(100.0 - i * 0.8, oi=1000.0 + i * 200)
            )

        signals = detector.detect_all()
        cascade_signals = [
            s for s in signals if s.loop_type == FeedbackLoopType.LIQUIDATION_CASCADE
        ]
        if cascade_signals:
            assert cascade_signals[0].metadata["open_interest_factor"] > 1.0

    def test_extreme_funding_amplifies(self) -> None:
        detector = ReflexivityDetector(liquidation_threshold=0.05)
        for i in range(10):
            detector.add_observation(
                _obs(100.0 - i * 0.8, fr=0.002)
            )

        signals = detector.detect_all()
        cascade_signals = [
            s for s in signals if s.loop_type == FeedbackLoopType.LIQUIDATION_CASCADE
        ]
        if cascade_signals:
            assert cascade_signals[0].metadata["funding_factor"] > 1.0


class TestForcedSelling:
    def test_detects_sustained_drawdown(self) -> None:
        detector = ReflexivityDetector(forced_selling_drawdown=0.10)
        # Peak at 100, then decline to ~85 (15% drawdown)
        prices = [100, 98, 96, 94, 92, 90, 88, 87, 86, 85]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        forced = [
            s for s in signals if s.loop_type == FeedbackLoopType.FORCED_SELLING
        ]
        assert len(forced) >= 1

    def test_no_forced_selling_on_recovery(self) -> None:
        detector = ReflexivityDetector(forced_selling_drawdown=0.10)
        # Small dip then recovery
        prices = [100, 98, 97, 98, 99, 100, 101, 102, 103, 104]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        forced = [
            s for s in signals if s.loop_type == FeedbackLoopType.FORCED_SELLING
        ]
        assert len(forced) == 0


class TestNarrativeFlows:
    def test_detects_bullish_streak(self) -> None:
        detector = ReflexivityDetector(narrative_streak=5)
        # 6 consecutive up moves
        prices = [100, 101, 102, 103, 104, 105, 106]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        narrative = [
            s for s in signals if s.loop_type == FeedbackLoopType.NARRATIVE_FLOWS
        ]
        assert len(narrative) >= 1
        if narrative:
            assert narrative[0].metadata["direction"] == 1.0

    def test_detects_bearish_streak(self) -> None:
        detector = ReflexivityDetector(narrative_streak=5)
        prices = [106, 105, 104, 103, 102, 101, 100]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        narrative = [
            s for s in signals if s.loop_type == FeedbackLoopType.NARRATIVE_FLOWS
        ]
        assert len(narrative) >= 1
        if narrative:
            assert narrative[0].metadata["direction"] == -1.0

    def test_no_narrative_on_mixed_moves(self) -> None:
        detector = ReflexivityDetector(narrative_streak=5)
        prices = [100, 102, 101, 103, 100, 104, 99]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        narrative = [
            s for s in signals if s.loop_type == FeedbackLoopType.NARRATIVE_FLOWS
        ]
        assert len(narrative) == 0


class TestPolicyResponse:
    def test_detects_high_volatility(self) -> None:
        detector = ReflexivityDetector(policy_vol_threshold=0.05)
        # Highly volatile prices
        prices = [100, 110, 95, 115, 85, 120, 80, 125, 75, 130, 70]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        policy = [
            s for s in signals if s.loop_type == FeedbackLoopType.POLICY_RESPONSE
        ]
        assert len(policy) >= 1

    def test_no_policy_on_calm_market(self) -> None:
        detector = ReflexivityDetector(policy_vol_threshold=0.05)
        prices = [100, 100.1, 100.2, 100.1, 100.3, 100.2, 100.4, 100.3, 100.5, 100.4, 100.6]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        policy = [
            s for s in signals if s.loop_type == FeedbackLoopType.POLICY_RESPONSE
        ]
        assert len(policy) == 0


class TestReversalExtreme:
    def test_detects_extended_upward_streak(self) -> None:
        detector = ReflexivityDetector(reversal_streak=7, reversal_magnitude=0.15)
        # 8 consecutive up moves with >15% total
        prices = [100, 103, 106, 109, 112, 115, 118, 121, 124]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        reversal = [
            s for s in signals if s.loop_type == FeedbackLoopType.REVERSAL_EXTREME
        ]
        assert len(reversal) >= 1

    def test_no_reversal_on_short_streak(self) -> None:
        detector = ReflexivityDetector(reversal_streak=7)
        # Only 4 up moves
        prices = [100, 101, 102, 103, 104]
        for p in prices:
            detector.add_observation(_obs(float(p)))

        signals = detector.detect_all()
        reversal = [
            s for s in signals if s.loop_type == FeedbackLoopType.REVERSAL_EXTREME
        ]
        assert len(reversal) == 0


class TestSeverityMapping:
    def test_severity_levels(self) -> None:
        assert ReflexivityDetector._severity_from_confidence(0.9) == ReflexivitySeverity.CRITICAL
        assert ReflexivityDetector._severity_from_confidence(0.7) == ReflexivitySeverity.HIGH
        assert ReflexivityDetector._severity_from_confidence(0.5) == ReflexivitySeverity.MEDIUM
        assert ReflexivityDetector._severity_from_confidence(0.2) == ReflexivitySeverity.LOW
