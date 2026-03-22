"""Tests for slippage estimation models used in execution and backtesting."""

from __future__ import annotations

import math

import pytest

from aiswarm.execution.slippage import (
    CompositeSlippage,
    FixedSlippage,
    HistoricalSlippage,
    RegimeAwareSlippage,
    SlippageEstimate,
    VolumeWeightedSlippage,
    default_slippage_model,
)


# ---------------------------------------------------------------------------
# Tests: FixedSlippage
# ---------------------------------------------------------------------------


class TestFixedSlippage:
    def test_always_returns_configured_bps(self) -> None:
        """FixedSlippage returns the same bps regardless of notional."""
        # Arrange
        model = FixedSlippage(bps=7.5)

        # Act
        est_small = model.estimate_bps(notional=100.0)
        est_large = model.estimate_bps(notional=10_000_000.0)

        # Assert
        assert est_small.bps == pytest.approx(7.5)
        assert est_large.bps == pytest.approx(7.5)
        assert est_small.model_name == "fixed"

    def test_default_bps_is_five(self) -> None:
        """Default fixed slippage is 5 bps."""
        # Arrange & Act
        model = FixedSlippage()
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(5.0)

    def test_details_contain_fixed_bps(self) -> None:
        """Details dict reports the fixed bps value."""
        # Arrange
        model = FixedSlippage(bps=3.0)

        # Act
        est = model.estimate_bps(notional=500.0)

        # Assert
        assert est.details["fixed_bps"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tests: VolumeWeightedSlippage
# ---------------------------------------------------------------------------


class TestVolumeWeightedSlippage:
    def test_small_order_low_slippage(self) -> None:
        """A small order relative to deep book has low slippage near base_bps."""
        # Arrange — $100 order against $1M book
        model = VolumeWeightedSlippage(
            base_bps=1.0, impact_coefficient=10.0, min_bps=0.5, max_bps=50.0
        )

        # Act
        est = model.estimate_bps(notional=100.0, orderbook_depth=1_000_000.0)

        # Assert — participation_rate = 0.0001, impact = 10 * sqrt(0.0001) = 0.1
        expected = 1.0 + 10.0 * math.sqrt(100.0 / 1_000_000.0)
        assert est.bps == pytest.approx(expected)
        assert est.bps < 5.0  # Definitely low

    def test_large_order_high_slippage(self) -> None:
        """A large order relative to thin book has high slippage."""
        # Arrange — $500k order against $100k book
        model = VolumeWeightedSlippage(
            base_bps=1.0, impact_coefficient=10.0, min_bps=0.5, max_bps=50.0
        )

        # Act
        est = model.estimate_bps(notional=500_000.0, orderbook_depth=100_000.0)

        # Assert — participation_rate = 5.0, impact = 10 * sqrt(5) = ~22.36
        expected = min(50.0, max(0.5, 1.0 + 10.0 * math.sqrt(5.0)))
        assert est.bps == pytest.approx(expected)
        assert est.bps > 20.0  # High slippage

    def test_zero_depth_returns_max_bps(self) -> None:
        """When orderbook depth is zero, max_bps is returned."""
        # Arrange
        model = VolumeWeightedSlippage(max_bps=42.0)

        # Act
        est = model.estimate_bps(notional=1000.0, orderbook_depth=0.0)

        # Assert
        assert est.bps == pytest.approx(42.0)
        assert est.details["reason"] == "zero_depth"

    def test_negative_depth_returns_max_bps(self) -> None:
        """Negative orderbook depth is treated as zero (returns max_bps)."""
        # Arrange
        model = VolumeWeightedSlippage(max_bps=42.0)

        # Act
        est = model.estimate_bps(notional=1000.0, orderbook_depth=-100.0)

        # Assert
        assert est.bps == pytest.approx(42.0)

    def test_respects_min_bound(self) -> None:
        """Result is clamped to min_bps even if computed value is lower."""
        # Arrange — tiny order, massive book, high min
        model = VolumeWeightedSlippage(
            base_bps=0.0, impact_coefficient=0.001, min_bps=3.0, max_bps=50.0
        )

        # Act
        est = model.estimate_bps(notional=1.0, orderbook_depth=100_000_000.0)

        # Assert
        assert est.bps == pytest.approx(3.0)

    def test_respects_max_bound(self) -> None:
        """Result is clamped to max_bps even if computed value is higher."""
        # Arrange — huge order, tiny book
        model = VolumeWeightedSlippage(
            base_bps=100.0, impact_coefficient=100.0, min_bps=0.5, max_bps=25.0
        )

        # Act
        est = model.estimate_bps(notional=1_000_000.0, orderbook_depth=1.0)

        # Assert
        assert est.bps == pytest.approx(25.0)

    def test_model_name_is_volume_weighted(self) -> None:
        """Model name is reported correctly."""
        # Arrange & Act
        model = VolumeWeightedSlippage()
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.model_name == "volume_weighted"

    def test_default_depth_when_not_provided(self) -> None:
        """When orderbook_depth is not passed, default of 1M is used."""
        # Arrange
        model = VolumeWeightedSlippage(base_bps=1.0, impact_coefficient=10.0)

        # Act
        est = model.estimate_bps(notional=10_000.0)

        # Assert — participation_rate = 10000/1000000 = 0.01
        expected = 1.0 + 10.0 * math.sqrt(0.01)
        assert est.bps == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: HistoricalSlippage
# ---------------------------------------------------------------------------


class TestHistoricalSlippage:
    def test_returns_default_when_no_samples(self) -> None:
        """Without any recorded fills, returns default_bps."""
        # Arrange
        model = HistoricalSlippage(default_bps=8.0, min_samples=5)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(8.0)
        assert est.details["reason"] == "insufficient_samples"
        assert est.model_name == "historical"

    def test_returns_default_when_insufficient_samples(self) -> None:
        """With fewer samples than min_samples, returns default_bps."""
        # Arrange
        model = HistoricalSlippage(default_bps=5.0, min_samples=10)
        for i in range(9):
            model.record_fill(reference_price=100.0, fill_price=100.05, side=1)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(5.0)
        assert est.details["samples"] == 9

    def test_calibrates_after_sufficient_samples(self) -> None:
        """After min_samples fills, uses EWMA-calibrated estimate."""
        # Arrange — 10 fills at 5 bps slippage each (buy at 100.05, ref 100.0)
        model = HistoricalSlippage(default_bps=10.0, min_samples=10, ewma_alpha=0.1)
        for _ in range(10):
            model.record_fill(reference_price=100.0, fill_price=100.05, side=1)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert — should be approximately 5 bps (100.05 - 100.0) / 100.0 * 10000
        assert est.bps == pytest.approx(5.0, abs=0.5)
        assert est.details["samples"] == 10

    def test_ewma_weighting_recent_fills_more(self) -> None:
        """EWMA gives recent fills more influence than old fills."""
        # Arrange — start with 10 fills at 5 bps, then 10 fills at 20 bps
        model = HistoricalSlippage(default_bps=10.0, min_samples=5, ewma_alpha=0.3)

        # Phase 1: low slippage fills
        for _ in range(10):
            model.record_fill(reference_price=100.0, fill_price=100.05, side=1)

        est_after_low = model.estimate_bps(notional=1000.0)

        # Phase 2: high slippage fills
        for _ in range(10):
            model.record_fill(reference_price=100.0, fill_price=100.20, side=1)

        est_after_high = model.estimate_bps(notional=1000.0)

        # Assert — EWMA should have moved significantly toward 20 bps
        assert est_after_high.bps > est_after_low.bps
        # With alpha=0.3 and 10 high-slippage samples, should be closer to 20 than 5
        assert est_after_high.bps > 12.0

    def test_sell_side_slippage_computation(self) -> None:
        """Sell side: slippage = (reference - fill) / reference * 10000."""
        # Arrange — sell at 99.95 when ref is 100.0 => 5 bps slippage
        model = HistoricalSlippage(default_bps=10.0, min_samples=1, ewma_alpha=1.0)
        model.record_fill(reference_price=100.0, fill_price=99.95, side=-1)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(5.0)

    def test_negative_slippage_clamped_to_zero(self) -> None:
        """Price improvement (negative slippage) is clamped to 0."""
        # Arrange — buy at 99.95 when ref is 100.0 => negative raw slippage
        model = HistoricalSlippage(default_bps=10.0, min_samples=1, ewma_alpha=1.0)
        model.record_fill(reference_price=100.0, fill_price=99.95, side=1)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(0.0)

    def test_zero_reference_price_ignored(self) -> None:
        """record_fill with reference_price <= 0 is silently ignored."""
        # Arrange
        model = HistoricalSlippage(default_bps=5.0, min_samples=1)
        model.record_fill(reference_price=0.0, fill_price=100.0, side=1)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert — still at default because no valid samples recorded
        assert est.bps == pytest.approx(5.0)
        assert est.details["reason"] == "insufficient_samples"


# ---------------------------------------------------------------------------
# Tests: RegimeAwareSlippage
# ---------------------------------------------------------------------------


class TestRegimeAwareSlippage:
    def test_normal_regime_multiplier_is_one(self) -> None:
        """NORMAL regime applies a 1x multiplier (no change)."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(base_model=base)

        # Act
        est = model.estimate_bps(notional=1000.0, regime="normal")

        # Assert
        assert est.bps == pytest.approx(10.0)
        assert est.details["multiplier"] == pytest.approx(1.0)
        assert est.model_name == "regime_aware"

    def test_volatile_regime_applies_multiplier(self) -> None:
        """VOLATILE regime applies the volatile_multiplier (default 2x)."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(base_model=base, volatile_multiplier=2.0)

        # Act
        est = model.estimate_bps(notional=1000.0, regime="volatile")

        # Assert
        assert est.bps == pytest.approx(20.0)
        assert est.details["multiplier"] == pytest.approx(2.0)

    def test_stressed_regime_applies_three_point_five_x(self) -> None:
        """STRESSED regime applies 3.5x multiplier by default."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(base_model=base, stressed_multiplier=3.5)

        # Act
        est = model.estimate_bps(notional=1000.0, regime="stressed")

        # Assert
        assert est.bps == pytest.approx(35.0)
        assert est.details["multiplier"] == pytest.approx(3.5)

    def test_unknown_regime_falls_back_to_normal(self) -> None:
        """An unrecognized regime string defaults to NORMAL (1x)."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(base_model=base)

        # Act
        est = model.estimate_bps(notional=1000.0, regime="unknown_regime")

        # Assert
        assert est.bps == pytest.approx(10.0)

    def test_no_regime_kwarg_defaults_to_normal(self) -> None:
        """When no regime kwarg is passed, defaults to 'normal'."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(base_model=base)

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(10.0)

    def test_custom_multipliers(self) -> None:
        """Custom multipliers are respected."""
        # Arrange
        base = FixedSlippage(bps=10.0)
        model = RegimeAwareSlippage(
            base_model=base,
            volatile_multiplier=5.0,
            stressed_multiplier=10.0,
        )

        # Act
        est = model.estimate_bps(notional=1000.0, regime="stressed")

        # Assert
        assert est.bps == pytest.approx(100.0)

    def test_details_include_base_model_info(self) -> None:
        """Details include the base model's bps and name."""
        # Arrange
        base = FixedSlippage(bps=8.0)
        model = RegimeAwareSlippage(base_model=base)

        # Act
        est = model.estimate_bps(notional=1000.0, regime="volatile")

        # Assert
        assert est.details["base_bps"] == pytest.approx(8.0)
        assert est.details["base_model"] == "fixed"
        assert est.details["regime"] == "volatile"


# ---------------------------------------------------------------------------
# Tests: CompositeSlippage
# ---------------------------------------------------------------------------


class TestCompositeSlippage:
    def test_weighted_average_of_components(self) -> None:
        """Composite returns weighted average of component estimates."""
        # Arrange — 60% fixed@10, 40% fixed@20 => 14 bps
        fixed_10 = FixedSlippage(bps=10.0)
        fixed_20 = FixedSlippage(bps=20.0)
        model = CompositeSlippage(models=[(fixed_10, 0.6), (fixed_20, 0.4)])

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(14.0)
        assert est.model_name == "composite"

    def test_weights_are_normalized(self) -> None:
        """Weights that don't sum to 1.0 are automatically normalized."""
        # Arrange — weights 3 and 7 => normalized to 0.3 and 0.7
        fixed_10 = FixedSlippage(bps=10.0)
        fixed_20 = FixedSlippage(bps=20.0)
        model = CompositeSlippage(models=[(fixed_10, 3), (fixed_20, 7)])

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert — 10*0.3 + 20*0.7 = 3 + 14 = 17
        assert est.bps == pytest.approx(17.0)

    def test_single_component_returns_its_estimate(self) -> None:
        """A composite with one component returns that component's estimate."""
        # Arrange
        fixed = FixedSlippage(bps=15.0)
        model = CompositeSlippage(models=[(fixed, 1.0)])

        # Act
        est = model.estimate_bps(notional=500.0)

        # Assert
        assert est.bps == pytest.approx(15.0)

    def test_raises_on_zero_total_weight(self) -> None:
        """Raises ValueError when all weights sum to zero."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Weights must sum to a positive number"):
            CompositeSlippage(models=[(FixedSlippage(bps=10.0), 0.0)])

    def test_raises_on_negative_total_weight(self) -> None:
        """Raises ValueError when weights sum to a negative number."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Weights must sum to a positive number"):
            CompositeSlippage(
                models=[
                    (FixedSlippage(bps=10.0), -2.0),
                    (FixedSlippage(bps=20.0), 1.0),
                ]
            )

    def test_details_contain_component_breakdown(self) -> None:
        """Details include per-component model name, bps, and weight."""
        # Arrange
        fixed_a = FixedSlippage(bps=5.0)
        fixed_b = FixedSlippage(bps=15.0)
        model = CompositeSlippage(models=[(fixed_a, 0.5), (fixed_b, 0.5)])

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        components = est.details["components"]
        assert len(components) == 2
        assert components[0]["model"] == "fixed"
        assert components[1]["model"] == "fixed"
        assert components[0]["weight"] == pytest.approx(0.5)

    def test_three_component_composite(self) -> None:
        """Composite works correctly with three components."""
        # Arrange — equal weights: (5 + 10 + 15) / 3 = 10
        model = CompositeSlippage(
            models=[
                (FixedSlippage(bps=5.0), 1.0),
                (FixedSlippage(bps=10.0), 1.0),
                (FixedSlippage(bps=15.0), 1.0),
            ]
        )

        # Act
        est = model.estimate_bps(notional=1000.0)

        # Assert
        assert est.bps == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Tests: default_slippage_model factory
# ---------------------------------------------------------------------------


class TestDefaultSlippageModel:
    def test_returns_regime_aware_model(self) -> None:
        """default_slippage_model returns a RegimeAwareSlippage instance."""
        # Arrange & Act
        model = default_slippage_model()

        # Assert
        assert isinstance(model, RegimeAwareSlippage)

    def test_wraps_composite_slippage(self) -> None:
        """The base model inside the regime-aware wrapper is CompositeSlippage."""
        # Arrange & Act
        model = default_slippage_model()

        # Assert
        assert isinstance(model._base, CompositeSlippage)

    def test_produces_valid_estimate_in_normal_regime(self) -> None:
        """The default model produces a reasonable estimate under normal conditions."""
        # Arrange
        model = default_slippage_model()

        # Act
        est = model.estimate_bps(notional=10_000.0, orderbook_depth=1_000_000.0)

        # Assert
        assert est.bps > 0.0
        assert est.model_name == "regime_aware"
        assert est.details["regime"] == "normal"

    def test_stressed_regime_higher_than_normal(self) -> None:
        """Stressed regime produces higher slippage than normal for same order."""
        # Arrange
        model = default_slippage_model()

        # Act
        normal = model.estimate_bps(notional=10_000.0, orderbook_depth=1_000_000.0, regime="normal")
        stressed = model.estimate_bps(
            notional=10_000.0, orderbook_depth=1_000_000.0, regime="stressed"
        )

        # Assert
        assert stressed.bps > normal.bps
        assert stressed.bps == pytest.approx(normal.bps * 3.5)


# ---------------------------------------------------------------------------
# Tests: SlippageEstimate dataclass
# ---------------------------------------------------------------------------


class TestSlippageEstimate:
    def test_is_frozen(self) -> None:
        """SlippageEstimate is immutable (frozen dataclass)."""
        # Arrange
        est = SlippageEstimate(bps=5.0, model_name="test")

        # Act & Assert
        with pytest.raises(AttributeError):
            est.bps = 10.0  # type: ignore[misc]

    def test_default_details_is_empty_dict(self) -> None:
        """Details defaults to an empty dict."""
        # Arrange & Act
        est = SlippageEstimate(bps=5.0, model_name="test")

        # Assert
        assert est.details == {}
