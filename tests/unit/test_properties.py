"""Property-based tests using Hypothesis for AIS core modules.

Tests invariants that must hold for ALL valid inputs, not just hand-picked examples.
Covers: Kelly criterion, HMAC risk tokens, slippage models, and Order serialization.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from aiswarm.execution.slippage import (
    CompositeSlippage,
    FixedSlippage,
    HistoricalSlippage,
    VolumeWeightedSlippage,
)
from aiswarm.quant.kelly import (
    expected_value,
    half_kelly,
    kelly_fraction,
    kelly_position_size,
    variance,
)
from aiswarm.risk.limits import sign_risk_token, verify_risk_token
from aiswarm.types.orders import Order, OrderStatus, Side

# ---------------------------------------------------------------------------
# Hypothesis profiles (registered once at import time)
# ---------------------------------------------------------------------------
settings.register_profile("ci", max_examples=200, deadline=5000)
settings.register_profile("dev", max_examples=50, deadline=3000)
settings.register_profile(
    "debug",
    max_examples=10,
    deadline=10000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# ---------------------------------------------------------------------------
# Custom strategies
# ---------------------------------------------------------------------------

# Win probability: strictly between 0 and 1 (exclusive) for meaningful Kelly
win_probs = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

# Payout ratio: must be > 1 for Kelly to have a positive bet zone
payout_ratios_positive = st.floats(
    min_value=1.01, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Payout ratio: full range for testing edge cases
payout_ratios_any = st.floats(
    min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
)

# Notional values for slippage
notionals = st.floats(min_value=1.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False)

# Orderbook depths for slippage
depths = st.floats(min_value=100.0, max_value=100_000_000.0, allow_nan=False, allow_infinity=False)

# Basis points for fixed slippage
bps_values = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

# Order IDs: printable ASCII, no colons (since token format uses colons as separators)
order_ids = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=64,
)


# ---------------------------------------------------------------------------
# Kelly Criterion Properties
# ---------------------------------------------------------------------------


class TestKellyProperties:
    """Property-based tests for the Kelly criterion module."""

    @given(p=win_probs, r=payout_ratios_positive)
    def test_kelly_fraction_bounded_for_positive_edge(self, p: float, r: float) -> None:
        """kelly_fraction(p, r) is in [0, 1] when the Kelly edge is non-negative.

        The Kelly edge is (b*p - q) where b = r - 1, q = 1 - p.
        When this edge is >= 0 the fraction must lie in [0, 1].
        Note: positive EV does NOT guarantee positive Kelly edge
        (they use different formulations), so we filter on Kelly >= 0.
        """
        # Arrange / Act
        f = kelly_fraction(p, r)
        assume(f >= 0)  # Only test the "bet" region

        # Assert
        assert 0.0 <= f <= 1.0, f"Kelly fraction {f} out of [0,1] for p={p}, r={r}"

    @given(p=win_probs, r=payout_ratios_positive)
    def test_half_kelly_leq_kelly_when_positive(self, p: float, r: float) -> None:
        """half_kelly(p, r) <= kelly_fraction(p, r) when Kelly fraction >= 0.

        For negative Kelly fractions, halving brings the value closer to
        zero (i.e., half > full), so the invariant only holds in the
        non-negative region where a position is recommended.
        """
        # Arrange / Act
        full = kelly_fraction(p, r)
        half = half_kelly(p, r)
        assume(full >= 0)

        # Assert
        assert half <= full + 1e-12, f"half_kelly={half} > kelly_fraction={full} for p={p}, r={r}"

    @given(p=win_probs, r=payout_ratios_positive)
    def test_half_kelly_closer_to_zero(self, p: float, r: float) -> None:
        """abs(half_kelly) <= abs(kelly_fraction) always -- half-Kelly is more conservative."""
        full = kelly_fraction(p, r)
        half = half_kelly(p, r)
        assert abs(half) <= abs(full) + 1e-12, (
            f"|half_kelly|={abs(half)} > |kelly_fraction|={abs(full)} for p={p}, r={r}"
        )

    @given(p=win_probs, r=payout_ratios_positive)
    def test_half_kelly_is_exactly_half(self, p: float, r: float) -> None:
        """half_kelly is exactly 0.5 * kelly_fraction."""
        full = kelly_fraction(p, r)
        half = half_kelly(p, r)
        assert abs(half - full * 0.5) < 1e-12

    @given(r=payout_ratios_positive)
    def test_kelly_monotonic_in_win_prob(self, r: float) -> None:
        """Kelly fraction is monotonically non-decreasing in win probability.

        For a fixed payout ratio, higher win probability should never
        produce a lower Kelly fraction.
        """
        # Arrange: sample a sequence of increasing win probs
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Act
        fractions = [kelly_fraction(p, r) for p in probs]

        # Assert
        for i in range(len(fractions) - 1):
            assert fractions[i] <= fractions[i + 1] + 1e-12, (
                f"Not monotonic: f({probs[i]}, {r})={fractions[i]} > "
                f"f({probs[i + 1]}, {r})={fractions[i + 1]}"
            )

    def test_kelly_breakeven_edge_case(self) -> None:
        """kelly_fraction(0.5, 2.0) == 0.0 (break-even: edge is exactly zero).

        With 50% win rate and 2x payout, expected value is zero, so Kelly
        recommends zero position.
        """
        f = kelly_fraction(0.5, 2.0)
        assert abs(f) < 1e-12, f"Expected 0.0 at breakeven, got {f}"

    @given(p=win_probs, r=payout_ratios_any)
    def test_kelly_fraction_is_finite(self, p: float, r: float) -> None:
        """Kelly fraction never returns NaN or infinity."""
        f = kelly_fraction(p, r)
        assert math.isfinite(f), f"Non-finite kelly_fraction={f} for p={p}, r={r}"

    @given(
        p=win_probs,
        r=payout_ratios_positive,
        capital=st.floats(min_value=1000.0, max_value=10_000_000.0),
        max_pct=st.floats(min_value=0.01, max_value=0.50),
    )
    def test_kelly_position_size_respects_cap(
        self, p: float, r: float, capital: float, max_pct: float
    ) -> None:
        """kelly_position_size never exceeds capital * max_position_pct."""
        size = kelly_position_size(p, r, capital, max_position_pct=max_pct)
        assert size <= capital * max_pct + 1e-6, (
            f"Position size {size} exceeds cap {capital * max_pct}"
        )
        assert size >= 0.0, f"Position size cannot be negative: {size}"

    @given(p=win_probs, r=payout_ratios_any)
    def test_expected_value_is_finite(self, p: float, r: float) -> None:
        """Expected value is always a finite number for valid inputs."""
        ev = expected_value(p, r)
        assert math.isfinite(ev)

    @given(p=win_probs, r=payout_ratios_positive)
    def test_variance_non_negative(self, p: float, r: float) -> None:
        """Variance of outcome is always >= 0."""
        v = variance(p, r)
        assert v >= -1e-12, f"Variance should be non-negative, got {v}"


# ---------------------------------------------------------------------------
# HMAC Risk Token Properties
# ---------------------------------------------------------------------------


class TestHMACTokenProperties:
    """Property-based tests for sign_risk_token / verify_risk_token round-trip."""

    def setup_method(self) -> None:
        os.environ["AIS_RISK_HMAC_SECRET"] = "test-property-secret"

    @given(oid=order_ids)
    def test_sign_verify_roundtrip(self, oid: str) -> None:
        """sign_risk_token + verify_risk_token round-trips for any order_id string."""
        # Arrange / Act
        token = sign_risk_token(oid)
        result = verify_risk_token(token, oid)

        # Assert
        assert result is True, f"Round-trip failed for order_id={oid!r}"

    @given(oid=order_ids)
    def test_token_has_four_colon_separated_parts(self, oid: str) -> None:
        """Signed tokens always contain exactly 3 colons (format: order_id:timestamp:key_id:signature)."""
        # Arrange / Act
        token = sign_risk_token(oid)
        parts = token.split(":")

        # Assert
        assert len(parts) == 4, f"Expected 4 parts, got {len(parts)} for token={token!r}"
        assert parts[0] == oid, "First part should be order_id"
        assert parts[1].isdigit(), f"Second part (timestamp) should be numeric: {parts[1]}"
        assert len(parts[3]) == 64, "Signature should be 64 hex chars (SHA256)"

    @given(oid=order_ids, wrong_oid=order_ids)
    def test_token_rejects_wrong_order_id(self, oid: str, wrong_oid: str) -> None:
        """A token signed for one order_id must not verify for a different one."""
        assume(oid != wrong_oid)

        token = sign_risk_token(oid)
        assert not verify_risk_token(token, wrong_oid)

    @given(oid=order_ids)
    def test_token_rejects_truncation(self, oid: str) -> None:
        """A truncated token must not verify."""
        token = sign_risk_token(oid)
        truncated = token[: len(token) // 2]
        # Truncated token might have wrong number of parts
        assert not verify_risk_token(truncated, oid)

    @given(oid=order_ids)
    def test_token_signature_is_hex(self, oid: str) -> None:
        """The signature portion is a valid hex string."""
        token = sign_risk_token(oid)
        sig = token.split(":")[-1]
        # Verify it is valid hex by attempting conversion
        int(sig, 16)  # Raises ValueError if not hex


# ---------------------------------------------------------------------------
# Slippage Model Properties
# ---------------------------------------------------------------------------


class TestFixedSlippageProperties:
    """Property-based tests for FixedSlippage."""

    @given(fixed_bps=bps_values, notional=notionals)
    def test_always_returns_configured_bps(self, fixed_bps: float, notional: float) -> None:
        """FixedSlippage always returns the same value regardless of notional."""
        # Arrange
        model = FixedSlippage(bps=fixed_bps)

        # Act
        est = model.estimate_bps(notional)

        # Assert
        assert est.bps == fixed_bps, f"Expected {fixed_bps}, got {est.bps}"
        assert est.model_name == "fixed"

    @given(
        fixed_bps=bps_values,
        n1=notionals,
        n2=notionals,
    )
    def test_independent_of_notional(self, fixed_bps: float, n1: float, n2: float) -> None:
        """Two different notionals produce identical bps from FixedSlippage."""
        model = FixedSlippage(bps=fixed_bps)
        e1 = model.estimate_bps(n1)
        e2 = model.estimate_bps(n2)
        assert e1.bps == e2.bps


class TestVolumeWeightedSlippageProperties:
    """Property-based tests for VolumeWeightedSlippage."""

    @given(notional=notionals, depth=depths)
    def test_result_within_bounds(self, notional: float, depth: float) -> None:
        """VolumeWeightedSlippage always returns bps in [min_bps, max_bps]."""
        # Arrange
        min_bps = 0.5
        max_bps = 50.0
        model = VolumeWeightedSlippage(
            base_bps=1.0,
            impact_coefficient=10.0,
            min_bps=min_bps,
            max_bps=max_bps,
        )

        # Act
        est = model.estimate_bps(notional, orderbook_depth=depth)

        # Assert
        assert min_bps <= est.bps <= max_bps, (
            f"bps={est.bps} outside [{min_bps}, {max_bps}] for notional={notional}, depth={depth}"
        )

    @given(
        notional=notionals,
        min_bps=st.floats(min_value=0.1, max_value=5.0),
        max_bps=st.floats(min_value=10.0, max_value=200.0),
    )
    def test_custom_bounds_respected(self, notional: float, min_bps: float, max_bps: float) -> None:
        """Custom min/max bps bounds are always respected."""
        model = VolumeWeightedSlippage(
            base_bps=1.0,
            impact_coefficient=10.0,
            min_bps=min_bps,
            max_bps=max_bps,
        )
        est = model.estimate_bps(notional, orderbook_depth=1_000_000.0)
        assert min_bps <= est.bps <= max_bps

    @given(depth=depths)
    def test_larger_order_more_slippage(self, depth: float) -> None:
        """Larger orders should produce >= slippage than smaller orders.

        Monotonicity: for fixed depth, slippage is non-decreasing in notional.
        """
        model = VolumeWeightedSlippage()
        small = model.estimate_bps(1000.0, orderbook_depth=depth)
        large = model.estimate_bps(1_000_000.0, orderbook_depth=depth)
        assert small.bps <= large.bps + 1e-12

    @given(notional=notionals)
    def test_zero_depth_returns_max(self, notional: float) -> None:
        """When orderbook depth is zero or negative, returns max_bps."""
        max_bps = 50.0
        model = VolumeWeightedSlippage(max_bps=max_bps)
        est = model.estimate_bps(notional, orderbook_depth=0.0)
        assert est.bps == max_bps
        assert est.details.get("reason") == "zero_depth"


class TestHistoricalSlippageProperties:
    """Property-based tests for HistoricalSlippage."""

    @given(
        default_bps=bps_values,
        notional=notionals,
        min_samples=st.integers(min_value=5, max_value=50),
    )
    def test_returns_default_until_min_samples_reached(
        self, default_bps: float, notional: float, min_samples: int
    ) -> None:
        """HistoricalSlippage returns default_bps until min_samples fills are recorded."""
        # Arrange
        model = HistoricalSlippage(
            default_bps=default_bps,
            min_samples=min_samples,
        )

        # Act -- record fewer samples than required
        for i in range(min_samples - 1):
            model.record_fill(
                reference_price=100.0,
                fill_price=100.05,
                side=1,
            )

        est = model.estimate_bps(notional)

        # Assert
        assert est.bps == default_bps, (
            f"Expected default_bps={default_bps}, got {est.bps} "
            f"with {min_samples - 1} samples < min_samples={min_samples}"
        )
        assert est.details.get("reason") == "insufficient_samples"

    @given(notional=notionals)
    def test_ewma_activates_after_min_samples(self, notional: float) -> None:
        """After recording min_samples fills, HistoricalSlippage uses EWMA estimate."""
        model = HistoricalSlippage(default_bps=5.0, min_samples=3)

        for _ in range(3):
            model.record_fill(reference_price=100.0, fill_price=100.10, side=1)

        est = model.estimate_bps(notional)
        # Should now use EWMA, not the default
        assert (
            est.details.get("reason") is None or est.details.get("reason") != "insufficient_samples"
        )
        assert est.bps >= 0.0  # EWMA of positive slippage should be non-negative


class TestCompositeSlippageProperties:
    """Property-based tests for CompositeSlippage."""

    @given(
        bps_a=st.floats(min_value=1.0, max_value=50.0),
        bps_b=st.floats(min_value=1.0, max_value=50.0),
        w_a=st.floats(min_value=0.1, max_value=10.0),
        w_b=st.floats(min_value=0.1, max_value=10.0),
        notional=notionals,
    )
    def test_composite_between_components(
        self, bps_a: float, bps_b: float, w_a: float, w_b: float, notional: float
    ) -> None:
        """CompositeSlippage result is always between min and max of components."""
        # Arrange
        model_a = FixedSlippage(bps=bps_a)
        model_b = FixedSlippage(bps=bps_b)
        composite = CompositeSlippage(models=[(model_a, w_a), (model_b, w_b)])

        # Act
        est = composite.estimate_bps(notional)

        # Assert
        lo = min(bps_a, bps_b)
        hi = max(bps_a, bps_b)
        assert lo - 1e-9 <= est.bps <= hi + 1e-9, f"Composite bps={est.bps} not in [{lo}, {hi}]"

    @given(
        bps_a=st.floats(min_value=1.0, max_value=50.0),
        bps_b=st.floats(min_value=1.0, max_value=50.0),
        bps_c=st.floats(min_value=1.0, max_value=50.0),
        notional=notionals,
    )
    def test_composite_three_models_bounded(
        self, bps_a: float, bps_b: float, bps_c: float, notional: float
    ) -> None:
        """Composite of three models stays within [min, max] of components."""
        models = [
            (FixedSlippage(bps=bps_a), 1.0),
            (FixedSlippage(bps=bps_b), 1.0),
            (FixedSlippage(bps=bps_c), 1.0),
        ]
        composite = CompositeSlippage(models=models)
        est = composite.estimate_bps(notional)

        lo = min(bps_a, bps_b, bps_c)
        hi = max(bps_a, bps_b, bps_c)
        assert lo - 1e-9 <= est.bps <= hi + 1e-9

    def test_composite_rejects_zero_weights(self) -> None:
        """CompositeSlippage raises ValueError when all weights sum to zero."""
        with pytest.raises(ValueError, match="Weights must sum to a positive number"):
            CompositeSlippage(models=[(FixedSlippage(bps=5.0), 0.0)])


# ---------------------------------------------------------------------------
# Order Model Properties
# ---------------------------------------------------------------------------


class TestOrderModelProperties:
    """Property-based tests for Order Pydantic model serialization."""

    @given(
        order_id=st.text(
            min_size=1, max_size=32, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        symbol=st.sampled_from(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]),
        side=st.sampled_from([Side.BUY, Side.SELL]),
        quantity=st.floats(
            min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        notional=st.floats(
            min_value=1.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False
        ),
        strategy=st.sampled_from(
            ["momentum_ma_crossover", "funding_rate_contrarian", "mean_reversion"]
        ),
    )
    def test_order_serialize_deserialize_roundtrip(
        self,
        order_id: str,
        symbol: str,
        side: Side,
        quantity: float,
        notional: float,
        strategy: str,
    ) -> None:
        """Valid orders serialize (model_dump) and deserialize (model_validate) correctly."""
        # Arrange
        order = Order(
            order_id=order_id,
            signal_id="sig1",
            symbol=symbol,
            side=side,
            quantity=quantity,
            notional=notional,
            strategy=strategy,
            thesis="A valid test thesis with enough length",
            created_at=datetime.now(timezone.utc),
        )

        # Act -- round-trip through dict
        dumped = order.model_dump(mode="python")
        restored = Order.model_validate(dumped)

        # Assert
        assert restored.order_id == order.order_id
        assert restored.symbol == order.symbol
        assert restored.side == order.side
        assert restored.quantity == order.quantity
        assert restored.notional == order.notional
        assert restored.strategy == order.strategy
        assert restored.status == OrderStatus.PENDING

    @given(
        order_id=st.text(
            min_size=1, max_size=32, alphabet=st.characters(whitelist_categories=("L", "N"))
        ),
        symbol=st.sampled_from(["BTCUSDT", "ETHUSDT"]),
        side=st.sampled_from([Side.BUY, Side.SELL]),
        quantity=st.floats(
            min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        notional=st.floats(
            min_value=1.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False
        ),
    )
    def test_order_json_roundtrip(
        self,
        order_id: str,
        symbol: str,
        side: Side,
        quantity: float,
        notional: float,
    ) -> None:
        """Orders survive JSON serialization and deserialization."""
        order = Order(
            order_id=order_id,
            signal_id="sig1",
            symbol=symbol,
            side=side,
            quantity=quantity,
            notional=notional,
            strategy="test_strat",
            thesis="Valid thesis for testing",
            created_at=datetime.now(timezone.utc),
        )

        json_str = order.model_dump_json()
        restored = Order.model_validate_json(json_str)

        assert restored.order_id == order.order_id
        assert restored.quantity == order.quantity
        assert restored.notional == order.notional

    @given(
        bad_qty=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    def test_order_rejects_non_positive_quantity(self, bad_qty: float) -> None:
        """Order model rejects quantity <= 0."""
        with pytest.raises(ValidationError):
            Order(
                order_id="o1",
                signal_id="s1",
                symbol="BTCUSDT",
                side=Side.BUY,
                quantity=bad_qty,
                notional=1000.0,
                strategy="test",
                thesis="Valid thesis for test",
                created_at=datetime.now(timezone.utc),
            )

    @given(
        bad_notional=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    def test_order_rejects_non_positive_notional(self, bad_notional: float) -> None:
        """Order model rejects notional <= 0."""
        with pytest.raises(ValidationError):
            Order(
                order_id="o1",
                signal_id="s1",
                symbol="BTCUSDT",
                side=Side.BUY,
                quantity=1.0,
                notional=bad_notional,
                strategy="test",
                thesis="Valid thesis for test",
                created_at=datetime.now(timezone.utc),
            )

    def test_order_rejects_short_thesis(self) -> None:
        """Order model rejects thesis shorter than 5 characters."""
        with pytest.raises(ValidationError):
            Order(
                order_id="o1",
                signal_id="s1",
                symbol="BTCUSDT",
                side=Side.BUY,
                quantity=1.0,
                notional=1000.0,
                strategy="test",
                thesis="Hi",  # Too short (min_length=5)
                created_at=datetime.now(timezone.utc),
            )

    def test_order_is_frozen(self) -> None:
        """Order model is immutable (frozen=True)."""
        order = Order(
            order_id="o1",
            signal_id="s1",
            symbol="BTCUSDT",
            side=Side.BUY,
            quantity=1.0,
            notional=1000.0,
            strategy="test",
            thesis="Valid thesis for test",
            created_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ValidationError):
            order.quantity = 2.0  # type: ignore[misc]
