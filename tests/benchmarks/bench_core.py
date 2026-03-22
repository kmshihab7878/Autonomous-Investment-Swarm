"""Benchmarks for AIS core hot paths using pytest-benchmark.

Run with:
    pytest tests/benchmarks/bench_core.py --benchmark-only -v
    pytest tests/benchmarks/bench_core.py --benchmark-compare
    pytest tests/benchmarks/bench_core.py --benchmark-json=bench.json

Measures throughput of:
    - Kelly criterion computation
    - HMAC risk token signing + verification round-trip
    - Slippage model estimation (Fixed, VolumeWeighted, Composite)
    - RiskEngine.validate() call
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from aiswarm.execution.slippage import (
    CompositeSlippage,
    FixedSlippage,
    HistoricalSlippage,
    RegimeAwareSlippage,
    VolumeWeightedSlippage,
)
from aiswarm.quant.kelly import (
    expected_value,
    half_kelly,
    kelly_fraction,
    kelly_position_size,
    variance,
)
from aiswarm.risk.limits import (
    RiskEngine,
    sign_risk_token,
    verify_risk_token,
)
from aiswarm.types.orders import Order, Side
from aiswarm.types.portfolio import PortfolioSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ensure_hmac_secret() -> None:
    """Ensure AIS_RISK_HMAC_SECRET is available for risk engine benchmarks."""
    os.environ["AIS_RISK_HMAC_SECRET"] = "bench-secret-key"


@pytest.fixture()
def sample_order() -> Order:
    """A representative order for benchmark inputs."""
    return Order(
        order_id="bench-order-001",
        signal_id="bench-signal-001",
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=0.5,
        notional=25_000.0,
        strategy="momentum_ma_crossover",
        thesis="Benchmark test thesis for order validation",
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture()
def sample_snapshot() -> PortfolioSnapshot:
    """A representative portfolio snapshot for benchmark inputs."""
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        nav=1_000_000.0,
        cash=500_000.0,
        gross_exposure=0.3,
        net_exposure=0.2,
        positions=(),
    )


@pytest.fixture()
def risk_engine() -> RiskEngine:
    """A standard RiskEngine instance matching config/risk.yaml defaults."""
    return RiskEngine(
        max_position_weight=0.05,
        max_gross_exposure=1.00,
        max_daily_loss=0.02,
        max_rolling_drawdown=0.05,
        max_leverage=1.00,
        min_liquidity_score=0.50,
    )


@pytest.fixture()
def fixed_slippage() -> FixedSlippage:
    return FixedSlippage(bps=5.0)


@pytest.fixture()
def volume_weighted_slippage() -> VolumeWeightedSlippage:
    return VolumeWeightedSlippage(
        base_bps=1.0,
        impact_coefficient=10.0,
        min_bps=0.5,
        max_bps=50.0,
    )


@pytest.fixture()
def composite_slippage() -> CompositeSlippage:
    volume = VolumeWeightedSlippage(base_bps=1.0, impact_coefficient=10.0)
    fixed = FixedSlippage(bps=3.0)
    return CompositeSlippage(models=[(volume, 0.7), (fixed, 0.3)])


@pytest.fixture()
def regime_aware_slippage() -> RegimeAwareSlippage:
    volume = VolumeWeightedSlippage(base_bps=1.0, impact_coefficient=10.0)
    fixed = FixedSlippage(bps=3.0)
    composite = CompositeSlippage(models=[(volume, 0.7), (fixed, 0.3)])
    return RegimeAwareSlippage(base_model=composite)


@pytest.fixture()
def historical_slippage_primed() -> HistoricalSlippage:
    """A HistoricalSlippage model with enough fills to use EWMA."""
    model = HistoricalSlippage(default_bps=5.0, min_samples=10)
    for i in range(20):
        model.record_fill(
            reference_price=50_000.0,
            fill_price=50_000.0 + (i * 0.5),
            side=1,
        )
    return model


# ---------------------------------------------------------------------------
# Kelly Criterion Benchmarks
# ---------------------------------------------------------------------------


class TestBenchKelly:
    """Benchmarks for Kelly criterion computations."""

    def test_bench_kelly_fraction(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark kelly_fraction for a typical edge case."""
        benchmark(kelly_fraction, 0.6, 2.0)

    def test_bench_half_kelly(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark half_kelly computation."""
        benchmark(half_kelly, 0.6, 2.0)

    def test_bench_kelly_position_size(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark full kelly_position_size with constraints."""
        benchmark(
            kelly_position_size,
            win_prob=0.6,
            payout_ratio=2.0,
            capital=1_000_000.0,
            max_position_pct=0.05,
        )

    def test_bench_expected_value(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark expected_value calculation."""
        benchmark(expected_value, 0.6, 2.0)

    def test_bench_variance(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark variance calculation."""
        benchmark(variance, 0.6, 2.0)

    def test_bench_kelly_no_edge(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark kelly_fraction for no-edge (payout <= 1) fast path."""
        benchmark(kelly_fraction, 0.6, 0.9)


# ---------------------------------------------------------------------------
# HMAC Token Benchmarks
# ---------------------------------------------------------------------------


class TestBenchHMAC:
    """Benchmarks for HMAC risk token signing and verification."""

    def test_bench_sign_risk_token(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark HMAC token signing."""
        benchmark(sign_risk_token, "order-bench-001")

    def test_bench_verify_risk_token(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark HMAC token verification (valid token)."""
        token = sign_risk_token("order-bench-001")
        benchmark(verify_risk_token, token, "order-bench-001")

    def test_bench_sign_and_verify_roundtrip(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark full sign + verify round-trip."""

        def roundtrip() -> bool:
            token = sign_risk_token("order-roundtrip-001")
            return verify_risk_token(token, "order-roundtrip-001")

        benchmark(roundtrip)

    def test_bench_verify_invalid_token(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark token verification rejection (tampered signature)."""
        token = sign_risk_token("order-bench-001")
        tampered = token[:-4] + "XXXX"
        benchmark(verify_risk_token, tampered, "order-bench-001")

    def test_bench_sign_long_order_id(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark signing with a long order ID (64 chars)."""
        long_id = "o" * 64
        benchmark(sign_risk_token, long_id)


# ---------------------------------------------------------------------------
# Slippage Model Benchmarks
# ---------------------------------------------------------------------------


class TestBenchSlippage:
    """Benchmarks for slippage model estimation."""

    def test_bench_fixed_slippage(
        self,
        benchmark: pytest.fixture,
        fixed_slippage: FixedSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark FixedSlippage.estimate_bps."""
        benchmark(fixed_slippage.estimate_bps, 50_000.0)

    def test_bench_volume_weighted_slippage(
        self,
        benchmark: pytest.fixture,
        volume_weighted_slippage: VolumeWeightedSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark VolumeWeightedSlippage.estimate_bps with typical inputs."""
        benchmark(
            volume_weighted_slippage.estimate_bps,
            50_000.0,
            orderbook_depth=5_000_000.0,
        )

    def test_bench_volume_weighted_zero_depth(
        self,
        benchmark: pytest.fixture,
        volume_weighted_slippage: VolumeWeightedSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark VolumeWeightedSlippage with zero depth (fast path)."""
        benchmark(
            volume_weighted_slippage.estimate_bps,
            50_000.0,
            orderbook_depth=0.0,
        )

    def test_bench_composite_slippage(
        self,
        benchmark: pytest.fixture,
        composite_slippage: CompositeSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark CompositeSlippage (volume_weighted + fixed) estimation."""
        benchmark(
            composite_slippage.estimate_bps,
            50_000.0,
            orderbook_depth=5_000_000.0,
        )

    def test_bench_regime_aware_slippage(
        self,
        benchmark: pytest.fixture,
        regime_aware_slippage: RegimeAwareSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark RegimeAwareSlippage (composite under normal regime)."""
        benchmark(
            regime_aware_slippage.estimate_bps,
            50_000.0,
            orderbook_depth=5_000_000.0,
            regime="normal",
        )

    def test_bench_regime_aware_stressed(
        self,
        benchmark: pytest.fixture,
        regime_aware_slippage: RegimeAwareSlippage,  # type: ignore[type-arg]
    ) -> None:
        """Benchmark RegimeAwareSlippage under stressed regime."""
        benchmark(
            regime_aware_slippage.estimate_bps,
            50_000.0,
            orderbook_depth=5_000_000.0,
            regime="stressed",
        )

    def test_bench_historical_slippage(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        historical_slippage_primed: HistoricalSlippage,
    ) -> None:
        """Benchmark HistoricalSlippage.estimate_bps with primed EWMA."""
        benchmark(historical_slippage_primed.estimate_bps, 50_000.0)

    def test_bench_historical_record_fill(self, benchmark: pytest.fixture) -> None:  # type: ignore[type-arg]
        """Benchmark HistoricalSlippage.record_fill (EWMA update)."""
        model = HistoricalSlippage(default_bps=5.0, min_samples=10)

        def record_one() -> None:
            model.record_fill(
                reference_price=50_000.0,
                fill_price=50_002.5,
                side=1,
            )

        benchmark(record_one)


# ---------------------------------------------------------------------------
# RiskEngine.validate() Benchmarks
# ---------------------------------------------------------------------------


class TestBenchRiskEngine:
    """Benchmarks for RiskEngine.validate() throughput."""

    def test_bench_risk_engine_approve(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        risk_engine: RiskEngine,
        sample_order: Order,
        sample_snapshot: PortfolioSnapshot,
    ) -> None:
        """Benchmark RiskEngine.validate() for a passing order (full pipeline)."""
        result = benchmark(
            risk_engine.validate,
            order=sample_order,
            snapshot=sample_snapshot,
            daily_pnl_fraction=0.0,
            rolling_drawdown=0.0,
            current_leverage=0.0,
            liquidity_score=1.0,
        )
        assert result.approved

    def test_bench_risk_engine_reject_kill_switch(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        risk_engine: RiskEngine,
        sample_order: Order,
        sample_snapshot: PortfolioSnapshot,
    ) -> None:
        """Benchmark RiskEngine.validate() for a kill-switch rejection (early exit)."""
        result = benchmark(
            risk_engine.validate,
            order=sample_order,
            snapshot=sample_snapshot,
            daily_pnl_fraction=-0.05,
            rolling_drawdown=0.0,
            current_leverage=0.0,
            liquidity_score=1.0,
        )
        assert not result.approved

    def test_bench_risk_engine_reject_multi_violation(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        risk_engine: RiskEngine,
        sample_order: Order,
        sample_snapshot: PortfolioSnapshot,
    ) -> None:
        """Benchmark RiskEngine.validate() with multiple violations (worst case path)."""
        result = benchmark(
            risk_engine.validate,
            order=sample_order,
            snapshot=sample_snapshot,
            daily_pnl_fraction=-0.05,
            rolling_drawdown=0.10,
            current_leverage=2.0,
            liquidity_score=0.1,
        )
        assert not result.approved
        assert len(result.reasons) >= 4

    def test_bench_risk_engine_no_snapshot(
        self,
        benchmark: pytest.fixture,  # type: ignore[type-arg]
        risk_engine: RiskEngine,
        sample_order: Order,
    ) -> None:
        """Benchmark RiskEngine.validate() without portfolio snapshot (paper mode path)."""
        result = benchmark(
            risk_engine.validate,
            order=sample_order,
            snapshot=None,
            daily_pnl_fraction=0.0,
        )
        assert result.approved
