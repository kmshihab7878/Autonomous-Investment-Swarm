"""Tests for PortfolioAllocator — position sizing from signals."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aiswarm.portfolio.allocator import PortfolioAllocator
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.types.orders import Side
from aiswarm.types.portfolio import PortfolioSnapshot
from aiswarm.utils.ids import new_id


def _make_signal(
    direction: int = 1,
    confidence: float = 0.7,
    expected_return: float = 0.02,
    reference_price: float = 50_000.0,
) -> Signal:
    return Signal(
        signal_id=new_id("sig"),
        agent_id="test_agent",
        symbol="BTCUSDT",
        strategy="momentum_ma_crossover",
        thesis="Test thesis",
        direction=direction,
        confidence=confidence,
        expected_return=expected_return,
        horizon_minutes=60,
        liquidity_score=0.8,
        regime=MarketRegime.RISK_ON,
        created_at=datetime.now(timezone.utc),
        reference_price=reference_price,
    )


def _make_snapshot(nav: float = 100_000.0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        cash=nav,
        gross_exposure=0.0,
        net_exposure=0.0,
        positions=[],
        timestamp=datetime.now(timezone.utc),
    )


class TestOrderFromSignal:
    def test_buy_signal_produces_buy_order(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        signal = _make_signal(direction=1)
        order = allocator.order_from_signal(signal, _make_snapshot())
        assert order.side == Side.BUY

    def test_sell_signal_produces_sell_order(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        signal = _make_signal(direction=-1)
        order = allocator.order_from_signal(signal, _make_snapshot())
        assert order.side == Side.SELL

    def test_notional_scales_with_nav(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        small = allocator.order_from_signal(_make_signal(), _make_snapshot(nav=10_000.0))
        large = allocator.order_from_signal(_make_signal(), _make_snapshot(nav=1_000_000.0))
        assert large.notional > small.notional

    def test_notional_has_minimum_floor(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.001)
        signal = _make_signal(confidence=0.01)
        order = allocator.order_from_signal(signal, _make_snapshot(nav=100.0))
        assert order.notional >= 100.0

    def test_quantity_derived_from_reference_price(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        signal = _make_signal(reference_price=50_000.0)
        order = allocator.order_from_signal(signal, _make_snapshot())
        assert order.quantity == pytest.approx(order.notional / 50_000.0, abs=0.001)

    def test_no_snapshot_uses_default_nav(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        order = allocator.order_from_signal(_make_signal(), None)
        # Default NAV is 1_000_000
        expected_notional = max(1_000_000.0 * 0.02 * 0.7, 100.0)
        assert order.notional == pytest.approx(expected_notional)

    def test_order_preserves_signal_metadata(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02)
        signal = _make_signal()
        order = allocator.order_from_signal(signal, _make_snapshot())
        assert order.signal_id == signal.signal_id
        assert order.symbol == signal.symbol
        assert order.strategy == signal.strategy


class TestKellyWeighting:
    def test_kelly_disabled_uses_target_weight(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.03, use_kelly=False)
        weight = allocator._compute_weight(_make_signal())
        assert weight == 0.03

    def test_kelly_enabled_computes_from_signal(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02, use_kelly=True, max_kelly_weight=0.10)
        weight = allocator._compute_weight(_make_signal(confidence=0.7, expected_return=0.05))
        assert weight > 0
        assert weight <= 0.10

    def test_kelly_capped_at_max(self) -> None:
        allocator = PortfolioAllocator(target_weight=0.02, use_kelly=True, max_kelly_weight=0.01)
        weight = allocator._compute_weight(_make_signal(confidence=0.9, expected_return=0.1))
        assert weight <= 0.01
