"""Tests for ExposureManager — pre-trade exposure checks."""

from __future__ import annotations

from datetime import datetime, timezone

from aiswarm.portfolio.exposure import ExposureManager
from aiswarm.types.orders import Order, Side
from aiswarm.types.portfolio import PortfolioSnapshot
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now


def _make_order(notional: float = 5_000.0) -> Order:
    return Order(
        order_id=new_id("ord"),
        signal_id=new_id("sig"),
        symbol="BTCUSDT",
        side=Side.BUY,
        quantity=0.1,
        limit_price=None,
        notional=notional,
        strategy="test_strategy",
        thesis="Test thesis",
        created_at=utc_now(),
    )


def _make_snapshot(
    nav: float = 100_000.0,
    gross_exposure: float = 0.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        nav=nav,
        cash=nav,
        gross_exposure=gross_exposure,
        net_exposure=0.0,
        positions=[],
        timestamp=datetime.now(timezone.utc),
    )


class TestCheckOrder:
    def test_order_within_limits_passes(self) -> None:
        mgr = ExposureManager(max_position_weight=0.10, max_gross_exposure=1.0)
        ok, reason = mgr.check_order(_make_order(5_000.0), _make_snapshot(nav=100_000.0))
        assert ok is True
        assert reason == "ok"

    def test_order_exceeding_position_weight_rejected(self) -> None:
        mgr = ExposureManager(max_position_weight=0.01, max_gross_exposure=1.0)
        ok, reason = mgr.check_order(_make_order(5_000.0), _make_snapshot(nav=100_000.0))
        assert ok is False
        assert "position_weight" in reason

    def test_order_exceeding_gross_exposure_rejected(self) -> None:
        mgr = ExposureManager(max_position_weight=0.10, max_gross_exposure=0.5)
        snapshot = _make_snapshot(nav=100_000.0, gross_exposure=0.48)
        ok, reason = mgr.check_order(_make_order(5_000.0), snapshot)
        assert ok is False
        assert "gross_exposure" in reason

    def test_no_snapshot_uses_default_nav(self) -> None:
        mgr = ExposureManager(max_position_weight=0.10, max_gross_exposure=1.0)
        ok, reason = mgr.check_order(_make_order(50_000.0), None)
        # 50k / 1M default = 0.05, under 0.10 limit
        assert ok is True

    def test_order_at_exact_limit_rejected(self) -> None:
        mgr = ExposureManager(max_position_weight=0.05, max_gross_exposure=1.0)
        # 5001 / 100_000 = 0.05001 > 0.05
        ok, _ = mgr.check_order(_make_order(5_001.0), _make_snapshot(nav=100_000.0))
        assert ok is False
