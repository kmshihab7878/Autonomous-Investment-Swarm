"""Tests for surgical reconciliation — cancel only mismatched symbols.

Verifies that reconciliation mismatches trigger cancellation of orders only
for the specific symbols that diverged, leaving orders for matching symbols
untouched.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

import pytest

from aiswarm.data.event_store import EventStore
from aiswarm.data.providers.aster import ExchangePosition
from aiswarm.data.providers.aster_config import AsterConfig
from aiswarm.execution.aster_executor import AsterExecutor, ExecutionMode
from aiswarm.execution.live_executor import LiveOrderExecutor
from aiswarm.execution.mcp_gateway import MockMCPGateway
from aiswarm.execution.order_store import OrderStore
from aiswarm.monitoring.reconciliation import (
    PositionReconciler,
    ReconciliationLoop,
)
from aiswarm.risk.limits import sign_risk_token
from aiswarm.types.orders import Order, OrderStatus, Side
from aiswarm.types.portfolio import PortfolioSnapshot, Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> EventStore:
    return EventStore(tempfile.mktemp(suffix=".db"))


def _make_order(order_id: str, symbol: str = "BTCUSDT") -> Order:
    return Order(
        order_id=order_id,
        signal_id=f"sig_{order_id}",
        symbol=symbol,
        side=Side.BUY,
        quantity=0.1,
        limit_price=None,
        notional=5000.0,
        strategy="momentum",
        thesis="valid test thesis",
        created_at=datetime.now(timezone.utc),
        risk_approval_token=sign_risk_token(order_id),
        status=OrderStatus.APPROVED,
    )


def _make_position(
    symbol: str = "BTCUSDT",
    quantity: float = 1.0,
    price: float = 50000.0,
) -> Position:
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_price=price,
        market_price=price,
        strategy="test",
    )


def _make_snapshot(positions: tuple[Position, ...] = ()) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        nav=100_000,
        cash=100_000,
        gross_exposure=0.0,
        net_exposure=0.0,
        positions=positions,
    )


def _make_exchange_position(
    symbol: str = "BTCUSDT",
    quantity: float = 1.0,
) -> ExchangePosition:
    return ExchangePosition(
        symbol=symbol,
        side="LONG",
        quantity=quantity,
        entry_price=50000.0,
        mark_price=50000.0,
        unrealized_pnl=0.0,
        leverage=1,
        margin_mode="ISOLATED",
    )


def _submit_live_order(
    live: LiveOrderExecutor,
    store: OrderStore,
    order_id: str,
    symbol: str,
) -> None:
    """Submit an order in live mode and verify it is tracked as SUBMITTED."""
    order = _make_order(order_id, symbol)
    result = live.submit_order(order)
    assert result.success
    record = store.get(order_id)
    assert record is not None
    assert record.order.status == OrderStatus.SUBMITTED


@pytest.fixture(autouse=True)
def _env() -> None:
    os.environ["AIS_RISK_HMAC_SECRET"] = "test-secret-for-ci"
    os.environ["AIS_ENABLE_LIVE_TRADING"] = "true"
    os.environ["ASTER_ACCOUNT_ID"] = "test_account"
    yield  # type: ignore[misc]
    os.environ.pop("AIS_ENABLE_LIVE_TRADING", None)
    os.environ.pop("ASTER_ACCOUNT_ID", None)


def _make_live_executor() -> tuple[LiveOrderExecutor, OrderStore, MockMCPGateway]:
    """Create a LiveOrderExecutor in LIVE mode with mock gateway."""
    config = AsterConfig(account_id="test_account")
    executor = AsterExecutor(config=config, mode=ExecutionMode.LIVE)
    gateway = MockMCPGateway()
    event_store = _make_store()
    store = OrderStore(event_store)
    live = LiveOrderExecutor(executor, gateway, store)
    return live, store, gateway


# ---------------------------------------------------------------------------
# ReconciliationReport.mismatched_symbols
# ---------------------------------------------------------------------------


class TestReconciliationReportMismatchedSymbols:
    """Tests for the mismatched_symbols property on ReconciliationReport."""

    def test_no_mismatches_returns_empty_list(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange = [_make_exchange_position("BTCUSDT", 1.0)]

        report = reconciler.run_full_reconciliation(snap, exchange)
        assert report.passed
        assert report.mismatched_symbols == []

    def test_single_mismatch_returns_that_symbol(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
            )
        )
        exchange = [
            _make_exchange_position("BTCUSDT", 1.0),  # match
            _make_exchange_position("ETHUSDT", 5.0),  # mismatch
        ]

        report = reconciler.run_full_reconciliation(snap, exchange)
        assert not report.passed
        assert report.mismatched_symbols == ["ETHUSDT"]

    def test_multiple_mismatches_returns_all_symbols(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
                _make_position("SOLUSDT", 3.0),
            )
        )
        exchange = [
            _make_exchange_position("BTCUSDT", 9.0),  # mismatch
            _make_exchange_position("ETHUSDT", 2.0),  # match
            _make_exchange_position("SOLUSDT", 9.0),  # mismatch
        ]

        report = reconciler.run_full_reconciliation(snap, exchange)
        assert not report.passed
        assert report.mismatched_symbols == ["BTCUSDT", "SOLUSDT"]

    def test_missing_exchange_counted_as_mismatch(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
            )
        )
        exchange = [_make_exchange_position("BTCUSDT", 1.0)]  # ETHUSDT missing

        report = reconciler.run_full_reconciliation(snap, exchange)
        assert not report.passed
        assert report.mismatched_symbols == ["ETHUSDT"]

    def test_missing_internal_counted_as_mismatch(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange = [
            _make_exchange_position("BTCUSDT", 1.0),
            _make_exchange_position("ETHUSDT", 2.0),  # not in internal
        ]

        report = reconciler.run_full_reconciliation(snap, exchange)
        assert not report.passed
        assert report.mismatched_symbols == ["ETHUSDT"]

    def test_portfolio_balance_mismatch_excluded_from_symbols(self) -> None:
        """Balance mismatches use symbol='PORTFOLIO' and should be excluded."""
        from aiswarm.data.providers.aster import AccountBalance

        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange_pos = [_make_exchange_position("BTCUSDT", 1.0)]
        exchange_bal = AccountBalance(
            asset="USDT",
            total_balance=50_000.0,  # big mismatch vs NAV=100_000
            available_balance=40_000.0,
            unrealized_pnl=0.0,
            margin_balance=50_000.0,
        )

        report = reconciler.run_full_reconciliation(snap, exchange_pos, exchange_bal)
        assert not report.passed
        # Balance mismatch should NOT appear in mismatched_symbols
        assert report.mismatched_symbols == []

    def test_to_dict_includes_mismatched_symbols(self) -> None:
        reconciler = PositionReconciler(tolerance=0.001)
        snap = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange = [_make_exchange_position("BTCUSDT", 5.0)]

        report = reconciler.run_full_reconciliation(snap, exchange)
        d = report.to_dict()
        assert "mismatched_symbols" in d
        assert d["mismatched_symbols"] == ["BTCUSDT"]


# ---------------------------------------------------------------------------
# OrderStore.get_open_orders_for_symbol
# ---------------------------------------------------------------------------


class TestOrderStoreGetOpenOrdersForSymbol:
    def test_returns_only_orders_for_requested_symbol(self) -> None:
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")
        _submit_live_order(live, store, "btc2", "BTCUSDT")

        btc_orders = store.get_open_orders_for_symbol("BTCUSDT")
        eth_orders = store.get_open_orders_for_symbol("ETHUSDT")

        assert len(btc_orders) == 2
        assert all(r.order.symbol == "BTCUSDT" for r in btc_orders)
        assert len(eth_orders) == 1
        assert eth_orders[0].order.symbol == "ETHUSDT"

    def test_excludes_filled_orders(self) -> None:
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        store.record_fill("btc1", 50000.0, 0.1)  # now filled

        _submit_live_order(live, store, "btc2", "BTCUSDT")

        open_btc = store.get_open_orders_for_symbol("BTCUSDT")
        assert len(open_btc) == 1
        assert open_btc[0].order.order_id == "btc2"

    def test_returns_empty_for_unknown_symbol(self) -> None:
        live, store, _ = _make_live_executor()
        _submit_live_order(live, store, "btc1", "BTCUSDT")

        assert store.get_open_orders_for_symbol("XYZUSDT") == []


# ---------------------------------------------------------------------------
# LiveOrderExecutor.cancel_for_symbols
# ---------------------------------------------------------------------------


class TestCancelForSymbols:
    def test_cancels_only_target_symbol_orders(self) -> None:
        live, store, gateway = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")
        _submit_live_order(live, store, "btc2", "BTCUSDT")

        # Clear call history from submissions
        gateway.call_history.clear()

        results = live.cancel_for_symbols(["BTCUSDT"])
        assert all(r.success for r in results)

        # BTCUSDT orders should be cancelled
        btc1 = store.get("btc1")
        btc2 = store.get("btc2")
        assert btc1 is not None and btc1.order.status == OrderStatus.CANCELLED
        assert btc2 is not None and btc2.order.status == OrderStatus.CANCELLED

        # ETHUSDT order must remain SUBMITTED (untouched)
        eth1 = store.get("eth1")
        assert eth1 is not None and eth1.order.status == OrderStatus.SUBMITTED

    def test_cancels_multiple_symbols(self) -> None:
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")
        _submit_live_order(live, store, "sol1", "SOLUSDT")

        live.cancel_for_symbols(["BTCUSDT", "ETHUSDT"])

        btc1 = store.get("btc1")
        eth1 = store.get("eth1")
        sol1 = store.get("sol1")

        assert btc1 is not None and btc1.order.status == OrderStatus.CANCELLED
        assert eth1 is not None and eth1.order.status == OrderStatus.CANCELLED
        assert sol1 is not None and sol1.order.status == OrderStatus.SUBMITTED

    def test_empty_symbols_list_cancels_nothing(self) -> None:
        live, store, _ = _make_live_executor()
        _submit_live_order(live, store, "btc1", "BTCUSDT")

        results = live.cancel_for_symbols([])
        assert results == []

        btc1 = store.get("btc1")
        assert btc1 is not None and btc1.order.status == OrderStatus.SUBMITTED

    def test_cancel_reason_includes_symbol(self) -> None:
        live, store, _ = _make_live_executor()
        _submit_live_order(live, store, "btc1", "BTCUSDT")

        live.cancel_for_symbols(["BTCUSDT"])

        # Check the event store for the cancel reason
        events = store.event_store.get_events(event_type="order_cancelled", limit=10)
        assert len(events) >= 1
        payload = events[0]["payload"]
        assert "surgical_reconciliation_cancel:BTCUSDT" in payload["reason"]


# ---------------------------------------------------------------------------
# ReconciliationLoop surgical mismatch handling (end-to-end)
# ---------------------------------------------------------------------------


class TestSurgicalReconciliationLoop:
    """Integration tests verifying the full surgical reconciliation flow."""

    def test_mismatch_in_btcusdt_only_cancels_btcusdt_orders(self) -> None:
        """Core test: BTCUSDT mismatch cancels only BTCUSDT, not ETHUSDT."""
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")

        cancelled_symbols: list[list[str]] = []

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=_make_store(),
            pause_callback=lambda: None,
            mismatch_callback=lambda syms: (
                cancelled_symbols.append(syms),
                live.cancel_for_symbols(syms),
            )[-1],
        )

        snapshot = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
            )
        )
        exchange_positions = [
            _make_exchange_position("BTCUSDT", 5.0),  # MISMATCH
            _make_exchange_position("ETHUSDT", 2.0),  # match
        ]

        report = recon_loop.on_fill(snapshot, exchange_positions)

        assert not report.passed
        assert report.mismatched_symbols == ["BTCUSDT"]
        assert cancelled_symbols == [["BTCUSDT"]]

        # BTCUSDT order cancelled
        btc1 = store.get("btc1")
        assert btc1 is not None and btc1.order.status == OrderStatus.CANCELLED

        # ETHUSDT order untouched
        eth1 = store.get("eth1")
        assert eth1 is not None and eth1.order.status == OrderStatus.SUBMITTED

    def test_multiple_mismatches_cancel_all_mismatched_symbols(self) -> None:
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")
        _submit_live_order(live, store, "sol1", "SOLUSDT")

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=_make_store(),
            pause_callback=lambda: None,
            mismatch_callback=lambda syms: live.cancel_for_symbols(syms),
        )

        snapshot = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
                _make_position("SOLUSDT", 3.0),
            )
        )
        exchange_positions = [
            _make_exchange_position("BTCUSDT", 9.0),  # mismatch
            _make_exchange_position("ETHUSDT", 2.0),  # match
            _make_exchange_position("SOLUSDT", 9.0),  # mismatch
        ]

        report = recon_loop.on_fill(snapshot, exchange_positions)

        assert not report.passed
        assert report.mismatched_symbols == ["BTCUSDT", "SOLUSDT"]

        # Mismatched symbols cancelled
        btc1 = store.get("btc1")
        sol1 = store.get("sol1")
        assert btc1 is not None and btc1.order.status == OrderStatus.CANCELLED
        assert sol1 is not None and sol1.order.status == OrderStatus.CANCELLED

        # Matched symbol untouched
        eth1 = store.get("eth1")
        assert eth1 is not None and eth1.order.status == OrderStatus.SUBMITTED

    def test_no_mismatch_cancels_nothing(self) -> None:
        live, store, _ = _make_live_executor()

        _submit_live_order(live, store, "btc1", "BTCUSDT")
        _submit_live_order(live, store, "eth1", "ETHUSDT")

        mismatch_called: list[bool] = []
        pause_called: list[bool] = []

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=_make_store(),
            pause_callback=lambda: pause_called.append(True),
            mismatch_callback=lambda syms: mismatch_called.append(True),
        )

        snapshot = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
            )
        )
        exchange_positions = [
            _make_exchange_position("BTCUSDT", 1.0),  # match
            _make_exchange_position("ETHUSDT", 2.0),  # match
        ]

        report = recon_loop.on_fill(snapshot, exchange_positions)

        assert report.passed
        assert mismatch_called == []
        assert pause_called == []

        # All orders remain SUBMITTED
        btc1 = store.get("btc1")
        eth1 = store.get("eth1")
        assert btc1 is not None and btc1.order.status == OrderStatus.SUBMITTED
        assert eth1 is not None and eth1.order.status == OrderStatus.SUBMITTED

    def test_fallback_to_pause_callback_when_no_mismatch_callback(self) -> None:
        """Without mismatch_callback, falls back to nuclear pause_callback."""
        pause_called: list[bool] = []

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=_make_store(),
            pause_callback=lambda: pause_called.append(True),
            # No mismatch_callback
        )

        snapshot = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange_positions = [_make_exchange_position("BTCUSDT", 5.0)]

        report = recon_loop.on_fill(snapshot, exchange_positions)

        assert not report.passed
        assert len(pause_called) == 1  # nuclear fallback triggered

    def test_mismatch_persists_event_with_symbols(self) -> None:
        """Reconciliation pause event should include mismatched_symbols."""
        event_store = _make_store()

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=event_store,
            pause_callback=lambda: None,
            mismatch_callback=lambda syms: None,
        )

        snapshot = _make_snapshot(
            positions=(
                _make_position("BTCUSDT", 1.0),
                _make_position("ETHUSDT", 2.0),
            )
        )
        exchange_positions = [
            _make_exchange_position("BTCUSDT", 5.0),  # mismatch
            _make_exchange_position("ETHUSDT", 2.0),  # match
        ]

        recon_loop.on_fill(snapshot, exchange_positions)

        pause_events = event_store.get_events(event_type="reconciliation_pause", limit=10)
        assert len(pause_events) >= 1
        payload = pause_events[0]["payload"]
        assert payload["mismatched_symbols"] == ["BTCUSDT"]

    def test_periodic_check_uses_surgical_callback(self) -> None:
        """run_periodic_check also uses the surgical path."""
        cancelled_symbols: list[list[str]] = []

        recon_loop = ReconciliationLoop(
            reconciler=PositionReconciler(provider=None, tolerance=0.001),
            event_store=_make_store(),
            pause_callback=lambda: None,
            mismatch_callback=lambda syms: cancelled_symbols.append(syms),
        )

        snapshot = _make_snapshot(positions=(_make_position("BTCUSDT", 1.0),))
        exchange_positions = [_make_exchange_position("BTCUSDT", 5.0)]

        report = recon_loop.run_periodic_check(snapshot, exchange_positions)

        assert not report.passed
        assert cancelled_symbols == [["BTCUSDT"]]
