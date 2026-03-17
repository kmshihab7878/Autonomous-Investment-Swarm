"""Tests for OrderStore crash recovery via event replay.

Validates that in-memory order state can be faithfully reconstructed
from persisted EventStore events after a process restart.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone

from aiswarm.data.event_store import EventStore
from aiswarm.execution.order_store import OrderStore
from aiswarm.types.orders import Order, OrderStatus, Side


def _make_event_store() -> EventStore:
    """Create a fresh EventStore backed by a temporary SQLite database."""
    return EventStore(tempfile.mktemp(suffix=".db"))


def _make_order(
    order_id: str = "o1",
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    status: OrderStatus = OrderStatus.APPROVED,
) -> Order:
    """Create a test Order with all required fields populated."""
    return Order(
        order_id=order_id,
        signal_id=f"sig_{order_id}",
        symbol=symbol,
        side=side,
        quantity=0.1,
        limit_price=None,
        notional=5000.0,
        strategy="momentum_ma_crossover",
        thesis="valid test thesis for recovery",
        created_at=datetime.now(timezone.utc),
        risk_approval_token="fake_token",
        mandate_id="m1",
        status=status,
    )


class TestRestoreFromEvents:
    """Tests for OrderStore.restore_from_events()."""

    def test_empty_event_store_produces_empty_order_store(self) -> None:
        """Restoring from an empty EventStore yields no orders."""
        es = _make_event_store()
        store = OrderStore(es)
        restored = store.restore_from_events()

        assert restored == 0
        assert store.get_all() == []
        assert store.known_exchange_ids == set()

    def test_restore_pending_order(self) -> None:
        """A tracked-but-not-submitted order is restored as PENDING-like."""
        es = _make_event_store()

        # Session 1: track an order then "crash"
        store1 = OrderStore(es)
        order = _make_order("o1")
        store1.track(order, venue="futures")

        # Session 2: new OrderStore from same EventStore
        store2 = OrderStore(es)
        restored = store2.restore_from_events()

        assert restored == 1
        record = store2.get("o1")
        assert record is not None
        assert record.order.order_id == "o1"
        assert record.order.symbol == "BTCUSDT"
        assert record.order.side == Side.BUY
        assert record.order.quantity == 0.1
        assert record.order.notional == 5000.0
        assert record.order.strategy == "momentum_ma_crossover"
        assert record.order.mandate_id == "m1"
        assert record.venue == "futures"
        assert record.exchange_order_id is None

    def test_restore_submitted_order(self) -> None:
        """A submitted order is restored with exchange ID and SUBMITTED status."""
        es = _make_event_store()

        # Session 1
        store1 = OrderStore(es)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")

        # Session 2
        store2 = OrderStore(es)
        restored = store2.restore_from_events()

        assert restored == 1
        record = store2.get("o1")
        assert record is not None
        assert record.order.status == OrderStatus.SUBMITTED
        assert record.exchange_order_id == "EX001"

        # Exchange ID reverse lookup should also work
        by_exchange = store2.get_by_exchange_id("EX001")
        assert by_exchange is not None
        assert by_exchange.order.order_id == "o1"
        assert "EX001" in store2.known_exchange_ids

    def test_filled_order_not_restored(self) -> None:
        """Orders in terminal FILLED state are pruned during restore."""
        es = _make_event_store()

        # Session 1: track -> submit -> fill
        store1 = OrderStore(es)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")
        store1.record_fill("o1", fill_price=50000.0, fill_quantity=0.1)

        # Session 2
        store2 = OrderStore(es)
        restored = store2.restore_from_events()

        assert restored == 0
        assert store2.get("o1") is None
        assert store2.get_by_exchange_id("EX001") is None

    def test_cancelled_order_not_restored(self) -> None:
        """Orders in terminal CANCELLED state are pruned during restore."""
        es = _make_event_store()

        # Session 1: track -> submit -> cancel
        store1 = OrderStore(es)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")
        store1.record_cancel("o1", reason="timeout")

        # Session 2
        store2 = OrderStore(es)
        restored = store2.restore_from_events()

        assert restored == 0
        assert store2.get("o1") is None

    def test_mixed_orders_only_active_restored(self) -> None:
        """Only non-terminal orders survive restore from a mix of states."""
        es = _make_event_store()

        # Session 1: create orders in different states
        store1 = OrderStore(es)

        # o1: filled (terminal)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")
        store1.record_fill("o1", fill_price=50000.0, fill_quantity=0.1)

        # o2: submitted (active — should survive)
        store1.track(_make_order("o2"))
        store1.record_submission("o2", "EX002")

        # o3: cancelled (terminal)
        store1.track(_make_order("o3"))
        store1.record_submission("o3", "EX003")
        store1.record_cancel("o3", reason="emergency")

        # o4: tracked but not submitted (active — should survive)
        store1.track(_make_order("o4"))

        # Session 2
        store2 = OrderStore(es)
        restored = store2.restore_from_events()

        assert restored == 2
        assert store2.get("o1") is None  # filled
        assert store2.get("o2") is not None  # submitted
        assert store2.get("o3") is None  # cancelled
        assert store2.get("o4") is not None  # tracked

        assert store2.get("o2").order.status == OrderStatus.SUBMITTED
        assert store2.get("o2").exchange_order_id == "EX002"
        assert "EX002" in store2.known_exchange_ids
        assert "EX001" not in store2.known_exchange_ids
        assert "EX003" not in store2.known_exchange_ids

    def test_restored_state_matches_original(self) -> None:
        """Restored order fields match what was originally tracked."""
        es = _make_event_store()
        original_order = _make_order(
            "o1",
            symbol="ETHUSDT",
            side=Side.SELL,
        )

        # Session 1
        store1 = OrderStore(es)
        store1.track(original_order, venue="spot")
        store1.record_submission("o1", "EX_ETH_001")

        # Session 2
        store2 = OrderStore(es)
        store2.restore_from_events()

        record = store2.get("o1")
        assert record is not None
        assert record.order.symbol == "ETHUSDT"
        assert record.order.side == Side.SELL
        assert record.order.quantity == 0.1
        assert record.order.notional == 5000.0
        assert record.order.strategy == "momentum_ma_crossover"
        assert record.order.thesis == "valid test thesis for recovery"
        assert record.order.signal_id == "sig_o1"
        assert record.order.mandate_id == "m1"
        assert record.venue == "spot"
        assert record.exchange_order_id == "EX_ETH_001"
        assert record.order.status == OrderStatus.SUBMITTED

    def test_open_orders_available_after_restore(self) -> None:
        """get_open_orders() returns submitted orders after restore."""
        es = _make_event_store()

        # Session 1: two submitted orders
        store1 = OrderStore(es)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")
        store1.track(_make_order("o2"))
        store1.record_submission("o2", "EX002")

        # Session 2
        store2 = OrderStore(es)
        store2.restore_from_events()

        open_orders = store2.get_open_orders()
        assert len(open_orders) == 2
        open_ids = {r.order.order_id for r in open_orders}
        assert open_ids == {"o1", "o2"}

    def test_restore_is_idempotent(self) -> None:
        """Calling restore_from_events() twice yields the same state."""
        es = _make_event_store()

        store1 = OrderStore(es)
        store1.track(_make_order("o1"))
        store1.record_submission("o1", "EX001")

        store2 = OrderStore(es)
        first_count = store2.restore_from_events()
        second_count = store2.restore_from_events()

        assert first_count == second_count == 1
        assert store2.get("o1") is not None

    def test_restore_handles_legacy_events_without_enriched_fields(self) -> None:
        """Events from before enrichment (missing signal_id, strategy, etc.)
        are handled gracefully with sensible defaults."""
        es = _make_event_store()

        # Manually insert a minimal legacy-style order_tracked event
        es.append(
            "order_tracked",
            {
                "order_id": "legacy_01",
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 0.5,
                "notional": 25000.0,
                "mandate_id": None,
                "venue": "futures",
                # No signal_id, strategy, thesis, created_at, etc.
            },
            source="order_store",
        )

        store = OrderStore(es)
        restored = store.restore_from_events()

        assert restored == 1
        record = store.get("legacy_01")
        assert record is not None
        assert record.order.symbol == "BTCUSDT"
        assert record.order.quantity == 0.5
        # Defaults should be applied for missing fields
        assert record.order.signal_id == ""
        assert record.order.strategy == "unknown"
        assert record.order.thesis == "restored from event log"


class TestPersistSnapshot:
    """Tests for OrderStore.persist_snapshot()."""

    def test_snapshot_saves_active_orders(self) -> None:
        """persist_snapshot() creates a checkpoint with active orders."""
        es = _make_event_store()
        store = OrderStore(es)

        store.track(_make_order("o1"))
        store.record_submission("o1", "EX001")

        checkpoint_id = store.persist_snapshot()
        assert checkpoint_id > 0

        # Verify checkpoint was saved
        cp = es.load_latest_checkpoint("order_store")
        assert cp is not None
        payload = cp["payload"]
        assert "orders" in payload
        assert len(payload["orders"]) == 1
        assert payload["orders"][0]["order_id"] == "o1"
        assert payload["orders"][0]["exchange_order_id"] == "EX001"

    def test_snapshot_excludes_terminal_orders(self) -> None:
        """persist_snapshot() does not include filled or cancelled orders."""
        es = _make_event_store()
        store = OrderStore(es)

        # Active order
        store.track(_make_order("o1"))
        store.record_submission("o1", "EX001")

        # Filled order
        store.track(_make_order("o2"))
        store.record_submission("o2", "EX002")
        store.record_fill("o2", 50000.0, 0.1)

        checkpoint_id = store.persist_snapshot()
        assert checkpoint_id > 0

        cp = es.load_latest_checkpoint("order_store")
        assert cp is not None
        orders = cp["payload"]["orders"]
        assert len(orders) == 1
        assert orders[0]["order_id"] == "o1"

    def test_snapshot_empty_store(self) -> None:
        """persist_snapshot() on empty store creates a checkpoint with no orders."""
        es = _make_event_store()
        store = OrderStore(es)

        checkpoint_id = store.persist_snapshot()
        assert checkpoint_id > 0

        cp = es.load_latest_checkpoint("order_store")
        assert cp is not None
        assert cp["payload"]["orders"] == []
