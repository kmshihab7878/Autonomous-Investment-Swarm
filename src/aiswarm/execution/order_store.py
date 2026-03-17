"""Persistent order store — tracks order lifecycle and exchange ID mapping.

Maps internal order IDs to exchange order IDs and tracks all state transitions.
Uses EventStore for persistence.  Supports crash recovery via
``restore_from_events()`` which replays the event log to rebuild in-memory
state after a process restart.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aiswarm.data.event_store import EventStore
from aiswarm.types.orders import Order, OrderStatus, Side
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

# Event types emitted by OrderStore — used for replay during recovery.
_ORDER_EVENT_TYPES = (
    "order_tracked",
    "order_submitted",
    "fill",
    "order_cancelled",
)

# Terminal statuses that do not require active tracking after restore.
_TERMINAL_STATUSES = frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED})


@dataclass
class OrderRecord:
    """Tracked order with exchange mapping."""

    order: Order
    exchange_order_id: str | None = None
    fill_price: float | None = None
    fill_quantity: float | None = None
    venue: str = "futures"
    submitted_at: float = 0.0


class OrderStore:
    """In-memory order store with EventStore persistence."""

    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store
        self._orders: dict[str, OrderRecord] = {}
        self._exchange_map: dict[str, str] = {}  # exchange_id -> internal_id

    def track(self, order: Order, venue: str = "futures") -> OrderRecord:
        """Start tracking an order."""
        record = OrderRecord(order=order, venue=venue)
        self._orders[order.order_id] = record
        self.event_store.append(
            "order_tracked",
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "notional": order.notional,
                "mandate_id": order.mandate_id,
                "venue": venue,
                # Fields required for crash-recovery reconstruction:
                "signal_id": order.signal_id,
                "strategy": order.strategy,
                "thesis": order.thesis,
                "created_at": order.created_at.isoformat(),
                "limit_price": order.limit_price,
                "risk_approval_token": order.risk_approval_token,
                "status": order.status.value,
            },
            source="order_store",
        )
        return record

    def record_submission(self, order_id: str, exchange_order_id: str) -> OrderRecord | None:
        """Record that an order was submitted to the exchange."""
        record = self._orders.get(order_id)
        if record is None:
            return None
        record.exchange_order_id = exchange_order_id
        record.submitted_at = time.monotonic()
        record.order = record.order.model_copy(update={"status": OrderStatus.SUBMITTED})
        self._exchange_map[exchange_order_id] = order_id
        self.event_store.append(
            "order_submitted",
            {
                "order_id": order_id,
                "exchange_order_id": exchange_order_id,
                "symbol": record.order.symbol,
            },
            source="order_store",
        )
        logger.info(
            "Order submitted to exchange",
            extra={
                "extra_json": {
                    "order_id": order_id,
                    "exchange_order_id": exchange_order_id,
                }
            },
        )
        return record

    def record_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_quantity: float,
    ) -> OrderRecord | None:
        """Record that an order was filled."""
        record = self._orders.get(order_id)
        if record is None:
            return None
        record.fill_price = fill_price
        record.fill_quantity = fill_quantity
        record.order = record.order.model_copy(update={"status": OrderStatus.FILLED})
        self.event_store.append(
            "fill",
            {
                "order_id": order_id,
                "exchange_order_id": record.exchange_order_id,
                "symbol": record.order.symbol,
                "side": record.order.side.value,
                "fill_price": fill_price,
                "fill_quantity": fill_quantity,
                "mandate_id": record.order.mandate_id,
                "pnl": 0.0,
            },
            source="order_store",
        )
        logger.info(
            "Order filled",
            extra={
                "extra_json": {
                    "order_id": order_id,
                    "price": fill_price,
                    "quantity": fill_quantity,
                }
            },
        )
        return record

    def record_cancel(self, order_id: str, reason: str = "") -> OrderRecord | None:
        """Record that an order was cancelled."""
        record = self._orders.get(order_id)
        if record is None:
            return None
        record.order = record.order.model_copy(update={"status": OrderStatus.CANCELLED})
        self.event_store.append(
            "order_cancelled",
            {
                "order_id": order_id,
                "exchange_order_id": record.exchange_order_id,
                "reason": reason,
            },
            source="order_store",
        )
        return record

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def restore_from_events(self) -> int:
        """Rebuild in-memory order state by replaying persisted events.

        Queries all order-lifecycle events from EventStore in chronological
        order and reconstructs ``_orders`` and ``_exchange_map``.  Orders
        that reached a terminal state (FILLED, CANCELLED) are discarded so
        only potentially-orphaned orders remain for active tracking.

        Returns:
            The number of non-terminal orders restored (PENDING or SUBMITTED).
        """
        events = self._query_order_events_chronological()
        if not events:
            logger.info("No order events found — nothing to restore")
            return 0

        # Replay events in chronological order to reconstruct state.
        for event in events:
            event_type: str = event["event_type"]
            payload: dict[str, Any] = event["payload"]
            self._apply_event(event_type, payload)

        # Prune terminal orders — they don't need active tracking.
        terminal_ids = [
            oid for oid, rec in self._orders.items() if rec.order.status in _TERMINAL_STATUSES
        ]
        for oid in terminal_ids:
            rec = self._orders.pop(oid)
            # Also clean exchange map for terminal orders.
            if rec.exchange_order_id and rec.exchange_order_id in self._exchange_map:
                del self._exchange_map[rec.exchange_order_id]

        restored_count = len(self._orders)
        logger.info(
            "Order store restored from events",
            extra={
                "extra_json": {
                    "restored": restored_count,
                    "terminal_pruned": len(terminal_ids),
                    "total_events_replayed": len(events),
                }
            },
        )
        return restored_count

    def persist_snapshot(self) -> int:
        """Save a checkpoint of current order state for faster recovery.

        Serialises all non-terminal orders into the EventStore checkpoint
        table so that future restores can start from the checkpoint rather
        than replaying the entire event log.

        Returns:
            The checkpoint ID.
        """
        records: list[dict[str, Any]] = []
        for oid, rec in self._orders.items():
            if rec.order.status in _TERMINAL_STATUSES:
                continue
            records.append(
                {
                    "order_id": oid,
                    "symbol": rec.order.symbol,
                    "side": rec.order.side.value,
                    "quantity": rec.order.quantity,
                    "notional": rec.order.notional,
                    "mandate_id": rec.order.mandate_id,
                    "venue": rec.venue,
                    "signal_id": rec.order.signal_id,
                    "strategy": rec.order.strategy,
                    "thesis": rec.order.thesis,
                    "created_at": rec.order.created_at.isoformat(),
                    "limit_price": rec.order.limit_price,
                    "risk_approval_token": rec.order.risk_approval_token,
                    "status": rec.order.status.value,
                    "exchange_order_id": rec.exchange_order_id,
                    "fill_price": rec.fill_price,
                    "fill_quantity": rec.fill_quantity,
                }
            )

        checkpoint_id = self.event_store.save_checkpoint(
            "order_store",
            {"orders": records},
        )
        logger.info(
            "Order store snapshot persisted",
            extra={
                "extra_json": {
                    "checkpoint_id": checkpoint_id,
                    "order_count": len(records),
                }
            },
        )
        return checkpoint_id

    # ------------------------------------------------------------------
    # Internal helpers for recovery
    # ------------------------------------------------------------------

    def _query_order_events_chronological(self) -> list[dict[str, Any]]:
        """Fetch all order-lifecycle events in chronological (ASC) order.

        The default ``EventStore.get_events`` returns DESC with a limit,
        which is unsuitable for full replay.  This method queries directly
        to get all events in insertion order.
        """
        placeholders = ",".join("?" for _ in _ORDER_EVENT_TYPES)
        query = f"SELECT * FROM events WHERE event_type IN ({placeholders}) ORDER BY id ASC"
        results: list[dict[str, Any]] = []
        with self.event_store._conn() as conn:
            rows = conn.execute(query, _ORDER_EVENT_TYPES).fetchall()
            for row in rows:
                results.append(
                    {
                        "id": row["id"],
                        "event_type": row["event_type"],
                        "timestamp": row["timestamp"],
                        "payload": json.loads(row["payload"]),
                        "source": row["source"],
                    }
                )
        return results

    def _apply_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Apply a single event to in-memory state during replay."""
        order_id: str = payload.get("order_id", "")
        if not order_id:
            return

        if event_type == "order_tracked":
            self._replay_tracked(order_id, payload)
        elif event_type == "order_submitted":
            self._replay_submitted(order_id, payload)
        elif event_type == "fill":
            self._replay_fill(order_id, payload)
        elif event_type == "order_cancelled":
            self._replay_cancelled(order_id, payload)

    def _replay_tracked(self, order_id: str, payload: dict[str, Any]) -> None:
        """Reconstruct an Order from an order_tracked event payload."""
        # Parse created_at — handle both ISO strings and missing values.
        created_at_raw = payload.get("created_at")
        if created_at_raw:
            created_at = datetime.fromisoformat(str(created_at_raw))
        else:
            created_at = datetime.now(timezone.utc)

        # Parse status — default to PENDING for older events without this field.
        status_raw = payload.get("status", OrderStatus.PENDING.value)
        try:
            status = OrderStatus(status_raw)
        except ValueError:
            status = OrderStatus.PENDING

        # Parse side.
        side_raw = payload.get("side", Side.BUY.value)
        try:
            side = Side(side_raw)
        except ValueError:
            side = Side.BUY

        order = Order(
            order_id=order_id,
            signal_id=payload.get("signal_id", ""),
            symbol=payload.get("symbol", ""),
            side=side,
            quantity=payload.get("quantity", 0.01),
            limit_price=payload.get("limit_price"),
            notional=payload.get("notional", 0.01),
            strategy=payload.get("strategy", "unknown"),
            thesis=payload.get("thesis", "restored from event log"),
            mandate_id=payload.get("mandate_id"),
            risk_approval_token=payload.get("risk_approval_token"),
            status=status,
            created_at=created_at,
        )
        venue = payload.get("venue", "futures")
        self._orders[order_id] = OrderRecord(order=order, venue=venue)

    def _replay_submitted(self, order_id: str, payload: dict[str, Any]) -> None:
        """Apply a submission event to an existing tracked order."""
        record = self._orders.get(order_id)
        if record is None:
            return
        exchange_order_id = payload.get("exchange_order_id", "")
        record.exchange_order_id = exchange_order_id
        # Use wall-clock time for restored orders so stale-order detection
        # still works (monotonic times are meaningless across restarts).
        record.submitted_at = time.monotonic()
        record.order = record.order.model_copy(update={"status": OrderStatus.SUBMITTED})
        if exchange_order_id:
            self._exchange_map[exchange_order_id] = order_id

    def _replay_fill(self, order_id: str, payload: dict[str, Any]) -> None:
        """Apply a fill event to an existing tracked order."""
        record = self._orders.get(order_id)
        if record is None:
            return
        record.fill_price = payload.get("fill_price")
        record.fill_quantity = payload.get("fill_quantity")
        record.order = record.order.model_copy(update={"status": OrderStatus.FILLED})

    def _replay_cancelled(self, order_id: str, payload: dict[str, Any]) -> None:
        """Apply a cancellation event to an existing tracked order."""
        record = self._orders.get(order_id)
        if record is None:
            return
        record.order = record.order.model_copy(update={"status": OrderStatus.CANCELLED})

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, order_id: str) -> OrderRecord | None:
        """Get an order record by internal ID."""
        return self._orders.get(order_id)

    def get_by_exchange_id(self, exchange_order_id: str) -> OrderRecord | None:
        """Look up an order by its exchange order ID."""
        internal_id = self._exchange_map.get(exchange_order_id)
        if internal_id is None:
            return None
        return self._orders.get(internal_id)

    def get_open_orders(self) -> list[OrderRecord]:
        """Get all orders in SUBMITTED status (not yet filled or cancelled)."""
        return [r for r in self._orders.values() if r.order.status == OrderStatus.SUBMITTED]

    def get_open_orders_for_symbol(self, symbol: str) -> list[OrderRecord]:
        """Get open orders filtered to a specific symbol."""
        return [
            r
            for r in self._orders.values()
            if r.order.status == OrderStatus.SUBMITTED and r.order.symbol == symbol
        ]

    def get_stale_orders(self, max_age_seconds: float = 300.0) -> list[OrderRecord]:
        """Get submitted orders older than max_age_seconds."""
        now = time.monotonic()
        return [
            r
            for r in self._orders.values()
            if r.order.status == OrderStatus.SUBMITTED
            and r.submitted_at > 0
            and (now - r.submitted_at) > max_age_seconds
        ]

    def get_all(self) -> list[OrderRecord]:
        """Get all tracked orders."""
        return list(self._orders.values())

    @property
    def known_exchange_ids(self) -> set[str]:
        """All exchange order IDs we know about."""
        return set(self._exchange_map.keys())
