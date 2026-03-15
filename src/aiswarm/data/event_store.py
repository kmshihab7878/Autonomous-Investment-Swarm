"""Event store for decision and order persistence.

Uses SQLite for dev simplicity (no external dependencies). Provides:
- Append-only event log for all decisions, orders, and risk events
- Portfolio snapshot checkpointing for state recovery
- Query interface for replay and attribution analysis

All writes are append-only — events are never modified or deleted.
This is the foundation for replay, audit trail, and recovery.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

DEFAULT_DB_PATH = "data/ais_events.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    source TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON checkpoints(checkpoint_type);
"""


class EventStore:
    """Append-only event store backed by SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- Event operations ---

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        source: str = "",
        timestamp: datetime | None = None,
    ) -> int:
        """Append an event to the store. Returns the event ID."""
        ts = timestamp or utc_now()
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO events (event_type, timestamp, payload, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    event_type,
                    ts.isoformat(),
                    json.dumps(payload, default=str),
                    source,
                    utc_now().isoformat(),
                ),
            )
            event_id = cursor.lastrowid or 0
        logger.info(
            "Event appended",
            extra={"extra_json": {"event_id": event_id, "type": event_type}},
        )
        return event_id

    def append_decision(self, decision: dict[str, Any]) -> int:
        """Append a decision log event."""
        return self.append("decision", decision, source="coordinator")

    def append_order(self, order: dict[str, Any]) -> int:
        """Append an order event."""
        return self.append("order", order, source="oms")

    def append_risk_event(self, risk_event: dict[str, Any]) -> int:
        """Append a risk event."""
        return self.append("risk_event", risk_event, source="risk_engine")

    def append_fill(self, fill: dict[str, Any]) -> int:
        """Append an execution fill event."""
        return self.append("fill", fill, source="executor")

    def append_reconciliation(self, result: dict[str, Any]) -> int:
        """Append a reconciliation result event."""
        return self.append("reconciliation", result, source="reconciliation")

    # --- Query operations ---

    def get_events(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query events with optional type and time filters."""
        query = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "event_type": row["event_type"],
                "timestamp": row["timestamp"],
                "payload": json.loads(row["payload"]),
                "source": row["source"],
            }
            for row in rows
        ]

    def get_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent decisions."""
        return self.get_events(event_type="decision", limit=limit)

    def get_orders(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent orders."""
        return self.get_events(event_type="order", limit=limit)

    def count_events(self, event_type: str | None = None) -> int:
        """Count events, optionally filtered by type."""
        if event_type:
            query = "SELECT COUNT(*) FROM events WHERE event_type = ?"
            params: tuple[Any, ...] = (event_type,)
        else:
            query = "SELECT COUNT(*) FROM events"
            params = ()

        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
        return row[0] if row else 0

    # --- Checkpoint operations ---

    def save_checkpoint(
        self,
        checkpoint_type: str,
        payload: dict[str, Any],
    ) -> int:
        """Save a state checkpoint for recovery."""
        ts = utc_now()
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO checkpoints (checkpoint_type, timestamp, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    checkpoint_type,
                    ts.isoformat(),
                    json.dumps(payload, default=str),
                    ts.isoformat(),
                ),
            )
            cp_id = cursor.lastrowid or 0
        logger.info(
            "Checkpoint saved",
            extra={"extra_json": {"checkpoint_id": cp_id, "type": checkpoint_type}},
        )
        return cp_id

    def load_latest_checkpoint(
        self,
        checkpoint_type: str,
    ) -> dict[str, Any] | None:
        """Load the most recent checkpoint of a given type."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload, timestamp FROM checkpoints "
                "WHERE checkpoint_type = ? ORDER BY id DESC LIMIT 1",
                (checkpoint_type,),
            ).fetchone()

        if row is None:
            return None

        return {
            "payload": json.loads(row["payload"]),
            "timestamp": row["timestamp"],
        }

    def save_portfolio_checkpoint(self, snapshot: dict[str, Any]) -> int:
        """Save a portfolio state checkpoint."""
        return self.save_checkpoint("portfolio", snapshot)

    def load_portfolio_checkpoint(self) -> dict[str, Any] | None:
        """Load the latest portfolio checkpoint."""
        return self.load_latest_checkpoint("portfolio")

    def save_memory_checkpoint(self, memory_state: dict[str, Any]) -> int:
        """Save SharedMemory state for recovery."""
        return self.save_checkpoint("shared_memory", memory_state)

    def load_memory_checkpoint(self) -> dict[str, Any] | None:
        """Load the latest SharedMemory checkpoint."""
        return self.load_latest_checkpoint("shared_memory")
