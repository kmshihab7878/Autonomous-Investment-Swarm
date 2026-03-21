"""Persistence layer for Alpha Intelligence data.

Uses SQLite (same pattern as EventStore) to store trader profiles,
trade activities, strategy fingerprints, and leaderboard snapshots.
All writes are append-only or upsert — trader profiles are updated
in place as new data arrives.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from aiswarm.intelligence.models import (
    LeaderboardEntry,
    StrategyFingerprint,
    TradeActivity,
    TraderProfile,
    TraderTier,
)
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

DEFAULT_ALPHA_DB_PATH = "data/ais_alpha.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trader_profiles (
    trader_id TEXT PRIMARY KEY,
    exchange TEXT NOT NULL,
    display_name TEXT DEFAULT '',
    tier TEXT DEFAULT 'average',
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_exchange ON trader_profiles(exchange);
CREATE INDEX IF NOT EXISTS idx_profiles_tier ON trader_profiles(tier);

CREATE TABLE IF NOT EXISTS trade_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id TEXT NOT NULL UNIQUE,
    trader_id TEXT NOT NULL,
    exchange TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    timestamp TEXT NOT NULL,
    source TEXT NOT NULL,
    pnl REAL,
    holding_minutes INTEGER,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activities_trader ON trade_activities(trader_id);
CREATE INDEX IF NOT EXISTS idx_activities_symbol ON trade_activities(symbol);
CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON trade_activities(timestamp);

CREATE TABLE IF NOT EXISTS strategy_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trader_id TEXT NOT NULL,
    style TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fingerprints_trader ON strategy_fingerprints(trader_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_style ON strategy_fingerprints(style);

CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trader_id TEXT NOT NULL,
    exchange TEXT NOT NULL,
    rank INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_time TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_leaderboard_exchange ON leaderboard_snapshots(exchange);
CREATE INDEX IF NOT EXISTS idx_leaderboard_time ON leaderboard_snapshots(snapshot_time);
"""


class AlphaStore:
    """SQLite-backed store for Alpha Intelligence data."""

    def __init__(self, db_path: str = DEFAULT_ALPHA_DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- Trader Profiles ---

    def upsert_profile(self, profile: TraderProfile) -> None:
        """Insert or update a trader profile."""
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO trader_profiles
                   (trader_id, exchange, display_name, tier, profile_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trader_id) DO UPDATE SET
                     exchange = excluded.exchange,
                     display_name = excluded.display_name,
                     tier = excluded.tier,
                     profile_json = excluded.profile_json,
                     updated_at = excluded.updated_at""",
                (
                    profile.trader_id,
                    profile.exchange,
                    profile.display_name,
                    profile.tier.value,
                    profile.model_dump_json(),
                    now,
                    now,
                ),
            )

    def get_profile(self, trader_id: str) -> TraderProfile | None:
        """Get a single trader profile."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM trader_profiles WHERE trader_id = ?",
                (trader_id,),
            ).fetchone()
        if row is None:
            return None
        return TraderProfile.model_validate_json(row["profile_json"])

    def get_top_traders(
        self,
        tier: TraderTier | None = None,
        exchange: str | None = None,
        limit: int = 50,
    ) -> list[TraderProfile]:
        """Get top-ranked traders, optionally filtered by tier or exchange."""
        query = "SELECT profile_json FROM trader_profiles WHERE 1=1"
        params: list[Any] = []
        if tier is not None:
            query += " AND tier = ?"
            params.append(tier.value)
        if exchange is not None:
            query += " AND exchange = ?"
            params.append(exchange)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [TraderProfile.model_validate_json(r["profile_json"]) for r in rows]

    # --- Trade Activities ---

    def append_activity(self, activity: TradeActivity) -> None:
        """Record a new trade activity observation."""
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO trade_activities
                   (activity_id, trader_id, exchange, symbol, side, quantity,
                    price, notional, timestamp, source, pnl, holding_minutes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity.activity_id,
                    activity.trader_id,
                    activity.exchange,
                    activity.symbol,
                    activity.side,
                    activity.quantity,
                    activity.price,
                    activity.notional,
                    activity.timestamp.isoformat(),
                    activity.source.value,
                    activity.pnl,
                    activity.holding_minutes,
                    now,
                ),
            )

    def get_activities(
        self,
        trader_id: str | None = None,
        symbol: str | None = None,
        limit: int = 500,
    ) -> list[TradeActivity]:
        """Query trade activities with optional filters."""
        query = "SELECT * FROM trade_activities WHERE 1=1"
        params: list[Any] = []
        if trader_id is not None:
            query += " AND trader_id = ?"
            params.append(trader_id)
        if symbol is not None:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for r in rows:
            results.append(
                TradeActivity(
                    activity_id=r["activity_id"],
                    trader_id=r["trader_id"],
                    exchange=r["exchange"],
                    symbol=r["symbol"],
                    side=r["side"],
                    quantity=r["quantity"],
                    price=r["price"],
                    notional=r["notional"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    source=r["source"],
                    pnl=r["pnl"],
                    holding_minutes=r["holding_minutes"],
                )
            )
        return results

    def get_activity_count(self, trader_id: str) -> int:
        """Get total trade count for a trader."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trade_activities WHERE trader_id = ?",
                (trader_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    # --- Strategy Fingerprints ---

    def save_fingerprint(self, fingerprint: StrategyFingerprint) -> None:
        """Save a strategy fingerprint for a trader."""
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO strategy_fingerprints
                   (trader_id, style, fingerprint_json, sample_size, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    fingerprint.trader_id,
                    fingerprint.style.value,
                    fingerprint.model_dump_json(),
                    fingerprint.sample_size,
                    fingerprint.confidence,
                    now,
                ),
            )

    def get_latest_fingerprint(self, trader_id: str) -> StrategyFingerprint | None:
        """Get the most recent fingerprint for a trader."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT fingerprint_json FROM strategy_fingerprints
                   WHERE trader_id = ? ORDER BY created_at DESC LIMIT 1""",
                (trader_id,),
            ).fetchone()
        if row is None:
            return None
        return StrategyFingerprint.model_validate_json(row["fingerprint_json"])

    # --- Leaderboard Snapshots ---

    def save_leaderboard_snapshot(self, entry: LeaderboardEntry) -> None:
        """Record a leaderboard snapshot."""
        now = utc_now().isoformat()
        snapshot_time = entry.snapshot_time.isoformat() if entry.snapshot_time else now
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO leaderboard_snapshots
                   (trader_id, exchange, rank, snapshot_json, snapshot_time, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry.trader_id,
                    entry.exchange,
                    entry.rank,
                    entry.model_dump_json(),
                    snapshot_time,
                    now,
                ),
            )

    def get_rank_history(self, trader_id: str, limit: int = 30) -> list[LeaderboardEntry]:
        """Get historical rank snapshots for a trader."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT snapshot_json FROM leaderboard_snapshots
                   WHERE trader_id = ? ORDER BY snapshot_time DESC LIMIT ?""",
                (trader_id, limit),
            ).fetchall()
        return [LeaderboardEntry.model_validate_json(r["snapshot_json"]) for r in rows]
