"""Leaderboard Tracker — monitors exchange copy-trading leaderboards.

Periodically captures leaderboard snapshots from exchanges that support
copy-trading (Binance, Bybit, Bitget). Tracks rank changes over time
and identifies consistently top-performing traders.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import LeaderboardEntry, TraderTier
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class LeaderboardTracker:
    """Tracks copy-trading leaderboard positions across exchanges.

    Captures periodic snapshots, detects rank changes, and identifies
    traders who consistently maintain top positions.
    """

    def __init__(self, store: AlphaStore) -> None:
        self.store = store
        self._latest_entries: dict[
            str, dict[str, LeaderboardEntry]
        ] = {}  # exchange → {trader_id → entry}

    def ingest_leaderboard(
        self,
        exchange: str,
        entries: list[dict[str, Any]],
    ) -> list[LeaderboardEntry]:
        """Process a raw leaderboard response from an exchange.

        Args:
            exchange: Exchange identifier (e.g., "binance", "bybit")
            entries: Raw leaderboard data (exchange-specific format)

        Returns:
            Parsed LeaderboardEntry objects that were persisted.
        """
        now = utc_now()
        parsed: list[LeaderboardEntry] = []

        for raw in entries:
            entry = self._parse_entry(exchange, raw, now)
            if entry is None:
                continue

            parsed.append(entry)
            self.store.save_leaderboard_snapshot(entry)

        # Update latest state
        if exchange not in self._latest_entries:
            self._latest_entries[exchange] = {}
        for entry in parsed:
            self._latest_entries[exchange][entry.trader_id] = entry

        logger.info(
            "Leaderboard snapshot captured",
            extra={
                "extra_json": {
                    "exchange": exchange,
                    "traders_captured": len(parsed),
                }
            },
        )
        return parsed

    def _parse_entry(
        self,
        exchange: str,
        raw: dict[str, Any],
        snapshot_time: datetime,
    ) -> LeaderboardEntry | None:
        """Parse a raw leaderboard entry into canonical format.

        Supports multiple exchange formats. Returns None if the entry
        cannot be parsed.
        """
        try:
            trader_id = str(raw.get("encryptedUid") or raw.get("leaderId") or raw.get("uid", ""))
            if not trader_id:
                return None

            return LeaderboardEntry(
                trader_id=f"{exchange}:{trader_id}",
                exchange=exchange,
                rank=int(raw.get("rank", raw.get("position", 0))),
                display_name=str(raw.get("nickName") or raw.get("nickname") or raw.get("name", "")),
                pnl_pct=float(raw.get("pnl", raw.get("roi", 0))),
                pnl_usd=float(raw.get("pnlAmount", raw.get("profit", 0))),
                roi_7d=float(raw.get("roi7d", raw.get("weeklyRoi", 0))),
                roi_30d=float(raw.get("roi30d", raw.get("monthlyRoi", 0))),
                roi_90d=float(raw.get("roi90d", raw.get("quarterlyRoi", 0))),
                followers=int(raw.get("followerCount", raw.get("copierNum", 0))),
                win_rate=float(raw.get("winRate", raw.get("winRatio", 0))),
                snapshot_time=snapshot_time,
            )
        except (ValueError, TypeError, KeyError) as exc:
            logger.debug("Failed to parse leaderboard entry: %s", exc)
            return None

    def get_consistent_leaders(
        self,
        exchange: str,
        min_snapshots: int = 5,
        max_rank: int = 50,
    ) -> list[str]:
        """Identify traders who consistently appear in the top N.

        Returns trader_ids of traders who have appeared in the top
        ``max_rank`` positions across at least ``min_snapshots`` snapshots.
        """
        # Query historical snapshots
        with self.store._connect() as conn:
            rows = conn.execute(
                """SELECT trader_id, COUNT(*) as appearances
                   FROM leaderboard_snapshots
                   WHERE exchange = ? AND rank <= ?
                   GROUP BY trader_id
                   HAVING appearances >= ?
                   ORDER BY appearances DESC""",
                (exchange, max_rank, min_snapshots),
            ).fetchall()

        return [row["trader_id"] for row in rows]

    def classify_tier(
        self,
        trader_id: str,
        exchange: str,
    ) -> TraderTier:
        """Classify a trader's tier based on their leaderboard history."""
        history = self.store.get_rank_history(trader_id, limit=30)
        if not history:
            return TraderTier.AVERAGE

        avg_rank = sum(e.rank for e in history) / len(history)
        appearances = len(history)

        if avg_rank <= 10 and appearances >= 10:
            return TraderTier.ELITE
        elif avg_rank <= 25 and appearances >= 7:
            return TraderTier.STRONG
        elif avg_rank <= 50 and appearances >= 5:
            return TraderTier.NOTABLE
        elif appearances >= 3:
            return TraderTier.AVERAGE
        else:
            return TraderTier.WEAK
