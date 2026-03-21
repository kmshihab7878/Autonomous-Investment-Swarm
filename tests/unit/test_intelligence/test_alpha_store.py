"""Tests for AlphaStore persistence layer."""

import tempfile
from datetime import datetime, timezone

import pytest

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import (
    ActivitySource,
    LeaderboardEntry,
    StrategyFingerprint,
    TradeActivity,
    TraderProfile,
    TraderTier,
    TradingStyle,
)


@pytest.fixture()
def store() -> AlphaStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return AlphaStore(db_path=f.name)


def _make_activity(
    activity_id: str = "act_1",
    trader_id: str = "t1",
    pnl: float | None = 100.0,
) -> TradeActivity:
    return TradeActivity(
        activity_id=activity_id,
        trader_id=trader_id,
        exchange="binance",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=50000.0,
        notional=5000.0,
        timestamp=datetime.now(timezone.utc),
        source=ActivitySource.TRADE_FEED,
        pnl=pnl,
    )


class TestProfileOperations:
    def test_upsert_and_get(self, store: AlphaStore) -> None:
        profile = TraderProfile(
            trader_id="t1",
            exchange="binance",
            display_name="TestTrader",
            tier=TraderTier.STRONG,
            win_rate=0.65,
        )
        store.upsert_profile(profile)
        retrieved = store.get_profile("t1")
        assert retrieved is not None
        assert retrieved.trader_id == "t1"
        assert retrieved.tier == TraderTier.STRONG
        assert retrieved.win_rate == 0.65

    def test_get_nonexistent(self, store: AlphaStore) -> None:
        assert store.get_profile("nonexistent") is None

    def test_upsert_updates(self, store: AlphaStore) -> None:
        profile1 = TraderProfile(trader_id="t1", exchange="binance", win_rate=0.5)
        store.upsert_profile(profile1)

        profile2 = TraderProfile(trader_id="t1", exchange="binance", win_rate=0.75)
        store.upsert_profile(profile2)

        retrieved = store.get_profile("t1")
        assert retrieved is not None
        assert retrieved.win_rate == 0.75

    def test_get_top_traders(self, store: AlphaStore) -> None:
        for i in range(5):
            tier = TraderTier.ELITE if i < 2 else TraderTier.AVERAGE
            store.upsert_profile(TraderProfile(trader_id=f"t{i}", exchange="binance", tier=tier))

        elite = store.get_top_traders(tier=TraderTier.ELITE)
        assert len(elite) == 2

        all_traders = store.get_top_traders(limit=10)
        assert len(all_traders) == 5


class TestActivityOperations:
    def test_append_and_get(self, store: AlphaStore) -> None:
        activity = _make_activity()
        store.append_activity(activity)

        results = store.get_activities(trader_id="t1")
        assert len(results) == 1
        assert results[0].activity_id == "act_1"

    def test_dedup_on_activity_id(self, store: AlphaStore) -> None:
        activity = _make_activity()
        store.append_activity(activity)
        store.append_activity(activity)  # Same ID — should be ignored

        results = store.get_activities(trader_id="t1")
        assert len(results) == 1

    def test_filter_by_symbol(self, store: AlphaStore) -> None:
        store.append_activity(_make_activity(activity_id="a1"))
        store.append_activity(
            TradeActivity(
                activity_id="a2",
                trader_id="t1",
                exchange="binance",
                symbol="ETHUSDT",
                side="SELL",
                quantity=1.0,
                price=3000.0,
                notional=3000.0,
                timestamp=datetime.now(timezone.utc),
                source=ActivitySource.TRADE_FEED,
            )
        )

        btc = store.get_activities(symbol="BTCUSDT")
        assert len(btc) == 1
        assert btc[0].symbol == "BTCUSDT"

    def test_activity_count(self, store: AlphaStore) -> None:
        for i in range(10):
            store.append_activity(_make_activity(activity_id=f"act_{i}"))
        assert store.get_activity_count("t1") == 10


class TestFingerprintOperations:
    def test_save_and_get(self, store: AlphaStore) -> None:
        fp = StrategyFingerprint(
            trader_id="t1",
            style=TradingStyle.MOMENTUM,
            sample_size=50,
            confidence=0.8,
        )
        store.save_fingerprint(fp)
        retrieved = store.get_latest_fingerprint("t1")
        assert retrieved is not None
        assert retrieved.style == TradingStyle.MOMENTUM

    def test_latest_fingerprint(self, store: AlphaStore) -> None:
        store.save_fingerprint(
            StrategyFingerprint(trader_id="t1", style=TradingStyle.SCALPER, confidence=0.5)
        )
        store.save_fingerprint(
            StrategyFingerprint(trader_id="t1", style=TradingStyle.MOMENTUM, confidence=0.8)
        )
        latest = store.get_latest_fingerprint("t1")
        assert latest is not None
        assert latest.style == TradingStyle.MOMENTUM


class TestLeaderboardOperations:
    def test_save_and_get_history(self, store: AlphaStore) -> None:
        for i in range(3):
            entry = LeaderboardEntry(
                trader_id="t1",
                exchange="binance",
                rank=i + 1,
                snapshot_time=datetime.now(timezone.utc),
            )
            store.save_leaderboard_snapshot(entry)

        history = store.get_rank_history("t1")
        assert len(history) == 3
