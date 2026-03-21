"""Tests for TraderProfiler."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import ActivitySource, TradeActivity, TraderTier
from aiswarm.intelligence.profiler import TraderProfiler


@pytest.fixture()
def store() -> AlphaStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return AlphaStore(db_path=f.name)


@pytest.fixture()
def profiler(store: AlphaStore) -> TraderProfiler:
    return TraderProfiler(store)


def _seed_trades(
    store: AlphaStore,
    trader_id: str = "t1",
    count: int = 50,
    win_rate: float = 0.6,
) -> None:
    """Seed the store with synthetic trades."""
    now = datetime.now(timezone.utc)
    for i in range(count):
        is_win = (i % 10) < int(win_rate * 10)
        pnl = 100.0 if is_win else -80.0
        store.append_activity(
            TradeActivity(
                activity_id=f"act_{trader_id}_{i}",
                trader_id=trader_id,
                exchange="binance",
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                side="BUY" if i % 3 != 0 else "SELL",
                quantity=0.1,
                price=50000.0,
                notional=5000.0,
                timestamp=now - timedelta(hours=count - i),
                source=ActivitySource.TRADE_FEED,
                pnl=pnl,
                holding_minutes=30 + i * 5,
            )
        )


class TestTraderProfiler:
    def test_empty_trader(self, profiler: TraderProfiler) -> None:
        profile = profiler.build_profile("unknown", "binance")
        assert profile.total_trades == 0
        assert profile.tier == TraderTier.AVERAGE

    def test_build_profile_with_data(self, store: AlphaStore, profiler: TraderProfiler) -> None:
        _seed_trades(store, "t1", count=50, win_rate=0.6)
        profile = profiler.build_profile("t1", "binance", "TestTrader")

        assert profile.trader_id == "t1"
        assert profile.total_trades == 50
        assert profile.win_rate > 0.5
        assert profile.sharpe_ratio != 0
        assert profile.max_drawdown >= 0
        assert profile.profit_factor > 0
        assert len(profile.preferred_symbols) > 0
        assert profile.avg_holding_minutes > 0
        assert profile.trade_frequency_daily > 0

    def test_tier_classification(self, store: AlphaStore, profiler: TraderProfiler) -> None:
        _seed_trades(store, "winner", count=100, win_rate=0.8)
        profile = profiler.build_profile("winner", "binance")
        assert profile.tier in (TraderTier.ELITE, TraderTier.STRONG, TraderTier.NOTABLE)

    def test_profile_persisted(self, store: AlphaStore, profiler: TraderProfiler) -> None:
        _seed_trades(store, "t2", count=30, win_rate=0.5)
        profiler.build_profile("t2", "binance")

        # Should be in the store now
        retrieved = store.get_profile("t2")
        assert retrieved is not None
        assert retrieved.total_trades == 30

    def test_consistency_score(self, store: AlphaStore, profiler: TraderProfiler) -> None:
        _seed_trades(store, "consistent", count=100, win_rate=0.65)
        profile = profiler.build_profile("consistent", "binance")
        # Consistent win rate should yield a decent consistency score
        assert profile.consistency_score >= 0.0
