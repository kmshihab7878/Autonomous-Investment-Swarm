"""Tests for AlphaFollowerAgent."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from aiswarm.intelligence.agents.alpha_follower import AlphaFollowerAgent
from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import (
    ActivitySource,
    TradeActivity,
    TraderProfile,
    TraderTier,
    TradingStyle,
)


@pytest.fixture()
def store() -> AlphaStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return AlphaStore(db_path=f.name)


@pytest.fixture()
def agent(store: AlphaStore) -> AlphaFollowerAgent:
    return AlphaFollowerAgent(store=store)


def _seed_elite_trader(store: AlphaStore) -> None:
    """Create an elite trader with recent activity."""
    now = datetime.now(timezone.utc)

    # Profile
    store.upsert_profile(
        TraderProfile(
            trader_id="elite_1",
            exchange="binance",
            display_name="EliteTrader",
            tier=TraderTier.ELITE,
            total_trades=500,
            win_rate=0.72,
            sharpe_ratio=2.5,
            consistency_score=0.85,
            primary_style=TradingStyle.MOMENTUM,
        )
    )

    # Recent activity
    store.append_activity(
        TradeActivity(
            activity_id="act_elite_recent",
            trader_id="elite_1",
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=55000.0,
            notional=55000.0,
            timestamp=now - timedelta(minutes=5),
            source=ActivitySource.TRADE_FEED,
            pnl=None,
        )
    )


def _seed_weak_trader(store: AlphaStore) -> None:
    """Create a weak trader with recent activity."""
    now = datetime.now(timezone.utc)

    store.upsert_profile(
        TraderProfile(
            trader_id="weak_1",
            exchange="binance",
            tier=TraderTier.WEAK,
            win_rate=0.3,
        )
    )

    store.append_activity(
        TradeActivity(
            activity_id="act_weak_recent",
            trader_id="weak_1",
            exchange="binance",
            symbol="BTCUSDT",
            side="SELL",
            quantity=0.5,
            price=55000.0,
            notional=27500.0,
            timestamp=now - timedelta(minutes=10),
            source=ActivitySource.TRADE_FEED,
        )
    )


class TestAlphaFollowerAgent:
    def test_no_activity_returns_none(self, agent: AlphaFollowerAgent) -> None:
        result = agent.analyze({"symbol": "BTCUSDT"})
        assert result["signal"] is None
        assert result["reason"] == "no_recent_activity"

    def test_signal_from_elite_trader(self, store: AlphaStore, agent: AlphaFollowerAgent) -> None:
        _seed_elite_trader(store)
        result = agent.analyze({"symbol": "BTCUSDT"})
        assert result["signal"] is not None
        signal = result["signal"]
        assert signal.direction == 1  # BUY → long
        assert signal.confidence > 0.5
        assert signal.strategy == "alpha_follower"
        assert "elite" in signal.thesis.lower()

    def test_weak_trader_filtered(self, store: AlphaStore, agent: AlphaFollowerAgent) -> None:
        _seed_weak_trader(store)
        result = agent.analyze({"symbol": "BTCUSDT"})
        # Weak traders should not generate signals (min tier = NOTABLE)
        assert result["signal"] is None

    def test_elite_preferred_over_weak(self, store: AlphaStore, agent: AlphaFollowerAgent) -> None:
        _seed_elite_trader(store)
        _seed_weak_trader(store)
        result = agent.analyze({"symbol": "BTCUSDT"})
        assert result["signal"] is not None
        # Signal should come from the elite trader
        assert result["signal"].direction == 1  # Elite trader's direction

    def test_stale_activity_ignored(self, store: AlphaStore, agent: AlphaFollowerAgent) -> None:
        # Create an elite trader with OLD activity (beyond max_activity_age)
        store.upsert_profile(
            TraderProfile(
                trader_id="stale_1",
                exchange="binance",
                tier=TraderTier.ELITE,
                win_rate=0.8,
            )
        )
        store.append_activity(
            TradeActivity(
                activity_id="act_stale",
                trader_id="stale_1",
                exchange="binance",
                symbol="BTCUSDT",
                side="BUY",
                quantity=1.0,
                price=50000.0,
                notional=50000.0,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=3),  # 3h old
                source=ActivitySource.TRADE_FEED,
            )
        )
        result = agent.analyze({"symbol": "BTCUSDT"})
        assert result["signal"] is None

    def test_validate_returns_true_with_data(
        self, store: AlphaStore, agent: AlphaFollowerAgent
    ) -> None:
        _seed_elite_trader(store)
        assert agent.validate({"symbol": "BTCUSDT"}) is True

    def test_validate_returns_false_without_data(self, agent: AlphaFollowerAgent) -> None:
        assert agent.validate({"symbol": "BTCUSDT"}) is False

    def test_propose_delegates_to_analyze(
        self, store: AlphaStore, agent: AlphaFollowerAgent
    ) -> None:
        _seed_elite_trader(store)
        result = agent.propose({"symbol": "BTCUSDT"})
        assert result["signal"] is not None
