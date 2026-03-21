"""Tests for Alpha Intelligence domain models."""

from datetime import datetime, timezone

import pytest

from aiswarm.intelligence.models import (
    ActivitySource,
    LeaderboardEntry,
    StrategyFingerprint,
    TradeActivity,
    TraderProfile,
    TraderSignal,
    TraderTier,
    TradingStyle,
)


class TestTradeActivity:
    def test_create_valid(self) -> None:
        activity = TradeActivity(
            activity_id="act_1",
            trader_id="trader_1",
            exchange="binance",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.5,
            price=50000.0,
            notional=25000.0,
            timestamp=datetime.now(timezone.utc),
            source=ActivitySource.TRADE_FEED,
        )
        assert activity.symbol == "BTCUSDT"
        assert activity.notional == 25000.0

    def test_frozen(self) -> None:
        activity = TradeActivity(
            activity_id="act_1",
            trader_id="t1",
            exchange="aster",
            symbol="ETHUSDT",
            side="SELL",
            quantity=1.0,
            price=3000.0,
            notional=3000.0,
            timestamp=datetime.now(timezone.utc),
            source=ActivitySource.WHALE_ALERT,
        )
        with pytest.raises(Exception):
            activity.symbol = "BTCUSDT"  # type: ignore[misc]

    def test_optional_pnl(self) -> None:
        activity = TradeActivity(
            activity_id="act_2",
            trader_id="t1",
            exchange="aster",
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.0,
            price=50000.0,
            notional=50000.0,
            timestamp=datetime.now(timezone.utc),
            source=ActivitySource.LEADERBOARD,
            pnl=500.0,
            holding_minutes=120,
        )
        assert activity.pnl == 500.0
        assert activity.holding_minutes == 120


class TestTraderProfile:
    def test_defaults(self) -> None:
        profile = TraderProfile(trader_id="t1", exchange="binance")
        assert profile.tier == TraderTier.AVERAGE
        assert profile.win_rate == 0.0
        assert profile.primary_style == TradingStyle.UNKNOWN
        assert profile.preferred_symbols == ()

    def test_full_profile(self) -> None:
        profile = TraderProfile(
            trader_id="t1",
            exchange="binance",
            display_name="TopTrader",
            tier=TraderTier.ELITE,
            total_trades=500,
            win_rate=0.72,
            sharpe_ratio=2.5,
            consistency_score=0.85,
            preferred_symbols=("BTCUSDT", "ETHUSDT"),
            primary_style=TradingStyle.MOMENTUM,
        )
        assert profile.tier == TraderTier.ELITE
        assert profile.win_rate == 0.72
        assert len(profile.preferred_symbols) == 2

    def test_win_rate_bounds(self) -> None:
        with pytest.raises(Exception):
            TraderProfile(trader_id="t1", exchange="x", win_rate=1.5)
        with pytest.raises(Exception):
            TraderProfile(trader_id="t1", exchange="x", win_rate=-0.1)


class TestStrategyFingerprint:
    def test_create(self) -> None:
        fp = StrategyFingerprint(
            trader_id="t1",
            style=TradingStyle.MOMENTUM,
            sample_size=50,
            confidence=0.75,
        )
        assert fp.style == TradingStyle.MOMENTUM
        assert fp.confidence == 0.75

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            StrategyFingerprint(trader_id="t1", style=TradingStyle.SCALPER, confidence=1.5)


class TestLeaderboardEntry:
    def test_create(self) -> None:
        entry = LeaderboardEntry(
            trader_id="binance:abc123",
            exchange="binance",
            rank=3,
            display_name="CryptoKing",
            pnl_pct=45.2,
            roi_30d=12.5,
            followers=1500,
            win_rate=0.68,
            snapshot_time=datetime.now(timezone.utc),
        )
        assert entry.rank == 3
        assert entry.followers == 1500


class TestTraderSignal:
    def test_create(self) -> None:
        sig = TraderSignal(
            trader_id="t1",
            trader_tier=TraderTier.ELITE,
            symbol="BTCUSDT",
            direction=1,
            trader_confidence=0.82,
            strategy_match=TradingStyle.MOMENTUM,
            reasoning="WR=72%, Sharpe=2.5",
        )
        assert sig.direction == 1
        assert sig.trader_confidence == 0.82
