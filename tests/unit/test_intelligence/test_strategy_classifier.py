"""Tests for StrategyClassifier."""

import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import ActivitySource, TradeActivity, TradingStyle
from aiswarm.intelligence.strategy_classifier import StrategyClassifier


@pytest.fixture()
def store() -> AlphaStore:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        return AlphaStore(db_path=f.name)


@pytest.fixture()
def classifier(store: AlphaStore) -> StrategyClassifier:
    return StrategyClassifier(store)


def _seed_scalper(store: AlphaStore, trader_id: str = "scalper") -> None:
    """Seed trades that look like scalping (short holds, many trades)."""
    now = datetime.now(timezone.utc)
    for i in range(30):
        store.append_activity(
            TradeActivity(
                activity_id=f"act_{trader_id}_{i}",
                trader_id=trader_id,
                exchange="binance",
                symbol="BTCUSDT",
                side="BUY" if i % 2 == 0 else "SELL",
                quantity=0.5,
                price=50000.0,
                notional=25000.0,
                timestamp=now - timedelta(minutes=i * 5),
                source=ActivitySource.TRADE_FEED,
                pnl=20.0 if i % 3 != 0 else -15.0,
                holding_minutes=5 + (i % 10),
            )
        )


def _seed_swing(store: AlphaStore, trader_id: str = "swing") -> None:
    """Seed trades that look like swing trading (multi-day holds)."""
    now = datetime.now(timezone.utc)
    for i in range(20):
        store.append_activity(
            TradeActivity(
                activity_id=f"act_{trader_id}_{i}",
                trader_id=trader_id,
                exchange="binance",
                symbol="ETHUSDT",
                side="BUY",
                quantity=2.0,
                price=3000.0,
                notional=6000.0,
                timestamp=now - timedelta(days=i),
                source=ActivitySource.TRADE_FEED,
                pnl=300.0 if i % 3 != 0 else -200.0,
                holding_minutes=1440 + i * 60,  # 1-2 days
            )
        )


class TestStrategyClassifier:
    def test_insufficient_data(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        fp = classifier.classify("empty_trader")
        assert fp.style == TradingStyle.UNKNOWN
        assert fp.confidence == 0.0

    def test_classify_scalper(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        _seed_scalper(store)
        fp = classifier.classify("scalper")
        assert fp.style == TradingStyle.SCALPER
        assert fp.sample_size >= 10
        assert fp.confidence > 0

    def test_classify_swing(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        _seed_swing(store)
        fp = classifier.classify("swing")
        assert fp.style == TradingStyle.SWING
        assert fp.sample_size >= 10

    def test_fingerprint_persisted(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        _seed_scalper(store, "persisted")
        classifier.classify("persisted")
        fp = store.get_latest_fingerprint("persisted")
        assert fp is not None
        assert fp.trader_id == "persisted"

    def test_entry_timing_detected(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        _seed_swing(store, "timing_test")
        fp = classifier.classify("timing_test")
        assert fp.entry_timing in ("dip_buyer", "rally_shorter", "mixed")

    def test_typical_hour_detected(self, store: AlphaStore, classifier: StrategyClassifier) -> None:
        _seed_scalper(store, "hour_test")
        fp = classifier.classify("hour_test")
        assert fp.typical_entry_hour_utc is not None
