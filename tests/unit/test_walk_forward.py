"""Tests for walk-forward optimization backtesting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aiswarm.backtest.engine import OHLCV
from aiswarm.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardOptimizer,
    WalkForwardResult,
)
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now

BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_candles(prices: list[float]) -> list[OHLCV]:
    return [
        OHLCV(
            timestamp=BASE_TS + timedelta(hours=i),
            open=p * 0.999,
            high=p * 1.01,
            low=p * 0.99,
            close=p,
            volume=1000.0,
        )
        for i, p in enumerate(prices)
    ]


class _AlwaysBuyGenerator:
    """Signal generator that always buys (for testing walk-forward mechanics)."""

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        if current_position is not None:
            return Signal(
                signal_id=new_id("sig"),
                agent_id="test",
                symbol=symbol,
                strategy="test",
                thesis="Close position for test",
                direction=-1,
                confidence=0.5,
                expected_return=0.01,
                horizon_minutes=60,
                liquidity_score=0.8,
                regime=MarketRegime.RISK_ON,
                created_at=utc_now(),
                reference_price=candles[-1].close,
            )
        return Signal(
            signal_id=new_id("sig"),
            agent_id="test",
            symbol=symbol,
            strategy="test",
            thesis="Always buy for test",
            direction=1,
            confidence=0.5,
            expected_return=0.01,
            horizon_minutes=60,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
            reference_price=candles[-1].close,
        )


class _NeverSignalGenerator:
    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        return None


class TestWalkForwardOptimizer:
    def test_produces_multiple_windows(self) -> None:
        prices = [100.0 + i * 0.1 for i in range(200)]
        candles = _make_candles(prices)
        config = WalkForwardConfig(train_bars=50, test_bars=30, step_bars=30)
        optimizer = WalkForwardOptimizer(config=config)

        result = optimizer.run("test", _AlwaysBuyGenerator(), "BTCUSDT", candles)

        assert isinstance(result, WalkForwardResult)
        assert result.total_windows >= 2
        assert result.strategy_name == "test"
        assert result.symbol == "BTCUSDT"

    def test_insufficient_data_raises(self) -> None:
        candles = _make_candles([100.0] * 10)
        config = WalkForwardConfig(train_bars=50, test_bars=30)
        optimizer = WalkForwardOptimizer(config=config)

        with pytest.raises(ValueError, match="Need at least"):
            optimizer.run("test", _AlwaysBuyGenerator(), "BTCUSDT", candles)

    def test_aggregate_metrics_computed(self) -> None:
        prices = [100.0 + i * 0.5 for i in range(150)]
        candles = _make_candles(prices)
        config = WalkForwardConfig(train_bars=30, test_bars=20, step_bars=20)
        optimizer = WalkForwardOptimizer(config=config)

        result = optimizer.run("test", _AlwaysBuyGenerator(), "BTCUSDT", candles)

        assert result.aggregate_total_trades >= 0
        assert isinstance(result.aggregate_return_pct, float)
        assert isinstance(result.aggregate_sharpe, float)
        assert isinstance(result.aggregate_max_drawdown_pct, float)
        assert isinstance(result.aggregate_win_rate, float)

    def test_no_signals_produces_zero_trades(self) -> None:
        prices = [100.0 + i * 0.1 for i in range(150)]
        candles = _make_candles(prices)
        config = WalkForwardConfig(train_bars=30, test_bars=20, step_bars=20)
        optimizer = WalkForwardOptimizer(config=config)

        result = optimizer.run("test", _NeverSignalGenerator(), "BTCUSDT", candles)

        assert result.aggregate_total_trades == 0

    def test_windows_have_correct_indices(self) -> None:
        prices = [100.0] * 200
        candles = _make_candles(prices)
        config = WalkForwardConfig(train_bars=50, test_bars=30, step_bars=30)
        optimizer = WalkForwardOptimizer(config=config)

        result = optimizer.run("test", _NeverSignalGenerator(), "BTCUSDT", candles)

        for w in result.windows:
            assert w.train_end_idx == w.train_start_idx + 50
            assert w.test_start_idx == w.train_end_idx
            assert w.test_end_idx <= len(candles)
            assert w.test_end_idx - w.test_start_idx <= 30

    def test_summary_returns_string(self) -> None:
        prices = [100.0] * 150
        candles = _make_candles(prices)
        config = WalkForwardConfig(train_bars=30, test_bars=20, step_bars=20)
        optimizer = WalkForwardOptimizer(config=config)

        result = optimizer.run("test", _NeverSignalGenerator(), "BTCUSDT", candles)
        summary = result.summary()

        assert "Walk-Forward" in summary
        assert "test" in summary
