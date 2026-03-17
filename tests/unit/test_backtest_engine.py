"""Comprehensive tests for the backtesting engine.

Covers:
- Uptrend / downtrend PnL correctness
- Look-ahead bias prevention
- Commission and slippage impact
- Equity curve accuracy
- Trade recording (entry, exit, PnL)
- Risk metric delegation (Sharpe, Sortino, drawdown)
- Force-close at end of backtest
- Edge cases: insufficient candles, no signals, flat market
- Adapter integration with MomentumAgent and FundingRateAgent
- CSV data loader
"""

from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aiswarm.backtest.adapters import (
    FundingRateSignalGenerator,
    MomentumSignalGenerator,
)
from aiswarm.backtest.data_loader import load_candles_from_csv
from aiswarm.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    OHLCV,
)
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now


# ---------------------------------------------------------------------------
# Test helpers: deterministic signal generators
# ---------------------------------------------------------------------------

BASE_TS = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_candles(
    prices: list[float],
    base: datetime = BASE_TS,
    interval_minutes: int = 60,
) -> list[OHLCV]:
    """Build OHLCV candles from a list of close prices."""
    candles: list[OHLCV] = []
    for i, price in enumerate(prices):
        candles.append(
            OHLCV(
                timestamp=base + timedelta(minutes=i * interval_minutes),
                open=price * 0.999,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000.0,
            )
        )
    return candles


def _make_signal(direction: int, confidence: float = 0.7) -> Signal:
    """Create a minimal valid Signal."""
    return Signal(
        signal_id=new_id("sig"),
        agent_id="test_agent",
        symbol="BTCUSDT",
        strategy="test_strategy",
        thesis="backtest test signal with direction",
        direction=direction,
        confidence=confidence,
        expected_return=0.01,
        horizon_minutes=60,
        liquidity_score=0.8,
        regime=MarketRegime.RISK_ON,
        created_at=utc_now(),
        reference_price=100.0,
    )


class AlwaysLongGenerator:
    """Generates a long signal on every bar (no position check)."""

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        if current_position is not None:
            return None  # already positioned
        return _make_signal(direction=1)


class AlwaysShortGenerator:
    """Generates a short signal on every bar (no position check)."""

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        if current_position is not None:
            return None
        return _make_signal(direction=-1)


class AlternatingGenerator:
    """Alternates between long and short: opens on even bars, closes on odd."""

    def __init__(self) -> None:
        self.call_count = 0

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        self.call_count += 1
        if current_position is None:
            return _make_signal(direction=1)
        return _make_signal(direction=-1)


class NeverSignalGenerator:
    """Never generates any signal."""

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        return None


class LookAheadDetector:
    """Records how many candles it receives on each call.

    Used to verify the engine does not pass future candles.
    """

    def __init__(self) -> None:
        self.candle_counts: list[int] = []

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        self.candle_counts.append(len(candles))
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBacktestEngineEdgeCases:
    """Edge cases and input validation."""

    def test_fewer_than_two_candles_raises(self) -> None:
        engine = BacktestEngine()
        candles = _make_candles([100.0])
        with pytest.raises(ValueError, match="at least 2 candles"):
            engine.run("test", NeverSignalGenerator(), "BTCUSDT", candles)

    def test_empty_candles_raises(self) -> None:
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="at least 2 candles"):
            engine.run("test", NeverSignalGenerator(), "BTCUSDT", [])

    def test_no_signals_returns_flat_equity(self) -> None:
        engine = BacktestEngine(BacktestConfig(initial_capital=10_000.0))
        candles = _make_candles([100.0, 101.0, 102.0, 103.0, 104.0])
        result = engine.run("no_signal", NeverSignalGenerator(), "BTCUSDT", candles)

        assert result.total_trades == 0
        assert result.total_return_pct == pytest.approx(0.0)
        assert result.final_capital == pytest.approx(10_000.0)
        # Equity curve should be flat at initial capital
        assert all(v == pytest.approx(10_000.0) for v in result.equity_curve)


class TestLookAheadBias:
    """Verify the engine prevents look-ahead bias."""

    def test_candle_count_increases_monotonically(self) -> None:
        detector = LookAheadDetector()
        engine = BacktestEngine()
        candles = _make_candles([100.0 + i for i in range(10)])
        engine.run("look_ahead", detector, "BTCUSDT", candles)

        # The engine starts at index 1, so first call gets 2 candles, etc.
        assert len(detector.candle_counts) == 9  # one per bar after the first
        for idx, count in enumerate(detector.candle_counts):
            expected = idx + 2  # bar 1 sees [0..1], bar 2 sees [0..2], ...
            assert count == expected, f"Bar {idx + 1} received {count} candles, expected {expected}"


class TestUptrendPnL:
    """Always-long on an uptrend should be profitable."""

    def test_positive_return_on_uptrend(self) -> None:
        # Prices steadily rise 100 -> 120
        prices = [100.0 + i * 1.0 for i in range(21)]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("long_uptrend", AlwaysLongGenerator(), "BTCUSDT", candles)

        assert result.total_return_pct > 0.0
        assert result.final_capital > 10_000.0
        # There should be at least one closing trade (force-close at end)
        assert result.total_trades >= 1

    def test_trades_recorded_correctly(self) -> None:
        prices = [100.0 + i * 2.0 for i in range(5)]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("test", AlwaysLongGenerator(), "BTCUSDT", candles)

        # Should have an opening BUY trade and a closing SELL trade (force-close)
        buy_trades = [t for t in result.trades if t.side == "BUY"]
        sell_trades = [t for t in result.trades if t.side == "SELL"]
        assert len(buy_trades) >= 1
        assert len(sell_trades) >= 1

        # The force-close SELL should have positive PnL on an uptrend
        assert sell_trades[-1].pnl > 0.0


class TestDowntrendPnL:
    """Always-short on a downtrend should be profitable."""

    def test_positive_return_on_downtrend(self) -> None:
        prices = [200.0 - i * 1.0 for i in range(21)]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("short_downtrend", AlwaysShortGenerator(), "BTCUSDT", candles)

        assert result.total_return_pct > 0.0
        assert result.final_capital > 10_000.0


class TestCommissionSlippage:
    """Commission and slippage should reduce returns."""

    def test_commission_reduces_returns(self) -> None:
        prices = [100.0 + i * 1.0 for i in range(21)]
        candles = _make_candles(prices)

        no_cost = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        with_cost = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=50.0,  # very high commission to make impact obvious
            slippage_bps=0.0,
        )

        result_free = BacktestEngine(no_cost).run("free", AlwaysLongGenerator(), "BTCUSDT", candles)
        result_costly = BacktestEngine(with_cost).run(
            "costly", AlwaysLongGenerator(), "BTCUSDT", candles
        )

        assert result_free.final_capital > result_costly.final_capital

    def test_slippage_reduces_returns(self) -> None:
        prices = [100.0 + i * 1.0 for i in range(21)]
        candles = _make_candles(prices)

        no_slip = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        with_slip = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=100.0,  # 1% slippage
        )

        result_free = BacktestEngine(no_slip).run("free", AlwaysLongGenerator(), "BTCUSDT", candles)
        result_slippy = BacktestEngine(with_slip).run(
            "slippy", AlwaysLongGenerator(), "BTCUSDT", candles
        )

        assert result_free.final_capital > result_slippy.final_capital


class TestEquityCurve:
    """Equity curve should track capital + unrealized PnL correctly."""

    def test_equity_curve_length(self) -> None:
        n = 15
        candles = _make_candles([100.0 + i for i in range(n)])
        engine = BacktestEngine()
        result = engine.run("test", NeverSignalGenerator(), "BTCUSDT", candles)

        # Initial point + one per bar = n points total
        assert len(result.equity_curve) == n

    def test_equity_curve_starts_at_initial_capital(self) -> None:
        config = BacktestConfig(initial_capital=5_000.0)
        candles = _make_candles([100.0, 101.0, 102.0])
        result = BacktestEngine(config).run("test", NeverSignalGenerator(), "BTCUSDT", candles)
        assert result.equity_curve[0] == pytest.approx(5_000.0)

    def test_equity_curve_reflects_unrealized_pnl(self) -> None:
        # With a long position on rising prices, equity should increase
        prices = [100.0, 100.0, 110.0, 120.0, 130.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=1.0,  # 100% of capital
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("test", AlwaysLongGenerator(), "BTCUSDT", candles)

        # After opening long at bar 1 (price 100), equity should grow
        # Each equity point after entry should be >= previous (rising prices)
        # The opening trade happens at bar 1, so equity_curve[2] onward reflects position
        for i in range(2, len(result.equity_curve)):
            assert result.equity_curve[i] >= result.equity_curve[i - 1], (
                f"Equity decreased at bar {i}: "
                f"{result.equity_curve[i]} < {result.equity_curve[i - 1]}"
            )


class TestForceClose:
    """Open positions should be force-closed at the last bar."""

    def test_position_force_closed_at_end(self) -> None:
        prices = [100.0, 100.0, 105.0, 110.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("test", AlwaysLongGenerator(), "BTCUSDT", candles)

        # Last trade should be a force-close SELL with signal_confidence=0.0
        last_trade = result.trades[-1]
        assert last_trade.side == "SELL"
        assert last_trade.signal_confidence == pytest.approx(0.0)
        assert last_trade.pnl > 0.0  # price went up

    def test_force_close_short_position(self) -> None:
        prices = [100.0, 100.0, 95.0, 90.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("test", AlwaysShortGenerator(), "BTCUSDT", candles)

        last_trade = result.trades[-1]
        assert last_trade.side == "BUY"
        assert last_trade.signal_confidence == pytest.approx(0.0)
        assert last_trade.pnl > 0.0  # price went down, short profits


class TestTradeStatistics:
    """Win/loss statistics should be computed correctly."""

    def test_alternating_trades_stats(self) -> None:
        # Up-down-up-down pattern: some wins, some losses
        prices = [100.0, 100.0, 110.0, 90.0, 120.0, 80.0, 130.0, 70.0, 140.0, 60.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("alternating", AlternatingGenerator(), "BTCUSDT", candles)

        # Verify trade count consistency
        assert result.total_trades == result.winning_trades + result.losing_trades
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 100.0

    def test_all_winning_trades(self) -> None:
        # Steady uptrend with always-long
        prices = [100.0 + i * 5.0 for i in range(10)]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("all_wins", AlwaysLongGenerator(), "BTCUSDT", candles)

        assert result.winning_trades >= 1
        assert result.losing_trades == 0
        assert result.win_rate == pytest.approx(100.0)
        assert result.profit_factor == math.inf

    def test_profit_factor_finite_with_losses(self) -> None:
        # Mix of wins and losses
        prices = [100.0, 100.0, 110.0, 90.0, 110.0, 90.0, 110.0, 90.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("mixed", AlternatingGenerator(), "BTCUSDT", candles)

        if result.losing_trades > 0:
            assert result.profit_factor < math.inf
            assert result.profit_factor >= 0.0


class TestRiskMetrics:
    """Sharpe, Sortino, and max drawdown should be computed."""

    def test_metrics_computed_for_sufficient_data(self) -> None:
        prices = [100.0 + i * 0.5 for i in range(30)]
        candles = _make_candles(prices)
        engine = BacktestEngine()
        result = engine.run("test", AlwaysLongGenerator(), "BTCUSDT", candles)

        # On a monotonic uptrend, Sharpe should be positive
        assert result.sharpe_ratio > 0.0
        assert result.sortino_ratio > 0.0
        # Drawdown on a pure uptrend should be very small
        assert result.max_drawdown_pct >= 0.0

    def test_metrics_zero_for_no_trades(self) -> None:
        prices = [100.0] * 5
        candles = _make_candles(prices)
        engine = BacktestEngine()
        result = engine.run("flat", NeverSignalGenerator(), "BTCUSDT", candles)

        assert result.sharpe_ratio == pytest.approx(0.0)
        assert result.sortino_ratio == pytest.approx(0.0)


class TestBacktestResult:
    """BacktestResult dataclass and summary."""

    def test_summary_format(self) -> None:
        prices = [100.0 + i for i in range(10)]
        candles = _make_candles(prices)
        engine = BacktestEngine()
        result = engine.run("format_test", AlwaysLongGenerator(), "BTCUSDT", candles)

        summary = result.summary()
        assert "format_test" in summary
        assert "BTCUSDT" in summary
        assert "Return:" in summary
        assert "Sharpe:" in summary
        assert "Sortino:" in summary
        assert "Max Drawdown:" in summary
        assert "Win Rate:" in summary
        assert "Profit Factor:" in summary

    def test_result_dates_match_candle_range(self) -> None:
        prices = [100.0 + i for i in range(5)]
        candles = _make_candles(prices)
        engine = BacktestEngine()
        result = engine.run("dates", NeverSignalGenerator(), "BTCUSDT", candles)

        assert result.start_date == candles[0].timestamp
        assert result.end_date == candles[-1].timestamp
        assert result.symbol == "BTCUSDT"
        assert result.strategy_name == "dates"


class TestBacktestConfig:
    """Configuration defaults and overrides."""

    def test_default_config(self) -> None:
        config = BacktestConfig()
        assert config.initial_capital == 10_000.0
        assert config.position_size_pct == 0.1
        assert config.commission_bps == 5.0
        assert config.slippage_bps == 2.0

    def test_custom_config(self) -> None:
        config = BacktestConfig(
            initial_capital=50_000.0,
            position_size_pct=0.25,
            commission_bps=10.0,
            slippage_bps=5.0,
        )
        assert config.initial_capital == 50_000.0
        assert config.position_size_pct == 0.25


class TestCSVDataLoader:
    """CSV data loading."""

    def test_load_valid_csv(self) -> None:
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01 00:00:00,100.0,105.0,98.0,103.0,5000\n"
            "2024-01-01 01:00:00,103.0,108.0,101.0,106.0,6000\n"
            "2024-01-01 02:00:00,106.0,110.0,104.0,109.0,7000\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        candles = load_candles_from_csv(csv_path)

        assert len(candles) == 3
        assert candles[0].open == pytest.approx(100.0)
        assert candles[0].close == pytest.approx(103.0)
        assert candles[1].volume == pytest.approx(6000.0)
        # Sorted chronologically
        assert candles[0].timestamp < candles[1].timestamp < candles[2].timestamp

        Path(csv_path).unlink()

    def test_load_csv_sorts_by_timestamp(self) -> None:
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            "2024-01-01 02:00:00,106.0,110.0,104.0,109.0,7000\n"
            "2024-01-01 00:00:00,100.0,105.0,98.0,103.0,5000\n"
            "2024-01-01 01:00:00,103.0,108.0,101.0,106.0,6000\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        candles = load_candles_from_csv(csv_path)

        assert candles[0].close == pytest.approx(103.0)  # earliest
        assert candles[-1].close == pytest.approx(109.0)  # latest

        Path(csv_path).unlink()

    def test_load_nonexistent_csv_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_candles_from_csv("/nonexistent/path/data.csv")


class TestMomentumAdapter:
    """MomentumSignalGenerator adapter integration."""

    def test_generates_long_signal_on_uptrend(self) -> None:
        gen = MomentumSignalGenerator(fast_period=5, slow_period=10, min_candles=10)
        # Clear uptrend
        prices = [100.0 + i * 2.0 for i in range(20)]
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is not None
        assert signal.direction == 1

    def test_generates_short_signal_on_downtrend(self) -> None:
        gen = MomentumSignalGenerator(fast_period=5, slow_period=10, min_candles=10)
        prices = [200.0 - i * 2.0 for i in range(20)]
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is not None
        assert signal.direction == -1

    def test_no_signal_on_insufficient_data(self) -> None:
        gen = MomentumSignalGenerator(min_candles=50)
        prices = [100.0 + i for i in range(10)]
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is None

    def test_full_backtest_with_momentum_adapter(self) -> None:
        gen = MomentumSignalGenerator(fast_period=5, slow_period=10, min_candles=10)
        prices = [100.0 + i * 1.5 for i in range(60)]
        candles = _make_candles(prices)

        engine = BacktestEngine(
            BacktestConfig(
                initial_capital=10_000.0,
                position_size_pct=0.2,
                commission_bps=5.0,
                slippage_bps=2.0,
            )
        )
        result = engine.run("momentum_bt", gen, "BTCUSDT", candles)

        assert result.strategy_name == "momentum_bt"
        assert len(result.equity_curve) == 60


class TestFundingRateAdapter:
    """FundingRateSignalGenerator adapter integration."""

    def test_generates_contrarian_signal_on_strong_uptrend(self) -> None:
        # Strong uptrend -> positive synthetic funding -> contrarian short
        gen = FundingRateSignalGenerator(
            lookback=5,
            rate_multiplier=0.5,  # amplify so threshold is exceeded
            extreme_threshold=0.001,
        )
        prices = [100.0 + i * 10.0 for i in range(20)]
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is not None
        assert signal.direction == -1  # contrarian short

    def test_generates_contrarian_signal_on_strong_downtrend(self) -> None:
        gen = FundingRateSignalGenerator(
            lookback=5,
            rate_multiplier=0.5,
            extreme_threshold=0.001,
        )
        prices = [200.0 - i * 10.0 for i in range(20)]
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is not None
        assert signal.direction == 1  # contrarian long

    def test_no_signal_on_flat_market(self) -> None:
        gen = FundingRateSignalGenerator(
            lookback=5,
            rate_multiplier=0.01,
            extreme_threshold=0.001,
        )
        # Flat prices -> near-zero synthetic funding
        prices = [100.0] * 20
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is None

    def test_no_signal_on_insufficient_data(self) -> None:
        gen = FundingRateSignalGenerator(lookback=20)
        prices = [100.0] * 5
        candles = _make_candles(prices)

        signal = gen.generate_signal("BTCUSDT", candles, None)
        assert signal is None


class TestPnLCalculation:
    """Verify PnL math is correct in specific scenarios."""

    def test_long_pnl_exact(self) -> None:
        """Open long at 100, close at 120, 50% position size, no costs."""
        prices = [100.0, 100.0, 120.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("exact", AlwaysLongGenerator(), "BTCUSDT", candles)

        # Position: 5000 / 100 = 50 units
        # PnL: (120 - 100) * 50 = 1000
        # Final capital: 10000 + 1000 = 11000
        assert result.final_capital == pytest.approx(11_000.0)
        assert result.total_return_pct == pytest.approx(10.0)

    def test_short_pnl_exact(self) -> None:
        """Open short at 100, close at 80, 50% position size, no costs."""
        prices = [100.0, 100.0, 80.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=0.0,
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("exact_short", AlwaysShortGenerator(), "BTCUSDT", candles)

        # Position: 5000 / 100 = 50 units
        # PnL: (100 - 80) * 50 = 1000
        assert result.final_capital == pytest.approx(11_000.0)
        assert result.total_return_pct == pytest.approx(10.0)

    def test_commission_deducted_correctly(self) -> None:
        """Verify commission is deducted from capital on entry."""
        prices = [100.0, 100.0, 100.0]
        candles = _make_candles(prices)
        config = BacktestConfig(
            initial_capital=10_000.0,
            position_size_pct=0.5,
            commission_bps=100.0,  # 1% = 100 bps
            slippage_bps=0.0,
        )
        engine = BacktestEngine(config)
        result = engine.run("comm", AlwaysLongGenerator(), "BTCUSDT", candles)

        # Commission: 5000 * 100/10000 = 50
        # PnL from position: 0 (flat price)
        # Final: 10000 - 50 = 9950
        assert result.final_capital == pytest.approx(9_950.0)


class TestOHLCV:
    """OHLCV dataclass."""

    def test_frozen(self) -> None:
        candle = OHLCV(
            timestamp=BASE_TS,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        with pytest.raises(AttributeError):
            candle.close = 200.0  # type: ignore[misc]

    def test_fields(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        candle = OHLCV(
            timestamp=ts,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        assert candle.timestamp == ts
        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 95.0
        assert candle.close == 102.0
        assert candle.volume == 1000.0
