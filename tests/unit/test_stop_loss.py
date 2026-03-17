"""Tests for per-position stop-loss monitoring."""

from __future__ import annotations

from datetime import datetime, timezone

from aiswarm.risk.stop_loss import StopLossMonitor
from aiswarm.types.orders import Side
from aiswarm.types.portfolio import Position, PortfolioSnapshot


def _make_position(
    symbol: str = "BTCUSDT",
    quantity: float = 1.0,
    avg_price: float = 50_000.0,
    market_price: float = 50_000.0,
    strategy: str = "test",
) -> Position:
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_price=avg_price,
        market_price=market_price,
        strategy=strategy,
    )


def _make_snapshot(
    positions: tuple[Position, ...] = (),
    nav: float = 100_000.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        nav=nav,
        cash=nav,
        gross_exposure=0.0,
        net_exposure=0.0,
        positions=positions,
    )


class TestStopLossMonitorWithinLimit:
    """Positions within the loss limit should produce no stop-loss orders."""

    def test_no_orders_when_position_is_flat(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=50_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_no_orders_when_position_is_profitable(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=55_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_no_orders_when_loss_below_threshold(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # 2% loss (below 5% threshold)
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=49_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_no_orders_when_loss_exactly_at_threshold(self) -> None:
        """At exactly the threshold, do NOT trigger (must exceed)."""
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=47_500.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_no_orders_when_short_position_is_profitable(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # Short position: negative quantity, price dropped = profit
        pos = _make_position(quantity=-1.0, avg_price=50_000.0, market_price=48_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_no_orders_on_empty_portfolio(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        snapshot = _make_snapshot(positions=())
        orders = monitor.check_positions(snapshot)
        assert orders == []


class TestStopLossMonitorExceedingLimit:
    """Positions exceeding the loss limit should generate closing orders."""

    def test_long_position_generates_sell_close(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # 10% loss on a long position
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=45_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.SELL
        assert order.quantity == 1.0

    def test_short_position_generates_buy_close(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # Short position: price went UP 10% = loss
        pos = _make_position(quantity=-1.0, avg_price=50_000.0, market_price=55_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "BTCUSDT"
        assert order.side == Side.BUY
        assert order.quantity == 1.0

    def test_closing_order_has_full_quantity(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=3.5, avg_price=50_000.0, market_price=45_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        assert orders[0].quantity == 3.5

    def test_closing_order_has_correct_notional(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=2.0, avg_price=50_000.0, market_price=45_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        # notional = quantity * market_price = 2.0 * 45_000.0 = 90_000.0
        assert orders[0].notional == 90_000.0

    def test_closing_order_fields_are_valid(self) -> None:
        """The generated order must be a valid Order model (Pydantic validates)."""
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(
            symbol="ETHUSDT",
            quantity=10.0,
            avg_price=3_000.0,
            market_price=2_800.0,
            strategy="momentum_ma_crossover",
        )
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        order = orders[0]
        assert order.symbol == "ETHUSDT"
        assert order.strategy == "momentum_ma_crossover"
        assert order.thesis.startswith("stop_loss:")
        assert order.order_id.startswith("sl_")
        assert order.signal_id.startswith("sl_sig_")


class TestStopLossMonitorClosingSide:
    """The closing order must have the opposite side to flatten the position."""

    def test_buy_position_closed_with_sell(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=40_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders[0].side == Side.SELL

    def test_sell_position_closed_with_buy(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        pos = _make_position(quantity=-1.0, avg_price=50_000.0, market_price=60_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders[0].side == Side.BUY


class TestStopLossMonitorMultiplePositions:
    """Only positions exceeding the loss limit should generate orders."""

    def test_only_losing_positions_generate_orders(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        winning = _make_position(
            symbol="BTCUSDT", quantity=1.0, avg_price=50_000.0, market_price=55_000.0
        )
        small_loss = _make_position(
            symbol="ETHUSDT", quantity=5.0, avg_price=3_000.0, market_price=2_950.0
        )
        big_loss = _make_position(
            symbol="SOLUSDT", quantity=100.0, avg_price=100.0, market_price=90.0
        )
        snapshot = _make_snapshot(positions=(winning, small_loss, big_loss))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        assert orders[0].symbol == "SOLUSDT"
        assert orders[0].side == Side.SELL

    def test_multiple_positions_exceeding_limit(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        loss_btc = _make_position(
            symbol="BTCUSDT", quantity=1.0, avg_price=50_000.0, market_price=45_000.0
        )
        loss_eth = _make_position(
            symbol="ETHUSDT", quantity=-10.0, avg_price=3_000.0, market_price=3_500.0
        )
        snapshot = _make_snapshot(positions=(loss_btc, loss_eth))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 2
        symbols = {o.symbol for o in orders}
        assert symbols == {"BTCUSDT", "ETHUSDT"}


class TestStopLossEntryPriceTracking:
    """StopLossMonitor can track entry prices via set_entry_prices and record_entry."""

    def test_set_entry_prices(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        monitor.set_entry_prices({"BTCUSDT": 50_000.0, "ETHUSDT": 3_000.0})
        assert monitor.entry_prices["BTCUSDT"] == 50_000.0
        assert monitor.entry_prices["ETHUSDT"] == 3_000.0

    def test_record_entry(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        monitor.record_entry("BTCUSDT", 48_000.0, "buy")
        assert monitor.entry_prices["BTCUSDT"] == 48_000.0

    def test_record_entry_overwrites(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        monitor.record_entry("BTCUSDT", 48_000.0, "buy")
        monitor.record_entry("BTCUSDT", 52_000.0, "buy")
        assert monitor.entry_prices["BTCUSDT"] == 52_000.0

    def test_entry_price_used_when_available(self) -> None:
        """When an entry price is recorded, it overrides avg_price for PnL calc."""
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # avg_price is 50k (would show 10% loss at market 45k)
        # but recorded entry at 44k means position is actually in profit
        monitor.record_entry("BTCUSDT", 44_000.0, "buy")
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=45_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert orders == []

    def test_entry_price_triggers_stop_loss(self) -> None:
        """When entry price makes the real loss exceed threshold, trigger stop-loss."""
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        # avg_price is 50k (would show 2% loss at market 49k, within limit)
        # but recorded entry at 55k means actual loss is ~10.9%
        monitor.record_entry("BTCUSDT", 55_000.0, "buy")
        pos = _make_position(quantity=1.0, avg_price=50_000.0, market_price=49_000.0)
        snapshot = _make_snapshot(positions=(pos,))
        orders = monitor.check_positions(snapshot)
        assert len(orders) == 1
        assert orders[0].side == Side.SELL


class TestStopLossMonitorConstructor:
    """Verify constructor validation."""

    def test_default_threshold(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        assert monitor.max_position_loss_pct == 0.05

    def test_custom_threshold(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.10)
        assert monitor.max_position_loss_pct == 0.10

    def test_empty_entry_prices(self) -> None:
        monitor = StopLossMonitor(max_position_loss_pct=0.05)
        assert monitor.entry_prices == {}
