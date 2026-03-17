"""Per-position stop-loss monitoring.

Scans portfolio positions for unrealized losses exceeding a configurable
threshold and generates closing orders to flatten breaching positions.

This module runs *after* order entry — it is not a pre-trade check but a
continuous position-level risk control.
"""

from __future__ import annotations

from aiswarm.types.orders import Order, Side
from aiswarm.types.portfolio import Position, PortfolioSnapshot
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class StopLossMonitor:
    """Monitor open positions and generate closing orders when per-position
    unrealized loss exceeds the configured threshold.

    Args:
        max_position_loss_pct: Maximum tolerated unrealized loss as a fraction
            (e.g. 0.05 = 5%). Positions whose loss *exceeds* this value will
            trigger a stop-loss close order.
    """

    def __init__(self, max_position_loss_pct: float) -> None:
        self.max_position_loss_pct = max_position_loss_pct
        self.entry_prices: dict[str, float] = {}

    def set_entry_prices(self, entries: dict[str, float]) -> None:
        """Bulk-set entry prices for multiple symbols.

        Args:
            entries: Mapping of symbol to entry price.
        """
        self.entry_prices.update(entries)

    def record_entry(self, symbol: str, price: float, side: str) -> None:
        """Record an entry price when an order fills.

        Args:
            symbol: Trading pair symbol.
            price: Fill price.
            side: Order side (``"buy"`` or ``"sell"``). Stored for reference
                  but not used for PnL — position quantity sign determines
                  directionality.
        """
        self.entry_prices[symbol] = price
        logger.info(
            "Entry price recorded",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "price": price,
                    "side": side,
                }
            },
        )

    def check_positions(self, snapshot: PortfolioSnapshot) -> list[Order]:
        """Check all positions for stop-loss breaches.

        For each position, calculates unrealized PnL percentage using the
        recorded entry price (falling back to ``Position.avg_price``) and
        the current market price. If the loss exceeds
        ``max_position_loss_pct``, a closing order is generated.

        Args:
            snapshot: Current portfolio snapshot with positions and prices.

        Returns:
            List of closing ``Order`` objects for positions that breached
            the stop-loss threshold. Empty if no breaches.
        """
        close_orders: list[Order] = []

        for position in snapshot.positions:
            unrealized_loss_pct = self._unrealized_loss_pct(position)

            if unrealized_loss_pct > self.max_position_loss_pct:
                order = self._build_close_order(position, unrealized_loss_pct)
                close_orders.append(order)
                logger.warning(
                    "Stop-loss triggered",
                    extra={
                        "extra_json": {
                            "symbol": position.symbol,
                            "unrealized_loss_pct": round(unrealized_loss_pct, 6),
                            "threshold": self.max_position_loss_pct,
                            "quantity": position.quantity,
                            "entry_price": self._effective_entry_price(position),
                            "market_price": position.market_price,
                        }
                    },
                )

        return close_orders

    def _effective_entry_price(self, position: Position) -> float:
        """Return the recorded entry price for the symbol, or fall back to avg_price."""
        return self.entry_prices.get(position.symbol, position.avg_price)

    def _unrealized_loss_pct(self, position: Position) -> float:
        """Calculate the unrealized loss percentage for a position.

        For a long position (quantity > 0):
            loss = (entry_price - market_price) / entry_price

        For a short position (quantity < 0):
            loss = (market_price - entry_price) / entry_price

        Returns a positive value when the position is losing money,
        negative (or zero) when profitable.
        """
        entry_price = self._effective_entry_price(position)
        if entry_price <= 0:
            return 0.0

        # side_multiplier: +1 for long, -1 for short
        side_multiplier = 1.0 if position.quantity > 0 else -1.0

        # Positive return means profit; we want loss as a positive number
        pnl_pct = (position.market_price - entry_price) / entry_price * side_multiplier

        # Return loss as positive (negate the PnL)
        return -pnl_pct

    def _build_close_order(self, position: Position, loss_pct: float) -> Order:
        """Build a closing order for a position that breached the stop-loss.

        The closing side is opposite to the position direction:
        - Long (qty > 0) -> SELL
        - Short (qty < 0) -> BUY

        Quantity is the absolute position size (full close).
        """
        close_side = Side.SELL if position.quantity > 0 else Side.BUY
        abs_quantity = abs(position.quantity)
        notional = abs_quantity * position.market_price

        return Order(
            order_id=new_id("sl"),
            signal_id=new_id("sl_sig"),
            symbol=position.symbol,
            side=close_side,
            quantity=abs_quantity,
            limit_price=None,
            notional=notional,
            strategy=position.strategy,
            thesis=f"stop_loss: unrealized loss {loss_pct:.4f} exceeds {self.max_position_loss_pct:.4f}",
            created_at=utc_now(),
        )
