"""Live order executor — submits orders to Aster DEX via MCP gateway.

Bridges the gap between the OMS (token validation) and actual exchange execution.
Uses the MCPGateway protocol for testability with MockMCPGateway.
"""

from __future__ import annotations

from dataclasses import dataclass

from aiswarm.data.providers.aster_config import Venue
from aiswarm.execution.aster_executor import AsterExecutor, ExecutionMode
from aiswarm.execution.mcp_gateway import MCPGateway
from aiswarm.execution.order_store import OrderStore
from aiswarm.types.orders import Order
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SubmissionResult:
    """Result of submitting an order to the exchange."""

    success: bool
    order_id: str
    exchange_order_id: str | None
    message: str


class LiveOrderExecutor:
    """Submits orders to Aster DEX via MCP and tracks their lifecycle.

    Wraps AsterExecutor (parameter builder) + MCPGateway (tool invoker)
    + OrderStore (persistence) into a single execution service.
    """

    def __init__(
        self,
        executor: AsterExecutor,
        gateway: MCPGateway,
        order_store: OrderStore,
        default_venue: Venue = Venue.FUTURES,
    ) -> None:
        self.executor = executor
        self.gateway = gateway
        self.order_store = order_store
        self.default_venue = default_venue

    def submit_order(
        self,
        order: Order,
        venue: Venue | None = None,
    ) -> SubmissionResult:
        """Submit an order to the exchange.

        1. Track the order in OrderStore
        2. Prepare MCP parameters via AsterExecutor
        3. Invoke the MCP tool via gateway
        4. Record the exchange order ID
        """
        venue = venue or self.default_venue

        # Paper mode: simulate fill, don't call MCP
        if self.executor.mode == ExecutionMode.PAPER:
            return self._handle_paper(order)

        # Track the order
        self.order_store.track(order, venue=venue.value)

        # Prepare and submit
        try:
            if venue == Venue.FUTURES:
                params = self.executor.prepare_futures_order(order)
                tool_name = "mcp__aster__create_order"
            else:
                params = self.executor.prepare_spot_order(order)
                tool_name = "mcp__aster__create_spot_order"

            response = self.gateway.call_tool(tool_name, params)
            exchange_order_id = response.get("orderId", "")

            if not exchange_order_id:
                logger.error(
                    "Exchange returned no order ID",
                    extra={"extra_json": {"order_id": order.order_id, "response": response}},
                )
                return SubmissionResult(
                    success=False,
                    order_id=order.order_id,
                    exchange_order_id=None,
                    message=f"Exchange returned no order ID: {response}",
                )

            self.order_store.record_submission(order.order_id, exchange_order_id)

            logger.info(
                "Order submitted to exchange",
                extra={
                    "extra_json": {
                        "order_id": order.order_id,
                        "exchange_order_id": exchange_order_id,
                        "symbol": order.symbol,
                    }
                },
            )
            return SubmissionResult(
                success=True,
                order_id=order.order_id,
                exchange_order_id=exchange_order_id,
                message="Order submitted successfully",
            )

        except Exception as e:
            logger.error(
                "Order submission failed",
                extra={"extra_json": {"order_id": order.order_id, "error": str(e)}},
            )
            self.order_store.record_cancel(order.order_id, reason=f"submission_failed: {e}")
            return SubmissionResult(
                success=False,
                order_id=order.order_id,
                exchange_order_id=None,
                message=f"Submission failed: {e}",
            )

    def cancel_order(
        self,
        order_id: str,
        venue: Venue | None = None,
    ) -> SubmissionResult:
        """Cancel an order on the exchange."""
        venue = venue or self.default_venue
        record = self.order_store.get(order_id)
        if record is None:
            return SubmissionResult(
                success=False,
                order_id=order_id,
                exchange_order_id=None,
                message="Order not found in store",
            )
        if record.exchange_order_id is None:
            return SubmissionResult(
                success=False,
                order_id=order_id,
                exchange_order_id=None,
                message="Order has no exchange ID (not yet submitted?)",
            )

        try:
            params = self.executor.prepare_cancel_order(
                symbol=record.order.symbol,
                order_id=record.exchange_order_id,
                venue=venue,
            )
            tool_name = params.pop("tool")
            self.gateway.call_tool(tool_name, params)
            self.order_store.record_cancel(order_id, reason="operator_cancel")
            return SubmissionResult(
                success=True,
                order_id=order_id,
                exchange_order_id=record.exchange_order_id,
                message="Order cancelled",
            )
        except Exception as e:
            return SubmissionResult(
                success=False,
                order_id=order_id,
                exchange_order_id=record.exchange_order_id,
                message=f"Cancel failed: {e}",
            )

    def cancel_for_symbols(self, symbols: list[str]) -> list[SubmissionResult]:
        """Surgical cancel — cancel orders only for the specified symbols.

        Unlike ``cancel_all`` which is a nuclear option across all whitelisted
        symbols, this method targets specific symbols (e.g. those flagged by
        reconciliation mismatches).
        """
        results: list[SubmissionResult] = []

        for symbol in symbols:
            for venue in (Venue.FUTURES, Venue.SPOT):
                try:
                    params = self.executor.prepare_cancel_all(symbol, venue)
                    tool_name = params.pop("tool")
                    self.gateway.call_tool(tool_name, params)
                    results.append(
                        SubmissionResult(
                            success=True,
                            order_id=f"cancel_symbol_{symbol}_{venue.value}",
                            exchange_order_id=None,
                            message=f"Surgical cancel {venue.value} for {symbol}",
                        )
                    )
                except Exception as e:
                    results.append(
                        SubmissionResult(
                            success=False,
                            order_id=f"cancel_symbol_{symbol}_{venue.value}",
                            exchange_order_id=None,
                            message=f"Surgical cancel failed: {e}",
                        )
                    )

            # Mark only this symbol's open orders as cancelled
            for record in self.order_store.get_open_orders_for_symbol(symbol):
                self.order_store.record_cancel(
                    record.order.order_id,
                    reason=f"surgical_reconciliation_cancel:{symbol}",
                )

        logger.info(
            "Surgical cancel completed",
            extra={
                "extra_json": {
                    "symbols": symbols,
                    "results_count": len(results),
                }
            },
        )
        return results

    def cancel_all(self, symbols: list[str] | None = None) -> list[SubmissionResult]:
        """Emergency cancel all orders across symbols."""
        from aiswarm.data.providers.aster_config import WHITELISTED_SYMBOLS

        symbols = symbols or WHITELISTED_SYMBOLS
        results: list[SubmissionResult] = []

        for symbol in symbols:
            for venue in (Venue.FUTURES, Venue.SPOT):
                try:
                    params = self.executor.prepare_cancel_all(symbol, venue)
                    tool_name = params.pop("tool")
                    self.gateway.call_tool(tool_name, params)
                    results.append(
                        SubmissionResult(
                            success=True,
                            order_id=f"cancel_all_{symbol}_{venue.value}",
                            exchange_order_id=None,
                            message=f"Cancel all {venue.value} for {symbol}",
                        )
                    )
                except Exception as e:
                    results.append(
                        SubmissionResult(
                            success=False,
                            order_id=f"cancel_all_{symbol}_{venue.value}",
                            exchange_order_id=None,
                            message=f"Cancel all failed: {e}",
                        )
                    )

        # Mark all tracked open orders as cancelled
        for record in self.order_store.get_open_orders():
            self.order_store.record_cancel(record.order.order_id, reason="emergency_cancel_all")

        return results

    def _handle_paper(self, order: Order) -> SubmissionResult:
        """Handle paper mode: simulate fill immediately."""
        self.order_store.track(order, venue="paper")
        result = self.executor.simulate_paper_fill(order, current_price=0.0)
        exchange_id = result.exchange_order_id or f"paper_{order.order_id}"
        self.order_store.record_submission(order.order_id, exchange_id)
        if result.fill_price is not None and result.fill_quantity is not None:
            self.order_store.record_fill(order.order_id, result.fill_price, result.fill_quantity)
        return SubmissionResult(
            success=True,
            order_id=order.order_id,
            exchange_order_id=exchange_id,
            message="Paper fill simulated",
        )
