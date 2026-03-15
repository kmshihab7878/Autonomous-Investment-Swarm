"""Market data service — fetches live data via MCP gateway for agent consumption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiswarm.data.providers.aster import AsterDataProvider
from aiswarm.execution.mcp_gateway import MCPGateway
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SymbolData:
    """Aggregated market data for a single symbol."""

    symbol: str
    klines_raw: Any | None = None
    funding_raw: Any | None = None
    ticker_raw: Any | None = None
    orderbook_raw: Any | None = None


class MarketDataService:
    """Fetches market data from Aster DEX via MCP gateway.

    Provides structured data contexts for agents to consume.
    """

    def __init__(
        self,
        gateway: MCPGateway,
        provider: AsterDataProvider | None = None,
    ) -> None:
        self.gateway = gateway
        self.provider = provider or AsterDataProvider()

    def fetch_symbol_data(
        self,
        symbol: str,
        klines_interval: str = "1h",
        klines_limit: int = 100,
    ) -> SymbolData:
        """Fetch all market data for a symbol.

        Calls klines, funding rate, ticker, and order book MCP tools.
        Returns raw responses for agents to parse.
        """
        klines_raw = self._fetch_safe(
            "mcp__aster__get_klines",
            {"symbol": symbol, "interval": klines_interval, "limit": klines_limit},
        )
        funding_raw = self._fetch_safe(
            "mcp__aster__get_funding_rate",
            {"symbol": symbol},
        )
        ticker_raw = self._fetch_safe(
            "mcp__aster__get_ticker",
            {"symbol": symbol},
        )
        orderbook_raw = self._fetch_safe(
            "mcp__aster__get_order_book",
            {"symbol": symbol},
        )

        return SymbolData(
            symbol=symbol,
            klines_raw=klines_raw,
            funding_raw=funding_raw,
            ticker_raw=ticker_raw,
            orderbook_raw=orderbook_raw,
        )

    def build_agent_context(self, data: SymbolData) -> dict[str, Any]:
        """Build context dict for agents from fetched market data."""
        ctx: dict[str, Any] = {"symbol": data.symbol}
        if data.klines_raw is not None:
            ctx["klines_data"] = data.klines_raw
        if data.funding_raw is not None:
            ctx["funding_data"] = data.funding_raw
        if data.ticker_raw is not None:
            ctx["ticker_data"] = data.ticker_raw
        if data.orderbook_raw is not None:
            ctx["orderbook_data"] = data.orderbook_raw
        return ctx

    def compute_liquidity_score(self, data: SymbolData, notional: float = 10000.0) -> float:
        """Compute liquidity score from order book data."""
        if data.orderbook_raw is None:
            return 0.5  # Default moderate score
        orderbook = self.provider.parse_orderbook_response(data.orderbook_raw, data.symbol)
        if orderbook is None:
            return 0.5
        return float(self.provider.compute_liquidity_score(orderbook, notional))

    def _fetch_safe(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Fetch data, returning None on error."""
        try:
            return self.gateway.call_tool(tool_name, params)
        except Exception as e:
            logger.warning(
                "Market data fetch failed",
                extra={"extra_json": {"tool": tool_name, "error": str(e)}},
            )
            return None
