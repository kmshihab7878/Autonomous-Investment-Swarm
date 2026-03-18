"""Aster DEX data provider.

Wraps Aster DEX MCP tool calls into a clean Python interface.
Returns canonical types compatible with the AIS type system.

NOTE: Canonical data types have been moved to ``aiswarm.exchange.types``.
They are re-exported here for backward compatibility. New code should
import from ``aiswarm.exchange.types`` directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aiswarm.data.providers.aster_config import (
    AsterConfig,
    normalize_symbol,
    to_canonical_symbol,
)

# Re-export canonical types for backward compatibility
from aiswarm.exchange.types import (  # noqa: F401
    AccountBalance,
    ExchangeInfo,
    ExchangePosition,
    FundingRate,
    IncomeRecord,
    IncomeType,
    LeverageBracket,
    OHLCV,
    OrderBook,
    OrderBookLevel,
    Ticker,
    TradeRecord,
)

__all__ = [
    "AsterConfig",
    "AsterDataProvider",
    "normalize_symbol",
    "to_canonical_symbol",
    "AccountBalance",
    "ExchangeInfo",
    "ExchangePosition",
    "FundingRate",
    "IncomeRecord",
    "IncomeType",
    "LeverageBracket",
    "OHLCV",
    "OrderBook",
    "OrderBookLevel",
    "Ticker",
    "TradeRecord",
]
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


# --- Parser functions: convert raw MCP responses to canonical types ---


def parse_ohlcv(raw: dict[str, Any], symbol: str) -> OHLCV:
    """Parse a single kline entry from Aster DEX response."""
    ts = raw.get("openTime") or raw.get("timestamp") or raw.get("t", 0)
    ts_float = float(ts)
    if ts_float > 1e12:
        ts_float = ts_float / 1000  # ms to seconds
    return OHLCV(
        timestamp=datetime.fromtimestamp(ts_float, tz=timezone.utc),
        open=float(raw.get("open") or raw.get("o", 0)),
        high=float(raw.get("high") or raw.get("h", 0)),
        low=float(raw.get("low") or raw.get("l", 0)),
        close=float(raw.get("close") or raw.get("c", 0)),
        volume=float(raw.get("volume") or raw.get("v", 0)),
        symbol=to_canonical_symbol(symbol),
    )


def parse_ohlcv_list(raw_list: list[dict[str, Any]], symbol: str) -> list[OHLCV]:
    """Parse a list of kline entries."""
    return [parse_ohlcv(entry, symbol) for entry in raw_list]


def parse_ticker(raw: dict[str, Any]) -> Ticker:
    """Parse ticker response."""
    symbol = raw.get("symbol", "")
    return Ticker(
        symbol=to_canonical_symbol(symbol),
        last_price=float(raw.get("lastPrice", 0)),
        high_24h=float(raw.get("highPrice", 0)),
        low_24h=float(raw.get("lowPrice", 0)),
        volume_24h=float(raw.get("volume", 0)),
        price_change_pct=float(raw.get("priceChangePercent", 0)),
        timestamp=datetime.now(timezone.utc),
    )


def parse_order_book(raw: dict[str, Any], symbol: str) -> OrderBook:
    """Parse order book response."""
    bids = tuple(
        OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in raw.get("bids", [])
    )
    asks = tuple(
        OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in raw.get("asks", [])
    )
    return OrderBook(
        symbol=to_canonical_symbol(symbol),
        bids=bids,
        asks=asks,
        timestamp=datetime.now(timezone.utc),
    )


def parse_funding_rate(raw: dict[str, Any]) -> FundingRate:
    """Parse funding rate response."""
    symbol = raw.get("symbol", "")
    nft = raw.get("nextFundingTime")
    next_time = None
    if nft and isinstance(nft, (int, float)) and nft > 0:
        if nft > 1e12:
            nft = nft / 1000
        next_time = datetime.fromtimestamp(nft, tz=timezone.utc)
    return FundingRate(
        symbol=to_canonical_symbol(symbol),
        funding_rate=float(raw.get("lastFundingRate", 0)),
        mark_price=float(raw.get("markPrice", 0)),
        next_funding_time=next_time,
        timestamp=datetime.now(timezone.utc),
    )


def parse_balance(raw: dict[str, Any]) -> AccountBalance:
    """Parse balance response."""
    return AccountBalance(
        total_balance=float(raw.get("totalBalance") or raw.get("balance", 0)),
        available_balance=float(raw.get("availableBalance") or raw.get("withdrawAvailable", 0)),
        unrealized_pnl=float(raw.get("unrealizedProfit") or raw.get("crossUnPnl", 0)),
        margin_balance=float(raw.get("marginBalance", 0)),
        asset=raw.get("asset", "USDT"),
    )


def parse_position(raw: dict[str, Any]) -> ExchangePosition:
    """Parse a single position entry."""
    qty = float(raw.get("positionAmt", 0))
    side = "LONG" if qty >= 0 else "SHORT"
    return ExchangePosition(
        symbol=to_canonical_symbol(raw.get("symbol", "")),
        side=side,
        quantity=abs(qty),
        entry_price=float(raw.get("entryPrice", 0)),
        mark_price=float(raw.get("markPrice", 0)),
        unrealized_pnl=float(raw.get("unrealizedProfit", 0)),
        leverage=int(raw.get("leverage", 1)),
        margin_mode=raw.get("marginType", "CROSSED").upper(),
    )


def parse_trade(raw: dict[str, Any]) -> TradeRecord:
    """Parse a single trade/fill entry."""
    ts = raw.get("time", 0)
    if isinstance(ts, (int, float)) and ts > 1e12:
        ts = ts / 1000
    return TradeRecord(
        trade_id=str(raw.get("id", "")),
        symbol=to_canonical_symbol(raw.get("symbol", "")),
        side=raw.get("side", ""),
        price=float(raw.get("price", 0)),
        quantity=float(raw.get("qty", 0)),
        commission=float(raw.get("commission", 0)),
        commission_asset=raw.get("commissionAsset", "USDT"),
        realized_pnl=float(raw.get("realizedPnl", 0)),
        timestamp=datetime.fromtimestamp(float(ts), tz=timezone.utc),
        order_id=str(raw.get("orderId", "")),
    )


def parse_income(raw: dict[str, Any]) -> IncomeRecord:
    """Parse a single income entry."""
    ts = raw.get("time", 0)
    if isinstance(ts, (int, float)) and ts > 1e12:
        ts = ts / 1000
    return IncomeRecord(
        income_type=raw.get("incomeType", ""),
        amount=float(raw.get("income", 0)),
        asset=raw.get("asset", "USDT"),
        symbol=to_canonical_symbol(raw.get("symbol", "")),
        timestamp=datetime.fromtimestamp(float(ts), tz=timezone.utc),
    )


def parse_exchange_info(raw: dict[str, Any]) -> ExchangeInfo:
    """Parse a single symbol's exchange info."""
    filters = {f["filterType"]: f for f in raw.get("filters", [])}
    lot_filter = filters.get("LOT_SIZE", {})
    price_filter = filters.get("PRICE_FILTER", {})
    return ExchangeInfo(
        symbol=to_canonical_symbol(raw.get("symbol", "")),
        base_asset=raw.get("baseAsset", ""),
        quote_asset=raw.get("quoteAsset", ""),
        price_precision=int(raw.get("pricePrecision", 8)),
        quantity_precision=int(raw.get("quantityPrecision", 8)),
        min_quantity=float(lot_filter.get("minQty", 0)),
        max_quantity=float(lot_filter.get("maxQty", 0)),
        tick_size=float(price_filter.get("tickSize", 0)),
        status=raw.get("status", "TRADING"),
    )


def parse_leverage_bracket(raw: dict[str, Any]) -> LeverageBracket:
    """Parse a single leverage bracket entry."""
    return LeverageBracket(
        bracket=int(raw.get("bracket", 0)),
        initial_leverage=int(raw.get("initialLeverage", 1)),
        notional_cap=float(raw.get("notionalCap", 0)),
        notional_floor=float(raw.get("notionalFloor", 0)),
        maintenance_margin_rate=float(raw.get("maintMarginRatio", 0)),
    )


class AsterDataProvider:
    """High-level interface for Aster DEX market and account data.

    This class provides methods that accept raw MCP tool responses
    and return canonical AIS data types. It does NOT make MCP calls
    directly — the calling code is responsible for invoking the MCP
    tools and passing the responses here for parsing.

    Usage pattern:
        # Via ExchangeProvider (preferred):
        candles = exchange_provider.get_klines("BTCUSDT", interval="1h", limit=100)
        # Or direct parsing from raw MCP responses:
        candles = provider.parse_klines(raw_response, "BTCUSDT")
    """

    def __init__(self, config: AsterConfig | None = None) -> None:
        self.config = config or AsterConfig.from_env()

    # --- Market Data Parsers ---

    def parse_klines(self, raw_response: Any, symbol: str) -> list[OHLCV]:
        """Parse klines/candle response into OHLCV list."""
        if isinstance(raw_response, list):
            return parse_ohlcv_list(raw_response, normalize_symbol(symbol))
        if isinstance(raw_response, dict) and "data" in raw_response:
            return parse_ohlcv_list(raw_response["data"], normalize_symbol(symbol))
        logger.warning(
            "Unexpected klines response format",
            extra={"extra_json": {"type": type(raw_response).__name__}},
        )
        return []

    def parse_ticker_response(self, raw_response: Any) -> Ticker | None:
        """Parse ticker response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            return parse_ticker(data)
        return None

    def parse_orderbook_response(self, raw_response: Any, symbol: str) -> OrderBook | None:
        """Parse order book response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            return parse_order_book(data, normalize_symbol(symbol))
        return None

    def parse_funding_response(self, raw_response: Any) -> FundingRate | None:
        """Parse funding rate response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list) and data:
                return parse_funding_rate(data[0])
            return parse_funding_rate(data)
        return None

    def parse_exchange_info_response(self, raw_response: Any) -> list[ExchangeInfo]:
        """Parse exchange info response."""
        if isinstance(raw_response, dict):
            symbols = raw_response.get("symbols", [])
            if isinstance(symbols, list):
                return [parse_exchange_info(s) for s in symbols]
        return []

    # --- Account Data Parsers ---

    def parse_balance_response(self, raw_response: Any) -> AccountBalance | None:
        """Parse balance response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list) and data:
                return parse_balance(data[0])
            return parse_balance(data)
        return None

    def parse_positions_response(self, raw_response: Any) -> list[ExchangePosition]:
        """Parse positions response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list):
                return [parse_position(p) for p in data if float(p.get("positionAmt", 0)) != 0]
        if isinstance(raw_response, list):
            return [parse_position(p) for p in raw_response if float(p.get("positionAmt", 0)) != 0]
        return []

    def parse_trades_response(self, raw_response: Any) -> list[TradeRecord]:
        """Parse trade history response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list):
                return [parse_trade(t) for t in data]
        if isinstance(raw_response, list):
            return [parse_trade(t) for t in raw_response]
        return []

    def parse_income_response(self, raw_response: Any) -> list[IncomeRecord]:
        """Parse income/P&L response."""
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list):
                return [parse_income(i) for i in data]
        if isinstance(raw_response, list):
            return [parse_income(i) for i in raw_response]
        return []

    def parse_leverage_brackets_response(self, raw_response: Any) -> list[LeverageBracket]:
        """Parse leverage bracket response."""
        brackets: list[dict[str, Any]] = []
        if isinstance(raw_response, dict):
            data = raw_response.get("data", raw_response)
            if isinstance(data, list):
                for entry in data:
                    if "brackets" in entry:
                        brackets.extend(entry["brackets"])
                    else:
                        brackets.append(entry)
        if isinstance(raw_response, list):
            for entry in raw_response:
                if "brackets" in entry:
                    brackets.extend(entry["brackets"])
                else:
                    brackets.append(entry)
        return [parse_leverage_bracket(b) for b in brackets]

    # --- Utility ---

    def compute_liquidity_score(self, orderbook: OrderBook, notional: float) -> float:
        """Compute a 0-1 liquidity score based on order book depth vs order size.

        Returns 1.0 if the order book can easily absorb the notional,
        0.0 if it would consume all visible liquidity.
        """
        if notional <= 0:
            return 1.0
        depth = min(orderbook.bid_depth, orderbook.ask_depth)
        if depth <= 0:
            return 0.0
        # Score = 1 - (notional / depth), clamped to [0, 1]
        ratio = notional / depth
        return max(0.0, min(1.0, 1.0 - ratio))

    def is_funding_rate_extreme(
        self,
        funding: FundingRate,
        threshold: float = 0.001,
    ) -> tuple[bool, str]:
        """Check if funding rate is extreme (potential contrarian signal).

        Returns (is_extreme, direction) where direction is 'long' or 'short'.
        Extreme positive funding → contrarian short signal.
        Extreme negative funding → contrarian long signal.
        """
        if abs(funding.funding_rate) >= threshold:
            direction = "short" if funding.funding_rate > 0 else "long"
            return True, direction
        return False, ""
