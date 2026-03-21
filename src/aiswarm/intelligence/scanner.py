"""Trade Scanner — monitors exchanges for notable trades and whale activity.

Polls connected exchanges for recent trades, identifies large or unusual
activity, and feeds observations into the Alpha Intelligence pipeline.
"""

from __future__ import annotations


from aiswarm.exchange.provider import ExchangeProvider
from aiswarm.intelligence.models import ActivitySource, TradeActivity
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class TradeScanner:
    """Scans exchange trade feeds for notable activity.

    Identifies:
    - Large trades (whale activity) above a notional threshold
    - Volume anomalies (sudden spikes)
    - Unusual trading patterns across symbols
    """

    def __init__(
        self,
        whale_threshold_usd: float = 50_000.0,
        volume_spike_multiplier: float = 3.0,
    ) -> None:
        self.whale_threshold_usd = whale_threshold_usd
        self.volume_spike_multiplier = volume_spike_multiplier
        self._recent_volumes: dict[str, list[float]] = {}  # symbol → recent notionals
        self._seen_trade_ids: set[str] = set()

    def scan_exchange(
        self,
        provider: ExchangeProvider,
        symbols: list[str],
    ) -> list[TradeActivity]:
        """Scan an exchange for notable trades across the given symbols.

        Returns a list of TradeActivity observations for trades that
        exceed the whale threshold or represent volume anomalies.
        """
        activities: list[TradeActivity] = []

        for symbol in symbols:
            try:
                trades = provider.get_my_trades(symbol)
            except (NotImplementedError, Exception) as exc:
                logger.debug(
                    "Cannot fetch trades for %s on %s: %s",
                    symbol,
                    provider.exchange_id,
                    exc,
                )
                continue

            for trade in trades:
                if trade.trade_id in self._seen_trade_ids:
                    continue
                self._seen_trade_ids.add(trade.trade_id)

                notional = trade.price * trade.quantity
                is_whale = notional >= self.whale_threshold_usd
                is_spike = self._is_volume_spike(symbol, notional)

                if is_whale or is_spike:
                    activity = TradeActivity(
                        activity_id=new_id("act"),
                        trader_id=f"{provider.exchange_id}:{trade.order_id or trade.trade_id}",
                        exchange=provider.exchange_id,
                        symbol=symbol,
                        side=trade.side,
                        quantity=trade.quantity,
                        price=trade.price,
                        notional=notional,
                        timestamp=trade.timestamp,
                        source=ActivitySource.TRADE_FEED,
                        pnl=trade.realized_pnl if trade.realized_pnl != 0 else None,
                    )
                    activities.append(activity)
                    logger.info(
                        "Notable trade detected",
                        extra={
                            "extra_json": {
                                "exchange": provider.exchange_id,
                                "symbol": symbol,
                                "side": trade.side,
                                "notional": round(notional, 2),
                                "is_whale": is_whale,
                                "is_spike": is_spike,
                            }
                        },
                    )

                # Track volume for spike detection
                self._track_volume(symbol, notional)

        return activities

    def _is_volume_spike(self, symbol: str, notional: float) -> bool:
        """Check if a trade notional is a volume spike relative to recent history."""
        history = self._recent_volumes.get(symbol, [])
        if len(history) < 10:
            return False
        avg = sum(history) / len(history)
        if avg <= 0:
            return False
        return notional >= avg * self.volume_spike_multiplier

    def _track_volume(self, symbol: str, notional: float) -> None:
        """Track recent trade volumes for spike detection."""
        if symbol not in self._recent_volumes:
            self._recent_volumes[symbol] = []
        self._recent_volumes[symbol].append(notional)
        # Keep last 100 trades for rolling average
        if len(self._recent_volumes[symbol]) > 100:
            self._recent_volumes[symbol] = self._recent_volumes[symbol][-100:]

    @property
    def seen_count(self) -> int:
        """Total number of unique trades seen."""
        return len(self._seen_trade_ids)
