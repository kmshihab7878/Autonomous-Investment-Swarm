"""Adapters that wrap existing AIS agents as backtest SignalGenerators.

Each adapter translates the backtest engine's ``(symbol, candles, position)``
interface into the ``context: dict`` that production agents expect, then
extracts the resulting ``Signal`` from the agent's response dict.

The adapters avoid any exchange / MCP dependency by constructing the raw
kline dicts that ``AsterDataProvider.parse_klines`` expects directly from
the backtest ``OHLCV`` bars.
"""

from __future__ import annotations

from aiswarm.agents.market_intelligence.funding_rate_agent import FundingRateAgent
from aiswarm.agents.strategy.momentum_agent import MomentumAgent
from aiswarm.backtest.engine import OHLCV
from aiswarm.types.market import Signal


class MomentumSignalGenerator:
    """Wraps ``MomentumAgent`` for use with the backtest engine.

    Converts backtest OHLCV bars into the raw kline dict format that the
    momentum agent's ``AsterDataProvider.parse_klines`` expects, calls
    ``analyze()``, and returns the resulting ``Signal`` (or ``None``).

    Parameters:
        fast_period: Fast moving-average window.
        slow_period: Slow moving-average window.
        min_candles: Minimum bars required before generating signals.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        min_candles: int = 50,
    ) -> None:
        self.agent = MomentumAgent(
            fast_period=fast_period,
            slow_period=slow_period,
            min_candles=min_candles,
        )

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        """Generate a momentum signal from backtest candle history."""
        # Convert backtest OHLCV to raw kline dicts the agent expects
        raw_klines = _ohlcv_to_raw_klines(candles)

        context: dict[str, object] = {
            "klines_data": raw_klines,
            "symbol": symbol,
        }
        result = self.agent.analyze(context)
        return result.get("signal")


class FundingRateSignalGenerator:
    """Wraps ``FundingRateAgent`` for use with the backtest engine.

    Since historical funding rate data is not available in OHLCV candles,
    this adapter synthesises a proxy funding rate from price momentum:
    strong recent uptrend implies positive funding (longs pay shorts),
    strong recent downtrend implies negative funding. This is a
    reasonable approximation for backtesting purposes because extreme
    price moves and extreme funding rates are highly correlated in
    perpetual futures markets.

    Parameters:
        lookback: Number of bars to measure price change over.
        rate_multiplier: Scaling factor from price return to funding rate.
        extreme_threshold: Passed through to ``FundingRateAgent``.
        high_threshold: Passed through to ``FundingRateAgent``.
    """

    def __init__(
        self,
        lookback: int = 8,
        rate_multiplier: float = 0.01,
        extreme_threshold: float = 0.001,
        high_threshold: float = 0.0005,
    ) -> None:
        self.agent = FundingRateAgent(
            extreme_threshold=extreme_threshold,
            high_threshold=high_threshold,
        )
        self.lookback = lookback
        self.rate_multiplier = rate_multiplier

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None:
        """Generate a funding-rate contrarian signal from price data.

        Derives a synthetic funding rate from recent price momentum,
        then feeds it through the real ``FundingRateAgent``.
        """
        if len(candles) < self.lookback + 1:
            return None

        # Synthetic funding rate from recent price change
        current_price = candles[-1].close
        past_price = candles[-(self.lookback + 1)].close
        if past_price <= 0:
            return None

        price_return = (current_price - past_price) / past_price
        synthetic_rate = price_return * self.rate_multiplier

        funding_data: dict[str, object] = {
            "symbol": symbol,
            "lastFundingRate": str(synthetic_rate),
            "markPrice": str(current_price),
            "nextFundingTime": 0,
        }

        context: dict[str, object] = {
            "funding_data": funding_data,
            "symbol": symbol,
        }
        result = self.agent.analyze(context)
        return result.get("signal")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ohlcv_to_raw_klines(candles: list[OHLCV]) -> list[dict[str, str]]:
    """Convert backtest OHLCV bars to the raw kline dict format.

    The ``AsterDataProvider.parse_klines`` method expects dicts with
    string values and keys like ``openTime``, ``open``, ``high``, etc.
    """
    raw: list[dict[str, str]] = []
    for c in candles:
        ts_ms = int(c.timestamp.timestamp() * 1000)
        raw.append(
            {
                "openTime": str(ts_ms),
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
            }
        )
    return raw
