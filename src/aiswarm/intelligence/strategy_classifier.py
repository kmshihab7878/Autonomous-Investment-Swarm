"""Strategy Classifier — reverse-engineers trading styles from trade patterns.

Analyzes a trader's historical trades to classify their approach:
momentum, mean reversion, breakout, scalping, swing, trend following,
or contrarian. Extracts timing, sizing, and exit patterns into a
StrategyFingerprint.
"""

from __future__ import annotations

import math
from collections import Counter

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import (
    StrategyFingerprint,
    TradeActivity,
    TradingStyle,
)
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

# Holding time thresholds (minutes)
_SCALPER_MAX = 15
_INTRADAY_MAX = 480  # 8 hours
_SWING_MAX = 10080  # 7 days


class StrategyClassifier:
    """Classifies a trader's strategy from their trade history."""

    def __init__(self, store: AlphaStore) -> None:
        self.store = store

    def classify(self, trader_id: str) -> StrategyFingerprint:
        """Analyze trade history and produce a strategy fingerprint.

        Requires at least 10 trades with P&L data for meaningful
        classification.
        """
        activities = self.store.get_activities(trader_id=trader_id, limit=2000)
        with_pnl = [a for a in activities if a.pnl is not None]

        if len(with_pnl) < 10:
            fp = StrategyFingerprint(
                trader_id=trader_id,
                style=TradingStyle.UNKNOWN,
                sample_size=len(with_pnl),
                confidence=0.0,
            )
            self.store.save_fingerprint(fp)
            return fp

        # Classify primary style
        style = self._classify_style(with_pnl)

        # Extract patterns
        entry_timing = self._detect_entry_timing(with_pnl)
        avg_entry_dist = self._avg_entry_distance(with_pnl)
        typical_hour = self._most_common_hour(with_pnl)

        # Exit patterns
        winners = [a for a in with_pnl if a.pnl is not None and a.pnl > 0]
        losers = [a for a in with_pnl if a.pnl is not None and a.pnl < 0]
        avg_winner_hold = self._avg_holding(winners)
        avg_loser_hold = self._avg_holding(losers)
        uses_stop = avg_loser_hold < avg_winner_hold * 0.5 if avg_winner_hold > 0 else False
        avg_winner_return = self._avg_return(winners)
        avg_loser_return = abs(self._avg_return(losers))

        # Sizing patterns
        scales_in = self._detects_scaling_in(activities)
        scales_out = self._detects_scaling_out(activities)

        # Market condition preferences
        prefers_trending = style in (
            TradingStyle.MOMENTUM,
            TradingStyle.TREND_FOLLOWING,
            TradingStyle.BREAKOUT,
        )
        prefers_ranging = style in (TradingStyle.MEAN_REVERSION, TradingStyle.SCALPER)

        # Confidence based on sample size
        confidence = min(1.0, len(with_pnl) / 100)

        fp = StrategyFingerprint(
            trader_id=trader_id,
            style=style,
            entry_timing=entry_timing,
            avg_entry_distance_from_ma_pct=avg_entry_dist,
            typical_entry_hour_utc=typical_hour,
            avg_winner_hold_minutes=avg_winner_hold,
            avg_loser_hold_minutes=avg_loser_hold,
            uses_stop_loss=uses_stop,
            avg_stop_distance_pct=avg_loser_return * 100,
            avg_take_profit_pct=avg_winner_return * 100,
            scales_in=scales_in,
            scales_out=scales_out,
            prefers_trending=prefers_trending,
            prefers_ranging=prefers_ranging,
            sample_size=len(with_pnl),
            confidence=confidence,
        )

        self.store.save_fingerprint(fp)
        logger.info(
            "Strategy classified",
            extra={
                "extra_json": {
                    "trader_id": trader_id,
                    "style": style.value,
                    "confidence": round(confidence, 2),
                    "sample_size": len(with_pnl),
                }
            },
        )
        return fp

    def _classify_style(self, activities: list[TradeActivity]) -> TradingStyle:
        """Classify the primary trading style from trade patterns."""
        holding_times = [a.holding_minutes for a in activities if a.holding_minutes is not None]
        avg_hold = sum(holding_times) / len(holding_times) if holding_times else 60

        # Win/loss characteristics
        winners = [a for a in activities if a.pnl is not None and a.pnl > 0]
        losers = [a for a in activities if a.pnl is not None and a.pnl < 0]
        win_rate = len(winners) / len(activities) if activities else 0

        # Return distribution
        returns = [a.pnl / a.notional for a in activities if a.pnl is not None and a.notional > 0]
        avg_return = sum(returns) / len(returns) if returns else 0

        # Classify by holding period first
        if avg_hold <= _SCALPER_MAX:
            return TradingStyle.SCALPER

        # Check for contrarian signals (low win rate but large winners)
        if win_rate < 0.4 and avg_return > 0:
            avg_winner = self._avg_return(winners)
            avg_loser = abs(self._avg_return(losers))
            if avg_loser > 0 and avg_winner / avg_loser > 3.0:
                return TradingStyle.CONTRARIAN

        # Check for mean reversion (high win rate, small returns)
        if win_rate > 0.65 and self._return_std(returns) < 0.02:
            return TradingStyle.MEAN_REVERSION

        # Check for breakout (lower win rate, larger winners)
        if win_rate < 0.5:
            avg_winner_r = self._avg_return(winners)
            avg_loser_r = abs(self._avg_return(losers))
            if avg_loser_r > 0 and avg_winner_r / avg_loser_r > 2.0:
                return TradingStyle.BREAKOUT

        # Swing vs momentum vs trend following by holding period
        if avg_hold > _INTRADAY_MAX:
            if avg_hold > _SWING_MAX:
                return TradingStyle.TREND_FOLLOWING
            return TradingStyle.SWING

        return TradingStyle.MOMENTUM

    def _detect_entry_timing(self, activities: list[TradeActivity]) -> str:
        """Detect whether entries tend to happen on breakouts, pullbacks, etc."""
        # Simplified: use side distribution and holding period patterns
        buys = sum(1 for a in activities if a.side == "BUY")
        total = len(activities)

        if total == 0:
            return "unknown"

        buy_ratio = buys / total
        if buy_ratio > 0.7:
            return "dip_buyer"
        elif buy_ratio < 0.3:
            return "rally_shorter"
        else:
            return "mixed"

    def _avg_entry_distance(self, activities: list[TradeActivity]) -> float:
        """Estimate average distance from moving average at entry (approximation)."""
        if len(activities) < 20:
            return 0.0
        prices = [a.price for a in activities]
        ma = sum(prices) / len(prices)
        distances = [abs(p - ma) / ma for p in prices]
        return sum(distances) / len(distances) * 100

    def _most_common_hour(self, activities: list[TradeActivity]) -> int | None:
        """Find the most common trading hour (UTC)."""
        if not activities:
            return None
        hours = [a.timestamp.hour for a in activities]
        counter = Counter(hours)
        return counter.most_common(1)[0][0]

    def _avg_holding(self, activities: list[TradeActivity]) -> float:
        """Average holding time in minutes."""
        times = [a.holding_minutes for a in activities if a.holding_minutes is not None]
        return sum(times) / len(times) if times else 0.0

    def _avg_return(self, activities: list[TradeActivity]) -> float:
        """Average return as a fraction."""
        returns = [a.pnl / a.notional for a in activities if a.pnl is not None and a.notional > 0]
        return sum(returns) / len(returns) if returns else 0.0

    def _return_std(self, returns: list[float]) -> float:
        """Standard deviation of returns."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    def _detects_scaling_in(self, activities: list[TradeActivity]) -> bool:
        """Detect if a trader tends to add to positions (scale in)."""
        if len(activities) < 10:
            return False
        # Look for consecutive same-side trades on the same symbol
        consecutive = 0
        for i in range(1, len(activities)):
            if (
                activities[i].symbol == activities[i - 1].symbol
                and activities[i].side == activities[i - 1].side
            ):
                consecutive += 1
        return consecutive / len(activities) > 0.2

    def _detects_scaling_out(self, activities: list[TradeActivity]) -> bool:
        """Detect if a trader tends to exit in parts (scale out)."""
        if len(activities) < 10:
            return False
        # Look for decreasing quantities on same-side exits
        partial_exits = 0
        for i in range(1, len(activities)):
            if (
                activities[i].symbol == activities[i - 1].symbol
                and activities[i].side != activities[i - 1].side
                and activities[i].quantity < activities[i - 1].quantity
            ):
                partial_exits += 1
        return partial_exits / len(activities) > 0.1
