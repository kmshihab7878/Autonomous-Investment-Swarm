"""Trader Profiler — builds statistical profiles from trade data.

Analyzes a trader's historical trades to compute performance metrics,
behavioral patterns, and consistency scores. Profiles are used to
rank traders and inform signal confidence.
"""

from __future__ import annotations

import math

from aiswarm.intelligence.alpha_store import AlphaStore
from aiswarm.intelligence.models import (
    TradeActivity,
    TraderProfile,
    TraderTier,
)
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class TraderProfiler:
    """Builds and updates statistical profiles for tracked traders."""

    def __init__(self, store: AlphaStore) -> None:
        self.store = store

    def build_profile(
        self,
        trader_id: str,
        exchange: str,
        display_name: str = "",
    ) -> TraderProfile:
        """Build or rebuild a complete profile from stored trade activities."""
        activities = self.store.get_activities(trader_id=trader_id, limit=5000)

        if not activities:
            return TraderProfile(
                trader_id=trader_id,
                exchange=exchange,
                display_name=display_name,
                last_updated=utc_now(),
            )

        # Performance metrics
        returns = self._compute_returns(activities)
        win_rate = self._compute_win_rate(activities)
        total_return = sum(returns) if returns else 0.0
        avg_return = total_return / len(returns) if returns else 0.0
        sharpe = self._compute_sharpe(returns)
        sortino = self._compute_sortino(returns)
        max_dd = self._compute_max_drawdown(returns)
        profit_factor = self._compute_profit_factor(activities)

        # Behavioral metrics
        holding_times = [a.holding_minutes for a in activities if a.holding_minutes is not None]
        avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0.0
        symbols = [a.symbol for a in activities]
        preferred_symbols = tuple(self._top_n(symbols, 5))
        sides = [a.side for a in activities]
        buy_pct = sides.count("BUY") / len(sides) if sides else 0.5
        preferred_side = "LONG" if buy_pct > 0.6 else ("SHORT" if buy_pct < 0.4 else "BOTH")
        notionals = [a.notional for a in activities]
        avg_size = sum(notionals) / len(notionals) if notionals else 0.0
        max_size = max(notionals) if notionals else 0.0

        # Trade frequency
        if len(activities) >= 2:
            first = min(a.timestamp for a in activities)
            last = max(a.timestamp for a in activities)
            days = max((last - first).total_seconds() / 86400, 1.0)
            frequency = len(activities) / days
        else:
            frequency = 0.0

        # Tier classification
        tier = self._classify_tier(win_rate, sharpe, len(activities), max_dd)

        # Consistency score: how stable is the win rate over time?
        consistency = self._compute_consistency(activities)

        profile = TraderProfile(
            trader_id=trader_id,
            exchange=exchange,
            display_name=display_name,
            tier=tier,
            total_trades=len(activities),
            win_rate=win_rate,
            avg_return_pct=avg_return * 100,
            total_return_pct=total_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_holding_minutes=avg_holding,
            trade_frequency_daily=frequency,
            preferred_symbols=preferred_symbols,
            preferred_side=preferred_side,
            avg_position_size_usd=avg_size,
            max_position_size_usd=max_size,
            consistency_score=consistency,
            first_seen=min(a.timestamp for a in activities),
            last_seen=max(a.timestamp for a in activities),
            last_updated=utc_now(),
        )

        # Persist
        self.store.upsert_profile(profile)
        return profile

    def _compute_returns(self, activities: list[TradeActivity]) -> list[float]:
        """Compute per-trade returns from PnL and notional."""
        returns = []
        for a in activities:
            if a.pnl is not None and a.notional > 0:
                returns.append(a.pnl / a.notional)
        return returns

    def _compute_win_rate(self, activities: list[TradeActivity]) -> float:
        """Fraction of trades with positive P&L."""
        with_pnl = [a for a in activities if a.pnl is not None]
        if not with_pnl:
            return 0.0
        wins = sum(1 for a in with_pnl if a.pnl is not None and a.pnl > 0)
        return wins / len(with_pnl)

    def _compute_sharpe(self, returns: list[float], periods_per_year: float = 365.0) -> float:
        """Annualized Sharpe ratio from trade returns."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        return (mean / std) * math.sqrt(periods_per_year)

    def _compute_sortino(self, returns: list[float], periods_per_year: float = 365.0) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf") if mean > 0 else 0.0
        down_var = sum(r**2 for r in downside) / len(downside)
        down_std = math.sqrt(down_var) if down_var > 0 else 0.0
        if down_std == 0:
            return 0.0
        return (mean / down_std) * math.sqrt(periods_per_year)

    def _compute_max_drawdown(self, returns: list[float]) -> float:
        """Maximum drawdown from cumulative returns."""
        if not returns:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _compute_profit_factor(self, activities: list[TradeActivity]) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(a.pnl for a in activities if a.pnl is not None and a.pnl > 0)
        gross_loss = abs(sum(a.pnl for a in activities if a.pnl is not None and a.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _classify_tier(
        self,
        win_rate: float,
        sharpe: float,
        total_trades: int,
        max_drawdown: float,
    ) -> TraderTier:
        """Classify a trader into a performance tier."""
        if total_trades < 10:
            return TraderTier.WEAK

        score = 0.0
        # Win rate contribution (0-30 points)
        score += min(win_rate * 40, 30)
        # Sharpe contribution (0-30 points)
        score += min(max(sharpe, 0) * 10, 30)
        # Trade count contribution (0-20 points, logarithmic)
        score += min(math.log(total_trades + 1) * 4, 20)
        # Drawdown penalty (0 to -20 points)
        score -= min(max_drawdown * 100, 20)

        if score >= 60:
            return TraderTier.ELITE
        elif score >= 45:
            return TraderTier.STRONG
        elif score >= 30:
            return TraderTier.NOTABLE
        elif score >= 15:
            return TraderTier.AVERAGE
        else:
            return TraderTier.WEAK

    def _compute_consistency(self, activities: list[TradeActivity]) -> float:
        """Measure how consistent a trader's performance is over time.

        Splits trades into chunks and measures variance of win rates.
        Low variance = high consistency.
        """
        with_pnl = [a for a in activities if a.pnl is not None]
        if len(with_pnl) < 20:
            return 0.0

        chunk_size = max(len(with_pnl) // 5, 5)
        chunk_win_rates = []
        for i in range(0, len(with_pnl), chunk_size):
            chunk = with_pnl[i : i + chunk_size]
            if len(chunk) < 3:
                continue
            wr = sum(1 for a in chunk if a.pnl is not None and a.pnl > 0) / len(chunk)
            chunk_win_rates.append(wr)

        if len(chunk_win_rates) < 2:
            return 0.0

        mean = sum(chunk_win_rates) / len(chunk_win_rates)
        variance = sum((wr - mean) ** 2 for wr in chunk_win_rates) / len(chunk_win_rates)
        # Convert variance to 0-1 score (lower variance = higher consistency)
        return max(0.0, min(1.0, 1.0 - math.sqrt(variance) * 3))

    def _top_n(self, items: list[str], n: int) -> list[str]:
        """Return the top N most frequent items."""
        counts: dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        sorted_items = sorted(counts, key=lambda x: counts[x], reverse=True)
        return sorted_items[:n]
