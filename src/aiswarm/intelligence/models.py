"""Domain models for the Alpha Intelligence Engine.

All models use Pydantic v2 with frozen=True to match AIS conventions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TradingStyle(str, Enum):
    """Classified trading strategy style."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPER = "scalper"
    SWING = "swing"
    TREND_FOLLOWING = "trend_following"
    CONTRARIAN = "contrarian"
    UNKNOWN = "unknown"


class TraderTier(str, Enum):
    """Trader ranking tier based on consistency and performance."""

    ELITE = "elite"  # Top 1% — sustained edge, high Sharpe
    STRONG = "strong"  # Top 5% — consistently profitable
    NOTABLE = "notable"  # Top 15% — above average
    AVERAGE = "average"  # Median performers
    WEAK = "weak"  # Below average


class ActivitySource(str, Enum):
    """Where trade activity data was sourced from."""

    LEADERBOARD = "leaderboard"
    TRADE_FEED = "trade_feed"
    ON_CHAIN = "on_chain"
    WHALE_ALERT = "whale_alert"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class TradeActivity(BaseModel, frozen=True):
    """A single observed trade from a tracked trader."""

    activity_id: str
    trader_id: str
    exchange: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    notional: float = Field(gt=0)
    timestamp: datetime
    source: ActivitySource
    pnl: float | None = None  # Realized P&L if known
    holding_minutes: int | None = None  # How long position was held


class TraderProfile(BaseModel, frozen=True):
    """Statistical profile of a tracked trader."""

    trader_id: str
    exchange: str
    display_name: str = ""
    tier: TraderTier = TraderTier.AVERAGE

    # Performance metrics
    total_trades: int = 0
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_return_pct: float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss

    # Behavioral metrics
    avg_holding_minutes: float = 0.0
    trade_frequency_daily: float = 0.0
    preferred_symbols: tuple[str, ...] = ()
    preferred_side: str = ""  # "LONG", "SHORT", or "BOTH"
    avg_position_size_usd: float = 0.0
    max_position_size_usd: float = 0.0

    # Classification
    primary_style: TradingStyle = TradingStyle.UNKNOWN
    secondary_style: TradingStyle | None = None
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Metadata
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    last_updated: datetime | None = None


class StrategyFingerprint(BaseModel, frozen=True):
    """Extracted strategy characteristics from a trader's behavior."""

    trader_id: str
    style: TradingStyle

    # Entry patterns
    entry_timing: str = ""  # "breakout", "pullback", "reversal", "momentum_continuation"
    avg_entry_distance_from_ma_pct: float = 0.0  # Distance from moving average at entry
    prefers_limit_orders: bool = False
    typical_entry_hour_utc: int | None = None  # Most common trading hour

    # Exit patterns
    avg_winner_hold_minutes: float = 0.0
    avg_loser_hold_minutes: float = 0.0
    uses_stop_loss: bool = False
    avg_stop_distance_pct: float = 0.0
    avg_take_profit_pct: float = 0.0

    # Sizing patterns
    scales_in: bool = False  # Adds to winning positions
    scales_out: bool = False  # Partial exits
    size_increases_with_confidence: bool = False

    # Market condition preferences
    prefers_trending: bool = False
    prefers_ranging: bool = False
    avoids_high_volatility: bool = False

    # Confidence in this fingerprint
    sample_size: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LeaderboardEntry(BaseModel, frozen=True):
    """A snapshot of a trader's position on an exchange leaderboard."""

    trader_id: str
    exchange: str
    rank: int = Field(gt=0)
    display_name: str = ""
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    roi_7d: float = 0.0
    roi_30d: float = 0.0
    roi_90d: float = 0.0
    followers: int = 0
    win_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    snapshot_time: datetime | None = None


class TraderSignal(BaseModel, frozen=True):
    """A signal derived from a top trader's recent activity.

    This is an internal representation before conversion to an AIS Signal.
    """

    trader_id: str
    trader_tier: TraderTier
    symbol: str
    direction: int  # 1 = long, -1 = short
    trader_confidence: float = Field(ge=0.0, le=1.0)
    strategy_match: TradingStyle = TradingStyle.UNKNOWN
    reasoning: str = ""
    source_activity_id: str = ""
    detected_at: datetime | None = None
