from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from aiswarm.types.portfolio import PortfolioSnapshot
from aiswarm.types.risk import RiskEvent

_MAX_RISK_EVENTS = 500


@dataclass
class MandatePnLTracker:
    """Per-mandate P&L and exposure tracking."""

    mandate_id: str
    daily_pnl: float = 0.0
    peak_nav: float = 0.0
    current_nav: float = 0.0
    gross_exposure: float = 0.0

    @property
    def drawdown(self) -> float:
        if self.peak_nav <= 0:
            return 0.0
        return (self.peak_nav - self.current_nav) / self.peak_nav


@dataclass
class SharedMemory:
    latest_snapshot: PortfolioSnapshot | None = None
    latest_pnl: float = 0.0
    rolling_drawdown: float = 0.0
    current_leverage: float = 0.0
    peak_nav: float = 0.0
    risk_events: deque[RiskEvent] = field(default_factory=lambda: deque(maxlen=_MAX_RISK_EVENTS))
    metadata: dict[str, str] = field(default_factory=dict)
    mandate_trackers: dict[str, MandatePnLTracker] = field(default_factory=dict)

    def update_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self.latest_snapshot = snapshot
        # Track peak NAV for drawdown calculation
        if snapshot.nav > self.peak_nav:
            self.peak_nav = snapshot.nav
        if self.peak_nav > 0:
            self.rolling_drawdown = (self.peak_nav - snapshot.nav) / self.peak_nav
        # Derive leverage from exposure and NAV
        if snapshot.nav > 0:
            self.current_leverage = snapshot.gross_exposure / snapshot.nav

    def record_risk_event(self, event: RiskEvent) -> None:
        self.risk_events.append(event)

    def get_mandate_tracker(self, mandate_id: str) -> MandatePnLTracker:
        """Get or create a per-mandate P&L tracker."""
        if mandate_id not in self.mandate_trackers:
            self.mandate_trackers[mandate_id] = MandatePnLTracker(mandate_id=mandate_id)
        return self.mandate_trackers[mandate_id]

    def update_mandate_pnl(self, mandate_id: str, pnl_delta: float) -> None:
        """Update P&L for a specific mandate."""
        tracker = self.get_mandate_tracker(mandate_id)
        tracker.daily_pnl += pnl_delta
        tracker.current_nav += pnl_delta
        if tracker.current_nav > tracker.peak_nav:
            tracker.peak_nav = tracker.current_nav

    def reset_daily_mandate_pnl(self) -> None:
        """Reset daily P&L for all mandates (called at session start)."""
        for tracker in self.mandate_trackers.values():
            tracker.daily_pnl = 0.0
