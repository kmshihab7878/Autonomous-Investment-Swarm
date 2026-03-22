"""Grid trading agent — places contrarian orders at regular price intervals.

Strategy:
  - Divides a price range into grid levels
  - Generates long signals when price drops to a lower grid level
  - Generates short signals when price rises to an upper grid level
  - Confidence is constant per grid level (grid trading is systematic)
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import register_agent
from aiswarm.data.providers.aster import AsterDataProvider
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


@register_agent("grid_trading")
class GridAgent(Agent):
    """Generates grid trading signals at regular price intervals."""

    def __init__(
        self,
        agent_id: str = "grid_agent",
        cluster: str = "strategy",
        grid_levels: int = 10,
        grid_range_pct: float = 0.10,
        min_candles: int = 20,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.grid_levels = grid_levels
        self.grid_range_pct = grid_range_pct
        self.min_candles = min_candles
        self.provider = AsterDataProvider()
        self._last_grid_level: dict[str, int] = {}

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_klines = context.get("klines_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_klines is None:
            return {"signal": None, "reason": "no_klines_data"}

        candles = self.provider.parse_klines(raw_klines, symbol)
        if len(candles) < self.min_candles:
            return {"signal": None, "reason": f"insufficient_data: {len(candles)}"}

        # Compute grid center from recent average
        recent = candles[-self.min_candles :]
        center = sum(c.close for c in recent) / len(recent)
        price = candles[-1].close

        # Grid boundaries
        half_range = center * self.grid_range_pct / 2
        grid_low = center - half_range
        grid_high = center + half_range
        grid_step = (grid_high - grid_low) / self.grid_levels if self.grid_levels > 0 else 1

        # Determine current grid level (0 = bottom, grid_levels = top)
        if grid_step == 0:
            return {"signal": None, "reason": "zero_grid_step"}
        current_level = int((price - grid_low) / grid_step)
        current_level = max(0, min(self.grid_levels, current_level))

        prev_level = self._last_grid_level.get(symbol)
        self._last_grid_level[symbol] = current_level

        if prev_level is None:
            return {"signal": None, "reason": "first_observation", "grid_level": current_level}

        # No level change → no signal
        if current_level == prev_level:
            return {"signal": None, "reason": "same_grid_level", "grid_level": current_level}

        # Price dropped to lower level → buy (contrarian)
        if current_level < prev_level:
            direction = 1
        else:
            direction = -1

        levels_moved = abs(current_level - prev_level)
        confidence = min(0.70, 0.45 + levels_moved * 0.05)
        expected_return = levels_moved * grid_step / price if price > 0 else 0

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="grid_trading",
            thesis=(
                f"Grid {direction_str}: level {prev_level}→{current_level}, "
                f"price={price:.2f}, grid=[{grid_low:.2f},{grid_high:.2f}]"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=60,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
            reference_price=price,
        )

        logger.info(
            "Grid trading signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "direction": direction_str,
                    "from_level": prev_level,
                    "to_level": current_level,
                    "levels_moved": levels_moved,
                }
            },
        )
        return {"signal": signal, "grid_level": current_level, "levels_moved": levels_moved}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
