from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from aiswarm.data.providers.aster_config import (
    WHITELISTED_SYMBOLS,
)
from aiswarm.utils.logging import get_logger

if TYPE_CHECKING:
    from aiswarm.execution.live_executor import LiveOrderExecutor

logger = get_logger(__name__)


class KillSwitch:
    """Emergency trading halt mechanism.

    Triggers when daily P&L fraction breaches the max_daily_loss threshold.
    When triggered in live mode, prepares cancel-all instructions for
    all active symbols across both spot and futures venues.

    If an executor is injected via ``set_executor()``, the kill switch is
    **self-enforcing**: it will call ``cancel_all()`` directly when triggered,
    independent of the trading loop.  If no executor is set, it behaves
    identically to the original advisory-only mode (backward compatible).
    """

    def __init__(self, max_daily_loss: float) -> None:
        self.max_daily_loss = max_daily_loss
        self._triggered = False
        self._executor: LiveOrderExecutor | None = None
        self._redis_client: Any | None = None

    # ------------------------------------------------------------------
    # Executor injection (post-construction to avoid circular deps)
    # ------------------------------------------------------------------

    def set_executor(self, executor: LiveOrderExecutor) -> None:
        """Inject a LiveOrderExecutor for self-enforcing cancellations.

        Must be called after both KillSwitch and LiveOrderExecutor are
        constructed.  Safe to call multiple times (last writer wins).
        """
        self._executor = executor
        logger.info("Kill switch executor configured for auto-cancel")

    def set_redis_client(self, redis_client: Any) -> None:
        """Optionally inject a Redis client for trigger notifications."""
        self._redis_client = redis_client

    # ------------------------------------------------------------------
    # Core trigger logic
    # ------------------------------------------------------------------

    def triggered(self, daily_pnl_fraction: float) -> bool:
        if daily_pnl_fraction <= -abs(self.max_daily_loss):
            if not self._triggered:
                logger.critical(
                    "KILL SWITCH TRIGGERED",
                    extra={
                        "extra_json": {
                            "daily_pnl_fraction": daily_pnl_fraction,
                            "threshold": -abs(self.max_daily_loss),
                        }
                    },
                )
                self._triggered = True
                self._execute_emergency_cancels()
                self._notify_redis()
            return True
        return False

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        """Reset kill switch (requires manual intervention)."""
        logger.warning("Kill switch manually reset")
        self._triggered = False

    # ------------------------------------------------------------------
    # Emergency cancel execution
    # ------------------------------------------------------------------

    def execute_emergency_cancels(
        self,
        executor: LiveOrderExecutor | None = None,
    ) -> list[dict[str, Any]]:
        """Cancel all open orders via the executor.

        Uses the injected executor by default, or an explicit override.
        Returns a list of result dicts summarising each cancel attempt.

        Never raises -- logs errors but does not crash (same pattern as
        AlertDispatcher).
        """
        target_executor = executor or self._executor
        if target_executor is None:
            logger.warning("Kill switch has no executor — cancel is advisory only")
            return []

        try:
            results = target_executor.cancel_all()
            summary = [
                {
                    "order_id": r.order_id,
                    "success": r.success,
                    "message": r.message,
                }
                for r in results
            ]
            succeeded = sum(1 for r in results if r.success)
            failed = len(results) - succeeded
            logger.critical(
                "Kill switch emergency cancel executed",
                extra={
                    "extra_json": {
                        "total": len(results),
                        "succeeded": succeeded,
                        "failed": failed,
                    }
                },
            )
            return summary
        except Exception as exc:
            logger.error(
                "Kill switch emergency cancel failed",
                extra={"extra_json": {"error": str(exc)}},
            )
            return [{"error": str(exc), "success": False}]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_emergency_cancels(self) -> None:
        """Auto-execute cancels on first trigger (fire-and-forget)."""
        if self._executor is not None:
            self.execute_emergency_cancels()

    def _notify_redis(self) -> None:
        """Publish trigger timestamp to Redis (best-effort)."""
        if self._redis_client is None:
            return
        try:
            self._redis_client.set(
                "ais:kill_switch:triggered",
                str(time.time()),
            )
            logger.info("Kill switch trigger published to Redis")
        except Exception as exc:
            logger.error(
                "Failed to publish kill switch trigger to Redis",
                extra={"extra_json": {"error": str(exc)}},
            )

    # ------------------------------------------------------------------
    # Legacy helper (unchanged API)
    # ------------------------------------------------------------------

    def prepare_emergency_cancels(
        self,
        account_id: str,
        symbols: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Prepare MCP cancel-all-orders calls for emergency shutdown.

        Returns a list of MCP tool parameter dicts that the caller
        should execute to cancel all open orders.
        """
        targets = symbols or WHITELISTED_SYMBOLS
        cancels: list[dict[str, str]] = []
        for symbol in targets:
            cancels.append(
                {
                    "tool": "mcp__aster__cancel_all_orders",
                    "account_id": account_id,
                    "symbol": symbol,
                }
            )
            cancels.append(
                {
                    "tool": "mcp__aster__cancel_spot_all_orders",
                    "account_id": account_id,
                    "symbol": symbol,
                }
            )
        return cancels
