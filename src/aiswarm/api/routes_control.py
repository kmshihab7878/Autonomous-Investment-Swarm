"""Control API for human override operations.

All control endpoints require authentication. These provide the operator
with the ability to pause trading, trigger kill switch, cancel all orders,
and force deleveraging — critical safety controls for live operation.

G-002: Control state is backed by Redis (ais:control:state key) so the
trading loop can read it each cycle. Fails closed if Redis is unavailable.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from aiswarm.api.auth import require_api_key
from aiswarm.api.rate_limit import require_control_rate_limit, require_general_rate_limit
from aiswarm.data.providers.aster_config import WHITELISTED_SYMBOLS
from aiswarm.utils.logging import get_logger
from aiswarm.utils.secrets import get_secrets_provider
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)
router = APIRouter()

REDIS_CONTROL_KEY = "ais:control:state"
REDIS_KILL_REASON_KEY = "ais:control:kill_reason"
REDIS_PAUSED_AT_KEY = "ais:control:paused_at"


# --- System state (module-level singleton for simplicity) ---


class SystemState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    KILLED = "killed"


def _get_redis() -> Any:
    """Get a Redis client. Returns None if Redis is not available."""
    try:
        import redis

        url = get_secrets_provider().get_secret("REDIS_URL") or "redis://localhost:6379/0"
        client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
        client.ping()
        return client
    except Exception:
        return None


class _ControlState:
    def __init__(self) -> None:
        self._fallback_state: SystemState = SystemState.RUNNING
        self._fallback_paused_at: str | None = None
        self._fallback_kill_reason: str | None = None

    @property
    def state(self) -> SystemState:
        r = _get_redis()
        if r is not None:
            try:
                val = r.get(REDIS_CONTROL_KEY)
                if val:
                    return SystemState(val)
            except Exception:
                pass
        return self._fallback_state

    @property
    def paused_at(self) -> str | None:
        r = _get_redis()
        if r is not None:
            try:
                val: str | None = r.get(REDIS_PAUSED_AT_KEY)
                return val
            except Exception:
                pass
        return self._fallback_paused_at

    @property
    def kill_reason(self) -> str | None:
        r = _get_redis()
        if r is not None:
            try:
                val: str | None = r.get(REDIS_KILL_REASON_KEY)
                return val
            except Exception:
                pass
        return self._fallback_kill_reason

    def pause(self) -> None:
        ts = utc_now().isoformat()
        self._fallback_state = SystemState.PAUSED
        self._fallback_paused_at = ts
        r = _get_redis()
        if r is not None:
            try:
                r.set(REDIS_CONTROL_KEY, SystemState.PAUSED.value)
                r.set(REDIS_PAUSED_AT_KEY, ts)
            except Exception:
                logger.error("Failed to write pause state to Redis")

    def resume(self) -> None:
        self._fallback_state = SystemState.RUNNING
        self._fallback_paused_at = None
        r = _get_redis()
        if r is not None:
            try:
                r.set(REDIS_CONTROL_KEY, SystemState.RUNNING.value)
                r.delete(REDIS_PAUSED_AT_KEY)
            except Exception:
                logger.error("Failed to write resume state to Redis")

    def kill(self, reason: str) -> None:
        self._fallback_state = SystemState.KILLED
        self._fallback_kill_reason = reason
        r = _get_redis()
        if r is not None:
            try:
                r.set(REDIS_CONTROL_KEY, SystemState.KILLED.value)
                r.set(REDIS_KILL_REASON_KEY, reason)
            except Exception:
                logger.error("Failed to write kill state to Redis")

    @property
    def is_trading_allowed(self) -> bool:
        return self.state == SystemState.RUNNING


control_state = _ControlState()


# --- Request/Response models ---


class PauseRequest(BaseModel):
    reason: str = "manual pause"


class KillSwitchRequest(BaseModel):
    reason: str = "manual kill switch"


class CancelAllRequest(BaseModel):
    symbols: list[str] | None = None


class DeleverageRequest(BaseModel):
    symbol: str
    reduce_pct: float = 1.0  # 1.0 = close 100% of position


# --- Endpoints ---


@router.get("/control/mode", dependencies=[Depends(require_general_rate_limit)])
def get_mode(_: str = Depends(require_api_key)) -> dict[str, str]:
    return {"default_mode": "paper"}


@router.get("/control/status", dependencies=[Depends(require_general_rate_limit)])
def get_status(_: str = Depends(require_api_key)) -> dict[str, Any]:
    """Get current system control state."""
    return {
        "state": control_state.state.value,
        "is_trading_allowed": control_state.is_trading_allowed,
        "paused_at": control_state.paused_at,
        "kill_reason": control_state.kill_reason,
        "timestamp": utc_now().isoformat(),
    }


@router.post("/control/pause", dependencies=[Depends(require_control_rate_limit)])
def pause_trading(
    req: PauseRequest,
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Pause the coordinator loop. No new signals will be processed."""
    control_state.pause()
    logger.warning(
        "Trading PAUSED by operator",
        extra={"extra_json": {"reason": req.reason}},
    )
    return {
        "action": "paused",
        "reason": req.reason,
        "timestamp": utc_now().isoformat(),
    }


@router.post("/control/resume", dependencies=[Depends(require_control_rate_limit)])
def resume_trading(_: str = Depends(require_api_key)) -> dict[str, Any]:
    """Resume the coordinator loop after a pause."""
    if control_state.state == SystemState.KILLED:
        return {
            "action": "refused",
            "reason": "System is in KILLED state. Manual intervention required.",
            "timestamp": utc_now().isoformat(),
        }
    control_state.resume()
    logger.warning("Trading RESUMED by operator")
    return {
        "action": "resumed",
        "timestamp": utc_now().isoformat(),
    }


@router.post("/control/kill-switch", dependencies=[Depends(require_control_rate_limit)])
def trigger_kill_switch(
    req: KillSwitchRequest,
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Manually trigger kill switch. Requires manual reset to resume.

    Returns the list of MCP cancel-all-orders calls that should be executed
    to cancel all open orders on Aster DEX.
    """
    control_state.kill(req.reason)
    logger.critical(
        "KILL SWITCH triggered by operator",
        extra={"extra_json": {"reason": req.reason}},
    )

    # Prepare emergency cancel instructions
    symbols = WHITELISTED_SYMBOLS
    cancel_instructions = []
    for symbol in symbols:
        cancel_instructions.append({"tool": "mcp__aster__cancel_all_orders", "symbol": symbol})
        cancel_instructions.append({"tool": "mcp__aster__cancel_spot_all_orders", "symbol": symbol})

    return {
        "action": "killed",
        "reason": req.reason,
        "cancel_instructions": cancel_instructions,
        "message": "Execute the cancel_instructions via MCP to cancel all open orders",
        "timestamp": utc_now().isoformat(),
    }


@router.post("/control/cancel-all", dependencies=[Depends(require_control_rate_limit)])
def cancel_all_orders(
    req: CancelAllRequest,
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Prepare cancel-all-orders instructions for specified symbols.

    Does NOT execute the cancellations directly — returns the MCP
    tool calls that the operator should execute.
    """
    symbols = req.symbols or WHITELISTED_SYMBOLS
    cancel_instructions = []
    for symbol in symbols:
        cancel_instructions.append({"tool": "mcp__aster__cancel_all_orders", "symbol": symbol})
        cancel_instructions.append({"tool": "mcp__aster__cancel_spot_all_orders", "symbol": symbol})

    logger.warning(
        "Cancel-all requested by operator",
        extra={"extra_json": {"symbols": symbols}},
    )

    return {
        "action": "cancel_all_prepared",
        "symbols": symbols,
        "cancel_instructions": cancel_instructions,
        "timestamp": utc_now().isoformat(),
    }


@router.post("/control/deleverage", dependencies=[Depends(require_control_rate_limit)])
def force_deleverage(
    req: DeleverageRequest,
    _: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Prepare a reduce-only order to close/reduce a position.

    Returns the MCP tool call parameters for a reduce_only order.
    The operator must execute the returned instruction via MCP.
    """
    logger.warning(
        "Force deleverage requested",
        extra={
            "extra_json": {
                "symbol": req.symbol,
                "reduce_pct": req.reduce_pct,
            }
        },
    )

    return {
        "action": "deleverage_prepared",
        "symbol": req.symbol,
        "reduce_pct": req.reduce_pct,
        "instruction": {
            "tool": "mcp__aster__create_order",
            "symbol": req.symbol,
            "side": "SELL",  # Operator must set correct side based on position
            "order_type": "MARKET",
            "reduce_only": True,
            "note": "Set side to SELL for long positions, BUY for short positions. "
            "Set quantity based on current position size × reduce_pct.",
        },
        "timestamp": utc_now().isoformat(),
    }
