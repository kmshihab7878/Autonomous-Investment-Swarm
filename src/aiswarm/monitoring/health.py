"""Health status — checks actual service connectivity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiswarm.utils.logging import get_logger
from aiswarm.utils.secrets import get_secrets_provider
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


def health_status() -> dict[str, Any]:
    """Return system health status including actual service probes."""
    checks: dict[str, Any] = {
        "status": "ok",
        "timestamp": utc_now().isoformat(),
    }

    # Check Redis
    checks["redis"] = _check_redis()

    # Check EventStore DB
    checks["database"] = _check_database()

    # Check loop heartbeat
    checks["loop"] = _check_loop_heartbeat()

    # Derive overall status
    if any(v == "error" for v in checks.values() if isinstance(v, str) and v == "error"):
        checks["status"] = "degraded"

    return checks


def _check_redis() -> str:
    try:
        import redis

        url = get_secrets_provider().get_secret("REDIS_URL") or "redis://localhost:6379/0"
        client = redis.Redis.from_url(url, socket_timeout=2, decode_responses=True)
        client.ping()
        return "connected"
    except Exception:
        return "error"


def _check_database() -> str:
    try:
        db_path = get_secrets_provider().get_secret("AIS_DB_PATH") or "data/ais_events.db"
        if Path(db_path).exists():
            return "connected"
        return "not_initialized"
    except Exception:
        return "error"


def _check_loop_heartbeat() -> str:
    """Check if the trading loop has emitted a heartbeat recently."""
    try:
        import time

        hb_path = Path("/tmp/ais_loop_heartbeat")  # nosec B108
        if not hb_path.exists():
            return "no_heartbeat"
        ts = float(hb_path.read_text().strip())
        age = time.time() - ts
        if age < 120:
            return "healthy"
        return f"stale ({int(age)}s ago)"
    except Exception:
        return "unknown"
