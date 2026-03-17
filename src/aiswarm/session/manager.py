"""Session manager — lifecycle and trading gate.

Controls the trading session lifecycle:
PENDING_REVIEW -> APPROVED -> ACTIVE -> ENDED -> PENDING_REVIEW

Trading is only allowed when a session is ACTIVE.

Session state is persisted to Redis (key ``ais:session:state``) when a
``redis_client`` is provided so that state survives process restarts.
Falls back to in-memory only when Redis is unavailable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from aiswarm.data.event_store import EventStore
from aiswarm.session.models import SessionState, TradingSession
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

REDIS_SESSION_KEY = "ais:session:state"

_VALID_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.PENDING_REVIEW: {SessionState.APPROVED},
    SessionState.APPROVED: {SessionState.ACTIVE},
    SessionState.ACTIVE: {SessionState.ENDED},
    SessionState.ENDED: {SessionState.PENDING_REVIEW},
}


class SessionManager:
    """Manages trading session lifecycle and the is_trading_allowed gate."""

    def __init__(
        self,
        event_store: EventStore,
        default_duration_hours: int = 8,
        schedule: dict[str, Any] | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self.event_store = event_store
        self.default_duration_hours = default_duration_hours
        self.schedule = schedule
        self._redis = redis_client
        self._current: TradingSession | None = None
        self._restore_from_redis()

    @property
    def current_session(self) -> TradingSession | None:
        return self._current

    @property
    def is_trading_allowed(self) -> bool:
        """Trading is only allowed when a session is ACTIVE."""
        if self._current is None:
            return False
        return self._current.state == SessionState.ACTIVE

    # --- Redis persistence --------------------------------------------------

    def _session_to_dict(self, session: TradingSession) -> dict[str, Any]:
        """Serialize a TradingSession to a JSON-safe dictionary."""
        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "scheduled_start": session.scheduled_start.isoformat(),
            "scheduled_end": session.scheduled_end.isoformat(),
            "actual_start": session.actual_start.isoformat() if session.actual_start else None,
            "actual_end": session.actual_end.isoformat() if session.actual_end else None,
            "approved_by": session.approved_by,
            "approval_notes": session.approval_notes,
            "created_at": session.created_at.isoformat(),
            "total_fills": session.total_fills,
            "total_pnl": session.total_pnl,
        }

    def _dict_to_session(self, data: dict[str, Any]) -> TradingSession:
        """Deserialize a dictionary back to a TradingSession."""
        return TradingSession(
            session_id=data["session_id"],
            state=SessionState(data["state"]),
            scheduled_start=datetime.fromisoformat(data["scheduled_start"]),
            scheduled_end=datetime.fromisoformat(data["scheduled_end"]),
            actual_start=(
                datetime.fromisoformat(data["actual_start"]) if data.get("actual_start") else None
            ),
            actual_end=(
                datetime.fromisoformat(data["actual_end"]) if data.get("actual_end") else None
            ),
            approved_by=data.get("approved_by"),
            approval_notes=data.get("approval_notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            total_fills=data.get("total_fills", 0),
            total_pnl=data.get("total_pnl", 0.0),
        )

    def _persist_to_redis(self) -> None:
        """Write current session state to Redis. Fails silently."""
        if self._redis is None or self._current is None:
            return
        try:
            payload = json.dumps(self._session_to_dict(self._current))
            self._redis.set(REDIS_SESSION_KEY, payload)
        except Exception:
            logger.error("Failed to persist session state to Redis")

    def _restore_from_redis(self) -> None:
        """Attempt to restore session state from Redis on startup."""
        if self._redis is None:
            return
        try:
            raw: str | None = self._redis.get(REDIS_SESSION_KEY)
            if raw is None:
                logger.info("No session state found in Redis — starting fresh")
                return
            data = json.loads(raw)
            self._current = self._dict_to_session(data)
            logger.info(
                "Session state restored from Redis",
                extra={
                    "extra_json": {
                        "session_id": self._current.session_id,
                        "state": self._current.state.value,
                    }
                },
            )
        except Exception:
            logger.error("Failed to restore session state from Redis — starting fresh")

    # --- Lifecycle transitions -----------------------------------------------

    def _transition(self, new_state: SessionState) -> TradingSession:
        """Apply a state transition with validation."""
        if self._current is None:
            raise ValueError("No current session")
        valid = _VALID_TRANSITIONS.get(self._current.state, set())
        if new_state not in valid:
            raise ValueError(
                f"Session not in {new_state.value} prerequisite state "
                f"(current: {self._current.state.value})"
            )
        updates: dict[str, object] = {"state": new_state}
        if new_state == SessionState.ACTIVE:
            updates["actual_start"] = utc_now()
        elif new_state == SessionState.ENDED:
            updates["actual_end"] = utc_now()
        self._current = self._current.model_copy(update=updates)
        self.event_store.append(
            "session",
            {
                "action": "state_changed",
                "session_id": self._current.session_id,
                "new_state": new_state.value,
            },
            source="session_manager",
        )
        logger.info(
            "Session state changed",
            extra={
                "extra_json": {
                    "session_id": self._current.session_id,
                    "new_state": new_state.value,
                }
            },
        )
        self._persist_to_redis()
        return self._current

    def start_session(
        self,
        scheduled_start: datetime | None = None,
        scheduled_end: datetime | None = None,
    ) -> TradingSession:
        """Create a new session in PENDING_REVIEW state."""
        now = utc_now()
        start = scheduled_start or now
        end = scheduled_end or (start + timedelta(hours=self.default_duration_hours))
        session = TradingSession(
            session_id=new_id("session"),
            state=SessionState.PENDING_REVIEW,
            scheduled_start=start,
            scheduled_end=end,
            created_at=now,
        )
        self._current = session
        self.event_store.append(
            "session",
            {
                "action": "created",
                "session_id": session.session_id,
                "scheduled_start": start.isoformat(),
                "scheduled_end": end.isoformat(),
            },
            source="session_manager",
        )
        logger.info(
            "Session created",
            extra={"extra_json": {"session_id": session.session_id}},
        )
        self._persist_to_redis()
        return session

    def approve_session(self, operator: str, notes: str = "") -> TradingSession:
        """Approve session — operator sign-off before activation."""
        if self._current is None:
            raise ValueError("No current session to approve")
        self._current = self._current.model_copy(
            update={"approved_by": operator, "approval_notes": notes}
        )
        return self._transition(SessionState.APPROVED)

    def activate_session(self) -> TradingSession:
        """Activate session — trading is now allowed."""
        return self._transition(SessionState.ACTIVE)

    def end_session(self) -> TradingSession:
        """End the current session — trading stops."""
        return self._transition(SessionState.ENDED)

    def check_session_end(self) -> bool:
        """Check if the session should auto-end based on schedule.

        Returns True if session was ended, False otherwise.
        """
        if self._current is None:
            return False
        if self._current.state != SessionState.ACTIVE:
            return False
        now = datetime.now(timezone.utc)
        if now >= self._current.scheduled_end:
            self.end_session()
            logger.info(
                "Session auto-ended on schedule",
                extra={"extra_json": {"session_id": self._current.session_id}},
            )
            return True
        return False
