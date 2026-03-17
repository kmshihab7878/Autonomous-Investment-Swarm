"""Tests for Redis-backed session state persistence.

Verifies that SessionManager persists session state to Redis on every
state transition and restores it on initialization, with graceful
fallback when Redis is unavailable.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from aiswarm.data.event_store import EventStore
from aiswarm.session.manager import REDIS_SESSION_KEY, SessionManager
from aiswarm.session.models import SessionState


def _make_store() -> EventStore:
    return EventStore(tempfile.mktemp(suffix=".db"))


def _make_mock_redis(store: dict[str, str] | None = None) -> MagicMock:
    """Create a mock Redis client backed by an in-memory dict.

    This mirrors the interface used by SessionManager: .get() and .set().
    """
    _store: dict[str, str] = store if store is not None else {}
    mock = MagicMock()

    def _get(key: str) -> str | None:
        return _store.get(key)

    def _set(key: str, value: str) -> None:
        _store[key] = value

    mock.get = MagicMock(side_effect=_get)
    mock.set = MagicMock(side_effect=_set)
    mock._store = _store  # exposed for test assertions
    return mock


class TestSessionPersistToRedis:
    """Session state is saved to Redis on every state transition."""

    def test_start_session_persists(self) -> None:
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()

        redis_mock.set.assert_called()
        raw = redis_mock._store[REDIS_SESSION_KEY]
        data = json.loads(raw)
        assert data["state"] == SessionState.PENDING_REVIEW.value
        assert "session_id" in data

    def test_approve_session_persists(self) -> None:
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()
        mgr.approve_session("operator_1", "looks good")

        raw = redis_mock._store[REDIS_SESSION_KEY]
        data = json.loads(raw)
        assert data["state"] == SessionState.APPROVED.value
        assert data["approved_by"] == "operator_1"
        assert data["approval_notes"] == "looks good"

    def test_activate_session_persists(self) -> None:
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()
        mgr.approve_session("operator_1")
        mgr.activate_session()

        raw = redis_mock._store[REDIS_SESSION_KEY]
        data = json.loads(raw)
        assert data["state"] == SessionState.ACTIVE.value
        assert data["actual_start"] is not None

    def test_end_session_persists(self) -> None:
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()
        mgr.approve_session("operator_1")
        mgr.activate_session()
        mgr.end_session()

        raw = redis_mock._store[REDIS_SESSION_KEY]
        data = json.loads(raw)
        assert data["state"] == SessionState.ENDED.value
        assert data["actual_end"] is not None

    def test_full_lifecycle_persistence_count(self) -> None:
        """Redis.set is called once for each state mutation."""
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()  # 1 set
        mgr.approve_session("op")  # 1 set (via _transition)
        mgr.activate_session()  # 1 set
        mgr.end_session()  # 1 set

        assert redis_mock.set.call_count == 4

    def test_persisted_data_contains_all_fields(self) -> None:
        redis_mock = _make_mock_redis()
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        mgr.start_session()
        mgr.approve_session("operator_1", "notes here")
        mgr.activate_session()

        raw = redis_mock._store[REDIS_SESSION_KEY]
        data = json.loads(raw)

        expected_keys = {
            "session_id",
            "state",
            "scheduled_start",
            "scheduled_end",
            "actual_start",
            "actual_end",
            "approved_by",
            "approval_notes",
            "created_at",
            "total_fills",
            "total_pnl",
        }
        assert set(data.keys()) == expected_keys


class TestSessionRestoreFromRedis:
    """Session state is restored from Redis on initialization."""

    def test_restore_active_session(self) -> None:
        """A new SessionManager restores an ACTIVE session from Redis."""
        now = datetime.now(timezone.utc)
        session_data = {
            "session_id": "session_test_123",
            "state": "active",
            "scheduled_start": now.isoformat(),
            "scheduled_end": (now + timedelta(hours=8)).isoformat(),
            "actual_start": now.isoformat(),
            "actual_end": None,
            "approved_by": "operator_1",
            "approval_notes": "approved for live",
            "created_at": now.isoformat(),
            "total_fills": 5,
            "total_pnl": 123.45,
        }
        redis_store: dict[str, str] = {REDIS_SESSION_KEY: json.dumps(session_data)}
        redis_mock = _make_mock_redis(store=redis_store)

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is not None
        assert mgr.current_session.session_id == "session_test_123"
        assert mgr.current_session.state == SessionState.ACTIVE
        assert mgr.is_trading_allowed
        assert mgr.current_session.approved_by == "operator_1"
        assert mgr.current_session.total_fills == 5
        assert mgr.current_session.total_pnl == 123.45

    def test_restore_approved_session(self) -> None:
        """Restoring an APPROVED session means trading is NOT yet allowed."""
        now = datetime.now(timezone.utc)
        session_data = {
            "session_id": "session_approved_1",
            "state": "approved",
            "scheduled_start": now.isoformat(),
            "scheduled_end": (now + timedelta(hours=8)).isoformat(),
            "actual_start": None,
            "actual_end": None,
            "approved_by": "ops_lead",
            "approval_notes": "",
            "created_at": now.isoformat(),
            "total_fills": 0,
            "total_pnl": 0.0,
        }
        redis_store: dict[str, str] = {REDIS_SESSION_KEY: json.dumps(session_data)}
        redis_mock = _make_mock_redis(store=redis_store)

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is not None
        assert mgr.current_session.state == SessionState.APPROVED
        assert not mgr.is_trading_allowed

    def test_restore_ended_session(self) -> None:
        """Restoring an ENDED session means trading is NOT allowed."""
        now = datetime.now(timezone.utc)
        session_data = {
            "session_id": "session_ended_1",
            "state": "ended",
            "scheduled_start": now.isoformat(),
            "scheduled_end": (now + timedelta(hours=8)).isoformat(),
            "actual_start": now.isoformat(),
            "actual_end": (now + timedelta(hours=7)).isoformat(),
            "approved_by": "ops_lead",
            "approval_notes": "",
            "created_at": now.isoformat(),
            "total_fills": 10,
            "total_pnl": -50.0,
        }
        redis_store: dict[str, str] = {REDIS_SESSION_KEY: json.dumps(session_data)}
        redis_mock = _make_mock_redis(store=redis_store)

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is not None
        assert mgr.current_session.state == SessionState.ENDED
        assert not mgr.is_trading_allowed

    def test_no_redis_data_starts_fresh(self) -> None:
        """When Redis has no session key, manager starts with no session."""
        redis_mock = _make_mock_redis()  # empty store
        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is None
        assert not mgr.is_trading_allowed

    def test_roundtrip_persist_then_restore(self) -> None:
        """Persist in one manager, restore in another — full roundtrip."""
        redis_mock = _make_mock_redis()
        store = _make_store()

        # First manager: create and activate a session
        mgr1 = SessionManager(store, redis_client=redis_mock)
        mgr1.start_session()
        mgr1.approve_session("operator_1", "roundtrip test")
        mgr1.activate_session()
        original_id = mgr1.current_session.session_id  # type: ignore[union-attr]

        # Second manager: simulate process restart
        mgr2 = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr2.current_session is not None
        assert mgr2.current_session.session_id == original_id
        assert mgr2.current_session.state == SessionState.ACTIVE
        assert mgr2.is_trading_allowed
        assert mgr2.current_session.approved_by == "operator_1"
        assert mgr2.current_session.approval_notes == "roundtrip test"


class TestSessionRedisFallback:
    """Graceful degradation when Redis is unavailable."""

    def test_no_redis_client_works_in_memory(self) -> None:
        """SessionManager works fine without Redis (redis_client=None)."""
        mgr = SessionManager(_make_store(), redis_client=None)

        mgr.start_session()
        mgr.approve_session("op")
        mgr.activate_session()
        assert mgr.is_trading_allowed

        mgr.end_session()
        assert not mgr.is_trading_allowed

    def test_redis_get_failure_on_restore(self) -> None:
        """If Redis raises on get during restore, manager starts fresh."""
        redis_mock = MagicMock()
        redis_mock.get.side_effect = ConnectionError("Redis down")

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is None
        assert not mgr.is_trading_allowed
        # Manager should still be fully functional in-memory
        mgr.start_session()
        assert mgr.current_session is not None

    def test_redis_set_failure_on_persist(self) -> None:
        """If Redis raises on set during persist, manager continues in-memory."""
        redis_mock = MagicMock()
        redis_mock.get.return_value = None  # no restore needed
        redis_mock.set.side_effect = ConnectionError("Redis down")

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        # Operations succeed despite Redis write failures
        mgr.start_session()
        assert mgr.current_session is not None
        assert mgr.current_session.state == SessionState.PENDING_REVIEW

        mgr.approve_session("op")
        assert mgr.current_session.state == SessionState.APPROVED

        mgr.activate_session()
        assert mgr.is_trading_allowed

    def test_corrupt_redis_data_on_restore(self) -> None:
        """If Redis returns invalid JSON, manager starts fresh."""
        redis_store: dict[str, str] = {REDIS_SESSION_KEY: "not-valid-json{{{"}
        redis_mock = _make_mock_redis(store=redis_store)

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is None
        assert not mgr.is_trading_allowed

    def test_incomplete_redis_data_on_restore(self) -> None:
        """If Redis returns JSON missing required fields, manager starts fresh."""
        redis_store: dict[str, str] = {REDIS_SESSION_KEY: json.dumps({"state": "active"})}
        redis_mock = _make_mock_redis(store=redis_store)

        mgr = SessionManager(_make_store(), redis_client=redis_mock)

        assert mgr.current_session is None
        assert not mgr.is_trading_allowed


class TestSessionRedisKeyConstant:
    """The Redis key used for session state is the documented constant."""

    def test_redis_key_value(self) -> None:
        assert REDIS_SESSION_KEY == "ais:session:state"


class TestExistingBehaviorPreserved:
    """Ensure adding Redis does not break existing SessionManager behavior."""

    def test_existing_tests_pass_without_redis(self) -> None:
        """All original lifecycle transitions work without redis_client."""
        store = _make_store()
        mgr = SessionManager(store)

        session = mgr.start_session()
        assert session.state == SessionState.PENDING_REVIEW

        mgr.approve_session("operator_1", "good")
        assert mgr.current_session is not None
        assert mgr.current_session.state == SessionState.APPROVED

        mgr.activate_session()
        assert mgr.is_trading_allowed

        mgr.end_session()
        assert not mgr.is_trading_allowed

    def test_cannot_approve_without_session_still_works(self) -> None:
        mgr = SessionManager(_make_store())
        with pytest.raises(ValueError, match="No current session"):
            mgr.approve_session("operator")

    def test_invalid_transition_still_raises(self) -> None:
        mgr = SessionManager(_make_store())
        mgr.start_session()
        with pytest.raises(ValueError, match="not in.*prerequisite"):
            mgr.activate_session()
