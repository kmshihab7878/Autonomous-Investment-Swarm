"""Tests for kill switch auto-execution of emergency cancels.

Validates that the kill switch is self-enforcing when an executor is
injected, backward compatible when no executor is set, and resilient
to errors in the cancel path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aiswarm.execution.live_executor import LiveOrderExecutor, SubmissionResult
from aiswarm.risk.kill_switch import KillSwitch


def _make_mock_executor(
    cancel_results: list[SubmissionResult] | None = None,
) -> MagicMock:
    """Build a mock LiveOrderExecutor with configurable cancel_all results."""
    mock = MagicMock(spec=LiveOrderExecutor)
    if cancel_results is None:
        cancel_results = [
            SubmissionResult(
                success=True,
                order_id="cancel_all_BTCUSDT_futures",
                exchange_order_id=None,
                message="Cancel all futures for BTCUSDT",
            ),
        ]
    mock.cancel_all.return_value = cancel_results
    return mock


class TestKillSwitchAutoExecute:
    """Kill switch auto-cancels orders when an executor is configured."""

    def test_trigger_auto_cancels_orders(self) -> None:
        """When triggered with an executor set, cancel_all is called."""
        ks = KillSwitch(0.02)
        mock_executor = _make_mock_executor()

        ks.set_executor(mock_executor)
        result = ks.triggered(-0.03)

        assert result is True
        assert ks.is_triggered
        mock_executor.cancel_all.assert_called_once()

    def test_trigger_auto_cancel_called_only_on_first_trigger(self) -> None:
        """cancel_all fires only on the first trigger, not on subsequent calls."""
        ks = KillSwitch(0.02)
        mock_executor = _make_mock_executor()

        ks.set_executor(mock_executor)
        ks.triggered(-0.03)
        ks.triggered(-0.05)
        ks.triggered(-0.10)

        assert mock_executor.cancel_all.call_count == 1

    def test_trigger_auto_cancel_fires_again_after_reset(self) -> None:
        """After reset, the next trigger fires cancel_all again."""
        ks = KillSwitch(0.02)
        mock_executor = _make_mock_executor()

        ks.set_executor(mock_executor)
        ks.triggered(-0.03)
        assert mock_executor.cancel_all.call_count == 1

        ks.reset()
        ks.triggered(-0.04)
        assert mock_executor.cancel_all.call_count == 2


class TestKillSwitchBackwardCompatible:
    """Kill switch works without an executor (advisory-only mode)."""

    def test_trigger_without_executor_just_sets_flag(self) -> None:
        """Without an executor, triggered() sets flag but does not crash."""
        ks = KillSwitch(0.02)
        result = ks.triggered(-0.03)

        assert result is True
        assert ks.is_triggered

    def test_not_triggered_stays_false(self) -> None:
        """PnL above threshold does not trigger."""
        ks = KillSwitch(0.02)
        assert not ks.triggered(-0.01)
        assert not ks.is_triggered

    def test_execute_emergency_cancels_without_executor_returns_empty(self) -> None:
        """Calling execute_emergency_cancels with no executor returns empty list."""
        ks = KillSwitch(0.02)
        result = ks.execute_emergency_cancels()

        assert result == []

    def test_existing_api_unchanged(self) -> None:
        """prepare_emergency_cancels still works as before."""
        ks = KillSwitch(0.02)
        cancels = ks.prepare_emergency_cancels("acc123", ["BTCUSDT"])
        assert len(cancels) == 2
        assert cancels[0]["tool"] == "mcp__aster__cancel_all_orders"
        assert cancels[1]["tool"] == "mcp__aster__cancel_spot_all_orders"


class TestKillSwitchErrorResilience:
    """Errors in cancel execution must not crash the kill switch."""

    def test_cancel_all_exception_does_not_propagate(self) -> None:
        """If cancel_all raises, triggered() still returns True without raising."""
        ks = KillSwitch(0.02)
        mock_executor = MagicMock(spec=LiveOrderExecutor)
        mock_executor.cancel_all.side_effect = ConnectionError("MCP server down")

        ks.set_executor(mock_executor)
        result = ks.triggered(-0.03)

        assert result is True
        assert ks.is_triggered
        mock_executor.cancel_all.assert_called_once()

    def test_cancel_all_exception_returns_error_summary(self) -> None:
        """execute_emergency_cancels returns an error dict on exception."""
        ks = KillSwitch(0.02)
        mock_executor = MagicMock(spec=LiveOrderExecutor)
        mock_executor.cancel_all.side_effect = RuntimeError("gateway timeout")

        ks.set_executor(mock_executor)
        summary = ks.execute_emergency_cancels()

        assert len(summary) == 1
        assert summary[0]["success"] is False
        assert "gateway timeout" in summary[0]["error"]

    def test_execute_with_explicit_executor_override(self) -> None:
        """An explicit executor parameter overrides the injected one."""
        ks = KillSwitch(0.02)
        injected = _make_mock_executor()
        override = _make_mock_executor()

        ks.set_executor(injected)
        ks.execute_emergency_cancels(executor=override)

        injected.cancel_all.assert_not_called()
        override.cancel_all.assert_called_once()


class TestKillSwitchRedisNotification:
    """Kill switch publishes to Redis on trigger (when available)."""

    def test_redis_notification_on_trigger(self) -> None:
        """When Redis client is set, trigger publishes timestamp to key."""
        ks = KillSwitch(0.02)
        mock_redis = MagicMock()
        mock_executor = _make_mock_executor()

        ks.set_executor(mock_executor)
        ks.set_redis_client(mock_redis)
        ks.triggered(-0.03)

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "ais:kill_switch:triggered"
        # Value should be a stringified timestamp
        ts_value = call_args[0][1]
        float(ts_value)  # should not raise

    def test_no_redis_no_notification(self) -> None:
        """Without Redis client, trigger works but no notification is sent."""
        ks = KillSwitch(0.02)
        ks.triggered(-0.03)
        # No assertion needed beyond no exception

    def test_redis_error_does_not_crash(self) -> None:
        """Redis failure during notification is logged but not raised."""
        ks = KillSwitch(0.02)
        mock_redis = MagicMock()
        mock_redis.set.side_effect = ConnectionError("Redis down")
        mock_executor = _make_mock_executor()

        ks.set_executor(mock_executor)
        ks.set_redis_client(mock_redis)
        result = ks.triggered(-0.03)

        assert result is True
        assert ks.is_triggered
        mock_redis.set.assert_called_once()


class TestKillSwitchExecuteEmergencyCancelsResults:
    """Validate the summary returned by execute_emergency_cancels."""

    def test_returns_summary_of_results(self) -> None:
        """Successful cancels return structured summaries."""
        cancel_results = [
            SubmissionResult(
                success=True,
                order_id="cancel_all_BTCUSDT_futures",
                exchange_order_id=None,
                message="Cancel all futures for BTCUSDT",
            ),
            SubmissionResult(
                success=False,
                order_id="cancel_all_ETHUSDT_spot",
                exchange_order_id=None,
                message="Exchange rejected",
            ),
        ]
        ks = KillSwitch(0.02)
        mock_executor = _make_mock_executor(cancel_results)
        ks.set_executor(mock_executor)

        summary = ks.execute_emergency_cancels()

        assert len(summary) == 2
        assert summary[0]["success"] is True
        assert summary[0]["order_id"] == "cancel_all_BTCUSDT_futures"
        assert summary[1]["success"] is False
        assert summary[1]["order_id"] == "cancel_all_ETHUSDT_spot"

    def test_set_executor_last_writer_wins(self) -> None:
        """Calling set_executor multiple times uses the last one."""
        ks = KillSwitch(0.02)
        first = _make_mock_executor()
        second = _make_mock_executor()

        ks.set_executor(first)
        ks.set_executor(second)
        ks.triggered(-0.03)

        first.cancel_all.assert_not_called()
        second.cancel_all.assert_called_once()
