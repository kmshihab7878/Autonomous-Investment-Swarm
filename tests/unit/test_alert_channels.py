"""Tests for multi-channel alert dispatching.

Covers:
  - Dispatch to multiple channels
  - Per-channel severity filtering
  - Slack payload formatting
  - Backward compatibility with single webhook_url
  - Isolated channel failures (one channel failing does not block others)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from aiswarm.monitoring.alerts import (
    AlertChannel,
    AlertDispatcher,
    AlertSeverity,
    _format_generic_payload,
    _format_slack_payload,
    _resolve_severity,
)


# ---------------------------------------------------------------------------
# Severity resolution
# ---------------------------------------------------------------------------


class TestResolveSeverity:
    def test_standard_labels(self) -> None:
        assert _resolve_severity("info") == AlertSeverity.INFO
        assert _resolve_severity("warning") == AlertSeverity.WARNING
        assert _resolve_severity("error") == AlertSeverity.ERROR
        assert _resolve_severity("critical") == AlertSeverity.CRITICAL

    def test_alias_labels(self) -> None:
        """low/medium/high map to the corresponding AlertSeverity values."""
        assert _resolve_severity("low") == AlertSeverity.INFO
        assert _resolve_severity("medium") == AlertSeverity.WARNING
        assert _resolve_severity("high") == AlertSeverity.ERROR

    def test_case_insensitive(self) -> None:
        assert _resolve_severity("CRITICAL") == AlertSeverity.CRITICAL
        assert _resolve_severity("High") == AlertSeverity.ERROR

    def test_unknown_defaults_to_warning(self) -> None:
        assert _resolve_severity("banana") == AlertSeverity.WARNING


# ---------------------------------------------------------------------------
# Slack payload formatting
# ---------------------------------------------------------------------------


class TestSlackPayloadFormatting:
    def test_slack_payload_has_text_and_blocks(self) -> None:
        payload = _format_slack_payload("Server on fire", "critical", {})
        assert "text" in payload
        assert "blocks" in payload
        assert isinstance(payload["blocks"], list)

    def test_slack_text_contains_severity_and_message(self) -> None:
        payload = _format_slack_payload("Drawdown exceeded", "error", {})
        assert "CRITICAL" not in payload["text"]
        assert "ERROR" in payload["text"]
        assert "Drawdown exceeded" in payload["text"]

    def test_slack_header_block(self) -> None:
        payload = _format_slack_payload("Test alert", "warning", {})
        header = payload["blocks"][0]
        assert header["type"] == "header"
        assert "WARNING" in header["text"]["text"]

    def test_slack_context_rendered(self) -> None:
        ctx = {"cycle": 42, "reason": "kill switch"}
        payload = _format_slack_payload("Halted", "critical", ctx)
        # Context should appear in a section block
        blocks_text = str(payload["blocks"])
        assert "cycle" in blocks_text
        assert "kill switch" in blocks_text

    def test_slack_no_context_block_when_empty(self) -> None:
        payload = _format_slack_payload("Simple alert", "info", {})
        # With no context, we expect: header, section (message), context (timestamp)
        # but NOT an extra section for empty context dict
        assert len(payload["blocks"]) == 3

    def test_slack_timestamp_in_context_element(self) -> None:
        payload = _format_slack_payload("Alert", "info", {})
        last_block = payload["blocks"][-1]
        assert last_block["type"] == "context"
        assert "Timestamp:" in last_block["elements"][0]["text"]


# ---------------------------------------------------------------------------
# Generic payload formatting
# ---------------------------------------------------------------------------


class TestGenericPayloadFormatting:
    def test_generic_payload_structure(self) -> None:
        payload = _format_generic_payload("test msg", "error", {"k": "v"})
        assert payload["severity"] == "error"
        assert payload["message"] == "test msg"
        assert "timestamp" in payload
        assert payload["context"] == {"k": "v"}

    def test_generic_empty_context(self) -> None:
        payload = _format_generic_payload("msg", "info", {})
        assert payload["context"] == {}


# ---------------------------------------------------------------------------
# AlertChannel dataclass
# ---------------------------------------------------------------------------


class TestAlertChannel:
    def test_defaults(self) -> None:
        ch = AlertChannel(name="test", url="http://example.com/hook")
        assert ch.format == "generic"
        assert ch.min_severity == "low"

    def test_frozen(self) -> None:
        ch = AlertChannel(name="test", url="http://example.com/hook")
        try:
            ch.name = "mutated"  # type: ignore[misc]
            assert False, "Should not allow mutation on frozen dataclass"
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Multi-channel dispatch
# ---------------------------------------------------------------------------


class TestMultiChannelDispatch:
    def _make_dispatcher(self, channels: list[AlertChannel]) -> AlertDispatcher:
        return AlertDispatcher(channels=channels, enabled=True)

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_dispatch_to_multiple_channels(self, mock_post: MagicMock) -> None:
        """Alert dispatched to all channels that meet the severity threshold."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        channels = [
            AlertChannel(name="ch1", url="http://a.com/hook", min_severity="low"),
            AlertChannel(name="ch2", url="http://b.com/hook", min_severity="low"),
        ]
        dispatcher = self._make_dispatcher(channels)

        result = dispatcher.send("test", severity="error")

        assert result is True
        assert mock_post.call_count == 2
        urls_called = {call.args[0] for call in mock_post.call_args_list}
        assert urls_called == {"http://a.com/hook", "http://b.com/hook"}

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_severity_filtering_per_channel(self, mock_post: MagicMock) -> None:
        """A low-severity alert should NOT reach a high-severity channel."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        channels = [
            AlertChannel(name="all-alerts", url="http://a.com/hook", min_severity="low"),
            AlertChannel(name="critical-only", url="http://b.com/hook", min_severity="critical"),
        ]
        dispatcher = self._make_dispatcher(channels)

        result = dispatcher.send("routine check", severity="info")

        assert result is True
        assert mock_post.call_count == 1
        assert mock_post.call_args.args[0] == "http://a.com/hook"

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_all_channels_filtered_returns_true(self, mock_post: MagicMock) -> None:
        """If alert is below ALL channel thresholds, send returns True (not an error)."""
        channels = [
            AlertChannel(name="high-only", url="http://a.com/hook", min_severity="high"),
            AlertChannel(name="critical-only", url="http://b.com/hook", min_severity="critical"),
        ]
        dispatcher = self._make_dispatcher(channels)

        result = dispatcher.send("low noise", severity="info")

        assert result is True
        mock_post.assert_not_called()

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_slack_format_channel(self, mock_post: MagicMock) -> None:
        """Slack-format channels receive Slack Block Kit payloads."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        channels = [
            AlertChannel(
                name="slack", url="http://slack.com/hook", format="slack", min_severity="low"
            ),
        ]
        dispatcher = self._make_dispatcher(channels)

        result = dispatcher.send("kill switch triggered", severity="critical", context={"a": 1})

        assert result is True
        posted_payload = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json"))
        assert "text" in posted_payload
        assert "blocks" in posted_payload
        assert "CRITICAL" in posted_payload["text"]

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_generic_format_channel(self, mock_post: MagicMock) -> None:
        """Generic-format channels receive the standard JSON payload."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        channels = [
            AlertChannel(
                name="generic", url="http://pager.com/hook", format="generic", min_severity="low"
            ),
        ]
        dispatcher = self._make_dispatcher(channels)

        result = dispatcher.send("test", severity="warning", context={"k": "v"})

        assert result is True
        posted_payload = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json"))
        assert posted_payload["severity"] == "warning"
        assert posted_payload["message"] == "test"
        assert posted_payload["context"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Isolated channel failures
# ---------------------------------------------------------------------------


class TestIsolatedChannelFailures:
    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_one_channel_failure_does_not_block_others(self, mock_post: MagicMock) -> None:
        """If channel A fails, channel B should still receive the alert."""
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.raise_for_status = MagicMock()

        def _side_effect(url: str, **kwargs: Any) -> MagicMock:
            if url == "http://failing.com/hook":
                raise httpx.ConnectError("Connection refused")
            return success_resp

        mock_post.side_effect = _side_effect

        channels = [
            AlertChannel(name="failing", url="http://failing.com/hook", min_severity="low"),
            AlertChannel(name="ok", url="http://ok.com/hook", min_severity="low"),
        ]
        dispatcher = AlertDispatcher(channels=channels, enabled=True)

        result = dispatcher.send("alert", severity="error")

        # Overall result is False because one channel failed
        assert result is False
        # But both channels were attempted
        assert mock_post.call_count == 2

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_all_channels_fail_returns_false(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        channels = [
            AlertChannel(name="ch1", url="http://a.com/hook", min_severity="low"),
            AlertChannel(name="ch2", url="http://b.com/hook", min_severity="low"),
        ]
        dispatcher = AlertDispatcher(channels=channels, enabled=True)

        result = dispatcher.send("alert", severity="critical")

        assert result is False
        assert mock_post.call_count == 2

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_http_error_status_is_failure(self, mock_post: MagicMock) -> None:
        """A 500 response should be treated as a failure for that channel."""
        bad_resp = MagicMock()
        bad_resp.status_code = 500
        bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=bad_resp
        )
        mock_post.return_value = bad_resp

        channels = [
            AlertChannel(name="broken", url="http://broken.com/hook", min_severity="low"),
        ]
        dispatcher = AlertDispatcher(channels=channels, enabled=True)

        result = dispatcher.send("alert", severity="error")

        assert result is False


# ---------------------------------------------------------------------------
# Backward compatibility — single webhook_url
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_single_webhook_url_wraps_as_channel(self) -> None:
        """Old-style single webhook_url should be wrapped as a generic channel."""
        dispatcher = AlertDispatcher(
            webhook_url="http://legacy.com/hook",
            severity_filter="warning",
            enabled=True,
        )
        assert len(dispatcher.channels) == 1
        ch = dispatcher.channels[0]
        assert ch.name == "default"
        assert ch.url == "http://legacy.com/hook"
        assert ch.format == "generic"
        assert ch.min_severity == "warning"

    def test_disabled_with_no_url_and_no_channels(self) -> None:
        dispatcher = AlertDispatcher(webhook_url="", enabled=True)
        assert dispatcher.enabled is False
        assert len(dispatcher.channels) == 0

    def test_send_returns_true_when_disabled(self) -> None:
        dispatcher = AlertDispatcher(webhook_url="", enabled=False)
        result = dispatcher.send("test", severity="critical")
        assert result is True

    @patch("aiswarm.monitoring.alerts.httpx.post")
    def test_single_url_dispatch(self, mock_post: MagicMock) -> None:
        """Old-style single webhook_url dispatches via the generic format."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        dispatcher = AlertDispatcher(
            webhook_url="http://legacy.com/hook",
            severity_filter="warning",
            enabled=True,
        )

        result = dispatcher.send("test alert", severity="error", context={"x": 1})

        assert result is True
        mock_post.assert_called_once()
        posted_payload = mock_post.call_args.kwargs.get("json", mock_post.call_args[1].get("json"))
        assert posted_payload["severity"] == "error"
        assert posted_payload["message"] == "test alert"

    def test_channels_take_precedence_over_webhook_url(self) -> None:
        """When both channels and webhook_url are provided, channels win."""
        channels = [
            AlertChannel(name="explicit", url="http://explicit.com/hook"),
        ]
        dispatcher = AlertDispatcher(
            webhook_url="http://legacy.com/hook",
            channels=channels,
            enabled=True,
        )
        assert len(dispatcher.channels) == 1
        assert dispatcher.channels[0].name == "explicit"

    def test_legacy_min_severity_attribute(self) -> None:
        """The .min_severity attribute is preserved for any callers using it."""
        dispatcher = AlertDispatcher(severity_filter="error")
        assert dispatcher.min_severity == AlertSeverity.ERROR

    def test_legacy_webhook_url_attribute(self) -> None:
        """The .webhook_url attribute is preserved for any callers using it."""
        dispatcher = AlertDispatcher(webhook_url="http://legacy.com/hook")
        assert dispatcher.webhook_url == "http://legacy.com/hook"


# ---------------------------------------------------------------------------
# Bootstrap helper: _build_alert_channels
# ---------------------------------------------------------------------------


class TestBuildAlertChannels:
    def test_build_channels_from_config(self) -> None:
        from aiswarm.bootstrap import _build_alert_channels

        cfg: dict[str, Any] = {
            "alert_channels": [
                {
                    "name": "slack-ops",
                    "url": "http://slack.example.com/hook",
                    "format": "slack",
                    "min_severity": "high",
                },
                {
                    "name": "pager",
                    "url": "http://pager.example.com/hook",
                    "format": "generic",
                    "min_severity": "critical",
                },
            ]
        }
        channels = _build_alert_channels(cfg)
        assert len(channels) == 2
        assert channels[0].name == "slack-ops"
        assert channels[0].format == "slack"
        assert channels[1].min_severity == "critical"

    def test_env_var_resolution(self) -> None:
        import os

        from aiswarm.bootstrap import _build_alert_channels

        os.environ["_TEST_WEBHOOK"] = "http://resolved.example.com/hook"
        try:
            cfg: dict[str, Any] = {
                "alert_channels": [
                    {
                        "name": "env-channel",
                        "url": "${_TEST_WEBHOOK}",
                        "format": "generic",
                        "min_severity": "low",
                    }
                ]
            }
            channels = _build_alert_channels(cfg)
            assert len(channels) == 1
            assert channels[0].url == "http://resolved.example.com/hook"
        finally:
            os.environ.pop("_TEST_WEBHOOK", None)

    def test_unresolved_env_var_skipped(self) -> None:
        import os

        from aiswarm.bootstrap import _build_alert_channels

        os.environ.pop("_NONEXISTENT_WEBHOOK_VAR", None)
        cfg: dict[str, Any] = {
            "alert_channels": [
                {
                    "name": "missing-env",
                    "url": "${_NONEXISTENT_WEBHOOK_VAR}",
                }
            ]
        }
        channels = _build_alert_channels(cfg)
        assert len(channels) == 0

    def test_empty_url_skipped(self) -> None:
        from aiswarm.bootstrap import _build_alert_channels

        cfg: dict[str, Any] = {
            "alert_channels": [
                {"name": "empty", "url": ""},
            ]
        }
        channels = _build_alert_channels(cfg)
        assert len(channels) == 0

    def test_no_channels_section(self) -> None:
        from aiswarm.bootstrap import _build_alert_channels

        channels = _build_alert_channels({})
        assert channels == []
