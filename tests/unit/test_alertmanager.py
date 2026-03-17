"""Tests for Alertmanager integration in AlertDispatcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiswarm.monitoring.alerts import (
    AlertChannel,
    AlertDispatcher,
    _format_alertmanager_payload,
)


class TestAlertmanagerPayload:
    def test_basic_payload_structure(self) -> None:
        payload = _format_alertmanager_payload("test message", "warning", {})
        assert isinstance(payload, list)
        assert len(payload) == 1
        alert = payload[0]
        assert "labels" in alert
        assert "annotations" in alert

    def test_labels_contain_required_fields(self) -> None:
        payload = _format_alertmanager_payload("msg", "critical", {})
        labels = payload[0]["labels"]
        assert labels["alertname"] == "ais_alert"
        assert labels["severity"] == "critical"
        assert labels["source"] == "ais"

    def test_severity_normalized_to_lowercase(self) -> None:
        payload = _format_alertmanager_payload("msg", "CRITICAL", {})
        assert payload[0]["labels"]["severity"] == "critical"

    def test_symbol_promoted_to_label(self) -> None:
        payload = _format_alertmanager_payload("msg", "high", {"symbol": "BTCUSDT"})
        assert payload[0]["labels"]["symbol"] == "BTCUSDT"
        # symbol should NOT be in annotations
        assert "symbol" not in payload[0]["annotations"]

    def test_strategy_promoted_to_label(self) -> None:
        payload = _format_alertmanager_payload(
            "msg", "warning", {"strategy": "momentum_ma_crossover"}
        )
        assert payload[0]["labels"]["strategy"] == "momentum_ma_crossover"
        assert "strategy" not in payload[0]["annotations"]

    def test_context_in_annotations(self) -> None:
        payload = _format_alertmanager_payload(
            "test", "warning", {"cycle": 42, "reason": "drawdown breach"}
        )
        annotations = payload[0]["annotations"]
        assert annotations["summary"] == "test"
        assert annotations["cycle"] == "42"
        assert annotations["reason"] == "drawdown breach"
        assert "timestamp" in annotations

    def test_empty_context(self) -> None:
        payload = _format_alertmanager_payload("msg", "info", {})
        annotations = payload[0]["annotations"]
        assert annotations["summary"] == "msg"
        assert "timestamp" in annotations


class TestAlertmanagerDispatch:
    def test_alertmanager_url_gets_api_path(self) -> None:
        channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093",
            format="alertmanager",
            min_severity="low",
        )
        dispatcher = AlertDispatcher(channels=[channel])

        with patch("aiswarm.monitoring.alerts.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            result = dispatcher.send("test alert", severity="warning")
            assert result is True

            # Verify URL has /api/v1/alerts appended
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://alertmanager:9093/api/v1/alerts"

    def test_alertmanager_url_trailing_slash_handled(self) -> None:
        channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093/",
            format="alertmanager",
            min_severity="low",
        )
        dispatcher = AlertDispatcher(channels=[channel])

        with patch("aiswarm.monitoring.alerts.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            dispatcher.send("test", severity="critical")
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://alertmanager:9093/api/v1/alerts"

    def test_alertmanager_payload_is_list(self) -> None:
        channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093",
            format="alertmanager",
            min_severity="low",
        )
        dispatcher = AlertDispatcher(channels=[channel])

        with patch("aiswarm.monitoring.alerts.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            dispatcher.send("test alert", severity="error", context={"symbol": "ETHUSDT"})
            payload = mock_post.call_args[1]["json"]
            assert isinstance(payload, list)
            assert payload[0]["labels"]["symbol"] == "ETHUSDT"

    def test_alertmanager_failure_isolated(self) -> None:
        channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093",
            format="alertmanager",
            min_severity="low",
        )
        dispatcher = AlertDispatcher(channels=[channel])

        with patch(
            "aiswarm.monitoring.alerts.httpx.post",
            side_effect=ConnectionError("refused"),
        ):
            result = dispatcher.send("test", severity="critical")
            assert result is False

    def test_mixed_channels_alertmanager_and_slack(self) -> None:
        am_channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093",
            format="alertmanager",
            min_severity="low",
        )
        slack_channel = AlertChannel(
            name="slack",
            url="http://slack.example.com/hook",
            format="slack",
            min_severity="high",
        )
        dispatcher = AlertDispatcher(channels=[am_channel, slack_channel])

        with patch("aiswarm.monitoring.alerts.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            # Warning-level: only AM (min_severity=low), not Slack (min_severity=high)
            dispatcher.send("test", severity="warning")
            assert mock_post.call_count == 1
            assert "/api/v1/alerts" in mock_post.call_args[0][0]

    def test_mixed_channels_critical_hits_both(self) -> None:
        am_channel = AlertChannel(
            name="am",
            url="http://alertmanager:9093",
            format="alertmanager",
            min_severity="low",
        )
        slack_channel = AlertChannel(
            name="slack",
            url="http://slack.example.com/hook",
            format="slack",
            min_severity="high",
        )
        dispatcher = AlertDispatcher(channels=[am_channel, slack_channel])

        with patch("aiswarm.monitoring.alerts.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            dispatcher.send("critical alert", severity="critical")
            assert mock_post.call_count == 2
