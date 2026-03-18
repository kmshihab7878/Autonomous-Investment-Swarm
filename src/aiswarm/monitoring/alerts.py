"""Alert dispatcher — sends webhook notifications for critical events.

G-003: Replaces the 2-line stub with a real AlertDispatcher that posts
JSON payloads to a configurable webhook URL.

Multi-channel support: dispatches to multiple notification channels with
per-channel severity filtering and format customisation (generic, Slack).

Severity filter: only dispatches alerts at or above the configured level.
Graceful failure: network errors are logged but never crash the loop.
Channel failures are isolated — one channel failing does not block others.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import httpx

from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class AlertSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3


SEVERITY_MAP: dict[str, AlertSeverity] = {
    "info": AlertSeverity.INFO,
    "low": AlertSeverity.INFO,
    "warning": AlertSeverity.WARNING,
    "medium": AlertSeverity.WARNING,
    "error": AlertSeverity.ERROR,
    "high": AlertSeverity.ERROR,
    "critical": AlertSeverity.CRITICAL,
}

# Emoji indicators for Slack severity rendering
_SLACK_SEVERITY_EMOJI: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: ":information_source:",
    AlertSeverity.WARNING: ":warning:",
    AlertSeverity.ERROR: ":rotating_light:",
    AlertSeverity.CRITICAL: ":fire:",
}


@dataclass(frozen=True)
class AlertChannel:
    """A single notification channel with its own URL, format, and severity gate."""

    name: str
    url: str
    format: str = "generic"  # "generic" or "slack"
    min_severity: str = "low"


def _resolve_severity(label: str) -> AlertSeverity:
    """Resolve a severity label string to an AlertSeverity enum value."""
    return SEVERITY_MAP.get(label.lower(), AlertSeverity.WARNING)


def _format_generic_payload(
    message: str,
    severity: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a generic JSON webhook payload."""
    return {
        "severity": severity,
        "message": message,
        "timestamp": utc_now().isoformat(),
        "context": context,
    }


def _format_alertmanager_payload(
    message: str,
    severity: str,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build an Alertmanager-compatible alert payload.

    Alertmanager expects a JSON array of alert objects at ``POST /api/v1/alerts``.
    Each alert has ``labels`` (used for routing/dedup) and ``annotations``
    (human-readable metadata).
    """
    labels: dict[str, str] = {
        "alertname": "ais_alert",
        "severity": severity.lower(),
        "source": "ais",
    }
    # Promote symbol/strategy to labels for routing rules
    if "symbol" in context:
        labels["symbol"] = str(context["symbol"])
    if "strategy" in context:
        labels["strategy"] = str(context["strategy"])

    annotations: dict[str, str] = {
        "summary": message,
        "timestamp": utc_now().isoformat(),
    }
    for k, v in context.items():
        if k not in ("symbol", "strategy"):
            annotations[k] = str(v)

    return [{"labels": labels, "annotations": annotations}]


def _format_slack_payload(
    message: str,
    severity: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build a Slack-compatible webhook payload with blocks.

    Slack incoming webhooks expect a JSON body with a top-level ``text`` field
    (used as fallback) and an optional ``blocks`` array for rich formatting.
    """
    sev = _resolve_severity(severity)
    emoji = _SLACK_SEVERITY_EMOJI.get(sev, ":bell:")
    ts = utc_now().isoformat()

    text = f"{emoji} *[{severity.upper()}]* {message}"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"AIS Alert — {severity.upper()}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} {message}",
            },
        },
    ]

    if context:
        context_lines = "\n".join(f"• *{k}*: {v}" for k, v in context.items())
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": context_lines,
                },
            }
        )

    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Timestamp: {ts}",
                }
            ],
        }
    )

    return {"text": text, "blocks": blocks}


class AlertDispatcher:
    """Dispatches alerts to one or more notification channels.

    Supports two channel formats:
      - ``generic``: plain JSON payload with severity, message, timestamp, context
      - ``slack``: Slack Block Kit payload with rich formatting

    Backward-compatible: if constructed with a single ``webhook_url`` (old style),
    it is automatically wrapped as a generic channel.
    """

    def __init__(
        self,
        webhook_url: str = "",
        severity_filter: str = "warning",
        enabled: bool = True,
        timeout: float = 5.0,
        channels: list[AlertChannel] | None = None,
    ) -> None:
        self.timeout = timeout
        self._channels: list[AlertChannel] = []

        if channels:
            self._channels = list(channels)

        # Backward compatibility: single webhook_url wraps into a generic channel
        if webhook_url and not channels:
            self._channels = [
                AlertChannel(
                    name="default",
                    url=webhook_url,
                    format="generic",
                    min_severity=severity_filter,
                )
            ]

        # Preserve the old .enabled semantics for callers that check it
        self.enabled = enabled and len(self._channels) > 0

        # Legacy attributes for backward compatibility
        self.webhook_url = webhook_url
        self.min_severity = _resolve_severity(severity_filter)

    @property
    def channels(self) -> list[AlertChannel]:
        """Return a copy of the registered channels."""
        return list(self._channels)

    def send(
        self,
        message: str,
        severity: str = "warning",
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Send an alert to all channels whose severity threshold is met.

        Returns True if all targeted channels succeeded (or the alert was
        filtered / disabled), False if any channel encountered an error.
        Never raises — failures are logged and isolated per channel.
        """
        ctx = context or {}
        sev = _resolve_severity(severity)

        if not self.enabled:
            payload = _format_generic_payload(message, severity, ctx)
            logger.info(
                "Alert (not dispatched — disabled)",
                extra={"extra_json": payload},
            )
            return True

        all_ok = True
        dispatched_any = False

        for channel in self._channels:
            channel_min = _resolve_severity(channel.min_severity)
            if sev < channel_min:
                continue

            dispatched_any = True
            ok = self._dispatch_to_channel(channel, message, severity, ctx)
            if not ok:
                all_ok = False

        if not dispatched_any:
            # Alert was below all channel thresholds — that is not an error
            return True

        return all_ok

    def _dispatch_to_channel(
        self,
        channel: AlertChannel,
        message: str,
        severity: str,
        context: dict[str, Any],
    ) -> bool:
        """Post a payload to a single channel. Never raises."""
        payload: dict[str, Any] | list[dict[str, Any]]
        if channel.format == "alertmanager":
            payload = _format_alertmanager_payload(message, severity, context)
            url = channel.url.rstrip("/") + "/api/v1/alerts"
        elif channel.format == "slack":
            payload = _format_slack_payload(message, severity, context)
            url = channel.url
        else:
            payload = _format_generic_payload(message, severity, context)
            url = channel.url

        try:
            resp = httpx.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(
                "Alert dispatched",
                extra={
                    "extra_json": {
                        "channel": channel.name,
                        "severity": severity,
                        "status": resp.status_code,
                    }
                },
            )
            return True
        except Exception as e:
            logger.error(
                "Alert dispatch failed",
                extra={
                    "extra_json": {
                        "channel": channel.name,
                        "error": str(e),
                        "message": message,
                    }
                },
            )
            return False


def build_alert(message: str) -> dict[str, str]:
    """Legacy helper — builds a simple alert dict."""
    return {"severity": "warning", "message": message}
