"""Tests for HTTPMCPGateway — resilience, redaction, and call history capping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aiswarm.execution.http_mcp_gateway import HTTPMCPGateway


@pytest.fixture
def gateway() -> HTTPMCPGateway:
    return HTTPMCPGateway(
        server_url="http://localhost:9999",
        exchange_name="test",
        timeout=1.0,
        rate_limit_rps=100.0,
        circuit_failure_threshold=3,
        circuit_recovery_timeout=1.0,
    )


class TestCallHistoryRedaction:
    def test_account_id_redacted_in_call_history(self, gateway: HTTPMCPGateway) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(gateway._client, "post", return_value=mock_response):
            gateway.call_tool("get_balance", {"account_id": "secret-123", "symbol": "BTCUSDT"})

        assert len(gateway.call_history) == 1
        record = gateway.call_history[0]
        assert record.params["account_id"] == "***"
        assert record.params["symbol"] == "BTCUSDT"

    def test_api_key_redacted_in_call_history(self, gateway: HTTPMCPGateway) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(gateway._client, "post", return_value=mock_response):
            gateway.call_tool("connect", {"api_key": "my-key", "api_secret": "my-secret"})

        record = gateway.call_history[0]
        assert record.params["api_key"] == "***"
        assert record.params["api_secret"] == "***"


class TestCallHistoryCap:
    def test_call_history_capped_at_max(self, gateway: HTTPMCPGateway) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(gateway._client, "post", return_value=mock_response):
            for i in range(gateway._MAX_CALL_HISTORY + 50):
                gateway.call_tool("ping", {"seq": i})

        assert len(gateway.call_history) <= gateway._MAX_CALL_HISTORY


class TestCircuitBreaker:
    def test_circuit_opens_after_failures(self, gateway: HTTPMCPGateway) -> None:
        with patch.object(gateway._client, "post", side_effect=ConnectionError("down")):
            for _ in range(gateway._circuit_breaker.failure_threshold):
                with pytest.raises(ConnectionError):
                    gateway.call_tool("ping", {})

        # Circuit should now be open
        with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
            gateway.call_tool("ping", {})


class TestRateLimiter:
    def test_gateway_has_rate_limiter(self, gateway: HTTPMCPGateway) -> None:
        assert gateway.rate_limiter is not None
        assert gateway.circuit_breaker is not None
