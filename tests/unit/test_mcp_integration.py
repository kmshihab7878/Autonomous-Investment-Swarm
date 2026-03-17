"""Integration tests for AsterMCPGateway with simulated HTTP scenarios.

Verifies that the real AsterMCPGateway class handles connection failures,
timeouts, malformed responses, and interacts correctly with the circuit
breaker and rate limiter under adverse conditions.

All HTTP calls are intercepted via unittest.mock patches on httpx.Client.post
so no real MCP server is required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from aiswarm.execution.mcp_gateway import AsterMCPGateway
from aiswarm.resilience.circuit_breaker import CircuitState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gateway(
    *,
    failure_threshold: int = 3,
    recovery_timeout: float = 0.2,
    rate_limit_rps: float = 100.0,
    timeout: float = 5.0,
) -> AsterMCPGateway:
    """Create an AsterMCPGateway with fast-failing resilience settings."""
    return AsterMCPGateway(
        server_url="http://fake-mcp:8080",
        timeout=timeout,
        rate_limit_rps=rate_limit_rps,
        circuit_failure_threshold=failure_threshold,
        circuit_recovery_timeout=recovery_timeout,
    )


def _ok_response(body: dict | None = None, status: int = 200) -> httpx.Response:
    """Build a realistic httpx.Response with JSON body."""
    body = body or {}
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
    )


def _error_response(status: int = 500, body: str = "Internal Server Error") -> httpx.Response:
    """Build an httpx error response."""
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
    )


# ---------------------------------------------------------------------------
# 1. Successful order submission
# ---------------------------------------------------------------------------


class TestSuccessfulOrderSubmission:
    """Mock server returns valid order ID; verify gateway parses correctly."""

    @patch.object(httpx.Client, "post")
    def test_parses_valid_order_response(self, mock_post: MagicMock) -> None:
        expected = {
            "orderId": "EX00000001",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "status": "NEW",
        }
        mock_post.return_value = _ok_response(expected)

        gw = _make_gateway()
        result = gw.call_tool(
            "mcp__aster__create_order",
            {"symbol": "BTCUSDT", "side": "BUY", "quantity": 0.1},
        )

        assert result == expected
        assert result["orderId"] == "EX00000001"
        assert result["status"] == "NEW"

    @patch.object(httpx.Client, "post")
    def test_records_call_in_history(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"orderId": "EX1"})

        gw = _make_gateway()
        gw.call_tool("mcp__aster__create_order", {"symbol": "ETHUSDT"})

        assert len(gw.call_history) == 1
        assert gw.call_history[0].tool_name == "mcp__aster__create_order"
        assert gw.call_history[0].params == {"symbol": "ETHUSDT"}
        assert gw.call_history[0].response == {"orderId": "EX1"}

    @patch.object(httpx.Client, "post")
    def test_circuit_breaker_records_success(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"orderId": "EX1"})

        gw = _make_gateway()
        gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        stats = gw.circuit_breaker.stats()
        assert stats.success_count == 1
        assert stats.failure_count == 0
        assert stats.state == CircuitState.CLOSED

    @patch.object(httpx.Client, "post")
    def test_posts_to_correct_endpoint_with_payload(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"orderId": "EX1"})

        gw = _make_gateway()
        gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT", "side": "BUY"})

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == "http://fake-mcp:8080/call-tool"
        assert call_args.kwargs["json"] == {
            "tool_name": "mcp__aster__create_order",
            "params": {"symbol": "BTCUSDT", "side": "BUY"},
        }


# ---------------------------------------------------------------------------
# 2. Connection timeout
# ---------------------------------------------------------------------------


class TestConnectionTimeout:
    """Mock server does not respond within timeout."""

    @patch.object(httpx.Client, "post")
    def test_timeout_raises_and_counts_failure(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.TimeoutException("Connection timed out")

        gw = _make_gateway(failure_threshold=3)

        with pytest.raises(httpx.TimeoutException, match="Connection timed out"):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1

    @patch.object(httpx.Client, "post")
    def test_repeated_timeouts_open_circuit(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.TimeoutException("timed out")

        gw = _make_gateway(failure_threshold=3)

        for _ in range(3):
            with pytest.raises(httpx.TimeoutException):
                gw.call_tool("mcp__aster__get_balance", {})

        assert gw.circuit_breaker.state == CircuitState.OPEN

    @patch.object(httpx.Client, "post")
    def test_open_circuit_rejects_without_http_call(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.TimeoutException("timed out")

        gw = _make_gateway(failure_threshold=3)

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(httpx.TimeoutException):
                gw.call_tool("mcp__aster__get_balance", {})

        # Reset mock to track subsequent calls
        mock_post.reset_mock()

        # Next call should be rejected by circuit breaker, not HTTP
        with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
            gw.call_tool("mcp__aster__get_balance", {})

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# 3. HTTP 500 from server
# ---------------------------------------------------------------------------


class TestHttp500Response:
    """Server returns 500; verify gateway treats as failure."""

    @patch.object(httpx.Client, "post")
    def test_500_raises_http_status_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _error_response(500)

        gw = _make_gateway()

        with pytest.raises(httpx.HTTPStatusError):
            gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

    @patch.object(httpx.Client, "post")
    def test_500_counts_circuit_breaker_failure(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _error_response(500)

        gw = _make_gateway(failure_threshold=3)

        with pytest.raises(httpx.HTTPStatusError):
            gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1
        assert stats.state == CircuitState.CLOSED

    @patch.object(httpx.Client, "post")
    def test_500_does_not_record_in_call_history(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _error_response(500)

        gw = _make_gateway()

        with pytest.raises(httpx.HTTPStatusError):
            gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        assert len(gw.call_history) == 0

    @patch.object(httpx.Client, "post")
    def test_repeated_500s_open_circuit(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _error_response(500)

        gw = _make_gateway(failure_threshold=3)

        for _ in range(3):
            with pytest.raises(httpx.HTTPStatusError):
                gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        assert gw.circuit_breaker.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# 4. Malformed JSON response
# ---------------------------------------------------------------------------


class TestMalformedJsonResponse:
    """Server returns invalid JSON; gateway must not crash."""

    @patch.object(httpx.Client, "post")
    def test_invalid_json_raises_without_crash(self, mock_post: MagicMock) -> None:
        response = httpx.Response(
            status_code=200,
            content=b"not valid json {{{",
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
        )
        mock_post.return_value = response

        gw = _make_gateway()

        with pytest.raises(Exception):
            gw.call_tool("mcp__aster__get_balance", {})

    @patch.object(httpx.Client, "post")
    def test_invalid_json_counts_circuit_breaker_failure(self, mock_post: MagicMock) -> None:
        response = httpx.Response(
            status_code=200,
            content=b"not json!",
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
        )
        mock_post.return_value = response

        gw = _make_gateway()

        with pytest.raises(Exception):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1

    @patch.object(httpx.Client, "post")
    def test_empty_body_200_raises(self, mock_post: MagicMock) -> None:
        response = httpx.Response(
            status_code=200,
            content=b"",
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
        )
        mock_post.return_value = response

        gw = _make_gateway()

        with pytest.raises(Exception):
            gw.call_tool("mcp__aster__get_balance", {})


# ---------------------------------------------------------------------------
# 5. Rate limiter integration
# ---------------------------------------------------------------------------


class TestRateLimiterIntegration:
    """Verify rate limiter gates requests correctly."""

    @patch.object(httpx.Client, "post")
    def test_requests_within_limit_succeed(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"status": "ok"})

        # High rate limit so calls easily pass
        gw = _make_gateway(rate_limit_rps=100.0)

        for _ in range(5):
            result = gw.call_tool("mcp__aster__get_balance", {})
            assert result["status"] == "ok"

        assert mock_post.call_count == 5

    @patch.object(httpx.Client, "post")
    def test_exhausted_tokens_raises_timeout_error(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"status": "ok"})

        # Very low burst: max_tokens = 0.5 * 2 = 1.0, refill = 0.5/s
        # With a very short wait timeout, second call should fail
        gw = _make_gateway(rate_limit_rps=0.5, timeout=0.1)

        # First call consumes the one available token
        gw.call_tool("mcp__aster__get_balance", {})

        # Second call: rate limiter has no tokens and wait_and_acquire
        # will time out after 0.1s
        with pytest.raises(TimeoutError, match="Rate limiter timeout"):
            gw.call_tool("mcp__aster__get_balance", {})

    @patch.object(httpx.Client, "post")
    def test_rate_limiter_stats_reflect_throttling(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({"status": "ok"})

        gw = _make_gateway(rate_limit_rps=0.5, timeout=0.1)

        # First call succeeds
        gw.call_tool("mcp__aster__get_balance", {})

        # Second call gets rate-limited
        with pytest.raises(TimeoutError):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.rate_limiter.stats()
        assert stats.total_allowed >= 1
        # The rate limiter internally retries via wait_and_acquire,
        # so total_throttled should be > 0 from the failed attempt
        assert stats.total_throttled > 0


# ---------------------------------------------------------------------------
# 6. Circuit breaker recovery (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
# ---------------------------------------------------------------------------


class TestCircuitBreakerRecovery:
    """Full lifecycle: failures open, timeout -> half-open, success -> closed."""

    @patch.object(httpx.Client, "post")
    def test_full_recovery_lifecycle(self, mock_post: MagicMock) -> None:
        gw = _make_gateway(failure_threshold=2, recovery_timeout=0.15)

        # Phase 1: Two failures open the circuit
        mock_post.side_effect = httpx.TimeoutException("down")
        for _ in range(2):
            with pytest.raises(httpx.TimeoutException):
                gw.call_tool("mcp__aster__get_balance", {})
        assert gw.circuit_breaker.state == CircuitState.OPEN

        # Phase 2: Calls are rejected immediately while OPEN
        with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
            gw.call_tool("mcp__aster__get_balance", {})

        # Phase 3: Wait for recovery timeout -> HALF_OPEN
        time.sleep(0.2)
        assert gw.circuit_breaker.state == CircuitState.HALF_OPEN

        # Phase 4: Successful probe closes the circuit
        mock_post.side_effect = None
        mock_post.return_value = _ok_response({"status": "ok"})
        result = gw.call_tool("mcp__aster__get_balance", {})
        assert result == {"status": "ok"}
        assert gw.circuit_breaker.state == CircuitState.CLOSED

    @patch.object(httpx.Client, "post")
    def test_half_open_failure_reopens_circuit(self, mock_post: MagicMock) -> None:
        gw = _make_gateway(failure_threshold=2, recovery_timeout=0.15)

        # Open the circuit
        mock_post.side_effect = httpx.TimeoutException("down")
        for _ in range(2):
            with pytest.raises(httpx.TimeoutException):
                gw.call_tool("mcp__aster__get_balance", {})
        assert gw.circuit_breaker.state == CircuitState.OPEN

        # Wait for half-open
        time.sleep(0.2)
        assert gw.circuit_breaker.state == CircuitState.HALF_OPEN

        # Probe fails -> re-opens
        with pytest.raises(httpx.TimeoutException):
            gw.call_tool("mcp__aster__get_balance", {})
        assert gw.circuit_breaker.state == CircuitState.OPEN

    @patch.object(httpx.Client, "post")
    def test_circuit_rejects_multiple_calls_while_open(self, mock_post: MagicMock) -> None:
        gw = _make_gateway(failure_threshold=2, recovery_timeout=60.0)

        # Open the circuit
        mock_post.side_effect = httpx.TimeoutException("down")
        for _ in range(2):
            with pytest.raises(httpx.TimeoutException):
                gw.call_tool("mcp__aster__get_balance", {})

        mock_post.reset_mock()

        # All subsequent calls are rejected without HTTP
        for _ in range(5):
            with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
                gw.call_tool("mcp__aster__get_balance", {})

        mock_post.assert_not_called()

        stats = gw.circuit_breaker.stats()
        assert stats.total_rejections == 5


# ---------------------------------------------------------------------------
# 7. Partial / unexpected schema response
# ---------------------------------------------------------------------------


class TestPartialResponse:
    """Server returns 200 but body has unexpected schema."""

    @patch.object(httpx.Client, "post")
    def test_missing_order_id_returns_empty_string(self, mock_post: MagicMock) -> None:
        """Gateway returns the raw response even without orderId field.

        The LiveOrderExecutor layer checks for missing orderId; the gateway
        itself just returns whatever JSON the server sends.
        """
        mock_post.return_value = _ok_response({"unexpected": "payload"})

        gw = _make_gateway()
        result = gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        # Gateway passes through the raw response
        assert result == {"unexpected": "payload"}
        assert "orderId" not in result

    @patch.object(httpx.Client, "post")
    def test_empty_json_object_is_valid(self, mock_post: MagicMock) -> None:
        mock_post.return_value = _ok_response({})

        gw = _make_gateway()
        result = gw.call_tool("mcp__aster__get_positions", {})

        assert result == {}
        # Should still count as success for circuit breaker
        stats = gw.circuit_breaker.stats()
        assert stats.success_count == 1
        assert stats.failure_count == 0

    @patch.object(httpx.Client, "post")
    def test_null_values_in_response_pass_through(self, mock_post: MagicMock) -> None:
        body = {"orderId": None, "status": None, "extra": "data"}
        mock_post.return_value = _ok_response(body)

        gw = _make_gateway()
        result = gw.call_tool("mcp__aster__create_order", {"symbol": "BTCUSDT"})

        assert result["orderId"] is None
        assert result["extra"] == "data"

    @patch.object(httpx.Client, "post")
    def test_html_error_page_on_200_raises(self, mock_post: MagicMock) -> None:
        """Some proxies return 200 with HTML error pages."""
        response = httpx.Response(
            status_code=200,
            content=b"<html><body>502 Bad Gateway</body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("POST", "http://fake-mcp:8080/call-tool"),
        )
        mock_post.return_value = response

        gw = _make_gateway()

        with pytest.raises(Exception):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1


# ---------------------------------------------------------------------------
# 8. Connection refused (server down)
# ---------------------------------------------------------------------------


class TestConnectionRefused:
    """Server is unreachable; ConnectionError must be handled."""

    @patch.object(httpx.Client, "post")
    def test_connection_error_is_raised(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        gw = _make_gateway()

        with pytest.raises(httpx.ConnectError, match="Connection refused"):
            gw.call_tool("mcp__aster__get_balance", {})

    @patch.object(httpx.Client, "post")
    def test_connection_error_counts_failure(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        gw = _make_gateway(failure_threshold=3)

        with pytest.raises(httpx.ConnectError):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1
        assert stats.state == CircuitState.CLOSED

    @patch.object(httpx.Client, "post")
    def test_connection_refused_opens_circuit_after_threshold(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        gw = _make_gateway(failure_threshold=3)

        for _ in range(3):
            with pytest.raises(httpx.ConnectError):
                gw.call_tool("mcp__aster__get_balance", {})

        assert gw.circuit_breaker.state == CircuitState.OPEN

    @patch.object(httpx.Client, "post")
    def test_dns_resolution_failure(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = httpx.ConnectError("Name or service not known")

        gw = _make_gateway()

        with pytest.raises(httpx.ConnectError, match="Name or service not known"):
            gw.call_tool("mcp__aster__get_balance", {})

        stats = gw.circuit_breaker.stats()
        assert stats.failure_count == 1


# ---------------------------------------------------------------------------
# 9. Mixed scenarios: resilience components working together
# ---------------------------------------------------------------------------


class TestResilienceCombined:
    """Verify circuit breaker and rate limiter interact correctly."""

    @patch.object(httpx.Client, "post")
    def test_circuit_open_takes_priority_over_rate_limiter(self, mock_post: MagicMock) -> None:
        """When circuit is open, request should fail at circuit breaker
        without consuming a rate limiter token."""
        mock_post.side_effect = httpx.ConnectError("down")

        gw = _make_gateway(failure_threshold=2, rate_limit_rps=100.0)

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(httpx.ConnectError):
                gw.call_tool("mcp__aster__get_balance", {})

        assert gw.circuit_breaker.state == CircuitState.OPEN

        rl_stats_before = gw.rate_limiter.stats()

        # This call should be rejected by circuit breaker
        with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
            gw.call_tool("mcp__aster__get_balance", {})

        # Rate limiter should not have consumed an additional token
        rl_stats_after = gw.rate_limiter.stats()
        assert rl_stats_after.total_allowed == rl_stats_before.total_allowed

    @patch.object(httpx.Client, "post")
    def test_success_after_failures_keeps_circuit_closed(self, mock_post: MagicMock) -> None:
        """Intermittent failures below threshold should not open circuit."""
        gw = _make_gateway(failure_threshold=5)

        # 2 failures
        mock_post.side_effect = httpx.ConnectError("down")
        for _ in range(2):
            with pytest.raises(httpx.ConnectError):
                gw.call_tool("mcp__aster__get_balance", {})

        assert gw.circuit_breaker.state == CircuitState.CLOSED

        # 1 success -- note: circuit breaker failure_count is NOT reset by
        # success in CLOSED state (only reset when transitioning from
        # HALF_OPEN to CLOSED), but state remains CLOSED since we haven't
        # hit threshold
        mock_post.side_effect = None
        mock_post.return_value = _ok_response({"status": "ok"})
        result = gw.call_tool("mcp__aster__get_balance", {})
        assert result["status"] == "ok"
        assert gw.circuit_breaker.state == CircuitState.CLOSED

    @patch.object(httpx.Client, "post")
    def test_different_error_types_all_count_as_failures(self, mock_post: MagicMock) -> None:
        """Timeout, connect error, and HTTP 500 all increment failure count."""
        gw = _make_gateway(failure_threshold=3)

        # Error 1: timeout
        mock_post.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(httpx.TimeoutException):
            gw.call_tool("mcp__aster__get_balance", {})

        # Error 2: connection refused
        mock_post.side_effect = httpx.ConnectError("refused")
        with pytest.raises(httpx.ConnectError):
            gw.call_tool("mcp__aster__get_balance", {})

        # Error 3: HTTP 500
        mock_post.side_effect = None
        mock_post.return_value = _error_response(500)
        with pytest.raises(httpx.HTTPStatusError):
            gw.call_tool("mcp__aster__get_balance", {})

        # Three different error types but all should trip the breaker
        assert gw.circuit_breaker.state == CircuitState.OPEN

    @patch.object(httpx.Client, "post")
    def test_multiple_tools_share_circuit_breaker(self, mock_post: MagicMock) -> None:
        """Failures across different tool names share the same breaker."""
        gw = _make_gateway(failure_threshold=3)
        mock_post.side_effect = httpx.ConnectError("down")

        tools = [
            "mcp__aster__get_balance",
            "mcp__aster__create_order",
            "mcp__aster__get_positions",
        ]
        for tool in tools:
            with pytest.raises(httpx.ConnectError):
                gw.call_tool(tool, {})

        assert gw.circuit_breaker.state == CircuitState.OPEN

        # All tools are now blocked
        for tool in tools:
            with pytest.raises(ConnectionError, match="Circuit breaker OPEN"):
                gw.call_tool(tool, {})
