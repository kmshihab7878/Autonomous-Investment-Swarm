"""Generic HTTP MCP gateway — exchange-agnostic HTTP gateway for MCP servers.

Parameterized version of AsterMCPGateway that works with any MCP server.
The ``exchange_name`` parameter is used for metrics labels and logging.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from aiswarm.execution.mcp_gateway import MCPCallRecord
from aiswarm.resilience.circuit_breaker import CircuitBreaker
from aiswarm.resilience.rate_limiter import TokenBucketRateLimiter
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class HTTPMCPGateway:
    """Production MCP gateway — calls any MCP server via HTTP.

    Integrates circuit breaker and rate limiter for resilience against
    exchange rate limits and outages. The ``exchange_name`` is used for
    Prometheus metric labels and circuit breaker identification.
    """

    _MAX_CALL_HISTORY = 1000
    _REDACT_KEYS = frozenset({"account_id", "api_key", "api_secret", "password", "token"})

    def __init__(
        self,
        server_url: str,
        exchange_name: str = "default",
        timeout: float = 10.0,
        rate_limit_rps: float = 5.0,
        circuit_failure_threshold: int = 5,
        circuit_recovery_timeout: float = 30.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.exchange_name = exchange_name
        self.timeout = timeout
        self.call_history: list[MCPCallRecord] = []
        self._client = httpx.Client(timeout=timeout)

        self._rate_limiter = TokenBucketRateLimiter(
            name=f"{exchange_name}_mcp",
            max_tokens=rate_limit_rps * 2,
            refill_rate=rate_limit_rps,
        )
        self._circuit_breaker = CircuitBreaker(
            name=f"{exchange_name}_mcp",
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
        )

    def call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Invoke an MCP tool on the server.

        Respects rate limits and circuit breaker. Raises on failure.
        """
        from aiswarm.monitoring import metrics as m

        if not self._circuit_breaker.allow_request():
            m.EXCHANGE_ERRORS.labels(exchange=self.exchange_name, tool=tool_name).inc()
            raise ConnectionError(
                f"Circuit breaker OPEN for {self.exchange_name}_mcp — skipping {tool_name}"
            )

        if not self._rate_limiter.wait_and_acquire(timeout=self.timeout):
            m.EXCHANGE_ERRORS.labels(exchange=self.exchange_name, tool=tool_name).inc()
            raise TimeoutError(f"Rate limiter timeout for {tool_name}")

        start = time.monotonic()
        try:
            response = self._client.post(
                f"{self.server_url}/call-tool",
                json={"tool_name": tool_name, "params": params},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            self._circuit_breaker.record_success()
            elapsed = time.monotonic() - start
            m.EXCHANGE_LATENCY.labels(exchange=self.exchange_name, tool=tool_name).observe(elapsed)

            safe_params = {k: ("***" if k in self._REDACT_KEYS else v) for k, v in params.items()}
            record = MCPCallRecord(tool_name=tool_name, params=safe_params, response=result)
            self.call_history.append(record)
            if len(self.call_history) > self._MAX_CALL_HISTORY:
                self.call_history = self.call_history[-self._MAX_CALL_HISTORY :]

            logger.info(
                "MCP call",
                extra={
                    "extra_json": {
                        "exchange": self.exchange_name,
                        "tool": tool_name,
                        "latency_ms": round(elapsed * 1000, 1),
                    }
                },
            )
            return result

        except Exception as e:
            self._circuit_breaker.record_failure()
            elapsed = time.monotonic() - start
            m.EXCHANGE_LATENCY.labels(exchange=self.exchange_name, tool=tool_name).observe(elapsed)
            m.EXCHANGE_ERRORS.labels(exchange=self.exchange_name, tool=tool_name).inc()

            logger.error(
                "MCP call failed",
                extra={
                    "extra_json": {
                        "exchange": self.exchange_name,
                        "tool": tool_name,
                        "error": str(e),
                        "latency_ms": round(elapsed * 1000, 1),
                    }
                },
            )
            raise

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker

    @property
    def rate_limiter(self) -> TokenBucketRateLimiter:
        return self._rate_limiter
