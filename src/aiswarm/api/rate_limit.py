"""Per-IP rate limiting for FastAPI endpoints.

Reuses the existing TokenBucketRateLimiter from the resilience layer to
enforce per-client request limits.  Returns HTTP 429 with a Retry-After
header when a client exceeds its allocation.

Two pre-configured dependency factories are exported:

* ``require_general_rate_limit``  — 60 requests / minute (burst 60)
* ``require_control_rate_limit``  — 10 requests / minute (burst 10)

Health-check and Prometheus metrics endpoints are intentionally excluded
from rate limiting (they are consumed by monitoring infrastructure).
"""

import math
import threading
from typing import Any

from fastapi import HTTPException, Request, status

from aiswarm.resilience.rate_limiter import TokenBucketRateLimiter
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitDependency:
    """FastAPI dependency that enforces per-IP rate limiting.

    Instantiate once per tier and inject via ``Depends(instance)``.

    Args:
        max_requests_per_minute: Sustained request budget per client IP.
        name: Identifier used in logging and diagnostics.
    """

    def __init__(
        self,
        max_requests_per_minute: float = 60.0,
        name: str = "api",
    ) -> None:
        self.name = name
        self.max_requests_per_minute = max_requests_per_minute

        # Token-bucket parameters derived from the per-minute budget.
        # burst  = max_requests_per_minute  (full minute of tokens available)
        # refill = max_requests_per_minute / 60  (tokens per second)
        self._max_tokens = max_requests_per_minute
        self._refill_rate = max_requests_per_minute / 60.0

        self._limiters: dict[str, TokenBucketRateLimiter] = {}
        self._lock = threading.Lock()

    # -- FastAPI dependency protocol ----------------------------------------

    async def __call__(self, request: Request) -> None:
        """Acquire a token for the caller's IP or raise HTTP 429."""
        client_ip = self._extract_client_ip(request)
        limiter = self._get_limiter(client_ip)

        if not limiter.acquire():
            retry_after = self._estimate_retry_after(limiter)
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "extra_json": {
                        "limiter": self.name,
                        "client_ip": client_ip,
                        "retry_after": retry_after,
                    }
                },
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )

    # -- Internals ----------------------------------------------------------

    @staticmethod
    def _extract_client_ip(request: Request) -> str:
        """Return the client IP from the ASGI scope.

        Falls back to ``"unknown"`` when running behind certain test
        harnesses where ``request.client`` is ``None``.
        """
        if request.client is not None:
            return request.client.host
        return "unknown"

    def _get_limiter(self, client_ip: str) -> TokenBucketRateLimiter:
        """Get or create a per-IP rate limiter (thread-safe)."""
        with self._lock:
            if client_ip not in self._limiters:
                self._limiters[client_ip] = TokenBucketRateLimiter(
                    name=f"{self.name}:{client_ip}",
                    max_tokens=self._max_tokens,
                    refill_rate=self._refill_rate,
                )
            return self._limiters[client_ip]

    def _estimate_retry_after(self, limiter: TokenBucketRateLimiter) -> int:
        """Estimate seconds until at least one token is available."""
        stats = limiter.stats()
        if stats.refill_rate <= 0:
            return 60
        deficit = 1.0 - stats.tokens_available
        if deficit <= 0:
            return 1
        return max(1, math.ceil(deficit / stats.refill_rate))

    # -- Diagnostics --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return per-IP stats for observability."""
        with self._lock:
            return {ip: limiter.stats() for ip, limiter in self._limiters.items()}


# ---------------------------------------------------------------------------
# Pre-configured dependency instances
# ---------------------------------------------------------------------------

require_general_rate_limit = RateLimitDependency(
    max_requests_per_minute=60.0,
    name="api_general",
)

require_control_rate_limit = RateLimitDependency(
    max_requests_per_minute=10.0,
    name="api_control",
)
