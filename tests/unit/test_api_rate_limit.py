"""Tests for per-IP API rate limiting.

Covers:
  - Requests within limit succeed (200)
  - Requests exceeding limit return 429 with Retry-After header
  - Different IPs have independent rate-limit buckets
  - Health and metrics endpoints are NOT rate-limited
  - Control endpoints use the stricter control limit
"""

from __future__ import annotations

import os

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from aiswarm.api.rate_limit import RateLimitDependency


# ---------------------------------------------------------------------------
# Fixtures: isolated FastAPI app with rate-limited and exempt routes
# ---------------------------------------------------------------------------


@pytest.fixture
def rate_limiter() -> RateLimitDependency:
    """A tight limiter: 3 requests/min for deterministic tests."""
    return RateLimitDependency(max_requests_per_minute=3.0, name="test")


@pytest.fixture
def app(rate_limiter: RateLimitDependency) -> FastAPI:
    """Minimal FastAPI app with rate-limited and unprotected routes."""
    _app = FastAPI()

    @_app.get("/limited", dependencies=[Depends(rate_limiter)])
    def limited_endpoint() -> dict[str, str]:
        return {"status": "ok"}

    @_app.get("/health")
    def health_endpoint() -> dict[str, str]:
        return {"status": "healthy"}

    @_app.get("/metrics")
    def metrics_endpoint() -> dict[str, str]:
        return {"metrics": "data"}

    return _app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRateLimitWithinBudget:
    """Requests within the allowed budget should succeed."""

    def test_requests_within_limit_succeed(self, client: TestClient) -> None:
        for _ in range(3):
            resp = client.get("/limited")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestRateLimitExceeded:
    """Requests exceeding the budget should receive HTTP 429."""

    def test_exceeding_limit_returns_429(self, client: TestClient) -> None:
        # Exhaust the 3-token budget
        for _ in range(3):
            resp = client.get("/limited")
            assert resp.status_code == 200

        # Next request should be rate-limited
        resp = client.get("/limited")
        assert resp.status_code == 429
        assert "Rate limit exceeded" in resp.json()["detail"]

    def test_429_includes_retry_after_header(self, client: TestClient) -> None:
        for _ in range(3):
            client.get("/limited")

        resp = client.get("/limited")
        assert resp.status_code == 429
        retry_after = resp.headers.get("retry-after")
        assert retry_after is not None
        assert int(retry_after) >= 1


class TestPerIpIsolation:
    """Different client IPs must have independent rate-limit buckets."""

    def test_different_ips_have_independent_limits(self) -> None:
        """Simulate two IPs by patching _extract_client_ip per request."""
        from unittest.mock import patch

        rl = RateLimitDependency(max_requests_per_minute=3.0, name="ip_test")
        _app = FastAPI()

        @_app.get("/limited", dependencies=[Depends(rl)])
        def limited() -> dict[str, str]:
            return {"ok": "true"}

        client = TestClient(_app)

        # Exhaust budget for IP "10.0.0.1"
        with patch.object(RateLimitDependency, "_extract_client_ip", return_value="10.0.0.1"):
            for _ in range(3):
                resp = client.get("/limited")
                assert resp.status_code == 200

            # 10.0.0.1 is now rate-limited
            resp = client.get("/limited")
            assert resp.status_code == 429

        # IP "10.0.0.2" should still have its own full budget
        with patch.object(RateLimitDependency, "_extract_client_ip", return_value="10.0.0.2"):
            for _ in range(3):
                resp = client.get("/limited")
                assert resp.status_code == 200


class TestHealthAndMetricsExempt:
    """Health and metrics endpoints must never be rate-limited."""

    def test_health_endpoint_not_rate_limited(self, client: TestClient) -> None:
        # Well beyond any reasonable limit
        for _ in range(100):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_metrics_endpoint_not_rate_limited(self, client: TestClient) -> None:
        for _ in range(100):
            resp = client.get("/metrics")
            assert resp.status_code == 200


class TestControlEndpointRateLimit:
    """Control endpoints in the real app should use the stricter limit."""

    def setup_method(self) -> None:
        os.environ["AIS_API_KEY"] = "test-api-key"
        self.headers = {"Authorization": "Bearer test-api-key"}

    def test_control_pause_rate_limited(self) -> None:
        """The /control/pause endpoint uses require_control_rate_limit (10 req/min).

        We patch the control rate limiter's acquire method to simulate exhaustion.
        """
        from aiswarm.api.app import app as real_app
        from aiswarm.api.rate_limit import require_control_rate_limit

        client = TestClient(real_app)

        # Drain the control limiter for our test IP
        ip = "testclient"
        limiter = require_control_rate_limit._get_limiter(ip)
        # Exhaust all tokens
        while limiter.acquire():
            pass

        resp = client.post(
            "/control/pause",
            headers=self.headers,
            json={"reason": "test"},
        )
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

        # Reset the limiter state so other tests are not affected
        with require_control_rate_limit._lock:
            require_control_rate_limit._limiters.pop(ip, None)


class TestRateLimitDependencyUnit:
    """Unit tests for the RateLimitDependency class itself."""

    def test_configurable_parameters(self) -> None:
        dep = RateLimitDependency(max_requests_per_minute=120.0, name="custom")
        assert dep.name == "custom"
        assert dep.max_requests_per_minute == 120.0
        assert dep._max_tokens == 120.0
        assert dep._refill_rate == pytest.approx(2.0)

    def test_stats_returns_per_ip_data(self, rate_limiter: RateLimitDependency) -> None:
        # Access a limiter for an IP to populate the registry
        limiter = rate_limiter._get_limiter("192.168.1.1")
        limiter.acquire()

        stats = rate_limiter.stats()
        assert "192.168.1.1" in stats
        assert stats["192.168.1.1"].total_allowed == 1

    def test_estimate_retry_after_minimum_is_one(self, rate_limiter: RateLimitDependency) -> None:
        limiter = rate_limiter._get_limiter("10.0.0.1")
        # Even with tokens available, retry_after should be at least 1
        retry = rate_limiter._estimate_retry_after(limiter)
        assert retry >= 1

    def test_extract_client_ip_none_client(self) -> None:
        """When request.client is None, should return 'unknown'."""
        from unittest.mock import MagicMock

        request = MagicMock()
        request.client = None
        assert RateLimitDependency._extract_client_ip(request) == "unknown"
