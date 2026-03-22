"""Tests for the retry decorator with exponential backoff and circuit breaker integration."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from aiswarm.resilience.circuit_breaker import CircuitBreaker, CircuitState
from aiswarm.resilience.retry import (
    RetryExhausted,
    _compute_delay,
    retry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TransientError(Exception):
    """Simulates a retryable transient failure."""


class PermanentError(Exception):
    """Simulates a non-retryable failure."""


def _make_open_breaker(name: str = "test_breaker") -> CircuitBreaker:
    """Create a circuit breaker that is already in OPEN state."""
    breaker = CircuitBreaker(name, failure_threshold=1, recovery_timeout=9999)
    breaker.record_failure()  # Triggers OPEN since threshold is 1
    assert breaker.state == CircuitState.OPEN
    return breaker


# ---------------------------------------------------------------------------
# Tests: sync retry behavior
# ---------------------------------------------------------------------------


class TestRetrySync:
    def test_successful_call_returns_immediately(self) -> None:
        """A function that succeeds on the first call returns without retry."""
        # Arrange
        call_count = 0

        @retry(max_attempts=3, retryable=(TransientError,), jitter=False)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        # Act
        result = succeed()

        # Assert
        assert result == "ok"
        assert call_count == 1

    def test_retryable_exception_triggers_retry_up_to_max_attempts(self) -> None:
        """A retryable exception causes the function to be retried max_attempts times."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise TransientError("boom")

        # Act & Assert
        with pytest.raises(RetryExhausted) as exc_info:
            always_fail()

        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, TransientError)

    def test_non_retryable_exception_is_not_retried(self) -> None:
        """An exception NOT in the retryable tuple propagates immediately."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        def fail_permanently() -> None:
            nonlocal call_count
            call_count += 1
            raise PermanentError("fatal")

        # Act & Assert
        with pytest.raises(PermanentError, match="fatal"):
            fail_permanently()

        assert call_count == 1

    def test_succeeds_after_transient_failures(self) -> None:
        """Function succeeds on attempt 3 after failing twice."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=5,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("transient")
            return "recovered"

        # Act
        result = flaky()

        # Assert
        assert result == "recovered"
        assert call_count == 3

    def test_custom_retryable_tuple(self) -> None:
        """Only exceptions in the retryable tuple trigger retries."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable=(ValueError, TypeError),
            jitter=False,
        )
        def fail_with_value_error() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("retryable")

        # Act & Assert
        with pytest.raises(RetryExhausted):
            fail_with_value_error()

        assert call_count == 3


# ---------------------------------------------------------------------------
# Tests: exponential backoff
# ---------------------------------------------------------------------------


class TestBackoffComputation:
    def test_exponential_backoff_delays_increase(self) -> None:
        """Delays double with each attempt (no jitter)."""
        # Arrange & Act
        d1 = _compute_delay(attempt=1, backoff_base=1.0, backoff_max=60.0, jitter=False)
        d2 = _compute_delay(attempt=2, backoff_base=1.0, backoff_max=60.0, jitter=False)
        d3 = _compute_delay(attempt=3, backoff_base=1.0, backoff_max=60.0, jitter=False)

        # Assert
        assert d1 == pytest.approx(1.0)
        assert d2 == pytest.approx(2.0)
        assert d3 == pytest.approx(4.0)

    def test_backoff_respects_max_cap(self) -> None:
        """Delay never exceeds backoff_max."""
        # Arrange & Act
        delay = _compute_delay(attempt=20, backoff_base=1.0, backoff_max=30.0, jitter=False)

        # Assert
        assert delay == pytest.approx(30.0)

    def test_jitter_produces_non_deterministic_delays(self) -> None:
        """With jitter enabled, multiple calls produce varying delays."""
        # Arrange & Act
        delays = {
            _compute_delay(attempt=3, backoff_base=1.0, backoff_max=60.0, jitter=True)
            for _ in range(50)
        }

        # Assert — with 50 samples and random jitter, we expect >1 unique value
        assert len(delays) > 1

    def test_jitter_stays_within_bounds(self) -> None:
        """Jitter produces delays between 0.5*base_delay and base_delay."""
        # Arrange
        base_delay = 4.0  # attempt=3: 1.0 * 2^2 = 4.0

        # Act
        delays = [
            _compute_delay(attempt=3, backoff_base=1.0, backoff_max=60.0, jitter=True)
            for _ in range(200)
        ]

        # Assert — jitter formula: delay * (0.5 + rand * 0.5) => [0.5*delay, delay)
        for d in delays:
            assert d >= base_delay * 0.5
            assert d <= base_delay  # upper bound is exclusive but float rounding


# ---------------------------------------------------------------------------
# Tests: circuit breaker integration
# ---------------------------------------------------------------------------


class TestRetryWithCircuitBreaker:
    def test_open_breaker_skips_retries_no_prior_exception(self) -> None:
        """When breaker is OPEN from the start, RetryExhausted is raised with 0 attempts."""
        # Arrange
        breaker = _make_open_breaker("open_skip_test")

        @retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable=(TransientError,),
            breaker_name="open_skip_test",
            jitter=False,
        )
        def guarded_call() -> str:
            return "should not reach"

        # Act & Assert
        with patch("aiswarm.resilience.retry.get_breaker", return_value=breaker):
            with pytest.raises(RetryExhausted) as exc_info:
                guarded_call()

        assert exc_info.value.attempts == 0
        assert "OPEN" in str(exc_info.value.last_exception)

    def test_open_breaker_with_prior_exception_reports_last_error(self) -> None:
        """If breaker opens mid-retry, the last real exception is reported."""
        # Arrange
        call_count = 0
        breaker = CircuitBreaker("mid_open_test", failure_threshold=1, recovery_timeout=9999)

        @retry(
            max_attempts=5,
            backoff_base=0.001,
            retryable=(TransientError,),
            breaker_name="mid_open_test",
            jitter=False,
        )
        def flaky_with_breaker() -> str:
            nonlocal call_count
            call_count += 1
            raise TransientError("transient")

        # Act & Assert — breaker opens after 1 failure, second attempt sees OPEN
        with patch("aiswarm.resilience.retry.get_breaker", return_value=breaker):
            with pytest.raises(RetryExhausted) as exc_info:
                flaky_with_breaker()

        # First attempt fails, breaker records failure -> OPEN
        # Second attempt blocked by breaker
        assert exc_info.value.attempts == 1
        assert isinstance(exc_info.value.last_exception, TransientError)

    def test_circuit_breaker_records_success(self) -> None:
        """A successful call records success on the breaker."""
        # Arrange
        breaker = MagicMock()
        breaker.allow_request.return_value = True

        @retry(
            max_attempts=3,
            retryable=(TransientError,),
            breaker_name="success_record_test",
            jitter=False,
        )
        def succeed() -> str:
            return "ok"

        # Act
        with patch("aiswarm.resilience.retry.get_breaker", return_value=breaker):
            result = succeed()

        # Assert
        assert result == "ok"
        breaker.record_success.assert_called_once()

    def test_circuit_breaker_records_failure(self) -> None:
        """A failed retryable call records failure on the breaker each time."""
        # Arrange
        breaker = MagicMock()
        breaker.allow_request.return_value = True

        @retry(
            max_attempts=2,
            backoff_base=0.001,
            retryable=(TransientError,),
            breaker_name="failure_record_test",
            jitter=False,
        )
        def always_fail() -> None:
            raise TransientError("boom")

        # Act
        with patch("aiswarm.resilience.retry.get_breaker", return_value=breaker):
            with pytest.raises(RetryExhausted):
                always_fail()

        # Assert — called once per attempt
        assert breaker.record_failure.call_count == 2


# ---------------------------------------------------------------------------
# Tests: async retry
# ---------------------------------------------------------------------------


class TestRetryAsync:
    def test_async_successful_call_returns_immediately(self) -> None:
        """Async function that succeeds returns without retry."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=3,
            retryable=(TransientError,),
            jitter=False,
        )
        async def async_succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "async_ok"

        # Act
        result = asyncio.get_event_loop().run_until_complete(async_succeed())

        # Assert
        assert result == "async_ok"
        assert call_count == 1

    def test_async_retries_on_transient_failure(self) -> None:
        """Async function retries and eventually succeeds."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=5,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        async def async_flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("transient")
            return "recovered"

        # Act
        result = asyncio.get_event_loop().run_until_complete(async_flaky())

        # Assert
        assert result == "recovered"
        assert call_count == 3

    def test_async_raises_retry_exhausted(self) -> None:
        """Async function raises RetryExhausted after all attempts fail."""

        # Arrange
        @retry(
            max_attempts=2,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        async def async_always_fail() -> None:
            raise TransientError("async_boom")

        # Act & Assert
        with pytest.raises(RetryExhausted) as exc_info:
            asyncio.get_event_loop().run_until_complete(async_always_fail())

        assert exc_info.value.attempts == 2
        assert isinstance(exc_info.value.last_exception, TransientError)

    def test_async_non_retryable_not_retried(self) -> None:
        """Async: non-retryable exceptions propagate immediately."""
        # Arrange
        call_count = 0

        @retry(
            max_attempts=3,
            backoff_base=0.001,
            retryable=(TransientError,),
            jitter=False,
        )
        async def async_permanent_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise PermanentError("fatal")

        # Act & Assert
        with pytest.raises(PermanentError, match="fatal"):
            asyncio.get_event_loop().run_until_complete(async_permanent_fail())

        assert call_count == 1

    def test_async_circuit_breaker_open_skips(self) -> None:
        """Async: OPEN breaker causes immediate RetryExhausted."""
        # Arrange
        breaker = _make_open_breaker("async_open_test")

        @retry(
            max_attempts=3,
            retryable=(TransientError,),
            breaker_name="async_open_test",
            jitter=False,
        )
        async def async_guarded() -> str:
            return "should not reach"

        # Act & Assert
        with patch("aiswarm.resilience.retry.get_breaker", return_value=breaker):
            with pytest.raises(RetryExhausted) as exc_info:
                asyncio.get_event_loop().run_until_complete(async_guarded())

        assert exc_info.value.attempts == 0


# ---------------------------------------------------------------------------
# Tests: RetryExhausted exception
# ---------------------------------------------------------------------------


class TestRetryExhausted:
    def test_attributes(self) -> None:
        """RetryExhausted stores attempts and last_exception."""
        # Arrange
        cause = TransientError("root cause")

        # Act
        exc = RetryExhausted(5, cause)

        # Assert
        assert exc.attempts == 5
        assert exc.last_exception is cause
        assert "5 attempts" in str(exc)

    def test_is_an_exception(self) -> None:
        """RetryExhausted inherits from Exception."""
        assert issubclass(RetryExhausted, Exception)
