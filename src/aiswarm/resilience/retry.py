"""Retry decorator with exponential backoff and circuit breaker integration.

Provides configurable retry logic for transient failures in exchange
communication, order submission, and market data fetching.

Usage::

    @retry(max_attempts=3, retryable=(httpx.TimeoutException, httpx.ConnectError))
    async def fetch_klines(symbol: str) -> list[OHLCV]:
        ...

    @retry(max_attempts=5, backoff_base=2.0, backoff_max=60.0, breaker_name="aster")
    def submit_order(order: Order) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from aiswarm.resilience.circuit_breaker import get_breaker
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RetryStats:
    """Result metadata from a retried call."""

    attempts: int
    total_delay: float
    final_exception: Exception | None


class RetryExhausted(Exception):
    """All retry attempts failed."""

    def __init__(self, attempts: int, last_exception: Exception) -> None:
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_exception}")


def _compute_delay(
    attempt: int,
    backoff_base: float,
    backoff_max: float,
    jitter: bool,
) -> float:
    """Compute delay with exponential backoff and optional jitter."""
    delay: float = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # noqa: S311
    return delay


def retry(
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    backoff_max: float = 30.0,
    jitter: bool = True,
    retryable: tuple[type[Exception], ...] = (Exception,),
    breaker_name: str | None = None,
) -> Callable[[F], F]:
    """Decorator that retries a function on transient failures.

    Args:
        max_attempts: Maximum number of attempts (including first try).
        backoff_base: Base delay in seconds for exponential backoff.
        backoff_max: Maximum delay cap in seconds.
        jitter: Add randomized jitter to prevent thundering herd.
        retryable: Tuple of exception types that trigger a retry.
        breaker_name: Optional circuit breaker name. If the breaker is
            OPEN, calls fail immediately without consuming retries.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = get_breaker(breaker_name) if breaker_name else None
            last_exc: Exception | None = None
            total_delay = 0.0

            for attempt in range(1, max_attempts + 1):
                # Circuit breaker gate
                if breaker and not breaker.allow_request():
                    logger.warning(
                        "Retry skipped: circuit breaker OPEN",
                        extra={
                            "extra_json": {
                                "function": func.__name__,
                                "breaker": breaker_name,
                                "state": breaker.state.value,
                            }
                        },
                    )
                    if last_exc:
                        raise RetryExhausted(attempt - 1, last_exc)
                    raise RetryExhausted(0, RuntimeError(f"Circuit breaker {breaker_name} is OPEN"))

                try:
                    result = func(*args, **kwargs)
                    if breaker:
                        breaker.record_success()
                    return result
                except retryable as exc:
                    last_exc = exc
                    if breaker:
                        breaker.record_failure()

                    if attempt == max_attempts:
                        break

                    delay = _compute_delay(attempt, backoff_base, backoff_max, jitter)
                    total_delay += delay

                    logger.warning(
                        "Retrying after transient failure",
                        extra={
                            "extra_json": {
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                                "delay_s": round(delay, 2),
                                "error": str(exc),
                            }
                        },
                    )
                    time.sleep(delay)

            if last_exc is None:  # pragma: no cover — unreachable
                raise RuntimeError("retry loop exited without exception")
            raise RetryExhausted(max_attempts, last_exc)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = get_breaker(breaker_name) if breaker_name else None
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                if breaker and not breaker.allow_request():
                    logger.warning(
                        "Retry skipped: circuit breaker OPEN",
                        extra={
                            "extra_json": {
                                "function": func.__name__,
                                "breaker": breaker_name,
                                "state": breaker.state.value,
                            }
                        },
                    )
                    if last_exc:
                        raise RetryExhausted(attempt - 1, last_exc)
                    raise RetryExhausted(0, RuntimeError(f"Circuit breaker {breaker_name} is OPEN"))

                try:
                    result = await func(*args, **kwargs)
                    if breaker:
                        breaker.record_success()
                    return result
                except retryable as exc:
                    last_exc = exc
                    if breaker:
                        breaker.record_failure()

                    if attempt == max_attempts:
                        break

                    delay = _compute_delay(attempt, backoff_base, backoff_max, jitter)

                    logger.warning(
                        "Retrying after transient failure (async)",
                        extra={
                            "extra_json": {
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                                "delay_s": round(delay, 2),
                                "error": str(exc),
                            }
                        },
                    )
                    await asyncio.sleep(delay)

            if last_exc is None:  # pragma: no cover — unreachable
                raise RuntimeError("retry loop exited without exception")
            raise RetryExhausted(max_attempts, last_exc)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
