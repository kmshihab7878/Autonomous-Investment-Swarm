"""Circuit breaker pattern for external service calls.

Prevents cascading failures by halting calls to a service after
repeated failures. Transitions through three states:

  CLOSED  -> call passes through normally
  OPEN    -> calls fail immediately (service is assumed down)
  HALF_OPEN -> one probe call allowed to test recovery

After ``failure_threshold`` consecutive failures the breaker opens.
After ``recovery_timeout`` seconds it transitions to half-open.
A successful half-open call closes the breaker; a failure re-opens it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStats:
    """Observable statistics for a circuit breaker."""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    total_calls: int
    total_rejections: int


_STATE_VALUE = {
    CircuitState.CLOSED: 0,
    CircuitState.OPEN: 1,
    CircuitState.HALF_OPEN: 2,
}


class CircuitBreaker:
    """Thread-safe circuit breaker for external service calls.

    Usage::

        breaker = CircuitBreaker("aster_api", failure_threshold=5, recovery_timeout=30)

        if breaker.allow_request():
            try:
                result = call_external_service()
                breaker.record_success()
            except Exception:
                breaker.record_failure()
        else:
            # Service is down — skip or use fallback
            ...
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._total_calls = 0
        self._total_rejections = 0
        self._lock = threading.Lock()

        # Set initial state gauge
        self._emit_state_gauge(CircuitState.CLOSED)

    # --- Prometheus helpers (lazy import to avoid circular deps) ----------

    def _emit_state_gauge(self, state: CircuitState) -> None:
        """Update the Prometheus state gauge for this breaker."""
        from aiswarm.monitoring import metrics as m

        m.CIRCUIT_BREAKER_STATE.labels(name=self.name).set(_STATE_VALUE[state])

    def _emit_transition(self, from_state: CircuitState, to_state: CircuitState) -> None:
        """Record a state transition in Prometheus."""
        from aiswarm.monitoring import metrics as m

        self._emit_state_gauge(to_state)
        m.CIRCUIT_BREAKER_TRANSITIONS.labels(
            name=self.name,
            from_state=from_state.value,
            to_state=to_state.value,
        ).inc()

    # --- Public API -------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through."""
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                return True

            # OPEN
            self._total_rejections += 1
            from aiswarm.monitoring import metrics as m

            m.CIRCUIT_BREAKER_REJECTIONS.labels(name=self.name).inc()
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        from aiswarm.monitoring import metrics as m

        with self._lock:
            self._success_count += 1
            self._total_calls += 1
            self._last_success_time = time.monotonic()
            m.CIRCUIT_BREAKER_SUCCESSES.labels(name=self.name).inc()

            if self._state == CircuitState.HALF_OPEN:
                self._emit_transition(CircuitState.HALF_OPEN, CircuitState.CLOSED)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(
                    "Circuit breaker CLOSED (recovered)",
                    extra={"extra_json": {"breaker": self.name}},
                )

    def record_failure(self) -> None:
        """Record a failed call."""
        from aiswarm.monitoring import metrics as m

        with self._lock:
            self._failure_count += 1
            self._total_calls += 1
            self._last_failure_time = time.monotonic()
            m.CIRCUIT_BREAKER_FAILURES.labels(name=self.name).inc()

            if self._state == CircuitState.HALF_OPEN:
                self._emit_transition(CircuitState.HALF_OPEN, CircuitState.OPEN)
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker re-OPENED (half-open probe failed)",
                    extra={"extra_json": {"breaker": self.name}},
                )
            elif (
                self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold
            ):
                self._emit_transition(CircuitState.CLOSED, CircuitState.OPEN)
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker OPENED",
                    extra={
                        "extra_json": {
                            "breaker": self.name,
                            "failures": self._failure_count,
                            "threshold": self.failure_threshold,
                        }
                    },
                )

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            if old_state != CircuitState.CLOSED:
                self._emit_transition(old_state, CircuitState.CLOSED)
            logger.info(
                "Circuit breaker manually reset",
                extra={"extra_json": {"breaker": self.name}},
            )

    def stats(self) -> CircuitStats:
        """Return observable statistics snapshot."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return CircuitStats(
                name=self.name,
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                last_failure_time=self._last_failure_time,
                last_success_time=self._last_success_time,
                total_calls=self._total_calls,
                total_rejections=self._total_rejections,
            )

    def _maybe_transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN after recovery timeout."""
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and (time.monotonic() - self._last_failure_time) >= self.recovery_timeout
        ):
            self._emit_transition(CircuitState.OPEN, CircuitState.HALF_OPEN)
            self._state = CircuitState.HALF_OPEN
            logger.info(
                "Circuit breaker HALF_OPEN (recovery timeout elapsed)",
                extra={"extra_json": {"breaker": self.name}},
            )


# --- Convenience: per-service breaker registry ---

_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Get or create a named circuit breaker (singleton per name)."""
    with _registry_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _breakers[name]


def all_breaker_stats() -> list[CircuitStats]:
    """Return stats for all registered breakers."""
    with _registry_lock:
        return [b.stats() for b in _breakers.values()]
