"""Tests for circuit breaker Prometheus metric instrumentation."""

from __future__ import annotations

import time

from aiswarm.monitoring import metrics as m
from aiswarm.resilience.circuit_breaker import CircuitBreaker, CircuitState


def _get_gauge_value(gauge, labels: dict[str, str]) -> float:
    """Read the current value of a labelled Gauge."""
    return gauge.labels(**labels)._value.get()


def _get_counter_value(counter, labels: dict[str, str]) -> float:
    """Read the current value of a labelled Counter."""
    return counter.labels(**labels)._value.get()


class TestCircuitBreakerMetricsStateGauge:
    """Verify the CIRCUIT_BREAKER_STATE gauge reflects breaker state."""

    def test_initial_state_is_closed(self) -> None:
        cb = CircuitBreaker("gauge_init", failure_threshold=3)
        value = _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_init"})
        assert value == 0  # CLOSED=0
        assert cb.state == CircuitState.CLOSED

    def test_gauge_set_to_open_after_threshold(self) -> None:
        cb = CircuitBreaker("gauge_open", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        value = _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_open"})
        assert value == 1  # OPEN=1
        assert cb.state == CircuitState.OPEN

    def test_gauge_set_to_half_open_after_recovery_timeout(self) -> None:
        cb = CircuitBreaker("gauge_half", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_half"}) == 1
        time.sleep(0.06)
        # Access state to trigger the transition
        assert cb.state == CircuitState.HALF_OPEN
        value = _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_half"})
        assert value == 2  # HALF_OPEN=2

    def test_gauge_returns_to_closed_on_recovery(self) -> None:
        cb = CircuitBreaker("gauge_recover", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # triggers half-open transition
        cb.record_success()  # closes breaker
        value = _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_recover"})
        assert value == 0  # CLOSED=0

    def test_gauge_returns_to_closed_on_reset(self) -> None:
        cb = CircuitBreaker("gauge_reset", failure_threshold=1)
        cb.record_failure()
        assert _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_reset"}) == 1
        cb.reset()
        assert _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "gauge_reset"}) == 0


class TestCircuitBreakerMetricsFailures:
    """Verify the CIRCUIT_BREAKER_FAILURES counter increments on failure."""

    def test_failures_increment(self) -> None:
        cb = CircuitBreaker("fail_count", failure_threshold=10)
        before = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_count"})
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        after = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_count"})
        assert after - before == 3

    def test_failures_use_correct_label(self) -> None:
        cb_a = CircuitBreaker("fail_a", failure_threshold=10)
        cb_b = CircuitBreaker("fail_b", failure_threshold=10)
        before_a = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_a"})
        before_b = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_b"})
        cb_a.record_failure()
        cb_a.record_failure()
        cb_b.record_failure()
        assert _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_a"}) - before_a == 2
        assert _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "fail_b"}) - before_b == 1


class TestCircuitBreakerMetricsSuccesses:
    """Verify the CIRCUIT_BREAKER_SUCCESSES counter increments on success."""

    def test_successes_increment(self) -> None:
        cb = CircuitBreaker("succ_count", failure_threshold=10)
        before = _get_counter_value(m.CIRCUIT_BREAKER_SUCCESSES, {"name": "succ_count"})
        cb.record_success()
        cb.record_success()
        after = _get_counter_value(m.CIRCUIT_BREAKER_SUCCESSES, {"name": "succ_count"})
        assert after - before == 2


class TestCircuitBreakerMetricsRejections:
    """Verify the CIRCUIT_BREAKER_REJECTIONS counter increments when open."""

    def test_rejections_counted_when_open(self) -> None:
        cb = CircuitBreaker("rej_count", failure_threshold=1)
        cb.record_failure()  # opens the breaker
        before = _get_counter_value(m.CIRCUIT_BREAKER_REJECTIONS, {"name": "rej_count"})
        cb.allow_request()  # rejected
        cb.allow_request()  # rejected
        cb.allow_request()  # rejected
        after = _get_counter_value(m.CIRCUIT_BREAKER_REJECTIONS, {"name": "rej_count"})
        assert after - before == 3

    def test_no_rejections_when_closed(self) -> None:
        cb = CircuitBreaker("rej_closed", failure_threshold=10)
        before = _get_counter_value(m.CIRCUIT_BREAKER_REJECTIONS, {"name": "rej_closed"})
        cb.allow_request()
        cb.allow_request()
        after = _get_counter_value(m.CIRCUIT_BREAKER_REJECTIONS, {"name": "rej_closed"})
        assert after - before == 0


class TestCircuitBreakerMetricsTransitions:
    """Verify the CIRCUIT_BREAKER_TRANSITIONS counter tracks state changes."""

    def test_closed_to_open_transition(self) -> None:
        cb = CircuitBreaker("trans_co", failure_threshold=2)
        labels = {"name": "trans_co", "from_state": "closed", "to_state": "open"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        cb.record_failure()
        cb.record_failure()  # triggers CLOSED -> OPEN
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 1

    def test_open_to_half_open_transition(self) -> None:
        cb = CircuitBreaker("trans_oh", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()  # CLOSED -> OPEN
        labels = {"name": "trans_oh", "from_state": "open", "to_state": "half_open"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        time.sleep(0.06)
        _ = cb.state  # triggers OPEN -> HALF_OPEN
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 1

    def test_half_open_to_closed_transition(self) -> None:
        cb = CircuitBreaker("trans_hc", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # triggers half-open
        labels = {"name": "trans_hc", "from_state": "half_open", "to_state": "closed"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        cb.record_success()  # HALF_OPEN -> CLOSED
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 1

    def test_half_open_to_open_transition_on_failure(self) -> None:
        cb = CircuitBreaker("trans_ho", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # triggers half-open
        labels = {"name": "trans_ho", "from_state": "half_open", "to_state": "open"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        cb.record_failure()  # HALF_OPEN -> OPEN
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 1

    def test_reset_records_transition(self) -> None:
        cb = CircuitBreaker("trans_reset", failure_threshold=1)
        cb.record_failure()  # CLOSED -> OPEN
        labels = {"name": "trans_reset", "from_state": "open", "to_state": "closed"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        cb.reset()
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 1

    def test_reset_no_transition_when_already_closed(self) -> None:
        cb = CircuitBreaker("trans_noop", failure_threshold=10)
        labels = {"name": "trans_noop", "from_state": "closed", "to_state": "closed"}
        before = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        cb.reset()  # already closed — no transition
        after = _get_counter_value(m.CIRCUIT_BREAKER_TRANSITIONS, labels)
        assert after - before == 0


class TestCircuitBreakerMetricLabels:
    """Verify metric labels include the breaker name correctly."""

    def test_different_breakers_have_independent_metrics(self) -> None:
        cb_x = CircuitBreaker("label_x", failure_threshold=1)
        cb_y = CircuitBreaker("label_y", failure_threshold=1)

        before_x = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "label_x"})
        before_y = _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "label_y"})

        cb_x.record_failure()
        cb_x.record_failure()
        cb_y.record_failure()

        assert _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "label_x"}) - before_x == 2
        assert _get_counter_value(m.CIRCUIT_BREAKER_FAILURES, {"name": "label_y"}) - before_y == 1

        # State gauges are also per-name
        assert _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "label_x"}) == 1  # OPEN
        assert _get_gauge_value(m.CIRCUIT_BREAKER_STATE, {"name": "label_y"}) == 1  # OPEN
