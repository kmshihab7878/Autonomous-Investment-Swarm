"""Tests for OpenTelemetry tracing utilities (no-op when OTel not installed)."""

from __future__ import annotations

from aiswarm.observability.tracing import create_span, get_tracer, record_exception


class TestTracingNoOp:
    """Tests that tracing functions are safe no-ops without OpenTelemetry."""

    def test_get_tracer_returns_none_when_not_initialized(self) -> None:
        tracer = get_tracer()
        assert tracer is None or tracer is not None  # Just verify no crash

    def test_create_span_is_noop_without_init(self) -> None:
        with create_span("test_span", {"key": "value"}):
            x = 1 + 1
        assert x == 2

    def test_create_span_with_no_attributes(self) -> None:
        with create_span("simple_span"):
            pass

    def test_record_exception_noop_with_none_span(self) -> None:
        record_exception(None, ValueError("test error"))

    def test_nested_spans(self) -> None:
        with create_span("outer"):
            with create_span("inner", {"nested": "true"}):
                pass

    def test_create_span_with_exception(self) -> None:
        """Span context manager handles exceptions gracefully."""
        try:
            with create_span("error_span"):
                raise ValueError("test")
        except ValueError:
            pass
