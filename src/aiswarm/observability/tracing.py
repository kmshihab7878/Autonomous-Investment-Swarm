"""OpenTelemetry tracing initialization and span utilities.

Provides optional distributed tracing for the AIS trading loop.
When OpenTelemetry packages are not installed, all tracing functions
are no-ops — the system runs without any tracing overhead.

Usage::

    from aiswarm.observability.tracing import init_tracing, create_span

    init_tracing(service_name="ais-loop")

    with create_span("trading_cycle", attributes={"cycle": 42}):
        # ... trading logic ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

# Try optional OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

_tracer: Any = None
_initialized = False


def init_tracing(
    service_name: str = "ais",
    endpoint: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for trace identification.
        endpoint: OTLP exporter endpoint (e.g., "http://localhost:4317").
            If None, uses console exporter for development.

    Returns:
        True if tracing was initialized, False if OTel not available.
    """
    global _tracer, _initialized

    if not _HAS_OTEL:
        logger.info("OpenTelemetry not installed — tracing disabled")
        return False

    if _initialized:
        return True

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(
                "OTLP trace exporter configured",
                extra={"extra_json": {"endpoint": endpoint}},
            )
        except ImportError:
            logger.warning("OTLP exporter not available — using console")
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    _initialized = True

    logger.info(
        "OpenTelemetry tracing initialized",
        extra={"extra_json": {"service": service_name}},
    )
    return True


def get_tracer() -> Any:
    """Return the global tracer (or None if not initialized)."""
    return _tracer


@contextmanager
def create_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Create a trace span. No-op if tracing is not initialized.

    Usage::

        with create_span("risk_validation", {"order_id": "o1"}):
            result = risk_engine.validate(order, snapshot)
    """
    if _tracer is not None and _HAS_OTEL:
        with _tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
    else:
        yield None


def record_exception(span: Any, exception: Exception) -> None:
    """Record an exception on a span (no-op if span is None)."""
    if span is not None and _HAS_OTEL:
        span.record_exception(exception)
        span.set_status(trace.StatusCode.ERROR, str(exception))
