"""Tests for G-006: Prometheus metrics are defined and accessible."""

from __future__ import annotations


class TestMetricsDefinitions:
    def test_all_portfolio_metrics_exist(self) -> None:
        from aiswarm.monitoring.metrics import (
            PNL_GAUGE,
            EXPOSURE_GAUGE,
            NAV_GAUGE,
            DRAWDOWN_GAUGE,
            LEVERAGE_GAUGE,
        )

        # All should be Gauge instances
        for gauge in (PNL_GAUGE, EXPOSURE_GAUGE, NAV_GAUGE, DRAWDOWN_GAUGE, LEVERAGE_GAUGE):
            assert hasattr(gauge, "set")

    def test_all_agent_metrics_exist(self) -> None:
        from aiswarm.monitoring.metrics import (
            AGENT_LATENCY,
            SIGNALS_GENERATED,
            SIGNALS_APPROVED,
            SIGNALS_REJECTED,
        )

        assert hasattr(AGENT_LATENCY, "observe")
        assert hasattr(SIGNALS_GENERATED, "inc")
        assert hasattr(SIGNALS_APPROVED, "inc")
        assert hasattr(SIGNALS_REJECTED, "inc")

    def test_all_execution_metrics_exist(self) -> None:
        from aiswarm.monitoring.metrics import (
            ORDERS_SUBMITTED,
            ORDERS_FILLED,
            PAPER_FILLS,
        )

        for counter in (ORDERS_SUBMITTED, ORDERS_FILLED, PAPER_FILLS):
            assert hasattr(counter, "inc")

    def test_all_loop_metrics_exist(self) -> None:
        from aiswarm.monitoring.metrics import (
            LOOP_CYCLES,
            LOOP_CYCLE_DURATION,
            LOOP_ERRORS,
            LOOP_HEARTBEAT,
            LOOP_HALTED,
        )

        assert hasattr(LOOP_CYCLES, "inc")
        assert hasattr(LOOP_CYCLE_DURATION, "observe")
        assert hasattr(LOOP_ERRORS, "inc")
        assert hasattr(LOOP_HEARTBEAT, "set")
        assert hasattr(LOOP_HALTED, "set")

    def test_mandate_metrics_exist(self) -> None:
        from aiswarm.monitoring.metrics import (
            MANDATE_PNL,
            MANDATE_EXPOSURE,
            MANDATE_DRAWDOWN,
        )

        assert hasattr(MANDATE_PNL, "labels")
        assert hasattr(MANDATE_EXPOSURE, "labels")
        assert hasattr(MANDATE_DRAWDOWN, "labels")

    def test_metric_count(self) -> None:
        """Verify we have at least 27 Prometheus metrics defined."""
        from aiswarm.monitoring import metrics as m

        metric_attrs = [attr for attr in dir(m) if attr.isupper() and not attr.startswith("_")]
        assert len(metric_attrs) >= 27, (
            f"Expected >=27 metrics, found {len(metric_attrs)}: {metric_attrs}"
        )
