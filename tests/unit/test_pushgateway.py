"""Tests for Prometheus pushgateway support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from aiswarm.monitoring.metrics import push_metrics


class TestPushMetrics:
    def test_push_success(self) -> None:
        with patch("aiswarm.monitoring.metrics.push_to_gateway") as mock_push:
            result = push_metrics("http://pushgateway:9091", job="test")
            assert result is True
            mock_push.assert_called_once()
            args, kwargs = mock_push.call_args
            assert args[0] == "http://pushgateway:9091"
            assert kwargs["job"] == "test"

    def test_push_with_grouping_key(self) -> None:
        with patch("aiswarm.monitoring.metrics.push_to_gateway") as mock_push:
            result = push_metrics(
                "http://gw:9091",
                job="ais-backtest",
                grouping_key={"strategy": "momentum", "symbol": "BTCUSDT"},
            )
            assert result is True
            _, kwargs = mock_push.call_args
            assert kwargs["grouping_key"] == {"strategy": "momentum", "symbol": "BTCUSDT"}

    def test_push_failure_returns_false(self) -> None:
        with patch(
            "aiswarm.monitoring.metrics.push_to_gateway",
            side_effect=ConnectionError("refused"),
        ):
            result = push_metrics("http://bad:9091", job="test")
            assert result is False

    def test_push_timeout_returns_false(self) -> None:
        with patch(
            "aiswarm.monitoring.metrics.push_to_gateway",
            side_effect=TimeoutError("timed out"),
        ):
            result = push_metrics("http://slow:9091", job="test")
            assert result is False

    def test_push_uses_default_registry(self) -> None:
        with patch("aiswarm.monitoring.metrics.push_to_gateway") as mock_push:
            from prometheus_client.registry import REGISTRY

            push_metrics("http://gw:9091", job="test")
            _, kwargs = mock_push.call_args
            assert kwargs["registry"] is REGISTRY

    def test_empty_grouping_key_default(self) -> None:
        with patch("aiswarm.monitoring.metrics.push_to_gateway") as mock_push:
            push_metrics("http://gw:9091", job="test")
            _, kwargs = mock_push.call_args
            assert kwargs["grouping_key"] == {}

    def test_push_default_job_name(self) -> None:
        with patch("aiswarm.monitoring.metrics.push_to_gateway") as mock_push:
            push_metrics("http://gw:9091")
            _, kwargs = mock_push.call_args
            assert kwargs["job"] == "ais"


class TestBacktestPushgatewayIntegration:
    """Verify that BacktestEngine pushes metrics when AIS_PUSHGATEWAY_URL is set."""

    def test_backtest_pushes_when_url_set(self) -> None:
        from datetime import datetime

        from aiswarm.backtest.engine import OHLCV, BacktestEngine

        candles = [
            OHLCV(datetime(2024, 1, 1), 100, 105, 95, 102, 1000),
            OHLCV(datetime(2024, 1, 2), 102, 110, 100, 108, 1200),
            OHLCV(datetime(2024, 1, 3), 108, 112, 106, 110, 900),
        ]

        gen = MagicMock()
        gen.generate_signal.return_value = None

        with patch.dict("os.environ", {"AIS_PUSHGATEWAY_URL": "http://gw:9091"}):
            with patch("aiswarm.backtest.engine.push_metrics") as mock_push:
                engine = BacktestEngine()
                engine.run("test_strat", gen, "BTCUSDT", candles)
                mock_push.assert_called_once_with(
                    "http://gw:9091",
                    job="ais-backtest",
                    grouping_key={"strategy": "test_strat", "symbol": "BTCUSDT"},
                )

    def test_backtest_skips_push_when_no_url(self) -> None:
        from datetime import datetime

        from aiswarm.backtest.engine import OHLCV, BacktestEngine

        candles = [
            OHLCV(datetime(2024, 1, 1), 100, 105, 95, 102, 1000),
            OHLCV(datetime(2024, 1, 2), 102, 110, 100, 108, 1200),
            OHLCV(datetime(2024, 1, 3), 108, 112, 106, 110, 900),
        ]

        gen = MagicMock()
        gen.generate_signal.return_value = None

        with patch.dict("os.environ", {}, clear=False):
            with patch("aiswarm.backtest.engine.push_metrics") as mock_push:
                engine = BacktestEngine()
                engine.run("test_strat", gen, "BTCUSDT", candles)
                mock_push.assert_not_called()
