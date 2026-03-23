"""Example AIS plugin: Simple Moving Average Crossover.

This demonstrates how to build a plugin-based strategy for AIS.
Drop this file in a plugin directory or install via entry points.

Usage:
    # As a directory plugin:
    python -m aiswarm --mode paper --plugin-dir ./examples/plugins/

    # Or register via pyproject.toml entry points:
    # [project.entry-points."ais.plugins"]
    # sma_cross = "my_package.sma_plugin:SMACrossPlugin"
"""

from __future__ import annotations

from typing import Any

from aiswarm.plugins.base import Plugin, PluginType


class SMACrossPlugin(Plugin):
    """Simple Moving Average crossover plugin.

    Generates a buy signal when the fast SMA crosses above the slow SMA,
    and a sell signal when it crosses below.
    """

    plugin_type = PluginType.STRATEGY
    plugin_name = "sma_cross_plugin"
    plugin_version = "1.0.0"
    plugin_description = "Simple SMA crossover strategy (example plugin)"
    plugin_author = "AIS Examples"

    def __init__(self) -> None:
        self.fast_period = 10
        self.slow_period = 30

    def on_load(self, config: dict[str, Any]) -> None:
        """Configure periods from plugin config."""
        self.fast_period = config.get("fast_period", 10)
        self.slow_period = config.get("slow_period", 30)

    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Check for SMA crossover and return signal dict."""
        prices = context.get("close_prices", [])
        if len(prices) < self.slow_period:
            return None

        fast_sma = sum(prices[-self.fast_period :]) / self.fast_period
        slow_sma = sum(prices[-self.slow_period :]) / self.slow_period

        if fast_sma > slow_sma:
            return {
                "direction": 1,
                "confidence": 0.55,
                "reason": f"Fast SMA ({fast_sma:.2f}) > Slow SMA ({slow_sma:.2f})",
            }
        elif fast_sma < slow_sma:
            return {
                "direction": -1,
                "confidence": 0.55,
                "reason": f"Fast SMA ({fast_sma:.2f}) < Slow SMA ({slow_sma:.2f})",
            }
        return None
