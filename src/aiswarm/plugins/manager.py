"""Plugin lifecycle manager.

Manages plugin discovery, loading, and lifecycle hooks for the
trading loop integration.

Usage::

    manager = PluginManager()
    manager.discover_and_load(config)
    manager.start_all()

    # In trading loop:
    for plugin in manager.strategy_plugins:
        result = plugin.on_cycle(context)
"""

from __future__ import annotations

from typing import Any

from aiswarm.plugins.base import Plugin, PluginInfo, PluginType
from aiswarm.plugins.loader import discover_plugins, load_plugin
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class PluginManager:
    """Manages plugin lifecycle: discover, load, start, cycle, shutdown."""

    def __init__(self) -> None:
        self._plugins: list[Plugin] = []

    @property
    def plugins(self) -> list[Plugin]:
        return list(self._plugins)

    @property
    def strategy_plugins(self) -> list[Plugin]:
        return [p for p in self._plugins if p.plugin_type == PluginType.STRATEGY]

    @property
    def data_source_plugins(self) -> list[Plugin]:
        return [p for p in self._plugins if p.plugin_type == PluginType.DATA_SOURCE]

    @property
    def risk_guard_plugins(self) -> list[Plugin]:
        return [p for p in self._plugins if p.plugin_type == PluginType.RISK_GUARD]

    @property
    def integration_plugins(self) -> list[Plugin]:
        return [p for p in self._plugins if p.plugin_type == PluginType.INTEGRATION]

    def list_plugins(self) -> list[PluginInfo]:
        """Return metadata for all loaded plugins."""
        return [p.info() for p in self._plugins]

    def discover_and_load(
        self,
        config: dict[str, Any] | None = None,
        plugin_dir: str | None = None,
    ) -> int:
        """Discover and load all available plugins.

        Returns the number of plugins loaded.
        """
        config = config or {}
        classes = discover_plugins(plugin_dir=plugin_dir)

        for cls in classes:
            try:
                plugin_config = config.get(cls.plugin_name, {})
                plugin = load_plugin(cls, plugin_config)
                self._plugins.append(plugin)
            except Exception:
                logger.exception(
                    "Failed to load plugin",
                    extra={"extra_json": {"class": cls.__name__}},
                )

        logger.info(
            "Plugin discovery complete",
            extra={
                "extra_json": {
                    "total": len(self._plugins),
                    "strategies": len(self.strategy_plugins),
                    "data_sources": len(self.data_source_plugins),
                    "risk_guards": len(self.risk_guard_plugins),
                    "integrations": len(self.integration_plugins),
                }
            },
        )
        return len(self._plugins)

    def register(self, plugin: Plugin) -> None:
        """Manually register a pre-loaded plugin."""
        self._plugins.append(plugin)

    def start_all(self) -> None:
        """Call on_start for all loaded plugins."""
        for plugin in self._plugins:
            try:
                plugin.on_start()
            except Exception:
                logger.exception(
                    "Plugin start failed",
                    extra={"extra_json": {"name": plugin.plugin_name}},
                )

    def shutdown_all(self) -> None:
        """Call on_shutdown for all loaded plugins."""
        for plugin in self._plugins:
            try:
                plugin.on_shutdown()
            except Exception:
                logger.exception(
                    "Plugin shutdown failed",
                    extra={"extra_json": {"name": plugin.plugin_name}},
                )

    def run_cycle(
        self,
        plugin_type: PluginType,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run on_cycle for all plugins of a given type.

        Returns list of non-None results.
        """
        results: list[dict[str, Any]] = []
        plugins = [p for p in self._plugins if p.plugin_type == plugin_type]

        for plugin in plugins:
            try:
                result = plugin.on_cycle(context)
                if result is not None:
                    results.append(result)
            except Exception:
                logger.exception(
                    "Plugin cycle error",
                    extra={"extra_json": {"name": plugin.plugin_name}},
                )

        return results
