"""Tests for the plugin system: base, loader, and manager."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


from aiswarm.plugins.base import Plugin, PluginInfo, PluginType
from aiswarm.plugins.loader import discover_directory_plugins, load_plugin
from aiswarm.plugins.manager import PluginManager


# --- Test plugin implementations ---


class _MockStrategy(Plugin):
    plugin_type = PluginType.STRATEGY
    plugin_name = "mock_strategy"
    plugin_version = "1.0.0"
    plugin_description = "A mock strategy for testing"
    plugin_author = "test"

    def __init__(self) -> None:
        self.loaded = False
        self.started = False
        self.shutdown_called = False
        self.cycle_count = 0
        self.config: dict[str, Any] = {}

    def on_load(self, config: dict[str, Any]) -> None:
        self.loaded = True
        self.config = config

    def on_start(self) -> None:
        self.started = True

    def on_shutdown(self) -> None:
        self.shutdown_called = True

    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        self.cycle_count += 1
        return {"signal": "buy", "symbol": context.get("symbol", "BTCUSDT")}


class _MockDataSource(Plugin):
    plugin_type = PluginType.DATA_SOURCE
    plugin_name = "mock_data_source"
    plugin_version = "0.1.0"
    plugin_description = "Mock data source"

    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return {"enriched": True}


class _MockRiskGuard(Plugin):
    plugin_type = PluginType.RISK_GUARD
    plugin_name = "mock_risk_guard"
    plugin_version = "0.1.0"
    plugin_description = "Mock risk guard"

    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return {"approved": True, "reason": "ok"}


class _NonePlugin(Plugin):
    plugin_type = PluginType.INTEGRATION
    plugin_name = "none_plugin"
    plugin_version = "0.1.0"
    plugin_description = "Returns None on cycle"

    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        return None


# --- Tests: Plugin base ---


class TestPluginBase:
    def test_info_returns_metadata(self) -> None:
        plugin = _MockStrategy()
        info = plugin.info()
        assert isinstance(info, PluginInfo)
        assert info.name == "mock_strategy"
        assert info.version == "1.0.0"
        assert info.plugin_type == PluginType.STRATEGY

    def test_lifecycle_hooks(self) -> None:
        plugin = _MockStrategy()
        assert not plugin.loaded

        plugin.on_load({"key": "val"})
        assert plugin.loaded
        assert plugin.config == {"key": "val"}

        plugin.on_start()
        assert plugin.started

        result = plugin.on_cycle({"symbol": "ETHUSDT"})
        assert result is not None
        assert result["symbol"] == "ETHUSDT"
        assert plugin.cycle_count == 1

        plugin.on_shutdown()
        assert plugin.shutdown_called


# --- Tests: Plugin loader ---


class TestPluginLoader:
    def test_load_plugin_calls_on_load(self) -> None:
        plugin = load_plugin(_MockStrategy, {"threshold": 0.5})
        assert plugin.loaded
        assert plugin.config["threshold"] == 0.5

    def test_load_plugin_default_config(self) -> None:
        plugin = load_plugin(_MockStrategy)
        assert plugin.loaded
        assert plugin.config == {}

    def test_discover_directory_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plugins = discover_directory_plugins(tmpdir)
            assert plugins == []

    def test_discover_directory_nonexistent(self) -> None:
        plugins = discover_directory_plugins("/nonexistent/path")
        assert plugins == []

    def test_discover_directory_finds_plugin(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_file = Path(tmpdir) / "my_plugin.py"
            plugin_file.write_text(
                "from aiswarm.plugins.base import Plugin, PluginType\n"
                "class TestPlugin(Plugin):\n"
                "    plugin_type = PluginType.STRATEGY\n"
                "    plugin_name = 'dir_test'\n"
                "    plugin_version = '0.1.0'\n"
                "    plugin_description = 'test'\n"
                "    def on_cycle(self, context):\n"
                "        return None\n"
            )
            plugins = discover_directory_plugins(tmpdir)
            assert len(plugins) == 1
            assert plugins[0].plugin_name == "dir_test"

    def test_discover_skips_underscored_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "__init__.py").write_text("")
            (Path(tmpdir) / "_private.py").write_text("x = 1")
            plugins = discover_directory_plugins(tmpdir)
            assert plugins == []


# --- Tests: Plugin manager ---


class TestPluginManager:
    def test_register_and_list(self) -> None:
        mgr = PluginManager()
        mgr.register(_MockStrategy())
        mgr.register(_MockDataSource())

        assert len(mgr.plugins) == 2
        assert len(mgr.strategy_plugins) == 1
        assert len(mgr.data_source_plugins) == 1

    def test_start_all(self) -> None:
        mgr = PluginManager()
        s = _MockStrategy()
        mgr.register(s)
        mgr.start_all()
        assert s.started

    def test_shutdown_all(self) -> None:
        mgr = PluginManager()
        s = _MockStrategy()
        mgr.register(s)
        mgr.shutdown_all()
        assert s.shutdown_called

    def test_run_cycle_filters_by_type(self) -> None:
        mgr = PluginManager()
        mgr.register(_MockStrategy())
        mgr.register(_MockDataSource())
        mgr.register(_MockRiskGuard())

        results = mgr.run_cycle(PluginType.STRATEGY, {"symbol": "BTCUSDT"})
        assert len(results) == 1
        assert results[0]["signal"] == "buy"

    def test_run_cycle_skips_none_results(self) -> None:
        mgr = PluginManager()
        mgr.register(_NonePlugin())
        results = mgr.run_cycle(PluginType.INTEGRATION, {})
        assert results == []

    def test_list_plugins_returns_info(self) -> None:
        mgr = PluginManager()
        mgr.register(_MockStrategy())
        infos = mgr.list_plugins()
        assert len(infos) == 1
        assert infos[0].name == "mock_strategy"

    def test_type_filtered_properties(self) -> None:
        mgr = PluginManager()
        mgr.register(_MockStrategy())
        mgr.register(_MockDataSource())
        mgr.register(_MockRiskGuard())
        mgr.register(_NonePlugin())

        assert len(mgr.risk_guard_plugins) == 1
        assert len(mgr.integration_plugins) == 1
