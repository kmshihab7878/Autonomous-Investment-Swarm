"""Plugin discovery and loading from entry points and directories.

Discovers plugins via:
1. Python entry points (``ais.plugins`` group) — for pip-installed plugins
2. Directory scanning — for local plugin development

Usage::

    from aiswarm.plugins.loader import discover_plugins, load_plugin

    plugins = discover_plugins()
    for plugin in plugins:
        plugin.on_load(config)
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from aiswarm.plugins.base import Plugin
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "ais.plugins"


def discover_entry_point_plugins() -> list[type[Plugin]]:
    """Discover plugins registered via Python entry points."""
    plugins: list[type[Plugin]] = []

    from importlib.metadata import entry_points

    try:
        eps: list[Any] = list(entry_points(group=ENTRY_POINT_GROUP))
    except TypeError:
        # Python < 3.12 fallback: entry_points() returns a dict
        all_eps = entry_points()
        eps = list(getattr(all_eps, "get", lambda *a: [])(ENTRY_POINT_GROUP, []))

    for ep in eps:
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Plugin) and cls is not Plugin:
                plugins.append(cls)
                logger.info(
                    "Discovered entry point plugin",
                    extra={"extra_json": {"name": ep.name, "class": cls.__name__}},
                )
        except Exception:
            logger.exception(
                "Failed to load entry point plugin",
                extra={"extra_json": {"name": ep.name}},
            )

    return plugins


def discover_directory_plugins(plugin_dir: str | Path) -> list[type[Plugin]]:
    """Discover plugins from Python files in a directory."""
    plugin_dir = Path(plugin_dir)
    if not plugin_dir.is_dir():
        return []

    plugins: list[type[Plugin]] = []
    for py_file in sorted(plugin_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue

        module_name = f"ais_plugin_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                    and hasattr(attr, "plugin_name")
                ):
                    plugins.append(attr)
                    logger.info(
                        "Discovered directory plugin",
                        extra={
                            "extra_json": {
                                "file": str(py_file),
                                "class": attr.__name__,
                                "name": attr.plugin_name,
                            }
                        },
                    )
        except Exception:
            logger.exception(
                "Failed to load plugin file",
                extra={"extra_json": {"file": str(py_file)}},
            )

    return plugins


def discover_plugins(
    plugin_dir: str | Path | None = None,
) -> list[type[Plugin]]:
    """Discover all plugins from entry points and optional directory."""
    plugins = discover_entry_point_plugins()
    if plugin_dir is not None:
        plugins.extend(discover_directory_plugins(plugin_dir))
    return plugins


def load_plugin(
    cls: type[Plugin],
    config: dict[str, Any] | None = None,
) -> Plugin:
    """Instantiate and initialize a plugin."""
    plugin = cls()
    plugin.on_load(config or {})
    logger.info(
        "Plugin loaded",
        extra={
            "extra_json": {
                "name": plugin.plugin_name,
                "type": plugin.plugin_type.value,
                "version": plugin.plugin_version,
            }
        },
    )
    return plugin
