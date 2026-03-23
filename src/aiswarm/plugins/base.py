"""Plugin base classes and lifecycle hooks.

Plugins extend AIS with custom strategies, data sources, risk guards,
and integrations. Each plugin type has specific hooks called at
different points in the trading loop lifecycle.

Usage::

    from aiswarm.plugins.base import Plugin, PluginType

    class MyStrategy(Plugin):
        plugin_type = PluginType.STRATEGY
        plugin_name = "my_awesome_strategy"
        plugin_version = "1.0.0"

        def on_load(self, config: dict) -> None:
            self.threshold = config.get("threshold", 0.5)

        def on_cycle(self, context: dict) -> dict | None:
            # Return a signal dict or None
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class PluginType(str, Enum):
    """Types of plugins supported by AIS."""

    STRATEGY = "strategy"
    DATA_SOURCE = "data_source"
    RISK_GUARD = "risk_guard"
    INTEGRATION = "integration"


@dataclass(frozen=True)
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    version: str
    plugin_type: PluginType
    description: str
    author: str = ""


class Plugin(ABC):
    """Base class for all AIS plugins.

    Subclasses must set class-level attributes:
        plugin_type: PluginType
        plugin_name: str
        plugin_version: str
        plugin_description: str
    """

    plugin_type: PluginType
    plugin_name: str = "unnamed"
    plugin_version: str = "0.0.0"
    plugin_description: str = ""
    plugin_author: str = ""

    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        return PluginInfo(
            name=self.plugin_name,
            version=self.plugin_version,
            plugin_type=self.plugin_type,
            description=self.plugin_description,
            author=self.plugin_author,
        )

    def on_load(self, config: dict[str, Any]) -> None:
        """Called when the plugin is loaded. Override to initialize state."""

    def on_start(self) -> None:
        """Called when the trading loop starts."""

    def on_shutdown(self) -> None:
        """Called when the trading loop shuts down."""

    @abstractmethod
    def on_cycle(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Called each trading cycle. Return data or None.

        For STRATEGY plugins: return a signal dict or None.
        For DATA_SOURCE plugins: return enriched context data.
        For RISK_GUARD plugins: return {"approved": bool, "reason": str}.
        For INTEGRATION plugins: return status dict.
        """
        raise NotImplementedError
