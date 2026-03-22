"""Dynamic agent registry for config-driven strategy loading.

Provides a decorator-based registration system that allows agents to
declare their strategy name. The bootstrap module uses this registry
to instantiate only the agents specified in configuration.

Usage::

    # In agent module:
    @register_agent("momentum_ma_crossover")
    class MomentumAgent(Agent):
        ...

    # In bootstrap:
    agents = AgentRegistry.build_agents(config)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

# Global registry: strategy_name -> (agent_class, default_kwargs)
_AGENT_REGISTRY: dict[str, tuple[type[Agent], dict[str, Any]]] = {}


def register_agent(
    strategy: str,
    **default_kwargs: Any,
) -> Callable[[type[Agent]], type[Agent]]:
    """Decorator to register an Agent class under a strategy name.

    Args:
        strategy: The strategy name this agent produces (must match mandates).
        **default_kwargs: Default constructor arguments for the agent.

    Example::

        @register_agent("momentum_ma_crossover")
        class MomentumAgent(Agent):
            ...
    """

    def decorator(cls: type[Agent]) -> type[Agent]:
        if strategy in _AGENT_REGISTRY:
            existing_cls = _AGENT_REGISTRY[strategy][0]
            logger.warning(
                "Overwriting agent registration",
                extra={
                    "extra_json": {
                        "strategy": strategy,
                        "old_class": existing_cls.__name__,
                        "new_class": cls.__name__,
                    }
                },
            )
        _AGENT_REGISTRY[strategy] = (cls, default_kwargs)
        return cls

    return decorator


def get_registered_strategies() -> list[str]:
    """Return all registered strategy names."""
    return sorted(_AGENT_REGISTRY.keys())


def get_agent_class(strategy: str) -> type[Agent] | None:
    """Look up the agent class for a given strategy name."""
    entry = _AGENT_REGISTRY.get(strategy)
    return entry[0] if entry else None


def build_from_registry(
    strategies: list[str],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[Agent]:
    """Instantiate agents from the registry for the given strategy names.

    Args:
        strategies: List of strategy names to activate.
        overrides: Optional per-strategy constructor overrides.
            Example: {"momentum_ma_crossover": {"fast_period": 10}}

    Returns:
        List of instantiated Agent objects.

    Raises:
        ValueError: If a requested strategy is not registered.
    """
    overrides = overrides or {}
    agents: list[Agent] = []

    for strategy in strategies:
        entry = _AGENT_REGISTRY.get(strategy)
        if entry is None:
            available = get_registered_strategies()
            raise ValueError(f"Strategy '{strategy}' not registered. Available: {available}")

        cls, default_kwargs = entry
        kwargs = {**default_kwargs, **overrides.get(strategy, {})}

        try:
            agent = cls(**kwargs)
            agents.append(agent)
            logger.info(
                "Agent instantiated from registry",
                extra={
                    "extra_json": {
                        "strategy": strategy,
                        "class": cls.__name__,
                        "agent_id": agent.agent_id,
                    }
                },
            )
        except Exception:
            logger.exception(
                "Failed to instantiate agent",
                extra={"extra_json": {"strategy": strategy, "class": cls.__name__}},
            )
            raise

    return agents


def discover_agents() -> None:
    """Import all agent modules to trigger @register_agent decorators.

    This function imports all known agent packages so their decorators
    execute and populate the registry. Call this before build_from_registry.
    """
    import importlib

    agent_modules = [
        "aiswarm.agents.strategy.momentum_agent",
        "aiswarm.agents.market_intelligence.funding_rate_agent",
    ]

    for module_name in agent_modules:
        try:
            importlib.import_module(module_name)
        except ImportError:
            logger.debug(
                "Optional agent module not available",
                extra={"extra_json": {"module": module_name}},
            )
