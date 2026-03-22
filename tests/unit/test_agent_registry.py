"""Tests for the dynamic agent registry: registration, lookup, and instantiation."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import (
    _AGENT_REGISTRY,
    build_from_registry,
    discover_agents,
    get_agent_class,
    get_registered_strategies,
    register_agent,
)


# ---------------------------------------------------------------------------
# Test helpers: concrete Agent subclasses
# ---------------------------------------------------------------------------


class _FakeAgentA(Agent):
    """Minimal concrete Agent for testing registration."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(agent_id="fake_a", cluster="test")
        self.kwargs = kwargs

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def validate(self, context: dict[str, Any]) -> bool:
        return True


class _FakeAgentB(Agent):
    """Second concrete Agent for testing overwrite detection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(agent_id="fake_b", cluster="test")
        self.kwargs = kwargs

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def validate(self, context: dict[str, Any]) -> bool:
        return True


class _FakeAgentWithParams(Agent):
    """Agent that accepts constructor parameters for override testing."""

    def __init__(self, fast_period: int = 5, slow_period: int = 20, **kwargs: Any) -> None:
        super().__init__(agent_id="fake_params", cluster="test")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return {}

    def validate(self, context: dict[str, Any]) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixture: isolate the global registry between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry() -> Any:
    """Save and restore the global agent registry around each test.

    This prevents test-registered agents from leaking into other tests.
    """
    saved = dict(_AGENT_REGISTRY)
    yield
    _AGENT_REGISTRY.clear()
    _AGENT_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Tests: @register_agent decorator
# ---------------------------------------------------------------------------


class TestRegisterAgent:
    def test_decorator_registers_class(self) -> None:
        """@register_agent stores the class in the global registry."""

        # Arrange & Act
        @register_agent("test_strategy_alpha")
        class AlphaAgent(_FakeAgentA):
            pass

        # Assert
        assert "test_strategy_alpha" in _AGENT_REGISTRY
        assert _AGENT_REGISTRY["test_strategy_alpha"][0] is AlphaAgent

    def test_decorator_returns_original_class(self) -> None:
        """The decorator returns the class unchanged (no wrapping)."""

        # Arrange & Act
        @register_agent("test_strategy_passthrough")
        class PassthroughAgent(_FakeAgentA):
            pass

        # Assert
        assert PassthroughAgent.__name__ == "PassthroughAgent"
        assert issubclass(PassthroughAgent, Agent)

    def test_decorator_stores_default_kwargs(self) -> None:
        """Default kwargs are stored alongside the class."""

        # Arrange & Act
        @register_agent("test_strategy_with_defaults", fast_period=10, slow_period=30)
        class DefaultsAgent(_FakeAgentWithParams):
            pass

        # Assert
        _, defaults = _AGENT_REGISTRY["test_strategy_with_defaults"]
        assert defaults == {"fast_period": 10, "slow_period": 30}

    def test_overwriting_registration_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Re-registering the same strategy name logs a warning."""

        # Arrange
        @register_agent("test_overwrite_strategy")
        class FirstAgent(_FakeAgentA):
            pass

        # Act
        with caplog.at_level(logging.WARNING):

            @register_agent("test_overwrite_strategy")
            class SecondAgent(_FakeAgentB):
                pass

        # Assert — the new class replaces the old one
        assert _AGENT_REGISTRY["test_overwrite_strategy"][0] is SecondAgent
        assert any("Overwriting" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: get_registered_strategies
# ---------------------------------------------------------------------------


class TestGetRegisteredStrategies:
    def test_returns_sorted_list(self) -> None:
        """Strategy names are returned sorted alphabetically."""
        # Arrange
        register_agent("zebra")(_FakeAgentA)
        register_agent("alpha")(_FakeAgentA)
        register_agent("mango")(_FakeAgentA)

        # Act
        strategies = get_registered_strategies()

        # Assert — filter to only our test strategies to avoid leakage
        test_strategies = [s for s in strategies if s in {"zebra", "alpha", "mango"}]
        assert test_strategies == ["alpha", "mango", "zebra"]

    def test_empty_when_nothing_registered(self) -> None:
        """Returns empty list when no agents are registered."""
        # Arrange — clear registry (fixture restores it after)
        _AGENT_REGISTRY.clear()

        # Act
        strategies = get_registered_strategies()

        # Assert
        assert strategies == []


# ---------------------------------------------------------------------------
# Tests: get_agent_class
# ---------------------------------------------------------------------------


class TestGetAgentClass:
    def test_returns_correct_class(self) -> None:
        """Returns the Agent subclass registered under the given strategy."""
        # Arrange
        register_agent("test_lookup")(_FakeAgentA)

        # Act
        cls = get_agent_class("test_lookup")

        # Assert
        assert cls is _FakeAgentA

    def test_returns_none_for_unknown_strategy(self) -> None:
        """Returns None when the strategy name is not in the registry."""
        # Arrange (nothing registered for this name)

        # Act
        cls = get_agent_class("nonexistent_strategy_xyz")

        # Assert
        assert cls is None


# ---------------------------------------------------------------------------
# Tests: build_from_registry
# ---------------------------------------------------------------------------


class TestBuildFromRegistry:
    def test_creates_agent_instances(self) -> None:
        """Builds a list of Agent instances from registered strategies."""
        # Arrange
        register_agent("test_build_a")(_FakeAgentA)
        register_agent("test_build_b")(_FakeAgentB)

        # Act
        agents = build_from_registry(["test_build_a", "test_build_b"])

        # Assert
        assert len(agents) == 2
        assert isinstance(agents[0], _FakeAgentA)
        assert isinstance(agents[1], _FakeAgentB)

    def test_raises_value_error_for_unknown_strategy(self) -> None:
        """Raises ValueError when a requested strategy is not registered."""
        # Arrange (nothing registered for this name)

        # Act & Assert
        with pytest.raises(ValueError, match="not registered"):
            build_from_registry(["totally_unknown_strategy"])

    def test_applies_overrides(self) -> None:
        """Constructor overrides are passed to the agent."""
        # Arrange
        register_agent("test_override", fast_period=5, slow_period=20)(_FakeAgentWithParams)

        # Act
        agents = build_from_registry(
            ["test_override"],
            overrides={"test_override": {"fast_period": 10, "slow_period": 50}},
        )

        # Assert
        agent = agents[0]
        assert isinstance(agent, _FakeAgentWithParams)
        assert agent.fast_period == 10
        assert agent.slow_period == 50

    def test_default_kwargs_used_when_no_overrides(self) -> None:
        """Default kwargs from registration are used when no overrides given."""
        # Arrange
        register_agent("test_defaults", fast_period=7, slow_period=21)(_FakeAgentWithParams)

        # Act
        agents = build_from_registry(["test_defaults"])

        # Assert
        agent = agents[0]
        assert isinstance(agent, _FakeAgentWithParams)
        assert agent.fast_period == 7
        assert agent.slow_period == 21

    def test_overrides_merge_with_defaults(self) -> None:
        """Overrides merge with defaults; override wins on conflict."""
        # Arrange
        register_agent("test_merge", fast_period=5, slow_period=20)(_FakeAgentWithParams)

        # Act — override only fast_period
        agents = build_from_registry(
            ["test_merge"],
            overrides={"test_merge": {"fast_period": 12}},
        )

        # Assert
        agent = agents[0]
        assert isinstance(agent, _FakeAgentWithParams)
        assert agent.fast_period == 12
        assert agent.slow_period == 20  # default preserved

    def test_empty_strategy_list_returns_empty(self) -> None:
        """An empty strategies list returns an empty agents list."""
        # Arrange & Act
        agents = build_from_registry([])

        # Assert
        assert agents == []


# ---------------------------------------------------------------------------
# Tests: discover_agents
# ---------------------------------------------------------------------------


class TestDiscoverAgents:
    def test_imports_known_modules(self) -> None:
        """discover_agents attempts to import the known agent modules."""
        # Arrange
        with patch("importlib.import_module") as mock_import:
            # Act
            discover_agents()

        # Assert — should have attempted both known modules
        imported_names = [call.args[0] for call in mock_import.call_args_list]
        assert "aiswarm.agents.strategy.momentum_agent" in imported_names
        assert "aiswarm.agents.market_intelligence.funding_rate_agent" in imported_names

    def test_handles_import_error_gracefully(self) -> None:
        """discover_agents does not raise when a module is unavailable."""
        # Arrange
        with patch("importlib.import_module", side_effect=ImportError("no such module")):
            # Act — should not raise
            discover_agents()

        # Assert — reaching this point without exception is the test
