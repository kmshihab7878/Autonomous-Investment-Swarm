"""Tests for simulation engine."""

from __future__ import annotations

import pytest

from aiswarm.simulation.engine import (
    AgentAdapter,
    AgentPrediction,
    PredictionOutcome,
    SimulationEngine,
    SimulationSummary,
)
from aiswarm.simulation.futures_generator import CryptoFuturesGenerator, ScenarioBranch
from aiswarm.simulation.reflexivity import ReflexivityDetector
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.time import utc_now


STARTING_PRICES = {
    "BTC": 65000.0,
    "ETH": 3500.0,
    "SOL": 150.0,
    "BNB": 600.0,
    "AVAX": 35.0,
    "LINK": 18.0,
    "DOGE": 0.15,
}


def _bullish_agent(context: dict) -> dict:
    """Mock agent that always predicts bullish."""
    return {
        "signal": Signal(
            signal_id=new_id("sig"),
            agent_id="bullish_agent",
            symbol=context.get("symbol", "BTCUSDT"),
            strategy="test_bullish",
            thesis="Always bullish for testing",
            direction=1,
            confidence=0.8,
            expected_return=0.05,
            horizon_minutes=240,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_ON,
            created_at=utc_now(),
        ),
    }


def _bearish_agent(context: dict) -> dict:
    """Mock agent that always predicts bearish."""
    return {
        "signal": Signal(
            signal_id=new_id("sig"),
            agent_id="bearish_agent",
            symbol=context.get("symbol", "BTCUSDT"),
            strategy="test_bearish",
            thesis="Always bearish for testing",
            direction=-1,
            confidence=0.7,
            expected_return=0.03,
            horizon_minutes=240,
            liquidity_score=0.8,
            regime=MarketRegime.RISK_OFF,
            created_at=utc_now(),
        ),
    }


def _neutral_agent(context: dict) -> dict:
    """Mock agent that returns no signal."""
    return {"signal": None, "reason": "neutral"}


def _failing_agent(context: dict) -> dict:
    """Mock agent that raises an exception."""
    raise RuntimeError("Agent failure")


class TestScorePrediction:
    def test_correct_long_prediction(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        pred = AgentPrediction(
            agent_id="a",
            scenario=ScenarioBranch.BASE,
            symbol="BTC",
            direction=1,
            confidence=0.8,
            rationale="bullish",
        )
        scored = engine.score_prediction(pred, actual_return=0.05)
        assert scored.outcome == PredictionOutcome.CORRECT
        assert scored.score > 0.5

    def test_incorrect_long_prediction(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        pred = AgentPrediction(
            agent_id="a",
            scenario=ScenarioBranch.BASE,
            symbol="BTC",
            direction=1,
            confidence=0.9,
            rationale="bullish but wrong",
        )
        scored = engine.score_prediction(pred, actual_return=-0.05)
        assert scored.outcome == PredictionOutcome.INCORRECT
        assert scored.score < 0.3

    def test_neutral_prediction(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        pred = AgentPrediction(
            agent_id="a",
            scenario=ScenarioBranch.BASE,
            symbol="BTC",
            direction=0,
            confidence=0.5,
            rationale="neutral",
        )
        scored = engine.score_prediction(pred, actual_return=0.05)
        assert scored.outcome == PredictionOutcome.NEUTRAL
        assert scored.score == pytest.approx(0.3)

    def test_low_confidence_treated_as_neutral(self) -> None:
        engine = SimulationEngine(
            CryptoFuturesGenerator(seed=42),
            min_confidence_threshold=0.5,
        )
        pred = AgentPrediction(
            agent_id="a",
            scenario=ScenarioBranch.BASE,
            symbol="BTC",
            direction=1,
            confidence=0.2,
            rationale="low conviction",
        )
        scored = engine.score_prediction(pred, actual_return=0.10)
        assert scored.outcome == PredictionOutcome.NEUTRAL

    def test_correct_short_prediction(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        pred = AgentPrediction(
            agent_id="a",
            scenario=ScenarioBranch.BEAR,
            symbol="BTC",
            direction=-1,
            confidence=0.8,
            rationale="bearish",
        )
        scored = engine.score_prediction(pred, actual_return=-0.05)
        assert scored.outcome == PredictionOutcome.CORRECT
        assert scored.score > 0.5


class TestAgentAdapter:
    def test_predict_bullish(self) -> None:
        adapter = AgentAdapter(
            agent_id="bull",
            strategy="test",
            analyze_fn=_bullish_agent,
        )
        pred = adapter.predict(ScenarioBranch.BASE, "BTC", [100, 101, 102])
        assert pred.direction == 1
        assert pred.confidence > 0

    def test_predict_neutral(self) -> None:
        adapter = AgentAdapter(
            agent_id="neutral",
            strategy="test",
            analyze_fn=_neutral_agent,
        )
        pred = adapter.predict(ScenarioBranch.BASE, "BTC", [100, 99, 98])
        assert pred.direction == 0
        assert pred.confidence == 0.0

    def test_predict_handles_failure(self) -> None:
        adapter = AgentAdapter(
            agent_id="fail",
            strategy="test",
            analyze_fn=_failing_agent,
        )
        pred = adapter.predict(ScenarioBranch.BASE, "BTC", [100])
        assert pred.direction == 0
        assert pred.rationale == "analysis_failed"


class TestSimulationEngine:
    def test_run_simulation(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        engine = SimulationEngine(gen)

        agents = [
            AgentAdapter("bull", "test", _bullish_agent),
            AgentAdapter("bear", "test", _bearish_agent),
        ]

        summary = engine.run_simulation(
            agents=agents,
            starting_prices=STARTING_PRICES,
            symbols=["BTC", "ETH"],
            horizon_days=10,
        )

        assert isinstance(summary, SimulationSummary)
        assert summary.scenarios_simulated == 5
        assert summary.agents_tested == 2
        assert summary.total_predictions > 0
        assert 0 <= summary.avg_score <= 1.0
        assert summary.best_agent in ("bull", "bear")
        assert summary.worst_agent in ("bull", "bear")

    def test_run_count_increments(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        assert engine.run_count == 0

        agents = [AgentAdapter("a", "test", _neutral_agent)]
        engine.run_simulation(agents, STARTING_PRICES, ["BTC"], 5)
        assert engine.run_count == 1

    def test_history_accumulates(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        agents = [AgentAdapter("a", "test", _neutral_agent)]

        engine.run_simulation(agents, STARTING_PRICES, ["BTC"], 5)
        engine.run_simulation(agents, STARTING_PRICES, ["BTC"], 5)

        assert len(engine.history) == 2

    def test_probability_weighted_forecast(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        forecasts = engine.probability_weighted_forecast(
            STARTING_PRICES, ["BTC", "ETH"], horizon_days=10
        )

        assert "BTC" in forecasts
        assert "ETH" in forecasts
        assert "expected_return" in forecasts["BTC"]
        assert "tail_risk_return" in forecasts["BTC"]

    def test_scenario_returns_populated(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        agents = [AgentAdapter("a", "test", _bullish_agent)]

        summary = engine.run_simulation(
            agents, STARTING_PRICES, ["BTC"], horizon_days=5
        )

        assert "base" in summary.scenario_returns
        assert "BTC" in summary.scenario_returns["base"]

    def test_empty_agents_list(self) -> None:
        engine = SimulationEngine(CryptoFuturesGenerator(seed=42))
        summary = engine.run_simulation(
            agents=[], starting_prices=STARTING_PRICES, symbols=["BTC"], horizon_days=5
        )
        assert summary.total_predictions == 0
        assert summary.avg_score == 0.0
