"""Simulation engine — forward scenario testing for strategy validation.

Adapted from ATLAS-GIC's MiroFish simulation bridge. Generates scenarios
from current market state, runs agents through simulated futures, scores
predictions against synthetic outcomes, and feeds results back into the
evolution system.

The engine connects:
- CryptoFuturesGenerator for price path generation
- DarwinianWeightManager for agent scoring
- ReflexivityDetector for feedback loop awareness
- Existing AIS agents for forward-testing decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from aiswarm.simulation.futures_generator import (
    CryptoFuturesGenerator,
    ScenarioBranch,
)
from aiswarm.simulation.reflexivity import (
    ReflexivityDetector,
    ReflexivitySignal,
)
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class PredictionOutcome(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class AgentPrediction:
    """An agent's prediction for a specific scenario."""

    agent_id: str
    scenario: ScenarioBranch
    symbol: str
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0.0 - 1.0
    rationale: str


@dataclass(frozen=True)
class ScoredPrediction:
    """A prediction scored against actual synthetic outcome."""

    prediction: AgentPrediction
    actual_return: float
    score: float  # 0.0 - 1.0
    outcome: PredictionOutcome


@dataclass(frozen=True)
class SimulationSummary:
    """Summary of a complete simulation run."""

    run_id: str
    timestamp: datetime
    scenarios_simulated: int
    agents_tested: int
    total_predictions: int
    avg_score: float
    best_agent: str
    worst_agent: str
    reflexivity_signals: list[ReflexivitySignal]
    scenario_returns: dict[str, dict[str, float]]  # scenario -> symbol -> return


@dataclass
class AgentAdapter:
    """Adapter to present scenarios to an agent and collect predictions.

    Since AIS agents have an `analyze(context)` interface, this adapter
    formats scenario data into the expected context dict.
    """

    agent_id: str
    strategy: str
    analyze_fn: object  # Callable[[dict], dict] — the agent's analyze method

    def predict(
        self,
        scenario: ScenarioBranch,
        symbol: str,
        price_path: list[float],
    ) -> AgentPrediction:
        """Ask the agent to predict based on a price path.

        Formats the price path as mock klines data that the agent's
        analyze method can consume.
        """
        # Build context that mimics the standard agent context
        context = {
            "symbol": symbol,
            "scenario": scenario.value,
            "price_path": price_path,
            "simulation_mode": True,
        }

        try:
            result = self.analyze_fn(context)  # type: ignore[operator]
        except Exception:
            logger.debug(
                "Agent analysis failed in simulation",
                extra={"extra_json": {"agent_id": self.agent_id, "symbol": symbol}},
            )
            return AgentPrediction(
                agent_id=self.agent_id,
                scenario=scenario,
                symbol=symbol,
                direction=0,
                confidence=0.0,
                rationale="analysis_failed",
            )

        signal = result.get("signal")
        if signal is None:
            return AgentPrediction(
                agent_id=self.agent_id,
                scenario=scenario,
                symbol=symbol,
                direction=0,
                confidence=0.0,
                rationale=result.get("reason", "no_signal"),
            )

        return AgentPrediction(
            agent_id=self.agent_id,
            scenario=scenario,
            symbol=symbol,
            direction=signal.direction,
            confidence=signal.confidence,
            rationale=signal.thesis,
        )


class SimulationEngine:
    """Forward scenario simulation engine.

    Orchestrates the full simulation pipeline:
    1. Generate price paths via CryptoFuturesGenerator
    2. Present scenarios to agents via AgentAdapter
    3. Score predictions against synthetic outcomes
    4. Aggregate results for the evolution system
    """

    def __init__(
        self,
        futures_generator: CryptoFuturesGenerator,
        reflexivity_detector: ReflexivityDetector | None = None,
        min_confidence_threshold: float = 0.3,
    ) -> None:
        self._generator = futures_generator
        self._reflexivity = reflexivity_detector or ReflexivityDetector()
        self._min_confidence = min_confidence_threshold
        self._run_counter = 0
        self._history: list[SimulationSummary] = []

    @property
    def run_count(self) -> int:
        return self._run_counter

    @property
    def history(self) -> list[SimulationSummary]:
        return list(self._history)

    def score_prediction(
        self,
        prediction: AgentPrediction,
        actual_return: float,
    ) -> ScoredPrediction:
        """Score a single prediction against actual outcome.

        Scoring:
        - Correct direction + high confidence = high score
        - Wrong direction + high confidence = penalized
        - Neutral / low confidence = moderate score (not rewarded for abstaining)
        """
        if prediction.direction == 0 or prediction.confidence < self._min_confidence:
            return ScoredPrediction(
                prediction=prediction,
                actual_return=actual_return,
                score=0.3,  # Neutral baseline
                outcome=PredictionOutcome.NEUTRAL,
            )

        direction_correct = (
            (prediction.direction > 0 and actual_return > 0)
            or (prediction.direction < 0 and actual_return < 0)
        )

        if direction_correct:
            # Reward scales with confidence and magnitude
            magnitude_bonus = min(0.3, abs(actual_return) * 2)
            score = 0.5 + prediction.confidence * 0.3 + magnitude_bonus
            score = min(1.0, score)
            outcome = PredictionOutcome.CORRECT
        else:
            # Penalty scales with confidence (high-conviction wrong = worst)
            score = max(0.0, 0.3 - prediction.confidence * 0.3)
            outcome = PredictionOutcome.INCORRECT

        return ScoredPrediction(
            prediction=prediction,
            actual_return=actual_return,
            score=score,
            outcome=outcome,
        )

    def run_simulation(
        self,
        agents: list[AgentAdapter],
        starting_prices: dict[str, float],
        symbols: list[str],
        horizon_days: int = 30,
    ) -> SimulationSummary:
        """Run a complete forward simulation.

        Generates all scenario branches, presents each to all agents,
        scores predictions, and returns an aggregated summary.
        """
        self._run_counter += 1
        run_id = f"sim_{self._run_counter:04d}"

        # Generate all scenario paths
        all_scenarios = self._generator.generate_all_scenarios(
            starting_prices=starting_prices,
            horizon_days=horizon_days,
        )

        all_scored: list[ScoredPrediction] = []
        agent_scores: dict[str, list[float]] = {a.agent_id: [] for a in agents}
        scenario_returns: dict[str, dict[str, float]] = {}

        for scenario_result in all_scenarios:
            scenario_name = scenario_result.scenario.value
            scenario_returns[scenario_name] = {}

            for symbol in symbols:
                path = scenario_result.paths.get(symbol)
                if path is None or len(path.prices) < 2:
                    continue

                actual_return = (path.prices[-1] - path.prices[0]) / path.prices[0]
                scenario_returns[scenario_name][symbol] = actual_return

                for agent in agents:
                    prediction = agent.predict(
                        scenario=scenario_result.scenario,
                        symbol=symbol,
                        price_path=path.prices,
                    )
                    scored = self.score_prediction(prediction, actual_return)
                    all_scored.append(scored)
                    agent_scores[agent.agent_id].append(scored.score)

        # Identify best/worst agents
        avg_by_agent = {
            aid: float(np.mean(scores)) if scores else 0.0
            for aid, scores in agent_scores.items()
        }
        best_agent = max(avg_by_agent, key=lambda k: avg_by_agent[k]) if avg_by_agent else ""
        worst_agent = min(avg_by_agent, key=lambda k: avg_by_agent[k]) if avg_by_agent else ""

        # Check reflexivity from the base scenario
        reflexivity_signals: list[ReflexivitySignal] = []
        if self._reflexivity is not None:
            base_scenarios = [s for s in all_scenarios if s.scenario == ScenarioBranch.BASE]
            if base_scenarios:
                # Use BTC path (or first available) for reflexivity detection
                for symbol in symbols[:1]:
                    path = base_scenarios[0].paths.get(symbol)
                    if path:
                        from aiswarm.simulation.reflexivity import PriceObservation

                        for i, price in enumerate(path.prices):
                            self._reflexivity.add_observation(
                                PriceObservation(
                                    timestamp=utc_now(),
                                    price=price,
                                    volume=1000.0,
                                )
                            )
                        reflexivity_signals = self._reflexivity.detect_all()

        overall_avg = float(np.mean([s.score for s in all_scored])) if all_scored else 0.0

        summary = SimulationSummary(
            run_id=run_id,
            timestamp=utc_now(),
            scenarios_simulated=len(all_scenarios),
            agents_tested=len(agents),
            total_predictions=len(all_scored),
            avg_score=overall_avg,
            best_agent=best_agent,
            worst_agent=worst_agent,
            reflexivity_signals=reflexivity_signals,
            scenario_returns=scenario_returns,
        )

        self._history.append(summary)

        logger.info(
            "Simulation complete",
            extra={
                "extra_json": {
                    "run_id": run_id,
                    "scenarios": len(all_scenarios),
                    "predictions": len(all_scored),
                    "avg_score": round(overall_avg, 4),
                    "best_agent": best_agent,
                    "worst_agent": worst_agent,
                    "reflexivity_signals": len(reflexivity_signals),
                }
            },
        )

        return summary

    def probability_weighted_forecast(
        self,
        starting_prices: dict[str, float],
        symbols: list[str],
        horizon_days: int = 30,
    ) -> dict[str, dict[str, float]]:
        """Generate probability-weighted return forecasts per symbol.

        Returns a dict of symbol -> {expected_return, tail_risk_return}.
        No agent involvement — pure scenario-weighted price paths.
        """
        all_scenarios = self._generator.generate_all_scenarios(
            starting_prices=starting_prices,
            horizon_days=horizon_days,
        )

        forecasts: dict[str, dict[str, float]] = {}
        for symbol in symbols:
            expected = self._generator.probability_weighted_return(all_scenarios, symbol)
            tail = self._generator.tail_risk_return(all_scenarios, symbol)
            forecasts[symbol] = {
                "expected_return": expected,
                "tail_risk_return": tail,
            }

        return forecasts
