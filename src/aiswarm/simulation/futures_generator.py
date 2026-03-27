"""Correlated crypto price path generator — Cholesky-based Monte Carlo.

Adapted from ATLAS-GIC's MiroFish futures generator for crypto markets.
Generates synthetic future price paths with realistic cross-asset
correlations using Cholesky decomposition of the correlation matrix.

Features:
- 7 crypto assets with calibrated volatility and drift parameters
- 5 scenario branches: base, bull, bear, liquidation_cascade, regulatory_shock
- Event injection: halving, ETF decision, exchange hack, defi exploit
- Correlated returns via Cholesky decomposition of a 7x7 matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class ScenarioBranch(str, Enum):
    BASE = "base"
    BULL = "bull"
    BEAR = "bear"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    REGULATORY_SHOCK = "regulatory_shock"


# Scenario probability weights (must sum to 1.0)
SCENARIO_WEIGHTS: dict[ScenarioBranch, float] = {
    ScenarioBranch.BASE: 0.50,
    ScenarioBranch.BULL: 0.20,
    ScenarioBranch.BEAR: 0.20,
    ScenarioBranch.LIQUIDATION_CASCADE: 0.05,
    ScenarioBranch.REGULATORY_SHOCK: 0.05,
}


@dataclass(frozen=True)
class AssetParams:
    """Parameters for a single crypto asset."""

    symbol: str
    name: str
    annual_volatility: float  # Annualized volatility (e.g. 0.65 = 65%)
    annual_drift: float  # Annualized expected drift


@dataclass(frozen=True)
class ScheduledEvent:
    """A market event injected into the simulation at a specific day."""

    day: int
    name: str
    affected_assets: list[str]
    impact_pct: float  # Signed: positive = bullish, negative = bearish
    volatility_multiplier: float  # Applied to daily vol on event day


@dataclass(frozen=True)
class PricePath:
    """Generated price path for a single asset."""

    symbol: str
    prices: list[float]  # Day 0 = starting price, Day N = final price
    returns: list[float]  # Daily returns


@dataclass(frozen=True)
class ScenarioResult:
    """Full simulation result for one scenario."""

    scenario: ScenarioBranch
    probability: float
    horizon_days: int
    paths: dict[str, PricePath]  # symbol -> PricePath
    events_applied: list[str]
    generated_at: datetime


# Default crypto asset parameters (annual)
DEFAULT_ASSETS: list[AssetParams] = [
    AssetParams("BTC", "Bitcoin", annual_volatility=0.65, annual_drift=0.15),
    AssetParams("ETH", "Ethereum", annual_volatility=0.80, annual_drift=0.20),
    AssetParams("SOL", "Solana", annual_volatility=1.00, annual_drift=0.25),
    AssetParams("BNB", "BNB", annual_volatility=0.70, annual_drift=0.10),
    AssetParams("AVAX", "Avalanche", annual_volatility=0.95, annual_drift=0.15),
    AssetParams("LINK", "Chainlink", annual_volatility=0.85, annual_drift=0.12),
    AssetParams("DOGE", "Dogecoin", annual_volatility=1.20, annual_drift=0.00),
]

# Cross-asset correlation matrix (7x7)
# BTC is the anchor; alts correlate heavily with BTC but less with each other
DEFAULT_CORRELATION = np.array(
    [
        # BTC   ETH   SOL   BNB   AVAX  LINK  DOGE
        [1.00, 0.85, 0.75, 0.70, 0.72, 0.68, 0.55],  # BTC
        [0.85, 1.00, 0.82, 0.72, 0.78, 0.75, 0.60],  # ETH
        [0.75, 0.82, 1.00, 0.65, 0.80, 0.70, 0.55],  # SOL
        [0.70, 0.72, 0.65, 1.00, 0.62, 0.58, 0.50],  # BNB
        [0.72, 0.78, 0.80, 0.62, 1.00, 0.72, 0.52],  # AVAX
        [0.68, 0.75, 0.70, 0.58, 0.72, 1.00, 0.50],  # LINK
        [0.55, 0.60, 0.55, 0.50, 0.52, 0.50, 1.00],  # DOGE
    ],
    dtype=np.float64,
)

# Default events for scenario simulation
DEFAULT_EVENTS: dict[ScenarioBranch, list[ScheduledEvent]] = {
    ScenarioBranch.BASE: [
        ScheduledEvent(7, "CPI Release", ["BTC", "ETH"], 0.02, 1.5),
        ScheduledEvent(15, "FOMC Decision", ["BTC", "ETH", "SOL"], 0.01, 2.0),
    ],
    ScenarioBranch.BULL: [
        ScheduledEvent(5, "ETF Inflow Surge", ["BTC", "ETH"], 0.05, 1.8),
        ScheduledEvent(12, "Major Adoption News", ["BTC", "ETH", "SOL", "LINK"], 0.03, 1.5),
    ],
    ScenarioBranch.BEAR: [
        ScheduledEvent(3, "Whale Dump", ["BTC", "ETH"], -0.04, 2.0),
        ScheduledEvent(10, "Regulatory FUD", ["BTC", "ETH", "BNB", "SOL"], -0.03, 1.8),
    ],
    ScenarioBranch.LIQUIDATION_CASCADE: [
        ScheduledEvent(2, "Flash Crash", ["BTC", "ETH", "SOL"], -0.08, 3.0),
        ScheduledEvent(
            3, "Cascading Liquidations", ["BTC", "ETH", "SOL", "AVAX", "DOGE"], -0.06, 2.5
        ),
        ScheduledEvent(8, "Dead Cat Bounce", ["BTC", "ETH"], 0.04, 2.0),
    ],
    ScenarioBranch.REGULATORY_SHOCK: [
        ScheduledEvent(4, "Exchange Ban Announcement", ["BTC", "ETH", "BNB"], -0.10, 3.5),
        ScheduledEvent(7, "Stablecoin Depeg Scare", ["BTC", "ETH", "SOL", "AVAX"], -0.05, 2.5),
    ],
}


class CryptoFuturesGenerator:
    """Generates correlated crypto price paths for forward simulation.

    Uses Cholesky decomposition to produce correlated random returns
    across multiple crypto assets, then applies scenario-specific drift
    adjustments and event injections.
    """

    def __init__(
        self,
        assets: list[AssetParams] | None = None,
        correlation_matrix: np.ndarray | None = None,
        events: dict[ScenarioBranch, list[ScheduledEvent]] | None = None,
        scenario_weights: dict[ScenarioBranch, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._assets = assets or DEFAULT_ASSETS
        self._correlation = (
            correlation_matrix if correlation_matrix is not None else DEFAULT_CORRELATION
        )
        self._events = events or DEFAULT_EVENTS
        self._scenario_weights = scenario_weights or SCENARIO_WEIGHTS
        self._rng = np.random.default_rng(seed)

        # Validate dimensions
        n = len(self._assets)
        if self._correlation.shape != (n, n):
            raise ValueError(
                f"Correlation matrix shape {self._correlation.shape} does not match {n} assets"
            )

        # Compute Cholesky decomposition (lower triangular)
        self._cholesky = np.linalg.cholesky(self._correlation)

    @property
    def asset_symbols(self) -> list[str]:
        return [a.symbol for a in self._assets]

    def _daily_params(self, asset: AssetParams) -> tuple[float, float]:
        """Convert annual parameters to daily."""
        daily_vol = asset.annual_volatility / np.sqrt(365)  # Crypto: 365 days
        daily_drift = asset.annual_drift / 365
        return float(daily_drift), float(daily_vol)

    def _scenario_drift_adjustment(
        self,
        scenario: ScenarioBranch,
        asset: AssetParams,
    ) -> float:
        """Scenario-specific drift adjustment multiplier."""
        if scenario == ScenarioBranch.BASE:
            return 1.0
        if scenario == ScenarioBranch.BULL:
            return 2.0  # Double drift in bull scenario
        if scenario == ScenarioBranch.BEAR:
            return -1.5  # Invert and amplify drift
        if scenario == ScenarioBranch.LIQUIDATION_CASCADE:
            return -3.0  # Strong negative drift
        if scenario == ScenarioBranch.REGULATORY_SHOCK:
            return -2.5
        return 1.0

    def generate_scenario(
        self,
        scenario: ScenarioBranch,
        starting_prices: dict[str, float],
        horizon_days: int = 30,
    ) -> ScenarioResult:
        """Generate price paths for one scenario.

        Args:
            scenario: Which scenario branch to simulate.
            starting_prices: Current prices keyed by symbol.
            horizon_days: Number of days to simulate forward.

        Returns:
            ScenarioResult with price paths for all assets.
        """
        n_assets = len(self._assets)

        # Generate correlated random normals for all days
        uncorrelated = self._rng.standard_normal((horizon_days, n_assets))
        correlated = uncorrelated @ self._cholesky.T  # Apply Cholesky

        # Build event lookup: day -> list of events
        scenario_events = self._events.get(scenario, [])
        event_by_day: dict[int, list[ScheduledEvent]] = {}
        for evt in scenario_events:
            if evt.day < horizon_days:
                event_by_day.setdefault(evt.day, []).append(evt)

        events_applied: list[str] = []
        paths: dict[str, PricePath] = {}

        for i, asset in enumerate(self._assets):
            daily_drift, daily_vol = self._daily_params(asset)
            drift_adj = self._scenario_drift_adjustment(scenario, asset)
            adjusted_drift = daily_drift * drift_adj

            prices = [starting_prices.get(asset.symbol, 100.0)]
            returns: list[float] = []

            for day in range(horizon_days):
                vol_mult = 1.0
                event_impact = 0.0

                # Apply events for this day
                for evt in event_by_day.get(day, []):
                    if asset.symbol in evt.affected_assets:
                        event_impact += evt.impact_pct
                        vol_mult = max(vol_mult, evt.volatility_multiplier)
                        if evt.name not in events_applied:
                            events_applied.append(evt.name)

                # GBM-like return: drift + vol * correlated_normal + event
                daily_return = (
                    adjusted_drift + daily_vol * vol_mult * float(correlated[day, i]) + event_impact
                )
                returns.append(daily_return)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(0.001, new_price))  # Floor at 0.001

            paths[asset.symbol] = PricePath(
                symbol=asset.symbol,
                prices=prices,
                returns=returns,
            )

        probability = self._scenario_weights.get(scenario, 0.0)

        logger.info(
            "Scenario generated",
            extra={
                "extra_json": {
                    "scenario": scenario.value,
                    "horizon": horizon_days,
                    "events": len(events_applied),
                    "assets": n_assets,
                }
            },
        )

        return ScenarioResult(
            scenario=scenario,
            probability=probability,
            horizon_days=horizon_days,
            paths=paths,
            events_applied=events_applied,
            generated_at=utc_now(),
        )

    def generate_all_scenarios(
        self,
        starting_prices: dict[str, float],
        horizon_days: int = 30,
    ) -> list[ScenarioResult]:
        """Generate price paths for all scenario branches."""
        return [
            self.generate_scenario(scenario, starting_prices, horizon_days)
            for scenario in ScenarioBranch
        ]

    def probability_weighted_return(
        self,
        results: list[ScenarioResult],
        symbol: str,
    ) -> float:
        """Compute probability-weighted expected return for a symbol.

        Averages the final return across all scenarios, weighted by
        scenario probability.
        """
        total = 0.0
        total_weight = 0.0
        for result in results:
            path = result.paths.get(symbol)
            if path and len(path.prices) >= 2:
                final_return = (path.prices[-1] - path.prices[0]) / path.prices[0]
                total += final_return * result.probability
                total_weight += result.probability

        return total / total_weight if total_weight > 0 else 0.0

    def tail_risk_return(
        self,
        results: list[ScenarioResult],
        symbol: str,
    ) -> float:
        """Return from the worst-case scenario for a symbol."""
        worst = 0.0
        for result in results:
            path = result.paths.get(symbol)
            if path and len(path.prices) >= 2:
                final_return = (path.prices[-1] - path.prices[0]) / path.prices[0]
                worst = min(worst, final_return)
        return worst
