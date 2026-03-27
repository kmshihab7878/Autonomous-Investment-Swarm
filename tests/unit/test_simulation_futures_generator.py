"""Tests for correlated crypto price path generator."""

from __future__ import annotations

import numpy as np
import pytest

from aiswarm.simulation.futures_generator import (
    DEFAULT_ASSETS,
    DEFAULT_CORRELATION,
    AssetParams,
    CryptoFuturesGenerator,
    ScenarioBranch,
    ScenarioResult,
)


STARTING_PRICES = {
    "BTC": 65000.0,
    "ETH": 3500.0,
    "SOL": 150.0,
    "BNB": 600.0,
    "AVAX": 35.0,
    "LINK": 18.0,
    "DOGE": 0.15,
}


class TestCryptoFuturesGenerator:
    def test_init_default_assets(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        assert len(gen.asset_symbols) == 7
        assert "BTC" in gen.asset_symbols

    def test_init_custom_assets(self) -> None:
        assets = [
            AssetParams("A", "Asset A", 0.5, 0.1),
            AssetParams("B", "Asset B", 0.6, 0.05),
        ]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        gen = CryptoFuturesGenerator(assets=assets, correlation_matrix=corr, seed=42)
        assert gen.asset_symbols == ["A", "B"]

    def test_init_mismatched_dimensions(self) -> None:
        assets = [AssetParams("A", "A", 0.5, 0.1)]
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="does not match"):
            CryptoFuturesGenerator(assets=assets, correlation_matrix=corr)

    def test_generate_scenario_base(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        result = gen.generate_scenario(
            ScenarioBranch.BASE, STARTING_PRICES, horizon_days=10
        )

        assert result.scenario == ScenarioBranch.BASE
        assert result.horizon_days == 10
        assert len(result.paths) == 7

        btc_path = result.paths["BTC"]
        assert len(btc_path.prices) == 11  # Starting + 10 days
        assert btc_path.prices[0] == 65000.0
        assert len(btc_path.returns) == 10

    def test_generate_scenario_all_prices_positive(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        result = gen.generate_scenario(
            ScenarioBranch.LIQUIDATION_CASCADE, STARTING_PRICES, horizon_days=30
        )

        for symbol, path in result.paths.items():
            for price in path.prices:
                assert price > 0, f"{symbol} has non-positive price: {price}"

    def test_generate_all_scenarios(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        results = gen.generate_all_scenarios(STARTING_PRICES, horizon_days=10)

        assert len(results) == 5  # 5 scenario branches
        scenarios = {r.scenario for r in results}
        assert scenarios == set(ScenarioBranch)

    def test_probabilities_sum_to_one(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        results = gen.generate_all_scenarios(STARTING_PRICES, horizon_days=5)

        total_prob = sum(r.probability for r in results)
        assert total_prob == pytest.approx(1.0, abs=0.01)

    def test_probability_weighted_return(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        results = gen.generate_all_scenarios(STARTING_PRICES, horizon_days=10)

        expected = gen.probability_weighted_return(results, "BTC")
        # Should be a finite float
        assert np.isfinite(expected)

    def test_tail_risk_return_negative(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        results = gen.generate_all_scenarios(STARTING_PRICES, horizon_days=30)

        tail = gen.tail_risk_return(results, "BTC")
        # Tail risk should be the worst-case, likely negative
        assert tail <= 0.0

    def test_bear_scenario_has_negative_drift(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        # Run multiple bear scenarios to get statistical tendency
        total_return = 0.0
        n = 20
        for i in range(n):
            gen_i = CryptoFuturesGenerator(seed=i)
            result = gen_i.generate_scenario(
                ScenarioBranch.BEAR, STARTING_PRICES, horizon_days=30
            )
            btc = result.paths["BTC"]
            total_return += (btc.prices[-1] - btc.prices[0]) / btc.prices[0]

        avg_return = total_return / n
        assert avg_return < 0.05  # Bear scenario should tend negative

    def test_liquidation_cascade_has_events(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        result = gen.generate_scenario(
            ScenarioBranch.LIQUIDATION_CASCADE, STARTING_PRICES, horizon_days=30
        )

        assert len(result.events_applied) > 0
        assert any("Crash" in e or "Liquidation" in e for e in result.events_applied)

    def test_deterministic_with_same_seed(self) -> None:
        gen1 = CryptoFuturesGenerator(seed=123)
        gen2 = CryptoFuturesGenerator(seed=123)

        r1 = gen1.generate_scenario(ScenarioBranch.BASE, STARTING_PRICES, horizon_days=5)
        r2 = gen2.generate_scenario(ScenarioBranch.BASE, STARTING_PRICES, horizon_days=5)

        for symbol in STARTING_PRICES:
            assert r1.paths[symbol].prices == pytest.approx(r2.paths[symbol].prices)

    def test_different_seeds_produce_different_paths(self) -> None:
        gen1 = CryptoFuturesGenerator(seed=1)
        gen2 = CryptoFuturesGenerator(seed=999)

        r1 = gen1.generate_scenario(ScenarioBranch.BASE, STARTING_PRICES, horizon_days=10)
        r2 = gen2.generate_scenario(ScenarioBranch.BASE, STARTING_PRICES, horizon_days=10)

        # Very unlikely to be identical
        assert r1.paths["BTC"].prices[-1] != pytest.approx(r2.paths["BTC"].prices[-1])

    def test_missing_symbol_uses_default_price(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        result = gen.generate_scenario(
            ScenarioBranch.BASE, {"BTC": 65000.0}, horizon_days=5
        )

        # ETH not in starting_prices → defaults to 100.0
        assert result.paths["ETH"].prices[0] == 100.0

    def test_correlation_matrix_is_valid(self) -> None:
        # Verify default correlation matrix is positive semi-definite
        eigenvalues = np.linalg.eigvalsh(DEFAULT_CORRELATION)
        assert all(v >= -1e-10 for v in eigenvalues)

    def test_tail_risk_unknown_symbol(self) -> None:
        gen = CryptoFuturesGenerator(seed=42)
        results = gen.generate_all_scenarios(STARTING_PRICES, horizon_days=5)
        tail = gen.tail_risk_return(results, "UNKNOWN")
        assert tail == 0.0
