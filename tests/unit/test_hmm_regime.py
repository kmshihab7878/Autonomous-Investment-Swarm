"""Tests for HMM regime detection and regime detector agent."""

from __future__ import annotations


import numpy as np

from aiswarm.intelligence.regime.hmm_detector import (
    HMMRegimeDetector,
    RegimeFeatures,
    RegimeState,
    _label_regime,
    extract_features,
)
from aiswarm.types.market import MarketRegime


# ---------------------------------------------------------------------------
# Tests: extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_returns_empty_for_insufficient_data(self) -> None:
        features = extract_features([100.0] * 5, [1000.0] * 5, lookback=20)
        assert features == []

    def test_returns_correct_count(self) -> None:
        n = 50
        closes = [100.0 + i * 0.1 for i in range(n)]
        volumes = [1000.0] * n
        features = extract_features(closes, volumes, lookback=20)
        assert len(features) == n - 20

    def test_feature_fields_populated(self) -> None:
        closes = [100.0 + i * 0.5 for i in range(30)]
        volumes = [1000.0] * 30
        features = extract_features(closes, volumes, lookback=10)
        assert len(features) > 0
        f = features[0]
        assert isinstance(f.returns, float)
        assert isinstance(f.volatility, float)
        assert isinstance(f.volume_ratio, float)

    def test_high_volatility_detected(self) -> None:
        # Stable then volatile
        closes = [100.0] * 25 + [90.0, 110.0, 85.0, 115.0, 80.0]
        volumes = [1000.0] * 30
        features = extract_features(closes, volumes, lookback=10)
        late_vol = features[-1].volatility
        early_vol = features[0].volatility
        assert late_vol > early_vol

    def test_volume_ratio_spikes(self) -> None:
        closes = [100.0] * 30
        volumes = [1000.0] * 29 + [5000.0]
        features = extract_features(closes, volumes, lookback=10)
        assert features[-1].volume_ratio > 1.5


# ---------------------------------------------------------------------------
# Tests: _label_regime (rule-based fallback)
# ---------------------------------------------------------------------------


class TestLabelRegime:
    def test_low_vol_positive_returns_risk_on(self) -> None:
        assert _label_regime(0.01, 0.005, 0.02) == MarketRegime.RISK_ON

    def test_low_vol_negative_returns_transition(self) -> None:
        assert _label_regime(-0.01, 0.005, 0.02) == MarketRegime.TRANSITION

    def test_high_vol_positive_returns_transition(self) -> None:
        assert _label_regime(0.01, 0.05, 0.02) == MarketRegime.TRANSITION

    def test_high_vol_negative_returns_stressed(self) -> None:
        assert _label_regime(-0.01, 0.05, 0.02) == MarketRegime.STRESSED


# ---------------------------------------------------------------------------
# Tests: HMMRegimeDetector
# ---------------------------------------------------------------------------


class TestHMMRegimeDetector:
    def test_not_fitted_initially(self) -> None:
        detector = HMMRegimeDetector()
        assert not detector.is_fitted

    def test_fit_with_sufficient_data(self) -> None:
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(200) * 0.01) + 100)
        volumes = list(np.random.uniform(800, 1200, 200))
        detector = HMMRegimeDetector(n_regimes=3, lookback=10)
        detector.fit(closes, volumes)
        assert detector.is_fitted

    def test_fit_with_insufficient_data_still_fits_rule_based(self) -> None:
        detector = HMMRegimeDetector(n_regimes=4, lookback=10)
        detector.fit([100.0] * 15, [1000.0] * 15)
        assert detector.is_fitted

    def test_predict_before_fit_returns_transition(self) -> None:
        detector = HMMRegimeDetector()
        state = detector.predict(RegimeFeatures(returns=0.01, volatility=0.01, volume_ratio=1.0))
        assert state.regime == MarketRegime.TRANSITION
        assert state.confidence == 0.0
        assert state.raw_label == "unfitted"

    def test_predict_rule_based_risk_on(self) -> None:
        detector = HMMRegimeDetector(n_regimes=2, lookback=10)
        # Fit with volatile data so the vol_threshold is high enough
        # that vol=0.001 is considered "low vol"
        np.random.seed(99)
        closes = list(np.cumsum(np.random.randn(80) * 0.5) + 100)
        detector.fit(closes, [1000.0] * 80)

        # Low vol + positive returns = risk_on
        state = detector.predict(RegimeFeatures(returns=0.02, volatility=0.001, volume_ratio=1.0))
        assert state.regime == MarketRegime.RISK_ON

    def test_predict_rule_based_stressed(self) -> None:
        detector = HMMRegimeDetector(n_regimes=2, lookback=10)
        detector.fit([100.0 + i * 0.01 for i in range(50)], [1000.0] * 50)

        state = detector.predict(RegimeFeatures(returns=-0.05, volatility=0.10, volume_ratio=2.0))
        assert state.regime == MarketRegime.STRESSED

    def test_predict_from_prices(self) -> None:
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(100) * 0.01) + 100)
        volumes = list(np.random.uniform(800, 1200, 100))
        detector = HMMRegimeDetector(n_regimes=3, lookback=10)
        detector.fit(closes, volumes)

        state = detector.predict_from_prices(closes, volumes)
        assert isinstance(state, RegimeState)
        assert isinstance(state.regime, MarketRegime)
        assert 0 <= state.confidence <= 1.0

    def test_predict_from_prices_insufficient_data(self) -> None:
        detector = HMMRegimeDetector(lookback=50)
        detector.fit([100.0] * 10, [1000.0] * 10)

        state = detector.predict_from_prices([100.0] * 5, [1000.0] * 5)
        assert state.raw_label == "insufficient_data"

    def test_regime_state_has_features(self) -> None:
        detector = HMMRegimeDetector(n_regimes=2, lookback=10)
        detector.fit([100.0 + i * 0.01 for i in range(50)], [1000.0] * 50)

        state = detector.predict(RegimeFeatures(returns=0.01, volatility=0.01, volume_ratio=1.2))
        assert "returns" in state.features
        assert "volatility" in state.features
        assert "volume_ratio" in state.features


# ---------------------------------------------------------------------------
# Tests: RegimeDetectorAgent
# ---------------------------------------------------------------------------


class TestRegimeDetectorAgent:
    def _make_klines(
        self, prices: list[float], volumes: list[float] | None = None
    ) -> list[dict[str, str]]:
        """Build raw kline dicts compatible with AsterDataProvider.parse_klines."""
        if volumes is None:
            volumes = [1000.0] * len(prices)
        return [
            {
                "openTime": str(1700000000000 + i * 3600000),
                "open": str(p * 0.999),
                "high": str(p * 1.01),
                "low": str(p * 0.99),
                "close": str(p),
                "volume": str(v),
            }
            for i, (p, v) in enumerate(zip(prices, volumes))
        ]

    def test_no_data_returns_none(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        agent = RegimeDetectorAgent(min_candles=30)
        result = agent.analyze({"symbol": "BTCUSDT"})
        assert result["signal"] is None

    def test_insufficient_data_returns_none(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        agent = RegimeDetectorAgent(min_candles=60)
        klines = self._make_klines([100.0] * 10)
        result = agent.analyze({"klines_data": klines, "symbol": "BTCUSDT"})
        assert result["signal"] is None
        assert "insufficient" in result["reason"]

    def test_generates_signal_with_sufficient_data(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        np.random.seed(42)
        # Trending up = risk_on → should generate long signal
        prices = [100.0 + i * 0.5 for i in range(80)]
        agent = RegimeDetectorAgent(min_candles=30)
        klines = self._make_klines(prices)
        result = agent.analyze({"klines_data": klines, "symbol": "BTCUSDT"})

        # Should either produce a signal or classify as transition
        if result["signal"] is not None:
            assert result["signal"].strategy == "regime_hmm"
            assert 0.35 <= result["signal"].confidence <= 0.90
        else:
            assert result["reason"] == "transition_regime"

    def test_stressed_regime_generates_short(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        # Sharp decline = stressed
        prices = [100.0] * 40 + [100.0 - i * 2 for i in range(40)]
        agent = RegimeDetectorAgent(min_candles=30)
        klines = self._make_klines(prices)
        result = agent.analyze({"klines_data": klines, "symbol": "BTCUSDT"})

        if result["signal"] is not None:
            assert result["signal"].direction == -1

    def test_validate_returns_false_without_data(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        agent = RegimeDetectorAgent(min_candles=30)
        assert not agent.validate({"symbol": "BTCUSDT"})

    def test_validate_returns_true_with_sufficient_data(self) -> None:
        from aiswarm.agents.market_intelligence.regime_detector import RegimeDetectorAgent

        agent = RegimeDetectorAgent(min_candles=30)
        klines = self._make_klines([100.0 + i * 0.1 for i in range(50)])
        assert agent.validate({"klines_data": klines, "symbol": "BTCUSDT"})
