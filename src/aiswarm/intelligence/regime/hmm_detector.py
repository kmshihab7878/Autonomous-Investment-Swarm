"""Hidden Markov Model regime detector for market state classification.

Classifies market conditions into discrete regimes (risk_on, risk_off,
transition, stressed) using returns, realized volatility, and volume
features. Works with or without the optional ``hmmlearn`` package —
falls back to a rule-based classifier when hmmlearn is unavailable.

Usage::

    detector = HMMRegimeDetector(n_regimes=4)
    detector.fit(returns, volatilities, volumes)
    regime = detector.predict_current(returns[-1], vol[-1], volume[-1])
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from aiswarm.types.market import MarketRegime
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)

# Try optional hmmlearn import
try:
    from hmmlearn.hmm import GaussianHMM

    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False


class RegimeLabel(str, Enum):
    """Internal regime labels mapped to MarketRegime."""

    LOW_VOL_UP = "low_vol_up"  # risk_on
    LOW_VOL_DOWN = "low_vol_down"  # transition
    HIGH_VOL_UP = "high_vol_up"  # transition
    HIGH_VOL_DOWN = "high_vol_down"  # risk_off / stressed


@dataclass(frozen=True)
class RegimeState:
    """Current regime classification with metadata."""

    regime: MarketRegime
    confidence: float
    raw_label: str
    features: dict[str, float]


@dataclass(frozen=True)
class RegimeFeatures:
    """Feature vector for regime detection."""

    returns: float
    volatility: float
    volume_ratio: float


def extract_features(
    closes: list[float],
    volumes: list[float],
    lookback: int = 20,
) -> list[RegimeFeatures]:
    """Extract regime features from price and volume data.

    Args:
        closes: Close prices (chronological).
        volumes: Volume values aligned with closes.
        lookback: Window for volatility and volume ratio computation.

    Returns:
        List of RegimeFeatures (one per bar, starting at index lookback).
    """
    if len(closes) < lookback + 1 or len(volumes) < lookback + 1:
        return []

    features: list[RegimeFeatures] = []
    for i in range(lookback, len(closes)):
        # Log return
        ret = (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] != 0 else 0.0

        # Realized volatility (std of returns over lookback)
        window_returns = [
            (closes[j] - closes[j - 1]) / closes[j - 1] if closes[j - 1] != 0 else 0.0
            for j in range(i - lookback + 1, i + 1)
        ]
        vol: float = float(np.std(window_returns, ddof=1)) if len(window_returns) > 1 else 0.0

        # Volume ratio (current / avg)
        avg_vol: float = float(np.mean(volumes[i - lookback : i])) if lookback > 0 else 1.0
        vol_ratio = volumes[i] / avg_vol if avg_vol > 0 else 1.0

        features.append(RegimeFeatures(returns=ret, volatility=vol, volume_ratio=vol_ratio))

    return features


def _features_to_array(features: list[RegimeFeatures]) -> np.ndarray:
    """Convert feature list to numpy array (N, 3)."""
    return np.array(
        [[f.returns, f.volatility, f.volume_ratio] for f in features],
        dtype=np.float64,
    )


def _label_regime(ret: float, vol: float, vol_threshold: float) -> MarketRegime:
    """Rule-based regime classification (fallback when hmmlearn unavailable)."""
    high_vol = vol > vol_threshold
    positive = ret >= 0

    if not high_vol and positive:
        return MarketRegime.RISK_ON
    if not high_vol and not positive:
        return MarketRegime.TRANSITION
    if high_vol and positive:
        return MarketRegime.TRANSITION
    # high_vol and negative
    return MarketRegime.STRESSED


class HMMRegimeDetector:
    """Market regime detector using Gaussian Hidden Markov Model.

    Falls back to rule-based classification when hmmlearn is not installed.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 20,
        vol_threshold_percentile: float = 75.0,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.vol_threshold_percentile = vol_threshold_percentile
        self.random_state = random_state
        self._model: object | None = None
        self._vol_threshold: float = 0.02  # default, calibrated on fit()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def uses_hmm(self) -> bool:
        return _HAS_HMMLEARN and self._model is not None

    def fit(
        self,
        closes: list[float],
        volumes: list[float],
    ) -> None:
        """Fit the regime model on historical data.

        Args:
            closes: Historical close prices.
            volumes: Historical volume values.
        """
        features = extract_features(closes, volumes, self.lookback)
        if len(features) < self.n_regimes * 5:
            logger.warning(
                "Insufficient data for HMM fitting, using rule-based fallback",
                extra={
                    "extra_json": {"features": len(features), "min_required": self.n_regimes * 5}
                },
            )
            vols = [f.volatility for f in features] if features else [0.02]
            self._vol_threshold = float(np.percentile(vols, self.vol_threshold_percentile))
            self._fitted = True
            return

        # Calibrate volatility threshold for rule-based fallback
        vols = [f.volatility for f in features]
        self._vol_threshold = float(np.percentile(vols, self.vol_threshold_percentile))

        if _HAS_HMMLEARN:
            X = _features_to_array(features)
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=self.random_state,
            )
            try:
                model.fit(X)
                self._model = model
                logger.info(
                    "HMM regime model fitted",
                    extra={
                        "extra_json": {
                            "n_regimes": self.n_regimes,
                            "n_samples": len(features),
                            "converged": model.monitor_.converged,
                        }
                    },
                )
            except Exception:
                logger.exception("HMM fitting failed, using rule-based fallback")
                self._model = None
        else:
            logger.info("hmmlearn not installed, using rule-based regime detection")

        self._fitted = True

    def predict(self, features: RegimeFeatures) -> RegimeState:
        """Classify the current market regime from a single feature vector.

        Args:
            features: Current bar's regime features.

        Returns:
            RegimeState with classification and confidence.
        """
        if not self._fitted:
            return RegimeState(
                regime=MarketRegime.TRANSITION,
                confidence=0.0,
                raw_label="unfitted",
                features={
                    "returns": features.returns,
                    "volatility": features.volatility,
                    "volume_ratio": features.volume_ratio,
                },
            )

        feature_dict = {
            "returns": features.returns,
            "volatility": features.volatility,
            "volume_ratio": features.volume_ratio,
        }

        if self._model is not None and _HAS_HMMLEARN:
            return self._predict_hmm(features, feature_dict)

        return self._predict_rule_based(features, feature_dict)

    def predict_from_prices(
        self,
        closes: list[float],
        volumes: list[float],
    ) -> RegimeState:
        """Classify regime from raw price/volume data (extracts features internally)."""
        features = extract_features(closes, volumes, self.lookback)
        if not features:
            return RegimeState(
                regime=MarketRegime.TRANSITION,
                confidence=0.0,
                raw_label="insufficient_data",
                features={},
            )
        return self.predict(features[-1])

    def _predict_hmm(
        self,
        features: RegimeFeatures,
        feature_dict: dict[str, float],
    ) -> RegimeState:
        """Predict using fitted HMM model."""
        X = np.array([[features.returns, features.volatility, features.volume_ratio]])
        state = int(self._model.predict(X)[0])  # type: ignore[union-attr]
        proba = self._model.predict_proba(X)[0]  # type: ignore[union-attr]
        confidence = float(proba[state])

        # Map HMM state to MarketRegime based on state means
        means = self._model.means_  # type: ignore[union-attr]
        state_ret = float(means[state][0])
        state_vol = float(means[state][1])

        regime = _label_regime(state_ret, state_vol, self._vol_threshold)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            raw_label=f"hmm_state_{state}",
            features=feature_dict,
        )

    def _predict_rule_based(
        self,
        features: RegimeFeatures,
        feature_dict: dict[str, float],
    ) -> RegimeState:
        """Predict using rule-based fallback."""
        regime = _label_regime(features.returns, features.volatility, self._vol_threshold)

        # Confidence heuristic: how far from the threshold boundaries
        vol_distance = abs(features.volatility - self._vol_threshold) / max(
            self._vol_threshold, 0.001
        )
        confidence = min(0.90, 0.50 + vol_distance * 0.3)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            raw_label=f"rule_{regime.value}",
            features=feature_dict,
        )
