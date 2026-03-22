"""HMM-based market regime detection agent.

Classifies market regime using a Hidden Markov Model (or rule-based
fallback) and generates signals when regime transitions occur.
Risk-off and stressed regimes produce short signals; risk-on produces
long signals; transitions produce no signal.

This agent maintains its own HMM detector instance and fits it
on the first call with sufficient data.
"""

from __future__ import annotations

from typing import Any

from aiswarm.agents.base import Agent
from aiswarm.agents.registry import register_agent
from aiswarm.data.providers.aster import AsterDataProvider
from aiswarm.intelligence.regime.hmm_detector import (
    HMMRegimeDetector,
)
from aiswarm.types.market import MarketRegime, Signal
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)

# Regime → signal direction mapping
_REGIME_DIRECTION: dict[MarketRegime, int | None] = {
    MarketRegime.RISK_ON: 1,
    MarketRegime.RISK_OFF: -1,
    MarketRegime.STRESSED: -1,
    MarketRegime.TRANSITION: None,  # No signal during transition
}

# Confidence floor by regime (stressed has higher confidence)
_REGIME_CONFIDENCE_BASE: dict[MarketRegime, float] = {
    MarketRegime.RISK_ON: 0.50,
    MarketRegime.RISK_OFF: 0.55,
    MarketRegime.STRESSED: 0.65,
    MarketRegime.TRANSITION: 0.0,
}


@register_agent("regime_hmm")
class RegimeDetectorAgent(Agent):
    """Generates signals based on HMM market regime classification.

    On the first call with sufficient data, fits the HMM model.
    Subsequent calls use the fitted model for regime prediction.
    Signals are generated only when a clear regime is detected
    (not during transitions).
    """

    def __init__(
        self,
        agent_id: str = "regime_detector_agent",
        cluster: str = "market_intelligence",
        n_regimes: int = 4,
        lookback: int = 20,
        min_candles: int = 60,
    ) -> None:
        super().__init__(agent_id=agent_id, cluster=cluster)
        self.min_candles = min_candles
        self.detector = HMMRegimeDetector(
            n_regimes=n_regimes,
            lookback=lookback,
        )
        self.provider = AsterDataProvider()
        self._last_regime: MarketRegime | None = None

    def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_klines = context.get("klines_data")
        symbol = context.get("symbol", "BTCUSDT")

        if raw_klines is None:
            return {"signal": None, "reason": "no_klines_data"}

        candles = self.provider.parse_klines(raw_klines, symbol)
        if len(candles) < self.min_candles:
            return {"signal": None, "reason": f"insufficient_data: {len(candles)}"}

        closes = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # Fit on first call with sufficient data
        if not self.detector.is_fitted:
            self.detector.fit(closes, volumes)

        state = self.detector.predict_from_prices(closes, volumes)
        direction = _REGIME_DIRECTION.get(state.regime)

        # Track regime transitions
        regime_changed = self._last_regime is not None and state.regime != self._last_regime
        self._last_regime = state.regime

        if direction is None:
            return {
                "signal": None,
                "reason": "transition_regime",
                "regime": state.regime.value,
                "confidence": state.confidence,
            }

        # Boost confidence on regime transitions
        base_confidence = _REGIME_CONFIDENCE_BASE.get(state.regime, 0.50)
        confidence = min(0.85, base_confidence + state.confidence * 0.20)
        if regime_changed:
            confidence = min(0.90, confidence + 0.10)
        confidence = max(0.35, confidence)

        expected_return = confidence * 0.015
        price = candles[-1].close

        direction_str = "long" if direction == 1 else "short"
        signal = Signal(
            signal_id=new_id("sig"),
            agent_id=self.agent_id,
            symbol=symbol,
            strategy="regime_hmm",
            thesis=(
                f"Regime {state.regime.value}: {direction_str}, "
                f"HMM confidence={state.confidence:.2f}, "
                f"vol={state.features.get('volatility', 0):.4f}"
            ),
            direction=direction,
            confidence=confidence,
            expected_return=expected_return,
            horizon_minutes=360,
            liquidity_score=0.8,
            regime=state.regime,
            created_at=utc_now(),
            reference_price=price,
        )

        logger.info(
            "Regime signal generated",
            extra={
                "extra_json": {
                    "symbol": symbol,
                    "regime": state.regime.value,
                    "direction": direction_str,
                    "confidence": round(confidence, 4),
                    "hmm_confidence": round(state.confidence, 4),
                    "regime_changed": regime_changed,
                    "uses_hmm": self.detector.uses_hmm,
                }
            },
        )

        return {
            "signal": signal,
            "regime": state.regime.value,
            "regime_state": state,
            "regime_changed": regime_changed,
        }

    def propose(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.analyze(context)

    def validate(self, context: dict[str, Any]) -> bool:
        raw_klines = context.get("klines_data")
        if raw_klines is None:
            return False
        candles = self.provider.parse_klines(raw_klines, context.get("symbol", ""))
        return len(candles) >= self.min_candles
