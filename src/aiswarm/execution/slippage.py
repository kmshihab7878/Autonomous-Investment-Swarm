"""Slippage models for order execution cost estimation.

Provides pluggable slippage estimation used by the backtest engine
and portfolio allocator to model realistic execution costs.

Models:
  - FixedSlippage: Constant bps (baseline)
  - VolumeWeightedSlippage: Scales with order size relative to book depth
  - HistoricalSlippage: Calibrated from realized vs expected fill prices
  - RegimeAwareSlippage: Adjusts slippage based on market regime
  - CompositeSlippage: Weighted combination of multiple models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class SlippageRegime(str, Enum):
    """Market regime for slippage estimation."""

    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESSED = "stressed"


@dataclass(frozen=True)
class SlippageEstimate:
    """Result of a slippage estimation."""

    bps: float
    model_name: str
    details: dict[str, Any] = field(default_factory=dict)


class SlippageModel(ABC):
    """Abstract base class for slippage models."""

    @abstractmethod
    def estimate_bps(
        self,
        notional: float,
        **kwargs: Any,
    ) -> SlippageEstimate:
        """Estimate slippage in basis points.

        Args:
            notional: Order notional value in USD.
            **kwargs: Model-specific parameters (orderbook_depth,
                      avg_volume, regime, historical_fills, etc.)

        Returns:
            SlippageEstimate with bps, model name, and computation details.
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Constant slippage in basis points.

    Simple baseline model. Useful for conservative estimates or when
    market microstructure data is unavailable.
    """

    def __init__(self, bps: float = 5.0) -> None:
        self._bps = bps

    def estimate_bps(self, notional: float, **kwargs: Any) -> SlippageEstimate:
        return SlippageEstimate(
            bps=self._bps,
            model_name="fixed",
            details={"fixed_bps": self._bps},
        )


class VolumeWeightedSlippage(SlippageModel):
    """Slippage scales with order size relative to available liquidity.

    Uses a square-root market impact model:
        slippage_bps = base_bps + impact_coeff * sqrt(notional / depth)

    This captures the empirical observation that market impact grows
    sub-linearly with order size (Almgren-Chriss model).
    """

    def __init__(
        self,
        base_bps: float = 1.0,
        impact_coefficient: float = 10.0,
        min_bps: float = 0.5,
        max_bps: float = 50.0,
    ) -> None:
        self._base_bps = base_bps
        self._impact_coeff = impact_coefficient
        self._min_bps = min_bps
        self._max_bps = max_bps

    def estimate_bps(self, notional: float, **kwargs: Any) -> SlippageEstimate:
        orderbook_depth = kwargs.get("orderbook_depth", 1_000_000.0)

        if orderbook_depth <= 0:
            return SlippageEstimate(
                bps=self._max_bps,
                model_name="volume_weighted",
                details={"reason": "zero_depth", "depth": orderbook_depth},
            )

        # Square-root market impact
        participation_rate = notional / orderbook_depth
        impact = self._impact_coeff * (participation_rate**0.5)
        bps = self._base_bps + impact
        bps = max(self._min_bps, min(self._max_bps, bps))

        return SlippageEstimate(
            bps=bps,
            model_name="volume_weighted",
            details={
                "participation_rate": round(participation_rate, 6),
                "impact_bps": round(impact, 2),
                "base_bps": self._base_bps,
                "depth": orderbook_depth,
            },
        )


class HistoricalSlippage(SlippageModel):
    """Slippage calibrated from historical realized fills.

    Maintains a running estimate of actual slippage by comparing
    fill prices to reference prices (mid-price at signal time).
    Falls back to a default when insufficient history exists.
    """

    def __init__(
        self,
        default_bps: float = 5.0,
        min_samples: int = 10,
        ewma_alpha: float = 0.1,
    ) -> None:
        self._default_bps = default_bps
        self._min_samples = min_samples
        self._ewma_alpha = ewma_alpha
        self._ewma_bps: float | None = None
        self._sample_count: int = 0

    def record_fill(
        self,
        reference_price: float,
        fill_price: float,
        side: int,
    ) -> None:
        """Record an observed fill for calibration.

        Args:
            reference_price: Mid-price at signal time.
            fill_price: Actual execution price.
            side: 1 for buy, -1 for sell.
        """
        if reference_price <= 0:
            return

        # Slippage is adverse price movement
        if side == 1:  # Buy: slippage = fill above reference
            slippage_bps = (fill_price - reference_price) / reference_price * 10_000
        else:  # Sell: slippage = fill below reference
            slippage_bps = (reference_price - fill_price) / reference_price * 10_000

        slippage_bps = max(0.0, slippage_bps)

        if self._ewma_bps is None:
            self._ewma_bps = slippage_bps
        else:
            self._ewma_bps = (
                self._ewma_alpha * slippage_bps + (1 - self._ewma_alpha) * self._ewma_bps
            )
        self._sample_count += 1

    def estimate_bps(self, notional: float, **kwargs: Any) -> SlippageEstimate:
        if self._ewma_bps is None or self._sample_count < self._min_samples:
            return SlippageEstimate(
                bps=self._default_bps,
                model_name="historical",
                details={
                    "reason": "insufficient_samples",
                    "samples": self._sample_count,
                    "min_required": self._min_samples,
                },
            )

        return SlippageEstimate(
            bps=self._ewma_bps,
            model_name="historical",
            details={
                "ewma_bps": round(self._ewma_bps, 2),
                "samples": self._sample_count,
            },
        )


class RegimeAwareSlippage(SlippageModel):
    """Adjusts a base model's slippage estimate by market regime.

    In stressed/volatile markets, slippage is empirically 2-5x higher
    due to wider spreads, thinner books, and adverse selection.
    """

    def __init__(
        self,
        base_model: SlippageModel,
        volatile_multiplier: float = 2.0,
        stressed_multiplier: float = 3.5,
    ) -> None:
        self._base = base_model
        self._multipliers = {
            SlippageRegime.NORMAL: 1.0,
            SlippageRegime.VOLATILE: volatile_multiplier,
            SlippageRegime.STRESSED: stressed_multiplier,
        }

    def estimate_bps(self, notional: float, **kwargs: Any) -> SlippageEstimate:
        regime_str = kwargs.get("regime", "normal")
        try:
            regime = SlippageRegime(regime_str)
        except ValueError:
            regime = SlippageRegime.NORMAL

        base_estimate = self._base.estimate_bps(notional, **kwargs)
        multiplier = self._multipliers[regime]
        adjusted_bps = base_estimate.bps * multiplier

        return SlippageEstimate(
            bps=adjusted_bps,
            model_name="regime_aware",
            details={
                "regime": regime.value,
                "multiplier": multiplier,
                "base_bps": base_estimate.bps,
                "base_model": base_estimate.model_name,
            },
        )


class CompositeSlippage(SlippageModel):
    """Weighted combination of multiple slippage models.

    Produces a blended estimate by averaging across models, optionally
    weighted. This allows combining a fast model (fixed/volume) with
    a slow-adapting model (historical) for robustness.
    """

    def __init__(
        self,
        models: list[tuple[SlippageModel, float]],
    ) -> None:
        """Args:
        models: List of (model, weight) tuples. Weights are normalized.
        """
        total = sum(w for _, w in models)
        if total <= 0:
            raise ValueError("Weights must sum to a positive number")
        self._models = [(m, w / total) for m, w in models]

    def estimate_bps(self, notional: float, **kwargs: Any) -> SlippageEstimate:
        weighted_bps = 0.0
        component_details: list[dict[str, Any]] = []

        for model, weight in self._models:
            est = model.estimate_bps(notional, **kwargs)
            weighted_bps += est.bps * weight
            component_details.append(
                {
                    "model": est.model_name,
                    "bps": round(est.bps, 2),
                    "weight": round(weight, 3),
                }
            )

        return SlippageEstimate(
            bps=weighted_bps,
            model_name="composite",
            details={"components": component_details},
        )


def default_slippage_model() -> SlippageModel:
    """Build the default slippage model stack.

    Returns a regime-aware composite of volume-weighted and fixed models.
    """
    volume = VolumeWeightedSlippage(base_bps=1.0, impact_coefficient=10.0)
    fixed = FixedSlippage(bps=3.0)
    composite = CompositeSlippage(models=[(volume, 0.7), (fixed, 0.3)])
    return RegimeAwareSlippage(base_model=composite)
