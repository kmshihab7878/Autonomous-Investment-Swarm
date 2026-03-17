from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

from aiswarm.mandates.models import Mandate
from aiswarm.orchestration.memory import MandatePnLTracker
from aiswarm.portfolio.exposure import ExposureManager
from aiswarm.risk.drawdown import DrawdownGuard
from aiswarm.risk.kill_switch import KillSwitch
from aiswarm.risk.leverage import LeverageGuard
from aiswarm.risk.liquidity import LiquidityGuard
from aiswarm.types.orders import Order, OrderStatus
from aiswarm.types.portfolio import PortfolioSnapshot
from aiswarm.utils.logging import get_logger
from aiswarm.utils.secrets import get_secrets_provider

logger = get_logger(__name__)

TOKEN_TTL_SECONDS = 300  # 5 minutes

# Default key identifier when none is configured via environment.
_DEFAULT_KEY_ID = "v1"


def _hmac_secret() -> str:
    """Return the current (primary) HMAC secret. Fails closed if unset."""
    secret = get_secrets_provider().get_secret("AIS_RISK_HMAC_SECRET") or ""
    if not secret:
        raise RuntimeError(
            "AIS_RISK_HMAC_SECRET is not set (checked env vars and secrets provider). "
            "Risk token signing requires an explicit secret — no fallback is allowed."
        )
    return secret


def _hmac_secret_previous() -> str | None:
    """Return the previous HMAC secret for key rotation, or None if not configured."""
    return get_secrets_provider().get_secret("AIS_RISK_HMAC_SECRET_PREVIOUS") or None


def _hmac_key_id() -> str:
    """Return the key identifier for the current HMAC secret."""
    return get_secrets_provider().get_secret("AIS_RISK_HMAC_KEY_ID") or _DEFAULT_KEY_ID


def sign_risk_token(order_id: str) -> str:
    """Create an HMAC-signed, time-bound risk approval token.

    Token format: ``{order_id}:{timestamp}:{key_id}:{signature}``

    Always signs with the current (primary) key.  The ``key_id`` field
    makes the token self-identifying so verification can select the
    correct key during a rotation window.
    """
    timestamp = str(int(time.time()))
    key_id = _hmac_key_id()
    payload = f"{order_id}:{timestamp}:{key_id}"
    sig = hmac.new(
        _hmac_secret().encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}:{sig}"


def _verify_with_secret(
    tok_order_id: str,
    timestamp_str: str,
    key_id: str,
    sig: str,
    secret: str,
) -> bool:
    """Check HMAC signature against a specific secret."""
    payload = f"{tok_order_id}:{timestamp_str}:{key_id}"
    expected_sig = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(sig, expected_sig)


def verify_risk_token(token: str, order_id: str) -> bool:
    """Verify that a risk approval token is valid and not expired.

    Tries the current HMAC key first.  If verification fails and a
    previous key is configured (``AIS_RISK_HMAC_SECRET_PREVIOUS``),
    retries with the previous key so tokens issued before a key
    rotation can still be honoured within their TTL window.
    """
    try:
        parts = token.split(":")
        if len(parts) != 4:
            return False
        tok_order_id, timestamp_str, key_id, sig = parts
        if tok_order_id != order_id:
            return False

        # TTL check (shared across both keys)
        elapsed = time.time() - int(timestamp_str)
        if elapsed > TOKEN_TTL_SECONDS or elapsed < 0:
            return False

        # Try current key
        if _verify_with_secret(tok_order_id, timestamp_str, key_id, sig, _hmac_secret()):
            return True

        # Try previous key if configured (rotation window)
        previous = _hmac_secret_previous()
        if previous is not None:
            if _verify_with_secret(tok_order_id, timestamp_str, key_id, sig, previous):
                logger.warning(
                    "Risk token verified with previous HMAC key — rotation in progress",
                    extra={
                        "extra_json": {
                            "order_id": order_id,
                            "key_id": key_id,
                        }
                    },
                )
                return True

        return False
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class RiskApproval:
    approved: bool
    reasons: list[str]
    order: Order | None


class RiskEngine:
    """Unified risk validation engine.

    Integrates all risk guards: KillSwitch, ExposureManager, DrawdownGuard,
    LeverageGuard, and LiquidityGuard. No order may leave this engine
    approved without passing every active control.
    """

    def __init__(
        self,
        max_position_weight: float,
        max_gross_exposure: float,
        max_daily_loss: float,
        max_rolling_drawdown: float = 0.05,
        max_leverage: float = 1.0,
        min_liquidity_score: float = 0.50,
    ) -> None:
        self.exposure = ExposureManager(max_position_weight, max_gross_exposure)
        self.kill_switch = KillSwitch(max_daily_loss)
        self.drawdown_guard = DrawdownGuard()
        self.leverage_guard = LeverageGuard()
        self.liquidity_guard = LiquidityGuard()

        self.max_rolling_drawdown = max_rolling_drawdown
        self.max_leverage = max_leverage
        self.min_liquidity_score = min_liquidity_score

    def validate(
        self,
        order: Order,
        snapshot: PortfolioSnapshot | None,
        daily_pnl_fraction: float,
        rolling_drawdown: float = 0.0,
        current_leverage: float = 0.0,
        liquidity_score: float = 1.0,
    ) -> RiskApproval:
        reasons: list[str] = []

        # Kill switch: daily loss circuit breaker
        if self.kill_switch.triggered(daily_pnl_fraction):
            reasons.append("kill_switch_triggered")
            logger.warning(
                "Kill switch triggered",
                extra={"extra_json": {"daily_pnl": daily_pnl_fraction}},
            )

        # Exposure limits: position weight and gross exposure
        ok, reason = self.exposure.check_order(order, snapshot)
        if not ok:
            reasons.append(reason)

        # Drawdown guard: rolling drawdown limit
        if self.drawdown_guard.breached(rolling_drawdown, self.max_rolling_drawdown):
            reasons.append(
                f"drawdown_breached: {rolling_drawdown:.4f} >= {self.max_rolling_drawdown:.4f}"
            )

        # Leverage guard: max leverage check
        if self.leverage_guard.breached(current_leverage, self.max_leverage):
            reasons.append(f"leverage_breached: {current_leverage:.2f} > {self.max_leverage:.2f}")

        # Liquidity guard: minimum liquidity score
        if self.liquidity_guard.breached(liquidity_score, self.min_liquidity_score):
            reasons.append(
                f"liquidity_insufficient: {liquidity_score:.4f} < {self.min_liquidity_score:.4f}"
            )

        if reasons:
            logger.info(
                "Risk validation rejected order",
                extra={"extra_json": {"order_id": order.order_id, "reasons": reasons}},
            )
            return RiskApproval(approved=False, reasons=reasons, order=None)

        # All checks passed — sign and approve
        token = sign_risk_token(order.order_id)
        approved_order = order.model_copy(
            update={
                "risk_approval_token": token,
                "status": OrderStatus.APPROVED,
            }
        )
        logger.info(
            "Risk validation approved order",
            extra={"extra_json": {"order_id": order.order_id}},
        )
        return RiskApproval(approved=True, reasons=["approved"], order=approved_order)

    def validate_with_mandate(
        self,
        order: Order,
        snapshot: PortfolioSnapshot | None,
        daily_pnl_fraction: float,
        rolling_drawdown: float,
        current_leverage: float,
        liquidity_score: float,
        mandate: Mandate,
        mandate_tracker: MandatePnLTracker,
    ) -> RiskApproval:
        """Run global risk checks, then layer on mandate-specific constraints.

        The stricter-of-two wins for each check.
        """
        # Run global checks first
        global_result = self.validate(
            order=order,
            snapshot=snapshot,
            daily_pnl_fraction=daily_pnl_fraction,
            rolling_drawdown=rolling_drawdown,
            current_leverage=current_leverage,
            liquidity_score=liquidity_score,
        )
        if not global_result.approved:
            return global_result

        # Now apply mandate-specific checks
        reasons: list[str] = []
        budget = mandate.risk_budget

        # Mandate daily loss check
        if mandate_tracker.daily_pnl < 0:
            mandate_daily_loss_frac = abs(mandate_tracker.daily_pnl) / budget.max_capital
            if mandate_daily_loss_frac >= budget.max_daily_loss:
                reasons.append(
                    f"mandate_daily_loss_breached: "
                    f"{mandate_daily_loss_frac:.4f} >= {budget.max_daily_loss:.4f}"
                )

        # Mandate drawdown check
        if mandate_tracker.drawdown >= budget.max_drawdown:
            reasons.append(
                f"mandate_drawdown_breached: "
                f"{mandate_tracker.drawdown:.4f} >= {budget.max_drawdown:.4f}"
            )

        # Mandate exposure/capital check
        new_exposure = mandate_tracker.gross_exposure + order.notional
        if new_exposure > budget.max_capital:
            reasons.append(
                f"mandate_capital_exceeded: {new_exposure:.2f} > {budget.max_capital:.2f}"
            )

        # Per-position notional check
        if order.notional > budget.effective_position_notional:
            reasons.append(
                f"mandate_position_notional_exceeded: "
                f"{order.notional:.2f} > {budget.effective_position_notional:.2f}"
            )

        if reasons:
            logger.info(
                "Mandate risk validation rejected order",
                extra={
                    "extra_json": {
                        "order_id": order.order_id,
                        "mandate_id": mandate.mandate_id,
                        "reasons": reasons,
                    }
                },
            )
            return RiskApproval(approved=False, reasons=reasons, order=None)

        # All mandate checks passed — use the already-signed order from global validation
        assert global_result.order is not None
        return global_result
