from __future__ import annotations

from typing import Iterable

from aiswarm.mandates.validator import MandateValidator
from aiswarm.orchestration.arbitration import WeightedArbitration
from aiswarm.orchestration.memory import SharedMemory
from aiswarm.portfolio.allocator import PortfolioAllocator
from aiswarm.risk.limits import RiskEngine, verify_risk_token
from aiswarm.session.manager import SessionManager
from aiswarm.types.decisions import DecisionLog
from aiswarm.types.market import Signal
from aiswarm.types.orders import Order, OrderStatus
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import append_jsonl, get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class Coordinator:
    def __init__(
        self,
        arbitration: WeightedArbitration,
        allocator: PortfolioAllocator,
        risk_engine: RiskEngine,
        memory: SharedMemory,
        decision_log_path: str,
        mandate_validator: MandateValidator | None = None,
        session_manager: SessionManager | None = None,
        staging_enabled: bool = False,
    ) -> None:
        self.arbitration = arbitration
        self.allocator = allocator
        self.risk_engine = risk_engine
        self.memory = memory
        self.decision_log_path = decision_log_path
        self.mandate_validator = mandate_validator
        self.session_manager = session_manager
        self.staging_enabled = staging_enabled
        self._staged_orders: dict[str, Order] = {}

    def coordinate(self, signals: Iterable[Signal]) -> Order | None:
        # Session gate: check if trading is allowed
        if self.session_manager is not None and not self.session_manager.is_trading_allowed:
            logger.info("Session gate closed — skipping signal processing")
            return None

        signal_list = list(signals)
        selected = self.arbitration.select_signal(signal_list)
        if selected is None:
            return None
        order = self.allocator.order_from_signal(selected, self.memory.latest_snapshot)

        # Mandate validation (if configured)
        mandate = None
        if self.mandate_validator is not None:
            validation = self.mandate_validator.validate_order(order)
            if not validation.ok:
                decision = DecisionLog(
                    decision_id=new_id("decision"),
                    timestamp=utc_now(),
                    decision_type="order_intent",
                    summary=f"Mandate rejected: {validation.reason}",
                    agent_votes={s.agent_id: s.confidence for s in signal_list},
                    selected_signal_id=selected.signal_id,
                    selected_order_id=order.order_id,
                    risk_passed=False,
                    risk_reasons=(validation.reason,),
                )
                append_jsonl(self.decision_log_path, decision.model_dump())
                return None
            mandate = validation.mandate
            # Stamp mandate_id onto order
            order = order.model_copy(update={"mandate_id": mandate.mandate_id})  # type: ignore[union-attr]

        # Risk validation — with mandate if available
        if mandate is not None:
            tracker = self.memory.get_mandate_tracker(mandate.mandate_id)
            approval = self.risk_engine.validate_with_mandate(
                order=order,
                snapshot=self.memory.latest_snapshot,
                daily_pnl_fraction=self.memory.latest_pnl,
                rolling_drawdown=self.memory.rolling_drawdown,
                current_leverage=self.memory.current_leverage,
                liquidity_score=selected.liquidity_score,
                mandate=mandate,
                mandate_tracker=tracker,
            )
        else:
            approval = self.risk_engine.validate(
                order=order,
                snapshot=self.memory.latest_snapshot,
                daily_pnl_fraction=self.memory.latest_pnl,
                rolling_drawdown=self.memory.rolling_drawdown,
                current_leverage=self.memory.current_leverage,
                liquidity_score=selected.liquidity_score,
            )

        decision = DecisionLog(
            decision_id=new_id("decision"),
            timestamp=utc_now(),
            decision_type="order_intent",
            summary=f"Selected {selected.symbol} from {selected.agent_id}",
            agent_votes={s.agent_id: s.confidence for s in signal_list},
            selected_signal_id=selected.signal_id,
            selected_order_id=order.order_id,
            risk_passed=approval.approved,
            risk_reasons=tuple(approval.reasons),
        )
        append_jsonl(self.decision_log_path, decision.model_dump())

        if not approval.approved:
            return None

        approved_order = approval.order
        if approved_order is None:
            return None

        # Staging mode: hold order for operator review
        if self.staging_enabled:
            staged = approved_order.model_copy(update={"status": OrderStatus.STAGED})
            self._staged_orders[staged.order_id] = staged
            logger.info(
                "Order staged for review",
                extra={"extra_json": {"order_id": staged.order_id}},
            )
            return None

        return approved_order

    # --- Staging API ---

    def get_staged_orders(self) -> list[Order]:
        """Return all currently staged orders, removing expired ones."""
        valid: list[Order] = []
        expired: list[str] = []
        for order_id, order in self._staged_orders.items():
            if order.risk_approval_token and verify_risk_token(
                order.risk_approval_token, order.order_id
            ):
                valid.append(order)
            else:
                expired.append(order_id)
        for oid in expired:
            del self._staged_orders[oid]
        return valid

    def execute_staged(self, order_id: str) -> Order | None:
        """Execute a staged order if its risk token is still valid.

        If the token has expired, re-validates the order through the risk engine.
        """
        order = self._staged_orders.pop(order_id, None)
        if order is None:
            return None
        # Check if token is still valid
        if order.risk_approval_token and verify_risk_token(
            order.risk_approval_token, order.order_id
        ):
            return order.model_copy(update={"status": OrderStatus.APPROVED})
        # Token expired — need re-validation
        logger.info(
            "Staged order token expired, re-validating",
            extra={"extra_json": {"order_id": order_id}},
        )
        return None

    def inject_external_signal(self, signal: Signal) -> Order | None:
        """Process a single externally-injected signal through the full pipeline.

        Used for TradingView webhooks and other external signal sources.
        The signal bypasses arbitration (it's already selected) but goes
        through allocation, mandate validation, and risk checks.
        """
        return self.coordinate([signal])

    def reject_staged(self, order_id: str, reason: str) -> Order | None:
        """Reject a staged order."""
        order = self._staged_orders.pop(order_id, None)
        if order is None:
            return None
        rejected = order.model_copy(update={"status": OrderStatus.REJECTED})
        logger.info(
            "Staged order rejected",
            extra={"extra_json": {"order_id": order_id, "reason": reason}},
        )
        return rejected
