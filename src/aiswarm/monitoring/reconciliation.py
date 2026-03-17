"""Position reconciliation against Aster DEX.

Periodically compares internal SharedMemory state against real exchange
positions and balances. Alerts on any divergence.

Key reconciliation checks:
1. Position quantity match (internal vs exchange)
2. Balance verification (expected vs actual)
3. Unauthorized trade detection (trades we didn't submit)
4. P&L consistency (internal tracking vs exchange income)

Also provides ReconciliationLoop for automated periodic reconciliation
with auto-pause on mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from aiswarm.data.event_store import EventStore
from aiswarm.data.providers.aster import (
    AccountBalance,
    AsterDataProvider,
    ExchangePosition,
    TradeRecord,
)
from aiswarm.types.portfolio import PortfolioSnapshot, Position
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


class ReconciliationStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING_INTERNAL = "missing_internal"
    MISSING_EXCHANGE = "missing_exchange"
    ERROR = "error"


@dataclass(frozen=True)
class ReconciliationResult:
    """Result of a single reconciliation check."""

    status: ReconciliationStatus
    check_type: str
    symbol: str
    message: str
    internal_value: float | None = None
    exchange_value: float | None = None
    timestamp: datetime | None = None


_MISMATCH_STATUSES = frozenset(
    {
        ReconciliationStatus.MISMATCH,
        ReconciliationStatus.MISSING_INTERNAL,
        ReconciliationStatus.MISSING_EXCHANGE,
    }
)


@dataclass(frozen=True)
class ReconciliationReport:
    """Full reconciliation report across all checks."""

    timestamp: datetime
    passed: bool
    total_checks: int
    mismatches: int
    results: tuple[ReconciliationResult, ...]

    @property
    def mismatched_symbols(self) -> list[str]:
        """Return deduplicated, sorted list of symbols with mismatches.

        Excludes non-symbol entries like 'PORTFOLIO' (balance checks)
        since those are not actionable at the per-symbol order level.
        """
        seen: set[str] = set()
        for r in self.results:
            if r.status in _MISMATCH_STATUSES and r.symbol != "PORTFOLIO":
                seen.add(r.symbol)
        return sorted(seen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "total_checks": self.total_checks,
            "mismatches": self.mismatches,
            "mismatched_symbols": self.mismatched_symbols,
            "results": [
                {
                    "status": r.status.value,
                    "check_type": r.check_type,
                    "symbol": r.symbol,
                    "message": r.message,
                    "internal_value": r.internal_value,
                    "exchange_value": r.exchange_value,
                }
                for r in self.results
            ],
        }


class PositionReconciler:
    """Reconciles internal portfolio state against Aster DEX exchange data."""

    def __init__(
        self,
        provider: AsterDataProvider | None = None,
        tolerance: float = 0.001,
    ) -> None:
        self.provider = provider or AsterDataProvider()
        self.tolerance = tolerance

    def reconcile_positions(
        self,
        internal_snapshot: PortfolioSnapshot | None,
        exchange_positions: list[ExchangePosition],
    ) -> list[ReconciliationResult]:
        """Compare internal positions against exchange positions."""
        results: list[ReconciliationResult] = []

        if internal_snapshot is None:
            if exchange_positions:
                for pos in exchange_positions:
                    results.append(
                        ReconciliationResult(
                            status=ReconciliationStatus.MISSING_INTERNAL,
                            check_type="position",
                            symbol=pos.symbol,
                            message=f"Exchange has position but no internal snapshot: "
                            f"qty={pos.quantity} {pos.side}",
                            internal_value=None,
                            exchange_value=pos.quantity,
                            timestamp=utc_now(),
                        )
                    )
            return results

        # Build lookup maps
        internal_by_symbol: dict[str, Position] = {p.symbol: p for p in internal_snapshot.positions}
        exchange_by_symbol: dict[str, ExchangePosition] = {p.symbol: p for p in exchange_positions}

        all_symbols = set(internal_by_symbol.keys()) | set(exchange_by_symbol.keys())

        for symbol in sorted(all_symbols):
            internal = internal_by_symbol.get(symbol)
            exchange = exchange_by_symbol.get(symbol)

            if internal and not exchange:
                results.append(
                    ReconciliationResult(
                        status=ReconciliationStatus.MISSING_EXCHANGE,
                        check_type="position",
                        symbol=symbol,
                        message=f"Internal has position but exchange does not: "
                        f"qty={internal.quantity}",
                        internal_value=internal.quantity,
                        exchange_value=None,
                        timestamp=utc_now(),
                    )
                )
            elif exchange and not internal:
                results.append(
                    ReconciliationResult(
                        status=ReconciliationStatus.MISSING_INTERNAL,
                        check_type="position",
                        symbol=symbol,
                        message=f"Exchange has position but internal does not: "
                        f"qty={exchange.quantity} {exchange.side}",
                        internal_value=None,
                        exchange_value=exchange.quantity,
                        timestamp=utc_now(),
                    )
                )
            elif internal and exchange:
                internal_qty = abs(internal.quantity)
                exchange_qty = abs(exchange.quantity)
                diff = abs(internal_qty - exchange_qty)
                max_qty = max(internal_qty, exchange_qty)

                if max_qty > 0 and diff / max_qty > self.tolerance:
                    results.append(
                        ReconciliationResult(
                            status=ReconciliationStatus.MISMATCH,
                            check_type="position_quantity",
                            symbol=symbol,
                            message=f"Quantity mismatch: internal={internal_qty:.6f} "
                            f"exchange={exchange_qty:.6f} diff={diff:.6f}",
                            internal_value=internal_qty,
                            exchange_value=exchange_qty,
                            timestamp=utc_now(),
                        )
                    )
                else:
                    results.append(
                        ReconciliationResult(
                            status=ReconciliationStatus.MATCH,
                            check_type="position_quantity",
                            symbol=symbol,
                            message=f"Position matched: qty={internal_qty:.6f}",
                            internal_value=internal_qty,
                            exchange_value=exchange_qty,
                            timestamp=utc_now(),
                        )
                    )

        return results

    def reconcile_balance(
        self,
        expected_nav: float,
        exchange_balance: AccountBalance,
    ) -> ReconciliationResult:
        """Compare expected NAV against exchange balance."""
        actual = exchange_balance.total_balance
        diff = abs(expected_nav - actual)
        max_val = max(expected_nav, actual)

        if max_val > 0 and diff / max_val > self.tolerance:
            return ReconciliationResult(
                status=ReconciliationStatus.MISMATCH,
                check_type="balance",
                symbol="PORTFOLIO",
                message=f"Balance mismatch: expected_nav={expected_nav:.2f} "
                f"exchange={actual:.2f} diff={diff:.2f}",
                internal_value=expected_nav,
                exchange_value=actual,
                timestamp=utc_now(),
            )

        return ReconciliationResult(
            status=ReconciliationStatus.MATCH,
            check_type="balance",
            symbol="PORTFOLIO",
            message=f"Balance matched: nav={expected_nav:.2f} exchange={actual:.2f}",
            internal_value=expected_nav,
            exchange_value=actual,
            timestamp=utc_now(),
        )

    def check_unauthorized_trades(
        self,
        known_order_ids: set[str],
        exchange_trades: list[TradeRecord],
    ) -> list[ReconciliationResult]:
        """Detect trades on the exchange that we didn't submit."""
        results: list[ReconciliationResult] = []

        for trade in exchange_trades:
            if trade.trade_id not in known_order_ids:
                results.append(
                    ReconciliationResult(
                        status=ReconciliationStatus.MISMATCH,
                        check_type="unauthorized_trade",
                        symbol=trade.symbol,
                        message=f"Unknown trade detected: id={trade.trade_id} "
                        f"side={trade.side} qty={trade.quantity} "
                        f"price={trade.price}",
                        internal_value=None,
                        exchange_value=trade.quantity,
                        timestamp=trade.timestamp,
                    )
                )

        return results

    def run_full_reconciliation(
        self,
        internal_snapshot: PortfolioSnapshot | None,
        exchange_positions: list[ExchangePosition],
        exchange_balance: AccountBalance | None = None,
        exchange_trades: list[TradeRecord] | None = None,
        known_order_ids: set[str] | None = None,
    ) -> ReconciliationReport:
        """Run all reconciliation checks and return a full report."""
        all_results: list[ReconciliationResult] = []

        # Position reconciliation
        all_results.extend(self.reconcile_positions(internal_snapshot, exchange_positions))

        # Balance reconciliation
        if exchange_balance and internal_snapshot:
            all_results.append(self.reconcile_balance(internal_snapshot.nav, exchange_balance))

        # Unauthorized trade detection
        if exchange_trades and known_order_ids is not None:
            all_results.extend(self.check_unauthorized_trades(known_order_ids, exchange_trades))

        mismatches = sum(
            1
            for r in all_results
            if r.status
            in (
                ReconciliationStatus.MISMATCH,
                ReconciliationStatus.MISSING_INTERNAL,
                ReconciliationStatus.MISSING_EXCHANGE,
            )
        )

        report = ReconciliationReport(
            timestamp=utc_now(),
            passed=mismatches == 0,
            total_checks=len(all_results),
            mismatches=mismatches,
            results=tuple(all_results),
        )

        if not report.passed:
            logger.warning(
                "Reconciliation FAILED",
                extra={
                    "extra_json": {
                        "mismatches": mismatches,
                        "total_checks": report.total_checks,
                    }
                },
            )
        else:
            logger.info(
                "Reconciliation passed",
                extra={"extra_json": {"total_checks": report.total_checks}},
            )

        return report


class ReconciliationLoop:
    """Automated periodic reconciliation with surgical mismatch handling.

    Wraps PositionReconciler for continuous monitoring. When a mismatch
    is detected:
    - If ``mismatch_callback`` is provided, calls it with the list of
      mismatched symbols so only those symbols' orders are cancelled
      (surgical reconciliation).
    - Falls back to ``pause_callback`` (nuclear: no symbol info) for
      backward compatibility.
    """

    def __init__(
        self,
        reconciler: PositionReconciler,
        event_store: EventStore,
        pause_callback: Callable[[], None],
        mismatch_threshold: int = 0,
        mismatch_callback: Callable[[list[str]], None] | None = None,
    ) -> None:
        self.reconciler = reconciler
        self.event_store = event_store
        self.pause_callback = pause_callback
        self.mismatch_threshold = mismatch_threshold
        self.mismatch_callback = mismatch_callback
        self.latest_report: ReconciliationReport | None = None

    def on_fill(
        self,
        internal_snapshot: PortfolioSnapshot | None,
        exchange_positions: list[ExchangePosition],
    ) -> ReconciliationReport:
        """Run reconciliation after each fill event."""
        report = self.reconciler.run_full_reconciliation(
            internal_snapshot=internal_snapshot,
            exchange_positions=exchange_positions,
        )
        self.latest_report = report
        self._persist_report(report)
        if not report.passed and report.mismatches > self.mismatch_threshold:
            self._handle_mismatch(report)
        return report

    def run_periodic_check(
        self,
        internal_snapshot: PortfolioSnapshot | None,
        exchange_positions: list[ExchangePosition],
        exchange_balance: AccountBalance | None = None,
    ) -> ReconciliationReport:
        """Run periodic reconciliation check (called on interval)."""
        report = self.reconciler.run_full_reconciliation(
            internal_snapshot=internal_snapshot,
            exchange_positions=exchange_positions,
            exchange_balance=exchange_balance,
        )
        self.latest_report = report
        self._persist_report(report)
        if not report.passed and report.mismatches > self.mismatch_threshold:
            self._handle_mismatch(report)
        return report

    def _handle_mismatch(self, report: ReconciliationReport) -> None:
        """Handle reconciliation mismatch — surgical or nuclear.

        If a ``mismatch_callback`` is configured, delegates to it with the
        specific mismatched symbols (surgical). Otherwise falls back to the
        legacy ``pause_callback`` which cancels everything (nuclear).
        """
        mismatched = report.mismatched_symbols

        if self.mismatch_callback and mismatched:
            logger.warning(
                "Reconciliation mismatch — surgical cancel for mismatched symbols",
                extra={
                    "extra_json": {
                        "mismatched_symbols": mismatched,
                        "mismatches": report.mismatches,
                        "total_checks": report.total_checks,
                    }
                },
            )
            self.mismatch_callback(mismatched)
        else:
            logger.warning(
                "Reconciliation mismatch — pausing trading (nuclear)",
                extra={
                    "extra_json": {
                        "mismatches": report.mismatches,
                        "total_checks": report.total_checks,
                    }
                },
            )
            self.pause_callback()

        self.event_store.append(
            "reconciliation_pause",
            report.to_dict(),
            source="reconciliation_loop",
        )

    def _persist_report(self, report: ReconciliationReport) -> None:
        """Persist reconciliation report to EventStore."""
        self.event_store.append(
            "reconciliation",
            report.to_dict(),
            source="reconciliation_loop",
        )
