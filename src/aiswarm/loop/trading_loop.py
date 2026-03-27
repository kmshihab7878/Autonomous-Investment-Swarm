"""Autonomous trading loop — wires all components into a self-running system.

Cycle: sync portfolio → fetch market data → run agents → coordinate →
       submit orders → sync fills → reconcile → update metrics.

The loop is session-aware and respects the kill switch, graceful shutdown,
and configurable intervals for each operation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from aiswarm.agents.base import Agent
from aiswarm.evolution.autoresearch import AutoresearchLoop
from aiswarm.evolution.darwinian import DarwinianWeightManager
from aiswarm.exchange.provider import ExchangeProvider
from aiswarm.execution.account_setup import AccountSetupService
from aiswarm.execution.fill_tracker import FillTracker
from aiswarm.execution.live_executor import LiveOrderExecutor
from aiswarm.execution.portfolio_sync import PortfolioSyncService
from aiswarm.loop.config import LoopConfig
from aiswarm.loop.market_data import MarketDataService
from aiswarm.monitoring import metrics as m
from aiswarm.monitoring.alerts import AlertDispatcher
from aiswarm.monitoring.reconciliation import ReconciliationLoop
from aiswarm.orchestration.coordinator import Coordinator
from aiswarm.orchestration.janus import JanusMetaWeighting
from aiswarm.orchestration.memory import SharedMemory
from aiswarm.resilience.shutdown import GracefulShutdown
from aiswarm.risk.stop_loss import StopLossMonitor
from aiswarm.session.manager import SessionManager
from aiswarm.types.market import Signal
from aiswarm.types.risk import RiskEvent, RiskSeverity
from aiswarm.utils.ids import new_id
from aiswarm.utils.logging import get_logger
from aiswarm.utils.time import utc_now

logger = get_logger(__name__)


@dataclass(frozen=True)
class CycleResult:
    """Result of a single trading loop cycle."""

    cycle_number: int
    signals_generated: int
    order_submitted: bool
    fills_matched: int
    reconciliation_passed: bool | None
    errors: tuple[str, ...] = ()
    duration_seconds: float = 0.0


@dataclass
class LoopState:
    """Mutable state of the trading loop."""

    cycle_count: int = 0
    consecutive_errors: int = 0
    last_portfolio_sync: float = 0.0
    last_fill_sync: float = 0.0
    last_reconciliation: float = 0.0
    last_heartbeat: float = 0.0
    total_orders_submitted: int = 0
    total_fills: int = 0
    halted: bool = False
    halt_reason: str = ""


class TradingLoop:
    """Self-running autonomous trading loop.

    Wires together: agents -> coordinator -> executor -> fill tracker ->
    reconciliation in a continuous cycle with configurable intervals.
    """

    def __init__(
        self,
        coordinator: Coordinator,
        live_executor: LiveOrderExecutor,
        fill_tracker: FillTracker,
        portfolio_sync: PortfolioSyncService,
        account_setup: AccountSetupService,
        session_manager: SessionManager,
        reconciliation_loop: ReconciliationLoop,
        shutdown: GracefulShutdown,
        exchange_provider: ExchangeProvider,
        memory: SharedMemory,
        agents: list[Agent],
        market_data: MarketDataService,
        config: LoopConfig | None = None,
        alert_dispatcher: AlertDispatcher | None = None,
        stop_loss_monitor: StopLossMonitor | None = None,
        darwinian: DarwinianWeightManager | None = None,
        autoresearch: AutoresearchLoop | None = None,
        janus: JanusMetaWeighting | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.live_executor = live_executor
        self.fill_tracker = fill_tracker
        self.portfolio_sync = portfolio_sync
        self.account_setup = account_setup
        self.session_manager = session_manager
        self.reconciliation_loop = reconciliation_loop
        self.shutdown = shutdown
        self.exchange_provider = exchange_provider
        self.memory = memory
        self.agents = agents
        self.market_data = market_data
        self.config = config or LoopConfig()
        self.state = LoopState()
        self.alert_dispatcher = alert_dispatcher or AlertDispatcher()
        self.stop_loss_monitor = stop_loss_monitor
        self.darwinian = darwinian
        self.autoresearch = autoresearch
        self.janus = janus

    def run(self) -> LoopState:
        """Run the trading loop until shutdown or session end.

        Returns the final loop state.
        """
        logger.info(
            "Trading loop starting",
            extra={
                "extra_json": {
                    "symbols": list(self.config.symbols),
                    "cycle_interval": self.config.cycle_interval,
                }
            },
        )

        # Setup: configure account for all symbols
        self._setup_account()

        # Initial portfolio sync
        self.portfolio_sync.sync_account()
        self.state.last_portfolio_sync = time.monotonic()

        # Main loop
        while self.shutdown.is_running and not self.state.halted:
            # Check session lifecycle
            self.session_manager.check_session_end()
            if not self.session_manager.is_trading_allowed:
                self._wait_for_session()
                continue

            try:
                result = self._run_cycle()
                self.state.consecutive_errors = 0

                if result.errors:
                    logger.warning(
                        "Cycle completed with errors",
                        extra={
                            "extra_json": {
                                "cycle": result.cycle_number,
                                "errors": list(result.errors),
                            }
                        },
                    )

            except Exception as e:
                self.state.consecutive_errors += 1
                logger.error(
                    "Cycle failed",
                    extra={
                        "extra_json": {
                            "cycle": self.state.cycle_count,
                            "error": str(e),
                            "consecutive_errors": self.state.consecutive_errors,
                        }
                    },
                )

                if self.state.consecutive_errors >= self.config.max_consecutive_errors:
                    self._halt(f"Max consecutive errors reached: {self.state.consecutive_errors}")
                    break

            # Wait for next cycle
            self._sleep_interruptible(self.config.cycle_interval)

        logger.info(
            "Trading loop stopped",
            extra={
                "extra_json": {
                    "total_cycles": self.state.cycle_count,
                    "total_orders": self.state.total_orders_submitted,
                    "total_fills": self.state.total_fills,
                    "halted": self.state.halted,
                    "halt_reason": self.state.halt_reason,
                }
            },
        )
        return self.state

    def _check_control_state(self) -> bool:
        """Check Redis control state. Returns True if trading is allowed.

        Fails closed: if Redis is unreachable, returns False.
        """
        try:
            from aiswarm.api.routes_control import control_state

            if not control_state.is_trading_allowed:
                logger.info(
                    "Control state blocks trading",
                    extra={"extra_json": {"state": control_state.state.value}},
                )
                return False
        except Exception as e:
            logger.error(
                "Control state check failed — failing closed",
                extra={"extra_json": {"error": str(e)}},
            )
            return False
        return True

    def _run_cycle(self) -> CycleResult:
        """Execute a single trading cycle."""
        cycle_start = time.monotonic()
        self.state.cycle_count += 1
        errors: list[str] = []
        signals_generated = 0
        order_submitted = False
        fills_matched = 0
        recon_passed: bool | None = None
        now = time.monotonic()

        # 1. Portfolio sync (if interval elapsed)
        if now - self.state.last_portfolio_sync >= self.config.portfolio_sync_interval:
            try:
                sync_start = time.monotonic()
                self.portfolio_sync.sync_account()
                m.PORTFOLIO_SYNC_LATENCY.observe(time.monotonic() - sync_start)
                # G-005: daily P&L sync
                self.portfolio_sync.sync_daily_pnl()
                self.state.last_portfolio_sync = now
                # Update portfolio metrics
                if self.memory.latest_snapshot:
                    m.NAV_GAUGE.set(self.memory.latest_snapshot.nav)
                    m.EXPOSURE_GAUGE.set(self.memory.latest_snapshot.gross_exposure)
                m.PNL_GAUGE.set(self.memory.latest_pnl)
                m.DRAWDOWN_GAUGE.set(self.memory.rolling_drawdown)
                m.LEVERAGE_GAUGE.set(self.memory.current_leverage)
            except Exception as e:
                m.LOOP_ERRORS.labels(component="portfolio_sync").inc()
                errors.append(f"portfolio_sync: {e}")

        # 1b. Per-position stop-loss check (after portfolio sync)
        if self.stop_loss_monitor is not None and self.memory.latest_snapshot:
            try:
                self._run_stop_loss_check(errors)
            except Exception as e:
                m.LOOP_ERRORS.labels(component="stop_loss").inc()
                errors.append(f"stop_loss: {e}")

        # 2. Check control state (G-002: Redis-backed)
        if not self._check_control_state():
            duration = time.monotonic() - cycle_start
            return CycleResult(
                cycle_number=self.state.cycle_count,
                signals_generated=0,
                order_submitted=False,
                fills_matched=0,
                reconciliation_passed=None,
                errors=("control_state_blocked",),
                duration_seconds=round(duration, 4),
            )

        # 3. Fetch market data and run agents for each symbol
        all_signals: list[Signal] = []

        # 3a. Drain external signals (TradingView webhooks, etc.)
        try:
            from aiswarm.integrations.tradingview.webhook import drain_signals

            external = drain_signals()
            all_signals.extend(external)
            if external:
                logger.info(
                    "External signals received",
                    extra={"extra_json": {"count": len(external)}},
                )
        except ImportError:
            pass  # TradingView integration not available

        for symbol in self.config.symbols:
            try:
                symbol_signals = self._process_symbol(symbol)
                all_signals.extend(symbol_signals)
            except Exception as e:
                errors.append(f"process_{symbol}: {e}")

        signals_generated = len(all_signals)

        # 3. Coordinate: arbitrate + risk check → approved order
        if all_signals:
            try:
                order = self.coordinator.coordinate(all_signals)
                if order is not None:
                    m.SIGNALS_APPROVED.inc()
                    # 4. Submit order
                    result = self.live_executor.submit_order(order)
                    if result.success:
                        order_submitted = True
                        self.state.total_orders_submitted += 1
                        m.ORDERS_SUBMITTED.labels(symbol=order.symbol, side=order.side.value).inc()
                        logger.info(
                            "Order submitted",
                            extra={
                                "extra_json": {
                                    "order_id": order.order_id,
                                    "symbol": order.symbol,
                                    "side": order.side.value,
                                    "exchange_id": result.exchange_order_id,
                                }
                            },
                        )
                    else:
                        errors.append(f"submit_order: {result.message}")
            except Exception as e:
                m.LOOP_ERRORS.labels(component="coordinator").inc()
                errors.append(f"coordinate: {e}")

        # 5. Fill sync (if interval elapsed)
        if now - self.state.last_fill_sync >= self.config.fill_sync_interval:
            for symbol in self.config.symbols:
                try:
                    fill_result = self.fill_tracker.sync_fills(symbol)
                    fills_matched += fill_result.matched_fills
                    self.state.total_fills += fill_result.matched_fills
                    if fill_result.matched_fills > 0:
                        m.ORDERS_FILLED.labels(symbol=symbol).inc(fill_result.matched_fills)
                except Exception as e:
                    m.LOOP_ERRORS.labels(component="fill_sync").inc()
                    errors.append(f"fill_sync_{symbol}: {e}")
            self.state.last_fill_sync = now

        # 6. Reconciliation (if interval elapsed)
        if now - self.state.last_reconciliation >= self.config.reconciliation_interval:
            try:
                recon_passed = self._run_reconciliation()
                self.state.last_reconciliation = now
            except Exception as e:
                errors.append(f"reconciliation: {e}")

        # 7. Cancel stale orders (G-015)
        stale = self.live_executor.order_store.get_stale_orders(self.config.order_timeout_seconds)
        for record in stale:
            try:
                self.live_executor.cancel_order(record.order.order_id)
                logger.warning(
                    "Stale order cancelled",
                    extra={"extra_json": {"order_id": record.order.order_id}},
                )
            except Exception as e:
                errors.append(f"stale_cancel_{record.order.order_id}: {e}")

        # 8. Heartbeat
        if now - self.state.last_heartbeat >= self.config.heartbeat_interval:
            self._emit_heartbeat()
            self.state.last_heartbeat = now

        duration = time.monotonic() - cycle_start
        m.LOOP_CYCLES.inc()
        m.LOOP_CYCLE_DURATION.observe(duration)

        return CycleResult(
            cycle_number=self.state.cycle_count,
            signals_generated=signals_generated,
            order_submitted=order_submitted,
            fills_matched=fills_matched,
            reconciliation_passed=recon_passed,
            errors=tuple(errors),
            duration_seconds=round(duration, 4),
        )

    def _process_symbol(self, symbol: str) -> list[Signal]:
        """Fetch market data and run all agents for a single symbol."""
        data = self.market_data.fetch_symbol_data(
            symbol,
            klines_interval=self.config.klines_interval,
            klines_limit=self.config.klines_limit,
        )
        context = self.market_data.build_agent_context(data)
        signals: list[Signal] = []

        for agent in self.agents:
            try:
                result = agent.analyze(context)
                signal = result.get("signal")
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(
                    "Agent analysis failed",
                    extra={
                        "extra_json": {
                            "agent": agent.agent_id,
                            "symbol": symbol,
                            "error": str(e),
                        }
                    },
                )

        return signals

    def _run_stop_loss_check(self, errors: list[str]) -> None:
        """Scan positions for stop-loss breaches and submit closing orders.

        Stop-loss orders bypass the normal signal/coordinator flow and are
        submitted directly to the executor. Each trigger is recorded as a
        risk event in SharedMemory and as a Prometheus counter increment.
        """
        assert self.stop_loss_monitor is not None
        assert self.memory.latest_snapshot is not None

        close_orders = self.stop_loss_monitor.check_positions(self.memory.latest_snapshot)

        for order in close_orders:
            try:
                result = self.live_executor.submit_order(order)
                m.STOP_LOSS_TRIGGERS.labels(symbol=order.symbol).inc()
                if result.success:
                    self.state.total_orders_submitted += 1
                    m.ORDERS_SUBMITTED.labels(symbol=order.symbol, side=order.side.value).inc()
                    logger.warning(
                        "Stop-loss order submitted",
                        extra={
                            "extra_json": {
                                "order_id": order.order_id,
                                "symbol": order.symbol,
                                "side": order.side.value,
                                "quantity": order.quantity,
                                "exchange_id": result.exchange_order_id,
                            }
                        },
                    )
                else:
                    errors.append(f"stop_loss_submit_{order.symbol}: {result.message}")

                # Record risk event regardless of submission outcome
                self.memory.record_risk_event(
                    RiskEvent(
                        event_id=new_id("rev"),
                        severity=RiskSeverity.WARNING,
                        rule="per_position_stop_loss",
                        message=order.thesis,
                        symbol=order.symbol,
                        strategy=order.strategy,
                        created_at=utc_now(),
                    )
                )
            except Exception as e:
                errors.append(f"stop_loss_submit_{order.symbol}: {e}")

    def _run_reconciliation(self) -> bool:
        """Run position reconciliation against exchange state."""
        # Fetch exchange positions
        try:
            exchange_positions = self.exchange_provider.get_positions()
        except Exception:
            exchange_positions = []

        report = self.reconciliation_loop.run_periodic_check(
            internal_snapshot=self.memory.latest_snapshot,
            exchange_positions=exchange_positions,
        )
        return report.passed

    def _setup_account(self) -> None:
        """Configure leverage and margin mode for all symbols."""
        results = self.account_setup.setup_all_symbols(
            list(self.config.symbols),
            leverage=self.config.default_leverage,
            margin_mode=self.config.default_margin_mode,
        )
        for r in results:
            if not r.leverage_set or not r.margin_mode_set:
                logger.warning(
                    "Account setup incomplete",
                    extra={"extra_json": {"symbol": r.symbol, "message": r.message}},
                )

    def _halt(self, reason: str) -> None:
        """Halt the trading loop and cancel all orders."""
        self.state.halted = True
        self.state.halt_reason = reason
        logger.error(
            "Trading loop halted",
            extra={"extra_json": {"reason": reason}},
        )
        self.alert_dispatcher.send(
            f"Trading loop halted: {reason}",
            severity="critical",
            context={"cycle": self.state.cycle_count},
        )
        try:
            self.live_executor.cancel_all(list(self.config.symbols))
        except Exception as e:
            logger.error(
                "Emergency cancel failed during halt",
                extra={"extra_json": {"error": str(e)}},
            )

    def _wait_for_session(self) -> None:
        """Wait for an active trading session, checking periodically."""
        self._sleep_interruptible(self.config.cycle_interval)

    def _sleep_interruptible(self, duration: float) -> None:
        """Sleep in small increments so we can respond to shutdown signals."""
        end = time.monotonic() + duration
        while time.monotonic() < end and self.shutdown.is_running and not self.state.halted:
            time.sleep(min(1.0, end - time.monotonic()))

    def _emit_heartbeat(self) -> None:
        """Emit heartbeat log, update metrics, and touch heartbeat file."""
        logger.debug(
            "Heartbeat",
            extra={
                "extra_json": {
                    "cycle": self.state.cycle_count,
                    "orders": self.state.total_orders_submitted,
                    "fills": self.state.total_fills,
                    "consecutive_errors": self.state.consecutive_errors,
                }
            },
        )
        m.LOOP_HEARTBEAT.set(time.time())
        m.LOOP_HALTED.set(1.0 if self.state.halted else 0.0)
        # Update mandate metrics
        for mid, tracker in self.memory.mandate_trackers.items():
            m.MANDATE_PNL.labels(mandate_id=mid).set(tracker.daily_pnl)
            m.MANDATE_EXPOSURE.labels(mandate_id=mid).set(tracker.gross_exposure)
            m.MANDATE_DRAWDOWN.labels(mandate_id=mid).set(tracker.drawdown)
        # Write heartbeat file for Docker health check (G-016)
        try:
            hb_path = Path("/tmp/ais_loop_heartbeat")  # nosec B108
            hb_path.write_text(str(time.time()))
        except OSError:
            pass
