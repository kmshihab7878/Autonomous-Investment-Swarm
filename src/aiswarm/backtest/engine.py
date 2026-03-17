"""Backtesting engine for AIS strategies.

Event-driven simulation that replays historical candle data through a
signal generator, simulates order execution with commission and slippage,
and computes comprehensive performance metrics.

Design decisions:
- Uses its own lightweight OHLCV dataclass (no ``symbol`` field) to
  decouple backtest data from Aster-specific types.
- Tracks positions internally rather than using the production
  ``Position`` Pydantic model, since backtesting needs mutable state
  and fields (side, entry_price) that the production model omits.
- Delegates risk-metric computation to ``aiswarm.quant.risk_metrics``
  so the backtest benefits from the same battle-tested math.
- Prevents look-ahead bias by only feeding candles up to and including
  the current bar to the signal generator.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np

from aiswarm.monitoring.metrics import push_metrics
from aiswarm.quant.risk_metrics import compute_risk_metrics
from aiswarm.types.market import Signal
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OHLCV:
    """Single candlestick bar for backtesting.

    Intentionally separate from ``aiswarm.data.providers.aster.OHLCV``
    to avoid coupling backtest data loading to exchange-specific types.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestTrade:
    """Record of a simulated trade execution."""

    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    notional: float
    signal_confidence: float
    pnl: float = 0.0  # realized PnL (nonzero only for closing trades)


@dataclass
class BacktestResult:
    """Complete backtest results with performance metrics."""

    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # gross_profit / gross_loss
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of backtest results."""
        return (
            f"\n{'=' * 60}\n"
            f"Backtest: {self.strategy_name} on {self.symbol}\n"
            f"Period: {self.start_date:%Y-%m-%d} to {self.end_date:%Y-%m-%d}\n"
            f"{'=' * 60}\n"
            f"Return: {self.total_return_pct:+.2f}%\n"
            f"Sharpe: {self.sharpe_ratio:.3f}\n"
            f"Sortino: {self.sortino_ratio:.3f}\n"
            f"Max Drawdown: {self.max_drawdown_pct:.2f}%\n"
            f"Trades: {self.total_trades} "
            f"(W:{self.winning_trades} L:{self.losing_trades})\n"
            f"Win Rate: {self.win_rate:.1f}%\n"
            f"Avg Win: ${self.avg_win:.2f} | Avg Loss: ${self.avg_loss:.2f}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Final Capital: ${self.final_capital:,.2f}\n"
            f"{'=' * 60}\n"
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SignalGenerator(Protocol):
    """Protocol for anything that generates signals from candle data.

    Implementors receive:
    - ``symbol``: the instrument being backtested
    - ``candles``: all candles up to and including the current bar
    - ``current_position``: a dict describing the open position (or None)

    The ``current_position`` dict uses the keys ``side`` ("LONG" / "SHORT"),
    ``quantity``, and ``entry_price`` so adapters do not need to construct
    Pydantic models.
    """

    def generate_signal(
        self,
        symbol: str,
        candles: list[OHLCV],
        current_position: dict[str, object] | None,
    ) -> Signal | None: ...


# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------


@dataclass
class _SimPosition:
    """Mutable internal position tracker for the backtest engine."""

    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    initial_capital: float = 10_000.0
    position_size_pct: float = 0.1  # fraction of capital per trade
    commission_bps: float = 5.0  # basis points per trade
    slippage_bps: float = 2.0  # basis points slippage


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Event-driven backtesting engine.

    Replays historical candles through a ``SignalGenerator``, simulates
    order execution with commission and slippage, and computes
    performance metrics via ``aiswarm.quant.risk_metrics``.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    # -- public API --

    def run(
        self,
        strategy_name: str,
        signal_generator: SignalGenerator,
        symbol: str,
        candles: list[OHLCV],
    ) -> BacktestResult:
        """Run a backtest over historical candle data.

        Args:
            strategy_name: Human label for the strategy.
            signal_generator: Produces ``Signal | None`` per bar.
            symbol: Instrument symbol (e.g. "BTCUSDT").
            candles: Chronologically ordered OHLCV bars.

        Returns:
            ``BacktestResult`` with metrics, trades, and equity curve.

        Raises:
            ValueError: If fewer than 2 candles are provided.
        """
        if len(candles) < 2:
            raise ValueError("Need at least 2 candles for backtest")

        capital = self.config.initial_capital
        equity_curve: list[float] = [capital]
        trades: list[BacktestTrade] = []
        position: _SimPosition | None = None

        for i in range(1, len(candles)):
            # Feed candles up to current bar (look-ahead bias prevention)
            history = candles[: i + 1]
            current = candles[i]

            # Build position dict for signal generator
            pos_dict: dict[str, object] | None = None
            if position is not None:
                pos_dict = {
                    "side": position.side,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                }

            signal = signal_generator.generate_signal(symbol, history, pos_dict)

            if signal is None:
                equity_curve.append(self._mark_to_market(capital, position, current.close))
                continue

            exec_price = self._apply_slippage(current.close, signal.direction)
            commission = self._compute_commission(capital, exec_price)

            if position is None:
                # Open a new position
                position, trade = self._open_position(
                    signal, symbol, capital, exec_price, current.timestamp
                )
                capital -= commission
                trades.append(trade)

            else:
                # Check whether signal closes the position (opposite direction)
                should_close = (position.side == "LONG" and signal.direction <= -1) or (
                    position.side == "SHORT" and signal.direction >= 1
                )
                if should_close:
                    pnl = self._calc_realized_pnl(position, exec_price)
                    capital += pnl - commission
                    close_side = "SELL" if position.side == "LONG" else "BUY"
                    trades.append(
                        BacktestTrade(
                            timestamp=current.timestamp,
                            symbol=symbol,
                            side=close_side,
                            price=exec_price,
                            quantity=position.quantity,
                            notional=position.quantity * exec_price,
                            signal_confidence=signal.confidence,
                            pnl=pnl,
                        )
                    )
                    position = None

            equity_curve.append(self._mark_to_market(capital, position, current.close))

        # Force-close any open position at last bar
        if position is not None:
            last_price = candles[-1].close
            pnl = self._calc_realized_pnl(position, last_price)
            capital += pnl
            close_side = "SELL" if position.side == "LONG" else "BUY"
            trades.append(
                BacktestTrade(
                    timestamp=candles[-1].timestamp,
                    symbol=symbol,
                    side=close_side,
                    price=last_price,
                    quantity=position.quantity,
                    notional=position.quantity * last_price,
                    signal_confidence=0.0,
                    pnl=pnl,
                )
            )
            # Update final equity point to reflect closed position
            equity_curve[-1] = capital

        result = self._compute_results(
            strategy_name, symbol, candles, capital, equity_curve, trades
        )

        # Push metrics to Pushgateway if configured (short-lived process support)
        gw_url = os.environ.get("AIS_PUSHGATEWAY_URL", "")
        if gw_url:
            push_metrics(
                gw_url,
                job="ais-backtest",
                grouping_key={"strategy": strategy_name, "symbol": symbol},
            )

        return result

    # -- private helpers --

    def _apply_slippage(self, price: float, direction: int) -> float:
        """Apply slippage: buyers pay more, sellers receive less."""
        slip = price * self.config.slippage_bps / 10_000
        if direction >= 1:
            return price + slip
        return price - slip

    def _compute_commission(self, capital: float, exec_price: float) -> float:
        """Compute commission in dollar terms."""
        notional = capital * self.config.position_size_pct
        return notional * self.config.commission_bps / 10_000

    def _open_position(
        self,
        signal: Signal,
        symbol: str,
        capital: float,
        exec_price: float,
        timestamp: datetime,
    ) -> tuple[_SimPosition, BacktestTrade]:
        """Open a new position from a signal."""
        notional = capital * self.config.position_size_pct
        quantity = notional / exec_price
        side_str = "LONG" if signal.direction >= 1 else "SHORT"
        trade_side = "BUY" if signal.direction >= 1 else "SELL"

        position = _SimPosition(
            side=side_str,
            quantity=quantity,
            entry_price=exec_price,
        )
        trade = BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            side=trade_side,
            price=exec_price,
            quantity=quantity,
            notional=notional,
            signal_confidence=signal.confidence,
        )
        return position, trade

    @staticmethod
    def _calc_realized_pnl(position: _SimPosition, exit_price: float) -> float:
        if position.side == "LONG":
            return (exit_price - position.entry_price) * position.quantity
        return (position.entry_price - exit_price) * position.quantity

    @staticmethod
    def _mark_to_market(
        capital: float,
        position: _SimPosition | None,
        current_price: float,
    ) -> float:
        """Compute equity = cash + unrealized PnL."""
        if position is None:
            return capital
        if position.side == "LONG":
            unrealized = (current_price - position.entry_price) * position.quantity
        else:
            unrealized = (position.entry_price - current_price) * position.quantity
        return capital + unrealized

    def _compute_results(
        self,
        strategy_name: str,
        symbol: str,
        candles: list[OHLCV],
        final_capital: float,
        equity_curve: list[float],
        trades: list[BacktestTrade],
    ) -> BacktestResult:
        initial = self.config.initial_capital
        total_return = ((final_capital - initial) / initial) * 100

        # Compute per-bar returns for risk metrics
        returns: list[float] = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            if prev > 0:
                returns.append((equity_curve[i] - prev) / prev)

        # Use the production risk_metrics module
        returns_arr = np.array(returns, dtype=np.float64)
        metrics = compute_risk_metrics(returns_arr)

        # Trade statistics (only closing trades have nonzero pnl)
        closing_trades = [t for t in trades if t.pnl != 0.0]
        wins = [t for t in closing_trades if t.pnl > 0]
        losses = [t for t in closing_trades if t.pnl < 0]

        gross_profit = sum(t.pnl for t in wins) if wins else 0.0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=candles[0].timestamp,
            end_date=candles[-1].timestamp,
            initial_capital=initial,
            final_capital=final_capital,
            total_return_pct=total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown_pct=metrics.max_drawdown * 100,
            total_trades=len(closing_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=(len(wins) / len(closing_trades) * 100) if closing_trades else 0.0,
            avg_win=(gross_profit / len(wins)) if wins else 0.0,
            avg_loss=(gross_loss / len(losses)) if losses else 0.0,
            profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else math.inf,
            trades=trades,
            equity_curve=equity_curve,
        )
