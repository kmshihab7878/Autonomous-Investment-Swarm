"""Comprehensive tests for the AIS risk engine and all risk guards."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from aiswarm.risk.drawdown import DrawdownGuard
from aiswarm.risk.kill_switch import KillSwitch
from aiswarm.risk.leverage import LeverageGuard
from aiswarm.risk.limits import (
    RiskEngine,
    sign_risk_token,
    verify_risk_token,
)
from aiswarm.risk.liquidity import LiquidityGuard
from aiswarm.types.orders import Order, Side
from aiswarm.types.portfolio import PortfolioSnapshot


def _make_order(
    order_id: str = "o1",
    symbol: str = "BTCUSDT",
    notional: float = 1000.0,
    quantity: float = 1.0,
    side: Side = Side.BUY,
) -> Order:
    return Order(
        order_id=order_id,
        signal_id="s1",
        symbol=symbol,
        side=side,
        quantity=quantity,
        limit_price=None,
        notional=notional,
        strategy="test",
        thesis="valid test thesis",
        created_at=datetime.now(timezone.utc),
    )


def _make_snapshot(
    nav: float = 100_000.0,
    gross_exposure: float = 0.0,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp=datetime.now(timezone.utc),
        nav=nav,
        cash=nav,
        gross_exposure=gross_exposure,
        net_exposure=0.0,
        positions=(),
    )


# --- KillSwitch Tests ---


class TestKillSwitch:
    def test_not_triggered_on_zero_pnl(self) -> None:
        ks = KillSwitch(0.02)
        assert not ks.triggered(0.0)

    def test_not_triggered_on_positive_pnl(self) -> None:
        ks = KillSwitch(0.02)
        assert not ks.triggered(0.05)

    def test_triggered_on_loss_breach(self) -> None:
        ks = KillSwitch(0.02)
        assert ks.triggered(-0.03)

    def test_triggered_at_exact_threshold(self) -> None:
        ks = KillSwitch(0.02)
        assert ks.triggered(-0.02)

    def test_not_triggered_just_above_threshold(self) -> None:
        ks = KillSwitch(0.02)
        assert not ks.triggered(-0.019)

    def test_is_triggered_property(self) -> None:
        ks = KillSwitch(0.02)
        assert not ks.is_triggered
        ks.triggered(-0.03)
        assert ks.is_triggered

    def test_reset(self) -> None:
        ks = KillSwitch(0.02)
        ks.triggered(-0.03)
        assert ks.is_triggered
        ks.reset()
        assert not ks.is_triggered

    def test_emergency_cancels(self) -> None:
        ks = KillSwitch(0.02)
        cancels = ks.prepare_emergency_cancels("acc123", ["BTCUSDT", "ETHUSDT"])
        assert len(cancels) == 4  # 2 symbols × 2 venues
        assert cancels[0]["tool"] == "mcp__aster__cancel_all_orders"
        assert cancels[1]["tool"] == "mcp__aster__cancel_spot_all_orders"


# --- DrawdownGuard Tests ---


class TestDrawdownGuard:
    def test_not_breached(self) -> None:
        guard = DrawdownGuard()
        assert not guard.breached(0.03, 0.05)

    def test_breached_at_threshold(self) -> None:
        guard = DrawdownGuard()
        assert guard.breached(0.05, 0.05)

    def test_breached_above_threshold(self) -> None:
        guard = DrawdownGuard()
        assert guard.breached(0.10, 0.05)

    def test_zero_drawdown(self) -> None:
        guard = DrawdownGuard()
        assert not guard.breached(0.0, 0.05)


# --- LeverageGuard Tests ---


class TestLeverageGuard:
    def test_not_breached(self) -> None:
        guard = LeverageGuard()
        assert not guard.breached(0.5, 1.0)

    def test_breached(self) -> None:
        guard = LeverageGuard()
        assert guard.breached(1.5, 1.0)

    def test_at_threshold_not_breached(self) -> None:
        guard = LeverageGuard()
        assert not guard.breached(1.0, 1.0)

    def test_validate_against_brackets(self) -> None:
        from aiswarm.data.providers.aster import LeverageBracket

        brackets = [
            LeverageBracket(
                bracket=1,
                initial_leverage=20,
                notional_cap=50000,
                notional_floor=0,
                maintenance_margin_rate=0.005,
            ),
            LeverageBracket(
                bracket=2,
                initial_leverage=10,
                notional_cap=250000,
                notional_floor=50000,
                maintenance_margin_rate=0.01,
            ),
        ]
        guard = LeverageGuard()

        # Within first bracket, leverage ok
        valid, max_lev = guard.validate_against_brackets(10000, 15, brackets)
        assert valid
        assert max_lev == 20

        # Within first bracket, leverage too high
        valid, max_lev = guard.validate_against_brackets(10000, 25, brackets)
        assert not valid
        assert max_lev == 20

        # In second bracket, leverage ok
        valid, max_lev = guard.validate_against_brackets(100000, 8, brackets)
        assert valid
        assert max_lev == 10


# --- LiquidityGuard Tests ---


class TestLiquidityGuard:
    def test_not_breached(self) -> None:
        guard = LiquidityGuard()
        assert not guard.breached(0.8, 0.5)

    def test_breached(self) -> None:
        guard = LiquidityGuard()
        assert guard.breached(0.3, 0.5)

    def test_at_threshold_not_breached(self) -> None:
        guard = LiquidityGuard()
        assert not guard.breached(0.5, 0.5)

    def test_check_orderbook_depth(self) -> None:
        from aiswarm.data.providers.aster import OrderBook, OrderBookLevel

        ob = OrderBook(
            symbol="BTC/USDT",
            bids=(
                OrderBookLevel(price=50000, quantity=10),
                OrderBookLevel(price=49999, quantity=5),
            ),
            asks=(
                OrderBookLevel(price=50001, quantity=10),
                OrderBookLevel(price=50002, quantity=5),
            ),
            timestamp=datetime.now(timezone.utc),
        )
        guard = LiquidityGuard()

        # Small order should pass
        safe, ratio = guard.check_orderbook_depth(ob, 10000, max_depth_consumption=0.10)
        assert safe

        # Huge order should fail
        safe, ratio = guard.check_orderbook_depth(ob, 500000, max_depth_consumption=0.10)
        assert not safe


# --- HMAC Token Tests ---


class TestHMACTokens:
    def setup_method(self) -> None:
        os.environ["AIS_RISK_HMAC_SECRET"] = "test-secret-for-ci"

    def test_sign_and_verify(self) -> None:
        token = sign_risk_token("order123")
        assert verify_risk_token(token, "order123")

    def test_reject_wrong_order_id(self) -> None:
        token = sign_risk_token("order123")
        assert not verify_risk_token(token, "order456")

    def test_reject_tampered_token(self) -> None:
        token = sign_risk_token("order123")
        tampered = token[:-4] + "XXXX"
        assert not verify_risk_token(tampered, "order123")

    def test_reject_empty_token(self) -> None:
        assert not verify_risk_token("", "order123")

    def test_reject_malformed_token(self) -> None:
        assert not verify_risk_token("not:valid", "order123")
        assert not verify_risk_token("a:b:c:d:e", "order123")

    def test_token_format(self) -> None:
        token = sign_risk_token("order123")
        parts = token.split(":")
        assert len(parts) == 4
        assert parts[0] == "order123"
        assert parts[2] == "v1"  # default key_id


# --- RiskEngine Integration Tests ---


class TestRiskEngine:
    def setup_method(self) -> None:
        os.environ["AIS_RISK_HMAC_SECRET"] = "test-secret-for-ci"

    def test_blocks_on_kill_switch(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02)
        order = _make_order()
        approval = engine.validate(order, None, -0.03)
        assert not approval.approved
        assert "kill_switch_triggered" in approval.reasons

    def test_approves_with_token(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02)
        order = _make_order()
        approval = engine.validate(order, None, 0.0)
        assert approval.approved
        assert approval.order is not None
        assert approval.order.risk_approval_token is not None
        # Verify the token is valid
        assert verify_risk_token(approval.order.risk_approval_token, order.order_id)

    def test_blocks_on_drawdown_breach(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02, max_rolling_drawdown=0.05)
        order = _make_order()
        approval = engine.validate(order, None, 0.0, rolling_drawdown=0.06)
        assert not approval.approved
        assert any("drawdown_breached" in r for r in approval.reasons)

    def test_blocks_on_leverage_breach(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02, max_leverage=1.0)
        order = _make_order()
        approval = engine.validate(order, None, 0.0, current_leverage=1.5)
        assert not approval.approved
        assert any("leverage_breached" in r for r in approval.reasons)

    def test_blocks_on_liquidity_insufficient(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02, min_liquidity_score=0.50)
        order = _make_order()
        approval = engine.validate(order, None, 0.0, liquidity_score=0.3)
        assert not approval.approved
        assert any("liquidity_insufficient" in r for r in approval.reasons)

    def test_blocks_on_position_weight_exceeded(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02)
        # Notional of 60k against 100k NAV = 60% weight, way above 5%
        order = _make_order(notional=60000)
        snapshot = _make_snapshot(nav=100000)
        approval = engine.validate(order, snapshot, 0.0)
        assert not approval.approved
        assert any("position_weight" in r for r in approval.reasons)

    def test_blocks_on_gross_exposure_exceeded(self) -> None:
        engine = RiskEngine(1.0, 0.50, 0.02)
        order = _make_order(notional=60000)
        snapshot = _make_snapshot(nav=100000, gross_exposure=0.45)
        approval = engine.validate(order, snapshot, 0.0)
        assert not approval.approved
        assert any("gross_exposure" in r for r in approval.reasons)

    def test_multiple_violations_all_reported(self) -> None:
        engine = RiskEngine(
            0.05, 1.0, 0.02, max_rolling_drawdown=0.05, max_leverage=1.0, min_liquidity_score=0.50
        )
        order = _make_order()
        approval = engine.validate(
            order,
            None,
            -0.03,
            rolling_drawdown=0.10,
            current_leverage=2.0,
            liquidity_score=0.1,
        )
        assert not approval.approved
        assert len(approval.reasons) >= 4  # kill_switch + drawdown + leverage + liquidity

    def test_approved_order_has_correct_status(self) -> None:
        engine = RiskEngine(0.05, 1.0, 0.02)
        order = _make_order()
        approval = engine.validate(order, None, 0.0)
        assert approval.approved
        assert approval.order is not None
        assert approval.order.status.value == "approved"

    def test_config_driven_risk_engine(self) -> None:
        """Test creating RiskEngine with config/risk.yaml-like parameters."""
        engine = RiskEngine(
            max_position_weight=0.05,
            max_gross_exposure=1.00,
            max_daily_loss=0.02,
            max_rolling_drawdown=0.05,
            max_leverage=1.00,
            min_liquidity_score=0.50,
        )
        order = _make_order(notional=1000)
        snapshot = _make_snapshot(nav=100000)
        approval = engine.validate(order, snapshot, 0.0)
        assert approval.approved
