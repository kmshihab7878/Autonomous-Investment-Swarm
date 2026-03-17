"""Tests for centralized YAML config validation."""

from __future__ import annotations

import copy

import pytest

from aiswarm.utils.config_schema import (
    AISConfig,
    AlertChannelConfig,
    ConfigValidationError,
    ExecutionConfig,
    MandateConfig,
    RiskConfig,
    validate_config,
)


# ---------------------------------------------------------------------------
# Fixture: valid minimal config
# ---------------------------------------------------------------------------

VALID_CONFIG: dict = {
    "environment": "paper",
    "default_currency": "USD",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "risk": {
        "max_position_weight": 0.05,
        "max_gross_exposure": 1.0,
        "max_daily_loss": 0.02,
        "max_rolling_drawdown": 0.05,
        "max_leverage": 1.0,
        "min_liquidity_score": 0.5,
        "max_concentration_hhi": 0.18,
        "max_position_loss_pct": 0.05,
        "max_strategy_weight": 0.20,
        "max_net_exposure": 0.50,
    },
    "execution": {
        "timeout_seconds": 5,
        "retry_attempts": 3,
        "allow_live": False,
        "scheduler_interval_seconds": 10,
    },
    "audit": {"decision_log_path": "logs/decision_log.jsonl"},
    "orchestration": {
        "arbitration_mode": "weighted_voting",
        "required_risk_approval": True,
    },
    "portfolio": {
        "target_gross_exposure": 0.75,
        "max_single_position_weight": 0.05,
        "rebalance_interval_minutes": 60,
    },
    "monitoring": {
        "prometheus_port": 9001,
        "health_interval_seconds": 30,
        "decision_log_json": True,
    },
    "alerting": {
        "enabled": False,
        "webhook_url": "",
        "severity_filter": "warning",
        "alert_channels": [],
    },
    "session": {
        "default_duration_hours": 8,
        "auto_end_on_schedule": True,
        "require_approval": True,
    },
    "staging": {"enabled": True, "auto_expire_seconds": 300},
    "mandates": [
        {
            "mandate_id": "mandate_btc",
            "strategy": "momentum_ma_crossover",
            "symbols": ["BTCUSDT"],
            "risk_budget": {
                "max_capital": 10000.0,
                "max_daily_loss": 0.02,
                "max_drawdown": 0.05,
                "max_open_orders": 3,
                "max_position_notional": 5000.0,
            },
        }
    ],
}


def _cfg(**overrides: object) -> dict:
    """Create a config with overrides applied at the top level."""
    c = copy.deepcopy(VALID_CONFIG)
    c.update(overrides)
    return c


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestValidConfig:
    def test_valid_config_passes(self) -> None:
        result = validate_config(VALID_CONFIG)
        assert isinstance(result, AISConfig)
        assert result.environment == "paper"
        assert len(result.mandates) == 1

    def test_defaults_applied(self) -> None:
        """Minimal empty config should pass with defaults."""
        result = validate_config({})
        assert result.environment == "paper"
        assert result.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_mode_field_accepted(self) -> None:
        result = validate_config({"mode": "live"})
        assert result.mode == "live"

    def test_empty_mode_accepted(self) -> None:
        result = validate_config({"mode": ""})
        assert result.mode == ""


# ---------------------------------------------------------------------------
# Schema enforcement (extra keys rejected)
# ---------------------------------------------------------------------------


class TestExtraKeysRejected:
    def test_unknown_top_level_key(self) -> None:
        with pytest.raises(ConfigValidationError, match="validation failed"):
            validate_config({"bogus_key": 42})

    def test_unknown_risk_key(self) -> None:
        with pytest.raises(ConfigValidationError, match="validation failed"):
            validate_config({"risk": {"max_position_weight": 0.05, "typo_field": 1}})

    def test_unknown_execution_key(self) -> None:
        with pytest.raises(ConfigValidationError, match="validation failed"):
            validate_config({"execution": {"timeout_seconds": 5, "unknown": True}})

    def test_unknown_mandate_field(self) -> None:
        with pytest.raises(ConfigValidationError, match="validation failed"):
            validate_config(
                {
                    "mandates": [
                        {
                            "mandate_id": "m1",
                            "strategy": "s1",
                            "extra_field": "bad",
                        }
                    ]
                }
            )


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------


class TestRangeValidation:
    def test_negative_max_position_weight(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"risk": {"max_position_weight": -0.1}})

    def test_weight_over_one(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"risk": {"max_position_weight": 1.5}})

    def test_leverage_zero(self) -> None:
        """Leverage of 0.0 is valid (no trading)."""
        result = validate_config({"risk": {"max_leverage": 0.0}})
        assert result.risk.max_leverage == 0.0

    def test_leverage_max(self) -> None:
        result = validate_config({"risk": {"max_leverage": 125.0}})
        assert result.risk.max_leverage == 125.0

    def test_leverage_over_max(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"risk": {"max_leverage": 200.0}})

    def test_negative_cycle_interval(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"loop": {"cycle_interval": 0.0}})

    def test_port_zero(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"monitoring": {"prometheus_port": 0}})

    def test_port_over_max(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"monitoring": {"prometheus_port": 70000}})

    def test_klines_limit_over_max(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"loop": {"klines_limit": 2000}})


# ---------------------------------------------------------------------------
# Enum validation
# ---------------------------------------------------------------------------


class TestEnumValidation:
    def test_invalid_mode(self) -> None:
        with pytest.raises(ConfigValidationError, match="mode"):
            validate_config({"mode": "invalid_mode"})

    def test_valid_modes(self) -> None:
        for mode in ("paper", "shadow", "live", ""):
            result = validate_config({"mode": mode})
            assert result.mode == mode

    def test_invalid_alert_format(self) -> None:
        with pytest.raises(ConfigValidationError, match="format"):
            validate_config(
                {
                    "alerting": {
                        "alert_channels": [{"name": "ch1", "url": "http://x", "format": "invalid"}]
                    }
                }
            )

    def test_valid_alert_formats(self) -> None:
        for fmt in ("generic", "slack", "alertmanager"):
            result = validate_config(
                {
                    "alerting": {
                        "alert_channels": [{"name": "ch1", "url": "http://x", "format": fmt}]
                    }
                }
            )
            assert result.alerting.alert_channels[0].format == fmt


# ---------------------------------------------------------------------------
# Mandate validation
# ---------------------------------------------------------------------------


class TestMandateValidation:
    def test_mandate_missing_id(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"mandates": [{"strategy": "s1"}]})

    def test_mandate_missing_strategy(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config({"mandates": [{"mandate_id": "m1"}]})

    def test_valid_mandate(self) -> None:
        result = validate_config(
            {
                "mandates": [
                    {
                        "mandate_id": "m1",
                        "strategy": "s1",
                        "symbols": ["BTCUSDT"],
                        "risk_budget": {"max_capital": 5000.0},
                    }
                ]
            }
        )
        assert result.mandates[0].mandate_id == "m1"
        assert result.mandates[0].risk_budget.max_capital == 5000.0

    def test_risk_budget_invalid_daily_loss(self) -> None:
        with pytest.raises(ConfigValidationError):
            validate_config(
                {
                    "mandates": [
                        {
                            "mandate_id": "m1",
                            "strategy": "s1",
                            "risk_budget": {"max_daily_loss": 2.0},
                        }
                    ]
                }
            )


# ---------------------------------------------------------------------------
# Pydantic model unit tests
# ---------------------------------------------------------------------------


class TestModelDirect:
    def test_risk_config_defaults(self) -> None:
        r = RiskConfig()
        assert r.max_position_weight == 0.05
        assert r.max_leverage == 1.0

    def test_execution_config_defaults(self) -> None:
        e = ExecutionConfig()
        assert e.timeout_seconds == 5
        assert e.allow_live is False

    def test_alert_channel_config(self) -> None:
        ch = AlertChannelConfig(name="test", url="http://x", format="slack")
        assert ch.format == "slack"

    def test_mandate_config(self) -> None:
        m = MandateConfig(mandate_id="m1", strategy="s1")
        assert m.symbols == []
        assert m.notes == ""
