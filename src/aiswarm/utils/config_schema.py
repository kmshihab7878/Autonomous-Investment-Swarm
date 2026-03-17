"""Centralized YAML config validation.

Validates the merged configuration dictionary against Pydantic schemas at
startup, catching typos, missing keys, and invalid values before any
component is wired.  Called from ``bootstrap.load_config()`` so that every
code path benefits from validation.

Usage::

    from aiswarm.utils.config_schema import validate_config
    merged = load_config("config/")
    validated = validate_config(merged)  # raises ConfigValidationError on failure
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when config validation fails."""


# ---------------------------------------------------------------------------
# Section schemas
# ---------------------------------------------------------------------------


class RiskConfig(BaseModel):
    max_position_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    max_strategy_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    max_gross_exposure: float = Field(default=1.00, ge=0.0, le=10.0)
    max_net_exposure: float = Field(default=0.50, ge=0.0, le=10.0)
    max_leverage: float = Field(default=1.00, ge=0.0, le=125.0)
    max_daily_loss: float = Field(default=0.02, ge=0.0, le=1.0)
    max_rolling_drawdown: float = Field(default=0.05, ge=0.0, le=1.0)
    min_liquidity_score: float = Field(default=0.50, ge=0.0, le=1.0)
    max_concentration_hhi: float = Field(default=0.18, ge=0.0, le=1.0)
    max_position_loss_pct: float = Field(default=0.05, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class ExecutionConfig(BaseModel):
    timeout_seconds: int = Field(default=5, ge=1)
    retry_attempts: int = Field(default=3, ge=0)
    allow_live: bool = False
    scheduler_interval_seconds: int = Field(default=10, ge=1)
    endpoint: str = ""
    require_explicit_flag: bool = False
    default_leverage: int = Field(default=1, ge=1, le=125)
    default_margin_mode: str = "ISOLATED"

    model_config = {"extra": "forbid"}


class AuditConfig(BaseModel):
    decision_log_path: str = "logs/decision_log.jsonl"

    model_config = {"extra": "forbid"}


class OrchestrationConfig(BaseModel):
    arbitration_mode: str = "weighted_voting"
    required_risk_approval: bool = True

    model_config = {"extra": "forbid"}


class PortfolioConfig(BaseModel):
    target_gross_exposure: float = Field(default=0.75, ge=0.0, le=10.0)
    max_single_position_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    rebalance_interval_minutes: int = Field(default=60, ge=1)
    use_kelly: bool = False
    max_kelly_weight: float = Field(default=0.05, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class MonitoringConfig(BaseModel):
    prometheus_port: int = Field(default=9001, ge=1, le=65535)
    health_interval_seconds: int = Field(default=30, ge=1)
    decision_log_json: bool = True

    model_config = {"extra": "forbid"}


class LoopConfigSchema(BaseModel):
    cycle_interval: float = Field(default=60.0, ge=1.0)
    portfolio_sync_interval: float = Field(default=30.0, ge=1.0)
    fill_sync_interval: float = Field(default=15.0, ge=1.0)
    reconciliation_interval: float = Field(default=60.0, ge=1.0)
    klines_interval: str = "1h"
    klines_limit: int = Field(default=100, ge=1, le=1500)
    max_consecutive_errors: int = Field(default=5, ge=1)
    heartbeat_interval: float = Field(default=10.0, ge=1.0)

    model_config = {"extra": "forbid"}


class RiskBudgetConfig(BaseModel):
    max_capital: float = Field(default=10000.0, ge=0.0)
    max_daily_loss: float = Field(default=0.02, ge=0.0, le=1.0)
    max_drawdown: float = Field(default=0.05, ge=0.0, le=1.0)
    max_open_orders: int = Field(default=3, ge=1)
    max_position_notional: float = Field(default=5000.0, ge=0.0)

    model_config = {"extra": "forbid"}


class MandateConfig(BaseModel):
    mandate_id: str
    strategy: str
    symbols: list[str] = []
    risk_budget: RiskBudgetConfig = RiskBudgetConfig()
    notes: str = ""

    model_config = {"extra": "forbid"}


class AlertChannelConfig(BaseModel):
    name: str = "unnamed"
    url: str = ""
    format: str = "generic"
    min_severity: str = "low"

    @model_validator(mode="after")
    def validate_format(self) -> AlertChannelConfig:
        allowed = {"generic", "slack", "alertmanager"}
        if self.format not in allowed:
            raise ValueError(f"Alert channel format must be one of {allowed}, got '{self.format}'")
        return self

    model_config = {"extra": "forbid"}


class AlertingConfig(BaseModel):
    enabled: bool = False
    webhook_url: str = ""
    severity_filter: str = "warning"
    alert_channels: list[AlertChannelConfig] = []
    alertmanager_url: str = ""

    model_config = {"extra": "forbid"}


class SessionConfig(BaseModel):
    default_duration_hours: int = Field(default=8, ge=1)
    auto_end_on_schedule: bool = True
    require_approval: bool = True

    model_config = {"extra": "forbid"}


class StagingConfig(BaseModel):
    enabled: bool = False
    auto_expire_seconds: int = Field(default=300, ge=0)

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


class AISConfig(BaseModel):
    """Top-level configuration schema for the Autonomous Investment Swarm."""

    # Top-level scalars
    environment: str = "paper"
    mode: str = ""
    default_currency: str = "USD"
    symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Sections
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    audit: AuditConfig = AuditConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    loop: LoopConfigSchema = LoopConfigSchema()
    alerting: AlertingConfig = AlertingConfig()
    session: SessionConfig = SessionConfig()
    staging: StagingConfig = StagingConfig()

    # Mandates is a top-level list
    mandates: list[MandateConfig] = []

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_mode(self) -> AISConfig:
        allowed = {"paper", "shadow", "live", ""}
        if self.mode not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got '{self.mode}'")
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_config(config: dict[str, Any]) -> AISConfig:
    """Validate a merged config dict against the AIS schema.

    Returns the validated ``AISConfig`` model.
    Raises ``ConfigValidationError`` with a human-readable message on failure.
    """
    try:
        validated = AISConfig(**config)
        logger.info(
            "Configuration validated",
            extra={
                "extra_json": {
                    "mandates": len(validated.mandates),
                    "symbols": validated.symbols,
                    "mode": validated.mode or validated.environment,
                }
            },
        )
        return validated
    except Exception as e:
        msg = f"Configuration validation failed: {e}"
        logger.error(msg)
        raise ConfigValidationError(msg) from e
