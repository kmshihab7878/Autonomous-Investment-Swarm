from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Portfolio metrics
PNL_GAUGE = Gauge("ais_pnl", "Portfolio pnl fraction")
EXPOSURE_GAUGE = Gauge("ais_exposure", "Gross exposure fraction")
NAV_GAUGE = Gauge("ais_nav", "Net asset value in quote currency")
DRAWDOWN_GAUGE = Gauge("ais_drawdown", "Current rolling drawdown fraction")
LEVERAGE_GAUGE = Gauge("ais_leverage", "Current portfolio leverage")

# Agent metrics
AGENT_LATENCY = Histogram("ais_agent_latency_seconds", "Agent analysis latency", ["agent_id"])
SIGNALS_GENERATED = Counter(
    "ais_signals_total", "Total signals generated", ["agent_id", "direction"]
)
SIGNALS_APPROVED = Counter("ais_signals_approved_total", "Signals that passed risk validation")
SIGNALS_REJECTED = Counter(
    "ais_signals_rejected_total", "Signals rejected by risk engine", ["reason"]
)

# Execution metrics
ORDERS_SUBMITTED = Counter(
    "ais_orders_submitted_total", "Orders submitted to OMS", ["symbol", "side"]
)
ORDERS_FILLED = Counter("ais_orders_filled_total", "Orders filled", ["symbol"])
PAPER_FILLS = Counter("ais_paper_fills_total", "Paper trading fills", ["symbol"])

# Aster DEX connectivity metrics
ASTER_LATENCY = Histogram("ais_aster_latency_seconds", "Aster DEX MCP call latency", ["tool"])
ASTER_ERRORS = Counter("ais_aster_errors_total", "Aster DEX MCP call errors", ["tool"])
ASTER_DATA_FRESHNESS = Gauge(
    "ais_aster_data_age_seconds", "Age of latest data from Aster DEX", ["data_type"]
)

# Risk metrics
KILL_SWITCH_TRIGGERS = Counter(
    "ais_kill_switch_triggers_total", "Number of kill switch activations"
)
RISK_REJECTIONS = Counter("ais_risk_rejections_total", "Risk validation rejections", ["guard"])
STOP_LOSS_TRIGGERS = Counter(
    "ais_stop_loss_triggers_total", "Per-position stop-loss closings", ["symbol"]
)

# Mandate metrics
MANDATE_PNL = Gauge("ais_mandate_pnl", "Per-mandate P&L", ["mandate_id"])
MANDATE_EXPOSURE = Gauge("ais_mandate_exposure", "Per-mandate gross exposure", ["mandate_id"])
MANDATE_DRAWDOWN = Gauge("ais_mandate_drawdown", "Per-mandate drawdown fraction", ["mandate_id"])

# Session metrics
SESSION_STATE = Gauge("ais_session_state", "Current session state (1=active, 0=inactive)")
STAGED_ORDERS = Gauge("ais_staged_orders", "Number of staged orders awaiting review")

# Live execution metrics
LIVE_SUBMISSIONS = Counter(
    "ais_live_submissions_total", "Orders submitted to exchange", ["symbol", "venue"]
)
LIVE_FILLS = Counter("ais_live_fills_total", "Orders filled on exchange", ["symbol"])
LIVE_CANCELS = Counter("ais_live_cancels_total", "Orders cancelled on exchange", ["reason"])
PORTFOLIO_SYNC_LATENCY = Histogram(
    "ais_portfolio_sync_seconds", "Portfolio sync latency from exchange"
)

# Loop health metrics
LOOP_CYCLES = Counter("ais_loop_cycles_total", "Total trading loop cycles completed")
LOOP_CYCLE_DURATION = Histogram("ais_loop_cycle_seconds", "Trading loop cycle duration")
LOOP_ERRORS = Counter("ais_loop_errors_total", "Trading loop errors", ["component"])
LOOP_HEARTBEAT = Gauge("ais_loop_heartbeat_epoch", "Last heartbeat timestamp (epoch)")
LOOP_HALTED = Gauge("ais_loop_halted", "Whether the loop is halted (1=halted, 0=running)")

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "ais_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["name"],
)
CIRCUIT_BREAKER_FAILURES = Counter(
    "ais_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["name"],
)
CIRCUIT_BREAKER_SUCCESSES = Counter(
    "ais_circuit_breaker_successes_total",
    "Total circuit breaker successes",
    ["name"],
)
CIRCUIT_BREAKER_REJECTIONS = Counter(
    "ais_circuit_breaker_rejections_total",
    "Requests rejected by open circuit",
    ["name"],
)
CIRCUIT_BREAKER_TRANSITIONS = Counter(
    "ais_circuit_breaker_transitions_total",
    "State transitions",
    ["name", "from_state", "to_state"],
)
