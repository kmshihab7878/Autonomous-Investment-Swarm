import os
from pathlib import Path

from typing import Any, Callable

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aiswarm import __version__
from aiswarm.api.rate_limit import require_general_rate_limit
from aiswarm.api.routes_control import router as control_router
from aiswarm.api.routes_health import router as health_router
from aiswarm.api.routes_mandates import router as mandates_router
from aiswarm.api.routes_metrics import router as metrics_router
from aiswarm.api.routes_reports import router as reports_router
from aiswarm.api.routes_session import router as session_router
from aiswarm.api.routes_ws import router as ws_router

_is_live = os.environ.get("AIS_EXECUTION_MODE", "").lower() == "live"

app = FastAPI(
    title="Autonomous Investment Swarm",
    version=__version__,
    description=(
        "Risk-gated autonomous trading API. "
        "Every order requires an HMAC-signed approval token from the risk engine."
    ),
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    docs_url=None if _is_live else "/docs",
    redoc_url=None if _is_live else "/redoc",
    openapi_url=None if _is_live else "/openapi.json",
)

# CORS — restrict origins in production, permissive in dev
_cors_origins = os.environ.get("AIS_CORS_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or (["*"] if not _is_live else []),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization"],
)


# Security headers middleware
@app.middleware("http")
async def security_headers(request: Request, call_next: Callable[[Request], Any]) -> Response:
    """Add security headers to all HTTP responses."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if _is_live:
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


# Static files (dashboard)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(_static_dir / "dashboard.html"))


# Public routes — no rate limiting (health is for monitoring, metrics for Prometheus)
app.include_router(health_router, tags=["health"])
app.include_router(metrics_router, tags=["metrics"])

# Authenticated routes — control router has per-endpoint rate limits (see routes_control.py),
# other authenticated routers get the general rate limit (60 req/min per IP).
app.include_router(control_router, tags=["control"])
app.include_router(
    reports_router, tags=["reports"], dependencies=[Depends(require_general_rate_limit)]
)
app.include_router(
    mandates_router, tags=["mandates"], dependencies=[Depends(require_general_rate_limit)]
)
app.include_router(
    session_router, tags=["session"], dependencies=[Depends(require_general_rate_limit)]
)
app.include_router(ws_router, tags=["websocket"])
