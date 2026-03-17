from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aiswarm.utils.logging import get_logger
from aiswarm.utils.secrets import get_secrets_provider

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_api_key() -> str:
    key = get_secrets_provider().get_secret("AIS_API_KEY") or ""
    if not key or key == "change-me-to-a-secure-random-string":
        return ""
    return key


def _is_live_mode() -> bool:
    return os.environ.get("AIS_EXECUTION_MODE", "").lower() == "live"


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """FastAPI dependency that enforces Bearer token auth.

    If AIS_API_KEY is not configured:
      - In live mode: reject with 503 (misconfiguration).
      - Otherwise: allow all (dev mode).
    """
    expected = _get_api_key()
    if not expected:
        if _is_live_mode():
            logger.error("AIS_API_KEY is not set while AIS_EXECUTION_MODE=live")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AIS_API_KEY must be configured in live mode",
            )
        return "dev"

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return credentials.credentials
