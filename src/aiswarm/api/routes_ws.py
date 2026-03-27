"""WebSocket endpoint for real-time dashboard updates.

Broadcasts trading events (signals, orders, fills, risk, portfolio snapshots)
to connected dashboard clients via a lightweight pub/sub pattern.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from aiswarm.api.auth import _get_api_key
from aiswarm.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Connected WebSocket clients
_clients: set[WebSocket] = set()


async def broadcast(event_type: str, data: dict[str, Any]) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    if not _clients:
        return
    message = json.dumps({"type": event_type, "data": data})
    disconnected: list[WebSocket] = []
    for ws in _clients:
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _clients.discard(ws)


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket connection for real-time dashboard updates.

    Requires a valid API key via query parameter ``token`` when
    ``AIS_API_KEY`` is configured. Connections without a valid token
    are rejected with close code 4001.
    """
    expected = _get_api_key()
    if expected:
        token = ws.query_params.get("token", "")
        if not token or not secrets.compare_digest(token, expected):
            logger.warning("WebSocket auth failed — invalid or missing token")
            await ws.close(code=4001)
            return

    await ws.accept()
    _clients.add(ws)
    logger.info(
        "Dashboard WebSocket connected",
        extra={"extra_json": {"clients": len(_clients)}},
    )
    try:
        while True:
            # Keep connection alive; clients don't send data
            await ws.receive_text()
    except WebSocketDisconnect:
        _clients.discard(ws)
        logger.info(
            "Dashboard WebSocket disconnected",
            extra={"extra_json": {"clients": len(_clients)}},
        )
