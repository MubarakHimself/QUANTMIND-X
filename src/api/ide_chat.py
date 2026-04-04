"""
QuantMind IDE legacy chat compatibility endpoint.

This module keeps the old `/api/chat` surface alive as an ingress-only alias
while routing requests to the canonical Floor Manager / department chat paths.
It must not import deprecated LangChain/LangGraph-era agents.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException

from src.api.chat_endpoints import (
    ChatMessageRequest,
    department_chat,
    floor_manager_chat,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

_LEGACY_AGENT_ROUTE_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "copilot": ("floor-manager", None),
    "floor_manager": ("floor-manager", None),
    "workshop": ("floor-manager", None),
    "analyst": ("department", "research"),
    "research": ("department", "research"),
    "quantcode": ("department", "development"),
    "development": ("department", "development"),
    "risk": ("department", "risk"),
    "trading": ("department", "trading"),
    "portfolio": ("department", "portfolio"),
}


def _coerce_legacy_context(raw_context: Any, *, model: str) -> tuple[Dict[str, Any], Optional[list[dict[str, str]]]]:
    """Normalize legacy request context into canonical chat request fields."""
    history = raw_context if isinstance(raw_context, list) else None
    context = dict(raw_context) if isinstance(raw_context, dict) else {}
    if model:
        context.setdefault("model", model)
    return context, history


def _resolve_legacy_route(agent: str) -> tuple[str, Optional[str]]:
    normalized = (agent or "floor_manager").strip().lower()
    return _LEGACY_AGENT_ROUTE_MAP.get(normalized, ("floor-manager", None))


@router.post("/chat")
async def chat(request: dict):
    """Compatibility wrapper for the removed legacy `/api/chat` endpoint."""
    message = str(request.get("message", "")).strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    agent = str(request.get("agent", "floor_manager"))
    model = str(request.get("model", "")).strip()
    raw_context = request.get("context", {})
    normalized_context, history = _coerce_legacy_context(raw_context, model=model)
    normalized_context.setdefault("legacy_agent", agent)

    route_type, route_target = _resolve_legacy_route(agent)
    compat_request = ChatMessageRequest(
        message=message,
        context=normalized_context,
        history=history,
        stream=False,
    )

    try:
        if route_type == "department" and route_target:
            result = await department_chat(route_target, compat_request)
        else:
            result = await floor_manager_chat(compat_request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Legacy chat compatibility routing failed for agent %s: %s", agent, exc)
        raise HTTPException(status_code=502, detail=f"Legacy chat routing failed: {exc}") from exc

    reply = getattr(result, "reply", None)
    if reply is None and isinstance(result, dict):
        reply = result.get("reply")
    if reply is None:
        raise HTTPException(status_code=502, detail="Legacy chat routing returned no reply")

    return {
        "response": reply,
        "agent": agent,
        "model": model or normalized_context.get("model") or "",
    }
