"""
Chat API Endpoints - Session Management & Per-Agent Chat

This module provides:
- Session CRUD: /api/chat/sessions
- Per-Agent Chat: /api/chat/workshop/message, /api/chat/floor-manager/message, /api/chat/departments/{dept}/message

NOTE: Legacy endpoints (/send, /async, /pinescript, etc.) have been removed.
Use /api/workshop/* or /api/floor-manager/* for assistant queries
Use /api/floor-manager/* for Floor Manager queries
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.services.chat_session_service import ChatSessionService
from src.agents.departments.types import Department

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Service instance
_session_service = ChatSessionService()


# =============================================================================
# Request/Response Models
# =============================================================================


class CreateSessionRequest(BaseModel):
    """Request to create a new chat session."""
    agent_type: str
    agent_id: str
    user_id: str
    title: Optional[str] = None
    context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class UpdateSessionRequest(BaseModel):
    """Request to update mutable chat session fields."""
    title: str


class SessionResponse(BaseModel):
    """Response model for chat session."""
    id: str
    agent_type: str
    agent_id: str
    title: str
    user_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    context: Dict[str, Any] = {}  # includes prior_session_memory, canvas_context


class StoredChatMessageResponse(BaseModel):
    """Stored chat message payload for session history hydration."""
    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    metadata: Dict[str, Any] = {}


class ChatMessageRequest(BaseModel):
    """Request model for per-agent chat messages."""
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None
    stream: bool = False


class ChatMessageResponse(BaseModel):
    """Response model for per-agent chat messages."""
    session_id: str
    message_id: str
    reply: str
    artifacts: List[dict] = []
    action_taken: Optional[str] = None
    delegation: Optional[Dict[str, Any] | str] = None


# =============================================================================
# Context Normalization Helpers (manifest-first contract)
# =============================================================================

_ALLOWED_CONTEXT_KEYS = {
    "canvas",
    "department",
    "model",
    "llm_model",
    "provider",
    "llm_provider",
    "session_type",
    "workflow_id",
    "workflow_step",
    "workflow_run_id",
    "request_id",
}
_MAX_HINTS = 32
_MAX_ATTACHMENTS = 24
_MAX_CONTEXT_STR = 512
_MAX_ID_LEN = 256
_GLOBAL_HINT_CANVASES = {"workshop", "flowforge", "floor-manager"}


def _safe_text(value: Any, *, limit: int = _MAX_CONTEXT_STR) -> str:
    """Best-effort bounded string coercion for context values."""
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _compact_hint(hint: Any) -> Optional[Dict[str, Any]]:
    """Compact one workspace resource hint into a bounded descriptor."""
    if not isinstance(hint, dict):
        return None
    compact = {
        "id": _safe_text(hint.get("id"), limit=_MAX_ID_LEN),
        "canvas": _safe_text(hint.get("canvas"), limit=64),
        "path": _safe_text(hint.get("path"), limit=384),
        "label": _safe_text(hint.get("label"), limit=160),
        "type": _safe_text(hint.get("type"), limit=48),
        "description": _safe_text(hint.get("description"), limit=240),
    }
    relevance = hint.get("relevance")
    if isinstance(relevance, (int, float)):
        compact["relevance"] = max(0.0, min(float(relevance), 1.0))
    return {k: v for k, v in compact.items() if v not in ("", None)}


def _compact_attachment_context(item: Any) -> Optional[Dict[str, Any]]:
    """
    Preserve attachment semantics while stripping heavy payloads.

    We keep references (ids, paths, canvases, memory identifiers) and discard
    full inlined blobs to keep chat context deployment-safe.
    """
    if not isinstance(item, dict):
        return None

    payload = item.get("context") if isinstance(item.get("context"), dict) else item

    compact: Dict[str, Any] = {
        "canvas": _safe_text(item.get("canvas") or payload.get("canvas"), limit=64),
        "label": _safe_text(item.get("label") or payload.get("label"), limit=160),
        "strategy": _safe_text(item.get("strategy") or payload.get("strategy"), limit=64),
        "manifest_version": _safe_text(
            payload.get("manifest_version") or payload.get("version"), limit=32
        ),
    }

    attachment_type = _safe_text(payload.get("attachment_type"), limit=32)
    if attachment_type:
        compact["attachment_type"] = attachment_type

    resource = payload.get("resource")
    if isinstance(resource, dict):
        compact_resource = _compact_hint(resource)
        if compact_resource:
            compact["resource"] = compact_resource

    template = payload.get("template") or payload.get("template_manifest")
    if isinstance(template, dict):
        compact["template"] = {
            "base_descriptor": _safe_text(template.get("base_descriptor"), limit=160),
            "canvas": _safe_text(template.get("canvas"), limit=64),
        }

    # Keep memory references if present (used by existing tests/logic)
    memory_identifiers = payload.get("memory_identifiers")
    if isinstance(memory_identifiers, list):
        compact["memory_identifiers"] = [
            _safe_text(mid, limit=_MAX_ID_LEN)
            for mid in memory_identifiers[:64]
            if mid
        ]

    resources = payload.get("resources")
    if isinstance(resources, list):
        compact_resources: List[Dict[str, Any]] = []
        for resource in resources[:_MAX_HINTS]:
            compact_resource = _compact_hint(resource)
            if compact_resource:
                compact_resources.append(compact_resource)
        if compact_resources:
            compact["resources"] = compact_resources

    runtime_manifest = payload.get("runtime_manifest")
    if isinstance(runtime_manifest, dict):
        compact["runtime_manifest"] = {
            "canvas": _safe_text(runtime_manifest.get("canvas"), limit=64),
            "total_resources": runtime_manifest.get("total_resources"),
            "by_type": runtime_manifest.get("by_type"),
            "sample": [
                hint
                for hint in (
                    _compact_hint(sample) for sample in (runtime_manifest.get("sample") or [])[:12]
                )
                if hint
            ],
        }
        compact["runtime_manifest"] = {
            key: value
            for key, value in compact["runtime_manifest"].items()
            if value not in ("", None, [], {})
        }

    # Remove empty keys
    compact = {
        k: v
        for k, v in compact.items()
        if v not in ("", None, [], {})
    }
    return compact or None


def _normalize_workspace_contract(raw: Any) -> Dict[str, Any]:
    """Normalize workspace contract and force safe defaults."""
    if not isinstance(raw, dict):
        raw = {}

    version = _safe_text(raw.get("version"), limit=32) or "manifest-v1"
    strategy = _safe_text(raw.get("strategy"), limit=48) or "manifest-first"
    if strategy not in {"manifest-first", "legacy-full"}:
        strategy = "manifest-first"

    return {
        "version": version,
        "strategy": strategy,
        "natural_resource_search": bool(raw.get("natural_resource_search", True)),
    }


def _normalize_session_type(raw: Any, context: Optional[Dict[str, Any]] = None) -> str:
    """Normalize request session type to contract values."""
    aliases = {
        "interactive": "interactive_session",
        "interactive_session": "interactive_session",
        "chat": "interactive_session",
        "workflow": "workflow_session",
        "workflow_session": "workflow_session",
        "harness": "workflow_session",
        "autonomous": "workflow_session",
        "background": "workflow_session",
    }
    value = _safe_text(raw, limit=64).lower()
    if value in aliases:
        return aliases[value]
    context = context or {}
    if any(context.get(key) for key in ("workflow_id", "workflow_run_id", "workflow_step")):
        return "workflow_session"
    return "interactive_session"


def _normalize_chat_context(
    context: Optional[Dict[str, Any]],
    *,
    default_canvas: Optional[str] = None,
    default_department: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize incoming request context to a deployment-safe compact contract.

    - Keep model/provider/canvas routing keys.
    - Keep only lightweight resource descriptors for canvas attachments.
    - Keep workspace_resource_hints for natural search guidance.
    """
    if not isinstance(context, dict):
        context = {}

    normalized: Dict[str, Any] = {}
    for key in _ALLOWED_CONTEXT_KEYS:
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = (
                _safe_text(value, limit=_MAX_CONTEXT_STR)
                if isinstance(value, str)
                else value
            )

    normalized["workspace_contract"] = _normalize_workspace_contract(
        context.get("workspace_contract")
    )
    normalized["session_type"] = _normalize_session_type(
        normalized.get("session_type") or context.get("session_type"),
        normalized,
    )

    allowed_hint_canvases = set()
    active_canvas = _safe_text(
        context.get("active_canvas") or context.get("canvas") or default_canvas,
        limit=64,
    )
    normalized_department = _safe_text(default_department, limit=64)
    if active_canvas and active_canvas not in _GLOBAL_HINT_CANVASES:
        allowed_hint_canvases.add(active_canvas)
        allowed_hint_canvases.add("shared-assets")
    elif normalized_department and normalized_department not in _GLOBAL_HINT_CANVASES:
        allowed_hint_canvases.add(normalized_department)
        allowed_hint_canvases.add("shared-assets")

    hints = context.get("workspace_resource_hints")
    if isinstance(hints, list):
        compact_hints: List[Dict[str, Any]] = []
        for hint in hints[:_MAX_HINTS]:
            compact_hint = _compact_hint(hint)
            if compact_hint:
                hint_canvas = _safe_text(compact_hint.get("canvas"), limit=64)
                if allowed_hint_canvases and hint_canvas and hint_canvas not in allowed_hint_canvases:
                    continue
                compact_hints.append(compact_hint)
        if compact_hints:
            normalized["workspace_resource_hints"] = compact_hints

    attachments: List[Dict[str, Any]] = []
    for key in ("attached_contexts", "attachments"):
        raw_attachments = context.get(key)
        if not isinstance(raw_attachments, list):
            continue
        for item in raw_attachments[:_MAX_ATTACHMENTS]:
            compact_item = _compact_attachment_context(item)
            if compact_item:
                attachments.append(compact_item)
    if attachments:
        normalized["attached_contexts"] = attachments

    # Keep minimal canvas metadata from preloaded session context only.
    canvas_context = context.get("canvas_context")
    if isinstance(canvas_context, dict):
        normalized_canvas_context = {
            "canvas": _safe_text(canvas_context.get("canvas"), limit=64),
            "display_name": _safe_text(canvas_context.get("display_name"), limit=128),
            "department_head": _safe_text(
                canvas_context.get("department_head"), limit=96
            ),
            "workflow_namespaces": canvas_context.get("workflow_namespaces", [])[:12]
            if isinstance(canvas_context.get("workflow_namespaces"), list)
            else [],
            "skills": canvas_context.get("skills", [])[:64]
            if isinstance(canvas_context.get("skills"), list)
            else [],
        }
        normalized["canvas_context"] = {
            k: v for k, v in normalized_canvas_context.items() if v not in ("", None, [])
        }

    if default_canvas:
        normalized.setdefault("canvas", default_canvas)
    if default_department:
        normalized.setdefault("department", default_department)

    return normalized


async def _ensure_chat_session(
    *,
    requested_session_id: Optional[str],
    agent_type: str,
    agent_id: str,
    context: Optional[Dict[str, Any]] = None,
    user_id: str = "anonymous",
) -> str:
    """
    Resolve a requested session id, creating a new session when missing/stale.

    Prevents runtime 500s when the frontend sends a cached session_id that no
    longer exists in persistent storage.
    """
    get_session = getattr(_session_service, "get_session", None)
    create_session = getattr(_session_service, "create_session", None)

    # Test stubs sometimes monkeypatch a partial session service. In that case,
    # preserve the caller-provided session id instead of forcing recreation.
    if requested_session_id and not callable(get_session):
        return requested_session_id

    if requested_session_id:
        try:
            existing = await get_session(requested_session_id)
            if existing:
                return requested_session_id
            logger.warning(
                "Requested session %s not found for %s/%s; creating a new session.",
                requested_session_id,
                agent_type,
                agent_id,
            )
        except Exception as exc:
            logger.warning(
                "Session lookup failed for %s (%s/%s): %s; creating a new session.",
                requested_session_id,
                agent_type,
                agent_id,
                exc,
            )

    if not callable(create_session):
        if requested_session_id:
            return requested_session_id
        raise RuntimeError(
            "Session service cannot create sessions (create_session unavailable)."
        )

    session = await create_session(
        agent_type=agent_type,
        agent_id=agent_id,
        user_id=user_id,
        context=dict(context or {}),
    )
    return session.id


def _normalize_sse_event(event: Dict[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize chat streaming events to one consistent frontend contract.

    Contract:
    - content: {type: "content", delta: str}
    - tool: {type: "tool", tool: str, status?: str}
    - status: {type: "status", phase: str, status: str, message?: str}
    - error: {type: "error", error: str}
    - done: {type: "done", session_id?: str}
    """
    event_type = str(event.get("type") or "").strip().lower()

    if event_type == "content":
        raw_delta = event.get("delta")
        if raw_delta is None:
            raw_delta = event.get("content")
        if raw_delta is None:
            delta = ""
        else:
            delta = str(raw_delta)
            if len(delta) > 4096:
                delta = delta[:4096]
        return {
            "type": "content",
            "delta": delta,
        }

    if event_type == "tool":
        normalized = {
            "type": "tool",
            "tool": _safe_text(event.get("tool"), limit=96) or "tool",
        }
        status = _safe_text(event.get("status"), limit=64)
        if status:
            normalized["status"] = status
            normalized["phase"] = normalized["tool"]
        return normalized

    if event_type in {"thinking", "thinking_start", "thinking_chunk", "status"}:
        status = _safe_text(event.get("status"), limit=64)
        message = _safe_text(event.get("message") or event.get("content"), limit=2048)
        phase = _safe_text(event.get("phase"), limit=64) or "thinking"

        if event_type == "thinking_start":
            status = status or "started"
        elif event_type == "thinking_chunk":
            status = status or "streaming"
        elif event_type == "thinking":
            status = status or ("streaming" if message else "started")
        else:
            status = status or "progress"

        normalized = {"type": "status", "phase": phase, "status": status}
        if message:
            normalized["message"] = message
        return normalized

    if event_type == "delegation":
        normalized = {"type": "delegation"}
        for key in ("department", "task_id", "status"):
            value = event.get(key)
            if value is not None:
                normalized[key] = value
        return normalized

    if event_type == "done":
        normalized = {"type": "done"}
        sid = event.get("session_id") or session_id
        if sid:
            normalized["session_id"] = sid
        return normalized

    if event_type == "error":
        return {
            "type": "error",
            "error": _safe_text(event.get("error") or event.get("content") or "Streaming error", limit=2048),
        }

    # Unknown event: emit as status for observability.
    return {
        "type": "status",
        "phase": _safe_text(event_type or "stream", limit=64),
        "status": "progress",
        "message": _safe_text(event.get("content") or event.get("message"), limit=1024),
    }


_PROVISIONAL_STREAM_PREFIXES = (
    "i'll ",
    "i will ",
    "let me ",
    "first, i'll ",
    "first i will ",
    "i am going to ",
    "i'm going to ",
)


def _looks_like_provisional_stream_reply(content: Optional[str]) -> bool:
    """
    Detect short streaming preambles that are not useful final assistant replies.

    Some providers emit a single acknowledgement chunk like "I'll search the
    workspace..." and then terminate the stream without sending a real answer.
    Those should be buffered so the endpoint can fall back to the non-streaming
    completion path instead of persisting the preamble as the final reply.
    """
    if not content:
        return False

    normalized = " ".join(str(content).strip().split()).lower()
    if not normalized or len(normalized) > 280:
        return False

    return normalized.startswith(_PROVISIONAL_STREAM_PREFIXES)


# =============================================================================
# Session CRUD Endpoints
# =============================================================================


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    normalized_context = _normalize_chat_context(request.context)
    session = await _session_service.create_session(
        agent_type=request.agent_type,
        agent_id=request.agent_id,
        user_id=request.user_id,
        title=request.title,
        context=normalized_context,
        metadata=request.metadata
    )
    return SessionResponse(
        id=session.id,
        agent_type=session.agent_type,
        agent_id=session.agent_id,
        title=session.title,
        user_id=session.user_id,
        created_at=session.created_at.isoformat() if session.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now(timezone.utc).isoformat(),
        message_count=0,
        context=getattr(session, "context", {}) or {},
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get a chat session by ID."""
    session = await _session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        id=session.id,
        agent_type=session.agent_type,
        agent_id=session.agent_id,
        title=session.title,
        user_id=session.user_id,
        created_at=session.created_at.isoformat() if session.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now(timezone.utc).isoformat(),
        message_count=len(await _session_service.get_messages(session_id=session.id, limit=1000)),
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update mutable fields for a chat session."""
    title = (request.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title must not be empty")

    session = await _session_service.update_session_title(session_id, title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session.id,
        agent_type=session.agent_type,
        agent_id=session.agent_id,
        title=session.title,
        user_id=session.user_id,
        created_at=session.created_at.isoformat() if session.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now(timezone.utc).isoformat(),
        message_count=len(await _session_service.get_messages(session_id=session.id, limit=1000)),
        context=getattr(session, "context", {}) or {},
    )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    user_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    exclude_empty: bool = False,
):
    """List chat sessions with optional filtering."""
    sessions = await _session_service.list_sessions(
        user_id=user_id,
        agent_type=agent_type,
        agent_id=agent_id,
        exclude_empty=exclude_empty,
    )
    return [
        SessionResponse(
            id=s.id,
            agent_type=s.agent_type,
            agent_id=s.agent_id,
            title=s.title,
            user_id=s.user_id,
            created_at=s.created_at.isoformat() if s.created_at else datetime.now(timezone.utc).isoformat(),
            updated_at=s.updated_at.isoformat() if s.updated_at else datetime.now(timezone.utc).isoformat(),
            message_count=int(getattr(s, "message_count", 0) or 0),
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages", response_model=List[StoredChatMessageResponse])
async def get_session_messages(session_id: str, limit: int = 100):
    """Get stored messages for a chat session in chronological order."""
    session = await _session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await _session_service.get_messages(session_id=session_id, limit=limit)
    return [
        StoredChatMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            created_at=msg.created_at.isoformat()
            if msg.created_at else datetime.now(timezone.utc).isoformat(),
            metadata={
                key: value for key, value in {
                    "artifacts": msg.artifacts or [],
                    "tool_calls": msg.tool_calls or [],
                    "token_count": msg.token_count,
                }.items() if value not in (None, [], {})
            },
        )
        for msg in messages
    ]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    success = await _session_service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/sessions/{session_id}/compact")
async def compact_session(session_id: str):
    """
    POST /api/chat/sessions/{session_id}/compact

    Compact session context by summarising older messages and saving a summary
    to graph memory as an OPINION node.  Returns the number of messages kept
    and a short summary string so the frontend can display a system notice.
    """
    session = await _session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Fetch all messages for this session
    try:
        all_messages = await _session_service.get_messages(session_id=session_id, limit=500)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch messages: {e}")

    KEEP_RECENT = 20
    if len(all_messages) <= KEEP_RECENT:
        return {
            "compacted": False,
            "messages_kept": len(all_messages),
            "summary": "Context is already compact — no action taken.",
        }

    older = all_messages[:-KEEP_RECENT]

    # Build a brief summary from the older messages
    lines: List[str] = []
    for msg in older[-40:]:  # cap summary source at 40 msgs to keep it readable
        prefix = "User" if msg.role == "user" else "Assistant"
        snippet = (msg.content or "")[:120].replace("\n", " ")
        lines.append(f"{prefix}: {snippet}")
    summary_text = "Context compaction summary:\n" + "\n".join(lines)

    # Save to graph memory as an OPINION node
    try:
        import os
        from src.memory.graph.facade import get_graph_memory
        from src.memory.graph.types import MemoryNodeType, MemoryCategory, MemoryNode

        db_path = os.environ.get("GRAPH_MEMORY_DB", "data/graph_memory.db")
        facade = get_graph_memory(db_path=db_path)

        node = MemoryNode(
            node_type=MemoryNodeType.OPINION,
            category=MemoryCategory.EXPERIENTIAL,
            title=f"Session compaction — {session_id[:8]}",
            content=summary_text,
            confidence=0.9,
            importance=0.7,
            agent_id="system",
            session_id=session_id,
            tags=["compaction", "session-summary"],
        )
        facade.store.create_node(node)
    except Exception as e:
        logger.warning(f"Could not save compaction summary to graph memory: {e}")

    short_summary = f"Summarised {len(older)} older messages and saved to memory."
    return {
        "compacted": True,
        "messages_kept": KEEP_RECENT,
        "total_before": len(all_messages),
        "summary": short_summary,
    }


# =============================================================================
# Per-Agent Chat Endpoints
# =============================================================================


@router.post("/workshop/message", response_model=ChatMessageResponse)
async def workshop_chat(request: ChatMessageRequest):
    """Compatibility workshop chat endpoint backed by Floor Manager sessions."""
    from src.api.services.workshop_copilot_service import (
        WorkshopCopilotRequest,
        get_workshop_copilot_service,
    )

    normalized_context = _normalize_chat_context(request.context)
    session_id = await _ensure_chat_session(
        requested_session_id=request.session_id,
        agent_type="floor-manager",
        agent_id="floor-manager",
        context=normalized_context,
    )

    # Add user message
    await _session_service.add_message(
        session_id=session_id,
        role="user",
        content=request.message
    )

    history = []
    try:
        messages = await _session_service.get_messages(session_id=session_id, limit=20)
        history = [{"role": msg.role, "content": msg.content} for msg in messages]
    except Exception:
        pass

    service = get_workshop_copilot_service()
    result = await service.handle_message(
        WorkshopCopilotRequest(
            message=request.message,
            history=history,
            session_id=session_id,
        )
    )
    reply = result.reply

    # Add assistant message
    assistant_msg = await _session_service.add_message(
        session_id=session_id,
        role="assistant",
        content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply,
        action_taken=result.action_taken,
        delegation=result.delegation,
    )


@router.post("/floor-manager/message", response_model=ChatMessageResponse)
async def floor_manager_chat(request: ChatMessageRequest):
    """Chat with Floor Manager."""
    normalized_context = _normalize_chat_context(request.context)
    model_name = normalized_context.get("model") or normalized_context.get("llm_model")
    preferred_provider = normalized_context.get("provider") or normalized_context.get("llm_provider")
    session_id = await _ensure_chat_session(
        requested_session_id=request.session_id,
        agent_type="floor-manager",
        agent_id="floor-manager",
        context=normalized_context,
    )

    # Add user message to session
    await _session_service.add_message(session_id=session_id, role="user", content=request.message)

    # Get conversation history for context
    history = request.history or []
    try:
        messages = await _session_service.get_messages(session_id=session_id, limit=20)
        history = [{"role": msg.role, "content": msg.content} for msg in messages]
    except Exception:
        pass

    if request.stream:
        async def event_generator():
            full_content = ""
            buffered_content = ""
            buffered_content_emitted = False
            content_chunk_count = 0
            done_emitted = False
            last_status_signature: Optional[tuple[str, str, str]] = None
            try:
                async for event in fm.chat_stream(
                    message=request.message,
                    context=normalized_context,
                    history=history,
                    model_name=model_name,
                    preferred_provider=preferred_provider,
                ):
                    normalized_event = _normalize_sse_event(event, session_id=session_id)
                    if normalized_event.get("type") == "status":
                        status_signature = (
                            str(normalized_event.get("phase") or ""),
                            str(normalized_event.get("status") or ""),
                            str(normalized_event.get("message") or ""),
                        )
                        if status_signature == last_status_signature:
                            continue
                        last_status_signature = status_signature
                    else:
                        last_status_signature = None
                    if normalized_event.get("type") == "content":
                        delta = str(normalized_event.get("delta") or "")
                        full_content += delta
                        buffered_content += delta
                        content_chunk_count += 1

                        should_flush_buffer = (
                            content_chunk_count > 1
                            or not _looks_like_provisional_stream_reply(buffered_content)
                        )
                        if buffered_content_emitted or should_flush_buffer:
                            emit_delta = buffered_content if not buffered_content_emitted else delta
                            if emit_delta:
                                yield f"data: {json.dumps({'type': 'content', 'delta': emit_delta})}\n\n"
                            buffered_content = ""
                            buffered_content_emitted = True
                        continue
                    if normalized_event.get("type") == "done":
                        done_emitted = True
                        continue
                    yield f"data: {json.dumps(normalized_event)}\n\n"

                # Some providers may stream only status/thinking events.
                # Ensure the UI always receives a final assistant reply.
                needs_fallback = (
                    not full_content.strip()
                    or (
                        not buffered_content_emitted
                        and _looks_like_provisional_stream_reply(buffered_content)
                    )
                )
                if needs_fallback:
                    fallback = await fm.chat(
                        message=request.message,
                        context=normalized_context,
                        history=history,
                        model_name=model_name,
                        preferred_provider=preferred_provider,
                        stream=False,
                    )
                    fallback_reply = str(fallback.get("content") or "").strip()
                    if fallback_reply:
                        full_content = fallback_reply
                        yield f"data: {json.dumps({'type': 'content', 'delta': fallback_reply})}\n\n"
                        buffered_content = ""
                        buffered_content_emitted = True
                elif buffered_content and not buffered_content_emitted:
                    yield f"data: {json.dumps({'type': 'content', 'delta': buffered_content})}\n\n"
                    buffered_content = ""
                    buffered_content_emitted = True
                if done_emitted or full_content:
                    yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            except Exception as e:
                logger.error(f"Floor Manager streaming failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            finally:
                if full_content:
                    try:
                        await _session_service.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=full_content,
                        )
                    except Exception:
                        pass

        from src.agents.departments.floor_manager import get_floor_manager
        fm = get_floor_manager()
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Call Floor Manager with LLM
    from src.agents.departments.floor_manager import get_floor_manager
    fm = get_floor_manager()
    result = await fm.chat(
        message=request.message,
        context=normalized_context,
        history=history,
        model_name=model_name,
        preferred_provider=preferred_provider,
        stream=False,
    )

    reply = result.get("content", "Floor Manager: Message received.")
    if result.get("delegation"):
        delegation = result["delegation"]
    else:
        delegation = None

    assistant_msg = await _session_service.add_message(
        session_id=session_id, role="assistant", content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply,
        delegation=delegation,
    )


@router.post("/departments/{dept}/message")
async def department_chat(dept: str, request: ChatMessageRequest):
    """Chat with a department agent - each department has its own LLM-backed endpoint."""
    valid_depts = ["research", "development", "risk", "trading", "portfolio"]
    if dept not in valid_depts:
        raise HTTPException(status_code=400, detail=f"Invalid department: {dept}")

    normalized_context = _normalize_chat_context(
        request.context,
        default_canvas=dept,
        default_department=dept,
    )
    model_name = normalized_context.get("model") or normalized_context.get("llm_model")
    preferred_provider = normalized_context.get("provider") or normalized_context.get("llm_provider")
    session_context = dict(normalized_context)
    session_context.setdefault("canvas", dept)
    session_id = await _ensure_chat_session(
        requested_session_id=request.session_id,
        agent_type="department",
        agent_id=dept,
        context=session_context,
    )

    await _session_service.add_message(session_id=session_id, role="user", content=request.message)

    # Get conversation history for context
    history = []
    try:
        messages = await _session_service.get_messages(session_id=session_id, limit=20)
        history = [{"role": msg.role, "content": msg.content} for msg in messages]
    except Exception:
        pass

    from src.agents.departments.floor_manager import get_floor_manager
    fm = get_floor_manager()
    try:
        dept_key = Department(dept)
    except ValueError:
        dept_key = None
    dept_head = fm._department_heads.get(dept_key) if dept_key else None

    merged_canvas_context = {
        **normalized_context,
        "history": history,
        "department": dept,
    }
    if model_name:
        merged_canvas_context["model"] = model_name
    if preferred_provider:
        merged_canvas_context["provider"] = preferred_provider
    merged_canvas_context.setdefault("canvas", dept)

    # Get department system prompt for the streaming path
    dept_system_prompt = None
    if dept_head:
        try:
            dept_system_prompt = dept_head._build_system_prompt(
                canvas_context=merged_canvas_context,
            )
        except Exception:
            dept_system_prompt = dept_head.system_prompt if hasattr(dept_head, 'system_prompt') else None

    # SSE streaming path
    if request.stream:
        async def event_generator():
            full_content = ""
            buffered_content = ""
            buffered_content_emitted = False
            content_chunk_count = 0
            done_emitted = False
            last_status_signature: Optional[tuple[str, str, str]] = None
            initial_status = {'type': 'status', 'phase': 'thinking', 'status': 'started'}
            yield f"data: {json.dumps(initial_status)}\n\n"
            last_status_signature = ("thinking", "started", "")
            try:
                if dept_head:
                    yield f"data: {json.dumps({'type': 'status', 'phase': 'thinking', 'status': 'streaming'})}\n\n"
                    last_status_signature = ("thinking", "streaming", "")
                    result = await dept_head._invoke_claude(
                        task=request.message,
                        canvas_context=merged_canvas_context,
                        tools=None,
                    )
                    full_content = str(result.get("content") or "").strip()
                    if full_content:
                        yield f"data: {json.dumps({'type': 'content', 'delta': full_content})}\n\n"
                        buffered_content_emitted = True
                    elif result.get("error"):
                        yield f"data: {json.dumps({'type': 'error', 'error': str(result['error'])})}\n\n"
                else:
                    async for event in fm._invoke_llm_stream(
                        message=request.message,
                        history=history,
                        model_name=model_name,
                        system_prompt=dept_system_prompt,
                        preferred_provider=preferred_provider,
                    ):
                        normalized_event = _normalize_sse_event(event, session_id=session_id)
                        if normalized_event.get("type") == "status":
                            status_signature = (
                                str(normalized_event.get("phase") or ""),
                                str(normalized_event.get("status") or ""),
                                str(normalized_event.get("message") or ""),
                            )
                            if status_signature == last_status_signature:
                                continue
                            last_status_signature = status_signature
                        else:
                            last_status_signature = None
                        if normalized_event.get("type") == "content":
                            delta = str(normalized_event.get("delta") or "")
                            full_content += delta
                            buffered_content += delta
                            content_chunk_count += 1

                            should_flush_buffer = (
                                content_chunk_count > 1
                                or not _looks_like_provisional_stream_reply(buffered_content)
                            )
                            if buffered_content_emitted or should_flush_buffer:
                                emit_delta = buffered_content if not buffered_content_emitted else delta
                                if emit_delta:
                                    yield f"data: {json.dumps({'type': 'content', 'delta': emit_delta})}\n\n"
                                buffered_content = ""
                                buffered_content_emitted = True
                            continue
                        if normalized_event.get("type") == "done":
                            done_emitted = True
                            continue
                        yield f"data: {json.dumps(normalized_event)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

            # Some providers may stream only status/thinking events.
            # Ensure a final assistant reply is still emitted to the UI.
            needs_fallback = (
                not full_content.strip()
                or (
                    not buffered_content_emitted
                    and _looks_like_provisional_stream_reply(buffered_content)
                )
            )
            if needs_fallback:
                try:
                    if dept_head:
                        fallback_result = await dept_head._invoke_claude(
                            task=request.message,
                            canvas_context=merged_canvas_context,
                            tools=None,
                        )
                        fallback_reply = str(fallback_result.get("content") or "").strip()
                    else:
                        fallback_result = await fm.chat(
                            message=f"[{dept.upper()}] {request.message}",
                            context=merged_canvas_context,
                            history=history,
                            stream=False,
                            model_name=model_name,
                            preferred_provider=preferred_provider,
                        )
                        fallback_reply = str(fallback_result.get("content") or "").strip()
                    if fallback_reply:
                        full_content = fallback_reply
                        yield f"data: {json.dumps({'type': 'content', 'delta': fallback_reply})}\n\n"
                        buffered_content = ""
                        buffered_content_emitted = True
                except Exception as fallback_error:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(fallback_error)})}\n\n"
            elif buffered_content and not buffered_content_emitted:
                yield f"data: {json.dumps({'type': 'content', 'delta': buffered_content})}\n\n"
                buffered_content = ""
                buffered_content_emitted = True

            yield f"data: {json.dumps({'type': 'status', 'phase': 'thinking', 'status': 'completed'})}\n\n"
            if done_emitted or full_content:
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"
            # Persist assistant message
            if full_content:
                try:
                    await _session_service.add_message(
                        session_id=session_id, role="assistant", content=full_content
                    )
                except Exception:
                    pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming path
    if dept_head:
        result = await dept_head._invoke_claude(
            task=request.message,
            canvas_context=merged_canvas_context,
            tools=None,
        )
        reply = result.get("content", f"{dept.title()} Department response")
    else:
        result = await fm.chat(
            message=f"[{dept.upper()}] {request.message}",
            context=merged_canvas_context,
            history=history,
            stream=False,
        )
        reply = result.get("content", f"{dept.title()} Department: Message received.")

    assistant_msg = await _session_service.add_message(
        session_id=session_id, role="assistant", content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply
    )
