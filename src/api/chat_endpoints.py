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

    compact: Dict[str, Any] = {
        "canvas": _safe_text(item.get("canvas"), limit=64),
        "strategy": _safe_text(item.get("strategy"), limit=64),
        "manifest_version": _safe_text(
            item.get("manifest_version") or item.get("version"), limit=32
        ),
    }

    template = item.get("template")
    if isinstance(template, dict):
        compact["template"] = {
            "base_descriptor": _safe_text(template.get("base_descriptor"), limit=160),
            "canvas": _safe_text(template.get("canvas"), limit=64),
        }

    # Keep memory references if present (used by existing tests/logic)
    memory_identifiers = item.get("memory_identifiers")
    if isinstance(memory_identifiers, list):
        compact["memory_identifiers"] = [
            _safe_text(mid, limit=_MAX_ID_LEN)
            for mid in memory_identifiers[:64]
            if mid
        ]

    resources = item.get("resources")
    if isinstance(resources, list):
        compact_resources: List[Dict[str, Any]] = []
        for resource in resources[:_MAX_HINTS]:
            compact_resource = _compact_hint(resource)
            if compact_resource:
                compact_resources.append(compact_resource)
        if compact_resources:
            compact["resources"] = compact_resources

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

    hints = context.get("workspace_resource_hints")
    if isinstance(hints, list):
        compact_hints: List[Dict[str, Any]] = []
        for hint in hints[:_MAX_HINTS]:
            compact_hint = _compact_hint(hint)
            if compact_hint:
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
        updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now(timezone.utc).isoformat()
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
        context=getattr(session, "context", {}) or {},
    )


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(user_id: Optional[str] = None, agent_type: Optional[str] = None):
    """List chat sessions with optional filtering."""
    sessions = await _session_service.list_sessions(user_id=user_id, agent_type=agent_type)
    return [
        SessionResponse(
            id=s.id,
            agent_type=s.agent_type,
            agent_id=s.agent_id,
            title=s.title,
            user_id=s.user_id,
            created_at=s.created_at.isoformat() if s.created_at else datetime.now(timezone.utc).isoformat(),
            updated_at=s.updated_at.isoformat() if s.updated_at else datetime.now(timezone.utc).isoformat()
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

    session_id = request.session_id
    if not session_id:
        session = await _session_service.create_session(
            agent_type="floor-manager",
            agent_id="floor-manager",
            user_id="anonymous"
        )
        session_id = session.id

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
    session_id = request.session_id
    normalized_context = _normalize_chat_context(request.context)
    model_name = normalized_context.get("model") or normalized_context.get("llm_model")
    preferred_provider = normalized_context.get("provider") or normalized_context.get("llm_provider")
    if not session_id:
        session_context = dict(normalized_context)
        session = await _session_service.create_session(
            agent_type="floor-manager",
            agent_id="floor-manager",
            user_id="anonymous",
            context=session_context,
        )
        session_id = session.id

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
            try:
                async for event in fm.chat_stream(
                    message=request.message,
                    context=normalized_context,
                    history=history,
                    model_name=model_name,
                    preferred_provider=preferred_provider,
                ):
                    if event.get("type") == "content" and "delta" not in event:
                        event = {
                            **event,
                            "delta": event.get("content", ""),
                        }
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "content":
                        full_content += event.get("delta", "")
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

    session_id = request.session_id
    normalized_context = _normalize_chat_context(
        request.context,
        default_canvas=dept,
        default_department=dept,
    )
    model_name = normalized_context.get("model") or normalized_context.get("llm_model")
    preferred_provider = normalized_context.get("provider") or normalized_context.get("llm_provider")
    if not session_id:
        session_context = dict(normalized_context)
        session_context.setdefault("canvas", dept)
        session = await _session_service.create_session(
            agent_type="department",
            agent_id=dept,
            user_id="anonymous",
            context=session_context,
        )
        session_id = session.id

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
            yield f"data: {json.dumps({'type': 'tool', 'tool': 'thinking', 'status': 'started'})}\n\n"
            try:
                async for event in fm._invoke_llm_stream(
                    message=request.message,
                    history=history,
                    model_name=model_name,
                    system_prompt=dept_system_prompt,
                    preferred_provider=preferred_provider,
                ):
                    if event.get("type") == "content" and "delta" not in event:
                        event = {
                            **event,
                            "delta": event.get("content", ""),
                        }
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") == "content":
                        full_content += event.get("delta", "")
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'tool', 'tool': 'thinking', 'status': 'completed'})}\n\n"
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
