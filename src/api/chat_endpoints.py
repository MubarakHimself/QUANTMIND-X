"""
Chat API Endpoints - Session Management & Per-Agent Chat

This module provides:
- Session CRUD: /api/chat/sessions
- Per-Agent Chat: /api/chat/workshop/message, /api/chat/floor-manager/message, /api/chat/departments/{dept}/message

NOTE: Legacy endpoints (/send, /async, /pinescript, etc.) have been removed.
Use /api/workshop/copilot/* for Copilot queries
Use /api/floor-manager/* for Floor Manager queries
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.services.chat_session_service import ChatSessionService

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


class SessionResponse(BaseModel):
    """Response model for chat session."""
    id: str
    agent_type: str
    agent_id: str
    title: str
    user_id: str
    created_at: str
    updated_at: str


class ChatMessageRequest(BaseModel):
    """Request model for per-agent chat messages."""
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ChatMessageResponse(BaseModel):
    """Response model for per-agent chat messages."""
    session_id: str
    message_id: str
    reply: str
    artifacts: List[dict] = []
    action_taken: Optional[str] = None
    delegation: Optional[str] = None


# =============================================================================
# Session CRUD Endpoints
# =============================================================================


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new chat session."""
    session = await _session_service.create_session(
        agent_type=request.agent_type,
        agent_id=request.agent_id,
        user_id=request.user_id,
        title=request.title,
        context=request.context,
        metadata=request.metadata
    )
    return SessionResponse(
        id=session.id,
        agent_type=session.agent_type,
        agent_id=session.agent_id,
        title=session.title,
        user_id=session.user_id,
        created_at=session.created_at.isoformat() if session.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=session.updated_at.isoformat() if session.updated_at else datetime.now(timezone.utc).isoformat()
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


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session."""
    success = await _session_service.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# =============================================================================
# Per-Agent Chat Endpoints
# =============================================================================


@router.post("/workshop/message", response_model=ChatMessageResponse)
async def workshop_chat(request: ChatMessageRequest):
    """Chat with Workshop Copilot."""
    session_id = request.session_id
    if not session_id:
        session = await _session_service.create_session(
            agent_type="workshop",
            agent_id="copilot",
            user_id="anonymous"
        )
        session_id = session.id

    # Add user message
    await _session_service.add_message(
        session_id=session_id,
        role="user",
        content=request.message
    )

    # TODO: Process with Workshop Copilot service (placeholder)
    reply = "Workshop Copilot: I received your message."

    # Add assistant message
    assistant_msg = await _session_service.add_message(
        session_id=session_id,
        role="assistant",
        content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply
    )


@router.post("/floor-manager/message", response_model=ChatMessageResponse)
async def floor_manager_chat(request: ChatMessageRequest):
    """Chat with Floor Manager."""
    session_id = request.session_id
    if not session_id:
        session = await _session_service.create_session(
            agent_type="floor-manager",
            agent_id="floor-manager",
            user_id="anonymous"
        )
        session_id = session.id

    await _session_service.add_message(session_id=session_id, role="user", content=request.message)

    # TODO: Delegate to Floor Manager service (placeholder)
    reply = "Floor Manager: Message received."

    assistant_msg = await _session_service.add_message(
        session_id=session_id, role="assistant", content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply
    )


@router.post("/departments/{dept}/message", response_model=ChatMessageResponse)
async def department_chat(dept: str, request: ChatMessageRequest):
    """Chat with a department agent."""
    valid_depts = ["research", "development", "risk", "trading", "portfolio"]
    if dept not in valid_depts:
        raise HTTPException(status_code=400, detail=f"Invalid department: {dept}")

    session_id = request.session_id
    if not session_id:
        session = await _session_service.create_session(
            agent_type="department",
            agent_id=dept,
            user_id="anonymous"
        )
        session_id = session.id

    await _session_service.add_message(session_id=session_id, role="user", content=request.message)

    # TODO: Delegate to department service (placeholder)
    reply = f"Department {dept}: Message received."

    assistant_msg = await _session_service.add_message(
        session_id=session_id, role="assistant", content=reply
    )

    return ChatMessageResponse(
        session_id=session_id,
        message_id=assistant_msg.id,
        reply=reply
    )
