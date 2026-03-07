"""
Agent Session REST API Endpoints

Provides endpoints for:
- Creating new agent sessions
- Listing saved sessions
- Getting session details
- Updating session state
- Deleting sessions
- Exporting/importing sessions
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import logging

from src.agents.memory.agent_session_persistence import (
    AgentSessionPersistence,
    get_session_persistence,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/agent-sessions", tags=["agent-sessions"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a new agent session."""
    name: str = Field(..., description="Human-readable session name")
    agent_type: str = Field(..., description="Type of agent (analyst, quant, copilot, claude)")
    session_id: Optional[str] = Field(None, description="Optional custom session ID")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session metadata")
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session variables")


class UpdateSessionRequest(BaseModel):
    """Request to update session metadata."""
    name: Optional[str] = Field(None, description="New session name")
    status: Optional[str] = Field(None, description="New status (active, completed, failed, archived)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    variables: Optional[Dict[str, Any]] = Field(None, description="Updated variables")


class AddMessageRequest(BaseModel):
    """Request to add a message to session."""
    role: str = Field(..., description="Message role (system, user, assistant, tool)")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Message metadata")


class SetHistoryRequest(BaseModel):
    """Request to replace conversation history."""
    history: List[Dict[str, Any]] = Field(..., description="Complete conversation history")


class SessionResponse(BaseModel):
    """Session response."""
    session_id: str
    name: str
    agent_type: str
    status: str
    conversation_history: List[Dict[str, Any]]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str
    modified_at: str
    completed_at: Optional[str]
    message_count: int


class SessionSummaryResponse(BaseModel):
    """Session summary for listing."""
    session_id: str
    name: str
    agent_type: str
    status: str
    created_at: str
    modified_at: str
    completed_at: Optional[str]
    message_count: int


class SessionListResponse(BaseModel):
    """List of sessions response."""
    sessions: List[SessionSummaryResponse]
    total: int
    limit: int
    offset: int


class SessionExportResponse(BaseModel):
    """Session export response."""
    session_id: str
    name: str
    agent_type: str
    exported_at: str
    export_version: str
    conversation_history: List[Dict[str, Any]]
    variables: Dict[str, Any]
    metadata: Dict[str, Any]


class ImportSessionRequest(BaseModel):
    """Request to import session."""
    session_data: Dict[str, Any] = Field(..., description="Exported session data")
    new_session_id: Optional[str] = Field(None, description="New session ID")


class MessageResponse(BaseModel):
    """Response for adding message."""
    success: bool
    session_id: str
    message_count: int


class DeleteResponse(BaseModel):
    """Delete response."""
    success: bool
    session_id: str


class StatsResponse(BaseModel):
    """Session statistics response."""
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    archived_sessions: int
    by_agent_type: Dict[str, int]


# =============================================================================
# Helper Functions
# =============================================================================

async def get_persistence() -> AgentSessionPersistence:
    """Get persistence instance."""
    return get_session_persistence()


# =============================================================================
# Session Endpoints
# =============================================================================

@router.post("/", response_model=SessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """
    Create a new agent session.

    Example:
        POST /api/agent-sessions
        {
            "name": "BTC Analysis Session",
            "agent_type": "analyst",
            "metadata": {"project": "crypto-research"},
            "variables": {"context": "initial"}
        }

        Response:
        {
            "session_id": "abc-123",
            "name": "BTC Analysis Session",
            "agent_type": "analyst",
            "status": "active",
            "conversation_history": [],
            "variables": {"context": "initial"},
            "metadata": {"project": "crypto-research"},
            "created_at": "2026-03-06T10:00:00Z",
            "modified_at": "2026-03-06T10:00:00Z",
            "completed_at": null,
            "message_count": 0
        }
    """
    try:
        persistence = await get_persistence()
        session_id = await persistence.create_session(
            name=request.name,
            agent_type=request.agent_type,
            session_id=request.session_id,
            metadata=request.metadata,
            variables=request.variables,
        )

        session = await persistence.get_session(session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")

        return SessionResponse(**session)
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> SessionListResponse:
    """
    List saved sessions with optional filtering.

    Example:
        GET /api/agent-sessions?agent_type=analyst&limit=10

        Response:
        {
            "sessions": [...],
            "total": 25,
            "limit": 10,
            "offset": 0
        }
    """
    try:
        persistence = await get_persistence()
        sessions = await persistence.list_sessions(
            agent_type=agent_type,
            status=status,
            limit=limit,
            offset=offset,
        )
        total = await persistence.get_session_count(
            agent_type=agent_type,
            status=status,
        )

        return SessionListResponse(
            sessions=[SessionSummaryResponse(**s) for s in sessions],
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_session_stats() -> StatsResponse:
    """
    Get session statistics.

    Example:
        GET /api/agent-sessions/stats

        Response:
        {
            "total_sessions": 100,
            "active_sessions": 5,
            "completed_sessions": 80,
            "archived_sessions": 15,
            "by_agent_type": {"analyst": 40, "quant": 30, "copilot": 30}
        }
    """
    try:
        persistence = await get_persistence()

        total = await persistence.get_session_count()
        active = await persistence.get_session_count(status="active")
        completed = await persistence.get_session_count(status="completed")
        archived = await persistence.get_session_count(status="archived")

        # Get count by agent type
        agent_types = ["analyst", "quant", "copilot", "claude"]
        by_agent = {}
        for at in agent_types:
            count = await persistence.get_session_count(agent_type=at)
            if count > 0:
                by_agent[at] = count

        return StatsResponse(
            total_sessions=total,
            active_sessions=active,
            completed_sessions=completed,
            archived_sessions=archived,
            by_agent_type=by_agent,
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    include_history: bool = Query(True, description="Include conversation history"),
) -> SessionResponse:
    """
    Get session details.

    Example:
        GET /api/agent-sessions/abc-123

        Response:
        {
            "session_id": "abc-123",
            "name": "BTC Analysis Session",
            ...
            "message_count": 50
        }
    """
    try:
        persistence = await get_persistence()
        session = await persistence.get_session(session_id, include_history=include_history)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionResponse(**session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
) -> Dict[str, Any]:
    """
    Update session metadata.

    Example:
        PATCH /api/agent-sessions/abc-123
        {
            "name": "Updated Name",
            "status": "completed"
        }

        Response:
        {
            "success": true,
            "session_id": "abc-123"
        }
    """
    try:
        persistence = await get_persistence()

        # Check if session exists
        existing = await persistence.get_session(session_id, include_history=False)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")

        success = await persistence.update_session(
            session_id=session_id,
            name=request.name,
            status=request.status,
            metadata=request.metadata,
            variables=request.variables,
        )

        return {"success": success, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str) -> DeleteResponse:
    """
    Delete a session.

    Example:
        DELETE /api/agent-sessions/abc-123

        Response:
        {
            "success": true,
            "session_id": "abc-123"
        }
    """
    try:
        persistence = await get_persistence()
        success = await persistence.delete_session(session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return DeleteResponse(success=success, session_id=session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post("/{session_id}/messages", response_model=MessageResponse)
async def add_message(
    session_id: str,
    request: AddMessageRequest,
) -> MessageResponse:
    """
    Add a message to session conversation.

    Example:
        POST /api/agent-sessions/abc-123/messages
        {
            "role": "user",
            "content": "Analyze BTC trend",
            "metadata": {"timestamp": "..."}
        }

        Response:
        {
            "success": true,
            "session_id": "abc-123",
            "message_count": 51
        }
    """
    try:
        persistence = await get_persistence()

        # Check if session exists
        existing = await persistence.get_session(session_id, include_history=False)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")

        success = await persistence.add_message(
            session_id=session_id,
            role=request.role,
            content=request.content,
            metadata=request.metadata,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add message")

        # Get updated message count
        session = await persistence.get_session(session_id, include_history=False)

        return MessageResponse(
            success=success,
            session_id=session_id,
            message_count=session["message_count"] if session else 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")


@router.put("/{session_id}/history", response_model=MessageResponse)
async def set_conversation_history(
    session_id: str,
    request: SetHistoryRequest,
) -> MessageResponse:
    """
    Replace entire conversation history.

    Example:
        PUT /api/agent-sessions/abc-123/history
        {
            "history": [
                {"role": "system", "content": "You are helpful", "timestamp": "..."},
                {"role": "user", "content": "Hello", "timestamp": "..."}
            ]
        }

        Response:
        {
            "success": true,
            "session_id": "abc-123",
            "message_count": 2
        }
    """
    try:
        persistence = await get_persistence()

        # Check if session exists
        existing = await persistence.get_session(session_id, include_history=False)
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")

        success = await persistence.set_conversation_history(
            session_id=session_id,
            history=request.history,
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to set history")

        session = await persistence.get_session(session_id, include_history=False)

        return MessageResponse(
            success=success,
            session_id=session_id,
            message_count=session["message_count"] if session else 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set history: {str(e)}")


# =============================================================================
# Export/Import Endpoints
# =============================================================================

@router.get("/{session_id}/export", response_model=SessionExportResponse)
async def export_session(session_id: str) -> SessionExportResponse:
    """
    Export session for backup.

    Example:
        GET /api/agent-sessions/abc-123/export

        Response:
        {
            "session_id": "abc-123",
            "name": "BTC Analysis",
            "agent_type": "analyst",
            "exported_at": "2026-03-06T10:00:00Z",
            "export_version": "1.0",
            "conversation_history": [...],
            "variables": {...},
            "metadata": {...}
        }
    """
    try:
        persistence = await get_persistence()
        session_data = await persistence.export_session(session_id)

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionExportResponse(**session_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")


@router.post("/import", response_model=SessionResponse)
async def import_session(request: ImportSessionRequest) -> SessionResponse:
    """
    Import session from backup.

    Example:
        POST /api/agent-sessions/import
        {
            "session_data": {
                "session_id": "old-123",
                "name": "Old Session",
                "agent_type": "analyst",
                "conversation_history": [...],
                ...
            },
            "new_session_id": "new-456"
        }

        Response:
        {
            "session_id": "new-456",
            "name": "Old Session",
            ...
        }
    """
    try:
        persistence = await get_persistence()
        new_session_id = await persistence.import_session(
            session_data=request.session_data,
            new_session_id=request.new_session_id,
        )

        session = await persistence.get_session(new_session_id)
        if not session:
            raise HTTPException(status_code=500, detail="Failed to import session")

        return SessionResponse(**session)
    except Exception as e:
        logger.error(f"Error importing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import session: {str(e)}")
