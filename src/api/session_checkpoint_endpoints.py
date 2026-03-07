"""
Session Checkpoint REST API Endpoints

Provides endpoints for:
- Creating session checkpoints (manual and auto)
- Listing checkpoints
- Getting checkpoint details
- Restoring sessions from checkpoints
- Deleting checkpoints
- Managing old checkpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import logging

from src.agents.memory.session_checkpoint_service import (
    SessionCheckpointService,
    get_checkpoint_service,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/session-checkpoints", tags=["session-checkpoints"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateCheckpointRequest(BaseModel):
    """Request to create a checkpoint."""
    session_id: str = Field(..., description="Session ID to checkpoint")
    checkpoint_type: str = Field(default="manual", description="Type: manual, auto, scheduled")
    progress_percent: float = Field(default=0.0, description="Progress percentage")
    current_step: Optional[str] = Field(None, description="Current step description")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class RestoreCheckpointRequest(BaseModel):
    """Request to restore from checkpoint."""
    checkpoint_id: int = Field(..., description="Checkpoint ID to restore from")
    target_session_id: Optional[str] = Field(None, description="Target session ID (creates new if not provided)")


class DeleteOldCheckpointsRequest(BaseModel):
    """Request to delete old checkpoints."""
    session_id: str = Field(..., description="Session ID to clean up")
    keep_last: int = Field(default=5, description="Number of recent checkpoints to keep")


class CheckpointResponse(BaseModel):
    """Checkpoint response."""
    id: int
    session_id: str
    checkpoint_number: int
    checkpoint_type: str
    conversation_history: List[Dict[str, Any]]
    variables: Dict[str, Any]
    progress_percent: float
    current_step: Optional[str]
    metadata: Dict[str, Any]
    created_at: str
    size_bytes: int
    message_count: int


class CheckpointSummaryResponse(BaseModel):
    """Checkpoint summary for listing."""
    id: int
    session_id: str
    checkpoint_number: int
    checkpoint_type: str
    progress_percent: float
    current_step: Optional[str]
    created_at: str
    size_bytes: int
    message_count: int


class CheckpointListResponse(BaseModel):
    """List of checkpoints response."""
    checkpoints: List[CheckpointSummaryResponse]
    total: int
    limit: int
    offset: int


class CreateCheckpointResponse(BaseModel):
    """Response for creating checkpoint."""
    success: bool
    checkpoint_id: str
    session_id: str
    checkpoint_number: int


class RestoreCheckpointResponse(BaseModel):
    """Response for restoring checkpoint."""
    success: bool
    session_id: str
    restored_from_checkpoint: int


class DeleteResponse(BaseModel):
    """Delete response."""
    success: bool
    deleted_count: int


class CheckpointStatsResponse(BaseModel):
    """Checkpoint statistics response."""
    total_checkpoints: int
    by_type: Dict[str, int]
    total_size_mb: float
    oldest_checkpoint: Optional[str]
    newest_checkpoint: Optional[str]


# =============================================================================
# Helper Functions
# =============================================================================

async def get_service() -> SessionCheckpointService:
    """Get checkpoint service instance."""
    return get_checkpoint_service()


# =============================================================================
# Checkpoint Endpoints
# =============================================================================

@router.post("/", response_model=CreateCheckpointResponse, status_code=201)
async def create_checkpoint(request: CreateCheckpointRequest) -> CreateCheckpointResponse:
    """
    Create a checkpoint for a session.

    Example:
        POST /api/session-checkpoints
        {
            "session_id": "abc-123",
            "checkpoint_type": "manual",
            "progress_percent": 50.0,
            "current_step": "Processing data"
        }

        Response:
        {
            "success": true,
            "checkpoint_id": "1",
            "session_id": "abc-123",
            "checkpoint_number": 3
        }
    """
    try:
        service = await get_service()

        checkpoint_id = await service.create_checkpoint(
            session_id=request.session_id,
            checkpoint_type=request.checkpoint_type,
            progress_percent=request.progress_percent,
            current_step=request.current_step,
            metadata=request.metadata,
        )

        # Get checkpoint details
        checkpoint = await service.get_checkpoint(int(checkpoint_id))

        return CreateCheckpointResponse(
            success=True,
            checkpoint_id=checkpoint_id,
            session_id=request.session_id,
            checkpoint_number=checkpoint["checkpoint_number"],
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create checkpoint: {str(e)}")


@router.get("/", response_model=CheckpointListResponse)
async def list_checkpoints(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    checkpoint_type: Optional[str] = Query(None, description="Filter by checkpoint type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> CheckpointListResponse:
    """
    List checkpoints with optional filtering.

    Example:
        GET /api/session-checkpoints?session_id=abc-123&limit=10

        Response:
        {
            "checkpoints": [...],
            "total": 25,
            "limit": 10,
            "offset": 0
        }
    """
    try:
        service = await get_service()

        checkpoints = await service.list_checkpoints(
            session_id=session_id,
            checkpoint_type=checkpoint_type,
            limit=limit,
            offset=offset,
        )

        total = await service.get_checkpoint_count(session_id=session_id)

        return CheckpointListResponse(
            checkpoints=[CheckpointSummaryResponse(**cp) for cp in checkpoints],
            total=total,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list checkpoints: {str(e)}")


@router.get("/stats", response_model=CheckpointStatsResponse)
async def get_checkpoint_stats() -> CheckpointStatsResponse:
    """
    Get checkpoint statistics.

    Example:
        GET /api/session-checkpoints/stats

        Response:
        {
            "total_checkpoints": 100,
            "by_type": {"manual": 20, "auto": 80},
            "total_size_mb": 12.5,
            "oldest_checkpoint": "2026-01-01T00:00:00Z",
            "newest_checkpoint": "2026-03-06T10:00:00Z"
        }
    """
    try:
        service = await get_service()

        total = await service.get_checkpoint_count()
        checkpoints = await service.list_checkpoints(limit=1000)

        # Calculate stats
        by_type = {}
        total_size = 0

        for cp in checkpoints:
            cp_type = cp.get("checkpoint_type", "unknown")
            by_type[cp_type] = by_type.get(cp_type, 0) + 1
            total_size += cp.get("size_bytes", 0)

        oldest = checkpoints[-1]["created_at"] if checkpoints else None
        newest = checkpoints[0]["created_at"] if checkpoints else None

        return CheckpointStatsResponse(
            total_checkpoints=total,
            by_type=by_type,
            total_size_mb=total_size / (1024 * 1024),
            oldest_checkpoint=oldest,
            newest_checkpoint=newest,
        )
    except Exception as e:
        logger.error(f"Error getting checkpoint stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{checkpoint_id}", response_model=CheckpointResponse)
async def get_checkpoint(
    checkpoint_id: int,
    include_history: bool = Query(True, description="Include conversation history"),
) -> CheckpointResponse:
    """
    Get checkpoint details.

    Example:
        GET /api/session-checkpoints/1

        Response:
        {
            "id": 1,
            "session_id": "abc-123",
            "checkpoint_number": 3,
            "checkpoint_type": "manual",
            ...
        }
    """
    try:
        service = await get_service()

        checkpoint = await service.get_checkpoint(checkpoint_id, include_history=include_history)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        return CheckpointResponse(**checkpoint)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get checkpoint: {str(e)}")


@router.get("/session/{session_id}/latest")
async def get_latest_checkpoint(
    session_id: str,
    include_history: bool = Query(True, description="Include conversation history"),
) -> CheckpointResponse:
    """
    Get the latest checkpoint for a session.

    Example:
        GET /api/session-checkpoints/session/abc-123/latest

        Response:
        {
            "id": 3,
            "session_id": "abc-123",
            "checkpoint_number": 3,
            ...
        }
    """
    try:
        service = await get_service()

        checkpoint = await service.get_latest_checkpoint(session_id, include_history=include_history)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="No checkpoints found for session")

        return CheckpointResponse(**checkpoint)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest checkpoint: {str(e)}")


@router.post("/restore", response_model=RestoreCheckpointResponse)
async def restore_from_checkpoint(request: RestoreCheckpointRequest) -> RestoreCheckpointResponse:
    """
    Restore a session from a checkpoint.

    Example:
        POST /api/session-checkpoints/restore
        {
            "checkpoint_id": 3,
            "target_session_id": "new-session-456"
        }

        Response:
        {
            "success": true,
            "session_id": "new-session-456",
            "restored_from_checkpoint": 3
        }
    """
    try:
        service = await get_service()

        session_id = await service.restore_from_checkpoint(
            checkpoint_id=request.checkpoint_id,
            target_session_id=request.target_session_id,
        )

        return RestoreCheckpointResponse(
            success=True,
            session_id=session_id,
            restored_from_checkpoint=request.checkpoint_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error restoring from checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore: {str(e)}")


@router.delete("/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: int) -> DeleteResponse:
    """
    Delete a checkpoint.

    Example:
        DELETE /api/session-checkpoints/3

        Response:
        {
            "success": true,
            "deleted_count": 1
        }
    """
    try:
        service = await get_service()

        success = await service.delete_checkpoint(checkpoint_id)

        if not success:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        return DeleteResponse(success=True, deleted_count=1)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting checkpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete checkpoint: {str(e)}")


@router.post("/cleanup-old", response_model=DeleteResponse)
async def cleanup_old_checkpoints(request: DeleteOldCheckpointsRequest) -> DeleteResponse:
    """
    Delete old checkpoints, keeping only the most recent ones.

    Example:
        POST /api/session-checkpoints/cleanup-old
        {
            "session_id": "abc-123",
            "keep_last": 5
        }

        Response:
        {
            "success": true,
            "deleted_count": 10
        }
    """
    try:
        service = await get_service()

        deleted_count = await service.delete_old_checkpoints(
            session_id=request.session_id,
            keep_last=request.keep_last,
        )

        return DeleteResponse(success=True, deleted_count=deleted_count)
    except Exception as e:
        logger.error(f"Error cleaning up old checkpoints: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {str(e)}")


@router.post("/cleanup-orphaned", response_model=DeleteResponse)
async def cleanup_orphaned_checkpoints() -> DeleteResponse:
    """
    Delete checkpoints for sessions that no longer exist.

    Example:
        POST /api/session-checkpoints/cleanup-orphaned

        Response:
        {
            "success": true,
            "deleted_count": 5
        }
    """
    try:
        service = await get_service()

        deleted_count = await service.cleanup_orphaned_checkpoints()

        return DeleteResponse(success=True, deleted_count=deleted_count)
    except Exception as e:
        logger.error(f"Error cleaning up orphaned checkpoints: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup: {str(e)}")
