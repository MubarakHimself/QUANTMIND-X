"""
Video-to-EA Workflow API Endpoints

Provides REST API endpoints for the automated video-to-EA workflow:
- POST /api/video-to-ea/start - Start a new video-to-EA workflow
- GET /api/video-to-ea/{workflow_id} - Get workflow status
- GET /api/video-to-ea - List all workflows
- DELETE /api/video-to-ea/{workflow_id} - Cancel a workflow

Integrates with:
- src.router.video_to_ea_workflow - Workflow orchestration
- src.agents.departments.department_mail - Notifications
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/video-to-ea", tags=["video-to-ea"])

# Import the orchestrator
from src.router.video_to_ea_workflow import (
    get_video_to_ea_orchestrator,
    VideoToEAStage,
    VideoToEAWorkflowOrchestrator,
)

# Initialize orchestrator
_orchestrator: Optional[VideoToEAWorkflowOrchestrator] = None


def _get_orchestrator() -> VideoToEAWorkflowOrchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = get_video_to_ea_orchestrator()
    return _orchestrator


# ============================================================================
# Request/Response Models
# ============================================================================


class WorkflowStatus(str, Enum):
    """Status of video-to-EA workflow."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoToEAStartRequest(BaseModel):
    """Request model for starting a video-to-EA workflow."""
    video_url: str = Field(..., description="YouTube URL to process")
    strategy_name: Optional[str] = Field(None, description="Name for the strategy (derived from video if not provided)")
    analysis_depth: str = Field("standard", description="Analysis depth: quick, standard, or deep")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")


class VideoToEAResponse(BaseModel):
    """Response model for video-to-EA operations."""
    success: bool
    workflow_id: str
    video_url: str
    strategy_name: str
    current_stage: str
    progress_percent: float
    status: str
    ea_workflow_id: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class WorkflowListResponse(BaseModel):
    """Response model for listing workflows."""
    workflows: List[Dict[str, Any]]
    total: int


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/start", response_model=VideoToEAResponse)
async def start_video_to_ea_workflow(
    request: VideoToEAStartRequest,
):
    """
    Start a new video-to-EA workflow.

    This creates an automated pipeline that:
    1. Downloads and processes the YouTube video
    2. Analyzes the video for trading strategy elements
    3. Generates a Trading Requirements Document (TRD)
    4. Triggers EA creation workflow
    5. Sends notifications via department mail at each stage
    """
    orchestrator = _get_orchestrator()

    try:
        result = await orchestrator.start_workflow(
            video_url=request.video_url,
            strategy_name=request.strategy_name,
            analysis_depth=request.analysis_depth,
            metadata=request.metadata,
        )

        return VideoToEAResponse(
            success=result.success,
            workflow_id=result.workflow_id,
            video_url=result.video_url,
            strategy_name=result.strategy_name,
            current_stage=result.current_stage.value,
            progress_percent=result.progress_percent,
            status="running" if result.success else "failed",
            error=result.error,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.error(f"Failed to start video-to-EA workflow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}", response_model=VideoToEAResponse)
async def get_video_to_ea_workflow(workflow_id: str):
    """
    Get status of a video-to-EA workflow.
    """
    orchestrator = _get_orchestrator()
    workflow_status = orchestrator.get_workflow_status(workflow_id)

    if not workflow_status:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    return VideoToEAResponse(
        success=workflow_status.get("status") != "failed",
        workflow_id=workflow_status.get("workflow_id", ""),
        video_url=workflow_status.get("video_url", ""),
        strategy_name=workflow_status.get("strategy_name", ""),
        current_stage=workflow_status.get("current_stage", ""),
        progress_percent=workflow_status.get("progress_percent", 0.0),
        status=workflow_status.get("status", ""),
        ea_workflow_id=workflow_status.get("ea_workflow_id"),
        error=workflow_status.get("error"),
        created_at=workflow_status.get("created_at"),
        started_at=workflow_status.get("started_at"),
        completed_at=workflow_status.get("completed_at"),
    )


@router.get("/", response_model=WorkflowListResponse)
async def list_video_to_ea_workflows(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of workflows to return"),
):
    """
    List all video-to-EA workflows.
    """
    orchestrator = _get_orchestrator()
    workflows = orchestrator.get_all_workflows()

    # Filter by status if provided
    if status:
        workflows = [w for w in workflows if w.get("status") == status.value]

    # Limit results
    workflows = workflows[:limit]

    return WorkflowListResponse(
        workflows=workflows,
        total=len(workflows),
    )


@router.delete("/{workflow_id}")
async def cancel_video_to_ea_workflow(workflow_id: str):
    """
    Cancel a running video-to-EA workflow.

    Note: This cancels the parent workflow only. Any already-triggered
    EA creation workflow will continue independently.
    """
    orchestrator = _get_orchestrator()
    workflow_status = orchestrator.get_workflow_status(workflow_id)

    if not workflow_status:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow_status.get("status") not in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel workflow with status: {workflow_status.get('status')}"
        )

    # Update workflow status
    workflow_status["status"] = "cancelled"
    workflow_status["completed_at"] = datetime.now(timezone.utc).isoformat()

    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": "cancelled",
        "message": "Workflow cancelled successfully"
    }


@router.get("/{workflow_id}/wait")
async def wait_for_video_to_ea_workflow(
    workflow_id: str,
    timeout: int = Query(300, ge=1, le=3600, description="Maximum wait time in seconds"),
):
    """
    Wait for a video-to-EA workflow to complete.
    """
    import asyncio

    orchestrator = _get_orchestrator()
    workflow_status = orchestrator.get_workflow_status(workflow_id)

    if not workflow_status:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    # Check if already completed or failed
    if workflow_status.get("status") in ["completed", "failed", "cancelled"]:
        return workflow_status

    # Poll for completion
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        await asyncio.sleep(2)  # Poll every 2 seconds
        workflow_status = orchestrator.get_workflow_status(workflow_id)

        if workflow_status.get("status") in ["completed", "failed", "cancelled"]:
            return workflow_status

    # Timeout
    raise HTTPException(
        status_code=408,
        detail=f"Workflow did not complete within {timeout} seconds"
    )
