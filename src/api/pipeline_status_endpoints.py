"""
Pipeline Status API Endpoints

Alpha Forge Pipeline Status Board - FastAPI router for fetching pipeline state.
Provides REST API for tracking strategy runs through the 9-stage Alpha Forge pipeline.

Endpoints:
- GET /api/pipeline/status - Get all active pipeline runs
- GET /api/pipeline/status/{strategy_id} - Get specific strategy pipeline status
- GET /api/pipeline/stages - Get pipeline stage definitions
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pipeline", tags=["pipeline-status"])


# =============================================================================
# Enums
# =============================================================================

class PipelineStage(str, Enum):
    """Alpha Forge 9-stage pipeline."""
    VIDEO_INGEST = "VIDEO_INGEST"
    RESEARCH = "RESEARCH"
    TRD = "TRD"
    DEVELOPMENT = "DEVELOPMENT"
    COMPILE = "COMPILE"
    BACKTEST = "BACKTEST"
    VALIDATION = "VALIDATION"
    EA_LIFECYCLE = "EA_LIFECYCLE"
    APPROVAL = "APPROVAL"


class StageStatus(str, Enum):
    """Stage status within pipeline."""
    WAITING = "waiting"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    """Human approval status."""
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NONE = "none"


# =============================================================================
# Pydantic Models
# =============================================================================

class PipelineStageInfo(BaseModel):
    """Information about a single pipeline stage."""
    stage: PipelineStage
    status: StageStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PipelineRun(BaseModel):
    """Complete pipeline run state."""
    strategy_id: str
    strategy_name: str
    current_stage: PipelineStage
    stage_status: StageStatus
    stages: List[PipelineStageInfo]
    approval_status: ApprovalStatus
    started_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status list."""
    runs: List[PipelineRun]
    total: int
    active_count: int


class SinglePipelineResponse(BaseModel):
    """Response for single pipeline run."""
    run: Optional[PipelineRun] = None


class PipelineStagesResponse(BaseModel):
    """Response for pipeline stage definitions."""
    stages: List[str]
    stage_order: List[str]
    human_gates: List[str]


# =============================================================================
# In-Memory Pipeline State Store
# =============================================================================

# In production, this would query Prefect API (workflows.db on Contabo)
# For local development and testing, use in-memory store
_pipeline_runs: Dict[str, Dict[str, Any]] = {}


def _initialize_sample_data():
    """Initialize sample pipeline data for development/testing."""
    if _pipeline_runs:
        return  # Already initialized

    # Sample strategy for UI development
    sample_runs = [
        {
            "strategy_id": "strat-001",
            "strategy_name": "YouTube: RSI Divergence Strategy",
            "current_stage": PipelineStage.BACKTEST,
            "stage_status": StageStatus.RUNNING,
            "stages": [
                {"stage": PipelineStage.VIDEO_INGEST, "status": StageStatus.PASSED, "started_at": "2026-03-15T10:00:00Z", "completed_at": "2026-03-15T10:30:00Z"},
                {"stage": PipelineStage.RESEARCH, "status": StageStatus.PASSED, "started_at": "2026-03-15T10:30:00Z", "completed_at": "2026-03-15T11:00:00Z"},
                {"stage": PipelineStage.TRD, "status": StageStatus.PASSED, "started_at": "2026-03-15T11:00:00Z", "completed_at": "2026-03-15T11:30:00Z"},
                {"stage": PipelineStage.DEVELOPMENT, "status": StageStatus.PASSED, "started_at": "2026-03-15T11:30:00Z", "completed_at": "2026-03-15T12:00:00Z"},
                {"stage": PipelineStage.COMPILE, "status": StageStatus.PASSED, "started_at": "2026-03-15T12:00:00Z", "completed_at": "2026-03-15T12:15:00Z"},
                {"stage": PipelineStage.BACKTEST, "status": StageStatus.RUNNING, "started_at": "2026-03-15T12:15:00Z", "completed_at": None},
                {"stage": PipelineStage.VALIDATION, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.EA_LIFECYCLE, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.APPROVAL, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
            ],
            "approval_status": ApprovalStatus.NONE,
            "started_at": "2026-03-15T10:00:00Z",
            "updated_at": "2026-03-15T12:15:00Z",
        },
        {
            "strategy_id": "strat-002",
            "strategy_name": "YouTube: Moving Average Crossover",
            "current_stage": PipelineStage.APPROVAL,
            "stage_status": StageStatus.WAITING,
            "stages": [
                {"stage": PipelineStage.VIDEO_INGEST, "status": StageStatus.PASSED, "started_at": "2026-03-14T09:00:00Z", "completed_at": "2026-03-14T09:30:00Z"},
                {"stage": PipelineStage.RESEARCH, "status": StageStatus.PASSED, "started_at": "2026-03-14T09:30:00Z", "completed_at": "2026-03-14T10:00:00Z"},
                {"stage": PipelineStage.TRD, "status": StageStatus.PASSED, "started_at": "2026-03-14T10:00:00Z", "completed_at": "2026-03-14T10:30:00Z"},
                {"stage": PipelineStage.DEVELOPMENT, "status": StageStatus.PASSED, "started_at": "2026-03-14T10:30:00Z", "completed_at": "2026-03-14T11:00:00Z"},
                {"stage": PipelineStage.COMPILE, "status": StageStatus.PASSED, "started_at": "2026-03-14T11:00:00Z", "completed_at": "2026-03-14T11:15:00Z"},
                {"stage": PipelineStage.BACKTEST, "status": StageStatus.PASSED, "started_at": "2026-03-14T11:15:00Z", "completed_at": "2026-03-14T12:00:00Z"},
                {"stage": PipelineStage.VALIDATION, "status": StageStatus.PASSED, "started_at": "2026-03-14T12:00:00Z", "completed_at": "2026-03-14T12:30:00Z"},
                {"stage": PipelineStage.EA_LIFECYCLE, "status": StageStatus.PASSED, "started_at": "2026-03-14T12:30:00Z", "completed_at": "2026-03-14T13:00:00Z"},
                {"stage": PipelineStage.APPROVAL, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
            ],
            "approval_status": ApprovalStatus.PENDING_REVIEW,
            "started_at": "2026-03-14T09:00:00Z",
            "updated_at": "2026-03-14T13:00:00Z",
        },
        {
            "strategy_id": "strat-003",
            "strategy_name": "Custom: Mean Reversion Bot",
            "current_stage": PipelineStage.DEVELOPMENT,
            "stage_status": StageStatus.RUNNING,
            "stages": [
                {"stage": PipelineStage.VIDEO_INGEST, "status": StageStatus.PASSED, "started_at": "2026-03-16T08:00:00Z", "completed_at": "2026-03-16T08:15:00Z"},
                {"stage": PipelineStage.RESEARCH, "status": StageStatus.PASSED, "started_at": "2026-03-16T08:15:00Z", "completed_at": "2026-03-16T09:00:00Z"},
                {"stage": PipelineStage.TRD, "status": StageStatus.PASSED, "started_at": "2026-03-16T09:00:00Z", "completed_at": "2026-03-16T09:30:00Z"},
                {"stage": PipelineStage.DEVELOPMENT, "status": StageStatus.RUNNING, "started_at": "2026-03-16T09:30:00Z", "completed_at": None},
                {"stage": PipelineStage.COMPILE, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.BACKTEST, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.VALIDATION, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.EA_LIFECYCLE, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.APPROVAL, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
            ],
            "approval_status": ApprovalStatus.NONE,
            "started_at": "2026-03-16T08:00:00Z",
            "updated_at": "2026-03-16T09:30:00Z",
        },
    ]

    for run in sample_runs:
        _pipeline_runs[run["strategy_id"]] = run


def _get_pipeline_run(strategy_id: str) -> Optional[Dict[str, Any]]:
    """Get pipeline run by strategy ID."""
    return _pipeline_runs.get(strategy_id)


def _get_all_runs() -> List[Dict[str, Any]]:
    """Get all pipeline runs."""
    return list(_pipeline_runs.values())


def _get_active_runs() -> List[Dict[str, Any]]:
    """Get only active (running) pipeline runs."""
    return [
        run for run in _pipeline_runs.values()
        if run.get("stage_status") == StageStatus.RUNNING.value
    ]


# Initialize sample data on module load
_initialize_sample_data()


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    active_only: bool = Query(False, description="Return only active/running runs"),
    limit: int = Query(50, description="Maximum number of runs to return"),
) -> PipelineStatusResponse:
    """
    Get all pipeline runs or active runs only.

    In production, this would query Prefect API for workflow state.
    For development, returns in-memory sample data.

    Args:
        active_only: If True, return only runs with RUNNING status
        limit: Maximum number of runs to return

    Returns:
        Pipeline runs with pagination info
    """
    logger.info(f"Getting pipeline status (active_only={active_only}, limit={limit})")

    runs = _get_active_runs() if active_only else _get_all_runs()

    # Sort by updated_at descending (most recent first)
    runs = sorted(runs, key=lambda r: r.get("updated_at", ""), reverse=True)
    runs = runs[:limit]

    # Build response
    pipeline_runs = []
    for run in runs:
        stages = []
        for stage_info in run.get("stages", []):
            stages.append(PipelineStageInfo(
                stage=stage_info["stage"],
                status=stage_info["status"],
                started_at=stage_info.get("started_at"),
                completed_at=stage_info.get("completed_at"),
                error=stage_info.get("error")
            ))

        pipeline_runs.append(PipelineRun(
            strategy_id=run["strategy_id"],
            strategy_name=run["strategy_name"],
            current_stage=run["current_stage"],
            stage_status=run["stage_status"],
            stages=stages,
            approval_status=run["approval_status"],
            started_at=run["started_at"],
            updated_at=run["updated_at"],
            metadata=run.get("metadata")
        ))

    active_count = len(_get_active_runs())

    return PipelineStatusResponse(
        runs=pipeline_runs,
        total=len(_pipeline_runs),
        active_count=active_count
    )


@router.get("/status/{strategy_id}", response_model=SinglePipelineResponse)
async def get_strategy_pipeline_status(strategy_id: str) -> SinglePipelineResponse:
    """
    Get pipeline status for a specific strategy.

    Args:
        strategy_id: The strategy identifier

    Returns:
        Pipeline run details or 404 if not found
    """
    logger.info(f"Getting pipeline status for strategy: {strategy_id}")

    run = _get_pipeline_run(strategy_id)

    if not run:
        raise HTTPException(
            status_code=404,
            detail=f"Pipeline run not found for strategy: {strategy_id}"
        )

    stages = []
    for stage_info in run.get("stages", []):
        stages.append(PipelineStageInfo(
            stage=stage_info["stage"],
            status=stage_info["status"],
            started_at=stage_info.get("started_at"),
            completed_at=stage_info.get("completed_at"),
            error=stage_info.get("error")
        ))

    return SinglePipelineResponse(
        run=PipelineRun(
            strategy_id=run["strategy_id"],
            strategy_name=run["strategy_name"],
            current_stage=run["current_stage"],
            stage_status=run["stage_status"],
            stages=stages,
            approval_status=run["approval_status"],
            started_at=run["started_at"],
            updated_at=run["updated_at"],
            metadata=run.get("metadata")
        )
    )


@router.get("/stages", response_model=PipelineStagesResponse)
async def get_pipeline_stages() -> PipelineStagesResponse:
    """
    Get pipeline stage definitions.

    Returns:
        List of stages and their order, including human approval gates
    """
    return PipelineStagesResponse(
        stages=[stage.value for stage in PipelineStage],
        stage_order=[
            "VIDEO_INGEST",
            "RESEARCH",
            "TRD",
            "DEVELOPMENT",
            "COMPILE",
            "BACKTEST",
            "VALIDATION",
            "EA_LIFECYCLE",
            "APPROVAL"
        ],
        human_gates=["APPROVAL"]
    )


@router.get("/pending-approvals")
async def get_pending_approvals() -> Dict[str, Any]:
    """
    Get strategies awaiting human approval (PENDING_REVIEW status).

    Used for ApprovalGateBadge integration.

    Returns:
        List of strategies requiring approval
    """
    pending = [
        run for run in _pipeline_runs.values()
        if run.get("approval_status") == ApprovalStatus.PENDING_REVIEW.value
    ]

    return {
        "pending_count": len(pending),
        "strategies": [
            {
                "strategy_id": run["strategy_id"],
                "strategy_name": run["strategy_name"],
                "current_stage": run["current_stage"].value,
                "updated_at": run["updated_at"]
            }
            for run in pending
        ]
    }


@router.post("/run")
async def create_pipeline_run(
    strategy_id: str,
    strategy_name: str,
) -> Dict[str, Any]:
    """
    Create a new pipeline run (for testing/development).

    In production, this would be triggered by Prefect when a new workflow starts.

    Args:
        strategy_id: Unique strategy identifier
        strategy_name: Display name for the strategy

    Returns:
        Created pipeline run details
    """
    if strategy_id in _pipeline_runs:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline run already exists for strategy: {strategy_id}"
        )

    # Create new run
    now = datetime.utcnow().isoformat() + "Z"
    new_run = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "current_stage": PipelineStage.VIDEO_INGEST,
        "stage_status": StageStatus.RUNNING,
        "stages": [
            {
                "stage": stage,
                "status": StageStatus.WAITING if stage != PipelineStage.VIDEO_INGEST else StageStatus.RUNNING,
                "started_at": now if stage == PipelineStage.VIDEO_INGEST else None,
                "completed_at": None
            }
            for stage in PipelineStage
        ],
        "approval_status": ApprovalStatus.NONE,
        "started_at": now,
        "updated_at": now,
    }

    _pipeline_runs[strategy_id] = new_run

    return {
        "status": "created",
        "strategy_id": strategy_id,
        "strategy_name": strategy_name
    }


# =============================================================================
# Router Registration
# =============================================================================

def get_router() -> APIRouter:
    """Return the API router for registration."""
    return router