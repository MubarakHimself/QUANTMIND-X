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
from datetime import datetime, timezone
from enum import Enum
import logging
import json

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
# Canonical WF1 Read Model
# =============================================================================

PIPELINE_STAGE_ORDER: List[PipelineStage] = [
    PipelineStage.VIDEO_INGEST,
    PipelineStage.RESEARCH,
    PipelineStage.TRD,
    PipelineStage.DEVELOPMENT,
    PipelineStage.COMPILE,
    PipelineStage.BACKTEST,
    PipelineStage.VALIDATION,
    PipelineStage.EA_LIFECYCLE,
    PipelineStage.APPROVAL,
]

_WORKFLOW_STAGE_TO_PIPELINE_STAGE: Dict[str, PipelineStage] = {
    "video ingest": PipelineStage.VIDEO_INGEST,
    "research": PipelineStage.RESEARCH,
    "development": PipelineStage.DEVELOPMENT,
    "backtesting": PipelineStage.BACKTEST,
    "paper trading": PipelineStage.EA_LIFECYCLE,
    "paper trading monitor": PipelineStage.EA_LIFECYCLE,
    "sit gate": PipelineStage.VALIDATION,
    "human approval gate": PipelineStage.APPROVAL,
    "completed": PipelineStage.APPROVAL,
}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _stage_status_for_workflow_state(
    workflow_state: str,
    stage_index: int,
    current_index: int,
) -> StageStatus:
    state = (workflow_state or "").upper()
    if stage_index < current_index:
        return StageStatus.PASSED
    if stage_index > current_index:
        return StageStatus.WAITING
    if state == "RUNNING":
        return StageStatus.RUNNING
    if state in {"DONE"}:
        return StageStatus.PASSED
    if state in {"CANCELLED", "EXPIRED_REVIEW"}:
        return StageStatus.FAILED
    return StageStatus.WAITING


def _approval_status_for_workflow_state(workflow_state: str) -> ApprovalStatus:
    state = (workflow_state or "").upper()
    if state == "PENDING_REVIEW":
        return ApprovalStatus.PENDING_REVIEW
    if state == "DONE":
        return ApprovalStatus.APPROVED
    if state in {"CANCELLED", "EXPIRED_REVIEW"}:
        return ApprovalStatus.REJECTED
    return ApprovalStatus.NONE


def _pipeline_stage_for_workflow(current_stage: Optional[str]) -> PipelineStage:
    key = (current_stage or "").strip().lower()
    return _WORKFLOW_STAGE_TO_PIPELINE_STAGE.get(key, PipelineStage.VIDEO_INGEST)


def _strategy_name(strategy_id: str) -> str:
    from src.api.wf1_artifacts import find_strategy_root

    strategy_root = find_strategy_root(strategy_id)
    if strategy_root:
        meta_path = strategy_root / ".meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = (meta.get("name") or "").strip()
                if name:
                    return name
            except Exception:
                pass
    return strategy_id


def _workflow_to_pipeline_run(workflow: Dict[str, Any]) -> Dict[str, Any]:
    strategy_id = workflow.get("strategy_id") or workflow.get("id")
    workflow_state = workflow.get("state") or "PENDING"
    current_pipeline_stage = _pipeline_stage_for_workflow(workflow.get("current_stage"))
    current_index = PIPELINE_STAGE_ORDER.index(current_pipeline_stage)
    started_at = workflow.get("started_at") or workflow.get("updated_at") or datetime.now(timezone.utc).isoformat()
    updated_at = workflow.get("updated_at") or started_at

    stages: List[Dict[str, Any]] = []
    for idx, stage in enumerate(PIPELINE_STAGE_ORDER):
        status = _stage_status_for_workflow_state(workflow_state, idx, current_index)
        stage_started_at = started_at if idx <= current_index else None
        stage_completed_at = updated_at if status == StageStatus.PASSED else None
        stages.append(
            {
                "stage": stage,
                "status": status,
                "started_at": stage_started_at,
                "completed_at": stage_completed_at,
                "error": workflow.get("blocking_error") if idx == current_index else None,
            }
        )

    return {
        "strategy_id": strategy_id,
        "strategy_name": _strategy_name(strategy_id),
        "current_stage": current_pipeline_stage,
        "stage_status": _stage_status_for_workflow_state(workflow_state, current_index, current_index),
        "stages": stages,
        "approval_status": _approval_status_for_workflow_state(workflow_state),
        "started_at": started_at,
        "updated_at": updated_at,
        "metadata": {
            "workflow_id": workflow.get("id"),
            "workflow_state": workflow_state,
            "current_stage_label": workflow.get("current_stage"),
            "department": workflow.get("department"),
            "latest_artifact": workflow.get("latest_artifact"),
            "blocking_error": workflow.get("blocking_error"),
        },
    }


async def _load_alpha_forge_runs() -> List[Dict[str, Any]]:
    from src.api.prefect_workflow_endpoints import _merged_workflows

    workflows = list((await _merged_workflows()).values())
    workflows = [workflow for workflow in workflows if workflow.get("name") == "WF1 Creation"]
    workflows.sort(key=lambda workflow: workflow.get("updated_at") or workflow.get("started_at") or "", reverse=True)
    return [_workflow_to_pipeline_run(workflow) for workflow in workflows]


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

    Reads canonical WF1 workflow state from the backend workflow read model.

    Args:
        active_only: If True, return only runs with RUNNING status
        limit: Maximum number of runs to return

    Returns:
        Pipeline runs with pagination info
    """
    logger.info(f"Getting pipeline status (active_only={active_only}, limit={limit})")

    runs = await _load_alpha_forge_runs()
    if active_only:
        runs = [run for run in runs if run.get("stage_status") == StageStatus.RUNNING]

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

    active_count = len([run for run in runs if run.get("stage_status") == StageStatus.RUNNING])

    return PipelineStatusResponse(
        runs=pipeline_runs,
        total=len(runs),
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

    runs = await _load_alpha_forge_runs()
    run = next((run for run in runs if run.get("strategy_id") == strategy_id), None)

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
        run for run in await _load_alpha_forge_runs()
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
    """Pipeline runs are created by the workflow coordinator, not ad hoc here."""
    raise HTTPException(
        status_code=501,
        detail="Create pipeline runs via the canonical workflow coordinator",
    )


# =============================================================================
# Router Registration
# =============================================================================

def get_router() -> APIRouter:
    """Return the API router for registration."""
    return router
