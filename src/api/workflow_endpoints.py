"""
Workflow API Endpoints

Provides REST API endpoints for workflow management and execution.

The canonical workflow authority is the department workflow coordinator
(`wf1_creation`–`wf4_weekend_update`). The legacy `/api/workflows/*`
compatibility endpoints below now serialize coordinator state into the older
workflowStore.ts shape instead of dispatching to the obsolete private
orchestrator stack.

Endpoints still match the expected shapes from workflowStore.ts:
- workflow_id, workflow_type, status, progress_percent, steps, final_result, timestamps
"""

import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


# ============================================================================
# Enums and Models
# ============================================================================

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    step_id: str
    name: str
    description: str
    agent_type: str
    status: StepStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    error: Optional[str] = None
    retries: int = 0
    duration_seconds: Optional[float] = None


class Workflow(BaseModel):
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    current_step_index: int = 0
    progress_percent: float = 0.0
    steps: List[WorkflowStep] = []
    input_data: Dict[str, Any] = {}
    intermediate_results: Dict[str, Any] = {}
    final_result: Dict[str, Any] = {}
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


class WorkflowSummary(BaseModel):
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    progress_percent: float
    created_at: str
    error: Optional[str] = None


class VideoIngestToEARequest(BaseModel):
    video_ingest_content: str
    metadata: Optional[Dict[str, Any]] = None


class TRDToEARequest(BaseModel):
    trd_content: str
    trd_file: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowListResponse(BaseModel):
    workflows: List[Workflow]


# ============================================================================
# Coordinator Integration
# ============================================================================

def _get_coordinator():
    """Get the department workflow coordinator instance."""
    from src.agents.departments.workflow_coordinator import get_workflow_coordinator

    return get_workflow_coordinator()


def _convert_coordinator_status(status: str) -> WorkflowStatus:
    """Convert coordinator/canonical status to the legacy workflow API enum."""
    status_map = {
        "pending": WorkflowStatus.PENDING,
        "running": WorkflowStatus.RUNNING,
        "waiting": WorkflowStatus.PAUSED,
        "pending_review": WorkflowStatus.PAUSED,
        "expired_review": WorkflowStatus.PAUSED,
        "paused": WorkflowStatus.PAUSED,
        "completed": WorkflowStatus.COMPLETED,
        "failed": WorkflowStatus.FAILED,
        "cancelled": WorkflowStatus.CANCELLED,
    }
    return status_map.get(status, WorkflowStatus.PENDING)


def _step_status_from_task(status: str) -> StepStatus:
    status_map = {
        "pending": StepStatus.PENDING,
        "running": StepStatus.RUNNING,
        "waiting": StepStatus.RUNNING,
        "completed": StepStatus.COMPLETED,
        "failed": StepStatus.FAILED,
        "cancelled": StepStatus.SKIPPED,
    }
    return status_map.get(status, StepStatus.PENDING)


def _task_to_step(task: Dict[str, Any], index: int, workflow_id: str) -> WorkflowStep:
    return WorkflowStep(
        step_id=f"{workflow_id}_step_{index}",
        name=task.get("stage", f"step_{index}"),
        description=f"Execute {task.get('stage', f'step_{index}')} stage",
        agent_type=task.get("to_dept", "system"),
        status=_step_status_from_task(task.get("status", "pending")),
        started_at=task.get("created_at"),
        completed_at=task.get("completed_at"),
        input_data=task.get("payload") or {},
        output_data=task.get("result") or {},
        error=task.get("error"),
        retries=0,
    )


def _convert_coordinator_workflow(workflow_data: Dict[str, Any]) -> Workflow:
    """Convert coordinator workflow format to the legacy workflow API format."""
    steps = []
    tasks = workflow_data.get("tasks", [])
    for index, task in enumerate(tasks, start=1):
        steps.append(_task_to_step(task, index, workflow_data["workflow_id"]))

    completed_steps = sum(1 for step in steps if step.status == StepStatus.COMPLETED)
    workflow_type = workflow_data.get("workflow_type", "wf1_creation")
    legacy_type = workflow_type
    if workflow_type == "wf1_creation":
        entry_stage = (workflow_data.get("metadata") or {}).get("entry_stage")
        if entry_stage == "development":
            legacy_type = "trd_to_ea"
        else:
            legacy_type = "video_ingest_to_ea"

    created_at = workflow_data.get("created_at")
    updated_at = workflow_data.get("updated_at")
    duration_seconds = None
    if created_at and updated_at:
        try:
            duration_seconds = (
                datetime.fromisoformat(updated_at) - datetime.fromisoformat(created_at)
            ).total_seconds()
        except ValueError:
            duration_seconds = None

    return Workflow(
        workflow_id=workflow_data["workflow_id"],
        workflow_type=legacy_type,
        status=_convert_coordinator_status(workflow_data.get("status", "pending")),
        current_step_index=completed_steps,
        progress_percent=float(workflow_data.get("progress_percent", 0.0)),
        steps=steps,
        input_data=(workflow_data.get("metadata") or {}).get("initial_payload", {}),
        intermediate_results=workflow_data.get("metadata", {}),
        final_result=(workflow_data.get("metadata") or {}).get("latest_results", {}),
        created_at=created_at or "",
        started_at=steps[0].started_at if steps else created_at,
        completed_at=updated_at if workflow_data.get("status") in {"completed", "failed", "cancelled"} else None,
        duration_seconds=duration_seconds,
        error=workflow_data.get("error"),
    )


def _get_agent_type_for_step(step_name: str) -> str:
    """Get the agent type for a workflow step."""
    agent_map = {
        "video_ingest_processing": "system",
        "analyst": "analyst",
        "trd_generation": "analyst",
        "quantcode": "quantcode",
        "compilation": "quantcode",
        "backtest": "quantcode",
        "validation": "analyst"
    }
    return agent_map.get(step_name, "system")


def _serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model across v1/v2 without pulling in HTTP self-calls."""
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


def _get_workflow_gate_payload(workflow_id: str, pending_only: bool = False) -> Dict[str, Any]:
    """
    Read workflow approval gates directly from the local database.

    This avoids split-host breakage from self-calling `localhost:8000`.
    """
    from src.api.approval_gate import (
        ApprovalGateModel,
        ApprovalStatus,
        _model_to_response,
    )
    from src.database.engine import get_session

    session = get_session()
    try:
        query = session.query(ApprovalGateModel).filter(ApprovalGateModel.workflow_id == workflow_id)
        if pending_only:
            query = query.filter(
                ApprovalGateModel.status.in_([
                    ApprovalStatus.PENDING,
                    ApprovalStatus.PENDING_REVIEW,
                ])
            )

        gates = query.order_by(ApprovalGateModel.created_at.asc()).all()
        payload = [_serialize_model(_model_to_response(gate)) for gate in gates]
        return {"gates": payload, "total": len(payload)}
    finally:
        session.close()


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/video-ingest-to-ea", response_model=Dict[str, Any])
async def start_video_ingest_to_ea_workflow(
    request: VideoIngestToEARequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new VideoIngest to EA workflow.

    This creates a workflow that:
    1. Parses the VideoIngest
    2. Generates a TRD (using Analyst agent)
    3. Generates MQL5 code (using QuantCode agent)
    4. Compiles the EA (using MT5 Compiler MCP)
    5. Runs a backtest (using Backtest MCP)
    6. Analyzes and validates results

    Compatibility entrypoint for legacy clients. This now starts the canonical
    WF1 coordinator workflow instead of the retired private video-to-EA stack.
    """
    from src.agents.departments.workflow_models import TaskPriority, WorkflowType

    coordinator = _get_coordinator()
    payload = {
        "video_ingest_content": request.video_ingest_content,
        "source_type": "video_ingest",
    }
    workflow_id = coordinator.start_workflow(
        workflow_type=WorkflowType.WF1_CREATION,
        initial_payload=payload,
        priority=TaskPriority.MEDIUM,
        metadata=request.metadata or {},
    )

    return {
        "workflow_id": workflow_id,
        "workflow_type": "video_ingest_to_ea",
        "status": "pending",
        "message": "Canonical WF1 workflow started successfully",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/trd-to-ea", response_model=Dict[str, Any])
async def start_trd_to_ea_workflow(
    request: TRDToEARequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new TRD to EA workflow.

    This creates a workflow that skips VideoIngest processing and Analyst stages,
    starting directly from QuantCode:
    1. Generates MQL5 code from TRD (using QuantCode agent)
    2. Compiles the EA (using MT5 Compiler MCP)
    3. Runs a backtest (using Backtest MCP)
    4. Analyzes and validates results

    Compatibility entrypoint for legacy clients. This now starts the canonical
    WF1 coordinator workflow at the development stage using an existing TRD
    artifact instead of the retired private orchestrator.
    """
    from src.agents.departments.workflow_models import (
        TaskPriority,
        WorkflowStage,
        WorkflowType,
    )

    coordinator = _get_coordinator()
    payload = {
        "trd_content": request.trd_content,
        "trd_file": request.trd_file,
        "source_type": "trd",
    }
    workflow_id = coordinator.start_workflow(
        workflow_type=WorkflowType.WF1_CREATION,
        initial_payload=payload,
        priority=TaskPriority.MEDIUM,
        metadata=request.metadata or {},
        initial_stage=WorkflowStage.DEVELOPMENT,
    )

    return {
        "workflow_id": workflow_id,
        "workflow_type": "trd_to_ea",
        "status": "pending",
        "message": "Canonical WF1 workflow started from TRD successfully",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: str):
    """
    Get workflow status and details.
    
    Retrieves the current state from the canonical coordinator and projects it
    into the legacy workflowStore shape.
    """
    coordinator = _get_coordinator()
    workflow_data = coordinator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return _convert_coordinator_workflow(workflow_data)


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status")
):
    """
    List all workflows.
    
    Retrieves canonical workflows from the department workflow coordinator and
    projects them into the legacy workflowStore shape.
    """
    coordinator = _get_coordinator()
    all_workflows = coordinator.get_all_workflows()
    
    workflows = []
    for wf_data in all_workflows:
        workflow = _convert_coordinator_workflow(wf_data)
        if status is None or workflow.status == status:
            workflows.append(workflow)
    
    return WorkflowListResponse(workflows=workflows)


@router.post("/{workflow_id}/cancel", response_model=Dict[str, Any])
async def cancel_workflow(workflow_id: str):
    """
    Cancel a running workflow.
    
    Delegates cancellation to the canonical workflow coordinator.
    """
    coordinator = _get_coordinator()
    
    # Check if workflow exists
    workflow_data = coordinator.get_workflow_status(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    success = coordinator.cancel_workflow(workflow_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel workflow with status {workflow_data.get('status')}"
        )
    
    logger.info(f"Cancelled workflow: {workflow_id}")
    
    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": "cancelled",
    }


@router.post("/{workflow_id}/pause", response_model=Dict[str, Any])
async def pause_workflow(workflow_id: str):
    """
    Pause a running workflow.

    The workflow will stop at the current step and can be resumed later.
    """
    coordinator = _get_coordinator()
    workflow_data = coordinator.get_workflow_status(workflow_id)

    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow_data.get("status") != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Can only pause running workflows, current status: {workflow_data.get('status')}"
        )

    raise HTTPException(
        status_code=501,
        detail="Manual pause is not wired for canonical workflows yet",
    )


@router.post("/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(workflow_id: str):
    """
    Resume a paused workflow.

    The workflow will continue from where it was paused.
    """
    coordinator = _get_coordinator()
    workflow_data = coordinator.get_workflow_status(workflow_id)

    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow_data.get("status") != "waiting":
        raise HTTPException(
            status_code=400,
            detail=f"Can only resume waiting workflows, current status: {workflow_data.get('status')}"
        )

    raise HTTPException(
        status_code=501,
        detail="Manual resume is not wired for canonical workflows yet; use approval resolution or canonical endpoints",
    )


@router.get("/{workflow_id}/wait", response_model=Workflow)
async def wait_for_workflow(
    workflow_id: str,
    timeout: int = Query(300, description="Maximum wait time in seconds")
):
    """
    Wait for a workflow to complete.
    
    Returns the final workflow state when completed, failed, or cancelled.
    """
    coordinator = _get_coordinator()
    deadline = datetime.now(timezone.utc).timestamp() + timeout
    while datetime.now(timezone.utc).timestamp() < deadline:
        workflow_data = coordinator.get_workflow_status(workflow_id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail=str(f"Workflow {workflow_id} not found"))
        if workflow_data.get("status") in {"completed", "failed", "cancelled"}:
            return _convert_coordinator_workflow(workflow_data)
        await asyncio.sleep(1)

    raise HTTPException(
        status_code=408,
        detail=f"Workflow did not complete within {timeout} seconds"
    )


@router.get("/{workflow_id}/results", response_model=Dict[str, Any])
async def get_workflow_results(workflow_id: str):
    """
    Get the results of a completed workflow.
    
    Returns backtest results, validation status, and generated files.
    """
    coordinator = _get_coordinator()
    workflow_data = coordinator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if workflow_data.get("status") not in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Workflow not yet complete. Current status: {workflow_data.get('status')}"
        )
    
    latest_results = (workflow_data.get("metadata") or {}).get("latest_results", {})
    
    return {
        "workflow_id": workflow_id,
        "status": workflow_data.get("status"),
        "backtest_results": latest_results.get("backtest_results", {}),
        "validation": latest_results.get("validation", {}),
        "output_files": latest_results.get("output_files", {}),
        "metadata": workflow_data.get("metadata", {}),
        "error": workflow_data.get("error")
    }


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow and its associated files.
    
    Only allowed for completed, failed, or cancelled workflows.
    """
    coordinator = _get_coordinator()
    workflow_data = coordinator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if workflow_data.get("status") in ["pending", "running", "waiting"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running workflow. Cancel it first."
        )

    if workflow_id in coordinator._workflows:
        del coordinator._workflows[workflow_id]
    
    return {
        "success": True,
        "workflow_id": workflow_id,
        "message": "Workflow deleted successfully"
    }


@router.get("/{workflow_id}/approvals", response_model=Dict[str, Any])
async def get_workflow_approvals(workflow_id: str):
    """
    Get all approval gates for a specific workflow.

    This endpoint proxies to the approval-gates API to get workflow approvals.
    """
    try:
        return _get_workflow_gate_payload(workflow_id)
    except Exception as e:
        logger.error("Failed to fetch approvals for workflow %s: %s", workflow_id, e)
        return {"gates": [], "total": 0, "error": str(e)}


class WorkflowApprovalRequest(BaseModel):
    approved: bool
    notes: Optional[str] = None


@router.post("/{workflow_id}/approve", response_model=Dict[str, Any])
async def approve_workflow(
    workflow_id: str,
    request: WorkflowApprovalRequest,
):
    """
    Approve or reject a workflow pending review.

    This endpoint:
    1. Updates the workflow record in the database with the approval decision
    2. If approved, resumes the suspended Prefect flow via resume_flow_run()

    The suspended flow will only resume if approval was granted.
    """
    from flows.database import get_workflow_database

    db = get_workflow_database()
    workflow = db.get_workflow_run(workflow_id)

    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow.get("status") != "pending_review":
        raise HTTPException(
            status_code=400,
            detail=f"Workflow is not pending review. Current status: {workflow.get('status')}"
        )

    new_status = "approved" if request.approved else "rejected"

    # Parse existing output_data to preserve prefect_flow_run_id
    existing_output = {}
    if workflow.get("output_data"):
        try:
            existing_output = json.loads(workflow["output_data"])
        except json.JSONDecodeError:
            pass

    # Merge new approval data with existing output_data
    approval_data = {
        **existing_output,
        "approved": request.approved,
        "notes": request.notes,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }

    db.update_workflow_status(
        workflow_id=workflow_id,
        status=new_status,
        output_data=json.dumps(approval_data),
    )

    logger.info(f"Workflow {workflow_id} {new_status} by user")

    # If approved, resume the suspended Prefect flow
    if request.approved:
        prefect_flow_run_id = existing_output.get("prefect_flow_run_id")
        if prefect_flow_run_id:
            try:
                from prefect import resume_flow_run
                resume_flow_run(flow_run_id=prefect_flow_run_id)
                logger.info(f"Resumed Prefect flow run {prefect_flow_run_id} for workflow {workflow_id}")
            except Exception as e:
                logger.error(f"Failed to resume Prefect flow {prefect_flow_run_id}: {e}")
                # Don't fail the request - the flow might not be suspended or already resumed
        else:
            logger.warning(f"No prefect_flow_run_id found for workflow {workflow_id} - flow may not be suspended")

    return {
        "status": new_status,
        "workflow_id": workflow_id,
        "notes": request.notes,
    }


@router.get("/{workflow_id}/approvals/pending", response_model=Dict[str, Any])
async def get_workflow_pending_approval(workflow_id: str):
    """
    Check if a workflow has a pending approval gate.

    Returns the pending gate if one exists.
    """
    try:
        data = _get_workflow_gate_payload(workflow_id, pending_only=True)
        gates = data.get("gates", [])
        return {"has_pending": len(gates) > 0, "gates": gates, "total": len(gates)}
    except Exception as e:
        logger.error("Failed to fetch pending approvals for workflow %s: %s", workflow_id, e)
        return {"has_pending": False, "gates": [], "total": 0, "error": str(e)}


# ============================================================================
# Canonical Workflow Endpoints — WF1-WF4 (architecture §20)
# ============================================================================

class CanonicalWorkflowRequest(BaseModel):
    """Request body for starting a canonical workflow (WF1-WF4)."""
    workflow_type: str = Field(
        ..., description="Workflow type: wf1_creation, wf2_enhancement, wf3_performance_intel, wf4_weekend_update"
    )
    strategy_id: Optional[str] = Field(None, description="Strategy ID (required for WF2)")
    priority: str = Field("medium", description="Task priority: high, medium, low")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Initial workflow payload")
    metadata: Optional[Dict[str, Any]] = None


def _get_dept_coordinator():
    """Get the department workflow coordinator instance."""
    from src.agents.departments.workflow_coordinator import get_workflow_coordinator
    return get_workflow_coordinator()


def _map_workflow_type(wf_type_str: str):
    """Map string workflow type to WorkflowType enum."""
    from src.agents.departments.workflow_models import WorkflowType
    type_map = {
        "wf1_creation": WorkflowType.WF1_CREATION,
        "wf2_enhancement": WorkflowType.WF2_ENHANCEMENT,
        "wf3_performance_intel": WorkflowType.WF3_PERFORMANCE_INTEL,
        "wf4_weekend_update": WorkflowType.WF4_WEEKEND_UPDATE,
    }
    wf_type = type_map.get(wf_type_str.lower())
    if not wf_type:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown workflow type '{wf_type_str}'. "
                   f"Valid types: {', '.join(type_map.keys())}"
        )
    return wf_type


@router.post("/canonical/start", response_model=Dict[str, Any])
async def start_canonical_workflow(
    request: CanonicalWorkflowRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a canonical workflow (WF1-WF4).

    This endpoint triggers the department workflow coordinator which:
    1. Creates the workflow with proper EA lifecycle state
    2. Dispatches the first task via department mail
    3. Persists state to Prefect database for durability
    4. Records the decision in graph memory
    5. Updates Kanban board status

    Workflow Types:
    - wf1_creation: AlphaForge Creation Pipeline (YouTube → EA in library)
    - wf2_enhancement: AlphaForge Enhancement Loop (EA library → improved EA → live)
    - wf3_performance_intel: Daily Performance Intelligence (Dead Zone 16:15-18:00 GMT)
    - wf4_weekend_update: Weekend Update Cycle (Friday→Monday)
    """
    from src.agents.departments.workflow_models import TaskPriority

    coordinator = _get_dept_coordinator()
    wf_type = _map_workflow_type(request.workflow_type)

    priority_map = {
        "high": TaskPriority.HIGH,
        "medium": TaskPriority.MEDIUM,
        "low": TaskPriority.LOW,
    }
    priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)

    workflow_id = coordinator.start_workflow(
        workflow_type=wf_type,
        initial_payload=request.payload,
        strategy_id=request.strategy_id,
        priority=priority,
        metadata=request.metadata,
    )

    # Optionally trigger Prefect flow in background for durable execution
    async def _trigger_prefect():
        try:
            await coordinator._trigger_prefect_flow(
                wf_type, workflow_id, request.payload,
            )
        except Exception as e:
            logger.warning(f"Background Prefect trigger failed: {e}")

    background_tasks.add_task(_trigger_prefect)

    logger.info(f"Started canonical workflow: {request.workflow_type} → {workflow_id}")

    return {
        "workflow_id": workflow_id,
        "workflow_type": request.workflow_type,
        "status": "pending",
        "message": f"Canonical workflow {request.workflow_type} started",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/canonical/{workflow_id}/advance", response_model=Dict[str, Any])
async def advance_canonical_workflow(workflow_id: str):
    """
    Advance a canonical workflow to its next stage.

    Uses the routing tables defined in workflow_models.py to determine
    the next stage and target department.
    """
    coordinator = _get_dept_coordinator()
    result = coordinator.advance_workflow(workflow_id)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/canonical/{workflow_id}/respond", response_model=Dict[str, Any])
async def handle_department_response(
    workflow_id: str,
    from_dept: str = Query(..., description="Department that completed the task"),
    result: Dict[str, Any] = {},
):
    """
    Handle a department's response to a workflow task.

    Called when a department head completes its assigned stage. Updates
    EA lifecycle state, persists to Prefect DB, and records in memory.
    """
    coordinator = _get_dept_coordinator()
    response = coordinator.handle_department_response(
        workflow_id=workflow_id,
        from_dept=from_dept,
        result=result,
    )

    if "error" in response:
        raise HTTPException(status_code=400, detail=response["error"])

    return response


@router.get("/canonical/active", response_model=Dict[str, Any])
async def list_active_canonical_workflows():
    """List all active canonical workflows (WF1-WF4)."""
    coordinator = _get_dept_coordinator()
    active = coordinator.get_active_workflows()
    return {"workflows": active, "count": len(active)}


@router.get("/canonical/by-type/{workflow_type}", response_model=Dict[str, Any])
async def list_workflows_by_type(workflow_type: str):
    """List all canonical workflows of a specific type."""
    wf_type = _map_workflow_type(workflow_type)
    coordinator = _get_dept_coordinator()
    workflows = coordinator.get_workflows_by_type(wf_type)
    return {"workflows": workflows, "count": len(workflows)}


@router.get("/canonical/{workflow_id}/status", response_model=Dict[str, Any])
async def get_canonical_workflow_status(workflow_id: str):
    """Get detailed status of a canonical workflow including EA lifecycle state."""
    coordinator = _get_dept_coordinator()
    status = coordinator.get_workflow_status(workflow_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    return status


@router.post("/canonical/{workflow_id}/cancel", response_model=Dict[str, Any])
async def cancel_canonical_workflow(workflow_id: str):
    """Cancel a running canonical workflow."""
    coordinator = _get_dept_coordinator()
    success = coordinator.cancel_workflow(workflow_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel workflow {workflow_id}"
        )
    return {"success": True, "workflow_id": workflow_id, "status": "cancelled"}


@router.post("/canonical/{workflow_id}/wf2-loop", response_model=Dict[str, Any])
async def check_and_restart_wf2_loop(workflow_id: str):
    """
    Check if WF2 Enhancement Loop should iterate and restart if needed.

    WF2 is a continuous improvement loop. After each round, this endpoint
    checks if further improvement is warranted and restarts from
    RESEARCH_IMPROVEMENT if so.
    """
    coordinator = _get_dept_coordinator()

    if not coordinator.should_loop_wf2(workflow_id):
        return {
            "workflow_id": workflow_id,
            "should_loop": False,
            "message": "No further improvement needed",
        }

    result = coordinator.restart_wf2_round(workflow_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "workflow_id": workflow_id,
        "should_loop": True,
        "restart_result": result,
    }
