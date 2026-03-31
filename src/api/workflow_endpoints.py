"""
Workflow API Endpoints

Provides REST API endpoints for workflow management and execution.
Supports VideoIngest to EA workflow, status tracking, and control operations.

Delegates all workflow operations to the WorkflowOrchestrator for consistent
state management and real agent/MCP integration.

Endpoints match the expected shapes from workflowStore.ts:
- workflow_id, workflow_type, status, progress_percent, steps, final_result, timestamps
"""

import json
import logging
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from pathlib import Path
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
# Orchestrator Integration
# ============================================================================

def _get_orchestrator():
    """Get the workflow orchestrator instance."""
    from src.router.workflow_orchestrator import get_orchestrator
    return get_orchestrator()


def _convert_orchestrator_status(status: str) -> WorkflowStatus:
    """Convert orchestrator status to API status."""
    status_map = {
        "pending": WorkflowStatus.PENDING,
        "running": WorkflowStatus.RUNNING,
        "paused": WorkflowStatus.PAUSED,
        "completed": WorkflowStatus.COMPLETED,
        "failed": WorkflowStatus.FAILED,
        "cancelled": WorkflowStatus.CANCELLED,
    }
    return status_map.get(status, WorkflowStatus.PENDING)


def _convert_orchestrator_workflow(orch_workflow: Dict[str, Any]) -> Workflow:
    """Convert orchestrator workflow format to API format."""
    steps = []
    step_names = list(orch_workflow.get("steps", {}).keys())
    
    for i, (step_name, step_data) in enumerate(orch_workflow.get("steps", {}).items()):
        step_status = StepStatus.PENDING
        if step_data.get("status") == "running":
            step_status = StepStatus.RUNNING
        elif step_data.get("status") == "completed":
            step_status = StepStatus.COMPLETED
        elif step_data.get("status") == "failed":
            step_status = StepStatus.FAILED
        
        steps.append(WorkflowStep(
            step_id=f"{orch_workflow['workflow_id']}_step_{i}",
            name=step_name,
            description=f"Execute {step_name} stage",
            agent_type=_get_agent_type_for_step(step_name),
            status=step_status,
            started_at=step_data.get("started_at"),
            completed_at=step_data.get("completed_at"),
            output_data=step_data.get("output", {}),
            error=step_data.get("error"),
            retries=step_data.get("retry_count", 0)
        ))
    
    # Calculate progress
    completed_steps = sum(1 for s in steps if s.status == StepStatus.COMPLETED)
    progress = (completed_steps / len(steps) * 100) if steps else 0
    
    # Determine workflow type
    workflow_type = "video_ingest_to_ea"
    if "video_ingest_processing" not in orch_workflow.get("steps", {}):
        workflow_type = "trd_to_ea"
    
    return Workflow(
        workflow_id=orch_workflow["workflow_id"],
        workflow_type=workflow_type,
        status=_convert_orchestrator_status(orch_workflow.get("status", "pending")),
        current_step_index=completed_steps,
        progress_percent=progress,
        steps=steps,
        input_data=orch_workflow.get("metadata", {}),
        intermediate_results=orch_workflow.get("metadata", {}),
        final_result=orch_workflow.get("steps", {}).get("validation", {}).get("output", {}),
        created_at=orch_workflow.get("created_at", ""),
        started_at=orch_workflow.get("steps", {}).get("video_ingest_processing", {}).get("started_at") or
                   orch_workflow.get("steps", {}).get("quantcode", {}).get("started_at"),
        completed_at=orch_workflow.get("steps", {}).get("validation", {}).get("completed_at"),
        error=orch_workflow.get("error")
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

    Delegates to the WorkflowOrchestrator for execution.
    """
    orchestrator = _get_orchestrator()
    
    # Save VideoIngest content to a temporary file
    import tempfile
    import json
    
    temp_dir = Path("workflows/video_ingest_inputs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    video_ingest_file = temp_dir / f"video_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    
    # Parse VideoIngest content if it's a string
    try:
        if isinstance(request.video_ingest_content, str):
            video_ingest_data = json.loads(request.video_ingest_content)
        else:
            video_ingest_data = request.video_ingest_content
    except json.JSONDecodeError:
        video_ingest_data = {"content": request.video_ingest_content}
    
    with open(video_ingest_file, 'w') as f:
        json.dump(video_ingest_data, f, indent=2)
    
    # Submit to orchestrator
    workflow_id = await orchestrator.submit_video_ingest_task(video_ingest_file, request.metadata)
    
    logger.info(f"Started VideoIngest to EA workflow: {workflow_id}")
    
    return {
        "workflow_id": workflow_id,
        "workflow_type": "video_ingest_to_ea",
        "status": "pending",
        "message": "Workflow started successfully",
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

    Delegates to the WorkflowOrchestrator for execution.
    """
    orchestrator = _get_orchestrator()
    
    # Save TRD content to a file
    temp_dir = Path("workflows/trd_inputs")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    trd_file = Path(request.trd_file) if request.trd_file else \
               temp_dir / f"trd_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.md"
    
    if not request.trd_file:
        with open(trd_file, 'w') as f:
            f.write(request.trd_content)
    
    # Submit to orchestrator
    workflow_id = await orchestrator.submit_trd_task(trd_file, request.trd_content, request.metadata)
    
    logger.info(f"Started TRD to EA workflow: {workflow_id}")
    
    return {
        "workflow_id": workflow_id,
        "workflow_type": "trd_to_ea",
        "status": "pending",
        "message": "Workflow started successfully",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{workflow_id}", response_model=Workflow)
async def get_workflow(workflow_id: str):
    """
    Get workflow status and details.
    
    Retrieves the current state from the WorkflowOrchestrator.
    """
    orchestrator = _get_orchestrator()
    workflow_data = orchestrator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return _convert_orchestrator_workflow(workflow_data)


@router.get("/", response_model=WorkflowListResponse)
async def list_workflows(
    status: Optional[WorkflowStatus] = Query(None, description="Filter by status")
):
    """
    List all workflows.
    
    Retrieves all workflows from the WorkflowOrchestrator.
    """
    orchestrator = _get_orchestrator()
    all_workflows = orchestrator.get_all_workflows()
    
    workflows = []
    for wf_data in all_workflows:
        workflow = _convert_orchestrator_workflow(wf_data)
        if status is None or workflow.status == status:
            workflows.append(workflow)
    
    return WorkflowListResponse(workflows=workflows)


@router.post("/{workflow_id}/cancel", response_model=Dict[str, Any])
async def cancel_workflow(workflow_id: str):
    """
    Cancel a running workflow.
    
    Delegates cancellation to the WorkflowOrchestrator.
    """
    orchestrator = _get_orchestrator()
    
    # Check if workflow exists
    workflow_data = orchestrator.get_workflow_status(workflow_id)
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    # Cancel via orchestrator
    success = await orchestrator.cancel_workflow(workflow_id)
    
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
    orchestrator = _get_orchestrator()
    workflow_data = orchestrator.get_workflow_status(workflow_id)

    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow_data.get("status") != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Can only pause running workflows, current status: {workflow_data.get('status')}"
        )

    # Call the orchestrator to pause the workflow
    success = await orchestrator.pause_workflow(workflow_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to pause workflow {workflow_id}"
        )

    logger.info(f"Paused workflow: {workflow_id}")

    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": "paused",
        "message": "Workflow paused successfully"
    }


@router.post("/{workflow_id}/resume", response_model=Dict[str, Any])
async def resume_workflow(workflow_id: str):
    """
    Resume a paused workflow.

    The workflow will continue from where it was paused.
    """
    orchestrator = _get_orchestrator()
    workflow_data = orchestrator.get_workflow_status(workflow_id)

    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

    if workflow_data.get("status") != "paused":
        raise HTTPException(
            status_code=400,
            detail=f"Can only resume paused workflows, current status: {workflow_data.get('status')}"
        )

    # Call the orchestrator to resume the workflow
    success = await orchestrator.resume_workflow(workflow_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to resume workflow {workflow_id}"
        )

    logger.info(f"Resumed workflow: {workflow_id}")

    return {
        "success": True,
        "workflow_id": workflow_id,
        "status": "running",
        "message": "Workflow resumed successfully"
    }


@router.get("/{workflow_id}/wait", response_model=Workflow)
async def wait_for_workflow(
    workflow_id: str,
    timeout: int = Query(300, description="Maximum wait time in seconds")
):
    """
    Wait for a workflow to complete.
    
    Returns the final workflow state when completed, failed, or cancelled.
    """
    orchestrator = _get_orchestrator()
    
    try:
        result = await orchestrator.wait_for_completion(workflow_id, timeout)
        return _convert_orchestrator_workflow(result)
    except TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Workflow did not complete within {timeout} seconds"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{workflow_id}/results", response_model=Dict[str, Any])
async def get_workflow_results(workflow_id: str):
    """
    Get the results of a completed workflow.
    
    Returns backtest results, validation status, and generated files.
    """
    orchestrator = _get_orchestrator()
    workflow_data = orchestrator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if workflow_data.get("status") not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Workflow not yet complete. Current status: {workflow_data.get('status')}"
        )
    
    # Extract results from workflow
    steps = workflow_data.get("steps", {})
    backtest_results = steps.get("backtest", {}).get("output", {})
    validation_results = steps.get("validation", {}).get("output", {})
    
    return {
        "workflow_id": workflow_id,
        "status": workflow_data.get("status"),
        "backtest_results": backtest_results,
        "validation": validation_results,
        "output_files": workflow_data.get("output_files", {}),
        "metadata": workflow_data.get("metadata", {}),
        "error": workflow_data.get("error")
    }


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow and its associated files.
    
    Only allowed for completed, failed, or cancelled workflows.
    """
    orchestrator = _get_orchestrator()
    workflow_data = orchestrator.get_workflow_status(workflow_id)
    
    if not workflow_data:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if workflow_data.get("status") in ["pending", "running"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running workflow. Cancel it first."
        )
    
    # Delete workflow files
    import shutil
    workflow_dir = Path("workflows") / workflow_id
    if workflow_dir.exists():
        shutil.rmtree(workflow_dir)
    
    # Remove from orchestrator's memory
    if workflow_id in orchestrator._workflows:
        del orchestrator._workflows[workflow_id]
    
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
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:8000/api/approval-gates/workflow/{workflow_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    return {"gates": [], "total": 0, "error": "Failed to fetch approvals"}
    except Exception as e:
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
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://localhost:8000/api/approval-gates/pending?workflow_id={workflow_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Filter to only this workflow's gates
                    gates = [g for g in data.get("gates", []) if g.get("workflow_id") == workflow_id]
                    return {"has_pending": len(gates) > 0, "gates": gates, "total": len(gates)}
                else:
                    return {"has_pending": False, "gates": [], "total": 0}
    except Exception as e:
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
