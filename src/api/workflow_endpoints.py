"""
Workflow API Endpoints

Provides REST API endpoints for workflow management and execution.
Supports VideoIngest to EA workflow, status tracking, and control operations.

Delegates all workflow operations to the WorkflowOrchestrator for consistent
state management and real agent/MCP integration.

Endpoints match the expected shapes from workflowStore.ts:
- workflow_id, workflow_type, status, progress_percent, steps, final_result, timestamps
"""

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
