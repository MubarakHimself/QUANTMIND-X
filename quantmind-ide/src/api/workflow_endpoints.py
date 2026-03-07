"""
Workflow API Endpoints.

REST API for workflow and queue management.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..workflows.state import (
    WorkflowState,
    WorkflowStatus,
    get_workflow_store,
)
from ..workflows.video_ingest_to_ea import (
    VideoIngestToEAWorkflow,
    create_video_ingest_to_ea_workflow,
    run_video_ingest_to_ea_workflow,
)
from ..agents.queue import (
    QueueManager,
    Task,
    TaskPriority,
    get_queue_manager,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["workflow", "queue"])


# Request Models
class VideoIngestToEARequest(BaseModel):
    """Request to start VideoIngest to EA workflow."""
    video_ingest_content: str
    workspace_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AddTaskRequest(BaseModel):
    """Request to add a task."""
    agent_type: str
    name: str
    description: str = ""
    input_data: Dict[str, Any] = {}
    priority: int = 5
    dependencies: List[str] = []


# Workflow Endpoints

@router.post("/workflows/video-ingest-to-ea")
async def start_video_ingest_to_ea_workflow(
    request: VideoIngestToEARequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Start VideoIngest to EA workflow.

    Processes VideoIngest through Analyst to generate TRD,
    then through QuantCode to generate MQL5 EA.
    """
    workflow = create_video_ingest_to_ea_workflow(request.workspace_path)
    state = workflow.create_state(request.video_ingest_content, request.metadata)

    # Run in background
    async def run_workflow():
        try:
            await workflow.run(request.video_ingest_content, request.metadata)
        except Exception as e:
            logger.error(f"Workflow error: {e}")

    background_tasks.add_task(run_workflow)

    return {
        "workflow_id": state.workflow_id,
        "status": state.status.value,
        "message": "Workflow started",
    }


@router.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """
    Get workflow status.

    Returns current progress, step, and intermediate results.
    """
    store = get_workflow_store()
    state = store.get(workflow_id)

    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return state.to_dict()


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Cancel a running workflow.
    """
    store = get_workflow_store()
    state = store.get(workflow_id)

    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if state.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PENDING):
        raise HTTPException(status_code=400, detail="Workflow cannot be cancelled")

    state.cancel()
    store.save(state)

    return {
        "cancelled": True,
        "workflow_id": workflow_id,
    }


@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Pause a running workflow.
    """
    store = get_workflow_store()
    state = store.get(workflow_id)

    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if state.status != WorkflowStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Workflow is not running")

    state.pause()
    store.save(state)

    return {
        "paused": True,
        "workflow_id": workflow_id,
    }


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Resume a paused workflow.
    """
    store = get_workflow_store()
    state = store.get(workflow_id)

    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if state.status != WorkflowStatus.PAUSED:
        raise HTTPException(status_code=400, detail="Workflow is not paused")

    state.resume()
    store.save(state)

    return {
        "resumed": True,
        "workflow_id": workflow_id,
    }


@router.get("/workflows")
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, description="Max results"),
) -> Dict[str, Any]:
    """
    List workflows.

    Optionally filter by status.
    """
    store = get_workflow_store()

    if status:
        try:
            status_enum = WorkflowStatus(status)
            states = store.list_by_status(status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status")
    else:
        states = store.list_all()

    return {
        "workflows": [s.to_dict() for s in states[:limit]],
        "total": len(states),
    }


# Agent Endpoints

@router.get("/agents/{agent}/status")
async def get_agent_status(agent: str) -> Dict[str, Any]:
    """
    Get agent status.

    Returns queue status and current task.
    """
    queue_manager = get_queue_manager()
    queue_status = queue_manager.get_queue_status(agent)

    return {
        "agent_type": agent,
        "queue": queue_status,
    }


@router.get("/agents/{agent}/queue")
async def get_agent_queue(agent: str) -> Dict[str, Any]:
    """
    Get agent task queue.
    """
    queue_manager = get_queue_manager()
    queue_status = queue_manager.get_queue_status(agent)

    if not queue_status.get("exists"):
        raise HTTPException(status_code=404, detail="Agent not found")

    return queue_status


@router.post("/agents/{agent}/queue")
async def add_task_to_queue(
    agent: str,
    request: AddTaskRequest,
) -> Dict[str, Any]:
    """
    Add task to agent queue.
    """
    queue_manager = get_queue_manager()

    task = Task(
        name=request.name,
        description=request.description,
        agent_type=agent,
        priority=request.priority,
        input_data=request.input_data,
        dependencies=request.dependencies,
    )

    task_id = queue_manager.add_task(task)

    return {
        "task_id": task_id,
        "status": "queued",
        "agent_type": agent,
    }


@router.delete("/agents/{agent}/queue/{task_id}")
async def cancel_task(agent: str, task_id: str) -> Dict[str, Any]:
    """
    Cancel a task.
    """
    queue_manager = get_queue_manager()
    success = queue_manager.cancel_task(task_id)

    if not success:
        raise HTTPException(status_code=400, detail="Could not cancel task")

    return {
        "cancelled": True,
        "task_id": task_id,
    }


@router.post("/agents/{agent}/queue/{task_id}/retry")
async def retry_task(agent: str, task_id: str) -> Dict[str, Any]:
    """
    Retry a failed task.
    """
    queue_manager = get_queue_manager()
    success = queue_manager.retry_task(task_id)

    if not success:
        raise HTTPException(status_code=400, detail="Could not retry task")

    return {
        "retried": True,
        "task_id": task_id,
    }
