"""
Agent Queue API Endpoints

Provides REST API endpoints for agent task queue management.
Uses the AgentQueueManager for real task execution and persistence.
Supports queue status, task listing, cancellation, and retry.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

# Import queue manager
try:
    from src.agents.queue_manager import (
        get_queue_manager,
        initialize_queue_manager,
        AgentQueueManager,
        Task,
        TaskStatus,
        AgentQueueConfig,
    )
    QUEUE_MANAGER_AVAILABLE = True
except ImportError:
    QUEUE_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


# ============================================================================
# Enums and Models
# ============================================================================

class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueueStatusResponse(BaseModel):
    agent_type: str
    exists: bool
    pending_count: int
    running_count: int
    max_concurrent: int
    is_full: bool


class TaskResponse(BaseModel):
    task_id: str
    name: str
    description: str
    agent_type: str
    priority: int
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    retries: int


class TaskListResponse(BaseModel):
    agent_type: str
    tasks: List[TaskResponse]
    total_count: int


class SubmitTaskRequest(BaseModel):
    name: str
    description: str = ""
    priority: int = 5
    input_data: Dict[str, Any] = {}


# ============================================================================
# Fallback In-Memory Storage (when queue manager unavailable)
# ============================================================================

_agent_tasks: Dict[str, List[Dict[str, Any]]] = {
    "copilot": [],
    "analyst": [],
    "quantcode": [],
}

_agent_config: Dict[str, Dict[str, Any]] = {
    "copilot": {"max_concurrent": 2},
    "analyst": {"max_concurrent": 1},
    "quantcode": {"max_concurrent": 1},
}


def _get_fallback_queue_status(agent_type: str) -> Dict[str, Any]:
    """Get queue status for an agent type (fallback)."""
    tasks = _agent_tasks.get(agent_type, [])
    config = _agent_config.get(agent_type, {"max_concurrent": 1})

    pending_count = len([t for t in tasks if t["status"] in ["pending", "queued"]])
    running_count = len([t for t in tasks if t["status"] == "running"])
    max_concurrent = config.get("max_concurrent", 1)

    return {
        "agent_type": agent_type,
        "exists": True,
        "pending_count": pending_count,
        "running_count": running_count,
        "max_concurrent": max_concurrent,
        "is_full": running_count >= max_concurrent,
    }


def _task_to_response(task) -> TaskResponse:
    """Convert task to response model."""
    if isinstance(task, dict):
        return TaskResponse(
            task_id=task["task_id"],
            name=task["name"],
            description=task.get("description", ""),
            agent_type=task["agent_type"],
            priority=task.get("priority", 5),
            status=task["status"],
            created_at=task["created_at"],
            started_at=task.get("started_at"),
            completed_at=task.get("completed_at"),
            duration_seconds=task.get("duration_seconds"),
            error=task.get("error"),
            retries=task.get("retries", 0),
        )
    else:
        # Task object from queue manager
        return TaskResponse(
            task_id=task.task_id,
            name=task.name,
            description=task.description,
            agent_type=task.agent_type,
            priority=task.priority,
            status=task.status,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration_seconds=task.duration_seconds,
            error=task.error,
            retries=task.retries,
        )


def _get_manager() -> Optional[AgentQueueManager]:
    """Get queue manager if available."""
    if not QUEUE_MANAGER_AVAILABLE:
        return None
    try:
        return get_queue_manager()
    except Exception as e:
        logger.warning(f"Failed to get queue manager: {e}")
        return None


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/{agent_type}/queue", response_model=QueueStatusResponse)
async def get_queue_status(agent_type: str):
    """
    Get queue status for a specific agent type.
    """
    manager = _get_manager()

    if manager:
        if agent_type not in ["copilot", "analyst", "quantcode"]:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        status = manager.get_queue_status(agent_type)
        return QueueStatusResponse(**status)
    else:
        # Fallback
        if agent_type not in _agent_tasks:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        status = _get_fallback_queue_status(agent_type)
        return QueueStatusResponse(**status)


@router.get("/{agent_type}/queue/tasks", response_model=TaskListResponse)
async def get_queue_tasks(
    agent_type: str,
    status: Optional[str] = Query(None),
    limit: int = Query(50),
):
    """
    Get tasks in queue for a specific agent type.
    """
    manager = _get_manager()

    if manager:
        if agent_type not in ["copilot", "analyst", "quantcode"]:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        tasks = manager.get_tasks(agent_type, status=status, limit=limit)
        return TaskListResponse(
            agent_type=agent_type,
            tasks=[_task_to_response(t) for t in tasks],
            total_count=len(tasks),
        )
    else:
        # Fallback
        if agent_type not in _agent_tasks:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        tasks = _agent_tasks[agent_type]

        if status:
            tasks = [t for t in tasks if t["status"] == status]

        tasks = sorted(tasks, key=lambda t: (t.get("priority", 5), t["created_at"]))
        tasks = tasks[:limit]

        return TaskListResponse(
            agent_type=agent_type,
            tasks=[_task_to_response(t) for t in tasks],
            total_count=len(tasks),
        )


@router.delete("/{agent_type}/queue/{task_id}", response_model=Dict[str, Any])
async def cancel_task(agent_type: str, task_id: str):
    """
    Cancel a task in the queue.
    """
    manager = _get_manager()

    if manager:
        success = manager.cancel_task(task_id)
        if not success:
            task = manager.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel task with status {task.status}"
                )

        logger.info(f"Cancelled task {task_id} for agent {agent_type}")
        return {"success": True, "task_id": task_id, "status": "cancelled"}
    else:
        # Fallback
        if agent_type not in _agent_tasks:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        tasks = _agent_tasks[agent_type]
        task = next((t for t in tasks if t["task_id"] == task_id), None)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task["status"] not in ["pending", "queued"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status {task['status']}"
            )

        task["status"] = "cancelled"
        task["completed_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Cancelled task {task_id} for agent {agent_type}")
        return {"success": True, "task_id": task_id, "status": "cancelled"}


@router.post("/{agent_type}/queue/{task_id}/retry", response_model=Dict[str, Any])
async def retry_task(agent_type: str, task_id: str):
    """
    Retry a failed task.
    """
    manager = _get_manager()

    if manager:
        success = manager.retry_task(task_id)
        if not success:
            task = manager.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
            elif task.status != "failed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Can only retry failed tasks, current status: {task.status}"
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum retries exceeded"
                )

        task = manager.get_task(task_id)
        logger.info(f"Retrying task {task_id} for agent {agent_type}")
        return {
            "success": True,
            "task_id": task_id,
            "status": "queued",
            "retries": task.retries if task else 0,
        }
    else:
        # Fallback
        if agent_type not in _agent_tasks:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        tasks = _agent_tasks[agent_type]
        task = next((t for t in tasks if t["task_id"] == task_id), None)

        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        if task["status"] != "failed":
            raise HTTPException(
                status_code=400,
                detail=f"Can only retry failed tasks, current status: {task['status']}"
            )

        if task.get("retries", 0) >= 3:
            raise HTTPException(
                status_code=400,
                detail="Maximum retries exceeded"
            )

        task["status"] = "queued"
        task["retries"] = task.get("retries", 0) + 1
        task["error"] = None
        task["started_at"] = None
        task["completed_at"] = None

        logger.info(f"Retrying task {task_id} for agent {agent_type}")
        return {
            "success": True,
            "task_id": task_id,
            "status": "queued",
            "retries": task["retries"],
        }


@router.post("/{agent_type}/queue/submit", response_model=TaskResponse)
async def submit_task(
    agent_type: str,
    request: SubmitTaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a new task to the agent queue.

    The task will be executed by the queue manager if available.
    """
    manager = _get_manager()

    if manager:
        if agent_type not in ["copilot", "analyst", "quantcode"]:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        try:
            task = manager.submit_task(
                agent_type=agent_type,
                name=request.name,
                description=request.description,
                priority=request.priority,
                input_data=request.input_data
            )

            logger.info(f"Submitted task {task.task_id} to {agent_type} queue")
            return _task_to_response(task)

        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Fallback
        if agent_type not in _agent_tasks:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "task_id": task_id,
            "name": request.name,
            "description": request.description,
            "agent_type": agent_type,
            "priority": request.priority,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "error": None,
            "retries": 0,
        }

        _agent_tasks[agent_type].append(task)

        logger.info(f"Submitted task {task_id} to {agent_type} queue (fallback)")
        return _task_to_response(task)


@router.get("/queue/all", response_model=Dict[str, Any])
async def get_all_queue_statuses():
    """
    Get queue status for all agent types.
    """
    manager = _get_manager()

    if manager:
        statuses = manager.get_all_queue_statuses()
        return {"queues": statuses}
    else:
        # Fallback
        statuses = {}
        for agent_type in _agent_tasks:
            statuses[agent_type] = _get_fallback_queue_status(agent_type)
        return {"queues": statuses}


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get status of a specific task by ID.
    """
    manager = _get_manager()

    if manager:
        task = manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return _task_to_response(task)
    else:
        # Fallback - search all agent queues
        for agent_type, tasks in _agent_tasks.items():
            for task in tasks:
                if task["task_id"] == task_id:
                    return _task_to_response(task)

        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")


# ============================================================================
# Startup/Shutdown Events (called from server.py)
# ============================================================================

async def startup_queue_manager():
    """Initialize and start the queue manager on app startup."""
    if not QUEUE_MANAGER_AVAILABLE:
        logger.warning("Queue manager not available, using fallback mode")
        return

    try:
        # Get websocket manager
        try:
            from src.api.websocket_endpoints import manager as ws_manager
        except ImportError:
            ws_manager = None

        manager = await initialize_queue_manager(
            persistence_path="./data/agent_queues",
            websocket_manager=ws_manager
        )

        logger.info("Queue manager initialized and started")

    except Exception as e:
        logger.error(f"Failed to initialize queue manager: {e}")


async def shutdown_queue_manager():
    """Stop the queue manager on app shutdown."""
    if not QUEUE_MANAGER_AVAILABLE:
        return

    try:
        manager = get_queue_manager()
        await manager.stop()
        logger.info("Queue manager stopped")
    except Exception as e:
        logger.error(f"Failed to stop queue manager: {e}")
