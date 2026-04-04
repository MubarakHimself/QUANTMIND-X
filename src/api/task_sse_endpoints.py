"""
Task SSE Endpoints

Provides Server-Sent Events (SSE) for real-time task updates.
Endpoint: GET /api/sse/tasks/{department}
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.departments.types import Department
from src.agents.departments.mail_consumer import TodoStatus, get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ============================================================================
# Data Models
# ============================================================================

class TaskStatusUpdate(BaseModel):
    """Task status update model for SSE."""
    task_id: str
    status: str  # TODO, IN_PROGRESS, BLOCKED, DONE
    timestamp: str


class DepartmentTaskResponse(BaseModel):
    """Response model for department tasks."""
    department: str
    tasks: List[Dict]


def _resolve_departments(department: str) -> List[str]:
    """Resolve a frontend department name into one or more real task-manager departments."""
    normalized = department.lower()
    task_manager = get_task_manager()

    if normalized in task_manager.DEPARTMENTS:
        return [normalized]
    if normalized == "flowforge":
        return list(task_manager.DEPARTMENTS)
    if normalized == "shared-assets":
        return []

    valid = [*task_manager.DEPARTMENTS, "flowforge", "shared-assets"]
    raise HTTPException(
        status_code=400,
        detail=f"Invalid department: {department}. Must be one of: {valid}",
    )


def _map_todo_to_task(todo: Dict[str, Any]) -> Dict[str, Any]:
    status_map = {
        TodoStatus.PENDING.value: "TODO",
        TodoStatus.IN_PROGRESS.value: "IN_PROGRESS",
        TodoStatus.BLOCKED.value: "BLOCKED",
        TodoStatus.COMPLETED.value: "DONE",
        TodoStatus.CANCELLED.value: "DONE",
    }
    priority = str(todo.get("priority", "medium")).upper()
    if priority not in {"LOW", "MEDIUM", "HIGH"}:
        priority = "MEDIUM"

    task: Dict[str, Any] = {
        "task_id": todo["id"],
        "task_name": todo.get("title") or "Untitled task",
        "description": todo.get("description") or "",
        "department": todo.get("department", "research"),
        "priority": priority,
        "status": status_map.get(todo.get("status", ""), "TODO"),
        "source_dept": todo.get("source_dept") or "",
        "message_type": todo.get("message_type") or "",
        "workflow_id": todo.get("workflow_id"),
        "kanban_card_id": todo.get("kanban_card_id"),
        "mail_message_id": todo.get("mail_message_id"),
        "created_at": todo.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": todo.get("updated_at") or todo.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }

    if task["status"] == "IN_PROGRESS":
        task["started_at"] = todo.get("updated_at") or task["created_at"]
    if task["status"] == "DONE":
        task["completed_at"] = todo.get("updated_at") or task["created_at"]
    return task


def _get_department_tasks(department: str) -> List[Dict]:
    """
    Get all real tasks for a department or aggregate canvas.

    Source of truth is the task manager populated by the background mail
    consumers. No mock/sample tasks are returned.
    """
    task_manager = get_task_manager()
    tasks: List[Dict[str, Any]] = []

    for resolved_department in _resolve_departments(department):
        todos = task_manager.get_todos(resolved_department)
        tasks.extend(_map_todo_to_task(todo.to_dict()) for todo in todos)

    return sorted(tasks, key=lambda task: task["created_at"], reverse=True)


async def _stream_task_updates(department: str) -> AsyncGenerator[str, None]:
    """
    Stream task updates via SSE.

    Uses Redis pub/sub for real-time updates in production.
    Falls back to polling if Redis is unavailable.
    """
    import redis
    from redis import Redis

    redis_client: Optional[Redis] = None

    # Try to connect to Redis
    try:
        redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis for task SSE")
    except Exception as e:
        logger.warning(f"Redis unavailable for task SSE: {e}. Using fallback polling.")
        redis_client = None

    # Subscribe to Redis channel for task updates
    channel_name = f"task:dept:{department}:updates"

    pubsub = None
    if redis_client:
        try:
            pubsub = redis_client.pubsub()
            pubsub.subscribe(channel_name)
        except Exception as e:
            logger.warning(f"Failed to subscribe to Redis channel: {e}")
            pubsub = None

    try:
        # Send initial tasks
        initial_tasks = _get_department_tasks(department)
        yield f"data: {json.dumps({'type': 'initial', 'tasks': initial_tasks})}\n\n"

        # If Redis pub/sub available, poll it in a worker thread so the
        # synchronous redis client cannot block the FastAPI event loop.
        if pubsub:
            while True:
                message = await asyncio.to_thread(
                    pubsub.get_message,
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message and message.get("type") == "message":
                    update_data = json.loads(message["data"])
                    yield f"data: {json.dumps(update_data)}\n\n"
        else:
            # Fallback: poll every 5 seconds
            last_check = datetime.now(timezone.utc)
            while True:
                await asyncio.sleep(5)

                # In production: check Redis stream for new task updates
                # For now, send a heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"

    except asyncio.CancelledError:
        logger.info(f"SSE connection closed for department: {department}")
        raise
    finally:
        if pubsub:
            try:
                pubsub.unsubscribe(channel_name)
                pubsub.close()
            except Exception:
                pass


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/sse/tasks/{department}")
async def stream_department_tasks(department: str):
    """
    SSE endpoint for real-time task updates.

    Stream format:
    - Initial: { "type": "initial", "tasks": [...] }
    - Update: { "task_id": "...", "status": "...", "timestamp": "..." }
    - Heartbeat: { "type": "heartbeat", "timestamp": "..." }

    Args:
        department: Department name (research, development, risk, trading, portfolio)

    Returns:
        StreamingResponse with SSE data
    """
    _resolve_departments(department)

    logger.info(f"SSE connection opened for department: {department}")

    return StreamingResponse(
        _stream_task_updates(department),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/tasks/{department}")
async def get_department_tasks(department: str):
    """
    REST endpoint to get current tasks for a department.

    Args:
        department: Department name

    Returns:
        DepartmentTaskResponse with current tasks
    """
    _resolve_departments(department)

    tasks = _get_department_tasks(department)

    return DepartmentTaskResponse(
        department=department,
        tasks=tasks
    )


@router.patch("/tasks/{task_id}/status")
async def update_task_status(task_id: str, update: TaskStatusUpdate):
    """
    Update task status via REST (alternative to SSE for status changes).

    Args:
        task_id: Task ID
        update: New status

    Returns:
        Updated task
    """
    logger.info(f"Task {task_id} status update: {update.status}")

    status_map = {
        "TODO": TodoStatus.PENDING,
        "IN_PROGRESS": TodoStatus.IN_PROGRESS,
        "BLOCKED": TodoStatus.BLOCKED,
        "DONE": TodoStatus.COMPLETED,
    }
    todo_status = status_map.get(update.status.upper())
    if todo_status is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid status. Must be one of: TODO, IN_PROGRESS, BLOCKED, DONE",
        )

    updated = get_task_manager().update_todo_status(task_id, todo_status)
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": updated.id,
        "status": update.status.upper(),
        "updated_at": updated.updated_at,
    }


# ============================================================================
# Utility: Publish task update (called by TaskRouter)
# ============================================================================

def publish_task_update(department: str, task_id: str, status: str) -> None:
    """
    Publish a task update to Redis for SSE distribution.

    This function should be called by TaskRouter when task status changes.
    """
    import redis
    from redis import Redis

    try:
        redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()

        channel = f"task:dept:{department}:updates"
        update_data = json.dumps({
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        redis_client.publish(channel, update_data)

        logger.info(f"Published task update: {task_id} -> {status}")
    except Exception as e:
        logger.warning(f"Failed to publish task update: {e}")
