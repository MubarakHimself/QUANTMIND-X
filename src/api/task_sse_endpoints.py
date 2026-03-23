"""
Task SSE Endpoints

Provides Server-Sent Events (SSE) for real-time task updates.
Endpoint: GET /api/sse/tasks/{department}
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.agents.departments.types import Department

logger = logging.getLogger(__name__)

router = APIRouter()


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


# ============================================================================
# In-Memory Task State (mirrors Redis from TaskRouter)
# ============================================================================

# Global task state store (in production, this would come from Redis)
_task_state: Dict[str, Dict[str, Dict]] = {}


def _get_department_tasks(department: str) -> List[Dict]:
    """
    Get all tasks for a department from the task state.

    In production, this would query Redis via TaskRouter.
    For now, returns mock data demonstrating the interface.
    """
    # Try to get from global state
    if department in _task_state:
        return list(_task_state[department].values())

    # Return sample data for demonstration
    # In production: use TaskRouter.get_department_status()
    sample_tasks = [
        {
            "task_id": "task_001",
            "task_name": "Research market data",
            "department": department,
            "priority": "HIGH",
            "status": "IN_PROGRESS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "task_id": "task_002",
            "task_name": "Analyze strategy performance",
            "department": department,
            "priority": "MEDIUM",
            "status": "TODO",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "task_id": "task_003",
            "task_name": "Review risk metrics",
            "department": department,
            "priority": "LOW",
            "status": "BLOCKED",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "task_id": "task_004",
            "task_name": "Generate report",
            "department": department,
            "priority": "MEDIUM",
            "status": "DONE",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    return sample_tasks


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

        # If Redis pub/sub available, wait for messages
        if pubsub:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    update_data = json.loads(message['data'])
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
    # Validate department
    try:
        dept = Department(department.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department: {department}. Must be one of: {[d.value for d in Department]}"
        )

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
    try:
        dept = Department(department.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department: {department}"
        )

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

    # In production: update Redis via TaskRouter
    # For now, just acknowledge

    return {
        "task_id": task_id,
        "status": update.status,
        "updated_at": datetime.now(timezone.utc).isoformat()
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