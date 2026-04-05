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


_WORKFLOW_STATE_TO_TASK_STATUS = {
    "PENDING": "TODO",
    "RUNNING": "IN_PROGRESS",
    "PENDING_REVIEW": "BLOCKED",
    "EXPIRED_REVIEW": "BLOCKED",
    "CANCELLED": "BLOCKED",
    "DONE": "DONE",
}


def _workflow_task_to_task_status(status: Optional[str]) -> str:
    state = (status or "").strip().lower()
    if state in {"running", "in_progress"}:
        return "IN_PROGRESS"
    if state in {"completed", "done"}:
        return "DONE"
    if state in {"failed", "cancelled", "canceled", "expired_review", "pending_review", "waiting"}:
        return "BLOCKED"
    return "TODO"


def _canonical_task_signature(tasks: List[Dict[str, Any]]) -> List[tuple[str, str, str]]:
    signature: List[tuple[str, str, str]] = []
    for task in tasks:
        signature.append(
            (
                str(task.get("task_id", "")),
                str(task.get("status", "")),
                str(task.get("updated_at") or task.get("created_at") or ""),
            )
        )
    return sorted(signature)


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


def _normalize_department_name(value: Optional[str]) -> str:
    return (value or "").strip().lower().replace(" ", "-").replace("_", "-")


def _humanize_stage_label(value: Optional[str]) -> str:
    text = (value or "").strip().replace("_", " ")
    return text.title() if text else "Task"


def _workflow_description(workflow: Dict[str, Any]) -> str:
    parts: List[str] = []
    strategy_id = workflow.get("strategy_id")
    if strategy_id:
        parts.append(f"Strategy: {strategy_id}.")
    current_stage = workflow.get("current_stage")
    if current_stage:
        parts.append(f"Stage: {current_stage}.")
    next_step = workflow.get("next_step")
    if next_step and next_step != current_stage:
        parts.append(f"Next: {next_step}.")
    waiting_reason = workflow.get("waiting_reason")
    if waiting_reason:
        parts.append(f"Waiting: {waiting_reason}.")
    latest_artifact = workflow.get("latest_artifact") or {}
    artifact_path = latest_artifact.get("path")
    if artifact_path:
        parts.append(f"Latest artifact: {artifact_path}.")
    blocking_error = workflow.get("blocking_error")
    if blocking_error:
        parts.append(f"Blocking error: {blocking_error}.")
    return " ".join(parts).strip()


def _load_canonical_workflows() -> List[Dict[str, Any]]:
    from src.api.prefect_workflow_endpoints import (
        _load_db_workflows,
        _load_wf1_manifest_workflows,
    )

    workflows: Dict[str, Dict[str, Any]] = {}
    workflows.update(_load_db_workflows())
    workflows.update(_load_wf1_manifest_workflows())

    # Use the full coordinator payload here rather than the lossy FlowForge card
    # serializer so department kanban decomposition preserves task routing/status.
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()

        # First hydrate/override any persisted DB or manifest workflows so task
        # decomposition remains correct after API restarts.
        for workflow_id in list(workflows.keys()):
            hydrated = coordinator.get_workflow_status(workflow_id)
            if hydrated:
                workflows[workflow_id] = hydrated

        # Then add any additional in-memory live workflows that may not exist in
        # durable persistence yet.
        for workflow in coordinator.get_all_workflows():
            workflow_id = workflow.get("workflow_id")
            if workflow_id:
                workflows[workflow_id] = workflow
    except Exception as exc:
        logger.info(f"Coordinator task workflow metadata unavailable: {exc}")

    return list(workflows.values())


def _map_workflow_to_task(workflow: Dict[str, Any]) -> Dict[str, Any]:
    state = str(workflow.get("state", "PENDING")).upper()
    department = _normalize_department_name(workflow.get("department")) or "flowforge"
    workflow_id = workflow.get("workflow_id") or workflow.get("id") or workflow.get("flow_id") or "workflow"
    strategy_id = workflow.get("strategy_id")
    current_stage = workflow.get("current_stage") or workflow.get("next_step") or "Pending"
    latest_artifact = workflow.get("latest_artifact")
    blocking_error = workflow.get("blocking_error")
    waiting_reason = workflow.get("waiting_reason")

    priority = "HIGH" if blocking_error else "MEDIUM"
    task_name_root = strategy_id or workflow.get("name") or workflow_id
    task_name = f"{task_name_root} · {current_stage}"

    task: Dict[str, Any] = {
        "task_id": f"workflow:{workflow_id}:{department}",
        "task_name": task_name,
        "description": _workflow_description(workflow),
        "department": department,
        "priority": priority,
        "status": _WORKFLOW_STATE_TO_TASK_STATUS.get(state, "TODO"),
        "source_dept": "flowforge",
        "message_type": workflow.get("name") or "workflow",
        "workflow_id": workflow_id,
        "kanban_card_id": None,
        "mail_message_id": None,
        "created_at": workflow.get("started_at") or workflow.get("updated_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": workflow.get("updated_at") or workflow.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "started_at": workflow.get("started_at"),
        "completed_at": workflow.get("updated_at") if state == "DONE" else None,
        "strategy_id": strategy_id,
        "current_stage": current_stage,
        "next_step": workflow.get("next_step"),
        "blocking_error": blocking_error,
        "waiting_reason": waiting_reason,
        "latest_artifact": latest_artifact,
        "source_kind": "workflow",
        "read_only": True,
    }

    if task["status"] == "IN_PROGRESS" and not task["started_at"]:
        task["started_at"] = task["updated_at"]

    return task


def _map_workflow_to_subtasks(
    workflow: Dict[str, Any],
    department: str,
) -> List[Dict[str, Any]]:
    normalized_department = _normalize_department_name(department)
    workflow_id = str(workflow.get("workflow_id") or workflow.get("id") or workflow.get("flow_id") or "workflow")
    workflow_name = str(workflow.get("name") or "workflow")
    workflow_strategy = workflow.get("strategy_id")
    workflow_stage = workflow.get("current_stage") or workflow.get("next_step")
    workflow_updated_at = workflow.get("updated_at") or workflow.get("started_at") or datetime.now(timezone.utc).isoformat()
    workflow_error = workflow.get("blocking_error")
    workflow_waiting = workflow.get("waiting_reason")
    workflow_artifact = workflow.get("latest_artifact")

    subtasks: List[Dict[str, Any]] = []
    for raw_task in workflow.get("tasks", []):
        to_dept = _normalize_department_name(raw_task.get("to_dept")) or _normalize_department_name(workflow.get("department"))
        if normalized_department != "flowforge" and to_dept != normalized_department:
            continue

        from_dept = _normalize_department_name(raw_task.get("from_dept")) or "flowforge"
        stage = raw_task.get("stage")
        task_id = str(raw_task.get("task_id") or raw_task.get("id") or f"{workflow_id}:task")
        task_priority = str(raw_task.get("priority", "medium")).upper()
        if task_priority not in {"LOW", "MEDIUM", "HIGH"}:
            task_priority = "MEDIUM"

        stage_label = _humanize_stage_label(stage or "task")
        task_name = f"{stage_label} · {workflow_name}"
        payload = raw_task.get("payload") or {}
        message_id = raw_task.get("message_id")

        subtasks.append(
            {
                "task_id": f"workflow:{workflow_id}:{to_dept}:{task_id}",
                "task_name": task_name,
                "description": _workflow_description(workflow) or f"{stage_label} task routed from {from_dept} to {to_dept}.",
                "department": to_dept or normalized_department,
                "priority": task_priority,
                "status": _workflow_task_to_task_status(raw_task.get("status")),
                "source_dept": from_dept,
                "message_type": stage or workflow_name,
                "workflow_id": workflow_id,
                "kanban_card_id": None,
                "mail_message_id": message_id,
                "created_at": raw_task.get("created_at") or workflow.get("started_at") or workflow_updated_at,
                "updated_at": raw_task.get("completed_at") or workflow_updated_at,
                "started_at": raw_task.get("created_at"),
                "completed_at": raw_task.get("completed_at"),
                "strategy_id": workflow_strategy,
                "current_stage": stage or workflow_stage,
                "next_step": workflow.get("next_step"),
                "blocking_error": raw_task.get("error") or workflow_error,
                "waiting_reason": workflow_waiting,
                "latest_artifact": workflow_artifact,
                "source_kind": "workflow",
                "read_only": True,
                "subagent": payload.get("agent_id") or payload.get("subagent") or payload.get("owner"),
            }
        )

    return subtasks


def _workflow_matches_department(workflow: Dict[str, Any], department: str) -> bool:
    normalized = _normalize_department_name(department)
    if normalized == "shared-assets":
        return False
    if normalized == "flowforge":
        return True

    top_level_department = _normalize_department_name(workflow.get("department"))
    if top_level_department == normalized:
        return True

    for task in workflow.get("tasks", []):
        if _normalize_department_name(task.get("to_dept")) == normalized:
            return True

    current_stage = _normalize_department_name(workflow.get("current_stage"))
    if current_stage == normalized:
        return True

    return False


def _merge_task_sources(
    workflow_tasks: List[Dict[str, Any]],
    mail_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}

    for task in workflow_tasks:
        key = (
            task.get("workflow_id") or task["task_id"],
            _normalize_department_name(task.get("department")),
        )
        merged[key] = task

    result: List[Dict[str, Any]] = []
    for mail_task in mail_tasks:
        key = (
            mail_task.get("workflow_id") or mail_task["task_id"],
            _normalize_department_name(mail_task.get("department")),
        )
        workflow_task = merged.pop(key, None)
        if workflow_task:
            merged_task = {
                **workflow_task,
                "task_id": mail_task["task_id"],
                "task_name": mail_task.get("task_name") or workflow_task["task_name"],
                "description": mail_task.get("description") or workflow_task.get("description"),
                "source_dept": mail_task.get("source_dept") or workflow_task.get("source_dept"),
                "message_type": mail_task.get("message_type") or workflow_task.get("message_type"),
                "kanban_card_id": mail_task.get("kanban_card_id"),
                "mail_message_id": mail_task.get("mail_message_id"),
                "created_at": mail_task.get("created_at") or workflow_task.get("created_at"),
                "updated_at": mail_task.get("updated_at") or workflow_task.get("updated_at"),
                "source_kind": "workflow+mail",
                "read_only": False,
            }
            if merged_task["status"] == "IN_PROGRESS":
                merged_task["started_at"] = mail_task.get("updated_at") or workflow_task.get("started_at") or merged_task["created_at"]
            if merged_task["status"] == "DONE":
                merged_task["completed_at"] = mail_task.get("updated_at") or workflow_task.get("completed_at") or merged_task["updated_at"]
            result.append(merged_task)
        else:
            mail_task["source_kind"] = "mail"
            mail_task["read_only"] = False
            result.append(mail_task)

    result.extend(merged.values())
    return sorted(result, key=lambda task: task.get("updated_at") or task.get("created_at") or "", reverse=True)


def _get_department_tasks(department: str) -> List[Dict]:
    """
    Get all real tasks for a department or aggregate canvas.

    Source of truth is the task manager populated by the background mail
    consumers. No mock/sample tasks are returned.
    """
    task_manager = get_task_manager()
    mail_tasks: List[Dict[str, Any]] = []

    for resolved_department in _resolve_departments(department):
        todos = task_manager.get_todos(resolved_department)
        mail_tasks.extend(_map_todo_to_task(todo.to_dict()) for todo in todos)

    workflow_tasks: List[Dict[str, Any]] = []
    for workflow in _load_canonical_workflows():
        if not _workflow_matches_department(workflow, department):
            continue
        subtasks = _map_workflow_to_subtasks(workflow, department)
        if subtasks:
            workflow_tasks.extend(subtasks)
        else:
            workflow_tasks.append(_map_workflow_to_task(workflow))

    return _merge_task_sources(workflow_tasks, mail_tasks)


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
            last_signature = _canonical_task_signature(initial_tasks)
            last_snapshot_at = datetime.now(timezone.utc)
            while True:
                message = await asyncio.to_thread(
                    pubsub.get_message,
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message and message.get("type") == "message":
                    update_data = json.loads(message["data"])
                    yield f"data: {json.dumps(update_data)}\n\n"

                now = datetime.now(timezone.utc)
                if (now - last_snapshot_at).total_seconds() >= 5:
                    latest_tasks = _get_department_tasks(department)
                    latest_signature = _canonical_task_signature(latest_tasks)
                    if latest_signature != last_signature:
                        yield f"data: {json.dumps({'type': 'initial', 'tasks': latest_tasks})}\n\n"
                        last_signature = latest_signature
                    else:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': now.isoformat()})}\n\n"
                    last_snapshot_at = now
        else:
            # Fallback: poll every 5 seconds
            last_signature = _canonical_task_signature(initial_tasks)
            while True:
                await asyncio.sleep(5)
                now = datetime.now(timezone.utc)
                latest_tasks = _get_department_tasks(department)
                latest_signature = _canonical_task_signature(latest_tasks)
                if latest_signature != last_signature:
                    yield f"data: {json.dumps({'type': 'initial', 'tasks': latest_tasks})}\n\n"
                    last_signature = latest_signature
                else:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': now.isoformat()})}\n\n"

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

    if task_id.startswith("workflow:"):
        raise HTTPException(
            status_code=409,
            detail="Workflow-derived tasks are read-only. Update workflow state through the coordinator.",
        )

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
