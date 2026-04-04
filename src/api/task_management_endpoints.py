"""
Task Management API Endpoints — To-Do List + Kanban Board

Issues addressed: #11, #12, #13, #14, #15

Endpoints:
    GET  /api/tasks/todos/{department}              — Get to-do list
    GET  /api/tasks/todos/counts                    — Get counts per dept
    PATCH /api/tasks/todos/{todo_id}/status          — Update to-do status
    GET  /api/tasks/kanban/{department}              — Get Kanban cards
    GET  /api/tasks/kanban/summary                   — Get board summary
    PATCH /api/tasks/kanban/{card_id}/status          — Update card status
    POST /api/tasks/kanban/card                       — Create standalone card
    POST /api/tasks/consumers/start                   — Start mail consumers
    POST /api/tasks/consumers/stop                    — Stop mail consumers
    POST /api/tasks/process/{department}/{message_id} — Process mail manually
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.agents.departments.mail_consumer import (
    KanbanStatus,
    TodoStatus,
    get_task_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["task-management"])


# ── Request Models ──────────────────────────────────────────────────

class UpdateTodoStatusRequest(BaseModel):
    status: str = Field(..., description="New status: pending, in_progress, blocked, completed, cancelled")


class UpdateKanbanStatusRequest(BaseModel):
    status: str = Field(..., description="New status: inbox, processing, review, pending_approval, completed, failed")
    result: Optional[str] = Field(None, description="Optional result/notes")


class CreateKanbanCardRequest(BaseModel):
    department: str = Field(..., description="Department for the card")
    title: str = Field(..., description="Card title")
    description: str = Field("", description="Card description")
    priority: str = Field("normal", description="Priority: low, normal, high, urgent")
    workflow_id: Optional[str] = None
    strategy_id: Optional[str] = None


class ProcessMailRequest(BaseModel):
    result: str = Field(..., description="Processing result to send back")
    next_steps: Optional[str] = Field(None, description="Optional next steps")


# ── To-Do List Endpoints (Issue #12) ────────────────────────────────

@router.get("/todos/{department}")
async def get_todos(
    department: str,
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """Get to-do list for a department."""
    mgr = get_task_manager()
    todo_status = TodoStatus(status) if status else None
    todos = mgr.get_todos(department, status=todo_status)
    return {
        "department": department,
        "count": len(todos),
        "todos": [t.to_dict() for t in todos],
    }


@router.get("/todos/counts/all")
async def get_todo_counts():
    """Get to-do counts per department per status."""
    return get_task_manager().get_todo_counts()


@router.patch("/todos/{todo_id}/status")
async def update_todo_status(todo_id: str, body: UpdateTodoStatusRequest):
    """Update a to-do item's status."""
    try:
        status = TodoStatus(body.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {body.status}. Valid: {[s.value for s in TodoStatus]}",
        )

    todo = get_task_manager().update_todo_status(todo_id, status)
    if not todo:
        raise HTTPException(status_code=404, detail="Todo not found")
    return todo.to_dict()


# ── Kanban Board Endpoints (Issue #13) ──────────────────────────────

@router.get("/kanban/{department}")
async def get_kanban_cards(
    department: str,
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """Get Kanban cards for a department."""
    mgr = get_task_manager()
    kb_status = KanbanStatus(status) if status else None
    cards = mgr.get_kanban_cards(department, status=kb_status)
    return {
        "department": department,
        "count": len(cards),
        "cards": [c.to_dict() for c in cards],
    }


@router.get("/kanban/summary/all")
async def get_kanban_summary():
    """Get Kanban board summary (counts per department per status)."""
    return get_task_manager().get_kanban_summary()


@router.patch("/kanban/{card_id}/status")
async def update_kanban_status(card_id: str, body: UpdateKanbanStatusRequest):
    """Update a Kanban card's status."""
    try:
        status = KanbanStatus(body.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {body.status}. Valid: {[s.value for s in KanbanStatus]}",
        )

    card = get_task_manager().update_kanban_status(
        card_id, status, result=body.result,
    )
    if not card:
        raise HTTPException(status_code=404, detail="Kanban card not found")
    return card.to_dict()


@router.post("/kanban/card")
async def create_kanban_card(body: CreateKanbanCardRequest):
    """Create a standalone Kanban card."""
    card = get_task_manager().create_kanban_card(
        department=body.department,
        title=body.title,
        description=body.description,
        priority=body.priority,
        workflow_id=body.workflow_id,
        strategy_id=body.strategy_id,
    )
    return card.to_dict()


# ── Response Mail (Issue #14) ───────────────────────────────────────

@router.post("/process/{department}/{message_id}")
async def process_mail_and_respond(
    department: str,
    message_id: str,
    body: ProcessMailRequest,
):
    """
    Process a mail message and send a response back.

    Marks the task as complete and sends result to original sender.
    """
    mgr = get_task_manager()

    # Find and update the associated todo/kanban
    for todo in mgr.get_todos(department):
        if todo.mail_message_id == message_id:
            mgr.update_todo_status(todo.id, TodoStatus.COMPLETED)
            if todo.kanban_card_id:
                mgr.update_kanban_status(
                    todo.kanban_card_id,
                    KanbanStatus.COMPLETED,
                    result=body.result,
                )
            break

    # Send response mail
    mgr.send_response_mail(
        department=department,
        original_message_id=message_id,
        result=body.result,
        next_steps=body.next_steps,
    )

    return {
        "status": "success",
        "department": department,
        "message_id": message_id,
    }


# ── Mail Consumer Management (Issue #11) ────────────────────────────

@router.post("/consumers/start")
async def start_mail_consumers(
    poll_interval: float = Query(10.0, description="Poll interval in seconds"),
):
    """Start background mail consumers for all departments."""
    await get_task_manager().start_consumers(poll_interval=poll_interval)
    return {"status": "started", "departments": get_task_manager().DEPARTMENTS}


@router.post("/consumers/stop")
async def stop_mail_consumers():
    """Stop all background mail consumers."""
    await get_task_manager().stop_consumers()
    return {"status": "stopped"}
