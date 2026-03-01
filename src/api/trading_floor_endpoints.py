"""
Trading Floor API Endpoints

Provides REST and WebSocket endpoints for the Trading Floor visualization.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.agents.departments.floor_manager import FloorManager
from src.agents.departments.types import Department

router = APIRouter(prefix="/api/trading-floor", tags=["trading-floor"])

# Global floor manager instance
_floor_manager: Optional[FloorManager] = None


def get_floor_manager() -> FloorManager:
    """Get or create Floor Manager instance."""
    global _floor_manager
    if _floor_manager is None:
        _floor_manager = FloorManager()
    return _floor_manager


# Request/Response Models
class TaskRequest(BaseModel):
    """Request to process a task."""
    task: str
    context: Optional[Dict[str, Any]] = None


class DispatchRequest(BaseModel):
    """Request to dispatch a task to a department."""
    department: str
    task: str
    priority: str = "normal"
    context: Optional[Dict[str, Any]] = None


class DelegationRequest(BaseModel):
    """Request to delegate a task from Copilot to the Trading Floor."""
    from_department: str
    task: str
    suggested_department: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class AgentStateResponse(BaseModel):
    """Response with agent state."""
    id: str
    department: str
    status: str
    position: Dict[str, float]
    sub_agents: List[str]


class FloorStateResponse(BaseModel):
    """Response with full floor state."""
    departments: List[Dict[str, Any]]
    agents: List[AgentStateResponse]
    stats: Dict[str, int]


@router.get("/state", response_model=FloorStateResponse)
async def get_floor_state():
    """Get current Trading Floor state."""
    manager = get_floor_manager()

    departments = []
    for dept, config in manager.departments.items():
        departments.append({
            "id": dept.value,
            "name": dept.value.capitalize(),
            "agent_type": config.agent_type,
            "sub_agents": config.sub_agents,
        })

    agents = [
        AgentStateResponse(
            id="floor-manager",
            department="coordination",
            status="idle",
            position={"x": 450, "y": 10},
            sub_agents=[],
        )
    ]

    stats = {
        "total_tasks": 0,
        "active_tasks": 0,
        "completed_tasks": 0,
        "pending_mail": 0,
    }

    return FloorStateResponse(
        departments=departments,
        agents=agents,
        stats=stats,
    )


@router.post("/process")
async def process_task(request: TaskRequest):
    """Process a task through the Floor Manager."""
    manager = get_floor_manager()

    result = manager.process(request.task, request.context)

    return {
        "status": "success",
        "result": result,
    }


@router.post("/dispatch")
async def dispatch_task(request: DispatchRequest):
    """Dispatch a task directly to a department."""
    manager = get_floor_manager()

    try:
        dept = Department(request.department)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department: {request.department}"
        )

    result = manager.dispatch(
        to_dept=dept,
        task=request.task,
        priority=request.priority,
        context=request.context,
    )

    return {
        "status": "success",
        "dispatch": result,
    }


@router.get("/departments")
async def list_departments():
    """List all departments with their configurations."""
    manager = get_floor_manager()

    departments = []
    for dept, config in manager.departments.items():
        departments.append({
            "id": dept.value,
            "name": dept.value.capitalize(),
            "agent_type": config.agent_type,
            "sub_agents": config.sub_agents,
            "memory_namespace": config.memory_namespace,
        })

    return {"departments": departments}


@router.get("/mail/{department}")
async def get_department_mail(department: str):
    """Get mail for a specific department."""
    manager = get_floor_manager()

    # Allow floor_manager as special case (not a Department enum)
    if department == "floor_manager":
        # Return floor_manager's inbox (messages sent TO floor_manager)
        messages = manager.mail_service.check_inbox("floor_manager")
        return {
            "department": department,
            "messages": [
                {
                    "id": msg.id,
                    "from": msg.from_dept,
                    "to": msg.to_dept,
                    "type": msg.type.value,
                    "subject": msg.subject,
                    "body": msg.body,
                    "priority": msg.priority.value,
                    "timestamp": msg.timestamp.isoformat(),
                    "read": msg.read,
                }
                for msg in messages
            ],
        }

    try:
        dept = Department(department)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department: {department}"
        )

    messages = manager.mail_service.check_inbox(dept.value)

    return {
        "department": department,
        "messages": [
            {
                "id": msg.id,
                "from": msg.from_dept,
                "to": msg.to_dept,
                "type": msg.type.value,
                "subject": msg.subject,
                "body": msg.body,
                "priority": msg.priority.value,
                "timestamp": msg.timestamp.isoformat(),
                "read": msg.read,
            }
            for msg in messages
        ],
    }


@router.post("/mail/{message_id}/read")
async def mark_mail_read(message_id: str):
    """Mark a mail message as read."""
    manager = get_floor_manager()

    message = manager.mail_service.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    manager.mail_service.mark_as_read(message_id)

    return {"status": "success", "message_id": message_id}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "trading-floor",
    }


@router.post("/delegate")
async def delegate_task(request: DelegationRequest):
    """
    Delegate a task from Copilot or another agent to the Trading Floor.

    This endpoint allows external agents (like Copilot) to delegate tasks
    to the Trading Floor. The Floor Manager will route the task to the
    appropriate department based on the suggested_department or auto-classify.

    Args:
        request: Delegation request with from_department, task, and optional
                 suggested_department and context

    Returns:
        Delegation result with dispatch information
    """
    manager = get_floor_manager()

    result = manager.handle_dispatch(
        from_department=request.from_department,
        task=request.task,
        suggested_department=request.suggested_department,
        context=request.context,
    )

    return {
        "status": "success",
        "dispatch": result,
    }
