"""
Floor Manager API Endpoints

Provides REST and WebSocket endpoints for the QuantMind Floor Manager.
The Floor Manager is the main copilot interface that routes tasks to departments.

Endpoints:
- POST /api/floor-manager/task - Submit a task for routing
- GET /api/floor-manager/status - Get floor manager status
- POST /api/floor-manager/chat - Chat with floor manager (main copilot)
- GET /api/floor-manager/departments - List all departments with status
- POST /api/floor-manager/delegate - Delegate task to specific department
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import logging
import json

from src.agents.departments.floor_manager import get_floor_manager, FloorManager
from src.agents.departments.types import Department, get_personality, get_all_personalities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/floor-manager", tags=["floor-manager"])


# ========== Request/Response Models ==========

class TaskRequest(BaseModel):
    """Request to submit a task for routing."""
    task: str = Field(..., description="Task description to route")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")


class ChatRequest(BaseModel):
    """Request to chat with the floor manager."""
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Conversation history for context")
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    status: str
    content: Optional[str] = None
    delegation: Optional[Dict[str, Any]] = None


class CommandRequest(BaseModel):
    """Request to handle a natural language command with context."""
    message: str = Field(..., description="User command message")
    canvas_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Canvas context with canvas type, session_id, entity"
    )
    confirmed: bool = Field(
        default=False,
        description="Whether user has confirmed the action"
    )
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    error: Optional[str] = None


class DelegateRequest(BaseModel):
    """Request to delegate a task to a specific department."""
    department: str = Field(..., description="Target department (development, research, risk, trading, portfolio)")
    task: str = Field(..., description="Task description")
    priority: str = Field(default="normal", description="Task priority (low, normal, high, urgent)")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")
    spawn_worker: bool = Field(default=False, description="Whether to spawn a worker for this task")


class DepartmentResponse(BaseModel):
    """Response with department info."""
    id: str
    name: str
    agent_type: str
    sub_agents: List[str]
    memory_namespace: str
    model_tier: str
    max_workers: int
    pending_tasks: int
    status: str


class FloorManagerStatusResponse(BaseModel):
    """Response with floor manager status."""
    status: str
    model_tier: str
    departments: Dict[str, Dict[str, Any]]
    stats: Dict[str, int]
    timestamp: str


# ========== Helper Functions ==========

def get_manager() -> FloorManager:
    """Get the Floor Manager instance."""
    return get_floor_manager()


def validate_department(department: str) -> Department:
    """Validate and convert department string to enum."""
    try:
        return Department(department.lower())
    except ValueError:
        valid = [d.value for d in Department]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. Valid options: {valid}"
        )


# ========== API Endpoints ==========

@router.post("/task")
async def submit_task(request: TaskRequest):
    """
    Submit a task for routing.

    The Floor Manager will:
    1. Classify the task to determine the appropriate department
    2. Dispatch the task to that department via mail

    Returns the classification result and dispatch status.
    """
    manager = get_manager()

    try:
        result = manager.process(request.task, request.context)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=FloorManagerStatusResponse)
async def get_status():
    """
    Get Floor Manager status.

    Returns current status, department configurations, and statistics.
    """
    manager = get_manager()
    return manager.get_status()


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat with the Floor Manager (main copilot).

    This is the primary interface for interacting with the QuantMind system.
    The Floor Manager can:
    - Answer questions about trading operations
    - Route tasks to appropriate departments
    - Provide status updates

    For streaming responses, set stream: true in the request body.
    """
    manager = get_manager()

    # If streaming requested, use SSE
    if request.stream:
        async def event_generator():
            try:
                async for event in manager.chat_stream(
                    message=request.message,
                    context=request.context,
                    history=request.history,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"Streaming chat failed: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming response
    try:
        result = await manager.chat(
            message=request.message,
            context=request.context,
            history=request.history,
            stream=request.stream,
        )
        return result
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return ChatResponse(
            status="error",
            error=str(e),
        )


@router.post("/command")
async def handle_command(request: CommandRequest):
    """
    Handle a natural language command with canvas context awareness.

    This endpoint:
    - Classifies the user message into an intent
    - Requires confirmation for destructive commands (pause, close, stop)
    - Asks for clarification when confidence is low
    - Executes the command via appropriate APIs

    Args:
        request: CommandRequest with message, canvas_context, and confirmed flag

    Returns:
        Dict with response type (confirmation_needed, clarification_needed, success, error)
    """
    manager = get_manager()

    try:
        result = await manager.handle_command(
            message=request.message,
            canvas_context=request.canvas_context or {},
            confirmed=request.confirmed,
        )
        return result
    except Exception as e:
        logger.error(f"Command handling failed: {e}")
        return {
            "type": "error",
            "message": str(e),
        }


@router.get("/departments")
async def list_departments():
    """
    List all departments with their status.

    Returns department configurations, available workers, and pending task counts.
    """
    manager = get_manager()
    departments = manager.get_departments()

    return {
        "departments": departments,
        "total": len(departments),
    }


@router.get("/departments/{department}/personality")
async def get_department_personality(department: str):
    """
    Get personality for a specific department.

    Returns the department's persona name, traits, communication style,
    strengths, weaknesses, color, and icon.
    """
    try:
        dept = Department(department.lower())
    except ValueError:
        valid = [d.value for d in Department]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{department}'. Valid options: {valid}"
        )

    personality = get_personality(dept)
    if personality is None:
        raise HTTPException(
            status_code=404,
            detail=f"No personality found for department '{department}'"
        )

    return {
        "department": department,
        "name": personality.name,
        "tagline": personality.tagline,
        "traits": personality.traits,
        "communication_style": personality.communication_style,
        "strengths": personality.strengths,
        "weaknesses": personality.weaknesses,
        "color": personality.color,
        "icon": personality.icon,
    }


@router.get("/departments/personality")
async def list_department_personalities():
    """
    List all department personalities.

    Returns personality information for all 5 quant departments.
    """
    personalities = get_all_personalities()

    return {
        "personalities": [
            {
                "department": dept_key,
                "name": p.name,
                "tagline": p.tagline,
                "traits": p.traits,
                "communication_style": p.communication_style,
                "strengths": p.strengths,
                "weaknesses": p.weaknesses,
                "color": p.color,
                "icon": p.icon,
            }
            for dept_key, p in personalities.items()
        ],
        "total": len(personalities),
    }


@router.post("/delegate")
async def delegate_task(request: DelegateRequest):
    """
    Delegate a task to a specific department.

    This explicitly routes a task to a department rather than relying on
    automatic classification. Optionally spawns a worker to handle the task.

    Valid departments: development, research, risk, trading, portfolio
    """
    manager = get_manager()

    # Validate department
    validate_department(request.department)

    try:
        result = manager.delegate(
            department=request.department,
            task=request.task,
            priority=request.priority,
            context=request.context,
            spawn_worker=request.spawn_worker,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("error"))

        return {
            "status": "success",
            "delegation": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delegation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Concurrent Task Routing (Story 7.7) ==========

class ConcurrentTaskRequest(BaseModel):
    """Request to dispatch multiple tasks concurrently."""
    tasks: List[Dict[str, Any]] = Field(
        ...,
        description="List of tasks with keys: task_type, department, payload, priority"
    )
    session_id: Optional[str] = Field(default=None, description="Session ID for isolation")


@router.post("/concurrent")
async def dispatch_concurrent_tasks(request: ConcurrentTaskRequest):
    """
    Dispatch multiple tasks concurrently to different departments (AC-1).

    Routes tasks to appropriate departments in parallel via Redis Streams.
    Returns task IDs and initial status for tracking.
    """
    try:
        manager = get_manager()

        result = manager.dispatch_concurrent(
            tasks=request.tasks,
            session_id=request.session_id,
        )

        return {
            "status": "dispatched",
            "tasks": result,
            "session_id": request.session_id or result[0].get("session_id") if result else None,
        }
    except Exception as e:
        logger.error(f"Concurrent dispatch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/status/{session_id}")
async def get_task_status(session_id: str):
    """
    Get status of all tasks for a session (AC-1).

    Returns department-level status for Agent Panel display:
    "Research: running | Development: queued | Risk: running..."
    """
    try:
        manager = get_manager()

        status = manager.get_concurrent_task_status_display(session_id)

        return {
            "session_id": session_id,
            "departments": status,
        }
    except Exception as e:
        logger.error(f"Get task status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/concurrent/execute")
async def execute_concurrent_tasks(
    session_id: str,
    max_concurrent: int = 5,
):
    """
    Execute all pending tasks for a session concurrently (AC-2).

    Returns aggregated results with parallelism metrics.
    """
    try:
        manager = get_manager()

        result = await manager.execute_concurrent(
            session_id=session_id,
            max_concurrent=max_concurrent,
        )

        return result
    except Exception as e:
        logger.error(f"Concurrent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Story 7.5: Session Workspace Isolation ==========


class SessionCommitRequest(BaseModel):
    """Request to commit a session's draft nodes (Story 7.5)."""
    session_id: str = Field(..., description="Session ID to commit")
    department: Optional[str] = Field(default=None, description="Optional department filter")
    entity_id: Optional[str] = Field(default=None, description="Entity/strategy ID for conflict detection")


class SessionCommitResponse(BaseModel):
    """Response from session commit endpoint."""
    status: str  # 'success' or 'pending_review' or 'error'
    session_id: str
    node_count: Optional[int] = None
    committed_at_utc: Optional[str] = None
    department: Optional[str] = None
    entity_id: Optional[str] = None
    conflict_count: Optional[int] = None
    conflicts: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
    error: Optional[str] = None


@router.post("/session/commit", response_model=SessionCommitResponse)
async def commit_session(request: SessionCommitRequest):
    """
    Commit all draft nodes for a session (AC #2).

    Commits draft nodes making them visible to all subsequent sessions.
    If entity_id provided and conflicts detected (AC #3), returns 'pending_review'
    instead of immediately committing.
    """
    try:
        manager = get_manager()

        result = manager.commit_session(
            session_id=request.session_id,
            department=request.department,
            entity_id=request.entity_id,
        )

        return SessionCommitResponse(**result)
    except Exception as e:
        logger.error(f"Session commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/departments/{department}")
async def get_department(department: str):
    """
    Get details for a specific department.

    Includes department configuration, available workers, and pending tasks.
    """
    manager = get_manager()

    # Validate department
    dept = validate_department(department)

    departments = manager.get_departments()
    dept_info = next((d for d in departments if d["id"] == dept.value), None)

    if not dept_info:
        raise HTTPException(status_code=404, detail=f"Department not found: {department}")

    # Get pending mail for this department
    mail = manager.mail_service.check_inbox(dept.value, unread_only=True, limit=50)

    return {
        "department": dept_info,
        "pending_mail": [
            {
                "id": msg.id,
                "from": msg.from_dept,
                "subject": msg.subject,
                "priority": msg.priority.value,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in mail
        ],
    }


@router.post("/classify")
async def classify_task(request: TaskRequest):
    """
    Classify a task to see which department it would be routed to.

    This is a dry-run that only returns the classification without dispatching.
    """
    manager = get_manager()

    dept = manager.classify_task(request.task)

    return {
        "task": request.task,
        "classified_department": dept.value,
        "department_info": {
            "agent_type": manager.departments[dept].agent_type,
            "sub_agents": manager.departments[dept].sub_agents,
        },
    }


@router.post("/spawn")
async def spawn_worker(
    department: str,
    worker_type: str,
    task: str,
    input_data: Optional[Dict[str, Any]] = None,
):
    """
    Spawn a worker agent for a specific department.

    This creates a new worker to handle a specific task within a department.
    The worker type must be valid for the specified department.
    """
    manager = get_manager()

    # Validate department
    dept = validate_department(department)

    result = manager.spawn_worker(
        department=dept,
        worker_type=worker_type,
        task=task,
        input_data=input_data,
    )

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("error"))

    return {
        "status": "success",
        "spawn": result,
    }


@router.post("/conversation/clear")
async def clear_conversation():
    """Clear the conversation history with the Floor Manager."""
    manager = get_manager()
    manager.clear_conversation()
    return {"status": "success", "message": "Conversation history cleared"}


@router.get("/mail/{department}")
async def get_department_mail(department: str, unread_only: bool = True, limit: int = 50):
    """
    Get mail for a specific department.

    Returns messages sent to the department's inbox.
    """
    manager = get_manager()

    # Validate department
    validate_department(department)

    messages = manager.mail_service.check_inbox(
        dept=department,
        unread_only=unread_only,
        limit=limit,
    )

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
        "count": len(messages),
    }


@router.post("/mail/{message_id}/read")
async def mark_mail_read(message_id: str):
    """Mark a mail message as read."""
    manager = get_manager()

    message = manager.mail_service.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    manager.mail_service.mark_read(message_id)

    return {"status": "success", "message_id": message_id}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    manager = get_manager()
    return {
        "status": "healthy",
        "service": "floor-manager",
        "model_tier": manager.model_tier,
        "departments": len(manager.departments),
    }


# ========== WebSocket Endpoint for Streaming Chat ==========

@router.websocket("/ws")
async def floor_manager_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat with the Floor Manager.

    Message format (send):
    {
        "type": "chat",
        "message": "Your message here",
        "context": {}  // optional
    }

    Message format (receive):
    {
        "type": "started" | "text" | "completed" | "error",
        "delta": "...",  // for text events
        "output": "...",  // for completed events
        "error": "..."  // for error events
    }
    """
    await websocket.accept()
    manager = get_manager()

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON message",
                })
                continue

            msg_type = message.get("type")

            if msg_type == "chat":
                # Handle chat message
                user_message = message.get("message", "")
                context = message.get("context")

                if not user_message:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Message is required",
                    })
                    continue

                # Stream response
                try:
                    async for event in manager.chat_stream(
                        message=user_message,
                        context=context,
                    ):
                        await websocket.send_json(event)
                except Exception as e:
                    logger.error(f"Streaming chat failed: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })

            elif msg_type == "clear":
                # Clear conversation
                manager.clear_conversation()
                await websocket.send_json({
                    "type": "system",
                    "message": "Conversation cleared",
                })

            elif msg_type == "ping":
                # Ping/pong for connection keepalive
                await websocket.send_json({
                    "type": "pong",
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        logger.info("Floor Manager WebSocket disconnected")
    except Exception as e:
        logger.error(f"Floor Manager WebSocket error: {e}")
