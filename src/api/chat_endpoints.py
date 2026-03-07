"""
Chat API Endpoints

Provides the backend for the QuantMind Copilot chat interface.
Routes messages to the appropriate agent via ClaudeOrchestrator.
Uses Claude CLI with MCP configurations for agent execution.

Business logic has been extracted to src.api.services.chat_service.

**Validates: Requirements 8.1, 8.4**
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Import service (business logic extracted here)
from src.api.services.chat_service import (
    get_chat_service,
    build_messages,
    invoke_claude_agent,
    invoke_claude_agent_async,
    execute_tool,
    generate_pine_script,
    convert_mql5_to_pine,
    SDK_ORCHESTRATOR_AVAILABLE,
    CLAUDE_ORCHESTRATOR_AVAILABLE,
    VALID_AGENT_TYPES,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# =============================================================================
# Models (kept in endpoints for request/response handling)
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    agent_id: str = "copilot"
    history: List[ChatMessage] = []
    model: str = "claude-3-5-sonnet"
    api_keys: Dict[str, str] = {}
    session_id: Optional[str] = None
    stream: bool = False
    skill_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    agent_id: str
    task_id: Optional[str] = None
    action_taken: Optional[str] = None
    artifacts: List[Dict[str, Any]] = []


class TaskStatusResponse(BaseModel):
    task_id: str
    agent_id: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []


class PineScriptGenerateRequest(BaseModel):
    """Request model for generating Pine Script from natural language."""
    query: str


class PineScriptConvertRequest(BaseModel):
    """Request model for converting MQL5 to Pine Script."""
    mql5_code: str


class PineScriptResponse(BaseModel):
    """Response model for Pine Script endpoints."""
    pine_script: Optional[str] = None
    status: str
    errors: List[str] = []


class ToolExecutionRequest(BaseModel):
    """Request for direct tool execution."""
    tool_name: str
    parameters: Dict[str, Any] = {}
    agent_id: str = "copilot"


class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = []


# =============================================================================
# Endpoints (thin, delegating to service)
# =============================================================================

@router.post("/send", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat messages from the UI.
    Routes to specific agent workflows via Claude Orchestrator.
    """
    logger.info(f"Received chat request for agent: {request.agent_id}")

    # Validate agent type
    if request.agent_id not in VALID_AGENT_TYPES:
        return ChatResponse(
            reply=f"Unknown agent type: {request.agent_id}. Valid types: {', '.join(VALID_AGENT_TYPES)}",
            agent_id="system"
        )

    try:
        # Use the chat service
        chat_service = get_chat_service()

        # Convert history to dict format for service
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]

        result = await chat_service.process_chat(
            message=request.message,
            agent_id=request.agent_id,
            history=history_dicts,
            session_id=request.session_id,
            skill_id=request.skill_id,
        )

        return ChatResponse(
            reply=result["reply"],
            agent_id=result["agent_id"],
            task_id=result.get("task_id"),
            action_taken=result.get("action_taken"),
            artifacts=result.get("artifacts", [])
        )

    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        return ChatResponse(
            reply=f"I encountered an error processing your request: {str(e)}",
            agent_id="system"
        )


async def handle_tool_request_from_chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat messages that look like tool requests.
    Parses the message and executes tools directly.
    """
    chat_service = get_chat_service()

    result = await chat_service.handle_tool_request(request.message, request.agent_id)

    return ChatResponse(
        reply=result["reply"],
        agent_id=result["agent_id"],
        action_taken=result.get("action_taken"),
        artifacts=result.get("artifacts", [])
    )


@router.post("/async", response_model=ChatResponse)
async def chat_async_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Submit a chat task asynchronously and return task ID immediately.
    Use /task/{task_id} to check status and get results.
    """
    logger.info(f"Received async chat request for agent: {request.agent_id}")

    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Claude Orchestrator not available"
        )

    if request.agent_id not in VALID_AGENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {request.agent_id}. Must be one of: {', '.join(VALID_AGENT_TYPES)}"
        )

    try:
        # Convert history to dict format
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # Submit task and get task ID immediately
        task_id = await invoke_claude_agent_async(
            request.agent_id,
            request.message,
            history_dicts,
            request.session_id,
        )

        return ChatResponse(
            reply=f"Task submitted. Use /api/chat/task/{task_id} to check status.",
            agent_id=request.agent_id,
            task_id=task_id,
        )

    except Exception as e:
        logger.error(f"Async task submission error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, agent_id: Optional[str] = None):
    """
    Get status and result of an async task.
    If agent_id is not provided, searches all agents for the task.
    """
    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Claude Orchestrator not available"
        )

    from src.agents.claude_orchestrator import get_orchestrator
    orchestrator = get_orchestrator()

    if agent_id:
        if agent_id not in VALID_AGENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent type: {agent_id}"
            )

        status = orchestrator.get_task_status(agent_id, task_id)
        result = await orchestrator.get_result(agent_id, task_id)

        if result:
            return TaskStatusResponse(
                task_id=task_id,
                agent_id=agent_id,
                status=result.get("status", status),
                output=result.get("output"),
                error=result.get("error"),
                tool_calls=result.get("tool_calls", []),
            )

        return TaskStatusResponse(
            task_id=task_id,
            agent_id=agent_id,
            status=status,
        )

    # Search all agents for the task
    for aid in VALID_AGENT_TYPES:
        status = orchestrator.get_task_status(aid, task_id)
        if status != "unknown":
            result = await orchestrator.get_result(aid, task_id)
            if result:
                return TaskStatusResponse(
                    task_id=task_id,
                    agent_id=aid,
                    status=result.get("status", status),
                    output=result.get("output"),
                    error=result.get("error"),
                    tool_calls=result.get("tool_calls", []),
                )
            return TaskStatusResponse(
                task_id=task_id,
                agent_id=aid,
                status=status,
            )

    raise HTTPException(
        status_code=404,
        detail=f"Task {task_id} not found"
    )


@router.delete("/task/{task_id}")
async def cancel_task(task_id: str, agent_id: str):
    """Cancel a running task."""
    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Claude Orchestrator not available"
        )

    from src.agents.claude_orchestrator import get_orchestrator
    orchestrator = get_orchestrator()

    success = await orchestrator.cancel_task(agent_id, task_id)

    if success:
        return {"status": "cancelled", "task_id": task_id}
    else:
        raise HTTPException(
            status_code=400,
            detail="Task could not be cancelled (may already be complete)"
        )


@router.post("/{agent_type}/invoke", response_model=ChatResponse)
async def invoke_agent_direct(
    agent_type: str,
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    Directly invoke a specific agent type.

    Args:
        agent_type: One of 'copilot', 'analyst', 'quantcode', 'pinescript', 'router', 'executor'
        request: Chat request with message and optional history
    """
    if agent_type not in VALID_AGENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {agent_type}. Must be one of: {', '.join(VALID_AGENT_TYPES)}"
        )

    try:
        # Convert history to dict format
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]

        if CLAUDE_ORCHESTRATOR_AVAILABLE:
            result = await invoke_claude_agent(
                agent_type,
                request.message,
                history_dicts,
                request.session_id,
            )
            return ChatResponse(
                reply=result["reply"],
                agent_id=agent_type,
                task_id=result.get("task_id"),
                artifacts=result.get("artifacts", [])
            )

        # Legacy fallback
        from src.agents.copilot import run_copilot_workflow
        from src.agents.analyst import run_analyst_workflow

        if agent_type == "copilot":
            result = run_copilot_workflow(request.message)
            return ChatResponse(
                reply=str(result.get("messages", ["Completed"])[-1]),
                agent_id=agent_type
            )
        elif agent_type == "analyst":
            result = run_analyst_workflow(request.message)
            return ChatResponse(
                reply=result.get("synthesis_result", "Analysis complete"),
                agent_id=agent_type
            )

        return ChatResponse(
            reply=f"Agent {agent_type} processing: {request.message}",
            agent_id=agent_type
        )

    except Exception as e:
        logger.error(f"Agent invocation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
async def list_available_agents():
    """List all available agent types."""
    chat_service = get_chat_service()
    info = chat_service.get_available_agents()
    return info


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Stream chat response using SDK streaming.
    Returns Server-Sent Events (SSE) with real-time updates.
    """
    if not SDK_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SDK Orchestrator not available for streaming"
        )

    from src.agents.sdk_orchestrator import get_sdk_orchestrator
    orchestrator = get_sdk_orchestrator()

    # Convert history to dict format
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    messages = build_messages(request.message, history_dicts)

    async def event_generator():
        try:
            async for event in orchestrator.stream(
                agent_id=request.agent_id,
                messages=messages,
                session_id=request.session_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool_directly(request: ToolExecutionRequest):
    """Execute a tool directly without spawning Claude CLI."""
    result = await execute_tool(
        request.tool_name,
        request.parameters,
        request.agent_id
    )

    return ToolExecutionResponse(
        success=result.success,
        result=result.result,
        error=result.error,
        artifacts=result.artifacts
    )


@router.post("/pinescript", response_model=PineScriptResponse)
async def generate_pine_script_endpoint(request: PineScriptGenerateRequest):
    """Generate Pine Script v5 code from natural language query."""
    result = await generate_pine_script(request.query)

    return PineScriptResponse(
        pine_script=result.get("pine_script"),
        status=result.get("status", "complete"),
        errors=result.get("errors", [])
    )


@router.post("/pinescript/convert", response_model=PineScriptResponse)
async def convert_mql5_to_pinescript_endpoint(request: PineScriptConvertRequest):
    """Convert MQL5 code to Pine Script v5."""
    result = await convert_mql5_to_pine(request.mql5_code)

    return PineScriptResponse(
        pine_script=result.get("pine_script"),
        status=result.get("status", "complete"),
        errors=result.get("errors", [])
    )
