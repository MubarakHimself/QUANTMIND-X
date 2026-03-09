"""
Claude Agent v2 API Endpoints - DEPRECATED

Use /api/floor-manager endpoints instead.
This module used the legacy orchestrator which has been removed.

**Phase 6.1 - Claude Agent Endpoints**
"""

import logging
import warnings
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import asyncio
import json

warnings.warn(
    "claude_agent_endpoints is deprecated. Use /api/floor-manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Stub implementations
def get_orchestrator(*args, **kwargs):
    raise NotImplementedError(
        "Claude Orchestrator is deprecated. Use /api/floor-manager instead."
    )


class ClaudeOrchestrator:
    """Deprecated - use FloorManager instead."""
    pass


def get_agent_config(*args, **kwargs):
    """Deprecated - use floor_manager instead."""
    return None


def get_all_agent_ids():
    """Deprecated - use floor_manager instead."""
    return []


class ClaudeAgentConfig:
    """Deprecated - use department configs instead."""
    pass

logger = logging.getLogger(__name__)

# Create router with v2 prefix
router = APIRouter(prefix="/api/v2/agents", tags=["agents-v2"])


# =============================================================================
# Request/Response Models
# =============================================================================

class RunAgentRequest(BaseModel):
    """Request model for running an agent."""
    messages: List[Dict[str, Any]] = Field(..., description="List of messages for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")


class RunAgentResponse(BaseModel):
    """Response model for agent run."""
    task_id: str
    agent_id: str
    status: str
    poll_url: str
    stream_url: str


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    task_id: str
    agent_id: str
    status: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None


class AgentListResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[str]
    count: int


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """
    List all available agents.
    
    Returns:
        List of agent IDs
    """
    agent_ids = get_all_agent_ids()
    return AgentListResponse(
        agents=agent_ids,
        count=len(agent_ids)
    )


@router.get("/{agent_id}/config")
async def get_agent_configuration(agent_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.
    
    Args:
        agent_id: Agent identifier
        
    Returns:
        Agent configuration
    """
    config = get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    return {
        "success": True,
        "data": {
            "agent_id": config.agent_id,
            "workspace": str(config.workspace),
            "timeout_seconds": config.timeout_seconds,
            "max_retries": config.max_retries,
        }
    }


@router.post("/{agent_id}/run", response_model=RunAgentResponse)
async def run_agent(
    agent_id: str,
    request: RunAgentRequest
) -> RunAgentResponse:
    """
    Run an agent with the given messages.
    
    This endpoint submits a task and returns immediately with URLs for
    polling or streaming the result.
    
    Args:
        agent_id: Agent identifier
        request: Run request with messages and context
        
    Returns:
        Task ID and URLs for result retrieval
    """
    # Validate agent exists
    config = get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    try:
        orchestrator = get_orchestrator()
        
        # Submit task
        task_id = await orchestrator.submit_task(
            agent_id=agent_id,
            messages=request.messages,
            context=request.context,
            session_id=request.session_id
        )
        
        return RunAgentResponse(
            task_id=task_id,
            agent_id=agent_id,
            status="pending",
            poll_url=f"/api/v2/agents/{agent_id}/status/{task_id}",
            stream_url=f"/api/v2/agents/{agent_id}/stream/{task_id}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error running agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/status/{task_id}", response_model=AgentStatusResponse)
async def get_task_status(
    agent_id: str,
    task_id: str
) -> AgentStatusResponse:
    """
    Get the status of a task.
    
    Poll this endpoint to check task progress. Returns the result
    when the task is complete.
    
    Args:
        agent_id: Agent identifier
        task_id: Task identifier
        
    Returns:
        Task status and result (if complete)
    """
    try:
        orchestrator = get_orchestrator()
        
        # Get current status
        status = orchestrator.get_task_status(agent_id, task_id)
        
        # Try to get result
        result = await orchestrator.get_result(agent_id, task_id)
        
        if result:
            return AgentStatusResponse(
                task_id=task_id,
                agent_id=agent_id,
                status=result.get("status", status),
                completed_at=result.get("completed_at"),
                output=result.get("output"),
                error=result.get("error")
            )
        else:
            return AgentStatusResponse(
                task_id=task_id,
                agent_id=agent_id,
                status=status
            )
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting status for {agent_id}/{task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}/tasks/{task_id}")
async def cancel_task(
    agent_id: str,
    task_id: str
) -> Dict[str, Any]:
    """
    Cancel a running task.
    
    Args:
        agent_id: Agent identifier
        task_id: Task identifier
        
    Returns:
        Cancellation status
    """
    try:
        orchestrator = get_orchestrator()
        success = await orchestrator.cancel_task(agent_id, task_id)
        
        if success:
            return {
                "success": True,
                "message": f"Task {task_id} cancelled"
            }
        else:
            return {
                "success": False,
                "message": f"Task {task_id} not running or already complete"
            }
            
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/{agent_id}/stream/{task_id}")
async def stream_task_result(
    websocket: WebSocket,
    agent_id: str,
    task_id: str
):
    """
    Stream task result events via WebSocket.
    
    Emits events:
    - started: Task has begun
    - tool_call: Tool was called
    - progress: Progress update
    - completed: Task finished
    
    Args:
        websocket: WebSocket connection
        agent_id: Agent identifier
        task_id: Task identifier
    """
    await websocket.accept()
    
    try:
        orchestrator = get_orchestrator()
        
        # Stream events
        async for event in orchestrator.stream_result(agent_id, task_id):
            await websocket.send_json(event)
            
            # Break on completion
            if event.get("type") == "completed":
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {agent_id}/{task_id}")
    except ValueError as e:
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
    except Exception as e:
        logger.error(f"WebSocket error for {agent_id}/{task_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
    finally:
        await websocket.close()


# =============================================================================
# Backward Compatibility Helper
# =============================================================================

async def run_agent_sync(
    agent_id: str,
    messages: List[Dict[str, Any]],
    context: Dict[str, Any] = None,
    timeout: float = 300.0
) -> Dict[str, Any]:
    """
    Run an agent synchronously (blocking).
    
    This is used by the v1 API for backward compatibility.
    
    Args:
        agent_id: Agent identifier
        messages: List of messages
        context: Additional context
        timeout: Maximum wait time in seconds
        
    Returns:
        Agent result
    """
    orchestrator = get_orchestrator()
    
    # Submit task
    task_id = await orchestrator.submit_task(
        agent_id=agent_id,
        messages=messages,
        context=context or {}
    )
    
    # Poll until complete or timeout
    start_time = asyncio.get_event_loop().time()
    while True:
        result = await orchestrator.get_result(agent_id, task_id)
        
        if result:
            return result
        
        # Check timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= timeout:
            # Cancel the task
            await orchestrator.cancel_task(agent_id, task_id)
            raise TimeoutError(f"Agent {agent_id} timed out after {timeout}s")
        
        # Wait before next poll
        await asyncio.sleep(0.5)