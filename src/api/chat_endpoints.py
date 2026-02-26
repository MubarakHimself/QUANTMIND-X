"""
Chat API Endpoints

Provides the backend for the QuantMind Copilot chat interface.
Routes messages to the appropriate agent via ClaudeOrchestrator.
Uses Claude CLI with MCP configurations for agent execution.

**Validates: Requirements 8.1, 8.4**
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Pine Script agent imports (keep for backward compatibility)
try:
    from src.agents.pinescript import (
        generate_pine_script_from_query,
        convert_mql5_to_pinescript
    )
    PINESCRIPT_AGENT_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pine Script agent not available: {e}")
    PINESCRIPT_AGENT_AVAILABLE = False

# Claude Orchestrator imports
try:
    from src.agents.claude_orchestrator import get_orchestrator, ClaudeOrchestrator
    from src.agents.claude_config import get_agent_config, get_all_agent_ids
    CLAUDE_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Claude Orchestrator not available: {e}")
    CLAUDE_ORCHESTRATOR_AVAILABLE = False

# Legacy Agent Workflows (fallback)
try:
    from src.agents.copilot import run_copilot_workflow
    from src.agents.analyst import run_analyst_workflow
    LEGACY_AGENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Legacy agents not available: {e}")
    LEGACY_AGENTS_AVAILABLE = False

# SDK Orchestrator (preferred for Claude Code SDK integration)
try:
    from src.agents.sdk_orchestrator import get_sdk_orchestrator
    SDK_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SDK Orchestrator not available: {e}")
    SDK_ORCHESTRATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# =============================================================================
# Configuration Flags
# =============================================================================

# Use Claude Orchestrator by default
USE_CLAUDE_ORCHESTRATOR = os.getenv("QUANTMIND_USE_CLAUDE", "true").lower() == "true"

# Use SDK Orchestrator first (falls back to CLI if unavailable)
USE_SDK_ORCHESTRATOR = os.getenv("USE_SDK_ORCHESTRATOR", "true").lower() == "true"

# Supported agent types
VALID_AGENT_TYPES = ["copilot", "analyst", "quantcode", "pinescript", "router", "executor"]


# =============================================================================
# Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    agent_id: str = "copilot"
    history: List[ChatMessage] = []
    model: str = "claude-3-5-sonnet"  # Default to Claude model
    api_keys: Dict[str, str] = {}
    session_id: Optional[str] = None  # For conversation continuity
    stream: bool = False  # Enable streaming response


class ChatResponse(BaseModel):
    reply: str
    agent_id: str
    task_id: Optional[str] = None  # Task ID for async tracking
    action_taken: Optional[str] = None
    artifacts: List[Dict[str, Any]] = []


class TaskStatusResponse(BaseModel):
    task_id: str
    agent_id: str
    status: str  # pending, running, completed, failed, cancelled
    output: Optional[str] = None
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []


# =============================================================================
# Pine Script Request Models
# =============================================================================

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


# =============================================================================
# Helper Functions
# =============================================================================

def _build_messages(message: str, history: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Build messages list from current message and history.

    Args:
        message: Current user message
        history: Conversation history

    Returns:
        List of message dicts with role and content
    """
    messages = []

    # Add history if provided
    if history:
        for msg in history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

    # Add current message
    messages.append({
        "role": "user",
        "content": message
    })

    return messages


def _extract_action_from_message(message: str) -> Optional[str]:
    """
    Extract action type from message content.

    Args:
        message: User message

    Returns:
        Action type or None
    """
    msg_lower = message.lower()
    if msg_lower.startswith("/deploy"):
        return "deployment"
    elif msg_lower.startswith("/analyze"):
        return "analysis"
    elif msg_lower.startswith("/code") or msg_lower.startswith("/generate"):
        return "code_generation"
    return None


async def _invoke_claude_agent(
    agent_type: str,
    message: str,
    history: List[ChatMessage] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke an agent via SDK Orchestrator (preferred) or CLI Orchestrator (fallback).

    Args:
        agent_type: One of 'copilot', 'analyst', 'quantcode', etc.
        message: User message to process
        history: Optional conversation history
        session_id: Optional session ID for continuity

    Returns:
        Dict with reply, task_id, and any artifacts
    """
    # Build messages
    messages = _build_messages(message, history or [])

    # Try SDK Orchestrator first (faster, no CLI subprocess)
    if USE_SDK_ORCHESTRATOR and SDK_ORCHESTRATOR_AVAILABLE:
        try:
            logger.info(f"Trying SDK orchestrator for {agent_type}")
            orchestrator = get_sdk_orchestrator()
            logger.info(f"SDK orchestrator provider: {orchestrator._provider_config['provider']}")
            result = await orchestrator.invoke(
                agent_id=agent_type,
                messages=messages,
                session_id=session_id,
            )
            logger.info(f"SDK orchestrator result: {result.get('status', 'unknown')}")
            return {
                "reply": result.get("output", result.get("reply", "No response generated")),
                "task_id": result.get("task_id"),
                "agent_id": agent_type,
                "artifacts": result.get("artifacts", []),
                "tool_calls": result.get("tool_calls", []),
            }
        except Exception as e:
            logger.warning(f"SDK orchestrator failed, falling back to CLI: {e}", exc_info=True)

    # Fall back to CLI Orchestrator
    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise RuntimeError("Neither SDK nor CLI Orchestrator available")

    orchestrator = get_orchestrator()

    # Submit task to orchestrator
    task_id = await orchestrator.submit_task(
        agent_id=agent_type,
        messages=messages,
        session_id=session_id,
    )

    # Poll for result with timeout
    max_wait = 300  # 5 minutes max
    poll_interval = 0.5
    elapsed = 0

    while elapsed < max_wait:
        result = await orchestrator.get_result(agent_type, task_id)

        if result is not None:
            # Task completed
            if result.get("status") == "completed":
                return {
                    "reply": result.get("output", "No response generated"),
                    "task_id": task_id,
                    "agent_id": agent_type,
                    "artifacts": result.get("artifacts", []),
                    "tool_calls": result.get("tool_calls", []),
                }
            elif result.get("status") == "failed":
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"Agent failed: {error_msg}")
            elif result.get("status") == "cancelled":
                raise RuntimeError("Task was cancelled")

        # Wait before next poll
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout - return task ID for async tracking
    return {
        "reply": f"Task {task_id} is still processing. Check status with /api/chat/task/{task_id}",
        "task_id": task_id,
        "agent_id": agent_type,
        "artifacts": [],
    }


async def _invoke_claude_agent_async(
    agent_type: str,
    message: str,
    history: List[ChatMessage] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Submit a task asynchronously and return task ID immediately.

    Args:
        agent_type: Agent identifier
        message: User message
        history: Conversation history
        session_id: Session ID

    Returns:
        task_id for tracking
    """
    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise RuntimeError("Claude Orchestrator not available")

    orchestrator = get_orchestrator()
    messages = _build_messages(message, history or [])

    task_id = await orchestrator.submit_task(
        agent_id=agent_type,
        messages=messages,
        session_id=session_id,
    )

    return task_id


# =============================================================================
# Legacy Tools (for fallback)
# =============================================================================

def tool_deploy_strategy(strategy_name: str):
    """Deploys a trading strategy using the Copilot workflow."""
    logger.info(f"Tool invoked: deploy_strategy({strategy_name})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_copilot_workflow(f"Deploy strategy {strategy_name}")
        return f"Deployment initiated. Status: {result.get('monitoring_data', {}).get('status', 'Unknown')}"
    else:
        return f"Deployment simulation for strategy: {strategy_name}"


def tool_analyze_market(query: str):
    """Analyzes market conditions using the Analyst workflow."""
    logger.info(f"Tool invoked: analyze_market({query})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_analyst_workflow(query)
        return result.get("synthesis_result", "Analysis failed to produce result.")
    else:
        return f"Market analysis simulation for: {query}"


# =============================================================================
# Endpoints
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
        # Check for Claude Orchestrator availability
        if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
            # Extract action type
            action = _extract_action_from_message(request.message)

            # Determine target agent based on message or explicit request
            target_agent = request.agent_id
            if request.message.startswith("/deploy"):
                target_agent = "copilot"
            elif request.message.startswith("/analyze"):
                target_agent = "analyst"
            elif request.message.startswith("/code") or request.message.startswith("/generate"):
                target_agent = "quantcode"

            # Invoke via Claude Orchestrator
            result = await _invoke_claude_agent(
                target_agent,
                request.message,
                request.history,
                request.session_id,
            )

            return ChatResponse(
                reply=result["reply"],
                agent_id=result["agent_id"],
                task_id=result.get("task_id"),
                action_taken=action,
                artifacts=result.get("artifacts", [])
            )

        # Fallback to legacy agents
        if LEGACY_AGENTS_AVAILABLE:
            if request.message.startswith("/deploy"):
                reply = tool_deploy_strategy(request.message.replace("/deploy", "").strip())
                return ChatResponse(reply=reply, agent_id="copilot", action_taken="deployment")

            if request.message.startswith("/analyze"):
                reply = tool_analyze_market(request.message.replace("/analyze", "").strip())
                return ChatResponse(reply=reply, agent_id="analyst", action_taken="analysis")

            # Default to copilot for general queries
            result = run_copilot_workflow(request.message)
            return ChatResponse(
                reply=str(result.get("messages", ["Completed"])[-1]),
                agent_id=request.agent_id
            )

        # No orchestrator or legacy available - mock response
        return mock_chat_response(request)

    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        # Check if it's a nested session error - provide helpful QuantMind-aware response
        if "nested session" in str(e).lower() or "claude code cannot be launched" in str(e).lower():
            return quantmind_aware_response(request)
        return ChatResponse(
            reply=f"I encountered an error processing your request: {str(e)}",
            agent_id="system"
        )


async def handle_tool_request_from_chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat messages that look like tool requests.
    Parses the message and executes tools directly.
    """
    import re
    from pathlib import Path

    message = request.message.lower()

    # Detect file write requests
    write_pattern = r"(?:write|create|save)\s+(?:a\s+)?(?:file|ea|strategy)\s+(?:to\s+)?[`'\"]?([^\s`'\"]+)[`'\"]?(?:\s+with\s+(?:content|text)?\s*[:=]?\s*(.+))?"
    match = re.search(write_pattern, request.message, re.IGNORECASE | re.DOTALL)

    if match:
        file_path = match.group(1)
        content = match.group(2) or f"Generated by {request.agent_id}\nTimestamp: {datetime.utcnow().isoformat()}"

        # Execute write_file tool
        tool_request = ToolExecutionRequest(
            tool_name="write_file",
            parameters={"path": file_path, "content": content.strip()},
            agent_id=request.agent_id
        )
        result = await execute_tool_directly(tool_request)

        if result.success:
            return ChatResponse(
                reply=f"I've written the file to {file_path}. Size: {len(content)} bytes.",
                agent_id=request.agent_id,
                action_taken="file_write",
                artifacts=result.artifacts
            )
        else:
            return ChatResponse(
                reply=f"Failed to write file: {result.error}",
                agent_id="system"
            )

    # Detect directory listing requests
    if "list" in message and ("directory" in message or "folder" in message or "files" in message):
        # Extract path if mentioned
        path_match = re.search(r"[`'\"]?(/[^\s`'\"]+)[`'\"]?", request.message)
        path = path_match.group(1) if path_match else "/home/mubarkahimself/Desktop/QUANTMINDX"

        tool_request = ToolExecutionRequest(
            tool_name="list_directory",
            parameters={"path": path},
            agent_id=request.agent_id
        )
        result = await execute_tool_directly(tool_request)

        if result.success:
            return ChatResponse(
                reply=f"Directory listing for {path}:\n{result.result}",
                agent_id=request.agent_id,
                action_taken="list_directory"
            )
        else:
            return ChatResponse(
                reply=f"Failed to list directory: {result.error}",
                agent_id="system"
            )

    # Detect read file requests (handles "read file", "read the file", "show me file", etc.)
    read_pattern = r"(?:read|show|display|cat)\s+(?:the\s+)?(?:file\s+)?[`'\"]?(/[^\s`'\"]+)[`'\"]?"
    match = re.search(read_pattern, request.message, re.IGNORECASE)

    if match:
        file_path = match.group(1)

        tool_request = ToolExecutionRequest(
            tool_name="read_file",
            parameters={"path": file_path},
            agent_id=request.agent_id
        )
        result = await execute_tool_directly(tool_request)

        if result.success:
            return ChatResponse(
                reply=f"Contents of {file_path}:\n```\n{result.result}\n```",
                agent_id=request.agent_id,
                action_taken="read_file"
            )
        else:
            return ChatResponse(
                reply=f"Failed to read file: {result.error}",
                agent_id="system"
            )

    # Default: return helpful message
    return ChatResponse(
        reply=f"I understand you want to: {request.message[:100]}...\n\n"
              f"I can help with:\n"
              f"- Write a file to [path] with content [text]\n"
              f"- Read file [path]\n"
              f"- List directory [path]\n"
              f"- Run command [command] (restricted)\n\n"
              f"Try: 'Write a file to /path/to/file.txt with content: Hello World'",
        agent_id=request.agent_id
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

    # Validate agent type
    if request.agent_id not in VALID_AGENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {request.agent_id}. Must be one of: {', '.join(VALID_AGENT_TYPES)}"
        )

    try:
        # Submit task and get task ID immediately
        task_id = await _invoke_claude_agent_async(
            request.agent_id,
            request.message,
            request.history,
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

    orchestrator = get_orchestrator()

    # If agent_id provided, check that specific agent
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
    """
    Cancel a running task.
    """
    if not CLAUDE_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Claude Orchestrator not available"
        )

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
        if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
            result = await _invoke_claude_agent(
                agent_type,
                request.message,
                request.history,
                request.session_id,
            )
            return ChatResponse(
                reply=result["reply"],
                agent_id=agent_type,
                task_id=result.get("task_id"),
                artifacts=result.get("artifacts", [])
            )

        # Legacy fallback
        if LEGACY_AGENTS_AVAILABLE:
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
    return {
        "agents": VALID_AGENT_TYPES,
        "orchestrator_available": CLAUDE_ORCHESTRATOR_AVAILABLE,
        "sdk_orchestrator_available": SDK_ORCHESTRATOR_AVAILABLE,
        "legacy_available": LEGACY_AGENTS_AVAILABLE,
    }


@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Stream chat response using SDK streaming.

    Returns Server-Sent Events (SSE) with real-time updates.
    Falls back to CLI orchestrator if SDK is unavailable.
    """
    if not SDK_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SDK Orchestrator not available for streaming"
        )

    orchestrator = get_sdk_orchestrator()
    messages = _build_messages(request.message, request.history)

    async def event_generator():
        try:
            async for event in orchestrator.stream(
                agent_id=request.agent_id,
                messages=messages,
                session_id=request.session_id,
            ):
                # Format as SSE
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# =============================================================================
# Direct Tool Execution (bypasses Claude CLI for nested sessions)
# =============================================================================

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


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool_directly(request: ToolExecutionRequest):
    """
    Execute a tool directly without spawning Claude CLI.
    This is useful when running inside Claude Code (nested session).
    """
    import os
    from pathlib import Path
    import json
    from datetime import datetime

    tool_name = request.tool_name
    params = request.parameters
    base_dir = Path(os.getenv("WORKSPACES_DIR", "/home/mubarkahimself/Desktop/QUANTMINDX"))

    try:
        # File system tools
        if tool_name == "write_file":
            file_path = Path(params.get("path", ""))
            content = params.get("content", "")

            # Security: ensure path is within allowed directories
            if not str(file_path.absolute()).startswith(str(base_dir.absolute())):
                # Also allow direct project access
                project_dir = Path("/home/mubarkahimself/Desktop/QUANTMINDX")
                if not str(file_path.absolute()).startswith(str(project_dir.absolute())):
                    return ToolExecutionResponse(
                        success=False,
                        error=f"Access denied: path must be within {base_dir} or {project_dir}"
                    )

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(file_path, "w") as f:
                f.write(content)

            return ToolExecutionResponse(
                success=True,
                result=f"File written successfully: {file_path}",
                artifacts=[{"type": "file", "path": str(file_path), "size": len(content)}]
            )

        elif tool_name == "read_file":
            file_path = Path(params.get("path", ""))

            if not file_path.exists():
                return ToolExecutionResponse(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            with open(file_path, "r") as f:
                content = f.read()

            return ToolExecutionResponse(
                success=True,
                result=content,
                artifacts=[{"type": "file_content", "path": str(file_path)}]
            )

        elif tool_name == "list_directory":
            dir_path = Path(params.get("path", str(base_dir)))

            if not dir_path.exists():
                return ToolExecutionResponse(
                    success=False,
                    error=f"Directory not found: {dir_path}"
                )

            items = []
            for item in dir_path.iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })

            return ToolExecutionResponse(
                success=True,
                result=json.dumps(items, indent=2),
                artifacts=[{"type": "directory_listing", "path": str(dir_path), "count": len(items)}]
            )

        elif tool_name == "delete_file":
            file_path = Path(params.get("path", ""))

            if not file_path.exists():
                return ToolExecutionResponse(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            file_path.unlink()

            return ToolExecutionResponse(
                success=True,
                result=f"File deleted: {file_path}"
            )

        elif tool_name == "create_directory":
            dir_path = Path(params.get("path", ""))
            dir_path.mkdir(parents=True, exist_ok=True)

            return ToolExecutionResponse(
                success=True,
                result=f"Directory created: {dir_path}"
            )

        # Shell command execution (restricted)
        elif tool_name == "run_command":
            import subprocess

            command = params.get("command", "")
            cwd = params.get("cwd", str(base_dir))

            # Security: block dangerous commands
            blocked = ["rm -rf", "sudo", "chmod 777", "> /dev/", "mkfs", "dd if="]
            if any(b in command for b in blocked):
                return ToolExecutionResponse(
                    success=False,
                    error=f"Command blocked for security: contains dangerous pattern"
                )

            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )

            return ToolExecutionResponse(
                success=result.returncode == 0,
                result=result.stdout or result.stderr,
                error=result.stderr if result.returncode != 0 else None
            )

        else:
            return ToolExecutionResponse(
                success=False,
                error=f"Unknown tool: {tool_name}. Available: write_file, read_file, list_directory, delete_file, create_directory, run_command"
            )

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return ToolExecutionResponse(
            success=False,
            error=str(e)
        )


# =============================================================================
# Pine Script Endpoints (keep for backward compatibility)
# =============================================================================

@router.post("/pinescript", response_model=PineScriptResponse)
async def generate_pine_script_endpoint(request: PineScriptGenerateRequest):
    """
    Generate Pine Script v5 code from natural language query.

    Args:
        request: PineScriptGenerateRequest with query field

    Returns:
        PineScriptResponse with pine_script, status, and errors
    """
    logger.info(f"Received Pine Script generation request: {request.query[:100]}...")

    # Try Claude Orchestrator first
    if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
        try:
            result = await _invoke_claude_agent(
                "pinescript",
                f"Generate Pine Script v5 code for: {request.query}",
            )
            return PineScriptResponse(
                pine_script=result["reply"],
                status="complete",
                errors=[]
            )
        except Exception as e:
            logger.error(f"Claude Orchestrator Pine Script error: {e}")

    # Fallback to legacy agent
    if PINESCRIPT_AGENT_AVAILABLE:
        try:
            result = generate_pine_script_from_query(request.query)
            return PineScriptResponse(
                pine_script=result.get("pine_script"),
                status=result.get("status", "complete"),
                errors=result.get("errors", [])
            )
        except Exception as e:
            logger.error(f"Pine Script generation error: {e}", exc_info=True)
            return PineScriptResponse(
                pine_script=None,
                status="error",
                errors=[str(e)]
            )

    return PineScriptResponse(
        pine_script=None,
        status="error",
        errors=["Pine Script agent is not available. Please check server configuration."]
    )


@router.post("/pinescript/convert", response_model=PineScriptResponse)
async def convert_mql5_to_pinescript_endpoint(request: PineScriptConvertRequest):
    """
    Convert MQL5 code to Pine Script v5.

    Args:
        request: PineScriptConvertRequest with mql5_code field

    Returns:
        PineScriptResponse with pine_script, status, and errors
    """
    logger.info("Received MQL5 to Pine Script conversion request")

    # Try Claude Orchestrator first
    if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
        try:
            result = await _invoke_claude_agent(
                "pinescript",
                f"Convert this MQL5 code to Pine Script v5:\n\n```mql5\n{request.mql5_code}\n```",
            )
            return PineScriptResponse(
                pine_script=result["reply"],
                status="complete",
                errors=[]
            )
        except Exception as e:
            logger.error(f"Claude Orchestrator conversion error: {e}")

    # Fallback to legacy agent
    if PINESCRIPT_AGENT_AVAILABLE:
        try:
            result = convert_mql5_to_pinescript(request.mql5_code)
            return PineScriptResponse(
                pine_script=result.get("pine_script"),
                status=result.get("status", "complete"),
                errors=result.get("errors", [])
            )
        except Exception as e:
            logger.error(f"MQL5 to Pine Script conversion error: {e}", exc_info=True)
            return PineScriptResponse(
                pine_script=None,
                status="error",
                errors=[str(e)]
            )

    return PineScriptResponse(
        pine_script=None,
        status="error",
        errors=["Pine Script agent is not available. Please check server configuration."]
    )


def mock_chat_response(request: ChatRequest) -> ChatResponse:
    """Fallback response when no orchestrator is available."""
    msg = request.message.lower()

    if "deploy" in msg:
        return ChatResponse(
            reply="I can help with deployment. Since I'm in mock mode, I'll simulate a deployment check. Please verify your strategy parameters.",
            agent_id="copilot"
        )
    elif "analyze" in msg or "summary" in msg:
        return ChatResponse(
            reply="Market analysis requires a connection to live data sources. In this demo mode, I can confirm that the 'analyst' agent is registered and ready.",
            agent_id="analyst"
        )
    elif "code" in msg or "generate" in msg:
        return ChatResponse(
            reply="Code generation requires the QuantCode agent. In this demo mode, I can confirm the agent is registered and ready to generate MQL5 code.",
            agent_id="quantcode"
        )
    else:
        return ChatResponse(
            reply=f"I received your message: '{request.message}'. To get real responses, please ensure Claude Orchestrator is properly configured.",
            agent_id=request.agent_id
        )


def quantmind_aware_response(request: ChatRequest) -> ChatResponse:
    """
    QuantMind-aware fallback response when Claude CLI can't run (nested session).
    Provides intelligent responses based on QuantMindX system capabilities.
    """
    import os
    from pathlib import Path
    from datetime import datetime

    msg = request.message.lower().strip()
    agent_id = request.agent_id

    # Position and trading queries
    if any(kw in msg for kw in ["position", "positions", "trade", "trades", "holding"]):
        return ChatResponse(
            reply="**Position Query**\n\n"
                  "I can help you check positions, but I'm currently in restricted mode.\n\n"
                  "**Available endpoints:**\n"
                  "- `GET /api/router/status` - Bot status\n"
                  "- `GET /api/brokers/accounts` - Account info\n"
                  "- `GET /api/virtual-accounts` - Paper trading accounts\n\n"
                  "**To check live positions:**\n"
                  "1. Open MetaTrader 5 terminal\n"
                  "2. Or use the QuantMind dashboard at `/metrics`\n\n"
                  "Would you like me to help you set up position monitoring?",
            agent_id=agent_id
        )

    # Backtest queries
    elif any(kw in msg for kw in ["backtest", "back test", "test strategy", "historical"]):
        return ChatResponse(
            reply="**Backtest Request**\n\n"
                  "I can help you run backtests through the QuantMind system.\n\n"
                  "**Available backtest engines:**\n"
                  "- **Core Engine** (Backtrader-based)\n"
                  "- **Monte Carlo** (1,000+ simulations)\n"
                  "- **Walk-Forward** (out-of-sample validation)\n\n"
                  "**To run a backtest:**\n"
                  "```\n"
                  "POST /api/v1/backtest\n"
                  "{\n"
                  '  "strategy": "your_strategy",\n'
                  '  "symbol": "EURUSD",\n'
                  '  "timeframe": "H1",\n'
                  '  "start_date": "2024-01-01"\n'
                  "}\n"
                  "```\n\n"
                  "What strategy would you like to backtest?",
            agent_id=agent_id
        )

    # Strategy queries
    elif any(kw in msg for kw in ["strategy", "strategies", "ea", "expert advisor"]):
        strategies_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/strategies-yt")
        strategy_count = len(list(strategies_path.glob("*"))) if strategies_path.exists() else 0

        return ChatResponse(
            reply=f"**Strategy Management**\n\n"
                  f"I can help you with trading strategies.\n\n"
                  f"**Current strategies:** {strategy_count} in `/strategies-yt/`\n\n"
                  f"**Available actions:**\n"
                  f"- List strategies: `GET /api/strategies`\n"
                  f"- Create strategy: `POST /api/strategies?name=MyStrategy`\n"
                  f"- Deploy to paper: `POST /api/paper-trading/deploy`\n\n"
                  f"**Strategy families:**\n"
                  f"- SCALPER (fast, high frequency)\n"
                  f"- STRUCTURAL (ICT/SMC based)\n"
                  f"- SWING (multi-day holds)\n"
                  f"- HFT (sub-second execution)\n\n"
                  f"What would you like to do?",
            agent_id=agent_id
        )

    # Market regime / HMM queries
    elif any(kw in msg for kw in ["regime", "market condition", "hmm", "trending", "ranging"]):
        return ChatResponse(
            reply="**Market Regime Analysis**\n\n"
                  "The QuantMind system uses Hidden Markov Models (HMM) for regime detection.\n\n"
                  "**Detected regimes:**\n"
                  "- Trending (bullish/bearish)\n"
                  "- Ranging (consolidation)\n"
                  "- Volatile/Choppy (high uncertainty)\n\n"
                  "**API endpoints:**\n"
                  "- `GET /api/hmm/status` - HMM training status\n"
                  "- `POST /api/hmm/inference` - Query current regime\n"
                  "- `GET /api/hmm/models` - Available models\n\n"
                  "**Training schedule:**\n"
                  "- Every Saturday on Contabo server\n\n"
                  "Would you like me to check the current market regime?",
            agent_id=agent_id
        )

    # Bot management
    elif any(kw in msg for kw in ["bot", "bots", "ea status", "lifecycle"]):
        return ChatResponse(
            reply="**Bot Management**\n\n"
                  "I can help you manage trading bots.\n\n"
                  "**Bot tags:**\n"
                  "- `@primal` - Entry level (new bots)\n"
                  "- `@pending` - Validated (20+ trades, 50% win rate)\n"
                  "- `@perfect` - Advanced (Sharpe > 1.5)\n"
                  "- `@live` - Production ready\n"
                  "- `@quarantine` - Suspended\n"
                  "- `@dead` - Terminated\n\n"
                  "**API endpoints:**\n"
                  "- `GET /api/router/lifecycle/status` - All bot status\n"
                  "- `POST /api/router/lifecycle/promote/{bot_id}` - Promote bot\n"
                  "- `POST /api/router/lifecycle/quarantine/{bot_id}` - Quarantine\n\n"
                  "What bot operation do you need?",
            agent_id=agent_id
        )

    # Risk management
    elif any(kw in msg for kw in ["risk", "kelly", "position size", "lot size"]):
        return ChatResponse(
            reply="**Risk Management**\n\n"
                  "QuantMind uses Enhanced Kelly Criterion with 3-layer protection.\n\n"
                  "**Risk layers:**\n"
                  "1. **Base Kelly** - Mathematical optimal fraction\n"
                  "2. **Drawdown Adjustment** - Reduces size during losses\n"
                  "3. **Regime Filter** - Reduces in unfavorable conditions\n\n"
                  "**Safety limits:**\n"
                  "- Max 2% risk per trade\n"
                  "- 3% daily portfolio risk\n"
                  "- 10% weekly loss limit\n\n"
                  "**House Money Effect:**\n"
                  "- 0.5x conservative (down >3%)\n"
                  "- 1.0x baseline\n"
                  "- 1.5x aggressive (up >5%)\n\n"
                  "What risk calculation do you need?",
            agent_id=agent_id
        )

    # Session/timing queries
    elif any(kw in msg for kw in ["session", "london", "new york", "ny", "asian", "kill zone", "timing"]):
        now = datetime.utcnow()
        hour = now.hour

        # Simple session detection
        if 0 <= hour < 9:
            current = "Asian"
        elif 8 <= hour < 16:
            current = "London"
        elif 13 <= hour < 21:
            current = "New York"
        elif 13 <= hour < 16:
            current = "London/NY Overlap"
        else:
            current = "Closed/Weekend"

        return ChatResponse(
            reply=f"**Trading Sessions** (UTC: {now.strftime('%H:%M')})\n\n"
                  f"**Current session:** {current}\n\n"
                  f"**Session schedule:**\n"
                  f"- **Asian**: 00:00-09:00 Tokyo\n"
                  f"- **London**: 08:00-16:00 GMT\n"
                  f"- **New York**: 13:00-21:00 UTC\n"
                  f"- **Overlap**: 13:00-16:00 UTC (high volatility)\n\n"
                  f"**Kill Zones:**\n"
                  f"- London Open: 08:00-10:00 GMT\n"
                  f"- NY Open: 13:00-15:00 UTC\n\n"
                  f"**DST-aware:** Automatically adjusts for US/EU daylight saving.",
            agent_id=agent_id
        )

    # Help/greeting
    elif any(kw in msg for kw in ["help", "hello", "hi", "hey", "what can you"]):
        return ChatResponse(
            reply="**QuantMind Copilot** - General-purpose trading assistant\n\n"
                  "I can help you with:\n\n"
                  "**📊 Analysis:**\n"
                  "- Market regime detection (HMM)\n"
                  "- Session timing & kill zones\n"
                  "- Article research (1,806+ MQL5 articles)\n\n"
                  "**🔧 Strategy:**\n"
                  "- Strategy development & deployment\n"
                  "- Backtesting (Backtrader, Monte Carlo, Walk-forward)\n"
                  "- Paper trading & bot promotion\n\n"
                  "**💰 Risk:**\n"
                  "- Enhanced Kelly position sizing\n"
                  "- Portfolio risk management\n"
                  "- Fee monitoring\n\n"
                  "**🤖 Operations:**\n"
                  "- Bot lifecycle management\n"
                  "- Log monitoring\n"
                  "- System health checks\n\n"
                  "**⚠️ Note:** I'm in restricted mode (nested session).\n"
                  "For full AI capabilities, run outside Claude Code.\n\n"
                  "What would you like to explore?",
            agent_id=agent_id
        )

    # Default response
    else:
        return ChatResponse(
            reply=f"**Query:** {request.message}\n\n"
                  "I'm currently in restricted mode, but I can still help with:\n"
                  "- Strategy management\n"
                  "- Backtesting\n"
                  "- Risk calculations\n"
                  "- Bot lifecycle\n"
                  "- Session timing\n"
                  "- Market regime analysis\n\n"
                  "Try asking about:\n"
                  "- \"Show my positions\"\n"
                  "- \"Run a backtest\"\n"
                  "- \"Check bot status\"\n"
                  "- \"What session is it?\"\n"
                  "- \"Help with risk management\"",
            agent_id=agent_id
        )
