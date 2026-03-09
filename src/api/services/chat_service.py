"""
Chat Service - DEPRECATED

Use /api/floor-manager endpoints instead.
This module used the legacy agent system which has been removed.
"""

import os
import logging
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logging.warning(
    "chat_service is deprecated. Use /api/floor-manager endpoints instead."
)

# All legacy agents are now deprecated
PINESCRIPT_AGENT_AVAILABLE = False
CLAUDE_ORCHESTRATOR_AVAILABLE = False
LEGACY_AGENTS_AVAILABLE = False
SDK_ORCHESTRATOR_AVAILABLE = False

# Stub imports to prevent import errors
def get_orchestrator(*args, **kwargs):
    raise NotImplementedError(
        "Claude Orchestrator is deprecated. Use /api/floor-manager instead."
    )


class ClaudeOrchestrator:
    """Deprecated - use FloorManager instead."""
    pass


def get_agent_config(*args, **kwargs):
    raise NotImplementedError(
        "Agent config is deprecated. Use /api/floor-manager instead."
    )


def get_all_agent_ids():
    return []


def get_sdk_orchestrator(*args, **kwargs):
    raise NotImplementedError(
        "SDK Orchestrator is deprecated. Use /api/floor-manager instead."
    )

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

USE_CLAUDE_ORCHESTRATOR = False  # Deprecated
USE_SDK_ORCHESTRATOR = False  # Deprecated
# Valid agent types - deprecated, use floor_manager instead
VALID_AGENT_TYPES = ["copilot", "analyst", "quantcode", "pinescript", "router", "executor"]
# New agent types via floor_manager
VALID_DEPARTMENTS = ["development", "research", "risk", "trading", "portfolio"]


# =============================================================================
# Message Building
# =============================================================================

def build_messages(message: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build messages list from current message and history.

    Args:
        message: Current user message
        history: Conversation history

    Returns:
        List of message dicts with role and content
    """
    messages = []

    if history:
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

    messages.append({
        "role": "user",
        "content": message
    })

    return messages


def extract_action_from_message(message: str) -> Optional[str]:
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


def determine_target_agent(message: str, default_agent: str) -> str:
    """
    Determine target agent based on message content or default.

    Args:
        message: User message
        default_agent: Default agent to use

    Returns:
        Target agent type
    """
    logger.warning(
        "determine_target_agent is deprecated. "
        "Use WorkshopCopilotService instead."
    )
    if message.startswith("/deploy"):
        return "copilot"
    elif message.startswith("/analyze"):
        return "analyst"
    elif message.startswith("/code") or message.startswith("/generate"):
        return "quantcode"
    return default_agent


# =============================================================================
# Agent Invocation
# =============================================================================

async def invoke_claude_agent(
    agent_type: str,
    message: str,
    history: List[Dict[str, Any]] = None,
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
    messages = build_messages(message, history or [])

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

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Timeout - return task ID for async tracking
    return {
        "reply": f"Task {task_id} is still processing. Check status with /api/chat/task/{task_id}",
        "task_id": task_id,
        "agent_id": agent_type,
        "artifacts": [],
    }


async def invoke_claude_agent_async(
    agent_type: str,
    message: str,
    history: List[Dict[str, Any]] = None,
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
    messages = build_messages(message, history or [])

    task_id = await orchestrator.submit_task(
        agent_id=agent_type,
        messages=messages,
        session_id=session_id,
    )

    return task_id


# =============================================================================
# Legacy Tool Functions
# =============================================================================

def deploy_strategy(strategy_name: str) -> str:
    """Deploys a trading strategy using the Copilot workflow."""
    logger.info(f"Tool invoked: deploy_strategy({strategy_name})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_copilot_workflow(f"Deploy strategy {strategy_name}")
        return f"Deployment initiated. Status: {result.get('monitoring_data', {}).get('status', 'Unknown')}"
    else:
        return f"Deployment simulation for strategy: {strategy_name}"


def analyze_market(query: str) -> str:
    """Analyzes market conditions using the Analyst workflow."""
    logger.info(f"Tool invoked: analyze_market({query})")

    if LEGACY_AGENTS_AVAILABLE:
        result = run_analyst_workflow(query)
        return result.get("synthesis_result", "Analysis failed to produce result.")
    else:
        return f"Market analysis simulation for: {query}"


# =============================================================================
# Tool Execution
# =============================================================================

class ToolExecutionResult:
    """Result of tool execution."""
    def __init__(self, success: bool, result: Optional[str] = None,
                 error: Optional[str] = None, artifacts: List[Dict[str, Any]] = None):
        self.success = success
        self.result = result
        self.error = error
        self.artifacts = artifacts or []


async def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: str = "copilot"
) -> ToolExecutionResult:
    """
    Execute a tool directly without spawning Claude CLI.

    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters
        agent_id: Agent ID for context

    Returns:
        ToolExecutionResult with success, result, error, and artifacts
    """
    base_dir = Path(os.getenv("WORKSPACES_DIR", "/home/mubarkahimself/Desktop/QUANTMINDX"))

    try:
        # File system tools
        if tool_name == "write_file":
            file_path = Path(parameters.get("path", ""))
            content = parameters.get("content", "")

            # Security: ensure path is within allowed directories
            if not str(file_path.absolute()).startswith(str(base_dir.absolute())):
                project_dir = Path("/home/mubarkahimself/Desktop/QUANTMINDX")
                if not str(file_path.absolute()).startswith(str(project_dir.absolute())):
                    return ToolExecutionResult(
                        success=False,
                        error=f"Access denied: path must be within {base_dir} or {project_dir}"
                    )

            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(content)

            return ToolExecutionResult(
                success=True,
                result=f"File written successfully: {file_path}",
                artifacts=[{"type": "file", "path": str(file_path), "size": len(content)}]
            )

        elif tool_name == "read_file":
            file_path = Path(parameters.get("path", ""))

            if not file_path.exists():
                return ToolExecutionResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            with open(file_path, "r") as f:
                content = f.read()

            return ToolExecutionResult(
                success=True,
                result=content,
                artifacts=[{"type": "file_content", "path": str(file_path)}]
            )

        elif tool_name == "list_directory":
            dir_path = Path(parameters.get("path", str(base_dir)))

            if not dir_path.exists():
                return ToolExecutionResult(
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

            return ToolExecutionResult(
                success=True,
                result=json.dumps(items, indent=2),
                artifacts=[{"type": "directory_listing", "path": str(dir_path), "count": len(items)}]
            )

        elif tool_name == "delete_file":
            file_path = Path(parameters.get("path", ""))

            if not file_path.exists():
                return ToolExecutionResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            file_path.unlink()

            return ToolExecutionResult(
                success=True,
                result=f"File deleted: {file_path}"
            )

        elif tool_name == "create_directory":
            dir_path = Path(parameters.get("path", ""))
            dir_path.mkdir(parents=True, exist_ok=True)

            return ToolExecutionResult(
                success=True,
                result=f"Directory created: {dir_path}"
            )

        elif tool_name == "run_command":
            import subprocess

            command = parameters.get("command", "")
            cwd = parameters.get("cwd", str(base_dir))

            # Security: block dangerous commands
            blocked = ["rm -rf", "sudo", "chmod 777", "> /dev/", "mkfs", "dd if="]
            if any(b in command for b in blocked):
                return ToolExecutionResult(
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

            return ToolExecutionResult(
                success=result.returncode == 0,
                result=result.stdout or result.stderr,
                error=result.stderr if result.returncode != 0 else None
            )

        else:
            return ToolExecutionResult(
                success=False,
                error=f"Unknown tool: {tool_name}. Available: write_file, read_file, list_directory, delete_file, create_directory, run_command"
            )

    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return ToolExecutionResult(
            success=False,
            error=str(e)
        )


# =============================================================================
# Pine Script Generation
# =============================================================================

async def generate_pine_script(query: str) -> Dict[str, Any]:
    """
    Generate Pine Script v5 code from natural language query.

    Args:
        query: Natural language description

    Returns:
        Dict with pine_script, status, and errors
    """
    logger.info(f"Generating Pine Script for: {query[:100]}...")

    # Try Claude Orchestrator first
    if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
        try:
            result = await invoke_claude_agent(
                "pinescript",
                f"Generate Pine Script v5 code for: {query}",
            )
            return {
                "pine_script": result["reply"],
                "status": "complete",
                "errors": []
            }
        except Exception as e:
            logger.error(f"Claude Orchestrator Pine Script error: {e}")

    # Fallback to legacy agent
    if PINESCRIPT_AGENT_AVAILABLE:
        try:
            result = generate_pine_script_from_query(query)
            return {
                "pine_script": result.get("pine_script"),
                "status": result.get("status", "complete"),
                "errors": result.get("errors", [])
            }
        except Exception as e:
            logger.error(f"Pine Script generation error: {e}", exc_info=True)
            return {
                "pine_script": None,
                "status": "error",
                "errors": [str(e)]
            }

    return {
        "pine_script": None,
        "status": "error",
        "errors": ["Pine Script agent is not available. Please check server configuration."]
    }


async def convert_mql5_to_pine(mql5_code: str) -> Dict[str, Any]:
    """
    Convert MQL5 code to Pine Script v5.

    Args:
        mql5_code: MQL5 source code

    Returns:
        Dict with pine_script, status, and errors
    """
    logger.info("Converting MQL5 to Pine Script")

    # Try Claude Orchestrator first
    if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
        try:
            result = await invoke_claude_agent(
                "pinescript",
                f"Convert this MQL5 code to Pine Script v5:\n\n```mql5\n{mql5_code}\n```",
            )
            return {
                "pine_script": result["reply"],
                "status": "complete",
                "errors": []
            }
        except Exception as e:
            logger.error(f"Claude Orchestrator conversion error: {e}")

    # Fallback to legacy agent
    if PINESCRIPT_AGENT_AVAILABLE:
        try:
            result = convert_mql5_to_pinescript(mql5_code)
            return {
                "pine_script": result.get("pine_script"),
                "status": result.get("status", "complete"),
                "errors": result.get("errors", [])
            }
        except Exception as e:
            logger.error(f"MQL5 to Pine Script conversion error: {e}", exc_info=True)
            return {
                "pine_script": None,
                "status": "error",
                "errors": [str(e)]
            }

    return {
        "pine_script": None,
        "status": "error",
        "errors": ["Pine Script agent is not available. Please check server configuration."]
    }


# =============================================================================
# Chat Processing
# =============================================================================

class ChatService:
    """Main chat service class."""

    def __init__(self):
        self.logger = logger

    async def process_chat(
        self,
        message: str,
        agent_id: str = "copilot",
        history: List[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        skill_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return response.

        Args:
            message: User message
            agent_id: Target agent ID
            history: Conversation history
            session_id: Session ID for continuity
            skill_id: Optional skill ID to use for processing

        Returns:
            Dict with reply, agent_id, task_id, action_taken, artifacts
        """
        # Log skill usage if provided
        if skill_id:
            logger.info(f"Processing chat with skill: {skill_id}")
        # Validate agent type
        if agent_id not in VALID_AGENT_TYPES:
            return {
                "reply": f"Unknown agent type: {agent_id}. Valid types: {', '.join(VALID_AGENT_TYPES)}",
                "agent_id": "system",
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Extract action type
        action = extract_action_from_message(message)

        # Determine target agent
        target_agent = determine_target_agent(message, agent_id)

        try:
            # Try Claude Orchestrator
            if USE_CLAUDE_ORCHESTRATOR and CLAUDE_ORCHESTRATOR_AVAILABLE:
                result = await invoke_claude_agent(
                    target_agent,
                    message,
                    history,
                    session_id,
                )

                return {
                    "reply": result["reply"],
                    "agent_id": result["agent_id"],
                    "task_id": result.get("task_id"),
                    "action_taken": action,
                    "artifacts": result.get("artifacts", [])
                }

            # Fallback to legacy agents
            if LEGACY_AGENTS_AVAILABLE:
                if message.startswith("/deploy"):
                    reply = deploy_strategy(message.replace("/deploy", "").strip())
                    return {
                        "reply": reply,
                        "agent_id": "copilot",
                        "task_id": None,
                        "action_taken": "deployment",
                        "artifacts": []
                    }

                if message.startswith("/analyze"):
                    reply = analyze_market(message.replace("/analyze", "").strip())
                    return {
                        "reply": reply,
                        "agent_id": "analyst",
                        "task_id": None,
                        "action_taken": "analysis",
                        "artifacts": []
                    }

                result = run_copilot_workflow(message)
                return {
                    "reply": str(result.get("messages", ["Completed"])[-1]),
                    "agent_id": agent_id,
                    "task_id": None,
                    "action_taken": None,
                    "artifacts": []
                }

            # Mock response
            return self._mock_response(message, agent_id)

        except Exception as e:
            self.logger.error(f"Chat processing error: {e}", exc_info=True)
            # Check for nested session error
            if "nested session" in str(e).lower() or "claude code cannot be launched" in str(e).lower():
                return self._quantmind_aware_response(message, agent_id)

            return {
                "reply": f"I encountered an error processing your request: {str(e)}",
                "agent_id": "system",
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

    async def handle_tool_request(self, message: str, agent_id: str = "copilot") -> Dict[str, Any]:
        """
        Handle chat messages that look like tool requests.

        Args:
            message: User message
            agent_id: Agent ID

        Returns:
            Dict with reply, agent_id, action_taken, artifacts
        """
        message_lower = message.lower()

        # Detect file write requests
        write_pattern = r"(?:write|create|save)\s+(?:a\s+)?(?:file|ea|strategy)\s+(?:to\s+)?[`'\"]?([^\s`'\"]+)[`'\"]?(?:\s+with\s+(?:content|text)?\s*[:=]?\s*(.+))?"
        match = re.search(write_pattern, message, re.IGNORECASE | re.DOTALL)

        if match:
            file_path = match.group(1)
            content = match.group(2) or f"Generated by {agent_id}\nTimestamp: {datetime.utcnow().isoformat()}"

            result = await execute_tool(
                "write_file",
                {"path": file_path, "content": content.strip()},
                agent_id
            )

            if result.success:
                return {
                    "reply": f"I've written the file to {file_path}. Size: {len(content)} bytes.",
                    "agent_id": agent_id,
                    "action_taken": "file_write",
                    "artifacts": result.artifacts
                }
            else:
                return {
                    "reply": f"Failed to write file: {result.error}",
                    "agent_id": "system",
                    "action_taken": None,
                    "artifacts": []
                }

        # Detect directory listing requests
        if "list" in message_lower and ("directory" in message_lower or "folder" in message_lower or "files" in message_lower):
            path_match = re.search(r"[`'\"]?(/[^\s`'\"]+)[`'\"]?", message)
            path = path_match.group(1) if path_match else "/home/mubarkahimself/Desktop/QUANTMINDX"

            result = await execute_tool("list_directory", {"path": path}, agent_id)

            if result.success:
                return {
                    "reply": f"Directory listing for {path}:\n{result.result}",
                    "agent_id": agent_id,
                    "action_taken": "list_directory",
                    "artifacts": result.artifacts
                }
            else:
                return {
                    "reply": f"Failed to list directory: {result.error}",
                    "agent_id": "system",
                    "action_taken": None,
                    "artifacts": []
                }

        # Detect read file requests
        read_pattern = r"(?:read|show|display|cat)\s+(?:the\s+)?(?:file\s+)?[`'\"]?(/[^\s`'\"]+)[`'\"]?"
        match = re.search(read_pattern, message, re.IGNORECASE)

        if match:
            file_path = match.group(1)

            result = await execute_tool("read_file", {"path": file_path}, agent_id)

            if result.success:
                return {
                    "reply": f"Contents of {file_path}:\n```\n{result.result}\n```",
                    "agent_id": agent_id,
                    "action_taken": "read_file",
                    "artifacts": result.artifacts
                }
            else:
                return {
                    "reply": f"Failed to read file: {result.error}",
                    "agent_id": "system",
                    "action_taken": None,
                    "artifacts": []
                }

        # Default: return helpful message
        return {
            "reply": f"I understand you want to: {message[:100]}...\n\n"
                     f"I can help with:\n"
                     f"- Write a file to [path] with content [text]\n"
                     f"- Read file [path]\n"
                     f"- List directory [path]\n"
                     f"- Run command [command] (restricted)\n\n"
                     f"Try: 'Write a file to /path/to/file.txt with content: Hello World'",
            "agent_id": agent_id,
            "action_taken": None,
            "artifacts": []
        }

    def _mock_response(self, message: str, agent_id: str) -> Dict[str, Any]:
        """Fallback response when no orchestrator is available."""
        msg = message.lower()

        if "deploy" in msg:
            return {
                "reply": "I can help with deployment. Since I'm in mock mode, I'll simulate a deployment check. Please verify your strategy parameters.",
                "agent_id": "copilot",
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }
        elif "analyze" in msg or "summary" in msg:
            return {
                "reply": "Market analysis requires a connection to live data sources. In this demo mode, I can confirm that the 'analyst' agent is registered and ready.",
                "agent_id": "analyst",
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }
        elif "code" in msg or "generate" in msg:
            return {
                "reply": "Code generation requires the QuantCode agent. In this demo mode, I can confirm the agent is registered and ready to generate MQL5 code.",
                "agent_id": "quantcode",
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }
        else:
            return {
                "reply": f"I received your message: '{message}'. To get real responses, please ensure Claude Orchestrator is properly configured.",
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

    def _quantmind_aware_response(self, message: str, agent_id: str) -> Dict[str, Any]:
        """
        QuantMind-aware fallback response when Claude CLI can't run.

        Provides intelligent responses based on QuantMindX system capabilities.
        """
        msg = message.lower().strip()

        # Position and trading queries
        if any(kw in msg for kw in ["position", "positions", "trade", "trades", "holding"]):
            return {
                "reply": "**Position Query**\n\n"
                         "I can help you check positions, but I'm currently in restricted mode.\n\n"
                         "**Available endpoints:**\n"
                         "- `GET /api/router/status` - Bot status\n"
                         "- `GET /api/brokers/accounts` - Account info\n"
                         "- `GET /api/virtual-accounts` - Paper trading accounts\n\n"
                         "**To check live positions:**\n"
                         "1. Open MetaTrader 5 terminal\n"
                         "2. Or use the QuantMind dashboard at `/metrics`\n\n"
                         "Would you like me to help you set up position monitoring?",
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Backtest queries
        elif any(kw in msg for kw in ["backtest", "back test", "test strategy", "historical"]):
            return {
                "reply": "**Backtest Request**\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Strategy queries
        elif any(kw in msg for kw in ["strategy", "strategies", "ea", "expert advisor"]):
            strategies_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/strategies-yt")
            strategy_count = len(list(strategies_path.glob("*"))) if strategies_path.exists() else 0

            return {
                "reply": f"**Strategy Management**\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Market regime / HMM queries
        elif any(kw in msg for kw in ["regime", "market condition", "hmm", "trending", "ranging"]):
            return {
                "reply": "**Market Regime Analysis**\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Bot management
        elif any(kw in msg for kw in ["bot", "bots", "ea status", "lifecycle"]):
            return {
                "reply": "**Bot Management**\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Risk management
        elif any(kw in msg for kw in ["risk", "kelly", "position size", "lot size"]):
            return {
                "reply": "**Risk Management**\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Session/timing queries
        elif any(kw in msg for kw in ["session", "london", "new york", "ny", "asian", "kill zone", "timing"]):
            now = datetime.utcnow()
            hour = now.hour

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

            return {
                "reply": f"**Trading Sessions** (UTC: {now.strftime('%H:%M')})\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Help/greeting
        elif any(kw in msg for kw in ["help", "hello", "hi", "hey", "what can you"]):
            return {
                "reply": "**QuantMind Copilot** - General-purpose trading assistant\n\n"
                         "I can help you with:\n\n"
                         "**Analysis:**\n"
                         "- Market regime detection (HMM)\n"
                         "- Session timing & kill zones\n"
                         "- Article research (1,806+ MQL5 articles)\n\n"
                         "**Strategy:**\n"
                         "- Strategy development & deployment\n"
                         "- Backtesting (Backtrader, Monte Carlo, Walk-forward)\n"
                         "- Paper trading & bot promotion\n\n"
                         "**Risk:**\n"
                         "- Enhanced Kelly position sizing\n"
                         "- Portfolio risk management\n"
                         "- Fee monitoring\n\n"
                         "**Operations:**\n"
                         "- Bot lifecycle management\n"
                         "- Log monitoring\n"
                         "- System health checks\n\n"
                         "**Note:** I'm in restricted mode (nested session).\n"
                         "For full AI capabilities, run outside Claude Code.\n\n"
                         "What would you like to explore?",
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

        # Default response
        else:
            return {
                "reply": f"**Query:** {message}\n\n"
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
                "agent_id": agent_id,
                "task_id": None,
                "action_taken": None,
                "artifacts": []
            }

    def get_available_agents(self) -> Dict[str, Any]:
        """Get information about available agents."""
        return {
            "agents": VALID_AGENT_TYPES,
            "orchestrator_available": CLAUDE_ORCHESTRATOR_AVAILABLE,
            "sdk_orchestrator_available": SDK_ORCHESTRATOR_AVAILABLE,
            "legacy_available": LEGACY_AGENTS_AVAILABLE,
        }


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
