"""
Tool Call Logging Endpoint

API endpoints for serving tool call logs with terminal-friendly formatting.
Supports filtering by agent, time range, and other parameters.

**Validates: Phase 6 Task 11 - Terminal UI Endpoint for Tool Calls**
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tool-calls", tags=["tool-calls"])

# In-memory storage for tool call logs (in production, use a database)
_tool_call_logs: List[Dict[str, Any]] = []
_log_lock = None  # Will use threading.Lock if available


def _get_lock():
    """Get or create the lock for thread safety."""
    global _log_lock
    if _log_lock is None:
        try:
            import threading
            _log_lock = threading.Lock()
        except ImportError:
            pass
    return _log_lock


# =============================================================================
# MODELS
# =============================================================================

@dataclass
class ToolCallLog:
    """Represents a single tool call log entry."""
    id: str
    timestamp: datetime
    agent_id: str
    agent_type: str
    tool_name: str
    args: Dict[str, Any]
    result: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }

    def to_terminal_line(self, max_arg_length: int = 100) -> str:
        """
        Format as a terminal-friendly line.

        Args:
            max_arg_length: Maximum length for args display

        Returns:
            Formatted string suitable for terminal display
        """
        # Truncate args if too long
        args_str = str(self.args)
        if len(args_str) > max_arg_length:
            args_str = args_str[:max_arg_length - 3] + "..."

        # Format timestamp
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Format status indicator
        status = "OK" if self.success else "ERR"

        # Format duration if available
        dur_str = ""
        if self.duration_ms is not None:
            dur_str = f" [{self.duration_ms:.1f}ms]"

        return f"[{ts}] {self.agent_id:<20} {self.tool_name:<30} {status}{dur_str} {args_str}"


@dataclass
class ToolCallLogEntryRequest:
    """Request model for creating a tool call log entry."""
    agent_id: str
    agent_type: str
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ToolCallSummary:
    """Summary statistics for tool calls."""
    total_calls: int
    successful_calls: int
    failed_calls: int
    unique_agents: int
    unique_tools: int
    agents: Dict[str, int] = field(default_factory=dict)
    tools: Dict[str, int] = field(default_factory=dict)
    time_range: Optional[Dict[str, str]] = None


# =============================================================================
# LOGGING FUNCTIONS (for integration with existing observer system)
# =============================================================================

def log_tool_call(
    agent_id: str,
    agent_type: str,
    tool_name: str,
    args: Dict[str, Any],
    result: Optional[str] = None,
    duration_ms: Optional[float] = None,
    success: bool = True,
    error: Optional[str] = None,
) -> str:
    """
    Log a tool call to the in-memory store.

    Args:
        agent_id: ID of the agent making the call
        agent_type: Type of agent (analyst, quantcode, etc.)
        tool_name: Name of the tool being called
        args: Arguments passed to the tool
        result: Optional result string
        duration_ms: Optional duration in milliseconds
        success: Whether the call succeeded
        error: Optional error message

    Returns:
        The log entry ID
    """
    import uuid

    log_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc)

    entry = ToolCallLog(
        id=log_id,
        timestamp=timestamp,
        agent_id=agent_id,
        agent_type=agent_type,
        tool_name=tool_name,
        args=args,
        result=result,
        duration_ms=duration_ms,
        success=success,
        error=error,
    )

    lock = _get_lock()
    if lock:
        with lock:
            _tool_call_logs.append(entry.to_dict())
    else:
        _tool_call_logs.append(entry.to_dict())

    # Keep only last 10000 entries
    if len(_tool_call_logs) > 10000:
        if lock:
            with lock:
                _tool_call_logs[:] = _tool_call_logs[-10000:]
        else:
            _tool_call_logs[:] = _tool_call_logs[-10000:]

    logger.debug(f"Logged tool call: {agent_id} -> {tool_name}")
    return log_id


def get_tool_call_logs(
    agent_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    tool_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get tool call logs with optional filtering.

    Args:
        agent_id: Filter by agent ID
        agent_type: Filter by agent type
        tool_name: Filter by tool name
        start_time: Filter by start time
        end_time: Filter by end time
        limit: Maximum number of entries to return
        offset: Offset for pagination

    Returns:
        List of tool call log entries
    """
    lock = _get_lock()

    def _get_logs():
        logs = list(_tool_call_logs)

        # Apply filters
        if agent_id:
            logs = [l for l in logs if l.get("agent_id") == agent_id]
        if agent_type:
            logs = [l for l in logs if l.get("agent_type") == agent_type]
        if tool_name:
            logs = [l for l in logs if l.get("tool_name") == tool_name]
        if start_time:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")) >= start_time]
        if end_time:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")) <= end_time]

        # Sort by timestamp descending (most recent first)
        logs.sort(key=lambda x: x["timestamp"], reverse=True)

        return logs[offset:offset + limit]

    if lock:
        with lock:
            return _get_logs()
    else:
        return _get_logs()


def get_tool_call_summary(
    agent_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> ToolCallSummary:
    """
    Get summary statistics for tool calls.

    Args:
        agent_id: Filter by agent ID
        start_time: Filter by start time
        end_time: Filter by end time

    Returns:
        Summary statistics
    """
    lock = _get_lock()

    def _get_summary():
        logs = list(_tool_call_logs)

        # Apply time filters
        if start_time:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")) >= start_time]
        if end_time:
            logs = [l for l in logs if datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")) <= end_time]

        if agent_id:
            logs = [l for l in logs if l.get("agent_id") == agent_id]

        # Calculate statistics
        total = len(logs)
        successful = sum(1 for l in logs if l.get("success", True))
        failed = total - successful

        # Count by agent
        agents: Dict[str, int] = defaultdict(int)
        for l in logs:
            agents[l.get("agent_id", "unknown")] += 1

        # Count by tool
        tools: Dict[str, int] = defaultdict(int)
        for l in logs:
            tools[l.get("tool_name", "unknown")] += 1

        # Time range
        time_range = None
        if logs:
            timestamps = [datetime.fromisoformat(l["timestamp"].replace("Z", "+00:00")) for l in logs]
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
            }

        return ToolCallSummary(
            total_calls=total,
            successful_calls=successful,
            failed_calls=failed,
            unique_agents=len(agents),
            unique_tools=len(tools),
            agents=dict(agents),
            tools=dict(tools),
            time_range=time_range,
        )

    if lock:
        with lock:
            return _get_summary()
    else:
        return _get_summary()


def get_terminal_output(
    agent_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    tool_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 50,
) -> str:
    """
    Get tool call logs formatted for terminal display.

    Args:
        agent_id: Filter by agent ID
        agent_type: Filter by agent type
        tool_name: Filter by tool name
        start_time: Filter by start time
        end_time: Filter by end time
        limit: Maximum number of lines

    Returns:
        Formatted string for terminal display
    """
    logs = get_tool_call_logs(
        agent_id=agent_id,
        agent_type=agent_type,
        tool_name=tool_name,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )

    if not logs:
        return "No tool calls found matching the specified filters."

    # Build header
    output_lines = [
        "=" * 120,
        f"{'TIMESTAMP':<20} {'AGENT_ID':<20} {'TOOL_NAME':<30} {'STATUS':<8} {'DURATION':<12} {'ARGS':<30}",
        "=" * 120,
    ]

    # Add each log entry
    for log in logs:
        ts = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        agent = log.get("agent_id", "unknown")[:20]
        tool = log.get("tool_name", "unknown")[:30]
        status = "OK" if log.get("success", True) else "ERR"
        dur = f"{log.get('duration_ms', 0):.1f}ms" if log.get("duration_ms") else "-"

        # Format args
        args = str(log.get("args", {}))
        if len(args) > 28:
            args = args[:25] + "..."
        args = args[:30]

        output_lines.append(f"{ts_str:<20} {agent:<20} {tool:<30} {status:<8} {dur:<12} {args:<30}")

    # Add footer with summary
    output_lines.append("=" * 120)
    output_lines.append(f"Total: {len(logs)} entries")

    return "\n".join(output_lines)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/logs")
async def get_logs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    tool_name: Optional[str] = Query(None, description="Filter by tool name"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entries to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """
    Get tool call logs with optional filtering.

    Supports filtering by:
    - agent_id: Specific agent identifier
    - agent_type: Type of agent (analyst, quantcode, copilot, router)
    - tool_name: Name of the tool
    - start_time/end_time: Time range in ISO format

    Returns paginated results with metadata.
    """
    # Parse time filters
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format.")

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format.")

    logs = get_tool_call_logs(
        agent_id=agent_id,
        agent_type=agent_type,
        tool_name=tool_name,
        start_time=start_dt,
        end_time=end_dt,
        limit=limit,
        offset=offset,
    )

    return {
        "logs": logs,
        "count": len(logs),
        "limit": limit,
        "offset": offset,
        "filters": {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "tool_name": tool_name,
            "start_time": start_time,
            "end_time": end_time,
        }
    }


@router.get("/terminal")
async def get_terminal_logs(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    tool_name: Optional[str] = Query(None, description="Filter by tool name"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum lines to return"),
) -> Dict[str, Any]:
    """
    Get tool call logs formatted for terminal display.

    Returns logs in a format suitable for display in terminal/console.
    Includes formatted output string and raw data.
    """
    # Parse time filters
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format.")

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format.")

    # Get terminal-formatted output
    terminal_output = get_terminal_output(
        agent_id=agent_id,
        agent_type=agent_type,
        tool_name=tool_name,
        start_time=start_dt,
        end_time=end_dt,
        limit=limit,
    )

    # Also get raw data for programmatic access
    logs = get_tool_call_logs(
        agent_id=agent_id,
        agent_type=agent_type,
        tool_name=tool_name,
        start_time=start_dt,
        end_time=end_dt,
        limit=limit,
    )

    return {
        "terminal_output": terminal_output,
        "logs": logs,
        "count": len(logs),
        "filters": {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "tool_name": tool_name,
            "start_time": start_time,
            "end_time": end_time,
        }
    }


@router.get("/summary")
async def get_summary(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
) -> Dict[str, Any]:
    """
    Get summary statistics for tool calls.

    Returns aggregated statistics including:
    - Total/successful/failed call counts
    - Unique agents and tools
    - Breakdown by agent and tool
    - Time range of logged calls
    """
    # Parse time filters
    start_dt = None
    end_dt = None

    if start_time:
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format.")

    if end_time:
        try:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format.")

    summary = get_tool_call_summary(
        agent_id=agent_id,
        start_time=start_dt,
        end_time=end_dt,
    )

    return {
        "total_calls": summary.total_calls,
        "successful_calls": summary.successful_calls,
        "failed_calls": summary.failed_calls,
        "success_rate": round(summary.successful_calls / summary.total_calls, 4) if summary.total_calls > 0 else 0,
        "unique_agents": summary.unique_agents,
        "unique_tools": summary.unique_tools,
        "by_agent": summary.agents,
        "by_tool": summary.tools,
        "time_range": summary.time_range,
        "filters": {
            "agent_id": agent_id,
            "start_time": start_time,
            "end_time": end_time,
        }
    }


@router.get("/agents")
async def get_agents() -> Dict[str, Any]:
    """
    Get list of unique agents that have made tool calls.

    Returns:
        List of agent IDs and their tool call counts
    """
    lock = _get_lock()

    def _get_agents():
        agent_counts: Dict[str, int] = defaultdict(int)
        agent_types: Dict[str, str] = {}

        for log in _tool_call_logs:
            agent_id = log.get("agent_id", "unknown")
            agent_counts[agent_id] += 1
            if "agent_type" in log:
                agent_types[agent_id] = log["agent_type"]

        return [
            {
                "agent_id": agent_id,
                "agent_type": agent_types.get(agent_id, "unknown"),
                "call_count": count,
            }
            for agent_id, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    if lock:
        with lock:
            agents = _get_agents()
    else:
        agents = _get_agents()

    return {
        "agents": agents,
        "count": len(agents),
    }


@router.get("/tools")
async def get_tools() -> Dict[str, Any]:
    """
    Get list of unique tools that have been called.

    Returns:
        List of tool names and their call counts
    """
    lock = _get_lock()

    def _get_tools():
        tool_counts: Dict[str, int] = defaultdict(int)

        for log in _tool_call_logs:
            tool_name = log.get("tool_name", "unknown")
            tool_counts[tool_name] += 1

        return [
            {
                "tool_name": tool_name,
                "call_count": count,
            }
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        ]

    if lock:
        with lock:
            tools = _get_tools()
    else:
        tools = _get_tools()

    return {
        "tools": tools,
        "count": len(tools),
    }


@router.post("/log")
async def create_log_entry(request: ToolCallLogEntryRequest) -> Dict[str, Any]:
    """
    Manually log a tool call entry.

    This endpoint allows manual insertion of tool call logs,
    useful for testing or integrating with external systems.
    """
    log_id = log_tool_call(
        agent_id=request.agent_id,
        agent_type=request.agent_type,
        tool_name=request.tool_name,
        args=request.args,
        result=request.result,
        duration_ms=request.duration_ms,
        success=request.success,
        error=request.error,
    )

    return {
        "success": True,
        "log_id": log_id,
        "message": "Tool call logged successfully",
    }


@router.delete("/logs")
async def clear_logs() -> Dict[str, Any]:
    """
    Clear all tool call logs.

    Use with caution - this will delete all logged tool calls.
    """
    global _tool_call_logs

    lock = _get_lock()
    if lock:
        with lock:
            count = len(_tool_call_logs)
            _tool_call_logs = []
    else:
        count = len(_tool_call_logs)
        _tool_call_logs = []

    return {
        "success": True,
        "cleared_count": count,
        "message": f"Cleared {count} tool call logs",
    }
