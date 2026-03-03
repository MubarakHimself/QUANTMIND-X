"""
Tool Call Logger for Department Agents

Logs tool calls with timestamp, agent name, tool name, and input parameters.
Stores logs in memory (can be extended to database later).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """
    Represents a single tool call entry.

    Attributes:
        timestamp: When the tool was called
        agent_name: Name of the agent that made the call
        tool_name: Name of the tool being called
        input_params: Parameters passed to the tool
        result: Optional result from the tool call
        success: Whether the tool call succeeded
        duration_ms: Duration of the tool call in milliseconds
    """
    timestamp: datetime
    agent_name: str
    tool_name: str
    input_params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    success: bool = True
    duration_ms: Optional[float] = None


class ToolCallLogger:
    """
    Logger for tracking tool calls made by department agents.

    Provides in-memory storage of tool call logs with methods to
    log calls, retrieve logs, and clear logs.

    Example:
        >>> logger = ToolCallLogger()
        >>> logger.log_call("research_head", "memory_tools", {"key": "test"})
        >>> logs = logger.get_logs()
        >>> print(len(logs))
        1
    """

    def __init__(self, max_entries: int = 1000):
        """
        Initialize the ToolCallLogger.

        Args:
            max_entries: Maximum number of log entries to keep in memory
        """
        self._logs: List[ToolCall] = []
        self._max_entries = max_entries

    def log_call(
        self,
        agent_name: str,
        tool_name: str,
        input_params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        duration_ms: Optional[float] = None,
    ) -> None:
        """
        Log a tool call.

        Args:
            agent_name: Name of the agent making the call
            tool_name: Name of the tool being called
            input_params: Parameters passed to the tool
            result: Optional result from the tool call
            success: Whether the tool call succeeded
            duration_ms: Duration of the tool call in milliseconds
        """
        tool_call = ToolCall(
            timestamp=datetime.now(),
            agent_name=agent_name,
            tool_name=tool_name,
            input_params=input_params or {},
            result=result,
            success=success,
            duration_ms=duration_ms,
        )

        self._logs.append(tool_call)

        # Trim oldest entries if we exceed max
        if len(self._logs) > self._max_entries:
            self._logs = self._logs[-self._max_entries:]

        logger.debug(
            f"Tool call logged: {agent_name} -> {tool_name} "
            f"(success={success}, duration={duration_ms}ms)"
        )

    def get_logs(
        self,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ToolCall]:
        """
        Get tool call logs with optional filtering.

        Args:
            agent_name: Filter by agent name
            tool_name: Filter by tool name
            limit: Maximum number of entries to return

        Returns:
            List of ToolCall entries
        """
        logs = self._logs

        if agent_name:
            logs = [log for log in logs if log.agent_name == agent_name]

        if tool_name:
            logs = [log for log in logs if log.tool_name == tool_name]

        # Return most recent first
        logs = list(reversed(logs))

        if limit:
            logs = logs[:limit]

        return logs

    def clear_logs(self, agent_name: Optional[str] = None) -> int:
        """
        Clear tool call logs.

        Args:
            agent_name: If provided, only clear logs for this agent.
                       Otherwise, clear all logs.

        Returns:
            Number of entries cleared
        """
        if agent_name:
            original_count = len(self._logs)
            self._logs = [log for log in self._logs if log.agent_name != agent_name]
            cleared = original_count - len(self._logs)
        else:
            cleared = len(self._logs)
            self._logs = []

        logger.debug(f"Cleared {cleared} tool call logs")
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about logged tool calls.

        Returns:
            Dictionary containing statistics
        """
        if not self._logs:
            return {
                "total_calls": 0,
                "by_agent": {},
                "by_tool": {},
                "success_rate": 0.0,
            }

        # Count by agent
        by_agent: Dict[str, int] = {}
        for log in self._logs:
            by_agent[log.agent_name] = by_agent.get(log.agent_name, 0) + 1

        # Count by tool
        by_tool: Dict[str, int] = {}
        for log in self._logs:
            by_tool[log.tool_name] = by_tool.get(log.tool_name, 0) + 1

        # Success rate
        successful = sum(1 for log in self._logs if log.success)
        success_rate = successful / len(self._logs) if self._logs else 0.0

        # Average duration
        durations = [log.duration_ms for log in self._logs if log.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_calls": len(self._logs),
            "by_agent": by_agent,
            "by_tool": by_tool,
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration,
        }

    def get_recent_logs(self, count: int = 10) -> List[ToolCall]:
        """
        Get the most recent tool call logs.

        Args:
            count: Number of recent entries to return

        Returns:
            List of recent ToolCall entries
        """
        return list(reversed(self._logs[-count:]))

    def __len__(self) -> int:
        """Return the number of logged tool calls."""
        return len(self._logs)


# Singleton instance
_tool_call_logger: Optional[ToolCallLogger] = None


def get_tool_call_logger() -> ToolCallLogger:
    """
    Get the singleton ToolCallLogger instance.

    Returns:
        ToolCallLogger instance
    """
    global _tool_call_logger
    if _tool_call_logger is None:
        _tool_call_logger = ToolCallLogger()
    return _tool_call_logger


def reset_tool_call_logger() -> None:
    """Reset the singleton logger (useful for testing)."""
    global _tool_call_logger
    _tool_call_logger = None


__all__ = [
    "ToolCall",
    "ToolCallLogger",
    "get_tool_call_logger",
    "reset_tool_call_logger",
]
