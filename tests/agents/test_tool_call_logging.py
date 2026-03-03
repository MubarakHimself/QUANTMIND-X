"""
Tests for Tool Call Logging

Tests the ToolCallLogger functionality including:
- Logging tool calls
- Retrieving logs with filtering
- Statistics calculation
- Clearing logs
"""

import pytest
from datetime import datetime

from src.agents.departments.tool_call_logger import (
    ToolCall,
    ToolCallLogger,
    get_tool_call_logger,
    reset_tool_call_logger,
)


class TestToolCall:
    """Tests for the ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall entry."""
        timestamp = datetime.now()
        tool_call = ToolCall(
            timestamp=timestamp,
            agent_name="research_head",
            tool_name="memory_tools",
            input_params={"key": "test"},
            result={"success": True},
            success=True,
            duration_ms=100.5,
        )

        assert tool_call.timestamp == timestamp
        assert tool_call.agent_name == "research_head"
        assert tool_call.tool_name == "memory_tools"
        assert tool_call.input_params == {"key": "test"}
        assert tool_call.result == {"success": True}
        assert tool_call.success is True
        assert tool_call.duration_ms == 100.5

    def test_tool_call_defaults(self):
        """Test ToolCall with default values."""
        tool_call = ToolCall(
            timestamp=datetime.now(),
            agent_name="trading_head",
            tool_name="broker_tools",
        )

        assert tool_call.input_params == {}
        assert tool_call.result is None
        assert tool_call.success is True
        assert tool_call.duration_ms is None


class TestToolCallLogger:
    """Tests for the ToolCallLogger class."""

    def setup_method(self):
        """Reset the logger before each test."""
        reset_tool_call_logger()

    def teardown_method(self):
        """Clean up after each test."""
        reset_tool_call_logger()

    def test_logger_initialization(self):
        """Test creating a new logger instance."""
        logger = ToolCallLogger()
        assert len(logger) == 0

    def test_log_call(self):
        """Test logging a tool call."""
        logger = ToolCallLogger()
        logger.log_call(
            agent_name="research_head",
            tool_name="memory_tools",
            input_params={"key": "test", "value": "data"},
            success=True,
        )

        assert len(logger) == 1

    def test_log_call_with_result(self):
        """Test logging a tool call with result."""
        logger = ToolCallLogger()
        logger.log_call(
            agent_name="risk_head",
            tool_name="risk_tools",
            input_params={"symbol": "BTCUSD"},
            result={"position_size": 0.1},
            success=True,
            duration_ms=50.0,
        )

        logs = logger.get_logs()
        assert len(logs) == 1
        assert logs[0].result == {"position_size": 0.1}
        assert logs[0].duration_ms == 50.0

    def test_get_logs_all(self):
        """Test retrieving all logs."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("trading_head", "broker_tools")
        logger.log_call("risk_head", "risk_tools")

        logs = logger.get_logs()
        assert len(logs) == 3

    def test_get_logs_filter_by_agent(self):
        """Test filtering logs by agent name."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("trading_head", "broker_tools")
        logger.log_call("research_head", "knowledge_tools")

        research_logs = logger.get_logs(agent_name="research_head")
        assert len(research_logs) == 2

        trading_logs = logger.get_logs(agent_name="trading_head")
        assert len(trading_logs) == 1

    def test_get_logs_filter_by_tool(self):
        """Test filtering logs by tool name."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("trading_head", "memory_tools")
        logger.log_call("risk_head", "risk_tools")

        memory_logs = logger.get_logs(tool_name="memory_tools")
        assert len(memory_logs) == 2

    def test_get_logs_filter_by_both(self):
        """Test filtering logs by both agent and tool."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("research_head", "knowledge_tools")
        logger.log_call("trading_head", "memory_tools")

        logs = logger.get_logs(agent_name="research_head", tool_name="memory_tools")
        assert len(logs) == 1

    def test_get_logs_with_limit(self):
        """Test limiting number of returned logs."""
        logger = ToolCallLogger()
        for i in range(10):
            logger.log_call(f"agent_{i}", f"tool_{i}")

        logs = logger.get_logs(limit=5)
        assert len(logs) == 5

    def test_get_logs_returns_recent_first(self):
        """Test that logs are returned with most recent first."""
        logger = ToolCallLogger()
        logger.log_call("first", "tool_a")
        logger.log_call("second", "tool_b")
        logger.log_call("third", "tool_c")

        logs = logger.get_logs()
        assert logs[0].agent_name == "third"
        assert logs[1].agent_name == "second"
        assert logs[2].agent_name == "first"

    def test_clear_all_logs(self):
        """Test clearing all logs."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("trading_head", "broker_tools")

        cleared = logger.clear_logs()
        assert cleared == 2
        assert len(logger) == 0

    def test_clear_logs_by_agent(self):
        """Test clearing logs for a specific agent."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools")
        logger.log_call("trading_head", "broker_tools")
        logger.log_call("research_head", "knowledge_tools")

        cleared = logger.clear_logs(agent_name="research_head")
        assert cleared == 2
        assert len(logger) == 1

    def test_get_stats_empty(self):
        """Test statistics with no logs."""
        logger = ToolCallLogger()
        stats = logger.get_stats()

        assert stats["total_calls"] == 0
        assert stats["by_agent"] == {}
        assert stats["by_tool"] == {}
        assert stats["success_rate"] == 0.0

    def test_get_stats(self):
        """Test statistics calculation."""
        logger = ToolCallLogger()
        logger.log_call("research_head", "memory_tools", success=True)
        logger.log_call("research_head", "knowledge_tools", success=True)
        logger.log_call("trading_head", "broker_tools", success=False)

        stats = logger.get_stats()

        assert stats["total_calls"] == 3
        assert stats["by_agent"]["research_head"] == 2
        assert stats["by_agent"]["trading_head"] == 1
        assert stats["by_tool"]["memory_tools"] == 1
        assert stats["success_rate"] == pytest.approx(2/3)
        assert "avg_duration_ms" in stats

    def test_get_recent_logs(self):
        """Test getting recent logs."""
        logger = ToolCallLogger()
        for i in range(15):
            logger.log_call(f"agent_{i}", f"tool_{i}")

        recent = logger.get_recent_logs(5)
        assert len(recent) == 5
        assert recent[0].agent_name == "agent_14"
        assert recent[4].agent_name == "agent_10"

    def test_max_entries(self):
        """Test that logger respects max_entries limit."""
        logger = ToolCallLogger(max_entries=5)
        for i in range(10):
            logger.log_call(f"agent_{i}", f"tool_{i}")

        assert len(logger) == 5
        # Should have the last 5 entries
        logs = logger.get_logs()
        assert logs[0].agent_name == "agent_9"

    def test_singleton_getter(self):
        """Test the singleton get_tool_call_logger function."""
        logger1 = get_tool_call_logger()
        logger2 = get_tool_call_logger()

        assert logger1 is logger2

    def test_reset_singleton(self):
        """Test resetting the singleton."""
        logger1 = get_tool_call_logger()
        logger1.log_call("test", "test")

        reset_tool_call_logger()

        logger2 = get_tool_call_logger()
        assert logger2 is not logger1
        assert len(logger2) == 0


class TestToolCallLoggerIntegration:
    """Integration tests for tool call logging with department agents."""

    def setup_method(self):
        """Reset before each test."""
        reset_tool_call_logger()

    def teardown_method(self):
        """Clean up after each test."""
        reset_tool_call_logger()

    def test_log_multiple_department_calls(self):
        """Test logging calls from multiple departments."""
        logger = get_tool_call_logger()

        # Simulate research department calls
        logger.log_call("research_head", "memory_tools", {"key": "strategy1"})
        logger.log_call("research_head", "knowledge_tools", {"query": "patterns"})

        # Simulate trading department calls
        logger.log_call("trading_head", "broker_tools", {"symbol": "BTCUSD"})

        # Simulate risk department calls
        logger.log_call("risk_head", "risk_tools", {"position": 0.1})

        stats = logger.get_stats()
        assert stats["total_calls"] == 4
        assert stats["by_agent"]["research_head"] == 2
        assert stats["by_agent"]["trading_head"] == 1
        assert stats["by_agent"]["risk_head"] == 1
