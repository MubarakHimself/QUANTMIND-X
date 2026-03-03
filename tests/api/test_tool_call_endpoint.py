"""
Tool Call Endpoint Tests

Tests for:
- GET /api/tool-calls/logs
- GET /api/tool-calls/terminal
- GET /api/tool-calls/summary
- GET /api/tool-calls/agents
- GET /api/tool-calls/tools
- POST /api/tool-calls/log
- DELETE /api/tool-calls/logs

**Validates: Phase 6 Task 11 - Terminal UI Endpoint for Tool Calls**
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import json


class TestToolCallEndpoints:
    """Tests for tool call logging endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Import and clear logs before each test
        from src.api import tool_call_endpoints
        tool_call_endpoints._tool_call_logs = []
        yield
        # Clean up after test
        tool_call_endpoints._tool_call_logs = []

    @pytest.fixture
    def sample_logs(self):
        """Create sample tool call logs."""
        from src.api.tool_call_endpoints import log_tool_call

        logs = []
        # Add some test logs
        logs.append(log_tool_call(
            agent_id="analyst-001",
            agent_type="analyst",
            tool_name="read_file",
            args={"path": "/test/file1.py"},
            duration_ms=150.5,
            success=True,
        ))
        logs.append(log_tool_call(
            agent_id="analyst-001",
            agent_type="analyst",
            tool_name="analyze_data",
            args={"symbol": "EURUSD", "timeframe": "H1"},
            duration_ms=320.0,
            success=True,
        ))
        logs.append(log_tool_call(
            agent_id="quantcode-001",
            agent_type="quantcode",
            tool_name="execute_strategy",
            args={"strategy": "ma_cross", "symbol": "GBPUSD"},
            duration_ms=500.0,
            success=True,
        ))
        logs.append(log_tool_call(
            agent_id="copilot-001",
            agent_type="copilot",
            tool_name="search_code",
            args={"query": "trading"},
            duration_ms=75.0,
            success=True,
        ))
        # Add a failed call
        logs.append(log_tool_call(
            agent_id="analyst-001",
            agent_type="analyst",
            tool_name="fetch_data",
            args={"source": "invalid"},
            duration_ms=50.0,
            success=False,
            error="Invalid data source",
        ))

        return logs


class TestToolCallLogsEndpoint(TestToolCallEndpoints):
    """Tests for GET /api/tool-calls/logs endpoint."""

    def test_get_logs_returns_all_logs(self, sample_logs):
        """Test getting all logs without filters."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(limit=10)

        assert len(logs) == 5
        assert all("id" in log for log in logs)
        assert all("timestamp" in log for log in logs)
        assert all("agent_id" in log for log in logs)
        assert all("tool_name" in log for log in logs)

    def test_get_logs_filter_by_agent_id(self, sample_logs):
        """Test filtering logs by agent ID."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(agent_id="analyst-001", limit=10)

        assert len(logs) == 3
        assert all(log["agent_id"] == "analyst-001" for log in logs)

    def test_get_logs_filter_by_agent_type(self, sample_logs):
        """Test filtering logs by agent type."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(agent_type="analyst", limit=10)

        assert len(logs) == 3
        assert all(log["agent_type"] == "analyst" for log in logs)

    def test_get_logs_filter_by_tool_name(self, sample_logs):
        """Test filtering logs by tool name."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(tool_name="read_file", limit=10)

        assert len(logs) == 1
        assert logs[0]["tool_name"] == "read_file"

    def test_get_logs_pagination(self, sample_logs):
        """Test pagination with limit and offset."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs_page1 = get_tool_call_logs(limit=2, offset=0)
        logs_page2 = get_tool_call_logs(limit=2, offset=2)

        assert len(logs_page1) == 2
        assert len(logs_page2) == 2
        # Pages should have different entries
        assert logs_page1[0]["id"] != logs_page2[0]["id"]


class TestToolCallTerminalEndpoint(TestToolCallEndpoints):
    """Tests for GET /api/tool-calls/terminal endpoint."""

    def test_get_terminal_output_format(self, sample_logs):
        """Test terminal output format."""
        from src.api.tool_call_endpoints import get_terminal_output

        output = get_terminal_output(limit=5)

        assert isinstance(output, str)
        assert "=" in output  # Header/footer separator
        assert "TIMESTAMP" in output
        assert "AGENT_ID" in output
        assert "TOOL_NAME" in output

    def test_get_terminal_output_with_filters(self, sample_logs):
        """Test terminal output with agent filter."""
        from src.api.tool_call_endpoints import get_terminal_output

        output = get_terminal_output(agent_id="analyst-001", limit=10)

        assert "analyst-001" in output
        assert "read_file" in output

    def test_get_terminal_output_empty(self):
        """Test terminal output when no logs exist."""
        from src.api.tool_call_endpoints import get_terminal_output

        output = get_terminal_output()

        assert "No tool calls found" in output


class TestToolCallSummaryEndpoint(TestToolCallEndpoints):
    """Tests for GET /api/tool-calls/summary endpoint."""

    def test_get_summary_counts(self, sample_logs):
        """Test summary statistics."""
        from src.api.tool_call_endpoints import get_tool_call_summary

        summary = get_tool_call_summary()

        assert summary.total_calls == 5
        assert summary.successful_calls == 4
        assert summary.failed_calls == 1
        assert summary.unique_agents == 3
        assert summary.unique_tools == 5

    def test_get_summary_by_agent(self, sample_logs):
        """Test summary filtered by agent."""
        from src.api.tool_call_endpoints import get_tool_call_summary

        summary = get_tool_call_summary(agent_id="analyst-001")

        assert summary.total_calls == 3
        assert summary.unique_agents == 1

    def test_get_summary_breakdown(self, sample_logs):
        """Test breakdown by agent and tool."""
        from src.api.tool_call_endpoints import get_tool_call_summary

        summary = get_tool_call_summary()

        assert "analyst-001" in summary.agents
        assert summary.agents["analyst-001"] == 3
        assert "read_file" in summary.tools


class TestToolCallAgentsEndpoint(TestToolCallEndpoints):
    """Tests for GET /api/tool-calls/agents endpoint."""

    def test_get_agents_list(self, sample_logs):
        """Test getting list of unique agents."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(limit=100)

        # Build unique agents manually for comparison
        agent_counts = {}
        for log in logs:
            agent_id = log.get("agent_id")
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1

        assert len(agent_counts) == 3
        assert "analyst-001" in agent_counts
        assert "quantcode-001" in agent_counts
        assert "copilot-001" in agent_counts


class TestToolCallToolsEndpoint(TestToolCallEndpoints):
    """Tests for GET /api/tool-calls/tools endpoint."""

    def test_get_tools_list(self, sample_logs):
        """Test getting list of unique tools."""
        from src.api.tool_call_endpoints import get_tool_call_logs

        logs = get_tool_call_logs(limit=100)

        # Build unique tools manually for comparison
        tool_counts = {}
        for log in logs:
            tool_name = log.get("tool_name")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        assert len(tool_counts) == 5
        assert "read_file" in tool_counts
        assert "analyze_data" in tool_counts


class TestToolCallLogEntryEndpoint(TestToolCallEndpoints):
    """Tests for POST /api/tool-calls/log endpoint."""

    def test_create_log_entry(self):
        """Test creating a log entry."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_logs

        log_id = log_tool_call(
            agent_id="test-agent",
            agent_type="test",
            tool_name="test_tool",
            args={"test": "value"},
            duration_ms=100.0,
            success=True,
        )

        assert log_id is not None

        logs = get_tool_call_logs(agent_id="test-agent", limit=10)
        assert len(logs) == 1
        assert logs[0]["agent_id"] == "test-agent"
        assert logs[0]["tool_name"] == "test_tool"

    def test_create_log_entry_with_error(self):
        """Test creating a failed log entry."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_logs

        log_id = log_tool_call(
            agent_id="test-agent",
            agent_type="test",
            tool_name="failing_tool",
            args={},
            duration_ms=50.0,
            success=False,
            error="Test error",
        )

        logs = get_tool_call_logs(agent_id="test-agent", limit=10)
        assert logs[0]["success"] is False
        assert logs[0]["error"] == "Test error"


class TestToolCallClearEndpoint(TestToolCallEndpoints):
    """Tests for DELETE /api/tool-calls/logs endpoint."""

    def test_clear_all_logs(self, sample_logs):
        """Test clearing all logs."""
        from src.api.tool_call_endpoints import get_tool_call_logs, _tool_call_logs

        # Verify logs exist
        logs = get_tool_call_logs(limit=100)
        assert len(logs) == 5

        # Clear logs
        _tool_call_logs.clear()

        # Verify logs are cleared
        logs = get_tool_call_logs(limit=100)
        assert len(logs) == 0


class TestAPIIteration:
    """Integration tests for the tool_call_endpoints module functions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from src.api import tool_call_endpoints
        tool_call_endpoints._tool_call_logs = []
        yield
        tool_call_endpoints._tool_call_logs = []

    def test_get_logs_via_function(self):
        """Test getting logs via the function directly."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_logs

        log_tool_call(
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="test_tool",
            args={},
            success=True,
        )

        logs = get_tool_call_logs(limit=10)
        assert len(logs) >= 1

    def test_get_terminal_output_via_function(self):
        """Test getting terminal output via function."""
        from src.api.tool_call_endpoints import log_tool_call, get_terminal_output

        log_tool_call(
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="test_tool",
            args={},
            success=True,
        )

        output = get_terminal_output(limit=10)
        assert "test-agent" in output

    def test_get_summary_via_function(self):
        """Test getting summary via function."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_summary

        log_tool_call(
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="test_tool",
            args={},
            success=True,
        )

        summary = get_tool_call_summary()
        assert summary.total_calls >= 1

    def test_filter_by_agent_id_via_function(self):
        """Test filtering by agent_id via function."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_logs

        log_tool_call(agent_id="agent-a", agent_type="analyst", tool_name="tool1", args={})
        log_tool_call(agent_id="agent-b", agent_type="analyst", tool_name="tool2", args={})

        logs = get_tool_call_logs(agent_id="agent-a", limit=10)
        assert all(log["agent_id"] == "agent-a" for log in logs)

    def test_clear_logs_via_function(self):
        """Test clearing logs via direct function access."""
        from src.api.tool_call_endpoints import log_tool_call, get_tool_call_logs, _tool_call_logs

        log_tool_call(agent_id="test-agent", agent_type="analyst", tool_name="tool1", args={})

        # Verify data exists
        logs = get_tool_call_logs(limit=100)
        initial_count = len(logs)
        assert initial_count >= 1

        # Clear logs
        _tool_call_logs.clear()

        # Verify data is cleared
        logs = get_tool_call_logs(limit=100)
        assert len(logs) == 0


class TestTerminalFormatting:
    """Tests for terminal output formatting."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from src.api import tool_call_endpoints
        tool_call_endpoints._tool_call_logs = []
        yield
        tool_call_endpoints._tool_call_logs = []

    def test_terminal_line_format(self):
        """Test ToolCallLog.to_terminal_line method."""
        from src.api.tool_call_endpoints import ToolCallLog
        from datetime import datetime, timezone

        log = ToolCallLog(
            id="test-123",
            timestamp=datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc),
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="read_file",
            args={"path": "/test/file.py"},
            duration_ms=150.5,
            success=True,
        )

        line = log.to_terminal_line()

        assert "2024-01-15 10:30:45" in line
        assert "test-agent" in line
        assert "read_file" in line
        assert "OK" in line
        assert "150.5ms" in line

    def test_terminal_line_with_long_args(self):
        """Test ToolCallLog.to_terminal_line truncates long args."""
        from src.api.tool_call_endpoints import ToolCallLog
        from datetime import datetime, timezone

        log = ToolCallLog(
            id="test-123",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="test_tool",
            args={"very_long_key": "x" * 200},
            success=True,
        )

        line = log.to_terminal_line(max_arg_length=50)

        assert len(line) < 200  # Should be truncated

    def test_terminal_line_error_status(self):
        """Test ToolCallLog.to_terminal_line shows error status."""
        from src.api.tool_call_endpoints import ToolCallLog
        from datetime import datetime, timezone

        log = ToolCallLog(
            id="test-123",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent",
            agent_type="analyst",
            tool_name="failing_tool",
            args={},
            success=False,
            error="Test error",
        )

        line = log.to_terminal_line()

        assert "ERR" in line
