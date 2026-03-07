"""
Unit Tests for MCP Dynamic Tool Loader

Tests dynamic tool loading from MCP servers.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.mcp.loader import (
    ToolDefinition,
    ToolExecution,
    DynamicToolLoader,
    get_tool_loader,
)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_create_definition(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_id="filesystem",
        )

        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert tool.server_id == "filesystem"
        assert tool.call_count == 0
        assert tool.loaded_at is not None

    def test_update_stats(self):
        """Test updating tool statistics."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            input_schema={},
            server_id="test",
        )

        tool.call_count = 1
        tool.last_called = datetime.now()

        assert tool.call_count == 1
        assert tool.last_called is not None


class TestToolExecution:
    """Tests for ToolExecution dataclass."""

    def test_create_execution(self):
        """Test creating a tool execution."""
        execution = ToolExecution(
            call_id="call-123",
            tool_name="read_file",
            server_id="filesystem",
            arguments={"path": "/tmp/test.txt"},
        )

        assert execution.call_id == "call-123"
        assert execution.tool_name == "read_file"
        assert execution.started_at is not None
        assert execution.completed_at is None
        assert execution.result is None

    def test_execution_completion(self):
        """Test marking execution as complete."""
        execution = ToolExecution(
            call_id="call-123",
            tool_name="test",
            server_id="test",
            arguments={},
        )

        execution.result = {"success": True}
        execution.completed_at = datetime.now()

        assert execution.result is not None
        assert execution.completed_at is not None


class TestDynamicToolLoader:
    """Tests for DynamicToolLoader class."""

    @pytest.fixture
    def mock_discovery(self):
        """Create mock discovery."""
        discovery = Mock()
        discovery.discover_all = AsyncMock(return_value=[])
        discovery.get_enabled_servers = Mock(return_value=[])
        return discovery

    @pytest.fixture
    def mock_sse(self):
        """Create mock SSE handler."""
        sse = Mock()
        sse.start = AsyncMock()
        sse.stop = AsyncMock()
        sse.publish_tool_discovered = AsyncMock()
        sse.publish_tool_called = AsyncMock()
        sse.publish_tool_result = AsyncMock()
        sse.publish_connection_event = AsyncMock()
        return sse

    @pytest.mark.asyncio
    async def test_initialization(self, mock_discovery, mock_sse):
        """Test loader initialization."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        await loader.initialize()

        assert loader._initialized is True
        mock_discovery.discover_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tools_cached(self, mock_discovery, mock_sse):
        """Test getting tools from cache."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        # Pre-populate cache
        tool = ToolDefinition(
            name="cached_tool",
            description="Cached",
            input_schema={},
            server_id="test",
        )
        loader._tools["test"] = {"cached_tool": tool}

        tools = await loader.get_tools("test")

        assert len(tools) == 1
        assert tools[0].name == "cached_tool"

    @pytest.mark.asyncio
    async def test_list_all_tools(self, mock_discovery, mock_sse):
        """Test listing all loaded tools."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        # Add tools to multiple servers
        loader._tools["server1"] = {
            "tool1": ToolDefinition("tool1", "Desc1", {}, "server1"),
            "tool2": ToolDefinition("tool2", "Desc2", {}, "server1"),
        }
        loader._tools["server2"] = {
            "tool3": ToolDefinition("tool3", "Desc3", {}, "server2"),
        }

        all_tools = loader.list_all_tools()

        assert len(all_tools) == 3

    @pytest.mark.asyncio
    async def test_list_servers_with_tools(self, mock_discovery, mock_sse):
        """Test listing servers with tools."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        loader._tools["server1"] = {"tool1": Mock()}
        loader._tools["server2"] = {"tool2": Mock()}

        servers = loader.list_servers_with_tools()

        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    def test_get_tool_stats(self, mock_discovery, mock_sse):
        """Test getting tool statistics."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        # Add tools with call counts
        tool1 = ToolDefinition("tool1", "Desc1", {}, "server1")
        tool1.call_count = 5
        tool2 = ToolDefinition("tool2", "Desc2", {}, "server1")
        tool2.call_count = 3

        loader._tools["server1"] = {"tool1": tool1, "tool2": tool2}

        stats = loader.get_tool_stats()

        assert stats["total_servers"] == 1
        assert stats["total_tools"] == 2
        assert stats["total_calls"] == 8
        assert stats["servers"]["server1"]["tools_count"] == 2
        assert stats["servers"]["server1"]["total_calls"] == 8

    @pytest.mark.asyncio
    async def test_get_execution(self, mock_discovery, mock_sse):
        """Test getting execution by call ID."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        execution = ToolExecution(
            call_id="call-123",
            tool_name="test",
            server_id="test",
            arguments={},
        )
        loader._executions["call-123"] = execution

        result = await loader.get_execution("call-123")

        assert result is not None
        assert result.call_id == "call-123"

    @pytest.mark.asyncio
    async def test_get_execution_not_found(self, mock_discovery, mock_sse):
        """Test getting non-existent execution."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        result = await loader.get_execution("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_tools(self, mock_discovery, mock_sse):
        """Test refreshing tools from a server."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        # Add cached tools
        loader._tools["test"] = {"old_tool": ToolDefinition("old", "Old", {}, "test")}

        # Mock MCP client
        mock_tool = Mock()
        mock_tool.name = "new_tool"
        mock_tool.description = "New"
        mock_tool.input_schema = {}

        loader._mcp_client = Mock()
        loader._mcp_client.get_status = Mock(return_value=Mock(value="disconnected"))
        loader._mcp_client.connect = AsyncMock(return_value=True)
        loader._mcp_client.list_tools = AsyncMock(return_value=[mock_tool])

        await loader.refresh_tools("test")

        # Old tools should be removed
        assert "old_tool" not in loader._tools.get("test", {})

    @pytest.mark.asyncio
    async def test_disconnect_server(self, mock_discovery, mock_sse):
        """Test disconnecting from a server."""
        loader = DynamicToolLoader(mock_discovery, mock_sse)

        # Add cached tools
        loader._tools["test"] = {"tool": ToolDefinition("tool", "Desc", {}, "test")}

        # Mock MCP client
        loader._mcp_client = Mock()
        loader._mcp_client.disconnect = AsyncMock()

        await loader.disconnect_server("test")

        # Tools should be cleared
        assert "test" not in loader._tools


class TestToolLoaderIntegration:
    """Integration tests for tool loader with mocked MCP client."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        # Reset global
        import src.agents.mcp.loader as loader_module
        loader_module._tool_loader = None

        # Create mock components
        mock_discovery = Mock()
        mock_discovery.discover_all = AsyncMock(return_value=[])
        mock_discovery.get_enabled_servers = Mock(return_value=[])

        mock_sse = Mock()
        mock_sse.start = AsyncMock()
        mock_sse.stop = AsyncMock()
        mock_sse.publish_tool_called = AsyncMock()
        mock_sse.publish_tool_result = AsyncMock()

        # Create mock MCP client
        mock_client = Mock()
        mock_client.get_status = Mock(return_value=Mock(value="connected"))
        mock_client.call_tool = AsyncMock(return_value={"result": "success"})

        with patch("src.agents.mcp.loader.get_mcp_client", return_value=mock_client):
            loader = DynamicToolLoader(mock_discovery, mock_sse)
            loader._mcp_client = mock_client
            loader._tools["test"] = {
                "test_tool": ToolDefinition("test_tool", "Test", {}, "test")
            }

            result = await loader.call_tool("test", "test_tool", {"arg": "value"})

            assert result == {"result": "success"}

        # Cleanup
        loader_module._tool_loader = None

    @pytest.mark.asyncio
    async def test_call_tool_fallback(self):
        """Test tool call fallback when no MCP client."""
        # Reset global
        import src.agents.mcp.loader as loader_module
        loader_module._tool_loader = None

        mock_discovery = Mock()
        mock_discovery.discover_all = AsyncMock(return_value=[])
        mock_discovery.get_enabled_servers = Mock(return_value=[])

        mock_sse = Mock()
        mock_sse.start = AsyncMock()
        mock_sse.stop = AsyncMock()
        mock_sse.publish_tool_called = AsyncMock()
        mock_sse.publish_tool_result = AsyncMock()

        loader = DynamicToolLoader(mock_discovery, mock_sse)
        loader._mcp_client = None  # No client
        loader._tools["test"] = {
            "test_tool": ToolDefinition("test_tool", "Test", {}, "test")
        }

        result = await loader.call_tool("test", "test_tool", {})

        # Should return mock result
        assert "mock" in result["status"]

        # Cleanup
        loader_module._tool_loader = None


class TestGetToolLoader:
    """Tests for get_tool_loader function."""

    def test_get_tool_loader_singleton(self):
        """Test that get_tool_loader returns singleton."""
        # Reset global
        import src.agents.mcp.loader as loader_module
        loader_module._tool_loader = None

        loader1 = get_tool_loader()
        loader2 = get_tool_loader()

        assert loader1 is loader2

        # Cleanup
        loader_module._tool_loader = None
