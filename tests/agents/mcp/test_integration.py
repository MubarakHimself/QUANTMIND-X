"""
Unit Tests for MCP Integration Module

Tests MCP integration with agent spawner.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.agents.mcp.integration import (
    MCPAgentConfig,
    MCPIntegration,
    get_mcp_integration,
)


class TestMCPAgentConfig:
    """Tests for MCPAgentConfig dataclass."""

    def test_create_config(self):
        """Test creating MCP agent config."""
        config = MCPAgentConfig(
            agent_id="agent-1",
            agent_type="coder",
            mcp_servers=["filesystem", "context7"],
        )

        assert config.agent_id == "agent-1"
        assert config.agent_type == "coder"
        assert config.mcp_servers == ["filesystem", "context7"]
        assert config.auto_load_tools is True

    def test_config_defaults(self):
        """Test config default values."""
        config = MCPAgentConfig(
            agent_id="test",
            agent_type="test",
        )

        assert config.mcp_servers == []
        assert config.auto_load_tools is True
        assert config.tool_timeout_seconds == 30


class TestMCPIntegration:
    """Tests for MCPIntegration class."""

    @pytest.fixture
    def mock_discovery(self):
        """Create mock discovery."""
        discovery = Mock()
        discovery.get_enabled_servers = Mock(return_value=[])
        return discovery

    @pytest.fixture
    def mock_sse(self):
        """Create mock SSE handler."""
        sse = Mock()
        sse.start = AsyncMock()
        sse.stop = AsyncMock()
        sse.stream_tool_events = Mock(return_value=AsyncIterator())
        return sse

    @pytest.fixture
    def mock_loader(self):
        """Create mock tool loader."""
        loader = Mock()
        loader.initialize = AsyncMock()
        loader.get_tools = AsyncMock(return_value=[])
        loader.list_servers_with_tools = Mock(return_value=[])
        loader.list_all_tools = Mock(return_value=[])
        loader.get_tool_stats = Mock(return_value={
            "total_tools": 0,
            "total_calls": 0,
        })
        return loader

    @pytest.mark.asyncio
    async def test_initialization(self, mock_discovery, mock_sse, mock_loader):
        """Test integration initialization."""
        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()
                    await integration.initialize()

                    assert integration._initialized is True
                    mock_loader.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_agent_tools(self, mock_discovery, mock_sse, mock_loader):
        """Test getting tools for an agent."""
        mock_discovery.get_enabled_servers = Mock(return_value=[
            Mock(server_id="filesystem"),
        ])

        mock_loader.get_tools = AsyncMock(return_value=[
            Mock(name="read_file", description="Read a file"),
            Mock(name="write_file", description="Write a file"),
        ])

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()
                    integration._initialized = True
                    integration._tool_loader = mock_loader

                    tools = await integration.get_agent_tools("agent-1", ["filesystem"])

                    assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_get_agent_tools_auto_discover(self, mock_discovery, mock_sse, mock_loader):
        """Test auto-discovering tools for agent."""
        mock_discovery.get_enabled_servers = Mock(return_value=[
            Mock(server_id="filesystem"),
            Mock(server_id="context7"),
        ])

        mock_loader.get_tools = AsyncMock(return_value=[])

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()
                    integration._initialized = True
                    integration._tool_loader = mock_loader

                    # No servers specified - should use all enabled
                    tools = await integration.get_agent_tools("agent-1")

                    mock_loader.get_tools.assert_called()

    def test_get_agent_tool_names(self, mock_discovery, mock_sse, mock_loader):
        """Test getting tool names for an agent."""
        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()
                    integration._agent_tools["agent-1"] = ["tool1", "tool2"]

                    names = integration.get_agent_tool_names("agent-1")

                    assert names == ["tool1", "tool2"]

    def test_get_agent_tool_names_not_found(self, mock_discovery, mock_sse, mock_loader):
        """Test getting tools for unknown agent."""
        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    names = integration.get_agent_tool_names("unknown-agent")

                    assert names == []

    def test_get_integration_stats(self, mock_discovery, mock_sse, mock_loader):
        """Test getting integration statistics."""
        mock_discovery.get_enabled_servers = Mock(return_value=[
            Mock(server_id="fs"),
        ])
        mock_loader.list_servers_with_tools = Mock(return_value=["fs"])
        mock_loader.list_all_tools = Mock(return_value=[Mock(), Mock()])
        mock_loader.get_tool_stats = Mock(return_value={
            "total_calls": 10,
        })

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    stats = integration.get_integration_stats()

                    assert stats["servers_discovered"] == 1
                    assert stats["servers_with_tools"] == 1
                    assert stats["tools_loaded"] == 2

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_discovery, mock_sse, mock_loader):
        """Test integration shutdown."""
        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    await integration.shutdown()

                    mock_sse.stop.assert_called_once()


class TestGetMcpIntegration:
    """Tests for get_mcp_integration function."""

    def test_get_mcp_integration_singleton(self):
        """Test that get_mcp_integration returns singleton."""
        # Reset global
        import src.agents.mcp.integration as integration_module
        integration_module._mcp_integration = None

        integration1 = get_mcp_integration()
        integration2 = get_mcp_integration()

        assert integration1 is integration2

        # Cleanup
        integration_module._mcp_integration = None


class TestIntegrationProperties:
    """Tests for integration properties."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        mock_discovery = Mock()
        mock_sse = Mock()
        mock_loader = Mock()

        return mock_discovery, mock_sse, mock_loader

    def test_discovery_property(self, mock_components):
        """Test discovery property."""
        mock_discovery, mock_sse, mock_loader = mock_components

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    assert integration.discovery is mock_discovery

    def test_sse_handler_property(self, mock_components):
        """Test SSE handler property."""
        mock_discovery, mock_sse, mock_loader = mock_components

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    assert integration.sse_handler is mock_sse

    def test_tool_loader_property(self, mock_components):
        """Test tool loader property."""
        mock_discovery, mock_sse, mock_loader = mock_components

        with patch("src.agents.mcp.integration.get_mcp_discovery", return_value=mock_discovery):
            with patch("src.agents.mcp.integration.get_sse_handler", return_value=mock_sse):
                with patch("src.agents.mcp.integration.get_tool_loader", return_value=mock_loader):
                    integration = MCPIntegration()

                    assert integration.tool_loader is mock_loader
