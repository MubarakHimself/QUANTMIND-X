"""
Unit Tests for MCP Discovery Module

Tests MCP server discovery from various sources.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.agents.mcp.discovery import (
    MCPServerDiscovery,
    DiscoveredServer,
    DiscoverySource,
    get_mcp_discovery,
)


class TestDiscoveredServer:
    """Tests for DiscoveredServer dataclass."""

    def test_create_server(self):
        """Test creating a discovered server."""
        server = DiscoveredServer(
            server_id="test-server",
            name="Test Server",
            description="A test MCP server",
            source=DiscoverySource.DEFAULT,
            command="npx",
            args=["-y", "@test/mcp-server"],
        )

        assert server.server_id == "test-server"
        assert server.name == "Test Server"
        assert server.enabled is True
        assert server.auto_connect is False

    def test_server_defaults(self):
        """Test server default values."""
        server = DiscoveredServer(
            server_id="minimal",
            name="Minimal",
        )

        assert server.description == ""
        assert server.source == DiscoverySource.DEFAULT
        assert server.transport == "stdio"
        assert server.args == []
        assert server.env == {}


class TestMCPServerDiscovery:
    """Tests for MCPServerDiscovery class."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def discovery(self, temp_config_dir):
        """Create a discovery instance with temp directory."""
        return MCPServerDiscovery(project_root=temp_config_dir)

    def test_initialization(self, discovery):
        """Test discovery initialization."""
        assert discovery is not None
        assert discovery._project_root is not None

    @pytest.mark.asyncio
    async def test_discover_from_defaults(self, discovery):
        """Test discovering default servers."""
        await discovery.discover_all()

        # Should have default servers
        servers = discovery.get_all_servers()
        assert len(servers) > 0

        # Check for known default servers
        server_ids = [s.server_id for s in servers]
        assert "filesystem" in server_ids
        assert "context7" in server_ids
        assert "sequential_thinking" in server_ids

    @pytest.mark.asyncio
    async def test_discover_from_config_file(self, temp_config_dir, discovery):
        """Test discovering servers from config file."""
        config_path = temp_config_dir / "mcp_config.json"
        config_data = {
            "mcpServers": {
                "custom-server": {
                    "name": "Custom Server",
                    "description": "A custom MCP server",
                    "command": "node",
                    "args": ["./server.js"],
                    "env": {"DEBUG": "true"},
                    "autoConnect": True,
                }
            }
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        await discovery.discover_all()

        servers = discovery.get_all_servers()
        server_ids = [s.server_id for s in servers]

        assert "custom-server" in server_ids

        custom = discovery.get_server("custom-server")
        assert custom is not None
        assert custom.name == "Custom Server"
        assert custom.command == "node"
        assert custom.auto_connect is True

    def test_discover_from_environment(self, temp_config_dir):
        """Test discovering servers from environment variables."""
        # Set environment variable
        env_config = json.dumps({
            "name": "Env Server",
            "command": "python",
            "args": ["-m", "server"],
        })
        os.environ["MCP_SERVER_ENV_TEST"] = env_config

        try:
            discovery = MCPServerDiscovery(project_root=temp_config_dir)
            discovery._discover_from_environment()

            server = discovery.get_server("env_test")
            assert server is not None
            assert server.name == "Env Server"
            assert server.source == DiscoverySource.ENVIRONMENT

        finally:
            del os.environ["MCP_SERVER_ENV_TEST"]

    def test_get_enabled_servers(self, discovery):
        """Test getting enabled servers."""
        # Add test servers
        discovery._discovered["enabled"] = DiscoveredServer(
            server_id="enabled",
            name="Enabled",
            enabled=True,
        )
        discovery._discovered["disabled"] = DiscoveredServer(
            server_id="disabled",
            name="Disabled",
            enabled=False,
        )

        enabled = discovery.get_enabled_servers()
        assert len(enabled) == 1
        assert enabled[0].server_id == "enabled"

    def test_get_servers_by_source(self, discovery):
        """Test getting servers by source."""
        discovery._discovered["env1"] = DiscoveredServer(
            server_id="env1",
            name="Env1",
            source=DiscoverySource.ENVIRONMENT,
        )
        discovery._discovered["default1"] = DiscoveredServer(
            server_id="default1",
            name="Default1",
            source=DiscoverySource.DEFAULT,
        )

        env_servers = discovery.get_servers_by_source(DiscoverySource.ENVIRONMENT)
        assert len(env_servers) == 1
        assert env_servers[0].server_id == "env1"

    def test_to_mcp_client_configs(self, discovery):
        """Test converting to MCP client configs."""
        discovery._discovered["test"] = DiscoveredServer(
            server_id="test",
            name="Test",
            description="Test server",
            command="npx",
            args=["-y", "test-server"],
        )

        configs = discovery.to_mcp_client_configs()
        assert len(configs) == 1
        assert configs[0].server_id == "test"
        assert configs[0].name == "Test"

    def test_known_mcp_packages(self):
        """Test known MCP packages list."""
        discovery = MCPServerDiscovery()

        # Check known packages are defined
        assert "@anthropic-ai/mcp-server-mt5" in discovery.KNOWN_MCP_PACKAGES
        assert "@anthropic-ai/mcp-server-filesystem" in discovery.KNOWN_MCP_PACKAGES

        # Check descriptions exist
        for package, desc in discovery.KNOWN_MCP_PACKAGES.items():
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestGetMcpDiscovery:
    """Tests for get_mcp_discovery function."""

    def test_get_discovery_singleton(self):
        """Test that get_mcp_discovery returns singleton."""
        # Reset global
        import src.agents.mcp.discovery as discovery_module
        discovery_module._discovery = None

        discovery1 = get_mcp_discovery()
        discovery2 = get_mcp_discovery()

        assert discovery1 is discovery2

        # Cleanup
        discovery_module._discovery = None


class TestNpmDiscovery:
    """Tests for npm-based discovery."""

    @pytest.mark.asyncio
    async def test_discover_from_npm_known_packages(self):
        """Test discovering known npm packages."""
        discovery = MCPServerDiscovery()
        servers = await discovery.discover_from_npm()

        # Should return known packages
        assert len(servers) > 0

        # Check structure
        for server in servers:
            assert server.server_id
            assert server.name
            assert server.source == DiscoverySource.NPM_REGISTRY

    @pytest.mark.asyncio
    async def test_discover_from_npm_with_search(self):
        """Test npm discovery with search term."""
        discovery = MCPServerDiscovery()

        # Search for filesystem
        servers = await discovery.discover_from_npm("@anthropic-ai/mcp-server-filesystem")

        # Should find the package
        assert len(servers) > 0
        assert any("filesystem" in s.server_id for s in servers)
