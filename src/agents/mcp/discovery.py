"""
MCP Server Discovery

Discovers MCP servers from multiple sources:
- Configuration files
- Environment variables
- npm registry (for available MCP packages)
- Local project configs
"""

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DiscoverySource(str, Enum):
    """Sources for MCP server discovery."""
    CONFIG_FILE = "config_file"
    ENVIRONMENT = "environment"
    NPM_REGISTRY = "npm_registry"
    LOCAL_PROJECT = "local_project"
    DEFAULT = "default"


@dataclass
class DiscoveredServer:
    """Represents a discovered MCP server."""
    server_id: str
    name: str
    description: str = ""
    source: DiscoverySource = DiscoverySource.DEFAULT
    transport: str = "stdio"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    auto_connect: bool = False
    enabled: bool = True


class MCPServerDiscovery:
    """
    Discovers MCP servers from multiple sources.

    Discovery priority:
    1. Local project config (.analyst/mcp_config.json)
    2. Environment variables (MCP_SERVER_*)
    3. Default config files (mcp_config.json, ~/.mcp/servers.json)
    4. npm registry lookup

    Usage:
        discovery = MCPServerDiscovery()
        servers = await discovery.discover_all()

        # Get servers from specific source
        env_servers = discovery.discover_from_env()
    """

    DEFAULT_CONFIG_PATHS = [
        ".analyst/mcp_config.json",
        "mcp_config.json",
        ".mcp/config.json",
        "~/.mcp/servers.json",
    ]

    ENV_PREFIX = "MCP_SERVER_"

    # Known MCP packages on npm (for reference)
    KNOWN_MCP_PACKAGES = {
        "@anthropic-ai/mcp-server-mt5": "MetaTrader 5 trading platform integration",
        "@anthropic-ai/mcp-server-filesystem": "Local filesystem access",
        "@anthropic-ai/mcp-server-github": "GitHub API integration",
        "@anthropic-ai/mcp-server-brave-search": "Brave web search",
        "@anthropic-ai/mcp-server-puppeteer": "Browser automation via Puppeteer",
        "@context7/mcp-server": "Context7 documentation search",
        "@modelcontextprotocol/server-sequential-thinking": "Sequential thinking/reasoning",
        "@pageindex/mcp-server": "PDF indexing and search",
    }

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize MCP server discovery.

        Args:
            project_root: Root directory for project-specific config lookup
        """
        self._project_root = project_root or Path.cwd()
        self._discovered: Dict[str, DiscoveredServer] = {}
        self._lock = asyncio.Lock()

    async def discover_all(self) -> List[DiscoveredServer]:
        """
        Discover all MCP servers from all sources.

        Returns:
            List of discovered servers (deduplicated by server_id)
        """
        async with self._lock:
            self._discovered.clear()

            # Discover from each source
            await self._discover_from_config_files()
            self._discover_from_environment()
            self._discover_from_defaults()

            return list(self._discovered.values())

    async def _discover_from_config_files(self) -> None:
        """Discover servers from configuration files."""
        for config_path in self.DEFAULT_CONFIG_PATHS:
            path = Path(config_path).expanduser()

            # Handle project-relative paths
            if not path.is_absolute() and not str(path).startswith("~"):
                path = self._project_root / path

            if path.exists():
                try:
                    with open(path, "r") as f:
                        config = json.load(f)

                    servers = config.get("mcpServers", {})
                    for server_id, server_config in servers.items():
                        self._add_discovered_server(
                            server_id,
                            DiscoveredServer(
                                server_id=server_id,
                                name=server_config.get("name", server_id),
                                description=server_config.get("description", ""),
                                source=DiscoverySource.CONFIG_FILE,
                                transport="stdio" if "url" not in server_config else "http",
                                command=server_config.get("command"),
                                args=server_config.get("args", []),
                                env=server_config.get("env", {}),
                                url=server_config.get("url"),
                                auto_connect=server_config.get("autoConnect", False),
                                enabled=server_config.get("enabled", True),
                            ),
                        )

                    logger.info(f"Discovered {len(servers)} servers from {path}")

                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to parse config {path}: {e}")

    def _discover_from_environment(self) -> None:
        """Discover servers from environment variables."""
        env_servers = {}

        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Parse MCP_SERVER_NAME=json_config
            server_id = key[len(self.ENV_PREFIX):].lower()

            try:
                config = json.loads(value)
                env_servers[server_id] = DiscoveredServer(
                    server_id=server_id,
                    name=config.get("name", server_id),
                    description=config.get("description", ""),
                    source=DiscoverySource.ENVIRONMENT,
                    transport=config.get("transport", "stdio"),
                    command=config.get("command"),
                    args=config.get("args", []),
                    env=config.get("env", {}),
                    url=config.get("url"),
                    auto_connect=config.get("autoConnect", False),
                    enabled=config.get("enabled", True),
                )
            except json.JSONDecodeError:
                # Simple URL=value format
                env_servers[server_id] = DiscoveredServer(
                    server_id=server_id,
                    name=server_id,
                    description="From environment",
                    source=DiscoverySource.ENVIRONMENT,
                    transport="http",
                    url=value,
                    auto_connect=False,
                    enabled=True,
                )

        for server_id, server in env_servers.items():
            self._add_discovered_server(server_id, server)

        if env_servers:
            logger.info(f"Discovered {len(env_servers)} servers from environment")

    def _discover_from_defaults(self) -> None:
        """Add default/built-in MCP servers."""
        defaults = [
            DiscoveredServer(
                server_id="filesystem",
                name="Filesystem MCP",
                description="Local filesystem access",
                command="npx",
                args=["-y", "@anthropic-ai/mcp-server-filesystem", "--root", "./workspace"],
                auto_connect=False,
                enabled=True,
            ),
            DiscoveredServer(
                server_id="context7",
                name="Context7 MCP",
                description="MQL5 documentation retrieval",
                command="npx",
                args=["-y", "@context7/mcp-server"],
                auto_connect=False,
                enabled=True,
            ),
            DiscoveredServer(
                server_id="sequential_thinking",
                name="Sequential Thinking MCP",
                description="Task decomposition and reasoning",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
                auto_connect=False,
                enabled=True,
            ),
        ]

        for server in defaults:
            if server.server_id not in self._discovered:
                self._add_discovered_server(server.server_id, server)

    def _add_discovered_server(self, server_id: str, server: DiscoveredServer) -> None:
        """Add a discovered server, avoiding duplicates."""
        # Prefer earlier sources
        if server_id in self._discovered:
            existing = self._discovered[server_id]
            source_priority = {
                DiscoverySource.ENVIRONMENT: 0,
                DiscoverySource.CONFIG_FILE: 1,
                DiscoverySource.LOCAL_PROJECT: 2,
                DiscoverySource.NPM_REGISTRY: 3,
                DiscoverySource.DEFAULT: 4,
            }

            if source_priority.get(server.source, 99) >= source_priority.get(existing.source, 99):
                return

        self._discovered[server_id] = server

    async def discover_from_npm(self, search_term: Optional[str] = None) -> List[DiscoveredServer]:
        """
        Discover MCP servers from npm registry.

        Args:
            search_term: Optional search term to filter packages

        Returns:
            List of discovered servers from npm
        """
        servers = []

        # Check for known packages if no search term
        packages_to_check = (
            [search_term] if search_term else list(self.KNOWN_MCP_PACKAGES.keys())
        )

        for package in packages_to_check:
            if package not in self.KNOWN_MCP_PACKAGES:
                continue

            servers.append(DiscoveredServer(
                server_id=package.replace("@", "").replace("/", "-").lower(),
                name=package,
                description=self.KNOWN_MCP_PACKAGES[package],
                source=DiscoverySource.NPM_REGISTRY,
                command="npx",
                args=["-y", package],
                auto_connect=False,
                enabled=True,
            ))

        return servers

    def get_server(self, server_id: str) -> Optional[DiscoveredServer]:
        """Get a discovered server by ID."""
        return self._discovered.get(server_id)

    def get_all_servers(self) -> List[DiscoveredServer]:
        """Get all discovered servers."""
        return list(self._discovered.values())

    def get_enabled_servers(self) -> List[DiscoveredServer]:
        """Get all enabled discovered servers."""
        return [s for s in self._discovered.values() if s.enabled]

    def get_servers_by_source(self, source: DiscoverySource) -> List[DiscoveredServer]:
        """Get servers from a specific discovery source."""
        return [s for s in self._discovered.values() if s.source == source]

    def to_mcp_client_configs(self) -> List[Dict[str, Any]]:
        """Convert discovered servers to MCP client config format."""
        from src.mcp.client import MCPServerConfig, MCPTransportType

        configs = []
        for server in self._discovered.values():
            if not server.enabled:
                continue

            transport = (
                MCPTransportType.HTTP if server.transport == "http"
                else MCPTransportType.STDIO
            )

            configs.append(MCPServerConfig(
                server_id=server.server_id,
                name=server.name,
                description=server.description,
                transport=transport,
                command=server.command,
                args=server.args,
                env=server.env,
                url=server.url,
                auto_connect=server.auto_connect,
            ))

        return configs


# Global discovery instance
_discovery: Optional[MCPServerDiscovery] = None


def get_mcp_discovery(project_root: Optional[Path] = None) -> MCPServerDiscovery:
    """
    Get the global MCP server discovery instance.

    Args:
        project_root: Optional project root for discovery

    Returns:
        MCPServerDiscovery instance
    """
    global _discovery
    if _discovery is None:
        _discovery = MCPServerDiscovery(project_root)
    return _discovery
