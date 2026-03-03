"""
MCP Client Manager Module.

Provides MCPClientManager class for managing MCP server connections.
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import real MCP client
try:
    from src.mcp.client import (
        get_mcp_client,
        MCPServerConfig,
        MCPServerStatus,
        MCPTransportType,
    )
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    logger.warning("Real MCP client not available, using fallback")


# Path to MCP config file
MCP_CONFIG_PATH = Path(__file__).parent.parent.parent / ".analyst" / "mcp_config.json"


def _load_mcp_config() -> Dict[str, Any]:
    """Load MCP server configurations from the config file."""
    config_path = MCP_CONFIG_PATH
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            logger.info(f"Loaded MCP config from {config_path}")
            return config_data.get("mcpServers", {})
        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")
    return {}


class MCPClientManager:
    """
    Manager for MCP server connections and tool invocations.

    Uses the real MCP client layer for actual server connections.
    Supports both stdio and HTTP transports.
    """

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        """
        Initialize MCP client manager.

        Args:
            configs: Optional custom server configurations
        """
        self.configs = configs or _load_mcp_config()
        self._real_client: Optional[Any] = None
        self._http_sessions: Dict[str, aiohttp.ClientSession] = {}
        self._status: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

        logger.info(f"MCPClientManager initialized with {len(self.configs)} servers")

    async def initialize(self) -> None:
        """Initialize connections to all MCP servers using real client."""
        if self._initialized:
            return

        if MCP_CLIENT_AVAILABLE:
            self._real_client = get_mcp_client()

            # Register servers from config
            for server_id, config in self.configs.items():
                if config.get("enabled", True):
                    await self._register_server(server_id, config)
        else:
            logger.warning("MCP client unavailable, tools will raise errors on invocation")

        self._initialized = True

    async def _register_server(self, server_id: str, config: Dict[str, Any]) -> None:
        """Register a server based on its configuration."""
        url = config.get("url", "")

        # Check if this is an HTTP-based server
        if url and url.startswith("http"):
            # HTTP transport - create session
            if server_id not in self._http_sessions:
                self._http_sessions[server_id] = aiohttp.ClientSession(
                    base_url=url,
                    timeout=aiohttp.ClientTimeout(total=config.get("config", {}).get("timeout", 30))
                )
            self._status[server_id] = {
                "status": "online",
                "last_check": datetime.now().isoformat(),
                "error": None,
                "transport": "http",
                "url": url
            }
            logger.info(f"Registered HTTP MCP server: {server_id} at {url}")
        elif url and url.startswith("local://"):
            # Local/stdio transport - use real MCP client
            if MCP_CLIENT_AVAILABLE and self._real_client:
                try:
                    # Determine command based on server type
                    command, args = self._get_stdio_command(server_id, config)
                    if command:
                        mcp_config = MCPServerConfig(
                            server_id=server_id,
                            name=config.get("name", server_id),
                            description=config.get("description", ""),
                            transport=MCPTransportType.STDIO,
                            command=command,
                            args=args,
                            env=config.get("env", {}),
                            enabled=True,
                            auto_connect=False,
                        )
                        self._real_client.register_server(mcp_config)
                        logger.info(f"Registered stdio MCP server: {server_id}")
                except Exception as e:
                    logger.error(f"Failed to register MCP server {server_id}: {e}")
        else:
            # Default: try stdio with real client
            if MCP_CLIENT_AVAILABLE and self._real_client:
                try:
                    command, args = self._get_stdio_command(server_id, config)
                    if command:
                        mcp_config = MCPServerConfig(
                            server_id=server_id,
                            name=config.get("name", server_id),
                            description=config.get("description", ""),
                            transport=MCPTransportType.STDIO,
                            command=command,
                            args=args,
                            env=config.get("env", {}),
                            enabled=True,
                            auto_connect=False,
                        )
                        self._real_client.register_server(mcp_config)
                        logger.info(f"Registered stdio MCP server: {server_id}")
                except Exception as e:
                    logger.error(f"Failed to register MCP server {server_id}: {e}")

    def _get_stdio_command(self, server_id: str, config: Dict[str, Any]) -> tuple:
        """Get stdio command and args for a server type."""
        server_type = config.get("type", server_id)

        # Map server types to their npx packages
        type_to_package = {
            "context7": ("npx", ["-y", "@context7/mcp-server"]),
            "sequential_thinking": ("npx", ["-y", "@modelcontextprotocol/server-sequential-thinking"]),
            "pageindex": ("npx", ["-y", "@pageindex/mcp-server"]),
            "backtest": ("python", ["-m", "mcp_servers.backtest_mcp_server.main"]),
            "mt5_compiler": ("python", ["-m", "mcp_servers.mt5_compiler"]),
        }

        if server_type in type_to_package:
            return type_to_package[server_type]

        # Check if config has explicit command/args
        if "command" in config:
            return config["command"], config.get("args", [])

        return None, []

    async def connect_server(self, server_id: str) -> bool:
        """Connect to a specific MCP server."""
        config = self.configs.get(server_id)
        if not config:
            raise ValueError(f"Unknown server: {server_id}")

        url = config.get("url", "")

        # HTTP servers are "always connected" via session
        if url and url.startswith("http"):
            if server_id in self._http_sessions:
                self._status[server_id] = {
                    "status": "online",
                    "last_check": datetime.now().isoformat(),
                    "error": None
                }
                return True
            # Create session if not exists
            self._http_sessions[server_id] = aiohttp.ClientSession(
                base_url=url,
                timeout=aiohttp.ClientTimeout(total=config.get("config", {}).get("timeout", 30))
            )
            return True

        # Stdio servers need real connection
        if MCP_CLIENT_AVAILABLE and self._real_client:
            server_config = self._real_client.get_server_config(server_id)
            if server_config:
                success = await self._real_client.connect(server_id)
                if success:
                    self._status[server_id] = {
                        "status": "online",
                        "last_check": datetime.now().isoformat(),
                        "error": None
                    }
                else:
                    state = self._real_client._servers.get(server_id)
                    self._status[server_id] = {
                        "status": "offline",
                        "last_check": datetime.now().isoformat(),
                        "error": state.last_error if state else "Unknown error"
                    }
                return success
        return False

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server using real client or HTTP.

        Args:
            server_id: Server identifier
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result from actual MCP server
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        config = self.configs.get(server_id)
        if not config:
            raise ValueError(f"Unknown server: {server_id}")

        url = config.get("url", "")

        # HTTP transport
        if url and url.startswith("http"):
            return await self._call_http_tool(server_id, tool_name, arguments)

        # Stdio transport via real MCP client
        if MCP_CLIENT_AVAILABLE and self._real_client:
            # Check if server is connected
            status = self._real_client.get_status(server_id)
            if status != MCPServerStatus.CONNECTED:
                # Try to connect
                connected = await self.connect_server(server_id)
                if not connected:
                    raise ConnectionError(f"Failed to connect to MCP server: {server_id}")

            # Call the tool using real client
            try:
                result = await self._real_client.call_tool(server_id, tool_name, arguments)
                logger.debug(f"Called tool {tool_name} on {server_id}: success")
                return result
            except Exception as e:
                logger.error(f"Tool call failed on {server_id}/{tool_name}: {e}")
                raise

        raise RuntimeError(f"MCP client not available - cannot call {server_id}/{tool_name}")

    async def _call_http_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool via HTTP transport."""
        session = self._http_sessions.get(server_id)
        if not session:
            await self.connect_server(server_id)
            session = self._http_sessions.get(server_id)

        if not session:
            raise ConnectionError(f"No HTTP session for server: {server_id}")

        # MCP over HTTP uses JSON-RPC
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }

        try:
            async with session.post("/mcp", json=request_body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")

                result = await response.json()

                # Check for JSON-RPC error
                if "error" in result:
                    raise RuntimeError(f"MCP error: {result['error']}")

                return result.get("result", {})

        except aiohttp.ClientError as e:
            logger.error(f"HTTP call failed to {server_id}: {e}")
            raise ConnectionError(f"Failed to call HTTP MCP server {server_id}: {e}")

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers."""
        statuses = {}

        for server_id, config in self.configs.items():
            url = config.get("url", "")

            if url and url.startswith("http"):
                # HTTP server status
                statuses[server_id] = self._status.get(server_id, {
                    "status": "unknown",
                    "last_check": datetime.now().isoformat(),
                    "error": None
                })
            elif MCP_CLIENT_AVAILABLE and self._real_client:
                # Stdio server status
                status = self._real_client.get_status(server_id)
                state = self._real_client._servers.get(server_id)
                statuses[server_id] = {
                    "status": status.value,
                    "last_check": datetime.now().isoformat(),
                    "error": state.last_error if state else None
                }
            else:
                statuses[server_id] = self._status.get(server_id, {
                    "status": "unavailable",
                    "last_check": datetime.now().isoformat(),
                    "error": "MCP client not available"
                })

        return statuses

    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all servers."""
        results = {}

        for server_id, config in self.configs.items():
            url = config.get("url", "")

            if url and url.startswith("http"):
                # HTTP health check
                session = self._http_sessions.get(server_id)
                if session:
                    try:
                        async with session.get("/health") as response:
                            results[server_id] = response.status == 200
                    except:
                        results[server_id] = False
                else:
                    results[server_id] = False
            elif MCP_CLIENT_AVAILABLE and self._real_client:
                status = self._real_client.get_status(server_id)
                results[server_id] = status == MCPServerStatus.CONNECTED
            else:
                results[server_id] = False

        return results

    async def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
        """List tools available from a server."""
        config = self.configs.get(server_id)
        if not config:
            return []

        url = config.get("url", "")

        # HTTP transport
        if url and url.startswith("http"):
            session = self._http_sessions.get(server_id)
            if session:
                try:
                    request_body = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }
                    async with session.post("/mcp", json=request_body) as response:
                        if response.status == 200:
                            result = await response.json()
                            tools = result.get("result", {}).get("tools", [])
                            return [
                                {
                                    "name": t.get("name", "unknown"),
                                    "description": t.get("description", ""),
                                    "input_schema": t.get("inputSchema", {}),
                                    "server_id": server_id
                                }
                                for t in tools
                            ]
                except Exception as e:
                    logger.error(f"Failed to list tools from {server_id}: {e}")
            return []

        # Stdio transport
        if MCP_CLIENT_AVAILABLE and self._real_client:
            tools = await self._real_client.list_tools(server_id)
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "server_id": t.server_id
                }
                for t in tools
            ]
        return []

    async def close(self) -> None:
        """Close all connections."""
        # Close HTTP sessions
        for session in self._http_sessions.values():
            await session.close()
        self._http_sessions.clear()

        # Disconnect stdio servers
        if MCP_CLIENT_AVAILABLE and self._real_client:
            for server_id in list(self._real_client._servers.keys()):
                try:
                    await self._real_client.disconnect(server_id)
                except Exception as e:
                    logger.warning(f"Error disconnecting {server_id}: {e}")

        self._initialized = False
        logger.info("MCPClientManager closed")


# Singleton instance
_mcp_manager: Optional[MCPClientManager] = None


async def get_mcp_manager() -> MCPClientManager:
    """Get or create the singleton MCP manager instance."""
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPClientManager()
        await _mcp_manager.initialize()
    return _mcp_manager
