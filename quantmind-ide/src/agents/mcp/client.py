"""
MCP Client for QuantMind agents.

Provides connectivity to MCP (Model Context Protocol) servers,
enabling dynamic tool discovery and execution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """MCP server connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    server_id: str
    name: str
    transport: str = "stdio"  # stdio, sse, websocket
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # for SSE/WebSocket
    timeout: int = 30000  # milliseconds
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class MCPServerConnection:
    """Represents a connection to an MCP server."""
    config: MCPServerConfig
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    request_id: int = 0
    _process: Optional[Any] = None
    _reader: Optional[Any] = None
    _writer: Optional[Any] = None

    def get_next_request_id(self) -> int:
        """Get next request ID."""
        self.request_id += 1
        return self.request_id


@dataclass
class MCPToolSchema:
    """Schema for an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_id: str


class MCPClient:
    """
    Client for connecting to and interacting with MCP servers.

    Features:
    - Multiple server connections
    - Connection pooling
    - Tool discovery
    - Error handling and retry logic
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerConnection] = {}
        self._tool_cache: Dict[str, MCPToolSchema] = {}
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None

    async def connect(self, config: MCPServerConfig) -> MCPServerConnection:
        """
        Connect to an MCP server.

        Args:
            config: Server configuration

        Returns:
            Connection object
        """
        connection = MCPServerConnection(config=config)
        connection.status = ConnectionStatus.CONNECTING

        try:
            if config.transport == "stdio":
                await self._connect_stdio(connection)
            elif config.transport == "sse":
                await self._connect_sse(connection)
            elif config.transport == "websocket":
                await self._connect_websocket(connection)
            else:
                raise ValueError(f"Unknown transport: {config.transport}")

            connection.status = ConnectionStatus.CONNECTED
            connection.connected_at = datetime.now()

            # Discover tools
            await self._discover_tools(connection)

            self._servers[config.server_id] = connection
            logger.info(f"Connected to MCP server: {config.name}")

        except Exception as e:
            connection.status = ConnectionStatus.ERROR
            connection.last_error = str(e)
            logger.error(f"Failed to connect to MCP server {config.name}: {e}")
            raise

        return connection

    async def _connect_stdio(self, connection: MCPServerConnection) -> None:
        """Connect via stdio transport."""
        config = connection.config

        # Start subprocess
        process = await asyncio.create_subprocess_exec(
            config.command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**dict(__import__('os').environ), **config.env},
        )

        connection._process = process
        connection._reader = process.stdout
        connection._writer = process.stdin

        # Send initialize request
        await self._send_request(connection, "initialize", {
            "protocolVersion": "2024-11-05",
            "clientInfo": {
                "name": "QuantMind",
                "version": "1.0.0"
            }
        })

    async def _connect_sse(self, connection: MCPServerConnection) -> None:
        """Connect via SSE transport."""
        # Would use httpx or aiohttp for SSE
        raise NotImplementedError("SSE transport not yet implemented")

    async def _connect_websocket(self, connection: MCPServerConnection) -> None:
        """Connect via WebSocket transport."""
        # Would use websockets library
        raise NotImplementedError("WebSocket transport not yet implemented")

    async def _send_request(
        self,
        connection: MCPServerConnection,
        method: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send JSON-RPC request to server."""
        request_id = connection.get_next_request_id()

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        request_json = json.dumps(request) + "\n"

        if connection._writer:
            connection._writer.write(request_json.encode())
            await connection._writer.drain()

        # Read response
        if connection._reader:
            response_line = await connection._reader.readline()
            response = json.loads(response_line.decode())

            if "error" in response:
                raise Exception(response["error"].get("message", "Unknown error"))

            return response.get("result", {})

        raise Exception("No connection reader available")

    async def _discover_tools(self, connection: MCPServerConnection) -> None:
        """Discover available tools from server."""
        try:
            result = await self._send_request(connection, "tools/list", {})
            tools = result.get("tools", [])

            connection.tools = tools

            # Cache tool schemas
            for tool in tools:
                schema = MCPToolSchema(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    input_schema=tool.get("inputSchema", {}),
                    server_id=connection.config.server_id,
                )
                self._tool_cache[f"{connection.config.server_id}:{tool['name']}"] = schema

            logger.info(f"Discovered {len(tools)} tools from {connection.config.name}")

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.

        Args:
            server_id: Server to call
            tool_name: Name of tool to call
            arguments: Tool arguments
            timeout: Override timeout (milliseconds)

        Returns:
            Tool result
        """
        connection = self._servers.get(server_id)
        if not connection:
            raise ValueError(f"Server not found: {server_id}")

        if connection.status != ConnectionStatus.CONNECTED:
            raise Exception(f"Server not connected: {server_id}")

        # Retry logic
        retries = 0
        max_retries = connection.config.retry_count

        while retries <= max_retries:
            try:
                result = await asyncio.wait_for(
                    self._send_request(connection, "tools/call", {
                        "name": tool_name,
                        "arguments": arguments,
                    }),
                    timeout=(timeout or connection.config.timeout) / 1000,
                )
                return result

            except asyncio.TimeoutError:
                retries += 1
                if retries > max_retries:
                    raise
                logger.warning(f"Tool call timeout, retry {retries}/{max_retries}")
                await asyncio.sleep(connection.config.retry_delay * retries)

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    connection.last_error = str(e)
                    raise
                logger.warning(f"Tool call error, retry {retries}/{max_retries}: {e}")
                await asyncio.sleep(connection.config.retry_delay * retries)

        raise Exception("Max retries exceeded")

    async def disconnect(self, server_id: str) -> bool:
        """Disconnect from a server."""
        connection = self._servers.get(server_id)
        if not connection:
            return False

        try:
            if connection._process:
                connection._process.terminate()
                await connection._process.wait()

            connection.status = ConnectionStatus.DISCONNECTED
            connection._process = None
            connection._reader = None
            connection._writer = None

            logger.info(f"Disconnected from MCP server: {server_id}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from {server_id}: {e}")
            return False

    def get_server_status(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a server."""
        connection = self._servers.get(server_id)
        if not connection:
            return None

        return {
            "server_id": server_id,
            "name": connection.config.name,
            "status": connection.status.value,
            "connected_at": connection.connected_at.isoformat() if connection.connected_at else None,
            "tools_count": len(connection.tools),
            "last_error": connection.last_error,
        }

    def get_all_servers(self) -> List[Dict[str, Any]]:
        """Get status of all servers."""
        return [
            self.get_server_status(server_id)
            for server_id in self._servers
        ]

    def get_tools(self, server_id: Optional[str] = None) -> List[MCPToolSchema]:
        """Get available tools."""
        if server_id:
            return [
                schema for key, schema in self._tool_cache.items()
                if schema.server_id == server_id
            ]
        return list(self._tool_cache.values())

    def get_tool(self, tool_name: str, server_id: Optional[str] = None) -> Optional[MCPToolSchema]:
        """Get a specific tool schema."""
        if server_id:
            return self._tool_cache.get(f"{server_id}:{tool_name}")

        # Search all servers
        for key, schema in self._tool_cache.items():
            if schema.name == tool_name:
                return schema

        return None

    async def start_health_checks(self) -> None:
        """Start periodic health checks."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_health_checks(self) -> None:
        """Stop health checks."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                for server_id, connection in self._servers.items():
                    if connection.status == ConnectionStatus.CONNECTED:
                        try:
                            # Simple ping
                            await self._send_request(connection, "ping", {})
                        except Exception as e:
                            logger.warning(f"Health check failed for {server_id}: {e}")
                            connection.status = ConnectionStatus.ERROR
                            connection.last_error = str(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    @classmethod
    def from_config_file(cls, config_path: Union[str, Path]) -> "MCPClient":
        """Create client from configuration file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        config_data = json.loads(config_path.read_text())
        client = cls()

        # Parse server configurations
        for server_data in config_data.get("mcpServers", []):
            server_config = MCPServerConfig(
                server_id=server_data.get("serverId", str(uuid.uuid4())),
                name=server_data.get("name", "Unknown"),
                transport=server_data.get("transport", "stdio"),
                command=server_data.get("command"),
                args=server_data.get("args", []),
                env=server_data.get("env", {}),
                url=server_data.get("url"),
                timeout=server_data.get("timeout", 30000),
            )
            # Note: Connection must be done async
            client._servers[server_config.server_id] = MCPServerConnection(config=server_config)

        return client


# Global client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the global MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client
