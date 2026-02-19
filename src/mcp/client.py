"""
MCP Client Layer

Implements real MCP (Model Context Protocol) client connections.
Provides configuration-backed discovery, live status checks, and tool execution.

This replaces the hardcoded MCP_SERVERS_REGISTRY with a proper client layer
that can connect to real MCP servers via stdio or WebSocket transports.
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class MCPServerStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MCPTransportType(str, Enum):
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    HTTP = "http"


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_id: str


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    server_id: str
    name: str
    description: str = ""
    transport: MCPTransportType = MCPTransportType.STDIO
    command: Optional[str] = None  # For stdio transport
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # For websocket/http transport
    enabled: bool = True
    auto_connect: bool = False
    timeout_seconds: int = 30


@dataclass
class MCPServerState:
    """Runtime state for an MCP server connection."""
    config: MCPServerConfig
    status: MCPServerStatus = MCPServerStatus.DISCONNECTED
    connected_at: Optional[datetime] = None
    last_error: Optional[str] = None
    tools: List[MCPTool] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    websocket: Optional[Any] = None


class MCPTransport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to MCP server."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        pass

    @abstractmethod
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass


class StdioTransport(MCPTransport):
    """MCP transport over stdio (subprocess communication)."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Start the MCP server subprocess and establish connection."""
        if not self.config.command:
            raise ValueError("No command specified for stdio transport")

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)

            # Find the command (could be npx, python, etc.)
            command = self.config.command
            if command == "npx":
                # Use npm/npx from PATH
                npx_path = shutil.which("npx") or shutil.which("npm")
                if npx_path:
                    command = npx_path

            # Start subprocess
            self.process = subprocess.Popen(
                [command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Start reader task
            self._reader_task = asyncio.create_task(self._read_responses())

            logger.info(f"Started MCP server: {self.config.server_id} (PID: {self.process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to start MCP server {self.config.server_id}: {e}")
            return False

    async def disconnect(self) -> None:
        """Terminate the MCP server subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
            finally:
                self.process = None

    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request via stdin."""
        if not self.process or not self.process.stdin:
            raise ConnectionError("Not connected to MCP server")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {}
        }

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[self._request_id] = future

        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()

            # Wait for response with timeout
            return await asyncio.wait_for(future, timeout=self.config.timeout_seconds)

        except asyncio.TimeoutError:
            self._pending_requests.pop(self._request_id, None)
            raise TimeoutError(f"Request {method} timed out")
        except Exception as e:
            self._pending_requests.pop(self._request_id, None)
            raise

    async def is_connected(self) -> bool:
        """Check if subprocess is running."""
        return self.process is not None and self.process.poll() is None

    async def _read_responses(self) -> None:
        """Read responses from stdout in background."""
        loop = asyncio.get_event_loop()

        while self.process and self.process.stdout:
            try:
                # Read line from stdout
                line = await loop.run_in_executor(None, self.process.stdout.readline)
                if not line:
                    break

                response = json.loads(line.strip())

                # Match response to pending request
                request_id = response.get("id")
                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)
                    if not future.done():
                        if "error" in response:
                            future.set_exception(Exception(response["error"]))
                        else:
                            future.set_result(response.get("result", {}))

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response: {e}")
            except Exception as e:
                logger.error(f"Error reading response: {e}")
                break


class MCPClient:
    """
    MCP Client for connecting to and interacting with MCP servers.

    Usage:
        client = MCPClient()
        await client.register_server("metatrader5", config)
        await client.connect("metatrader5")
        tools = await client.list_tools("metatrader5")
        result = await client.call_tool("metatrader5", "get_account_info", {})
    """

    def __init__(self):
        self._servers: Dict[str, MCPServerState] = {}
        self._transports: Dict[str, MCPTransport] = {}
        self._tool_registry: Dict[str, MCPTool] = {}  # tool_name -> MCPTool

    def register_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration."""
        state = MCPServerState(config=config)
        self._servers[config.server_id] = state
        logger.info(f"Registered MCP server: {config.server_id}")

    def unregister_server(self, server_id: str) -> None:
        """Unregister and disconnect an MCP server."""
        if server_id in self._servers:
            asyncio.create_task(self.disconnect(server_id))
            del self._servers[server_id]

    def get_server_config(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get server configuration."""
        if server_id in self._servers:
            return self._servers[server_id].config
        return None

    def list_servers(self) -> List[MCPServerConfig]:
        """List all registered server configurations."""
        return [state.config for state in self._servers.values()]

    async def connect(self, server_id: str) -> bool:
        """Connect to an MCP server."""
        if server_id not in self._servers:
            raise ValueError(f"Server {server_id} not registered")

        state = self._servers[server_id]
        state.status = MCPServerStatus.CONNECTING

        try:
            # Create appropriate transport
            config = state.config
            if config.transport == MCPTransportType.STDIO:
                transport = StdioTransport(config)
            else:
                raise ValueError(f"Unsupported transport: {config.transport}")

            self._transports[server_id] = transport

            # Connect
            success = await transport.connect()
            if not success:
                state.status = MCPServerStatus.ERROR
                state.last_error = "Failed to establish connection"
                return False

            # Initialize MCP protocol
            await self._initialize_protocol(server_id)

            # Discover tools
            await self._discover_tools(server_id)

            state.status = MCPServerStatus.CONNECTED
            state.connected_at = datetime.now(timezone.utc)
            state.last_error = None

            logger.info(f"Connected to MCP server: {server_id}")
            return True

        except Exception as e:
            state.status = MCPServerStatus.ERROR
            state.last_error = str(e)
            logger.error(f"Failed to connect to MCP server {server_id}: {e}")
            return False

    async def disconnect(self, server_id: str) -> None:
        """Disconnect from an MCP server."""
        if server_id in self._transports:
            await self._transports[server_id].disconnect()
            del self._transports[server_id]

        if server_id in self._servers:
            state = self._servers[server_id]
            state.status = MCPServerStatus.DISCONNECTED
            state.connected_at = None

            # Remove tools from registry
            for tool in state.tools:
                self._tool_registry.pop(tool.name, None)
            state.tools = []

        logger.info(f"Disconnected from MCP server: {server_id}")

    async def _initialize_protocol(self, server_id: str) -> None:
        """Initialize MCP protocol handshake."""
        transport = self._transports.get(server_id)
        if not transport:
            raise ConnectionError(f"Not connected to {server_id}")

        # Send initialize request
        result = await transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "QuantMind",
                "version": "1.0.0"
            }
        })

        logger.debug(f"MCP initialize result for {server_id}: {result}")

    async def _discover_tools(self, server_id: str) -> None:
        """Discover available tools from the server."""
        transport = self._transports.get(server_id)
        if not transport:
            raise ConnectionError(f"Not connected to {server_id}")

        state = self._servers[server_id]

        try:
            result = await transport.send_request("tools/list", {})
            tools_data = result.get("tools", [])

            state.tools = []
            for tool_info in tools_data:
                tool = MCPTool(
                    name=tool_info.get("name", "unknown"),
                    description=tool_info.get("description", ""),
                    input_schema=tool_info.get("inputSchema", {}),
                    server_id=server_id,
                )
                state.tools.append(tool)
                self._tool_registry[tool.name] = tool

            logger.info(f"Discovered {len(state.tools)} tools from {server_id}")

        except Exception as e:
            logger.warning(f"Failed to discover tools from {server_id}: {e}")
            state.tools = []

    async def list_tools(self, server_id: str) -> List[MCPTool]:
        """List tools available from a server."""
        if server_id not in self._servers:
            return []

        state = self._servers[server_id]
        if state.status != MCPServerStatus.CONNECTED:
            return []

        return state.tools

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Call a tool on an MCP server."""
        transport = self._transports.get(server_id)
        if not transport:
            raise ConnectionError(f"Not connected to {server_id}")

        result = await transport.send_request("tools/call", {
            "name": tool_name,
            arguments: arguments or {}
        })

        return result

    def get_status(self, server_id: str) -> MCPServerStatus:
        """Get connection status for a server."""
        if server_id in self._servers:
            return self._servers[server_id].status
        return MCPServerStatus.DISCONNECTED

    def get_all_statuses(self) -> Dict[str, MCPServerStatus]:
        """Get status for all servers."""
        return {sid: state.status for sid, state in self._servers.items()}

    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name from the registry."""
        return self._tool_registry.get(tool_name)

    def get_all_tools(self) -> List[MCPTool]:
        """Get all registered tools."""
        return list(self._tool_registry.values())


# Global MCP client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Get the global MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


async def initialize_mcp_client_from_config(config_path: Optional[str] = None) -> MCPClient:
    """
    Initialize the MCP client from configuration file.

    Args:
        config_path: Path to MCP config file (defaults to ./mcp_config.json)

    Returns:
        Initialized MCP client
    """
    client = get_mcp_client()

    # Default config path
    if config_path is None:
        config_path = os.getenv("MCP_CONFIG_PATH", "./mcp_config.json")

    # Load config if exists
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            servers = config_data.get("mcpServers", {})
            for server_id, server_config in servers.items():
                mcp_config = MCPServerConfig(
                    server_id=server_id,
                    name=server_config.get("name", server_id),
                    description=server_config.get("description", ""),
                    transport=MCPTransportType.STDIO,
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    enabled=server_config.get("enabled", True),
                    auto_connect=server_config.get("autoConnect", False),
                )
                client.register_server(mcp_config)

                # Auto-connect if configured
                if mcp_config.auto_connect:
                    await client.connect(server_id)

        except Exception as e:
            logger.error(f"Failed to load MCP config from {config_path}: {e}")

    return client
