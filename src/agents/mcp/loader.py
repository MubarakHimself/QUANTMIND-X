"""
Dynamic Tool Loader

Dynamically loads and manages tools from MCP servers:
- Discovers available tools
- Caches tool definitions
- Provides tool execution interface
- Integrates with agent spawning
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.agents.mcp.discovery import MCPServerDiscovery, get_mcp_discovery
from src.agents.mcp.sse import MCPSSEHandler, get_sse_handler

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Represents a loaded tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_id: str
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    call_count: int = 0
    last_called: Optional[datetime] = None


@dataclass
class ToolExecution:
    """Represents a tool execution context."""
    call_id: str
    tool_name: str
    server_id: str
    arguments: Dict[str, Any]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


class DynamicToolLoader:
    """
    Dynamically loads and manages tools from MCP servers.

    Features:
    - Lazy tool loading (only load when needed)
    - Tool caching with TTL
    - Automatic server connection management
    - Tool execution tracking
    - Integration with SSE for real-time updates

    Usage:
        loader = DynamicToolLoader()
        await loader.initialize()

        # Get tools from a server
        tools = await loader.get_tools("filesystem")

        # Call a tool
        result = await loader.call_tool("filesystem", "read_file", {"path": "/tmp/test.txt"})

        # Get all available tools
        all_tools = loader.list_all_tools()
    """

    def __init__(
        self,
        discovery: Optional[MCPServerDiscovery] = None,
        sse_handler: Optional[MCPSSEHandler] = None,
    ):
        """
        Initialize the dynamic tool loader.

        Args:
            discovery: MCP server discovery instance
            sse_handler: SSE handler for events
        """
        self._discovery = discovery or get_mcp_discovery()
        self._sse_handler = sse_handler or get_sse_handler()
        self._tools: Dict[str, Dict[str, ToolDefinition]] = {}  # server_id -> tool_name -> ToolDefinition
        self._executions: Dict[str, ToolExecution] = {}  # call_id -> execution
        self._mcp_client = None
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the loader and discover servers."""
        if self._initialized:
            return

        # Start SSE handler
        await self._sse_handler.start()

        # Discover MCP servers
        await self._discovery.discover_all()

        # Get MCP client
        try:
            from src.mcp.client import get_mcp_client, MCPServerConfig, MCPTransportType
            self._mcp_client = get_mcp_client()

            # Register discovered servers with MCP client
            for server in self._discovery.get_enabled_servers():
                config = MCPServerConfig(
                    server_id=server.server_id,
                    name=server.name,
                    description=server.description,
                    transport=(
                        MCPTransportType.HTTP if server.transport == "http"
                        else MCPTransportType.STDIO
                    ),
                    command=server.command,
                    args=server.args,
                    env=server.env,
                    url=server.url,
                    auto_connect=server.auto_connect,
                )
                self._mcp_client.register_server(config)

            logger.info(f"Initialized tool loader with {len(self._discovery.get_enabled_servers())} servers")

        except ImportError:
            logger.warning("MCP client not available, using fallback mode")

        self._initialized = True

    async def get_tools(self, server_id: str) -> List[ToolDefinition]:
        """
        Get tools from a specific server.

        Args:
            server_id: Server identifier

        Returns:
            List of available tools
        """
        # Check cache first
        if server_id in self._tools:
            return list(self._tools[server_id].values())

        # Try to load from MCP client
        if self._mcp_client:
            try:
                # Ensure connected
                status = self._mcp_client.get_status(server_id)
                if status.value == "disconnected":
                    await self._mcp_client.connect(server_id)

                # Get tools
                tools = await self._mcp_client.list_tools(server_id)

                async with self._lock:
                    self._tools[server_id] = {}

                    for tool in tools:
                        definition = ToolDefinition(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.input_schema,
                            server_id=server_id,
                        )
                        self._tools[server_id][tool.name] = definition

                        # Publish SSE event
                        await self._sse_handler.publish_tool_discovered(
                            server_id,
                            tool.name,
                            tool.description,
                            tool.input_schema,
                        )

                return list(self._tools[server_id].values())

            except Exception as e:
                logger.error(f"Failed to get tools from {server_id}: {e}")

        return []

    async def call_tool(
        self,
        server_id: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            server_id: Server identifier
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found
            RuntimeError: If tool execution fails
        """
        call_id = str(uuid.uuid4())
        arguments = arguments or {}

        execution = ToolExecution(
            call_id=call_id,
            tool_name=tool_name,
            server_id=server_id,
            arguments=arguments,
        )

        async with self._lock:
            self._executions[call_id] = execution

        # Publish tool call event
        await self._sse_handler.publish_tool_called(
            server_id,
            tool_name,
            call_id,
            arguments,
        )

        try:
            # Ensure connected
            if self._mcp_client:
                status = self._mcp_client.get_status(server_id)
                if status.value == "disconnected":
                    await self._mcp_client.connect(server_id)
                    await self._sse_handler.publish_connection_event(server_id, "connected")

                # Call the tool
                result = await self._mcp_client.call_tool(server_id, tool_name, arguments)

                # Update execution
                execution.result = result
                execution.completed_at = datetime.now(timezone.utc)

                # Update tool stats
                async with self._lock:
                    if server_id in self._tools and tool_name in self._tools[server_id]:
                        tool = self._tools[server_id][tool_name]
                        tool.call_count += 1
                        tool.last_called = datetime.now(timezone.utc)

                # Publish result event
                await self._sse_handler.publish_tool_result(
                    server_id,
                    tool_name,
                    call_id,
                    result,
                )

                return result

            # Fallback - return mock result
            return {
                "status": "mock",
                "message": f"Tool {tool_name} called on {server_id}",
                "call_id": call_id,
                "arguments": arguments,
            }

        except Exception as e:
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)

            # Publish error event
            await self._sse_handler.publish_tool_result(
                server_id,
                tool_name,
                call_id,
                None,
                error=str(e),
            )

            raise RuntimeError(f"Tool execution failed: {e}") from e

    def list_all_tools(self) -> List[ToolDefinition]:
        """
        Get all loaded tools across all servers.

        Returns:
            List of all tool definitions
        """
        tools = []
        for server_tools in self._tools.values():
            tools.extend(server_tools.values())
        return tools

    def list_servers_with_tools(self) -> List[str]:
        """Get list of server IDs that have loaded tools."""
        return list(self._tools.keys())

    def get_tool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded tools.

        Returns:
            Dictionary with tool statistics
        """
        total_tools = sum(len(tools) for tools in self._tools.values())
        total_calls = sum(
            tool.call_count
            for tools in self._tools.values()
            for tool in tools.values()
        )

        return {
            "total_servers": len(self._tools),
            "total_tools": total_tools,
            "total_calls": total_calls,
            "servers": {
                server_id: {
                    "tools_count": len(tools),
                    "total_calls": sum(t.call_count for t in tools.values()),
                }
                for server_id, tools in self._tools.items()
            },
        }

    async def get_execution(self, call_id: str) -> Optional[ToolExecution]:
        """Get a tool execution by call ID."""
        return self._executions.get(call_id)

    async def refresh_tools(self, server_id: str) -> List[ToolDefinition]:
        """
        Refresh tools from a server (clear cache and reload).

        Args:
            server_id: Server identifier

        Returns:
            List of refreshed tools
        """
        async with self._lock:
            if server_id in self._tools:
                del self._tools[server_id]

        return await self.get_tools(server_id)

    async def disconnect_server(self, server_id: str) -> None:
        """
        Disconnect from a server and clear its tools.

        Args:
            server_id: Server identifier
        """
        if self._mcp_client:
            await self._mcp_client.disconnect(server_id)
            await self._sse_handler.publish_connection_event(server_id, "disconnected")

        async with self._lock:
            if server_id in self._tools:
                del self._tools[server_id]


# Global tool loader instance
_tool_loader: Optional[DynamicToolLoader] = None


def get_tool_loader(
    discovery: Optional[MCPServerDiscovery] = None,
    sse_handler: Optional[MCPSSEHandler] = None,
) -> DynamicToolLoader:
    """
    Get the global dynamic tool loader instance.

    Args:
        discovery: Optional MCP server discovery
        sse_handler: Optional SSE handler

    Returns:
        DynamicToolLoader instance
    """
    global _tool_loader
    if _tool_loader is None:
        _tool_loader = DynamicToolLoader(discovery, sse_handler)
    return _tool_loader
