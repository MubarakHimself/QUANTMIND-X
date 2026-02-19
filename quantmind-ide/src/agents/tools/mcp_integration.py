"""
MCP Integration Tools for QuantMind agents.

These tools provide unified access to MCP server functionality:
- call_mt5_tool: Call MT5 MCP server tools
- call_backtest_tool: Call backtest MCP server tools
- call_kb_tool: Call knowledge base MCP server tools
- list_mcp_servers: List available MCP servers
- get_mcp_status: Get MCP server status
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType
from ..mcp.client import get_mcp_client, MCPClient


logger = logging.getLogger(__name__)


class CallMCPToolInput(BaseModel):
    """Input schema for call_mcp_tool."""
    server_id: str = Field(
        description="MCP server ID"
    )
    tool_name: str = Field(
        description="Name of the tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )


class ListMCPServersInput(BaseModel):
    """Input schema for list_mcp_servers."""
    include_tools: bool = Field(
        default=False,
        description="Include list of tools for each server"
    )


class GetMCPStatusInput(BaseModel):
    """Input schema for get_mcp_status."""
    server_id: Optional[str] = Field(
        default=None,
        description="Specific server ID (all if not specified)"
    )


# Specific server tool inputs
class CallMT5ToolInput(BaseModel):
    """Input schema for call_mt5_tool."""
    tool_name: str = Field(
        description="Name of the MT5 tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )


class CallBacktestToolInput(BaseModel):
    """Input schema for call_backtest_tool."""
    tool_name: str = Field(
        description="Name of the backtest tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )


class CallKBToolInput(BaseModel):
    """Input schema for call_kb_tool."""
    tool_name: str = Field(
        description="Name of the knowledge base tool to call"
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments"
    )


@register_tool(
    agent_types=[AgentType.ALL],
    tags=["mcp", "dynamic", "tool"],
)
class CallMCPTool(QuantMindTool):
    """Call any MCP tool dynamically."""

    name: str = "call_mcp_tool"
    description: str = """Call a tool on an MCP server dynamically.
    Specify the server ID, tool name, and arguments.
    Returns the tool result or error."""

    args_schema: type[BaseModel] = CallMCPToolInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        """Execute MCP tool call."""
        arguments = arguments or {}

        try:
            # Run async call
            result = asyncio.run(self._async_call(server_id, tool_name, arguments))

            return ToolResult.ok(
                data=result,
                metadata={
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "called_at": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            raise ToolError(
                f"MCP tool call failed: {e}",
                tool_name=self.name,
                error_code="MCP_CALL_FAILED",
                details={"server_id": server_id, "tool_name": tool_name}
            )

    async def _async_call(
        self,
        server_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Async tool call."""
        return await self._client.call_tool(
            server_id=server_id,
            tool_name=tool_name,
            arguments=arguments
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["mcp", "servers", "list"],
)
class ListMCPServersTool(QuantMindTool):
    """List available MCP servers."""

    name: str = "list_mcp_servers"
    description: str = """List all configured MCP servers and their status.
    Optionally includes list of available tools per server."""

    args_schema: type[BaseModel] = ListMCPServersInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        include_tools: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute server list."""
        servers = self._client.get_all_servers()

        if include_tools:
            for server in servers:
                server["tools"] = [
                    {
                        "name": t.name,
                        "description": t.description,
                    }
                    for t in self._client.get_tools(server["server_id"])
                ]

        return ToolResult.ok(
            data={
                "servers": servers,
                "total_servers": len(servers),
            },
            metadata={
                "include_tools": include_tools,
                "listed_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["mcp", "status", "health"],
)
class GetMCPStatusTool(QuantMindTool):
    """Get MCP server status."""

    name: str = "get_mcp_status"
    description: str = """Get the connection status of MCP servers.
    Shows connection state, last error, and available tools count."""

    args_schema: type[BaseModel] = GetMCPStatusInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        server_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute status check."""
        if server_id:
            status = self._client.get_server_status(server_id)
            if not status:
                raise ToolError(
                    f"Server not found: {server_id}",
                    tool_name=self.name,
                    error_code="SERVER_NOT_FOUND"
                )
            statuses = [status]
        else:
            statuses = self._client.get_all_servers()

        return ToolResult.ok(
            data={
                "statuses": statuses,
            },
            metadata={
                "checked_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["mcp", "mt5", "trading"],
)
class CallMT5Tool(QuantMindTool):
    """Call MT5 MCP server tools."""

    name: str = "call_mt5_tool"
    description: str = """Call a tool on the MT5 MCP server.
    Provides access to trading operations, account info, and market data."""

    args_schema: type[BaseModel] = CallMT5ToolInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.HIGH

    # Default server ID for MT5
    MT5_SERVER_ID = "mt5"

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        """Execute MT5 tool call."""
        arguments = arguments or {}

        try:
            result = asyncio.run(
                self._client.call_tool(
                    server_id=self.MT5_SERVER_ID,
                    tool_name=tool_name,
                    arguments=arguments
                )
            )

            return ToolResult.ok(
                data=result,
                metadata={
                    "server": "mt5",
                    "tool": tool_name,
                }
            )

        except Exception as e:
            raise ToolError(
                f"MT5 tool call failed: {e}",
                tool_name=self.name,
                error_code="MT5_TOOL_ERROR"
            )


@register_tool(
    agent_types=[AgentType.ANALYST, AgentType.COPILOT],
    tags=["mcp", "backtest", "testing"],
)
class CallBacktestTool(QuantMindTool):
    """Call backtest MCP server tools."""

    name: str = "call_backtest_tool"
    description: str = """Call a tool on the backtest MCP server.
    Provides access to backtesting, optimization, and analysis operations."""

    args_schema: type[BaseModel] = CallBacktestToolInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.HIGH

    # Default server ID for backtest
    BACKTEST_SERVER_ID = "backtest"

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        """Execute backtest tool call."""
        arguments = arguments or {}

        try:
            result = asyncio.run(
                self._client.call_tool(
                    server_id=self.BACKTEST_SERVER_ID,
                    tool_name=tool_name,
                    arguments=arguments
                )
            )

            return ToolResult.ok(
                data=result,
                metadata={
                    "server": "backtest",
                    "tool": tool_name,
                }
            )

        except Exception as e:
            raise ToolError(
                f"Backtest tool call failed: {e}",
                tool_name=self.name,
                error_code="BACKTEST_TOOL_ERROR"
            )


@register_tool(
    agent_types=[AgentType.ANALYST, AgentType.QUANTCODE],
    tags=["mcp", "knowledge", "search"],
)
class CallKBTool(QuantMindTool):
    """Call knowledge base MCP server tools."""

    name: str = "call_kb_tool"
    description: str = """Call a tool on the knowledge base MCP server.
    Provides access to document search, retrieval, and knowledge management."""

    args_schema: type[BaseModel] = CallKBToolInput
    category: ToolCategory = ToolCategory.MCP
    priority: ToolPriority = ToolPriority.NORMAL

    # Default server ID for knowledge base
    KB_SERVER_ID = "knowledge_base"

    def __init__(self, **data):
        super().__init__(**data)
        self._client = get_mcp_client()

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        **kwargs
    ) -> ToolResult:
        """Execute KB tool call."""
        arguments = arguments or {}

        try:
            result = asyncio.run(
                self._client.call_tool(
                    server_id=self.KB_SERVER_ID,
                    tool_name=tool_name,
                    arguments=arguments
                )
            )

            return ToolResult.ok(
                data=result,
                metadata={
                    "server": "knowledge_base",
                    "tool": tool_name,
                }
            )

        except Exception as e:
            raise ToolError(
                f"Knowledge base tool call failed: {e}",
                tool_name=self.name,
                error_code="KB_TOOL_ERROR"
            )


# Export all tools
__all__ = [
    "CallMCPTool",
    "ListMCPServersTool",
    "GetMCPStatusTool",
    "CallMT5Tool",
    "CallBacktestTool",
    "CallKBTool",
]
