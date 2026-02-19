"""
MCP Tool Adapter for LangChain integration.

Converts MCP tool schemas to LangChain tool format,
enabling seamless integration with ToolNode.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, create_model

from ..tools.base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
)
from ..tools.registry import AgentType, register_tool
from .client import MCPClient, MCPToolSchema, get_mcp_client


logger = logging.getLogger(__name__)


def json_schema_to_pydantic_field(
    name: str,
    schema: Dict[str, Any],
    required: bool = False,
) -> tuple:
    """
    Convert JSON Schema property to Pydantic field definition.

    Args:
        name: Field name
        schema: JSON Schema for the field
        required: Whether the field is required

    Returns:
        Tuple of (type, Field) for create_model
    """
    json_type = schema.get("type", "string")
    description = schema.get("description", "")
    default = ... if required else None

    # Map JSON Schema types to Python types
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    python_type = type_map.get(json_type, str)

    # Handle optional
    if not required:
        python_type = Optional[python_type]

    return (python_type, Field(default=default, description=description))


def create_pydantic_model_from_schema(
    model_name: str,
    schema: Dict[str, Any],
) -> Type[BaseModel]:
    """
    Create a Pydantic model from JSON Schema.

    Args:
        model_name: Name for the model class
        schema: JSON Schema object

    Returns:
        Pydantic model class
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for prop_name, prop_schema in properties.items():
        field_type, field_info = json_schema_to_pydantic_field(
            prop_name,
            prop_schema,
            required=prop_name in required
        )
        fields[prop_name] = (field_type, field_info)

    return create_model(model_name, **fields)


class MCPToolAdapter(QuantMindTool):
    """
    Adapter that wraps an MCP tool as a QuantMind tool.

    Enables MCP tools to work seamlessly with LangChain ToolNode.
    """

    # MCP-specific fields
    mcp_server_id: str = ""
    mcp_tool_name: str = ""
    mcp_schema: Optional[MCPToolSchema] = None

    # Override category
    category: ToolCategory = ToolCategory.MCP

    def __init__(
        self,
        server_id: str,
        tool_schema: MCPToolSchema,
        **kwargs
    ):
        # Set attributes before super init
        kwargs["name"] = f"mcp_{server_id}_{tool_schema.name}"
        kwargs["description"] = tool_schema.description
        kwargs["mcp_server_id"] = server_id
        kwargs["mcp_tool_name"] = tool_schema.name
        kwargs["mcp_schema"] = tool_schema

        # Create Pydantic model from schema
        if tool_schema.input_schema:
            kwargs["args_schema"] = create_pydantic_model_from_schema(
                f"{tool_schema.name.title()}Input",
                tool_schema.input_schema
            )

        super().__init__(**kwargs)

        self._client = get_mcp_client()

    def execute(self, **kwargs) -> ToolResult:
        """Execute the MCP tool."""
        # MCP client is async, need to run in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._async_execute(**kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self._async_execute(**kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._async_execute(**kwargs))

    async def _async_execute(self, **kwargs) -> ToolResult:
        """Async execution of MCP tool."""
        try:
            result = await self._client.call_tool(
                server_id=self.mcp_server_id,
                tool_name=self.mcp_tool_name,
                arguments=kwargs,
            )

            return ToolResult.ok(
                data=result.get("content", result),
                metadata={
                    "mcp_server": self.mcp_server_id,
                    "mcp_tool": self.mcp_tool_name,
                }
            )

        except Exception as e:
            raise ToolError(
                f"MCP tool execution failed: {e}",
                tool_name=self.name,
                error_code="MCP_EXECUTION_ERROR"
            )

    @classmethod
    def from_schema(
        cls,
        server_id: str,
        tool_schema: MCPToolSchema,
    ) -> "MCPToolAdapter":
        """Create adapter from MCP tool schema."""
        return cls(server_id=server_id, tool_schema=tool_schema)


def adapt_mcp_tool(
    server_id: str,
    tool_schema: MCPToolSchema,
    register: bool = True,
    agent_types: Optional[List[Union[AgentType, str]]] = None,
) -> MCPToolAdapter:
    """
    Adapt an MCP tool to a QuantMind tool.

    Args:
        server_id: MCP server ID
        tool_schema: Tool schema from MCP server
        register: Whether to register in tool registry
        agent_types: Agent types that can use this tool

    Returns:
        Adapted tool
    """
    adapter = MCPToolAdapter.from_schema(server_id, tool_schema)

    if register:
        from ..tools.registry import tool_registry
        tool_registry.register(
            adapter,
            agent_types=agent_types or [AgentType.ALL],
            tags=["mcp", f"mcp:{server_id}"],
        )

    return adapter


def adapt_all_mcp_tools(
    client: MCPClient,
    server_id: Optional[str] = None,
    agent_types: Optional[List[Union[AgentType, str]]] = None,
) -> List[MCPToolAdapter]:
    """
    Adapt all tools from MCP client.

    Args:
        client: MCP client
        server_id: Specific server (all if not specified)
        agent_types: Agent types for tools

    Returns:
        List of adapted tools
    """
    tools = client.get_tools(server_id)
    adapters = []

    for tool_schema in tools:
        adapter = adapt_mcp_tool(
            server_id=tool_schema.server_id,
            tool_schema=tool_schema,
            register=True,
            agent_types=agent_types,
        )
        adapters.append(adapter)

    logger.info(f"Adapted {len(adapters)} MCP tools")
    return adapters


class MCPToolWrapper:
    """
    Wrapper class for calling MCP tools dynamically.

    Provides a simple interface for calling any MCP tool
    without pre-registration.
    """

    def __init__(self, client: Optional[MCPClient] = None):
        self._client = client or get_mcp_client()

    async def call(
        self,
        server_id: str,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Call an MCP tool by name."""
        return await self._client.call_tool(
            server_id=server_id,
            tool_name=tool_name,
            arguments=kwargs,
        )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available MCP tools."""
        tools = self._client.get_tools()
        return [
            {
                "name": t.name,
                "server_id": t.server_id,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]
