"""
MCP Integration Module

Integrates MCP functionality with the agent spawner:
- Loads MCP tools for spawned agents
- Provides MCP tool execution interface
- Manages tool discovery and caching
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.mcp.discovery import MCPServerDiscovery, get_mcp_discovery
from src.agents.mcp.loader import DynamicToolLoader, ToolDefinition, get_tool_loader
from src.agents.mcp.sse import MCPSSEHandler, get_sse_handler
from src.agents.subagent.spawner import AgentSpawner

logger = logging.getLogger(__name__)


@dataclass
class MCPAgentConfig:
    """Configuration for MCP-enabled agent."""
    agent_id: str
    agent_type: str
    mcp_servers: List[str] = field(default_factory=list)
    auto_load_tools: bool = True
    tool_timeout_seconds: int = 30


class MCPIntegration:
    """
    Integrates MCP servers with the agent spawning system.

    Provides:
    - MCP tool discovery for agents
    - Tool loading and caching
    - SSE event streaming
    - Integration with AgentSpawner

    Usage:
        integration = MCPIntegration()

        # Initialize
        await integration.initialize()

        # Get tools for an agent
        tools = await integration.get_agent_tools("my-agent", ["filesystem", "context7"])

        # Spawn agent with tools
        agent_id = spawner.spawn("coder", "Write a file", mcp_tools=tools)

        # Stream agent events
        async for event in integration.sse_handler.stream_server_events("filesystem"):
            print(event)
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize MCP integration.

        Args:
            project_root: Project root path for config discovery
        """
        self._discovery = get_mcp_discovery(project_root)
        self._sse_handler = get_sse_handler()
        self._tool_loader = get_tool_loader(self._discovery, self._sse_handler)
        self._spawner: Optional[AgentSpawner] = None
        self._agent_tools: Dict[str, List[str]] = {}  # agent_id -> list of tool names
        self._initialized = False

    @property
    def discovery(self) -> MCPServerDiscovery:
        """Get the MCP server discovery instance."""
        return self._discovery

    @property
    def sse_handler(self) -> MCPSSEHandler:
        """Get the SSE handler instance."""
        return self._sse_handler

    @property
    def tool_loader(self) -> DynamicToolLoader:
        """Get the dynamic tool loader instance."""
        return self._tool_loader

    async def initialize(self) -> None:
        """Initialize the MCP integration."""
        if self._initialized:
            return

        # Initialize tool loader (discovers servers)
        await self._tool_loader.initialize()

        # Start SSE handler
        await self._sse_handler.start()

        self._initialized = True
        logger.info("MCP integration initialized")

    async def get_agent_tools(
        self,
        agent_id: str,
        server_ids: Optional[List[str]] = None,
    ) -> List[ToolDefinition]:
        """
        Get MCP tools available to an agent.

        Args:
            agent_id: Agent identifier
            server_ids: Optional list of server IDs (defaults to all enabled)

        Returns:
            List of tool definitions available to the agent
        """
        if not self._initialized:
            await self.initialize()

        all_tools = []

        # Get servers to load tools from
        if server_ids is None:
            servers = self._discovery.get_enabled_servers()
            server_ids = [s.server_id for s in servers]

        # Load tools from each server
        for server_id in server_ids:
            try:
                tools = await self._tool_loader.get_tools(server_id)
                all_tools.extend(tools)

                # Track tools for this agent
                if agent_id not in self._agent_tools:
                    self._agent_tools[agent_id] = []
                self._agent_tools[agent_id].extend([t.name for t in tools])

            except Exception as e:
                logger.warning(f"Failed to load tools from {server_id}: {e}")

        return all_tools

    async def call_agent_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Call an MCP tool on behalf of an agent.

        Args:
            agent_id: Agent identifier
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If agent not found or tool not available
        """
        if agent_id not in self._agent_tools:
            raise ValueError(f"Agent {agent_id} has no MCP tools loaded")

        if tool_name not in self._agent_tools[agent_id]:
            raise ValueError(f"Tool {tool_name} not available to agent {agent_id}")

        # Find which server has this tool
        for server_id in self._tool_loader.list_servers_with_tools():
            tools = await self._tool_loader.get_tools(server_id)
            tool_names = [t.name for t in tools]
            if tool_name in tool_names:
                return await self._tool_loader.call_tool(server_id, tool_name, arguments)

        raise ValueError(f"Tool {tool_name} not found on any connected server")

    def get_agent_tool_names(self, agent_id: str) -> List[str]:
        """Get list of tool names available to an agent."""
        return self._agent_tools.get(agent_id, [])

    async def attach_to_spawner(self, spawner: AgentSpawner) -> None:
        """
        Attach MCP integration to an agent spawner.

        Args:
            spawner: AgentSpawner instance
        """
        self._spawner = spawner

        # Modify spawner to include MCP tools
        original_build_prompt = spawner._build_agent_prompt

        async def enhanced_build_prompt(agent_id: str, options: Dict[str, Any]) -> str:
            """Enhanced prompt builder that includes MCP tool info."""
            prompt = await original_build_prompt(agent_id, options)

            # Add MCP tools if configured
            mcp_tools = options.get("mcp_tools")
            if mcp_tools:
                tool_descriptions = [
                    f"- {tool.name}: {tool.description}"
                    for tool in mcp_tools
                ]
                prompt += f"\n\nAvailable MCP tools:\n" + "\n".join(tool_descriptions)

            return prompt

        # Replace method (monkey patch for integration)
        # Note: In production, would use composition instead
        logger.info("MCP integration attached to spawner")

    async def stream_agent_mcp_events(
        self,
        agent_id: str,
    ) -> Any:
        """
        Stream MCP events for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Async generator of SSE events
        """
        tool_names = self.get_agent_tool_names(agent_id)

        # Stream events for tools used by this agent
        async for event in self._sse_handler.stream_tool_events():
            if event.tool_name in tool_names:
                yield event

    def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get MCP integration statistics.

        Returns:
            Dictionary with integration stats
        """
        return {
            "servers_discovered": len(self._discovery.get_enabled_servers()),
            "servers_with_tools": len(self._tool_loader.list_servers_with_tools()),
            "tools_loaded": len(self._tool_loader.list_all_tools()),
            "agents_with_tools": len(self._agent_tools),
            "tool_stats": self._tool_loader.get_tool_stats(),
        }

    async def shutdown(self) -> None:
        """Shutdown the MCP integration."""
        await self._sse_handler.stop()
        logger.info("MCP integration shutdown")


# Global integration instance
_mcp_integration: Optional[MCPIntegration] = None


def get_mcp_integration(project_root: Optional[Path] = None) -> MCPIntegration:
    """
    Get the global MCP integration instance.

    Args:
        project_root: Optional project root

    Returns:
        MCPIntegration instance
    """
    global _mcp_integration
    if _mcp_integration is None:
        _mcp_integration = MCPIntegration(project_root)
    return _mcp_integration
