"""
Agent MCP Package

Provides enhanced MCP integration for agents including:
- Server discovery from multiple sources
- Dynamic tool loading
- SSE support for real-time updates
"""

from src.agents.mcp.discovery import MCPServerDiscovery, get_mcp_discovery
from src.agents.mcp.loader import DynamicToolLoader, get_tool_loader
from src.agents.mcp.sse import MCPSSEHandler, get_sse_handler

__all__ = [
    "MCPServerDiscovery",
    "get_mcp_discovery",
    "DynamicToolLoader",
    "get_tool_loader",
    "MCPSSEHandler",
    "get_sse_handler",
]
