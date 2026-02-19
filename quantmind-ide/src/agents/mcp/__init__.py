"""
MCP (Model Context Protocol) Integration Package.

This package provides MCP server connectivity for QuantMind agents.
"""

from .client import MCPClient, MCPServerConnection
from .adapter import MCPToolAdapter, adapt_mcp_tool

__all__ = [
    "MCPClient",
    "MCPServerConnection",
    "MCPToolAdapter",
    "adapt_mcp_tool",
]
