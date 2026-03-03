# src/agents/tools/knowledge/client.py
"""PageIndex MCP Client Integration."""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def get_pageindex_manager():
    """Get the MCP manager for PageIndex operations."""
    from src.agents.tools.mcp_tools import get_mcp_manager
    return await get_mcp_manager()


async def call_pageindex_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a PageIndex MCP tool with proper error handling.

    Args:
        tool_name: Name of the PageIndex tool to call
        arguments: Arguments to pass to the tool

    Returns:
        Tool result from PageIndex MCP server

    Raises:
        RuntimeError: If the MCP call fails
    """
    try:
        manager = await get_pageindex_manager()
        result = await manager.call_tool("pageindex", tool_name, arguments)
        return result
    except Exception as e:
        logger.error(f"PageIndex MCP call failed for {tool_name}: {e}")
        raise RuntimeError(f"PageIndex MCP error: {e}")


class PageIndexClient:
    """Client for PageIndex MCP operations."""

    def __init__(self):
        """Initialize the PageIndex client."""
        pass

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a PageIndex tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        return await call_pageindex_tool(tool_name, arguments)

    async def search(self, query: str, namespace: str, max_results: int = 10,
                     include_content: bool = True) -> Dict[str, Any]:
        """
        Search a namespace.

        Args:
            query: Search query
            namespace: Namespace to search
            max_results: Maximum results
            include_content: Include content snippets

        Returns:
            Search results
        """
        return await self.call_tool("search", {
            "query": query,
            "namespace": namespace,
            "max_results": max_results,
            "include_content": include_content
        })

    async def get_page(self, namespace: str, page: int) -> Dict[str, Any]:
        """
        Get a specific page.

        Args:
            namespace: Namespace
            page: Page number

        Returns:
            Page content
        """
        return await self.call_tool("get-page", {
            "namespace": namespace,
            "page": page
        })

    async def list_namespaces(self) -> Dict[str, Any]:
        """
        List available namespaces.

        Returns:
            List of namespaces
        """
        return await self.call_tool("list-namespaces", {})
