"""
Context7 MCP Tools Module.

Provides tools for MQL5 documentation retrieval via Context7 MCP server.
"""

import logging
from typing import Dict, Any, Optional

from src.agents.tools.mcp.manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def get_mql5_documentation(
    query: str,
    context: Optional[str] = None,
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Retrieve MQL5 documentation from Context7.

    This tool queries the MQL5 documentation through Context7 MCP server
    to get relevant documentation, examples, and API references.

    Args:
        query: Search query for MQL5 documentation
        context: Optional context to narrow search (e.g., "indicators", "trading")
        max_results: Maximum number of results to return

    Returns:
        Dictionary containing:
        - results: List of documentation entries
        - total: Total number of matching documents
        - query: Original query
    """
    logger.info(f"Querying MQL5 documentation: {query}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "context7",
            "query-docs",
            {
                "query": query,
                "context": context,
                "max_results": max_results
            }
        )

        # Parse result from MCP server
        if isinstance(result, dict):
            return {
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "query": query
            }
        return {"results": [], "total": 0, "query": query}

    except Exception as e:
        logger.error(f"Context7 query failed: {e}")
        raise RuntimeError(f"Failed to query MQL5 documentation: {e}")


async def get_mql5_examples(
    topic: str,
    language: str = "mql5"
) -> Dict[str, Any]:
    """
    Get MQL5 code examples for a specific topic.

    Args:
        topic: Topic to get examples for (e.g., "moving average", "order send")
        language: Programming language (default: mql5)

    Returns:
        Dictionary containing code examples with explanations
    """
    logger.info(f"Getting MQL5 examples for: {topic}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "context7",
            "get-examples",
            {
                "topic": topic,
                "language": language
            }
        )

        if isinstance(result, dict):
            return {
                "examples": result.get("examples", []),
                "topic": topic,
                "language": language
            }
        return {"examples": [], "topic": topic, "language": language}

    except Exception as e:
        logger.error(f"Failed to get MQL5 examples: {e}")
        raise RuntimeError(f"Failed to get MQL5 examples: {e}")
