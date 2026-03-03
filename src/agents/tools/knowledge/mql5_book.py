# src/agents/tools/knowledge/mql5_book.py
"""MQL5 Book Search Tools."""

import logging
from typing import Dict, Any, Optional
from src.agents.tools.knowledge.client import call_pageindex_tool

logger = logging.getLogger(__name__)


async def search_mql5_book(
    query: str,
    max_results: int = 5,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search the MQL5 programming book for relevant content.

    This tool searches the indexed MQL5 book (mql5book.pdf) to find
    relevant documentation, examples, and explanations using PageIndex MCP.

    Args:
        query: Search query (e.g., "how to create indicator", "OrderSend function")
        max_results: Maximum number of results to return (default: 5)
        include_content: Whether to include full content snippets (default: True)

    Returns:
        Dictionary containing:
        - results: List of matching book sections
        - total: Total number of matches
        - query: Original query
    """
    logger.info(f"Searching MQL5 book: {query}")

    try:
        # Call PageIndex MCP to search the mql5_book namespace
        result = await call_pageindex_tool("search", {
            "query": query,
            "namespace": "mql5_book",
            "max_results": max_results,
            "include_content": include_content
        })

        if isinstance(result, dict):
            results = []
            for item in result.get("results", []):
                results.append({
                    "page": item.get("page", 0),
                    "chapter": item.get("chapter", ""),
                    "content": item.get("content") if include_content else None,
                    "relevance": item.get("relevance", 0.0),
                    "section_title": item.get("section_title", item.get("title", ""))
                })

            return {
                "success": True,
                "results": results,
                "total": result.get("total", len(results)),
                "query": query,
                "namespace": "mql5_book"
            }

        return {
            "success": True,
            "results": [],
            "total": 0,
            "query": query,
            "namespace": "mql5_book"
        }

    except Exception as e:
        logger.error(f"MQL5 book search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query,
            "namespace": "mql5_book"
        }


async def get_mql5_book_section(
    page_number: int
) -> Dict[str, Any]:
    """
    Get a specific section from the MQL5 book by page number.

    Args:
        page_number: Page number to retrieve

    Returns:
        Dictionary containing the page content
    """
    logger.info(f"Getting MQL5 book page: {page_number}")

    try:
        # Call PageIndex MCP to get specific page
        result = await call_pageindex_tool("get-page", {
            "namespace": "mql5_book",
            "page": page_number
        })

        if isinstance(result, dict):
            return {
                "success": True,
                "page": page_number,
                "content": result.get("content", ""),
                "chapter": result.get("chapter", ""),
                "section": result.get("section", "")
            }

        return {
            "success": False,
            "page": page_number,
            "content": "",
            "chapter": "",
            "section": "",
            "error": "Page not found"
        }

    except Exception as e:
        logger.error(f"Failed to get MQL5 book page {page_number}: {e}")
        return {
            "success": False,
            "page": page_number,
            "content": "",
            "chapter": "",
            "section": "",
            "error": str(e)
        }
