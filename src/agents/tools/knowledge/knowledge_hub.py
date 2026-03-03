# src/agents/tools/knowledge/knowledge_hub.py
"""Knowledge Hub Tools."""

import logging
from typing import Dict, Any, List, Optional
from src.agents.tools.knowledge.client import call_pageindex_tool

logger = logging.getLogger(__name__)


async def search_knowledge_hub(
    query: str,
    namespaces: Optional[List[str]] = None,
    max_results: int = 10,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search all indexed PDFs in the knowledge hub using PageIndex MCP.

    This tool searches across all indexed documents including:
    - mql5_book: MQL5 programming book
    - strategies: Trading strategies documentation
    - knowledge: General knowledge base

    Args:
        query: Search query
        namespaces: List of namespaces to search (default: all)
        max_results: Maximum results per namespace (default: 10)
        include_content: Whether to include content snippets (default: True)

    Returns:
        Dictionary containing:
        - results: List of matching documents grouped by namespace
        - total: Total number of matches
        - query: Original query
    """
    logger.info(f"Searching knowledge hub: {query}")

    if namespaces is None:
        namespaces = ["mql5_book", "strategies", "knowledge"]

    try:
        # Call PageIndex MCP for multi-namespace search
        result = await call_pageindex_tool("search-all", {
            "query": query,
            "namespaces": namespaces,
            "max_results": max_results,
            "include_content": include_content
        })

        results = []
        if isinstance(result, dict):
            # Group results by namespace
            namespace_results = {}
            for item in result.get("results", []):
                ns = item.get("namespace", "unknown")
                if ns not in namespace_results:
                    namespace_results[ns] = []
                namespace_results[ns].append({
                    "document": item.get("filename", item.get("document", "")),
                    "page": item.get("page", 1),
                    "content": item.get("content") if include_content else None,
                    "relevance": item.get("relevance", 0.0)
                })

            for ns, matches in namespace_results.items():
                results.append({
                    "namespace": ns,
                    "matches": matches
                })

        return {
            "success": True,
            "results": results,
            "total": sum(len(r["matches"]) for r in results),
            "query": query,
            "namespaces_searched": namespaces
        }

    except Exception as e:
        logger.error(f"Knowledge hub search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query,
            "namespaces_searched": namespaces
        }


async def get_article_content(
    article_id: str,
    namespace: str
) -> Dict[str, Any]:
    """
    Retrieve full content of an indexed article/document using PageIndex MCP.

    Args:
        article_id: Article or document identifier
        namespace: Namespace the article belongs to

    Returns:
        Dictionary containing the full article content
    """
    logger.info(f"Getting article content: {article_id} from {namespace}")

    try:
        # Call PageIndex MCP to get document content
        result = await call_pageindex_tool("get-document", {
            "document_id": article_id,
            "namespace": namespace
        })

        if isinstance(result, dict):
            return {
                "success": True,
                "article_id": article_id,
                "namespace": namespace,
                "title": result.get("title", f"Article {article_id}"),
                "content": result.get("content", ""),
                "metadata": {
                    "indexed_at": result.get("indexed_at", ""),
                    "pages": result.get("pages", 0)
                }
            }

        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": "Article not found"
        }

    except Exception as e:
        logger.error(f"Failed to get article content: {e}")
        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": str(e)
        }


async def list_knowledge_namespaces() -> Dict[str, Any]:
    """
    List all available knowledge namespaces using PageIndex MCP.

    Returns:
        Dictionary containing list of namespaces with metadata
    """
    logger.info("Listing knowledge namespaces")

    try:
        # Call PageIndex MCP to list namespaces
        result = await call_pageindex_tool("list-namespaces", {})

        namespaces = []
        if isinstance(result, dict):
            for ns in result.get("namespaces", []):
                namespaces.append({
                    "name": ns.get("name", ""),
                    "description": ns.get("description", ""),
                    "document_count": ns.get("document_count", 0),
                    "total_pages": ns.get("total_pages", 0),
                    "indexed_at": ns.get("indexed_at")
                })

        # Add default namespaces if not present
        default_namespaces = {
            "mql5_book": "MQL5 Programming Book",
            "strategies": "Trading strategies documentation",
            "knowledge": "General knowledge base"
        }

        existing_names = [ns["name"] for ns in namespaces]
        for name, desc in default_namespaces.items():
            if name not in existing_names:
                namespaces.append({
                    "name": name,
                    "description": desc,
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                })

        return {
            "success": True,
            "namespaces": sorted(namespaces, key=lambda x: x["name"]),
            "total": len(namespaces)
        }

    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}")
        return {
            "success": False,
            "error": str(e),
            "namespaces": [
                {
                    "name": "mql5_book",
                    "description": "MQL5 Programming Book",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "strategies",
                    "description": "Trading strategies documentation",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "knowledge",
                    "description": "General knowledge base",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                }
            ],
            "total": 3
        }
