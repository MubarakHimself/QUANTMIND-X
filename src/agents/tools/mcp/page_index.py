"""
PageIndex MCP Tools Module.

Provides tools for PDF indexing and search via PageIndex MCP server.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.tools.mcp.manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def index_pdf(
    pdf_path: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a PDF document using PageIndex.

    This tool indexes a PDF file into the PageIndex knowledge base
    for later retrieval and search.

    Args:
        pdf_path: Path to the PDF file
        namespace: Namespace to index into (e.g., "mql5_book", "strategies")
        metadata: Optional metadata for the document

    Returns:
        Dictionary containing:
        - job_id: Indexing job identifier
        - status: Indexing status
        - pages_indexed: Number of pages indexed
    """
    logger.info(f"Indexing PDF: {pdf_path} into namespace: {namespace}")

    # Validate PDF path
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "index-document",
            {
                "path": pdf_path,
                "namespace": namespace,
                "metadata": metadata or {}
            }
        )

        if isinstance(result, dict):
            return {
                "job_id": result.get("job_id", f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "completed"),
                "pdf_path": pdf_path,
                "namespace": namespace,
                "pages_indexed": result.get("pages_indexed", 0),
                "metadata": metadata or {}
            }
        return {
            "job_id": f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "unknown",
            "pdf_path": pdf_path,
            "namespace": namespace,
            "pages_indexed": 0,
            "metadata": metadata or {}
        }

    except Exception as e:
        logger.error(f"PDF indexing failed: {e}")
        raise RuntimeError(f"Failed to index PDF: {e}")


async def search_pdf(
    query: str,
    namespace: str,
    max_results: int = 5,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search indexed PDFs using PageIndex.

    Args:
        query: Search query
        namespace: Namespace to search in
        max_results: Maximum number of results
        include_content: Whether to include full content in results

    Returns:
        Dictionary containing search results
    """
    logger.info(f"Searching PDF namespace {namespace}: {query}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "search",
            {
                "query": query,
                "namespace": namespace,
                "max_results": max_results,
                "include_content": include_content
            }
        )

        if isinstance(result, dict):
            return {
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "query": query,
                "namespace": namespace
            }
        return {
            "results": [],
            "total": 0,
            "query": query,
            "namespace": namespace
        }

    except Exception as e:
        logger.error(f"PDF search failed: {e}")
        raise RuntimeError(f"Failed to search PDFs: {e}")


async def get_indexed_documents(namespace: str) -> Dict[str, Any]:
    """
    Get list of indexed documents in a namespace.

    Args:
        namespace: Namespace to list documents from

    Returns:
        Dictionary containing list of indexed documents
    """
    logger.info(f"Getting indexed documents for namespace: {namespace}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "pageindex",
            "list-documents",
            {"namespace": namespace}
        )

        if isinstance(result, dict):
            return {
                "namespace": namespace,
                "documents": result.get("documents", []),
                "total": result.get("total", 0)
            }
        return {"namespace": namespace, "documents": [], "total": 0}

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise RuntimeError(f"Failed to list indexed documents: {e}")
