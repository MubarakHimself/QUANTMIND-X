# src/agents/tools/knowledge/pdf_indexing.py
"""PDF Indexing Tools."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from src.agents.tools.knowledge.client import call_pageindex_tool

logger = logging.getLogger(__name__)


async def index_pdf_document(
    pdf_path: str,
    namespace: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Index a PDF document into the knowledge hub using PageIndex MCP.

    This tool indexes a PDF file for later retrieval and search.

    Args:
        pdf_path: Path to the PDF file
        namespace: Namespace to index into (e.g., "strategies", "knowledge")
        metadata: Optional metadata for the document

    Returns:
        Dictionary containing:
        - job_id: Indexing job identifier
        - status: Indexing status
        - message: Status message
    """
    logger.info(f"Indexing PDF: {pdf_path} into {namespace}")

    # Validate path
    path = Path(pdf_path)
    if not path.exists():
        return {
            "success": False,
            "error": f"PDF file not found: {pdf_path}",
            "status": "failed"
        }

    if not path.suffix.lower() == ".pdf":
        return {
            "success": False,
            "error": "File must be a PDF",
            "status": "failed"
        }

    try:
        # Call PageIndex MCP to index the document
        result = await call_pageindex_tool("index-document", {
            "path": pdf_path,
            "namespace": namespace,
            "metadata": metadata or {}
        })

        if isinstance(result, dict):
            return {
                "success": True,
                "job_id": result.get("job_id", f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                "status": result.get("status", "completed"),
                "pdf_path": pdf_path,
                "namespace": namespace,
                "pages_indexed": result.get("pages_indexed", 0),
                "message": f"Successfully indexed {path.name} into {namespace}"
            }

        return {
            "success": True,
            "job_id": f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "pdf_path": pdf_path,
            "namespace": namespace,
            "pages_indexed": 0,
            "message": f"Indexing completed for {path.name}"
        }

    except Exception as e:
        logger.error(f"PDF indexing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "status": "failed",
            "pdf_path": pdf_path,
            "namespace": namespace
        }


async def get_indexing_status(
    job_id: str
) -> Dict[str, Any]:
    """
    Get status of a PDF indexing job using PageIndex MCP.

    Args:
        job_id: Indexing job identifier

    Returns:
        Dictionary containing job status
    """
    logger.info(f"Getting indexing status: {job_id}")

    try:
        # Call PageIndex MCP to get job status
        result = await call_pageindex_tool("get-job-status", {
            "job_id": job_id
        })

        if isinstance(result, dict):
            return {
                "success": True,
                "job_id": job_id,
                "status": result.get("status", "unknown"),
                "progress": result.get("progress", 0),
                "pages_processed": result.get("pages_processed", 0),
                "pages_total": result.get("pages_total", 0),
                "started_at": result.get("started_at"),
                "completed_at": result.get("completed_at")
            }

        return {
            "success": True,
            "job_id": job_id,
            "status": "unknown",
            "progress": 0,
            "pages_processed": 0,
            "pages_total": 0,
            "started_at": None,
            "completed_at": None
        }

    except Exception as e:
        logger.error(f"Failed to get indexing status: {e}")
        return {
            "success": False,
            "job_id": job_id,
            "status": "error",
            "error": str(e)
        }


async def list_indexed_documents(
    namespace: str
) -> Dict[str, Any]:
    """
    List all indexed documents in a namespace using PageIndex MCP.

    Args:
        namespace: Namespace to list documents from

    Returns:
        Dictionary containing list of documents
    """
    logger.info(f"Listing indexed documents in {namespace}")

    try:
        # Call PageIndex MCP to list documents
        result = await call_pageindex_tool("list-documents", {
            "namespace": namespace
        })

        if isinstance(result, dict):
            documents = []
            for doc in result.get("documents", []):
                documents.append({
                    "id": doc.get("id", ""),
                    "filename": doc.get("filename", ""),
                    "pages": doc.get("pages", 0),
                    "indexed_at": doc.get("indexed_at", ""),
                    "size_bytes": doc.get("size_bytes", 0)
                })

            return {
                "success": True,
                "namespace": namespace,
                "documents": documents,
                "total": len(documents)
            }

        return {
            "success": True,
            "namespace": namespace,
            "documents": [],
            "total": 0
        }

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return {
            "success": False,
            "namespace": namespace,
            "documents": [],
            "total": 0,
            "error": str(e)
        }


async def remove_indexed_document(
    document_id: str,
    namespace: str
) -> Dict[str, Any]:
    """
    Remove an indexed document from the knowledge hub using PageIndex MCP.

    Args:
        document_id: Document identifier
        namespace: Namespace the document belongs to

    Returns:
        Dictionary containing removal status
    """
    logger.info(f"Removing document: {document_id} from {namespace}")

    try:
        # Call PageIndex MCP to remove document
        result = await call_pageindex_tool("remove-document", {
            "document_id": document_id,
            "namespace": namespace
        })

        if isinstance(result, dict):
            return {
                "success": result.get("success", True),
                "document_id": document_id,
                "namespace": namespace,
                "message": result.get("message", "Document removed successfully")
            }

        return {
            "success": True,
            "document_id": document_id,
            "namespace": namespace,
            "message": "Document removed successfully"
        }

    except Exception as e:
        logger.error(f"Failed to remove document: {e}")
        return {
            "success": False,
            "document_id": document_id,
            "namespace": namespace,
            "error": str(e)
        }
