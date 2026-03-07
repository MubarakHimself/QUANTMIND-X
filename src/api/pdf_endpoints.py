"""
PDF API Endpoints for Knowledge Hub.

This module provides REST API endpoints for PDF upload and indexing
in the QuantMind Knowledge Hub using the PageIndex MCP server.

Uses real PageIndex MCP calls for indexing and search operations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])


# =============================================================================
# Configuration
# =============================================================================

# Upload directory
UPLOAD_DIR = Path("uploads/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Job state persistence directory
JOB_STATE_DIR = Path("data/pdf_jobs")
JOB_STATE_DIR.mkdir(parents=True, exist_ok=True)

# Document metadata persistence directory
DOC_STATE_DIR = Path("data/pdf_documents")
DOC_STATE_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf"}

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


# =============================================================================
# Request/Response Models
# =============================================================================

class PDFUploadResponse(BaseModel):
    """Response for PDF upload."""
    success: bool
    file_id: str
    filename: str
    size_bytes: int
    namespace: str
    indexing_job_id: Optional[str] = None
    message: str


class IndexingStatusResponse(BaseModel):
    """Response for indexing status."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int
    pages_processed: int
    pages_total: int
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class PDFDocument(BaseModel):
    """PDF document metadata."""
    id: str
    filename: str
    namespace: str
    size_bytes: int
    pages: int
    indexed_at: str
    status: str


class PDFListResponse(BaseModel):
    """Response for PDF list."""
    documents: List[PDFDocument]
    total: int
    namespace: Optional[str] = None


class SearchRequest(BaseModel):
    """Request for PDF search."""
    query: str
    namespaces: Optional[List[str]] = None
    max_results: int = 10
    include_content: bool = True


class SearchResult(BaseModel):
    """Single search result."""
    document_id: str
    filename: str
    namespace: str
    page: int
    content: Optional[str] = None
    relevance: float


class SearchResponse(BaseModel):
    """Response for PDF search."""
    success: bool
    query: str
    results: List[SearchResult]
    total: int


# =============================================================================
# PageIndex MCP Client Integration
# =============================================================================

async def get_pageindex_client():
    """Get the PageIndex MCP client for PDF operations."""
    from src.agents.tools.mcp_tools import get_mcp_manager
    manager = await get_mcp_manager()
    return manager


async def call_pageindex_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a PageIndex MCP tool with error handling."""
    try:
        manager = await get_pageindex_client()
        result = await manager.call_tool("pageindex", tool_name, arguments)
        return result
    except Exception as e:
        logger.error(f"PageIndex MCP call failed for {tool_name}: {e}")
        raise RuntimeError(f"PageIndex MCP error: {e}")


# =============================================================================
# Job State Persistence
# =============================================================================

def _load_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job state from disk."""
    state_file = JOB_STATE_DIR / f"{job_id}.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load job state {job_id}: {e}")
    return None


def _save_job_state(job_id: str, state: Dict[str, Any]) -> None:
    """Save job state to disk."""
    state_file = JOB_STATE_DIR / f"{job_id}.json"
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save job state {job_id}: {e}")


def _load_document_state(file_id: str) -> Optional[Dict[str, Any]]:
    """Load document state from disk."""
    state_file = DOC_STATE_DIR / f"{file_id}.json"
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load document state {file_id}: {e}")
    return None


def _save_document_state(file_id: str, state: Dict[str, Any]) -> None:
    """Save document state to disk."""
    state_file = DOC_STATE_DIR / f"{file_id}.json"
    try:
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save document state {file_id}: {e}")


def _list_all_documents() -> List[Dict[str, Any]]:
    """List all document states from disk."""
    documents = []
    for state_file in DOC_STATE_DIR.glob("*.json"):
        try:
            with open(state_file, "r") as f:
                documents.append(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load document from {state_file}: {e}")
    return documents


# =============================================================================
# Helper Functions
# =============================================================================

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
        )


async def save_upload_file(file: UploadFile, file_id: str) -> Path:
    """Save uploaded file to disk."""
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        
        # Check size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024} MB"
            )
        
        buffer.write(content)
    
    return file_path


async def index_pdf_task(
    job_id: str,
    file_path: Path,
    namespace: str,
    file_id: str,
    filename: str
) -> None:
    """
    Background task to index a PDF using PageIndex MCP.
    
    This calls the real PageIndex MCP server to index the PDF.
    """
    logger.info(f"Starting indexing job {job_id} for {filename}")
    
    # Load job state
    job_state = _load_job_state(job_id)
    if not job_state:
        job_state = {
            "job_id": job_id,
            "file_id": file_id,
            "filename": filename,
            "namespace": namespace,
            "status": "pending",
            "progress": 0,
            "pages_processed": 0,
            "pages_total": 0,
            "started_at": None,
            "completed_at": None,
            "error": None
        }
    
    # Update job status
    job_state["status"] = "processing"
    job_state["started_at"] = datetime.now().isoformat()
    _save_job_state(job_id, job_state)
    
    try:
        # Get file size
        size_bytes = file_path.stat().st_size
        
        # Call PageIndex MCP to index the document
        result = await call_pageindex_tool("index-document", {
            "path": str(file_path),
            "namespace": namespace,
            "metadata": {
                "filename": filename,
                "file_id": file_id,
                "uploaded_at": datetime.now().isoformat()
            }
        })
        
        # Update progress from MCP response
        if isinstance(result, dict):
            job_state["progress"] = result.get("progress", 100)
            job_state["pages_processed"] = result.get("pages_processed", 0)
            job_state["pages_total"] = result.get("pages_total", 0)
            pages = result.get("pages_indexed", result.get("pages_total", 0))
        else:
            job_state["progress"] = 100
            pages = 0
        
        # Complete indexing
        job_state["status"] = "completed"
        job_state["progress"] = 100
        job_state["completed_at"] = datetime.now().isoformat()
        _save_job_state(job_id, job_state)
        
        # Store document metadata
        doc_state = {
            "id": file_id,
            "filename": filename,
            "namespace": namespace,
            "size_bytes": size_bytes,
            "pages": pages,
            "indexed_at": datetime.now().isoformat(),
            "status": "indexed",
            "file_path": str(file_path),
            "job_id": job_id
        }
        _save_document_state(file_id, doc_state)
        
        logger.info(f"Completed indexing job {job_id}: {pages} pages indexed")
        
    except Exception as e:
        logger.error(f"Indexing job {job_id} failed: {e}")
        job_state["status"] = "failed"
        job_state["error"] = str(e)
        job_state["completed_at"] = datetime.now().isoformat()
        _save_job_state(job_id, job_state)


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Query("knowledge", description="Namespace to index into"),
    auto_index: bool = Query(True, description="Automatically start indexing")
) -> PDFUploadResponse:
    """
    Upload a PDF file for indexing.
    
    Accepts a PDF file and optionally starts indexing it into the specified namespace
    using the PageIndex MCP server.
    
    Args:
        file: PDF file to upload
        namespace: Namespace to index into (default: "knowledge")
        auto_index: Whether to automatically start indexing (default: True)
        
    Returns:
        PDFUploadResponse with file ID and indexing job ID
    """
    logger.info(f"Uploading PDF: {file.filename} to namespace: {namespace}")
    
    # Validate file
    validate_file(file)
    
    # Generate file ID
    file_id = f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Save file
    file_path = await save_upload_file(file, file_id)
    size_bytes = file_path.stat().st_size
    
    # Create indexing job if auto-indexing
    indexing_job_id = None
    if auto_index:
        indexing_job_id = f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Store job state
        job_state = {
            "job_id": indexing_job_id,
            "file_id": file_id,
            "filename": file.filename,
            "namespace": namespace,
            "status": "pending",
            "progress": 0,
            "pages_processed": 0,
            "pages_total": 0,
            "started_at": None,
            "completed_at": None,
            "error": None
        }
        _save_job_state(indexing_job_id, job_state)
        
        # Start background indexing
        background_tasks.add_task(
            index_pdf_task,
            indexing_job_id,
            file_path,
            namespace,
            file_id,
            file.filename
        )
    
    return PDFUploadResponse(
        success=True,
        file_id=file_id,
        filename=file.filename,
        size_bytes=size_bytes,
        namespace=namespace,
        indexing_job_id=indexing_job_id,
        message="PDF uploaded successfully" + (" and indexing started" if auto_index else "")
    )


@router.get("/status/{job_id}", response_model=IndexingStatusResponse)
async def get_indexing_status(job_id: str) -> IndexingStatusResponse:
    """
    Get the status of a PDF indexing job.
    
    Args:
        job_id: Indexing job identifier
        
    Returns:
        IndexingStatusResponse with current status
    """
    job = _load_job_state(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Indexing job not found")
    
    return IndexingStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        pages_processed=job["pages_processed"],
        pages_total=job["pages_total"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


@router.get("/documents", response_model=PaginatedResponse[PDFDocument])
async def list_documents(
    namespace: Optional[str] = Query(None, description="Filter by namespace"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[PDFDocument]:
    """
    List all indexed PDF documents with pagination.

    Args:
        namespace: Optional namespace filter
        limit: Maximum items to return
        offset: Number of items to skip

    Returns:
        PaginatedResponse with list of documents
    """
    documents = _list_all_documents()

    filtered_docs = []
    for doc in documents:
        if namespace and doc.get("namespace") != namespace:
            continue

        filtered_docs.append(PDFDocument(
            id=doc["id"],
            filename=doc["filename"],
            namespace=doc["namespace"],
            size_bytes=doc["size_bytes"],
            pages=doc["pages"],
            indexed_at=doc["indexed_at"],
            status=doc["status"]
        ))

    total = len(filtered_docs)
    paginated_docs = filtered_docs[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated_docs,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/documents/{file_id}", response_model=PDFDocument)
async def get_document(file_id: str) -> PDFDocument:
    """
    Get details of a specific PDF document.
    
    Args:
        file_id: Document identifier
        
    Returns:
        PDFDocument with document details
    """
    doc = _load_document_state(file_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return PDFDocument(
        id=doc["id"],
        filename=doc["filename"],
        namespace=doc["namespace"],
        size_bytes=doc["size_bytes"],
        pages=doc["pages"],
        indexed_at=doc["indexed_at"],
        status=doc["status"]
    )


@router.delete("/documents/{file_id}")
async def delete_document(file_id: str) -> Dict[str, Any]:
    """
    Delete a PDF document and its index.
    
    Args:
        file_id: Document identifier
        
    Returns:
        Deletion status
    """
    doc = _load_document_state(file_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file
    file_path = Path(doc["file_path"])
    if file_path.exists():
        file_path.unlink()
    
    # Remove from PageIndex MCP
    try:
        await call_pageindex_tool("remove-document", {
            "document_id": file_id,
            "namespace": doc["namespace"]
        })
    except Exception as e:
        logger.warning(f"Failed to remove document from PageIndex: {e}")
    
    # Remove state file
    state_file = DOC_STATE_DIR / f"{file_id}.json"
    if state_file.exists():
        state_file.unlink()
    
    return {
        "success": True,
        "file_id": file_id,
        "message": "Document deleted successfully"
    }


@router.post("/search", response_model=SearchResponse)
async def search_pdfs(request: SearchRequest) -> SearchResponse:
    """
    Search indexed PDF documents using PageIndex MCP.
    
    Args:
        request: Search request with query and options
        
    Returns:
        SearchResponse with matching documents
    """
    logger.info(f"Searching PDFs: {request.query}")
    
    try:
        # Call PageIndex MCP to search
        if request.namespaces and len(request.namespaces) == 1:
            # Single namespace search
            result = await call_pageindex_tool("search", {
                "query": request.query,
                "namespace": request.namespaces[0],
                "max_results": request.max_results,
                "include_content": request.include_content
            })
        else:
            # Multi-namespace or all-namespace search
            result = await call_pageindex_tool("search-all", {
                "query": request.query,
                "namespaces": request.namespaces,
                "max_results": request.max_results,
                "include_content": request.include_content
            })
        
        # Parse results
        results = []
        if isinstance(result, dict):
            for item in result.get("results", []):
                results.append(SearchResult(
                    document_id=item.get("document_id", ""),
                    filename=item.get("filename", ""),
                    namespace=item.get("namespace", ""),
                    page=item.get("page", 1),
                    content=item.get("content") if request.include_content else None,
                    relevance=item.get("relevance", 0.0)
                ))
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        logger.error(f"PDF search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/namespaces")
async def list_namespaces() -> Dict[str, Any]:
    """
    List all available namespaces.
    
    Returns:
        Dictionary with list of namespaces
    """
    try:
        # Call PageIndex MCP to list namespaces
        result = await call_pageindex_tool("list-namespaces", {})
        
        namespaces = []
        if isinstance(result, dict):
            for ns in result.get("namespaces", []):
                namespaces.append({
                    "name": ns.get("name", ""),
                    "document_count": ns.get("document_count", 0)
                })
        
        # Add default namespaces if not present
        default_namespaces = ["mql5_book", "strategies", "knowledge"]
        existing_names = [ns["name"] for ns in namespaces]
        for default_ns in default_namespaces:
            if default_ns not in existing_names:
                namespaces.append({
                    "name": default_ns,
                    "document_count": 0
                })
        
        return {
            "namespaces": sorted(namespaces, key=lambda x: x["name"]),
            "total": len(namespaces)
        }
        
    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}")
        # Return defaults on error
        return {
            "namespaces": [
                {"name": "knowledge", "document_count": 0},
                {"name": "mql5_book", "document_count": 0},
                {"name": "strategies", "document_count": 0}
            ],
            "total": 3
        }


@router.post("/reindex/{file_id}")
async def reindex_document(
    file_id: str,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Re-index a PDF document.
    
    Args:
        file_id: Document identifier
        
    Returns:
        Re-indexing status
    """
    doc = _load_document_state(file_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = Path(doc["file_path"])
    if not file_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Original file no longer exists"
        )
    
    # Create new indexing job
    job_id = f"idx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Store job state
    job_state = {
        "job_id": job_id,
        "file_id": file_id,
        "filename": doc["filename"],
        "namespace": doc["namespace"],
        "status": "pending",
        "progress": 0,
        "pages_processed": 0,
        "pages_total": 0,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "is_reindex": True
    }
    _save_job_state(job_id, job_state)
    
    # Start background re-indexing
    background_tasks.add_task(
        index_pdf_task,
        job_id,
        file_path,
        doc["namespace"],
        file_id,
        doc["filename"]
    )
    
    return {
        "success": True,
        "file_id": file_id,
        "job_id": job_id,
        "message": "Re-indexing started"
    }
