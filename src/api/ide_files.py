"""
QuantMind IDE Files API for Compliance

API endpoints for file operations with compliance tracking.
Provides full CRUD operations for compliance documents, trade logs,
reports, and audit logs.

**Validates: Task MUB-89 - Files API for Compliance**
"""

import os
import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.agents.files.compliance_files import (
    FilesAPI,
    FileType,
    FileStatus,
    FileMetadata,
)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/files", tags=["files"])

# Initialize FilesAPI
DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
COMPLIANCE_DIR = DATA_DIR / "compliance_files"
files_api = FilesAPI(storage_path=str(COMPLIANCE_DIR), max_file_size_mb=100)


# Request/Response Models
class FileCreateRequest(BaseModel):
    """Request model for creating a file."""
    filename: str
    file_type: str
    content: str  # Base64 encoded content
    created_by: str = "system"
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FileUpdateRequest(BaseModel):
    """Request model for updating a file."""
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FileSearchRequest(BaseModel):
    """Request model for searching files."""
    file_type: Optional[str] = None
    tags: Optional[List[str]] = None
    created_by: Optional[str] = None
    status: Optional[str] = "active"
    limit: int = 100


class FileResponse(BaseModel):
    """Response model for file metadata."""
    file_id: str
    filename: str
    file_type: str
    content_hash: str
    size_bytes: int
    created_at: str
    updated_at: str
    created_by: str
    tags: List[str]
    status: str
    metadata: Dict[str, Any]


class FileContentResponse(BaseModel):
    """Response model for file content."""
    file_id: str
    filename: str
    content: str  # Base64 encoded
    content_type: str


# CRUD Operations

@router.post("", response_model=FileResponse)
async def create_file(
    filename: str = Form(...),
    file_type: str = Form(...),
    content: str = Form(...),  # Base64 encoded
    created_by: str = Form(default="system"),
    tags: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
):
    """
    Create a new file in the compliance system.

    Supports file types:
    - compliance_document
    - trade_log
    - report
    - audit_log
    - config
    """
    try:
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # Parse metadata
        meta_dict = {}
        if metadata:
            import json
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                pass

        # Decode content
        try:
            content_bytes = base64.b64decode(content)
        except Exception as e:
            raise HTTPException(400, f"Invalid base64 content: {str(e)}")

        # Convert file_type string to FileType enum
        try:
            file_type_enum = FileType(file_type)
        except ValueError:
            raise HTTPException(
                400,
                f"Invalid file_type. Must be one of: {[ft.value for ft in FileType]}"
            )

        # Store file
        file_metadata = files_api.store_file(
            content=content_bytes,
            filename=filename,
            file_type=file_type_enum,
            created_by=created_by,
            tags=tag_list,
            metadata=meta_dict,
        )

        return FileResponse(
            file_id=file_metadata.file_id,
            filename=file_metadata.filename,
            file_type=file_metadata.file_type.value,
            content_hash=file_metadata.content_hash,
            size_bytes=file_metadata.size_bytes,
            created_at=file_metadata.created_at.isoformat(),
            updated_at=file_metadata.updated_at.isoformat(),
            created_by=file_metadata.created_by,
            tags=file_metadata.tags,
            status=file_metadata.status.value,
            metadata=file_metadata.metadata,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error creating file: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.get("/{file_id}", response_model=FileResponse)
async def get_file(file_id: str):
    """Get file metadata by ID."""
    metadata = files_api.get_metadata(file_id)
    if not metadata:
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        file_type=metadata.file_type.value,
        content_hash=metadata.content_hash,
        size_bytes=metadata.size_bytes,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        created_by=metadata.created_by,
        tags=metadata.tags,
        status=metadata.status.value,
        metadata=metadata.metadata,
    )


@router.get("/{file_id}/content", response_model=FileContentResponse)
async def get_file_content(file_id: str):
    """Get file content by ID."""
    metadata = files_api.get_metadata(file_id)
    if not metadata:
        raise HTTPException(404, "File not found")

    content = files_api.retrieve_file(file_id)
    if content is None:
        raise HTTPException(404, "File content not found")

    # Encode content as base64
    content_b64 = base64.b64encode(content).decode("utf-8")

    # Determine content type based on extension
    ext = Path(metadata.filename).suffix.lower()
    content_type = "application/octet-stream"
    if ext in [".json"]:
        content_type = "application/json"
    elif ext in [".txt", ".log"]:
        content_type = "text/plain"
    elif ext in [".md"]:
        content_type = "text/markdown"
    elif ext in [".pdf"]:
        content_type = "application/pdf"
    elif ext in [".csv"]:
        content_type = "text/csv"

    return FileContentResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        content=content_b64,
        content_type=content_type,
    )


@router.put("/{file_id}", response_model=FileResponse)
async def update_file(file_id: str, request: FileUpdateRequest):
    """Update file metadata (tags, metadata)."""
    metadata = files_api.get_metadata(file_id)
    if not metadata:
        raise HTTPException(404, "File not found")

    # Update tags if provided
    if request.tags is not None:
        # Get current tags and update
        current_tags = metadata.tags
        # Add new tags
        for tag in request.tags:
            if tag not in current_tags:
                current_tags.append(tag)
        files_api.add_tags(file_id, request.tags)

    # Update metadata if provided
    if request.metadata is not None:
        # Merge metadata
        new_metadata = {**metadata.metadata, **request.metadata}
        # Need to update through the API
        metadata.metadata = new_metadata

    # Get updated metadata
    updated_metadata = files_api.get_metadata(file_id)
    if not updated_metadata:
        raise HTTPException(404, "File not found after update")

    return FileResponse(
        file_id=updated_metadata.file_id,
        filename=updated_metadata.filename,
        file_type=updated_metadata.file_type.value,
        content_hash=updated_metadata.content_hash,
        size_bytes=updated_metadata.size_bytes,
        created_at=updated_metadata.created_at.isoformat(),
        updated_at=updated_metadata.updated_at.isoformat(),
        created_by=updated_metadata.created_by,
        tags=updated_metadata.tags,
        status=updated_metadata.status.value,
        metadata=updated_metadata.metadata,
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str, hard_delete: bool = False):
    """Delete a file (soft delete by default, hard delete optional)."""
    success = files_api.delete_file(file_id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(404, "File not found")

    return {
        "success": True,
        "file_id": file_id,
        "deleted": "hard" if hard_delete else "soft"
    }


@router.post("/{file_id}/archive", response_model=FileResponse)
async def archive_file(file_id: str):
    """Archive a file."""
    metadata = files_api.update_file_status(file_id, FileStatus.ARCHIVED)
    if not metadata:
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        file_type=metadata.file_type.value,
        content_hash=metadata.content_hash,
        size_bytes=metadata.size_bytes,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        created_by=metadata.created_by,
        tags=metadata.tags,
        status=metadata.status.value,
        metadata=metadata.metadata,
    )


@router.post("/{file_id}/restore", response_model=FileResponse)
async def restore_file(file_id: str):
    """Restore an archived or deleted file."""
    metadata = files_api.update_file_status(file_id, FileStatus.ACTIVE)
    if not metadata:
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        file_type=metadata.file_type.value,
        content_hash=metadata.content_hash,
        size_bytes=metadata.size_bytes,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        created_by=metadata.created_by,
        tags=metadata.tags,
        status=metadata.status.value,
        metadata=metadata.metadata,
    )


@router.post("/{file_id}/tags", response_model=FileResponse)
async def add_file_tags(file_id: str, tags: List[str]):
    """Add tags to a file."""
    metadata = files_api.add_tags(file_id, tags)
    if not metadata:
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        file_type=metadata.file_type.value,
        content_hash=metadata.content_hash,
        size_bytes=metadata.size_bytes,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        created_by=metadata.created_by,
        tags=metadata.tags,
        status=metadata.status.value,
        metadata=metadata.metadata,
    )


@router.delete("/{file_id}/tags", response_model=FileResponse)
async def remove_file_tags(file_id: str, tags: List[str]):
    """Remove tags from a file."""
    metadata = files_api.remove_tags(file_id, tags)
    if not metadata:
        raise HTTPException(404, "File not found")

    return FileResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        file_type=metadata.file_type.value,
        content_hash=metadata.content_hash,
        size_bytes=metadata.size_bytes,
        created_at=metadata.created_at.isoformat(),
        updated_at=metadata.updated_at.isoformat(),
        created_by=metadata.created_by,
        tags=metadata.tags,
        status=metadata.status.value,
        metadata=metadata.metadata,
    )


# Search and List Endpoints

@router.get("")
async def list_files(
    file_type: Optional[str] = None,
    status: Optional[str] = "active",
    created_by: Optional[str] = None,
    limit: int = 100,
):
    """
    List files with optional filtering.

    Query parameters:
    - file_type: Filter by file type
    - status: Filter by status (active, archived, deleted)
    - created_by: Filter by creator
    - limit: Maximum number of results
    """
    # Convert status string to FileStatus enum
    status_enum = None
    if status:
        try:
            status_enum = FileStatus(status)
        except ValueError:
            pass

    # Convert file_type string to FileType enum
    file_type_enum = None
    if file_type:
        try:
            file_type_enum = FileType(file_type)
        except ValueError:
            pass

    results = files_api.search_files(
        file_type=file_type_enum,
        status=status_enum,
        created_by=created_by,
        limit=limit,
    )

    return {
        "files": [
            {
                "file_id": m.file_id,
                "filename": m.filename,
                "file_type": m.file_type.value,
                "size_bytes": m.size_bytes,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
                "created_by": m.created_by,
                "tags": m.tags,
                "status": m.status.value,
            }
            for m in results
        ],
        "total": len(results),
    }


@router.post("/search")
async def search_files(request: FileSearchRequest):
    """Search files with advanced filters."""
    # Convert status string to FileStatus enum
    status_enum = None
    if request.status:
        try:
            status_enum = FileStatus(request.status)
        except ValueError:
            pass

    # Convert file_type string to FileType enum
    file_type_enum = None
    if request.file_type:
        try:
            file_type_enum = FileType(request.file_type)
        except ValueError:
            pass

    results = files_api.search_files(
        file_type=file_type_enum,
        tags=request.tags,
        created_by=request.created_by,
        status=status_enum,
        limit=request.limit,
    )

    return {
        "files": [
            {
                "file_id": m.file_id,
                "filename": m.filename,
                "file_type": m.file_type.value,
                "size_bytes": m.size_bytes,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
                "created_by": m.created_by,
                "tags": m.tags,
                "status": m.status.value,
            }
            for m in results
        ],
        "total": len(results),
    }


# Compliance and Stats Endpoints

@router.get("/types")
async def list_file_types():
    """List all available file types."""
    return {"file_types": files_api.list_file_types()}


@router.get("/stats")
async def get_file_stats():
    """Get file storage statistics."""
    return files_api.get_stats()


# Legacy Endpoints (for backward compatibility)

@router.get("/content")
async def get_file_content_legacy(path: str):
    """Get content of any file (legacy endpoint)."""
    file_path = Path(path)

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    if file_path.suffix in ['.mq5', '.mqh', '.py', '.md', '.txt', '.json']:
        return {"content": file_path.read_text()}
    else:
        return {"content": f"Binary file: {file_path.name}", "binary": True}


# Plans directory
PLANS_DIR = Path(__file__).parent.parent.parent / "docs" / "plans"


@router.get("/plans")
async def list_plans():
    """List all implementation plans."""
    if not PLANS_DIR.exists():
        return {"plans": []}

    plans = []
    for f in PLANS_DIR.glob("*.md"):
        stat = f.stat()
        plans.append({
            "id": f.stem,
            "name": f.stem.replace("-", " ").replace("_", " ").title(),
            "filename": f.name,
            "path": str(f),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size": stat.st_size
        })

    plans.sort(key=lambda x: x["modified"], reverse=True)
    return {"plans": plans}


@router.get("/plans/{plan_id}")
async def get_plan_content(plan_id: str):
    """Get content of a specific plan."""
    plan_path = PLANS_DIR / f"{plan_id}.md"
    if not plan_path.exists():
        plan_path = PLANS_DIR / plan_id
        if not plan_path.exists():
            raise HTTPException(404, "Plan not found")

    content = plan_path.read_text()
    return {
        "id": plan_path.stem,
        "name": plan_path.stem.replace("-", " ").replace("_", " ").title(),
        "content": content,
        "path": str(plan_path)
    }
