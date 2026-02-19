"""
File History API Endpoints.

REST API for file history operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.file_history import (
    FileHistoryStore,
    get_file_history_store,
    init_file_history_store,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/files", tags=["file-history"])


# Request/Response Models
class AddVersionRequest(BaseModel):
    """Request to add a new file version."""
    file_path: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class RevertRequest(BaseModel):
    """Request to revert to a version."""
    version_id: str
    create_backup: bool = True


class DiffRequest(BaseModel):
    """Request for diff between versions."""
    version_a_id: Optional[str] = None
    version_b_id: Optional[str] = None
    context_lines: int = 3


def get_store() -> FileHistoryStore:
    """Get file history store."""
    return get_file_history_store()


@router.get("/history/{file_path:path}")
async def get_file_history(
    file_path: str,
    include_content: bool = Query(False, description="Include content in versions"),
    limit: Optional[int] = Query(None, description="Max versions to return"),
) -> Dict[str, Any]:
    """
    Get complete history for a file.

    Returns all versions with timestamps and optionally content.
    """
    store = get_store()
    history = store.get_history(file_path)

    if not history:
        raise HTTPException(status_code=404, detail="File history not found")

    versions = history.versions
    if limit:
        versions = versions[-limit:]

    return {
        "file_path": file_path,
        "file_id": history.file_id,
        "total_versions": history.total_versions,
        "current_version_id": history.current_version_id,
        "created_at": history.created_at.isoformat(),
        "updated_at": history.updated_at.isoformat(),
        "versions": [
            {
                "version_id": v.version_id,
                "created_at": v.created_at.isoformat(),
                "size_bytes": v.size_bytes,
                "content_hash": v.content_hash[:16],
                "metadata": v.metadata,
                **({"content": v.content} if include_content else {}),
            }
            for v in versions
        ],
    }


@router.get("/version/{file_path:path}")
async def get_file_version(
    file_path: str,
    version_id: Optional[str] = Query(None, description="Specific version ID"),
    version_index: Optional[int] = Query(None, description="Version index"),
) -> Dict[str, Any]:
    """
    Get a specific version of a file.

    Returns file content and version metadata.
    """
    store = get_store()

    if version_id:
        version = store.get_version(version_id)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
    elif version_index is not None:
        version = store.get_version_at(file_path, version_index)
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
    else:
        version = store.get_current_version(file_path)
        if not version:
            raise HTTPException(status_code=404, detail="No versions found")

    return {
        "version_id": version.version_id,
        "file_path": version.file_path,
        "content": version.content,
        "created_at": version.created_at.isoformat(),
        "size_bytes": version.size_bytes,
        "content_hash": version.content_hash,
        "metadata": version.metadata,
    }


@router.post("/diff")
async def get_file_diff(request: DiffRequest) -> Dict[str, Any]:
    """
    Get diff between two file versions.

    Returns unified diff with statistics.
    """
    store = get_store()

    if not request.version_a_id or not request.version_b_id:
        raise HTTPException(
            status_code=400,
            detail="Both version_a_id and version_b_id are required"
        )

    diff_result = store.get_diff(
        request.version_a_id,
        request.version_b_id,
        request.context_lines
    )

    if not diff_result:
        raise HTTPException(status_code=404, detail="Could not compute diff")

    return {
        "version_a_id": diff_result.version_a_id,
        "version_b_id": diff_result.version_b_id,
        "added_lines": diff_result.added_lines,
        "removed_lines": diff_result.removed_lines,
        "changed_lines": diff_result.changed_lines,
        "diff": diff_result.diff_text,
        "hunks": diff_result.hunks,
    }


@router.post("/revert")
async def revert_file(request: RevertRequest) -> Dict[str, Any]:
    """
    Revert file to a previous version.

    Creates new version with old content (non-destructive).
    """
    # This would need file_path from context
    raise HTTPException(status_code=501, detail="Use revert endpoint with file path")


@router.post("/revert/{file_path:path}")
async def revert_file_with_path(
    file_path: str,
    request: RevertRequest,
) -> Dict[str, Any]:
    """
    Revert file to a previous version.

    Creates new version with old content (non-destructive).
    """
    store = get_store()

    new_version = store.revert_to_version(file_path, request.version_id)

    if not new_version:
        raise HTTPException(status_code=400, detail="Revert failed")

    # Also write to file
    try:
        target_version = store.get_version(request.version_id)
        if target_version:
            Path(file_path).write_text(target_version.content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not write to file: {e}")

    return {
        "reverted_to": request.version_id,
        "new_version_id": new_version.version_id,
        "file_path": file_path,
    }


@router.get("/recent-changes")
async def get_recent_changes(
    limit: int = Query(50, description="Max changes to return"),
    file_pattern: Optional[str] = Query(None, description="Filter by pattern"),
    include_content: bool = Query(False, description="Include content preview"),
) -> Dict[str, Any]:
    """
    Get recent file changes across workspace.

    Returns most recent versions with timestamps.
    """
    store = get_store()
    versions = store.get_recent_changes(limit, file_pattern)

    return {
        "changes": [
            {
                "version_id": v.version_id,
                "file_path": v.file_path,
                "created_at": v.created_at.isoformat(),
                "size_bytes": v.size_bytes,
                "content_hash": v.content_hash[:16],
                **({"content_preview": v.content[:500]} if include_content else {}),
            }
            for v in versions
        ],
        "total_returned": len(versions),
    }


@router.post("/version")
async def add_file_version(request: AddVersionRequest) -> Dict[str, Any]:
    """
    Add a new version to file history.

    Creates version entry with provided content.
    """
    store = get_store()

    version = store.add_version(
        file_path=request.file_path,
        content=request.content,
        metadata=request.metadata,
    )

    return {
        "version_id": version.version_id,
        "file_path": version.file_path,
        "created_at": version.created_at.isoformat(),
        "size_bytes": version.size_bytes,
        "content_hash": version.content_hash,
    }


@router.delete("/history/{file_path:path}")
async def delete_file_history(file_path: str) -> Dict[str, Any]:
    """
    Delete entire history for a file.

    This cannot be undone.
    """
    store = get_store()

    success = store.delete_history(file_path)

    if not success:
        raise HTTPException(status_code=404, detail="History not found")

    return {
        "deleted": True,
        "file_path": file_path,
    }


@router.post("/prune/{file_path:path}")
async def prune_file_history(
    file_path: str,
    keep_count: int = Query(10, description="Number of versions to keep"),
) -> Dict[str, Any]:
    """
    Prune old versions, keeping only recent N.

    Removes oldest versions beyond keep_count.
    """
    store = get_store()

    removed_count = store.prune_old_versions(file_path, keep_count)

    return {
        "file_path": file_path,
        "removed_count": removed_count,
        "kept_count": keep_count,
    }


@router.get("/statistics")
async def get_statistics() -> Dict[str, Any]:
    """
    Get file history store statistics.

    Returns counts and sizes.
    """
    store = get_store()
    return store.get_statistics()
