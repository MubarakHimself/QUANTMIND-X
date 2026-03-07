"""
Files API for Compliance Documents

Provides file storage and retrieval for compliance documents,
trade logs, and reports within the agent SDK.

**Validates: Task 11 - Files API for Compliance**
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Types of files supported by the compliance system."""
    COMPLIANCE_DOCUMENT = "compliance_document"
    TRADE_LOG = "trade_log"
    REPORT = "report"
    AUDIT_LOG = "audit_log"
    CONFIG = "config"


class FileStatus(Enum):
    """Status of stored files."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class FileMetadata:
    """Metadata for stored files."""
    file_id: str
    filename: str
    file_type: FileType
    content_hash: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: List[str]
    status: FileStatus
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "file_type": self.file_type.value,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "tags": self.tags,
            "status": self.status.value,
            "metadata": self.metadata,
        }


class FilesAPI:
    """
    Files API for storing and retrieving compliance documents.

    Provides storage and retrieval for:
    - Compliance documents
    - Trade logs
    - Reports
    - Audit logs

    Features:
    - File metadata and search
    - Content hashing for integrity
    - Tag-based organization
    - Status management (active/archived/deleted)
    """

    def __init__(
        self,
        storage_path: str = "data/compliance_files",
        max_file_size_mb: int = 100,
    ):
        """
        Initialize the Files API.

        Args:
            storage_path: Base path for file storage
            max_file_size_mb: Maximum file size in megabytes
        """
        self.storage_path = Path(storage_path)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._index: Dict[str, FileMetadata] = {}
        self._index_file = self.storage_path / ".file_index.json"

        # Create storage directory structure
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialize storage directories and load index."""
        # Create type-specific subdirectories
        for file_type in FileType:
            (self.storage_path / file_type.value).mkdir(parents=True, exist_ok=True)

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load file index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    data = json.load(f)
                    for file_id, item in data.items():
                        item["created_at"] = datetime.fromisoformat(item["created_at"])
                        item["updated_at"] = datetime.fromisoformat(item["updated_at"])
                        item["file_type"] = FileType(item["file_type"])
                        item["status"] = FileStatus(item["status"])
                        self._index[file_id] = FileMetadata(**item)
            except Exception as e:
                logger.warning(f"Failed to load file index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save file index to disk."""
        data = {
            file_id: metadata.to_dict()
            for file_id, metadata in self._index.items()
        }
        with open(self._index_file, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def store_file(
        self,
        content: bytes,
        filename: str,
        file_type: FileType,
        created_by: str = "system",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FileMetadata:
        """
        Store a file in the compliance system.

        Args:
            content: File content as bytes
            filename: Original filename
            file_type: Type of file (compliance_document, trade_log, report, etc.)
            created_by: Identifier for who created the file
            tags: Optional list of tags for organization
            metadata: Optional additional metadata

        Returns:
            FileMetadata object with file information

        Raises:
            ValueError: If file exceeds max size
            ValueError: If file_type is invalid
        """
        # Validate file size
        if len(content) > self.max_file_size_bytes:
            raise ValueError(
                f"File size {len(content)} exceeds maximum {self.max_file_size_bytes}"
            )

        # Validate file type
        if not isinstance(file_type, FileType):
            raise ValueError(f"Invalid file type: {file_type}")

        # Generate unique file ID
        file_id = str(uuid.uuid4())
        content_hash = self._compute_hash(content)

        # Get current timestamp
        now = datetime.now(timezone.utc)

        # Create metadata
        file_metadata = FileMetadata(
            file_id=file_id,
            filename=filename,
            file_type=file_type,
            content_hash=content_hash,
            size_bytes=len(content),
            created_at=now,
            updated_at=now,
            created_by=created_by,
            tags=tags or [],
            status=FileStatus.ACTIVE,
            metadata=metadata or {},
        )

        # Determine storage path
        type_dir = self.storage_path / file_type.value
        file_path = type_dir / f"{file_id}_{filename}"

        # Write file content
        with open(file_path, "wb") as f:
            f.write(content)

        # Update index
        self._index[file_id] = file_metadata
        self._save_index()

        logger.info(f"Stored file {file_id} ({filename}) as {file_type.value}")
        return file_metadata

    def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """
        Retrieve file content by ID.

        Args:
            file_id: Unique file identifier

        Returns:
            File content as bytes, or None if not found
        """
        metadata = self._index.get(file_id)
        if not metadata or metadata.status == FileStatus.DELETED:
            return None

        file_path = self.storage_path / metadata.file_type.value / f"{file_id}_{metadata.filename}"
        if not file_path.exists():
            logger.warning(f"File {file_id} not found on disk")
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def get_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """
        Get file metadata by ID.

        Args:
            file_id: Unique file identifier

        Returns:
            FileMetadata object, or None if not found
        """
        metadata = self._index.get(file_id)
        if metadata and metadata.status != FileStatus.DELETED:
            return metadata
        return None

    def search_files(
        self,
        file_type: Optional[FileType] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        status: FileStatus = FileStatus.ACTIVE,
        limit: int = 100,
    ) -> List[FileMetadata]:
        """
        Search for files based on criteria.

        Args:
            file_type: Filter by file type
            tags: Filter by tags (any match)
            created_by: Filter by creator
            status: Filter by status (default: active)
            limit: Maximum results to return

        Returns:
            List of matching FileMetadata objects
        """
        results = []

        for metadata in self._index.values():
            # Skip deleted files unless explicitly requested
            if metadata.status == FileStatus.DELETED and status != FileStatus.DELETED:
                continue

            # Apply status filter
            if status and metadata.status != status:
                continue

            # Apply file type filter
            if file_type and metadata.file_type != file_type:
                continue

            # Apply creator filter
            if created_by and metadata.created_by != created_by:
                continue

            # Apply tags filter (any tag match)
            if tags:
                if not any(tag in metadata.tags for tag in tags):
                    continue

            results.append(metadata)

            if len(results) >= limit:
                break

        # Sort by creation date (newest first)
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results

    def update_file_status(
        self, file_id: str, status: FileStatus
    ) -> Optional[FileMetadata]:
        """
        Update file status (active/archived/deleted).

        Args:
            file_id: Unique file identifier
            status: New status

        Returns:
            Updated FileMetadata, or None if not found
        """
        metadata = self._index.get(file_id)
        if not metadata:
            return None

        metadata.status = status
        metadata.updated_at = datetime.now(timezone.utc)
        self._save_index()

        logger.info(f"Updated file {file_id} status to {status.value}")
        return metadata

    def delete_file(self, file_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a file (soft delete by default).

        Args:
            file_id: Unique file identifier
            hard_delete: If True, permanently delete the file

        Returns:
            True if successful, False if file not found
        """
        metadata = self._index.get(file_id)
        if not metadata:
            return False

        if hard_delete:
            # Remove from disk
            file_path = self.storage_path / metadata.file_type.value / f"{file_id}_{metadata.filename}"
            if file_path.exists():
                file_path.unlink()

            # Remove from index
            del self._index[file_id]
            logger.info(f"Hard deleted file {file_id}")
        else:
            # Soft delete
            metadata.status = FileStatus.DELETED
            metadata.updated_at = datetime.now(timezone.utc)
            logger.info(f"Soft deleted file {file_id}")

        self._save_index()
        return True

    def add_tags(self, file_id: str, tags: List[str]) -> Optional[FileMetadata]:
        """
        Add tags to a file.

        Args:
            file_id: Unique file identifier
            tags: Tags to add

        Returns:
            Updated FileMetadata, or None if not found
        """
        metadata = self._index.get(file_id)
        if not metadata:
            return None

        # Add new tags (avoiding duplicates)
        for tag in tags:
            if tag not in metadata.tags:
                metadata.tags.append(tag)

        metadata.updated_at = datetime.now(timezone.utc)
        self._save_index()

        return metadata

    def remove_tags(self, file_id: str, tags: List[str]) -> Optional[FileMetadata]:
        """
        Remove tags from a file.

        Args:
            file_id: Unique file identifier
            tags: Tags to remove

        Returns:
            Updated FileMetadata, or None if not found
        """
        metadata = self._index.get(file_id)
        if not metadata:
            return None

        metadata.tags = [t for t in metadata.tags if t not in tags]
        metadata.updated_at = datetime.now(timezone.utc)
        self._save_index()

        return metadata

    def list_file_types(self) -> List[str]:
        """
        List all available file types in the system.

        Returns:
            List of file type values
        """
        return [ft.value for ft in FileType]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored files.

        Returns:
            Dictionary with file statistics
        """
        stats = {
            "total_files": len(self._index),
            "active_files": 0,
            "archived_files": 0,
            "deleted_files": 0,
            "by_type": {},
            "total_size_bytes": 0,
        }

        for metadata in self._index.values():
            # Count by status
            if metadata.status == FileStatus.ACTIVE:
                stats["active_files"] += 1
            elif metadata.status == FileStatus.ARCHIVED:
                stats["archived_files"] += 1
            elif metadata.status == FileStatus.DELETED:
                stats["deleted_files"] += 1

            # Count by type
            type_key = metadata.file_type.value
            if type_key not in stats["by_type"]:
                stats["by_type"][type_key] = {"count": 0, "size_bytes": 0}
            stats["by_type"][type_key]["count"] += 1
            stats["by_type"][type_key]["size_bytes"] += metadata.size_bytes

            # Total size (only active files)
            if metadata.status == FileStatus.ACTIVE:
                stats["total_size_bytes"] += metadata.size_bytes

        return stats
