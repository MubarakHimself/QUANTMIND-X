"""
File History Service for QuantMind.

This service provides backend file version management,
replacing the frontend localStorage-based FileHistoryManager.
Uses database persistence for cross-session access.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class FileVersion:
    """Represents a single version of a file."""
    version_id: str
    file_id: str
    file_path: str
    content: str
    content_hash: str
    created_at: datetime
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version_id: Optional[str] = None


@dataclass
class FileHistory:
    """Complete history for a file."""
    file_id: str
    file_path: str
    versions: List[FileVersion]
    current_version_id: str
    total_versions: int
    created_at: datetime
    updated_at: datetime


@dataclass
class DiffResult:
    """Result of comparing two file versions."""
    version_a_id: str
    version_b_id: str
    added_lines: int
    removed_lines: int
    changed_lines: int
    diff_text: str
    hunks: List[Dict[str, Any]]


class FileHistoryStore:
    """
    Database-backed store for file history.

    In production, this would use SQLite or another database.
    For now, uses in-memory storage with optional JSON persistence.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self._histories: Dict[str, FileHistory] = {}
        self._versions: Dict[str, FileVersion] = {}

        if storage_path:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load histories from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())
            # Reconstruct histories from JSON
            for file_id, history_data in data.get("histories", {}).items():
                versions = [
                    FileVersion(
                        version_id=v["version_id"],
                        file_id=v["file_id"],
                        file_path=v["file_path"],
                        content=v["content"],
                        content_hash=v["content_hash"],
                        created_at=datetime.fromisoformat(v["created_at"]),
                        size_bytes=v["size_bytes"],
                        metadata=v.get("metadata", {}),
                        parent_version_id=v.get("parent_version_id"),
                    )
                    for v in history_data.get("versions", [])
                ]

                self._histories[file_id] = FileHistory(
                    file_id=history_data["file_id"],
                    file_path=history_data["file_path"],
                    versions=versions,
                    current_version_id=history_data["current_version_id"],
                    total_versions=history_data["total_versions"],
                    created_at=datetime.fromisoformat(history_data["created_at"]),
                    updated_at=datetime.fromisoformat(history_data["updated_at"]),
                )

                for version in versions:
                    self._versions[version.version_id] = version

            logger.info(f"Loaded {len(self._histories)} file histories from disk")
        except Exception as e:
            logger.error(f"Failed to load histories: {e}")

    def _save_to_disk(self) -> None:
        """Save histories to disk."""
        if not self.storage_path:
            return

        try:
            data = {
                "histories": {
                    file_id: {
                        "file_id": h.file_id,
                        "file_path": h.file_path,
                        "versions": [
                            {
                                "version_id": v.version_id,
                                "file_id": v.file_id,
                                "file_path": v.file_path,
                                "content": v.content,
                                "content_hash": v.content_hash,
                                "created_at": v.created_at.isoformat(),
                                "size_bytes": v.size_bytes,
                                "metadata": v.metadata,
                                "parent_version_id": v.parent_version_id,
                            }
                            for v in h.versions
                        ],
                        "current_version_id": h.current_version_id,
                        "total_versions": h.total_versions,
                        "created_at": h.created_at.isoformat(),
                        "updated_at": h.updated_at.isoformat(),
                    }
                    for file_id, h in self._histories.items()
                }
            }

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save histories: {e}")

    def add_version(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FileVersion:
        """Add a new version of a file."""
        # Generate IDs
        file_id = self._generate_file_id(file_path)
        version_id = self._generate_version_id()
        content_hash = self._hash_content(content)

        # Check if content changed
        history = self._histories.get(file_id)
        if history and history.versions:
            latest = history.versions[-1]
            if latest.content_hash == content_hash:
                # Content unchanged, return existing version
                return latest

        # Create version
        version = FileVersion(
            version_id=version_id,
            file_id=file_id,
            file_path=file_path,
            content=content,
            content_hash=content_hash,
            created_at=datetime.now(),
            size_bytes=len(content.encode("utf-8")),
            metadata=metadata or {},
            parent_version_id=history.current_version_id if history else None,
        )

        # Update or create history
        if history:
            history.versions.append(version)
            history.current_version_id = version_id
            history.total_versions = len(history.versions)
            history.updated_at = datetime.now()
        else:
            history = FileHistory(
                file_id=file_id,
                file_path=file_path,
                versions=[version],
                current_version_id=version_id,
                total_versions=1,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self._histories[file_id] = history

        self._versions[version_id] = version
        self._save_to_disk()

        logger.info(f"Added version {version_id} for file {file_path}")
        return version

    def get_history(self, file_path: str) -> Optional[FileHistory]:
        """Get complete history for a file."""
        file_id = self._generate_file_id(file_path)
        return self._histories.get(file_id)

    def get_version(self, version_id: str) -> Optional[FileVersion]:
        """Get a specific version by ID."""
        return self._versions.get(version_id)

    def get_version_at(self, file_path: str, index: int) -> Optional[FileVersion]:
        """Get version at specific index (0 = oldest, -1 = newest)."""
        history = self.get_history(file_path)
        if not history or not history.versions:
            return None

        try:
            return history.versions[index]
        except IndexError:
            return None

    def get_current_version(self, file_path: str) -> Optional[FileVersion]:
        """Get the current (latest) version."""
        return self.get_version_at(file_path, -1)

    def get_diff(
        self,
        version_a_id: str,
        version_b_id: str,
        context_lines: int = 3,
    ) -> Optional[DiffResult]:
        """Get diff between two versions."""
        version_a = self._versions.get(version_a_id)
        version_b = self._versions.get(version_b_id)

        if not version_a or not version_b:
            return None

        # Calculate diff
        lines_a = version_a.content.splitlines(keepends=True)
        lines_b = version_b.content.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"{version_a.file_path} (v{version_a.version_id[:8]})",
            tofile=f"{version_b.file_path} (v{version_b.version_id[:8]})",
            n=context_lines,
        ))

        # Parse hunks
        hunks = self._parse_hunks(diff)

        # Count changes
        added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

        return DiffResult(
            version_a_id=version_a_id,
            version_b_id=version_b_id,
            added_lines=added,
            removed_lines=removed,
            changed_lines=max(added, removed),
            diff_text="".join(diff),
            hunks=hunks,
        )

    def _parse_hunks(self, diff_lines: List[str]) -> List[Dict[str, Any]]:
        """Parse diff hunks."""
        hunks = []
        current_hunk = None

        for line in diff_lines:
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append(current_hunk)
                # Parse hunk header
                match = re.match(r"@@ -(\d+),?\d* \+(\d+),?\d* @@", line)
                if match:
                    current_hunk = {
                        "old_start": int(match.group(1)),
                        "new_start": int(match.group(2)),
                        "lines": [],
                    }
                else:
                    # Regex didn't match, skip this hunk header
                    current_hunk = None
            elif current_hunk is not None:
                current_hunk["lines"].append(line)

        if current_hunk:
            hunks.append(current_hunk)

        return hunks

    def revert_to_version(self, file_path: str, version_id: str) -> Optional[FileVersion]:
        """Revert file to a specific version (creates new version with old content)."""
        version = self._versions.get(version_id)
        if not version:
            return None

        # Create new version with old content
        return self.add_version(
            file_path=file_path,
            content=version.content,
            metadata={
                "reverted_from": version_id,
                "revert_timestamp": datetime.now().isoformat(),
            }
        )

    def get_recent_changes(
        self,
        limit: int = 50,
        file_pattern: Optional[str] = None,
    ) -> List[FileVersion]:
        """Get recent changes across all files."""
        all_versions = list(self._versions.values())

        # Filter by pattern if specified
        if file_pattern:
            all_versions = [
                v for v in all_versions
                if file_pattern.lower() in v.file_path.lower()
            ]

        # Sort by creation time, newest first
        all_versions.sort(key=lambda v: v.created_at, reverse=True)

        return all_versions[:limit]

    def delete_history(self, file_path: str) -> bool:
        """Delete entire history for a file."""
        file_id = self._generate_file_id(file_path)
        history = self._histories.get(file_id)

        if not history:
            return False

        # Remove versions
        for version in history.versions:
            self._versions.pop(version.version_id, None)

        # Remove history
        del self._histories[file_id]
        self._save_to_disk()

        return True

    def prune_old_versions(self, file_path: str, keep_count: int = 10) -> int:
        """Remove old versions, keeping only the most recent N."""
        history = self.get_history(file_path)
        if not history or len(history.versions) <= keep_count:
            return 0

        # Keep only the last N versions
        to_remove = history.versions[:-keep_count]
        history.versions = history.versions[-keep_count:]
        history.total_versions = len(history.versions)

        # Remove from versions dict
        for version in to_remove:
            self._versions.pop(version.version_id, None)

        self._save_to_disk()
        return len(to_remove)

    def _generate_file_id(self, file_path: str) -> str:
        """Generate consistent file ID from path."""
        normalized = str(Path(file_path).resolve())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        import uuid
        return str(uuid.uuid4())

    def _hash_content(self, content: str) -> str:
        """Generate hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_size = sum(v.size_bytes for v in self._versions.values())
        return {
            "total_files": len(self._histories),
            "total_versions": len(self._versions),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_versions_per_file": (
                len(self._versions) / len(self._histories)
                if self._histories else 0
            ),
        }


# Global store instance
_file_history_store: Optional[FileHistoryStore] = None


def get_file_history_store(storage_path: Optional[Path] = None) -> FileHistoryStore:
    """Get the global file history store instance."""
    global _file_history_store
    if _file_history_store is None:
        _file_history_store = FileHistoryStore(storage_path)
    return _file_history_store


def init_file_history_store(storage_path: Union[str, Path]) -> FileHistoryStore:
    """Initialize the global file history store."""
    global _file_history_store
    _file_history_store = FileHistoryStore(Path(storage_path))
    return _file_history_store
