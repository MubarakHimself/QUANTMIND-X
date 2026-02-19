"""
File History Tools for QuantMind agents.

These tools expose FileHistoryManager functionality to agents:
- get_file_history: Get complete history for a file
- get_file_version: Get specific version content
- compare_versions: Compare two file versions with diff
- revert_to_version: Revert file to previous version
- get_recent_changes: Get recent file changes across workspace
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType
from ..services.file_history import (
    FileHistoryStore,
    FileVersion,
    DiffResult,
    get_file_history_store,
)


logger = logging.getLogger(__name__)


class GetFileHistoryInput(BaseModel):
    """Input schema for get_file_history tool."""
    file_path: str = Field(
        description="Path to the file to get history for"
    )
    include_content: bool = Field(
        default=False,
        description="Whether to include content in each version"
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of versions to return"
    )


class GetFileVersionInput(BaseModel):
    """Input schema for get_file_version tool."""
    file_path: str = Field(
        description="Path to the file"
    )
    version_id: Optional[str] = Field(
        default=None,
        description="Specific version ID (returns current if not specified)"
    )
    version_index: Optional[int] = Field(
        default=None,
        description="Version index (0=oldest, -1=newest)"
    )


class CompareVersionsInput(BaseModel):
    """Input schema for compare_versions tool."""
    file_path: str = Field(
        description="Path to the file"
    )
    version_a_id: Optional[str] = Field(
        default=None,
        description="First version ID (defaults to previous)"
    )
    version_b_id: Optional[str] = Field(
        default=None,
        description="Second version ID (defaults to current)"
    )
    context_lines: int = Field(
        default=3,
        description="Number of context lines in diff"
    )


class RevertToVersionInput(BaseModel):
    """Input schema for revert_to_version tool."""
    file_path: str = Field(
        description="Path to the file to revert"
    )
    version_id: str = Field(
        description="Version ID to revert to"
    )
    create_backup: bool = Field(
        default=True,
        description="Create backup of current version before reverting"
    )


class GetRecentChangesInput(BaseModel):
    """Input schema for get_recent_changes tool."""
    limit: int = Field(
        default=50,
        description="Maximum number of recent changes to return"
    )
    file_pattern: Optional[str] = Field(
        default=None,
        description="Filter by file path pattern"
    )
    include_content: bool = Field(
        default=False,
        description="Include content preview in results"
    )


class AddFileVersionInput(BaseModel):
    """Input schema for add_file_version tool."""
    file_path: str = Field(
        description="Path to the file"
    )
    content: str = Field(
        description="File content to save"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for this version"
    )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "history", "version"],
)
class GetFileHistoryTool(QuantMindTool):
    """Get complete history for a file."""

    name: str = "get_file_history"
    description: str = """Get the complete version history for a file.
    Returns all versions with timestamps, sizes, and optionally content.
    Useful for understanding how a file has evolved over time."""

    args_schema: type[BaseModel] = GetFileHistoryInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        file_path: str,
        include_content: bool = False,
        limit: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute file history retrieval."""
        # Validate path is within workspace
        validated_path = self.validate_workspace_path(file_path)

        history = self._store.get_history(str(validated_path))

        if not history:
            raise ToolError(
                f"No history found for file '{file_path}'",
                tool_name=self.name,
                error_code="HISTORY_NOT_FOUND"
            )

        versions = history.versions
        if limit:
            versions = versions[-limit:]

        version_data = [
            {
                "version_id": v.version_id,
                "created_at": v.created_at.isoformat(),
                "size_bytes": v.size_bytes,
                "content_hash": v.content_hash[:16],
                "metadata": v.metadata,
                **({"content": v.content} if include_content else {}),
            }
            for v in versions
        ]

        return ToolResult.ok(
            data={
                "file_path": file_path,
                "file_id": history.file_id,
                "total_versions": history.total_versions,
                "current_version_id": history.current_version_id,
                "created_at": history.created_at.isoformat(),
                "updated_at": history.updated_at.isoformat(),
                "versions": version_data,
            },
            metadata={
                "versions_returned": len(version_data),
                "include_content": include_content,
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "version", "content"],
)
class GetFileVersionTool(QuantMindTool):
    """Get specific version of a file."""

    name: str = "get_file_version"
    description: str = """Get the content of a specific file version.
    Can retrieve by version ID or index (0=oldest, -1=newest).
    Returns the file content and version metadata."""

    args_schema: type[BaseModel] = GetFileVersionInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        file_path: str,
        version_id: Optional[str] = None,
        version_index: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute file version retrieval."""
        validated_path = self.validate_workspace_path(file_path)

        # Get version
        if version_id:
            version = self._store.get_version(version_id)
            if not version:
                raise ToolError(
                    f"Version '{version_id}' not found",
                    tool_name=self.name,
                    error_code="VERSION_NOT_FOUND"
                )
        elif version_index is not None:
            version = self._store.get_version_at(str(validated_path), version_index)
            if not version:
                raise ToolError(
                    f"Version at index {version_index} not found",
                    tool_name=self.name,
                    error_code="VERSION_NOT_FOUND"
                )
        else:
            version = self._store.get_current_version(str(validated_path))
            if not version:
                raise ToolError(
                    f"No versions found for file '{file_path}'",
                    tool_name=self.name,
                    error_code="NO_VERSIONS"
                )

        return ToolResult.ok(
            data={
                "version_id": version.version_id,
                "file_path": version.file_path,
                "content": version.content,
                "created_at": version.created_at.isoformat(),
                "size_bytes": version.size_bytes,
                "content_hash": version.content_hash,
                "metadata": version.metadata,
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE, AgentType.ANALYST],
    tags=["file", "diff", "compare", "version"],
)
class CompareVersionsTool(QuantMindTool):
    """Compare two file versions with diff."""

    name: str = "compare_versions"
    description: str = """Compare two versions of a file and get a unified diff.
    Defaults to comparing previous version with current version.
    Returns line-by-line differences with statistics."""

    args_schema: type[BaseModel] = CompareVersionsInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        file_path: str,
        version_a_id: Optional[str] = None,
        version_b_id: Optional[str] = None,
        context_lines: int = 3,
        **kwargs
    ) -> ToolResult:
        """Execute version comparison."""
        validated_path = self.validate_workspace_path(file_path)
        history = self._store.get_history(str(validated_path))

        if not history or len(history.versions) < 1:
            raise ToolError(
                f"Not enough versions to compare for file '{file_path}'",
                tool_name=self.name,
                error_code="INSUFFICIENT_VERSIONS"
            )

        # Default to comparing previous with current
        if not version_a_id and not version_b_id:
            if len(history.versions) < 2:
                raise ToolError(
                    "Need at least 2 versions to compare",
                    tool_name=self.name,
                    error_code="INSUFFICIENT_VERSIONS"
                )
            version_a_id = history.versions[-2].version_id
            version_b_id = history.versions[-1].version_id
        elif not version_b_id:
            version_b_id = history.current_version_id

        diff_result = self._store.get_diff(version_a_id, version_b_id, context_lines)

        if not diff_result:
            raise ToolError(
                f"Could not compare versions {version_a_id} and {version_b_id}",
                tool_name=self.name,
                error_code="DIFF_FAILED"
            )

        return ToolResult.ok(
            data={
                "version_a_id": diff_result.version_a_id,
                "version_b_id": diff_result.version_b_id,
                "added_lines": diff_result.added_lines,
                "removed_lines": diff_result.removed_lines,
                "changed_lines": diff_result.changed_lines,
                "diff": diff_result.diff_text,
                "hunks": diff_result.hunks,
            },
            metadata={
                "context_lines": context_lines,
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["file", "revert", "version"],
)
class RevertToVersionTool(QuantMindTool):
    """Revert file to a previous version."""

    name: str = "revert_to_version"
    description: str = """Revert a file to a previous version.
    Creates a new version with the old content (non-destructive).
    Optionally creates backup of current version."""

    args_schema: type[BaseModel] = RevertToVersionInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.HIGH

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        file_path: str,
        version_id: str,
        create_backup: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute version revert."""
        validated_path = self.validate_workspace_path(file_path)

        # Get version to revert to
        target_version = self._store.get_version(version_id)
        if not target_version:
            raise ToolError(
                f"Version '{version_id}' not found",
                tool_name=self.name,
                error_code="VERSION_NOT_FOUND"
            )

        # Create backup if requested
        current_version = self._store.get_current_version(str(validated_path))
        if create_backup and current_version:
            # Backup is implicit - the current version stays in history
            pass

        # Revert by creating new version with old content
        new_version = self._store.revert_to_version(str(validated_path), version_id)

        if not new_version:
            raise ToolError(
                f"Failed to revert to version '{version_id}'",
                tool_name=self.name,
                error_code="REVERT_FAILED"
            )

        # Also write to actual file
        try:
            validated_path.write_text(target_version.content, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not write reverted content to file: {e}")

        return ToolResult.ok(
            data={
                "reverted_to": version_id,
                "new_version_id": new_version.version_id,
                "file_path": file_path,
            },
            metadata={
                "create_backup": create_backup,
                "reverted_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["file", "history", "recent", "workspace"],
)
class GetRecentChangesTool(QuantMindTool):
    """Get recent file changes across workspace."""

    name: str = "get_recent_changes"
    description: str = """Get recent file changes across the entire workspace.
    Returns list of most recent file versions with timestamps.
    Can filter by file pattern."""

    args_schema: type[BaseModel] = GetRecentChangesInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        limit: int = 50,
        file_pattern: Optional[str] = None,
        include_content: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute recent changes retrieval."""
        versions = self._store.get_recent_changes(limit, file_pattern)

        changes = []
        for v in versions:
            change = {
                "version_id": v.version_id,
                "file_path": v.file_path,
                "created_at": v.created_at.isoformat(),
                "size_bytes": v.size_bytes,
                "content_hash": v.content_hash[:16],
            }
            if include_content:
                # Include preview (first 500 chars)
                change["content_preview"] = v.content[:500]
            changes.append(change)

        return ToolResult.ok(
            data={
                "changes": changes,
                "total_returned": len(changes),
            },
            metadata={
                "limit": limit,
                "file_pattern": file_pattern,
                "include_content": include_content,
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "history", "save"],
)
class AddFileVersionTool(QuantMindTool):
    """Add a new version to file history."""

    name: str = "add_file_version"
    description: str = """Add a new version to a file's history.
    Creates a new version entry with the provided content.
    Returns version ID and metadata."""

    args_schema: type[BaseModel] = AddFileVersionInput
    category: ToolCategory = ToolCategory.FILE_HISTORY
    priority: ToolPriority = ToolPriority.NORMAL

    def __init__(self, **data):
        super().__init__(**data)
        self._store = get_file_history_store()

    def execute(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute version addition."""
        validated_path = self.validate_workspace_path(file_path)

        version = self._store.add_version(
            file_path=str(validated_path),
            content=content,
            metadata=metadata,
        )

        return ToolResult.ok(
            data={
                "version_id": version.version_id,
                "file_path": version.file_path,
                "created_at": version.created_at.isoformat(),
                "size_bytes": version.size_bytes,
                "content_hash": version.content_hash,
            },
            metadata={
                "added_at": datetime.now().isoformat(),
            }
        )


# Export all tools
__all__ = [
    "GetFileHistoryTool",
    "GetFileVersionTool",
    "CompareVersionsTool",
    "RevertToVersionTool",
    "GetRecentChangesTool",
    "AddFileVersionTool",
]
