"""
File Operation Tools for QuantMind agents.

These tools provide secure file operations within workspace boundaries:
- read_file: Read file contents
- write_file: Write file with parent directory creation
- list_files: List directory contents
- delete_file: Delete file with confirmation
- copy_file: Copy file to another location
- move_file: Move/rename file
"""

from __future__ import annotations

import logging
import shutil
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


logger = logging.getLogger(__name__)


class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""
    file_path: str = Field(
        description="Path to the file to read, relative to workspace root"
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding (default: utf-8)"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Start line number for partial read (1-indexed)"
    )
    end_line: Optional[int] = Field(
        default=None,
        description="End line number for partial read (1-indexed)"
    )


class WriteFileInput(BaseModel):
    """Input schema for write_file tool."""
    file_path: str = Field(
        description="Path to the file to write, relative to workspace root"
    )
    content: str = Field(
        description="Content to write to the file"
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding (default: utf-8)"
    )
    overwrite: bool = Field(
        default=True,
        description="Whether to overwrite existing file"
    )
    create_dirs: bool = Field(
        default=True,
        description="Whether to create parent directories if they don't exist"
    )


class ListFilesInput(BaseModel):
    """Input schema for list_files tool."""
    directory: str = Field(
        default=".",
        description="Directory path to list, relative to workspace root"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Glob pattern to filter files (e.g., '*.mq5')"
    )
    recursive: bool = Field(
        default=False,
        description="Whether to list files recursively"
    )
    include_hidden: bool = Field(
        default=False,
        description="Whether to include hidden files/folders"
    )


class DeleteFileInput(BaseModel):
    """Input schema for delete_file tool."""
    file_path: str = Field(
        description="Path to the file to delete, relative to workspace root"
    )
    confirm: bool = Field(
        default=False,
        description="Confirmation flag to prevent accidental deletion"
    )


class CopyFileInput(BaseModel):
    """Input schema for copy_file tool."""
    source: str = Field(
        description="Source file path, relative to workspace root"
    )
    destination: str = Field(
        description="Destination file path, relative to workspace root"
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing destination file"
    )


class MoveFileInput(BaseModel):
    """Input schema for move_file tool."""
    source: str = Field(
        description="Source file path, relative to workspace root"
    )
    destination: str = Field(
        description="Destination file path, relative to workspace root"
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing destination file"
    )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "read", "io"],
)
class ReadFileTool(QuantMindTool):
    """Read file contents from the workspace."""

    name: str = "read_file"
    description: str = """Read the contents of a file from the workspace.
    Returns the file contents as a string.
    Use start_line and end_line for partial reads of large files.
    Supports various encodings (default: utf-8)."""

    args_schema: type[BaseModel] = ReadFileInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        file_path: str,
        encoding: str = "utf-8",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        **kwargs
    ) -> ToolResult:
        """Execute file read operation."""
        # Validate path
        validated_path = self.validate_workspace_path(file_path)

        # Check file exists
        if not validated_path.exists():
            raise ToolError(
                f"File '{file_path}' does not exist",
                tool_name=self.name,
                error_code="FILE_NOT_FOUND"
            )

        if not validated_path.is_file():
            raise ToolError(
                f"Path '{file_path}' is not a file",
                tool_name=self.name,
                error_code="NOT_A_FILE"
            )

        # Validate file size
        self.validate_file_size(validated_path)

        # Read file
        try:
            with open(validated_path, "r", encoding=encoding) as f:
                if start_line is not None or end_line is not None:
                    lines = f.readlines()
                    start = (start_line or 1) - 1  # Convert to 0-indexed
                    end = end_line or len(lines)
                    content = "".join(lines[start:end])
                    total_lines = len(lines)
                else:
                    content = f.read()
                    total_lines = content.count("\n") + 1

            return ToolResult.ok(
                data=content,
                metadata={
                    "file_path": file_path,
                    "total_lines": total_lines,
                    "encoding": encoding,
                    "partial_read": start_line is not None or end_line is not None,
                }
            )

        except UnicodeDecodeError as e:
            raise ToolError(
                f"Failed to decode file '{file_path}' with encoding '{encoding}': {e}",
                tool_name=self.name,
                error_code="ENCODING_ERROR"
            )
        except Exception as e:
            raise ToolError(
                f"Failed to read file '{file_path}': {e}",
                tool_name=self.name,
                error_code="READ_ERROR"
            )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "write", "io"],
)
class WriteFileTool(QuantMindTool):
    """Write content to a file in the workspace."""

    name: str = "write_file"
    description: str = """Write content to a file in the workspace.
    Creates parent directories if they don't exist.
    Can optionally not overwrite existing files.
    Returns the number of bytes written."""

    args_schema: type[BaseModel] = WriteFileInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = True,
        create_dirs: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute file write operation."""
        # Validate path
        validated_path = self.validate_workspace_path(file_path)

        # Check if file exists
        if validated_path.exists() and not overwrite:
            raise ToolError(
                f"File '{file_path}' already exists and overwrite is False",
                tool_name=self.name,
                error_code="FILE_EXISTS"
            )

        # Create parent directories
        if create_dirs:
            validated_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        try:
            with open(validated_path, "w", encoding=encoding) as f:
                bytes_written = f.write(content)

            return ToolResult.ok(
                data={
                    "bytes_written": bytes_written,
                    "file_path": file_path,
                },
                metadata={
                    "file_path": file_path,
                    "encoding": encoding,
                    "created_dirs": create_dirs and not validated_path.parent.exists(),
                }
            )

        except Exception as e:
            raise ToolError(
                f"Failed to write file '{file_path}': {e}",
                tool_name=self.name,
                error_code="WRITE_ERROR"
            )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "list", "directory", "io"],
)
class ListFilesTool(QuantMindTool):
    """List files and directories in a workspace path."""

    name: str = "list_files"
    description: str = """List files and directories in a workspace path.
    Can filter by glob pattern and list recursively.
    Returns a list of file/directory information including name, type, size."""

    args_schema: type[BaseModel] = ListFilesInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = NORMAL

    def execute(
        self,
        directory: str = ".",
        pattern: Optional[str] = None,
        recursive: bool = False,
        include_hidden: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute file list operation."""
        # Validate path
        validated_path = self.validate_workspace_path(directory)

        # Check directory exists
        if not validated_path.exists():
            raise ToolError(
                f"Directory '{directory}' does not exist",
                tool_name=self.name,
                error_code="DIRECTORY_NOT_FOUND"
            )

        if not validated_path.is_dir():
            raise ToolError(
                f"Path '{directory}' is not a directory",
                tool_name=self.name,
                error_code="NOT_A_DIRECTORY"
            )

        # Get file list
        try:
            items = []

            if recursive:
                glob_pattern = "**/*" if pattern is None else f"**/{pattern}"
            else:
                glob_pattern = "*" if pattern is None else pattern

            for item in validated_path.glob(glob_pattern):
                # Skip hidden files if not requested
                if not include_hidden and any(
                    part.startswith(".") for part in item.parts
                ):
                    continue

                # Get relative path
                rel_path = item.relative_to(validated_path)

                # Get item info
                item_info = {
                    "name": item.name,
                    "path": str(rel_path),
                    "type": "directory" if item.is_dir() else "file",
                }

                if item.is_file():
                    stat = item.stat()
                    item_info["size"] = stat.st_size
                    item_info["modified"] = stat.st_mtime

                items.append(item_info)

            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (x["type"] != "directory", x["name"].lower()))

            return ToolResult.ok(
                data=items,
                metadata={
                    "directory": directory,
                    "pattern": pattern,
                    "recursive": recursive,
                    "total_items": len(items),
                }
            )

        except Exception as e:
            raise ToolError(
                f"Failed to list directory '{directory}': {e}",
                tool_name=self.name,
                error_code="LIST_ERROR"
            )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["file", "delete", "io"],
)
class DeleteFileTool(QuantMindTool):
    """Delete a file from the workspace."""

    name: str = "delete_file"
    description: str = """Delete a file from the workspace.
    Requires confirmation flag to prevent accidental deletion.
    Cannot delete directories - use for files only."""

    args_schema: type[BaseModel] = DeleteFileInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        file_path: str,
        confirm: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute file delete operation."""
        if not confirm:
            raise ToolError(
                "Deletion requires confirmation. Set confirm=True to proceed.",
                tool_name=self.name,
                error_code="CONFIRMATION_REQUIRED"
            )

        # Validate path
        validated_path = self.validate_workspace_path(file_path)

        # Check file exists
        if not validated_path.exists():
            raise ToolError(
                f"File '{file_path}' does not exist",
                tool_name=self.name,
                error_code="FILE_NOT_FOUND"
            )

        if validated_path.is_dir():
            raise ToolError(
                f"Path '{file_path}' is a directory. Use delete_directory instead.",
                tool_name=self.name,
                error_code="IS_DIRECTORY"
            )

        # Delete file
        try:
            validated_path.unlink()

            return ToolResult.ok(
                data={"deleted": file_path},
                metadata={
                    "file_path": file_path,
                }
            )

        except Exception as e:
            raise ToolError(
                f"Failed to delete file '{file_path}': {e}",
                tool_name=self.name,
                error_code="DELETE_ERROR"
            )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "copy", "io"],
)
class CopyFileTool(QuantMindTool):
    """Copy a file to another location in the workspace."""

    name: str = "copy_file"
    description: str = """Copy a file to another location within the workspace.
    Creates destination directories if they don't exist.
    Can optionally overwrite existing files."""

    args_schema: type[BaseModel] = CopyFileInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute file copy operation."""
        # Validate paths
        source_path = self.validate_workspace_path(source)
        dest_path = self.validate_workspace_path(destination)

        # Check source exists
        if not source_path.exists():
            raise ToolError(
                f"Source file '{source}' does not exist",
                tool_name=self.name,
                error_code="SOURCE_NOT_FOUND"
            )

        if source_path.is_dir():
            raise ToolError(
                f"Source '{source}' is a directory. Use copy_directory instead.",
                tool_name=self.name,
                error_code="SOURCE_IS_DIRECTORY"
            )

        # Check destination
        if dest_path.exists() and not overwrite:
            raise ToolError(
                f"Destination '{destination}' already exists and overwrite is False",
                tool_name=self.name,
                error_code="DESTINATION_EXISTS"
            )

        # Create destination directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        try:
            shutil.copy2(source_path, dest_path)

            return ToolResult.ok(
                data={
                    "source": source,
                    "destination": destination,
                },
                metadata={
                    "bytes_copied": dest_path.stat().st_size,
                }
            )

        except Exception as e:
            raise ToolError(
                f"Failed to copy '{source}' to '{destination}': {e}",
                tool_name=self.name,
                error_code="COPY_ERROR"
            )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["file", "move", "rename", "io"],
)
class MoveFileTool(QuantMindTool):
    """Move or rename a file within the workspace."""

    name: str = "move_file"
    description: str = """Move or rename a file within the workspace.
    Creates destination directories if they don't exist.
    Can optionally overwrite existing files."""

    args_schema: type[BaseModel] = MoveFileInput
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute file move operation."""
        # Validate paths
        source_path = self.validate_workspace_path(source)
        dest_path = self.validate_workspace_path(destination)

        # Check source exists
        if not source_path.exists():
            raise ToolError(
                f"Source file '{source}' does not exist",
                tool_name=self.name,
                error_code="SOURCE_NOT_FOUND"
            )

        # Check destination
        if dest_path.exists() and not overwrite:
            raise ToolError(
                f"Destination '{destination}' already exists and overwrite is False",
                tool_name=self.name,
                error_code="DESTINATION_EXISTS"
            )

        # Create destination directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Move file
        try:
            shutil.move(str(source_path), str(dest_path))

            return ToolResult.ok(
                data={
                    "source": source,
                    "destination": destination,
                }
            )

        except Exception as e:
            raise ToolError(
                f"Failed to move '{source}' to '{destination}': {e}",
                tool_name=self.name,
                error_code="MOVE_ERROR"
            )


# Export all tools
__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListFilesTool",
    "DeleteFileTool",
    "CopyFileTool",
    "MoveFileTool",
]
