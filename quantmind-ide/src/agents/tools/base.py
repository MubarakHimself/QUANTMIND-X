"""
Base classes for QuantMind tools.

All tools extend LangChain's BaseTool with additional functionality:
- Security validation (path traversal, permissions)
- Execution metrics tracking
- Structured error handling
- Logging and audit trails
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, PrivateAttr, field_validator


logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Categories for tool organization."""
    FILE_OPERATIONS = "file_operations"
    DEPLOYMENT = "deployment"
    ANALYSIS = "analysis"
    NPRD_TRD = "nprd_trd"
    QUANTCODE = "quantcode"
    BROKER = "broker"
    WORKFLOW = "workflow"
    MCP = "mcp"
    KNOWLEDGE = "knowledge"
    BACKTEST = "backtest"
    DIFF = "diff"
    QUEUE = "queue"
    FILE_HISTORY = "file_history"
    STRATEGY = "strategy"


class ToolPriority(str, Enum):
    """Priority levels for tool execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    tool_name: str = ""
    tool_call_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
            "execution_time_ms": self.execution_time_ms,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
        }

    @classmethod
    def ok(cls, data: Any, metadata: Optional[Dict] = None, **kwargs) -> "ToolResult":
        """Create successful result."""
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
            **kwargs
        )

    @classmethod
    def error(cls, error: str, metadata: Optional[Dict] = None, **kwargs) -> "ToolResult":
        """Create error result."""
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
            **kwargs
        )


class ToolError(Exception):
    """Custom exception for tool errors."""
    def __init__(
        self,
        message: str,
        tool_name: str = "",
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.error_code = error_code or "TOOL_ERROR"
        self.details = details or {}


class QuantMindTool(LangChainBaseTool):
    """
    Base class for all QuantMind tools.

    Extends LangChain's BaseTool with:
    - Security validation
    - Execution metrics
    - Structured results
    - Categorized organization
    """

    # Tool metadata
    category: ToolCategory = ToolCategory.FILE_OPERATIONS
    priority: ToolPriority = ToolPriority.NORMAL
    version: str = "1.0.0"
    author: str = "QuantMind"
    tags: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)
    requirements: List[str] = Field(default_factory=list)

    # Security settings
    requires_workspace: bool = True
    allowed_extensions: Optional[List[str]] = None
    max_file_size_mb: int = 100

    # Execution settings
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Internal state
    _workspace_path: Optional[Path] = PrivateAttr(default=None)
    _execution_count: int = PrivateAttr(default=0)
    _total_execution_time_ms: float = PrivateAttr(default=0.0)
    _last_error: Optional[str] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._workspace_path = data.get("workspace_path")

    def set_workspace(self, workspace_path: Union[str, Path]) -> None:
        """Set the workspace path for path validation."""
        self._workspace_path = Path(workspace_path).resolve()

    def validate_workspace_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that a path is within the workspace boundaries.

        Raises:
            ToolError: If path is outside workspace or doesn't exist
        """
        if not self._workspace_path:
            raise ToolError(
                "Workspace not set. Call set_workspace() first.",
                tool_name=self.name,
                error_code="WORKSPACE_NOT_SET"
            )

        target_path = (self._workspace_path / path).resolve()

        # Check for path traversal
        try:
            target_path.relative_to(self._workspace_path)
        except ValueError:
            raise ToolError(
                f"Path '{path}' is outside workspace boundaries",
                tool_name=self.name,
                error_code="PATH_TRAVERSAL_DETECTED",
                details={"attempted_path": str(path), "workspace": str(self._workspace_path)}
            )

        return target_path

    def validate_file_extension(self, path: Path) -> None:
        """Validate file extension if restrictions are set."""
        if self.allowed_extensions:
            ext = path.suffix.lower()
            if ext not in [e.lower() for e in self.allowed_extensions]:
                raise ToolError(
                    f"File extension '{ext}' not allowed. Allowed: {self.allowed_extensions}",
                    tool_name=self.name,
                    error_code="INVALID_FILE_EXTENSION"
                )

    def validate_file_size(self, path: Path) -> None:
        """Validate file size is within limits."""
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.max_file_size_mb:
                raise ToolError(
                    f"File size ({size_mb:.2f}MB) exceeds limit ({self.max_file_size_mb}MB)",
                    tool_name=self.name,
                    error_code="FILE_TOO_LARGE"
                )

    def _run(self, *args, **kwargs) -> Any:
        """
        Execute the tool with metrics tracking.

        This wraps the actual execution with:
        - Timing metrics
        - Error handling
        - Logging
        - Retry logic
        """
        start_time = time.time()
        self._execution_count += 1
        tool_call_id = str(uuid.uuid4())

        logger.info(f"Tool '{self.name}' starting execution", extra={
            "tool_name": self.name,
            "tool_call_id": tool_call_id,
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if k != "run_manager"}
        })

        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                result = self.execute(*args, **kwargs)

                execution_time_ms = (time.time() - start_time) * 1000
                self._total_execution_time_ms += execution_time_ms

                # Wrap result if not already a ToolResult
                if not isinstance(result, ToolResult):
                    result = ToolResult.ok(
                        data=result,
                        execution_time_ms=execution_time_ms,
                        tool_name=self.name,
                        tool_call_id=tool_call_id
                    )
                else:
                    result.execution_time_ms = execution_time_ms
                    result.tool_name = self.name
                    result.tool_call_id = tool_call_id

                logger.info(f"Tool '{self.name}' completed successfully", extra={
                    "tool_name": self.name,
                    "tool_call_id": tool_call_id,
                    "execution_time_ms": execution_time_ms,
                    "success": result.success
                })

                return result.data if result.success else self._handle_tool_error(result.error)

            except ToolError as e:
                last_error = e
                logger.warning(f"Tool '{self.name}' error (attempt {retries + 1}): {e}")
                if retries < self.max_retries:
                    time.sleep(self.retry_delay_seconds * (retries + 1))
                    retries += 1
                else:
                    self._last_error = str(e)
                    raise

            except Exception as e:
                last_error = e
                logger.error(f"Tool '{self.name}' unexpected error: {e}", exc_info=True)
                self._last_error = str(e)
                raise ToolError(
                    f"Unexpected error in tool '{self.name}': {str(e)}",
                    tool_name=self.name,
                    error_code="UNEXPECTED_ERROR",
                    details={"exception_type": type(e).__name__}
                )

    @abstractmethod
    def execute(self, *args, **kwargs) -> Union[ToolResult, Any]:
        """
        Execute the tool's actual functionality.

        Must be implemented by subclasses.
        Should return either a ToolResult or raw data (will be wrapped).
        """
        pass

    def _handle_tool_error(self, error: str) -> None:
        """Handle tool errors by raising ToolException for LangChain."""
        raise ToolException(error)

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this tool."""
        avg_time = (
            self._total_execution_time_ms / self._execution_count
            if self._execution_count > 0
            else 0
        )
        return {
            "name": self.name,
            "category": self.category.value,
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time_ms,
            "average_execution_time_ms": avg_time,
            "last_error": self._last_error,
        }

    def to_langchain_tool(self) -> LangChainBaseTool:
        """Convert to standard LangChain tool for ToolNode integration."""
        return self


class ToolInputSchema(BaseModel):
    """Base schema for tool inputs."""
    pass


def create_tool_schema(
    name: str,
    description: str,
    fields: Dict[str, tuple]
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic schema for tool inputs.

    Args:
        name: Schema name
        description: Schema description
        fields: Dict of field_name -> (type, Field_args)

    Returns:
        A Pydantic BaseModel class
    """
    namespace = {"__annotations__": {}}

    for field_name, (field_type, field_args) in fields.items():
        namespace["__annotations__"][field_name] = field_type
        namespace[field_name] = Field(**field_args)

    namespace["__doc__"] = description

    return type(name, (BaseModel,), namespace)
