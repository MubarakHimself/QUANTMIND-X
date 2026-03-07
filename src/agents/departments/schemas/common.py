# src/agents/departments/schemas/common.py
"""
Common schemas shared across all departments.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Department(str, Enum):
    """Trading floor departments."""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TRADING = "trading"
    RISK = "risk"
    PORTFOLIO = "portfolio"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DepartmentOutput(BaseModel):
    """Base output model for department operations."""
    department: Department
    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    message: str = Field(default="", description="Status message or result summary")
    data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Generic task result for department operations."""
    success: bool = Field(..., description="Whether task succeeded")
    task_id: str = Field(..., description="Task identifier")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time_ms: Optional[int] = Field(default=None, description="Execution time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "task_001",
                "result": {"output": "data"},
                "metadata": {"processed_at": "2024-01-01T00:00:00Z"}
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameter",
                "details": {"field": "symbol", "reason": "Must be non-empty"}
            }
        }


class ToolCall(BaseModel):
    """Represents a tool call made by an agent."""
    tool_name: str = Field(..., description="Name of the tool called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: Optional[Any] = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error if tool failed")
    execution_time_ms: Optional[int] = Field(default=None)


class AgentMessage(BaseModel):
    """Message passed between agents."""
    sender: str = Field(..., description="Sending agent name")
    recipient: str = Field(..., description="Receiving agent name")
    subject: str = Field(..., description="Message subject")
    body: str = Field(..., description="Message body")
    priority: str = Field(default="normal", description="Message priority")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "sender": "research_head",
                "recipient": "development_head",
                "subject": "New strategy ready",
                "body": "Strategy STRAT_001 is ready for implementation",
                "priority": "high"
            }
        }


class ValidationResult(BaseModel):
    """Result of input validation."""
    valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation errors if any")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
