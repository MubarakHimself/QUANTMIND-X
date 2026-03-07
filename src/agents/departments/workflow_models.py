"""
Workflow Data Models

Contains all dataclasses and enums for the department workflow system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class WorkflowStage(str, Enum):
    """Department workflow stages."""
    VIDEO_INGEST = "video_ingest"
    RESEARCH = "research"  # TRD generation
    DEVELOPMENT = "development"  # EA implementation
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStatus(str, Enum):
    """Status of a workflow."""
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"  # Waiting for department response
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    """A single task in the workflow."""
    task_id: str
    stage: WorkflowStage
    from_dept: str
    to_dept: str
    message_id: str
    status: WorkflowStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DepartmentWorkflow:
    """Represents a complete department workflow."""
    workflow_id: str
    status: WorkflowStatus
    current_stage: WorkflowStage
    created_at: datetime
    updated_at: datetime
    tasks: List[WorkflowTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "current_stage": self.current_stage.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "stage": t.stage.value,
                    "from_dept": t.from_dept,
                    "to_dept": t.to_dept,
                    "message_id": t.message_id,
                    "status": t.status.value,
                    "created_at": t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                    "payload": t.payload,
                    "result": t.result,
                    "error": t.error,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
            "error": self.error,
        }

    def get_progress(self) -> float:
        """Calculate progress percentage."""
        stage_order = [
            WorkflowStage.VIDEO_INGEST,
            WorkflowStage.RESEARCH,
            WorkflowStage.DEVELOPMENT,
            WorkflowStage.BACKTESTING,
            WorkflowStage.PAPER_TRADING,
            WorkflowStage.COMPLETED,
        ]

        try:
            current_index = stage_order.index(self.current_stage)
            return (current_index / (len(stage_order) - 1)) * 100
        except ValueError:
            return 0.0
