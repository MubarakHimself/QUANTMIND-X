"""
Workflow State Management.

Defines state schema and management for multi-agent workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Individual step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentType(str, Enum):
    """Types of agents in workflows."""
    COPILOT = "copilot"
    ANALYST = "analyst"
    QUANTCODE = "quantcode"


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    name: str
    description: str
    agent_type: AgentType
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

    def start(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, output: Dict[str, Any]) -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output_data = output

    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def skip(self, reason: str = "") -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.now()
        self.error = reason

    def can_retry(self) -> bool:
        """Check if step can be retried."""
        return self.retries < self.max_retries

    def retry(self) -> None:
        """Reset step for retry."""
        self.retries += 1
        self.status = StepStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get step duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "retries": self.retries,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class WorkflowState:
    """
    Complete state for a workflow execution.

    Tracks:
    - Current step and progress
    - Intermediate results (NPRD, TRD, MQL5 code)
    - Metadata and timing
    """
    # Identity
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: str = "nprd_to_ea"

    # Status
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step_index: int = 0

    # Steps
    steps: List[WorkflowStep] = field(default_factory=list)

    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_result: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    error: Optional[str] = None

    # Configuration
    auto_continue: bool = True
    max_retries_per_step: int = 3

    @property
    def current_step(self) -> Optional[WorkflowStep]:
        """Get current step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return (completed / len(self.steps)) * 100

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get total workflow duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        if self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)

    def start(self) -> None:
        """Start the workflow."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        if self.steps:
            self.steps[0].start()

    def advance(self) -> bool:
        """
        Advance to next step.

        Returns:
            True if advanced, False if at end
        """
        if self.current_step_index >= len(self.steps) - 1:
            return False

        self.current_step_index += 1
        if self.current_step:
            self.current_step.start()
        return True

    def complete(self, result: Dict[str, Any] = None) -> None:
        """Mark workflow as completed."""
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        if result:
            self.final_result = result

    def fail(self, error: str) -> None:
        """Mark workflow as failed."""
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def pause(self) -> None:
        """Pause the workflow."""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.PAUSED

    def resume(self) -> None:
        """Resume a paused workflow."""
        if self.status == WorkflowStatus.PAUSED:
            self.status = WorkflowStatus.RUNNING

    def cancel(self) -> None:
        """Cancel the workflow."""
        self.status = WorkflowStatus.CANCELLED
        self.completed_at = datetime.now()

    def get_step_by_id(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_completed_steps(self) -> List[WorkflowStep]:
        """Get all completed steps."""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]

    def get_failed_steps(self) -> List[WorkflowStep]:
        """Get all failed steps."""
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def set_intermediate(self, key: str, value: Any) -> None:
        """Store intermediate result."""
        self.intermediate_results[key] = value

    def get_intermediate(self, key: str, default: Any = None) -> Any:
        """Get intermediate result."""
        return self.intermediate_results.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "status": self.status.value,
            "current_step_index": self.current_step_index,
            "progress_percent": self.progress_percent,
            "steps": [s.to_dict() for s in self.steps],
            "input_data": self.input_data,
            "intermediate_results": self.intermediate_results,
            "final_result": self.final_result,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


@dataclass
class WorkflowResult:
    """Final result of a workflow execution."""
    workflow_id: str
    success: bool
    status: WorkflowStatus
    final_output: Dict[str, Any]
    intermediate_outputs: Dict[str, Any]
    duration_seconds: float
    steps_completed: int
    steps_total: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "status": self.status.value,
            "final_output": self.final_output,
            "intermediate_outputs": self.intermediate_outputs,
            "duration_seconds": self.duration_seconds,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "error": self.error,
        }


class WorkflowStateStore:
    """
    In-memory store for workflow states.

    In production, this would use a database.
    """

    def __init__(self):
        self._states: Dict[str, WorkflowState] = {}

    def save(self, state: WorkflowState) -> None:
        """Save workflow state."""
        self._states[state.workflow_id] = state

    def get(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow state."""
        return self._states.get(workflow_id)

    def delete(self, workflow_id: str) -> bool:
        """Delete workflow state."""
        if workflow_id in self._states:
            del self._states[workflow_id]
            return True
        return False

    def list_all(self) -> List[WorkflowState]:
        """List all workflow states."""
        return list(self._states.values())

    def list_by_status(self, status: WorkflowStatus) -> List[WorkflowState]:
        """List workflows by status."""
        return [s for s in self._states.values() if s.status == status]


# Global store
_workflow_store: Optional[WorkflowStateStore] = None


def get_workflow_store() -> WorkflowStateStore:
    """Get the global workflow store."""
    global _workflow_store
    if _workflow_store is None:
        _workflow_store = WorkflowStateStore()
    return _workflow_store
