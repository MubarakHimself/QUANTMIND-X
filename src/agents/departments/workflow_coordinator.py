"""
Department Workflow Coordinator

Orchestrates the flow between Trading Floor departments using the department mail system.

Workflow stages:
- Video Ingest → Research (TRD generation)
- Research → Development (EA implementation)
- Development → Backtesting
- Backtesting → Paper Trading

Each stage uses the department mail system for communication.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from src.agents.departments.department_mail import (
    DepartmentMailService,
    DepartmentMessage,
    MessageType,
    Priority,
)
from src.agents.departments.types import Department

logger = logging.getLogger(__name__)


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


class DepartmentWorkflowCoordinator:
    """
    Orchestrates workflow between Trading Floor departments.

    Manages the complete pipeline:
    - Video Ingest → Research (TRD generation)
    - Research → Development (EA implementation)
    - Development → Backtesting
    - Backtesting → Paper Trading

    Uses department mail system for all cross-department communication.

    Usage:
        coordinator = DepartmentWorkflowCoordinator()

        # Start a new workflow
        workflow_id = coordinator.start_workflow(
            source="video_ingest",
            initial_payload={"video_path": "/path/to/video.mp4"}
        )

        # Process current stage
        coordinator.process_stage(workflow_id)

        # Check status
        status = coordinator.get_workflow_status(workflow_id)
    """

    # Stage to department mapping
    STAGE_DEPARTMENT_MAP = {
        WorkflowStage.VIDEO_INGEST: "video_ingest",
        WorkflowStage.RESEARCH: "research",
        WorkflowStage.DEVELOPMENT: "development",
        WorkflowStage.BACKTESTING: "backtesting",
        WorkflowStage.PAPER_TRADING: "trading",
    }

    # Next stage in workflow
    NEXT_STAGE = {
        WorkflowStage.VIDEO_INGEST: WorkflowStage.RESEARCH,
        WorkflowStage.RESEARCH: WorkflowStage.DEVELOPMENT,
        WorkflowStage.DEVELOPMENT: WorkflowStage.BACKTESTING,
        WorkflowStage.BACKTESTING: WorkflowStage.PAPER_TRADING,
        WorkflowStage.PAPER_TRADING: WorkflowStage.COMPLETED,
    }

    def __init__(
        self,
        mail_db_path: str = ".quantmind/department_mail.db",
        on_progress: Optional[Callable[[str, float, WorkflowStage], None]] = None,
    ):
        """
        Initialize the workflow coordinator.

        Args:
            mail_db_path: Path to department mail database
            on_progress: Optional progress callback
        """
        self.mail_service = DepartmentMailService(db_path=mail_db_path)
        self.on_progress = on_progress
        self._workflows: Dict[str, DepartmentWorkflow] = {}

        logger.info("DepartmentWorkflowCoordinator initialized")

    def start_workflow(
        self,
        source: str,
        initial_payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new department workflow.

        Args:
            source: Source of the workflow (e.g., "video_ingest")
            initial_payload: Initial data for the workflow
            metadata: Optional metadata

        Returns:
            Workflow ID
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        workflow = DepartmentWorkflow(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            current_stage=WorkflowStage.VIDEO_INGEST,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {"source": source, "initial_payload": initial_payload},
        )

        self._workflows[workflow_id] = workflow

        logger.info(f"Started workflow {workflow_id} from {source}")

        # Create initial task for video ingest
        self._create_task(
            workflow=workflow,
            stage=WorkflowStage.VIDEO_INGEST,
            from_dept="floor_manager",
            to_dept="video_ingest",
            payload=initial_payload,
        )

        return workflow_id

    def _create_task(
        self,
        workflow: DepartmentWorkflow,
        stage: WorkflowStage,
        from_dept: str,
        to_dept: str,
        payload: Dict[str, Any],
    ) -> WorkflowTask:
        """Create a new task in the workflow."""
        task_id = f"task_{len(workflow.tasks) + 1}"

        # Determine message type based on stage
        if stage == WorkflowStage.RESEARCH:
            msg_type = MessageType.STRATEGY_DISPATCH
        else:
            msg_type = MessageType.DISPATCH

        # Send message via mail service
        message = self.mail_service.send(
            from_dept=from_dept,
            to_dept=to_dept,
            type=msg_type,
            subject=f"Workflow {workflow.workflow_id}: {stage.value}",
            body=json.dumps({
                "workflow_id": workflow.workflow_id,
                "stage": stage.value,
                "payload": payload,
                "previous_results": workflow.metadata.get("latest_results", {}),
            }),
            priority=Priority.HIGH,
        )

        task = WorkflowTask(
            task_id=task_id,
            stage=stage,
            from_dept=from_dept,
            to_dept=to_dept,
            message_id=message.id,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(),
            payload=payload,
        )

        workflow.tasks.append(task)

        logger.info(f"Created task {task_id} for workflow {workflow.workflow_id}: {stage.value} -> {to_dept}")

        return task

    def process_stage(self, workflow_id: str) -> Dict[str, Any]:
        """
        Process the current stage of a workflow.

        Args:
            workflow_id: Workflow to process

        Returns:
            Processing result
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        if workflow.status == WorkflowStatus.COMPLETED:
            return {"error": "Workflow already completed"}

        if workflow.status == WorkflowStatus.FAILED:
            return {"error": "Workflow failed"}

        # Update status to running
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.now()

        current_stage = workflow.current_stage

        # Get the target department for current stage
        to_dept = self.STAGE_DEPARTMENT_MAP.get(current_stage)
        if not to_dept:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = f"No department mapping for stage: {current_stage}"
            return {"error": workflow.error}

        # Get previous results to pass along
        previous_results = {}
        if workflow.tasks:
            last_task = workflow.tasks[-1]
            if last_task.result:
                previous_results = last_task.result

        # Build payload for current stage
        payload = {
            **workflow.metadata.get("initial_payload", {}),
            "workflow_id": workflow_id,
            "stage": current_stage.value,
        }

        if previous_results:
            payload["from_previous_stage"] = previous_results

        # Create task for current stage
        from_dept = "floor_manager"
        if workflow.tasks:
            from_dept = workflow.tasks[-1].to_dept

        task = self._create_task(
            workflow=workflow,
            stage=current_stage,
            from_dept=from_dept,
            to_dept=to_dept,
            payload=payload,
        )

        # Mark task as running
        task.status = WorkflowStatus.RUNNING

        # Move to next stage
        next_stage = self.NEXT_STAGE.get(current_stage)
        if next_stage:
            workflow.current_stage = next_stage
            workflow.status = WorkflowStatus.WAITING
        else:
            workflow.status = WorkflowStatus.COMPLETED

        workflow.updated_at = datetime.now()

        # Notify progress
        if self.on_progress:
            self.on_progress(workflow_id, workflow.get_progress(), current_stage)

        return {
            "workflow_id": workflow_id,
            "task_id": task.task_id,
            "message_id": task.message_id,
            "current_stage": current_stage.value,
            "next_stage": next_stage.value if next_stage else None,
            "status": workflow.status.value,
        }

    def handle_department_response(
        self,
        workflow_id: str,
        from_dept: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle a response from a department.

        Called when a department completes its work and sends results back.

        Args:
            workflow_id: Workflow ID
            from_dept: Department sending the response
            result: Result data from the department

        Returns:
            Handling result
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow {workflow_id} not found"}

        # Find the pending task for this department
        pending_task = None
        for task in reversed(workflow.tasks):
            if task.to_dept == from_dept and task.status == WorkflowStatus.RUNNING:
                pending_task = task
                break

        if not pending_task:
            return {"error": f"No pending task for department {from_dept}"}

        # Update task with result
        pending_task.status = WorkflowStatus.COMPLETED
        pending_task.completed_at = datetime.now()
        pending_task.result = result

        # Store latest results in workflow metadata
        workflow.metadata["latest_results"] = result

        # If the department returned a TRD, route it
        if "trd_content" in result:
            logger.info(f"TRD received from {from_dept}, will route to next stage")

        workflow.updated_at = datetime.now()

        return {
            "workflow_id": workflow_id,
            "task_id": pending_task.task_id,
            "status": "completed",
            "result": result,
        }

    def check_department_inbox(
        self,
        workflow_id: str,
        department: str,
    ) -> List[Dict[str, Any]]:
        """
        Check inbox for department responses.

        Args:
            workflow_id: Workflow ID
            department: Department to check

        Returns:
            List of messages
        """
        messages = self.mail_service.check_inbox(
            dept=department,
            unread_only=True,
            limit=50,
        )

        workflow_messages = [
            msg for msg in messages
            if json.loads(msg.body).get("workflow_id") == workflow_id
        ]

        return [msg.to_dict() for msg in workflow_messages]

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow status dictionary or None if not found
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None

        return workflow.to_dict()

    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """
        Get all workflows.

        Returns:
            List of workflow status dictionaries
        """
        return [w.to_dict() for w in self._workflows.values()]

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a running workflow.

        Args:
            workflow_id: Workflow to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return False

        if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.WAITING]:
            return False

        workflow.status = WorkflowStatus.CANCELLED
        workflow.updated_at = datetime.now()

        # Mark running task as cancelled
        for task in workflow.tasks:
            if task.status == WorkflowStatus.RUNNING:
                task.status = WorkflowStatus.CANCELLED
                task.completed_at = datetime.now()

        logger.info(f"Workflow {workflow_id} cancelled")

        return True

    def close(self):
        """Clean up resources."""
        self.mail_service.close()
        logger.info("DepartmentWorkflowCoordinator closed")


# =============================================================================
# Singleton Instance
# =============================================================================

_coordinator: Optional[DepartmentWorkflowCoordinator] = None


def get_workflow_coordinator() -> DepartmentWorkflowCoordinator:
    """Get or create the global workflow coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = DepartmentWorkflowCoordinator()
    return _coordinator


def create_workflow_coordinator(
    mail_db_path: str = ".quantmind/department_mail.db",
    on_progress: Optional[Callable[[str, float, WorkflowStage], None]] = None,
) -> DepartmentWorkflowCoordinator:
    """
    Create a workflow coordinator instance.

    Args:
        mail_db_path: Path to department mail database
        on_progress: Optional progress callback

    Returns:
        Configured DepartmentWorkflowCoordinator instance
    """
    return DepartmentWorkflowCoordinator(
        mail_db_path=mail_db_path,
        on_progress=on_progress,
    )
