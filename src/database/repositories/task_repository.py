"""
Task Repository

Provides data access methods for AgentTasks model.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import AgentTasks
from ..engine import Session as SessionFactory


class TaskRepository:
    """
    Repository for AgentTasks data access.

    Handles all database operations related to agent tasks.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self._session = session

    @property
    def session(self) -> Session:
        """Get the session (creates new if not provided)."""
        if self._session is not None:
            return self._session
        return SessionFactory()

    def create(
        self,
        agent_type: str,
        task_type: str,
        task_data: Dict[str, Any],
        status: str = 'pending'
    ) -> AgentTasks:
        """
        Create a new agent task record.

        Args:
            agent_type: Type of agent (analyst/quant/copilot)
            task_type: Type of task being performed
            task_data: Dictionary containing task details
            status: Initial task status (default: pending)

        Returns:
            Created AgentTasks object
        """
        task = AgentTasks(
            agent_type=agent_type,
            task_type=task_type,
            task_data=task_data,
            status=status
        )
        self.session.add(task)
        self.session.flush()
        self.session.refresh(task)
        self.session.expunge(task)
        return task

    def update(
        self,
        task_id: int,
        status: str,
        completed_at: Optional[datetime] = None
    ) -> Optional[AgentTasks]:
        """
        Update the status of an agent task.

        Args:
            task_id: Task ID
            status: New status (pending/in_progress/completed/failed)
            completed_at: Completion timestamp (defaults to now if status is completed)

        Returns:
            Updated AgentTasks object or None if not found
        """
        task = self.session.query(AgentTasks).filter(
            AgentTasks.id == task_id
        ).first()

        if task is None:
            return None

        task.status = status
        if status == 'completed' and completed_at is None:
            task.completed_at = datetime.utcnow()
        elif completed_at is not None:
            task.completed_at = completed_at

        self.session.flush()
        self.session.refresh(task)
        self.session.expunge(task)
        return task

    def get_all(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentTasks]:
        """
        Retrieve agent tasks with optional filtering.

        Args:
            agent_type: Filter by agent type (optional)
            status: Filter by status (optional)
            limit: Maximum number of tasks to return

        Returns:
            List of AgentTasks objects
        """
        query = self.session.query(AgentTasks)

        if agent_type is not None:
            query = query.filter(AgentTasks.agent_type == agent_type)

        if status is not None:
            query = query.filter(AgentTasks.status == status)

        query = query.order_by(AgentTasks.created_at.desc()).limit(limit)

        tasks = query.all()
        # Expunge all tasks to avoid detached instance errors
        for task in tasks:
            self.session.expunge(task)
        return tasks
