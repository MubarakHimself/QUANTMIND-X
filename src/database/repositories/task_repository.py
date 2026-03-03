"""
Task Repository.

Provides database operations for agent tasks.
"""

from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.database.engine import Session
from src.database.models import AgentTasks


class TaskRepository:
    """Repository for AgentTasks database operations."""

    def __init__(self):
        """Initialize the task repository."""
        pass

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
        with self.get_session() as session:
            task = AgentTasks(
                agent_type=agent_type,
                task_type=task_type,
                task_data=task_data,
                status=status
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

    def update(
        self,
        task_id: int,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[AgentTasks]:
        """Update an existing task."""
        with self.get_session() as session:
            task = session.query(AgentTasks).filter(
                AgentTasks.id == task_id
            ).first()

            if task is None:
                return None

            if status is not None:
                task.status = status
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error

            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

    def get_by_id(self, task_id: int) -> Optional[AgentTasks]:
        """Get a task by ID."""
        with self.get_session() as session:
            task = session.query(AgentTasks).filter(
                AgentTasks.id == task_id
            ).first()
            if task is not None:
                session.expunge(task)
            return task

    def get_by_status(self, status: str) -> List[AgentTasks]:
        """Get all tasks with a specific status."""
        with self.get_session() as session:
            tasks = session.query(AgentTasks).filter(
                AgentTasks.status == status
            ).order_by(AgentTasks.created_at.desc()).all()
            for task in tasks:
                session.expunge(task)
            return tasks

    def get_by_agent_type(self, agent_type: str, status: Optional[str] = None) -> List[AgentTasks]:
        """Get all tasks for a specific agent type, optionally filtered by status."""
        with self.get_session() as session:
            query = session.query(AgentTasks).filter(
                AgentTasks.agent_type == agent_type
            )
            if status is not None:
                query = query.filter(AgentTasks.status == status)

            tasks = query.order_by(AgentTasks.created_at.desc()).all()
            for task in tasks:
                session.expunge(task)
            return tasks

    def get_all(self, limit: int = 100) -> List[AgentTasks]:
        """Get all tasks, ordered by creation date."""
        with self.get_session() as session:
            tasks = session.query(AgentTasks).order_by(
                AgentTasks.created_at.desc()
            ).limit(limit).all()
            for task in tasks:
                session.expunge(task)
            return tasks

    def delete(self, task_id: int) -> bool:
        """Delete a task by ID."""
        with self.get_session() as session:
            task = session.query(AgentTasks).filter(
                AgentTasks.id == task_id
            ).first()
            if task:
                session.delete(task)
                return True
            return False
