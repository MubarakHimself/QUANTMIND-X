"""
Agent models.

Contains models for agent task tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, JSON, Index
from ..models.base import Base


class AgentTasks(Base):
    """
    Agent task history for tracking agent operations and coordination.

    Attributes:
        id: Primary key
        agent_type: Type of agent (analyst/quant/copilot)
        task_type: Type of task being performed
        task_data: JSON data containing task details
        status: Task status (pending/in_progress/completed/failed)
        created_at: Task creation timestamp
        completed_at: Task completion timestamp
    """
    __tablename__ = 'agent_tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_type = Column(String(50), nullable=False, index=True)
    task_type = Column(String(100), nullable=False, index=True)
    task_data = Column(JSON, nullable=False)
    status = Column(String(20), server_default='pending', nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('ix_agent_tasks_agent_status', 'agent_type', 'status'),
    )

    def __repr__(self):
        return f"<AgentTasks(id={self.id}, agent={self.agent_type}, type={self.task_type}, status={self.status})>"
