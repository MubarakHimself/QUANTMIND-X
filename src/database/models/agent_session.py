"""
Agent Session Database Model

Stores agent conversation sessions with full state persistence.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index
from ..models.base import Base


class AgentSession(Base):
    """
    Agent session for conversation state persistence.

    Stores complete conversation history, variables, and progress
    for resuming agent sessions.

    Attributes:
        id: Primary key
        session_id: Unique session identifier (UUID or custom)
        name: Human-readable session name
        agent_type: Type of agent (claude, analyst, quant, copilot)
        status: Session status (active, completed, failed, archived)
        conversation_history: JSON array of messages
        variables: JSON object of session variables
        session_metadata: JSON object with additional metadata
        created_at: Session creation timestamp
        modified_at: Last modification timestamp
        completed_at: Session completion timestamp (nullable)
    """
    __tablename__ = 'agent_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    agent_type = Column(String(50), nullable=False, index=True)
    status = Column(String(20), server_default='active', nullable=False, index=True)
    conversation_history = Column(JSON, nullable=False, default=list)
    variables = Column(JSON, nullable=False, default=dict)
    session_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    modified_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('ix_agent_sessions_agent_status', 'agent_type', 'status'),
        Index('ix_agent_sessions_created', 'created_at'),
    )

    def __repr__(self):
        return f"<AgentSession(id={self.id}, session_id={self.session_id}, agent={self.agent_type}, status={self.status})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "conversation_history": self.conversation_history,
            "variables": self.variables,
            "metadata": self.session_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "message_count": len(self.conversation_history) if self.conversation_history else 0,
        }
