"""
Session Checkpoint Database Model

Stores session checkpoints for resuming agent sessions.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Index
from ..models.base import Base


class SessionCheckpoint(Base):
    """
    Session checkpoint for storing agent session state at specific points.

    Allows resuming agent sessions from any saved checkpoint with full
    conversation history and variables preserved.

    Attributes:
        id: Primary key
        session_id: Reference to the agent session
        checkpoint_number: Sequential checkpoint number for this session
        checkpoint_type: Type of checkpoint (manual, auto, scheduled)
        conversation_history: JSON array of messages at checkpoint time
        variables: JSON object of session variables at checkpoint time
        progress_percent: Progress percentage at checkpoint time
        current_step: Current step description at checkpoint time
        checkpoint_metadata: JSON object with additional checkpoint metadata
        created_at: Checkpoint creation timestamp
        size_bytes: Size of checkpoint data in bytes
    """
    __tablename__ = 'session_checkpoints'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    checkpoint_number = Column(Integer, nullable=False)
    checkpoint_type = Column(String(20), server_default='auto', nullable=False)
    conversation_history = Column(JSON, nullable=False, default=list)
    variables = Column(JSON, nullable=False, default=dict)
    progress_percent = Column(Float, server_default='0.0', nullable=False)
    current_step = Column(String(255), nullable=True)
    checkpoint_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index('ix_session_checkpoints_session_num', 'session_id', 'checkpoint_number'),
        Index('ix_session_checkpoints_type', 'checkpoint_type'),
    )

    def __repr__(self):
        return f"<SessionCheckpoint(id={self.id}, session_id={self.session_id}, checkpoint={self.checkpoint_number})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "checkpoint_number": self.checkpoint_number,
            "checkpoint_type": self.checkpoint_type,
            "conversation_history": self.conversation_history,
            "variables": self.variables,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "metadata": self.checkpoint_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "size_bytes": self.size_bytes,
            "message_count": len(self.conversation_history) if self.conversation_history else 0,
        }

    def to_summary(self):
        """Convert to summary dictionary for listing."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "checkpoint_number": self.checkpoint_number,
            "checkpoint_type": self.checkpoint_type,
            "progress_percent": self.progress_percent,
            "current_step": self.current_step,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "size_bytes": self.size_bytes,
            "message_count": len(self.conversation_history) if self.conversation_history else 0,
        }
