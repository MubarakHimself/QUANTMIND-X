"""
Activity Feed Models.

Contains database models for persistent activity feed storage.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, JSON
from ..models.base import Base


class ActivityEvent(Base):
    """
    Activity Event for persistent agent activity feed storage.

    Stores all agent activity events for audit, analysis, and real-time feed.

    Attributes:
        id: Primary key (UUID string)
        agent_id: Unique identifier for the agent
        agent_type: Type of agent (analyst, quantcode, copilot, etc.)
        agent_name: Human-readable name for the agent
        event_type: Type of event (action, decision, tool_call, tool_result)
        action: Description of the action taken
        timestamp: When the event occurred (ISO format)
        details: Additional event details (JSON)
        reasoning: Agent's reasoning for the action
        tool_name: Name of tool being called (if applicable)
        tool_result: Result from tool execution (if applicable)
        status: Current status of the event (pending, running, completed, failed)
        created_at: Record creation timestamp
    """
    __tablename__ = 'activity_events'

    id = Column(String(36), primary_key=True)  # UUID string
    agent_id = Column(String(100), nullable=False, index=True)
    agent_type = Column(String(50), nullable=False, index=True)
    agent_name = Column(String(100), nullable=False)
    event_type = Column(String(20), nullable=False, index=True)  # action, decision, tool_call, tool_result
    action = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    details = Column(JSON, nullable=True)
    reasoning = Column(Text, nullable=True)
    tool_name = Column(String(100), nullable=True)
    tool_result = Column(JSON, nullable=True)
    status = Column(String(20), nullable=False, default='pending', index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_activity_events_agent_timestamp', 'agent_id', 'timestamp'),
        Index('idx_activity_events_type_timestamp', 'event_type', 'timestamp'),
        Index('idx_activity_events_created', 'created_at'),
    )

    def __repr__(self):
        return f"<ActivityEvent(id={self.id}, agent_id={self.agent_id}, event_type={self.event_type})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "event_type": self.event_type,
            "action": self.action,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details,
            "reasoning": self.reasoning,
            "tool_name": self.tool_name,
            "tool_result": self.tool_result,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
