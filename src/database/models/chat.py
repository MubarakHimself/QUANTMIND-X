"""
Chat models.

Contains models for chat sessions and messages - stored in HOT tier (SQLite).
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Integer, Index
from sqlalchemy.orm import relationship
from ..models.base import Base


class ChatSession(Base):
    """Chat session model - stored in HOT tier (SQLite).

    Attributes:
        id: Primary key (string for SQLite)
        agent_type: Type of agent ('workshop', 'floor-manager', 'department')
        agent_id: Specific agent identifier ('copilot', 'research', etc.)
        title: Session title
        user_id: User who owns this session
        context: Session context (JSON)
        metadata: Additional metadata (JSON)
        created_at: Creation timestamp
        updated_at: Last update timestamp
        last_message_at: Timestamp of last message
    """
    __tablename__ = "chat_sessions"
    __table_args__ = (
        Index('ix_chat_sessions_user_id', 'user_id'),
        Index('ix_chat_sessions_last_message_at', 'last_message_at'),
        Index('ix_chat_sessions_user_agent', 'user_id', 'agent_type'),
    )

    id = Column(String, primary_key=True)
    agent_type = Column(String(50), nullable=False)  # 'workshop', 'floor-manager', 'department'
    agent_id = Column(String(50), nullable=False)    # 'copilot', 'research', etc.
    title = Column(String(200))
    user_id = Column(String(50))
    context = Column(JSON, default=dict)
    session_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_message_at = Column(DateTime)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        # Map 'metadata' to 'session_metadata' to avoid SQLAlchemy reserved name conflict
        if 'metadata' in kwargs:
            kwargs['session_metadata'] = kwargs.pop('metadata')
        super().__init__(**kwargs)

    def __repr__(self):
        return f"<ChatSession(id={self.id}, agent_type={self.agent_type}, title={self.title})>"


class ChatMessage(Base):
    """Chat message model - stored in HOT tier (SQLite).

    Attributes:
        id: Primary key (string for SQLite)
        session_id: Foreign key to ChatSession
        role: Message role ('user', 'assistant', 'system')
        content: Message content
        artifacts: List of artifacts (JSON)
        tool_calls: List of tool calls (JSON)
        token_count: Number of tokens in message
        created_at: Creation timestamp
    """
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    artifacts = Column(JSON, default=list)
    tool_calls = Column(JSON, default=list)
    token_count = Column(Integer)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role}, session_id={self.session_id})>"
