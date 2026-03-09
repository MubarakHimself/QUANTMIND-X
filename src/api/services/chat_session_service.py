import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from src.database.engine import Session
from src.database.models import ChatSession, ChatMessage


class ChatSessionService:
    """Service for managing chat sessions and messages."""

    def __init__(self):
        pass

    async def create_session(
        self,
        agent_type: str,
        agent_id: str,
        user_id: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        session = ChatSession(
            id=session_id,
            agent_type=agent_type,
            agent_id=agent_id,
            title=title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}",
            user_id=user_id,
            context=context or {},
            session_metadata=metadata or {},
            last_message_at=now
        )

        db = Session()
        try:
            db.add(session)
            db.commit()
            db.refresh(session)
            # Expunge to detach from session
            db.expunge(session)
            return session
        finally:
            db.close()

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID."""
        db = Session()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session:
                db.expunge(session)
            return session
        finally:
            db.close()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        artifacts: Optional[List[Dict]] = None,
        tool_calls: Optional[List[Dict]] = None,
        token_count: Optional[int] = None
    ) -> ChatMessage:
        """Add a message to a session."""
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        db = Session()
        try:
            # Validate session exists
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                raise ValueError(f"Session {session_id} not found")

            # Update session's last_message_at
            session.last_message_at = now

            message = ChatMessage(
                id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                artifacts=artifacts or [],
                tool_calls=tool_calls or [],
                token_count=token_count,
                created_at=now
            )
            db.add(message)
            db.commit()
            db.refresh(message)
            db.expunge(message)
            return message
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    async def get_messages(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """Get messages for a session."""
        db = Session()
        try:
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.asc()).limit(limit).all()
            for msg in messages:
                db.expunge(msg)
            return messages
        finally:
            db.close()

    async def list_sessions(
        self,
        user_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ChatSession]:
        """List sessions with optional filtering."""
        db = Session()
        try:
            query = db.query(ChatSession)

            if user_id:
                query = query.filter(ChatSession.user_id == user_id)
            if agent_type:
                query = query.filter(ChatSession.agent_type == agent_type)

            sessions = query.order_by(ChatSession.updated_at.desc()).limit(limit).all()

            for session in sessions:
                db.expunge(session)
            return sessions
        finally:
            db.close()

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        db = Session()
        try:
            session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
            if not session:
                return False
            db.delete(session)
            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
