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

    async def build_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Build context for a chat session from history and memory.

        HOT: Recent messages from SQLite
        WARM: Search memory facade for relevant context (if available)
        """
        # Get recent messages (HOT tier)
        messages = await self.get_messages(session_id, limit=20)

        # Format history
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Get session for context
        session = await self.get_session(session_id)

        # Try to get memory context (WARM tier via UnifiedMemoryFacade)
        memory_context = []
        try:
            from src.agents.memory.unified_memory_facade import get_memory_facade
            memory = get_memory_facade()

            if history:
                last_message = history[-1]["content"]
                # Search memory for relevant context
                results = await memory.search(last_message, limit=3)
                memory_context = results
        except Exception:
            pass  # Memory not available

        return {
            "session_id": session_id,
            "agent_type": session.agent_type if session else None,
            "agent_id": session.agent_id if session else None,
            "context": session.context if session else {},
            "history": history,
            "memory": memory_context
        }

    async def archive_old_sessions(self, days: int = 30) -> int:
        """
        Archive sessions older than specified days to cold storage.

        Moves session data to cold_storage.db for long-term retention.
        Returns count of archived sessions.
        """
        from datetime import timedelta
        import json
        from pathlib import Path
        import sqlite3

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        db = Session()
        try:
            old_sessions = db.query(ChatSession).filter(
                ChatSession.last_message_at < cutoff
            ).all()

            archived_count = 0
            for session in old_sessions:
                # Get all messages for this session
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session.id
                ).all()

                # Archive to cold storage
                self._archive_to_cold_storage(session, messages)
                archived_count += 1

                # Delete from hot storage
                db.delete(session)

            db.commit()
            return archived_count

        finally:
            db.close()

    def _archive_to_cold_storage(self, session: ChatSession, messages: List[ChatMessage]):
        """Archive session to cold storage (SQLite)."""
        import json
        from pathlib import Path
        import sqlite3

        # Use existing cold storage
        cold_db_path = Path("data/departments/cold_storage.db")
        cold_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(cold_db_path))
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_chat_sessions (
                id TEXT PRIMARY KEY,
                agent_type TEXT,
                agent_id TEXT,
                title TEXT,
                user_id TEXT,
                context TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS archived_chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                artifacts TEXT,
                tool_calls TEXT,
                created_at TEXT,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert session
        cursor.execute("""
            INSERT OR REPLACE INTO archived_chat_sessions
            (id, agent_type, agent_id, title, user_id, context, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session.id,
            session.agent_type,
            session.agent_id,
            session.title,
            session.user_id,
            json.dumps(session.context) if session.context else '{}',
            json.dumps(session.session_metadata) if session.session_metadata else '{}',
            session.created_at.isoformat() if session.created_at else '',
            session.last_message_at.isoformat() if session.last_message_at else ''
        ))

        # Insert messages
        for msg in messages:
            cursor.execute("""
                INSERT OR REPLACE INTO archived_chat_messages
                (id, session_id, role, content, artifacts, tool_calls, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                msg.id,
                msg.session_id,
                msg.role,
                msg.content,
                json.dumps(msg.artifacts) if msg.artifacts else '[]',
                json.dumps(msg.tool_calls) if msg.tool_calls else '[]',
                msg.created_at.isoformat() if msg.created_at else ''
            ))

        conn.commit()
        conn.close()
