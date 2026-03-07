"""
Agent Session Persistence Service

Database-backed session storage with support for:
- Save/restore conversation history
- Session metadata management
- Export/import functionality
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session as DbSession
from sqlalchemy.exc import SQLAlchemyError

from src.database.models.agent_session import AgentSession
from src.database.models import get_db_session

logger = logging.getLogger(__name__)


class AgentSessionPersistence:
    """
    Database-backed session persistence service.

    Provides methods for creating, retrieving, updating, and deleting
    agent sessions with full conversation history.

    Example:
        >>> persistence = AgentSessionPersistence()
        >>> session_id = await persistence.create_session(
        ...     name="Research Session",
        ...     agent_type="analyst"
        ... )
        >>> await persistence.add_message(session_id, "user", "Analyze BTC")
        >>> session = await persistence.get_session(session_id)
    """

    def __init__(self, db_session: Optional[DbSession] = None):
        """
        Initialize persistence service.

        Args:
            db_session: Optional SQLAlchemy session. If not provided,
                       creates a new one for each operation.
        """
        self._db_session = db_session

    def _get_session(self) -> DbSession:
        """Get database session."""
        if self._db_session:
            return self._db_session
        return get_db_session()

    async def create_session(
        self,
        name: str,
        agent_type: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new agent session.

        Args:
            name: Human-readable session name
            agent_type: Type of agent (analyst, quant, copilot, claude)
            session_id: Optional custom session ID (generates UUID if not provided)
            metadata: Optional session metadata
            variables: Optional session variables

        Returns:
            Created session ID
        """
        db = self._get_session()
        try:
            sid = session_id or str(uuid4())
            now = datetime.now(timezone.utc)

            session = AgentSession(
                session_id=sid,
                name=name,
                agent_type=agent_type,
                status="active",
                conversation_history=[],
                variables=variables or {},
                session_metadata=metadata or {},
                created_at=now,
                modified_at=now,
            )

            db.add(session)
            db.commit()
            db.refresh(session)

            logger.info(f"Created agent session: {sid} ({name})")
            return sid

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to create session: {e}")
            raise

    async def get_session(
        self,
        session_id: str,
        include_history: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session identifier
            include_history: Whether to include full conversation history

        Returns:
            Session data dict or None if not found
        """
        db = self._get_session()
        try:
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                return None

            result = session.to_dict()

            if not include_history:
                # Return summary without full history
                result["conversation_history"] = []

            return result

        except SQLAlchemyError as e:
            logger.error(f"Failed to get session: {e}")
            raise

    async def update_session(
        self,
        session_id: str,
        name: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update session metadata.

        Args:
            session_id: Session identifier
            name: New name (optional)
            status: New status (optional)
            metadata: Updated metadata (optional)
            variables: Updated variables (optional)

        Returns:
            True if updated, False if session not found
        """
        db = self._get_session()
        try:
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                return False

            now = datetime.now(timezone.utc)
            session.modified_at = now

            if name is not None:
                session.name = name
            if status is not None:
                session.status = status
                if status in ("completed", "failed", "archived"):
                    session.completed_at = now
            if metadata is not None:
                session.session_metadata = metadata
            if variables is not None:
                session.variables = variables

            db.commit()
            logger.info(f"Updated session: {session_id}")
            return True

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to update session: {e}")
            raise

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        db = self._get_session()
        try:
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                return False

            db.delete(session)
            db.commit()

            logger.info(f"Deleted session: {session_id}")
            return True

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to delete session: {e}")
            raise

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a message to the session conversation.

        Args:
            session_id: Session identifier
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Optional message metadata

        Returns:
            True if added, False if session not found
        """
        db = self._get_session()
        try:
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                return False

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            history = session.conversation_history or []
            history.append(message)
            session.conversation_history = history
            session.modified_at = datetime.now(timezone.utc)

            db.commit()
            return True

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to add message: {e}")
            raise

    async def set_conversation_history(
        self,
        session_id: str,
        history: List[Dict[str, Any]],
    ) -> bool:
        """
        Replace entire conversation history.

        Args:
            session_id: Session identifier
            history: List of message dicts

        Returns:
            True if updated, False if session not found
        """
        db = self._get_session()
        try:
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                return False

            session.conversation_history = history
            session.modified_at = datetime.now(timezone.utc)

            db.commit()
            return True

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to set conversation history: {e}")
            raise

    async def list_sessions(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.

        Args:
            agent_type: Filter by agent type
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of session summary dicts
        """
        db = self._get_session()
        try:
            query = db.query(AgentSession)

            if agent_type:
                query = query.filter(AgentSession.agent_type == agent_type)
            if status:
                query = query.filter(AgentSession.status == status)

            sessions = query.order_by(
                AgentSession.modified_at.desc()
            ).offset(offset).limit(limit).all()

            return [
                {
                    "session_id": s.session_id,
                    "name": s.name,
                    "agent_type": s.agent_type,
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "modified_at": s.modified_at.isoformat() if s.modified_at else None,
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "message_count": len(s.conversation_history) if s.conversation_history else 0,
                }
                for s in sessions
            ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to list sessions: {e}")
            raise

    async def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Export session data for backup/sharing.

        Args:
            session_id: Session identifier

        Returns:
            Complete session data dict or None if not found
        """
        session = await self.get_session(session_id, include_history=True)
        if not session:
            return None

        # Add export metadata
        session["exported_at"] = datetime.now(timezone.utc).isoformat()
        session["export_version"] = "1.0"

        return session

    async def import_session(
        self,
        session_data: Dict[str, Any],
        new_session_id: Optional[str] = None,
    ) -> str:
        """
        Import session data from backup.

        Args:
            session_data: Exported session data
            new_session_id: Optional new session ID (uses existing if not provided)

        Returns:
            New session ID
        """
        db = self._get_session()
        try:
            sid = new_session_id or str(uuid4())
            now = datetime.now(timezone.utc)

            session = AgentSession(
                session_id=sid,
                name=session_data.get("name", "Imported Session"),
                agent_type=session_data.get("agent_type", "unknown"),
                status="archived",  # Imported sessions are archived by default
                conversation_history=session_data.get("conversation_history", []),
                variables=session_data.get("variables", {}),
                session_metadata={
                    **session_data.get("metadata", {}),
                    "imported_from": session_data.get("session_id"),
                },
                created_at=now,
                modified_at=now,
            )

            db.add(session)
            db.commit()
            db.refresh(session)

            logger.info(f"Imported session as: {sid}")
            return sid

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to import session: {e}")
            raise

    async def get_session_count(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        """
        Get count of sessions matching filters.

        Args:
            agent_type: Filter by agent type
            status: Filter by status

        Returns:
            Count of matching sessions
        """
        db = self._get_session()
        try:
            query = db.query(AgentSession)

            if agent_type:
                query = query.filter(AgentSession.agent_type == agent_type)
            if status:
                query = query.filter(AgentSession.status == status)

            return query.count()

        except SQLAlchemyError as e:
            logger.error(f"Failed to count sessions: {e}")
            raise


# Global instance for convenience
_persistence: Optional[AgentSessionPersistence] = None


def get_session_persistence() -> AgentSessionPersistence:
    """Get the global session persistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = AgentSessionPersistence()
    return _persistence
