"""
Session Checkpoint Service

Database-backed checkpoint management for agent sessions.
Provides save, restore, list, and delete operations for session checkpoints.
"""

import json
import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session as DbSession
from sqlalchemy.exc import SQLAlchemyError

from src.database.models.session_checkpoint import SessionCheckpoint
from src.database.models.agent_session import AgentSession
from src.database.models import get_db_session

logger = logging.getLogger(__name__)


class SessionCheckpointService:
    """
    Database-backed checkpoint service for agent sessions.

    Provides methods for creating, retrieving, and managing session checkpoints
    with support for manual and automatic checkpoint creation.

    Example:
        >>> service = SessionCheckpointService()
        >>> checkpoint_id = await service.create_checkpoint(
        ...     session_id="abc-123",
        ...     checkpoint_type="manual"
        ... )
        >>> checkpoint = await service.get_checkpoint(checkpoint_id)
    """

    def __init__(self, db_session: Optional[DbSession] = None):
        """
        Initialize checkpoint service.

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

    async def create_checkpoint(
        self,
        session_id: str,
        checkpoint_type: str = "auto",
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        variables: Optional[Dict[str, Any]] = None,
        progress_percent: float = 0.0,
        current_step: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new checkpoint for a session.

        Args:
            session_id: Session to checkpoint
            checkpoint_type: Type (manual, auto, scheduled)
            conversation_history: Messages to save (fetched from session if not provided)
            variables: Variables to save (fetched from session if not provided)
            progress_percent: Progress percentage at checkpoint time
            current_step: Current step description
            metadata: Additional metadata

        Returns:
            Created checkpoint ID
        """
        db = self._get_session()
        try:
            # Get conversation history and variables from session if not provided
            if conversation_history is None or variables is None:
                session = db.query(AgentSession).filter(
                    AgentSession.session_id == session_id
                ).first()

                if not session:
                    raise ValueError(f"Session not found: {session_id}")

                if conversation_history is None:
                    conversation_history = session.conversation_history or []
                if variables is None:
                    variables = session.variables or {}

            # Get next checkpoint number
            last_checkpoint = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id == session_id
            ).order_by(SessionCheckpoint.checkpoint_number.desc()).first()

            checkpoint_number = (last_checkpoint.checkpoint_number + 1) if last_checkpoint else 1

            # Calculate size
            checkpoint_data = json.dumps({
                "conversation_history": conversation_history,
                "variables": variables,
            })
            size_bytes = len(checkpoint_data.encode('utf-8'))

            # Create checkpoint
            checkpoint = SessionCheckpoint(
                session_id=session_id,
                checkpoint_number=checkpoint_number,
                checkpoint_type=checkpoint_type,
                conversation_history=conversation_history,
                variables=variables,
                progress_percent=progress_percent,
                current_step=current_step,
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                size_bytes=size_bytes,
            )

            db.add(checkpoint)
            db.commit()
            db.refresh(checkpoint)

            logger.info(f"Created checkpoint {checkpoint.checkpoint_number} for session {session_id}")
            return str(checkpoint.id)

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    async def get_checkpoint(
        self,
        checkpoint_id: int,
        include_history: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier
            include_history: Whether to include full conversation history

        Returns:
            Checkpoint data dict or None if not found
        """
        db = self._get_session()
        try:
            checkpoint = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.id == checkpoint_id
            ).first()

            if not checkpoint:
                return None

            result = checkpoint.to_dict()

            if not include_history:
                result["conversation_history"] = []
                result["variables"] = {}

            return result

        except SQLAlchemyError as e:
            logger.error(f"Failed to get checkpoint: {e}")
            raise

    async def get_latest_checkpoint(
        self,
        session_id: str,
        include_history: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for a session.

        Args:
            session_id: Session identifier
            include_history: Whether to include full conversation history

        Returns:
            Latest checkpoint data dict or None if not found
        """
        db = self._get_session()
        try:
            checkpoint = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id == session_id
            ).order_by(SessionCheckpoint.checkpoint_number.desc()).first()

            if not checkpoint:
                return None

            result = checkpoint.to_dict()

            if not include_history:
                result["conversation_history"] = []
                result["variables"] = {}

            return result

        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            raise

    async def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        checkpoint_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List checkpoints with optional filtering.

        Args:
            session_id: Filter by session ID
            checkpoint_type: Filter by checkpoint type
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of checkpoint summary dicts
        """
        db = self._get_session()
        try:
            query = db.query(SessionCheckpoint)

            if session_id:
                query = query.filter(SessionCheckpoint.session_id == session_id)
            if checkpoint_type:
                query = query.filter(SessionCheckpoint.checkpoint_type == checkpoint_type)

            checkpoints = query.order_by(
                SessionCheckpoint.created_at.desc()
            ).offset(offset).limit(limit).all()

            return [cp.to_summary() for cp in checkpoints]

        except SQLAlchemyError as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise

    async def restore_from_checkpoint(
        self,
        checkpoint_id: int,
        target_session_id: Optional[str] = None,
    ) -> str:
        """
        Restore a session from a checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore from
            target_session_id: Optional target session ID (creates new if not provided)

        Returns:
            Session ID that was restored
        """
        db = self._get_session()
        try:
            checkpoint = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.id == checkpoint_id
            ).first()

            if not checkpoint:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")

            session_id = target_session_id or checkpoint.session_id

            # Find or create session
            session = db.query(AgentSession).filter(
                AgentSession.session_id == session_id
            ).first()

            if not session:
                # Create new session with checkpoint data
                session = AgentSession(
                    session_id=session_id,
                    name=f"Restored from checkpoint {checkpoint.checkpoint_number}",
                    agent_type="restored",
                    status="active",
                    conversation_history=checkpoint.conversation_history,
                    variables=checkpoint.variables,
                    session_metadata={
                        "restored_from_checkpoint": checkpoint.id,
                        "original_session_id": checkpoint.session_id,
                    },
                    created_at=datetime.now(timezone.utc),
                    modified_at=datetime.now(timezone.utc),
                )
                db.add(session)
            else:
                # Update existing session with checkpoint data
                session.conversation_history = checkpoint.conversation_history
                session.variables = checkpoint.variables
                session.session_metadata = {
                    **session.session_metadata,
                    "restored_from_checkpoint": checkpoint.id,
                    "restored_at": datetime.now(timezone.utc).isoformat(),
                }
                session.modified_at = datetime.now(timezone.utc)
                session.status = "active"

            db.commit()

            logger.info(f"Restored session {session_id} from checkpoint {checkpoint_id}")
            return session_id

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to restore from checkpoint: {e}")
            raise

    async def delete_checkpoint(self, checkpoint_id: int) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        db = self._get_session()
        try:
            checkpoint = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.id == checkpoint_id
            ).first()

            if not checkpoint:
                return False

            db.delete(checkpoint)
            db.commit()

            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to delete checkpoint: {e}")
            raise

    async def delete_old_checkpoints(
        self,
        session_id: str,
        keep_last: int = 5,
    ) -> int:
        """
        Delete old checkpoints, keeping only the most recent ones.

        Args:
            session_id: Session ID to clean up
            keep_last: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        db = self._get_session()
        try:
            # Get all checkpoints for session ordered by number
            checkpoints = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id == session_id
            ).order_by(SessionCheckpoint.checkpoint_number.desc()).all()

            if len(checkpoints) <= keep_last:
                return 0

            # Keep the most recent ones
            to_delete = checkpoints[keep_last:]
            count = len(to_delete)

            for cp in to_delete:
                db.delete(cp)

            db.commit()

            logger.info(f"Deleted {count} old checkpoints for session {session_id}")
            return count

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to delete old checkpoints: {e}")
            raise

    async def get_checkpoint_count(
        self,
        session_id: Optional[str] = None,
    ) -> int:
        """
        Get count of checkpoints.

        Args:
            session_id: Optional filter by session ID

        Returns:
            Count of matching checkpoints
        """
        db = self._get_session()
        try:
            query = db.query(SessionCheckpoint)

            if session_id:
                query = query.filter(SessionCheckpoint.session_id == session_id)

            return query.count()

        except SQLAlchemyError as e:
            logger.error(f"Failed to count checkpoints: {e}")
            raise

    async def cleanup_orphaned_checkpoints(self) -> int:
        """
        Delete checkpoints for sessions that no longer exist.

        Returns:
            Number of orphaned checkpoints deleted
        """
        db = self._get_session()
        try:
            # Find all checkpoint session_ids
            checkpoint_sessions = db.query(SessionCheckpoint.session_id).distinct().all()
            checkpoint_session_ids = {cs[0] for cs in checkpoint_sessions}

            # Find all existing session IDs
            existing_sessions = db.query(AgentSession.session_id).all()
            existing_session_ids = {s[0] for s in existing_sessions}

            # Find orphaned checkpoints
            orphaned = checkpoint_session_ids - existing_session_ids

            if not orphaned:
                return 0

            # Delete orphaned checkpoints
            count = db.query(SessionCheckpoint).filter(
                SessionCheckpoint.session_id.in_(orphaned)
            ).delete(synchronize_session=False)

            db.commit()

            logger.info(f"Deleted {count} orphaned checkpoints")
            return count

        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to cleanup orphaned checkpoints: {e}")
            raise


# Global instance for convenience
_checkpoint_service: Optional[SessionCheckpointService] = None


def get_checkpoint_service() -> SessionCheckpointService:
    """Get the global checkpoint service instance."""
    global _checkpoint_service
    if _checkpoint_service is None:
        _checkpoint_service = SessionCheckpointService()
    return _checkpoint_service
