"""
Activity Event Repository.

Provides database operations for activity events.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from src.database.engine import Session as BaseSession
from src.database.models.base import Base
from src.database.models.activity import ActivityEvent


class ActivityEventRepository:
    """
    Repository for ActivityEvent database operations.

    Provides CRUD operations and specialized queries for activity events.
    """

    def __init__(self):
        """Initialize the repository."""
        pass

    def _get_session(self) -> Session:
        """Get a new database session."""
        return BaseSession()

    def create(
        self,
        id: str,
        agent_id: str,
        agent_type: str,
        agent_name: str,
        event_type: str,
        action: str,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[Dict[str, Any]] = None,
        status: str = 'pending'
    ) -> ActivityEvent:
        """
        Create a new activity event.

        Args:
            id: Unique event ID (UUID)
            agent_id: Agent identifier
            agent_type: Type of agent
            agent_name: Agent name
            event_type: Type of event
            action: Action description
            timestamp: Event timestamp
            details: Additional details
            reasoning: Agent reasoning
            tool_name: Tool name
            tool_result: Tool result
            status: Event status

        Returns:
            Created ActivityEvent instance
        """
        session = self._get_session()
        try:
            event = ActivityEvent(
                id=id,
                agent_id=agent_id,
                agent_type=agent_type,
                agent_name=agent_name,
                event_type=event_type,
                action=action,
                timestamp=timestamp or datetime.now(timezone.utc),
                details=details,
                reasoning=reasoning,
                tool_name=tool_name,
                tool_result=tool_result,
                status=status,
                created_at=datetime.now(timezone.utc)
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            result = event.to_dict()
            return result
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an activity event by ID.

        Args:
            event_id: Event ID

        Returns:
            Event dict or None
        """
        session = self._get_session()
        try:
            event = session.query(ActivityEvent).filter(ActivityEvent.id == event_id).first()
            if event:
                session.expunge(event)
                return event.to_dict()
            return None
        finally:
            session.close()

    def get_all(
        self,
        limit: int = 100,
        agent_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get activity events with optional filters.

        Args:
            limit: Maximum events to return
            agent_id: Filter by agent ID
            agent_type: Filter by agent type
            event_type: Filter by event type

        Returns:
            List of event dicts
        """
        session = self._get_session()
        try:
            query = session.query(ActivityEvent)

            if agent_id:
                query = query.filter(ActivityEvent.agent_id == agent_id)
            if agent_type:
                query = query.filter(ActivityEvent.agent_type == agent_type)
            if event_type:
                query = query.filter(ActivityEvent.event_type == event_type)

            events = query.order_by(desc(ActivityEvent.timestamp)).limit(limit).all()
            for event in events:
                session.expunge(event)
            return [e.to_dict() for e in events]
        finally:
            session.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get activity statistics.

        Returns:
            Statistics dictionary
        """
        session = self._get_session()
        try:
            now = datetime.now(timezone.utc)
            one_hour_ago = now - timedelta(hours=1)

            # Total events
            total_events = session.query(func.count(ActivityEvent.id)).scalar() or 0

            # Events in last hour
            events_last_hour = session.query(func.count(ActivityEvent.id)).filter(
                ActivityEvent.timestamp >= one_hour_ago
            ).scalar() or 0

            # Active agents
            active_agents = session.query(
                func.count(func.distinct(ActivityEvent.agent_id))
            ).scalar() or 0

            # Events by type
            events_by_type = {}
            type_results = session.query(
                ActivityEvent.event_type,
                func.count(ActivityEvent.id)
            ).group_by(ActivityEvent.event_type).all()
            for event_type, count in type_results:
                events_by_type[event_type] = count

            # Events by agent
            events_by_agent = {}
            agent_results = session.query(
                ActivityEvent.agent_id,
                func.count(ActivityEvent.id)
            ).group_by(ActivityEvent.agent_id).all()
            for agent_id, count in agent_results:
                events_by_agent[agent_id] = count

            return {
                "total_events": total_events,
                "events_last_hour": events_last_hour,
                "active_agents": active_agents,
                "events_by_type": events_by_type,
                "events_by_agent": events_by_agent
            }
        finally:
            session.close()

    def delete_old_events(self, days: int = 30) -> int:
        """
        Delete events older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of deleted events
        """
        session = self._get_session()
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            deleted = session.query(ActivityEvent).filter(
                ActivityEvent.timestamp < cutoff
            ).delete()
            session.commit()
            return deleted
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_recent_events(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get most recent events (for maintaining in-memory cache).

        Args:
            limit: Maximum events to return

        Returns:
            List of event dicts
        """
        session = self._get_session()
        try:
            events = session.query(ActivityEvent).order_by(
                desc(ActivityEvent.timestamp)
            ).limit(limit).all()
            for event in events:
                session.expunge(event)
            return [e.to_dict() for e in events]
        finally:
            session.close()


# Singleton instance
activity_event_repository = ActivityEventRepository()
