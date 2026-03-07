"""
Agent Activity API Endpoints

Provides REST and WebSocket endpoints for real-time agent activity feed.
Uses database for persistent storage with in-memory cache for fast access.

**Validates: Task 13 - Live Agent Activity Feed UI**
"""

import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel

from src.database.repositories.activity_event_repository import activity_event_repository

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/agents/activity", tags=["agent-activity"])

# In-memory cache for fast access (loaded from database)
_activity_cache: deque = deque(maxlen=1000)
_activity_subscribers: List[WebSocket] = []


# =============================================================================
# Request/Response Models
# =============================================================================

class ActivityEvent(BaseModel):
    """Model for agent activity events."""
    id: str
    agent_id: str
    agent_type: str
    agent_name: str
    event_type: str  # action, decision, tool_call, tool_result
    action: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed


class ActivityFilter(BaseModel):
    """Filter for activity events."""
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    event_type: Optional[str] = None
    from_timestamp: Optional[str] = None
    to_timestamp: Optional[str] = None
    limit: int = 50


class ActivityStats(BaseModel):
    """Statistics about agent activity."""
    total_events: int
    events_last_hour: int
    active_agents: int
    events_by_type: Dict[str, int]
    events_by_agent: Dict[str, int]


# =============================================================================
# Activity Event Management
# =============================================================================

def add_activity_event(
    agent_id: str,
    agent_type: str,
    agent_name: str,
    event_type: str,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    reasoning: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_result: Optional[Dict[str, Any]] = None,
    status: str = "pending"
) -> ActivityEvent:
    """
    Add a new activity event to the store and broadcast to subscribers.

    Args:
        agent_id: Unique identifier for the agent
        agent_type: Type of agent (analyst, quantcode, copilot, etc.)
        agent_name: Human-readable name for the agent
        event_type: Type of event (action, decision, tool_call, tool_result)
        action: Description of the action taken
        details: Additional event details
        reasoning: Agent's reasoning for the action
        tool_name: Name of tool being called (if applicable)
        tool_result: Result from tool execution (if applicable)
        status: Current status of the event

    Returns:
        The created ActivityEvent
    """
    event_id = str(uuid.uuid4())

    # Save to database
    try:
        db_event = activity_event_repository.create(
            id=event_id,
            agent_id=agent_id,
            agent_type=agent_type,
            agent_name=agent_name,
            event_type=event_type,
            action=action,
            timestamp=datetime.utcnow(),
            details=details,
            reasoning=reasoning,
            tool_name=tool_name,
            tool_result=tool_result,
            status=status
        )
    except Exception as e:
        logger.error(f"Failed to save activity event to database: {e}")
        # Continue with in-memory only if database fails

    # Create Pydantic model for response and caching
    event = ActivityEvent(
        id=event_id,
        agent_id=agent_id,
        agent_type=agent_type,
        agent_name=agent_name,
        event_type=event_type,
        action=action,
        timestamp=datetime.utcnow().isoformat(),
        details=details,
        reasoning=reasoning,
        tool_name=tool_name,
        tool_result=tool_result,
        status=status
    )

    # Add to in-memory cache
    _activity_cache.append(event)

    # Broadcast to WebSocket subscribers
    _broadcast_event(event)

    logger.debug(f"Activity event added: {agent_id} - {action}")
    return event


async def _broadcast_event(event: ActivityEvent):
    """Broadcast event to all WebSocket subscribers."""
    if not _activity_subscribers:
        return

    message = event.model_dump_json()
    disconnected = []

    for ws in _activity_subscribers:
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to broadcast to subscriber: {e}")
            disconnected.append(ws)

    # Clean up disconnected subscribers
    for ws in disconnected:
        _activity_subscribers.remove(ws)


# =============================================================================
# REST Endpoints
# =============================================================================

@router.get("", response_model=Dict[str, Any])
async def get_activity_feed(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(50, ge=1, le=200, description="Max events to return")
) -> Dict[str, Any]:
    """
    Get the activity feed with optional filters.

    Args:
        agent_id: Filter by specific agent
        agent_type: Filter by agent type
        event_type: Filter by event type
        limit: Maximum number of events to return

    Returns:
        Dictionary with activity events
    """
    # Try database first
    try:
        events = activity_event_repository.get_all(
            limit=limit,
            agent_id=agent_id,
            agent_type=agent_type,
            event_type=event_type
        )
        return {
            "success": True,
            "data": {
                "events": events,
                "count": len(events)
            }
        }
    except Exception as e:
        logger.warning(f"Database query failed, falling back to cache: {e}")
        # Fallback to in-memory cache
        events = list(_activity_cache)

        # Apply filters
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        if agent_type:
            events = [e for e in events if e.agent_type == agent_type]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent events first
        events = events[-limit:][::-1]

        return {
            "success": True,
            "data": {
                "events": [e.model_dump() for e in events],
                "count": len(events)
            }
        }


@router.get("/stats", response_model=Dict[str, Any])
async def get_activity_stats() -> Dict[str, Any]:
    """
    Get statistics about agent activity.

    Returns:
        Dictionary with activity statistics
    """
    # Try database first
    try:
        stats = activity_event_repository.get_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.warning(f"Database query failed, falling back to cache: {e}")
        # Fallback to in-memory cache
        events = list(_activity_cache)
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)

        # Count events in last hour
        events_last_hour = 0
        active_agents = set()
        events_by_type: Dict[str, int] = {}
        events_by_agent: Dict[str, int] = {}

        for event in events:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time >= one_hour_ago:
                events_last_hour += 1

            active_agents.add(event.agent_id)

            # Count by type
            events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1

            # Count by agent
            events_by_agent[event.agent_id] = events_by_agent.get(event.agent_id, 0) + 1

        stats = ActivityStats(
            total_events=len(events),
            events_last_hour=events_last_hour,
            active_agents=len(active_agents),
            events_by_type=events_by_type,
            events_by_agent=events_by_agent
        )

        return {
            "success": True,
            "data": stats.model_dump()
        }


@router.post("", response_model=Dict[str, Any])
async def create_activity_event(event: ActivityEvent) -> Dict[str, Any]:
    """
    Manually create an activity event (for testing or external triggers).

    Args:
        event: The activity event to create

    Returns:
        Dictionary with the created event
    """
    # Save to database
    try:
        activity_event_repository.create(
            id=event.id,
            agent_id=event.agent_id,
            agent_type=event.agent_type,
            agent_name=event.agent_name,
            event_type=event.event_type,
            action=event.action,
            timestamp=datetime.fromisoformat(event.timestamp) if event.timestamp else datetime.utcnow(),
            details=event.details,
            reasoning=event.reasoning,
            tool_name=event.tool_name,
            tool_result=event.tool_result,
            status=event.status
        )
    except Exception as e:
        logger.error(f"Failed to save activity event to database: {e}")

    # Add to cache
    _activity_cache.append(event)
    await _broadcast_event(event)

    return {
        "success": True,
        "data": event.model_dump()
    }


@router.get("/event/{event_id}", response_model=Dict[str, Any])
async def get_event(event_id: str) -> Dict[str, Any]:
    """
    Get a specific activity event by ID.

    Args:
        event_id: The event ID to retrieve

    Returns:
        Dictionary with the event data

    Raises:
        HTTPException: If event not found
    """
    # Try database first
    try:
        event = activity_event_repository.get_by_id(event_id)
        if event:
            return {
                "success": True,
                "data": event
            }
    except Exception as e:
        logger.warning(f"Database query failed, falling back to cache: {e}")

    # Fallback to in-memory cache
    for event in reversed(_activity_cache):
        if event.id == event_id:
            return {
                "success": True,
                "data": event.model_dump()
            }

    raise HTTPException(status_code=404, detail="Event not found")


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/stream")
async def websocket_activity_feed(websocket: WebSocket):
    """
    WebSocket endpoint for real-time activity feed.

    Client can send:
    - {"action": "subscribe", "agent_id": "optional-filter"}
    - {"action": "ping"}

    Server sends:
    - ActivityEvent JSON for each new event
    """
    await websocket.accept()
    _activity_subscribers.append(websocket)

    try:
        # Send initial events (last 10)
        initial_events = list(_activity_cache)[-10:][::-1]
        await websocket.send_text("{\"type\": \"initial_events\", \"events\": [")
        for i, event in enumerate(initial_events):
            await websocket.send_text(event.model_dump_json() + ("," if i < len(initial_events) - 1 else ""))
        await websocket.send_text("]}")

        while True:
            data = await websocket.receive_text()
            import json
            message = json.loads(data)

            if message.get("action") == "ping":
                await websocket.send_text("{\"type\": \"pong\"}")
            elif message.get("action") == "subscribe":
                # Subscription confirmation (filters handled server-side)
                agent_id = message.get("agent_id")
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "agent_id": agent_id
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected from activity feed")
    except Exception as e:
        logger.error(f"WebSocket error in activity feed: {e}")
    finally:
        if websocket in _activity_subscribers:
            _activity_subscribers.remove(websocket)


# =============================================================================
# Utility Functions
# =============================================================================

def get_activity_store() -> deque:
    """Get the activity store for testing."""
    return _activity_cache


def clear_activity_store():
    """Clear all activity events (for testing)."""
    _activity_cache.clear()


__all__ = [
    'router',
    'ActivityEvent',
    'ActivityFilter',
    'ActivityStats',
    'add_activity_event',
    'get_activity_store',
    'clear_activity_store'
]
