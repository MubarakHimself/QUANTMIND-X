"""
Tests for Agent Activity API Endpoints.

Validates Task 13 - Live Agent Activity Feed UI
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
import json

from src.api.agent_activity import (
    router,
    ActivityEvent,
    ActivityStats,
    add_activity_event,
    get_activity_store,
    clear_activity_store,
    _broadcast_event
)


@pytest.fixture(autouse=True)
def clear_store():
    """Clear activity store before each test."""
    clear_activity_store()
    yield
    clear_activity_store()


class TestActivityEvent:
    """Tests for ActivityEvent model."""

    def test_activity_event_creation(self):
        """Test creating an activity event."""
        event = ActivityEvent(
            id="test-123",
            agent_id="agent-001",
            agent_type="analyst",
            agent_name="Market Analyst",
            event_type="action",
            action="Analyzing market data",
            timestamp=datetime.utcnow().isoformat(),
            status="completed"
        )

        assert event.id == "test-123"
        assert event.agent_id == "agent-001"
        assert event.agent_type == "analyst"
        assert event.event_type == "action"
        assert event.status == "completed"

    def test_activity_event_with_details(self):
        """Test activity event with optional details."""
        event = ActivityEvent(
            id="test-456",
            agent_id="agent-002",
            agent_type="quantcode",
            agent_name="Code Generator",
            event_type="tool_call",
            action="Running code analysis",
            timestamp=datetime.utcnow().isoformat(),
            tool_name="code_analyzer",
            details={"lines": 150, "complexity": "medium"},
            reasoning="Checking code quality before generation"
        )

        assert event.tool_name == "code_analyzer"
        assert event.details == {"lines": 150, "complexity": "medium"}
        assert event.reasoning == "Checking code quality before generation"


class TestAddActivityEvent:
    """Tests for add_activity_event function."""

    def test_add_simple_event(self):
        """Test adding a simple activity event."""
        event = add_activity_event(
            agent_id="agent-001",
            agent_type="analyst",
            agent_name="Market Analyst",
            event_type="action",
            action="Starting analysis"
        )

        assert event.agent_id == "agent-001"
        assert event.event_type == "action"
        assert event.action == "Starting analysis"
        assert event.id is not None

    def test_add_decision_event(self):
        """Test adding a decision event."""
        event = add_activity_event(
            agent_id="agent-002",
            agent_type="copilot",
            agent_name="Trading Copilot",
            event_type="decision",
            action="Choosing trading strategy",
            reasoning="Market conditions indicate high volatility, using mean-reversion strategy"
        )

        assert event.event_type == "decision"
        assert event.reasoning is not None

    def test_add_tool_call_event(self):
        """Test adding a tool call event."""
        event = add_activity_event(
            agent_id="agent-003",
            agent_type="router",
            agent_name="Task Router",
            event_type="tool_call",
            action="Fetching market data",
            tool_name="get_market_data",
            status="running"
        )

        assert event.event_type == "tool_call"
        assert event.tool_name == "get_market_data"
        assert event.status == "running"

    def test_add_tool_result_event(self):
        """Test adding a tool result event."""
        event = add_activity_event(
            agent_id="agent-003",
            agent_type="router",
            agent_name="Task Router",
            event_type="tool_result",
            action="Market data retrieved",
            tool_name="get_market_data",
            tool_result={"price": 1.2345, "volume": 10000},
            status="completed"
        )

        assert event.event_type == "tool_result"
        assert event.tool_result == {"price": 1.2345, "volume": 10000}

    def test_store_maintains_max_size(self):
        """Test that store maintains max size of 1000 events."""
        store = get_activity_store()

        # Add more than 1000 events
        for i in range(1050):
            add_activity_event(
                agent_id=f"agent-{i}",
                agent_type="test",
                agent_name="Test Agent",
                event_type="action",
                action=f"Action {i}"
            )

        # Store should contain max 1000 events
        assert len(store) == 1000


class TestGetActivityFeed:
    """Tests for activity feed retrieval."""

    @pytest.mark.asyncio
    async def test_get_empty_feed(self):
        """Test getting activity feed when empty."""
        from fastapi.testclient import TestClient
        from src.api.agent_activity import router

        # Create mock app for testing
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_get_feed_with_events(self):
        """Test getting activity feed with events."""
        # Add some events
        for i in range(5):
            add_activity_event(
                agent_id=f"agent-{i}",
                agent_type="analyst",
                agent_name=f"Agent {i}",
                event_type="action",
                action=f"Action {i}"
            )

        from fastapi.testclient import TestClient
        from src.api.agent_activity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 5

    @pytest.mark.asyncio
    async def test_filter_by_agent_id(self):
        """Test filtering by agent ID."""
        # Add events for different agents
        add_activity_event(
            agent_id="agent-1",
            agent_type="analyst",
            agent_name="Analyst 1",
            event_type="action",
            action="Action 1"
        )
        add_activity_event(
            agent_id="agent-2",
            agent_type="copilot",
            agent_name="Copilot 1",
            event_type="action",
            action="Action 2"
        )

        from fastapi.testclient import TestClient
        from src.api.agent_activity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity?agent_id=agent-1")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["count"] == 1
        assert data["data"]["events"][0]["agent_id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        """Test filtering by event type."""
        add_activity_event(
            agent_id="agent-1",
            agent_type="analyst",
            agent_name="Analyst",
            event_type="action",
            action="Action"
        )
        add_activity_event(
            agent_id="agent-2",
            agent_type="analyst",
            agent_name="Analyst",
            event_type="decision",
            action="Decision"
        )

        from fastapi.testclient import TestClient
        from src.api.agent_activity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity?event_type=decision")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["count"] == 1
        assert data["data"]["events"][0]["event_type"] == "decision"


class TestActivityStats:
    """Tests for activity statistics."""

    def test_calculate_stats_empty(self):
        """Test calculating stats with empty store."""
        from src.api.agent_activity import router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_events"] == 0
        assert data["data"]["active_agents"] == 0

    def test_calculate_stats_with_events(self):
        """Test calculating stats with events."""
        # Add events
        add_activity_event(
            agent_id="agent-1",
            agent_type="analyst",
            agent_name="Analyst",
            event_type="action",
            action="Action 1"
        )
        add_activity_event(
            agent_id="agent-1",
            agent_type="analyst",
            agent_name="Analyst",
            event_type="decision",
            action="Decision 1"
        )
        add_activity_event(
            agent_id="agent-2",
            agent_type="copilot",
            agent_name="Copilot",
            event_type="tool_call",
            action="Tool call"
        )

        from src.api.agent_activity import router
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)
        response = client.get("/api/agents/activity/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total_events"] == 3
        assert data["data"]["active_agents"] == 2
        assert data["data"]["events_by_type"]["action"] == 1
        assert data["data"]["events_by_type"]["decision"] == 1
        assert data["data"]["events_by_type"]["tool_call"] == 1


class TestCreateActivityEvent:
    """Tests for creating activity events via API."""

    @pytest.mark.asyncio
    async def test_create_event_via_api(self):
        """Test creating event via POST endpoint."""
        from fastapi.testclient import TestClient
        from src.api.agent_activity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)

        event_data = {
            "id": "custom-id",
            "agent_id": "agent-001",
            "agent_type": "analyst",
            "agent_name": "Test Agent",
            "event_type": "action",
            "action": "Test action",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }

        response = client.post("/api/agents/activity", json=event_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "custom-id"


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        from fastapi.testclient import TestClient
        from src.api.agent_activity import router
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        # Note: WebSocket testing requires special handling
        # This is a basic test to ensure the endpoint exists
        with patch('src.api.agent_activity._activity_subscribers', []):
            # Verify the router has the websocket endpoint
            routes = [r.path for r in router.routes]
            assert any('stream' in r for r in routes)
