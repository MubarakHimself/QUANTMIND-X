# tests/api/test_chat_endpoints.py
"""
Tests for Chat Session REST API Endpoints

Tests the session CRUD endpoints:
- POST /api/chat/sessions - Create session
- GET /api/chat/sessions/{session_id} - Get session
- GET /api/chat/sessions - List sessions
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


class TestCreateSessionEndpoint:
    """Test POST /api/chat/sessions endpoint."""

    def test_create_session_endpoint(self, client):
        """Test POST /api/chat/sessions creates a session."""
        response = client.post(
            "/api/chat/sessions",
            json={
                "agent_type": "workshop",
                "agent_id": "copilot",
                "user_id": "user-1",
                "title": "Test Chat"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "workshop"
        assert data["agent_id"] == "copilot"
        assert data["user_id"] == "user-1"
        assert "id" in data

    def test_create_session_with_defaults(self, client):
        """Test POST /api/chat/sessions with minimal data."""
        response = client.post(
            "/api/chat/sessions",
            json={
                "agent_type": "workshop",
                "agent_id": "copilot",
                "user_id": "user-1"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["agent_type"] == "workshop"
        assert data["agent_id"] == "copilot"
        assert data["title"] is not None  # Should have default title


class TestGetSessionEndpoint:
    """Test GET /api/chat/sessions/{session_id} endpoint."""

    def test_get_session_endpoint(self, client):
        """Test GET /api/chat/sessions/{id} retrieves a session."""
        # First create
        create_resp = client.post(
            "/api/chat/sessions",
            json={
                "agent_type": "workshop",
                "agent_id": "copilot",
                "user_id": "user-1"
            }
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

        # Then get
        response = client.get(f"/api/chat/sessions/{session_id}")
        assert response.status_code == 200
        assert response.json()["id"] == session_id

    def test_get_session_not_found(self, client):
        """Test GET /api/chat/sessions/{id} returns 404 for non-existent session."""
        response = client.get("/api/chat/sessions/non-existent-id")
        assert response.status_code == 404


class TestListSessionsEndpoint:
    """Test GET /api/chat/sessions endpoint."""

    def test_list_sessions_endpoint(self, client):
        """Test GET /api/chat/sessions returns session list."""
        response = client.get("/api/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
