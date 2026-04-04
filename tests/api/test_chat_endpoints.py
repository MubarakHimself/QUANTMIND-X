# tests/api/test_chat_endpoints.py
"""
Tests for Chat Session REST API Endpoints

Tests the session CRUD endpoints:
- POST /api/chat/sessions - Create session
- GET /api/chat/sessions/{session_id} - Get session
- GET /api/chat/sessions - List sessions
- PATCH /api/chat/sessions/{session_id} - Rename session
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.chat_endpoints import _normalize_chat_context


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


class TestUpdateSessionEndpoint:
    """Test PATCH /api/chat/sessions/{session_id} endpoint."""

    def test_patch_session_title(self, client):
        """PATCH updates session title."""
        create_resp = client.post(
            "/api/chat/sessions",
            json={
                "agent_type": "floor-manager",
                "agent_id": "floor-manager",
                "user_id": "user-1",
                "title": "Original title",
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

        patch_resp = client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"title": "Renamed title"},
        )
        assert patch_resp.status_code == 200
        data = patch_resp.json()
        assert data["id"] == session_id
        assert data["title"] == "Renamed title"

    def test_patch_session_rejects_empty_title(self, client):
        """PATCH rejects empty titles."""
        create_resp = client.post(
            "/api/chat/sessions",
            json={
                "agent_type": "floor-manager",
                "agent_id": "floor-manager",
                "user_id": "user-1",
            },
        )
        assert create_resp.status_code == 200
        session_id = create_resp.json()["id"]

        patch_resp = client.patch(
            f"/api/chat/sessions/{session_id}",
            json={"title": "   "},
        )
        assert patch_resp.status_code == 400


class TestContextNormalization:
    """Test deployment-safe context normalization."""

    def test_does_not_force_provider_or_model_defaults(self):
        normalized = _normalize_chat_context(
            {"canvas_context": {"canvas": "research"}},
            default_canvas="research",
            default_department="research",
        )

        assert normalized["canvas"] == "research"
        assert normalized["department"] == "research"
        assert "provider" not in normalized
        assert "model" not in normalized

    def test_filters_workspace_resource_hints_to_active_canvas_and_shared_assets(self):
        normalized = _normalize_chat_context(
            {
                "canvas": "development",
                "active_canvas": "development",
                "workspace_resource_hints": [
                    {"id": "dev-1", "canvas": "development", "path": "/tmp/dev.py", "label": "Dev file"},
                    {"id": "shared-1", "canvas": "shared-assets", "path": "/tmp/shared.md", "label": "Shared doc"},
                    {"id": "research-1", "canvas": "research", "path": "/tmp/research.md", "label": "Research doc"},
                ],
            },
            default_canvas="development",
            default_department="development",
        )

        hints = normalized["workspace_resource_hints"]
        assert [hint["canvas"] for hint in hints] == ["development", "shared-assets"]

    def test_keeps_cross_canvas_workspace_resource_hints_for_floor_manager(self):
        normalized = _normalize_chat_context(
            {
                "canvas": "flowforge",
                "active_canvas": "flowforge",
                "workspace_resource_hints": [
                    {"id": "risk-1", "canvas": "risk", "path": "/tmp/risk.json", "label": "Risk"},
                    {"id": "research-1", "canvas": "research", "path": "/tmp/research.md", "label": "Research"},
                ],
            },
            default_canvas="flowforge",
            default_department="floor-manager",
        )

        hints = normalized["workspace_resource_hints"]
        assert [hint["canvas"] for hint in hints] == ["risk", "research"]
