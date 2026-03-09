# tests/api/test_chat_per_agent.py
"""
Tests for Per-Agent Chat Endpoints

Tests the per-agent messaging endpoints:
- POST /api/chat/workshop/message - Chat with Workshop Copilot
- POST /api/chat/floor-manager/message - Chat with Floor Manager
- POST /api/chat/departments/{dept}/message - Chat with department agents
"""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_workshop_chat_endpoint(client):
    """Test POST /api/chat/workshop/message."""
    response = client.post(
        "/api/chat/workshop/message",
        json={
            "message": "Hello",
            "session_id": None
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data


def test_floor_manager_chat_endpoint(client):
    """Test POST /api/chat/floor-manager/message."""
    response = client.post(
        "/api/chat/floor-manager/message",
        json={
            "message": "Execute trade",
            "session_id": None
        }
    )
    assert response.status_code == 200


def test_department_chat_endpoint(client):
    """Test POST /api/chat/departments/research/message."""
    response = client.post(
        "/api/chat/departments/research/message",
        json={
            "message": "Analyze market",
            "session_id": None
        }
    )
    assert response.status_code == 200
