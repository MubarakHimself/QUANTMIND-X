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
from unittest.mock import AsyncMock
from types import SimpleNamespace

from src.api.server import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_workshop_chat_endpoint(client):
    """Test POST /api/chat/workshop/message."""
    class FakeWorkshopResponse:
        reply = "handled by workshop service"
        action_taken = "direct_response"
        delegation = None

    fake_service = AsyncMock()
    fake_service.handle_message = AsyncMock(return_value=FakeWorkshopResponse())

    from src.api.services import workshop_copilot_service as workshop_service_module

    original_service = workshop_service_module._workshop_copilot_service
    workshop_service_module._workshop_copilot_service = fake_service
    response = client.post(
        "/api/chat/workshop/message",
        json={
            "message": "Hello",
            "session_id": None
        }
    )
    workshop_service_module._workshop_copilot_service = original_service

    assert response.status_code == 200
    data = response.json()
    assert data["reply"] == "handled by workshop service"
    assert data["action_taken"] == "direct_response"


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


def test_department_chat_forwards_request_context_to_dept_head(monkeypatch, client):
    from src.api import chat_endpoints
    from src.agents.departments.types import Department

    captured: dict = {}
    fake_session_service = SimpleNamespace(
        add_message=AsyncMock(),
        get_messages=AsyncMock(return_value=[]),
    )

    class FakeHead:
        system_prompt = "research"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
            captured["task"] = task
            captured["canvas_context"] = canvas_context
            return {"content": "ok"}

    class FakeFloorManager:
        def __init__(self):
            self._department_heads = {Department.RESEARCH: FakeHead()}

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "add_message",
        AsyncMock(return_value=SimpleNamespace(id="msg-ctx-1")),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=[]),
    )

    response = client.post(
        "/api/chat/departments/research/message",
        json={
            "message": "Analyze the current research canvas",
            "session_id": "session-ctx-1",
            "context": {
                "canvas": "research",
                "attached_contexts": [
                    {
                        "canvas": "research",
                        "template": {"base_descriptor": "Research Department Copilot"},
                        "memory_identifiers": ["node-1", "node-2"],
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    assert captured["task"] == "Analyze the current research canvas"
    assert captured["canvas_context"]["canvas"] == "research"
    assert captured["canvas_context"]["attached_contexts"][0]["memory_identifiers"] == ["node-1", "node-2"]


def test_department_chat_stream_normalizes_content_chunks(monkeypatch, client):
    from src.agents.departments.types import Department
    from src.api import chat_endpoints

    class FakeHead:
        def _build_system_prompt(self):
            return "system"

    class FakeFloorManager:
        def __init__(self):
            self._department_heads = {Department.RESEARCH: FakeHead()}

        async def _invoke_llm_stream(self, message, history=None, system_prompt=None):
            yield {"type": "content", "content": "Hello "}
            yield {"type": "content", "content": "world"}

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "add_message",
        AsyncMock(return_value=SimpleNamespace(id="msg-stream-1")),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=[]),
    )

    with client.stream(
        "POST",
        "/api/chat/departments/research/message",
        json={"message": "Hello", "stream": True, "session_id": "test-session"},
    ) as response:
        assert response.status_code == 200
        payload = "\n".join(response.iter_text())

    assert '"delta": "Hello "' in payload
    assert '"delta": "world"' in payload
    assert '"session_id": "test-session"' in payload


def test_get_session_messages_endpoint_returns_chronological_history(monkeypatch, client):
    from src.api import chat_endpoints

    fake_session = SimpleNamespace(id="session-123")
    fake_messages = [
        SimpleNamespace(
            id="msg-1",
            session_id="session-123",
            role="user",
            content="hello",
            created_at=None,
            artifacts=[],
            tool_calls=[],
            token_count=None,
        ),
        SimpleNamespace(
            id="msg-2",
            session_id="session-123",
            role="assistant",
            content="world",
            created_at=None,
            artifacts=[{"kind": "report"}],
            tool_calls=[{"name": "lookup"}],
            token_count=42,
        ),
    ]

    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_session",
        AsyncMock(return_value=fake_session),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=fake_messages),
    )

    response = client.get("/api/chat/sessions/session-123/messages")

    assert response.status_code == 200
    data = response.json()
    assert [item["id"] for item in data] == ["msg-1", "msg-2"]
    assert data[0]["role"] == "user"
    assert data[0]["content"] == "hello"
    assert data[0]["metadata"] == {}
    assert data[1]["metadata"] == {
        "artifacts": [{"kind": "report"}],
        "tool_calls": [{"name": "lookup"}],
        "token_count": 42,
    }


def test_get_session_messages_endpoint_404s_for_unknown_session(monkeypatch, client):
    from src.api import chat_endpoints

    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_session",
        AsyncMock(return_value=None),
    )

    response = client.get("/api/chat/sessions/missing/messages")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_floor_manager_chat_stream_uses_session_backed_sse(monkeypatch, client):
    from src.api import chat_endpoints

    async def fake_chat_stream(message, context=None, history=None):
        yield {"type": "tool", "tool": "thinking", "status": "started"}
        yield {"type": "content", "delta": "hello "}
        yield {"type": "content", "delta": "world"}

    class FakeFloorManager:
        def chat_stream(self, message, context=None, history=None):
            return fake_chat_stream(message, context=context, history=history)

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    fake_session_service = SimpleNamespace(
        add_message=AsyncMock(return_value=SimpleNamespace(id="msg-floor-stream-1")),
        get_messages=AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        chat_endpoints,
        "_session_service",
        fake_session_service,
    )

    with client.stream(
        "POST",
        "/api/chat/floor-manager/message",
        json={
            "message": "hello",
            "session_id": "session-123",
            "context": {"canvas_context": "trading"},
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        payload = "\n".join(response.iter_text())

    assert '"delta": "hello "' in payload
    assert '"delta": "world"' in payload
    assert '"session_id": "session-123"' in payload
