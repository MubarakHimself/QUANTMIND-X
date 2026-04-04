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


def test_floor_manager_chat_endpoint(monkeypatch, client):
    """Test POST /api/chat/floor-manager/message."""
    class FakeFloorManager:
        async def chat(
            self,
            message,
            context=None,
            history=None,
            model_name=None,
            preferred_provider=None,
            stream=False,
        ):
            return {"content": "ok"}

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )

    response = client.post(
        "/api/chat/floor-manager/message",
        json={
            "message": "Execute trade",
            "session_id": None
        }
    )
    assert response.status_code == 200


def test_department_chat_endpoint(monkeypatch, client):
    """Test POST /api/chat/departments/research/message."""
    from src.agents.departments.types import Department

    class FakeHead:
        system_prompt = "system"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
            return {"content": "ok"}

    class FakeFloorManager:
        def __init__(self):
            self._department_heads = {Department.RESEARCH: FakeHead()}

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )

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


def test_department_chat_compacts_manifest_context(monkeypatch, client):
    from src.api import chat_endpoints
    from src.agents.departments.types import Department

    captured: dict = {}

    class FakeHead:
        system_prompt = "research"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
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
        AsyncMock(return_value=SimpleNamespace(id="msg-ctx-compact-1")),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=[]),
    )

    response = client.post(
        "/api/chat/departments/research/message",
        json={
            "message": "Use shared assets naturally",
            "session_id": "session-ctx-compact-1",
            "context": {
                "canvas": "research",
                "provider": "minimax",
                "model": "MiniMax-M2.7",
                "workspace_contract": {
                    "version": "manifest-v1",
                    "strategy": "manifest-first",
                    "natural_resource_search": True,
                },
                "workspace_resource_hints": [
                    {
                        "id": "articles/fx/alpha-note-1.md",
                        "canvas": "research",
                        "path": "/shared/articles/fx/alpha-note-1.md",
                        "label": "Alpha note 1",
                        "description": "FX alpha note",
                        "content": "x" * 10000,
                    }
                ],
                "attached_contexts": [
                    {
                        "canvas": "research",
                        "template": {"base_descriptor": "Research Department Copilot"},
                        "memory_identifiers": ["node-1"],
                        "resources": [
                            {
                                "id": "articles/fx/alpha-note-1.md",
                                "path": "/shared/articles/fx/alpha-note-1.md",
                                "label": "Alpha note 1",
                            }
                        ],
                        "visible_canvas_context": {"raw": "y" * 15000},
                    }
                ],
                "visible_canvas_context": {"raw": "z" * 15000},
            },
        },
    )

    assert response.status_code == 200
    compact_ctx = captured["canvas_context"]
    assert compact_ctx["workspace_contract"]["strategy"] == "manifest-first"
    assert compact_ctx["provider"] == "minimax"
    assert compact_ctx["model"] == "MiniMax-M2.7"
    assert len(compact_ctx["workspace_resource_hints"]) == 1
    assert "visible_canvas_context" not in compact_ctx
    assert compact_ctx["attached_contexts"][0]["memory_identifiers"] == ["node-1"]


def test_floor_manager_chat_uses_normalized_context(monkeypatch, client):
    from src.api import chat_endpoints

    captured: dict = {}

    class FakeFloorManager:
        async def chat(
            self,
            message,
            context=None,
            history=None,
            model_name=None,
            preferred_provider=None,
            stream=False,
        ):
            captured["context"] = context
            captured["model_name"] = model_name
            captured["preferred_provider"] = preferred_provider
            return {"content": "ok"}

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "add_message",
        AsyncMock(return_value=SimpleNamespace(id="msg-floor-norm-1")),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=[]),
    )

    response = client.post(
        "/api/chat/floor-manager/message",
        json={
            "message": "Check active workflows",
            "session_id": "session-floor-norm-1",
            "context": {
                "provider": "minimax",
                "model": "MiniMax-M2.5",
                "workspace_contract": {
                    "version": "manifest-v1",
                    "strategy": "manifest-first",
                },
                "workspace_resource_hints": [{"id": "wf-1", "canvas": "flowforge"}],
                "raw_canvas_dump": {"payload": "a" * 10000},
            },
        },
    )

    assert response.status_code == 200
    assert captured["preferred_provider"] == "minimax"
    assert captured["model_name"] == "MiniMax-M2.5"
    assert captured["context"]["workspace_contract"]["strategy"] == "manifest-first"
    assert "raw_canvas_dump" not in captured["context"]


def test_floor_manager_chat_recreates_stale_session(monkeypatch, client):
    from src.api import chat_endpoints

    class FakeFloorManager:
        async def chat(
            self,
            message,
            context=None,
            history=None,
            model_name=None,
            preferred_provider=None,
            stream=False,
        ):
            return {"content": "ok"}

    create_session_mock = AsyncMock(return_value=SimpleNamespace(id="session-recreated-1"))
    add_message_mock = AsyncMock(return_value=SimpleNamespace(id="msg-recreated-1"))

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(chat_endpoints._session_service, "get_session", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_endpoints._session_service, "create_session", create_session_mock)
    monkeypatch.setattr(chat_endpoints._session_service, "add_message", add_message_mock)
    monkeypatch.setattr(chat_endpoints._session_service, "get_messages", AsyncMock(return_value=[]))

    response = client.post(
        "/api/chat/floor-manager/message",
        json={
            "message": "stream check",
            "session_id": "stale-session-id",
            "context": {"canvas": "workshop"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session-recreated-1"
    create_session_mock.assert_called_once()
    assert add_message_mock.call_count >= 2


def test_department_chat_recreates_stale_session(monkeypatch, client):
    from src.api import chat_endpoints
    from src.agents.departments.types import Department

    class FakeHead:
        system_prompt = "system"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
            return {"content": "ok"}

    class FakeFloorManager:
        def __init__(self):
            self._department_heads = {Department.RESEARCH: FakeHead()}

    create_session_mock = AsyncMock(return_value=SimpleNamespace(id="dept-session-recreated-1"))
    add_message_mock = AsyncMock(return_value=SimpleNamespace(id="dept-msg-recreated-1"))

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(chat_endpoints._session_service, "get_session", AsyncMock(return_value=None))
    monkeypatch.setattr(chat_endpoints._session_service, "create_session", create_session_mock)
    monkeypatch.setattr(chat_endpoints._session_service, "add_message", add_message_mock)
    monkeypatch.setattr(chat_endpoints._session_service, "get_messages", AsyncMock(return_value=[]))

    response = client.post(
        "/api/chat/departments/research/message",
        json={
            "message": "Analyze market",
            "session_id": "stale-dept-session-id",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "dept-session-recreated-1"
    create_session_mock.assert_called_once()
    assert add_message_mock.call_count >= 2


def test_department_chat_stream_uses_department_head_runtime(monkeypatch, client):
    from src.agents.departments.types import Department
    from src.api import chat_endpoints

    class FakeHead:
        system_prompt = "system"
        invoked = False

        def _build_system_prompt(self, canvas_context=None, memory_nodes=None):
            return "system"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
            self.invoked = True
            return {"content": "Hello world from department head"}

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

    assert '"delta": "Hello world from department head"' in payload
    assert '"type": "status", "phase": "thinking", "status": "started"' in payload
    assert '"type": "status", "phase": "thinking", "status": "streaming"' in payload
    assert '"session_id": "' in payload
    assert payload.count('"type": "done"') == 1


def test_department_chat_stream_falls_back_when_only_provisional_preamble_streams(monkeypatch, client):
    from src.agents.departments.types import Department
    from src.api import chat_endpoints

    class FakeHead:
        system_prompt = "system"

        def _build_system_prompt(self, canvas_context=None, memory_nodes=None):
            return "system"

        async def _invoke_claude(self, task, canvas_context=None, tools=None):
            return {
                "content": (
                    "**Result:** `src/eas/example_ea.mq5`\n\n"
                    "**Sub-agent used:** `mql5_dev`"
                )
            }

    class FakeFloorManager:
        def __init__(self):
            self._department_heads = {Department.DEVELOPMENT: FakeHead()}

        async def _invoke_llm_stream(
            self,
            message,
            history=None,
            system_prompt=None,
            model_name=None,
            preferred_provider=None,
        ):
            yield {"type": "thinking_start"}
            yield {"type": "content", "content": "I'll search the workspace for EA-related files."}
            yield {"type": "done"}

    add_message_mock = AsyncMock(return_value=SimpleNamespace(id="msg-stream-fallback-1"))

    monkeypatch.setattr(
        "src.agents.departments.floor_manager.get_floor_manager",
        lambda: FakeFloorManager(),
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "add_message",
        add_message_mock,
    )
    monkeypatch.setattr(
        chat_endpoints._session_service,
        "get_messages",
        AsyncMock(return_value=[]),
    )

    with client.stream(
        "POST",
        "/api/chat/departments/development/message",
        json={"message": "Find an EA file", "stream": True, "session_id": "dev-stream-1"},
    ) as response:
        assert response.status_code == 200
        payload = "\n".join(response.iter_text())

    assert "src/eas/example_ea.mq5" in payload
    assert "mql5_dev" in payload
    assert "I'll search the workspace for EA-related files." not in payload
    assert payload.count('"type": "done"') == 1

    persisted_assistant_content = add_message_mock.await_args_list[-1].kwargs["content"]
    assert "src/eas/example_ea.mq5" in persisted_assistant_content
    assert "mql5_dev" in persisted_assistant_content


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

    async def fake_chat_stream(
        message,
        context=None,
        history=None,
        model_name=None,
        preferred_provider=None,
    ):
        yield {"type": "tool", "tool": "thinking", "status": "started"}
        yield {"type": "content", "delta": "hello "}
        yield {"type": "content", "delta": "world"}
        yield {"type": "done"}

    class FakeFloorManager:
        def chat_stream(
            self,
            message,
            context=None,
            history=None,
            model_name=None,
            preferred_provider=None,
        ):
            return fake_chat_stream(
                message,
                context=context,
                history=history,
                model_name=model_name,
                preferred_provider=preferred_provider,
            )

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
    assert payload.count('"type": "done"') == 1
