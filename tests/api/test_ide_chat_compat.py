from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_legacy_chat_endpoint_routes_copilot_to_floor_manager(monkeypatch):
    from src.api import ide_chat

    app = FastAPI()
    app.include_router(ide_chat.router)
    client = TestClient(app)

    async def fake_floor_manager_chat(request):
        return SimpleNamespace(
            session_id="sess_floor",
            message_id="msg_floor",
            reply="floor-manager reply",
            artifacts=[],
            action_taken=None,
            delegation=None,
        )

    monkeypatch.setattr(ide_chat, "floor_manager_chat", fake_floor_manager_chat, raising=False)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "agent": "copilot", "model": "compat-model"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "floor-manager reply"
    assert data["agent"] == "copilot"
    assert data["model"] == "compat-model"


def test_legacy_chat_endpoint_routes_analyst_to_research_department(monkeypatch):
    from src.api import ide_chat

    app = FastAPI()
    app.include_router(ide_chat.router)
    client = TestClient(app)

    async def fake_department_chat(dept, request):
        assert dept == "research"
        return SimpleNamespace(
            session_id="sess_research",
            message_id="msg_research",
            reply="research reply",
            artifacts=[],
            action_taken=None,
            delegation=None,
        )

    monkeypatch.setattr(ide_chat, "department_chat", fake_department_chat, raising=False)

    response = client.post(
        "/api/chat",
        json={"message": "analyze this", "agent": "analyst", "model": "compat-model"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "research reply"
    assert data["agent"] == "analyst"
    assert data["model"] == "compat-model"


def test_legacy_chat_endpoint_defaults_to_floor_manager_when_agent_missing(monkeypatch):
    from src.api import ide_chat

    app = FastAPI()
    app.include_router(ide_chat.router)
    client = TestClient(app)

    async def fake_floor_manager_chat(request):
        return SimpleNamespace(
            session_id="sess_floor",
            message_id="msg_floor",
            reply="floor-manager default reply",
            artifacts=[],
            action_taken=None,
            delegation=None,
        )

    monkeypatch.setattr(ide_chat, "floor_manager_chat", fake_floor_manager_chat, raising=False)

    response = client.post(
        "/api/chat",
        json={"message": "hello without explicit agent"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "floor-manager default reply"
    assert data["agent"] == "floor_manager"
