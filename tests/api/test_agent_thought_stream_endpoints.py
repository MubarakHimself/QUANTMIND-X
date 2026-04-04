from fastapi.testclient import TestClient

from src.api.server import app
from src.api.agent_thought_stream_endpoints import get_thought_publisher


def test_thought_history_filters_by_session_id():
    client = TestClient(app)
    publisher = get_thought_publisher()

    publisher.publish(
        department="research",
        thought="session-a-thought",
        session_id="session-a",
    )
    publisher.publish(
        department="risk",
        thought="session-b-thought",
        session_id="session-b",
    )

    response = client.get("/api/agent-thoughts/history", params={"session_id": "session-a", "limit": 10})

    assert response.status_code == 200
    data = response.json()
    assert any(item["content"] == "session-a-thought" for item in data)
    assert all(item.get("session_id") == "session-a" for item in data)
