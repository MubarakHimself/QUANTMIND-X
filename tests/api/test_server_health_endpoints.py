from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_server_health_metrics_degrades_when_cloudzy_unconfigured(monkeypatch):
    from src.api.server_health_endpoints import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.delenv("CLOUDZY_REACHABLE", raising=False)

    response = client.get("/api/server/health/metrics")

    assert response.status_code == 200
    data = response.json()
    assert data["contabo"]["status"] in {"healthy", "warning", "critical", "unknown", "error"}
    assert data["cloudzy"]["status"] == "disconnected"
    assert data["cloudzy"]["last_heartbeat"] in {"", None}
    assert "not configured" in data["cloudzy"]["status_detail"].lower()
