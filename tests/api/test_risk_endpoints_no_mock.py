from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app():
    from src.api.risk_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


def test_physics_endpoint_fails_honestly_without_mt5(monkeypatch):
    from src.api import risk_endpoints

    monkeypatch.setattr(risk_endpoints, "_fetch_mt5_candles", lambda **_: None)

    client = TestClient(_build_app())
    response = client.get("/api/risk/physics")

    assert response.status_code == 503
    assert "mt5" in response.json()["detail"].lower()


def test_compliance_endpoint_returns_empty_live_state_instead_of_demo(monkeypatch):
    from src.api import risk_endpoints

    monkeypatch.setattr(risk_endpoints, "_demo_account_tags", ["demo-a", "demo-b"])

    client = TestClient(_build_app())
    response = client.get("/api/risk/compliance")

    assert response.status_code == 200
    data = response.json()
    assert data["account_tags"] == []
    assert data["islamic"]["active_positions_count"] == 0


def test_calendar_blackout_endpoint_returns_no_synthetic_blackouts(monkeypatch):
    from src.api import risk_endpoints
    from src.risk.models.calendar import NewsItem, CalendarEventType, NewsImpact

    monkeypatch.setattr(
        risk_endpoints,
        "_calendar_events",
        [
            NewsItem(
                event_id="evt_live_1",
                title="CPI Release",
                event_type=CalendarEventType.CPI,
                impact=NewsImpact.HIGH,
                event_time=datetime.now(timezone.utc),
                currencies=["USD"],
                source="test",
            )
        ],
    )
    monkeypatch.setattr(risk_endpoints, "_demo_account_tags", ["demo-a", "demo-b"])

    client = TestClient(_build_app())
    response = client.get("/api/risk/calendar/blackout")

    assert response.status_code == 200
    data = response.json()
    assert len(data["events"]) == 1
    assert data["blackouts"] == []
