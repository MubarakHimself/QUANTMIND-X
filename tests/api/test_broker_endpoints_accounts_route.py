from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.broker_endpoints import router, broker_connections, account_switcher


def _reset_state() -> None:
    broker_connections.brokers.clear()
    broker_connections.pending.clear()
    account_switcher.clear_active_account()


def test_accounts_route_is_not_captured_as_broker_id():
    _reset_state()
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/brokers/accounts")
    assert response.status_code == 200
    assert response.json() == []


def test_broker_id_route_still_resolves():
    _reset_state()
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/brokers/nonexistent-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Broker not found"
