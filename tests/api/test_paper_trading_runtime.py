from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.paper_trading import agents, routes
from src.api.paper_trading.runtime import (
    PaperTradingUnavailableDeployer,
    ensure_paper_trading_runtime,
)


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_get_deployer_returns_unavailable_stub_when_mt5_runtime_missing():
    deployer = agents.get_deployer()

    assert getattr(deployer, "available", True) is False
    assert "paper trading runtime" in deployer.unavailable_reason.lower()


def test_active_endpoint_degrades_to_empty_payload_when_runtime_unavailable():
    client = build_client()
    client.app.dependency_overrides[agents.get_deployer] = (
        lambda: PaperTradingUnavailableDeployer("runtime unavailable for test")
    )

    response = client.get("/api/paper-trading/active")

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_runtime_guard_raises_503_for_write_paths():
    deployer = PaperTradingUnavailableDeployer("runtime unavailable for test")

    try:
        ensure_paper_trading_runtime(deployer)
    except Exception as exc:  # pragma: no cover - assertion follows
        assert getattr(exc, "status_code", None) == 503
        assert "runtime unavailable for test" in str(getattr(exc, "detail", exc))
    else:  # pragma: no cover
        raise AssertionError("expected 503 HTTPException for unavailable runtime")
