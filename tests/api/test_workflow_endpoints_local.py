from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_client() -> TestClient:
    from src.api.approval_gate import router as approval_gate_router
    from src.api.workflow_endpoints import router as workflow_router

    app = FastAPI()
    app.include_router(approval_gate_router)
    app.include_router(workflow_router)
    return TestClient(app)


def test_workflow_approvals_reads_local_gate_state():
    client = _make_client()
    workflow_id = "wf-local-approvals"

    create = client.post(
        "/api/approval-gates",
        json={
            "workflow_id": workflow_id,
            "workflow_type": "video_ingest_to_ea",
            "from_stage": "quantcode",
            "to_stage": "backtest",
            "gate_type": "stage_transition",
            "requester": "system",
        },
    )
    assert create.status_code == 201

    response = client.get(f"/api/workflows/{workflow_id}/approvals")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert any(gate["workflow_id"] == workflow_id for gate in data["gates"])


def test_workflow_pending_approvals_includes_pending_review():
    client = _make_client()
    workflow_id = "wf-local-pending-review"

    create = client.post(
        "/api/approval-gates",
        json={
            "workflow_id": workflow_id,
            "workflow_type": "video_ingest_to_ea",
            "from_stage": "backtest",
            "to_stage": "papertrade",
            "gate_type": "alpha_forge_backtest",
            "requester": "system",
            "strategy_id": "strat-1",
        },
    )
    assert create.status_code == 201

    response = client.get(f"/api/workflows/{workflow_id}/approvals/pending")
    assert response.status_code == 200
    data = response.json()
    assert data["has_pending"] is True
    assert data["total"] >= 1
    assert all(gate["workflow_id"] == workflow_id for gate in data["gates"])

