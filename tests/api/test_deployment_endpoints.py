"""
Tests for EA Deployment API Endpoints

Tests deployment request/response structures and endpoint behavior.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.deployment_endpoints import router, _deployments

    _deployments.clear()  # Reset state between tests

    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestDeploymentModels:
    """Test deployment request/response model structures."""

    def test_deployment_request_required_fields(self):
        from src.api.deployment_endpoints import DeploymentRequest

        req = DeploymentRequest(strategy_id="strat-001", ea_name="QuantMindX_strat001")
        assert req.strategy_id == "strat-001"
        assert req.ea_name == "QuantMindX_strat001"
        assert req.ea_params is None
        assert req.source_dir == "src/mql5/Experts"

    def test_deployment_response_structure(self):
        from src.api.deployment_endpoints import DeploymentResponse

        resp = DeploymentResponse(
            deployment_id="deploy_strat001_1700000000",
            strategy_id="strat-001",
            status="pending",
            status_detail="Deployment queued",
            started_at="2026-03-21T10:00:00",
        )
        assert resp.status == "pending"
        assert resp.completed_at is None
        assert resp.error is None


class TestDeploymentEndpoints:
    """Test deployment endpoints with in-memory state."""

    def test_deploy_ea_returns_pending(self):
        client = _make_client()
        response = client.post(
            "/api/deployment/deploy",
            json={"strategy_id": "strat-001", "ea_name": "TestEA"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["strategy_id"] == "strat-001"
        assert "deployment_id" in data

    def test_deploy_returns_deployment_id(self):
        client = _make_client()
        response = client.post(
            "/api/deployment/deploy",
            json={"strategy_id": "strat-xyz", "ea_name": "XYZ_EA"},
        )
        data = response.json()
        assert data["deployment_id"].startswith("deploy_strat-xyz")

    def test_get_deployment_status_not_found(self):
        client = _make_client()
        response = client.get("/api/deployment/status/nonexistent-id")
        assert response.status_code == 404

    def test_get_deployment_status_found(self):
        client = _make_client()
        # First create a deployment
        post_resp = client.post(
            "/api/deployment/deploy",
            json={"strategy_id": "strat-002", "ea_name": "EA_002"},
        )
        deployment_id = post_resp.json()["deployment_id"]

        # Then get its status
        get_resp = client.get(f"/api/deployment/status/{deployment_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["deployment_id"] == deployment_id

    def test_list_deployments_empty(self):
        client = _make_client()
        response = client.get("/api/deployment/list")
        assert response.status_code == 200
        data = response.json()
        assert "deployments" in data
        assert "total" in data

    def test_list_deployments_with_filter(self):
        client = _make_client()
        # Create a deployment first
        client.post(
            "/api/deployment/deploy",
            json={"strategy_id": "filterable-strat", "ea_name": "Filter_EA"},
        )

        response = client.get(
            "/api/deployment/list",
            params={"strategy_id": "filterable-strat"},
        )
        assert response.status_code == 200
        data = response.json()
        for d in data["deployments"]:
            assert d["strategy_id"] == "filterable-strat"

    def test_cancel_deployment_pending(self):
        from src.api.deployment_endpoints import _deployments
        from datetime import datetime

        client = _make_client()
        # Inject a pending deployment directly to avoid background task race
        deployment_id = "deploy_strat-cancel_9999999999"
        _deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "strategy_id": "strat-cancel",
            "ea_name": "Cancel_EA",
            "ea_params": None,
            "source_dir": "src/mql5/Experts",
            "status": "pending",
            "status_detail": "Deployment queued",
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
        }

        # Cancel it
        cancel_resp = client.post(f"/api/deployment/cancel/{deployment_id}")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_deployment_404(self):
        client = _make_client()
        response = client.post("/api/deployment/cancel/no-such-id")
        assert response.status_code == 404

    def test_approval_webhook_approved_triggers_deployment(self):
        client = _make_client()
        response = client.post(
            "/api/deployment/webhook/approval-trigger",
            params={"strategy_id": "strat-approved", "approval_status": "approved"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "triggered"
        assert "deployment_id" in data

    def test_approval_webhook_rejected_skips(self):
        client = _make_client()
        response = client.post(
            "/api/deployment/webhook/approval-trigger",
            params={"strategy_id": "strat-rejected", "approval_status": "rejected"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
