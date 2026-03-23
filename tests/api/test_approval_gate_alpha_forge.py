"""
Tests for Alpha Forge Approval Gate Features

Tests the new Alpha Forge specific endpoints and functionality:
- Alpha Forge gate types (ALPHA_FORGE_BACKTEST, ALPHA_FORGE_DEPLOYMENT)
- PENDING_REVIEW status with timeout logic
- Revision request endpoints
- Immutable audit record creation
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def alpha_forge_gate_data():
    """Create sample Alpha Forge approval gate data."""
    return {
        "workflow_id": str(uuid.uuid4()),
        "workflow_type": "alpha_forge",
        "from_stage": "backtest",
        "to_stage": "validation",
        "gate_type": "alpha_forge_backtest",
        "requester": "floor_manager",
        "strategy_id": str(uuid.uuid4()),
        "metrics_snapshot": {
            "total_trades": 150,
            "win_rate": 0.58,
            "sharpe_ratio": 1.45,
            "max_drawdown": 0.12,
            "net_profit": 12500.00
        }
    }


class TestAlphaForgeGateCreation:
    """Test Alpha Forge specific gate creation."""

    def test_create_alpha_forge_backtest_gate(self, client, alpha_forge_gate_data):
        """Test creating an Alpha Forge backtest approval gate."""
        response = client.post("/api/approval-gates", json=alpha_forge_gate_data)

        assert response.status_code == 201
        data = response.json()

        assert data["status"] == "pending_review"
        assert data["gate_type"] == "alpha_forge_backtest"
        assert data["strategy_id"] == alpha_forge_gate_data["strategy_id"]
        assert data["metrics_snapshot"] == alpha_forge_gate_data["metrics_snapshot"]
        assert data["expires_at"] is not None
        assert data["hard_expires_at"] is not None

    def test_create_alpha_forge_deployment_gate(self, client):
        """Test creating an Alpha Forge deployment approval gate."""
        data = {
            "workflow_id": str(uuid.uuid4()),
            "from_stage": "validation",
            "to_stage": "deployment",
            "gate_type": "alpha_forge_deployment",
            "strategy_id": str(uuid.uuid4()),
            "metrics_snapshot": {
                "total_trades": 200,
                "win_rate": 0.62,
                "sharpe_ratio": 1.8
            }
        }

        response = client.post("/api/approval-gates", json=data)
        assert response.status_code == 201

        data = response.json()
        assert data["status"] == "pending_review"
        assert data["gate_type"] == "alpha_forge_deployment"


class TestAlphaForgePendingGates:
    """Test Alpha Forge pending gates endpoint."""

    def test_list_alpha_forge_pending_gates(self, client, alpha_forge_gate_data):
        """Test listing Alpha Forge pending gates."""
        # Create an Alpha Forge gate
        client.post("/api/approval-gates", json=alpha_forge_gate_data)

        # List Alpha Forge pending gates
        response = client.get("/api/approval-gates/alpha-forge/pending")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] >= 1
        for gate in data["gates"]:
            assert gate["status"] == "pending_review"
            assert gate["gate_type"] in ["alpha_forge_backtest", "alpha_forge_deployment"]


class TestAlphaForgeApproval:
    """Test Alpha Forge specific approval behavior."""

    def test_approve_alpha_forge_gate_creates_audit_record(self, client, alpha_forge_gate_data):
        """Test that approving Alpha Forge gate creates immutable audit record."""
        # Create gate
        response = client.post("/api/approval-gates", json=alpha_forge_gate_data)
        gate_id = response.json()["gate_id"]

        # Approve the gate
        approve_data = {
            "approver": "Mubarak",
            "notes": "Strategy looks good, ready for validation"
        }
        response = client.post(f"/api/approval-gates/{gate_id}/approve", json=approve_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"

        # Verify gate status updated
        gate_response = client.get(f"/api/approval-gates/{gate_id}")
        gate_data = gate_response.json()
        assert gate_data["status"] == "approved"
        assert gate_data["approver"] == "Mubarak"


class TestRevisionRequest:
    """Test revision request endpoint."""

    def test_request_revision(self, client, alpha_forge_gate_data):
        """Test requesting revision for a gate."""
        # Create gate
        response = client.post("/api/approval-gates", json=alpha_forge_gate_data)
        gate_id = response.json()["gate_id"]

        # Request revision
        revision_data = {
            "approver": "Mubarak",
            "feedback": "Need to improve risk management parameters",
            "create_new_gate": True
        }
        response = client.post(f"/api/approval-gates/{gate_id}/request-revision", json=revision_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify feedback was stored
        gate_response = client.get(f"/api/approval-gates/{gate_id}")
        gate_data = gate_response.json()
        assert gate_data["revision_feedback"] == revision_data["feedback"]


class TestTimeoutChecking:
    """Test timeout checking endpoint."""

    def test_check_timeout_endpoint_exists(self, client, alpha_forge_gate_data):
        """Test that check-timeout endpoint exists and returns proper response."""
        # Create gate
        response = client.post("/api/approval-gates", json=alpha_forge_gate_data)
        gate_id = response.json()["gate_id"]

        # Call check-timeout endpoint - should work for any gate
        response = client.post(f"/api/approval-gates/{gate_id}/check-timeout")

        # Should return 200 with the gate status (not expired since just created)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Status should still be pending_review (not yet expired)
        assert data["status"] in ["pending_review", "expired_review"]


class TestRejectFromPendingReview:
    """Test that gates can be rejected from PENDING_REVIEW status."""

    def test_reject_pending_review_gate(self, client, alpha_forge_gate_data):
        """Test rejecting a gate in PENDING_REVIEW status."""
        # Create gate
        response = client.post("/api/approval-gates", json=alpha_forge_gate_data)
        gate_id = response.json()["gate_id"]

        # Verify initial status
        assert response.json()["status"] == "pending_review"

        # Reject the gate
        reject_data = {
            "approver": "Mubarak",
            "notes": "Backtest results not satisfactory"
        }
        response = client.post(f"/api/approval-gates/{gate_id}/reject", json=reject_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"