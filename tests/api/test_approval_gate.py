"""
Tests for Approval Gate API Endpoints

Tests /api/approval-gates endpoints for workflow stage approval gates.
"""

import pytest
import uuid
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.server import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_workflow_id():
    """Generate a sample workflow ID."""
    return str(uuid.uuid4())


@pytest.fixture
def sample_gate_data(sample_workflow_id):
    """Create sample approval gate data."""
    return {
        "workflow_id": sample_workflow_id,
        "workflow_type": "nprd_to_ea",
        "from_stage": "quantcode",
        "to_stage": "compilation",
        "gate_type": "stage_transition",
        "requester": "system",
        "reason": "Ready to compile the generated EA code"
    }


class TestCreateApprovalGate:
    """Test POST /api/approval-gates endpoint."""

    def test_create_approval_gate_success(self, client, sample_gate_data):
        """Test successful approval gate creation."""
        response = client.post("/api/approval-gates", json=sample_gate_data)

        assert response.status_code == 201
        data = response.json()

        assert "gate_id" in data
        assert data["workflow_id"] == sample_gate_data["workflow_id"]
        assert data["workflow_type"] == sample_gate_data["workflow_type"]
        assert data["from_stage"] == sample_gate_data["from_stage"]
        assert data["to_stage"] == sample_gate_data["to_stage"]
        assert data["status"] == "pending"
        assert data["requester"] == sample_gate_data["requester"]

    def test_create_approval_gate_minimal(self, client, sample_workflow_id):
        """Test approval gate creation with minimal data."""
        data = {
            "workflow_id": sample_workflow_id,
            "from_stage": "analyst",
            "to_stage": "quantcode"
        }

        response = client.post("/api/approval-gates", json=data)

        assert response.status_code == 201
        response_data = response.json()
        assert response_data["status"] == "pending"
        assert response_data["from_stage"] == "analyst"
        assert response_data["to_stage"] == "quantcode"

    def test_create_approval_gate_missing_workflow_id(self, client):
        """Test approval gate creation with missing workflow_id."""
        data = {
            "from_stage": "analyst",
            "to_stage": "quantcode"
        }

        response = client.post("/api/approval-gates", json=data)

        assert response.status_code == 422  # Validation error

    def test_create_approval_gate_missing_from_stage(self, client):
        """Test approval gate creation with missing from_stage."""
        data = {
            "workflow_id": str(uuid.uuid4()),
            "to_stage": "quantcode"
        }

        response = client.post("/api/approval-gates", json=data)

        assert response.status_code == 422  # Validation error

    def test_create_approval_gate_missing_to_stage(self, client):
        """Test approval gate creation with missing to_stage."""
        data = {
            "workflow_id": str(uuid.uuid4()),
            "from_stage": "analyst"
        }

        response = client.post("/api/approval-gates", json=data)

        assert response.status_code == 422  # Validation error


class TestGetApprovalGate:
    """Test GET /api/approval-gates/{gate_id} endpoint."""

    def test_get_approval_gate_success(self, client, sample_gate_data):
        """Test successful approval gate retrieval."""
        # First create a gate
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Then retrieve it
        response = client.get(f"/api/approval-gates/{gate_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["gate_id"] == gate_id
        assert data["workflow_id"] == sample_gate_data["workflow_id"]
        assert data["status"] == "pending"

    def test_get_approval_gate_not_found(self, client):
        """Test retrieval of non-existent gate."""
        fake_gate_id = str(uuid.uuid4())
        response = client.get(f"/api/approval-gates/{fake_gate_id}")

        assert response.status_code == 404


class TestApproveGate:
    """Test POST /api/approval-gates/{gate_id}/approve endpoint."""

    def test_approve_gate_success(self, client, sample_gate_data):
        """Test successful gate approval."""
        # First create a gate
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Then approve it
        approve_data = {
            "approver": "admin_user",
            "notes": "Code looks good, approved for compilation"
        }

        response = client.post(f"/api/approval-gates/{gate_id}/approve", json=approve_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "approved"
        assert data["approver"] == "admin_user"

    def test_approve_gate_not_found(self, client):
        """Test approval of non-existent gate."""
        fake_gate_id = str(uuid.uuid4())
        approve_data = {"approver": "admin_user"}

        response = client.post(f"/api/approval-gates/{fake_gate_id}/approve", json=approve_data)

        assert response.status_code == 404

    def test_approve_gate_already_approved(self, client, sample_gate_data):
        """Test approval of already approved gate."""
        # Create and approve first time
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        approve_data = {"approver": "admin_user"}
        client.post(f"/api/approval-gates/{gate_id}/approve", json=approve_data)

        # Try to approve again
        response = client.post(f"/api/approval-gates/{gate_id}/approve", json=approve_data)

        assert response.status_code == 400
        assert "already" in response.json()["detail"].lower()


class TestRejectGate:
    """Test POST /api/approval-gates/{gate_id}/reject endpoint."""

    def test_reject_gate_success(self, client, sample_gate_data):
        """Test successful gate rejection."""
        # First create a gate
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Then reject it
        reject_data = {
            "approver": "admin_user",
            "notes": "Code quality issues, needs revision"
        }

        response = client.post(f"/api/approval-gates/{gate_id}/reject", json=reject_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "rejected"
        assert data["approver"] == "admin_user"

    def test_reject_gate_not_found(self, client):
        """Test rejection of non-existent gate."""
        fake_gate_id = str(uuid.uuid4())
        reject_data = {"approver": "admin_user"}

        response = client.post(f"/api/approval-gates/{fake_gate_id}/reject", json=reject_data)

        assert response.status_code == 404


class TestGetWorkflowGates:
    """Test GET /api/approval-gates/workflow/{workflow_id} endpoint."""

    def test_get_workflow_gates_success(self, client, sample_gate_data):
        """Test successful retrieval of workflow gates."""
        # Create multiple gates for the same workflow
        workflow_id = sample_gate_data["workflow_id"]

        client.post("/api/approval-gates", json=sample_gate_data)

        second_gate = sample_gate_data.copy()
        second_gate["from_stage"] = "compilation"
        second_gate["to_stage"] = "backtest"
        client.post("/api/approval-gates", json=second_gate)

        # Get all gates for the workflow
        response = client.get(f"/api/approval-gates/workflow/{workflow_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 2
        assert len(data["gates"]) == 2

    def test_get_workflow_gates_empty(self, client):
        """Test retrieval of gates for workflow with no gates."""
        workflow_id = str(uuid.uuid4())

        response = client.get(f"/api/approval-gates/workflow/{workflow_id}")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] == 0
        assert data["gates"] == []


class TestListPendingGates:
    """Test GET /api/approval-gates/pending endpoint."""

    def test_list_pending_gates_success(self, client, sample_gate_data):
        """Test successful listing of pending gates."""
        # Create a pending gate
        client.post("/api/approval-gates", json=sample_gate_data)

        response = client.get("/api/approval-gates/pending")

        assert response.status_code == 200
        data = response.json()

        assert data["total"] >= 1
        assert all(gate["status"] == "pending" for gate in data["gates"])

    def test_list_pending_gates_filtered_by_workflow(self, client, sample_gate_data):
        """Test listing pending gates filtered by workflow."""
        workflow_id = sample_gate_data["workflow_id"]

        # Create gate for specific workflow
        client.post("/api/approval-gates", json=sample_gate_data)

        # Create another gate for different workflow
        other_workflow = sample_gate_data.copy()
        other_workflow["workflow_id"] = str(uuid.uuid4())
        client.post("/api/approval-gates", json=other_workflow)

        # Filter by workflow
        response = client.get(f"/api/approval-gates/pending?workflow_id={workflow_id}")

        assert response.status_code == 200
        data = response.json()

        # All gates returned should be for the filtered workflow
        for gate in data["gates"]:
            assert gate["workflow_id"] == workflow_id


class TestApprovalGateStatusTransitions:
    """Test approval status transitions."""

    def test_pending_to_approved_transition(self, client, sample_gate_data):
        """Test transition from pending to approved."""
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Verify initial status is pending
        get_response = client.get(f"/api/approval-gates/{gate_id}")
        assert get_response.json()["status"] == "pending"

        # Approve the gate
        client.post(f"/api/approval-gates/{gate_id}/approve", json={"approver": "admin"})

        # Verify status changed to approved
        get_response = client.get(f"/api/approval-gates/{gate_id}")
        assert get_response.json()["status"] == "approved"
        assert get_response.json()["approved_at"] is not None

    def test_pending_to_rejected_transition(self, client, sample_gate_data):
        """Test transition from pending to rejected."""
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Verify initial status is pending
        get_response = client.get(f"/api/approval-gates/{gate_id}")
        assert get_response.json()["status"] == "pending"

        # Reject the gate
        client.post(f"/api/approval-gates/{gate_id}/reject", json={"approver": "admin"})

        # Verify status changed to rejected
        get_response = client.get(f"/api/approval-gates/{gate_id}")
        assert get_response.json()["status"] == "rejected"
        assert get_response.json()["rejected_at"] is not None

    def test_cannot_reject_approved_gate(self, client, sample_gate_data):
        """Test that an approved gate cannot be rejected."""
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        # Approve first
        client.post(f"/api/approval-gates/{gate_id}/approve", json={"approver": "admin"})

        # Try to reject
        response = client.post(f"/api/approval-gates/{gate_id}/reject", json={"approver": "admin"})

        assert response.status_code == 400


class TestApprovalGateResponseFormat:
    """Test response format and timestamps."""

    def test_response_has_timestamps(self, client, sample_gate_data):
        """Test that responses include timestamps."""
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        data = create_response.json()

        assert "created_at" in data
        assert "updated_at" in data
        assert data["created_at"] is not None

    def test_approval_includes_timestamp(self, client, sample_gate_data):
        """Test that approval includes timestamp."""
        create_response = client.post("/api/approval-gates", json=sample_gate_data)
        gate_id = create_response.json()["gate_id"]

        approve_response = client.post(
            f"/api/approval-gates/{gate_id}/approve",
            json={"approver": "admin"}
        )

        assert approve_response.json()["timestamp"] is not None
