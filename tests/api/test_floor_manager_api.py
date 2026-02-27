# tests/api/test_floor_manager_api.py
"""
Tests for Floor Manager API Endpoints

Phase 3: Copilot -> Floor Manager Delegation UI
Tests for the delegation API endpoint that allows Copilot to delegate tasks to departments.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile
import os


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the trading floor router."""
    from src.api.trading_floor_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the test app."""
    return TestClient(test_app)


class TestFloorManagerDelegationEndpoint:
    """Test POST /api/trading-floor/delegate endpoint."""

    def test_delegate_endpoint_exists(self, client):
        """Should have a delegate endpoint that accepts POST requests."""
        response = client.post("/api/trading-floor/delegate", json={})

        # Should not return 404
        assert response.status_code != 404

    def test_delegate_accepts_required_fields(self, client):
        """Should accept from_department, task, and suggested_department fields."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Analyze EURUSD for opportunities",
            "suggested_department": "analysis",
        })

        # Should accept the request
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"

    def test_delegate_requires_from_department(self, client):
        """Should return error if from_department is missing."""
        response = client.post("/api/trading-floor/delegate", json={
            "task": "Analyze EURUSD",
        })

        assert response.status_code == 422  # Validation error

    def test_delegate_requires_task(self, client):
        """Should return error if task is missing."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
        })

        assert response.status_code == 422  # Validation error

    def test_delegate_returns_dispatch_result(self, client):
        """Should return dispatch result with message_id and routing info."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Backtest the RSI strategy",
            "suggested_department": "research",
        })

        assert response.status_code == 200

        data = response.json()
        assert "dispatch" in data
        assert "message_id" in data["dispatch"]
        assert data["dispatch"]["from_department"] == "copilot"
        assert data["dispatch"]["to_department"] == "research"
        assert data["dispatch"]["status"] == "dispatched"

    def test_delegate_auto_classifies_when_no_suggestion(self, client):
        """Should auto-classify task when suggested_department is not provided."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Calculate position size for EURUSD",
        })

        assert response.status_code == 200

        data = response.json()
        # Should classify to risk based on keywords
        assert data["dispatch"]["to_department"] == "risk"

    def test_delegate_with_context(self, client):
        """Should accept and include optional context in dispatch."""
        context = {
            "symbol": "GBPUSD",
            "timeframe": "H1",
        }

        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Analyze market",
            "suggested_department": "analysis",
            "context": context,
        })

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"

    def test_delegate_invalid_suggested_department_falls_back(self, client):
        """Should fall back to auto-classification for invalid department."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Execute buy order",
            "suggested_department": "invalid_dept",
        })

        assert response.status_code == 200

        data = response.json()
        # Should classify to execution
        assert data["dispatch"]["to_department"] == "execution"


class TestFloorManagerAPIIntegration:
    """Integration tests for Floor Manager API."""

    def test_delegate_creates_mail_in_department_inbox(self, client):
        """Delegation should create mail message in target department inbox."""
        response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Research alpha factors",
        })

        assert response.status_code == 200

        # Check that mail was created
        mail_response = client.get("/api/trading-floor/mail/research")
        assert mail_response.status_code == 200

        mail_data = mail_response.json()
        assert len(mail_data["messages"]) > 0

        # Find our message
        our_message = next(
            (m for m in mail_data["messages"] if m["from"] == "copilot"),
            None
        )
        assert our_message is not None
        assert "alpha" in our_message["body"]

    def test_delegate_workflow_from_copilot_perspective(self, client):
        """Simulate complete delegation workflow from Copilot to department."""
        # Step 1: Copilot sends delegation request
        delegation_response = client.post("/api/trading-floor/delegate", json={
            "from_department": "copilot",
            "task": "Check drawdown limits",
            "suggested_department": None,  # Let Floor Manager decide
        })

        assert delegation_response.status_code == 200

        delegation_data = delegation_response.json()
        assert delegation_data["status"] == "success"
        message_id = delegation_data["dispatch"]["message_id"]

        # Step 2: Verify mail was sent to correct department
        target_dept = delegation_data["dispatch"]["to_department"]
        mail_response = client.get(f"/api/trading-floor/mail/{target_dept}")

        assert mail_response.status_code == 200
        mail_data = mail_response.json()
        assert len(mail_data["messages"]) > 0

        # Step 3: Verify message content
        our_message = next(
            (m for m in mail_data["messages"] if m["id"] == message_id),
            None
        )
        assert our_message is not None
        assert our_message["from"] == "copilot"
        assert our_message["to"] == target_dept
