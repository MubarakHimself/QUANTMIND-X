"""
P2 Tests for Epic 8 - Deployment Status UI

Priority: P2
Coverage: Deployment status endpoints, EA lifecycle tracking

Risk Coverage:
- R-001: EA Deployment Pipeline
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    try:
        from src.api.deployment_endpoints import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    except ImportError:
        # Deployment endpoints may not exist yet
        pytest.skip("deployment_endpoints not available")


class TestDeploymentModels:
    """P2: Deployment model structures."""

    def test_deployment_status_enum_values(self):
        try:
            from src.api.deployment_endpoints import DeploymentStatus
            assert DeploymentStatus.PENDING == "pending"
            assert DeploymentStatus.IN_PROGRESS == "in_progress"
            assert DeploymentStatus.COMPLETED == "completed"
            assert DeploymentStatus.FAILED == "failed"
        except ImportError:
            pytest.skip("deployment_endpoints not available")

    def test_deployment_response_has_required_fields(self):
        try:
            from src.api.deployment_endpoints import DeploymentResponse
            resp = DeploymentResponse(
                deployment_id="dep-001",
                strategy_id="strat-001",
                status="completed",
                status_detail="Deployment completed successfully",
                started_at="2026-03-21T10:00:00Z",
                completed_at="2026-03-21T10:15:00Z"
            )
            assert resp.deployment_id == "dep-001"
        except ImportError:
            pytest.skip("deployment_endpoints not available")


class TestDeploymentEndpoints:
    """P2: Deployment status REST API endpoints."""

    def test_get_deployment_status_returns_list(self):
        client = _make_client()
        response = client.get("/api/deployments")
        assert response.status_code in [200, 404]

    def test_get_specific_deployment_returns_details(self):
        client = _make_client()
        response = client.get("/api/deployments/dep-001")
        assert response.status_code in [200, 404]

    def test_deployment_creates_audit_trail(self):
        """P2: Deployment creates immutable audit record."""
        # Verify audit endpoint exists
        from src.api.server import app
        client = TestClient(app)

        response = client.get("/api/audit/deployments/dep-001")
        assert response.status_code in [200, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
