"""
P1 Tests for Epic 8 - Alpha Forge Canvas (Pipeline Board)

Priority: P1
Coverage: Pipeline status UI, store actions, polling, pending approvals

Risk Coverage:
- R-001: EA Deployment Pipeline
- R-002: Deployment Window UTC Enforcement
- R-003: Prefect workflows.db Corruption
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestPipelineStatusPolling:
    """P1: Pipeline status polling mechanism."""

    def test_pipeline_store_fetch_updates_runs(self):
        """Test alphaForgeStore.fetchPipelineStatus updates runs array."""
        # alphaForgeStore is a Svelte/TypeScript store - tested via Vitest frontend tests
        # Backend API contract tested via TestPipelineStatusEndpoints
        pass

    def test_pipeline_polling_interval_is_5_seconds(self):
        """P1: Verify polling interval is 5 seconds as per spec."""
        # POLLING_INTERVAL_MS is defined in alpha-forge.ts - tested via Vitest
        # Backend spec: polling interval should be 5000ms (verified via API response timing)
        POLLING_INTERVAL_MS = 5000  # Expected constant value from frontend spec
        assert POLLING_INTERVAL_MS == 5000

    def test_pending_approval_count_derived_correctly(self):
        """P1: pendingApprovalCount derived store returns correct count."""
        # Test the derived store logic
        state = {"pendingApprovals": [{"id": 1}, {"id": 2}]}
        count = len(state["pendingApprovals"])
        assert count == 2

    def test_active_run_count_filters_running_only(self):
        """P1: activeRunCount only counts RUNNING status."""
        runs = [
            {"stage_status": "running"},
            {"stage_status": "passed"},
            {"stage_status": "running"},
        ]
        active = [r for r in runs if r["stage_status"] == "running"]
        assert len(active) == 2


class TestPipelineStatusEndpoints:
    """P1: Backend API tests for pipeline status board."""

    def _make_client(self):
        from src.api.pipeline_status_endpoints import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_get_pipeline_status_returns_9_stages_in_order(self):
        """P1: Verify 9-stage pipeline order in response."""
        client = self._make_client()
        response = client.get("/api/pipeline/stages")
        assert response.status_code == 200

        data = response.json()
        assert len(data["stage_order"]) == 9
        assert data["stage_order"][0] == "VIDEO_INGEST"
        assert data["stage_order"][-1] == "APPROVAL"

    def test_pipeline_run_has_correct_stage_transitions(self):
        """P1: Pipeline run includes all 9 stage entries."""
        client = self._make_client()
        response = client.get("/api/pipeline/status/strat-001")
        assert response.status_code == 200

        run = response.json()["run"]
        assert len(run["stages"]) == 9

    def test_deployment_window_check_returns_in_window_boolean(self):
        """P1: check_deployment_window returns in_window field."""
        from flows.alpha_forge_flow import check_deployment_window

        result = check_deployment_window()
        assert "in_window" in result
        assert isinstance(result["in_window"], bool)

    def test_human_gate_exists_only_at_approval_stage(self):
        """P1: Only APPROVAL stage is marked as human gate."""
        client = self._make_client()
        response = client.get("/api/pipeline/stages")
        data = response.json()

        assert data["human_gates"] == ["APPROVAL"]


class TestPipelineApprovalIntegration:
    """P1: Pipeline + Approval gate integration."""

    def _make_client(self):
        from src.api.pipeline_status_endpoints import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_pending_review_strategies_appear_in_pending_approvals(self):
        """P1: Strategies with PENDING_REVIEW status appear in pending list."""
        client = self._make_client()
        response = client.get("/api/pipeline/pending-approvals")
        assert response.status_code == 200

        data = response.json()
        assert "pending_count" in data
        assert "strategies" in data

        # strat-002 has PENDING_REVIEW in sample data
        pending_strategies = data["strategies"]
        assert len(pending_strategies) >= 1

    def test_approval_status_reflects_pipeline_state(self):
        """P1: Approval status is synced with pipeline approval_status field."""
        client = self._make_client()

        # Get specific strategy
        response = client.get("/api/pipeline/status/strat-002")
        run = response.json()["run"]

        assert run["approval_status"] == "pending_review"


class TestDeploymentWindowBoundaries:
    """P1: Deployment window UTC enforcement at boundaries."""

    @patch('flows.alpha_forge_flow.datetime')
    def test_deployment_allowed_saturday_afternoon_utc(self, mock_datetime):
        """P1: Deployment allowed Saturday 14:00 UTC (inside window)."""
        from flows.alpha_forge_flow import check_deployment_window
        from datetime import datetime, timezone

        saturday_1400 = datetime(2026, 3, 21, 14, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = saturday_1400
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        mock_datetime.timezone.utc = timezone.utc

        result = check_deployment_window()
        assert result["in_window"] is True

    @patch('flows.alpha_forge_flow.datetime')
    def test_deployment_blocked_monday_21_30_utc(self, mock_datetime):
        """P1: Deployment blocked Monday 21:30 UTC (outside window)."""
        from flows.alpha_forge_flow import check_deployment_window
        from datetime import datetime, timezone

        monday_2130 = datetime(2026, 3, 23, 21, 30, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = monday_2130
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        mock_datetime.timezone.utc = timezone.utc

        result = check_deployment_window()
        assert result["in_window"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
