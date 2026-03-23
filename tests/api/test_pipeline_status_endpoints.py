"""
Tests for Pipeline Status API Endpoints

Tests Alpha Forge 9-stage pipeline status board.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.pipeline_status_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestPipelineModels:
    """Test pipeline model structures."""

    def test_pipeline_stage_enum_values(self):
        from src.api.pipeline_status_endpoints import PipelineStage

        assert PipelineStage.VIDEO_INGEST == "VIDEO_INGEST"
        assert PipelineStage.APPROVAL == "APPROVAL"
        assert len(PipelineStage) == 9

    def test_stage_status_enum_values(self):
        from src.api.pipeline_status_endpoints import StageStatus

        assert StageStatus.WAITING == "waiting"
        assert StageStatus.RUNNING == "running"
        assert StageStatus.PASSED == "passed"
        assert StageStatus.FAILED == "failed"

    def test_approval_status_enum_values(self):
        from src.api.pipeline_status_endpoints import ApprovalStatus

        assert ApprovalStatus.PENDING_REVIEW == "pending_review"
        assert ApprovalStatus.APPROVED == "approved"
        assert ApprovalStatus.REJECTED == "rejected"

    def test_pipeline_run_structure(self):
        from src.api.pipeline_status_endpoints import (
            PipelineRun,
            PipelineStage,
            StageStatus,
            ApprovalStatus,
            PipelineStageInfo,
        )

        run = PipelineRun(
            strategy_id="strat-001",
            strategy_name="Test Strategy",
            current_stage=PipelineStage.BACKTEST,
            stage_status=StageStatus.RUNNING,
            stages=[
                PipelineStageInfo(stage=PipelineStage.VIDEO_INGEST, status=StageStatus.PASSED),
                PipelineStageInfo(stage=PipelineStage.RESEARCH, status=StageStatus.RUNNING),
            ],
            approval_status=ApprovalStatus.NONE,
            started_at="2026-03-21T10:00:00Z",
            updated_at="2026-03-21T11:00:00Z",
        )
        assert run.strategy_id == "strat-001"
        assert len(run.stages) == 2

    def test_pipeline_stages_response(self):
        from src.api.pipeline_status_endpoints import PipelineStagesResponse

        resp = PipelineStagesResponse(
            stages=["VIDEO_INGEST", "RESEARCH"],
            stage_order=["VIDEO_INGEST", "RESEARCH"],
            human_gates=["APPROVAL"],
        )
        assert "APPROVAL" in resp.human_gates


class TestPipelineEndpoints:
    """Test pipeline status REST endpoints."""

    def test_get_all_pipeline_status(self):
        client = _make_client()
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
        assert "active_count" in data

    def test_get_pipeline_status_has_sample_data(self):
        client = _make_client()
        response = client.get("/api/pipeline/status")
        data = response.json()
        # Sample data is seeded on module load
        assert data["total"] >= 3

    def test_get_pipeline_status_active_only(self):
        client = _make_client()
        response = client.get("/api/pipeline/status", params={"active_only": True})
        assert response.status_code == 200
        data = response.json()
        for run in data["runs"]:
            assert run["stage_status"] == "running"

    def test_get_specific_strategy_pipeline_status(self):
        client = _make_client()
        response = client.get("/api/pipeline/status/strat-001")
        assert response.status_code == 200
        data = response.json()
        assert data["run"]["strategy_id"] == "strat-001"

    def test_get_nonexistent_strategy_returns_404(self):
        client = _make_client()
        response = client.get("/api/pipeline/status/nonexistent-strat")
        assert response.status_code == 404

    def test_get_pipeline_stages(self):
        client = _make_client()
        response = client.get("/api/pipeline/stages")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        assert "stage_order" in data
        assert "human_gates" in data
        assert "APPROVAL" in data["human_gates"]
        assert len(data["stage_order"]) == 9

    def test_get_pending_approvals(self):
        client = _make_client()
        response = client.get("/api/pipeline/pending-approvals")
        assert response.status_code == 200
        data = response.json()
        assert "pending_count" in data
        assert "strategies" in data
        # strat-002 has PENDING_REVIEW approval status
        assert data["pending_count"] >= 1

    def test_create_pipeline_run(self):
        client = _make_client()
        response = client.post(
            "/api/pipeline/run",
            params={"strategy_id": "strat-new-100", "strategy_name": "New Test Strategy"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["strategy_id"] == "strat-new-100"

    def test_create_duplicate_run_returns_400(self):
        client = _make_client()
        # strat-001 already exists in sample data
        response = client.post(
            "/api/pipeline/run",
            params={"strategy_id": "strat-001", "strategy_name": "Duplicate"},
        )
        assert response.status_code == 400
