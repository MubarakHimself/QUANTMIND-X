"""
Tests for Pipeline Status API Endpoints

Tests Alpha Forge 9-stage pipeline status board.
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _fake_runs():
    from src.api.pipeline_status_endpoints import PipelineStage, StageStatus, ApprovalStatus

    return [
        {
            "strategy_id": "wf1-rsi-divergence",
            "strategy_name": "YouTube: RSI Divergence Strategy",
            "current_stage": PipelineStage.BACKTEST,
            "stage_status": StageStatus.RUNNING,
            "stages": [
                {"stage": PipelineStage.VIDEO_INGEST, "status": StageStatus.PASSED, "started_at": "2026-03-15T10:00:00Z", "completed_at": "2026-03-15T10:30:00Z"},
                {"stage": PipelineStage.RESEARCH, "status": StageStatus.PASSED, "started_at": "2026-03-15T10:30:00Z", "completed_at": "2026-03-15T11:00:00Z"},
                {"stage": PipelineStage.TRD, "status": StageStatus.PASSED, "started_at": "2026-03-15T11:00:00Z", "completed_at": "2026-03-15T11:30:00Z"},
                {"stage": PipelineStage.DEVELOPMENT, "status": StageStatus.PASSED, "started_at": "2026-03-15T11:30:00Z", "completed_at": "2026-03-15T12:00:00Z"},
                {"stage": PipelineStage.COMPILE, "status": StageStatus.PASSED, "started_at": "2026-03-15T12:00:00Z", "completed_at": "2026-03-15T12:15:00Z"},
                {"stage": PipelineStage.BACKTEST, "status": StageStatus.RUNNING, "started_at": "2026-03-15T12:15:00Z", "completed_at": None},
                {"stage": PipelineStage.VALIDATION, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.EA_LIFECYCLE, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
                {"stage": PipelineStage.APPROVAL, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
            ],
            "approval_status": ApprovalStatus.NONE,
            "started_at": "2026-03-15T10:00:00Z",
            "updated_at": "2026-03-15T12:15:00Z",
            "metadata": {"workflow_id": "wf1-rsi-divergence"},
        },
        {
            "strategy_id": "wf1-ma-crossover",
            "strategy_name": "YouTube: Moving Average Crossover",
            "current_stage": PipelineStage.APPROVAL,
            "stage_status": StageStatus.WAITING,
            "stages": [
                {"stage": PipelineStage.VIDEO_INGEST, "status": StageStatus.PASSED, "started_at": "2026-03-14T09:00:00Z", "completed_at": "2026-03-14T09:30:00Z"},
                {"stage": PipelineStage.RESEARCH, "status": StageStatus.PASSED, "started_at": "2026-03-14T09:30:00Z", "completed_at": "2026-03-14T10:00:00Z"},
                {"stage": PipelineStage.TRD, "status": StageStatus.PASSED, "started_at": "2026-03-14T10:00:00Z", "completed_at": "2026-03-14T10:30:00Z"},
                {"stage": PipelineStage.DEVELOPMENT, "status": StageStatus.PASSED, "started_at": "2026-03-14T10:30:00Z", "completed_at": "2026-03-14T11:00:00Z"},
                {"stage": PipelineStage.COMPILE, "status": StageStatus.PASSED, "started_at": "2026-03-14T11:00:00Z", "completed_at": "2026-03-14T11:15:00Z"},
                {"stage": PipelineStage.BACKTEST, "status": StageStatus.PASSED, "started_at": "2026-03-14T11:15:00Z", "completed_at": "2026-03-14T12:00:00Z"},
                {"stage": PipelineStage.VALIDATION, "status": StageStatus.PASSED, "started_at": "2026-03-14T12:00:00Z", "completed_at": "2026-03-14T12:30:00Z"},
                {"stage": PipelineStage.EA_LIFECYCLE, "status": StageStatus.PASSED, "started_at": "2026-03-14T12:30:00Z", "completed_at": "2026-03-14T13:00:00Z"},
                {"stage": PipelineStage.APPROVAL, "status": StageStatus.WAITING, "started_at": None, "completed_at": None},
            ],
            "approval_status": ApprovalStatus.PENDING_REVIEW,
            "started_at": "2026-03-14T09:00:00Z",
            "updated_at": "2026-03-14T13:00:00Z",
            "metadata": {"workflow_id": "wf1-ma-crossover"},
        },
    ]


def _make_client(monkeypatch):
    from src.api import pipeline_status_endpoints as endpoints
    from src.api.pipeline_status_endpoints import router

    async def _load_runs():
        return _fake_runs()

    monkeypatch.setattr(endpoints, "_load_alpha_forge_runs", _load_runs)

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

    def test_get_all_pipeline_status(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert "total" in data
        assert "active_count" in data

    def test_get_pipeline_status_uses_canonical_loader(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/status")
        data = response.json()
        assert data["total"] == 2
        assert {run["strategy_id"] for run in data["runs"]} == {
            "wf1-rsi-divergence",
            "wf1-ma-crossover",
        }

    def test_get_pipeline_status_active_only(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/status", params={"active_only": True})
        assert response.status_code == 200
        data = response.json()
        for run in data["runs"]:
            assert run["stage_status"] == "running"

    def test_get_specific_strategy_pipeline_status(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/status/wf1-rsi-divergence")
        assert response.status_code == 200
        data = response.json()
        assert data["run"]["strategy_id"] == "wf1-rsi-divergence"

    def test_get_nonexistent_strategy_returns_404(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/status/nonexistent-strat")
        assert response.status_code == 404

    def test_get_pipeline_stages(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/stages")
        assert response.status_code == 200
        data = response.json()
        assert "stages" in data
        assert "stage_order" in data
        assert "human_gates" in data
        assert "APPROVAL" in data["human_gates"]
        assert len(data["stage_order"]) == 9

    def test_get_pending_approvals(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.get("/api/pipeline/pending-approvals")
        assert response.status_code == 200
        data = response.json()
        assert "pending_count" in data
        assert "strategies" in data
        assert data["pending_count"] >= 1

    def test_create_pipeline_run_is_disabled(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.post(
            "/api/pipeline/run",
            params={"strategy_id": "strat-new-100", "strategy_name": "New Test Strategy"},
        )
        assert response.status_code == 501

    def test_create_duplicate_run_also_stays_disabled(self, monkeypatch):
        client = _make_client(monkeypatch)
        response = client.post(
            "/api/pipeline/run",
            params={"strategy_id": "wf1-rsi-divergence", "strategy_name": "Duplicate"},
        )
        assert response.status_code == 501
