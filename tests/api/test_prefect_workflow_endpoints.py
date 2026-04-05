import asyncio
import json

import pytest
from fastapi import HTTPException


class _FakeWorkflowDb:
    def __init__(self):
        self._runs = [
            {
                "id": "wf-real-1",
                "workflow_name": "wf1_creation",
                "status": "running",
                "created_at": "2026-03-31T12:00:00+00:00",
                "updated_at": "2026-03-31T12:05:00+00:00",
                "error_message": None,
            },
            {
                "id": "wf-real-2",
                "workflow_name": "wf2_enhancement",
                "status": "completed",
                "created_at": "2026-03-31T10:00:00+00:00",
                "updated_at": "2026-03-31T10:10:00+00:00",
                "error_message": None,
            },
        ]
        self._stages = {
            "wf-real-1": [
                {"workflow_run_id": "wf-real-1", "stage_name": "video_ingest", "status": "completed"},
                {"workflow_run_id": "wf-real-1", "stage_name": "research", "status": "running"},
            ],
            "wf-real-2": [
                {"workflow_run_id": "wf-real-2", "stage_name": "ea_library_scan", "status": "running"},
                {"workflow_run_id": "wf-real-2", "stage_name": "full_backtest_matrix", "status": "running"},
            ],
        }

    def list_workflow_runs(self, limit: int = 100):
        return self._runs[:limit]

    def get_stage_results(self, workflow_run_id: str):
        return self._stages.get(workflow_run_id, [])


@pytest.fixture
def endpoints(monkeypatch):
    from src.api import prefect_workflow_endpoints as endpoints

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _FakeWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)
    monkeypatch.setattr(endpoints, "_load_live_coordinator_workflows", lambda: {})
    monkeypatch.setattr(endpoints, "iter_workflow_manifests", lambda: [])
    monkeypatch.setattr(endpoints, "find_strategy_root_by_workflow_id", lambda _workflow_id: None)
    return endpoints


def test_list_workflows_reads_real_workflow_db(endpoints):
    data = asyncio.run(endpoints.list_workflows())

    assert data["total"] == 2
    assert {workflow["id"] for workflow in data["workflows"]} == {"wf-real-1", "wf-real-2"}
    assert len(data["by_state"]["RUNNING"]) == 1
    assert len(data["by_state"]["DONE"]) == 1


def test_get_workflow_builds_task_graph_from_stage_results(endpoints):
    workflow = asyncio.run(endpoints.get_workflow("wf-real-1"))

    assert workflow["department"] == "Research"
    assert workflow["state"] == "RUNNING"
    assert workflow["completed_steps"] == 1
    assert workflow["total_steps"] == 2
    assert workflow["next_step"] == "Research"
    assert workflow["can_pause"] is True
    assert workflow["can_resume"] is False
    assert [task["name"] for task in workflow["tasks"]] == ["Video Ingest", "Research"]
    assert workflow["dependencies"] == [
        {
            "from": workflow["tasks"][0]["id"],
            "to": workflow["tasks"][1]["id"],
        }
    ]


def test_completed_workflow_uses_real_db_not_mock_progress(endpoints):
    workflow = asyncio.run(endpoints.get_workflow("wf-real-2"))

    assert workflow["state"] == "DONE"
    assert workflow["completed_steps"] == workflow["total_steps"] == 2
    assert workflow["next_step"] == "Completed"


def test_get_workflow_missing_returns_404(endpoints):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(endpoints.get_workflow("missing"))

    assert exc_info.value.status_code == 404


def test_cancel_and_resume_are_honestly_not_found_without_live_coordinator_state(endpoints):
    with pytest.raises(HTTPException) as cancel_exc:
        asyncio.run(endpoints.cancel_workflow("wf-real-1"))
    with pytest.raises(HTTPException) as resume_exc:
        asyncio.run(endpoints.resume_workflow("wf-real-1"))

    assert cancel_exc.value.status_code == 404
    assert resume_exc.value.status_code == 404


def test_pause_resume_and_retry_use_live_coordinator_controls(monkeypatch):
    from src.api import prefect_workflow_endpoints as endpoints

    class _Coordinator:
        def get_workflow_status(self, workflow_id: str):
            if workflow_id == "wf-running":
                return {"workflow_id": workflow_id, "status": "running", "metadata": {}}
            if workflow_id == "wf-paused":
                return {
                    "workflow_id": workflow_id,
                    "status": "waiting",
                    "metadata": {"manual_pause": True, "pause_reason": "Paused by operator"},
                }
            if workflow_id == "wf-cancelled":
                return {"workflow_id": workflow_id, "status": "cancelled", "metadata": {}}
            return None

        def pause_workflow(self, workflow_id: str):
            if workflow_id != "wf-running":
                return {"error": "Workflow not found"}
            return {
                "workflow_id": workflow_id,
                "status": "waiting",
                "current_stage": "research",
                "waiting_reason": "Paused by operator",
                "manual_pause": True,
            }

        def resume_workflow(self, workflow_id: str):
            if workflow_id != "wf-paused":
                return {"error": "Workflow is waiting on a human approval gate and cannot be resumed directly"}
            return {"workflow_id": workflow_id, "status": "running", "current_stage": "research"}

        def retry_workflow(self, workflow_id: str):
            if workflow_id != "wf-cancelled":
                return {"error": "Workflow not found"}
            return {
                "workflow_id": workflow_id,
                "status": "running",
                "current_stage": "development",
                "retry_count": 1,
            }

    monkeypatch.setattr(
        "src.agents.departments.workflow_coordinator.get_workflow_coordinator",
        lambda: _Coordinator(),
    )

    pause_response = asyncio.run(endpoints.pause_workflow("wf-running"))
    assert pause_response["manual_pause"] is True

    resume_response = asyncio.run(endpoints.resume_workflow("wf-paused"))
    assert resume_response["status"] == "running"

    retry_response = asyncio.run(endpoints.retry_workflow("wf-cancelled"))
    assert retry_response["retry_count"] == 1


def test_list_workflows_includes_durable_wf1_manifest_state(monkeypatch, tmp_path):
    from src.api import prefect_workflow_endpoints as endpoints

    class _EmptyWorkflowDb:
        def list_workflow_runs(self, limit: int = 100):
            return []

        def get_stage_results(self, workflow_run_id: str):
            return []

    strategy_root = tmp_path / "strategies" / "scalping" / "playlists" / "playlist_batch"
    (strategy_root / "workflow").mkdir(parents=True)
    (strategy_root / "research" / "trd").mkdir(parents=True)
    (strategy_root / ".meta.json").write_text(
        json.dumps(
            {
                "name": "playlist_batch",
                "created_at": "2026-04-04T12:00:00+00:00",
                "updated_at": "2026-04-04T12:05:00+00:00",
                "workflow_id": "wf1_manifest_live",
                "strategy_id": "playlist_batch",
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "workflow_id": "wf1_manifest_live",
                "workflow_type": "wf1_creation",
                "strategy_id": "playlist_batch",
                "current_stage": "research",
                "status": "running",
                "updated_at": "2026-04-04T12:05:00+00:00",
                "waiting_reason": None,
                "blocking_error": None,
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "research" / "trd" / "playlist_batch.trd.md").write_text(
        "# TRD", encoding="utf-8"
    )

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _EmptyWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)
    monkeypatch.setattr(endpoints, "_load_live_coordinator_workflows", lambda: {})
    monkeypatch.setattr(
        endpoints,
        "iter_workflow_manifests",
        lambda: [(strategy_root, json.loads((strategy_root / "workflow" / "manifest.json").read_text(encoding="utf-8")))],
    )

    data = asyncio.run(endpoints.list_workflows())

    assert data["total"] == 1
    workflow = data["workflows"][0]
    assert workflow["id"] == "wf1_manifest_live"
    assert workflow["state"] == "RUNNING"
    assert workflow["current_stage"] == "Research"
    assert workflow["department"] == "Research"
    assert workflow["strategy_id"] == "playlist_batch"
    assert workflow["latest_artifact"]["name"] == "playlist_batch.trd.md"


def test_list_workflows_skips_empty_wf1_db_placeholders(monkeypatch):
    from src.api import prefect_workflow_endpoints as endpoints

    class _PlaceholderWorkflowDb:
        def list_workflow_runs(self, limit: int = 100):
            return [
                {
                    "id": "wf_placeholder",
                    "workflow_name": "wf1_creation",
                    "status": "pending",
                    "created_at": "2026-04-04T13:00:00+00:00",
                    "updated_at": "2026-04-04T13:00:01+00:00",
                    "error_message": None,
                }
            ]

        def get_stage_results(self, workflow_run_id: str):
            return []

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _PlaceholderWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)
    monkeypatch.setattr(endpoints, "_load_live_coordinator_workflows", lambda: {})
    monkeypatch.setattr(endpoints, "iter_workflow_manifests", lambda: [])
    monkeypatch.setattr(endpoints, "find_strategy_root_by_workflow_id", lambda _workflow_id: None)

    data = asyncio.run(endpoints.list_workflows())

    assert data["total"] == 0
    assert data["workflows"] == []


def test_list_workflows_skips_manifest_placeholders_without_source_or_artifacts(monkeypatch, tmp_path):
    from src.api import prefect_workflow_endpoints as endpoints

    class _EmptyWorkflowDb:
        def list_workflow_runs(self, limit: int = 100):
            return []

        def get_stage_results(self, workflow_run_id: str):
            return []

    strategy_root = tmp_path / "strategies" / "scalping" / "single-videos" / "probe"
    (strategy_root / "source").mkdir(parents=True)
    (strategy_root / "workflow").mkdir(parents=True)
    (strategy_root / ".meta.json").write_text(
        json.dumps(
            {
                "name": "probe",
                "created_at": "2026-04-04T12:00:00+00:00",
                "updated_at": "2026-04-04T12:05:00+00:00",
                "workflow_id": "wf1_manifest_placeholder",
                "strategy_id": "probe",
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "workflow_id": "wf1_manifest_placeholder",
                "workflow_type": "wf1_creation",
                "strategy_id": "probe",
                "current_stage": "video_ingest",
                "status": "pending",
                "updated_at": "2026-04-04T12:05:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "source" / "request.json").write_text(
        json.dumps(
            {
                "source_url": None,
                "workflow_id": "wf1_manifest_placeholder",
                "strategy_id": "probe",
            }
        ),
        encoding="utf-8",
    )

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _EmptyWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)
    monkeypatch.setattr(endpoints, "_load_live_coordinator_workflows", lambda: {})
    monkeypatch.setattr(
        endpoints,
        "iter_workflow_manifests",
        lambda: [(strategy_root, json.loads((strategy_root / "workflow" / "manifest.json").read_text(encoding="utf-8")))],
    )

    data = asyncio.run(endpoints.list_workflows())

    assert data["total"] == 0
    assert data["workflows"] == []


def test_list_workflows_skips_humanized_video_ingest_manifest_placeholders(monkeypatch, tmp_path):
    from src.api import prefect_workflow_endpoints as endpoints

    class _EmptyWorkflowDb:
        def list_workflow_runs(self, limit: int = 100):
            return []

        def get_stage_results(self, workflow_run_id: str):
            return []

    strategy_root = tmp_path / "strategies" / "scalping" / "single-videos" / "video_ingest"
    (strategy_root / "source").mkdir(parents=True)
    (strategy_root / "workflow").mkdir(parents=True)
    (strategy_root / ".meta.json").write_text(
        json.dumps(
            {
                "name": "video_ingest",
                "created_at": "2026-04-04T12:00:00+00:00",
                "updated_at": "2026-04-04T12:05:00+00:00",
                "workflow_id": "wf1_manifest_humanized_placeholder",
                "strategy_id": "video_ingest",
                "status": "pending",
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "workflow" / "manifest.json").write_text(
        json.dumps(
            {
                "workflow_id": "wf1_manifest_humanized_placeholder",
                "workflow_type": "wf1_creation",
                "strategy_id": "video_ingest",
                "current_stage": "Video Ingest",
                "status": "pending",
                "updated_at": "2026-04-04T12:05:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / "source" / "request.json").write_text(
        json.dumps(
            {
                "source_url": None,
                "workflow_id": "wf1_manifest_humanized_placeholder",
                "strategy_id": "video_ingest",
            }
        ),
        encoding="utf-8",
    )

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _EmptyWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)
    monkeypatch.setattr(endpoints, "_load_live_coordinator_workflows", lambda: {})
    monkeypatch.setattr(
        endpoints,
        "iter_workflow_manifests",
        lambda: [(strategy_root, json.loads((strategy_root / "workflow" / "manifest.json").read_text(encoding="utf-8")))],
    )

    data = asyncio.run(endpoints.list_workflows())

    assert data["total"] == 0
    assert data["workflows"] == []
