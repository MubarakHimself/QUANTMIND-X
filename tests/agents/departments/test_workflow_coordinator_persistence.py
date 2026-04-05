import json
from pathlib import Path

from src.agents.departments.workflow_coordinator import DepartmentWorkflowCoordinator


class _FakeWorkflowDb:
    def get_workflow_run(self, workflow_id: str):
        if workflow_id != "wf-persisted-1":
            return None
        return {
            "id": workflow_id,
            "workflow_name": "wf1_creation",
            "status": "running",
            "created_at": "2026-04-04T10:00:00+00:00",
            "updated_at": "2026-04-04T10:05:00+00:00",
            "error_message": None,
        }

    def get_stage_results(self, workflow_run_id: str):
        if workflow_run_id != "wf-persisted-1":
            return []
        return [
            {
                "workflow_run_id": workflow_run_id,
                "stage_name": "video_ingest",
                "status": "completed",
                "started_at": "2026-04-04T10:00:00+00:00",
                "completed_at": "2026-04-04T10:02:00+00:00",
                "result_data": json.dumps({"captions": True}),
                "error_message": None,
            },
            {
                "workflow_run_id": workflow_run_id,
                "stage_name": "research",
                "status": "running",
                "started_at": "2026-04-04T10:02:00+00:00",
                "completed_at": None,
                "result_data": None,
                "error_message": None,
            },
        ]


def test_persisted_workflow_can_be_hydrated_and_paused(monkeypatch, tmp_path):
    coordinator = DepartmentWorkflowCoordinator(
        mail_db_path=str(tmp_path / "mail.db"),
        use_prefect=False,
        use_memory=False,
    )

    strategy_root = tmp_path / "strategies" / "scalping" / "playlists" / "playlist_batch"
    workflow_dir = strategy_root / "workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "manifest.json").write_text(
        json.dumps(
            {
                "workflow_id": "wf-persisted-1",
                "workflow_type": "wf1_creation",
                "strategy_id": "playlist_batch",
                "status": "running",
                "current_stage": "research",
                "retry_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (strategy_root / ".meta.json").write_text(
        json.dumps({"strategy_id": "playlist_batch", "name": "playlist_batch"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(coordinator, "_get_workflow_db", lambda: _FakeWorkflowDb())
    monkeypatch.setattr(
        "src.api.wf1_artifacts.find_strategy_root_by_workflow_id",
        lambda workflow_id: strategy_root if workflow_id == "wf-persisted-1" else None,
    )
    monkeypatch.setattr(coordinator, "_update_prefect_db_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "_persist_workflow_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(coordinator, "_write_memory", lambda *args, **kwargs: None)

    status = coordinator.get_workflow_status("wf-persisted-1")

    assert status is not None
    assert status["workflow_id"] == "wf-persisted-1"
    assert status["current_stage"] == "research"
    assert status["status"] == "running"
    assert len(status["tasks"]) == 2

    paused = coordinator.pause_workflow("wf-persisted-1")

    assert paused["workflow_id"] == "wf-persisted-1"
    assert paused["status"] == "waiting"
    assert paused["manual_pause"] is True
