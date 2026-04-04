from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest


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
def app(monkeypatch):
    from src.api import prefect_workflow_endpoints as endpoints

    async def _no_prefect():
        return {}

    monkeypatch.setattr(
        "flows.database.get_workflow_database",
        lambda: _FakeWorkflowDb(),
    )
    monkeypatch.setattr(endpoints, "_load_prefect_deployments", _no_prefect)

    app = FastAPI()
    app.include_router(endpoints.router)
    return app


@pytest.mark.asyncio
async def test_list_workflows_reads_real_workflow_db(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/prefect/workflows")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert {workflow["id"] for workflow in data["workflows"]} == {"wf-real-1", "wf-real-2"}
    assert len(data["by_state"]["RUNNING"]) == 1
    assert len(data["by_state"]["DONE"]) == 1


@pytest.mark.asyncio
async def test_get_workflow_builds_task_graph_from_stage_results(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/prefect/workflows/wf-real-1")

    assert response.status_code == 200
    workflow = response.json()
    assert workflow["department"] == "Research"
    assert workflow["state"] == "RUNNING"
    assert workflow["completed_steps"] == 1
    assert workflow["total_steps"] == 2
    assert workflow["next_step"] == "Research"
    assert [task["name"] for task in workflow["tasks"]] == ["Video Ingest", "Research"]
    assert workflow["dependencies"] == [
        {
            "from": workflow["tasks"][0]["id"],
            "to": workflow["tasks"][1]["id"],
        }
    ]


@pytest.mark.asyncio
async def test_completed_workflow_uses_real_db_not_mock_progress(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/prefect/workflows/wf-real-2")

    assert response.status_code == 200
    workflow = response.json()
    assert workflow["state"] == "DONE"
    assert workflow["completed_steps"] == workflow["total_steps"] == 2
    assert workflow["next_step"] == "Completed"


@pytest.mark.asyncio
async def test_get_workflow_missing_returns_404(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/prefect/workflows/missing")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cancel_and_resume_are_honestly_unwired(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        cancel_response = await client.post("/api/prefect/workflows/wf-real-1/cancel")
        resume_response = await client.post("/api/prefect/workflows/wf-real-1/resume")

    assert cancel_response.status_code == 501
    assert resume_response.status_code == 501
