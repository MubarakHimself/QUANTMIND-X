from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.departments.workflow_coordinator import DepartmentWorkflowCoordinator
from src.agents.departments.workflow_models import WorkflowType


def _make_client(monkeypatch, tmp_path):
    from src.api.workflow_endpoints import router

    coordinator = DepartmentWorkflowCoordinator(
        mail_db_path=str(tmp_path / "department_mail.db")
    )
    monkeypatch.setattr("src.api.workflow_endpoints._get_coordinator", lambda: coordinator)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app), coordinator


def test_legacy_video_ingest_endpoint_starts_canonical_wf1(monkeypatch, tmp_path):
    client, coordinator = _make_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/workflows/video-ingest-to-ea",
        json={
            "video_ingest_content": "{\"title\":\"London scalper\"}",
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    workflow_id = data["workflow_id"]

    workflow = coordinator.get_workflow_status(workflow_id)
    assert workflow is not None
    assert workflow["workflow_type"] == "wf1_creation"
    assert workflow["current_stage"] == "video_ingest"
    assert workflow["metadata"]["entry_stage"] == "video_ingest"
    assert workflow["metadata"]["initial_payload"]["video_ingest_content"] == "{\"title\":\"London scalper\"}"
    assert workflow["tasks"][0]["to_dept"] == "research"

    serialized = client.get(f"/api/workflows/{workflow_id}")
    assert serialized.status_code == 200
    body = serialized.json()
    assert body["workflow_type"] == "video_ingest_to_ea"
    assert body["steps"][0]["agent_type"] == "research"

    coordinator.close()


def test_legacy_trd_endpoint_starts_canonical_wf1_at_development(monkeypatch, tmp_path):
    client, coordinator = _make_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/workflows/trd-to-ea",
        json={
            "trd_content": "# TRD\n\nMean reversion scalp",
            "metadata": {"source": "unit-test"},
        },
    )

    assert response.status_code == 200
    workflow_id = response.json()["workflow_id"]

    workflow = coordinator.get_workflow_status(workflow_id)
    assert workflow is not None
    assert workflow["workflow_type"] == "wf1_creation"
    assert workflow["current_stage"] == "development"
    assert workflow["metadata"]["entry_stage"] == "development"
    assert workflow["tasks"][0]["to_dept"] == "development"

    serialized = client.get(f"/api/workflows/{workflow_id}")
    assert serialized.status_code == 200
    body = serialized.json()
    assert body["workflow_type"] == "trd_to_ea"
    assert body["steps"][0]["name"] == "development"

    coordinator.close()


def test_list_workflows_uses_canonical_coordinator_state(monkeypatch, tmp_path):
    client, coordinator = _make_client(monkeypatch, tmp_path)

    workflow_id = coordinator.start_workflow(
        workflow_type=WorkflowType.WF1_CREATION,
        initial_payload={"video_ingest_content": "{}"},
    )

    response = client.get("/api/workflows")
    assert response.status_code == 200
    data = response.json()
    assert data["workflows"]
    assert any(workflow["workflow_id"] == workflow_id for workflow in data["workflows"])

    coordinator.close()
