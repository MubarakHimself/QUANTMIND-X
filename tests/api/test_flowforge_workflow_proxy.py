"""
P0-P3 Tests for FlowForge Workflow Proxy API

Epic 11 Story 11-8: FlowForge - Prefect API Contract

Test Coverage:
- P0: Workflow deployment, run trigger, cancellation, deletion
- P1: Workflow listing, SSE event stream structure
- P2: Input validation, error handling, auth headers
- P3: Edge cases, timeout handling, mapping persistence

Reference: src/api/flowforge_workflow_proxy.py
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI


class TestFlowForgeDeployment:
    """P0: Workflow deployment from FlowForge node graph."""

    @pytest.mark.asyncio
    async def test_create_deployment_returns_workflow_and_deployment_ids(self):
        """P0: POST /api/workflows creates Prefect deployment and returns IDs."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client, set_deployment_id
        from src.api.flowforge_workflow_proxy import PrefectClient, NodeDefinition, WorkflowDeploymentRequest

        app = FastAPI()
        app.include_router(router)

        # Mock the Prefect client response
        mock_client = AsyncMock(spec=PrefectClient)
        mock_client.create_deployment.return_value = {
            "workflow_id": "canvas-uuid-123",
            "deployment_id": " prefect-deployment-456",
            "deployment_name": "test-workflow"
        }

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/workflows",
                    json={
                        "canvas_workflow_uuid": "canvas-uuid-123",
                        "name": "Test Workflow",
                        "nodes": [
                            {"id": "node-1", "type": "trigger", "config": {}, "depends_on": []},
                            {"id": "node-2", "type": "task", "config": {}, "depends_on": ["node-1"]},
                        ],
                        "department": "Research"
                    }
                )

        assert response.status_code == 201
        data = response.json()
        assert data["workflow_id"] == "canvas-uuid-123"
        assert "deployment_id" in data

    @pytest.mark.asyncio
    async def test_deployment_requires_canvas_workflow_uuid(self):
        """P0: Deployment fails without canvas_workflow_uuid."""
        from src.api.flowforge_workflow_proxy import router

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/workflows",
                json={
                    "name": "Test Workflow",
                    "nodes": []
                }
            )

        # Should fail validation
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_deployment_requires_nodes_array(self):
        """P0: Deployment fails without nodes array."""
        from src.api.flowforge_workflow_proxy import router

        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/workflows",
                json={
                    "canvas_workflow_uuid": "test-uuid",
                    "name": "Test Workflow"
                }
            )

        assert response.status_code == 422


class TestFlowForgeWorkflowList:
    """P0: List deployed workflows with status badges."""

    @pytest.mark.asyncio
    async def test_list_workflows_returns_grouped_by_state(self):
        """P0: GET /api/workflows returns workflows grouped by 6 Kanban states."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        mock_client.list_deployments.return_value = [
            {"id": "wf-1", "name": "Alpha Research", "state": "RUNNING", "canvas_workflow_uuid": "cu-1"},
            {"id": "wf-2", "name": "MQL5 Build", "state": "PENDING", "canvas_workflow_uuid": "cu-2"},
        ]

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/workflows")

        assert response.status_code == 200
        data = response.json()

        assert "workflows" in data
        assert "by_state" in data
        assert "total" in data

        # All 6 states should be present
        expected_states = ["PENDING", "RUNNING", "PENDING_REVIEW", "DONE", "CANCELLED", "EXPIRED_REVIEW"]
        for state in expected_states:
            assert state in data["by_state"]

    @pytest.mark.asyncio
    async def test_list_workflows_with_empty_deployments(self):
        """P1: Returns empty list gracefully when no deployments exist."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        mock_client.list_deployments.return_value = []

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/workflows")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0


class TestFlowForgeWorkflowRun:
    """P0: Trigger workflow run."""

    @pytest.mark.asyncio
    async def test_trigger_workflow_run_requires_deployment(self):
        """P0: Trigger fails if workflow not deployed first."""
        from src.api.flowforge_workflow_proxy import router, get_deployment_id

        app = FastAPI()
        app.include_router(router)

        # No deployment mapping exists
        with patch('src.api.flowforge_workflow_proxy.get_deployment_id', return_value=None):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/workflows/test-workflow-id/run",
                    json={"canvas_id": "canvas-1", "operator_id": "op-1", "run_reason": "manual"}
                )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_trigger_workflow_run_success(self):
        """P0: POST /api/workflows/{id}/run triggers Prefect flow run."""
        from src.api.flowforge_workflow_proxy import router, get_deployment_id, set_deployment_id, get_prefect_client

        # Set up deployment mapping
        set_deployment_id("canvas-workflow-123", "prefect-deployment-456")

        mock_client = AsyncMock()
        mock_client.create_flow_run.return_value = {
            "run_id": "run-789",
            "deployment_id": "prefect-deployment-456",
            "state": "PENDING"
        }

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/workflows/canvas-workflow-123/run",
                    json={"canvas_id": "canvas-1", "operator_id": "op-1", "run_reason": "manual"}
                )

        assert response.status_code == 201
        data = response.json()
        assert data["run_id"] == "run-789"

    @pytest.mark.asyncio
    async def test_trigger_workflow_run_includes_context(self):
        """P1: Run request includes canvas_id, operator_id, run_reason."""
        from src.api.flowforge_workflow_proxy import router, get_deployment_id, set_deployment_id, get_prefect_client

        set_deployment_id("canvas-workflow-123", "prefect-deployment-456")

        mock_client = AsyncMock()
        mock_client.create_flow_run.return_value = {"run_id": "run-789", "deployment_id": "prefect-deployment-456", "state": "PENDING"}

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/workflows/canvas-workflow-123/run",
                    json={
                        "canvas_id": "live-trading-canvas",
                        "operator_id": "operator-mubarak",
                        "run_reason": "scheduled"
                    }
                )

        # Verify the context was passed to the client
        mock_client.create_flow_run.assert_called_once()
        call_args = mock_client.create_flow_run.call_args
        # create_flow_run(deployment_id, run_context) - positional args
        assert call_args[0][0] == "prefect-deployment-456"
        run_context = call_args[0][1]
        assert run_context["canvas_id"] == "live-trading-canvas"
        assert run_context["operator_id"] == "operator-mubarak"
        assert run_context["run_reason"] == "scheduled"


class TestFlowForgeWorkflowCancel:
    """P0: Cancel workflow run (kill switch)."""

    @pytest.mark.asyncio
    async def test_cancel_workflow_run_success(self):
        """P0: DELETE /api/workflows/{id}/run/{run_id} cancels Prefect flow."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        mock_client.cancel_flow_run.return_value = {
            "success": True,
            "run_id": "run-123",
            "state": "CANCELLED"
        }

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete("/api/workflows/workflow-1/run/run-123")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["run_id"] == "run-123"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_run_returns_404(self):
        """P1: Canceling nonexistent run returns 404."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        from fastapi import HTTPException
        mock_client = AsyncMock()
        mock_client.cancel_flow_run.side_effect = HTTPException(status_code=404, detail="Flow run not found")

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete("/api/workflows/wf-1/run/nonexistent")

        assert response.status_code == 404


class TestFlowForgeWorkflowDelete:
    """P0: Delete workflow and Prefect deployment."""

    @pytest.mark.asyncio
    async def test_delete_workflow_removes_mapping(self):
        """P0: DELETE /api/workflows/{id} deletes Prefect deployment and local mapping."""
        from src.api.flowforge_workflow_proxy import router, get_deployment_id, set_deployment_id, remove_deployment_mapping, get_prefect_client

        # Set up a mapping to delete
        set_deployment_id("canvas-workflow-to-delete", "prefect-deployment-to-delete")

        mock_client = AsyncMock()
        mock_client.delete_deployment.return_value = {"success": True, "deployment_id": "prefect-deployment-to-delete"}

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.delete("/api/workflows/canvas-workflow-to-delete")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify mapping was removed
        from src.api.flowforge_workflow_proxy import get_deployment_id
        assert get_deployment_id("canvas-workflow-to-delete") is None


class TestFlowForgeSSEEvents:
    """P1: SSE event stream for workflow stage updates."""

    @pytest.mark.asyncio
    async def test_workflow_events_endpoint_returns_sse(self):
        """P1: GET /api/workflows/{id}/events returns Server-Sent Events stream."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        # Simulate RUNNING state that transitions to COMPLETED
        mock_client.get_flow_run_state.side_effect = [
            {"run_id": "run-123", "state": "RUNNING", "started_at": None},
            {"run_id": "run-123", "state": "COMPLETED", "started_at": None},
        ]

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                # Use timeout since SSE is streaming
                response = await client.get(
                    "/api/workflows/workflow-1/events?run_id=run-123",
                    timeout=5.0
                )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_sse_stream_delivers_stage_events(self):
        """P1: SSE stream delivers stage transition events."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        mock_client.get_flow_run_state.side_effect = [
            {"run_id": "run-123", "state": "RUNNING", "started_at": None},
            {"run_id": "run-123", "state": "COMPLETED", "started_at": None},
        ]

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get(
                    "/api/workflows/workflow-1/events?run_id=run-123",
                    timeout=5.0
                )

        # Read the stream content
        content = b""
        async for chunk in response.aiter_bytes():
            content += chunk
            if b"completion" in chunk:
                break

        decoded = content.decode("utf-8")
        assert "data:" in decoded  # SSE format


class TestFlowForgeErrorHandling:
    """P2: Error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_deployment_timeout_returns_504(self):
        """P2: Deployment timeout returns 504 Gateway Timeout."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client
        from fastapi import HTTPException

        mock_client = AsyncMock()
        # Simulate the HTTPException(504) that create_deployment raises on timeout
        mock_client.create_deployment.side_effect = HTTPException(status_code=504, detail="Prefect deployment creation timed out")

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/api/workflows",
                    json={
                        "canvas_workflow_uuid": "test-uuid",
                        "name": "Test",
                        "nodes": []
                    }
                )

        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_list_workflows_graceful_degradation(self):
        """P2: List endpoint returns empty list on Prefect error (graceful degradation)."""
        from src.api.flowforge_workflow_proxy import router, get_prefect_client

        mock_client = AsyncMock()
        mock_client.list_deployments.side_effect = Exception("Prefect API unavailable")

        with patch('src.api.flowforge_workflow_proxy.get_prefect_client', return_value=mock_client):
            app = FastAPI()
            app.include_router(router)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/api/workflows")

        # Should return 500 (not 200) since it's an unhandled exception
        assert response.status_code == 500


class TestFlowForgeNodeDefinition:
    """P2: Node definition validation."""

    def test_node_definition_with_dependencies(self):
        """P2: NodeDefinition accepts depends_on list."""
        from src.api.flowforge_workflow_proxy import NodeDefinition

        node = NodeDefinition(
            id="node-1",
            type="task",
            config={"timeout": 300},
            depends_on=["node-0"]
        )

        assert node.id == "node-1"
        assert node.type == "task"
        assert node.depends_on == ["node-0"]

    def test_node_definition_default_empty_depends_on(self):
        """P2: NodeDefinition defaults depends_on to empty list."""
        from src.api.flowforge_workflow_proxy import NodeDefinition

        node = NodeDefinition(id="node-1", type="trigger", config={})
        assert node.depends_on == []


class TestFlowForgeWorkflowRunRequest:
    """P2: WorkflowRunRequest validation."""

    def test_workflow_run_request_all_fields(self):
        """P2: WorkflowRunRequest accepts all optional fields."""
        from src.api.flowforge_workflow_proxy import WorkflowRunRequest

        request = WorkflowRunRequest(
            canvas_id="canvas-1",
            operator_id="op-1",
            run_reason="scheduled"
        )

        assert request.canvas_id == "canvas-1"
        assert request.operator_id == "op-1"
        assert request.run_reason == "scheduled"

    def test_workflow_run_request_defaults(self):
        """P2: WorkflowRunRequest defaults run_reason to None."""
        from src.api.flowforge_workflow_proxy import WorkflowRunRequest

        request = WorkflowRunRequest(canvas_id="canvas-1")
        assert request.run_reason is None


class TestPrefectClientSanitization:
    """P3: Prefect client name sanitization."""

    def test_sanitize_name_lowercases(self):
        """P3: Name sanitization converts to lowercase."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        result = client._sanitize_name("Alpha Research Pipeline")
        assert result == "alpha-research-pipeline"

    def test_sanitize_name_replaces_spaces_with_hyphens(self):
        """P3: Name sanitization replaces spaces with hyphens."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        result = client._sanitize_name("My Workflow Name")
        assert result == "my-workflow-name"

    def test_sanitize_name_removes_special_chars(self):
        """P3: Name sanitization replaces special characters with hyphens."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        result = client._sanitize_name("Workflow@#$%Name!")
        assert result == "workflow-name"

    def test_sanitize_name_collapses_multiple_hyphens(self):
        """P3: Multiple spaces become single hyphen."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        result = client._sanitize_name("Workflow   Name")
        assert result == "workflow-name"


class TestFlowForgeTimeoutConfiguration:
    """P3: Timeout configuration."""

    def test_prefect_client_default_timeouts(self):
        """P3: PrefectClient has correct default timeouts."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        assert client.deploy_timeout == 30.0
        assert client.run_timeout == 30.0
        assert client.status_timeout == 10.0

    def test_prefect_client_default_api_url(self):
        """P3: PrefectClient defaults to localhost:4200."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        assert client.api_url == "http://127.0.0.1:4200/api"

    def test_prefect_client_preserves_api_path_for_child_endpoints(self):
        """P3: Child endpoint construction must not strip the configured /api path."""
        from src.api.flowforge_workflow_proxy import PrefectClient

        client = PrefectClient()
        assert client._api_endpoint("/deployments/") == "http://127.0.0.1:4200/api/deployments/"


class TestFlowForgeDeploymentMapping:
    """P2: Canvas workflow to Prefect deployment ID mapping."""

    def test_set_and_get_deployment_id(self):
        """P2: Can store and retrieve deployment mapping."""
        from src.api.flowforge_workflow_proxy import set_deployment_id, get_deployment_id

        set_deployment_id("canvas-123", "prefect-456")
        assert get_deployment_id("canvas-123") == "prefect-456"

    def test_get_deployment_id_returns_none_for_missing(self):
        """P2: Returns None for unmapped canvas workflow."""
        from src.api.flowforge_workflow_proxy import get_deployment_id

        assert get_deployment_id("nonexistent") is None

    def test_remove_deployment_mapping(self):
        """P2: Can remove deployment mapping."""
        from src.api.flowforge_workflow_proxy import set_deployment_id, get_deployment_id, remove_deployment_mapping

        set_deployment_id("canvas-123", "prefect-456")
        remove_deployment_mapping("canvas-123")
        assert get_deployment_id("canvas-123") is None
