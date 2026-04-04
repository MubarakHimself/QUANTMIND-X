# FlowForge Workflow Proxy API
"""
Canonical REST + SSE API contract between FlowForge canvas and Prefect execution engine.

FlowForge (N8N-style visual composer) -> Proxy API -> Prefect

This proxy provides:
- Workflow deployment (canvas node graph -> Prefect deployment)
- Workflow run triggering with SSE stage updates
- Workflow cancellation
- Workflow listing with status badges
- Workflow deletion

All FlowForge <-> Prefect interaction goes through this proxy.
FlowForge NEVER calls Prefect DTOs directly.

Environment Variables:
- PREFECT_API_URL: Base URL of Prefect API (e.g., http://localhost:4200/api)
- PREFECT_API_KEY: Prefect API authentication key
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import logging
import json
import asyncio
import os
import aiohttp
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["flowforge-workflow"])

# ============================================================================
# Pydantic Models (API Contract)
# ============================================================================


class NodeDefinition(BaseModel):
    """A node in the FlowForge canvas workflow graph."""

    id: str = Field(..., description="Unique node identifier")
    type: str = Field(..., description="Node type: trigger, task, etc.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    depends_on: Optional[List[str]] = Field(
        default_factory=list, description="List of node IDs this node depends on"
    )


class WorkflowDeploymentRequest(BaseModel):
    """Request payload for creating a Prefect deployment from a FlowForge node graph."""

    canvas_workflow_uuid: str = Field(..., description="UUID of the workflow in FlowForge canvas")
    name: str = Field(..., description="Workflow display name")
    nodes: List[NodeDefinition] = Field(..., description="List of nodes in the workflow graph")
    department: Optional[str] = Field(None, description="Department that owns this workflow")


class WorkflowRunRequest(BaseModel):
    """Request payload for triggering a workflow run."""

    canvas_id: Optional[str] = Field(None, description="Canvas ID that triggered this run")
    operator_id: Optional[str] = Field(None, description="Operator who triggered the run")
    run_reason: Optional[str] = Field(
        None, description="Reason for run: 'manual', 'scheduled', 'trigger', etc."
    )


class StageEvent(BaseModel):
    """SSE event for workflow stage transitions."""

    type: Literal["stage"] = "stage"
    run_id: str
    stage: str = Field(..., description="Task/stage name")
    status: Literal["running", "completed", "failed", "cancelled"]
    elapsed_s: int = Field(..., description="Seconds since run started")


class ErrorEvent(BaseModel):
    """SSE event for task errors."""

    type: Literal["error"] = "error"
    run_id: str
    task_name: str
    error_type: str
    error_message: str
    retry_count: int = 0


class CompletionEvent(BaseModel):
    """SSE event for workflow completion."""

    type: Literal["completion"] = "completion"
    run_id: str
    status: Literal["completed", "failed", "cancelled"]
    final_state: str
    elapsed_s: int


# ============================================================================
# Prefect Client Wrapper
# ============================================================================


class PrefectClient:
    """Wrapper for Prefect API client with authentication and timeouts."""

    def __init__(self):
        self.api_url = self._normalize_api_url(
            os.getenv("PREFECT_API_URL", "http://127.0.0.1:4200/api")
        )
        self.api_key = os.getenv("PREFECT_API_KEY", "")
        self._session = None
        # Timeouts in seconds
        self.deploy_timeout = 30.0
        self.run_timeout = 30.0
        self.status_timeout = 10.0

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        """Ensure the configured Prefect base URL keeps the API path."""
        parsed = urlsplit(api_url.strip())
        path = parsed.path.rstrip("/")
        if not path:
            path = "/api"
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    def _api_endpoint(self, path: str) -> str:
        """Build a child API URL without stripping the configured /api path."""
        base = self.api_url.rstrip("/")
        child = path if path.startswith("/") else f"/{path}"
        return f"{base}{child}"

    @staticmethod
    def _extract_state_name(payload: Dict[str, Any], default: str = "UNKNOWN") -> str:
        """Handle Prefect responses that nest state details under a dict."""
        state = payload.get("state")
        if isinstance(state, dict):
            return state.get("type") or state.get("name") or default
        if isinstance(state, str):
            return state
        return default

    async def _get_session(self):
        """Get or create aiohttp session with auth."""
        if self._session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Use default 30s timeout - individual methods override via timeout param
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session

    async def _post_with_timeout(self, url: str, json: dict, timeout: float) -> aiohttp.ClientResponse:
        """POST with specific timeout."""
        session = await self._get_session()
        return await session.post(url, json=json, timeout=aiohttp.ClientTimeout(total=timeout))

    async def _get_with_timeout(self, url: str, timeout: float) -> aiohttp.ClientResponse:
        """GET with specific timeout."""
        session = await self._get_session()
        return await session.get(url, timeout=aiohttp.ClientTimeout(total=timeout))

    async def _delete_with_timeout(self, url: str, timeout: float) -> aiohttp.ClientResponse:
        """DELETE with specific timeout."""
        session = await self._get_session()
        return await session.delete(url, timeout=aiohttp.ClientTimeout(total=timeout))

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def create_deployment(
        self, canvas_workflow_uuid: str, name: str, nodes: List[NodeDefinition], department: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a Prefect deployment from FlowForge node graph."""
        session = await self._get_session()

        # Map FlowForge node graph to Prefect deployment format
        prefect_payload = {
            "name": self._sanitize_name(name),
            "flow_name": "flowforge_workflow",
            "parameters": {
                "canvas_workflow_uuid": canvas_workflow_uuid,
                "nodes": [node.model_dump() for node in nodes],
            },
            "work_queue_name": "default",
        }

        if department:
            prefect_payload["description"] = f"FlowForge workflow for {department}"

        try:
            async with await self._post_with_timeout(
                self._api_endpoint("/deployments/"), prefect_payload, self.deploy_timeout
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    return {
                        "workflow_id": canvas_workflow_uuid,
                        "deployment_id": data.get("id"),
                        "deployment_name": data.get("name"),
                    }
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect deployment creation timed out")
        except Exception as e:
            logger.error(f"Error creating Prefect deployment: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def list_deployments(self) -> List[Dict[str, Any]]:
        """List all Prefect deployments."""
        session = await self._get_session()

        try:
            async with await self._get_with_timeout(
                self._api_endpoint("/deployments/"), self.status_timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    deployments = data if isinstance(data, list) else data.get("deployments", [])

                    # Map Prefect deployments to FlowForge workflow format
                    return [
                        {
                            "id": dep.get("id"),
                            "name": dep.get("name"),
                            "flow_name": dep.get("flow_name"),
                            "deployment_id": dep.get("id"),
                            "canvas_workflow_uuid": dep.get("parameters", {})
                            .get("canvas_workflow_uuid", dep.get("id")),
                            "state": "PENDING",  # Will be updated from flow runs
                        }
                        for dep in deployments
                        if dep.get("flow_name") == "flowforge_workflow"
                        or dep.get("name", "").startswith("flowforge-")
                    ]
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect deployment list timed out")
        except Exception as e:
            logger.error(f"Error listing Prefect deployments: {e}")
            # Return empty list on error to allow graceful degradation
            return []

    async def get_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment details."""
        session = await self._get_session()

        try:
            async with await self._get_with_timeout(
                self._api_endpoint(f"/deployments/{deployment_id}"), self.status_timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "flow_name": data.get("flow_name"),
                        "parameters": data.get("parameters", {}),
                        "canvas_workflow_uuid": data.get("parameters", {})
                        .get("canvas_workflow_uuid", data.get("id")),
                    }
                elif resp.status == 404:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect deployment fetch timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Prefect deployment: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Delete a Prefect deployment."""
        session = await self._get_session()

        try:
            async with await self._delete_with_timeout(
                self._api_endpoint(f"/deployments/{deployment_id}"), self.deploy_timeout
            ) as resp:
                if resp.status in (200, 204):
                    return {"success": True, "deployment_id": deployment_id}
                elif resp.status == 404:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect deployment deletion timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting Prefect deployment: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def create_flow_run(
        self, deployment_id: str, run_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger a flow run for a deployment."""
        session = await self._get_session()

        payload = {"deployment_id": deployment_id}
        if run_context:
            payload["parameters"] = run_context

        try:
            async with await self._post_with_timeout(
                self._api_endpoint("/flow_runs/"), payload, self.run_timeout
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    return {
                        "run_id": data.get("id"),
                        "deployment_id": deployment_id,
                        "state": self._extract_state_name(data, "PENDING"),
                        "created_at": data.get("created") or data.get("created_at"),
                    }
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect flow run creation timed out")
        except Exception as e:
            logger.error(f"Error creating Prefect flow run: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def cancel_flow_run(self, run_id: str) -> Dict[str, Any]:
        """Cancel a running flow."""
        session = await self._get_session()

        try:
            async with await self._post_with_timeout(
                self._api_endpoint(f"/flow_runs/{run_id}/cancel"), {}, self.run_timeout
            ) as resp:
                if resp.status in (200, 202):
                    data = await resp.json() if resp.content_length != 0 else {}
                    return {
                        "success": True,
                        "run_id": run_id,
                        "state": self._extract_state_name(data, "CANCELLED"),
                    }
                elif resp.status == 404:
                    raise HTTPException(status_code=404, detail="Flow run not found")
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect flow cancellation timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error cancelling Prefect flow run: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_flow_run_state(self, run_id: str) -> Dict[str, Any]:
        """Get current state of a flow run."""
        session = await self._get_session()

        try:
            async with await self._get_with_timeout(
                self._api_endpoint(f"/flow_runs/{run_id}"), self.status_timeout
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "run_id": run_id,
                        "state": self._extract_state_name(data, "UNKNOWN"),
                        "started_at": data.get("start_time") or data.get("expected_start_time"),
                        "ended_at": data.get("end_time"),
                    }
                elif resp.status == 404:
                    raise HTTPException(status_code=404, detail="Flow run not found")
                else:
                    error_text = await resp.text()
                    raise HTTPException(
                        status_code=resp.status, detail=f"Prefect API error: {error_text}"
                    )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Prefect state fetch timed out")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting Prefect flow run state: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _sanitize_name(self, name: str) -> str:
        """Sanitize workflow name for Prefect (lowercase, hyphenated)."""
        import re

        # Replace spaces and underscores with hyphens, lowercase
        sanitized = re.sub(r"[^\w\-]", "-", name)
        sanitized = re.sub(r"-+", "-", sanitized)  # Collapse multiple hyphens
        return sanitized.lower().strip("-")


# Global Prefect client instance
_prefect_client: Optional[PrefectClient] = None


def get_prefect_client() -> PrefectClient:
    """Get or create the global Prefect client."""
    global _prefect_client
    if _prefect_client is None:
        _prefect_client = PrefectClient()
    return _prefect_client


# ============================================================================
# Local State Store (canvas_workflow_uuid -> prefect_deployment_id mapping)
# ============================================================================

# In-memory store for UUID mapping (in production, persist to SQLite)
_deployment_mapping: Dict[str, str] = {}  # canvas_workflow_uuid -> prefect_deployment_id


def get_deployment_id(canvas_workflow_uuid: str) -> Optional[str]:
    """Get Prefect deployment ID for a canvas workflow UUID."""
    return _deployment_mapping.get(canvas_workflow_uuid)


def set_deployment_id(canvas_workflow_uuid: str, deployment_id: str) -> None:
    """Store mapping from canvas workflow UUID to Prefect deployment ID."""
    _deployment_mapping[canvas_workflow_uuid] = deployment_id


def remove_deployment_mapping(canvas_workflow_uuid: str) -> None:
    """Remove a deployment mapping."""
    if canvas_workflow_uuid in _deployment_mapping:
        del _deployment_mapping[canvas_workflow_uuid]


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("", status_code=201)
async def create_workflow_deployment(
    request: WorkflowDeploymentRequest,
) -> Dict[str, Any]:
    """
    Create a Prefect deployment from a FlowForge node graph.

    FlowForge POSTs the node graph serialized as a Prefect deployment payload.
    Prefect creates the deployment and returns workflow_id and deployment_id.
    FlowForge stores the mapping: canvas_workflow_uuid -> prefect_deployment_id.

    AC1: Workflow Deployment
    """
    client = get_prefect_client()

    try:
        result = await client.create_deployment(
            canvas_workflow_uuid=request.canvas_workflow_uuid,
            name=request.name,
            nodes=request.nodes,
            department=request.department,
        )

        # Store the mapping
        set_deployment_id(request.canvas_workflow_uuid, result["deployment_id"])

        logger.info(
            f"Workflow deployed: canvas_uuid={request.canvas_workflow_uuid}, "
            f"deployment_id={result['deployment_id']}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating workflow deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_workflows() -> Dict[str, Any]:
    """
    List all deployed FlowForge workflows with status badges.

    User-facing labels use "FlowForge workflow" not "Prefect deployment".

    AC6: Workflow List with Status Badges
    """
    client = get_prefect_client()

    try:
        deployments = await client.list_deployments()

        # Enrich with local mapping info
        for dep in deployments:
            canvas_uuid = dep.get("canvas_workflow_uuid")
            if canvas_uuid and canvas_uuid in _deployment_mapping:
                dep["local_deployment_id"] = _deployment_mapping[canvas_uuid]

        # Group by state for Kanban (all start as PENDING)
        workflows_by_state = {
            "PENDING": [],
            "RUNNING": [],
            "PENDING_REVIEW": [],
            "DONE": [],
            "CANCELLED": [],
            "EXPIRED_REVIEW": [],
        }

        for workflow in deployments:
            state = workflow.get("state", "PENDING")
            if state in workflows_by_state:
                workflows_by_state[state].append(workflow)
            else:
                workflows_by_state["PENDING"].append(workflow)

        return {
            "workflows": deployments,
            "by_state": workflows_by_state,
            "total": len(deployments),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}")
async def get_workflow_details(workflow_id: str) -> Dict[str, Any]:
    """
    Get detailed workflow information including deployment config.

    AC6: Workflow List with Status Badges (details endpoint)
    """
    client = get_prefect_client()

    # Try to find by deployment_id first, then by canvas_workflow_uuid
    deployment_id = get_deployment_id(workflow_id)
    if not deployment_id:
        deployment_id = workflow_id

    try:
        return await client.get_deployment(deployment_id)
    except HTTPException:
        raise


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> Dict[str, Any]:
    """
    Delete a workflow and its Prefect deployment.

    FlowForge removes the local state and refreshes the workflow list.

    AC9: Workflow Deletion
    """
    client = get_prefect_client()

    # Find the deployment_id from the mapping or use workflow_id directly
    deployment_id = get_deployment_id(workflow_id)
    if not deployment_id:
        deployment_id = workflow_id

    try:
        result = await client.delete_deployment(deployment_id)

        # Clean up local mapping
        remove_deployment_mapping(workflow_id)

        logger.info(f"Workflow deleted: workflow_id={workflow_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{workflow_id}/run", status_code=201)
async def trigger_workflow_run(
    workflow_id: str, request: WorkflowRunRequest
) -> Dict[str, Any]:
    """
    Trigger a workflow run.

    FlowForge POSTs to this endpoint with trigger context.
    Prefect enqueues the flow run and returns a run_id.
    FlowForge subscribes to SSE for real-time stage updates.

    AC2: Workflow Run Trigger
    """
    client = get_prefect_client()

    # Get deployment_id from mapping or use workflow_id
    deployment_id = get_deployment_id(workflow_id)
    if not deployment_id:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow {workflow_id} not found. Deploy it first.",
        )

    run_context = {
        "canvas_id": request.canvas_id,
        "operator_id": request.operator_id,
        "run_reason": request.run_reason or "manual",
    }

    try:
        result = await client.create_flow_run(deployment_id, run_context)

        logger.info(
            f"Workflow run triggered: workflow_id={workflow_id}, run_id={result['run_id']}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering workflow run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{workflow_id}/run/{run_id}")
async def cancel_workflow_run(
    workflow_id: str, run_id: str
) -> Dict[str, Any]:
    """
    Cancel a running workflow.

    FlowForge POSTs to cancel. Prefect cancels the flow run.
    FlowForge updates the node to a cancelled state with visual strikethrough.

    AC5: Workflow Cancellation (Kill)
    """
    client = get_prefect_client()

    try:
        result = await client.cancel_flow_run(run_id)

        logger.info(f"Workflow cancelled: workflow_id={workflow_id}, run_id={run_id}")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling workflow run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/events")
async def workflow_events(
    workflow_id: str, run_id: str
) -> StreamingResponse:
    """
    SSE stream for real-time workflow stage updates.

    Delivers stage transition events:
    - { type: "stage", run_id, stage, status, elapsed_s }
    - { type: "error", run_id, task_name, error_type, error_message, retry_count }
    - { type: "completion", run_id, status, final_state, elapsed_s }

    FlowForge subscribes to this endpoint for real-time updates.

    AC2: Workflow Run Trigger (SSE subscription)
    AC3: Real-time SSE Stage Updates
    AC4: Workflow Completion/Failure Notification
    AC7: Failed Task Error Display
    """
    client = get_prefect_client()

    async def event_generator():
        """Generate SSE events for workflow run progress."""
        start_time = datetime.now()
        completed_stages = set()
        failed_tasks = []

        try:
            # Poll Prefect for flow run state changes
            # In production, this would use Prefect's webhook/websocket system
            while True:
                try:
                    state_info = await client.get_flow_run_state(run_id)
                    current_state = state_info.get("state", "RUNNING")
                    elapsed_s = int((datetime.now() - start_time).total_seconds())

                    # Determine terminal state
                    if current_state in ("COMPLETED", "SUCCESS"):
                        # Emit completion event
                        yield f"data: {json.dumps({'type': 'completion', 'run_id': run_id, 'status': 'completed', 'final_state': current_state, 'elapsed_s': elapsed_s})}\n\n"
                        break
                    elif current_state in ("FAILED", "CRASHED"):
                        # AC7: Emit error event for failed tasks (if task-level info available from Prefect)
                        # Note: Basic polling only gets flow-level state. Task-level errors require Prefect webhooks.
                        yield f"data: {json.dumps({'type': 'error', 'run_id': run_id, 'task_name': 'workflow', 'error_type': current_state, 'error_message': f'Workflow run failed with state: {current_state}', 'retry_count': 0})}\n\n"
                        # AC4: Also emit completion
                        yield f"data: {json.dumps({'type': 'completion', 'run_id': run_id, 'status': 'failed', 'final_state': current_state, 'elapsed_s': elapsed_s})}\n\n"
                        break
                    elif current_state == "CANCELLED":
                        yield f"data: {json.dumps({'type': 'completion', 'run_id': run_id, 'status': 'cancelled', 'final_state': current_state, 'elapsed_s': elapsed_s})}\n\n"
                        break
                    elif current_state == "RUNNING":
                        # Emit running state with stage info
                        # In production, fetch actual task states from Prefect
                        yield f"data: {json.dumps({'type': 'stage', 'run_id': run_id, 'stage': 'workflow', 'status': 'running', 'elapsed_s': elapsed_s})}\n\n"

                    # Check if we should continue polling
                    if current_state in ("COMPLETED", "SUCCESS", "FAILED", "CRASHED", "CANCELLED"):
                        break

                    await asyncio.sleep(2)  # Poll every 2 seconds

                except HTTPException as e:
                    if e.status_code == 404:
                        # Flow run not found - might be completed and cleaned up
                        yield f"data: {json.dumps({'type': 'completion', 'run_id': run_id, 'status': 'completed', 'final_state': 'UNKNOWN', 'elapsed_s': int((datetime.now() - start_time).total_seconds())})}\n\n"
                        break
                    raise

        except Exception as e:
            logger.error(f"SSE stream error for run {run_id}: {e}")
            # Send error as SSE event
            error_event = {
                "type": "error",
                "run_id": run_id,
                "task_name": "workflow",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "retry_count": 0,
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )
