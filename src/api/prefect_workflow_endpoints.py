# Prefect Workflow API Endpoints
"""
Real workflow read-model endpoints for the FlowForge canvas.

The frontend expects workflow cards grouped by state. Those cards should be backed by
real workflow persistence, not canned examples. This module reads the local workflow DB
used by the coordinator/flows layer and optionally merges Prefect deployment metadata
when a live Prefect API is configured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prefect", tags=["prefect"])

WORKFLOW_STATES = {
    "PENDING": "PENDING",
    "RUNNING": "RUNNING",
    "PENDING_REVIEW": "PENDING_REVIEW",
    "DONE": "DONE",
    "CANCELLED": "CANCELLED",
    "EXPIRED_REVIEW": "EXPIRED_REVIEW",
}

_STATUS_TO_STATE = {
    "pending": "PENDING",
    "created": "PENDING",
    "scheduled": "PENDING",
    "running": "RUNNING",
    "in_progress": "RUNNING",
    "waiting": "PENDING_REVIEW",
    "pending_review": "PENDING_REVIEW",
    "paused": "PENDING_REVIEW",
    "completed": "DONE",
    "success": "DONE",
    "cancelled": "CANCELLED",
    "canceled": "CANCELLED",
    "failed": "CANCELLED",
    "expired_review": "EXPIRED_REVIEW",
}

_WORKFLOW_LABELS = {
    "wf1_creation": "WF1 Creation",
    "wf2_enhancement": "WF2 Enhancement",
    "wf3_performance_intel": "WF3 Performance Intel",
    "wf4_weekend_update": "WF4 Weekend Update",
}


def _empty_buckets() -> Dict[str, List[Dict[str, Any]]]:
    return {state: [] for state in WORKFLOW_STATES}


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _workflow_state(status: Optional[str]) -> str:
    return _STATUS_TO_STATE.get((status or "").strip().lower(), "PENDING")


def _humanize_stage(stage_name: str) -> str:
    return stage_name.replace("_", " ").title()


def _workflow_name(run: Dict[str, Any]) -> str:
    workflow_name = run.get("workflow_name") or "workflow"
    return _WORKFLOW_LABELS.get(workflow_name, workflow_name.replace("_", " ").title())


def _department_for_run(run: Dict[str, Any], stage_results: List[Dict[str, Any]]) -> str:
    workflow_name = (run.get("workflow_name") or "").strip()

    try:
        from src.agents.departments.workflow_models import (
            WorkflowStage,
            WorkflowType,
            WORKFLOW_STAGE_DEPARTMENTS,
        )

        workflow_type = WorkflowType(workflow_name)
        stage_department_map = WORKFLOW_STAGE_DEPARTMENTS.get(workflow_type, {})
        for stage in reversed(stage_results):
            stage_name = stage.get("stage_name")
            dept = stage_department_map.get(WorkflowStage(stage_name)) if stage_name else None
            if dept:
                return dept.replace("_", " ").title()
    except Exception:
        pass

    if workflow_name.startswith("wf1_") or workflow_name.startswith("wf3_"):
        return "Research"
    if workflow_name.startswith("wf2_"):
        return "Development"
    if workflow_name.startswith("wf4_"):
        return "Portfolio"
    return "FlowForge"


def _build_tasks(stage_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for index, stage in enumerate(stage_results):
        task_state = _workflow_state(stage.get("status"))
        if task_state == "DONE":
            task_state = "COMPLETED"
        tasks.append(
            {
                "id": f"{stage.get('workflow_run_id', 'workflow')}-task-{index + 1}",
                "name": _humanize_stage(stage.get("stage_name", f"stage_{index + 1}")),
                "state": task_state,
                "x": 120 + (index * 150),
                "y": 60,
            }
        )
    return tasks


def _build_dependencies(tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {"from": tasks[index]["id"], "to": tasks[index + 1]["id"]}
        for index in range(len(tasks) - 1)
    ]


def _serialize_db_workflow(run: Dict[str, Any], stage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = _build_tasks(stage_results)
    dependencies = _build_dependencies(tasks)
    state = _workflow_state(run.get("status"))

    completed_steps = sum(1 for stage in stage_results if _workflow_state(stage.get("status")) == "DONE")
    total_steps = len(stage_results)
    if state == "DONE" and total_steps > 0 and completed_steps == 0:
        completed_steps = total_steps
    if state == "DONE":
        next_stage = "Completed"
    else:
        next_stage = next(
            (
                _humanize_stage(stage.get("stage_name", ""))
                for stage in stage_results
                if _workflow_state(stage.get("status")) not in {"DONE", "CANCELLED", "EXPIRED_REVIEW"}
            ),
            "Awaiting Work",
        )

    started_at = _parse_timestamp(run.get("created_at"))
    ended_at = _parse_timestamp(run.get("updated_at"))
    duration_seconds = 0
    if started_at and ended_at:
        duration_seconds = max(0, int((ended_at - started_at).total_seconds()))

    return {
        "id": run["id"],
        "flow_id": run["id"],
        "name": _workflow_name(run),
        "department": _department_for_run(run, stage_results),
        "state": state,
        "started_at": run.get("created_at"),
        "duration_seconds": duration_seconds,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "next_step": next_stage,
        "tasks": tasks,
        "dependencies": dependencies,
        "source": "workflow_db",
        "updated_at": run.get("updated_at"),
        "error_message": run.get("error_message"),
    }


def _serialize_prefect_workflow(deployment: Dict[str, Any]) -> Dict[str, Any]:
    state = deployment.get("state") or "PENDING"
    return {
        "id": deployment.get("id"),
        "flow_id": deployment.get("flow_name") or deployment.get("id"),
        "name": deployment.get("name") or "Unnamed Workflow",
        "department": "FlowForge",
        "state": state if state in WORKFLOW_STATES else "PENDING",
        "started_at": None,
        "duration_seconds": 0,
        "completed_steps": 0,
        "total_steps": 0,
        "next_step": "Awaiting Run",
        "tasks": [],
        "dependencies": [],
        "source": "prefect_api",
    }


def _load_db_workflows() -> Dict[str, Dict[str, Any]]:
    from flows.database import get_workflow_database

    db = get_workflow_database()
    workflows: Dict[str, Dict[str, Any]] = {}
    for run in db.list_workflow_runs(limit=200):
        stage_results = db.get_stage_results(run["id"])
        workflows[run["id"]] = _serialize_db_workflow(run, stage_results)
    return workflows


async def _load_prefect_deployments() -> Dict[str, Dict[str, Any]]:
    try:
        from src.api.flowforge_workflow_proxy import get_prefect_client

        deployments = await get_prefect_client().list_deployments()
    except Exception as exc:
        logger.info(f"Prefect deployment metadata unavailable: {exc}")
        return {}

    return {
        deployment["id"]: _serialize_prefect_workflow(deployment)
        for deployment in deployments
        if deployment.get("id")
    }


async def _merged_workflows() -> Dict[str, Dict[str, Any]]:
    workflows = _load_db_workflows()
    for workflow_id, deployment in (await _load_prefect_deployments()).items():
        workflows.setdefault(workflow_id, deployment)
    return workflows


@router.get("/workflows")
async def list_workflows() -> Dict[str, Any]:
    """List real workflows grouped by FlowForge Kanban state."""
    workflows = list((await _merged_workflows()).values())
    workflows.sort(key=lambda workflow: workflow.get("started_at") or "", reverse=True)

    grouped = _empty_buckets()
    for workflow in workflows:
        grouped.setdefault(workflow["state"], []).append(workflow)

    return {
        "workflows": workflows,
        "by_state": grouped,
        "total": len(workflows),
    }


@router.get("/workflows/{workflow_id}")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Return one real workflow card with task graph details."""
    workflows = await _merged_workflows()
    workflow = workflows.get(workflow_id)
    if workflow:
        return workflow
    raise HTTPException(status_code=404, detail="Workflow not found")


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str) -> Dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Workflow cancellation is not wired to a live execution backend yet",
    )


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Workflow resume is not wired to a live execution backend yet",
    )
