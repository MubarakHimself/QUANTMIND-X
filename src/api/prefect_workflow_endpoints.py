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
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException

from src.api.wf1_artifacts import (
    find_strategy_root,
    find_strategy_root_by_workflow_id,
    iter_workflow_manifests,
)

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


def _is_manual_pause(manifest_like: Dict[str, Any]) -> bool:
    if manifest_like.get("manual_pause") is True:
        return True
    metadata = manifest_like.get("metadata") or {}
    if metadata.get("manual_pause") is True:
        return True
    status = (manifest_like.get("status") or "").strip().lower()
    if status == "paused":
        return True
    waiting_reason = (manifest_like.get("waiting_reason") or "").strip().lower()
    pause_reason = (manifest_like.get("pause_reason") or "").strip().lower()
    return "paused by" in waiting_reason or "paused by" in pause_reason


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


def _department_for_stage(workflow_name: str, stage_name: Optional[str]) -> str:
    if not stage_name:
        return "FlowForge"
    try:
        from src.agents.departments.workflow_models import (
            WorkflowStage,
            WorkflowType,
            WORKFLOW_STAGE_DEPARTMENTS,
        )

        workflow_type = WorkflowType(workflow_name)
        stage = WorkflowStage(stage_name)
        dept = WORKFLOW_STAGE_DEPARTMENTS.get(workflow_type, {}).get(stage)
        if dept:
            return dept.replace("_", " ").title()
    except Exception:
        pass
    return "FlowForge"


def _find_latest_artifact(strategy_root: Optional[Path]) -> Optional[Dict[str, Any]]:
    if not strategy_root or not strategy_root.exists():
        return None

    latest_file: Optional[Path] = None
    latest_mtime: float = -1.0
    ignored_names = {".meta.json", "manifest.json", "request.json"}

    for path in strategy_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name in ignored_names:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_file = path

    if not latest_file:
        return None

    return {
        "name": latest_file.name,
        "path": str(latest_file.relative_to(strategy_root)).replace("\\", "/"),
        "updated_at": datetime.fromtimestamp(latest_mtime).isoformat(),
    }


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


def _workflow_stage_sequence(workflow_name: str) -> List[str]:
    try:
        from src.agents.departments.workflow_models import (
            WORKFLOW_INITIAL_STAGES,
            WORKFLOW_NEXT_STAGES,
            WorkflowType,
        )

        workflow_type = WorkflowType(workflow_name)
        current = WORKFLOW_INITIAL_STAGES.get(workflow_type)
        next_map = WORKFLOW_NEXT_STAGES.get(workflow_type, {})
        sequence: List[str] = []
        visited: set[str] = set()
        while current and current.value not in visited:
            sequence.append(current.value)
            visited.add(current.value)
            current = next_map.get(current)
            if current and current.value == "completed":
                sequence.append(current.value)
                break
        return sequence
    except Exception:
        return []


def _progress_from_stage(workflow_name: str, stage_name: Optional[str], state: str) -> tuple[int, int, str]:
    sequence = _workflow_stage_sequence(workflow_name)
    if not sequence:
        human_stage = _humanize_stage(stage_name or "pending")
        return (0, 0, "Completed" if state == "DONE" else human_stage)

    completed_marker = "completed"
    total_steps = max(0, len([stage for stage in sequence if stage != completed_marker]))
    if state == "DONE":
        return total_steps, total_steps, "Completed"

    current_stage = (stage_name or "").strip().lower()
    if not current_stage:
        return 0, total_steps, "Awaiting Work"

    try:
        stage_index = sequence.index(current_stage)
    except ValueError:
        return 0, total_steps, _humanize_stage(current_stage)

    completed_steps = min(stage_index, total_steps)
    return completed_steps, total_steps, _humanize_stage(current_stage)


def _serialize_db_workflow(run: Dict[str, Any], stage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tasks = _build_tasks(stage_results)
    dependencies = _build_dependencies(tasks)
    state = _workflow_state(run.get("status"))
    workflow_name = run.get("workflow_name") or "workflow"

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

    current_stage_raw = next(
        (
            stage.get("stage_name")
            for stage in stage_results
            if _workflow_state(stage.get("status")) not in {"DONE", "CANCELLED", "EXPIRED_REVIEW"}
        ),
        stage_results[-1].get("stage_name") if stage_results else None,
    )
    current_stage = _humanize_stage(current_stage_raw or "")
    strategy_root = find_strategy_root_by_workflow_id(run["id"])
    latest_artifact = _find_latest_artifact(strategy_root)
    blocking_error = run.get("error_message") or next(
        (stage.get("error_message") for stage in reversed(stage_results) if stage.get("error_message")),
        None,
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
        "department": _department_for_stage(workflow_name, current_stage_raw) or _department_for_run(run, stage_results),
        "state": state,
        "started_at": run.get("created_at"),
        "duration_seconds": duration_seconds,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "next_step": next_stage,
        "current_stage": current_stage,
        "tasks": tasks,
        "dependencies": dependencies,
        "source": "workflow_db",
        "updated_at": run.get("updated_at"),
        "error_message": blocking_error,
        "blocking_error": blocking_error,
        "waiting_reason": (
            "Paused by operator"
            if (run.get("status") or "").strip().lower() == "paused"
            else next_stage if state == "PENDING_REVIEW" else None
        ),
        "latest_artifact": latest_artifact,
        "is_manual_pause": (run.get("status") or "").strip().lower() == "paused",
        "can_pause": state == "RUNNING",
        "can_resume": (run.get("status") or "").strip().lower() == "paused",
        "can_retry": state in {"CANCELLED", "EXPIRED_REVIEW"} or bool(blocking_error),
    }


def _serialize_manifest_workflow(strategy_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = manifest.get("workflow_id") or strategy_root.name
    workflow_type = manifest.get("workflow_type") or "wf1_creation"
    state = _workflow_state(manifest.get("status"))
    current_stage_raw = manifest.get("current_stage")
    latest_artifact = manifest.get("latest_artifact") or _find_latest_artifact(strategy_root)
    started_at = None

    meta_path = strategy_root / ".meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        started_at = meta.get("created_at") or meta.get("updated_at")

    if not started_at:
        started_at = manifest.get("created_at") or manifest.get("updated_at")

    completed_steps, total_steps, next_step = _progress_from_stage(
        workflow_type,
        current_stage_raw,
        state,
    )
    updated_at = manifest.get("updated_at")
    duration_seconds = 0
    if started_at and updated_at:
        started = _parse_timestamp(started_at)
        updated = _parse_timestamp(updated_at)
        if started and updated:
            duration_seconds = max(0, int((updated - started).total_seconds()))

    is_manual_pause = _is_manual_pause(manifest)
    return {
        "id": workflow_id,
        "flow_id": workflow_id,
        "name": _WORKFLOW_LABELS.get(workflow_type, workflow_type.replace("_", " ").title()),
        "department": _department_for_stage(workflow_type, current_stage_raw),
        "state": state,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "next_step": next_step,
        "current_stage": _humanize_stage(current_stage_raw or "pending"),
        "tasks": [],
        "dependencies": [],
        "source": "wf1_manifest",
        "updated_at": updated_at,
        "error_message": manifest.get("blocking_error"),
        "blocking_error": manifest.get("blocking_error"),
        "waiting_reason": manifest.get("waiting_reason"),
        "latest_artifact": latest_artifact,
        "strategy_id": manifest.get("strategy_id") or strategy_root.name,
        "is_manual_pause": is_manual_pause,
        "can_pause": state == "RUNNING",
        "can_resume": state == "PENDING_REVIEW" and is_manual_pause,
        "can_retry": state in {"CANCELLED", "EXPIRED_REVIEW"} or bool(manifest.get("blocking_error")),
    }


def _manifest_has_operator_visible_signal(strategy_root: Path, manifest: Dict[str, Any]) -> bool:
    if manifest.get("blocking_error"):
        return True
    if manifest.get("latest_artifact") or _find_latest_artifact(strategy_root):
        return True

    request_path = strategy_root / "source" / "request.json"
    if request_path.exists():
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception:
            request = {}
        if (request.get("source_url") or "").strip():
            return True

    status = (manifest.get("status") or "").strip().lower()
    current_stage = (manifest.get("current_stage") or "").strip().lower()
    normalized_stage = current_stage.replace(" ", "_").replace("-", "_")
    if status not in {"pending", "created", "scheduled"}:
        return True
    if normalized_stage not in {"", "video_ingest"}:
        return True
    return False


def _serialize_live_workflow(run: Dict[str, Any]) -> Dict[str, Any]:
    workflow_id = run["workflow_id"]
    workflow_type = run.get("workflow_type") or "custom"
    current_stage_raw = run.get("current_stage")
    strategy_id = run.get("strategy_id")
    strategy_root = find_strategy_root(strategy_id) or find_strategy_root_by_workflow_id(workflow_id)
    latest_artifact = _find_latest_artifact(strategy_root)
    tasks = [
        {
            "id": task.get("task_id", f"{workflow_id}-task"),
            "name": _humanize_stage(task.get("stage", "task")),
            "state": _workflow_state(task.get("status")).replace("DONE", "COMPLETED"),
            "x": 120 + (idx * 150),
            "y": 60,
        }
        for idx, task in enumerate(run.get("tasks", []))
    ]
    dependencies = _build_dependencies(tasks)
    blocking_error = run.get("error") or next(
        (
            task.get("error")
            for task in reversed(run.get("tasks", []))
            if task.get("error")
        ),
        None,
    )
    metadata = run.get("metadata") or {}
    waiting_reason = None
    if run.get("status") == "waiting":
        waiting_reason = (
            metadata.get("pause_reason")
            if metadata.get("manual_pause")
            else metadata.get("pending_gate") or _humanize_stage(current_stage_raw or "approval")
        )
    is_manual_pause = bool(metadata.get("manual_pause"))

    return {
        "id": workflow_id,
        "flow_id": workflow_id,
        "name": _WORKFLOW_LABELS.get(workflow_type, workflow_type.replace("_", " ").title()),
        "department": _department_for_stage(workflow_type, current_stage_raw),
        "state": _workflow_state(run.get("status")),
        "started_at": run.get("created_at"),
        "duration_seconds": max(
            0,
            int(
                (
                    _parse_timestamp(run.get("updated_at")) -
                    _parse_timestamp(run.get("created_at"))
                ).total_seconds()
            ),
        ) if run.get("created_at") and run.get("updated_at") else 0,
        "completed_steps": sum(1 for task in run.get("tasks", []) if task.get("status") == "completed"),
        "total_steps": len(run.get("tasks", [])),
        "next_step": _humanize_stage(current_stage_raw or "pending"),
        "current_stage": _humanize_stage(current_stage_raw or "pending"),
        "tasks": tasks,
        "dependencies": dependencies,
        "source": "workflow_coordinator",
        "updated_at": run.get("updated_at"),
        "error_message": blocking_error,
        "blocking_error": blocking_error,
        "waiting_reason": waiting_reason,
        "latest_artifact": latest_artifact,
        "strategy_id": strategy_id,
        "is_manual_pause": is_manual_pause,
        "can_pause": _workflow_state(run.get("status")) == "RUNNING",
        "can_resume": _workflow_state(run.get("status")) == "PENDING_REVIEW" and is_manual_pause,
        "can_retry": _workflow_state(run.get("status")) in {"CANCELLED", "EXPIRED_REVIEW"} or bool(blocking_error),
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
        strategy_root = find_strategy_root_by_workflow_id(run["id"])
        if (
            not stage_results
            and strategy_root is None
            and not run.get("error_message")
            and (run.get("status") or "").strip().lower() in {"pending", "created", "scheduled"}
        ):
            continue
        workflows[run["id"]] = _serialize_db_workflow(run, stage_results)
    return workflows


def _load_wf1_manifest_workflows() -> Dict[str, Dict[str, Any]]:
    entries: List[tuple[str, Dict[str, Any]]] = []
    for strategy_root, manifest in iter_workflow_manifests():
        workflow_id = manifest.get("workflow_id")
        if not workflow_id:
            continue
        if not _manifest_has_operator_visible_signal(strategy_root, manifest):
            continue
        entries.append((workflow_id, _serialize_manifest_workflow(strategy_root, manifest)))
    return dict(entries)


def _load_live_coordinator_workflows() -> Dict[str, Dict[str, Any]]:
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()
        return {
            workflow["workflow_id"]: _serialize_live_workflow(workflow)
            for workflow in coordinator.get_all_workflows()
        }
    except Exception as exc:
        logger.info(f"Coordinator workflow metadata unavailable: {exc}")
        return {}


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
    workflows.update(_load_wf1_manifest_workflows())
    workflows.update(_load_live_coordinator_workflows())
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
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()
        if coordinator.cancel_workflow(workflow_id):
            workflow = coordinator.get_workflow_status(workflow_id)
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": (workflow or {}).get("status", "cancelled"),
            }
        raise HTTPException(status_code=404, detail="Workflow not found or already terminal")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to cancel workflow: {exc}")


@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(workflow_id: str) -> Dict[str, Any]:
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()
        result = coordinator.pause_workflow(workflow_id)
        if result.get("error"):
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=409, detail=result["error"])
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to pause workflow: {exc}")


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(workflow_id: str) -> Dict[str, Any]:
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()
        workflow = coordinator.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        if workflow.get("status") == "cancelled":
            raise HTTPException(
                status_code=409,
                detail="Cancelled workflows must be retried, not resumed directly",
            )
        result = coordinator.resume_workflow(workflow_id)
        if result.get("error"):
            raise HTTPException(status_code=409, detail=result["error"])
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to resume workflow: {exc}")


@router.post("/workflows/{workflow_id}/retry")
async def retry_workflow(workflow_id: str) -> Dict[str, Any]:
    try:
        from src.agents.departments.workflow_coordinator import get_workflow_coordinator

        coordinator = get_workflow_coordinator()
        result = coordinator.retry_workflow(workflow_id)
        if result.get("error"):
            if "not found" in result["error"].lower():
                raise HTTPException(status_code=404, detail=result["error"])
            raise HTTPException(status_code=409, detail=result["error"])
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retry workflow: {exc}")
