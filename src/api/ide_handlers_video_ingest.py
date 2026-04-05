"""
QuantMind Video Ingest Handler

Business logic for queue-backed video ingest operations.
"""

import logging
import os
import re
import threading
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.api.wf1_artifacts import ensure_bundle, iter_strategy_roots
from src.video_ingest.exceptions import JobError
from src.video_ingest.job_queue import JobQueueManager
from src.video_ingest.models import JobOptions, VideoIngestConfig, JobState, JobStatus
from src.video_ingest.processor import VideoIngestProcessor
from src.video_ingest.providers import GeminiCLIProvider

logger = logging.getLogger(__name__)

_runtime_lock = threading.Lock()
_runtime: Dict[str, Any] = {}


def _sanitize_strategy_folder(strategy_name: str) -> str:
    """Normalize strategy folder names for filesystem safety."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (strategy_name or "").strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or "video_ingest"


def _update_job_status(
    job_queue: JobQueueManager,
    job_id: str,
    status,
    progress: int,
    message: str,
) -> None:
    """Forward processor status updates into the persistent job queue."""
    try:
        job_queue.update_job_status(
            job_id,
            status,
            progress=progress,
            log_message=message,
        )
    except Exception as exc:
        logger.debug("Video ingest status update skipped for %s: %s", job_id, exc)


def _load_json_file(path: Path, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return dict(fallback or {})


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_blocking_error(error: Optional[str]) -> Optional[str]:
    if error is None:
        return None
    raw = error if isinstance(error, str) else str(error)
    raw = raw.strip()
    if not raw:
        return None
    normalized = raw.lower()
    if "retryablequotaerror" in normalized or "exhausted your capacity" in normalized:
        return "Provider quota exhausted during analysis. Switch provider/model or wait for quota reset."
    if "authentication" in normalized or "unauthorized" in normalized or "invalid api key" in normalized:
        return "Provider authentication failed. Verify the configured API key and provider settings."
    if "no module named" in normalized or "modulenotfounderror" in normalized:
        return "Provider runtime is missing a required dependency. Repair the provider environment before retrying."
    first_line = next((line.strip() for line in raw.splitlines() if line.strip()), raw)
    return first_line[:277] + "..." if len(first_line) > 280 else first_line


def _strategy_meta_path(options: JobOptions) -> Optional[Path]:
    if not options.artifact_root:
        return None
    return options.artifact_root / ".meta.json"


def _workflow_manifest_path(options: JobOptions) -> Optional[Path]:
    if options.job_manifest_path:
        return options.job_manifest_path
    if options.artifact_root:
        return options.artifact_root / "workflow" / "manifest.json"
    return None


def _job_states_for_manifest(
    *,
    job_ids: list[str],
    current_job_id: Optional[str],
    current_job_state: Optional[str],
    job_queue: Optional[JobQueueManager],
) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for manifest_job_id in job_ids:
        if current_job_id and manifest_job_id == current_job_id and current_job_state:
            statuses[manifest_job_id] = current_job_state.upper()
            continue
        if job_queue is None:
            statuses[manifest_job_id] = "PENDING"
            continue
        try:
            statuses[manifest_job_id] = job_queue.get_job_status(manifest_job_id).status.value.upper()
        except Exception:
            statuses[manifest_job_id] = "PENDING"
    if current_job_id and current_job_state and current_job_id not in statuses:
        statuses[current_job_id] = current_job_state.upper()
    return statuses


def _sync_job_artifact_state(
    *,
    job_id: str,
    options: JobOptions,
    current_job_state: str,
    job_queue: Optional[JobQueueManager] = None,
    error: Optional[str] = None,
) -> None:
    manifest_path = _workflow_manifest_path(options)
    if manifest_path is None:
        return

    workflow_manifest = _load_json_file(
        manifest_path,
        {
            "workflow_id": options.workflow_id,
            "workflow_type": "wf1_creation",
            "strategy_id": options.strategy_id,
            "strategy_family": options.strategy_family,
            "source_bucket": options.source_bucket,
            "current_stage": "video_ingest",
            "status": "pending",
        },
    )
    job_ids = [
        str(manifest_job_id)
        for manifest_job_id in workflow_manifest.get("job_ids", [])
        if str(manifest_job_id).strip()
    ]
    if not job_ids:
        job_ids = [job_id]
        workflow_manifest["job_ids"] = job_ids

    job_states = _job_states_for_manifest(
        job_ids=job_ids,
        current_job_id=job_id,
        current_job_state=current_job_state,
        job_queue=job_queue,
    )
    normalized_states = {state.upper() for state in job_states.values()}
    any_failed = "FAILED" in normalized_states
    all_completed = bool(job_states) and normalized_states == {"COMPLETED"}

    now = datetime.now(timezone.utc).isoformat()
    workflow_manifest["job_statuses"] = job_states
    workflow_manifest["updated_at"] = now
    workflow_manifest["current_stage"] = "research" if all_completed else "video_ingest"
    workflow_manifest["status"] = "failed" if any_failed else "running"
    workflow_manifest["waiting_reason"] = None if any_failed else workflow_manifest["current_stage"]
    if any_failed:
        workflow_manifest["blocking_error"] = _summarize_blocking_error(error) or workflow_manifest.get("blocking_error")
        workflow_manifest["blocking_error_detail"] = error or workflow_manifest.get("blocking_error_detail")
    else:
        workflow_manifest["blocking_error"] = None
        workflow_manifest["blocking_error_detail"] = None
    if all_completed:
        workflow_manifest["video_ingest_completed_at"] = now

    _write_json_file(manifest_path, workflow_manifest)

    meta_path = _strategy_meta_path(options)
    if meta_path is None:
        return

    strategy_meta = _load_json_file(
        meta_path,
        {
            "name": options.strategy_id or "video_ingest",
            "created_at": now,
        },
    )
    strategy_meta.update(
        {
            "updated_at": now,
            "workflow_id": options.workflow_id or strategy_meta.get("workflow_id"),
            "strategy_id": options.strategy_id or strategy_meta.get("strategy_id"),
            "strategy_family": options.strategy_family or strategy_meta.get("strategy_family"),
            "source_bucket": options.source_bucket or strategy_meta.get("source_bucket"),
            "status": "quarantined" if any_failed else "processing",
            "has_video_ingest": all_completed,
        }
    )
    if any_failed:
        strategy_meta["blocking_error"] = _summarize_blocking_error(error) or strategy_meta.get("blocking_error")
        strategy_meta["blocking_error_detail"] = error or strategy_meta.get("blocking_error_detail")
    else:
        strategy_meta.pop("blocking_error", None)
        strategy_meta.pop("blocking_error_detail", None)
    _write_json_file(meta_path, strategy_meta)


def _find_job_artifact_context(job_id: str) -> Optional[JobOptions]:
    for strategy_root in iter_strategy_roots() or []:
        manifest_path = strategy_root / "workflow" / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_json_file(manifest_path)
        job_ids = [str(item) for item in manifest.get("job_ids", []) if str(item).strip()]
        if job_id not in job_ids:
            continue
        return JobOptions(
            workflow_id=manifest.get("workflow_id"),
            strategy_id=manifest.get("strategy_id") or strategy_root.name,
            strategy_family=manifest.get("strategy_family"),
            source_bucket=manifest.get("source_bucket"),
            job_manifest_path=manifest_path,
            artifact_root=strategy_root,
        )
    return None


def _asset_id_for_options(options: JobOptions) -> Optional[str]:
    if not options.strategy_family or not options.source_bucket or not options.strategy_id:
        return None
    return f"strategies/{options.strategy_family}/{options.source_bucket}/{options.strategy_id}"


def _job_to_dict(job: JobStatus) -> Dict[str, Any]:
    payload = job.to_dict() if hasattr(job, "to_dict") else None
    if isinstance(payload, dict):
        return payload
    return {
        "job_id": getattr(job, "job_id", None),
        "status": getattr(getattr(job, "status", None), "value", getattr(job, "status", None)),
        "progress": getattr(job, "progress", None),
        "video_url": getattr(job, "video_url", None),
        "created_at": getattr(getattr(job, "created_at", None), "isoformat", lambda: None)(),
        "updated_at": getattr(getattr(job, "updated_at", None), "isoformat", lambda: None)(),
        "logs": getattr(job, "logs", []),
        "result_path": getattr(job, "result_path", None),
        "error": getattr(job, "error", None),
    }


def _serialize_job_with_context(job: JobStatus, *, job_queue: Optional[JobQueueManager] = None) -> Dict[str, Any]:
    result = _job_to_dict(job)
    job_id = str(result.get("job_id") or getattr(job, "job_id", None) or "")
    job_state = str(result.get("status") or getattr(getattr(job, "status", None), "value", None) or "")
    result["current_stage"] = job_state

    options = _find_job_artifact_context(job_id) if job_id else None
    if options is None:
        fallback_error = getattr(job, "error", None)
        if fallback_error:
            result["blocking_error"] = _summarize_blocking_error(fallback_error)
            result["blocking_error_detail"] = fallback_error
        return result

    manifest = _load_json_file(_workflow_manifest_path(options) or Path("__missing__"))
    meta = _load_json_file(_strategy_meta_path(options) or Path("__missing__"))
    asset_id = _asset_id_for_options(options)

    if job_queue is not None:
        _sync_job_artifact_state(
            job_id=job_id,
            options=options,
            current_job_state=job_state,
            job_queue=job_queue,
            error=job.error,
        )
        manifest = _load_json_file(_workflow_manifest_path(options) or Path("__missing__"))
        meta = _load_json_file(_strategy_meta_path(options) or Path("__missing__"))

    result.update(
        {
            "workflow_id": options.workflow_id or manifest.get("workflow_id"),
            "strategy_id": options.strategy_id or manifest.get("strategy_id"),
            "strategy_folder": options.strategy_id or manifest.get("strategy_id"),
            "strategy_asset_id": asset_id,
            "strategy_family": options.strategy_family or manifest.get("strategy_family"),
            "source_bucket": options.source_bucket or manifest.get("source_bucket"),
            "workflow_status": manifest.get("status"),
            "waiting_reason": manifest.get("waiting_reason"),
            "blocking_error": manifest.get("blocking_error") or meta.get("blocking_error") or job.error,
            "blocking_error_detail": manifest.get("blocking_error_detail") or meta.get("blocking_error_detail") or job.error,
            "strategy_status": meta.get("status"),
            "has_video_ingest": bool(meta.get("has_video_ingest")),
            "current_stage": manifest.get("current_stage") or result["current_stage"],
        }
    )
    return result


def _build_job_processor(processor: VideoIngestProcessor, job_queue: JobQueueManager):
    """Wrap processor output for the queue manager callback contract and hydrate WF1 manifests."""

    def _run(job_id: str, url: str, options: JobOptions):
        try:
            result = processor.process(
                url,
                job_id=job_id,
                options=options,
            )
        except Exception as exc:
            _sync_job_artifact_state(
                job_id=job_id,
                options=options,
                current_job_state="FAILED",
                job_queue=job_queue,
                error=str(exc),
            )
            raise

        _sync_job_artifact_state(
            job_id=job_id,
            options=options,
            current_job_state="COMPLETED",
            job_queue=job_queue,
        )
        return result.timeline

    return _run


def _wire_runtime(
    config: VideoIngestConfig,
    processor: VideoIngestProcessor,
    job_queue: JobQueueManager,
) -> None:
    """Wire callbacks between queue and processor for the current runtime."""
    processor.set_status_callback(
        lambda job_id, status, progress, message: _update_job_status(
            job_queue,
            job_id,
            status,
            progress,
            message,
        )
    )
    job_queue.set_job_processor(_build_job_processor(processor, job_queue))
    job_queue.start()


def refresh_runtime(force: bool = False) -> Tuple[VideoIngestConfig, VideoIngestProcessor, JobQueueManager]:
    """
    Refresh the shared video ingest runtime when provider/config state changes.

    This keeps OpenRouter settings from the Providers panel hot-swappable
    without requiring an API restart.
    """
    with _runtime_lock:
        config = VideoIngestConfig.from_env()
        fingerprint = config.runtime_fingerprint()

        if (
            not force
            and _runtime.get("config") is not None
            and _runtime.get("fingerprint") == fingerprint
            and _runtime.get("processor") is not None
            and _runtime.get("job_queue") is not None
        ):
            job_queue = _runtime["job_queue"]
            job_queue.start()
            return _runtime["config"], _runtime["processor"], job_queue

        existing_queue: JobQueueManager | None = _runtime.get("job_queue")
        queue_can_be_reused = (
            existing_queue is not None
            and existing_queue.db_path == config.job_db_path
            and existing_queue.max_concurrent == config.max_concurrent_jobs
        )

        processor = VideoIngestProcessor(config=config)

        if queue_can_be_reused:
            job_queue = existing_queue
            job_queue.config = config
        else:
            if existing_queue is not None:
                wait_for_shutdown = existing_queue.get_active_job_count() == 0
                existing_queue.stop(wait=wait_for_shutdown)
            job_queue = JobQueueManager(config)

        _wire_runtime(config, processor, job_queue)
        _runtime["fingerprint"] = fingerprint
        _runtime["config"] = config
        _runtime["processor"] = processor
        _runtime["job_queue"] = job_queue
        return config, processor, job_queue


def _get_runtime() -> Tuple[VideoIngestConfig, VideoIngestProcessor, JobQueueManager]:
    """Initialize and return the shared video ingest runtime."""
    return refresh_runtime(force=False)


class VideoIngestAPIHandler:
    """Handler for video ingest operations."""

    def process_video(self, url: str, strategy_name: str, is_playlist: bool = False) -> Dict[str, Any]:
        """Submit a video or playlist to the real ingest queue."""
        _, processor, job_queue = _get_runtime()

        folder_name = _sanitize_strategy_folder(strategy_name)
        bundle = ensure_bundle(
            strategy_name=folder_name,
            is_playlist=is_playlist,
            source_url=url,
        )
        options = JobOptions(
            output_dir=bundle.timeline_dir,
            workflow_id=bundle.workflow_id,
            strategy_id=bundle.strategy_id,
            strategy_family=bundle.strategy_family,
            source_bucket=bundle.source_bucket,
            job_manifest_path=bundle.workflow_manifest_path,
            artifact_root=bundle.root,
        )

        if is_playlist:
            job_ids = [job_id for job_id in processor.create_playlist_jobs(url, job_queue, options=options) if job_id]
            if not job_ids:
                raise RuntimeError("Playlist ingest did not create any jobs")
            self._write_job_manifest(bundle.workflow_manifest_path, job_ids, bundle)
            return {
                "job_id": job_ids[0],
                "job_ids": job_ids,
                "status": "PENDING",
                "strategy_folder": folder_name,
            }

        job_id = job_queue.submit_job(url, options=options)
        self._write_job_manifest(bundle.workflow_manifest_path, [job_id], bundle)
        return {
            "job_id": job_id,
            "status": "PENDING",
            "strategy_folder": folder_name,
        }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status from the persistent job queue."""
        _, _, job_queue = _get_runtime()

        try:
            status = job_queue.get_job_status(job_id)
        except JobError:
            return {}
        return _serialize_job_with_context(status, job_queue=job_queue)

    def list_jobs(self, limit: int = 25, status: Optional[str] = None) -> list[Dict[str, Any]]:
        """List recent persistent ingest jobs enriched with WF1 artifact context."""
        _, _, job_queue = _get_runtime()

        status_filter = None
        if status:
            status_filter = JobState(status.upper())

        jobs = job_queue.list_jobs(status=status_filter, limit=limit)
        return [_serialize_job_with_context(job, job_queue=job_queue) for job in jobs]

    def get_auth_status(self) -> Dict[str, bool]:
        """Get authentication status for available ingest providers."""
        config = VideoIngestConfig.from_env()

        gemini_available = GeminiCLIProvider(
            yolo_mode=config.gemini_yolo_mode,
            api_key=config.gemini_api_key,
        ).is_authenticated()
        qwen_available = bool(config.qwen_api_key or os.getenv("QWEN_API_KEY"))
        if not qwen_available:
            qwen_available = (Path.home() / ".qwen" / "auth.json").exists()
        openrouter_available = bool(config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"))

        return {
            "openrouter": openrouter_available,
            "gemini": gemini_available,
            "qwen": qwen_available,
        }

    def _write_job_manifest(self, workflow_manifest_path: Path, job_ids: list[str], bundle) -> None:
        try:
            payload = json.loads(workflow_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = bundle.as_metadata()

        payload["job_ids"] = job_ids
        payload["job_statuses"] = {job_id: "PENDING" for job_id in job_ids}
        payload["status"] = "running"
        payload["current_stage"] = "video_ingest"
        payload["waiting_reason"] = "video_ingest"
        payload["blocking_error"] = None
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        workflow_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        meta = _load_json_file(
            bundle.meta_path,
            {
                "name": bundle.strategy_id,
                "created_at": payload["updated_at"],
            },
        )
        meta.update(
            {
                "updated_at": payload["updated_at"],
                "workflow_id": bundle.workflow_id,
                "strategy_id": bundle.strategy_id,
                "strategy_family": bundle.strategy_family,
                "source_bucket": bundle.source_bucket,
                "status": "processing",
                "has_video_ingest": False,
            }
        )
        meta.pop("blocking_error", None)
        _write_json_file(bundle.meta_path, meta)
