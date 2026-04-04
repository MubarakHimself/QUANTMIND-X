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
from typing import Any, Dict, Tuple

from src.api.wf1_artifacts import ensure_bundle
from src.video_ingest.exceptions import JobError
from src.video_ingest.job_queue import JobQueueManager
from src.video_ingest.models import JobOptions, VideoIngestConfig
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


def _build_job_processor(processor: VideoIngestProcessor):
    """Wrap processor output for the queue manager callback contract."""
    return lambda job_id, url, options: processor.process(  # noqa: E731
        url,
        job_id=job_id,
        options=options,
    ).timeline


def _get_runtime() -> Tuple[VideoIngestConfig, VideoIngestProcessor, JobQueueManager]:
    """Initialize and return the shared video ingest runtime."""
    with _runtime_lock:
        config = _runtime.get("config")
        if config is None:
            config = VideoIngestConfig.from_env()
            _runtime["config"] = config

        processor = _runtime.get("processor")
        if processor is None:
            processor = VideoIngestProcessor(config=config)
            _runtime["processor"] = processor

        job_queue = _runtime.get("job_queue")
        if job_queue is None:
            job_queue = JobQueueManager(config)
            _runtime["job_queue"] = job_queue

        if not _runtime.get("status_callback_registered"):
            processor.set_status_callback(
                lambda job_id, status, progress, message: _update_job_status(
                    job_queue,
                    job_id,
                    status,
                    progress,
                    message,
                )
            )
            _runtime["status_callback_registered"] = True

        if not _runtime.get("job_processor_registered"):
            job_queue.set_job_processor(_build_job_processor(processor))
            _runtime["job_processor_registered"] = True

        job_queue.start()
        return config, processor, job_queue


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

        result = status.to_dict()
        result["current_stage"] = status.status.value
        return result

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
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        workflow_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
