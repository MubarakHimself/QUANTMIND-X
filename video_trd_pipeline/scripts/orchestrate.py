#!/usr/bin/env python3
"""
orchestrate.py — Autonomous orchestrator for the video → TRD pipeline.

Manages:
  - Parallel per-video sub-processes (configurable concurrency)
  - Progress tracking + resume on restart
  - Per-video + aggregate status reporting
  - Auto-retry on failure (configurable attempts)
  - Signals: graceful shutdown on SIGINT/SIGTERM

Usage:
  python orchestrate.py --batch ~/Desktop/trading_videos.txt
  python orchestrate.py --playlist https://youtube.com/playlist?list=...
  python orchestrate.py --url https://youtube.com/watch?v=...

Each video gets its own folder under --output:
  video_out/<video_id>/
    video/           — downloaded mp4
    audio.mp3        — extracted audio
    frames/          — extracted frames
    timeline_gemini.json
    timeline_qwen.json
    TRD.md
    _status.json     — per-video stage status
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent.parent.resolve()
PIPELINE_SCRIPT = BASE_DIR / "scripts" / "video_pipeline.py"
VIDEO_IN    = BASE_DIR / "video_in"
VIDEO_OUT   = BASE_DIR / "video_out"
LOG_DIR     = BASE_DIR / "logs"

for d in [VIDEO_IN, VIDEO_OUT, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"orchestrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrate")


# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_CONCURRENCY = 2     # parallel video workers
DEFAULT_RETRIES     = 2     # retry attempts per video on stage failure
MAX_CONCURRENT_ENV = "PIPELINE_MAX_CONCURRENT"


@dataclass
class VideoJob:
    url:       str
    folder:    Path
    status:    str = "pending"   # pending | running | completed | failed | skipped
    attempts:  int = 0
    error:     str = ""
    started:   str = ""
    finished:  str = ""

    def to_dict(self):
        return asdict(self)


class GlobalStatus(str, Enum):
    IDLE       = "idle"
    RUNNING    = "running"
    PAUSED     = "paused"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class OrchestratorState:
    status:       GlobalStatus = GlobalStatus.IDLE
    jobs:         dict[str, VideoJob] = None   # url → VideoJob
    completed:    int = 0
    failed:       int = 0
    total:        int = 0
    started_at:   str = ""
    finished_at:  str = ""

    def __post_init__(self):
        if self.jobs is None:
            self.jobs = {}

    def to_dict(self):
        return {
            "status":       self.status.value,
            "total":        self.total,
            "completed":    self.completed,
            "failed":       self.failed,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "jobs":         {url: j.to_dict() for url, j in self.jobs.items()},
        }


# ─── State persistence ───────────────────────────────────────────────────────

def state_file(output_dir: Path) -> Path:
    return output_dir / "_orchestrator_state.json"


def load_state(output_dir: Path) -> Optional[OrchestratorState]:
    sf = state_file(output_dir)
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text())
        jobs = {url: VideoJob(**jd) for url, jd in data.get("jobs", {}).items()}
        state = OrchestratorState(
            status=GlobalStatus(data.get("status", "idle")),
            jobs=jobs,
            completed=data.get("completed", 0),
            failed=data.get("failed", 0),
            total=data.get("total", 0),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
        )
        return state
    except Exception as e:
        log.warning(f"Could not load state file: {e}")
        return None


def save_state(output_dir: Path, state: OrchestratorState):
    sf = state_file(output_dir)
    sf.write_text(json.dumps(state.to_dict(), indent=2))


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str:
    import re
    m = re.search(r'[?&]v=([^&]+)', url)
    if m:
        return m.group(1)
    m = re.search(r'youtu\.be/([^?]+)', url)
    if m:
        return m.group(1)
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', url)
    return safe[:80]


def get_batch_urls(file_path: Path) -> list[str]:
    urls = []
    for line in file_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def get_playlist_urls(playlist_url: str) -> list[str]:
    import subprocess
    result = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "url", playlist_url],
        capture_output=True, text=True, timeout=120,
    )
    urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return urls


def run_pipeline_worker(video_url: str, folder: Path, log_dir: Path,
                        max_retries: int) -> tuple[str, str]:
    """
    Run the video_pipeline.py as a subprocess for one video.
    Returns (status, error).
    """
    worker_log = log_dir / f"worker_{get_video_id(video_url)}.log"

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--url", video_url,
        "--output", str(folder),
    ]

    env = dict(os.environ)

    log.info(f"Starting worker: {video_url}")
    log.info(f"Worker log: {worker_log}")

    with open(worker_log, "w") as wf:
        proc = subprocess.Popen(
            cmd,
            stdout=wf,
            stderr=wf,
            env=env,
        )

    try:
        exit_code = proc.wait(timeout=3600)  # 1 hour per video max
        if exit_code == 0:
            return ("completed", "")
        else:
            # Read last lines of worker log for error
            try:
                last_lines = worker_log.read_text().splitlines()[-20:]
                error_summary = "\n".join(last_lines)[-500:]
            except Exception:
                error_summary = f"Exit code: {exit_code}"
            return ("failed", error_summary)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return ("failed", f"Worker timed out after 3600s for {video_url}")


# ─── Parallel runner ─────────────────────────────────────────────────────────

def run_parallel(jobs: list[VideoJob], concurrency: int,
                 output_dir: Path, max_retries: int) -> list[VideoJob]:
    """
    Process jobs with up to `concurrency` parallel workers.
    Tracks state to disk after each job completes (resume-safe).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    state = load_state(output_dir)
    if state and state.jobs:
        # Resume: update our jobs dict from saved state
        for job in jobs:
            if job.url in state.jobs:
                saved = state.jobs[job.url]
                job.status   = saved.status
                job.attempts = saved.attempts
                job.error    = saved.error
                job.started  = saved.started
                job.finished = saved.finished
        log.info(f"Resuming — {len([j for j in jobs if j.status in ('completed','failed')])} already done")

    now_ts = datetime.now().isoformat()

    pending = [j for j in jobs if j.status == "pending"]
    active  = [j for j in jobs if j.status == "running"]

    def _do(job: VideoJob) -> VideoJob:
        job.status   = "running"
        job.started   = datetime.now().isoformat()
        job.attempts += 1
        # Persist running state immediately
        _persist_jobs(output_dir, jobs)
        _persist_global_state(output_dir, GlobalStatus.RUNNING, jobs, started_at=now_ts)

        status, error = run_pipeline_worker(job.url, job.folder, LOG_DIR, max_retries)

        job.finished = datetime.now().isoformat()
        if status == "completed":
            job.status = "completed"
            job.error  = ""
        else:
            if job.attempts < max_retries + 1:
                job.status = "pending"   # will retry
                job.error  = f"Attempt {job.attempts} failed: {error[:200]}"
                log.warning(f"Attempt {job.attempts}/{max_retries+1} failed for {job.url}: {error[:200]}")
            else:
                job.status = "failed"
                job.error  = f"All {job.attempts} attempts failed: {error[:200]}"
                log.error(f"All attempts exhausted for {job.url}: {error[:200]}")

        # Persist after each job
        _persist_jobs(output_dir, jobs)
        return job

    # Run with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_do, job): job for job in pending + active}
        for future in as_completed(futures):
            try:
                updated_job = future.result()
            except Exception as e:
                log.error(f"Worker exception: {e}")

    return jobs


def _persist_jobs(output_dir: Path, jobs: list[VideoJob]):
    """Persist just the job list."""
    sf = state_file(output_dir)
    data = json.loads(sf.read_text()) if sf.exists() else {}
    data["jobs"] = {j.url: j.to_dict() for j in jobs}
    sf.write_text(json.dumps(data, indent=2))


def _persist_global_state(output_dir: Path, status: GlobalStatus,
                           jobs: list[VideoJob], started_at: str = ""):
    completed = sum(1 for j in jobs if j.status == "completed")
    failed    = sum(1 for j in jobs if j.status == "failed")
    total     = len(jobs)

    sf = state_file(output_dir)
    data = json.loads(sf.read_text()) if sf.exists() else {}
    data.update({
        "status":    status.value,
        "total":     total,
        "completed": completed,
        "failed":    failed,
        "started_at": started_at or data.get("started_at", ""),
    })
    sf.write_text(json.dumps(data, indent=2))


# ─── Graceful shutdown ───────────────────────────────────────────────────────

_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    log.warning(f"Received signal {signum} — will finish current jobs then exit")
    _shutdown_requested = True


# ─── Progress reporter ────────────────────────────────────────────────────────

def print_progress(jobs: list[VideoJob]):
    total    = len(jobs)
    done     = sum(1 for j in jobs if j.status == "completed")
    fail     = sum(1 for j in jobs if j.status == "failed")
    running  = sum(1 for j in jobs if j.status == "running")
    pending  = sum(1 for j in jobs if j.status == "pending")
    print(f"\n{'='*60}")
    print(f"  Progress: {done}/{total} completed | {fail} failed | {running} running | {pending} pending")
    print(f"{'='*60}")
    for j in jobs:
        icon = {"completed": "✅", "failed": "❌", "running": "🔄", "pending": "⏳"}.get(j.status, "?")
        short_id = get_video_id(j.url)[:40]
        print(f"  {icon} [{j.status.upper():<10}] {short_id}")
        if j.error and j.status == "failed":
            print(f"       ERROR: {j.error[:80]}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="orchestrate — Autonomous video → TRD orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url",      help="Single YouTube URL")
    group.add_argument("--playlist", help="YouTube playlist URL")
    group.add_argument("--batch",    type=Path, help="Text file with one URL per line")
    parser.add_argument("--output",  type=Path, default=VIDEO_OUT,
                        help=f"Output base directory (default: {VIDEO_OUT})")
    parser.add_argument("-j", "--concurrency", type=int,
                        default=int(os.environ.get(MAX_CONCURRENT_ENV, DEFAULT_CONCURRENCY)),
                        help="Parallel video workers (default: 2)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help=f"Max retry attempts per video (default: {DEFAULT_RETRIES})")
    parser.add_argument("--watch", action="store_true",
                        help="Re-run and watch progress indefinitely")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Resolve URLs
    if args.url:
        urls = [args.url]
    elif args.playlist:
        log.info(f"Fetching playlist: {args.playlist}")
        urls = get_playlist_urls(args.playlist)
    else:
        urls = get_batch_urls(args.batch)

    if not urls:
        log.error("No URLs found")
        sys.exit(1)

    log.info(f"Orchestrator starting — {len(urls)} video(s), concurrency={args.concurrency}")
    args.output.mkdir(parents=True, exist_ok=True)

    # Build job list (skip already-completed if resuming)
    existing_state = load_state(args.output)
    jobs = []
    for url in urls:
        if existing_state and url in existing_state.jobs:
            # Don't re-run completed videos on resume
            job = existing_state.jobs[url]
            if job.status == "completed":
                log.info(f"Skipping already-completed: {url}")
                jobs.append(job)
                continue
            elif job.status == "running":
                # Treat as pending (worker may have died)
                job.status = "pending"
        folder = args.output / get_video_id(url)
        jobs.append(VideoJob(url=url, folder=folder))

    now_ts = datetime.now().isoformat()

    if existing_state:
        state = existing_state
        state.status = GlobalStatus.RUNNING
        save_state(args.output, state)
    else:
        _persist_global_state(args.output, GlobalStatus.RUNNING, jobs, started_at=now_ts)

    try:
        run_parallel(jobs, args.concurrency, args.output, args.retries)
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt — pausing (state saved, can resume)")
        _persist_global_state(args.output, GlobalStatus.PAUSED, jobs)

    # Final summary
    completed = sum(1 for j in jobs if j.status == "completed")
    failed    = sum(1 for j in jobs if j.status == "failed")

    print_progress(jobs)

    if failed > 0:
        log.warning(f"DONE with {failed} failures — re-run with same args to retry")
    else:
        log.info(f"ALL {completed} VIDEOS COMPLETED SUCCESSFULLY")

    _persist_global_state(
        args.output,
        GlobalStatus.DONE if failed == 0 else GlobalStatus.FAILED,
        jobs,
    )

    summary_path = args.output / "_orchestrator_summary.json"
    summary_path.write_text(json.dumps({
        "total": len(jobs),
        "completed": completed,
        "failed": failed,
        "jobs": [j.to_dict() for j in jobs],
    }, indent=2))
    log.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
