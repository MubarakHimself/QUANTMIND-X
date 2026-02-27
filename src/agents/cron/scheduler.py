"""
Cron Job Scheduler using APScheduler

This module provides a comprehensive cron job scheduling system with:
- APScheduler-based scheduling engine
- Job persistence to JSON
- Job status tracking
- Support for interval and cron-style schedules
- Async job execution
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a cron job."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class JobSchedule:
    """
    Schedule configuration for a cron job.

    Use either:
    - cron: Cron expression (e.g., "0 2 * * *")
    - interval: Seconds between runs
    - job_time: Specific time to run (once)

    Attributes:
        cron: Cron expression (standard 5-field format)
        interval: Seconds between runs
        job_time: ISO format datetime for one-time execution
        timezone: Timezone for cron expressions
    """

    cron: Optional[str] = None
    interval: Optional[int] = None
    job_time: Optional[str] = None
    timezone: str = "UTC"

    def validate(self) -> bool:
        """Validate the schedule configuration."""
        count = sum([
            self.cron is not None,
            self.interval is not None,
            self.job_time is not None,
        ])
        return count == 1


@dataclass
class CronJob:
    """
    Represents a scheduled cron job.

    Attributes:
        id: Unique job identifier
        name: Human-readable job name
        description: Job description
        handler: Async function to execute
        schedule: Job schedule configuration
        enabled: Whether the job is enabled
        last_run: Last execution timestamp
        next_run: Next scheduled execution timestamp
        last_status: Last execution status
        last_error: Last error message (if failed)
        run_count: Total number of runs
        metadata: Additional job metadata
    """

    id: str
    name: str
    handler: Callable[[], Awaitable[Any]]
    schedule: JobSchedule
    description: str = ""
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    last_status: Optional[JobStatus] = None
    last_error: Optional[str] = None
    run_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Remove handler as it's not serializable
        data.pop("handler", None)
        # Convert schedule to dict
        if isinstance(data.get("schedule"), JobSchedule):
            data["schedule"] = asdict(self.schedule)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], handler: Callable[[], Awaitable[Any]]) -> "CronJob":
        """Create from dictionary."""
        schedule_data = data.get("schedule", {})
        schedule = JobSchedule(**schedule_data) if isinstance(schedule_data, dict) else None

        return cls(
            id=data["id"],
            name=data["name"],
            handler=handler,
            schedule=schedule or JobSchedule(),
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            last_status=JobStatus(data["last_status"]) if data.get("last_status") else None,
            last_error=data.get("last_error"),
            run_count=data.get("run_count", 0),
            metadata=data.get("metadata", {}),
        )


class Scheduler:
    """
    Cron job scheduler using APScheduler.

    Features:
    - Job registration and deregistration
    - Job persistence to JSON
    - Job status tracking
    - Support for interval and cron-style schedules
    - Async job execution
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        timezone: str = "UTC",
    ):
        """
        Initialize the scheduler.

        Args:
            persistence_path: Path for job persistence (JSON file)
            timezone: Default timezone for cron expressions
        """
        self._jobs: Dict[str, CronJob] = {}
        self._persistence_path = persistence_path or "./data/cron_jobs.json"
        self._timezone = timezone
        self._scheduler = None
        self._running = False
        self._handlers: Dict[str, str] = {}  # Maps handler name to job_id

        # Ensure persistence directory exists
        Path(self._persistence_path).parent.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler

            self._scheduler = AsyncIOScheduler(timezone=self._timezone)
            self._scheduler.start()
            self._running = True

            # Load and schedule persisted jobs
            await self._load_jobs()
            await self._schedule_all_jobs()

            logger.info("Cron scheduler started")

        except ImportError:
            logger.error("APScheduler not installed. Run: pip install apscheduler")
            raise
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def stop(self):
        """Stop the scheduler."""
        if not self._running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=True)

        self._running = False
        await self._save_jobs()

        logger.info("Cron scheduler stopped")

    def add_job(
        self,
        job_id: str,
        handler: Callable[[], Awaitable[Any]],
        *,
        name: str,
        cron: Optional[str] = None,
        interval: Optional[int] = None,
        job_time: Optional[str] = None,
        description: str = "",
        enabled: bool = True,
        metadata: Dict[str, Any] = None,
    ) -> CronJob:
        """
        Add a new job to the scheduler.

        Args:
            job_id: Unique job identifier
            handler: Async function to execute
            name: Human-readable job name
            cron: Cron expression (e.g., "0 2 * * *")
            interval: Seconds between runs
            job_time: ISO format datetime for one-time execution
            description: Job description
            enabled: Whether the job is initially enabled
            metadata: Additional job metadata

        Returns:
            The created CronJob

        Raises:
            ValueError: If schedule configuration is invalid
        """
        if job_id in self._jobs:
            raise ValueError(f"Job {job_id} already exists")

        schedule = JobSchedule(cron=cron, interval=interval, job_time=job_time)

        if not schedule.validate():
            raise ValueError("Must specify exactly one of: cron, interval, or job_time")

        job = CronJob(
            id=job_id,
            name=name,
            handler=handler,
            schedule=schedule,
            description=description,
            enabled=enabled,
            metadata=metadata or {},
        )

        self._jobs[job_id] = job

        # Store handler reference for persistence
        handler_name = handler.__name__ if hasattr(handler, "__name__") else str(id(handler))
        self._handlers[handler_name] = job_id

        # Schedule if enabled and scheduler is running
        if enabled and self._running:
            asyncio.create_task(self._schedule_job(job))

        # Persist
        asyncio.create_task(self._save_jobs())

        logger.info(f"Added job {job_id}: {name}")

        return job

    def schedule(
        self,
        *,
        cron: Optional[str] = None,
        interval: Optional[int] = None,
        job_time: Optional[str] = None,
        job_id: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        enabled: bool = True,
        metadata: Dict[str, Any] = None,
    ) -> Callable:
        """
        Decorator to register a job.

        Args:
            cron: Cron expression
            interval: Seconds between runs
            job_time: ISO format datetime
            job_id: Custom job ID (auto-generated if not provided)
            name: Job name (defaults to function name)
            description: Job description
            enabled: Whether initially enabled
            metadata: Additional metadata

        Returns:
            Decorator function

        Example:
            ```python
            @scheduler.schedule(cron="0 2 * * *", name="Nightly Task")
            async def my_job():
                print("Running at 2 AM!")
            ```
        """
        def decorator(func: Callable[[], Awaitable[Any]]) -> Callable[[], Awaitable[Any]]:
            job = self.add_job(
                job_id=job_id or f"job_{uuid4().hex[:12]}",
                handler=func,
                name=name or func.__name__,
                cron=cron,
                interval=interval,
                job_time=job_time,
                description=description,
                enabled=enabled,
                metadata=metadata,
            )
            return func

        return decorator

    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.

        Args:
            job_id: ID of job to remove

        Returns:
            True if job was found and removed
        """
        if job_id not in self._jobs:
            return False

        # Unschedule if running
        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass

        del self._jobs[job_id]

        # Persist
        asyncio.create_task(self._save_jobs())

        logger.info(f"Removed job {job_id}")
        return True

    def enable_job(self, job_id: str) -> bool:
        """Enable a job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.enabled = True

        if self._running:
            asyncio.create_task(self._schedule_job(job))

        asyncio.create_task(self._save_jobs())

        logger.info(f"Enabled job {job_id}")
        return True

    def disable_job(self, job_id: str) -> bool:
        """Disable a job."""
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.enabled = False

        if self._scheduler:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:
                pass

        asyncio.create_task(self._save_jobs())

        logger.info(f"Disabled job {job_id}")
        return True

    def get_job(self, job_id: str) -> Optional[CronJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[CronJob]:
        """Get all jobs."""
        return list(self._jobs.values())

    def get_enabled_jobs(self) -> List[CronJob]:
        """Get all enabled jobs."""
        return [j for j in self._jobs.values() if j.enabled]

    async def trigger_job(self, job_id: str) -> bool:
        """
        Trigger a job immediately.

        Args:
            job_id: ID of job to trigger

        Returns:
            True if job was triggered
        """
        job = self._jobs.get(job_id)
        if not job or not job.enabled:
            return False

        asyncio.create_task(self._execute_job(job))

        logger.info(f"Triggered job {job_id}")
        return True

    async def _schedule_job(self, job: CronJob):
        """Schedule a job with APScheduler."""
        if not self._scheduler:
            return

        # Remove existing job if any
        try:
            self._scheduler.remove_job(job.id)
        except Exception:
            pass

        # Add new schedule
        try:
            if job.schedule.cron:
                from apscheduler.triggers.cron import CronTrigger

                trigger = CronTrigger.from_crontab(job.schedule.cron, timezone=self._timezone)
                self._scheduler.add_job(
                    self._execute_job,
                    trigger=trigger,
                    id=job.id,
                    args=[job],
                )

            elif job.schedule.interval:
                from apscheduler.triggers.interval import IntervalTrigger

                trigger = IntervalTrigger(seconds=job.schedule.interval, timezone=self._timezone)
                self._scheduler.add_job(
                    self._execute_job,
                    trigger=trigger,
                    id=job.id,
                    args=[job],
                )

            elif job.schedule.job_time:
                from apscheduler.triggers.date import DateTrigger

                trigger = DateTrigger(
                    run_time=datetime.fromisoformat(job.schedule.job_time),
                    timezone=self._timezone,
                )
                self._scheduler.add_job(
                    self._execute_job,
                    trigger=trigger,
                    id=job.id,
                    args=[job],
                )

            # Update next_run
            job_data = self._scheduler.get_job(job.id)
            if job_data and job_data.next_run_time:
                job.next_run = job_data.next_run_time.isoformat()

        except Exception as e:
            logger.error(f"Failed to schedule job {job.id}: {e}")

    async def _schedule_all_jobs(self):
        """Schedule all enabled jobs."""
        for job in self.get_enabled_jobs():
            await self._schedule_job(job)

    async def _execute_job(self, job: CronJob):
        """Execute a job and update its status."""
        job.last_run = datetime.now(timezone.utc).isoformat()
        job.last_status = JobStatus.RUNNING

        try:
            logger.info(f"Executing job {job.id}: {job.name}")

            result = await job.handler()

            job.last_status = JobStatus.COMPLETED
            job.run_count += 1
            job.last_error = None

            logger.info(f"Job {job.id} completed successfully")

        except Exception as e:
            job.last_status = JobStatus.FAILED
            job.last_error = str(e)
            logger.error(f"Job {job.id} failed: {e}", exc_info=True)

        # Update next_run
        if self._scheduler:
            job_data = self._scheduler.get_job(job.id)
            if job_data and job_data.next_run_time:
                job.next_run = job_data.next_run_time.isoformat()

        # Persist
        await self._save_jobs()

    async def _save_jobs(self):
        """Persist jobs to JSON file."""
        try:
            data = {
                "timezone": self._timezone,
                "jobs": {
                    job_id: {
                        **job.to_dict(),
                        "handler_name": job.handler.__name__ if hasattr(job.handler, "__name__") else None,
                    }
                    for job_id, job in self._jobs.items()
                },
            }

            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    async def _load_jobs(self):
        """Load jobs from JSON file."""
        try:
            path = Path(self._persistence_path)
            if not path.exists():
                return

            with open(path, "r") as f:
                data = json.load(f)

            # Jobs are loaded but need handlers to be re-registered
            # The handlers will be connected by name
            for job_id, job_data in data.get("jobs", {}).items():
                logger.debug(f"Found persisted job {job_id}: {job_data.get('name')}")
                # Jobs without handlers are stored but not scheduled
                # They need to be re-registered with their handlers

            logger.info(f"Loaded {len(data.get('jobs', {}))} persisted jobs")

        except Exception as e:
            logger.error(f"Failed to load jobs: {e}")


# Global scheduler instance
_global_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get the global scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = Scheduler()
    return _global_scheduler


async def initialize_scheduler(
    persistence_path: Optional[str] = None,
    timezone: str = "UTC",
) -> Scheduler:
    """
    Initialize the global scheduler.

    Args:
        persistence_path: Path for job persistence
        timezone: Default timezone

    Returns:
        Initialized and started scheduler
    """
    global _global_scheduler

    _global_scheduler = Scheduler(
        persistence_path=persistence_path,
        timezone=timezone,
    )

    await _global_scheduler.start()

    return _global_scheduler
