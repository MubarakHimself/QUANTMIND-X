"""
Cron Job Scheduler for QuantMindX Agents

This module provides a comprehensive cron job scheduling system using APScheduler.

Key Features:
- CronJob dataclass for job definition
- Scheduler class with persistence
- Job registration and deregistration
- Support for interval and cron-style schedules
- Built-in jobs for common tasks

Example:
    ```python
    from src.agents.cron import Scheduler, get_scheduler

    scheduler = get_scheduler()
    await scheduler.start()

    @scheduler.schedule(cron="0 2 * * *")  # Run at 2 AM daily
    async def my_job():
        print("Running scheduled task!")

    # Or register manually
    scheduler.add_job(
        job_id="my-job",
        handler=my_job,
        cron="0 2 * * *",
    )
    ```
"""

from .scheduler import (
    CronJob,
    JobSchedule,
    Scheduler,
    get_scheduler,
    initialize_scheduler,
)

from .jobs import (
    MemoryConsolidationJob,
    SessionCleanupJob,
    EmbeddingSyncJob,
    HealthCheckJob,
)

__all__ = [
    # Scheduler
    "CronJob",
    "JobSchedule",
    "Scheduler",
    "get_scheduler",
    "initialize_scheduler",
    # Built-in jobs
    "MemoryConsolidationJob",
    "SessionCleanupJob",
    "EmbeddingSyncJob",
    "HealthCheckJob",
]
