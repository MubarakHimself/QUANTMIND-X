"""
Dead Zone Workflow Scheduler
============================

Schedules and triggers Workflow 3 at 16:15 GMT daily during Dead Zone.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class DeadZoneScheduler:
    """
    Dead Zone workflow scheduler — triggers Workflow 3 at 16:15 GMT daily.
    """

    DEAD_ZONE_OPEN = "16:00"  # GMT
    WORKFLOW_3_START = "16:15"  # GMT

    def __init__(self):
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        logger.info("DeadZoneScheduler initialized")

    async def start(self) -> None:
        """Start the Dead Zone scheduler."""
        self._running = True
        logger.info("DeadZoneScheduler starting")
        self._loop_task = asyncio.create_task(self._scheduler_loop())

    def stop(self) -> None:
        """Stop the Dead Zone scheduler."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
        logger.info("DeadZoneScheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_run = self._calc_next_run(now)
                sleep_seconds = (next_run - now).total_seconds()

                logger.info(
                    f"DeadZoneScheduler sleeping {sleep_seconds:.0f}s until {next_run.isoformat()}"
                )

                await asyncio.sleep(max(sleep_seconds, 0))

                if not self._running:
                    break

                # Trigger Workflow 3
                await self._trigger_workflow_3()

            except asyncio.CancelledError:
                logger.info("DeadZoneScheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _calc_next_run(self, now: datetime) -> datetime:
        """Calculate next 16:15 GMT trigger time."""
        h, m = 16, 15
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)

        if candidate <= now:
            candidate += timedelta(days=1)

        return candidate

    async def _trigger_workflow_3(self) -> None:
        """Trigger Workflow 3 execution."""
        from src.router.dead_zone_workflow_3 import get_dead_zone_workflow

        logger.info("Triggering Dead Zone Workflow 3")

        try:
            workflow = get_dead_zone_workflow()
            result = await workflow.execute()

            logger.info(
                f"Workflow 3 completed: status={result.status}, "
                f"run_id={result.run_id}"
            )

        except Exception as e:
            logger.error(f"Workflow 3 execution failed: {e}", exc_info=True)


# ============= Singleton Factory =============
_dead_zone_scheduler: Optional[DeadZoneScheduler] = None


def get_dead_zone_scheduler() -> DeadZoneScheduler:
    """Get singleton instance of DeadZoneScheduler."""
    global _dead_zone_scheduler
    if _dead_zone_scheduler is None:
        _dead_zone_scheduler = DeadZoneScheduler()
    return _dead_zone_scheduler
