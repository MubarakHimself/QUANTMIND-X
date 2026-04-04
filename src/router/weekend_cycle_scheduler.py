"""
Weekend Cycle Scheduler
=======================

Schedules and triggers Workflow 4 at:
- Friday 21:00 GMT: Start weekend cycle (Friday Analysis)
- Saturday 06:00 GMT: Saturday steps begin
- Sunday 06:00 GMT: Sunday steps begin
- Monday 05:00 GMT: Roster deployment

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WeekendCycleScheduler:
    """
    Weekend Update Cycle scheduler.

    Weekly triggers at:
    - Friday 21:00 GMT: Start weekend cycle (Friday Analysis)
    - Saturday 06:00 GMT: Saturday steps begin
    - Sunday 06:00 GMT: Sunday steps begin
    - Monday 05:00 GMT: Roster deployment
    """

    FRIDAY_START = "21:00"
    SATURDAY_START = "06:00"
    SUNDAY_START = "06:00"
    MONDAY_DEPLOY = "05:00"

    STEP_TRIGGERS = {
        "friday_analysis": (FRIDAY_START, "Friday"),
        "saturday_refinement": (SATURDAY_START, "Saturday"),
        "saturday_wfa": ("09:00", "Saturday"),
        "saturday_hmm_retrain": ("12:00", "Saturday"),
        "sunday_calibration": (SUNDAY_START, "Sunday"),
        "sunday_spread_profiles": ("09:00", "Sunday"),
        "sunday_sqs_refresh": ("12:00", "Sunday"),
        "sunday_kelly_calibration": ("15:00", "Sunday"),
        "monday_roster_deploy": (MONDAY_DEPLOY, "Monday"),
    }

    def __init__(self):
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._step_tasks: Dict[str, Optional[asyncio.Task]] = {}
        logger.info("WeekendCycleScheduler initialized")

    async def start(self) -> None:
        """Start the Weekend Cycle scheduler."""
        self._running = True
        logger.info("WeekendCycleScheduler starting")
        self._loop_task = asyncio.create_task(self._scheduler_loop())

    def stop(self) -> None:
        """Stop the Weekend Cycle scheduler."""
        self._running = False

        # Cancel all step tasks
        for task in self._step_tasks.values():
            if task and not task.done():
                task.cancel()

        if self._loop_task:
            self._loop_task.cancel()

        logger.info("WeekendCycleScheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop - checks every minute for triggers."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Check if any step should trigger
                await self._check_and_trigger_steps(now)

                # Sleep 60 seconds before next check
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                logger.info("WeekendCycleScheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _check_and_trigger_steps(self, now: datetime) -> None:
        """Check if any workflow step should trigger at the current time."""
        current_day = now.strftime("%A")
        current_time = now.strftime("%H:%M")

        for step_name, (trigger_time, trigger_day) in self.STEP_TRIGGERS.items():
            if current_day == trigger_day and current_time == trigger_time:
                # Check if already triggered recently (within last 10 minutes)
                if not self._should_trigger(step_name, now):
                    continue

                logger.info(f"Triggering weekend cycle step: {step_name}")
                await self._trigger_step(step_name)

    def _should_trigger(self, step_name: str, now: datetime) -> bool:
        """Check if step should trigger (not recently triggered)."""
        # Simple check - in production would track last trigger time
        # For now, always allow trigger
        return True

    async def _trigger_step(self, step_name: str) -> None:
        """Trigger a specific workflow step."""
        try:
            from src.router.weekend_update_cycle_workflow import get_weekend_update_cycle_workflow

            workflow = get_weekend_update_cycle_workflow()

            # Create task to run the step
            task = asyncio.create_task(
                self._run_step_with_timeout(workflow, step_name)
            )
            self._step_tasks[step_name] = task

            logger.info(f"Weekend cycle step triggered: {step_name}")

        except Exception as e:
            logger.error(f"Error triggering step {step_name}: {e}", exc_info=True)

    async def _run_step_with_timeout(
        self,
        workflow,
        step_name: str,
        timeout_seconds: int = 3600
    ) -> None:
        """Run a step with timeout."""
        try:
            await asyncio.wait_for(
                workflow.execute_single_step(step_name),
                timeout=timeout_seconds
            )
            logger.info(f"Step {step_name} completed")
        except asyncio.TimeoutError:
            logger.error(f"Step {step_name} timed out after {timeout_seconds}s")
        except Exception as e:
            logger.error(f"Step {step_name} failed: {e}", exc_info=True)

    async def trigger_friday_analysis(self) -> None:
        """Manually trigger Friday analysis."""
        logger.info("Manual trigger: Friday analysis")
        await self._trigger_step("friday_analysis")

    async def trigger_saturday_steps(self) -> None:
        """Manually trigger all Saturday steps."""
        logger.info("Manual trigger: Saturday steps")
        saturday_steps = ["saturday_refinement", "saturday_wfa", "saturday_hmm_retrain"]
        for step in saturday_steps:
            await self._trigger_step(step)

    async def trigger_sunday_steps(self) -> None:
        """Manually trigger all Sunday steps."""
        logger.info("Manual trigger: Sunday steps")
        sunday_steps = [
            "sunday_calibration", "sunday_spread_profiles",
            "sunday_sqs_refresh", "sunday_kelly_calibration"
        ]
        for step in sunday_steps:
            await self._trigger_step(step)

    async def trigger_monday_deploy(self) -> None:
        """Manually trigger Monday roster deployment."""
        logger.info("Manual trigger: Monday roster deployment")
        await self._trigger_step("monday_roster_deploy")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "friday_start": self.FRIDAY_START,
            "saturday_start": self.SATURDAY_START,
            "sunday_start": self.SUNDAY_START,
            "monday_deploy": self.MONDAY_DEPLOY,
            "active_tasks": {
                name: task.done() if task else None
                for name, task in self._step_tasks.items()
            },
        }


# ============= Singleton Factory =============
_scheduler_instance: Optional[WeekendCycleScheduler] = None


def get_weekend_cycle_scheduler() -> WeekendCycleScheduler:
    """Get singleton instance of WeekendCycleScheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = WeekendCycleScheduler()
    return _scheduler_instance
