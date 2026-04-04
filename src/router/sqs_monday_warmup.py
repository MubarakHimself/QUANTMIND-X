"""
SQS Monday Warmup
================

SQS Monday warm-up ramp.
At Monday 05:00 GMT: SQS baseline ramps from 0.75 to 0.60 over 15 minutes post-open.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC4
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    """State of SQS Monday warmup."""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    start_baseline: float = 0.75
    end_baseline: float = 0.60
    duration_minutes: int = 15
    current_baseline: float = 0.75
    is_complete: bool = False


class SqsMondayWarmup:
    """
    SQS Monday warm-up ramp.

    At Monday 05:00 GMT: SQS baseline ramps from 0.75 to 0.60 over 15 minutes post-open.
    """

    SQS_WARMUP_START = 0.75
    SQS_WARMUP_END = 0.60
    WARMUP_DURATION_MINUTES = 15

    def __init__(self):
        self._state = WarmupState()
        self._warmup_task: Optional[asyncio.Task] = None
        logger.info("SqsMondayWarmup initialized")

    async def execute_warmup(self) -> Dict[str, Any]:
        """
        Execute SQS Monday warm-up ramp.

        Returns:
            Warmup state information
        """
        logger.info("Starting SQS Monday warmup ramp")

        self._state = WarmupState(
            started_at=datetime.now(timezone.utc),
            start_baseline=self.SQS_WARMUP_START,
            end_baseline=self.SQS_WARMUP_END,
            duration_minutes=self.WARMUP_DURATION_MINUTES,
            current_baseline=self.SQS_WARMUP_START,
            is_complete=False,
        )

        # Start warmup task
        self._warmup_task = asyncio.create_task(self._run_warmup())

        return {
            "status": "started",
            "start_baseline": self.SQS_WARMUP_START,
            "end_baseline": self.SQS_WARMUP_END,
            "duration_minutes": self.WARMUP_DURATION_MINUTES,
            "started_at": self._state.started_at.isoformat(),
        }

    async def _run_warmup(self) -> None:
        """Run the warmup ramp over 15 minutes."""
        try:
            start_time = datetime.now(timezone.utc)
            end_time = start_time.replace(
                minute=start_time.minute + self.WARMUP_DURATION_MINUTES,
                second=0,
                microsecond=0
            )

            # Calculate total duration
            total_seconds = self.WARMUP_DURATION_MINUTES * 60
            baseline_delta = self.SQS_WARMUP_START - self.SQS_WARMUP_END

            while datetime.now(timezone.utc) < end_time:
                # Calculate current baseline
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                progress = min(elapsed / total_seconds, 1.0)

                current_baseline = self.SQS_WARMUP_START - (baseline_delta * progress)
                self._state.current_baseline = current_baseline

                # Apply to SQS service
                await self._apply_baseline(current_baseline)

                # Log progress every minute
                if int(elapsed) % 60 == 0:
                    logger.info(
                        f"SQS warmup progress: {progress*100:.1f}%, "
                        f"baseline={current_baseline:.4f}"
                    )

                # Wait 10 seconds before next update
                await asyncio.sleep(10)

            # Ensure final baseline is set
            await self._apply_baseline(self.SQS_WARMUP_END)
            self._state.current_baseline = self.SQS_WARMUP_END
            self._state.is_complete = True
            self._state.completed_at = datetime.now(timezone.utc)

            logger.info(
                f"SQS warmup complete: final_baseline={self.SQS_WARMUP_END}"
            )

        except asyncio.CancelledError:
            logger.info("SQS warmup cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in SQS warmup: {e}", exc_info=True)

    async def _apply_baseline(self, baseline: float) -> None:
        """
        Apply baseline to SQS service.

        Updates the SQS (Session Quality Score) baseline in the risk engine.
        The baseline affects position sizing and risk calculations during the
        Monday open period.
        """
        try:
            # Import the SQS engine to apply the baseline
            from src.risk.sqs_engine import get_sqs_engine

            engine = get_sqs_engine()
            await engine.update_baseline(baseline)

            logger.debug(f"SQS baseline applied: {baseline:.4f}")

        except ImportError:
            logger.warning("SQS engine not available, baseline not applied")
        except Exception as e:
            logger.error(f"Error applying SQS baseline {baseline}: {e}", exc_info=True)
            # Don't raise - we don't want to fail the warmup for baseline errors

    def get_warmup_status(self) -> Dict[str, Any]:
        """Get current warmup status."""
        return {
            "is_started": self._state.started_at is not None,
            "is_complete": self._state.is_complete,
            "current_baseline": round(self._state.current_baseline, 4),
            "started_at": self._state.started_at.isoformat() if self._state.started_at else None,
            "completed_at": self._state.completed_at.isoformat() if self._state.completed_at else None,
        }

    async def cancel_warmup(self) -> None:
        """Cancel the warmup if running."""
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass
            logger.info("SQS warmup cancelled")


# ============= Singleton Factory =============
_warmup_instance: Optional[SqsMondayWarmup] = None


def get_sqs_monday_warmup() -> SqsMondayWarmup:
    """Get singleton instance of SqsMondayWarmup."""
    global _warmup_instance
    if _warmup_instance is None:
        _warmup_instance = SqsMondayWarmup()
    return _warmup_instance
