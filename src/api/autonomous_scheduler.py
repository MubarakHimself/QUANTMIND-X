"""
Autonomous Overnight Research Scheduler

Fires a full research cycle on a configurable daily schedule (default 02:00 UTC).
Sends STRATEGY_DISPATCH department mail to Development for each hypothesis that
passes the confidence threshold (>= 0.75).

Weekend mode: Saturday at 00:00 UTC runs an extended scan.
"""
import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Deque, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SCHEDULE_TIME = "02:00"          # HH:MM UTC
WEEKEND_SCHEDULE_TIME = "00:00"          # Saturday 00:00 UTC
MAX_RUN_HISTORY = 50
CONFIDENCE_THRESHOLD = 0.75
RESEARCH_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

DAILY_TASK_TEXT = (
    "Daily market scan: Generate hypotheses for major forex pairs "
    "(EURUSD, GBPUSD, USDJPY, XAUUSD). Analyse recent price action, news sentiment, "
    "and technical signals. Generate TRDs for high-confidence opportunities "
    "(>0.75 confidence)."
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ScheduledRun:
    run_id: str
    trigger: str               # "scheduled" | "manual"
    status: str                # "queued" | "running" | "completed" | "failed"
    stage: str                 # "research" | "trd_dispatched" | "done"
    started_at: str            # ISO timestamp
    completed_at: Optional[str]
    hypotheses_count: int
    dispatched_count: int
    symbols: List[str]
    error: Optional[str]
    is_weekend_run: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_job_card(self) -> dict:
        """Format for FlowForge job card display."""
        status_map = {
            "completed": "COMPLETED",
            "failed": "FAILED",
            "running": "PENDING",
            "queued": "PENDING",
        }
        stage_map = {
            "research": "research",
            "trd_dispatched": "development",
            "done": "done",
        }
        dt_label = self.started_at[:16].replace("T", " ") + " UTC" if self.started_at else ""
        return {
            "job_id": self.run_id,
            "source": self.trigger,
            "title": f"Scheduled Research — {dt_label}",
            "status": status_map.get(self.status, "PENDING"),
            "alphaForgeStage": stage_map.get(self.stage, "research"),
            "submittedAt": self.started_at,
            "hypotheses_count": self.hypotheses_count,
            "dispatched_count": self.dispatched_count,
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class AutonomousResearchScheduler:
    """
    Background scheduler that fires overnight research cycles.

    Lifecycle:
      start()    — launches _scheduler_loop as asyncio task
      stop()     — signals loop to exit
      trigger()  — queues an immediate manual run
    """

    def __init__(self, schedule_time: str = DEFAULT_SCHEDULE_TIME):
        self._schedule_time = schedule_time          # "HH:MM"
        self._enabled: bool = True
        self._running: bool = False
        self._runs: Deque[ScheduledRun] = deque(maxlen=MAX_RUN_HISTORY)
        self._active_run: Optional[ScheduledRun] = None
        self._manual_trigger: asyncio.Event = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background scheduler loop."""
        self._running = True
        logger.info(
            f"AutonomousResearchScheduler starting — schedule={self._schedule_time} UTC, "
            f"enabled={self._enabled}"
        )
        await self._scheduler_loop()

    def stop(self) -> None:
        """Signal the scheduler loop to exit cleanly."""
        self._running = False
        self._manual_trigger.set()      # Unblock any sleeping wait

    def trigger_manual(self) -> str:
        """Queue an immediate manual run and return run_id."""
        run = self._create_run(trigger="manual")
        self._runs.appendleft(run)
        asyncio.create_task(self._fire_run_by_id(run.run_id))
        return run.run_id

    def get_status(self) -> dict:
        last = self._runs[0] if self._runs else None
        now = datetime.now(timezone.utc)
        next_run = self._calc_next_run(now)

        current_status = "idle"
        if self._active_run:
            current_status = "running" if self._active_run.status == "running" else "idle"
        elif last and last.status == "failed":
            current_status = "failed"

        return {
            "enabled": self._enabled,
            "schedule_time": self._schedule_time,
            "next_run_iso": next_run.isoformat(),
            "last_run": last.to_dict() if last else None,
            "status": current_status,
        }

    def get_runs(self) -> List[dict]:
        """Return last 50 runs as job cards (newest first)."""
        return [r.to_job_card() for r in self._runs]

    def configure(self, enabled: Optional[bool] = None, schedule_time: Optional[str] = None) -> None:
        if enabled is not None:
            self._enabled = enabled
        if schedule_time is not None:
            self._validate_time_format(schedule_time)
            self._schedule_time = schedule_time
        logger.info(f"Scheduler reconfigured: enabled={self._enabled}, schedule_time={self._schedule_time}")

    # ------------------------------------------------------------------
    # Scheduler loop
    # ------------------------------------------------------------------
    async def _scheduler_loop(self) -> None:
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_run = self._calc_next_run(now)
                sleep_seconds = (next_run - now).total_seconds()

                logger.info(
                    f"Autonomous scheduler sleeping {sleep_seconds:.0f}s until {next_run.isoformat()}"
                )

                # Sleep until next run or manual trigger
                try:
                    await asyncio.wait_for(
                        self._manual_trigger.wait(),
                        timeout=max(sleep_seconds, 0)
                    )
                    # Manual trigger fired — reset event but don't auto-fire here;
                    # trigger_manual() already queued the run via asyncio.create_task.
                    self._manual_trigger.clear()
                except asyncio.TimeoutError:
                    pass  # Normal timeout — time to fire scheduled run

                if not self._running:
                    break

                if self._enabled:
                    # Determine weekend flag
                    is_weekend = next_run.weekday() == 5  # Saturday
                    await self._fire_run(trigger="scheduled", is_weekend_run=is_weekend)

            except asyncio.CancelledError:
                logger.info("Autonomous scheduler loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in scheduler loop: {e}", exc_info=True)
                # Brief back-off before retrying to avoid tight error loops
                await asyncio.sleep(60)

    # ------------------------------------------------------------------
    # Scheduling maths
    # ------------------------------------------------------------------
    def _calc_next_run(self, now: datetime) -> datetime:
        """
        Calculate the next scheduled trigger time.

        Saturday uses WEEKEND_SCHEDULE_TIME (00:00), all other days use
        self._schedule_time.
        """
        h, m = self._parse_time(self._schedule_time)
        wh, wm = self._parse_time(WEEKEND_SCHEDULE_TIME)

        # Candidate today
        candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)

        # Walk forward up to 7 days to find real next trigger
        for delta in range(7):
            day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=delta)
            if day.weekday() == 5:
                trigger_time = day.replace(hour=wh, minute=wm)
            else:
                trigger_time = day.replace(hour=h, minute=m)

            if trigger_time > now:
                return trigger_time

        return candidate  # fallback (should never reach here)

    # ------------------------------------------------------------------
    # Run execution
    # ------------------------------------------------------------------
    async def _fire_run(self, trigger: str, is_weekend_run: bool = False) -> None:
        run = self._create_run(trigger=trigger, is_weekend_run=is_weekend_run)
        self._runs.appendleft(run)
        await self._execute_run(run)

    async def _fire_run_by_id(self, run_id: str) -> None:
        run = self._find_run(run_id)
        if run is None:
            logger.error(f"Cannot find run {run_id} to execute")
            return
        await self._execute_run(run)

    async def _execute_run(self, run: ScheduledRun) -> None:
        self._active_run = run
        run.status = "running"
        run.stage = "research"
        logger.info(f"[{run.run_id}] Research run started (trigger={run.trigger})")

        try:
            hypotheses = await self._run_research(run)
            run.hypotheses_count = len(hypotheses)
            run.stage = "trd_dispatched"

            dispatched = await self._dispatch_hypotheses(run, hypotheses)
            run.dispatched_count = dispatched
            run.stage = "done"
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"[{run.run_id}] Run completed — "
                f"hypotheses={run.hypotheses_count}, dispatched={run.dispatched_count}"
            )

        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.completed_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"[{run.run_id}] Run failed: {e}", exc_info=True)
        finally:
            self._active_run = None

    # ------------------------------------------------------------------
    # Research execution
    # ------------------------------------------------------------------
    async def _run_research(self, run: ScheduledRun) -> list:
        """
        Instantiate ResearchHead and call process_task for each symbol.
        Returns list of Hypothesis objects that pass the confidence threshold.
        """
        from src.agents.departments.heads.research_head import ResearchHead, ResearchTask

        research_head = ResearchHead()
        passing_hypotheses = []

        for symbol in run.symbols:
            try:
                task = ResearchTask(
                    query=DAILY_TASK_TEXT,
                    symbols=[symbol],
                    timeframes=["H4", "D1"],
                    session_id=run.run_id,
                )

                # process_task is synchronous — run in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                hypothesis = await loop.run_in_executor(None, research_head.process_task, task)

                logger.info(
                    f"[{run.run_id}] {symbol}: confidence={hypothesis.confidence_score:.2f}"
                )

                if hypothesis.confidence_score >= CONFIDENCE_THRESHOLD:
                    passing_hypotheses.append(hypothesis)

            except Exception as e:
                logger.warning(f"[{run.run_id}] Research failed for {symbol}: {e}")

        return passing_hypotheses

    # ------------------------------------------------------------------
    # Department mail dispatch
    # ------------------------------------------------------------------
    async def _dispatch_hypotheses(self, run: ScheduledRun, hypotheses: list) -> int:
        """
        Send STRATEGY_DISPATCH mail to Development for each passing hypothesis.
        Returns number of messages dispatched.
        """
        if not hypotheses:
            return 0

        dispatched = 0
        try:
            from src.agents.departments.department_mail import (
                DepartmentMailService,
                MessageType,
                Priority,
            )

            mail_service = DepartmentMailService()

            for hypothesis in hypotheses:
                try:
                    body = json.dumps({
                        "run_id": run.run_id,
                        "symbol": hypothesis.symbol,
                        "timeframe": hypothesis.timeframe,
                        "hypothesis": hypothesis.hypothesis,
                        "confidence_score": hypothesis.confidence_score,
                        "supporting_evidence": hypothesis.supporting_evidence,
                        "recommended_next_steps": hypothesis.recommended_next_steps,
                        "is_weekend_run": run.is_weekend_run,
                    })

                    mail_service.send(
                        from_dept="research",
                        to_dept="development",
                        type=MessageType.STRATEGY_DISPATCH,
                        subject=(
                            f"[Autonomous] High-confidence hypothesis: "
                            f"{hypothesis.symbol} ({hypothesis.confidence_score:.0%})"
                        ),
                        body=body,
                        priority=Priority.HIGH,
                        workflow_id=run.run_id,
                    )
                    dispatched += 1
                    logger.info(
                        f"[{run.run_id}] STRATEGY_DISPATCH sent for {hypothesis.symbol}"
                    )

                except Exception as e:
                    logger.warning(
                        f"[{run.run_id}] Failed to dispatch mail for {hypothesis.symbol}: {e}"
                    )

        except Exception as e:
            logger.warning(
                f"[{run.run_id}] Department mail service unavailable — "
                f"dispatching skipped: {e}"
            )

        return dispatched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _create_run(
        self,
        trigger: str,
        is_weekend_run: bool = False,
    ) -> ScheduledRun:
        return ScheduledRun(
            run_id=str(uuid.uuid4()),
            trigger=trigger,
            status="queued",
            stage="research",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
            hypotheses_count=0,
            dispatched_count=0,
            symbols=list(RESEARCH_SYMBOLS),
            error=None,
            is_weekend_run=is_weekend_run,
        )

    def _find_run(self, run_id: str) -> Optional[ScheduledRun]:
        for r in self._runs:
            if r.run_id == run_id:
                return r
        return None

    @staticmethod
    def _parse_time(time_str: str):
        h, m = time_str.split(":")
        return int(h), int(m)

    @staticmethod
    def _validate_time_format(time_str: str) -> None:
        parts = time_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid time format '{time_str}' — expected HH:MM")
        h, m = parts
        if not (h.isdigit() and m.isdigit() and 0 <= int(h) <= 23 and 0 <= int(m) <= 59):
            raise ValueError(f"Invalid time value '{time_str}'")


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_scheduler: Optional[AutonomousResearchScheduler] = None


def get_scheduler() -> AutonomousResearchScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AutonomousResearchScheduler()
    return _scheduler


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/autonomous-scheduler", tags=["autonomous-scheduler"])


class ConfigBody(BaseModel):
    enabled: Optional[bool] = None
    schedule_time: Optional[str] = None


@router.get("/status")
async def get_status():
    """Return scheduler status, next run time, and last run summary."""
    return get_scheduler().get_status()


@router.get("/runs")
async def get_runs():
    """Return the last 50 scheduled runs as FlowForge job cards (newest first)."""
    return get_scheduler().get_runs()


@router.post("/trigger")
async def trigger_manual_run():
    """Queue an immediate manual research run and return its run_id."""
    run_id = get_scheduler().trigger_manual()
    return {"run_id": run_id, "status": "queued"}


@router.post("/config")
async def update_config(body: ConfigBody):
    """Update scheduler configuration (enabled flag and/or schedule time)."""
    try:
        get_scheduler().configure(enabled=body.enabled, schedule_time=body.schedule_time)
        return {"ok": True, "status": get_scheduler().get_status()}
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(e))
