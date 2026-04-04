"""
Workflow 3 — Performance Intelligence (Dead Zone)
================================================

5-step agent-driven workflow executing at:
- 16:15 GMT: EOD Report
- 16:45 GMT: Session Performer ID
- 17:00 GMT: DPR Update
- 17:30 GMT: Queue Re-rank
- 18:00 GMT: Fortnight Accumulation

This workflow runs during the Dead Zone (16:00-18:00 GMT) to rank, score,
and prepare the portfolio for the next session (NY open).
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class WorkflowStepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DeadZoneWorkflowStep:
    """Single step in the Dead Zone Workflow 3."""
    step_name: str
    scheduled_time: str  # GMT timezone
    deadline_offset_minutes: int  # minutes after 16:15 GMT
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "scheduled_time": self.scheduled_time,
            "deadline_offset_minutes": self.deadline_offset_minutes,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class DeadZoneWorkflowResult:
    """Result of a complete Dead Zone workflow run."""
    run_id: str
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    steps: List[DeadZoneWorkflowStep] = field(default_factory=list)
    eod_report: Optional[Dict[str, Any]] = None
    dpr_scores: Optional[List[Dict[str, Any]]] = None
    queue_update: Optional[Dict[str, Any]] = None
    fortnight_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "eod_report": self.eod_report,
            "dpr_scores": self.dpr_scores,
            "queue_update": self.queue_update,
            "fortnight_result": self.fortnight_result,
            "error": self.error,
        }


class DeadZoneWorkflow3:
    """
    Workflow 3 — Performance Intelligence (Dead Zone).

    5-step agent-driven workflow executing at:
    - 16:15 GMT: EOD Report
    - 16:45 GMT: Session Performer ID
    - 17:00 GMT: DPR Update
    - 17:30 GMT: Queue Re-rank
    - 18:00 GMT: Fortnight Accumulation
    """

    WORKFLOW_STEPS = [
        ("eod_report", "16:15", 0),
        ("session_performer_id", "16:45", 30),
        ("dpr_update", "17:00", 45),
        ("queue_rerank", "17:30", 75),
        ("fortnight_accumulation", "18:00", 105),
    ]

    def __init__(self):
        self._steps: Dict[str, DeadZoneWorkflowStep] = {}
        for name, time, offset in self.WORKFLOW_STEPS:
            self._steps[name] = DeadZoneWorkflowStep(
                step_name=name,
                scheduled_time=time,
                deadline_offset_minutes=offset,
            )
        self._result: Optional[DeadZoneWorkflowResult] = None

    async def execute(self) -> DeadZoneWorkflowResult:
        """Execute the complete Dead Zone workflow."""
        run_id = str(uuid.uuid4())
        workflow_id = "workflow_3_dead_zone"

        self._result = DeadZoneWorkflowResult(
            run_id=run_id,
            workflow_id=workflow_id,
            started_at=datetime.now(timezone.utc),
            steps=list(self._steps.values()),
        )

        logger.info(f"[{run_id}] Dead Zone Workflow 3 starting")

        try:
            # Step 1: EOD Report at 16:15 GMT
            await self._execute_step("eod_report")

            # Step 2: Session Performer ID at 16:45 GMT
            await self._execute_step("session_performer_id")

            # Step 3: DPR Update at 17:00 GMT
            await self._execute_step("dpr_update")

            # Step 4: Queue Re-rank at 17:30 GMT
            await self._execute_step("queue_rerank")

            # Step 5: Fortnight Accumulation at 18:00 GMT
            await self._execute_step("fortnight_accumulation")

            self._result.status = "completed"
            self._result.completed_at = datetime.now(timezone.utc)
            logger.info(f"[{run_id}] Dead Zone Workflow 3 completed successfully")

        except Exception as e:
            self._result.status = "failed"
            self._result.error = str(e)
            self._result.completed_at = datetime.now(timezone.utc)
            logger.error(f"[{run_id}] Dead Zone Workflow 3 failed: {e}", exc_info=True)

        return self._result

    async def _execute_step(self, step_name: str) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step = self._steps[step_name]
        step.status = WorkflowStepStatus.RUNNING
        step.started_at = datetime.now(timezone.utc)

        logger.info(f"[{self._result.run_id}] Executing step: {step_name}")

        try:
            handler = getattr(self, f"_run_{step_name}", None)
            if handler is None:
                raise ValueError(f"No handler for step: {step_name}")

            output = await handler()

            step.status = WorkflowStepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            step.output = output

            # Store output in result for later steps
            if step_name == "eod_report":
                self._result.eod_report = output
            elif step_name == "dpr_update":
                self._result.dpr_scores = output.get("scores", [])
            elif step_name == "queue_rerank":
                self._result.queue_update = output
            elif step_name == "fortnight_accumulation":
                self._result.fortnight_result = output

            logger.info(f"[{self._result.run_id}] Step {step_name} completed")
            return output

        except Exception as e:
            step.status = WorkflowStepStatus.FAILED
            step.completed_at = datetime.now(timezone.utc)
            step.error = str(e)
            logger.error(f"[{self._result.run_id}] Step {step_name} failed: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Step Handlers
    # -------------------------------------------------------------------------
    async def _run_eod_report(self) -> Dict[str, Any]:
        """Generate EOD Report: trading day outcomes, regime states, anomaly events, P&L attribution."""
        from src.router.eod_report_generator import get_eod_report_generator

        generator = get_eod_report_generator()
        report = await generator.generate()

        return {
            "report": report.to_dict() if hasattr(report, 'to_dict') else report,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_session_performer_id(self) -> Dict[str, Any]:
        """Session Performer ID: score bots on session-specific metrics and apply SESSION_SPECIALIST tags."""
        from src.router.session_performer import get_session_performer_identifier

        identifier = get_session_performer_identifier()
        result = await identifier.run()

        return {
            "session_specialists": [r.to_dict() if hasattr(r, 'to_dict') else r for r in result.results],
            "total_bots_scored": len(result.results),
            "specialists_count": sum(1 for r in result.results if r.get("is_specialist", False)),
        }

    async def _run_dpr_update(self) -> Dict[str, Any]:
        """DPR Update: compute DPR scores and queue tier remix."""
        from src.router.dpr_scoring_engine import get_dpr_scoring_engine
        from src.router.queue_remix import get_queue_remix

        # Get active bot IDs (placeholder - would integrate with bot manifest)
        bot_ids = await self._get_active_bot_ids()

        # Score all bots
        scoring_engine = get_dpr_scoring_engine()
        dpr_scores = await scoring_engine.score_all_bots(bot_ids)

        # Compute queue tier remix
        remix_computer = get_queue_remix()
        remix_result = remix_computer.compute_remix(dpr_scores)

        return {
            "scores": [s.to_dict() if hasattr(s, 'to_dict') else {
                "bot_id": s.bot_id,
                "composite_score": s.composite_score,
                "tier": s.tier,
                "rank": s.rank,
            } for s in dpr_scores],
            "remix": remix_result,
            "total_bots": len(dpr_scores),
        }

    async def _run_queue_rerank(self) -> Dict[str, Any]:
        """Queue Re-rank: update ranked queue and apply SESSION_CONCERN flags."""
        from src.router.queue_reranker import get_queue_reranker

        # Get DPR scores from previous step
        dpr_scores = []
        if self._result and self._result.dpr_scores:
            from src.router.dpr_scoring_engine import DprScore
            for score_dict in self._result.dpr_scores:
                dpr_scores.append(DprScore(**score_dict))

        reranker = get_queue_reranker()
        result = await reranker.run(dpr_scores)

        return {
            "queue": result.queue,
            "concerns": result.concerns,
            "concerns_count": len(result.concerns),
        }

    async def _run_fortnight_accumulation(self) -> Dict[str, Any]:
        """Fortnight Accumulation: update 14-day rolling DPR data and write to cold storage."""
        from src.router.fortnight_accumulator import get_fortnight_accumulator

        # Get DPR scores from previous step
        dpr_scores = []
        if self._result and self._result.dpr_scores:
            from src.router.dpr_scoring_engine import DprScore
            for score_dict in self._result.dpr_scores:
                dpr_scores.append(DprScore(**score_dict))

        accumulator = get_fortnight_accumulator()
        result = await accumulator.run(dpr_scores)

        return {
            "fortnight_stats": result.fortnight_stats,
            "file_path": result.file_path,
            "bots_scored": result.bots_scored,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _get_active_bot_ids(self) -> List[str]:
        """Get list of active bot IDs from bot manifest."""
        try:
            from src.router.bot_manifest import BotRegistry
            registry = BotRegistry.get_instance()
            active_bots = registry.get_active_bots()
            return [bot.bot_id for bot in active_bots]
        except Exception as e:
            logger.warning(f"Could not get active bots from manifest: {e}")
            # Return empty list - workflow can still run with no bots
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        if self._result is None:
            return {"status": "not_started", "steps": {}}

        return {
            "run_id": self._result.run_id,
            "status": self._result.status,
            "started_at": self._result.started_at.isoformat(),
            "completed_at": self._result.completed_at.isoformat() if self._result.completed_at else None,
            "steps": {name: step.to_dict() for name, step in self._steps.items()},
            "error": self._result.error,
        }


# ============= Singleton Factory =============
_workflow_instance: Optional[DeadZoneWorkflow3] = None


def get_dead_zone_workflow() -> DeadZoneWorkflow3:
    """Get singleton instance of DeadZoneWorkflow3."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = DeadZoneWorkflow3()
    return _workflow_instance
