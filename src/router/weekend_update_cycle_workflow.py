"""
Workflow 4 — Weekend Update Cycle
=================================

3-phase agent-driven workflow executing at:
- Friday 21:00 GMT: Friday Analysis (performance review, regime behaviour, correlation shifts)
- Saturday 06:00-18:00 GMT: Workflow 2 Refinement + Walk-Forward Analysis + HMM retraining
- Sunday 06:00-18:00 GMT: Pre-market Calibration (spread profiles, SQS baselines, Kelly modifiers)
- Monday 05:00 GMT: Fresh roster deployment to SessionDetector

Weekday hard rule: No bot parameter changes on weekdays (architecturally enforced).

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle)
"""

import asyncio
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
class WeekendCycleStep:
    """Single step in the Weekend Update Cycle."""
    step_name: str
    scheduled_time: str  # GMT timezone (e.g., "21:00", "06:00")
    day: str  # Friday, Saturday, Sunday, Monday
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "scheduled_time": self.scheduled_time,
            "day": self.day,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class WeekendRoster:
    """Fresh roster prepared for Monday deployment."""
    created_at: datetime
    bots: List[str]  # List of bot IDs in deployment order
    session_configs: Dict[str, Any]  # Per-bot session configurations
    metadata: Dict[str, Any]  # Week number, year, analysis summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "bots": self.bots,
            "session_configs": self.session_configs,
            "metadata": self.metadata,
        }


@dataclass
class WeekendCycleWorkflowResult:
    """Result of a complete Weekend Update Cycle workflow run."""
    run_id: str
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    steps: List[WeekendCycleStep] = field(default_factory=list)
    friday_analysis: Optional[Dict[str, Any]] = None
    saturday_refinement: Optional[Dict[str, Any]] = None
    sunday_calibration: Optional[Dict[str, Any]] = None
    monday_roster: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "friday_analysis": self.friday_analysis,
            "saturday_refinement": self.saturday_refinement,
            "sunday_calibration": self.sunday_calibration,
            "monday_roster": self.monday_roster,
            "error": self.error,
        }


class WeekendUpdateCycleWorkflow:
    """
    Workflow 4 — Weekend Update Cycle.

    3-phase agent-driven workflow executing at:
    - Friday 21:00 GMT: Friday Analysis (performance review, regime behaviour, correlation shifts)
    - Saturday 06:00-18:00 GMT: Workflow 2 Refinement + Walk-Forward Analysis + HMM retraining
    - Sunday 06:00-18:00 GMT: Pre-market Calibration (spread profiles, SQS, Kelly modifiers)
    - Monday 05:00 GMT: Fresh roster deployment to SessionDetector

    Weekday hard rule: No bot parameter changes on weekdays (architecturally enforced).
    """

    WORKFLOW_STEPS = [
        # Friday
        ("friday_analysis", "21:00", "Friday"),
        # Saturday
        ("saturday_refinement", "06:00", "Saturday"),
        ("saturday_wfa", "09:00", "Saturday"),
        ("saturday_hmm_retrain", "12:00", "Saturday"),
        # Sunday
        ("sunday_calibration", "06:00", "Sunday"),
        ("sunday_spread_profiles", "09:00", "Sunday"),
        ("sunday_sqs_refresh", "12:00", "Sunday"),
        ("sunday_kelly_calibration", "15:00", "Sunday"),
        # Monday
        ("monday_roster_deploy", "05:00", "Monday"),
    ]

    # Weekday hard block hours (Monday 00:00 GMT to Thursday 23:59 GMT)
    WEEKDAY_BLOCK_START = 0  # Monday 00:00
    WEEKDAY_BLOCK_END = 23  # Thursday 23:59

    def __init__(self):
        self._steps: Dict[str, WeekendCycleStep] = {}
        for name, time, day in self.WORKFLOW_STEPS:
            self._steps[name] = WeekendCycleStep(
                step_name=name,
                scheduled_time=time,
                day=day,
            )
        self._result: Optional[WeekendCycleWorkflowResult] = None
        self._friday_candidates: List[str] = []
        logger.info("WeekendUpdateCycleWorkflow initialized")

    def is_weekday_blocked(self, utc_now: datetime) -> bool:
        """Check if current time is in weekday block period."""
        # Weekday = Monday(0) through Thursday(3)
        if utc_now.weekday() in [0, 1, 2, 3]:
            return True
        return False

    async def execute(self) -> WeekendCycleWorkflowResult:
        """Execute the complete Weekend Update Cycle."""
        run_id = str(uuid.uuid4())
        workflow_id = "workflow_4_weekend_update_cycle"

        self._result = WeekendCycleWorkflowResult(
            run_id=run_id,
            workflow_id=workflow_id,
            started_at=datetime.now(timezone.utc),
            steps=list(self._steps.values()),
        )

        logger.info(f"[{run_id}] Weekend Update Cycle Workflow starting")

        try:
            # Friday 21:00 GMT: Friday Analysis
            await self._execute_step("friday_analysis")

            # Store candidates for Saturday
            if self._result.friday_analysis:
                self._friday_candidates = self._result.friday_analysis.get(
                    "candidate_bots_for_refinement", []
                )

            # Saturday steps
            await self._execute_step("saturday_refinement")
            await self._execute_step("saturday_wfa")
            await self._execute_step("saturday_hmm_retrain")

            # Sunday steps
            await self._execute_step("sunday_calibration")
            await self._execute_step("sunday_spread_profiles")
            await self._execute_step("sunday_sqs_refresh")
            await self._execute_step("sunday_kelly_calibration")

            # Monday 05:00 GMT: Roster deployment
            await self._execute_step("monday_roster_deploy")

            self._result.status = "completed"
            self._result.completed_at = datetime.now(timezone.utc)
            logger.info(f"[{run_id}] Weekend Update Cycle Workflow completed successfully")

        except Exception as e:
            self._result.status = "failed"
            self._result.error = str(e)
            self._result.completed_at = datetime.now(timezone.utc)
            logger.error(f"[{run_id}] Weekend Update Cycle Workflow failed: {e}", exc_info=True)

        return self._result

    async def execute_single_step(self, step_name: str) -> Dict[str, Any]:
        """Execute a single workflow step by name (for manual triggering)."""
        return await self._execute_step(step_name)

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

            # Store step output in result
            if step_name == "friday_analysis":
                self._result.friday_analysis = output
            elif step_name.startswith("saturday"):
                if self._result.saturday_refinement is None:
                    self._result.saturday_refinement = {}
                if step_name == "saturday_refinement":
                    self._result.saturday_refinement["refinement"] = output
                elif step_name == "saturday_wfa":
                    self._result.saturday_refinement["wfa"] = output
                elif step_name == "saturday_hmm_retrain":
                    self._result.saturday_refinement["hmm_retrain"] = output
            elif step_name.startswith("sunday"):
                if self._result.sunday_calibration is None:
                    self._result.sunday_calibration = {}
                if step_name == "sunday_calibration":
                    self._result.sunday_calibration["calibration"] = output
                elif step_name == "sunday_spread_profiles":
                    self._result.sunday_calibration["spread_profiles"] = output
                elif step_name == "sunday_sqs_refresh":
                    self._result.sunday_calibration["sqs_refresh"] = output
                elif step_name == "sunday_kelly_calibration":
                    self._result.sunday_calibration["kelly_calibration"] = output
            elif step_name == "monday_roster_deploy":
                self._result.monday_roster = output

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
    async def _run_friday_analysis(self) -> Dict[str, Any]:
        """Friday Analysis at 21:00 GMT: full performance review, regime behaviour, correlation shifts."""
        from src.router.friday_analysis_service import get_friday_analysis_service

        service = get_friday_analysis_service()
        result = await service.run()

        return {
            "analysis": result.to_dict() if hasattr(result, 'to_dict') else result,
            "candidate_bots_for_refinement": getattr(result, 'candidate_bots_for_refinement', []),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_saturday_refinement(self) -> Dict[str, Any]:
        """Saturday 06:00 GMT: Workflow 2 refinement on selected bots."""
        from src.router.saturday_refinement_service import get_saturday_refinement_service

        service = get_saturday_refinement_service()
        candidates = self._friday_candidates[:5]  # Max 5 candidates
        result = await service.run(candidates)

        return {
            "refined_bots": [r.to_dict() if hasattr(r, 'to_dict') else r for r in result.results],
            "total_refined": len(result.results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_saturday_wfa(self) -> Dict[str, Any]:
        """Saturday 09:00 GMT: Walk-Forward Analysis on validated candidates."""
        from src.router.walk_forward_analyzer import get_walk_forward_analyzer

        analyzer = get_walk_forward_analyzer()
        candidates = self._friday_candidates[:5]
        wfa_results = []

        for bot_id in candidates:
            result = await analyzer.run(bot_id)
            wfa_results.append(result.to_dict() if hasattr(result, 'to_dict') else result)

        return {
            "wfa_results": wfa_results,
            "total_analyzed": len(wfa_results),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_saturday_hmm_retrain(self) -> Dict[str, Any]:
        """Saturday 12:00 GMT: HMM retraining trigger on Kamatera T2."""
        from src.router.hmm_retrain_trigger import get_hmm_retrain_trigger

        trigger = get_hmm_retrain_trigger()
        result = await trigger.trigger()

        return {
            "result": result.to_dict() if hasattr(result, 'to_dict') else result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_sunday_calibration(self) -> Dict[str, Any]:
        """Sunday 06:00 GMT: Pre-market calibration (spread profiles, SQS, Kelly modifiers)."""
        from src.router.sunday_calibration_service import get_sunday_calibration_service

        service = get_sunday_calibration_service()
        result = await service.run()

        return {
            "calibration": result.to_dict() if hasattr(result, 'to_dict') else result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_sunday_spread_profiles(self) -> Dict[str, Any]:
        """Sunday 09:00 GMT: Spread profiles update."""
        from src.router.sunday_calibration_service import get_sunday_calibration_service

        service = get_sunday_calibration_service()
        result = await service._update_spread_profiles()

        return {
            "spread_profiles": result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_sunday_sqs_refresh(self) -> Dict[str, Any]:
        """Sunday 12:00 GMT: SQS baselines refresh."""
        from src.router.sunday_calibration_service import get_sunday_calibration_service

        service = get_sunday_calibration_service()
        result = await service._refresh_sqs_baselines()

        return {
            "sqs_baselines": result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_sunday_kelly_calibration(self) -> Dict[str, Any]:
        """Sunday 15:00 GMT: Session Kelly modifiers calibration for coming week."""
        from src.router.sunday_calibration_service import get_sunday_calibration_service

        service = get_sunday_calibration_service()
        result = await service._calibrate_kelly_modifiers()

        return {
            "kelly_calibration": result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _run_monday_roster_deploy(self) -> Dict[str, Any]:
        """Monday 05:00 GMT: Fresh roster deployment to SessionDetector."""
        from src.router.weekend_roster_manager import get_weekend_roster_manager

        manager = get_weekend_roster_manager()
        result = await manager.deploy_roster()

        return {
            "deployment": result.to_dict() if hasattr(result, 'to_dict') else result,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

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
            "friday_analysis": self._result.friday_analysis,
            "saturday_refinement": self._result.saturday_refinement,
            "sunday_calibration": self._result.sunday_calibration,
            "monday_roster": self._result.monday_roster,
            "error": self._result.error,
        }

    def get_current_step(self) -> Optional[str]:
        """Get the current in-progress step name."""
        for name, step in self._steps.items():
            if step.status == WorkflowStepStatus.RUNNING:
                return name
        return None


# ============= Singleton Factory =============
_workflow_instance: Optional[WeekendUpdateCycleWorkflow] = None


def get_weekend_update_cycle_workflow() -> WeekendUpdateCycleWorkflow:
    """Get singleton instance of WeekendUpdateCycleWorkflow."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = WeekendUpdateCycleWorkflow()
    return _workflow_instance
