"""
QuantMindLib V1 — WF1 (AlgoForge) Library-Side Bridge

Phase 9 Packet 9A.

WF1 pipeline: TRD research -> BotSpec -> strategy code -> backtest evaluation
-> paper trading -> live promotion (3-day paper lag)

This bridge:
1. Converts TRD input -> BotSpec via TRDConverter
2. Triggers EvaluationOrchestrator for backtest
3. Tracks workflow state via WorkflowBridge
4. Handles paper -> live promotion decisions
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    LifecycleBridge,
    WorkflowArtifact,
    WorkflowBridge,
    WorkflowState,
)
from src.library.core.composition.trd_converter import TRDConverter
from src.library.workflows.stub_flows import AlgoForgeFlowStub

if TYPE_CHECKING:
    from src.library.core.domain.bot_spec import BotEvaluationProfile, BotSpec
    from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator


class WF1Bridge:
    """
    Library-side bridge for WF1 (AlgoForge) workflow.

    WF1 pipeline: TRD research -> BotSpec -> strategy code -> backtest evaluation
    -> paper trading -> live promotion (3-day paper lag)

    This bridge:
    1. Converts TRD input -> BotSpec
    2. Triggers EvaluationOrchestrator for backtest
    3. Tracks workflow state via WorkflowBridge
    4. Handles paper -> live promotion decisions
    """

    def __init__(
        self,
        trd_converter: Optional[TRDConverter] = None,
        evaluation_orchestrator: Optional[EvaluationOrchestrator] = None,
        workflow_bridge: Optional[WorkflowBridge] = None,
    ) -> None:
        """
        Initialize WF1Bridge with optional injected components.

        Args:
            trd_converter: Optional TRDConverter instance. Uses default if None.
            evaluation_orchestrator: Optional EvaluationOrchestrator instance.
                Uses default if None.
            workflow_bridge: Optional WorkflowBridge instance.
                Uses default if None.
        """
        self._trd_converter = trd_converter if trd_converter is not None else TRDConverter()
        self._evaluation_orchestrator = evaluation_orchestrator
        self._workflow_bridge = workflow_bridge if workflow_bridge is not None else WorkflowBridge()
        self._flow_stub = AlgoForgeFlowStub()

        # Cache for evaluation results keyed by workflow_id
        self._evaluation_cache: Dict[str, Optional[BotEvaluationProfile]] = {}
        # Cache for paper start times keyed by workflow_id
        self._paper_start_ms: Dict[str, int] = {}

    @property
    def evaluation_orchestrator(self) -> EvaluationOrchestrator:
        """Lazy-load EvaluationOrchestrator on first access."""
        if self._evaluation_orchestrator is None:
            from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator
            self._evaluation_orchestrator = EvaluationOrchestrator()
        return self._evaluation_orchestrator

    def submit_trd(
        self,
        trd_input: Dict[str, Any],
        workflow_id: Optional[str] = None,
    ) -> WorkflowState:
        """
        Submit a TRD document for WF1 processing.

        Steps:
            1. Convert TRD -> TRDRawData -> BotSpec
            2. Generate strategy code (validation)
            3. Register workflow with WorkflowBridge (status: RUNNING)
            4. Trigger evaluation
            5. Return WorkflowState

        Args:
            trd_input: Raw TRD dictionary with required fields
                (robot_id, strategy_type, symbol_scope).
            workflow_id: Optional workflow ID. Auto-generated if not provided.

        Returns:
            WorkflowState for the registered workflow.
        """
        # Generate workflow_id if not provided
        if workflow_id is None:
            workflow_id = f"wf1-{uuid.uuid4().hex[:8]}"

        # Step 1: Convert TRD -> BotSpec
        bot_spec: Optional[BotSpec] = None
        try:
            conversion_result = self._trd_converter.convert(trd_input)
            if not conversion_result.success:
                # TRD conversion failed — register as FAILED workflow
                self._workflow_bridge.register_workflow(
                    workflow_id=workflow_id,
                    workflow_name="WF1_ALGOFORGE",
                    input_artifacts=[],
                )
                self._workflow_bridge.fail_workflow(workflow_id)
                state = self._workflow_bridge.get_workflow(workflow_id)
                return state if state is not None else WorkflowState(
                    workflow_id=workflow_id,
                    workflow_name="WF1_ALGOFORGE",
                    status="FAILED",
                )
            bot_spec = conversion_result.bot_spec
        except ValueError as e:
            # TRD schema validation error
            self._workflow_bridge.register_workflow(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                input_artifacts=[],
            )
            self._workflow_bridge.fail_workflow(workflow_id)
            state = self._workflow_bridge.get_workflow(workflow_id)
            return state if state is not None else WorkflowState(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                status="FAILED",
            )

        if bot_spec is None:
            self._workflow_bridge.register_workflow(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                input_artifacts=[],
            )
            self._workflow_bridge.fail_workflow(workflow_id)
            state = self._workflow_bridge.get_workflow(workflow_id)
            return state if state is not None else WorkflowState(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                status="FAILED",
            )

        # Step 2: Generate strategy code (validation step)
        try:
            code_gen = self.evaluation_orchestrator._code_generator
            is_valid, errors = code_gen.validate_bot_spec(bot_spec)
            if not is_valid:
                self._workflow_bridge.register_workflow(
                    workflow_id=workflow_id,
                    workflow_name="WF1_ALGOFORGE",
                    input_artifacts=[],
                )
                self._workflow_bridge.fail_workflow(workflow_id)
                state = self._workflow_bridge.get_workflow(workflow_id)
                return state if state is not None else WorkflowState(
                    workflow_id=workflow_id,
                    workflow_name="WF1_ALGOFORGE",
                    status="FAILED",
                )
            code_gen.generate(bot_spec)
        except Exception:
            self._workflow_bridge.register_workflow(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                input_artifacts=[],
            )
            self._workflow_bridge.fail_workflow(workflow_id)
            state = self._workflow_bridge.get_workflow(workflow_id)
            return state if state is not None else WorkflowState(
                workflow_id=workflow_id,
                workflow_name="WF1_ALGOFORGE",
                status="FAILED",
            )

        # Step 3: Register workflow with WorkflowBridge
        now_ms = int(time.time() * 1000)
        input_artifact = WorkflowArtifact(
            artifact_id=f"{workflow_id}-trd-input",
            artifact_type="TRD",
            workflow_id=workflow_id,
            created_at_ms=now_ms,
        )
        spec_artifact = WorkflowArtifact(
            artifact_id=f"{workflow_id}-botspec-output",
            artifact_type="BotSpec",
            workflow_id=workflow_id,
            created_at_ms=now_ms,
        )
        self._workflow_bridge.register_workflow(
            workflow_id=workflow_id,
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[input_artifact, spec_artifact],
        )
        self._workflow_bridge.start_workflow(workflow_id)

        # Step 4: Trigger evaluation (async in production; synchronous for library scope)
        profile, warnings = self.evaluation_orchestrator.evaluate(bot_spec)
        self._evaluation_cache[workflow_id] = profile

        # Complete workflow if evaluation succeeded
        if profile is not None:
            output_artifact = WorkflowArtifact(
                artifact_id=f"{workflow_id}-eval-profile",
                artifact_type="EvaluationResult",
                workflow_id=workflow_id,
                created_at_ms=int(time.time() * 1000),
            )
            self._workflow_bridge.complete_workflow(
                workflow_id,
                output_artifacts=[output_artifact],
            )
        else:
            self._workflow_bridge.fail_workflow(workflow_id)

        # Step 5: Return workflow state
        state = self._workflow_bridge.get_workflow(workflow_id)
        return state if state is not None else WorkflowState(
            workflow_id=workflow_id,
            workflow_name="WF1_ALGOFORGE",
            status="FAILED",
        )

    def check_evaluation_ready(
        self,
        workflow_id: str,
    ) -> tuple[bool, Optional[BotEvaluationProfile]]:
        """
        Check if evaluation is complete and return the profile.

        Uses WorkflowBridge to track state and the internal cache for results.

        Args:
            workflow_id: The workflow ID to check.

        Returns:
            Tuple of (is_ready, evaluation_profile).
            is_ready=True when the workflow is COMPLETED.
        """
        state = self._workflow_bridge.get_workflow(workflow_id)
        if state is None:
            return False, None

        if state.status != "COMPLETED":
            return False, None

        profile = self._evaluation_cache.get(workflow_id)
        return True, profile

    def approve_for_paper(
        self,
        workflow_id: str,
    ) -> tuple[bool, str]:
        """
        Approve bot for paper trading based on evaluation results.

        Gate: robustness_score >= 0.6 AND passes_gate == True

        Args:
            workflow_id: The workflow ID to evaluate.

        Returns:
            Tuple of (approved, reason).
            approved=True if the strategy meets paper trading criteria.
        """
        state = self._workflow_bridge.get_workflow(workflow_id)
        if state is None:
            return False, "workflow_not_found"

        profile = self._evaluation_cache.get(workflow_id)
        if profile is None:
            return False, "no_evaluation_profile"

        # Check robustness gate
        if profile.robustness_score < 0.6:
            return False, f"robustness_too_low:{profile.robustness_score}"

        # Check passes_gate from backtest metrics
        # passes_gate is not directly on BotEvaluationProfile; derive from sharpe + drawdown
        sharpe = profile.backtest.sharpe_ratio
        drawdown = profile.backtest.max_drawdown
        passes_gate = sharpe >= 1.0 and drawdown <= 0.15

        if not passes_gate:
            return False, f"backtest_gate_failed:sharpe={sharpe},dd={drawdown}"

        return True, f"approved:robustness={profile.robustness_score:.3f}"

    def check_paper_ready_for_live(self, workflow_id: str) -> bool:
        """
        Check if paper trading phase is complete (3-day lag).

        Uses LifecycleBridge for phase timing.

        Args:
            workflow_id: The workflow ID to check.

        Returns:
            True if the paper trading phase has elapsed long enough for LIVE promotion.
        """
        state = self._workflow_bridge.get_workflow(workflow_id)
        if state is None:
            return False

        # Paper start time must have been recorded
        paper_start = self._paper_start_ms.get(workflow_id)
        if paper_start is None:
            return False

        lifecycle = LifecycleBridge()
        # Set the phase start to when paper began
        lifecycle.current_phase = "PAPER"
        lifecycle.phase_started_ms = paper_start

        return lifecycle.is_paper_ready()

    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get current workflow state from WorkflowBridge.

        Args:
            workflow_id: The workflow ID to look up.

        Returns:
            WorkflowState if found, None otherwise.
        """
        return self._workflow_bridge.get_workflow(workflow_id)


__all__ = ["WF1Bridge"]
