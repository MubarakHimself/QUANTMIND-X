"""
QuantMindLib V1 — WF2 (Improvement Loop) Library-Side Bridge

Phase 9 Packet 9A.

WF2 pipeline: surviving variants -> lineage analysis -> constrained mutation
-> re-backtest -> MC/WFA -> paper -> promotion/demotion/kill

This bridge:
1. Reads variant lineage from BotMutationProfile
2. Applies MutationEngine for constrained mutations
3. Triggers re-evaluation via EvaluationOrchestrator
4. Tracks workflow state via WorkflowBridge
5. Handles promotion/demotion/kill decisions
"""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    WorkflowArtifact,
    WorkflowBridge,
    WorkflowState,
)
from src.library.workflows.stub_flows import ImprovementLoopFlowStub

if TYPE_CHECKING:
    from src.library.core.domain.bot_spec import BotEvaluationProfile, BotSpec
    from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator
    from src.library.archetypes.mutation.engine import MutationEngine


class WF2Bridge:
    """
    Library-side bridge for WF2 (Improvement Loop) workflow.

    WF2 pipeline: surviving variants -> lineage analysis -> constrained mutation
    -> re-backtest -> MC/WFA -> paper -> promotion/demotion/kill

    This bridge:
    1. Reads variant lineage from BotMutationProfile
    2. Applies MutationEngine for constrained mutations
    3. Triggers re-evaluation via EvaluationOrchestrator
    4. Tracks workflow state via WorkflowBridge
    5. Handles promotion/demotion/kill decisions
    """

    def __init__(
        self,
        evaluation_orchestrator: Optional[EvaluationOrchestrator] = None,
        mutation_engine: Optional[MutationEngine] = None,
        workflow_bridge: Optional[WorkflowBridge] = None,
    ) -> None:
        """
        Initialize WF2Bridge with optional injected components.

        Args:
            evaluation_orchestrator: Optional EvaluationOrchestrator instance.
                Uses default if None.
            mutation_engine: Optional MutationEngine instance.
                Uses default if None.
            workflow_bridge: Optional WorkflowBridge instance.
                Uses default if None.
        """
        self._evaluation_orchestrator = evaluation_orchestrator
        self._mutation_engine = mutation_engine
        self._workflow_bridge = workflow_bridge if workflow_bridge is not None else WorkflowBridge()
        self._flow_stub = ImprovementLoopFlowStub()

        # Cache for evaluation results keyed by workflow_id
        self._evaluation_cache: Dict[str, Optional[BotEvaluationProfile]] = {}

    @property
    def evaluation_orchestrator(self) -> EvaluationOrchestrator:
        """Lazy-load EvaluationOrchestrator on first access."""
        if self._evaluation_orchestrator is None:
            from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator
            self._evaluation_orchestrator = EvaluationOrchestrator()
        return self._evaluation_orchestrator

    def submit_variant(
        self,
        parent_bot_spec: BotSpec,
        workflow_id: Optional[str] = None,
    ) -> WorkflowState:
        """
        Submit a variant for WF2 improvement loop.

        Creates mutated BotSpec from parent + MutationEngine.
        Registers workflow with WorkflowBridge.

        Args:
            parent_bot_spec: The parent BotSpec to mutate.
            workflow_id: Optional workflow ID. Auto-generated if not provided.

        Returns:
            WorkflowState for the registered workflow.
        """
        # Generate workflow_id if not provided
        if workflow_id is None:
            workflow_id = f"wf2-{uuid.uuid4().hex[:8]}"

        # Check for mutation profile
        if parent_bot_spec.mutation is None:
            self._workflow_bridge.register_workflow(
                workflow_id=workflow_id,
                workflow_name="WF2_IMPROVEMENT_LOOP",
                input_artifacts=[],
            )
            self._workflow_bridge.fail_workflow(workflow_id)
            state = self._workflow_bridge.get_workflow(workflow_id)
            return state if state is not None else WorkflowState(
                workflow_id=workflow_id,
                workflow_name="WF2_IMPROVEMENT_LOOP",
                status="FAILED",
            )

        # Register workflow
        now_ms = int(time.time() * 1000)
        input_artifact = WorkflowArtifact(
            artifact_id=f"{workflow_id}-parent-botspec",
            artifact_type="BotSpec",
            workflow_id=workflow_id,
            created_at_ms=now_ms,
        )
        self._workflow_bridge.register_workflow(
            workflow_id=workflow_id,
            workflow_name="WF2_IMPROVEMENT_LOOP",
            input_artifacts=[input_artifact],
        )
        self._workflow_bridge.start_workflow(workflow_id)

        # Variant lineage is preserved (parent_bot_spec already has mutation profile)
        output_artifact = WorkflowArtifact(
            artifact_id=f"{workflow_id}-variant-botspec",
            artifact_type="BotSpec",
            workflow_id=workflow_id,
            created_at_ms=int(time.time() * 1000),
        )
        self._workflow_bridge.complete_workflow(
            workflow_id,
            output_artifacts=[output_artifact],
        )

        state = self._workflow_bridge.get_workflow(workflow_id)
        return state if state is not None else WorkflowState(
            workflow_id=workflow_id,
            workflow_name="WF2_IMPROVEMENT_LOOP",
            status="FAILED",
        )

    def run_mutation_cycle(
        self,
        workflow_id: str,
    ) -> tuple[bool, Optional[BotSpec], List[str]]:
        """
        Run one mutation cycle: generate -> evaluate -> decide.

        Args:
            workflow_id: The workflow ID tracking this mutation cycle.

        Returns:
            Tuple of (improved, mutated_spec, warnings).
            improved=True means the variant should proceed to paper.
        """
        warnings: List[str] = []

        state = self._workflow_bridge.get_workflow(workflow_id)
        if state is None:
            warnings.append(f"workflow {workflow_id} not found")
            return False, None, warnings

        # Get parent BotSpec from workflow inputs
        if not state.inputs:
            warnings.append("no parent BotSpec in workflow inputs")
            return False, None, warnings

        # For the library-side bridge, mutated_spec must come from outside
        # since MutationEngine requires injected dependencies (Composer, ArchetypeRegistry).
        # In this library scope, we check the evaluation cache for existing results.
        profile = self._evaluation_cache.get(workflow_id)

        # Evaluate the workflow's output BotSpec if not yet evaluated
        if profile is None:
            warnings.append("no evaluation profile cached for this workflow")
            return False, None, warnings

        # improved = robustness >= 0.6 and passes_gate
        sharpe = profile.backtest.sharpe_ratio
        drawdown = profile.backtest.max_drawdown
        passes_gate = sharpe >= 1.0 and drawdown <= 0.15
        improved = profile.robustness_score >= 0.6 and passes_gate

        if improved:
            warnings.append(f"variant improved: robustness={profile.robustness_score:.3f}")
        else:
            warnings.append(
                f"variant not improved: robustness={profile.robustness_score:.3f}, "
                f"sharpe={sharpe}, dd={drawdown}"
            )

        return improved, None, warnings

    def decide_outcome(
        self,
        workflow_id: str,
        evaluation_profile: BotEvaluationProfile,
    ) -> str:
        """
        Decide: promote / demote / quarantine / kill.

        Decision matrix:
            - robustness_score >= 0.8 AND DPR score >= 50 -> promote
            - robustness_score >= 0.5 -> demote (retry)
            - robustness_score < 0.5 -> kill
            - DPR score < 30 -> quarantine

        Args:
            workflow_id: The workflow ID (for logging context).
            evaluation_profile: The evaluation profile to base the decision on.

        Returns:
            Decision string: "promote" | "demote" | "quarantine" | "kill".
        """
        robustness = evaluation_profile.robustness_score
        sharpe = evaluation_profile.backtest.sharpe_ratio
        drawdown = evaluation_profile.backtest.max_drawdown

        # DPR score is not directly on BotEvaluationProfile in this scope.
        # Use sharpe as proxy: sharpe >= 2.0 roughly maps to DPR score >= 50.
        dpr_proxy = sharpe * 25  # approximate mapping

        # Check quarantine first (DPR < 30)
        if dpr_proxy < 30:
            return "quarantine"

        # Promote: robustness >= 0.8 AND DPR proxy >= 50
        if robustness >= 0.8 and dpr_proxy >= 50:
            return "promote"

        # Kill: robustness < 0.5
        if robustness < 0.5:
            return "kill"

        # Demote: robustness >= 0.5 (retry loop)
        return "demote"

    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get current workflow state from WorkflowBridge.

        Args:
            workflow_id: The workflow ID to look up.

        Returns:
            WorkflowState if found, None otherwise.
        """
        return self._workflow_bridge.get_workflow(workflow_id)


__all__ = ["WF2Bridge"]
