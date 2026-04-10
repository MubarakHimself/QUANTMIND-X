"""
QuantMindLib V1 — LifecycleBridge + EvaluationBridge + WorkflowBridge

Phase 3 (Bridge Definitions) of QuantMindLib V1 packet delivery.
Packet 3C: Lifecycle state machine, evaluation result wiring, and workflow state tracking.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.library.core.types.enums import ActivationState, BotHealth


class LifecycleTransition(BaseModel):
    """Record of a lifecycle phase transition."""

    bot_id: str
    from_phase: str  # "BACKTEST" | "PAPER" | "LIVE" | "ARCHIVED"
    to_phase: str
    timestamp_ms: int
    reason: str = ""

    model_config = BaseModel.model_config


class LifecycleBridge(BaseModel):
    """
    Manages bot lifecycle state machine.
    Tracks current phase and transition history.
    """

    current_phase: str = "BACKTEST"  # default starting phase
    phase_started_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    transition_history: List[LifecycleTransition] = Field(default_factory=list)
    min_paper_days: int = 3  # minimum paper trading days before LIVE promotion

    def transition_to(self, bot_id: str, to_phase: str, reason: str = "") -> bool:
        """
        Attempt a phase transition.
        Returns True if transition succeeded, False if rejected.
        Rejects: BACKTEST -> LIVE directly (must go through PAPER).
        Rejects: any -> ARCHIVED if not allowed.
        """
        valid_phases = {"BACKTEST", "PAPER", "LIVE", "ARCHIVED"}
        if to_phase not in valid_phases:
            return False

        # BLOCK: BACKTEST -> LIVE (must go through PAPER)
        if self.current_phase == "BACKTEST" and to_phase == "LIVE":
            return False

        # BLOCK: PAPER -> LIVE unless paper phase is ready (min_paper_days elapsed)
        if self.current_phase == "PAPER" and to_phase == "LIVE":
            if not self.is_paper_ready():
                return False

        # BLOCK: going backward except to ARCHIVED
        phase_order = {"BACKTEST": 0, "PAPER": 1, "LIVE": 2}
        if to_phase != "ARCHIVED":
            if phase_order.get(to_phase, 99) < phase_order.get(self.current_phase, 0):
                return False

        transition = LifecycleTransition(
            bot_id=bot_id,
            from_phase=self.current_phase,
            to_phase=to_phase,
            timestamp_ms=int(time.time() * 1000),
            reason=reason,
        )
        self.transition_history.append(transition)
        self.current_phase = to_phase
        self.phase_started_ms = int(time.time() * 1000)
        return True

    def get_phase_age_ms(self) -> int:
        """Milliseconds since entering current phase."""
        return int(time.time() * 1000) - self.phase_started_ms

    def is_paper_ready(self) -> bool:
        """True if in PAPER phase long enough for LIVE promotion."""
        if self.current_phase != "PAPER":
            return False
        age_ms = self.get_phase_age_ms()
        return age_ms >= (self.min_paper_days * 24 * 60 * 60 * 1000)

    def can_promote_to_live(self) -> bool:
        """True if can transition PAPER -> LIVE."""
        return self.current_phase == "PAPER" and self.is_paper_ready()

    def get_transition_count(self) -> int:
        """Total number of transitions made."""
        return len(self.transition_history)

    def is_in_phase(self, phase: str) -> bool:
        """True if currently in the given phase."""
        return self.current_phase == phase

    model_config = BaseModel.model_config


class EvaluationBridge(BaseModel):
    """
    Wires evaluation results from backtest pipeline into EvaluationResult domain model.
    Derives BotEvaluationProfile from multi-mode evaluation outputs.
    """

    last_result: Optional[Any] = None  # avoid circular import: EvaluationResult
    last_profile: Optional[Any] = None  # avoid circular import: BotEvaluationProfile

    def from_backtest_result(
        self,
        bot_id: str,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        expectancy: float,
        total_trades: int,
    ) -> Any:  # EvaluationResult
        """
        Convert raw backtest metrics into an EvaluationResult (mode=BACKTEST).
        """
        from src.library.core.domain.evaluation_result import EvaluationResult

        result = EvaluationResult(
            bot_id=bot_id,
            mode="BACKTEST",
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_trades=total_trades,
            return_pct=sharpe_ratio * 10.0,  # approximate from sharpe
            kelly_score=min(
                1.0,
                max(
                    0.0,
                    win_rate - (1 - win_rate) / (profit_factor if profit_factor > 0 else 1),
                ),
            ),
            passes_gate=sharpe_ratio >= 1.0 and max_drawdown <= 0.15,
        )
        self.last_result = result
        return result

    def to_evaluation_profile(
        self,
        bot_id: str,
        backtest_result: Any,  # EvaluationResult
        monte_carlo_p5: float,
        monte_carlo_p95: float,
        monte_carlo_max_dd95: float,
        walk_forward_avg_sharpe: float,
        walk_forward_avg_return: float,
        walk_forward_stability: float,
    ) -> Any:  # BotEvaluationProfile
        """
        Derive a full BotEvaluationProfile from backtest + Monte Carlo + Walk-Forward results.
        """
        from src.library.core.domain.bot_spec import (
            BacktestMetrics,
            BotEvaluationProfile,
            MonteCarloMetrics,
            WalkForwardMetrics,
        )

        # Compute PBO score: probability of being out-of-sample from walk-forward stability
        pbo = max(0.0, 1.0 - walk_forward_stability) if walk_forward_stability >= 0 else 1.0

        # Robustness score: weighted 50% walk-forward + 30% Monte Carlo width + 20% backtest Sharpe
        mc_width = abs(monte_carlo_p95 - monte_carlo_p5)
        mc_quality = max(0.0, 1.0 - mc_width / 100.0)  # narrower = better
        robustness = (walk_forward_stability * 0.5) + (mc_quality * 0.3) + (
            min(1.0, backtest_result.sharpe_ratio / 2.0) * 0.2
        )
        robustness = max(0.0, min(1.0, robustness))

        backtest_metrics = BacktestMetrics(
            sharpe_ratio=backtest_result.sharpe_ratio,
            max_drawdown=backtest_result.max_drawdown,
            total_return=backtest_result.return_pct,
            win_rate=backtest_result.win_rate,
            profit_factor=backtest_result.profit_factor,
            expectancy=backtest_result.expectancy or backtest_result.sharpe_ratio,
            avg_bars_held=0.0,  # not available from backtest result
            total_trades=backtest_result.total_trades,
        )

        monte_carlo_metrics = MonteCarloMetrics(
            n_simulations=1000,
            percentile_5_return=monte_carlo_p5,
            percentile_95_return=monte_carlo_p95,
            max_drawdown_95=monte_carlo_max_dd95,
            sharpe_confidence_width=mc_width,
        )

        walk_forward_metrics = WalkForwardMetrics(
            n_splits=5,
            avg_sharpe=walk_forward_avg_sharpe,
            avg_return=walk_forward_avg_return,
            stability=walk_forward_stability,
        )

        profile = BotEvaluationProfile(
            bot_id=bot_id,
            backtest=backtest_metrics,
            monte_carlo=monte_carlo_metrics,
            walk_forward=walk_forward_metrics,
            pbo_score=pbo,
            robustness_score=robustness,
            spread_sensitivity=0.0,  # not computed in this simplified version
            session_scores={},
        )

        self.last_result = backtest_result
        self.last_profile = profile
        return profile

    def compute_spread_sensitivity(
        self,
        result_narrow: Any,  # EvaluationResult
        result_wide: Any,  # EvaluationResult
    ) -> float:
        """
        Compute spread sensitivity: how much performance degrades with wider spreads.
        Returns 0-1 score where 0 = no sensitivity, 1 = highly sensitive.
        """
        if result_wide.sharpe_ratio <= 0:
            return 1.0
        degradation = (result_narrow.sharpe_ratio - result_wide.sharpe_ratio) / max(
            result_narrow.sharpe_ratio, 0.01
        )
        return max(0.0, min(1.0, degradation))

    model_config = BaseModel.model_config


class WorkflowArtifact(BaseModel):
    """Represents a workflow input or output artifact."""

    artifact_id: str
    artifact_type: str  # "TRD" | "BotSpec" | "BacktestReport" | "EvaluationResult"
    path: str = ""
    workflow_id: str
    created_at_ms: int

    model_config = BaseModel.model_config


class WorkflowState(BaseModel):
    """Tracks workflow execution state."""

    workflow_id: str
    workflow_name: str  # "WF1_ALPHAFORGE" | "WF2_IMPROVEMENT_LOOP"
    status: str  # "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"
    started_at_ms: Optional[int] = None
    completed_at_ms: Optional[int] = None
    inputs: List[WorkflowArtifact] = Field(default_factory=list)
    outputs: List[WorkflowArtifact] = Field(default_factory=list)

    def is_handoff_ready(self) -> bool:
        """True if workflow is completed and outputs are available for next workflow."""
        return self.status == "COMPLETED" and len(self.outputs) > 0

    def is_blocked(self) -> bool:
        """True if workflow is waiting on missing inputs."""
        return self.status == "PENDING" and len(self.inputs) == 0

    model_config = BaseModel.model_config


class WorkflowBridge(BaseModel):
    """
    Workflow state tracker for WF1 (AlphaForge) and WF2 (Improvement Loop).
    Tracks workflow execution state and handoff readiness between workflows.
    """

    workflows: Dict[str, WorkflowState] = Field(default_factory=dict)

    def register_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        input_artifacts: Optional[List[WorkflowArtifact]] = None,
    ) -> WorkflowState:
        """Register a new workflow and set its initial state."""
        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            status="PENDING",
            inputs=input_artifacts or [],
            outputs=[],
        )
        self.workflows[workflow_id] = state
        return state

    def start_workflow(self, workflow_id: str) -> bool:
        """Mark a workflow as RUNNING."""
        if workflow_id not in self.workflows:
            return False
        wf = self.workflows[workflow_id]
        wf.status = "RUNNING"
        wf.started_at_ms = int(time.time() * 1000)
        return True

    def complete_workflow(
        self,
        workflow_id: str,
        output_artifacts: Optional[List[WorkflowArtifact]] = None,
    ) -> bool:
        """Mark a workflow as COMPLETED with its outputs."""
        if workflow_id not in self.workflows:
            return False
        wf = self.workflows[workflow_id]
        wf.status = "COMPLETED"
        wf.completed_at_ms = int(time.time() * 1000)
        if output_artifacts:
            wf.outputs = output_artifacts
        return True

    def fail_workflow(self, workflow_id: str) -> bool:
        """Mark a workflow as FAILED."""
        if workflow_id not in self.workflows:
            return False
        self.workflows[workflow_id].status = "FAILED"
        return True

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get workflow state by ID."""
        return self.workflows.get(workflow_id)

    def get_handoff_ready_workflows(self, next_workflow: str) -> List[WorkflowState]:
        """
        Get workflows whose outputs are ready to hand off to the next workflow.
        Only returns COMPLETED workflows.
        """
        return [
            wf
            for wf in self.workflows.values()
            if wf.is_handoff_ready() and wf.workflow_name != next_workflow
        ]

    def is_wf1_to_wf2_ready(self) -> bool:
        """True if AlphaForge (WF1) outputs are ready for Improvement Loop (WF2)."""
        wf1_outputs = [
            wf
            for wf in self.workflows.values()
            if wf.workflow_name == "WF1_ALPHAFORGE" and wf.is_handoff_ready()
        ]
        return len(wf1_outputs) > 0

    model_config = BaseModel.model_config


__all__ = [
    "LifecycleBridge",
    "LifecycleTransition",
    "EvaluationBridge",
    "WorkflowBridge",
    "WorkflowArtifact",
    "WorkflowState",
]
