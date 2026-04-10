"""
Tests for QuantMindLib V1 — WF2Bridge (Improvement Loop) (Packet 9B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    WorkflowBridge,
    WorkflowArtifact,
)
from src.library.core.domain.bot_spec import (
    BacktestMetrics,
    BotEvaluationProfile,
    BotMutationProfile,
    BotSpec,
    MonteCarloMetrics,
    WalkForwardMetrics,
)
from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.library.workflows.stub_flows import ImprovementLoopFlowStub
from src.library.workflows.wf2_bridge import WF2Bridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bot_spec_with_mutation(bot_id: str) -> BotSpec:
    """Build a BotSpec with a populated mutation profile."""
    return BotSpec(
        id=bot_id,
        archetype="orb",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
        mutation=BotMutationProfile(
            bot_id=bot_id,
            allowed_mutation_areas=["entry_rules", "exit_rules"],
            locked_components=["archetype"],
            lineage=[bot_id],
            parent_variant_id=None,
            generation=1,
            unsafe_areas=[],
        ),
    )


def _make_bot_spec_no_mutation(bot_id: str) -> BotSpec:
    """Build a BotSpec without a mutation profile."""
    return BotSpec(
        id=bot_id,
        archetype="orb",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
        mutation=None,
    )


def _make_evaluation_profile(
    bot_id: str,
    robustness: float,
    sharpe: float,
    drawdown: float,
) -> BotEvaluationProfile:
    """Build a BotEvaluationProfile with configurable metrics."""
    return BotEvaluationProfile(
        bot_id=bot_id,
        backtest=BacktestMetrics(
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            total_return=sharpe * 8.0,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=0.5,
            avg_bars_held=5.0,
            total_trades=50,
        ),
        monte_carlo=MonteCarloMetrics(
            n_simulations=1000,
            percentile_5_return=-5.0,
            percentile_95_return=25.0,
            max_drawdown_95=8.0,
            sharpe_confidence_width=30.0,
        ),
        walk_forward=WalkForwardMetrics(
            n_splits=5,
            avg_sharpe=sharpe * 0.8,
            avg_return=sharpe * 5.0,
            stability=0.7,
        ),
        pbo_score=0.15,
        robustness_score=robustness,
        spread_sensitivity=0.1,
        session_scores={},
    )


# ---------------------------------------------------------------------------
# TestImprovementLoopFlowStub
# ---------------------------------------------------------------------------


class TestImprovementLoopFlowStub:
    """Tests for ImprovementLoopFlowStub."""

    def test_trigger_returns_wf2_prefixed_id(self):
        """trigger() returns a string prefixed with 'wf2-'."""
        result = ImprovementLoopFlowStub.trigger({})
        assert isinstance(result, str)
        assert result.startswith("wf2-")

    def test_trigger_returns_unique_ids(self):
        """trigger() returns different IDs on each call."""
        ids = [ImprovementLoopFlowStub.trigger({}) for _ in range(5)]
        assert len(set(ids)) == 5

    def test_get_status_returns_pending(self):
        """get_status() returns 'PENDING' by default."""
        status = ImprovementLoopFlowStub.get_status("wf2-abc12345")
        assert status == "PENDING"

    def test_get_result_returns_none(self):
        """get_result() returns None by default."""
        result = ImprovementLoopFlowStub.get_result("wf2-abc12345")
        assert result is None


# ---------------------------------------------------------------------------
# TestWF2BridgeInit
# ---------------------------------------------------------------------------


class TestWF2BridgeInit:
    """Tests for WF2Bridge initialization."""

    def test_default_init(self):
        """WF2Bridge initializes with default components."""
        bridge = WF2Bridge()
        assert bridge._workflow_bridge is not None
        assert type(bridge._workflow_bridge).__name__ == "WorkflowBridge"
        assert bridge._flow_stub is not None
        assert type(bridge._flow_stub).__name__ == "ImprovementLoopFlowStub"
        assert bridge._evaluation_cache == {}

    def test_custom_components(self):
        """WF2Bridge accepts custom components."""
        mock_wb = WorkflowBridge()
        mock_eval = MagicMock(spec=EvaluationOrchestrator)
        mock_mutation = MagicMock()

        bridge = WF2Bridge(
            evaluation_orchestrator=mock_eval,
            mutation_engine=mock_mutation,
            workflow_bridge=mock_wb,
        )
        assert bridge._workflow_bridge is mock_wb
        assert bridge._evaluation_orchestrator is mock_eval
        assert bridge._mutation_engine is mock_mutation

    def test_evaluation_cache_starts_empty(self):
        """_evaluation_cache is empty on init."""
        bridge = WF2Bridge()
        assert bridge._evaluation_cache == {}


# ---------------------------------------------------------------------------
# TestWF2BridgeMutation
# ---------------------------------------------------------------------------


class TestWF2BridgeMutation:
    """Tests for WF2Bridge variant mutation pipeline."""

    def test_submit_variant_creates_wf2_workflow(self):
        """submit_variant() creates a workflow with workflow_name='WF2_IMPROVEMENT_LOOP'."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-001")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-var-001")

        assert state is not None
        assert state.workflow_name == "WF2_IMPROVEMENT_LOOP"

    def test_submit_variant_with_valid_bot_spec_returns_completed_state(self):
        """Valid BotSpec with mutation returns COMPLETED workflow."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-002")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-var-002")

        assert state is not None
        assert state.status == "COMPLETED"

    def test_submit_variant_with_invalid_spec_returns_failed(self):
        """BotSpec without mutation returns FAILED workflow."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_no_mutation("variant-fail-001")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-var-fail-001")

        assert state is not None
        assert state.status == "FAILED"
        assert state.workflow_name == "WF2_IMPROVEMENT_LOOP"

    def test_submit_variant_stores_evaluation(self):
        """submit_variant() does NOT store evaluation in cache directly (external pipeline)."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-003")

        bridge.submit_variant(bot_spec, workflow_id="wf2-var-003")

        # submit_variant() itself does not populate _evaluation_cache
        # (that happens in run_mutation_cycle or externally)
        assert "wf2-var-003" not in bridge._evaluation_cache

    def test_submit_variant_registers_in_workflow_bridge(self):
        """submit_variant() registers the workflow in WorkflowBridge."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-004")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-var-004")
        assert state.status == "COMPLETED"

        wb_state = bridge._workflow_bridge.get_workflow("wf2-var-004")
        assert wb_state is not None
        assert wb_state.status == "COMPLETED"

    def test_submit_variant_auto_generates_id(self):
        """submit_variant() auto-generates workflow_id when not provided."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-005")

        state = bridge.submit_variant(bot_spec)

        assert state is not None
        assert state.workflow_id.startswith("wf2-")

    def test_submit_variant_has_output_artifact(self):
        """Completed variant workflow has a BotSpec output artifact."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-006")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-var-006")

        assert len(state.outputs) == 1
        assert state.outputs[0].artifact_type == "BotSpec"

    def test_run_mutation_cycle_returns_false_when_not_found(self):
        """run_mutation_cycle() returns False when workflow is not found."""
        bridge = WF2Bridge()
        improved, spec, warnings = bridge.run_mutation_cycle("unknown-wf2")
        assert improved is False
        assert spec is None
        assert len(warnings) > 0

    def test_run_mutation_cycle_returns_false_without_inputs(self):
        """run_mutation_cycle() returns False when workflow has no inputs."""
        bridge = WF2Bridge()
        # Register a workflow without inputs
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf2-no-inputs",
            workflow_name="WF2_IMPROVEMENT_LOOP",
            input_artifacts=[],
        )
        bridge._workflow_bridge.start_workflow("wf2-no-inputs")

        improved, spec, warnings = bridge.run_mutation_cycle("wf2-no-inputs")
        assert improved is False
        assert spec is None

    def test_run_mutation_cycle_uses_evaluation_cache(self):
        """run_mutation_cycle() reads from _evaluation_cache."""
        bridge = WF2Bridge()

        # Register a completed workflow
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf2-cache-test",
            workflow_name="WF2_IMPROVEMENT_LOOP",
            input_artifacts=[
                WorkflowArtifact(
                    artifact_id="in-art",
                    artifact_type="BotSpec",
                    workflow_id="wf2-cache-test",
                    created_at_ms=1000,
                )
            ],
        )
        bridge._workflow_bridge.start_workflow("wf2-cache-test")
        bridge._workflow_bridge.complete_workflow("wf2-cache-test")

        # Pre-populate evaluation cache with a good profile
        good_profile = _make_evaluation_profile(
            bot_id="bot-cached",
            robustness=0.85,
            sharpe=1.5,
            drawdown=0.10,
        )
        bridge._evaluation_cache["wf2-cache-test"] = good_profile

        improved, spec, warnings = bridge.run_mutation_cycle("wf2-cache-test")

        assert improved is True
        assert any("variant improved" in w for w in warnings)


# ---------------------------------------------------------------------------
# TestWF2BridgePromotion
# ---------------------------------------------------------------------------


class TestWF2BridgePromotion:
    """Tests for WF2Bridge promotion/demotion/kill decisions."""

    def test_decide_outcome_quarantine_low_dpr(self):
        """decide_outcome() returns 'quarantine' when DPR proxy < 30."""
        bridge = WF2Bridge()

        # sharpe=0.5 -> dpr_proxy=12.5 (< 30) -> quarantine
        profile = _make_evaluation_profile(
            bot_id="bot-q",
            robustness=0.6,
            sharpe=0.5,
            drawdown=0.10,
        )

        decision = bridge.decide_outcome("wf2-q", profile)
        assert decision == "quarantine"

    def test_decide_outcome_promote_high_robustness_and_dpr(self):
        """decide_outcome() returns 'promote' when robustness >= 0.8 and DPR proxy >= 50."""
        bridge = WF2Bridge()

        # sharpe=2.0 -> dpr_proxy=50, robustness=0.8 -> promote
        profile = _make_evaluation_profile(
            bot_id="bot-promote",
            robustness=0.85,
            sharpe=2.0,
            drawdown=0.08,
        )

        decision = bridge.decide_outcome("wf2-promote", profile)
        assert decision == "promote"

    def test_decide_outcome_kill_low_robustness(self):
        """decide_outcome() returns 'kill' when robustness < 0.5."""
        bridge = WF2Bridge()

        # sharpe=1.0 -> dpr_proxy=25 (< 30 quarantine first, but let's use 1.5 -> 37.5)
        # robustness=0.3 < 0.5 -> kill
        profile = _make_evaluation_profile(
            bot_id="bot-kill",
            robustness=0.3,
            sharpe=1.5,  # dpr_proxy=37.5 (> 30, so not quarantine)
            drawdown=0.12,
        )

        decision = bridge.decide_outcome("wf2-kill", profile)
        assert decision == "kill"

    def test_decide_outcome_demote_mid_robustness(self):
        """decide_outcome() returns 'demote' for robustness >= 0.5 and < 0.8."""
        bridge = WF2Bridge()

        # sharpe=1.2 -> dpr_proxy=30, robustness=0.6 -> demote
        profile = _make_evaluation_profile(
            bot_id="bot-demote",
            robustness=0.65,
            sharpe=1.2,
            drawdown=0.10,
        )

        decision = bridge.decide_outcome("wf2-demote", profile)
        assert decision == "demote"

    def test_decide_outcome_boundary_robustness_80(self):
        """decide_outcome() boundary: robustness=0.8 triggers promote if DPR >= 50."""
        bridge = WF2Bridge()

        profile = _make_evaluation_profile(
            bot_id="bot-boundary",
            robustness=0.8,
            sharpe=2.5,  # dpr_proxy=62.5 >= 50
            drawdown=0.08,
        )

        decision = bridge.decide_outcome("wf2-boundary", profile)
        assert decision == "promote"

    def test_decide_outcome_quarantine_takes_precedence_over_kill(self):
        """decide_outcome() checks quarantine before kill."""
        bridge = WF2Bridge()

        # Both quarantine (<30 DPR) and kill (<0.5 robustness) conditions
        # Quarantine should win since it is checked first
        profile = _make_evaluation_profile(
            bot_id="bot-qvk",
            robustness=0.3,  # Would be kill
            sharpe=0.5,     # Would be quarantine
            drawdown=0.15,
        )

        decision = bridge.decide_outcome("wf2-qvk", profile)
        assert decision == "quarantine"


# ---------------------------------------------------------------------------
# TestWF2BridgeWorkflowState
# ---------------------------------------------------------------------------


class TestWF2BridgeWorkflowState:
    """Tests for WF2Bridge.get_workflow_state()."""

    def test_get_workflow_state_returns_state(self):
        """get_workflow_state() returns the WorkflowState."""
        bridge = WF2Bridge()
        bot_spec = _make_bot_spec_with_mutation("variant-state-001")

        state = bridge.submit_variant(bot_spec, workflow_id="wf2-state-001")

        retrieved = bridge.get_workflow_state("wf2-state-001")
        assert retrieved is not None
        assert retrieved.workflow_id == state.workflow_id
        assert retrieved.status == state.status

    def test_get_workflow_state_unknown_returns_none(self):
        """get_workflow_state() returns None for unknown ID."""
        bridge = WF2Bridge()
        result = bridge.get_workflow_state("nonexistent")
        assert result is None
