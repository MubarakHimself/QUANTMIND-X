"""
Tests for QuantMindLib V1 — WF1Bridge (AlgoForge) (Packet 9B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.bridges.lifecycle_eval_workflow_bridges import (
    LifecycleBridge,
    WorkflowBridge,
    WorkflowArtifact,
)
from src.library.core.composition.trd_converter import TRDConverter
from src.library.core.domain.bot_spec import (
    BacktestMetrics,
    BotEvaluationProfile,
    BotSpec,
    MonteCarloMetrics,
    WalkForwardMetrics,
)
from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator
from src.library.workflows.stub_flows import AlgoForgeFlowStub
from src.library.workflows.wf1_bridge import WF1Bridge


# ---------------------------------------------------------------------------
# Shared mock comparison and pipeline dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _MT5Result:
    sharpe: float
    return_pct: float
    drawdown: float
    trades: int
    log: str = ""
    initial_cash: float = 10000.0
    final_cash: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    win_rate: float = 0.0


@dataclass
class _MCResult:
    confidence_interval_5th: float = 0.0
    confidence_interval_95th: float = 0.0
    value_at_risk_95: float = 0.0


@dataclass
class _Comparison:
    vanilla_result: Optional[_MT5Result] = None
    spiced_result: Optional[_MT5Result] = None
    vanilla_full_result: Optional[_MT5Result] = None
    spiced_full_result: Optional[_MT5Result] = None
    vanilla_mc_result: Optional[_MCResult] = None
    spiced_mc_result: Optional[_MCResult] = None
    pbo: float = 0.5
    robustness_score: float = 0.0


def _make_mock_pipeline(
    sharpe: float = 1.5,
    drawdown: float = 0.10,
    robustness: float = 0.85,
    pbo: float = 0.15,
) -> MagicMock:
    """Build a mock pipeline returning a configured _Comparison."""
    mock_pipeline = MagicMock()
    mock_pipeline.run_all_variants.return_value = _Comparison(
        vanilla_result=_MT5Result(
            sharpe=sharpe,
            return_pct=12.0,
            drawdown=drawdown,
            trades=50,
            trade_history=[],
            win_rate=60.0,
        ),
        vanilla_mc_result=_MCResult(
            confidence_interval_5th=-5.0,
            confidence_interval_95th=25.0,
            value_at_risk_95=-8.0,
        ),
        pbo=pbo,
        robustness_score=robustness,
    )
    return mock_pipeline


def _make_good_profile(bot_id: str) -> BotEvaluationProfile:
    """Build a BotEvaluationProfile that passes the paper gate."""
    return BotEvaluationProfile(
        bot_id=bot_id,
        backtest=BacktestMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            total_return=12.0,
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
            avg_sharpe=1.2,
            avg_return=10.0,
            stability=0.8,
        ),
        pbo_score=0.15,
        robustness_score=0.85,
        spread_sensitivity=0.1,
        session_scores={},
    )


def _make_weak_profile(bot_id: str) -> BotEvaluationProfile:
    """Build a BotEvaluationProfile that fails robustness gate."""
    return BotEvaluationProfile(
        bot_id=bot_id,
        backtest=BacktestMetrics(
            sharpe_ratio=0.8,
            max_drawdown=0.10,
            total_return=5.0,
            win_rate=0.55,
            profit_factor=1.5,
            expectancy=0.2,
            avg_bars_held=5.0,
            total_trades=30,
        ),
        monte_carlo=MonteCarloMetrics(
            n_simulations=1000,
            percentile_5_return=-10.0,
            percentile_95_return=10.0,
            max_drawdown_95=15.0,
            sharpe_confidence_width=20.0,
        ),
        walk_forward=WalkForwardMetrics(
            n_splits=5,
            avg_sharpe=0.5,
            avg_return=3.0,
            stability=0.3,
        ),
        pbo_score=0.5,
        robustness_score=0.4,  # < 0.6 gate
        spread_sensitivity=0.3,
        session_scores={},
    )


def _make_sharpeless_profile(bot_id: str) -> BotEvaluationProfile:
    """Build a BotEvaluationProfile that fails the backtest gate (sharpe < 1.0)."""
    return BotEvaluationProfile(
        bot_id=bot_id,
        backtest=BacktestMetrics(
            sharpe_ratio=0.5,
            max_drawdown=0.10,
            total_return=3.0,
            win_rate=0.50,
            profit_factor=1.2,
            expectancy=0.1,
            avg_bars_held=5.0,
            total_trades=20,
        ),
        monte_carlo=MonteCarloMetrics(
            n_simulations=1000,
            percentile_5_return=-5.0,
            percentile_95_return=10.0,
            max_drawdown_95=10.0,
            sharpe_confidence_width=15.0,
        ),
        walk_forward=WalkForwardMetrics(
            n_splits=5,
            avg_sharpe=0.6,
            avg_return=4.0,
            stability=0.5,
        ),
        pbo_score=0.3,
        robustness_score=0.65,  # >= 0.6 but sharpe < 1.0 gate
        spread_sensitivity=0.2,
        session_scores={},
    )


# ---------------------------------------------------------------------------
# TestAlgoForgeFlowStub
# ---------------------------------------------------------------------------


class TestAlgoForgeFlowStub:
    """Tests for AlgoForgeFlowStub."""

    def test_trigger_returns_wf1_prefixed_id(self):
        """trigger() returns a string prefixed with 'wf1-'."""
        result = AlgoForgeFlowStub.trigger({})
        assert isinstance(result, str)
        assert result.startswith("wf1-")

    def test_trigger_returns_unique_ids(self):
        """trigger() returns different IDs on each call."""
        ids = [AlgoForgeFlowStub.trigger({}) for _ in range(5)]
        assert len(set(ids)) == 5

    def test_get_status_returns_pending(self):
        """get_status() returns 'PENDING' by default."""
        status = AlgoForgeFlowStub.get_status("wf1-abc12345")
        assert status == "PENDING"

    def test_get_result_returns_none(self):
        """get_result() returns None by default."""
        result = AlgoForgeFlowStub.get_result("wf1-abc12345")
        assert result is None


# ---------------------------------------------------------------------------
# TestWF1BridgeInit
# ---------------------------------------------------------------------------


class TestWF1BridgeInit:
    """Tests for WF1Bridge initialization."""

    def test_default_init(self):
        """WF1Bridge initializes with default components."""
        bridge = WF1Bridge()
        assert bridge._trd_converter is not None
        assert type(bridge._trd_converter).__name__ == "TRDConverter"
        assert bridge._workflow_bridge is not None
        assert type(bridge._workflow_bridge).__name__ == "WorkflowBridge"
        assert bridge._flow_stub is not None
        assert type(bridge._flow_stub).__name__ == "AlgoForgeFlowStub"
        assert bridge._evaluation_cache == {}

    def test_custom_components(self):
        """WF1Bridge accepts custom TRDConverter, EvaluationOrchestrator, WorkflowBridge."""
        mock_eval = MagicMock(spec=EvaluationOrchestrator)
        mock_wb = WorkflowBridge()
        mock_converter = TRDConverter()

        bridge = WF1Bridge(
            trd_converter=mock_converter,
            evaluation_orchestrator=mock_eval,
            workflow_bridge=mock_wb,
        )
        assert bridge._trd_converter is mock_converter
        assert bridge._evaluation_orchestrator is mock_eval
        assert bridge._workflow_bridge is mock_wb

    def test_evaluation_cache_starts_empty(self):
        """_evaluation_cache is empty on init."""
        bridge = WF1Bridge()
        assert bridge._evaluation_cache == {}
        assert bridge._paper_start_ms == {}


# ---------------------------------------------------------------------------
# TestWF1BridgeSubmitTRD
# ---------------------------------------------------------------------------


class TestWF1BridgeSubmitTRD:
    """Tests for WF1Bridge.submit_trd()."""

    def test_submit_trd_converts_to_wf1_algoforge_workflow(self):
        """submit_trd() creates a workflow with workflow_name='WF1_ALGOFORGE'."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-trd-001",
            "strategy_type": "orb",
            "symbol_scope": ["EURUSD"],
            "sessions": ["london"],
            "features": ["indicators/atr_14"],
            "confirmations": ["spread_ok"],
            "execution_profile": "orb_v1",
        }

        state = bridge.submit_trd(valid_trd, workflow_id="wf1-manual-001")

        assert state is not None
        assert state.workflow_name == "WF1_ALGOFORGE"

    def test_submit_trd_with_valid_trd_returns_workflow_state(self):
        """Valid TRD dict returns a WorkflowState."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-trd-002",
            "strategy_type": "orb",
            "symbol_scope": ["EURUSD"],
        }

        state = bridge.submit_trd(valid_trd, workflow_id="wf1-manual-002")

        assert state is not None
        assert isinstance(state.workflow_id, str)
        assert state.workflow_id == "wf1-manual-002"

    def test_submit_trd_with_invalid_trd_returns_failed_state(self):
        """Invalid TRD (missing required fields) returns FAILED workflow."""
        bridge = WF1Bridge()

        # Missing required fields: robot_id, strategy_type, symbol_scope
        invalid_trd = {
            "robot_name": "Incomplete Bot",
            # no robot_id, strategy_type, or symbol_scope
        }

        state = bridge.submit_trd(invalid_trd, workflow_id="wf1-fail-001")

        assert state is not None
        assert state.status == "FAILED"
        assert state.workflow_name == "WF1_ALGOFORGE"

    def test_submit_trd_validation_failure_returns_failed(self):
        """BotSpec that fails validation returns FAILED workflow."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        # archetype that will fail StrategyCodeGenerator validation
        invalid_trd = {
            "robot_id": "bot-trd-003",
            "strategy_type": "unsupported_archetype",
            "symbol_scope": ["EURUSD"],
        }

        state = bridge.submit_trd(invalid_trd, workflow_id="wf1-fail-002")

        assert state is not None
        assert state.status == "FAILED"
        assert state.workflow_name == "WF1_ALGOFORGE"

    def test_submit_trd_stores_evaluation_in_cache(self):
        """Completed workflow has evaluation cached by workflow_id."""
        mock_pipeline = _make_mock_pipeline(robustness=0.85)
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-trd-004",
            "strategy_type": "orb",
            "symbol_scope": ["EURUSD"],
        }

        state = bridge.submit_trd(valid_trd, workflow_id="wf1-cached-001")

        assert state.status == "COMPLETED"
        assert state.workflow_id in bridge._evaluation_cache
        profile = bridge._evaluation_cache[state.workflow_id]
        assert profile is not None
        assert profile.robustness_score == 0.85

    def test_submit_trd_registers_workflow_in_workflow_bridge(self):
        """submit_trd() registers the workflow in the internal WorkflowBridge."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-trd-005",
            "strategy_type": "scalper",
            "symbol_scope": ["GBPUSD"],
        }

        state = bridge.submit_trd(valid_trd, workflow_id="wf1-reg-001")
        assert state.status == "COMPLETED"

        # Also accessible via WorkflowBridge
        wb_state = bridge._workflow_bridge.get_workflow("wf1-reg-001")
        assert wb_state is not None
        assert wb_state.status == "COMPLETED"

    def test_submit_trd_with_auto_generated_id(self):
        """submit_trd() auto-generates workflow_id when not provided."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-trd-006",
            "strategy_type": "orb",
            "symbol_scope": ["EURUSD"],
        }

        state = bridge.submit_trd(valid_trd)

        assert state is not None
        assert state.workflow_id.startswith("wf1-")


# ---------------------------------------------------------------------------
# TestWF1BridgeApproval
# ---------------------------------------------------------------------------


class TestWF1BridgeApproval:
    """Tests for WF1Bridge approval methods."""

    def test_approve_for_paper_requires_robustness(self):
        """approve_for_paper returns False if robustness_score < 0.6."""
        bridge = WF1Bridge()

        # Pre-populate cache with weak robustness profile
        weak_profile = _make_weak_profile("bot-weak-001")
        bridge._evaluation_cache["wf1-weak-001"] = weak_profile

        # Register the workflow in WorkflowBridge (required by approve_for_paper)
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-weak-001",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-weak-001")

        approved, reason = bridge.approve_for_paper("wf1-weak-001")

        assert approved is False
        assert "robustness_too_low" in reason

    def test_approve_for_paper_requires_passes_gate(self):
        """approve_for_paper returns False if backtest gate fails."""
        bridge = WF1Bridge()

        # Profile with robustness >= 0.6 but sharpe < 1.0
        sharpe_less_profile = _make_sharpeless_profile("bot-sharpeless-001")
        bridge._evaluation_cache["wf1-sharpe-001"] = sharpe_less_profile

        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-sharpe-001",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-sharpe-001")

        approved, reason = bridge.approve_for_paper("wf1-sharpe-001")

        assert approved is False
        assert "backtest_gate_failed" in reason

    def test_approve_for_paper_approves_good_strategy(self):
        """Strategy with robustness >= 0.6 and sharpe >= 1.0 is approved."""
        bridge = WF1Bridge()

        good_profile = _make_good_profile("bot-good-001")
        bridge._evaluation_cache["wf1-good-001"] = good_profile

        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-good-001",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-good-001")

        approved, reason = bridge.approve_for_paper("wf1-good-001")

        assert approved is True
        assert "approved" in reason
        assert "0.850" in reason

    def test_approve_for_paper_unknown_workflow(self):
        """approve_for_paper returns False for unknown workflow_id."""
        bridge = WF1Bridge()
        approved, reason = bridge.approve_for_paper("unknown-workflow-id")
        assert approved is False
        assert reason == "workflow_not_found"

    def test_approve_for_paper_no_profile(self):
        """approve_for_paper returns False when no evaluation profile is cached."""
        bridge = WF1Bridge()
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-no-profile",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-no-profile")

        approved, reason = bridge.approve_for_paper("wf1-no-profile")
        assert approved is False
        assert reason == "no_evaluation_profile"

    def test_check_evaluation_ready_returns_false_for_unknown_workflow(self):
        """check_evaluation_ready returns (False, None) for unknown workflow_id."""
        bridge = WF1Bridge()
        is_ready, profile = bridge.check_evaluation_ready("nonexistent-id")
        assert is_ready is False
        assert profile is None

    def test_check_evaluation_ready_returns_false_for_incomplete(self):
        """check_evaluation_ready returns (False, None) for non-COMPLETED workflow."""
        bridge = WF1Bridge()
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-running",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.start_workflow("wf1-running")

        is_ready, profile = bridge.check_evaluation_ready("wf1-running")
        assert is_ready is False
        assert profile is None

    def test_check_evaluation_ready_returns_true_for_completed(self):
        """Completed workflow returns (True, profile)."""
        bridge = WF1Bridge()
        good_profile = _make_good_profile("bot-ready-001")
        bridge._evaluation_cache["wf1-ready-001"] = good_profile
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-ready-001",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-ready-001")

        is_ready, profile = bridge.check_evaluation_ready("wf1-ready-001")

        assert is_ready is True
        assert profile is not None
        assert profile.bot_id == "bot-ready-001"
        assert profile.robustness_score == 0.85


# ---------------------------------------------------------------------------
# TestWF1BridgePaperReady
# ---------------------------------------------------------------------------


class TestWF1BridgePaperReady:
    """Tests for paper -> live promotion."""

    def test_check_paper_ready_returns_false_when_not_started(self):
        """check_paper_ready returns False if paper hasn't started."""
        bridge = WF1Bridge()

        # Workflow exists but no paper_start_ms recorded
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-paper-none",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )
        bridge._workflow_bridge.complete_workflow("wf1-paper-none")

        result = bridge.check_paper_ready_for_live("wf1-paper-none")
        assert result is False

    def test_check_paper_ready_returns_false_for_unknown_workflow(self):
        """check_paper_ready returns False for unknown workflow_id."""
        bridge = WF1Bridge()
        result = bridge.check_paper_ready_for_live("unknown-id")
        assert result is False

    def test_check_paper_ready_uses_lifecycle_timing(self):
        """3-day paper lag is enforced via LifecycleBridge."""
        bridge = WF1Bridge()

        # Set paper_start_ms to 4 days ago (exceeds 3-day minimum)
        four_days_ms = 4 * 24 * 60 * 60 * 1000
        import time
        bridge._paper_start_ms["wf1-paper-ready"] = int(time.time() * 1000) - four_days_ms
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-paper-ready",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        result = bridge.check_paper_ready_for_live("wf1-paper-ready")
        assert result is True

    def test_check_paper_ready_within_3_days(self):
        """check_paper_ready returns False within the 3-day paper window."""
        bridge = WF1Bridge()

        # Set paper_start_ms to 1 day ago (within 3-day window)
        one_day_ms = 1 * 24 * 60 * 60 * 1000
        import time
        bridge._paper_start_ms["wf1-paper-young"] = int(time.time() * 1000) - one_day_ms
        bridge._workflow_bridge.register_workflow(
            workflow_id="wf1-paper-young",
            workflow_name="WF1_ALGOFORGE",
            input_artifacts=[],
        )

        result = bridge.check_paper_ready_for_live("wf1-paper-young")
        assert result is False


# ---------------------------------------------------------------------------
# TestWF1BridgeWorkflowState
# ---------------------------------------------------------------------------


class TestWF1BridgeWorkflowState:
    """Tests for WF1Bridge.get_workflow_state()."""

    def test_get_workflow_state_returns_state(self):
        """get_workflow_state() returns the WorkflowState."""
        mock_pipeline = _make_mock_pipeline()
        mock_eval = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        bridge = WF1Bridge(evaluation_orchestrator=mock_eval)

        valid_trd = {
            "robot_id": "bot-state-001",
            "strategy_type": "orb",
            "symbol_scope": ["EURUSD"],
        }
        state = bridge.submit_trd(valid_trd, workflow_id="wf1-state-001")

        retrieved = bridge.get_workflow_state("wf1-state-001")
        assert retrieved is not None
        assert retrieved.workflow_id == state.workflow_id
        assert retrieved.status == state.status

    def test_get_workflow_state_unknown_returns_none(self):
        """get_workflow_state() returns None for unknown ID."""
        bridge = WF1Bridge()
        result = bridge.get_workflow_state("nonexistent")
        assert result is None
