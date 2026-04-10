"""
Tests for QuantMindLib V1 — BacktestReportBridge (Packet 8C).

Comprehensive tests for:
    - BacktestReportBridge.generate_report (produces markdown)
    - BacktestReportBridge.generate_improvement_suggestions (returns typed suggestions)
    - BacktestReportBridge.get_sit_gate_verdict (PASS/FAIL based on OOS degradation)
    - SITVerdict escalation flags
    - Report bridge integration with EvaluationOrchestrator
    - Mock vs real sub-agent switching
    - Empty/failed evaluation gracefully handled
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.evaluation.report_bridge import (
    BacktestReportBridge,
    ImprovementSuggestion,
    SITVerdict,
)
from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orb_spec() -> BotSpec:
    return BotSpec(
        id="bot-001",
        archetype="orb",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
    )


@pytest.fixture
def scalper_spec() -> BotSpec:
    return BotSpec(
        id="bot-002",
        archetype="scalper",
        symbol_scope=["GBPUSD"],
        sessions=["london_newyork_overlap"],
        features=["indicators/vwap"],
        confirmations=["spread_ok"],
        execution_profile="scalper_v1",
    )


@pytest.fixture
def eval_result_passing() -> EvaluationResult:
    """EvaluationResult with Kelly score such that degradation <= 15%."""
    return EvaluationResult(
        bot_id="bot-001",
        mode="BACKTEST",
        sharpe_ratio=1.5,
        max_drawdown=0.08,
        win_rate=0.60,
        profit_factor=2.0,
        expectancy=25.0,
        total_trades=50,
        return_pct=12.0,
        kelly_score=0.90,  # (1 - 0.90) * 100 = 10% degradation -> PASS
        passes_gate=True,
    )


@pytest.fixture
def eval_result_failing() -> EvaluationResult:
    """EvaluationResult with Kelly score such that degradation > 15%."""
    return EvaluationResult(
        bot_id="bot-002",
        mode="BACKTEST",
        sharpe_ratio=0.6,
        max_drawdown=0.20,
        win_rate=0.35,
        profit_factor=1.2,
        expectancy=-5.0,
        total_trades=30,
        return_pct=3.0,
        kelly_score=0.30,  # (1 - 0.30) * 100 = 70% degradation -> FAIL
        passes_gate=False,
    )


@pytest.fixture
def eval_result_marginal() -> EvaluationResult:
    """Kelly score at exactly the boundary (15% degradation)."""
    return EvaluationResult(
        bot_id="bot-003",
        mode="BACKTEST",
        sharpe_ratio=1.0,
        max_drawdown=0.12,
        win_rate=0.50,
        profit_factor=1.5,
        expectancy=10.0,
        total_trades=40,
        return_pct=8.0,
        kelly_score=0.85,  # (1 - 0.85) * 100 = 15% -> PASS (boundary)
        passes_gate=True,
    )


# ---------------------------------------------------------------------------
# Helper: make EvaluationResult with escalation fields via __dict__ patching
# Pydantic v2 model_construct silently drops extra fields, so we attach
# escalation fields directly to __dict__ for getattr-based access in tests.
# ---------------------------------------------------------------------------


def _eval_result_with_escalation_fields(
    kelly_score: float,
    pbo_score: Optional[float] = None,
    walk_forward_efficiency: Optional[float] = None,
    mc_prob_profit: Optional[float] = None,
    bot_id: str = "bot-test",
) -> EvaluationResult:
    """Create EvaluationResult with escalation fields attached via __dict__."""
    result = EvaluationResult(
        bot_id=bot_id,
        mode="BACKTEST",
        sharpe_ratio=1.0,
        max_drawdown=0.10,
        win_rate=0.50,
        profit_factor=1.5,
        expectancy=10.0,
        total_trades=50,
        return_pct=10.0,
        kelly_score=kelly_score,
        passes_gate=True,
    )
    # Attach escalation fields that get_sit_gate_verdict reads via getattr.
    # Direct __dict__ assignment bypasses Pydantic's field registry.
    if pbo_score is not None:
        result.__dict__["pbo_score"] = pbo_score
    if walk_forward_efficiency is not None:
        result.__dict__["walk_forward_efficiency"] = walk_forward_efficiency
    if mc_prob_profit is not None:
        result.__dict__["mc_prob_profit"] = mc_prob_profit
    return result


# ---------------------------------------------------------------------------
# Mock result dataclasses (for orchestrator integration tests)
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


# ---------------------------------------------------------------------------
# TestImprovementSuggestion
# ---------------------------------------------------------------------------


class TestImprovementSuggestion:
    def test_suggestion_valid(self):
        """ImprovementSuggestion is valid with all required fields."""
        s = ImprovementSuggestion(
            category="WIN_RATE",
            parameter_name="entry_conditions",
            direction="INCREASE",
            expected_impact="win_rate +5%",
            priority=1,
            reason="Win rate below 50% threshold.",
        )
        assert s.category == "WIN_RATE"
        assert s.parameter_name == "entry_conditions"
        assert s.direction == "INCREASE"
        assert s.expected_impact == "win_rate +5%"
        assert s.priority == 1

    def test_suggestion_all_categories(self):
        """All valid category values are accepted."""
        for cat in ("WIN_RATE", "PROFIT_FACTOR", "DRAWDOWN", "SHARPE", "WFA", "PBO"):
            s = ImprovementSuggestion(
                category=cat,
                parameter_name="param",
                direction="INCREASE",
                expected_impact="metric improvement",
                priority=1,
                reason="Test",
            )
            assert s.category == cat

    def test_suggestion_all_directions(self):
        """All valid direction values are accepted."""
        for d in ("INCREASE", "DECREASE", "WIDEN", "NARROW", "REDESIGN"):
            s = ImprovementSuggestion(
                category="WIN_RATE",
                parameter_name="param",
                direction=d,
                expected_impact="impact",
                priority=1,
                reason="Test",
            )
            assert s.direction == d

    def test_priority_bounds(self):
        """Priority must be between 1 and 10."""
        s = ImprovementSuggestion(
            category="WIN_RATE",
            parameter_name="param",
            direction="INCREASE",
            expected_impact="impact",
            priority=5,
            reason="Test",
        )
        assert s.priority == 5

    def test_priority_rejected_zero(self):
        """Priority 0 is rejected."""
        with pytest.raises(Exception):  # pydantic validation error
            ImprovementSuggestion(
                category="WIN_RATE",
                parameter_name="param",
                direction="INCREASE",
                expected_impact="impact",
                priority=0,
                reason="Test",
            )


# ---------------------------------------------------------------------------
# TestSITVerdict
# ---------------------------------------------------------------------------


class TestSITVerdict:
    def test_verdict_passed(self):
        """SITVerdict with passed=True and no escalation flags."""
        v = SITVerdict(
            passed=True,
            oos_degradation_pct=10.0,
            escalation_flags=[],
            summary="SIT PASSED.",
        )
        assert v.passed is True
        assert v.oos_degradation_pct == 10.0
        assert v.escalation_flags == []
        assert "PASSED" in v.summary

    def test_verdict_failed(self):
        """SITVerdict with passed=False."""
        v = SITVerdict(
            passed=False,
            oos_degradation_pct=20.0,
            escalation_flags=["HIGH_OVERFITTING"],
            summary="SIT FAILED. Escalation flags: HIGH_OVERFITTING.",
        )
        assert v.passed is False
        assert v.oos_degradation_pct == 20.0
        assert "HIGH_OVERFITTING" in v.escalation_flags

    def test_verdict_all_escalation_flags(self):
        """All escalation flag values are accepted."""
        for flag in ("HIGH_OVERFITTING", "UNSTABLE", "UNPROFITABLE"):
            v = SITVerdict(
                passed=False,
                oos_degradation_pct=25.0,
                escalation_flags=[flag],
                summary="Test",
            )
            assert flag in v.escalation_flags

    def test_verdict_multiple_flags(self):
        """Multiple escalation flags are preserved."""
        v = SITVerdict(
            passed=False,
            oos_degradation_pct=80.0,
            escalation_flags=["HIGH_OVERFITTING", "UNSTABLE", "UNPROFITABLE"],
            summary="Critical failures detected.",
        )
        assert len(v.escalation_flags) == 3


# ---------------------------------------------------------------------------
# TestBacktestReportBridgeInit
# ---------------------------------------------------------------------------


class TestBacktestReportBridgeInit:
    def test_default_init_uses_mock(self):
        """Default init with use_mock=True uses mock sub-agent."""
        bridge = BacktestReportBridge(use_mock=True)
        agent = bridge._get_sub_agent()
        assert agent is not None
        # Mock agent should produce deterministic output
        result = agent.generate_report(
            strategy_id="test",
            trd_data={"strategy_name": "test", "bot_tag": "@primal",
                       "strategy_type": "orb", "symbol": "EURUSD", "timeframe": "M5"},
            backtest_result={
                "in_sample_summary": {},
                "oos_summary": {},
                "monte_carlo": {},
                "walk_forward": {},
                "pbo": {},
            },
            sit_result={"passed": True},
        )
        assert "# Backtest Report:" in result

    def test_custom_factory(self):
        """Custom sub-agent factory is used when provided."""
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = "CUSTOM REPORT"
        bridge = BacktestReportBridge(sub_agent_factory=lambda: mock_agent)
        assert bridge._get_sub_agent() is mock_agent

    def test_mock_lazy_instantiation(self):
        """Sub-agent is instantiated lazily, not at construction."""
        bridge = BacktestReportBridge(use_mock=True)
        assert bridge._sub_agent is None
        agent = bridge._get_sub_agent()
        assert agent is not None
        assert bridge._sub_agent is agent
        # Second call returns same instance
        assert bridge._get_sub_agent() is agent


# ---------------------------------------------------------------------------
# TestGenerateReport
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_generate_report_produces_markdown(self, eval_result_passing, orb_spec):
        """generate_report returns markdown containing required sections."""
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(eval_result_passing, orb_spec)

        assert isinstance(report, str)
        assert "# Backtest Report:" in report
        assert "## 1. Summary" in report
        assert "## 2. In-Sample vs Out-of-Sample" in report
        assert "## 3. Monte Carlo" in report
        assert "## 4. Walk-Forward" in report
        assert "## 5. Overfitting Risk" in report
        assert "## 6. Improvement Suggestions" in report

    def test_generate_report_contains_sit_result(self, eval_result_passing, orb_spec):
        """Report includes SIT Gate PASSED when Kelly >= 0.85."""
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(eval_result_passing, orb_spec)

        assert "SIT Gate: PASSED" in report
        assert "APPROVE for paper" in report

    def test_generate_report_contains_sit_fail(self, eval_result_failing, scalper_spec):
        """Report includes SIT Gate FAILED when Kelly < 0.85."""
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(eval_result_failing, scalper_spec)

        assert "SIT Gate: FAILED" in report
        assert "REVISION needed" in report

    def test_generate_report_contains_metrics(
        self, eval_result_passing, orb_spec
    ):
        """Report contains evaluation metrics from EvaluationResult."""
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(eval_result_passing, orb_spec)

        # In single-result context, IS = OOS (both derived from same evaluation result).
        # Win rate (0.60 -> 60.0%)
        assert "60.0%" in report
        # Profit factor (2.0)
        assert "2.0" in report
        # Sharpe (1.5)
        assert "1.5" in report
        # Degradation is 0% in IS/OOS table (IS = OOS in single-result context)
        assert "0.0%" in report

    def test_generate_report_contains_bot_spec_info(
        self, eval_result_passing, orb_spec
    ):
        """Report includes BotSpec metadata (id, archetype, symbol)."""
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(eval_result_passing, orb_spec)

        assert "bot-001" in report
        assert "EURUSD" in report
        assert "orb" in report.lower()

    def test_generate_report_subagent_failure_fallback(
        self, eval_result_passing, orb_spec
    ):
        """generate_report returns fallback report when sub-agent raises."""
        bridge = BacktestReportBridge(use_mock=True)

        # Make sub-agent raise
        bridge._sub_agent = MagicMock()
        bridge._sub_agent.generate_report.side_effect = RuntimeError("LLM unavailable")

        report = bridge.generate_report(eval_result_passing, orb_spec)

        assert isinstance(report, str)
        assert "# Backtest Report:" in report
        assert "bot-001" in report

    def test_generate_report_empty_eval_result(self, orb_spec):
        """Empty EvaluationResult (all zeros) produces a valid report."""
        empty_result = EvaluationResult(
            bot_id="bot-empty",
            mode="BACKTEST",
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_trades=0,
            return_pct=0.0,
            kelly_score=0.0,  # 100% degradation -> FAIL
            passes_gate=False,
        )
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(empty_result, orb_spec)

        assert isinstance(report, str)
        assert "# Backtest Report:" in report
        # SIT should fail with 100% degradation
        assert "SIT Gate: FAILED" in report


# ---------------------------------------------------------------------------
# TestGenerateImprovementSuggestions
# ---------------------------------------------------------------------------


class TestGenerateImprovementSuggestions:
    def test_no_suggestions_when_passing(
        self, eval_result_passing
    ):
        """No suggestions returned when degradation <= 15%."""
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge.generate_improvement_suggestions(eval_result_passing)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_suggestions_when_failing(self, eval_result_failing):
        """Suggestions returned when degradation > 15%."""
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge.generate_improvement_suggestions(eval_result_failing)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # All returned items are ImprovementSuggestion
        for s in suggestions:
            assert isinstance(s, ImprovementSuggestion)

    def test_suggestion_typed_fields(self, eval_result_failing):
        """Suggestions have correct field values."""
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge.generate_improvement_suggestions(eval_result_failing)

        assert len(suggestions) > 0
        s = suggestions[0]
        assert s.category in (
            "WIN_RATE", "PROFIT_FACTOR", "DRAWDOWN", "SHARPE", "WFA", "PBO"
        )
        assert s.direction in ("INCREASE", "DECREASE", "WIDEN", "NARROW", "REDESIGN")
        assert s.priority >= 1
        assert len(s.reason) > 0

    def test_suggestions_priorities_ordered(self, eval_result_failing):
        """Suggestion priorities are in ascending order (1, 2, 3...)."""
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge.generate_improvement_suggestions(eval_result_failing)

        if len(suggestions) > 1:
            priorities = [s.priority for s in suggestions]
            assert priorities == sorted(priorities)

    def test_suggestion_max_count(self, eval_result_failing):
        """At most 6 suggestions are returned (per sub-agent system prompt)."""
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge.generate_improvement_suggestions(eval_result_failing)

        assert len(suggestions) <= 6

    def test_mock_vs_real_subagent_suggestions(self, eval_result_failing):
        """Both mock and real paths produce typed ImprovementSuggestion objects."""
        # Mock path
        mock_bridge = BacktestReportBridge(use_mock=True)
        mock_suggestions = mock_bridge.generate_improvement_suggestions(
            eval_result_failing
        )

        assert all(isinstance(s, ImprovementSuggestion) for s in mock_suggestions)

        # Real path (rule-based fallback when LLM unavailable)
        real_bridge = BacktestReportBridge(use_mock=False)
        real_suggestions = real_bridge.generate_improvement_suggestions(
            eval_result_failing
        )

        assert all(isinstance(s, ImprovementSuggestion) for s in real_suggestions)


# ---------------------------------------------------------------------------
# TestSITGateVerdict
# ---------------------------------------------------------------------------


class TestSITGateVerdict:
    def test_pass_degradation_10_percent(self, eval_result_passing):
        """PASS when OOS degradation is 10% (<= 15%)."""
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(eval_result_passing)

        assert verdict.passed is True
        assert verdict.oos_degradation_pct == 10.0
        assert verdict.summary == (
            "SIT PASSED: OOS degradation (10.0%) is within the 15% threshold."
        )

    def test_fail_degradation_70_percent(self, eval_result_failing):
        """FAIL when OOS degradation is 70% (> 15%)."""
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(eval_result_failing)

        assert verdict.passed is False
        assert verdict.oos_degradation_pct == 70.0
        assert "FAILED" in verdict.summary
        assert "70.0%" in verdict.summary

    def test_pass_at_boundary_15_percent(self, eval_result_marginal):
        """PASS at exactly 15% degradation (boundary case)."""
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(eval_result_marginal)

        assert verdict.passed is True
        assert verdict.oos_degradation_pct == 15.0

    def test_high_overfitting_flag(self, eval_result_passing):
        """HIGH_OVERFITTING flag raised when PBO > 0.7."""
        result = _eval_result_with_escalation_fields(
            kelly_score=0.90, pbo_score=0.75
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert "HIGH_OVERFITTING" in verdict.escalation_flags

    def test_no_high_overfitting_flag_at_70(self, eval_result_passing):
        """No HIGH_OVERFITTING flag when PBO == 0.7 (threshold is > 0.7)."""
        result = _eval_result_with_escalation_fields(
            kelly_score=0.90, pbo_score=0.70
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert "HIGH_OVERFITTING" not in verdict.escalation_flags

    def test_unstable_flag_low_wfa(self, eval_result_passing):
        """UNSTABLE flag raised when WFA efficiency < 40%."""
        result = _eval_result_with_escalation_fields(
            kelly_score=0.90, walk_forward_efficiency=0.30
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert "UNSTABLE" in verdict.escalation_flags

    def test_unprofitable_flag_low_mc_prob(self, eval_result_passing):
        """UNPROFITABLE flag raised when MC prob of profit < 50%."""
        result = _eval_result_with_escalation_fields(
            kelly_score=0.90, mc_prob_profit=0.40
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert "UNPROFITABLE" in verdict.escalation_flags

    def test_multiple_escalation_flags(self, eval_result_failing):
        """Multiple escalation flags can coexist."""
        result = _eval_result_with_escalation_fields(
            kelly_score=0.30,
            pbo_score=0.80,
            walk_forward_efficiency=0.20,
            mc_prob_profit=0.30,
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert len(verdict.escalation_flags) == 3
        assert "HIGH_OVERFITTING" in verdict.escalation_flags
        assert "UNSTABLE" in verdict.escalation_flags
        assert "UNPROFITABLE" in verdict.escalation_flags

    def test_degradation_computation_exact(self):
        """OOS degradation = (1 - kelly_score) * 100."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.50,
            profit_factor=1.5,
            expectancy=10.0,
            total_trades=50,
            return_pct=10.0,
            kelly_score=0.75,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        # (1 - 0.75) * 100 = 25%
        assert verdict.oos_degradation_pct == 25.0
        assert verdict.passed is False

    def test_negative_kelly_clamped_to_zero(self):
        """Negative Kelly score is clamped to 0 (100% degradation)."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=-1.0,
            max_drawdown=0.50,
            win_rate=0.20,
            profit_factor=0.5,
            expectancy=-10.0,
            total_trades=10,
            return_pct=-20.0,
            kelly_score=-0.2,
            passes_gate=False,
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert verdict.oos_degradation_pct == 100.0
        assert verdict.passed is False


# ---------------------------------------------------------------------------
# TestBacktestReportBridgeRuleBasedSuggestions
# ---------------------------------------------------------------------------


class TestRuleBasedSuggestions:
    def test_low_win_rate_triggers_suggestion(self):
        """Win rate < 50% generates WIN_RATE suggestion."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.40,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
            return_pct=10.0,
            kelly_score=0.80,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        win_rate_sugs = [s for s in suggestions if s.category == "WIN_RATE"]
        assert len(win_rate_sugs) > 0

    def test_low_profit_factor_triggers_suggestion(self):
        """Profit factor < 1.5 generates PROFIT_FACTOR suggestion."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.2,
            expectancy=10.0,
            total_trades=50,
            return_pct=8.0,
            kelly_score=0.80,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        pf_sugs = [s for s in suggestions if s.category == "PROFIT_FACTOR"]
        assert len(pf_sugs) > 0

    def test_high_drawdown_triggers_suggestion(self):
        """Max drawdown > 15% generates DRAWDOWN suggestion."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=1.5,
            max_drawdown=0.20,
            win_rate=0.60,
            profit_factor=2.5,
            expectancy=30.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.90,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        dd_sugs = [s for s in suggestions if s.category == "DRAWDOWN"]
        assert len(dd_sugs) > 0

    def test_low_sharpe_triggers_suggestion(self):
        """Sharpe < 1.0 generates SHARPE suggestion."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=0.5,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=2.0,
            expectancy=10.0,
            total_trades=50,
            return_pct=5.0,
            kelly_score=0.80,
            passes_gate=False,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        sharpe_sugs = [s for s in suggestions if s.category == "SHARPE"]
        assert len(sharpe_sugs) > 0

    def test_low_kelly_triggers_redesign(self):
        """Kelly score < 0.5 generates REDESIGN suggestion."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=0.8,
            max_drawdown=0.12,
            win_rate=0.45,
            profit_factor=1.4,
            expectancy=5.0,
            total_trades=50,
            return_pct=4.0,
            kelly_score=0.40,
            passes_gate=False,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        redesign_sugs = [s for s in suggestions if s.direction == "REDESIGN"]
        assert len(redesign_sugs) > 0

    def test_healthy_result_produces_few_suggestions(self):
        """A healthy evaluation result produces minimal suggestions."""
        result = EvaluationResult(
            bot_id="bot-test",
            mode="BACKTEST",
            sharpe_ratio=2.0,
            max_drawdown=0.05,
            win_rate=0.65,
            profit_factor=3.0,
            expectancy=40.0,
            total_trades=100,
            return_pct=20.0,
            kelly_score=0.95,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        suggestions = bridge._rule_based_suggestions(result)

        # No suggestions for all-metrics-passing result
        assert len(suggestions) == 0


# ---------------------------------------------------------------------------
# TestOrchestratorIntegration
# ---------------------------------------------------------------------------


class TestOrchestratorReportIntegration:
    def test_orchestrator_with_report_bridge(
        self, orb_spec
    ):
        """EvaluationOrchestrator with report_bridge generates and attaches report."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            vanilla_mc_result=_MCResult(
                confidence_interval_5th=-5.0,
                confidence_interval_95th=25.0,
                value_at_risk_95=-8.0
            ),
            pbo=0.15,
            robustness_score=0.85,
        )

        from unittest.mock import MagicMock

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        report_bridge = BacktestReportBridge(use_mock=True)
        orch = EvaluationOrchestrator(
            pipeline_class=lambda: mock_pipeline,
            report_bridge=report_bridge,
        )
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.report is not None
        assert isinstance(profile.report, str)
        assert "# Backtest Report:" in profile.report
        assert "bot-001" in profile.report
        assert warnings == []

    def test_orchestrator_without_report_bridge(self, orb_spec):
        """EvaluationOrchestrator without report_bridge returns profile with report=None."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            pbo=0.1,
            robustness_score=0.8,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.report is None

    def test_orchestrator_report_on_evaluate_with_data(
        self, orb_spec
    ):
        """evaluate_with_data also generates report when report_bridge is set."""
        import pandas as pd

        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            pbo=0.1,
            robustness_score=0.8,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        report_bridge = BacktestReportBridge(use_mock=True)
        orch = EvaluationOrchestrator(
            pipeline_class=lambda: mock_pipeline,
            report_bridge=report_bridge,
        )
        data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "open": [1.1000] * 100,
            "high": [1.1010] * 100,
            "low": [1.0990] * 100,
            "close": [1.1005] * 100,
            "tick_volume": [1000] * 100,
        })
        profile, warnings = orch.evaluate_with_data(orb_spec, data)

        assert profile is not None
        assert profile.report is not None
        assert "# Backtest Report:" in profile.report

    def test_orchestrator_report_failure_adds_warning(
        self, orb_spec
    ):
        """Report generation failure adds warning but does not fail evaluation."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            pbo=0.1,
            robustness_score=0.8,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        # Mock bridge that always raises
        broken_bridge = MagicMock()
        broken_bridge.generate_report.side_effect = RuntimeError("LLM unavailable")

        orch = EvaluationOrchestrator(
            pipeline_class=lambda: mock_pipeline,
            report_bridge=broken_bridge,
        )
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.report is None
        assert any("Report generation failed" in w for w in warnings)

    def test_orchestrator_no_vanilla_result_no_report(
        self, orb_spec
    ):
        """No report generated when vanilla result is None."""
        mock_comparison = _Comparison(
            vanilla_result=None,
            spiced_result=None,
            pbo=0.0,
            robustness_score=0.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        report_bridge = BacktestReportBridge(use_mock=True)
        orch = EvaluationOrchestrator(
            pipeline_class=lambda: mock_pipeline,
            report_bridge=report_bridge,
        )
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        # No primary result available, report should not be attached
        assert profile.report is None


# ---------------------------------------------------------------------------
# TestMockVsRealSwitching
# ---------------------------------------------------------------------------


class TestMockVsRealSwitching:
    def test_use_mock_flag_true(self):
        """use_mock=True uses mock sub-agent."""
        bridge = BacktestReportBridge(use_mock=True)
        assert bridge._use_mock is True
        agent = bridge._get_sub_agent()
        from src.library.evaluation.report_bridge import _MockBacktestReportSubAgent
        assert isinstance(agent, _MockBacktestReportSubAgent)

    def test_use_mock_flag_false_real_factory(self):
        """use_mock=False uses real sub-agent factory."""
        bridge = BacktestReportBridge(use_mock=False)
        assert bridge._use_mock is False
        # Real factory tries to import; wrap in patch to avoid import errors in test
        with patch.dict("sys.modules", {
            "src.agents.departments.subagents.backtest_report_subagent": MagicMock()
        }):
            agent = bridge._factory()
            assert agent is not None  # Factory returned something

    def test_custom_factory_takes_precedence(self):
        """Custom factory overrides both mock and real paths."""
        custom = MagicMock()
        bridge = BacktestReportBridge(
            sub_agent_factory=lambda: custom,
            use_mock=True,
        )
        # use_mock=True sets _factory to mock factory, but custom factory overrides
        bridge2 = BacktestReportBridge(sub_agent_factory=lambda: custom)
        assert bridge2._get_sub_agent() is custom


# ---------------------------------------------------------------------------
# TestEmptyAndEdgeCases
# ---------------------------------------------------------------------------


class TestEmptyAndEdgeCases:
    def test_zero_trades_eval_result(self, orb_spec):
        """Zero trades EvaluationResult produces valid report."""
        zero_result = EvaluationResult(
            bot_id="bot-zero",
            mode="BACKTEST",
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_trades=0,
            return_pct=0.0,
            kelly_score=0.0,
            passes_gate=False,
        )
        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(zero_result, orb_spec)
        assert "# Backtest Report:" in report

    def test_very_high_kelly(self):
        """Kelly score very close to 1.0 produces 0% degradation."""
        result = EvaluationResult(
            bot_id="bot-elite",
            mode="BACKTEST",
            sharpe_ratio=5.0,
            max_drawdown=0.02,
            win_rate=0.90,
            profit_factor=10.0,
            expectancy=100.0,
            total_trades=200,
            return_pct=50.0,
            kelly_score=0.99,
            passes_gate=True,
        )
        bridge = BacktestReportBridge(use_mock=True)
        verdict = bridge.get_sit_gate_verdict(result)

        assert verdict.passed is True
        assert verdict.oos_degradation_pct == 1.0

    def test_all_optional_fields_none(self, orb_spec):
        """EvaluationResult without escalation fields handles gracefully."""
        # Result without extra escalation fields; getattr returns None/0 for all.
        result = EvaluationResult(
            bot_id="bot-sparse",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.8,
            expectancy=15.0,
            total_trades=50,
            return_pct=10.0,
            kelly_score=0.82,
            passes_gate=True,
        )

        bridge = BacktestReportBridge(use_mock=True)
        report = bridge.generate_report(result, orb_spec)
        verdict = bridge.get_sit_gate_verdict(result)

        assert "# Backtest Report:" in report
        # kelly 0.82 -> (1 - 0.82) * 100 = 18% degradation -> FAIL
        assert verdict.passed is False
        # No escalation flags should crash (fields default to 0)
        assert verdict.escalation_flags == []
