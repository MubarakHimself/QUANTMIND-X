"""
Tests for QuantMindLib V1 — EvaluationOrchestrator (Packet 8B).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.core.bridges.lifecycle_eval_workflow_bridges import EvaluationBridge
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
def unsupported_spec() -> BotSpec:
    return BotSpec(
        id="bot-003",
        archetype="unsupported_archetype",
        symbol_scope=["EURUSD"],
        sessions=[],
        features=[],
        confirmations=[],
        execution_profile="default",
    )


@dataclass
class _MT5Result:
    """Plain dataclass simulating MT5BacktestResult fields."""
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
class _SpicedResult(_MT5Result):
    """Plain dataclass simulating SpicedBacktestResult fields."""
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    filtered_trades: int = 0
    avg_regime_quality: float = 0.0


@dataclass
class _MCResult:
    """Plain dataclass simulating MonteCarloResult fields."""
    confidence_interval_5th: float = 0.0
    confidence_interval_95th: float = 0.0
    value_at_risk_95: float = 0.0


@dataclass
class _Comparison:
    """Plain dataclass simulating BacktestComparison fields."""
    vanilla_result: Optional[_MT5Result] = None
    spiced_result: Optional[_SpicedResult] = None
    vanilla_full_result: Optional[_MT5Result] = None
    spiced_full_result: Optional[_SpicedResult] = None
    vanilla_mc_result: Optional[_MCResult] = None
    spiced_mc_result: Optional[_MCResult] = None
    pbo: float = 0.5
    robustness_score: float = 0.0


# ---------------------------------------------------------------------------
# TestOrchestratorInit
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_default_init(self):
        """Default init creates internal StrategyCodeGenerator and EvaluationBridge."""
        orch = EvaluationOrchestrator()
        assert orch._code_generator is not None
        assert orch._evaluation_bridge is not None

    def test_custom_components(self):
        """Custom components are used when provided."""
        code_gen = MagicMock()
        eval_bridge = EvaluationBridge()
        orch = EvaluationOrchestrator(
            code_generator=code_gen,
            evaluation_bridge=eval_bridge,
        )
        assert orch._code_generator is code_gen
        assert orch._evaluation_bridge is eval_bridge


# ---------------------------------------------------------------------------
# TestEvaluateBotSpec
# ---------------------------------------------------------------------------


class TestEvaluateBotSpec:
    def test_valid_evaluation_produces_profile(self, orb_spec: BotSpec):
        """Valid BotSpec evaluation returns a BotEvaluationProfile."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            spiced_result=_SpicedResult(
                sharpe=1.8, return_pct=15.0, drawdown=0.07,
                trades=40, trade_history=[], win_rate=65.0,
                regime_distribution={"TRENDING": 20, "RANGING": 15},
                filtered_trades=5
            ),
            vanilla_full_result=_MT5Result(
                sharpe=1.2, return_pct=10.0, drawdown=0.12,
                trades=40, trade_history=[], win_rate=58.0
            ),
            spiced_full_result=_SpicedResult(
                sharpe=1.4, return_pct=12.0, drawdown=0.09,
                trades=32, trade_history=[], win_rate=62.0
            ),
            vanilla_mc_result=_MCResult(
                confidence_interval_5th=-5.0,
                confidence_interval_95th=25.0,
                value_at_risk_95=-8.0
            ),
            pbo=0.15,
            robustness_score=0.85,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.bot_id == "bot-001"
        assert profile.backtest.sharpe_ratio == 1.5
        assert profile.backtest.total_trades == 50
        assert profile.pbo_score == 0.15
        assert profile.robustness_score == 0.85
        assert warnings == []

    def test_invalid_spec_returns_none(self, unsupported_spec: BotSpec):
        """Unsupported archetype returns (None, errors)."""
        orch = EvaluationOrchestrator()
        profile, errors = orch.evaluate(unsupported_spec)

        assert profile is None
        assert len(errors) > 0
        assert any("Unsupported archetype" in e for e in errors)

    def test_missing_archetype_returns_errors(self):
        """Missing archetype in spec returns validation errors."""
        spec = BotSpec(
            id="bot-x",
            archetype="",
            symbol_scope=["EURUSD"],
            sessions=[],
            features=[],
            confirmations=[],
            execution_profile="default",
        )
        orch = EvaluationOrchestrator()
        profile, errors = orch.evaluate(spec)

        assert profile is None
        assert len(errors) > 0
        assert any("archetype" in e.lower() for e in errors)

    def test_valid_evaluation_with_warnings(self, scalper_spec: BotSpec):
        """Evaluation produces warnings when no backtest results are available."""
        mock_comparison = _Comparison(
            vanilla_result=None,
            spiced_result=None,
            vanilla_full_result=None,
            spiced_full_result=None,
            vanilla_mc_result=None,
            pbo=0.5,
            robustness_score=0.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(scalper_spec)

        assert profile is not None
        assert "No backtest results" in warnings[0]


# ---------------------------------------------------------------------------
# TestEvaluateWithData
# ---------------------------------------------------------------------------


class TestEvaluateWithData:
    def test_evaluation_with_provided_data(self, orb_spec: BotSpec):
        """evaluate_with_data uses provided DataFrame and runs pipeline."""
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

        data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "open": [1.1000] * 100,
            "high": [1.1010] * 100,
            "low": [1.0990] * 100,
            "close": [1.1005] * 100,
            "tick_volume": [1000] * 100,
        })

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate_with_data(orb_spec, data)

        assert profile is not None
        call_kwargs = mock_pipeline.run_all_variants.call_args.kwargs
        assert call_kwargs["data"] is data

    def test_data_param_is_used(self, scalper_spec: BotSpec):
        """The data parameter is passed to run_all_variants."""
        import pandas as pd

        mock_comparison = _Comparison(
            vanilla_result=None,
            pbo=0.0,
            robustness_score=0.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        data = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=50, freq="1h"),
            "open": [1.2000] * 50,
            "high": [1.2010] * 50,
            "low": [1.1990] * 50,
            "close": [1.2005] * 50,
            "tick_volume": [500] * 50,
        })

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        _ = orch.evaluate_with_data(scalper_spec, data)

        call_kwargs = mock_pipeline.run_all_variants.call_args.kwargs
        assert call_kwargs["data"] is data


# ---------------------------------------------------------------------------
# TestBacktestResultConversion
# ---------------------------------------------------------------------------


class TestBacktestResultConversion:
    def test_mt5_result_fields_extracted(self):
        """MT5BacktestResult fields are correctly extracted."""
        result = _MT5Result(
            sharpe=1.5, return_pct=12.5, drawdown=0.10,
            trades=50, trade_history=[], win_rate=60.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result(
            "bot-001", "VANILLA", result
        )

        assert eval_result.bot_id == "bot-001"
        assert eval_result.sharpe_ratio == 1.5
        assert eval_result.max_drawdown == 0.10
        assert eval_result.total_trades == 50
        assert eval_result.return_pct == 12.5
        assert eval_result.mode == "BACKTEST"

    def test_trade_history_computes_win_rate(self):
        """Trade history is used to compute win_rate when available."""
        result = _MT5Result(
            sharpe=1.2, return_pct=8.0, drawdown=0.08,
            trades=10,
            trade_history=[
                {"profit": 100.0},   # win
                {"profit": -50.0},   # loss
                {"profit": 80.0},    # win
                {"profit": -30.0},   # loss
                {"profit": 120.0},   # win
                {"profit": -20.0},   # loss
                {"profit": 90.0},    # win
                {"profit": -40.0},   # loss
                {"profit": 110.0},   # win
                {"profit": 60.0},    # win
            ],
            win_rate=0.0,  # Should be overridden by trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result(
            "bot-001", "VANILLA", result
        )

        # 6 winning trades out of 10 = 0.6 (100, 80, 120, 90, 110, 60)
        assert eval_result.win_rate == pytest.approx(0.6, rel=1e-3)
        # Expectancy: (100 - 50 + 80 - 30 + 120 - 20 + 90 - 40 + 110 + 60) / 10 = 42.0
        assert eval_result.expectancy == pytest.approx(42.0, rel=1e-3)
        # gross_profit = 100+80+120+90+110+60 = 560, gross_loss = 50+30+20+40 = 140
        # profit_factor = 560/140 = 4.0
        assert eval_result.profit_factor == pytest.approx(4.0, rel=1e-3)

    def test_passes_gate_true(self):
        """passes_gate is True when sharpe >= 1.0 and drawdown <= 0.15."""
        result = _MT5Result(
            sharpe=1.5, return_pct=12.5, drawdown=0.10,
            trades=50, trade_history=[], win_rate=60.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result(
            "bot-001", "VANILLA", result
        )

        assert eval_result.passes_gate is True
        assert eval_result.kelly_score > 0.0

    def test_passes_gate_false(self):
        """passes_gate is False when sharpe < 1.0 or drawdown > 0.15."""
        result = _MT5Result(
            sharpe=0.5, return_pct=2.0, drawdown=0.10,
            trades=5, trade_history=[], win_rate=40.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result(
            "bot-x", "VANILLA", result
        )

        assert eval_result.passes_gate is False
        assert eval_result.sharpe_ratio == 0.5

    def test_spiced_result_includes_regime_info(self):
        """SpicedBacktestResult -> EvaluationResult includes regime info."""
        result = _SpicedResult(
            sharpe=1.8, return_pct=15.0, drawdown=0.07,
            trades=40, trade_history=[], win_rate=65.0,
            regime_distribution={"TRENDING": 20, "RANGING": 15, "HIGH_CHAOS": 5},
            filtered_trades=5,
            avg_regime_quality=0.75
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._spiced_result_to_evaluation_result(
            "bot-001", result
        )

        assert eval_result.bot_id == "bot-001"
        assert eval_result.sharpe_ratio == 1.8
        assert eval_result.max_drawdown == 0.07
        assert eval_result.filtered_trades == 5
        assert eval_result.regime_distribution == {"TRENDING": 20, "RANGING": 15, "HIGH_CHAOS": 5}


# ---------------------------------------------------------------------------
# TestComparisonToProfile
# ---------------------------------------------------------------------------


class TestComparisonToProfile:
    def test_comparison_with_all_results(self):
        """Full BacktestComparison -> BotEvaluationProfile with all metrics."""
        comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            spiced_result=_SpicedResult(
                sharpe=1.8, return_pct=15.0, drawdown=0.07,
                trades=40, trade_history=[], win_rate=65.0,
                regime_distribution={}, filtered_trades=0
            ),
            vanilla_full_result=_MT5Result(
                sharpe=1.2, return_pct=10.0, drawdown=0.12,
                trades=40, trade_history=[], win_rate=58.0
            ),
            spiced_full_result=_SpicedResult(
                sharpe=1.4, return_pct=12.0, drawdown=0.09,
                trades=32, trade_history=[], win_rate=62.0,
                regime_distribution={}, filtered_trades=0
            ),
            vanilla_mc_result=_MCResult(
                confidence_interval_5th=-5.0,
                confidence_interval_95th=25.0,
                value_at_risk_95=-8.0
            ),
            pbo=0.15,
            robustness_score=0.85,
        )

        orch = EvaluationOrchestrator()
        profile = orch._comparison_to_profile("bot-001", comparison)

        assert profile.bot_id == "bot-001"
        assert profile.backtest.sharpe_ratio == 1.5
        assert profile.backtest.max_drawdown == 0.10
        assert profile.backtest.total_trades == 50
        assert profile.monte_carlo.percentile_5_return == -5.0
        assert profile.monte_carlo.percentile_95_return == 25.0
        assert profile.walk_forward.avg_sharpe == 1.2
        assert profile.walk_forward.avg_return == 10.0
        assert profile.pbo_score == 0.15
        assert profile.robustness_score == 0.85

    def test_comparison_with_partial_results(self):
        """Partial BacktestComparison returns defaults for missing results."""
        comparison = _Comparison(
            vanilla_result=None,
            spiced_result=None,
            vanilla_full_result=None,
            spiced_full_result=None,
            vanilla_mc_result=None,
            pbo=0.3,
            robustness_score=0.5,
        )

        orch = EvaluationOrchestrator()
        profile = orch._comparison_to_profile("bot-partial", comparison)

        assert profile.bot_id == "bot-partial"
        assert profile.backtest.sharpe_ratio == 0.0
        assert profile.walk_forward.avg_sharpe == 0.0
        assert profile.pbo_score == 0.3
        assert profile.robustness_score == 0.5

    def test_pbo_and_robustness_in_profile(self):
        """PBO and robustness from BacktestComparison are in profile."""
        comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            pbo=0.2,
            robustness_score=0.75,
        )

        orch = EvaluationOrchestrator()
        profile = orch._comparison_to_profile("bot-001", comparison)

        assert profile.pbo_score == 0.2
        assert profile.robustness_score == 0.75
        assert profile.spread_sensitivity == 0.0
        assert profile.session_scores == {}


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_backtest_failure_returns_errors(self, orb_spec: BotSpec):
        """Backtest pipeline exception returns (None, errors)."""
        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.side_effect = RuntimeError("Data fetch failed")

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, errors = orch.evaluate(orb_spec)

        assert profile is None
        assert len(errors) == 1
        assert "Data fetch failed" in errors[0]

    def test_empty_comparison_returns_defaults(self, orb_spec: BotSpec):
        """Empty BacktestComparison returns profile with defaults."""
        mock_comparison = _Comparison(
            vanilla_result=None,
            spiced_result=None,
            vanilla_full_result=None,
            spiced_full_result=None,
            vanilla_mc_result=None,
            spiced_mc_result=None,
            pbo=0.0,
            robustness_score=0.0,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.bot_id == "bot-001"
        assert profile.backtest.sharpe_ratio == 0.0
        assert profile.walk_forward.avg_sharpe == 0.0
        assert "No backtest results" in warnings[0]
