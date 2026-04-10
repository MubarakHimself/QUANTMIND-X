"""
Tests for QuantMindLib V1 — Evaluation Integration (Packet 8C).

Integration tests for the complete evaluation pipeline:
StrategyCodeGenerator + EvaluationOrchestrator + EvaluationBridge.

Uses MagicMock for FullBacktestPipeline and DataManager (they require Dukascopy data).
Uses real internal components (StrategyCodeGenerator, EvaluationOrchestrator, EvaluationBridge).
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

sys.path.insert(0, "src")

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.core.bridges.lifecycle_eval_workflow_bridges import EvaluationBridge
from src.library.evaluation.strategy_code_generator import StrategyCodeGenerator
from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator


# ---------------------------------------------------------------------------
# Mock result dataclasses (matching real result structures)
# ---------------------------------------------------------------------------


@dataclass
class _MT5Result:
    """Simulates MT5BacktestResult fields."""
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
    """Simulates SpicedBacktestResult fields."""
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    filtered_trades: int = 0
    avg_regime_quality: float = 0.0


@dataclass
class _MCResult:
    """Simulates MonteCarloResult fields."""
    confidence_interval_5th: float = 0.0
    confidence_interval_95th: float = 0.0
    value_at_risk_95: float = 0.0


@dataclass
class _Comparison:
    """Simulates BacktestComparison fields."""
    vanilla_result: Optional[_MT5Result] = None
    spiced_result: Optional[_SpicedResult] = None
    vanilla_full_result: Optional[_MT5Result] = None
    spiced_full_result: Optional[_SpicedResult] = None
    vanilla_mc_result: Optional[_MCResult] = None
    spiced_mc_result: Optional[_MCResult] = None
    pbo: float = 0.5
    robustness_score: float = 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def orb_spec() -> BotSpec:
    return BotSpec(
        id="test-orb-1",
        archetype="orb",
        symbol_scope=["EURUSD"],
        sessions=["LONDON_AM"],
        features=["indicators/macd", "indicators/vwap"],
        confirmations=[],
        execution_profile="PAPER",
    )


@pytest.fixture
def scalper_spec() -> BotSpec:
    return BotSpec(
        id="test-scalper-1",
        archetype="scalper",
        symbol_scope=["GBPUSD"],
        sessions=["london_newyork_overlap"],
        features=["indicators/vwap"],
        confirmations=["spread_ok"],
        execution_profile="PAPER",
    )


@pytest.fixture
def unsupported_spec() -> BotSpec:
    return BotSpec(
        id="test-unsupported-1",
        archetype="unsupported_archetype",
        symbol_scope=["EURUSD"],
        sessions=[],
        features=[],
        confirmations=[],
        execution_profile="default",
    )


@pytest.fixture
def invalid_symbol_spec() -> BotSpec:
    return BotSpec(
        id="test-invalid-sym-1",
        archetype="orb",
        symbol_scope=["INVALID"],
        sessions=["LONDON_AM"],
        features=["indicators/macd"],
        confirmations=[],
        execution_profile="PAPER",
    )


# ---------------------------------------------------------------------------
# TestStrategyCodeGeneratorIntegration
# ---------------------------------------------------------------------------


class TestStrategyCodeGeneratorIntegration:
    """Integration tests for StrategyCodeGenerator with real BotSpecs."""

    def test_orb_bot_spec_generates_valid_strategy(self, orb_spec: BotSpec) -> None:
        """ORB BotSpec generates a valid, executable-looking Python class."""
        gen = StrategyCodeGenerator()
        code = gen.generate(orb_spec)

        assert isinstance(code, str)
        assert len(code) > 0
        # Must parse as valid Python AST
        ast.parse(code)
        # Must contain the on_bar function
        assert "def on_bar" in code
        # Must reference the symbol
        assert "EURUSD" in code
        # Must reference the archetype name
        assert "ORB" in code or "OpeningRangeBreakout" in code

    def test_scalper_bot_spec_generates_vwap_strategy(self, scalper_spec: BotSpec) -> None:
        """Scalper BotSpec generates VWAP-based strategy code."""
        gen = StrategyCodeGenerator()
        code = gen.generate(scalper_spec)

        assert isinstance(code, str)
        ast.parse(code)
        assert "def on_bar" in code
        assert "GBPUSD" in code
        # Must include VWAP computation
        assert "vwap" in code.lower() or "_compute_vwap" in code
        # Must include RSI computation
        assert "rsi" in code.lower() or "_compute_rsi" in code
        # Must include position management
        assert "_position" in code
        # Must include trading calls
        assert "tester.buy" in code
        assert "tester.sell" in code

    def test_generated_code_includes_all_required_components(
        self, orb_spec: BotSpec
    ) -> None:
        """Generated code has __init__, on_bar, and signal logic."""
        gen = StrategyCodeGenerator()
        code = gen.generate(orb_spec)

        ast.parse(code)
        assert "def on_bar" in code
        # ORB-specific components
        assert "RANGE_BARS" in code
        assert "_range_high" in code
        assert "_range_low" in code
        assert "_in_range" in code
        assert "_position" in code
        # Breakout signal logic
        assert "close > _range_high" in code or "> _range_high" in code
        assert "close < _range_low" in code or "< _range_low" in code
        # Session awareness
        assert "LONDON_AM" in code or "london" in code.lower()

    def test_validation_rejects_invalid_symbol(self, invalid_symbol_spec: BotSpec) -> None:
        """Invalid symbol format is rejected during validation."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(invalid_symbol_spec)

        assert is_valid is False
        assert len(errors) > 0
        assert any("symbol" in e.lower() for e in errors)

    def test_validation_accepts_valid_bot_spec(self, orb_spec: BotSpec) -> None:
        """Valid ORB BotSpec passes validation."""
        gen = StrategyCodeGenerator()
        is_valid, errors = gen.validate_bot_spec(orb_spec)

        assert is_valid is True
        assert errors == []


# ---------------------------------------------------------------------------
# TestEvaluationOrchestratorIntegration
# ---------------------------------------------------------------------------


class TestEvaluationOrchestratorIntegration:
    """Integration tests for EvaluationOrchestrator with mocked backtest pipeline."""

    def test_evaluate_returns_profile_with_all_metrics(self, orb_spec: BotSpec) -> None:
        """Full evaluation returns BotEvaluationProfile with all required fields."""
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
        assert profile.bot_id == "test-orb-1"
        # Backtest metrics
        assert profile.backtest.sharpe_ratio == 1.5
        assert profile.backtest.max_drawdown == 0.10
        assert profile.backtest.total_trades == 50
        # Monte Carlo metrics
        assert profile.monte_carlo.percentile_5_return == -5.0
        assert profile.monte_carlo.percentile_95_return == 25.0
        assert profile.monte_carlo.max_drawdown_95 == -8.0
        # Walk-forward metrics
        assert profile.walk_forward.avg_sharpe == 1.2
        assert profile.walk_forward.avg_return == 10.0
        # PBO and robustness
        assert profile.pbo_score == 0.15
        assert profile.robustness_score == 0.85
        # Spread sensitivity
        assert profile.spread_sensitivity == 0.0
        # Session scores
        assert profile.session_scores == {}

    def test_evaluate_with_data_passes_data_to_pipeline(
        self, orb_spec: BotSpec
    ) -> None:
        """Provided data is passed to FullBacktestPipeline."""
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
        assert call_kwargs["symbol"] == "EURUSD"
        assert isinstance(call_kwargs["strategy_code"], str)
        assert len(call_kwargs["strategy_code"]) > 0

    def test_invalid_archetype_returns_errors(self, unsupported_spec: BotSpec) -> None:
        """Unsupported archetype returns validation errors."""
        orch = EvaluationOrchestrator()
        profile, errors = orch.evaluate(unsupported_spec)

        assert profile is None
        assert len(errors) > 0
        assert any("Unsupported archetype" in e for e in errors)

    def test_partial_results_produce_partial_profile(
        self, scalper_spec: BotSpec
    ) -> None:
        """Pipeline failure still produces profile with defaults."""
        mock_comparison = _Comparison(
            vanilla_result=None,
            spiced_result=None,
            vanilla_full_result=None,
            spiced_full_result=None,
            vanilla_mc_result=None,
            spiced_mc_result=None,
            pbo=0.3,
            robustness_score=0.5,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(scalper_spec)

        assert profile is not None
        assert profile.bot_id == "test-scalper-1"
        assert profile.backtest.sharpe_ratio == 0.0
        assert profile.walk_forward.avg_sharpe == 0.0
        assert profile.pbo_score == 0.3
        assert profile.robustness_score == 0.5
        assert "No backtest results" in warnings[0]

    def test_pbo_and_robustness_included_in_profile(self, orb_spec: BotSpec) -> None:
        """PBO and robustness scores are populated in BotEvaluationProfile."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            pbo=0.2,
            robustness_score=0.75,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.pbo_score == 0.2
        assert profile.robustness_score == 0.75

    def test_monte_carlo_metrics_populated(self, orb_spec: BotSpec) -> None:
        """MonteCarloMetrics are derived from backtest comparison."""
        mock_comparison = _Comparison(
            vanilla_result=_MT5Result(
                sharpe=1.5, return_pct=12.0, drawdown=0.10,
                trades=50, trade_history=[], win_rate=60.0
            ),
            vanilla_mc_result=_MCResult(
                confidence_interval_5th=-8.0,
                confidence_interval_95th=30.0,
                value_at_risk_95=-10.0
            ),
            vanilla_full_result=_MT5Result(
                sharpe=1.3, return_pct=11.0, drawdown=0.11,
                trades=45, trade_history=[], win_rate=62.0
            ),
            pbo=0.1,
            robustness_score=0.8,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.run_all_variants.return_value = mock_comparison

        orch = EvaluationOrchestrator(pipeline_class=lambda: mock_pipeline)
        profile, warnings = orch.evaluate(orb_spec)

        assert profile is not None
        assert profile.monte_carlo.n_simulations == 1000
        assert profile.monte_carlo.percentile_5_return == -8.0
        assert profile.monte_carlo.percentile_95_return == 30.0
        assert profile.monte_carlo.max_drawdown_95 == -10.0
        # sharpe_confidence_width = abs(30.0 - (-8.0)) = 38.0
        assert profile.monte_carlo.sharpe_confidence_width == 38.0


# ---------------------------------------------------------------------------
# TestEvaluationBridgeIntegration
# ---------------------------------------------------------------------------


class TestEvaluationBridgeIntegration:
    """Integration tests for EvaluationBridge with real data."""

    def test_from_backtest_result_computes_kelly_score(self) -> None:
        """Kelly score is computed correctly from win_rate and profit_factor."""
        bridge = EvaluationBridge()
        result = bridge.from_backtest_result(
            bot_id="test-bot-1",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
        )

        assert result.bot_id == "test-bot-1"
        assert result.sharpe_ratio == 1.5
        assert result.max_drawdown == 0.10
        assert result.win_rate == 0.60
        assert result.profit_factor == 2.0
        assert result.expectancy == 15.0
        assert result.total_trades == 50
        # Kelly score: win_rate - (1 - win_rate) / profit_factor
        # = 0.6 - 0.4 / 2.0 = 0.6 - 0.2 = 0.4, clamped to [0, 1]
        expected_kelly = min(1.0, max(0.0, 0.60 - (0.40 / 2.0)))
        assert result.kelly_score == pytest.approx(expected_kelly, rel=1e-3)

    def test_to_evaluation_profile_derives_robustness(self) -> None:
        """Robustness score is derived from walk-forward + Monte Carlo metrics."""
        bridge = EvaluationBridge()
        backtest_result = EvaluationResult(
            bot_id="test-bot-1",
            mode="BACKTEST",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
        )

        profile = bridge.to_evaluation_profile(
            bot_id="test-bot-1",
            backtest_result=backtest_result,
            monte_carlo_p5=-5.0,
            monte_carlo_p95=25.0,
            monte_carlo_max_dd95=-8.0,
            walk_forward_avg_sharpe=1.2,
            walk_forward_avg_return=10.0,
            walk_forward_stability=0.8,
        )

        assert profile.bot_id == "test-bot-1"
        assert profile.backtest.sharpe_ratio == 1.5
        assert profile.monte_carlo.percentile_5_return == -5.0
        assert profile.monte_carlo.percentile_95_return == 25.0
        assert profile.walk_forward.avg_sharpe == 1.2
        assert profile.walk_forward.avg_return == 10.0
        assert profile.walk_forward.stability == 0.8
        # PBO: max(0.0, 1.0 - walk_forward_stability)
        assert profile.pbo_score == pytest.approx(0.2, rel=1e-3)
        # Robustness: weighted formula
        mc_width = abs(25.0 - (-5.0))  # = 30.0
        mc_quality = max(0.0, 1.0 - mc_width / 100.0)  # = 0.7
        expected_robustness = (0.8 * 0.5) + (mc_quality * 0.3) + (
            min(1.0, 1.5 / 2.0) * 0.2
        )  # = 0.4 + 0.21 + 0.15 = 0.76
        assert profile.robustness_score == pytest.approx(expected_robustness, rel=1e-3)

    def test_passes_gate_true_for_good_strategy(self) -> None:
        """passes_gate is True when sharpe >= 1.0 and drawdown <= 0.15."""
        bridge = EvaluationBridge()
        result = bridge.from_backtest_result(
            bot_id="test-bot-good",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
        )

        assert result.passes_gate is True
        assert result.kelly_score > 0.0

    def test_passes_gate_false_for_poor_strategy(self) -> None:
        """passes_gate is False when sharpe < 1.0 or drawdown > 0.15."""
        # Case 1: sharpe < 1.0
        bridge = EvaluationBridge()
        result_low_sharpe = bridge.from_backtest_result(
            bot_id="test-bot-low-sharpe",
            sharpe_ratio=0.5,
            max_drawdown=0.10,
            win_rate=0.40,
            profit_factor=1.5,
            expectancy=5.0,
            total_trades=20,
        )
        assert result_low_sharpe.passes_gate is False

        # Case 2: drawdown > 0.15
        result_high_dd = bridge.from_backtest_result(
            bot_id="test-bot-high-dd",
            sharpe_ratio=1.5,
            max_drawdown=0.25,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
        )
        assert result_high_dd.passes_gate is False

    def test_spread_sensitivity_computed(self) -> None:
        """spread_sensitivity returns 0-1 based on performance degradation."""
        bridge = EvaluationBridge()

        result_narrow = EvaluationResult(
            bot_id="test-bot-spread",
            mode="BACKTEST",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
        )

        result_wide = EvaluationResult(
            bot_id="test-bot-spread",
            mode="BACKTEST",
            sharpe_ratio=0.75,
            max_drawdown=0.12,
            win_rate=0.55,
            profit_factor=1.8,
            expectancy=8.0,
            total_trades=45,
            return_pct=8.0,
            kelly_score=0.3,
            passes_gate=True,
        )

        sensitivity = bridge.compute_spread_sensitivity(result_narrow, result_wide)

        # degradation = (1.5 - 0.75) / 1.5 = 0.5
        expected = max(0.0, min(1.0, (1.5 - 0.75) / 1.5))
        assert sensitivity == pytest.approx(expected, rel=1e-3)

    def test_spread_sensitivity_zero_when_narrow_degrades_to_zero(self) -> None:
        """spread_sensitivity is 1.0 when wide result has zero or negative sharpe."""
        bridge = EvaluationBridge()

        result_narrow = EvaluationResult(
            bot_id="test-bot-spread",
            mode="BACKTEST",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=15.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
        )

        result_wide = EvaluationResult(
            bot_id="test-bot-spread",
            mode="BACKTEST",
            sharpe_ratio=0.0,
            max_drawdown=0.30,
            win_rate=0.40,
            profit_factor=1.0,
            expectancy=-2.0,
            total_trades=30,
            return_pct=-5.0,
            kelly_score=0.0,
            passes_gate=False,
        )

        sensitivity = bridge.compute_spread_sensitivity(result_narrow, result_wide)
        assert sensitivity == 1.0
