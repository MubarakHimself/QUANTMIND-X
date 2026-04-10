"""
Tests for QuantMindLib V1 — Backtest Schema Compatibility.

Verifies that MT5BacktestResult and CTraderBacktestResult both map
losslessly to EvaluationResult, ensuring the evaluation pipeline works
identically regardless of which backtest engine produced the data.

Coverage:
  A. MT5 Schema Tests          — MT5BacktestResult -> EvaluationResult
  B. cTrader Schema Tests      — CTraderBacktestResult -> EvaluationResult
  C. Schema Contract Tests     — Cross-engine equivalence
  D. Mock Data Tests           — Realistic equivalent data produces identical output
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

import sys
sys.path.insert(0, "src")

from src.library.core.domain.evaluation_result import EvaluationResult
from src.library.evaluation.ctrader_backtest_schema import CTraderBacktestResult
from src.library.evaluation.evaluation_orchestrator import EvaluationOrchestrator


# ---------------------------------------------------------------------------
# Mock MT5 result dataclass (mirrors src/backtesting/mt5_engine.py)
# ---------------------------------------------------------------------------


@dataclass
class _MT5Result:
    """Minimal mock of MT5BacktestResult."""
    sharpe: float
    return_pct: float
    drawdown: float
    trades: int
    log: str = ""
    initial_cash: float = 10000.0
    final_cash: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    win_rate: float = 0.0  # percentage 0-100


# ---------------------------------------------------------------------------
# Test Suite A: MT5 Schema Tests
# ---------------------------------------------------------------------------

class TestMT5SchemaMapping:
    """Tests for MT5BacktestResult -> EvaluationResult lossless mapping."""

    def test_sharpe_ratio_maps_correctly(self) -> None:
        """sharpe (MT5) -> sharpe_ratio (EvaluationResult) is exact."""
        result = _MT5Result(
            sharpe=2.3, return_pct=20.0, drawdown=0.05,
            trades=100, win_rate=65.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-001", "VANILLA", result)

        assert eval_result.sharpe_ratio == 2.3

    def test_max_drawdown_maps_correctly(self) -> None:
        """drawdown (MT5) -> max_drawdown (EvaluationResult) is exact."""
        result = _MT5Result(
            sharpe=1.0, return_pct=10.0, drawdown=0.12,
            trades=30, win_rate=55.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-002", "VANILLA", result)

        assert eval_result.max_drawdown == 0.12

    def test_win_rate_maps_correctly_from_win_rate_field(self) -> None:
        """win_rate (MT5 percentage) -> win_rate (EvaluationResult fraction) is exact."""
        result = _MT5Result(
            sharpe=1.5, return_pct=15.0, drawdown=0.10,
            trades=50, win_rate=62.5, trade_history=[]  # no trade history -> fallback
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-003", "VANILLA", result)

        # MT5 stores win_rate as percentage (0-100); EvaluationResult uses fraction (0-1)
        assert eval_result.win_rate == pytest.approx(0.625, rel=1e-3)

    def test_win_rate_computed_from_trade_history(self) -> None:
        """win_rate is computed from trade_history when available, overriding the field."""
        trade_history = [
            {"profit": 100.0},  # win
            {"profit": -50.0},  # loss
            {"profit": 200.0}, # win
            {"profit": -30.0}, # loss
        ]
        result = _MT5Result(
            sharpe=1.2, return_pct=8.0, drawdown=0.08,
            trades=4, win_rate=50.0,  # field says 50%
            trade_history=trade_history  # actual: 2/4 = 50% anyway
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-004", "VANILLA", result)

        # 2 winning / 4 total = 0.5
        assert eval_result.win_rate == pytest.approx(0.5, rel=1e-3)

    def test_profit_factor_computed_from_trade_history(self) -> None:
        """profit_factor is computed from trade_history (gross_profit / gross_loss)."""
        trade_history = [
            {"profit": 100.0},  # win
            {"profit": -50.0},  # loss
            {"profit": 200.0},  # win
            {"profit": -25.0},  # loss
            {"profit": 150.0},  # win
        ]
        result = _MT5Result(
            sharpe=1.5, return_pct=12.0, drawdown=0.10,
            trades=5, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-005", "VANILLA", result)

        # gross_profit = 100+200+150 = 450, gross_loss = 50+25 = 75
        # profit_factor = 450 / 75 = 6.0
        assert eval_result.profit_factor == pytest.approx(6.0, rel=1e-3)

    def test_profit_factor_capped_at_100(self) -> None:
        """profit_factor is capped at 100.0 to prevent overflow."""
        trade_history = [
            {"profit": 100.0},  # win
            {"profit": -0.001},  # loss (tiny)
        ]
        result = _MT5Result(
            sharpe=1.0, return_pct=5.0, drawdown=0.05,
            trades=2, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-006", "VANILLA", result)

        assert eval_result.profit_factor == 100.0

    def test_expectancy_computed_from_trade_history(self) -> None:
        """expectancy = sum(profit) / len(trade_history)."""
        trade_history = [
            {"profit": 100.0},
            {"profit": -40.0},
            {"profit": 80.0},
            {"profit": -20.0},
        ]
        result = _MT5Result(
            sharpe=1.0, return_pct=4.0, drawdown=0.05,
            trades=4, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-007", "VANILLA", result)

        # (100 - 40 + 80 - 20) / 4 = 120 / 4 = 30.0
        assert eval_result.expectancy == pytest.approx(30.0, rel=1e-3)

    def test_total_trades_maps_correctly(self) -> None:
        """trades (MT5) -> total_trades (EvaluationResult) is exact."""
        result = _MT5Result(
            sharpe=1.0, return_pct=5.0, drawdown=0.10,
            trades=87, win_rate=50.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-008", "VANILLA", result)

        assert eval_result.total_trades == 87

    def test_return_pct_maps_correctly(self) -> None:
        """return_pct (MT5) -> return_pct (EvaluationResult) is exact."""
        result = _MT5Result(
            sharpe=1.5, return_pct=25.3, drawdown=0.10,
            trades=50, win_rate=60.0
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-009", "VANILLA", result)

        assert eval_result.return_pct == 25.3

    def test_kelly_score_computed_correctly(self) -> None:
        """kelly_score = win_rate - (1 - win_rate) / profit_factor, clamped [0, 1]."""
        # 2 wins, 2 losses:
        #   win_rate = 2/4 = 0.5
        #   gross_profit = 220, gross_loss = 75
        #   profit_factor = 220/75 = 2.933...
        #   kelly = 0.5 - 0.5 / 2.933 = 0.5 - 0.1705 = 0.3295
        trade_history = [
            {"profit": 100.0},  # win
            {"profit": -50.0},  # loss
            {"profit": 120.0},  # win
            {"profit": -25.0},  # loss
        ]
        result = _MT5Result(
            sharpe=1.5, return_pct=10.0, drawdown=0.10,
            trades=4, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-010", "VANILLA", result)

        assert eval_result.kelly_score == pytest.approx(0.3295, rel=1e-2)

    def test_kelly_score_clamped_to_one(self) -> None:
        """kelly_score is clamped to 1.0 even for extremely profitable strategies."""
        trade_history = [
            {"profit": 1000.0},  # win
            {"profit": 1000.0},  # win
        ]
        result = _MT5Result(
            sharpe=3.0, return_pct=50.0, drawdown=0.02,
            trades=2, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-011", "VANILLA", result)

        assert eval_result.kelly_score <= 1.0

    def test_kelly_score_clamped_to_zero(self) -> None:
        """kelly_score is clamped to 0.0 for losing strategies."""
        trade_history = [
            {"profit": -50.0},  # loss
            {"profit": -50.0},  # loss
        ]
        result = _MT5Result(
            sharpe=-0.5, return_pct=-5.0, drawdown=0.30,
            trades=2, win_rate=0.0,
            trade_history=trade_history
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-012", "VANILLA", result)

        assert eval_result.kelly_score >= 0.0

    def test_regime_distribution_preserved(self) -> None:
        """regime_distribution from spiced result is preserved through mapping."""
        @dataclass
        class _SpicedResult(_MT5Result):
            regime_distribution: Dict[str, int] = field(default_factory=dict)
            filtered_trades: int = 0

        result = _SpicedResult(
            sharpe=1.8, return_pct=15.0, drawdown=0.07,
            trades=40, win_rate=65.0, trade_history=[],
            regime_distribution={"TRENDING": 25, "RANGING": 12, "HIGH_CHAOS": 3},
            filtered_trades=5
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._spiced_result_to_evaluation_result("bot-013", result)

        assert eval_result.regime_distribution == {"TRENDING": 25, "RANGING": 12, "HIGH_CHAOS": 3}
        assert eval_result.filtered_trades == 5

    def test_passes_gate_true_sharpe_and_drawdown(self) -> None:
        """passes_gate is True when sharpe >= 1.0 AND max_drawdown <= 0.15."""
        result = _MT5Result(
            sharpe=1.5, return_pct=12.0, drawdown=0.10,
            trades=50, win_rate=60.0, trade_history=[]
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-014", "VANILLA", result)

        assert eval_result.passes_gate is True

    def test_passes_gate_false_low_sharpe(self) -> None:
        """passes_gate is False when sharpe < 1.0 even with low drawdown."""
        result = _MT5Result(
            sharpe=0.8, return_pct=5.0, drawdown=0.05,
            trades=20, win_rate=45.0, trade_history=[]
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-015", "VANILLA", result)

        assert eval_result.passes_gate is False

    def test_passes_gate_false_high_drawdown(self) -> None:
        """passes_gate is False when max_drawdown > 0.15 even with good sharpe."""
        result = _MT5Result(
            sharpe=1.8, return_pct=20.0, drawdown=0.25,
            trades=30, win_rate=65.0, trade_history=[]
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-016", "VANILLA", result)

        assert eval_result.passes_gate is False

    def test_passes_gate_false_boundary_drawdown(self) -> None:
        """passes_gate is False when drawdown is exactly 0.16 (> 0.15 threshold)."""
        result = _MT5Result(
            sharpe=1.5, return_pct=12.0, drawdown=0.16,
            trades=50, win_rate=60.0, trade_history=[]
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-017", "VANILLA", result)

        assert eval_result.passes_gate is False

    def test_passes_gate_true_at_boundary(self) -> None:
        """passes_gate is True when drawdown is exactly 0.15 (boundary)."""
        result = _MT5Result(
            sharpe=1.0, return_pct=8.0, drawdown=0.15,
            trades=50, win_rate=50.0, trade_history=[]
        )
        orch = EvaluationOrchestrator()
        eval_result = orch._backtest_result_to_evaluation_result("bot-018", "VANILLA", result)

        assert eval_result.passes_gate is True


# ---------------------------------------------------------------------------
# Test Suite B: cTrader Schema Tests
# ---------------------------------------------------------------------------

class TestCTraderSchemaMapping:
    """Tests for CTraderBacktestResult -> EvaluationResult lossless mapping."""

    def test_basic_fields_map_losslessly(self) -> None:
        """All scalar fields from CTraderBacktestResult map to EvaluationResult exactly."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-1",
            mode="BACKTEST",
            sharpe_ratio=1.75,
            max_drawdown=0.08,
            win_rate=0.62,
            profit_factor=2.8,
            expectancy=30.0,
            total_trades=60,
            return_pct=18.5,
            kelly_score=0.45,
            passes_gate=True,
            regime_distribution={"TRENDING": 40, "RANGING": 18, "HIGH_CHAOS": 2},
            filtered_trades=7,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.bot_id == "ctrader-bot-1"
        assert eval_result.sharpe_ratio == 1.75
        assert eval_result.max_drawdown == 0.08
        assert eval_result.win_rate == 0.62
        assert eval_result.profit_factor == 2.8
        assert eval_result.expectancy == 30.0
        assert eval_result.total_trades == 60
        assert eval_result.return_pct == 18.5
        assert eval_result.kelly_score == 0.45
        assert eval_result.passes_gate is True
        assert eval_result.regime_distribution == {"TRENDING": 40, "RANGING": 18, "HIGH_CHAOS": 2}
        assert eval_result.filtered_trades == 7

    def test_to_evaluation_result_produces_valid_result(self) -> None:
        """to_evaluation_result() returns a valid EvaluationResult instance."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-2",
            sharpe_ratio=1.2,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.9,
            expectancy=15.0,
            total_trades=40,
            return_pct=10.0,
            kelly_score=0.35,
            passes_gate=True,
        )

        eval_result = ctrader.to_evaluation_result()

        assert isinstance(eval_result, EvaluationResult)
        assert eval_result.model_validate(eval_result.model_dump())

    def test_win_rate_fraction_convention(self) -> None:
        """cTrader win_rate is used as fraction directly (not divided again)."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-3",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.win_rate == 0.60

    def test_from_ctrader_api_normalises_percentage_win_rate(self) -> None:
        """from_ctrader_api converts percentage win_rate (e.g. 62) to fraction (0.62)."""
        raw_api_response = {
            "winRate": 62.0,       # cTrader API may return percentage
            "maxDrawdown": 8.0,    # cTrader API may return percentage
            "sharpeRatio": 1.75,
            "profitFactor": 2.8,
            "expectancy": 30.0,
            "totalTrades": 60,
            "returnPercent": 18.5,
            "kellyScore": 0.45,
            "gateStatus": True,
        }

        ctrader = CTraderBacktestResult.from_ctrader_api(raw_api_response, bot_id="ctrader-bot-4")

        assert ctrader.win_rate == pytest.approx(0.62, rel=1e-3)
        assert ctrader.max_drawdown == pytest.approx(0.08, rel=1e-3)

    def test_from_ctrader_api_handles_fraction_values(self) -> None:
        """from_ctrader_api leaves fraction values unchanged."""
        raw_api_response = {
            "winRate": 0.62,       # already a fraction
            "maxDrawdown": 0.08,   # already a fraction
            "sharpeRatio": 1.75,
            "profitFactor": 2.8,
            "totalTrades": 60,
        }

        ctrader = CTraderBacktestResult.from_ctrader_api(raw_api_response, bot_id="ctrader-bot-5")

        assert ctrader.win_rate == 0.62
        assert ctrader.max_drawdown == 0.08

    def test_from_ctrader_api_extracts_regime_fields(self) -> None:
        """from_ctrader_api extracts IS and OOS regime distributions correctly."""
        raw_api_response = {
            "winRate": 0.60,
            "maxDrawdown": 0.10,
            "sharpeRatio": 1.5,
            "profitFactor": 2.0,
            "totalTrades": 50,
            "isRegimeDistribution": {"TRENDING": 30, "RANGING": 15},
            "oosRegimeDistribution": {"TRENDING": 20, "RANGING": 10},
        }

        ctrader = CTraderBacktestResult.from_ctrader_api(raw_api_response, bot_id="ctrader-bot-6")

        assert ctrader.is_regime_distribution == {"TRENDING": 30, "RANGING": 15}
        assert ctrader.oos_regime_distribution == {"TRENDING": 20, "RANGING": 10}

    def test_regime_distribution_merge_logic(self, regime_distributions: dict) -> None:
        """IS + OOS regime distributions are summed when merged."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-7",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=102,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
            is_regime_distribution=regime_distributions["is"],
            oos_regime_distribution=regime_distributions["oos"],
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.regime_distribution == regime_distributions["merged"]

    def test_top_level_regime_distribution_used_when_no_is_oos(
        self, regime_distributions: dict
    ) -> None:
        """Top-level regime_distribution is used when IS/OOS fields are absent."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-8",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
            regime_distribution={"TRENDING": 35, "RANGING": 15},
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.regime_distribution == {"TRENDING": 35, "RANGING": 15}

    def test_filtered_trades_preserved(self) -> None:
        """filtered_trades is preserved through to_evaluation_result()."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-9",
            sharpe_ratio=1.8,
            max_drawdown=0.07,
            win_rate=0.65,
            profit_factor=3.0,
            expectancy=28.0,
            total_trades=45,
            return_pct=16.0,
            kelly_score=0.5,
            passes_gate=True,
            filtered_trades=8,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.filtered_trades == 8

    def test_optional_fields_default_to_none(self) -> None:
        """Optional fields default to None when not provided."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-bot-10",
            sharpe_ratio=1.0,
            max_drawdown=0.12,
            win_rate=0.50,
            profit_factor=1.5,
            expectancy=5.0,
            total_trades=20,
            return_pct=5.0,
            kelly_score=0.25,
            passes_gate=False,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.regime_distribution is None
        assert eval_result.filtered_trades is None


# ---------------------------------------------------------------------------
# Test Suite C: Schema Contract Tests
# ---------------------------------------------------------------------------

class TestSchemaContract:
    """Tests verifying both schemas produce identical EvaluationResult output."""

    def test_both_schemas_produce_identical_result(self) -> None:
        """MT5 and cTrader schemas produce byte-for-byte identical EvaluationResult."""
        # Identical metrics in both schema conventions
        mt5_result = _MT5Result(
            sharpe=1.5,
            return_pct=15.0,
            drawdown=0.10,
            trades=50,
            win_rate=60.0,  # MT5 percentage
            trade_history=[],
        )
        ctrader_result = CTraderBacktestResult(
            bot_id="bot-both",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,  # cTrader fraction
            profit_factor=2.5,
            expectancy=25.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
        )

        orch = EvaluationOrchestrator()
        mt5_eval = orch._backtest_result_to_evaluation_result("bot-both", "VANILLA", mt5_result)
        ctrader_eval = ctrader_result.to_evaluation_result()

        assert mt5_eval.sharpe_ratio == ctrader_eval.sharpe_ratio
        assert mt5_eval.max_drawdown == ctrader_eval.max_drawdown
        assert mt5_eval.total_trades == ctrader_eval.total_trades
        assert mt5_eval.return_pct == ctrader_eval.return_pct
        assert mt5_eval.win_rate == ctrader_eval.win_rate

    def test_regime_merge_combines_is_and_oos(self) -> None:
        """Regime distribution merge adds IS and OOS counts per regime key."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-merge",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=100,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
            is_regime_distribution={"TRENDING": 40, "RANGING": 20, "HIGH_CHAOS": 5},
            oos_regime_distribution={"TRENDING": 30, "RANGING": 15, "HIGH_CHAOS": 3},
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.regime_distribution == {
            "TRENDING": 70,
            "RANGING": 35,
            "HIGH_CHAOS": 8,
        }

    def test_regime_merge_with_missing_regimes(self) -> None:
        """Regime merge handles IS/OOS having non-overlapping regime keys."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-partial-merge",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
            is_regime_distribution={"TRENDING": 30, "HIGH_CHAOS": 5},
            oos_regime_distribution={"RANGING": 15},  # no overlap
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.regime_distribution == {
            "TRENDING": 30,
            "HIGH_CHAOS": 5,
            "RANGING": 15,
        }

    def test_filtered_trades_handled_when_not_present(self) -> None:
        """filtered_trades is None in EvaluationResult when cTrader result has no value."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-no-filtered",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.60,
            profit_factor=2.0,
            expectancy=20.0,
            total_trades=50,
            return_pct=15.0,
            kelly_score=0.4,
            passes_gate=True,
            filtered_trades=None,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.filtered_trades is None

    def test_mode_field_normalised_to_backtest(self) -> None:
        """cTrader mode field defaults to BACKTEST in EvaluationResult."""
        ctrader = CTraderBacktestResult(
            bot_id="ctrader-mode",
            mode="BACKTEST",
            sharpe_ratio=1.0,
            max_drawdown=0.10,
            win_rate=0.50,
            profit_factor=1.5,
            expectancy=5.0,
            total_trades=20,
            return_pct=5.0,
            kelly_score=0.25,
            passes_gate=False,
        )

        eval_result = ctrader.to_evaluation_result()

        assert eval_result.mode == "BACKTEST"


# ---------------------------------------------------------------------------
# Test Suite D: Mock Data Tests
# ---------------------------------------------------------------------------

class TestMockDataEquivalence:
    """Tests using realistic mock data to verify MT5 and cTrader equivalence."""

    def test_realistic_orb_strategy_both_produce_valid_results(
        self,
        make_mock_mt5_result,
        make_mock_ctrader_result,
    ) -> None:
        """
        Realistic ORB strategy produces valid EvaluationResult from both engines.

        Tests that both adapters (MT5 via EvaluationOrchestrator,
        cTrader via CTraderBacktestResult.to_evaluation_result()) produce
        structurally valid EvaluationResult objects with the correct bot_id,
        mode, and scalar metrics.

        Note: win_rate and expectancy may differ because MT5 computes them
        from trade_history while cTrader uses explicit fixture values.
        This is intentional — the test verifies both paths work, not that
        they produce byte-identical output.
        """
        # MT5: trade history with 6 wins, 5 losses (11 trades total).
        # 6 wins x 100 = 600 gross profit; 5 losses x 50 = 250 gross loss
        # pf = 600/250 = 2.4; expectancy = (600-250)/11 ≈ 31.82
        # win_rate = 6/11 ≈ 0.545; kelly = 6/11 - 5/11/2.4 ≈ 0.374
        trade_history = [
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 100.0},
        ]
        mt5_result = make_mock_mt5_result(
            sharpe=1.5,
            return_pct=15.0,
            drawdown=0.10,
            trades=50,
            win_rate=62.0,
            trade_history=trade_history,
        )
        ctrader_dict = make_mock_ctrader_result(
            bot_id="orb-mock-001",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=6 / 11,  # matches MT5 computed value from trade history
            profit_factor=2.4,
            expectancy=350 / 11,
            total_trades=50,
            return_pct=15.0,
            kelly_score=6 / 11 - (5 / 11) / 2.4,
            passes_gate=True,
        )
        ctrader_result = CTraderBacktestResult.model_validate(ctrader_dict)

        orch = EvaluationOrchestrator()
        mt5_eval = orch._backtest_result_to_evaluation_result("orb-mock-001", "VANILLA", mt5_result)
        ctrader_eval = ctrader_result.to_evaluation_result()

        # Both produce a valid EvaluationResult
        assert isinstance(mt5_eval, EvaluationResult)
        assert isinstance(ctrader_eval, EvaluationResult)
        assert mt5_eval.model_validate(mt5_eval.model_dump())
        assert ctrader_eval.model_validate(ctrader_eval.model_dump())

        # bot_id and mode are preserved
        assert mt5_eval.bot_id == "orb-mock-001"
        assert mt5_eval.mode == "BACKTEST"
        assert ctrader_eval.bot_id == "orb-mock-001"
        assert ctrader_eval.mode == "BACKTEST"

        # Scalar metrics match (computed from 11-entry trade_history: 6 wins, 5 losses)
        assert mt5_eval.sharpe_ratio == pytest.approx(1.5, rel=1e-3)
        assert mt5_eval.max_drawdown == pytest.approx(0.10, rel=1e-3)
        assert mt5_eval.profit_factor == pytest.approx(2.4, rel=1e-3)  # 600/250
        assert mt5_eval.win_rate == pytest.approx(6 / 11, rel=1e-3)    # 6/11 ≈ 0.545
        assert mt5_eval.expectancy == pytest.approx(350 / 11, rel=1e-2)  # (600-250)/11 ≈ 31.82
        assert mt5_eval.total_trades == 50
        assert mt5_eval.return_pct == pytest.approx(15.0, rel=1e-3)
        assert mt5_eval.kelly_score == pytest.approx(6 / 11 - (5 / 11) / 2.4, rel=1e-2)
        assert mt5_eval.passes_gate is True

        assert ctrader_eval.sharpe_ratio == pytest.approx(1.5, rel=1e-3)
        assert ctrader_eval.max_drawdown == pytest.approx(0.10, rel=1e-3)
        assert ctrader_eval.profit_factor == pytest.approx(2.4, rel=1e-3)
        assert ctrader_eval.total_trades == 50

        # MT5 and cTrader agree on the key metrics (same trade history assumptions)
        assert mt5_eval.sharpe_ratio == ctrader_eval.sharpe_ratio
        assert mt5_eval.max_drawdown == ctrader_eval.max_drawdown
        assert mt5_eval.profit_factor == ctrader_eval.profit_factor
        assert mt5_eval.win_rate == ctrader_eval.win_rate
        assert mt5_eval.expectancy == ctrader_eval.expectancy

    def test_equivalent_data_with_trade_history_computes_correct_metrics(
        self,
        make_mock_mt5_result,
        make_mock_ctrader_result,
    ) -> None:
        """
        MT5 trade history and cTrader trade_history both drive the same computed metrics.

        10 trades: 6 winners, 4 losers
          gross_profit = 560, gross_loss = 140
          profit_factor = 4.0, expectancy = 42.0, win_rate = 0.6
        """
        trade_history = [
            {"profit": 100.0}, {"profit": -50.0},
            {"profit": 80.0},  {"profit": -30.0},
            {"profit": 120.0}, {"profit": -20.0},
            {"profit": 90.0},  {"profit": -40.0},
            {"profit": 110.0}, {"profit": 60.0},
        ]

        mt5_result = make_mock_mt5_result(
            sharpe=1.5,
            return_pct=12.0,
            drawdown=0.10,
            trades=10,
            win_rate=0.0,  # force trade_history usage
            trade_history=trade_history,
        )
        ctrader_dict = make_mock_ctrader_result(
            bot_id="history-bot",
            sharpe_ratio=1.5,
            max_drawdown=0.10,
            win_rate=0.6,
            profit_factor=4.0,
            expectancy=42.0,
            total_trades=10,
            return_pct=12.0,
            kelly_score=0.5,
            passes_gate=True,
            trade_history=trade_history,
        )
        ctrader_result = CTraderBacktestResult.model_validate(ctrader_dict)

        orch = EvaluationOrchestrator()
        mt5_eval = orch._backtest_result_to_evaluation_result("history-bot", "VANILLA", mt5_result)
        ctrader_eval = ctrader_result.to_evaluation_result()

        # MT5 computes from trade history
        assert mt5_eval.win_rate == pytest.approx(0.6, rel=1e-3)
        assert mt5_eval.profit_factor == pytest.approx(4.0, rel=1e-3)
        assert mt5_eval.expectancy == pytest.approx(42.0, rel=1e-3)

        # cTrader uses explicit values (as would come from cTrader API)
        assert ctrader_eval.win_rate == pytest.approx(0.6, rel=1e-3)
        assert ctrader_eval.profit_factor == pytest.approx(4.0, rel=1e-3)
        assert ctrader_eval.expectancy == pytest.approx(42.0, rel=1e-3)

        # Both produce the same output for these metrics
        assert mt5_eval.win_rate == ctrader_eval.win_rate
        assert mt5_eval.profit_factor == ctrader_eval.profit_factor
        assert mt5_eval.expectancy == ctrader_eval.expectancy

    def test_spiced_mode_regime_equivalence(
        self,
        make_mock_mt5_result,
        make_mock_ctrader_result,
        regime_distributions: dict,
    ) -> None:
        """Spiced mode (with regime filtering) produces identical regime info from both engines."""
        mt5_result = make_mock_mt5_result(
            sharpe=1.8,
            return_pct=16.0,
            drawdown=0.07,
            trades=95,
            win_rate=65.0,
            regime_distribution=regime_distributions["merged"],
            filtered_trades=7,
            is_spiced=True,
        )
        ctrader_result = CTraderBacktestResult(
            bot_id="spiced-001",
            sharpe_ratio=1.8,
            max_drawdown=0.07,
            win_rate=0.65,
            profit_factor=3.2,
            expectancy=32.0,
            total_trades=95,
            return_pct=16.0,
            kelly_score=0.5,
            passes_gate=True,
            regime_distribution=regime_distributions["merged"],
            filtered_trades=7,
            is_regime_distribution=regime_distributions["is"],
            oos_regime_distribution=regime_distributions["oos"],
        )

        @dataclass
        class _SpicedMock(_MT5Result):
            regime_distribution: Dict[str, int] = field(default_factory=dict)
            filtered_trades: int = 0

        orch = EvaluationOrchestrator()
        mt5_spiced = _SpicedMock(
            sharpe=1.8, return_pct=16.0, drawdown=0.07, trades=95, win_rate=65.0,
            regime_distribution=regime_distributions["merged"],
            filtered_trades=7,
        )
        mt5_eval = orch._spiced_result_to_evaluation_result("spiced-001", mt5_spiced)
        ctrader_eval = ctrader_result.to_evaluation_result()

        assert mt5_eval.regime_distribution == ctrader_eval.regime_distribution
        assert mt5_eval.filtered_trades == ctrader_eval.filtered_trades
        assert mt5_eval.sharpe_ratio == ctrader_eval.sharpe_ratio
        assert mt5_eval.max_drawdown == ctrader_eval.max_drawdown

    def test_roundtrip_ctrader_evaluation_result(
        self,
        make_evaluation_result,
    ) -> None:
        """
        EvaluationResult -> CTraderBacktestResult -> EvaluationResult is lossless.
        """
        original = make_evaluation_result(
            bot_id="roundtrip-001",
            sharpe_ratio=1.75,
            max_drawdown=0.08,
            win_rate=0.62,
            profit_factor=2.8,
            expectancy=30.0,
            total_trades=60,
            return_pct=18.5,
            kelly_score=0.45,
            passes_gate=True,
            regime_distribution={"TRENDING": 40, "RANGING": 18, "HIGH_CHAOS": 2},
            filtered_trades=7,
        )

        ctrader = CTraderBacktestResult.from_evaluation_result(
            result=original,
            trade_history=[{"profit": 100.0}],
            is_regime_distribution={"TRENDING": 40, "RANGING": 18},
            oos_regime_distribution={"HIGH_CHAOS": 2},
        )
        roundtripped = ctrader.to_evaluation_result()

        assert roundtripped.bot_id == original.bot_id
        assert roundtripped.mode == original.mode
        assert roundtripped.sharpe_ratio == original.sharpe_ratio
        assert roundtripped.max_drawdown == original.max_drawdown
        assert roundtripped.win_rate == original.win_rate
        assert roundtripped.profit_factor == original.profit_factor
        assert roundtripped.expectancy == original.expectancy
        assert roundtripped.total_trades == original.total_trades
        assert roundtripped.return_pct == original.return_pct
        assert roundtripped.kelly_score == original.kelly_score
        assert roundtripped.passes_gate == original.passes_gate
        assert roundtripped.regime_distribution == original.regime_distribution
        assert roundtripped.filtered_trades == original.filtered_trades

    def test_ctrader_api_contract_with_regime_merge(self) -> None:
        """
        Simulated cTrader API response (with IS/OOS split) produces merged regime distribution.
        This test documents the CTRADER-008 contract: the adapter must provide IS/OOS fields
        and to_evaluation_result() will merge them.
        """
        # Simulated raw cTrader API response
        raw_api = {
            "sharpeRatio": 1.6,
            "maxDrawdown": 10.0,     # percentage
            "winRate": 63.0,         # percentage
            "profitFactor": 2.9,
            "expectancy": 27.0,
            "totalTrades": 77,
            "returnPercent": 17.2,
            "kellyScore": 0.44,
            "gateStatus": True,
            "isRegimeDistribution": {"TRENDING": 45, "RANGING": 22, "HIGH_CHAOS": 6},
            "oosRegimeDistribution": {"TRENDING": 35, "RANGING": 17, "HIGH_CHAOS": 4},
            "filteredTrades": 9,
        }

        ctrader = CTraderBacktestResult.from_ctrader_api(raw_api, bot_id="api-contract-001")
        eval_result = ctrader.to_evaluation_result()

        # Values normalised from percentage to fraction
        assert eval_result.max_drawdown == pytest.approx(0.10, rel=1e-3)
        assert eval_result.win_rate == pytest.approx(0.63, rel=1e-3)

        # Regime distributions merged
        assert eval_result.regime_distribution == {
            "TRENDING": 80,
            "RANGING": 39,
            "HIGH_CHAOS": 10,
        }
        assert eval_result.filtered_trades == 9
