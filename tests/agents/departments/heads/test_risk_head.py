"""
Tests for Risk Department Head - Backtest Evaluation

Story: 7-8-risk-trading-portfolio-department-real-implementations
Tests AC #1: Risk Department real backtest evaluation
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.departments.heads.risk_head import (
    RiskHead,
    BacktestEvaluationThresholds,
    ModeEvaluationResult,
    BacktestEvaluationResult,
)


class TestRiskHead:
    """Test suite for RiskHead."""

    @pytest.fixture
    def risk_head(self):
        """Create a RiskHead instance."""
        with patch("src.agents.departments.heads.risk_head.get_department_config"), \
             patch("src.agents.departments.heads.base.DepartmentHead._init_spawner"):
            return RiskHead()

    def test_initialization(self, risk_head):
        """Test RiskHead initializes with default thresholds."""
        assert risk_head.thresholds.min_sharpe == 1.0
        assert risk_head.thresholds.max_drawdown == 15.0
        assert risk_head.thresholds.min_win_rate == 50.0
        assert len(risk_head.BACKTEST_MODES) == 6

    def test_get_tools(self, risk_head):
        """Test tool definitions include backtest evaluation."""
        tools = risk_head.get_tools()
        tool_names = [t["name"] for t in tools]

        assert "run_backtest_evaluation" in tool_names
        assert "get_evaluation_thresholds" in tool_names
        assert "set_evaluation_thresholds" in tool_names

    def test_run_backtest_evaluation_returns_structure(self, risk_head):
        """Test evaluation returns expected structure."""
        result = risk_head.run_backtest_evaluation("TrendFollower_v2.1")

        assert "ea_name" in result
        assert "pass" in result
        assert "modes_passed" in result
        assert "modes_total" in result
        assert "mode_results" in result
        assert "thresholds" in result

    def test_modes_passed_count(self, risk_head):
        """Test modes_passed is correctly counted."""
        result = risk_head.run_backtest_evaluation("TrendFollower_v2.1")

        modes_passed = sum(1 for r in result["mode_results"] if r["passed"])
        assert result["modes_passed"] == modes_passed

    def test_pass_verdict_requires_4_of_6(self, risk_head):
        """Test pass verdict requires >= 4/6 modes to pass."""
        result = risk_head.run_backtest_evaluation("TrendFollower_v2.1")

        # According to AC #1: >=4/6 modes must pass
        assert result["modes_total"] == 6
        assert result["pass"] == (result["modes_passed"] >= 4)

    def test_get_evaluation_thresholds(self, risk_head):
        """Test getting current thresholds."""
        thresholds = risk_head.get_evaluation_thresholds()

        assert thresholds["min_sharpe"] == 1.0
        assert thresholds["max_drawdown"] == 15.0
        assert thresholds["min_win_rate"] == 50.0

    def test_set_evaluation_thresholds(self, risk_head):
        """Test setting custom thresholds."""
        result = risk_head.set_evaluation_thresholds(
            min_sharpe=1.5,
            max_drawdown=10.0,
            min_win_rate=55.0,
        )

        assert result["min_sharpe"] == 1.5
        assert result["max_drawdown"] == 10.0
        assert result["min_win_rate"] == 55.0

        # Verify they were applied
        assert risk_head.thresholds.min_sharpe == 1.5
        assert risk_head.thresholds.max_drawdown == 10.0
        assert risk_head.thresholds.min_win_rate == 55.0

    def test_calculate_position_size(self, risk_head):
        """Test position size calculation."""
        result = risk_head.calculate_position_size(
            symbol="EURUSD",
            account_balance=10000.0,
            risk_percent=2.0,
        )

        assert result["symbol"] == "EURUSD"
        assert result["account_balance"] == 10000.0
        assert result["risk_percent"] == 2.0
        assert result["risk_amount"] == 200.0
        assert result["position_size"] == 100.0  # 200 / 2%

    def test_check_drawdown(self, risk_head):
        """Test drawdown check."""
        result = risk_head.check_drawdown("acc_001")

        assert "account_id" in result
        assert "current_drawdown" in result
        assert "max_allowed_drawdown" in result
        assert "status" in result

    def test_calculate_var(self, risk_head):
        """Test VaR calculation."""
        portfolio = [
            {"asset": "EURUSD", "value": 5000},
            {"asset": "GBPUSD", "value": 3000},
            {"asset": "USDJPY", "value": 2000},
        ]

        result = risk_head.calculate_var(
            portfolio=portfolio,
            confidence=0.95,
            timeframe=10,
        )

        assert "var" in result
        assert "var_percentage" in result
        assert result["confidence"] == 0.95
        assert result["timeframe_days"] == 10

    def test_mode_evaluation_includes_all_modes(self, risk_head):
        """Test all 6 backtest modes are evaluated."""
        result = risk_head.run_backtest_evaluation("TrendFollower_v2.1")

        evaluated_modes = {r["mode"] for r in result["mode_results"]}
        expected_modes = {mode.value for mode in risk_head.BACKTEST_MODES}

        assert evaluated_modes == expected_modes


class TestModeEvaluationResult:
    """Test ModeEvaluationResult dataclass."""

    def test_dataclass_creation(self):
        """Test creating a ModeEvaluationResult."""
        result = ModeEvaluationResult(
            mode="VANILLA",
            passed=True,
            sharpe=1.5,
            max_drawdown=10.0,
            win_rate=65.0,
            net_pnl=25.0,
            total_trades=50,
            reason="All thresholds met",
        )

        assert result.mode == "VANILLA"
        assert result.passed is True
        assert result.sharpe == 1.5


class TestBacktestEvaluationThresholds:
    """Test BacktestEvaluationThresholds dataclass."""

    def test_default_values(self):
        """Test default threshold values."""
        thresholds = BacktestEvaluationThresholds()

        assert thresholds.min_sharpe == 1.0
        assert thresholds.max_drawdown == 15.0
        assert thresholds.min_win_rate == 50.0

    def test_custom_values(self):
        """Test custom threshold values."""
        thresholds = BacktestEvaluationThresholds(
            min_sharpe=1.5,
            max_drawdown=10.0,
            min_win_rate=55.0,
        )

        assert thresholds.min_sharpe == 1.5
        assert thresholds.max_drawdown == 10.0
        assert thresholds.min_win_rate == 55.0
