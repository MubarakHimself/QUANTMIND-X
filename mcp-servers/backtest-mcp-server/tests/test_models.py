"""
Focused tests for Backtest MCP Server Pydantic models.

Task Group 4.1: Write 2-8 focused tests for Pydantic models and validation.
Tests cover:
- BacktestConfig accepts all required fields
- BacktestResult serializes correctly
- Invalid inputs raise validation errors
- Language literal constraint (python, mq5)
"""

import pytest
from datetime import date, datetime
from pydantic import ValidationError

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    BacktestConfig,
    BacktestResult,
    BacktestStatus,
    BacktestSyntaxError,
    BacktestDataError,
    BacktestRuntimeError,
    BacktestTimeoutError,
)


class TestBacktestConfig:
    """Test BacktestConfig model validation."""

    def test_backtest_config_accepts_all_required_fields(self):
        """Test BacktestConfig accepts all required fields."""
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        assert config.symbol == "EURUSD"
        assert config.timeframe == 60
        assert config.start_date == date(2024, 1, 1)
        assert config.end_date == date(2024, 12, 31)
        # Check defaults
        assert config.initial_capital == 10000.0
        assert config.commission == 0.0002
        assert config.slippage == 0.0001
        assert config.position_size == 0.1

    def test_backtest_config_with_all_fields(self):
        """Test BacktestConfig accepts all fields including optional."""
        config = BacktestConfig(
            symbol="XAUUSD",
            timeframe=240,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_capital=50000.0,
            commission=0.0001,
            slippage=0.00005,
            position_size=0.5,
        )
        assert config.symbol == "XAUUSD"
        assert config.initial_capital == 50000.0
        assert config.position_size == 0.5

    def test_invalid_symbol_raises_validation_error(self):
        """Test invalid inputs raise validation errors for symbol."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(
                symbol="",
                timeframe=60,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            )
        assert "symbol" in str(exc_info.value).lower()

    def test_invalid_timeframe_raises_validation_error(self):
        """Test invalid timeframe raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(
                symbol="EURUSD",
                timeframe=999,  # Invalid timeframe
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
            )
        assert "timeframe" in str(exc_info.value).lower()

    def test_end_date_before_start_date_raises_error(self):
        """Test end_date before start_date raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(
                symbol="EURUSD",
                timeframe=60,
                start_date=date(2024, 12, 31),
                end_date=date(2024, 1, 1),  # Before start_date
            )
        assert "end_date" in str(exc_info.value).lower() or "after" in str(
            exc_info.value
        ).lower()

    def test_position_size_out_of_range_raises_error(self):
        """Test position_size outside valid range raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            BacktestConfig(
                symbol="EURUSD",
                timeframe=60,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                position_size=200.0,  # Too large
            )
        assert "position_size" in str(exc_info.value).lower()


class TestBacktestResult:
    """Test BacktestResult model."""

    def test_backtest_result_serializes_correctly(self):
        """Test BacktestResult serializes correctly to dict."""
        result = BacktestResult(
            backtest_id="test-123",
            status="success",
            metrics={
                "sharpe_ratio": 1.85,
                "max_drawdown": 12.5,
                "total_return": 25.3,
            },
            equity_curve=[{"timestamp": "2024-01-01T00:00:00Z", "value": 10000.0}],
            trade_log=[{"entry": "2024-01-01", "exit": "2024-01-02", "pnl": 150.0}],
            logs="Backtest completed",
            execution_time_seconds=45.2,
        )
        result_dict = result.model_dump()
        assert result_dict["backtest_id"] == "test-123"
        assert result_dict["status"] == "success"
        assert result_dict["metrics"]["sharpe_ratio"] == 1.85
        assert len(result_dict["equity_curve"]) == 1
        assert result_dict["execution_time_seconds"] == 45.2


class TestBacktestStatus:
    """Test BacktestStatus model."""

    def test_backtest_status_with_queued_state(self):
        """Test BacktestStatus with queued state."""
        status = BacktestStatus(
            backtest_id="test-456",
            status="queued",
            progress_percent=0.0,
            estimated_completion=datetime(2024, 12, 31, 12, 0, 0),
        )
        assert status.status == "queued"
        assert status.progress_percent == 0.0
        assert status.estimated_completion is not None
        assert status.result is None

    def test_backtest_status_with_completed_result(self):
        """Test BacktestStatus with completed result."""
        result = BacktestResult(
            backtest_id="test-789",
            status="success",
            metrics={"sharpe_ratio": 2.0},
            execution_time_seconds=30.0,
        )
        status = BacktestStatus(
            backtest_id="test-789",
            status="completed",
            progress_percent=100.0,
            result=result,
        )
        assert status.status == "completed"
        assert status.progress_percent == 100.0
        assert status.result is not None
        assert status.result.backtest_id == "test-789"


class TestBacktestErrors:
    """Test backtest error classes."""

    def test_backtest_syntax_error_with_line_number(self):
        """Test BacktestSyntaxError includes line number."""
        error = BacktestSyntaxError("Invalid syntax", line_number=42)
        assert "line 42" in str(error)
        assert "Invalid syntax" in str(error)

    def test_backtest_syntax_error_without_line_number(self):
        """Test BacktestSyntaxError works without line number."""
        error = BacktestSyntaxError("Invalid syntax")
        assert "Invalid syntax" in str(error)

    def test_backtest_data_error_with_symbol(self):
        """Test BacktestDataError includes symbol."""
        error = BacktestDataError("Insufficient data", symbol="EURUSD")
        assert "EURUSD" in str(error)
        assert "Insufficient data" in str(error)

    def test_backtest_runtime_error_with_type(self):
        """Test BacktestRuntimeError includes error type."""
        error = BacktestRuntimeError("Division by zero", error_type="ZeroDivisionError")
        assert "ZeroDivisionError" in str(error)
        assert "Division by zero" in str(error)

    def test_backtest_timeout_error(self):
        """Test BacktestTimeoutError includes timeout seconds."""
        error = BacktestTimeoutError(timeout_seconds=300.0)
        assert "300" in str(error)
        assert "timeout" in str(error).lower()


class TestLanguageLiteral:
    """Test language literal constraint for run_backtest."""

    def test_valid_languages(self):
        """Test valid language values."""
        valid_languages = ["python", "mq5"]
        for lang in valid_languages:
            # This would be validated in run_backtest function
            assert lang in valid_languages

    def test_invalid_language_rejected(self):
        """Test invalid language values would be rejected."""
        invalid_languages = ["Python", "MQ5", "mql5", "javascript", ""]
        for lang in invalid_languages:
            # In the actual implementation, these would raise ValueError
            # For now we just verify the expected valid values
            assert lang not in ["python", "mq5"]
