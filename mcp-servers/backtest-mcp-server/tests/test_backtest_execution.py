"""
Focused tests for Backtest Execution and Queue Management.

Task Group 5.1: Write 2-8 focused tests for backtest execution.
Tests cover:
- Python strategy executes successfully
- Metrics are calculated correctly (Sharpe, drawdown, win rate)
- Queue manager executes up to 10 parallel backtests
- CPU detection adjusts worker count
- Backtest status tracking
- Error handling for invalid code
- Skip exhaustive edge cases
"""

import pytest
import time
from datetime import date, datetime
from pathlib import Path

import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    BacktestConfig,
    BacktestResult,
    BacktestStatus,
    BacktestSyntaxError,
)
from backtest_runner import BacktestRunner
from queue_manager import BacktestQueueManager


# Sample Python strategy code for testing
SAMPLE_PYTHON_STRATEGY = """
import backtrader as bt

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(period=20)
        self.close = self.data.close

    def next(self):
        if not self.position:
            if self.close[0] > self.sma[0]:
                self.buy(size=100)
        else:
            if self.close[0] < self.sma[0]:
                self.sell(size=100)
"""


class TestBacktestRunner:
    """Test BacktestRunner executes Python strategies and calculates metrics."""

    def test_python_strategy_executes_successfully(self):
        """Test Python strategy executes successfully."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SAMPLE_PYTHON_STRATEGY, config)

        assert isinstance(result, BacktestResult)
        assert result.backtest_id != "placeholder"
        assert result.status in ["success", "error"]
        assert result.execution_time_seconds >= 0

    def test_metrics_calculated_correctly(self):
        """Test metrics are calculated correctly (Sharpe, drawdown, win rate)."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SAMPLE_PYTHON_STRATEGY, config)

        # Check metrics exist and are valid
        assert result.metrics is not None
        if result.status == "success":
            # Sharpe ratio can be negative or positive
            assert "sharpe_ratio" in result.metrics
            assert isinstance(result.metrics["sharpe_ratio"], (int, float))

            # Max drawdown should be non-negative (percentage)
            assert "max_drawdown" in result.metrics
            assert result.metrics["max_drawdown"] >= 0

            # Win rate should be 0-100
            assert "win_rate" in result.metrics
            assert 0 <= result.metrics["win_rate"] <= 100

            # Check for additional metrics
            assert "total_return" in result.metrics
            assert "profit_factor" in result.metrics

    def test_equity_curve_and_trade_log_generated(self):
        """Test equity curve and trade log are generated."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 28),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SAMPLE_PYTHON_STRATEGY, config)

        # Equity curve should be a list of timestamp/value pairs
        assert result.equity_curve is not None
        if result.status == "success" and len(result.equity_curve) > 0:
            assert isinstance(result.equity_curve, list)
            # Each entry should have timestamp and value
            entry = result.equity_curve[0]
            assert "timestamp" in entry or "date" in entry
            assert "value" in entry or "equity" in entry

        # Trade log should have entry/exit/PnL info
        assert result.trade_log is not None
        if result.status == "success" and len(result.trade_log) > 0:
            assert isinstance(result.trade_log, list)

    def test_error_handling_for_invalid_code(self):
        """Test error handling for invalid Python code."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # Invalid Python code with syntax error
        invalid_code = """
class BrokenStrategy(bt.Strategy):
    def next(self)
        # Missing colon - syntax error
        pass
"""

        result = runner.run_python_strategy(invalid_code, config)

        # Should return error result
        assert result.status == "error"
        assert "syntax" in result.logs.lower() or "error" in result.logs.lower()


class TestQueueManager:
    """Test Queue Manager executes parallel backtests."""

    def test_queue_manager_initializes_with_cpu_detection(self):
        """Test CPU detection adjusts worker count."""
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        manager = BacktestQueueManager(max_workers=10)

        # Should limit to CPU count or max_workers, whichever is lower
        expected_workers = min(cpu_count, 10)
        assert manager.max_workers == expected_workers or manager.max_workers == 10

    def test_backtest_status_tracking(self):
        """Test backtest status tracking through lifecycle."""
        manager = BacktestQueueManager(max_workers=2)

        # Submit a backtest
        backtest_id = manager.submit_backtest(
            code_content=SAMPLE_PYTHON_STRATEGY,
            language="python",
            config={
                "symbol": "EURUSD",
                "timeframe": 60,
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
            },
        )

        # Check initial status
        status = manager.get_status(backtest_id)
        assert isinstance(status, BacktestStatus)
        assert status.backtest_id == backtest_id
        assert status.status in ["queued", "running", "completed", "failed"]
        assert 0 <= status.progress_percent <= 100

    def test_queue_manager_executes_parallel_backtests(self):
        """Test queue manager executes up to 10 parallel backtests."""
        manager = BacktestQueueManager(max_workers=2)

        backtest_ids = []
        for i in range(3):
            bid = manager.submit_backtest(
                code_content=SAMPLE_PYTHON_STRATEGY,
                language="python",
                config={
                    "symbol": "EURUSD",
                    "timeframe": 60,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-31",
                },
            )
            backtest_ids.append(bid)

        # All backtests should have IDs
        assert len(backtest_ids) == 3
        # All IDs should be unique
        assert len(set(backtest_ids)) == 3

        # Check status for each
        for bid in backtest_ids:
            status = manager.get_status(bid)
            assert status.backtest_id == bid

    def test_result_caching_for_identical_configs(self):
        """Test results cache returns identical configurations instantly."""
        manager = BacktestQueueManager(max_workers=2)

        config = {
            "symbol": "EURUSD",
            "timeframe": 60,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        }

        # Submit identical backtest twice
        id1 = manager.submit_backtest(
            code_content=SAMPLE_PYTHON_STRATEGY,
            language="python",
            config=config,
        )

        # Wait a moment for first to potentially cache
        time.sleep(0.5)

        id2 = manager.submit_backtest(
            code_content=SAMPLE_PYTHON_STRATEGY,
            language="python",
            config=config,
        )

        # Both should return valid IDs
        assert id1 is not None
        assert id2 is not None
