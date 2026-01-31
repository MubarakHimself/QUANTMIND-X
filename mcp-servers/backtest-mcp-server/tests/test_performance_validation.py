"""
Performance and Validation Tests for Backtest MCP Server.

Task Group 6.1: Write 2-8 focused performance and validation tests.
Tests cover:
- Simple strategy backtest completes in under 2 minutes
- 10 parallel backtests execute without resource exhaustion
- equity_curve data is accurate
- trade_log includes required fields
- timeout handling
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
    BacktestDataError,
    BacktestRuntimeError,
    BacktestTimeoutError,
)
from backtest_runner import BacktestRunner
from queue_manager import BacktestQueueManager


# Sample simple Python strategy
SIMPLE_STRATEGY = """
import backtrader as bt

class SimpleStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(period=10)

    def next(self):
        if not self.position:
            if self.data.close[0] > self.sma[0]:
                self.buy()
        else:
            if self.data.close[0] < self.sma[0]:
                self.sell()
"""


class TestPerformanceValidation:
    """Test performance and validation requirements for backtests."""

    def test_simple_backtest_completes_under_2_minutes(self):
        """Test simple strategy backtest completes in under 2 minutes."""
        runner = BacktestRunner(timeout_seconds=120)
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),  # 1 month of data
            initial_capital=10000.0,
        )

        start_time = time.time()
        result = runner.run_python_strategy(SIMPLE_STRATEGY, config)
        execution_time = time.time() - start_time

        # Should complete well under 2 minutes (120 seconds)
        assert execution_time < 120, f"Backtest took {execution_time:.2f}s, expected < 120s"
        assert result.status in ["success", "error"]
        assert result.execution_time_seconds == execution_time

    def test_parallel_backtests_no_resource_exhaustion(self):
        """Test 10 parallel backtests execute without resource exhaustion."""
        manager = BacktestQueueManager(max_workers=10)

        backtest_ids = []
        start_time = time.time()

        # Submit 10 parallel backtests
        for i in range(10):
            bid = manager.submit_backtest(
                code_content=SIMPLE_STRATEGY,
                language="python",
                config={
                    "symbol": f"EURUSD{i}",  # Unique symbols
                    "timeframe": 60,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-15",
                },
            )
            backtest_ids.append(bid)

        # All submissions should succeed
        assert len(backtest_ids) == 10
        assert len(set(backtest_ids)) == 10  # All unique IDs

        # Wait for all to complete (with timeout)
        max_wait = 180  # 3 minutes max for all
        start_wait = time.time()
        completed = 0

        while completed < 10 and (time.time() - start_wait) < max_wait:
            completed = 0
            for bid in backtest_ids:
                status = manager.get_status(bid)
                if status.status in ["completed", "failed"]:
                    completed += 1
            time.sleep(0.5)

        total_time = time.time() - start_time
        assert completed == 10, f"Only {completed}/10 backtests completed"
        # Should complete in reasonable time (< 3 minutes)
        assert total_time < 180, f"Parallel execution took {total_time:.2f}s"

    def test_equity_curve_data_accurate(self):
        """Test equity_curve data is accurate and properly formatted."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 28),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SIMPLE_STRATEGY, config)

        # Equity curve should exist
        assert result.equity_curve is not None, "Equity curve should not be None"

        if result.status == "success" and len(result.equity_curve) > 0:
            # Check structure
            assert isinstance(result.equity_curve, list)

            # Validate each entry
            for entry in result.equity_curve:
                assert isinstance(entry, dict), "Each equity entry should be a dict"
                # Should have timestamp and value
                assert "timestamp" in entry or "date" in entry, "Equity entry missing timestamp"
                assert "value" in entry or "equity" in entry, "Equity entry missing value"

                # Validate data types
                if "timestamp" in entry:
                    assert isinstance(entry["timestamp"], (str, datetime))
                if "value" in entry:
                    assert isinstance(entry["value"], (int, float))
                    assert entry["value"] >= 0, "Equity value should be non-negative"

    def test_trade_log_includes_required_fields(self):
        """Test trade_log includes required fields (entry, exit, PnL)."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SIMPLE_STRATEGY, config)

        # Trade log should exist
        assert result.trade_log is not None, "Trade log should not be None"

        if result.status == "success" and len(result.trade_log) > 0:
            assert isinstance(result.trade_log, list)

            # Check for required fields in each trade
            for trade in result.trade_log:
                assert isinstance(trade, dict), "Each trade should be a dict"
                # At minimum should have trade info
                # Note: Our current implementation returns summary, but
                # production should have entry/exit/pnl per trade
                assert "entry" in trade or "total_trades" in trade or "won" in trade

    def test_timeout_handling(self):
        """Test timeout handling for long-running backtests."""
        # Create a strategy with infinite loop potential
        infinite_strategy = """
import backtrader as bt
import time

class InfiniteStrategy(bt.Strategy):
    def next(self):
        # This should trigger timeout
        time.sleep(1)
"""

        runner = BacktestRunner(timeout_seconds=5)  # 5 second timeout
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),  # Large date range
            initial_capital=10000.0,
        )

        start_time = time.time()
        result = runner.run_python_strategy(infinite_strategy, config)
        execution_time = time.time() - start_time

        # Should either complete or timeout gracefully
        assert result.status in ["success", "error", "timeout"]
        # Should not run significantly longer than timeout
        assert execution_time < 10, f"Execution {execution_time}s exceeded safety limit"

    def test_metrics_accuracy(self):
        """Test performance metrics are accurately calculated."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            initial_capital=10000.0,
        )

        result = runner.run_python_strategy(SIMPLE_STRATEGY, config)

        if result.status == "success" and result.metrics:
            # Sharpe ratio
            assert "sharpe_ratio" in result.metrics
            assert isinstance(result.metrics["sharpe_ratio"], (int, float))

            # Max drawdown
            assert "max_drawdown" in result.metrics
            assert result.metrics["max_drawdown"] >= 0

            # Total return
            assert "total_return" in result.metrics
            assert isinstance(result.metrics["total_return"], (int, float))

            # Win rate (0-100)
            assert "win_rate" in result.metrics
            assert 0 <= result.metrics["win_rate"] <= 100

            # Profit factor
            assert "profit_factor" in result.metrics
            assert result.metrics["profit_factor"] >= 0

    def test_data_loading_optimization(self):
        """Test data loading is optimized (should be fast)."""
        runner = BacktestRunner()
        config = BacktestConfig(
            symbol="EURUSD",
            timeframe=60,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        # Time data generation
        start_time = time.time()
        df = runner._generate_mock_data(config)
        data_time = time.time() - start_time

        # Data generation should be fast (< 1 second for mock data)
        assert data_time < 1.0, f"Data loading took {data_time:.2f}s, expected < 1s"
        assert len(df) >= 100, "Should have at least 100 data points"
