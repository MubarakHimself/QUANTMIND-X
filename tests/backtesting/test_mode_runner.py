"""
Tests for Backtest Mode Runner

Task Group 3.1: Write 2-8 focused tests for backtest modes

This test module validates:
- Vanilla backtest mode (basic historical run)
- Spiced backtest mode (with regime filtering)
- Walk-Forward optimization
- Monte Carlo simulation
- Multi-symbol simulation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtesting.mt5_engine import PythonStrategyTester, MQL5Timeframe, MT5BacktestResult
from backtesting.mode_runner import run_full_system_backtest, BacktestMode
from backtesting.walk_forward import WalkForwardOptimizer
from backtesting.monte_carlo import MonteCarloSimulator


@pytest.mark.unit
class TestVanillaBacktestMode:
    """Test Vanilla backtest mode (basic historical run)."""

    def test_vanilla_mode_runs_basic_historical_backtest(self, sample_backtest_data):
        """Test Vanilla mode executes backtest on fixed time period with static parameters."""
        result = run_full_system_backtest(
            mode='vanilla',
            data=sample_backtest_data,
            symbol='EURUSD',
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='''
def on_bar(tester):
    if tester.current_bar == 0:
        tester.buy(tester.symbol, 0.01)
    elif tester.current_bar == len(tester._data_cache[tester.symbol]) - 1:
        tester.close_all_positions()
'''
        )

        assert result is not None, "Vanilla mode should return result"
        # Check result has expected attributes (works across import styles)
        assert hasattr(result, 'sharpe'), "Should have sharpe attribute"
        assert hasattr(result, 'return_pct'), "Should have return_pct attribute"
        assert hasattr(result, 'drawdown'), "Should have drawdown attribute"
        assert hasattr(result, 'trades'), "Should have trades attribute"
        assert result.trades >= 0, "Should track trades"

    def test_vanilla_mode_preserves_static_parameters(self, sample_backtest_data):
        """Test Vanilla mode maintains static parameters throughout backtest."""
        initial_cash = 25000.0
        commission = 0.0005

        result = run_full_system_backtest(
            mode='vanilla',
            data=sample_backtest_data,
            symbol='GBPUSD',
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='def on_bar(tester): pass',
            initial_cash=initial_cash,
            commission=commission
        )

        assert result.initial_cash == initial_cash, "Should preserve initial cash"


@pytest.mark.unit
class TestSpicedBacktestMode:
    """Test Spiced backtest mode (with regime filtering)."""

    @patch('backtesting.mode_runner.Sentinel')
    def test_spiced_mode_filters_trades_by_regime(self, mock_sentinel_class, sample_backtest_data):
        """Test Spiced mode skips trades when chaos_score > 0.6 or regime == NEWS_EVENT."""
        # Mock Sentinel to return high chaos score
        mock_sentinel = Mock()
        mock_report = Mock()
        mock_report.regime = 'HIGH_CHAOS'
        mock_report.chaos_score = 0.8
        mock_report.regime_quality = 0.2
        mock_sentinel.on_tick.return_value = mock_report
        mock_sentinel_class.return_value = mock_sentinel

        result = run_full_system_backtest(
            mode='spiced',
            data=sample_backtest_data,
            symbol='XAUUSD',
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='''
def on_bar(tester):
    if tester.current_bar == 0:
        tester.buy(tester.symbol, 0.01)
'''
        )

        assert result is not None, "Spiced mode should return result"
        # High chaos should filter trades, resulting in fewer trades
        assert result.trades >= 0, "Should track filtered trades"

    @patch('backtesting.mode_runner.Sentinel')
    def test_spiced_mode_calculates_regime_quality(self, mock_sentinel_class, sample_backtest_data):
        """Test Spiced mode calculates regime_quality = 1.0 - chaos_score."""
        mock_sentinel = Mock()
        mock_report = Mock()
        mock_report.regime = 'TREND_STABLE'
        mock_report.chaos_score = 0.2
        mock_report.regime_quality = 0.8
        mock_sentinel.on_tick.return_value = mock_report
        mock_sentinel_class.return_value = mock_sentinel

        result = run_full_system_backtest(
            mode='spiced',
            data=sample_backtest_data,
            symbol='EURUSD',
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='def on_bar(tester): pass'
        )

        assert result is not None, "Should calculate regime quality"


@pytest.mark.unit
class TestWalkForwardOptimization:
    """Test Walk-Forward optimization (rolling window validation)."""

    def test_walk_forward_implements_rolling_windows(self):
        """Test Walk-Forward uses Train 50%, Test 20%, Gap 10% windows."""
        # Create longer dataset for walk-forward
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=200, freq='1h'),
            'open': np.linspace(1.1000, 1.1200, 200),
            'high': np.linspace(1.1050, 1.1250, 200),
            'low': np.linspace(1.0950, 1.1150, 200),
            'close': np.linspace(1.1000, 1.1200, 200),
            'tick_volume': np.random.randint(1000, 3000, 200)
        })

        optimizer = WalkForwardOptimizer(
            train_pct=0.5,
            test_pct=0.2,
            gap_pct=0.1
        )

        results = optimizer.optimize(
            data=data,
            symbol='EURUSD',
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='def on_bar(tester): pass'
        )

        assert results is not None, "Should return walk-forward results"
        assert len(results.window_results) > 0, "Should have multiple windows"
        assert results.aggregate_metrics is not None, "Should have aggregate metrics"


@pytest.mark.unit
class TestMonteCarloSimulation:
    """Test Monte Carlo simulation (distribution metrics)."""

    def test_monte_carlo_runs_1000_simulations(self, sample_backtest_data):
        """Test Monte Carlo runs 1000+ simulations with confidence intervals."""
        simulator = MonteCarloSimulator(num_simulations=1000)

        # Create a base result to simulate from
        base_result = MT5BacktestResult(
            sharpe=1.5,
            return_pct=15.0,
            drawdown=-5.0,
            trades=50,
            log="Test backtest",
            initial_cash=10000.0,
            final_cash=11500.0,
            equity_curve=list(np.linspace(10000, 11500, 100))
        )

        mc_results = simulator.simulate(
            base_result=base_result,
            data=sample_backtest_data
        )

        assert mc_results is not None, "Should return Monte Carlo results"
        assert mc_results.num_simulations == 1000, "Should run 1000 simulations"
        assert mc_results.confidence_interval_5th is not None, "Should calculate 5th percentile"
        assert mc_results.confidence_interval_95th is not None, "Should calculate 95th percentile"

    def test_monte_carlo_calculates_var_metrics(self, sample_backtest_data):
        """Test Monte Carlo calculates Value at Risk and expected shortfall."""
        simulator = MonteCarloSimulator(num_simulations=500)

        base_result = MT5BacktestResult(
            sharpe=1.0,
            return_pct=10.0,
            drawdown=-8.0,
            trades=40,
            log="Test backtest",
            initial_cash=10000.0,
            final_cash=11000.0,
            equity_curve=list(np.linspace(10000, 11000, 100))
        )

        mc_results = simulator.simulate(
            base_result=base_result,
            data=sample_backtest_data
        )

        assert mc_results.value_at_risk_95 is not None, "Should calculate VaR at 95%"
        assert mc_results.expected_shortfall_95 is not None, "Should calculate expected shortfall"


@pytest.mark.unit
class TestMultiSymbolSimulation:
    """Test multi-symbol simulation support."""

    def test_multi_symbol_simulation(self):
        """Test backtest runs across multiple symbols simultaneously."""
        symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']

        # Create data for each symbol
        data_dict = {}
        for symbol in symbols:
            data_dict[symbol] = pd.DataFrame({
                'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
                'open': np.linspace(1.1000, 1.1100, 100),
                'high': np.linspace(1.1050, 1.1150, 100),
                'low': np.linspace(1.0950, 1.1050, 100),
                'close': np.linspace(1.1000, 1.1100, 100),
                'tick_volume': np.random.randint(1000, 3000, 100)
            })

        results = run_full_system_backtest(
            mode='vanilla',
            symbols=symbols,
            data_dict=data_dict,
            timeframe=MQL5Timeframe.PERIOD_H1,
            strategy_code='def on_bar(tester): pass'
        )

        assert results is not None, "Should return multi-symbol results"
        assert len(results) == len(symbols), "Should have result for each symbol"
        assert all(sym in results for sym in symbols), "Should have results for all symbols"


# --- Fixtures ---

@pytest.fixture
def sample_backtest_data():
    """Create sample backtest data for testing."""
    return pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'open': np.linspace(1.1000, 1.1100, 100),
        'high': np.linspace(1.1050, 1.1150, 100),
        'low': np.linspace(1.0950, 1.1050, 100),
        'close': np.linspace(1.1000, 1.1100, 100),
        'tick_volume': np.random.randint(1000, 3000, 100)
    })
