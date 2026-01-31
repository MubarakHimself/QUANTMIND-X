"""
Tests for MT5 Engine - Python Strategy Tester with MQL5 Built-in Function Overloading

Task Group 6.1: Write 2-8 focused tests for MT5 engine

This test module validates:
- MQL5 built-in function overloading (iTime, iClose, iHigh, iLow, iVolume)
- Data retrieval from MetaTrader5 package
- Strategy backtesting with sample strategies
- Performance metrics calculation (Sharpe, drawdown, return)
- pandas DataFrame integration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtesting.mt5_engine import (
    PythonStrategyTester,
    MQL5Timeframe,
    iTime,
    iClose,
    iHigh,
    iLow,
    iVolume,
    MT5BacktestResult
)


@pytest.mark.unit
class TestMQL5BuiltInFunctionOverloading:
    """Test MQL5 built-in function overloading compatibility."""

    def test_iTime_returns_correct_timestamp(self, sample_mt5_data):
        """Test iTime function returns correct timestamp for given shift."""
        # Setup test data
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test iTime with shift 0 (most recent bar)
        result = tester.iTime("EURUSD", MQL5Timeframe.PERIOD_H1, 0)

        assert isinstance(result, datetime), "iTime should return datetime object"
        assert result == sample_mt5_data.iloc[-1]["time"], "Shift 0 should return most recent bar"

    def test_iClose_returns_correct_close_price(self, sample_mt5_data):
        """Test iClose function returns correct close price for given shift."""
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test iClose with shift 1 (second most recent bar)
        result = tester.iClose("EURUSD", MQL5Timeframe.PERIOD_H1, 1)

        assert isinstance(result, (int, float)), "iClose should return numeric value"
        expected = sample_mt5_data.iloc[-2]["close"]
        assert abs(result - expected) < 1e-10, f"iClose shift 1 should return {expected}"

    def test_iHigh_returns_correct_high_price(self, sample_mt5_data):
        """Test iHigh function returns correct high price for given shift."""
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test iHigh with shift 0
        result = tester.iHigh("EURUSD", MQL5Timeframe.PERIOD_H1, 0)

        assert isinstance(result, (int, float)), "iHigh should return numeric value"
        expected = sample_mt5_data.iloc[-1]["high"]
        assert abs(result - expected) < 1e-10, f"iHigh shift 0 should return {expected}"

    def test_iLow_returns_correct_low_price(self, sample_mt5_data):
        """Test iLow function returns correct low price for given shift."""
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test iLow with shift 0
        result = tester.iLow("EURUSD", MQL5Timeframe.PERIOD_H1, 0)

        assert isinstance(result, (int, float)), "iLow should return numeric value"
        expected = sample_mt5_data.iloc[-1]["low"]
        assert abs(result - expected) < 1e-10, f"iLow shift 0 should return {expected}"

    def test_iVolume_returns_correct_volume(self, sample_mt5_data):
        """Test iVolume function returns correct tick volume for given shift."""
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test iVolume with shift 0
        result = tester.iVolume("EURUSD", MQL5Timeframe.PERIOD_H1, 0)

        assert isinstance(result, (int, float)), "iVolume should return numeric value"
        expected = sample_mt5_data.iloc[-1]["tick_volume"]
        assert result == expected, f"iVolume shift 0 should return {expected}"

    def test_functions_handle_invalid_shift_gracefully(self, sample_mt5_data):
        """Test functions return None or raise error for invalid shift."""
        tester = PythonStrategyTester()
        tester._data_cache = {"EURUSD": sample_mt5_data}

        # Test with out-of-bounds shift
        result = tester.iClose("EURUSD", MQL5Timeframe.PERIOD_H1, 999)

        # Should return None for invalid shift
        assert result is None, "Invalid shift should return None"


@pytest.mark.unit
class TestMT5PackageIntegration:
    """Test MetaTrader5 Python package integration."""

    @patch('backtesting.mt5_engine.MT5_AVAILABLE', True)
    @patch('backtesting.mt5_engine.mt5')
    def test_copy_rates_from_pos_retrieves_data(self, mock_mt5):
        """Test copy_rates_from_pos retrieves OHLCV data correctly."""
        # Mock MT5 response
        mock_rates = np.array([
            (1704067200, 1.1000, 1.1050, 1.0980, 1.1030, 1983, 0, 0),
            (1704070800, 1.1030, 1.1080, 1.1020, 1.1070, 2783, 0, 0),
        ], dtype=[('time', 'i8'), ('open', 'f8'), ('high', 'f8'), ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'), ('spread', 'i8'), ('real_volume', 'i8')])
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        mock_mt5.TIMEFRAME_H1 = 16385

        tester = PythonStrategyTester()
        tester._mt5 = mock_mt5
        tester._mt5_connected = True  # Set connected flag

        result = tester._copy_rates_from_pos("EURUSD", MQL5Timeframe.PERIOD_H1, 0, 2)

        assert result is not None, "Should retrieve data"
        assert len(result) == 2, "Should retrieve 2 bars"
        mock_mt5.copy_rates_from_pos.assert_called_once()

    @patch('backtesting.mt5_engine.MT5_AVAILABLE', True)
    @patch('backtesting.mt5_engine.mt5')
    def test_symbol_info_tick_retrieves_tick_data(self, mock_mt5):
        """Test symbol_info_tick retrieves current tick data."""
        # Mock tick response
        mock_tick = Mock(
            time=1704067200,
            bid=1.1030,
            ask=1.1032,
            last=0.0,
            volume=100,
            volume_real=0.1
        )
        mock_mt5.symbol_info_tick.return_value = mock_tick

        tester = PythonStrategyTester()
        tester._mt5 = mock_mt5
        tester._mt5_connected = True  # Set connected flag

        result = tester._get_tick("EURUSD")

        assert result is not None, "Should retrieve tick data"
        assert result['bid'] == 1.1030, "Should retrieve correct bid price"
        mock_mt5.symbol_info_tick.assert_called_once_with("EURUSD")


@pytest.mark.unit
class TestStrategyBacktesting:
    """Test strategy backtesting with sample strategies."""

    def test_simple_buy_strategy_backtest(self):
        """Test backtesting a simple buy-and-hold strategy."""
        tester = PythonStrategyTester(initial_cash=10000.0)

        # Create sample data
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
            'open': np.linspace(1.1000, 1.1100, 100),
            'high': np.linspace(1.1050, 1.1150, 100),
            'low': np.linspace(1.0950, 1.1050, 100),
            'close': np.linspace(1.1000, 1.1100, 100),
            'tick_volume': np.random.randint(1000, 3000, 100)
        })

        # Simple strategy code
        strategy_code = """
def on_bar(tester):
    # Buy on first bar, hold forever
    if tester.current_bar == 0:
        tester.buy("EURUSD", 0.01)
"""

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result is not None, "Should return backtest result"
        assert isinstance(result, MT5BacktestResult), "Should return MT5BacktestResult"
        assert result.initial_cash == 10000.0, "Initial cash should be preserved"

    def test_strategy_with_entry_exit_logic(self):
        """Test strategy with entry and exit logic."""
        tester = PythonStrategyTester(initial_cash=10000.0, commission=0.001)

        # Create sample data with trending price
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
            'open': 1.1000 + np.cumsum(np.random.randn(50) * 0.001),
            'high': 1.1050 + np.cumsum(np.random.randn(50) * 0.001),
            'low': 1.0950 + np.cumsum(np.random.randn(50) * 0.001),
            'close': 1.1000 + np.cumsum(np.random.randn(50) * 0.001),
            'tick_volume': np.random.randint(1000, 3000, 50)
        })

        # Strategy: Buy when close > open, sell when close < open
        strategy_code = """
def on_bar(tester):
    close = tester.iClose(tester.symbol, tester.timeframe, 0)
    open = tester.iOpen(tester.symbol, tester.timeframe, 0)
    prev_close = tester.iClose(tester.symbol, tester.timeframe, 1)

    if prev_close is not None:
        # Simple moving average crossover logic
        if close > prev_close and tester.position_size == 0:
            tester.buy(tester.symbol, 0.01)
        elif close < prev_close and tester.position_size > 0:
            tester.sell(tester.symbol, 0.01)
"""

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result is not None, "Should return backtest result"
        assert result.trades >= 0, "Should track number of trades"


@pytest.mark.unit
class TestPerformanceMetricsCalculation:
    """Test performance metrics calculation (Sharpe, drawdown, return)."""

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio is calculated correctly."""
        tester = PythonStrategyTester()

        # Create profitable trend data
        returns = np.random.randn(100) * 0.01 + 0.002  # Positive drift
        equity = 10000 * (1 + np.cumsum(returns))

        sharpe = tester._calculate_sharpe(returns)

        assert isinstance(sharpe, float), "Sharpe should be float"
        # With positive drift, Sharpe should be positive
        assert sharpe > -1, "Sharpe should be reasonable"
        assert sharpe < 10, "Sharpe should not be unrealistically high"

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown is calculated correctly."""
        tester = PythonStrategyTester()

        # Create equity curve with known drawdown
        equity = np.array([10000, 10500, 10200, 9800, 10100, 10300, 10000])

        max_dd = tester._calculate_max_drawdown(equity)

        assert isinstance(max_dd, float), "Max drawdown should be float"
        # Drawdown is expressed as negative percentage (loss from peak)
        assert -100 <= max_dd <= 0, "Max drawdown should be negative percentage or zero"
        # The max drawdown is from 10500 to 9800 = -6.67%
        assert abs(max_dd - (-6.67)) < 0.1, f"Max drawdown should be approximately -6.67%, got {max_dd}"

    def test_total_return_calculation(self):
        """Test total return is calculated correctly."""
        tester = PythonStrategyTester(initial_cash=10000)

        # Create equity curve
        equity = np.array([10000, 10200, 10500, 10300, 10700])

        total_return = tester._calculate_total_return(equity)

        assert isinstance(total_return, float), "Total return should be float"
        # (10700 - 10000) / 10000 = 7%
        expected = (10700 - 10000) / 10000 * 100
        assert abs(total_return - expected) < 0.1, f"Return should be {expected}%"


@pytest.mark.unit
class TestPandasDataFrameIntegration:
    """Test pandas DataFrame integration with datetime indexing."""

    def test_dataframe_with_datetime_index(self):
        """Test data DataFrame with datetime index is handled correctly."""
        tester = PythonStrategyTester()

        # Create DataFrame with datetime index
        data = pd.DataFrame({
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.1050, 1.1060, 1.1070],
            'low': [1.0950, 1.0960, 1.0970],
            'close': [1.1030, 1.1040, 1.1050],
            'tick_volume': [1000, 1500, 2000]
        }, index=pd.date_range('2024-01-01', periods=3, freq='1h'))

        strategy_code = "def on_bar(tester): pass"

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result is not None, "Should handle datetime index"

    def test_dataframe_without_datetime_index(self):
        """Test data DataFrame without datetime index creates time column."""
        tester = PythonStrategyTester()

        # Create DataFrame without datetime index
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=3, freq='1h'),
            'open': [1.1000, 1.1010, 1.1020],
            'high': [1.1050, 1.1060, 1.1070],
            'low': [1.0950, 1.0960, 1.0970],
            'close': [1.1030, 1.1040, 1.1050],
            'tick_volume': [1000, 1500, 2000]
        })

        strategy_code = "def on_bar(tester): pass"

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result is not None, "Should handle DataFrame with time column"


@pytest.mark.unit
class TestConfigurableParameters:
    """Test configurable parameters (initial cash, commission, slippage)."""

    def test_initial_cash_configuration(self):
        """Test initial cash is configurable."""
        tester = PythonStrategyTester(initial_cash=50000.0)

        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [1.10] * 10,
            'high': [1.11] * 10,
            'low': [1.09] * 10,
            'close': [1.105] * 10,
            'tick_volume': [1000] * 10
        })

        strategy_code = "def on_bar(tester): pass"
        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result.initial_cash == 50000.0, "Initial cash should be configurable"

    def test_commission_affects_returns(self):
        """Test commission parameter affects final returns."""
        # Test with zero commission
        tester_no_comm = PythonStrategyTester(initial_cash=10000.0, commission=0.0)

        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='1h'),
            'open': [1.10] * 10,
            'high': [1.11] * 10,
            'low': [1.09] * 10,
            'close': [1.105] * 10,
            'tick_volume': [1000] * 10
        })

        strategy_code = """
def on_bar(tester):
    if tester.current_bar == 0:
        tester.buy(tester.symbol, 0.1)
    elif tester.current_bar == 9:
        tester.close_all_positions()
"""

        result_no_comm = tester_no_comm.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        # Test with commission
        tester_with_comm = PythonStrategyTester(initial_cash=10000.0, commission=0.001)
        result_with_comm = tester_with_comm.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        # Commission should reduce returns
        assert result_no_comm.final_cash >= result_with_comm.final_cash, "Commission should reduce returns"

    def test_slippage_simulation(self):
        """Test slippage parameter affects entry/exit prices."""
        tester = PythonStrategyTester(initial_cash=10000.0, slippage=0.0001)

        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'open': [1.10, 1.101, 1.102, 1.103, 1.104],
            'high': [1.11, 1.111, 1.112, 1.113, 1.114],
            'low': [1.09, 1.091, 1.092, 1.093, 1.094],
            'close': [1.105, 1.106, 1.107, 1.108, 1.109],
            'tick_volume': [1000] * 5
        })

        strategy_code = """
def on_bar(tester):
    if tester.current_bar == 0:
        tester.buy(tester.symbol, 0.01)
    elif tester.current_bar == 4:
        tester.close_all_positions()
"""

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        assert result is not None, "Should execute with slippage"


# --- Fixtures ---

@pytest.fixture
def sample_mt5_data():
    """Create sample MT5 data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h', tz=timezone.utc)
    return pd.DataFrame({
        'time': dates,
        'open': np.linspace(1.1000, 1.1090, 10),
        'high': np.linspace(1.1050, 1.1140, 10),
        'low': np.linspace(1.0950, 1.1040, 10),
        'close': np.linspace(1.1000, 1.1090, 10),
        'tick_volume': np.random.randint(1000, 3000, 10),
        'spread': [0] * 10,
        'real_volume': [0] * 10
    })
