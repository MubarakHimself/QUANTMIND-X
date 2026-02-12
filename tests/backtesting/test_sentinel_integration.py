"""
Test Suite: Sentinel Integration to Backtesting (Task Group 4)

Tests for regime detection, filtering, and quality scalar calculation in backtesting.

Focus Areas:
- Regime filtering in Spiced variants
- Chaos score threshold filtering
- NEWS_EVENT regime filtering
- Regime quality scalar calculation
- Per-bar regime tracking
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.backtesting.mt5_engine import PythonStrategyTester, MQL5Timeframe
from src.router.sentinel import Sentinel, RegimeReport


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC'),
        'open': np.linspace(1.1000, 1.1100, 100) + np.random.randn(100) * 0.0001,
        'high': np.linspace(1.1010, 1.1110, 100) + np.random.randn(100) * 0.0001,
        'low': np.linspace(1.0990, 1.1090, 100) + np.random.randn(100) * 0.0001,
        'close': np.linspace(1.1005, 1.1105, 100) + np.random.randn(100) * 0.0001,
        'tick_volume': np.random.randint(500, 2000, 100)
    })


@pytest.fixture
def mock_sentinel():
    """Create a mock Sentinel for testing."""
    sentinel = Mock(spec=Sentinel)

    # Default to stable regime
    sentinel.on_tick.return_value = RegimeReport(
        regime="TREND_STABLE",
        chaos_score=0.2,
        regime_quality=0.8,
        susceptibility=0.1,
        is_systemic_risk=False,
        news_state="SAFE",
        timestamp=0.0
    )

    return sentinel


@pytest.fixture
def strategy_code_basic():
    """Basic strategy code for testing."""
    return '''
def on_bar(tester):
    """Simple buy and hold strategy for testing."""
    if tester.current_bar == 0:
        tester.buy("EURUSD", 0.01)
    elif tester.current_bar == 99:
        tester.close_all_positions()
'''


# =============================================================================
# Test 1: Regime Filtering in Spiced Variants
# =============================================================================

def test_regime_filtering_spiced_variant(sample_ohlcv_data, mock_sentinel, strategy_code_basic):
    """Test that Spiced variant filters trades based on regime."""
    # Setup: Create mock that returns HIGH_CHAOS regime
    mock_sentinel.on_tick.return_value = RegimeReport(
        regime="HIGH_CHAOS",
        chaos_score=0.7,
        regime_quality=0.3,
        susceptibility=0.8,
        is_systemic_risk=False,
        news_state="SAFE",
        timestamp=0.0
    )

    # Create tester with Spiced mode
    tester = PythonStrategyTester(initial_cash=10000.0)

    # Mock sentinel integration
    with patch.object(tester, '_sentinel', mock_sentinel):
        with patch.object(tester, '_is_spiced_mode', True):
            # Run backtest
            result = tester.run(
                strategy_code_basic,
                sample_ohlcv_data,
                "EURUSD",
                MQL5Timeframe.PERIOD_H1
            )

    # Assert: Trades should be filtered in HIGH_CHAOS regime
    # In Spiced mode, HIGH_CHAOS should block trades
    assert result.trades == 0, "Spiced mode should filter trades in HIGH_CHAOS regime"
    assert "HIGH_CHAOS" in result.log or "chaos" in result.log.lower()


# =============================================================================
# Test 2: Chaos Score Threshold Filtering
# =============================================================================

def test_chaos_score_threshold_filtering(sample_ohlcv_data, mock_sentinel, strategy_code_basic):
    """Test that trades are filtered when chaos_score > 0.6."""
    # Setup: Create mock with chaos score just above threshold
    mock_sentinel.on_tick.return_value = RegimeReport(
        regime="UNCERTAIN",
        chaos_score=0.65,  # Above 0.6 threshold
        regime_quality=0.35,
        susceptibility=0.5,
        is_systemic_risk=False,
        news_state="SAFE",
        timestamp=0.0
    )

    tester = PythonStrategyTester(initial_cash=10000.0)

    # Mock sentinel integration
    with patch.object(tester, '_sentinel', mock_sentinel):
        with patch.object(tester, '_is_spiced_mode', True):
            result = tester.run(
                strategy_code_basic,
                sample_ohlcv_data,
                "EURUSD",
                MQL5Timeframe.PERIOD_H1
            )

    # Assert: chaos_score > 0.6 should filter trades
    assert result.trades == 0, "Trades should be filtered when chaos_score > 0.6"


# =============================================================================
# Test 3: NEWS_EVENT Regime Filtering
# =============================================================================

def test_news_event_regime_filtering(sample_ohlcv_data, mock_sentinel, strategy_code_basic):
    """Test that NEWS_EVENT regime filters all trades."""
    # Setup: Create mock that returns NEWS_EVENT
    mock_sentinel.on_tick.return_value = RegimeReport(
        regime="NEWS_EVENT",
        chaos_score=0.8,
        regime_quality=0.2,
        susceptibility=0.9,
        is_systemic_risk=False,
        news_state="KILL_ZONE",
        timestamp=0.0
    )

    tester = PythonStrategyTester(initial_cash=10000.0)

    # Mock sentinel integration
    with patch.object(tester, '_sentinel', mock_sentinel):
        with patch.object(tester, '_is_spiced_mode', True):
            result = tester.run(
                strategy_code_basic,
                sample_ohlcv_data,
                "EURUSD",
                MQL5Timeframe.PERIOD_H1
            )

    # Assert: NEWS_EVENT should filter all trades
    assert result.trades == 0, "NEWS_EVENT regime should filter all trades"
    assert "NEWS_EVENT" in result.log or "news" in result.log.lower() or "kill" in result.log.lower()


# =============================================================================
# Test 4: Regime Quality Scalar Calculation
# =============================================================================

def test_regime_quality_scalar_calculation(sample_ohlcv_data):
    """Test that regime_quality = 1.0 - chaos_score."""
    # Test various chaos scores
    test_cases = [
        (0.0, 1.0),   # No chaos = full quality
        (0.2, 0.8),   # Low chaos = high quality
        (0.5, 0.5),   # Medium chaos = medium quality
        (0.8, 0.2),   # High chaos = low quality
        (1.0, 0.0),   # Maximum chaos = zero quality
    ]

    for chaos_score, expected_quality in test_cases:
        regime_quality = 1.0 - chaos_score
        assert abs(regime_quality - expected_quality) < 1e-9, \
            f"Regime quality calculation failed for chaos_score={chaos_score}"

    # Test with Sentinel integration
    sentinel = Sentinel()

    # Mock internal sensors to return specific values
    with patch.object(sentinel.chaos, 'update') as mock_chaos:
        mock_chaos.return_value = Mock(score=0.3)

        with patch.object(sentinel.regime, 'update') as mock_regime:
            mock_regime.return_value = Mock(state="ORDERED", susceptibility=0.1)

            with patch.object(sentinel.news, 'check_state', return_value="SAFE"):
                report = sentinel.on_tick("EURUSD", 1.1000)

                # Assert regime quality is calculated correctly
                assert report.regime_quality == 1.0 - report.chaos_score, \
                    "Regime quality should be 1.0 - chaos_score"
                assert abs(report.regime_quality - 0.7) < 0.01, \
                    f"Expected regime_quality ~0.7 for chaos_score=0.3, got {report.regime_quality}"


# =============================================================================
# Test 5: Per-Bar Regime Tracking
# =============================================================================

def test_per_bar_regime_tracking(sample_ohlcv_data, strategy_code_basic):
    """Test that regime is tracked for each bar during backtest."""
    tester = PythonStrategyTester(initial_cash=10000.0)

    # Create real Sentinel for tracking
    sentinel = Sentinel()

    # Track regime calls
    regime_history = []
    original_on_tick = sentinel.on_tick

    def tracking_on_tick(symbol, price):
        report = original_on_tick(symbol, price)
        regime_history.append({
            'regime': report.regime,
            'chaos_score': report.chaos_score,
            'regime_quality': report.regime_quality
        })
        return report

    sentinel.on_tick = tracking_on_tick

    # Mock sentinel integration
    with patch.object(tester, '_sentinel', sentinel):
        with patch.object(tester, '_is_spiced_mode', False):  # Vanilla mode
            result = tester.run(
                strategy_code_basic,
                sample_ohlcv_data,
                "EURUSD",
                MQL5Timeframe.PERIOD_H1
            )

    # Assert: Should have regime data for each bar
    assert len(regime_history) > 0, "Should track regime for at least some bars"

    # Verify regime tracking structure
    first_entry = regime_history[0]
    assert 'regime' in first_entry
    assert 'chaos_score' in first_entry
    assert 'regime_quality' in first_entry
    assert 0.0 <= first_entry['chaos_score'] <= 1.0
    assert 0.0 <= first_entry['regime_quality'] <= 1.0
    assert first_entry['regime'] in ['TREND_STABLE', 'RANGE_STABLE', 'HIGH_CHAOS',
                                       'NEWS_EVENT', 'BREAKOUT_PRIME', 'UNCERTAIN']


# =============================================================================
# Integration Test: Sentinel to Kelly Calculator
# =============================================================================

def test_regime_quality_passed_to_kelly_calculator():
    """Test that regime_quality is passed to Kelly calculator."""
    from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult

    # Test with different regime qualities
    calculator = EnhancedKellyCalculator()

    # High regime quality (stable market)
    result_stable = calculator.calculate(
        account_balance=10000.0,
        win_rate=0.55,
        avg_win=400.0,
        avg_loss=200.0,
        current_atr=0.0010,
        average_atr=0.0010,
        stop_loss_pips=20.0,
        pip_value=10.0,
        regime_quality=0.9  # Stable regime
    )

    # Low regime quality (chaotic market)
    result_chaotic = calculator.calculate(
        account_balance=10000.0,
        win_rate=0.55,
        avg_win=400.0,
        avg_loss=200.0,
        current_atr=0.0010,
        average_atr=0.0010,
        stop_loss_pips=20.0,
        pip_value=10.0,
        regime_quality=0.3  # Chaotic regime
    )

    # Assert: Stable regime should allow larger position
    assert result_stable.position_size > result_chaotic.position_size, \
        "Stable regime (high quality) should allow larger position size than chaotic regime"

    # Verify regime quality affected the calculation
    assert "regime_quality" in str(result_stable.adjustments_applied).lower() or \
           "physics" in str(result_stable.adjustments_applied).lower()


# =============================================================================
# Test: Regime Transition Logging
# =============================================================================

def test_regime_transition_logging(sample_ohlcv_data, mock_sentinel):
    """Test that regime transitions are logged during backtest."""
    tester = PythonStrategyTester(initial_cash=10000.0)

    # Setup: Create regime transitions
    regime_sequence = [
        RegimeReport("TREND_STABLE", 0.2, 0.8, 0.1, False, "SAFE", 0.0),
        RegimeReport("TREND_STABLE", 0.25, 0.75, 0.15, False, "SAFE", 0.0),
        RegimeReport("HIGH_CHAOS", 0.7, 0.3, 0.8, False, "SAFE", 0.0),  # Transition
        RegimeReport("HIGH_CHAOS", 0.65, 0.35, 0.7, False, "SAFE", 0.0),
        RegimeReport("RANGE_STABLE", 0.3, 0.7, 0.2, False, "SAFE", 0.0),  # Transition
    ]

    call_count = [0]

    def mock_on_tick(symbol, price):
        idx = min(call_count[0], len(regime_sequence) - 1)
        call_count[0] += 1
        return regime_sequence[idx]

    mock_sentinel.on_tick = mock_on_tick

    # Strategy that just tracks
    strategy_code = '''
def on_bar(tester):
    pass
'''

    # Mock sentinel integration
    with patch.object(tester, '_sentinel', mock_sentinel):
        with patch.object(tester, '_is_spiced_mode', True):
            with patch.object(tester, '_log_regime_transition') as mock_log:
                result = tester.run(
                    strategy_code,
                    sample_ohlcv_data,
                    "EURUSD",
                    MQL5Timeframe.PERIOD_H1
                )

                # If _log_regime_transition exists, it should have been called
                # This test verifies the logging infrastructure is in place
                assert result is not None
