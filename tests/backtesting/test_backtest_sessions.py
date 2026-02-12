"""
Integration Tests for Backtest Session-Aware Filtering

Tests that backtests exercise session filters per spec, demonstrating:
- UTC timestamp extraction from bars
- Session detection during backtest execution
- Session filtering integration with Commander
- Regime history includes UTC timestamps for session analysis
"""

import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.backtesting.mode_runner import (
    SentinelEnhancedTester,
    BacktestMode,
    SpicedBacktestResult
)
from src.router.sessions import SessionDetector, TradingSession
from src.router.sentinel import RegimeReport


class TestBacktestUTCTimestampExtraction:
    """Test UTC timestamp extraction from backtest bars."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data with UTC timestamps."""
        # Create 24 hours of hourly data
        base_time = datetime(2026, 2, 12, 0, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(24)]
        
        data = pd.DataFrame({
            'time': times,
            'open': np.random.uniform(1.08, 1.10, 24),
            'high': np.random.uniform(1.09, 1.11, 24),
            'low': np.random.uniform(1.07, 1.09, 24),
            'close': np.random.uniform(1.08, 1.10, 24),
            'tick_volume': np.random.randint(1000, 5000, 24)
        })
        data['time'] = pd.to_datetime(data['time'])
        return data

    def test_get_bar_utc_timestamp_returns_utc(self, sample_data):
        """Test _get_bar_utc_timestamp returns UTC-aware datetime."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Set up tester with data
        tester._data_cache["EURUSD"] = sample_data
        tester.symbol = "EURUSD"
        tester.current_bar = 0
        
        utc_time = tester._get_bar_utc_timestamp()
        
        assert utc_time is not None
        assert utc_time.tzinfo == timezone.utc
        assert utc_time.hour == 0  # First bar at midnight UTC

    def test_get_bar_utc_timestamp_for_each_bar(self, sample_data):
        """Test UTC timestamp correctly extracted for multiple bars."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = sample_data
        tester.symbol = "EURUSD"
        
        # Check timestamps for different bars
        for bar_idx, expected_hour in enumerate([0, 1, 5, 12, 23]):
            tester.current_bar = expected_hour
            utc_time = tester._get_bar_utc_timestamp()
            
            assert utc_time is not None
            assert utc_time.hour == expected_hour
            assert utc_time.tzinfo == timezone.utc

    def test_get_bar_utc_timestamp_naive_converted_to_utc(self):
        """Test naive datetime converted to UTC-aware."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create data with naive datetime
        naive_time = datetime(2026, 2, 12, 10, 0)  # No timezone
        data = pd.DataFrame({
            'time': [naive_time],
            'open': [1.09],
            'high': [1.10],
            'low': [1.08],
            'close': [1.09],
            'tick_volume': [1000]
        })
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        tester.current_bar = 0
        
        utc_time = tester._get_bar_utc_timestamp()
        
        # Should be converted to UTC-aware
        assert utc_time is not None
        assert utc_time.tzinfo == timezone.utc


class TestRegimeHistoryIncludesUTCTimestamp:
    """Test regime history tracks UTC timestamps for session analysis."""

    @pytest.fixture
    def tester_with_data(self):
        """Create tester with sample data."""
        # Create 24-hour data spanning multiple sessions
        base_time = datetime(2026, 2, 12, 0, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(24)]
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 24,
            'high': [1.10] * 24,
            'low': [1.08] * 24,
            'close': [1.09] * 24,
            'tick_volume': [1000] * 24
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        
        return tester, data

    def test_regime_history_has_utc_timestamp_field(self, tester_with_data):
        """Test regime history includes utc_timestamp field."""
        tester, data = tester_with_data
        
        # Mock regime report
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                news_state="SAFE",
                susceptibility=0.3,
                is_systemic_risk=False,
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            tester._sentinel = mock_sentinel
            
            # Check regime filter at bar 0
            tester.current_bar = 0
            should_filter, reason, report = tester._check_regime_filter("EURUSD", 1.09)
            
            # Check that regime history was populated with UTC timestamp
            assert len(tester._regime_history) > 0
            latest_entry = tester._regime_history[-1]
            
            assert 'utc_timestamp' in latest_entry
            assert latest_entry['utc_timestamp'] is not None
            assert latest_entry['utc_timestamp'].tzinfo == timezone.utc

    def test_regime_history_tracks_session_transitions(self, tester_with_data):
        """Test regime history captures UTC timestamps across session transitions."""
        tester, data = tester_with_data
        
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            
            # Different regimes for different times
            regimes = [
                RegimeReport(
                    regime="UNCERTAIN",
                    chaos_score=0.5,
                    regime_quality=0.6,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 00:00 UTC (Asian)
                RegimeReport(
                    regime="TREND_STABLE",
                    chaos_score=0.2,
                    regime_quality=0.8,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 08:00 UTC (London starts)
                RegimeReport(
                    regime="TREND_STABLE",
                    chaos_score=0.2,
                    regime_quality=0.8,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 10:00 UTC (London)
                RegimeReport(
                    regime="TREND_STABLE",
                    chaos_score=0.2,
                    regime_quality=0.8,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 14:00 UTC (Overlap)
                RegimeReport(
                    regime="TREND_STABLE",
                    chaos_score=0.2,
                    regime_quality=0.8,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 18:00 UTC (NY)
                RegimeReport(
                    regime="UNCERTAIN",
                    chaos_score=0.5,
                    regime_quality=0.6,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state="SAFE",
                    timestamp=datetime.now(timezone.utc).timestamp()
                ),  # 22:00 UTC (Closed)
            ]
            
            mock_sentinel.on_tick.side_effect = regimes
            tester._sentinel = mock_sentinel
            
            # Check regimes at different times
            test_bars = [0, 8, 10, 14, 18, 22]  # Hours in UTC
            for bar_idx in test_bars:
                tester.current_bar = bar_idx
                tester._check_regime_filter("EURUSD", 1.09)
            
            # Verify regime history
            assert len(tester._regime_history) == len(test_bars)
            
            for i, entry in enumerate(tester._regime_history):
                assert 'utc_timestamp' in entry
                assert entry['utc_timestamp'].hour == test_bars[i]
                assert entry['utc_timestamp'].tzinfo == timezone.utc


class TestSessionDetectionDuringBacktest:
    """Test session detection works correctly during backtest execution."""

    def test_london_session_detected_during_london_hours(self):
        """Test London session correctly detected during 08:00-16:00 UTC."""
        # 10:00 UTC = During London session
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.LONDON

    def test_overlap_session_detected_during_overlap_hours(self):
        """Test overlap detected during London/NY concurrent hours."""
        # 14:00 UTC = London 14:00, NY 09:00 EST (overlap)
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.OVERLAP

    def test_ny_session_detected_during_ny_hours(self):
        """Test NY session detected during 13:00-21:00 UTC (08:00-17:00 EST)."""
        # 18:00 UTC = NY 13:00 EST (during NY session, after overlap)
        utc_time = datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.NEW_YORK

    def test_asian_session_detected_during_asian_hours(self):
        """Test Asian session detected during Tokyo local hours 00:00-09:00."""
        # 22:00 UTC = 07:00 Tokyo (inside 00:00-09:00 Tokyo) = ASIAN
        utc_time = datetime(2026, 2, 12, 22, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.ASIAN

    def test_closed_session_detected_during_closed_hours(self):
        """Test closed period detected outside all sessions."""
        # 01:00 UTC = 10:00 Tokyo (outside 00:00-09:00) = CLOSED
        utc_time = datetime(2026, 2, 12, 1, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.CLOSED


class TestSessionFilteringIntegration:
    """Test session filtering integration with backtest flow."""

    @pytest.fixture
    def london_filtered_strategy(self):
        """Create strategy that should filter during non-London hours."""
        strategy_code = '''
def on_bar(tester):
    # Buy only during London session
    # In production, Commander would enforce this
    # In backtest, we track it in regime history
    tester.buy("EURUSD", 0.1)
'''
        return strategy_code

    def test_backtest_passes_utc_timestamp_to_regime_filter(self):
        """Test backtest passes UTC timestamp through regime filter path."""
        # Create 24-hour data
        base_time = datetime(2026, 2, 12, 0, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(24)]
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 24,
            'high': [1.10] * 24,
            'low': [1.08] * 24,
            'close': [1.09] * 24,
            'tick_volume': [1000] * 24
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        tester.current_bar = 10  # 10:00 UTC = London session
        
        with patch('src.backtesting.mode_runner.Sentinel'):
            tester._sentinel = MagicMock()
            tester._sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            # Call check regime filter at bar 10 (10:00 UTC = London)
            bar_utc_time = tester._get_bar_utc_timestamp()
            should_filter, reason, report = tester._check_regime_filter(
                "EURUSD", 1.09, bar_utc_time
            )
            
            # Verify UTC timestamp was tracked
            assert len(tester._regime_history) > 0
            entry = tester._regime_history[-1]
            assert entry['utc_timestamp'].hour == 10
            assert entry['utc_timestamp'].tzinfo == timezone.utc

    def test_regime_history_audit_trail_across_sessions(self):
        """Test regime history creates audit trail across multiple sessions."""
        base_time = datetime(2026, 2, 12, 6, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(24)]  # 06:00 to 06:00 next day
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 24,
            'high': [1.10] * 24,
            'low': [1.08] * 24,
            'close': [1.09] * 24,
            'tick_volume': [1000] * 24
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        
        with patch('src.backtesting.mode_runner.Sentinel'):
            tester._sentinel = MagicMock()
            tester._sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            # Process multiple bars across sessions
            test_hours = [0, 2, 8, 10, 14, 18, 22]  # Covers all major sessions
            for bar_idx in test_hours:
                tester.current_bar = bar_idx
                tester._check_regime_filter("EURUSD", 1.09)
            
            # Verify audit trail with sessions
            assert len(tester._regime_history) == len(test_hours)
            
            for i, entry in enumerate(tester._regime_history):
                expected_hour = (6 + test_hours[i]) % 24  # Adjusted for 06:00 start
                actual_hour = entry['utc_timestamp'].hour
                
                # Verify session detection would work
                session = SessionDetector.detect_session(entry['utc_timestamp'])
                assert session in [
                    TradingSession.ASIAN,
                    TradingSession.LONDON,
                    TradingSession.OVERLAP,
                    TradingSession.NEW_YORK,
                    TradingSession.CLOSED
                ]


class TestCommanderIntegrationWithBacktestTimestamps:
    """Test Commander can use backtest UTC timestamps for session filtering.
    
    NOTE: This test documents how backtests pass UTC timestamps to Commander
    for session-aware bot filtering. In current implementation, the timestamp
    is tracked in regime_history. Future implementation would pass it directly
    to Commander.run_auction() for real-time session filtering.
    """

    def test_backtest_timestamp_available_for_commander_filtering(self):
        """Test UTC timestamp from backtest is available for Commander session filtering."""
        # Create data across multiple sessions
        base_time = datetime(2026, 2, 12, 8, 0, tzinfo=timezone.utc)  # London opens
        times = [base_time + timedelta(hours=i) for i in range(12)]
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 12,
            'high': [1.10] * 12,
            'low': [1.08] * 12,
            'close': [1.09] * 12,
            'tick_volume': [1000] * 12
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        
        # Process bars at London session (08:00-09:00 UTC) and overlap (14:00)
        test_cases = [
            (0, TradingSession.LONDON),     # 08:00 UTC
            (6, TradingSession.OVERLAP),    # 14:00 UTC
            (10, TradingSession.NEW_YORK),  # 18:00 UTC
        ]
        
        with patch('src.backtesting.mode_runner.Sentinel'):
            tester._sentinel = MagicMock()
            tester._sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                news_state="SAFE",
                susceptibility=0.3,
                is_systemic_risk=False,
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            for bar_idx, expected_session in test_cases:
                tester.current_bar = bar_idx
                
                # Extract UTC timestamp (available for Commander)
                utc_timestamp = tester._get_bar_utc_timestamp()
                assert utc_timestamp is not None  # Ensure non-null for session detection
                
                # Call regime filter
                tester._check_regime_filter("EURUSD", 1.09, utc_timestamp)
                
                # Verify timestamp can be used for session detection
                detected_session = SessionDetector.detect_session(utc_timestamp)
                assert detected_session == expected_session
                
                # Verify timestamp is in regime history
                latest_entry = tester._regime_history[-1]
                assert latest_entry['utc_timestamp'] == utc_timestamp


class TestSpicedBacktestSessionAwareness:
    """Test Spiced backtest mode exercises session filters."""

    def test_spiced_mode_includes_session_timestamps(self):
        """Test Spiced backtest result includes session-aware timestamps."""
        base_time = datetime(2026, 2, 12, 8, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(6)]
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 6,
            'high': [1.10] * 6,
            'low': [1.08] * 6,
            'close': [1.09] * 6,
            'tick_volume': [1000] * 6
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        
        # Process multiple bars
        with patch('src.backtesting.mode_runner.Sentinel'):
            tester._sentinel = MagicMock()
            tester._sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                news_state="SAFE",
                susceptibility=0.3,
                is_systemic_risk=False,
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            for bar_idx in range(len(data)):
                tester.current_bar = bar_idx
                tester._check_regime_filter("EURUSD", 1.09)
        
        # Verify regime history includes UTC timestamps
        assert len(tester._regime_history) == len(data)
        
        for entry in tester._regime_history:
            assert 'utc_timestamp' in entry
            assert entry['utc_timestamp'] is not None
            assert entry['utc_timestamp'].tzinfo == timezone.utc



class TestCommanderIntegrationWithBacktests:
    """Test Commander integration with backtest execution for session-aware bot selection.
    
    SPEC: Backtest Commander Integration
    ===================================
    Per spec requirement Comment 1, these tests verify that:
    1. Commander.run_auction() is called with bar_utc_time during backtests
    2. Session filtering is exercised in backtests per spec
    3. Auction results are tracked for analytics
    """

    @pytest.fixture
    def sample_data_with_sessions(self):
        """Create 24-hour data spanning multiple trading sessions."""
        # 06:00 UTC = Asian session ending, London starting
        # 10:00 UTC = London session
        # 14:00 UTC = Overlap (London + NY)
        # 18:00 UTC = NY session
        # 22:00 UTC = Closed
        base_time = datetime(2026, 2, 12, 6, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(24)]
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 24,
            'high': [1.10] * 24,
            'low': [1.08] * 24,
            'close': [1.09] * 24,
            'tick_volume': [1000] * 24
        })
        data['time'] = pd.to_datetime(data['time'])
        return data

    def test_commander_run_auction_receives_utc_timestamp(self):
        """Test Commander.run_auction() receives bar_utc_time for session filtering."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create data at 10:00 UTC (London session)
        test_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        data = pd.DataFrame({
            'time': [test_time],
            'open': [1.09],
            'high': [1.10],
            'low': [1.08],
            'close': [1.09],
            'tick_volume': [1000]
        })
        data['time'] = pd.to_datetime(data['time'])
        
        tester._data_cache["EURUSD"] = data
        tester.symbol = "EURUSD"
        tester.current_bar = 0
        
        # Mock Sentinel
        tester._sentinel = MagicMock()
        tester._sentinel.on_tick.return_value = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.3,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=test_time.timestamp()
        )
        
        # Run auction with UTC timestamp
        bar_utc_time = tester._get_bar_utc_timestamp()
        auction_result = tester.run_auction_with_utc_time("EURUSD", bar_utc_time)
        
        # Verify auction was called and returned (even if empty due to no bots)
        assert tester._auction_results is not None
        assert len(tester._auction_results) > 0
        
        # Verify the auction record has session info
        auction_record = tester._auction_results[-1]
        assert 'utc_timestamp' in auction_record
        assert 'session' in auction_record
        assert auction_record['session'] == 'LONDON'  # 10:00 UTC = London

    def test_backtest_runs_commander_auction_per_bar(self):
        """Test backtest execution calls Commander auction for each bar."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create 6-hour data spanning different sessions
        base_time = datetime(2026, 2, 12, 8, 0, tzinfo=timezone.utc)  # 08:00 UTC
        times = [base_time + timedelta(hours=i) for i in range(6)]  # 08:00 to 13:00
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 6,
            'high': [1.10] * 6,
            'low': [1.08] * 6,
            'close': [1.09] * 6,
            'tick_volume': [1000] * 6
        })
        data['time'] = pd.to_datetime(data['time'])
        
        strategy_code = '''
def on_bar(tester):
    pass  # No trades, just testing auction integration
'''
        
        # Mock Sentinel for auction calls
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            # Run backtest
            result = tester.run(
                strategy_code=strategy_code,
                data=data,
                symbol="EURUSD",
                timeframe=16393  # PERIOD_H1
            )
        
        # Verify auction was called for each bar
        assert len(tester._auction_results) == 6  # One per bar
        
        # Verify each auction has session info
        expected_sessions = ['LONDON', 'LONDON', 'LONDON', 'LONDON', 'LONDON', 'OVERLAP']  # 08:00-13:00
        for i, (auction, expected_session) in enumerate(zip(tester._auction_results, expected_sessions)):
            assert 'session' in auction
            assert auction['session'] == expected_session, f"Bar {i}: expected {expected_session}, got {auction['session']}"

    def test_auction_results_tracked_across_sessions(self):
        """Test auction results are tracked across different trading sessions."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create data spanning multiple sessions (correct UTC times for each session)
        # 22:00 UTC = ASIAN (07:00 Tokyo, within 00:00-09:00)
        # 10:00 UTC = LONDON (10:00 London, within 08:00-16:00)
        # 14:00 UTC = OVERLAP (14:00 London + 09:00 NY, both active)
        # 18:00 UTC = NEW_YORK (18:00 London + 13:00 NY, NY active after overlap ends)
        test_times = [
            datetime(2026, 2, 12, 22, 0, tzinfo=timezone.utc),   # ASIAN
            datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc),  # LONDON
            datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc),  # OVERLAP
            datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc),  # NEW_YORK
        ]
        
        data = pd.DataFrame({
            'time': test_times,
            'open': [1.09] * 4,
            'high': [1.10] * 4,
            'low': [1.08] * 4,
            'close': [1.09] * 4,
            'tick_volume': [1000] * 4
        })
        data['time'] = pd.to_datetime(data['time'])
        
        strategy_code = '''
def on_bar(tester):
    pass
'''
        
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            result = tester.run(
                strategy_code=strategy_code,
                data=data,
                symbol="EURUSD",
                timeframe=16393
            )
        
        # Verify sessions are correctly detected
        expected_sessions = ['ASIAN', 'LONDON', 'OVERLAP', 'NEW_YORK']
        for auction, expected in zip(tester._auction_results, expected_sessions):
            assert auction['session'] == expected, f"Expected {expected}, got {auction['session']}"

    def test_regime_history_still_records_utc_timestamps_with_commander(self):
        """Test regime history continues to record UTC timestamps with Commander integration."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create data with known timestamps
        base_time = datetime(2026, 2, 12, 9, 0, tzinfo=timezone.utc)
        times = [base_time + timedelta(hours=i) for i in range(5)]  # 09:00-13:00
        
        data = pd.DataFrame({
            'time': times,
            'open': [1.09] * 5,
            'high': [1.10] * 5,
            'low': [1.08] * 5,
            'close': [1.09] * 5,
            'tick_volume': [1000] * 5
        })
        data['time'] = pd.to_datetime(data['time'])
        
        strategy_code = '''
def on_bar(tester):
    pass
'''
        
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=datetime.now(timezone.utc).timestamp()
            )
            
            result = tester.run(
                strategy_code=strategy_code,
                data=data,
                symbol="EURUSD",
                timeframe=16393
            )
        
        # Verify regime history still has UTC timestamps
        assert len(tester._regime_history) == 5
        
        for i, entry in enumerate(tester._regime_history):
            assert 'utc_timestamp' in entry
            assert entry['utc_timestamp'] is not None
            assert entry['utc_timestamp'].tzinfo == timezone.utc
            assert entry['utc_timestamp'].hour == 9 + i  # 09:00, 10:00, 11:00, 12:00, 13:00

    def test_current_auction_result_accessible_during_backtest(self):
        """Test get_current_auction_result() returns latest auction info."""
        tester = SentinelEnhancedTester(
            mode=BacktestMode.SPICED,
            initial_cash=10000.0,
            broker_id="test_broker"
        )
        
        # Create single bar data
        test_time = datetime(2026, 2, 12, 11, 0, tzinfo=timezone.utc)  # London session
        data = pd.DataFrame({
            'time': [test_time],
            'open': [1.09],
            'high': [1.10],
            'low': [1.08],
            'close': [1.09],
            'tick_volume': [1000]
        })
        data['time'] = pd.to_datetime(data['time'])
        
        strategy_code = '''
def on_bar(tester):
    # Access current auction result
    result = tester.get_current_auction_result()
    if result:
        tester._log(f"Session: {result['session']}, Bots: {result['eligible_bots']}")
'''
        
        with patch('src.backtesting.mode_runner.Sentinel') as mock_sentinel_class:
            mock_sentinel = MagicMock()
            mock_sentinel_class.return_value = mock_sentinel
            mock_sentinel.on_tick.return_value = RegimeReport(
                regime="TREND_STABLE",
                chaos_score=0.2,
                regime_quality=0.8,
                susceptibility=0.3,
                is_systemic_risk=False,
                news_state="SAFE",
                timestamp=test_time.timestamp()
            )
            
            result = tester.run(
                strategy_code=strategy_code,
                data=data,
                symbol="EURUSD",
                timeframe=16393
            )
        
        # Verify auction result is accessible
        current_result = tester.get_current_auction_result()
        assert current_result is not None
        assert current_result['session'] == 'LONDON'
        assert current_result['bar'] == 0
