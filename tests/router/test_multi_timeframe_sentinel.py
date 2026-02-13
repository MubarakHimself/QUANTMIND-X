"""
Unit tests for MultiTimeframeSentinel

Tests tick-to-bar aggregation, regime updates per timeframe,
and dominant regime logic.
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.router.multi_timeframe_sentinel import (
    MultiTimeframeSentinel,
    Timeframe,
    TickAggregator,
    OHLCBar
)


class TestTickAggregator:
    """Tests for TickAggregator class."""
    
    def test_aggregator_creates_bar_on_first_tick(self):
        """Test that first tick creates a new bar."""
        aggregator = TickAggregator(Timeframe.M1)
        timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        result = aggregator.on_tick(1.1000, timestamp)
        
        # First tick should not return a completed bar
        assert result is None
        assert aggregator.current_bar is not None
        assert aggregator.current_bar.open == 1.1000
        assert aggregator.current_bar.close == 1.1000
    
    def test_aggregator_updates_bar_on_same_timeframe(self):
        """Test that ticks within same timeframe update the bar."""
        aggregator = TickAggregator(Timeframe.M1)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # First tick
        aggregator.on_tick(1.1000, base_time)
        
        # Second tick in same bar
        result = aggregator.on_tick(1.1005, base_time + timedelta(seconds=30))
        
        # Should not complete bar yet
        assert result is None
        assert aggregator.current_bar.high == 1.1005
        assert aggregator.current_bar.low == 1.1000
        assert aggregator.current_bar.close == 1.1005
    
    def test_aggregator_completes_bar_on_timeframe_boundary(self):
        """Test that bar completes when timeframe boundary is crossed."""
        aggregator = TickAggregator(Timeframe.M1)
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # First bar
        aggregator.on_tick(1.1000, base_time)
        aggregator.on_tick(1.1005, base_time + timedelta(seconds=30))
        
        # Cross to next minute
        next_minute = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        completed_bar = aggregator.on_tick(1.1010, next_minute)
        
        # Should return completed bar
        assert completed_bar is not None
        assert completed_bar.close == 1.1005
        assert completed_bar.timestamp == base_time
        
        # New bar should be started
        assert aggregator.current_bar is not None
        assert aggregator.current_bar.open == 1.1010
        assert aggregator.current_bar.timestamp == next_minute


class TestMultiTimeframeSentinel:
    """Tests for MultiTimeframeSentinel class."""
    
    def test_sentinel_initialization(self):
        """Test that sentinel initializes with correct timeframes."""
        sentinel = MultiTimeframeSentinel(timeframes=[Timeframe.M5, Timeframe.H1])
        
        assert len(sentinel.timeframes) == 2
        assert Timeframe.M5 in sentinel.aggregators
        assert Timeframe.H1 in sentinel.aggregators
        assert Timeframe.M5 in sentinel.sentinels
        assert Timeframe.H1 in sentinel.sentinels
    
    def test_sentinel_processes_tick(self):
        """Test that sentinel processes tick and updates regimes."""
        sentinel = MultiTimeframeSentinel(timeframes=[Timeframe.M5])
        symbol = "EURUSD"
        price = 1.1000
        timestamp = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # First tick should not complete any bar
        result = sentinel.on_tick(symbol, price, timestamp)
        assert result == {}  # No completed bars yet
        
        # Get regimes - should be empty initially
        regimes = sentinel.get_all_regimes()
        assert len(regimes) == 0
    
    def test_sentinel_bar_completion_updates_regime(self):
        """Test that bar completion triggers regime update."""
        sentinel = MultiTimeframeSentinel(timeframes=[Timeframe.M5])
        symbol = "EURUSD"
        
        # Send multiple ticks to complete a bar
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        # First bar
        for i in range(5):
            timestamp = base_time + timedelta(seconds=i*60)
            sentinel.on_tick(symbol, 1.1000 + i*0.0001, timestamp)
        
        # Now cross to next M5 bar
        next_bar_time = datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        result = sentinel.on_tick(symbol, 1.1005, next_bar_time)
        
        # Bar should be completed
        assert len(result) > 0
        assert Timeframe.M5 in result
        
        # Regime should be updated
        regimes = sentinel.get_all_regimes()
        assert Timeframe.M5 in regimes
        assert regimes[Timeframe.M5] is not None
    
    def test_get_dominant_regime_with_no_regimes(self):
        """Test dominant regime returns UNKNOWN when no regimes available."""
        sentinel = MultiTimeframeSentinel()
        
        dominant = sentinel.get_dominant_regime()
        
        assert dominant == "UNKNOWN"
    
    def test_get_dominant_regime_with_single_regime(self):
        """Test dominant regime returns single regime when only one available."""
        sentinel = MultiTimeframeSentinel(timeframes=[Timeframe.M5])
        symbol = "EURUSD"
        
        # Complete a bar to have a regime
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            timestamp = base_time + timedelta(seconds=i*60)
            sentinel.on_tick(symbol, 1.1000 + i*0.0001, timestamp)
        
        next_bar_time = datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        sentinel.on_tick(symbol, 1.1005, next_bar_time)
        
        dominant = sentinel.get_dominant_regime()
        
        # Should return the regime (not UNKNOWN)
        assert dominant != "UNKNOWN"
    
    def test_get_dominant_regime_voting(self):
        """Test dominant regime uses voting logic."""
        # This test verifies the voting mechanism works
        sentinel = MultiTimeframeSentinel(timeframes=[Timeframe.M5, Timeframe.H1, Timeframe.H4])
        
        # Manually set some regime reports for voting
        from src.router.sentinel import RegimeReport
        
        # All same regime - should return that regime
        sentinel.regime_reports[Timeframe.M5] = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=0.0
        )
        sentinel.regime_reports[Timeframe.H1] = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.3,
            regime_quality=0.7,
            susceptibility=0.2,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=0.0
        )
        sentinel.regime_reports[Timeframe.H4] = RegimeReport(
            regime="RANGE_STABLE",
            chaos_score=0.25,
            regime_quality=0.75,
            susceptibility=0.15,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=0.0
        )
        
        dominant = sentinel.get_dominant_regime()
        
        # TREND_STABLE has 2 votes, RANGE_STABLE has 1
        assert dominant == "TREND_STABLE"


class TestTimeframeEnum:
    """Tests for Timeframe enum."""
    
    def test_timeframe_seconds(self):
        """Test timeframe seconds property."""
        assert Timeframe.M1.seconds == 60
        assert Timeframe.M5.seconds == 300
        assert Timeframe.M15.seconds == 900
        assert Timeframe.H1.seconds == 3600
        assert Timeframe.H4.seconds == 14400
        assert Timeframe.D1.seconds == 86400
    
    def test_timeframe_from_mql5(self):
        """Test MQL5 timeframe conversion."""
        assert Timeframe.from_mql5_timeframe(1) == Timeframe.M1
        assert Timeframe.from_mql5_timeframe(5) == Timeframe.M5
        assert Timeframe.from_mql5_timeframe(15) == Timeframe.M15
        assert Timeframe.from_mql5_timeframe(60) == Timeframe.H1
        assert Timeframe.from_mql5_timeframe(240) == Timeframe.H4
        assert Timeframe.from_mql5_timeframe(1440) == Timeframe.D1
    
    def test_timeframe_pandas_freq(self):
        """Test pandas frequency conversion."""
        assert Timeframe.M1.to_pandas_freq() == '1min'
        assert Timeframe.M5.to_pandas_freq() == '5min'
        assert Timeframe.H1.to_pandas_freq() == '1H'
        assert Timeframe.H4.to_pandas_freq() == '4H'
        assert Timeframe.D1.to_pandas_freq() == '1D'
