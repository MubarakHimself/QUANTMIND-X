"""
Unit Tests for Volume Profile Indicator
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from svss.indicators.volume_profile import VolumeProfileIndicator, DEFAULT_BUCKET_SIZE


def create_mock_tick(symbol="EURUSD", last=1.0850, volume=100.0):
    """Create a mock tick for testing."""
    tick = MagicMock()
    tick.symbol = symbol
    tick.last = last
    tick.bid = last - 0.0001
    tick.ask = last + 0.0001
    tick.volume = volume
    tick.timestamp = datetime.now(timezone.utc)
    return tick


class TestVolumeProfileIndicator:
    """Tests for VolumeProfileIndicator."""

    def test_initialization(self):
        """Test Volume Profile indicator initialization."""
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session")

        assert indicator.name == "volume_profile"
        assert indicator.get_value() is None
        assert indicator.poc is None
        assert indicator.get_profile() == {}

    def test_volume_profile_accumulates_volume(self):
        """Test volume is accumulated at price levels."""
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session")

        # Tick at price 1.0850
        tick1 = create_mock_tick(last=1.0850, volume=100.0)
        indicator.compute(tick1)

        # Tick at same price level
        tick2 = create_mock_tick(last=1.0850, volume=50.0)
        indicator.compute(tick2)

        profile = indicator.get_profile()
        assert len(profile) == 1  # One price level

        # Get the bucket price
        bucket_price = list(profile.keys())[0]
        assert abs(bucket_price - 1.0850) < DEFAULT_BUCKET_SIZE
        assert profile[bucket_price] == 150.0  # Total volume

    def test_poc_identification(self):
        """Test POC (Point of Control) is correctly identified."""
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session")

        # Most volume at 1.0850
        for _ in range(5):
            tick1 = create_mock_tick(last=1.0850, volume=100.0)
            indicator.compute(tick1)

        # Less volume at 1.0860
        for _ in range(3):
            tick2 = create_mock_tick(last=1.0860, volume=50.0)
            indicator.compute(tick2)

        # POC should be at 1.0850 (highest volume)
        assert indicator.poc is not None
        assert abs(indicator.poc - 1.0850) < DEFAULT_BUCKET_SIZE

    def test_volume_profile_reset(self):
        """Test Volume Profile reset on session boundary."""
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session")

        tick = create_mock_tick(last=1.0850, volume=100.0)
        indicator.compute(tick)

        assert len(indicator.get_profile()) > 0

        indicator.reset()

        assert indicator.get_profile() == {}
        assert indicator.get_value() is None
        assert indicator.poc is None

    def test_multiple_price_levels(self):
        """Test volume profile with multiple price levels."""
        # Use smaller bucket size to distinguish 1.0850-1.0853
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session", bucket_size=0.0001)

        # Different price levels (10 pips apart = should be separate buckets)
        prices = [1.0850, 1.0860, 1.0870, 1.0880]
        for price in prices:
            tick = create_mock_tick(last=price, volume=100.0)
            indicator.compute(tick)

        profile = indicator.get_profile()
        assert len(profile) == 4  # 4 price levels

    def test_result_metadata_contains_profile(self):
        """Test result metadata contains serialized profile."""
        indicator = VolumeProfileIndicator(symbol="EURUSD", session_id="test_session")

        tick = create_mock_tick(last=1.0850, volume=100.0)
        result = indicator.compute(tick)

        assert "profile" in result.metadata
        assert result.metadata["profile"]["poc"] is not None
        assert result.metadata["profile"]["total_levels"] >= 1
