"""
Unit Tests for RVOL Indicator
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from svss.indicators.rvol import RVOLIndicator, DEFAULT_RVOL


def create_mock_tick(symbol="EURUSD", volume=100.0, hour=8, minute=30):
    """Create a mock tick for testing."""
    tick = MagicMock()
    tick.symbol = symbol
    tick.last = 1.0850
    tick.bid = 1.0849
    tick.ask = 1.0851
    tick.volume = volume
    tick.timestamp = datetime(2026, 3, 25, hour, minute, 0, tzinfo=timezone.utc)
    return tick


class TestRVOLIndicator:
    """Tests for RVOLIndicator."""

    def test_initialization(self):
        """Test RVOL indicator initialization."""
        indicator = RVOLIndicator(symbol="EURUSD", session_id="test_session")

        assert indicator.name == "rvvol"
        assert indicator.get_value() is None
        assert indicator.current_bar_volume == 0.0

    def test_rvol_with_rolling_average(self):
        """Test RVOL computation with rolling average available."""
        # Set up rolling average profile: 8:30 -> 100 lots
        rolling_avg = {8 * 60 + 30: 100.0}  # 510 minutes = 8:30

        indicator = RVOLIndicator(
            symbol="EURUSD",
            session_id="test_session",
            rolling_avg_volume_profile=rolling_avg
        )

        # Current tick: 125 lots at 8:30 -> RVOL = 125 / 100 = 1.25
        tick = create_mock_tick(volume=125.0, hour=8, minute=30)
        result = indicator.compute(tick)

        assert result.name == "rvvol"
        assert abs(result.value - 1.25) < 0.001
        assert result.metadata["current_volume"] == 125.0
        assert result.metadata["rolling_avg"] == 100.0

    def test_rvol_default_when_no_historical_data(self):
        """Test RVOL returns default when no rolling average available."""
        indicator = RVOLIndicator(symbol="EURUSD", session_id="test_session")

        # No rolling average set
        tick = create_mock_tick(volume=100.0, hour=8, minute=30)
        result = indicator.compute(tick)

        assert result.value == DEFAULT_RVOL

    def test_rvol_default_when_rolling_avg_is_zero(self):
        """Test RVOL returns default when rolling average is zero."""
        rolling_avg = {8 * 60 + 30: 0.0}  # Zero average

        indicator = RVOLIndicator(
            symbol="EURUSD",
            session_id="test_session",
            rolling_avg_volume_profile=rolling_avg
        )

        tick = create_mock_tick(volume=100.0, hour=8, minute=30)
        result = indicator.compute(tick)

        assert result.value == DEFAULT_RVOL

    def test_rvol_time_of_day_bucketing(self):
        """Test RVOL uses minute-of-day bucketing."""
        rolling_avg = {510: 100.0}  # 8:30 = 510 minutes

        indicator = RVOLIndicator(
            symbol="EURUSD",
            session_id="test_session",
            rolling_avg_volume_profile=rolling_avg
        )

        # 8:30:15 should still bucket to 8:30
        tick = create_mock_tick(volume=150.0, hour=8, minute=30)
        result = indicator.compute(tick)

        # Volume 150 / avg 100 = 1.5
        assert abs(result.value - 1.5) < 0.001

    def test_rvol_reset(self):
        """Test RVOL reset on session boundary."""
        rolling_avg = {510: 100.0}
        indicator = RVOLIndicator(
            symbol="EURUSD",
            session_id="old_session",
            rolling_avg_volume_profile=rolling_avg
        )

        tick = create_mock_tick(volume=100.0, hour=8, minute=30)
        indicator.compute(tick)
        assert indicator.current_bar_volume == 100.0

        # Reset with new session
        indicator.reset(new_session_id="new_session")

        assert indicator.current_bar_volume == 0.0
        assert indicator.get_value() is None

    def test_set_rolling_avg_profile(self):
        """Test setting rolling average profile."""
        indicator = RVOLIndicator(symbol="EURUSD", session_id="test_session")

        new_profile = {8 * 60 + 30: 200.0, 9 * 60 + 0: 150.0}
        indicator.set_rolling_avg_profile(new_profile)

        # Profile should be updated
        tick = create_mock_tick(volume=100.0, hour=8, minute=30)
        result = indicator.compute(tick)

        # 100 / 200 = 0.5
        assert abs(result.value - 0.5) < 0.001

    def test_rvol_bar_detection_minute_change(self):
        """Test RVOL resets bar volume when minute changes."""
        rolling_avg = {510: 100.0, 511: 150.0}  # 8:30 and 8:31

        indicator = RVOLIndicator(
            symbol="EURUSD",
            session_id="test_session",
            rolling_avg_volume_profile=rolling_avg
        )

        # First tick at 8:30
        tick1 = create_mock_tick(volume=100.0, hour=8, minute=30)
        indicator.compute(tick1)
        assert indicator.current_bar_volume == 100.0
        assert abs(indicator.get_value() - 1.0) < 0.001  # 100/100

        # Second tick at 8:30 (same minute, accumulating)
        tick2 = create_mock_tick(volume=50.0, hour=8, minute=30)
        indicator.compute(tick2)
        assert indicator.current_bar_volume == 150.0  # 100 + 50
        assert abs(indicator.get_value() - 1.5) < 0.001  # 150/100

        # Third tick at 8:31 (minute changed - bar should reset)
        tick3 = create_mock_tick(volume=150.0, hour=8, minute=31)
        indicator.compute(tick3)
        assert indicator.current_bar_volume == 150.0  # Reset! Now just 150
        assert abs(indicator.get_value() - 1.0) < 0.001  # 150/150
