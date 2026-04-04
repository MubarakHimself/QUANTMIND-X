"""
Unit Tests for VWAP Indicator
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from svss.indicators.vwap import VWAPIndicator


def create_mock_tick(symbol="EURUSD", last=1.0850, bid=1.0849, ask=1.0851, volume=100.0):
    """Create a mock tick for testing."""
    tick = MagicMock()
    tick.symbol = symbol
    tick.last = last
    tick.bid = bid
    tick.ask = ask
    tick.volume = volume
    tick.timestamp = datetime.now(timezone.utc)
    return tick


class TestVWAPIndicator:
    """Tests for VWAPIndicator."""

    def test_initialization(self):
        """Test VWAP indicator initialization."""
        indicator = VWAPIndicator(symbol="EURUSD", session_id="test_session")

        assert indicator.name == "vwap"
        assert indicator.get_value() is None
        assert indicator.cumulative_volume == 0.0
        assert indicator.cumulative_price_volume == 0.0

    def test_vwap_single_tick(self):
        """Test VWAP computation with single tick."""
        indicator = VWAPIndicator(symbol="EURUSD", session_id="test_session")
        tick = create_mock_tick(last=1.0850, volume=100.0)

        result = indicator.compute(tick)

        assert result.name == "vwap"
        assert result.value == 1.0850  # Single tick: price * volume / volume = price
        assert result.session_id == "test_session"
        assert result.metadata["symbol"] == "EURUSD"

    def test_vwap_multiple_ticks(self):
        """Test VWAP with multiple ticks."""
        indicator = VWAPIndicator(symbol="EURUSD", session_id="test_session")

        # Tick 1: price=1.0850, volume=100 -> pv=108.50
        tick1 = create_mock_tick(last=1.0850, volume=100.0)
        indicator.compute(tick1)

        # Tick 2: price=1.0860, volume=200 -> pv=217.20
        tick2 = create_mock_tick(last=1.0860, volume=200.0)
        indicator.compute(tick2)

        # Expected VWAP = (108.50 + 217.20) / (100 + 200) = 325.70 / 300 = 1.08567
        expected = (1.0850 * 100 + 1.0860 * 200) / 300
        assert abs(indicator.get_value() - expected) < 0.0001

    def test_vwap_with_zero_volume_tick(self):
        """Test VWAP ignores zero volume ticks."""
        indicator = VWAPIndicator(symbol="EURUSD", session_id="test_session")

        # First tick with volume
        tick1 = create_mock_tick(last=1.0850, volume=100.0)
        indicator.compute(tick1)

        # Zero volume tick
        tick2 = create_mock_tick(last=1.0900, volume=0.0)
        result = indicator.compute(tick2)

        # Should keep previous VWAP value
        assert result.value == 1.0850

    def test_vwap_reset(self):
        """Test VWAP reset on session boundary."""
        indicator = VWAPIndicator(symbol="EURUSD", session_id="test_session")

        # Accumulate some volume
        tick1 = create_mock_tick(last=1.0850, volume=100.0)
        indicator.compute(tick1)
        assert indicator.cumulative_volume == 100.0

        # Reset
        indicator.reset()

        assert indicator.cumulative_volume == 0.0
        assert indicator.cumulative_price_volume == 0.0
        assert indicator.get_value() is None
