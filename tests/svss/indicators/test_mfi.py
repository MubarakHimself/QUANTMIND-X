"""
Unit Tests for MFI Indicator
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from svss.indicators.mfi import MFIIndicator, MFI_PERIOD


def create_mock_tick(symbol="EURUSD", bid=1.0849, ask=1.0851, last=1.0850, volume=100.0):
    """Create a mock tick for testing."""
    tick = MagicMock()
    tick.symbol = symbol
    tick.bid = bid
    tick.ask = ask
    tick.last = last
    tick.volume = volume
    tick.timestamp = datetime.now(timezone.utc)
    return tick


class TestMFIIndicator:
    """Tests for MFIIndicator."""

    def test_initialization(self):
        """Test MFI indicator initialization."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        assert indicator.name == "mfi"
        assert indicator.get_value() is None
        assert indicator._period == MFI_PERIOD

    def test_mfi_neutral_default(self):
        """Test MFI returns neutral value when not enough data."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        # Less than MFI_PERIOD ticks
        for i in range(5):
            tick = create_mock_tick(last=1.0850 + i * 0.0001, volume=100.0)
            result = indicator.compute(tick)

        # Should return default 50 (neutral)
        assert result.value == 50.0

    def test_mfi_overbought_condition(self):
        """Test MFI indicates overbought when price rising."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        # Rising prices (positive money flow)
        for i in range(MFI_PERIOD):
            tick = create_mock_tick(
                bid=1.0850 + i * 0.0001,
                ask=1.0852 + i * 0.0001,
                last=1.0851 + i * 0.0001,
                volume=100.0
            )
            indicator.compute(tick)

        # MFI should be high (overbought)
        mfi_value = indicator.get_value()
        assert mfi_value is not None
        assert mfi_value > 50.0

    def test_mfi_oversold_condition(self):
        """Test MFI indicates oversold when price falling."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        # Falling prices (negative money flow)
        for i in range(MFI_PERIOD):
            tick = create_mock_tick(
                bid=1.0850 - i * 0.0001,
                ask=1.0852 - i * 0.0001,
                last=1.0851 - i * 0.0001,
                volume=100.0
            )
            indicator.compute(tick)

        # MFI should be low (oversold)
        mfi_value = indicator.get_value()
        assert mfi_value is not None
        assert mfi_value < 50.0

    def test_mfi_reset(self):
        """Test MFI reset on session boundary."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        # Accumulate some data
        for i in range(MFI_PERIOD):
            tick = create_mock_tick(last=1.0850 + i * 0.0001, volume=100.0)
            indicator.compute(tick)

        assert indicator.get_value() is not None

        indicator.reset()

        assert indicator.get_value() is None
        assert len(indicator._typical_prices) == 0
        assert len(indicator._money_flows) == 0
        assert indicator._positive_flow == 0.0
        assert indicator._negative_flow == 0.0

    def test_mfi_value_range(self):
        """Test MFI values are clamped to 0-100 range."""
        indicator = MFIIndicator(symbol="EURUSD", session_id="test_session")

        # Enough ticks to compute MFI
        for i in range(MFI_PERIOD + 5):
            tick = create_mock_tick(last=1.0850 + i * 0.0001, volume=100.0)
            result = indicator.compute(tick)

        mfi_value = indicator.get_value()
        assert mfi_value is not None
        assert 0.0 <= mfi_value <= 100.0
