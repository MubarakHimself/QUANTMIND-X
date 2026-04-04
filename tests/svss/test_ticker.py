"""
Unit Tests for SVSSTicker

Tests MT5 ZMQ tick subscription handler.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from svss.ticker import SVSSTicker, TickData


class TestTickData:
    """Tests for TickData dataclass."""

    def test_initialization(self):
        """Test TickData initialization."""
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert tick.symbol == "EURUSD"
        assert tick.bid == 1.0850
        assert tick.ask == 1.0852
        assert tick.last == 1.0851
        assert tick.volume == 100.0

    def test_typical_price(self):
        """Test typical price calculation."""
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
        )

        # typical_price = (high + low + close) / 3
        # high = ask = 1.0852, low = bid = 1.0850, close = last = 1.0851
        expected = (1.0852 + 1.0850 + 1.0851) / 3
        assert abs(tick.typical_price - expected) < 0.0001

    def test_spread_in_pips(self):
        """Test spread calculation in pips."""
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
        )

        # spread = (ask - bid) * 10000 = (1.0852 - 1.0850) * 10000 = 2 pips
        assert abs(tick.spread - 2.0) < 0.001

    def test_spread_large(self):
        """Test spread with larger difference."""
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0860,
            last=1.0855,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
        )

        # spread = (1.0860 - 1.0850) * 10000 = 10 pips
        assert abs(tick.spread - 10.0) < 0.001

    def test_raw_data_optional(self):
        """Test that raw_data is optional."""
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert tick.raw_data is None

    def test_raw_data_with_dict(self):
        """Test TickData with raw_data."""
        raw = {"original": "data"}
        tick = TickData(
            symbol="EURUSD",
            bid=1.0850,
            ask=1.0852,
            last=1.0851,
            volume=100.0,
            timestamp=datetime.now(timezone.utc),
            raw_data=raw,
        )

        assert tick.raw_data == raw


class TestSVSSTicker:
    """Tests for SVSSTicker."""

    def test_initialization_with_defaults(self):
        """Test SVSSTicker initialization with defaults."""
        ticker = SVSSTicker()

        assert ticker._zmq_endpoint == "tcp://localhost:5555"
        assert ticker._subscription_filter == "TICK"
        assert ticker._connected is False
        assert ticker._context is None
        assert ticker._socket is None

    def test_initialization_with_custom_values(self):
        """Test SVSSTicker with custom endpoint."""
        ticker = SVSSTicker(
            zmq_endpoint="tcp://custom:5555",
            subscription_filter="CUSTOM",
        )

        assert ticker._zmq_endpoint == "tcp://custom:5555"
        assert ticker._subscription_filter == "CUSTOM"

    def test_is_connected_initially_false(self):
        """Test is_connected returns False initially."""
        ticker = SVSSTicker()
        assert ticker.is_connected is False

    def test_set_tick_callback(self):
        """Test setting tick callback."""
        ticker = SVSSTicker()
        callback = MagicMock()

        ticker.set_tick_callback(callback)

        assert ticker._tick_callback == callback

    def test_parse_tick_valid_message(self):
        """Test parsing valid tick message."""
        ticker = SVSSTicker()
        message = 'TICK {"symbol": "EURUSD", "bid": 1.0850, "ask": 1.0852, "last": 1.0851, "volume": 100, "timestamp": 1711362600000}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        assert tick.symbol == "EURUSD"
        assert tick.bid == 1.0850
        assert tick.ask == 1.0852
        assert tick.last == 1.0851
        assert tick.volume == 100.0

    def test_parse_tick_without_filter_prefix(self):
        """Test parsing tick message without filter prefix."""
        ticker = SVSSTicker()
        message = '{"symbol": "EURUSD", "bid": 1.0850, "ask": 1.0852, "last": 1.0851, "volume": 100}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        assert tick.symbol == "EURUSD"

    def test_parse_tick_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        ticker = SVSSTicker()
        message = "INVALID JSON"

        tick = ticker._parse_tick(message)

        assert tick is None

    def test_parse_tick_missing_fields(self):
        """Test parsing tick with missing fields uses defaults."""
        ticker = SVSSTicker()
        message = '{"symbol": "EURUSD"}'  # Missing bid, ask, etc.

        tick = ticker._parse_tick(message)

        # Implementation uses defaults (0.0) when fields are missing
        assert tick is not None
        assert tick.symbol == "EURUSD"
        assert tick.bid == 0.0
        assert tick.ask == 0.0

    def test_parse_tick_missing_symbol(self):
        """Test parsing tick with missing symbol uses empty string."""
        ticker = SVSSTicker()
        message = '{"bid": 1.0850, "ask": 1.0852, "last": 1.0851, "volume": 100}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        assert tick.symbol == ""

    def test_parse_tick_with_zero_timestamp(self):
        """Test parsing tick with zero timestamp."""
        ticker = SVSSTicker()
        message = '{"symbol": "EURUSD", "bid": 1.0850, "ask": 1.0852, "last": 1.0851, "volume": 100, "timestamp": 0}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        # Should use current time when timestamp is 0

    def test_parse_tick_uses_last_as_close(self):
        """Test that tick uses last price as close."""
        ticker = SVSSTicker()
        message = '{"symbol": "EURUSD", "bid": 1.0850, "ask": 1.0852, "last": 1.0855, "volume": 100}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        assert tick.last == 1.0855

    def test_parse_tick_symbol_uppercase(self):
        """Test that symbol is normalized to uppercase."""
        ticker = SVSSTicker()
        message = '{"symbol": "eurusd", "bid": 1.0850, "ask": 1.0852, "last": 1.0851, "volume": 100}'

        tick = ticker._parse_tick(message)

        assert tick is not None
        assert tick.symbol == "EURUSD"

    def test_disconnect_when_not_connected(self):
        """Test disconnect when not connected does nothing."""
        ticker = SVSSTicker()
        ticker.disconnect()  # Should not raise

        assert ticker._connected is False
        assert ticker._socket is None
        assert ticker._context is None

    def test_poll_when_not_connected(self):
        """Test poll returns None when not connected."""
        ticker = SVSSTicker()

        result = ticker.poll()

        assert result is None

    def test_poll_returns_none_when_no_message(self):
        """Test poll returns None when no message available."""
        import zmq

        ticker = SVSSTicker()
        ticker._connected = True
        ticker._socket = MagicMock()
        ticker._socket.recv_string.side_effect = zmq.Again()

        result = ticker.poll()

        assert result is None

    def test_reconnect(self):
        """Test reconnect method."""
        ticker = SVSSTicker()

        with patch.object(ticker, 'disconnect') as mock_disconnect, \
             patch.object(ticker, 'connect', return_value=True) as mock_connect:
            result = ticker.reconnect()

            mock_disconnect.assert_called_once()
            mock_connect.assert_called_once()
            assert result is True

    def test_reconnect_failure(self):
        """Test reconnect failure."""
        ticker = SVSSTicker()

        with patch.object(ticker, 'disconnect'), \
             patch.object(ticker, 'connect', return_value=False):
            result = ticker.reconnect()

            assert result is False
