"""Tests for QuantMindLib V1 — SentinelBridgeUnavailableError."""

import pytest

from src.library.core.bridges.sentinel_dpr_bridges import (
    SentinelBridge,
    SentinelBridgeUnavailableError,
)
from src.library.core.errors import BridgeError, LibraryError


class TestSentinelBridgeUnavailableError:
    """Tests for SentinelBridgeUnavailableError."""

    def test_is_subclass_of_bridge_error(self):
        """SentinelBridgeUnavailableError inherits from BridgeError."""
        assert issubclass(SentinelBridgeUnavailableError, BridgeError)

    def test_is_subclass_of_library_error(self):
        """SentinelBridgeUnavailableError inherits from LibraryError."""
        assert issubclass(SentinelBridgeUnavailableError, LibraryError)

    def test_can_be_raised_and_caught(self):
        """The error can be raised and caught as SentinelBridgeUnavailableError."""
        with pytest.raises(SentinelBridgeUnavailableError):
            raise SentinelBridgeUnavailableError("Sentinel unavailable at tick time")

    def test_can_be_caught_as_bridge_error(self):
        """The error can be caught as a BridgeError."""
        with pytest.raises(BridgeError):
            raise SentinelBridgeUnavailableError("test message")

    def test_can_be_caught_as_library_error(self):
        """The error can be caught as a LibraryError."""
        with pytest.raises(LibraryError):
            raise SentinelBridgeUnavailableError("test message")

    def test_error_message_is_preserved(self):
        """The error message is preserved when the exception is raised."""
        msg = "Sentinel bridge cannot reach market intelligence system"
        try:
            raise SentinelBridgeUnavailableError(msg)
        except SentinelBridgeUnavailableError as e:
            assert str(e) == msg

    def test_can_be_raised_with_dynamic_context(self):
        """The error can include dynamic context in its message."""
        regime = "BREAKOUT_PRIME"
        msg = f"Sentinel unreachable during regime {regime}"
        with pytest.raises(SentinelBridgeUnavailableError, match=regime):
            raise SentinelBridgeUnavailableError(msg)