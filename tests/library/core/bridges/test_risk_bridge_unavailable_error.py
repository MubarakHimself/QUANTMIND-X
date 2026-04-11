"""Tests for QuantMindLib V1 — RiskBridgeUnavailableError."""

import pytest

from src.library.core.bridges.risk_execution_bridges import (
    RiskBridgeUnavailableError,
)
from src.library.core.errors import BridgeError, LibraryError


class TestRiskBridgeUnavailableError:
    """Tests for RiskBridgeUnavailableError."""

    def test_is_subclass_of_bridge_error(self):
        """RiskBridgeUnavailableError inherits from BridgeError."""
        assert issubclass(RiskBridgeUnavailableError, BridgeError)

    def test_is_subclass_of_library_error(self):
        """RiskBridgeUnavailableError inherits from LibraryError."""
        assert issubclass(RiskBridgeUnavailableError, LibraryError)

    def test_can_be_raised_and_caught(self):
        """The error can be raised and caught as RiskBridgeUnavailableError."""
        with pytest.raises(RiskBridgeUnavailableError):
            raise RiskBridgeUnavailableError("Risk bridge unavailable")

    def test_can_be_caught_as_bridge_error(self):
        """The error can be caught as a BridgeError."""
        with pytest.raises(BridgeError):
            raise RiskBridgeUnavailableError("test message")

    def test_can_be_caught_as_library_error(self):
        """The error can be caught as a LibraryError."""
        with pytest.raises(LibraryError):
            raise RiskBridgeUnavailableError("test message")

    def test_error_message_is_preserved(self):
        """The error message is preserved when the exception is raised."""
        msg = "Risk/governor system unavailable at decision time"
        try:
            raise RiskBridgeUnavailableError(msg)
        except RiskBridgeUnavailableError as e:
            assert str(e) == msg

    def test_can_be_raised_with_dynamic_context(self):
        """The error can include dynamic context in its message."""
        bot_id = "orb_v1"
        msg = f"Risk bridge unavailable for bot {bot_id}"
        with pytest.raises(RiskBridgeUnavailableError, match=bot_id):
            raise RiskBridgeUnavailableError(msg)