"""Tests for QuantMindLib V1 — RegistryBridgeUnavailableError."""

import pytest

from src.library.core.bridges.registry_journal_bridges import (
    RegistryBridge,
    RegistryBridgeUnavailableError,
)
from src.library.core.errors import BridgeError, LibraryError


class TestRegistryBridgeUnavailableError:
    """Tests for RegistryBridgeUnavailableError."""

    def test_is_subclass_of_bridge_error(self):
        """RegistryBridgeUnavailableError inherits from BridgeError."""
        assert issubclass(RegistryBridgeUnavailableError, BridgeError)

    def test_is_subclass_of_library_error(self):
        """RegistryBridgeUnavailableError inherits from LibraryError."""
        assert issubclass(RegistryBridgeUnavailableError, LibraryError)

    def test_can_be_raised_and_caught(self):
        """The error can be raised and caught as a BridgeError."""
        with pytest.raises(RegistryBridgeUnavailableError):
            raise RegistryBridgeUnavailableError("Registry unavailable at decision time")

    def test_can_be_caught_as_bridge_error(self):
        """The error can be caught as a BridgeError."""
        with pytest.raises(BridgeError):
            raise RegistryBridgeUnavailableError("test message")

    def test_can_be_caught_as_library_error(self):
        """The error can be caught as a LibraryError."""
        with pytest.raises(LibraryError):
            raise RegistryBridgeUnavailableError("test message")

    def test_error_message_is_preserved(self):
        """The error message is preserved when the exception is raised."""
        msg = "Registry unreachable for bot registration check"
        try:
            raise RegistryBridgeUnavailableError(msg)
        except RegistryBridgeUnavailableError as e:
            assert str(e) == msg

    def test_can_be_raised_with_format_string(self):
        """The error can include dynamic context in its message."""
        bot_id = "scalper_bot_001"
        msg = f"Registry unavailable for bot {bot_id} at decision time"
        with pytest.raises(RegistryBridgeUnavailableError, match=bot_id):
            raise RegistryBridgeUnavailableError(msg)