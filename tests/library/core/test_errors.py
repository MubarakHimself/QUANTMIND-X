"""Tests for QuantMindLib V1 core error hierarchy."""

import pytest

from src.library.core.errors import (
    BridgeError,
    BridgeUnavailableError,
    ContractValidationError,
    LibraryConfigError,
    LibraryError,
)


class TestLibraryErrorHierarchy:
    """Test the 5 core exception classes."""

    def test_library_error_exists(self):
        """LibraryError is importable."""
        assert LibraryError is not None

    def test_library_error_inherits_from_exception(self):
        """LibraryError inherits from Exception."""
        assert issubclass(LibraryError, Exception)

    def test_library_error_instantiable_with_message(self):
        """LibraryError can be instantiated with a message."""
        err = LibraryError("something went wrong")
        assert str(err) == "something went wrong"
        assert isinstance(err, Exception)

    def test_library_config_error_inherits_from_library_error(self):
        """LibraryConfigError inherits from LibraryError."""
        assert issubclass(LibraryConfigError, LibraryError)

    def test_library_config_error_instantiable(self):
        """LibraryConfigError is instantiatable with a message."""
        err = LibraryConfigError("invalid bot spec")
        assert str(err) == "invalid bot spec"
        assert isinstance(err, LibraryError)

    def test_contract_validation_error_inherits_from_library_error(self):
        """ContractValidationError inherits from LibraryError."""
        assert issubclass(ContractValidationError, LibraryError)

    def test_contract_validation_error_instantiable(self):
        """ContractValidationError is instantiatable with a message."""
        err = ContractValidationError("schema validation failed")
        assert str(err) == "schema validation failed"
        assert isinstance(err, LibraryError)

    def test_bridge_error_inherits_from_library_error(self):
        """BridgeError inherits from LibraryError."""
        assert issubclass(BridgeError, LibraryError)

    def test_bridge_error_instantiable(self):
        """BridgeError is instantiatable with a message."""
        err = BridgeError("bridge layer error")
        assert str(err) == "bridge layer error"
        assert isinstance(err, LibraryError)

    def test_bridge_unavailable_error_inherits_from_bridge_error(self):
        """BridgeUnavailableError inherits from BridgeError."""
        assert issubclass(BridgeUnavailableError, BridgeError)

    def test_bridge_unavailable_error_instantiable(self):
        """BridgeUnavailableError is instantiatable with a message."""
        err = BridgeUnavailableError("registry is down")
        assert str(err) == "registry is down"
        assert isinstance(err, BridgeError)

    def test_bridge_unavailable_error_is_also_instance_of_bridge_error_and_library_error(self):
        """BridgeUnavailableError is an instance of both BridgeError and LibraryError."""
        err = BridgeUnavailableError("DPR engine unavailable")
        assert isinstance(err, BridgeError)
        assert isinstance(err, LibraryError)
        assert isinstance(err, Exception)
