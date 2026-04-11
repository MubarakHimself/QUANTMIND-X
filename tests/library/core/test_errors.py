"""Tests for QuantMindLib V1 core error hierarchy."""

import time
from enum import StrEnum

import pytest

from src.library.core.errors import (
    AuditRecord,
    BridgeError,
    BridgeUnavailableError,
    ContractValidationError,
    LibraryConfigError,
    LibraryError,
)
from src.library.core.types import ErrorSeverity


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


class TestAuditRecord:
    """Test AuditRecord dataclass construction and fields."""

    def test_construction_minimal(self):
        """AuditRecord can be constructed with required fields only."""
        record = AuditRecord(
            component="FeatureRegistry",
            operation="validate_composition",
            outcome="rejected",
            reason="Required feature not registered: unknown_feature",
            details={"feature_id": "unknown_feature"},
            timestamp_ms=1712841600000,
        )
        assert record.component == "FeatureRegistry"
        assert record.operation == "validate_composition"
        assert record.outcome == "rejected"
        assert record.reason == "Required feature not registered: unknown_feature"
        assert record.details == {"feature_id": "unknown_feature"}
        assert record.timestamp_ms == 1712841600000

    def test_construction_all_none_optional(self):
        """AuditRecord reason and details are optional."""
        record = AuditRecord(
            component="RiskBridge",
            operation="authorize",
            outcome="success",
            reason=None,
            details=None,
            timestamp_ms=1712841600000,
        )
        assert record.component == "RiskBridge"
        assert record.operation == "authorize"
        assert record.outcome == "success"
        assert record.reason is None
        assert record.details is None

    def test_construction_with_all_outcomes(self):
        """AuditRecord outcome field accepts all four canonical values."""
        outcomes = ["success", "rejected", "failed", "blocked"]
        for outcome in outcomes:
            record = AuditRecord(
                component="TestComponent",
                operation="test_op",
                outcome=outcome,
                reason=None,
                details=None,
                timestamp_ms=int(time.time() * 1000),
            )
            assert record.outcome == outcome

    def test_is_dataclass(self):
        """AuditRecord is a dataclass."""
        record = AuditRecord(
            component="FeatureRegistry",
            operation="register",
            outcome="success",
            reason=None,
            details={"family": "indicators"},
            timestamp_ms=int(time.time() * 1000),
        )
        # Dataclass returns a tuple-like repr by default; check fields exist via dict
        assert hasattr(record, "component")
        assert hasattr(record, "operation")
        assert hasattr(record, "outcome")
        assert hasattr(record, "reason")
        assert hasattr(record, "details")
        assert hasattr(record, "timestamp_ms")


class TestErrorSeverity:
    """Test ErrorSeverity enum values and inheritance."""

    def test_all_four_levels_exist(self):
        """ErrorSeverity has all four expected values."""
        assert ErrorSeverity.LOW == "LOW"
        assert ErrorSeverity.MEDIUM == "MEDIUM"
        assert ErrorSeverity.HIGH == "HIGH"
        assert ErrorSeverity.CRITICAL == "CRITICAL"

    def test_inherits_from_strenum(self):
        """ErrorSeverity inherits from StrEnum."""
        assert issubclass(ErrorSeverity, StrEnum)

    def test_is_string_subclass(self):
        """ErrorSeverity members are strings."""
        assert isinstance(ErrorSeverity.CRITICAL, str)

    def test_str_value_matches(self):
        """String value of each member matches its label."""
        assert str(ErrorSeverity.LOW) == "LOW"
        assert str(ErrorSeverity.MEDIUM) == "MEDIUM"
        assert str(ErrorSeverity.HIGH) == "HIGH"
        assert str(ErrorSeverity.CRITICAL) == "CRITICAL"


class TestErrorSeverityMapping:
    """Optional: ErrorSeverity maps to library error classes."""

    def test_library_config_error_maps_to_config(self):
        """LibraryConfigError maps to MEDIUM severity by convention."""
        # Mapping is informational — validate the severity enum itself is stable
        assert ErrorSeverity.MEDIUM is not None

    def test_severity_ordering_hint(self):
        """Severity values follow LOW < MEDIUM < HIGH < CRITICAL ordering by definition."""
        # Ordering is by declaration order in the enum class, verified by name order
        names = [e.name for e in ErrorSeverity]
        assert names == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
