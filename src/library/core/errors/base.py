"""
QuantMindLib V1 error hierarchy — core base classes.

These are the foundation exceptions for all QuantMindLib V1 errors.
Bridge-specific subclasses live near their corresponding bridge files (ERR-002).
AuditRecord and ErrorSeverity are out of scope for this foundation ticket (ERR-003).
"""


class LibraryError(Exception):
    """Base exception for all QuantMindLib V1 errors."""

    pass


class LibraryConfigError(LibraryError):
    """Raised on invalid BotSpec, unknown archetype, or missing required capability at library initialization or bot registration time."""

    pass


class ContractValidationError(LibraryError):
    """Raised on pydantic/schema validation failure in library contracts (BotSpec, FeatureVector, MarketContext)."""

    pass


class BridgeError(LibraryError):
    """Base exception for bridge-layer errors. Subclasses live near their corresponding bridge files."""

    pass


class BridgeUnavailableError(BridgeError):
    """Raised when a bridge cannot reach its target system (registry down, DPR engine unavailable, sentinel unreachable) at decision time."""

    pass
