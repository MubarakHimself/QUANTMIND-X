"""
QuantMindLib V1 error hierarchy — core base classes.
"""

from .audit import AuditRecord
from .base import (
    BridgeError,
    BridgeUnavailableError,
    ContractValidationError,
    LibraryConfigError,
    LibraryError,
)

__all__ = [
    "AuditRecord",
    "LibraryError",
    "LibraryConfigError",
    "ContractValidationError",
    "BridgeError",
    "BridgeUnavailableError",
]
