"""
QuantMindLib V1 — AuditRecord schema.

Minimal diagnostic/journal entry for library operations.
Out of scope for V1: emission in every bridge and registry path.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AuditRecord:
    """Structured diagnostic/journal entry for library operations."""

    component: str  # "FeatureRegistry", "RiskBridge", "RegistryBridge"
    operation: str  # "register", "validate_composition", "authorize"
    outcome: str  # "success" | "rejected" | "failed" | "blocked"
    reason: Optional[str]  # Plain string rejection/failure message
    details: Optional[Dict[str, Any]]  # Structured context (bot_id, symbol, etc.)
    timestamp_ms: int  # Unix milliseconds


__all__ = ["AuditRecord"]
