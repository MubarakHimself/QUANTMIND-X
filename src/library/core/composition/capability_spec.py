"""
QuantMindLib V1 — CapabilitySpec Schema
CONTRACT-018
Sourced from: docs/planning/quantmindlib/07_shared_object_model.md §17
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class CapabilitySpec(BaseModel):
    """
    Declaration of what a module needs and what it produces — enables
    automatic composition validation.

    Mutable for runtime registration at module load time.
    """

    module_id: str = Field(min_length=1)
    provides: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    compatibility: List[str] = Field(default_factory=list)
    sync_mode: bool = Field(default=True)

    def provides_capability(self, cap_id: str) -> bool:
        """Return True if this spec provides the given capability ID."""
        return cap_id in self.provides

    def requires_capability(self, cap_id: str) -> bool:
        """Return True if this spec requires the given capability ID."""
        return cap_id in self.requires

    def is_compatible_with(self, archetype: str) -> bool:
        """Return True if this spec is compatible with the given archetype."""
        return archetype in self.compatibility


__all__ = ["CapabilitySpec"]
