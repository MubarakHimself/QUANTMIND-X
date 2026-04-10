"""
QuantMindLib V1 — DependencySpec Schema
CONTRACT-019
Sourced from: docs/planning/quantmindlib/07_shared_object_model.md §18
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DependencySpec(BaseModel):
    """
    Declaration of what a module requires to run — used for dependency
    resolution during composition.

    Mutable for runtime registration at module load time.
    """

    module_id: str = Field(min_length=1)
    requires: List[str] = Field(default_factory=list)
    optional: List[str] = Field(default_factory=list)
    conflict: List[str] = Field(default_factory=list)

    def has_required(self, cap_ids: set[str]) -> bool:
        """Return True if ALL required capabilities are present in cap_ids."""
        return all(req in cap_ids for req in self.requires)

    def has_optional(self, cap_ids: set[str]) -> set[str]:
        """Return the subset of optional capabilities that ARE present in cap_ids."""
        return {opt for opt in self.optional if opt in cap_ids}

    def has_conflict(self, cap_ids: set[str]) -> bool:
        """Return True if ANY conflicting capability is present in cap_ids."""
        return any(conf in cap_ids for conf in self.conflict)

    def missing_required(self, cap_ids: set[str]) -> set[str]:
        """Return the set of required capabilities NOT present in cap_ids."""
        return {req for req in self.requires if req not in cap_ids}


__all__ = ["DependencySpec"]
