"""
QuantMindLib V1 — ArchetypeSpec
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.library.archetypes.constraints import ConstraintSpec


class ArchetypeSpec(BaseModel):
    """
    Frozen spec contract for a bot archetype.

    Declares: required features, optional features, sessions, constraints,
    confirmations. Coding agents consume this to build bots.
    The Composer validates against this spec.

    Frozen — once defined, an archetype spec cannot be mutated.
    Use derive() or ArchetypeRegistry.derive() to create variants.
    """

    model_config = ConfigDict(frozen=True)

    archetype_id: str                          # unique ID, e.g. "opening_range_breakout"
    base_type: str                             # strategy family: "breakout", "scalping", etc.
    description: str                           # human-readable description
    v1_priority: bool = False                  # True for Phase 1 active archetypes
    quality_class: str = "native_supported"   # quality class per Phase 5

    # Feature declarations
    required_features: List[str] = Field(default_factory=list)              # mandatory feature_ids
    optional_features: List[str] = Field(default_factory=list)              # available optional feature_ids
    excluded_features: List[str] = Field(default_factory=list)              # features that cannot coexist

    # Session scope
    session_tags: List[str] = Field(default_factory=list)                    # applicable sessions

    # Constraints
    constraints: List[ConstraintSpec] = Field(default_factory=list)          # composition constraints
    confirmations: List[str] = Field(default_factory=list)                   # safety confirmation IDs

    # Execution
    execution_profile: str                    # execution profile name
    max_trades_per_hour: Optional[int] = Field(default=None)  # optional trade rate limit

    # Mutation boundaries (defaults -- override per BotMutationProfile)
    default_allowed_areas: List[str] = Field(default_factory=list)    # what can be mutated
    default_locked_components: List[str] = Field(default_factory=list)  # what cannot be changed

    # Derived archetype support
    base_archetype_id: Optional[str] = Field(default=None)  # for derived archetypes: base archetype ID

    def is_feature_required(self, feature_id: str) -> bool:
        """Return True if the given feature_id is a required feature."""
        return feature_id in self.required_features

    def is_feature_excluded(self, feature_id: str) -> bool:
        """Return True if the given feature_id is an excluded feature."""
        return feature_id in self.excluded_features

    def get_constraints(
        self, enabled_only: bool = True
    ) -> List[ConstraintSpec]:
        """
        Return constraints, optionally filtered to enabled only.

        Args:
            enabled_only: If True (default), return only enabled constraints.
        """
        if enabled_only:
            return [c for c in self.constraints if c.enabled]
        return list(self.constraints)

    def is_compatible_with(self, feature_id: str) -> bool:
        """
        Return True if a feature is compatible with this archetype.
        Compatible means: not excluded and either required or optional.
        """
        if feature_id in self.excluded_features:
            return False
        return feature_id in self.required_features or feature_id in self.optional_features

    @classmethod
    def derive(
        cls,
        base: "ArchetypeSpec",
        overrides: Dict[str, Any],
    ) -> "ArchetypeSpec":
        """
        Create a derived archetype spec from a base spec with overrides applied.

        The derived spec inherits all fields from the base and applies overrides.
        base_archetype_id is always set to the base archetype_id.
        archetype_id is always overridden (via overrides or derived from base).

        Args:
            base: The base ArchetypeSpec to derive from.
            overrides: Dict of field overrides.

        Returns:
            A new frozen ArchetypeSpec with overrides applied.
        """
        derived_data = base.model_dump()
        # Explicitly set base_archetype_id to the base archetype_id
        derived_data["base_archetype_id"] = base.archetype_id
        # Apply user overrides on top
        derived_data.update(overrides)
        return cls(**derived_data)


__all__ = ["ArchetypeSpec"]
