"""
QuantMindLib V1 — ArchetypeRegistry
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, PrivateAttr

from src.library.archetypes.archetype_spec import ArchetypeSpec


class ArchetypeNotFoundError(Exception):
    """Raised when an archetype_id is not found in the registry."""

    pass


class DuplicateArchetypeError(Exception):
    """Raised when attempting to register an archetype with a duplicate ID."""

    pass


class ArchetypeRegistry(BaseModel):
    """
    Singleton registry for all QuantMindLib archetypes.

    Manages registration, lookup, and derived archetype generation.
    Provides indexing by base_type, session, and V1 priority for fast queries.

    Usage:
        registry = ArchetypeRegistry()
        registry.register(my_archetype_spec)
        spec = registry.get("opening_range_breakout")
        all_ids = registry.list_all()
        v1_ids = registry.list_v1_priority()
    """

    model_config = BaseModel.model_config

    _archetypes: Dict[str, ArchetypeSpec] = PrivateAttr(default_factory=dict)
    _by_base_type: Dict[str, List[str]] = PrivateAttr(default_factory=dict)
    _by_session: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._archetypes = {}
        self._by_base_type = {}
        self._by_session = {}

    def register(self, spec: ArchetypeSpec) -> None:
        """
        Register an archetype spec.

        Raises:
            DuplicateArchetypeError: if archetype_id is already registered
        """
        aid = spec.archetype_id
        if aid in self._archetypes:
            raise DuplicateArchetypeError(
                f"Archetype {aid} already registered"
            )
        self._archetypes[aid] = spec

        # Index by base_type
        bt = spec.base_type
        if bt not in self._by_base_type:
            self._by_base_type[bt] = []
        if aid not in self._by_base_type[bt]:
            self._by_base_type[bt].append(aid)

        # Index by session tags
        for session in spec.session_tags:
            if session not in self._by_session:
                self._by_session[session] = []
            if aid not in self._by_session[session]:
                self._by_session[session].append(aid)

    def get(self, archetype_id: str) -> Optional[ArchetypeSpec]:
        """Get a registered archetype spec by archetype_id."""
        return self._archetypes.get(archetype_id)

    def list_all(self) -> List[str]:
        """List all registered archetype IDs."""
        return list(self._archetypes.keys())

    def list_by_base_type(self, base_type: str) -> List[str]:
        """List archetype IDs matching a given base type."""
        return list(self._by_base_type.get(base_type, []))

    def list_by_session(self, session: str) -> List[str]:
        """List archetype IDs valid for a given session tag."""
        return list(self._by_session.get(session, []))

    def list_v1_priority(self) -> List[str]:
        """List V1-priority archetype IDs."""
        return [
            aid
            for aid, spec in self._archetypes.items()
            if spec.v1_priority
        ]

    def compatible_features(self, archetype_id: str) -> List[str]:
        """
        Get all features compatible with an archetype.

        Returns required + optional features, excluding those already
        marked as excluded by the archetype.
        """
        spec = self._archetypes.get(archetype_id)
        if not spec:
            return []
        return list(
            set(spec.required_features) | set(spec.optional_features)
        )

    def derive(
        self,
        base_archetype_id: str,
        derived_id: str,
        overrides: Dict[str, Any],
    ) -> ArchetypeSpec:
        """
        Create a derived archetype spec from a base archetype.

        Overrides are applied on top of the base spec's frozen data.
        The derived spec's archetype_id is set to derived_id and
        base_archetype_id is set to base_archetype_id.

        Args:
            base_archetype_id: The archetype_id of the base archetype
            derived_id: The archetype_id for the new derived archetype
            overrides: Dict of field overrides

        Returns:
            A new frozen ArchetypeSpec

        Raises:
            ArchetypeNotFoundError: if base_archetype_id is not registered
        """
        base = self.get(base_archetype_id)
        if not base:
            raise ArchetypeNotFoundError(
                f"Base archetype {base_archetype_id} not found"
            )

        derived_data = base.model_dump()
        derived_data["archetype_id"] = derived_id
        derived_data["base_archetype_id"] = base_archetype_id
        derived_data.update(overrides)

        return ArchetypeSpec(**derived_data)


__all__ = [
    "ArchetypeRegistry",
    "ArchetypeNotFoundError",
    "DuplicateArchetypeError",
]
