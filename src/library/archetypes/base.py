"""
QuantMindLib V1 — BaseArchetype Abstract Base Class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

from src.library.core.composition.capability_spec import CapabilitySpec

if TYPE_CHECKING:
    from src.library.archetypes.archetype_spec import ArchetypeSpec
    from src.library.core.domain.bot_spec import BotSpec


class BaseArchetype(ABC, BaseModel):
    """
    Abstract base class for archetype implementations.

    Archetypes are primarily ArchetypeSpec data objects. The ABC provides
    archetype-specific behavior: validation, composition hints, capability derivation.

    Subclasses must implement spec, validate_bot_spec, and suggest_features.
    """

    @property
    @abstractmethod
    def spec(self) -> ArchetypeSpec:
        """Return the ArchetypeSpec for this archetype."""
        ...

    @abstractmethod
    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        """
        Validate that a BotSpec conforms to this archetype's spec.

        Returns list of error messages (empty = valid).
        Checks include:
        - All required features are present
        - No excluded features are present
        - Execution profile matches
        - Session tags are valid
        """
        ...

    @abstractmethod
    def suggest_features(
        self,
        context: Dict[str, Any],
    ) -> List[str]:
        """
        Given a runtime context, suggest optional features.

        Args:
            context: Dict with keys like 'regime', 'session', 'spread', 'volatility'

        Returns:
            List of feature_ids that may improve performance in this context.
        """
        ...

    def capability_spec(self) -> CapabilitySpec:
        """
        Derive a CapabilitySpec from this archetype's spec.

        Maps the archetype's feature declarations into a CapabilitySpec
        compatible with the composition system.
        """
        spec = self.spec
        return CapabilitySpec(
            module_id=f"archetype/{spec.archetype_id}",
            provides=[f"archetype/{spec.archetype_id}"],
            requires=spec.required_features,
            optional=spec.optional_features,
            outputs=["BotSpec"],
            compatibility=[spec.archetype_id],
            sync_mode=True,
        )


__all__ = ["BaseArchetype"]
