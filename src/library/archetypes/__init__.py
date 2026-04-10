"""
QuantMindLib V1 — Archetypes Package

Provides the archetype system for QuantMindLib bots.
Archetypes are frozen spec contracts consumed by coding agents
and validated by the Composer.
"""

from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.composer import Composer
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.orb import (
    OpeningRangeBreakout,
    ORB_ARCHETYPE,
    get_default_registry,
)
from src.library.archetypes.registry import (
    ArchetypeNotFoundError,
    ArchetypeRegistry,
    DuplicateArchetypeError,
)
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.mutation.engine import MutationEngine

__all__ = [
    "ArchetypeNotFoundError",
    "ArchetypeRegistry",
    "ArchetypeSpec",
    "BaseArchetype",
    "Composer",
    "ConstraintSpec",
    "DuplicateArchetypeError",
    "get_default_registry",
    "MutationEngine",
    "OpeningRangeBreakout",
    "ORB_ARCHETYPE",
]
