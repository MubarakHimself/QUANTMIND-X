"""
QuantMindLib V1 — Archetypes Package

Provides the archetype system for QuantMindLib bots.
Archetypes are frozen spec contracts consumed by coding agents
and validated by the Composer.
"""

from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.registry import (
    ArchetypeRegistry,
    ArchetypeNotFoundError,
    DuplicateArchetypeError,
)

__all__ = [
    "ConstraintSpec",
    "ArchetypeSpec",
    "BaseArchetype",
    "ArchetypeRegistry",
    "ArchetypeNotFoundError",
    "DuplicateArchetypeError",
]
