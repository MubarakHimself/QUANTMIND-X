"""
QuantMindLib V1 — Archetypes Package

Provides the archetype system for QuantMindLib bots.
Archetypes are frozen spec contracts consumed by coding agents
and validated by the Composer.
"""

from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.composer import Composer
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.derived import (
    LONDON_ORB_ARCHETYPE,
    NY_ORB_ARCHETYPE,
    SCALPER_M1_ARCHETYPE,
)
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
from src.library.archetypes.stubs import (
    BREAKOUT_SCALPER_ARCHETYPE,
    get_full_registry,
    MEAN_REVERSION_ARCHETYPE,
    PULLBACK_SCALPER_ARCHETYPE,
    SESSION_TRANSITION_ARCHETYPE,
)

__all__ = [
    # Registry + exceptions
    "ArchetypeNotFoundError",
    "ArchetypeRegistry",
    "DuplicateArchetypeError",
    # Core spec classes
    "ArchetypeSpec",
    "BaseArchetype",
    "Composer",
    "ConstraintSpec",
    "MutationEngine",
    # ORB
    "OpeningRangeBreakout",
    "ORB_ARCHETYPE",
    # Derived archetypes (Packet 6E)
    "LONDON_ORB_ARCHETYPE",
    "NY_ORB_ARCHETYPE",
    "SCALPER_M1_ARCHETYPE",
    # Structural stubs (Packet 6E)
    "BREAKOUT_SCALPER_ARCHETYPE",
    "MEAN_REVERSION_ARCHETYPE",
    "PULLBACK_SCALPER_ARCHETYPE",
    "SESSION_TRANSITION_ARCHETYPE",
    # Bootstrap singletons
    "get_default_registry",
    "get_full_registry",
]
