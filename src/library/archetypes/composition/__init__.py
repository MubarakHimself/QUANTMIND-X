"""
QuantMindLib V1 — Composition Package

Provides validation logic that ensures bots are composed correctly
before they run. Library scope: primitives/toolbox only. Validation
is spec-based, not executable.
"""

from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.composition.validator import CompositionValidator
from src.library.archetypes.composition.validation import ValidationResult

__all__ = [
    "CompositionResult",
    "CompositionValidator",
    "RequirementResolver",
    "ValidationResult",
]

# Lazy import to avoid circular dependency
from src.library.archetypes.composition.result import CompositionResult