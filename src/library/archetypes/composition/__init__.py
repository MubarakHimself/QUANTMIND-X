"""
QuantMindLib V1 — Composition Package

Provides validation logic that ensures bots are composed correctly
before they run. Library scope: primitives/toolbox only. Validation
is spec-based, not executable.
"""

from src.library.archetypes.composition.validation import ValidationResult
from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.composition.validator import CompositionValidator

__all__ = [
    "ValidationResult",
    "RequirementResolver",
    "CompositionValidator",
]