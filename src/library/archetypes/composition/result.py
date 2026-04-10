"""
QuantMindLib V1 — CompositionResult
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.composition.validation import ValidationResult
from src.library.core.domain.bot_spec import BotSpec


class CompositionResult(BaseModel):
    """
    Result of a composition attempt.

    bot_spec is None if validation failed.
    """

    bot_spec: Optional[BotSpec] = Field(default=None)
    validation: ValidationResult
    archetype: Optional[ArchetypeSpec] = Field(default=None)
    composition_notes: List[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        """True if composition succeeded (bot_spec is not None and valid)."""
        return self.bot_spec is not None and self.validation.valid

    @property
    def failed(self) -> bool:
        """True if composition failed."""
        return not self.succeeded


__all__ = ["CompositionResult"]
