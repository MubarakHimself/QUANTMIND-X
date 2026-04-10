"""
QuantMindLib V1 — ValidationResult
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Result holder for composition validation checks.

    Collects errors (fatal), warnings (non-fatal), and suggestions (optional)
    from all validation checks. Results can be merged to accumulate findings
    across multiple validation passes.
    """

    valid: bool = Field(default=True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """
        Merge two ValidationResult instances.

        Errors and warnings accumulate from both results.
        Valid is True only if both results are valid.
        Suggestions accumulate from both.

        Args:
            other: Another ValidationResult to merge into this one.

        Returns:
            A new ValidationResult with combined findings.
        """
        return ValidationResult(
            valid=self.valid and other.valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            suggestions=self.suggestions + other.suggestions,
        )

    @property
    def is_clean(self) -> bool:
        """
        True if no errors and no warnings.

        A clean result means the composition passed all checks with
        no issues at any severity level.
        """
        return self.valid and not self.errors and not self.warnings


__all__ = ["ValidationResult"]