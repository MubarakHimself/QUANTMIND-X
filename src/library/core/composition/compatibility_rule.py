"""
QuantMindLib V1 -- CompatibilityRule Schema
CONTRACT-020
Sourced from: docs/planning/quantmindlib/07_shared_object_model.md §19
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CompatibilityRule(BaseModel):
    """
    Rules governing what modules can or cannot be combined.

    Mutable for runtime rule registration at module load time.

    Attributes:
        rule_id: Unique rule identifier, e.g. "rule_no_vwap_with_rsi"
        subject: Capability/module ID this rule applies to, e.g. "orderflow.vwap"
        predicate: Rule type: COMBINES_WITH / EXCLUDES / REQUIRES
        target: Other capability/module ID, e.g. "indicators.rsi"
        condition: Optional condition expression, e.g. "regime==TREND_STABLE"
    """

    rule_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    predicate: Literal["COMBINES_WITH", "EXCLUDES", "REQUIRES"]
    target: str = Field(min_length=1)
    condition: Optional[str] = None

    def matches(self, subject_id: str, target_id: str) -> bool:
        """
        Return True if this rule applies to the given subject/target pair.

        The match is directional: subject must match self.subject AND
        target must match self.target.
        """
        return self.subject == subject_id and self.target == target_id

    def applies_to(self, cap_id: str) -> bool:
        """
        Return True if this rule applies to the given capability ID.

        A rule applies if the capability ID is either the subject or
        the target of the rule.
        """
        return self.subject == cap_id or self.target == cap_id


__all__ = ["CompatibilityRule"]
