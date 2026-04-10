"""
QuantMindLib V1 — ConstraintSpec
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConstraintSpec(BaseModel):
    """
    A single constraint check for bot composition.

    Used by archetypes to declare composition constraints (e.g. spread thresholds,
    news blackouts, session windows). The Composer evaluates these at validation time.
    """

    constraint_id: str           # unique identifier, e.g. "spread_filter", "news_blackout"
    operator: str                # comparison operator: "<", ">", "<=", ">=", "==", "!="
    value: Any                   # threshold value (float, str, int)
    enabled: bool = True         # can be toggled off

    def check(self, value: Any) -> bool:
        """
        Evaluate whether the given value satisfies this constraint.

        Returns True if the constraint is disabled, or if the comparison succeeds.
        """
        if not self.enabled:
            return True
        op = self.operator
        if op == "<":
            return value < self.value
        if op == ">":
            return value > self.value
        if op == "<=":
            return value <= self.value
        if op == ">=":
            return value >= self.value
        if op == "==":
            return value == self.value
        if op == "!=":
            return value != self.value
        return False


__all__ = ["ConstraintSpec"]
