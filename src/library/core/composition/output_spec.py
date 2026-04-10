"""
QuantMindLib V1 — OutputSpec Schema
CONTRACT-021
"""
from __future__ import annotations

from typing import List, Optional, Set

from pydantic import BaseModel, Field


class OutputSpec(BaseModel):
    """Output interface definition for a module — what schemas it produces."""

    output_id: str                                   # Unique output identifier, e.g. "FEATURE_VECTOR_OUT"
    output_type: str                                 # Schema type name, e.g. "FeatureVector"
    schema_version: str = Field(default="1.0.0")    # Semantic version
    description: str = ""
    fields: List[str] = Field(default_factory=list) # Field names this output produces
    mime_type: Optional[str] = None                 # Optional MIME type hint

    def produces_field(self, field_name: str) -> bool:
        """Return True if this output includes the given field name."""
        return field_name in self.fields

    def produces_any_field(self, field_names: Set[str]) -> Set[str]:
        """Return subset of field_names this output actually produces."""
        return field_names & set(self.fields)

    def is_version_compatible(self, other_version: str) -> bool:
        """Simple major version check — same major version means compatible."""
        self_major = self.schema_version.split(".")[0]
        other_major = other_version.split(".")[0]
        return self_major == other_major


__all__ = ["OutputSpec"]
