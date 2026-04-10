"""
QuantMindLib V1 — SpecRegistry
CONTRACT-024
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.composition.compatibility_rule import CompatibilityRule
from src.library.core.composition.dependency_spec import DependencySpec
from src.library.core.composition.output_spec import OutputSpec


class SpecRegistry(BaseModel):
    """
    Versioned, immutable registry for module specs.
    Stores CapabilitySpec, DependencySpec, CompatibilityRule, and OutputSpec objects
    keyed by module_id and version.
    """

    capability_specs: Dict[str, Dict[str, CapabilitySpec]] = Field(default_factory=dict)
    dependency_specs: Dict[str, Dict[str, DependencySpec]] = Field(default_factory=dict)
    compatibility_rules: Dict[str, Dict[str, CompatibilityRule]] = Field(default_factory=dict)
    output_specs: Dict[str, Dict[str, OutputSpec]] = Field(default_factory=dict)

    def register_capability(self, module_id: str, version: str, spec: CapabilitySpec) -> None:
        """Register a CapabilitySpec under module_id:version."""
        if module_id not in self.capability_specs:
            self.capability_specs[module_id] = {}
        self.capability_specs[module_id][version] = spec

    def register_dependency(self, module_id: str, version: str, spec: DependencySpec) -> None:
        """Register a DependencySpec under module_id:version."""
        if module_id not in self.dependency_specs:
            self.dependency_specs[module_id] = {}
        self.dependency_specs[module_id][version] = spec

    def register_compatibility(self, rule_id: str, rule: CompatibilityRule) -> None:
        """Register a CompatibilityRule by rule_id."""
        if rule_id not in self.compatibility_rules:
            self.compatibility_rules[rule_id] = {}
        self.compatibility_rules[rule_id]["1.0.0"] = rule

    def register_output(self, module_id: str, version: str, spec: OutputSpec) -> None:
        """Register an OutputSpec under module_id:version."""
        if module_id not in self.output_specs:
            self.output_specs[module_id] = {}
        self.output_specs[module_id][version] = spec

    def get_capability(self, module_id: str, version: str = "1.0.0") -> Optional[CapabilitySpec]:
        """Retrieve a CapabilitySpec by module_id and version."""
        return self.capability_specs.get(module_id, {}).get(version)

    def get_dependency(self, module_id: str, version: str = "1.0.0") -> Optional[DependencySpec]:
        """Retrieve a DependencySpec by module_id and version."""
        return self.dependency_specs.get(module_id, {}).get(version)

    def get_output(self, module_id: str, version: str = "1.0.0") -> Optional[OutputSpec]:
        """Retrieve an OutputSpec by module_id and version."""
        return self.output_specs.get(module_id, {}).get(version)

    def get_all_module_ids(self) -> List[str]:
        """Return all unique module_ids across capability, dependency, and output specs."""
        capability_ids = set(self.capability_specs.keys())
        dependency_ids = set(self.dependency_specs.keys())
        output_ids = set(self.output_specs.keys())
        return sorted(capability_ids | dependency_ids | output_ids)

    def get_versions(self, module_id: str) -> List[str]:
        """Return all known versions for a module_id across all spec types."""
        versions: set[str] = set()
        versions.update(self.capability_specs.get(module_id, {}).keys())
        versions.update(self.dependency_specs.get(module_id, {}).keys())
        versions.update(self.output_specs.get(module_id, {}).keys())
        return sorted(versions)


__all__ = ["SpecRegistry"]
