"""
QuantMindLib V1 — FeatureRegistry

Registry for all QuantMindLib feature modules.
Provides registration, lookup, dependency validation, and composition queries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, PrivateAttr

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.composition.spec_registry import SpecRegistry


class FeatureNotFoundError(Exception):
    """Raised when a feature_id is not found in the registry."""
    pass


class DependencyMissingError(Exception):
    """Raised when a required feature dependency is not registered."""
    pass


class FeatureRegistry(BaseModel):
    """
    Central registry for all QuantMindLib feature modules.
    Manages registration, lookup, and dependency validation.
    """
    model_config = BaseModel.model_config

    _features: Dict[str, FeatureModule] = PrivateAttr(default_factory=dict)
    _capability_specs: Dict[str, CapabilitySpec] = PrivateAttr(default_factory=dict)
    _output_specs: Dict[str, SpecRegistry] = PrivateAttr(default_factory=dict)
    _by_family: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._features = {}
        self._capability_specs = {}
        self._output_specs = {}
        self._by_family = {}

    def register(
        self,
        feature: FeatureModule,
        family: Optional[str] = None,
    ) -> None:
        """
        Register a feature module.

        Args:
            feature: FeatureModule instance to register
            family: Optional family identifier (e.g. 'indicators', 'volume')
        """
        fid = feature.feature_id
        self._features[fid] = feature
        self._capability_specs[fid] = feature.capability_spec()

        if family:
            if family not in self._by_family:
                self._by_family[family] = []
            if fid not in self._by_family[family]:
                self._by_family[family].append(fid)

    def get(self, feature_id: str) -> Optional[FeatureModule]:
        """Get a registered feature by feature_id."""
        return self._features.get(feature_id)

    def get_capability(self, feature_id: str) -> Optional[CapabilitySpec]:
        """Get the CapabilitySpec for a registered feature."""
        return self._capability_specs.get(feature_id)

    def list_all(self) -> List[str]:
        """List all registered feature_ids."""
        return list(self._features.keys())

    def list_by_family(self, family: str) -> List[str]:
        """List all feature_ids in a given family."""
        return list(self._by_family.get(family, []))

    def families(self) -> List[str]:
        """List all registered families."""
        return list(self._by_family.keys())

    def validate_composition(self, feature_ids: List[str]) -> List[str]:
        """
        Validate that all feature_ids are registered.

        Data inputs (like 'close_prices') are not feature dependencies and do not
        need to be registered — they come from market data feeds.

        Returns:
            Empty list if all features are registered.
            Raises DependencyMissingError on first missing feature.
        """
        for fid in feature_ids:
            if fid not in self._features:
                raise DependencyMissingError(
                    f"Required feature not registered: {fid}. "
                    f"Available features: {sorted(self._features.keys())}"
                )
        return []

    def resolve_inputs(self, feature_id: str, available_inputs: Set[str]) -> Set[str]:
        """
        Given available inputs, determine which features can be computed
        (all their required inputs are satisfied).
        """
        computable = set()
        for fid, feature in self._features.items():
            if fid in available_inputs:
                continue
            if feature.required_inputs.issubset(available_inputs):
                computable.add(fid)
        return computable

    def enabled_features(self) -> List[FeatureModule]:
        """Return all enabled features."""
        return [f for f in self._features.values() if f.enabled]

    def quality_filter(self, quality_class: str) -> List[FeatureModule]:
        """Return all features matching a quality class."""
        return [f for f in self._features.values() if f.quality_class == quality_class]

    def __len__(self) -> int:
        return len(self._features)