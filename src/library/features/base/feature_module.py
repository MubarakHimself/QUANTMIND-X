"""
QuantMindLib V1 — FeatureModule Base Class

FeatureModule is the abstract base for all feature compute primitives.
Each feature takes raw market data inputs and produces a FeatureVector output.
Features are stateless — no side effects, no trade execution.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel

from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.composition.output_spec import OutputSpec
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class FeatureConfig(BaseModel):
    """
    Per-feature configuration — inputs, window sizes, thresholds.
    Each feature subclass defines its own config fields.
    """
    model_config = BaseModel.model_config

    feature_id: str
    enabled: bool = True
    quality_class: str = "native_supported"  # "native_supported" | "proxy_inferred" | "external" | "deferred"
    source: str = "ctrader_native"           # Data source identifier
    notes: Optional[str] = None               # Mandatory for proxy_inferred


class FeatureModule(ABC, BaseModel):
    """
    Abstract base for all QuantMindLib feature modules.

    Features are stateless compute primitives. Subclasses implement `compute()`
    which takes raw inputs and returns a FeatureVector.

    Design principles:
    - Stateless: no instance state survives between calls
    - Pure: same inputs → same outputs
    - Labeled: every output carries quality metadata
    - Registered: every instance registers its CapabilitySpec

    Subclasses MUST:
    1. Define config_fields: the inputs this feature requires
    2. Implement compute(): pure function raw_inputs → FeatureVector
    3. Call super().__init__() with a FeatureConfig
    """
    model_config = BaseModel.model_config

    feature_id: str
    enabled: bool = True
    quality_class: str = "native_supported"
    source: str = "ctrader_native"
    notes: Optional[str] = None

    @property
    @abstractmethod
    def config(self) -> FeatureConfig:
        """Return this feature's configuration."""
        ...

    @property
    @abstractmethod
    def required_inputs(self) -> Set[str]:
        """Set of input keys this feature requires (e.g. {'close_prices', 'volume'})."""
        ...

    @property
    @abstractmethod
    def output_keys(self) -> Set[str]:
        """Set of feature keys this module produces (e.g. {'rsi_14', 'rsi_28'})."""
        ...

    @abstractmethod
    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        """
        Compute feature values from raw market data inputs.

        Args:
            inputs: Dict mapping input key names to their values.
                    e.g. {'close_prices': [1.0850, 1.0852, ...], 'volume': [...]}

        Returns:
            FeatureVector with computed feature values and quality metadata.

        Raises:
            ValueError: if required inputs are missing or invalid
        """
        ...

    def capability_spec(self) -> CapabilitySpec:
        """
        Generate a CapabilitySpec for this feature.
        Subclasses can override to provide a pre-built spec.
        """
        return CapabilitySpec(
            module_id=self.feature_id,
            provides=list(self.output_keys),
            requires=list(self.required_inputs),
            optional=[],
            compatibility=["scalping", "orb", "mean_reversion", "breakout"],
            sync_mode=True,
        )

    def output_spec(self) -> OutputSpec:
        """
        Generate an OutputSpec describing this feature's output schema.
        """
        return OutputSpec(
            output_id=f"{self.feature_id}_out",
            output_type="FeatureVector",
            schema_version="1.0.0",
            description=f"Feature output for {self.feature_id}",
            fields=list(self.output_keys),
        )

    def is_quality_labeled(self) -> bool:
        """True if this feature has a valid quality label."""
        valid_classes = {"native_supported", "proxy_inferred", "external", "deferred"}
        return self.quality_class in valid_classes

    def __init__(self, **data: Any) -> None:
        # Ensure feature_id is set before BaseModel init
        if "feature_id" not in data and hasattr(self, "feature_id"):
            data["feature_id"] = self.feature_id
        super().__init__(**data)