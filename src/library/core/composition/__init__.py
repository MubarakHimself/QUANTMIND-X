"""
QuantMindLib V1 — Composition module
"""

from src.library.core.composition.adapter_contracts import (
    IExecutionAdapter,
    IMarketDataAdapter,
    IRiskAdapter,
)
from src.library.core.composition.bridge_contracts import (
    IExecutionBridge,
    IFeatureBridge,
    IRiskBridge,
    ISentinelBridge,
)
from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.composition.compatibility_rule import CompatibilityRule
from src.library.core.composition.dependency_spec import DependencySpec
from src.library.core.composition.output_spec import OutputSpec
from src.library.core.composition.spec_registry import SpecRegistry
from src.library.core.composition.trd_converter import (
    TRDConversionResult,
    TRDConverter,
    TRDRawData,
)

__all__ = [
    "CapabilitySpec",
    "DependencySpec",
    "CompatibilityRule",
    "OutputSpec",
    "SpecRegistry",
    "TRDRawData",
    "TRDConversionResult",
    "TRDConverter",
    "IMarketDataAdapter",
    "IMarketDataAdapter",
    "IExecutionAdapter",
    "IRiskAdapter",
    "ISentinelBridge",
    "IExecutionBridge",
    "IFeatureBridge",
    "IRiskBridge",
]
