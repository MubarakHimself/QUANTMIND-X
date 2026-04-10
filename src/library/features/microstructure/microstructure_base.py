"""MicrostructureFeature — ABC base class for all microstructure order-flow features."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

from src.library.features.base import FeatureModule

if TYPE_CHECKING:
    from src.library.core.domain.feature_vector import FeatureConfidence


class MicrostructureFeature(FeatureModule, ABC):
    """
    Abstract base class for microstructure features that approximate order flow.

    Consolidates the proxy_inferred quality pattern for all order-flow
    approximated features. Subclasses derive signals from OHLCV price action
    rather than real exchange order book data.

    Subclasses should:
    - Set quality_class = "proxy_inferred" in config
    - Document the approximation method in the class docstring
    - Implement compute() / compute_batch() returning raw values
    - Provide confidence based on available lookback bars
    """

    @abstractmethod
    def compute(self, bar: dict[str, Any]) -> Any:
        """
        Compute the microstructure signal for a single bar.

        Args:
            bar: Dictionary with OHLCV fields (open, high, low, close, volume,
                 tick_count, spread, etc.). Exact keys vary by subclass.

        Returns:
            Raw feature value (float, str, or dict depending on subclass).
        """
        ...

    @abstractmethod
    def compute_batch(self, bars: List[dict[str, Any]]) -> List[Any]:
        """
        Compute over multiple bars with smoothing/historical context.

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of feature values, one per bar.
        """
        ...
