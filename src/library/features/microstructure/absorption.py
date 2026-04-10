"""AbsorptionProxyFeature — detects absorption of market orders by large players."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


class AbsorptionProxyFeature(MicrostructureFeature):
    """
    Detects absorption of market orders by large players.

    Absorption heuristic: price doesn't move despite high volume.
    When large players are actively absorbing supply/demand, the price
    remains flat while volume spikes. This signals institutional activity.

    Score = volume / price_range
    Higher score = more absorption detected (price movement suppressed
    relative to volume).

    Quality: proxy_inferred (heuristic — not a direct exchange metric)
    """

    def __init__(self, smoothing: float = 0.3, **kwargs: Any) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_smoothing"] = smoothing
        self.__dict__["_prev_score"] = 0.0

    def compute(self, bar: Dict[str, Any]) -> float:
        """
        Compute absorption score for a single bar.

        Args:
            bar: Dictionary with open, high, low, close, volume keys.

        Returns:
            float: Absorption score in range [0.0, ~10.0+].
                   Higher = more absorption detected.
                   0.0–2.0 = normal
                   2.0–5.0 = moderate absorption
                   5.0–10.0 = strong absorption
                   10.0+ = extreme absorption
        """
        volume = bar.get("volume", 0.0)
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)

        price_range = high - low

        if volume <= 0:
            return 0.0

        if price_range <= 0:
            # No range — could be extreme absorption or degenerate data
            # Treat flat bars with volume as potential absorption
            return 0.0

        score = volume / price_range

        # Clamp: very large scores indicate extreme absorption
        # Normalise to a reasonable upper range for display purposes
        return max(0.0, score)

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[float]:
        """
        Compute smoothed absorption over multiple bars.

        Uses EMA smoothing: smoothed = smoothing * current + (1-smoothing) * prev

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of absorption scores, one per bar.
        """
        if not bars:
            return []

        results: List[float] = []
        smoothed: float = self.__dict__["_prev_score"]
        smoothing: float = self.__dict__["_smoothing"]

        for bar in bars:
            raw = self.compute(bar)
            smoothed = smoothing * raw + (1.0 - smoothing) * smoothed
            results.append(smoothed)

        self.__dict__["_prev_score"] = smoothed
        return results

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/absorption_proxy",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Heuristic: high volume / narrow range = absorption by large players",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"open", "high", "low", "close", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"absorption_score"}

    def capability_spec(self) -> Dict[str, Any]:
        return {
            "feature_id": "microstructure/absorption_proxy",
            "quality_class": "proxy_inferred",
            "inputs": ["open", "high", "low", "close", "volume"],
            "outputs": ["absorption_score"],
            "lookback_required": 5,
        }

    def confidence(self, n_bars: int) -> FeatureConfidence:
        """
        Compute confidence based on number of available bars.

        Args:
            n_bars: Number of bars available for the computation.

        Returns:
            FeatureConfidence with quality score and feed tag.
        """
        if n_bars < 5:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.DISABLED.value,
            )
        elif n_bars < 10:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.3,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.LOW.value,
            )
        elif n_bars < 20:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.6,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.MEDIUM.value,
            )
        else:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.85,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.HIGH.value,
            )
