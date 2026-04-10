"""TickActivityFeature — measures tick rate and activity intensity."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


# Activity level constants — defined at module level to avoid Pydantic field conflicts
ACTIVITY_LOW: str = "LOW"
ACTIVITY_NORMAL: str = "NORMAL"
ACTIVITY_HIGH: str = "HIGH"
ACTIVITY_EXTREME: str = "EXTREME"


class TickActivityFeature(MicrostructureFeature):
    """
    Measures tick rate and activity intensity.

    Activity = tick_count / expected_ticks_per_bar
    Sustained high activity often precedes breakouts and liquidity events.

    Falls back to volume-based estimation when tick_count is unavailable.

    Quality: proxy_inferred (tick count from OHLCV, not true tick data)
    """

    def __init__(self, expected_ticks_per_bar: int = 50, **kwargs: Any) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_expected_ticks"] = expected_ticks_per_bar
        self.__dict__["_history"] = []
        self.__dict__["_volume_samples"] = []

    def compute(self, bar: Dict[str, Any]) -> float:
        """
        Compute activity ratio for a single bar.

        Args:
            bar: Dictionary with tick_count (optional) and volume keys.

        Returns:
            float: Activity ratio in range [0.0, ~3.0+].
                   > 2.0 = high activity, > 3.0 = extreme activity.
        """
        tick_count = bar.get("tick_count")
        volume = bar.get("volume", 0.0)

        if tick_count is not None and tick_count > 0 and volume > 0:
            # tick_count is primary when both are available (real tick data)
            activity = tick_count / self.__dict__["_expected_ticks"]
        elif volume > 0:
            # Fallback: estimate from volume (proxy approximation)
            activity = volume / self.__dict__["_expected_ticks"]
            self.__dict__["_volume_samples"].append(activity)
        else:
            activity = 0.0

        return max(0.0, activity)

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[float]:
        """
        Compute activity ratio over multiple bars.

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of float activity ratios.
        """
        if not bars:
            return []

        results: List[float] = []
        for bar in bars:
            activity = self.compute(bar)
            self.__dict__["_history"].append(activity)
            results.append(activity)
        return results

    def get_activity_level(self, activity: float) -> str:
        """
        Classify activity ratio into a human-readable level.

        Args:
            activity: Activity ratio from compute().

        Returns:
            str: One of LOW, NORMAL, HIGH, EXTREME.
        """
        if activity < 1.0:
            return ACTIVITY_LOW
        elif activity < 2.0:
            return ACTIVITY_NORMAL
        elif activity < 3.0:
            return ACTIVITY_HIGH
        else:
            return ACTIVITY_EXTREME

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/tick_activity",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Tick rate approximation; falls back to volume when tick_count unavailable",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"tick_count", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"activity_ratio", "activity_level"}

    def confidence(self, n_bars: int) -> FeatureConfidence:
        """
        Compute confidence based on number of available bars.

        Args:
            n_bars: Number of bars available for the computation.

        Returns:
            FeatureConfidence with quality score and feed tag.
        """
        if n_bars < 10:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.DISABLED.value,
            )
        elif n_bars < 20:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.35,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.LOW.value,
            )
        elif n_bars < 50:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.65,
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
