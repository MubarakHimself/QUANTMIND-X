"""VolumeImbalanceFeature — detects order flow imbalance from volume data."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


class VolumeImbalanceFeature(MicrostructureFeature):
    """
    Detects order flow imbalance from volume data.

    Imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    Buy/sell volume is derived from tick direction heuristic:
      - Up-tick bar (close > open): all volume counts as buy-side
      - Down-tick bar (close < open): all volume counts as sell-side
      - Flat bar (close == open): volume split evenly

    Optionally smoothed with EMA to reduce noise.

    Quality: proxy_inferred (derived from OHLCV tick direction, not true order flow)
    """

    def __init__(self, smoothing: float = 0.3, **kwargs: Any) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_smoothing"] = smoothing
        self.__dict__["_prev_imbalance"] = 0.0

    def _tick_direction(self, bar: Dict[str, Any]) -> float:
        """
        Determine buy-side fraction of bar volume.

        Returns:
            float: 1.0 = all buy, 0.0 = all sell, 0.5 = neutral.
        """
        open_price = bar.get("open", 0.0)
        close_price = bar.get("close", 0.0)

        if open_price == 0:
            return 0.5

        diff = close_price - open_price
        if diff > 0:
            return 1.0
        elif diff < 0:
            return 0.0
        else:
            return 0.5

    def compute(self, bar: Dict[str, Any]) -> float:
        """
        Compute volume imbalance for a single bar.

        Args:
            bar: Dictionary with keys open, close, volume.

        Returns:
            float: Imbalance in range [-1.0, 1.0].
                   Positive = buying pressure, negative = selling pressure.
        """
        volume = bar.get("volume", 0.0)
        if volume <= 0:
            return 0.0

        buy_fraction = self._tick_direction(bar)
        buy_volume = volume * buy_fraction
        sell_volume = volume * (1.0 - buy_fraction)

        total = buy_volume + sell_volume
        if total == 0:
            return 0.0

        imbalance = (buy_volume - sell_volume) / total
        return max(-1.0, min(1.0, imbalance))

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[float]:
        """
        Compute smoothed imbalance over multiple bars.

        Uses EMA smoothing: smoothed = smoothing * current + (1-smoothing) * prev

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of float imbalance values, one per bar.
        """
        if not bars:
            return []

        results: List[float] = []
        smoothed: float = self.__dict__["_prev_imbalance"]
        smoothing: float = self.__dict__["_smoothing"]

        for bar in bars:
            raw = self.compute(bar)
            smoothed = smoothing * raw + (1.0 - smoothing) * smoothed
            results.append(smoothed)

        self.__dict__["_prev_imbalance"] = smoothed
        return results

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/volume_imbalance",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Heuristic approximation of buy/sell volume from OHLCV tick direction",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"open", "close", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"imbalance"}

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
