"""VolumeProfileFeature — Point of Control and Value Area"""
from __future__ import annotations

from typing import Any, Dict, List, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class VolumeProfileFeature(FeatureModule):
    """
    Computes Volume Profile — POC (Point of Control), Value Area, and POC strength.

    POC is the price level with the highest volume-weighted activity.
    Value Area covers 70% of total volume.

    Inputs required: high_prices, low_prices, close_prices, volumes
    Outputs:        poc_price, poc_distance, value_area_high, value_area_low, poc_strength
    """
    num_bins: int = Field(default=20, ge=5, le=100)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="volume/profile",
            quality_class="native_supported",
            source="VolumeProfileFeature",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"high_prices", "low_prices", "close_prices", "volumes"}

    @property
    def output_keys(self) -> set[str]:
        return {
            "poc_price",
            "poc_distance",
            "value_area_high",
            "value_area_low",
            "poc_strength",
        }

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        high_prices: List[float] = inputs.get("high_prices", [])
        low_prices: List[float] = inputs.get("low_prices", [])
        close_prices: List[float] = inputs.get("close_prices", [])
        volumes: List[float] = inputs.get("volumes", [])

        n = len(close_prices)

        if n < self.num_bins:
            return self._insufficient_data()

        min_price = min(low_prices)
        max_price = max(high_prices)
        bin_width = (max_price - min_price) / self.num_bins if max_price > min_price else 0.0001

        # Build price-volume histogram
        bin_volumes: List[float] = [0.0] * self.num_bins
        total_volume = 0.0

        for h, l, v in zip(high_prices, low_prices, volumes):
            if h <= l:
                # Single-price bar
                bin_idx = min(int((h - min_price) / bin_width), self.num_bins - 1) if bin_width > 0 else 0
                bin_volumes[bin_idx] += v
                total_volume += v
            else:
                # Price range bar — distribute volume across covered bins
                start_bin = int((l - min_price) / bin_width) if bin_width > 0 else 0
                end_bin = int((h - min_price) / bin_width) if bin_width > 0 else self.num_bins - 1
                for b in range(max(0, start_bin), min(self.num_bins, end_bin + 1)):
                    bin_volumes[b] += v / (end_bin - start_bin + 1)
                total_volume += v

        # POC: bin with highest volume
        poc_bin = bin_volumes.index(max(bin_volumes))
        poc_price = min_price + (poc_bin + 0.5) * bin_width
        poc_price = round(poc_price, 5)

        # Value area: sorted bins descending, accumulate until 70% of total volume
        sorted_bins = sorted(enumerate(bin_volumes), key=lambda x: x[1], reverse=True)
        cumsum = 0.0
        value_area_bins: List[int] = []
        for bin_idx, vol in sorted_bins:
            cumsum += vol
            value_area_bins.append(bin_idx)
            if total_volume > 0 and cumsum / total_volume >= 0.70:
                break

        value_area_high = min_price + (max(value_area_bins) + 1) * bin_width
        value_area_low = min_price + min(value_area_bins) * bin_width
        value_area_high = round(value_area_high, 5)
        value_area_low = round(value_area_low, 5)

        # Distance from close to POC in pips (4-decimal FX convention)
        current_close = close_prices[-1]
        poc_distance = abs(current_close - poc_price) * 100000

        # POC strength: ratio of POC bin volume to total volume
        poc_strength = max(bin_volumes) / total_volume if total_volume > 0 else 0.0

        quality = min(1.0, (n - self.num_bins) / (self.num_bins * 5))
        quality_tag = "HIGH" if quality >= 0.7 else "MEDIUM"

        return FeatureVector(
            bot_id="SYSTEM",
            features={
                "poc_price": poc_price,
                "poc_distance": poc_distance,
                "value_area_high": value_area_high,
                "value_area_low": value_area_low,
                "poc_strength": poc_strength,
            },
            feature_confidence={
                "poc_price": FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "poc_distance": FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "value_area_high": FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "value_area_low": FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "poc_strength": FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
            },
        )

    def _insufficient_data(self) -> FeatureVector:
        return FeatureVector(
            bot_id="SYSTEM",
            features={
                "poc_price": 0.0,
                "poc_distance": 0.0,
                "value_area_high": 0.0,
                "value_area_low": 0.0,
                "poc_strength": 0.0,
            },
            feature_confidence={
                k: FeatureConfidence(
                    source="VolumeProfileFeature",
                    quality=0.0,
                    latency_ms=0.0,
                    feed_quality_tag="INSUFFICIENT_DATA",
                )
                for k in ("poc_price", "poc_distance", "value_area_high", "value_area_low", "poc_strength")
            },
        )
