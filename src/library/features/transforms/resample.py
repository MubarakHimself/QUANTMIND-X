"""ResampleTransform — timeframe resampling"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class ResampleTransform(FeatureModule):
    """
    Resamples a bar series to a target timeframe.
    Computes OHLCV aggregates for the new timeframe.

    Quality class: native_supported
    """
    source_tf: str = Field(default="M1")   # Source timeframe
    target_tf: str = Field(default="M5")   # Target timeframe

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"transforms/resample_{self.source_tf}_to_{self.target_tf}",
            quality_class="native_supported",
            source="library_transform",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"high", "low", "close_prices", "volume"}

    @property
    def output_keys(self) -> set[str]:
        return {f"resampled_{self.target_tf}_close", f"resampled_{self.target_tf}_volume"}

    @staticmethod
    def _get_ratio(source: str, target: str) -> int:
        mapping = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        return mapping.get(target, 5) // mapping.get(source, 1)

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        high = inputs.get("high", [])
        low = inputs.get("low", [])
        close = inputs.get("close_prices", [])
        volume = inputs.get("volume", [])

        if not all([high, low, close, volume]) or len(high) < 2:
            return self._neutral()

        bars = list(zip(high, low, close, volume))
        ratio = self._get_ratio(self.source_tf, self.target_tf)
        if len(bars) < ratio:
            return self._neutral()

        # Get the last complete group of `ratio` bars for the new timeframe
        last_group = bars[-ratio:]

        resampled_close = [b[2] for b in last_group]
        resampled_volume = [b[3] for b in last_group]

        close_key = f"resampled_{self.target_tf}_close"
        vol_key = f"resampled_{self.target_tf}_volume"

        features = {
            close_key: resampled_close[-1] if resampled_close else 0.0,
            vol_key: sum(resampled_volume) if resampled_volume else 0.0,
        }

        confidences = {
            k: FeatureConfidence(source="library_transform", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")
            for k in features
        }

        return FeatureVector(timestamp=0.0, bot_id="", features=features, feature_confidence=confidences)

    def _neutral(self) -> FeatureVector:
        close_key = f"resampled_{self.target_tf}_close"
        vol_key = f"resampled_{self.target_tf}_volume"
        features = {close_key: 0.0, vol_key: 0.0}
        confidences = {
            k: FeatureConfidence(source="library_transform", quality=0.0, latency_ms=0, feed_quality_tag="INSUFFICIENT_DATA")
            for k in features
        }
        return FeatureVector(timestamp=0.0, bot_id="", features=features, feature_confidence=confidences)
