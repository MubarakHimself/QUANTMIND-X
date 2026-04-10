"""RollingWindowTransform — rolling statistics"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class RollingWindowTransform(FeatureModule):
    """
    Computes rolling statistics over a window of a numeric series.
    Supports: mean, std, min, max.

    Quality class: native_supported
    """
    input_key: str = Field(default="value")
    window_size: int = Field(default=20, ge=2, le=500)
    stats: list[str] = Field(default_factory=lambda: ["mean", "std"])

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"transforms/rolling_{self.input_key}_{self.window_size}",
            quality_class="native_supported",
            source="library_transform",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {self.input_key}

    @property
    def output_keys(self) -> set[str]:
        prefix = f"rolling_{self.input_key}_{self.window_size}"
        return {f"{prefix}_{s}" for s in self.stats}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        series = inputs.get(self.input_key)
        if not series or len(series) < self.window_size:
            return self._neutral()

        window = list(series)[-self.window_size:]
        features = {}
        confidences = {}
        prefix = f"rolling_{self.input_key}_{self.window_size}"

        if "mean" in self.stats:
            k = f"{prefix}_mean"
            features[k] = sum(window) / len(window)
            confidences[k] = FeatureConfidence(source="library_transform", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")

        if "std" in self.stats:
            k = f"{prefix}_std"
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            features[k] = variance ** 0.5
            confidences[k] = FeatureConfidence(source="library_transform", quality=0.9, latency_ms=0, feed_quality_tag="HIGH")

        if "min" in self.stats:
            k = f"{prefix}_min"
            features[k] = min(window)
            confidences[k] = FeatureConfidence(source="library_transform", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")

        if "max" in self.stats:
            k = f"{prefix}_max"
            features[k] = max(window)
            confidences[k] = FeatureConfidence(source="library_transform", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")

        return FeatureVector(timestamp=0.0, bot_id="", features=features, feature_confidence=confidences)

    def _neutral(self) -> FeatureVector:
        features = {f"rolling_{self.input_key}_{self.window_size}_{s}": 0.0 for s in self.stats}
        confidences = {
            k: FeatureConfidence(source="library_transform", quality=0.0, latency_ms=0, feed_quality_tag="INSUFFICIENT_DATA")
            for k in features
        }
        return FeatureVector(timestamp=0.0, bot_id="", features=features, feature_confidence=confidences)
