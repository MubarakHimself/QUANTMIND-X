"""NormalizeTransform — min-max or z-score normalization"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class NormalizeTransform(FeatureModule):
    """
    Normalizes a numeric series using min-max or z-score.
    Operates on a single named input key.

    Quality class: native_supported
    """
    input_key: str = Field(default="value")
    method: str = Field(default="minmax")  # "minmax" | "zscore"

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"transforms/normalize_{self.input_key}",
            quality_class="native_supported",
            source="library_transform",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {self.input_key}

    @property
    def output_keys(self) -> set[str]:
        return {f"normalized_{self.input_key}"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        series = inputs.get(self.input_key)
        if not series or len(series) < 2:
            return self._neutral()

        vals = list(series)
        if self.method == "zscore":
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = variance ** 0.5
            if std == 0:
                normalized = 0.0
            else:
                normalized = (vals[-1] - mean) / std
        else:  # minmax
            mn, mx = min(vals), max(vals)
            if mx == mn:
                normalized = 0.0
            else:
                normalized = (vals[-1] - mn) / (mx - mn)

        key = f"normalized_{self.input_key}"
        return FeatureVector(
            timestamp=0.0,
            bot_id="",
            features={key: normalized},
            feature_confidence={
                key: FeatureConfidence(source="library_transform", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")
            },
        )

    def _neutral(self) -> FeatureVector:
        key = f"normalized_{self.input_key}"
        return FeatureVector(
            timestamp=0.0, bot_id="",
            features={key: 0.0},
            feature_confidence={key: FeatureConfidence(source="library_transform", quality=0.0, latency_ms=0, feed_quality_tag="INSUFFICIENT_DATA")},
        )
