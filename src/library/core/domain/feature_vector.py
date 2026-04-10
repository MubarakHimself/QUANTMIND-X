"""
QuantMindLib V1 — FeatureVector Schema
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.library.core.domain.market_context import MarketContext


class FeatureConfidence(BaseModel):
    """Quality metadata for a computed feature value."""

    source: str = "ctrader_native"  # Data source identifier
    quality: float = Field(default=0.0, ge=0.0, le=1.0)  # 0.0–1.0 confidence in data
    latency_ms: float = Field(default=0.0, ge=0.0)  # data freshness
    feed_quality_tag: str = "UNKNOWN"  # Quality tier: HIGH | MEDIUM | LOW | INSUFFICIENT_DATA | STALE

    model_config = BaseModel.model_config


class FeatureVector(BaseModel):
    """Computed feature values at a point in time — output of the feature evaluation pipeline."""

    bot_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    features: Dict[str, float] = Field(default_factory=dict)  # feature_id → computed value
    feature_confidence: Dict[str, FeatureConfidence] = Field(default_factory=dict)  # feature_id → quality metadata
    market_context_snapshot: Optional[MarketContext] = None

    model_config = BaseModel.model_config

    def get(self, key: str, default: float = 0.0) -> float:
        """Return feature value, or default if not present."""
        return self.features.get(key, default)

    def has_confidence(self, key: str) -> bool:
        """Return True if a confidence entry exists for the given feature key."""
        return key in self.feature_confidence

    def quality_score(self, key: str) -> float:
        """Return quality score for a feature, or 0.0 if not labeled."""
        if key in self.feature_confidence:
            return self.feature_confidence[key].quality
        return 0.0

    def weighted_value(self, key: str) -> float:
        """Return quality-weighted feature value: value * quality. Returns 0.0 if absent."""
        value = self.features.get(key)
        if value is None:
            return 0.0
        return value * self.quality_score(key)

    def merge(self, other: "FeatureVector") -> "FeatureVector":
        """Merge another FeatureVector, combining features and taking higher quality on conflict."""
        merged_features = dict(self.features)
        merged_features.update(other.features)
        merged_confidence = dict(self.feature_confidence)
        for k, v in other.feature_confidence.items():
            if k not in merged_confidence or v.quality > merged_confidence[k].quality:
                merged_confidence[k] = v
        return FeatureVector(
            bot_id=self.bot_id or other.bot_id,
            timestamp=self.timestamp or other.timestamp,
            features=merged_features,
            feature_confidence=merged_confidence,
        )

    __all__ = ["FeatureVector", "FeatureConfidence"]
