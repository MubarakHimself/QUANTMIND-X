"""
QuantMindLib V1 — FeatureVector Schema
CONTRACT-017
Sourced from: docs/planning/quantmindlib/07_shared_object_model.md §17
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class FeatureConfidence(BaseModel):
    """
    Quality metadata for a computed feature value.

    Provides source attribution, quality score (0-1), latency, and feed tag
    to allow downstream consumers to weight or filter features by reliability.
    """
    source: str = "ctrader_native"          # Data source identifier
    quality: float = Field(default=0.0, ge=0.0, le=1.0)  # 0=unreliable, 1=canonical
    latency_ms: float = Field(default=0.0, ge=0.0) # Compute/data latency in milliseconds
    feed_quality_tag: str = "UNKNOWN"        # Quality tier: HIGH | MEDIUM | LOW | INSUFFICIENT_DATA | STALE

    model_config = BaseModel.model_config


class FeatureVector(BaseModel):
    """
    Computed feature values at a point in time — output of the feature evaluation pipeline.

    A FeatureVector carries one or more computed feature values along with
    per-feature quality metadata. It is the standard output contract for all
    QuantMindLib feature modules.

    Design:
    - Timestamped: each vector carries a datetime for ordering
    - Typed: every feature value is a float; every quality entry is a FeatureConfidence
    - Feed-aware: feed_quality_tag propagates upstream data quality into the feature layer
    - Market-contextual: optional market regime/context snapshot for downstream gating
    """
    bot_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)
    features: Dict[str, float] = Field(default_factory=dict)   # feature_key → value
    feature_confidence: Dict[str, FeatureConfidence] = Field(default_factory=dict)
    market_context_snapshot: Optional[Any] = None  # Optional[MarketContext] at runtime; Any avoids import cycle

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
        """
        Return quality-weighted feature value: value * quality.
        Returns 0.0 if feature or quality is absent.
        """
        value = self.features.get(key)
        if value is None:
            return 0.0
        quality = self.quality_score(key)
        return value * quality

    def merge(self, other: "FeatureVector") -> "FeatureVector":
        """
        Merge another FeatureVector into this one, combining features and
        taking the higher quality score on conflict.
        """
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
