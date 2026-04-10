"""SessionBlackoutFeature — news blackout detection"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class SessionBlackoutFeature(FeatureModule):
    """
    Detects news blackout periods based on news_state input.
    Maps NewsState to is_blackout signal for session-aware trading.

    Quality class: native_supported (wraps NewsSensor via news_state input)
    """
    kill_zone_minutes: int = Field(default=15, ge=1, le=60)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="session/blackout",
            quality_class="native_supported",
            source="news_sensor",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"news_state", "minutes_to_event"}

    @property
    def output_keys(self) -> set[str]:
        return {"is_blackout", "blackout_severity", "kill_zone_active"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        news_state = inputs.get("news_state", "CLEAR")
        minutes_to_event = inputs.get("minutes_to_event")

        if news_state == "ACTIVE":
            severity = 1.0
            is_blackout = 1.0
        elif news_state == "KILL_ZONE":
            severity = 0.7
            is_blackout = 1.0
        else:
            severity = 0.0
            is_blackout = 0.0

        kill_zone = 0.0
        if minutes_to_event is not None and minutes_to_event <= self.kill_zone_minutes:
            kill_zone = 1.0 - (minutes_to_event / self.kill_zone_minutes)
            is_blackout = max(is_blackout, kill_zone)

        features = {
            "is_blackout": is_blackout,
            "blackout_severity": severity,
            "kill_zone_active": kill_zone,
        }

        return FeatureVector(
            timestamp=0.0,
            bot_id="",
            features=features,
            feature_confidence={
                k: FeatureConfidence(source="news_sensor", quality=0.9, latency_ms=0, feed_quality_tag="HIGH")
                for k in features
            },
        )
