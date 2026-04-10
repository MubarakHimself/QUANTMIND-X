"""RVOLFeature — Relative Volume"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class RVOLFeature(FeatureModule):
    """
    Computes Relative Volume — current volume normalised against expected volume
    for the active session and day of week.

    Expected volume = 1.0 * session_factor * day_factor
      session_factor: LONDON/LONDON_NY_OVERLAP → 2.0
                    LONDON_AM/NY_AM → 1.5
                    others → 1.0
      day_factor:   Wed → 1.2, others → 1.0

    Inputs required: volume, session_id, day_of_week, time_of_day_minutes
    Outputs:        rvol, rvol_category
    """
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="volume/rvol",
            quality_class="native_supported",
            source="RVOLFeature",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"volume", "session_id", "day_of_week", "time_of_day_minutes"}

    @property
    def output_keys(self) -> set[str]:
        return {"rvol", "rvol_category"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        volume = inputs.get("volume")
        session_id = inputs.get("session_id", "")
        day_of_week = inputs.get("day_of_week", 0)  # 0=Mon, 6=Sun

        if volume is None or volume <= 0:
            return FeatureVector(
                bot_id="SYSTEM",
                features={"rvol": 1.0, "rvol_category": 0.0},
                feature_confidence={
                    "rvol": FeatureConfidence(
                        source="RVOLFeature",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="NO_DATA",
                    ),
                    "rvol_category": FeatureConfidence(
                        source="RVOLFeature",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="NO_DATA",
                    ),
                },
            )

        # Session factor lookup
        session_factor_map = {
            "LONDON": 2.0,
            "LONDON_NY_OVERLAP": 2.0,
            "LONDON_AM": 1.5,
            "NY_AM": 1.5,
        }
        session_factor = session_factor_map.get(session_id, 1.0)

        # Day-of-week factor lookup (0=Mon)
        day_map = {0: "MON", 1: "TUE", 2: "WED", 3: "THU", 4: "FRI", 5: "SAT", 6: "SUN"}
        day_key = day_map.get(day_of_week, "MON")
        day_factor_map = {
            "MON": 1.0,
            "TUE": 1.0,
            "WED": 1.2,
            "THU": 1.0,
            "FRI": 1.0,
            "SAT": 1.0,
            "SUN": 1.0,
        }
        day_factor = day_factor_map.get(day_key, 1.0)

        expected = 1.0 * session_factor * day_factor
        rvol = volume / expected if expected > 0 else 1.0

        if rvol >= 2.0:
            category = "HIGH"
        elif rvol <= 0.5:
            category = "LOW"
        else:
            category = "NORMAL"

        # Encode category as float: HIGH=1.0, LOW=-1.0, NORMAL=0.0
        category_value = {"HIGH": 1.0, "LOW": -1.0, "NORMAL": 0.0}[category]

        return FeatureVector(
            bot_id="SYSTEM",
            features={"rvol": rvol, "rvol_category": category_value},
            feature_confidence={
                "rvol": FeatureConfidence(
                    source="RVOLFeature",
                    quality=0.85,
                    latency_ms=0.0,
                    feed_quality_tag="HIGH",
                ),
                "rvol_category": FeatureConfidence(
                    source="RVOLFeature",
                    quality=0.85,
                    latency_ms=0.0,
                    feed_quality_tag="HIGH",
                ),
            },
        )
