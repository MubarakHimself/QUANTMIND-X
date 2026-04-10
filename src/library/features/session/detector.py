"""SessionDetectorFeature — canonical session detection from server time"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class SessionDetectorFeature(FeatureModule):
    """
    Detects which canonical trading session is currently active from server time.
    Maps to the 10 canonical session windows (LONDON_AM, NY_AM, etc.).

    Quality class: native_supported
    """
    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="session/detector",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"server_time"}

    @property
    def output_keys(self) -> set[str]:
        return {"session_id", "session_active", "currencies_covered"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        server_time = inputs.get("server_time")
        if server_time is None:
            return self._neutral()

        if isinstance(server_time, (int, float)):
            # Unix timestamp in ms or seconds
            if server_time > 1e12:
                server_time = server_time / 1000.0
            dt = datetime.fromtimestamp(server_time)
        elif isinstance(server_time, datetime):
            dt = server_time
        else:
            return self._neutral()

        hour = dt.hour
        minute = dt.minute
        current_minutes = hour * 60 + minute

        sessions = [
            ("sydney", "22:00", "07:00"),
            ("tokyo", "00:00", "09:00"),
            ("london", "08:00", "17:00"),
            ("new_york", "13:00", "22:00"),
            ("london_ny_overlap", "13:00", "17:00"),
        ]

        active_sessions = []
        for name, start, end in sessions:
            sh, sm = map(int, start.split(":"))
            eh, em = map(int, end.split(":"))
            start_mins = sh * 60 + sm
            end_mins = eh * 60 + em

            if start_mins < end_mins:
                is_active = start_mins <= current_minutes < end_mins
            else:
                # Wraps midnight
                is_active = current_minutes >= start_mins or current_minutes < end_mins

            if is_active:
                session_key = name.upper()
                if name == "london_ny_overlap":
                    session_key = "LONDON_NY_OVERLAP"
                active_sessions.append(session_key)

        primary = active_sessions[0] if active_sessions else "NO_SESSION"

        currencies = []
        if primary in ("SYDNEY", "TOKYO"):
            currencies = ["JPY", "AUD", "NZD", "CNY"]
        elif primary in ("LONDON", "LONDON_NY_OVERLAP"):
            currencies = ["EUR", "GBP", "CHF"]
        elif primary == "NEW_YORK":
            currencies = ["USD", "CAD"]

        features = {
            "session_id": float(hash(primary) % 1000) / 1000.0,  # normalized
            "session_active": 1.0 if active_sessions else 0.0,
            "currencies_covered": len(currencies) / 5.0,  # normalized
        }

        return FeatureVector(
            timestamp=0.0,
            bot_id="",
            features=features,
            feature_confidence={
                k: FeatureConfidence(source="ctrader_native", quality=0.95, latency_ms=0, feed_quality_tag="HIGH")
                for k in features
            },
        )

    def _neutral(self) -> FeatureVector:
        return FeatureVector(
            timestamp=0.0, bot_id="",
            features={"session_id": 0.0, "session_active": 0.0, "currencies_covered": 0.0},
            feature_confidence={
                k: FeatureConfidence(source="ctrader_native", quality=0.0, latency_ms=0, feed_quality_tag="NO_DATA")
                for k in ["session_id", "session_active", "currencies_covered"]
            },
        )
