"""VWAP — Volume-Weighted Average Price"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class VWAPFeature(FeatureModule):
    """
    Computes Volume-Weighted Average Price over a bar series.

    Quality class: native_supported (computed from OHLCV bars + volume)
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="indicators/vwap",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"high", "low", "close_prices", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"vwap", "vwap_distance"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        high = inputs["high"]
        low = inputs["low"]
        close = inputs["close_prices"]
        volume = inputs["volume"]

        if not all([high, low, close, volume]) or len(high) < 1:
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={"vwap": 0.0, "vwap_distance": 0.0},
                feature_confidence={
                    k: FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    )
                    for k in ["vwap", "vwap_distance"]
                },
            )

        # VWAP = sum(typical_price * volume) / sum(volume)
        typical_prices = [(h + l + c) / 3.0 for h, l, c in zip(high, low, close)]
        total_pv = sum(tp * v for tp, v in zip(typical_prices, volume))
        total_vol = sum(volume)

        if total_vol == 0:
            vwap = typical_prices[-1]
        else:
            vwap = total_pv / total_vol

        current_price = close[-1] if close else 0.0
        vwap_distance = (current_price - vwap) / vwap if vwap != 0 else 0.0

        quality = min(1.0, len(high) / 50.0)
        fc = FeatureConfidence(
            source="ctrader_native",
            quality=quality,
            latency_ms=0.0,
            feed_quality_tag="HIGH",
        )

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={"vwap": vwap, "vwap_distance": vwap_distance},
            feature_confidence={k: fc for k in ["vwap", "vwap_distance"]},
        )
