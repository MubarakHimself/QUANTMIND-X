"""ATR — Average True Range"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class ATRFeature(FeatureModule):
    """
    Computes Average True Range over OHLCV bars.
    Returns ATR value in pips (assumes 10k/100k pip convention for FX).

    Quality class: native_supported (computed from OHLCV bars)
    """
    period: int = Field(default=14, ge=2, le=200)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"indicators/atr_{self.period}",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"high", "low", "close_prices"}

    @property
    def output_keys(self) -> Set[str]:
        return {f"atr_{self.period}"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        high = inputs["high"]
        low = inputs["low"]
        close = inputs["close_prices"]

        if (
            not all([high, low, close])
            or len(high) < self.period + 1
        ):
            return FeatureVector(
                timestamp=0.0,
                bot_id="",
                features={f"atr_{self.period}": 0.0},
                feature_confidence={
                    f"atr_{self.period}": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    )
                },
            )

        # True Range = max(H - L, |H - prev_close|, |L - prev_close|)
        trs = []
        for i in range(1, len(high)):
            h, l, c = high[i], low[i], close[i - 1]
            tr = max(h - l, abs(h - c), abs(l - c))
            # Convert to pips (FX convention: 4 decimal places = pip)
            trs.append(tr / 0.0001)

        if len(trs) < self.period:
            return FeatureVector(
                timestamp=0.0,
                bot_id="",
                features={f"atr_{self.period}": 0.0},
                feature_confidence={
                    f"atr_{self.period}": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    )
                },
            )

        # Wilder's smoothing
        atr = sum(trs[-self.period :]) / self.period
        quality = min(1.0, (len(trs) - self.period) / (self.period * 10))

        return FeatureVector(
            timestamp=0.0,
            bot_id="",
            features={f"atr_{self.period}": atr},
            feature_confidence={
                f"atr_{self.period}": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0,
                    feed_quality_tag="HIGH",
                )
            },
        )
