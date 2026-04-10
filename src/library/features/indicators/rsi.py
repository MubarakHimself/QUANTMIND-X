"""RSI — Relative Strength Index"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class RSIFeature(FeatureModule):
    """
    Computes Relative Strength Index over a price series.

    Quality class: native_supported (computed from OHLCV close prices)
    """
    period: int = Field(default=14, ge=2, le=200)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"indicators/rsi_{self.period}",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"close_prices"}

    @property
    def output_keys(self) -> Set[str]:
        return {f"rsi_{self.period}"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        closes = inputs["close_prices"]
        if not closes or len(closes) < self.period + 1:
            # Return neutral signal if insufficient data
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={f"rsi_{self.period}": 50.0},
                feature_confidence={
                    f"rsi_{self.period}": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    )
                },
            )

        # Wilder's RSI
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas[-self.period :]]
        losses = [-d if d < 0 else 0 for d in deltas[-self.period :]]

        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        quality = min(1.0, (len(closes) - self.period) / (self.period * 10))

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={f"rsi_{self.period}": rsi},
            feature_confidence={
                f"rsi_{self.period}": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag="HIGH",
                )
            },
        )
