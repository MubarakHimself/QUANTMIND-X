"""MFIFeature — Money Flow Index"""
from __future__ import annotations

from typing import Any, Dict, List, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class MFIFeature(FeatureModule):
    """
    Computes Money Flow Index — volume-weighted RSI analogue.
    Range 0–100. Default period=14.

    Inputs required: high_prices, low_prices, close_prices, volumes
    Outputs:        mfi_{period}, mfi_signal
    """
    period: int = Field(default=14, ge=2, le=200)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id=f"volume/mfi_{self.period}",
            quality_class="native_supported",
            source="MFIFeature",
        )

    @property
    def required_inputs(self) -> set[str]:
        return {"high_prices", "low_prices", "close_prices", "volumes"}

    @property
    def output_keys(self) -> set[str]:
        return {f"mfi_{self.period}", "mfi_signal"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        high_prices = inputs.get("high_prices", [])
        low_prices = inputs.get("low_prices", [])
        close_prices = inputs.get("close_prices", [])
        volumes = inputs.get("volumes", [])

        n = len(close_prices)

        if n < self.period + 1:
            return self._insufficient_data()

        # Compute typical prices
        typical_prices: List[float] = [
            (h + l + c) / 3.0
            for h, l, c in zip(high_prices, low_prices, close_prices)
        ]

        # Compute raw money flow
        raw_money_flow: List[float] = [
            tp * v for tp, v in zip(typical_prices, volumes)
        ]

        # Accumulate positive/negative money flow over the period
        positive_flow = 0.0
        negative_flow = 0.0
        for i in range(self.period, n):
            if typical_prices[i] > typical_prices[i - 1]:
                positive_flow += raw_money_flow[i]
            elif typical_prices[i] < typical_prices[i - 1]:
                negative_flow += raw_money_flow[i]

        if negative_flow == 0:
            mfi = 100.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi = 100.0 - (100.0 / (1.0 + money_ratio))

        # Determine signal
        if mfi >= 80:
            signal = "OVERBOUGHT"
        elif mfi <= 20:
            signal = "OVERSOLD"
        else:
            signal = "NEUTRAL"

        quality = min(1.0, (n - self.period) / (self.period * 10))
        quality_tag = "HIGH" if quality >= 0.7 else "MEDIUM"

        return FeatureVector(
            bot_id="SYSTEM",
            features={
                f"mfi_{self.period}": mfi,
                "mfi_signal": float(1.0),  # signal encoded as float for consistency
            },
            feature_confidence={
                f"mfi_{self.period}": FeatureConfidence(
                    source="MFIFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "mfi_signal": FeatureConfidence(
                    source="MFIFeature",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
            },
        )

    def _insufficient_data(self) -> FeatureVector:
        return FeatureVector(
            bot_id="SYSTEM",
            features={f"mfi_{self.period}": 50.0, "mfi_signal": float(1.0)},
            feature_confidence={
                f"mfi_{self.period}": FeatureConfidence(
                    source="MFIFeature",
                    quality=0.0,
                    latency_ms=0.0,
                    feed_quality_tag="INSUFFICIENT_DATA",
                ),
                "mfi_signal": FeatureConfidence(
                    source="MFIFeature",
                    quality=0.0,
                    latency_ms=0.0,
                    feed_quality_tag="INSUFFICIENT_DATA",
                ),
            },
        )
