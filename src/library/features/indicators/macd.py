"""MACD — Moving Average Convergence Divergence"""
from __future__ import annotations

from typing import Any, Dict, Set
from pydantic import Field

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class MACDFeature(FeatureModule):
    """
    Computes MACD (12, 26, 9) — line, signal, histogram.

    Quality class: native_supported (computed from OHLCV close prices)
    """
    fast_period: int = Field(default=12, ge=2)
    slow_period: int = Field(default=26, ge=2)
    signal_period: int = Field(default=9, ge=1)

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="indicators/macd",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"close_prices"}

    @property
    def output_keys(self) -> Set[str]:
        return {"macd_line", "macd_signal", "macd_histogram"}

    @staticmethod
    def _ema(data: list[float], period: int) -> float:
        if not data:
            return 0.0
        multiplier = 2.0 / (period + 1)
        ema = data[0]
        for value in data[1:]:
            ema = (value - ema) * multiplier + ema
        return ema

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        closes = inputs["close_prices"]
        min_required = max(self.slow_period, self.signal_period) + 1

        if not closes or len(closes) < min_required + 1:
            return FeatureVector(
                timestamp=0.0,
                bot_id="",
                features={"macd_line": 0.0, "macd_signal": 0.0, "macd_histogram": 0.0},
                feature_confidence={
                    k: FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    )
                    for k in ["macd_line", "macd_signal", "macd_histogram"]
                },
            )

        fast_ema = self._ema(closes, self.fast_period)
        slow_ema = self._ema(closes, self.slow_period)
        macd_line = fast_ema - slow_ema

        # Signal EMA on MACD line (simplified — use last N MACD values)
        macd_vals = []
        for i in range(self.slow_period, len(closes)):
            f = self._ema(closes[: i + 1], self.fast_period)
            s = self._ema(closes[: i + 1], self.slow_period)
            macd_vals.append(f - s)

        signal_line = (
            self._ema(macd_vals[-self.signal_period :], self.signal_period)
            if len(macd_vals) >= self.signal_period
            else macd_line
        )
        histogram = macd_line - signal_line

        quality = min(1.0, (len(closes) - min_required) / (min_required * 5))
        fc = FeatureConfidence(
            source="ctrader_native",
            quality=quality,
            latency_ms=0,
            feed_quality_tag="HIGH",
        )

        return FeatureVector(
            timestamp=0.0,
            bot_id="",
            features={
                "macd_line": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            feature_confidence={k: fc for k in ["macd_line", "macd_signal", "macd_histogram"]},
        )
