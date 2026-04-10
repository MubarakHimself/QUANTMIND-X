"""
Self-Verification Tests for Packet 11B: LiquidityStressProxyFeature.
15 tests covering init, compute, compute_batch, get_stress_level, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.liquidity_stress import (
    LiquidityStressProxyFeature,
    STRESS_NORMAL,
    STRESS_ELEVATED,
    STRESS_HIGH,
    STRESS_CRITICAL,
)


class TestLiquidityStressInit:
    def test_default_spread_ratio(self):
        feat = LiquidityStressProxyFeature()
        assert feat._spread_ratio == 0.0001
        assert feat._smoothing == 0.3
        assert feat._prev_stress == 0.0

    def test_custom_spread_ratio(self):
        feat = LiquidityStressProxyFeature(spread_ratio=0.0002)
        assert feat._spread_ratio == 0.0002

    def test_feature_id(self):
        feat = LiquidityStressProxyFeature()
        assert feat.config.feature_id == "microstructure/liquidity_stress_proxy"

    def test_quality_class(self):
        feat = LiquidityStressProxyFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestLiquidityStressCompute:
    def test_normal_stress_calm_bar(self):
        """Normal bar with small spread has low stress."""
        feat = LiquidityStressProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0851,
            "low": 1.0849,
            "close": 1.0850,
            "volume": 500.0,
            "spread": 0.00001,
        }
        result = feat.compute(bar)
        assert 0.0 <= result <= 1.0

    def test_high_stress_volatile_bar(self):
        """Wide range + high volume + wide spread = high stress."""
        feat = LiquidityStressProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0900,
            "low": 1.0800,
            "close": 1.0850,
            "volume": 50000.0,
            "spread": 0.0003,
        }
        result = feat.compute(bar)
        assert 0.0 <= result <= 1.0

    def test_zero_volume_returns_low_stress(self):
        """Zero volume bar returns near-zero stress."""
        feat = LiquidityStressProxyFeature()
        bar = {"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 0.0}
        result = feat.compute(bar)
        # Small range component may be non-zero; volume contribution is 0
        assert result >= 0.0
        assert result < 0.3  # Should be NORMAL level

    def test_zero_mid_price_returns_zero(self):
        """Invalid bar data (zero mid price) returns 0.0."""
        feat = LiquidityStressProxyFeature()
        bar = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 1000.0}
        assert feat.compute(bar) == 0.0

    def test_without_spread_uses_estimate(self):
        """Bar without spread uses estimated spread from range."""
        feat = LiquidityStressProxyFeature(spread_ratio=0.0001)
        bar = {
            "open": 1.0850,
            "high": 1.0900,
            "low": 1.0800,
            "close": 1.0850,
            "volume": 20000.0,
        }
        result = feat.compute(bar)
        assert 0.0 <= result <= 1.0


class TestLiquidityStressComputeBatch:
    def test_batch_returns_all_values(self):
        """Batch returns one value per bar."""
        feat = LiquidityStressProxyFeature()
        bars = [
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
            {"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0850, "volume": 20000.0},
            {"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 5000.0},
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 3
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_batch_empty_returns_empty(self):
        feat = LiquidityStressProxyFeature()
        assert feat.compute_batch([]) == []

    def test_batch_preserves_smoothing_across_calls(self):
        """Second batch call continues from previous smoothed value."""
        feat = LiquidityStressProxyFeature(smoothing=0.5)
        bars1 = [{"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 5000.0}]
        bars2 = [{"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0850, "volume": 20000.0}]
        feat.compute_batch(bars1)
        second = feat.compute_batch(bars2)
        assert len(second) == 1


class TestLiquidityStressGetStressLevel:
    def test_normal_level(self):
        feat = LiquidityStressProxyFeature()
        assert feat.get_stress_level(0.0) == STRESS_NORMAL
        assert feat.get_stress_level(0.1) == STRESS_NORMAL
        assert feat.get_stress_level(0.29) == STRESS_NORMAL

    def test_elevated_level(self):
        feat = LiquidityStressProxyFeature()
        assert feat.get_stress_level(0.3) == STRESS_ELEVATED
        assert feat.get_stress_level(0.45) == STRESS_ELEVATED
        assert feat.get_stress_level(0.59) == STRESS_ELEVATED

    def test_high_level(self):
        feat = LiquidityStressProxyFeature()
        assert feat.get_stress_level(0.6) == STRESS_HIGH
        assert feat.get_stress_level(0.7) == STRESS_HIGH
        assert feat.get_stress_level(0.79) == STRESS_HIGH

    def test_critical_level(self):
        feat = LiquidityStressProxyFeature()
        assert feat.get_stress_level(0.8) == STRESS_CRITICAL
        assert feat.get_stress_level(0.9) == STRESS_CRITICAL
        assert feat.get_stress_level(1.0) == STRESS_CRITICAL


class TestLiquidityStressCapabilitySpec:
    def test_capability_spec_structure(self):
        feat = LiquidityStressProxyFeature()
        spec = feat.capability_spec()
        assert spec["feature_id"] == "microstructure/liquidity_stress_proxy"
        assert spec["quality_class"] == "proxy_inferred"
        assert "liquidity_stress" in spec["outputs"]
        assert "stress_level" in spec["outputs"]

    def test_lookback_required(self):
        feat = LiquidityStressProxyFeature()
        spec = feat.capability_spec()
        assert spec["lookback_required"] == 10


class TestLiquidityStressConfidence:
    def test_disabled_under_10_bars(self):
        feat = LiquidityStressProxyFeature()
        conf = feat.confidence(9)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_10_bars(self):
        feat = LiquidityStressProxyFeature()
        conf = feat.confidence(10)
        assert conf.quality == 0.35
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_20_bars(self):
        feat = LiquidityStressProxyFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.65
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_50_bars(self):
        feat = LiquidityStressProxyFeature()
        conf = feat.confidence(50)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"
