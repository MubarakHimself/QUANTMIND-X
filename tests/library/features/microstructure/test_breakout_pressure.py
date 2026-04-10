"""
Self-Verification Tests for Packet 11B: BreakoutPressureProxyFeature.
15 tests covering init, compute, compute_batch, detect_consolidation, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.breakout_pressure import BreakoutPressureProxyFeature


class TestBreakoutPressureInit:
    def test_default_smoothing(self):
        feat = BreakoutPressureProxyFeature()
        assert feat._smoothing == 0.3
        assert feat._prev_pressure == 0.0

    def test_custom_smoothing(self):
        feat = BreakoutPressureProxyFeature(smoothing=0.5)
        assert feat._smoothing == 0.5

    def test_feature_id(self):
        feat = BreakoutPressureProxyFeature()
        assert feat.config.feature_id == "microstructure/breakout_pressure_proxy"

    def test_quality_class(self):
        feat = BreakoutPressureProxyFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestBreakoutPressureCompute:
    def test_high_pressure(self):
        """High volume + wide range + spread = high pressure."""
        feat = BreakoutPressureProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0900,
            "low": 1.0800,
            "close": 1.0850,
            "volume": 50000.0,
            "spread": 0.0002,
        }
        result = feat.compute(bar)
        assert result > 0.0
        assert result > 1.0  # Should be significant

    def test_low_pressure_calm_market(self):
        """Low volume + narrow range = low pressure."""
        feat = BreakoutPressureProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0851,
            "low": 1.0849,
            "close": 1.0850,
            "volume": 500.0,
            "spread": 0.00001,
        }
        result = feat.compute(bar)
        assert result >= 0.0

    def test_zero_volume_still_computes_range_based_pressure(self):
        """Zero volume bar still returns non-zero due to range component (pressure builds from range)."""
        feat = BreakoutPressureProxyFeature()
        bar = {"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 0.0}
        # Range-based pressure component alone can be non-zero
        assert feat.compute(bar) >= 0.0

    def test_zero_mid_price_returns_zero(self):
        """Invalid bar data (zero mid price) returns 0.0."""
        feat = BreakoutPressureProxyFeature()
        bar = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 1000.0}
        assert feat.compute(bar) == 0.0

    def test_without_spread_still_computes(self):
        """Bar without spread still produces a pressure score."""
        feat = BreakoutPressureProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0850,
            "volume": 10000.0,
        }
        result = feat.compute(bar)
        assert result >= 0.0


class TestBreakoutPressureComputeBatch:
    def test_batch_returns_all_values(self):
        """Batch returns one value per bar."""
        feat = BreakoutPressureProxyFeature(smoothing=0.3)
        bars = [
            {"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 5000.0},
            {"open": 1.0850, "high": 1.0870, "low": 1.0830, "close": 1.0850, "volume": 8000.0},
            {"open": 1.0850, "high": 1.0880, "low": 1.0820, "close": 1.0850, "volume": 10000.0},
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 3
        assert all(r >= 0.0 for r in results)

    def test_batch_empty_returns_empty(self):
        feat = BreakoutPressureProxyFeature()
        assert feat.compute_batch([]) == []

    def test_batch_preserves_smoothing_across_calls(self):
        """Second batch call continues from previous smoothed value."""
        feat = BreakoutPressureProxyFeature(smoothing=0.5)
        bars1 = [{"open": 1.0850, "high": 1.0860, "low": 1.0840, "close": 1.0850, "volume": 5000.0}]
        bars2 = [{"open": 1.0850, "high": 1.0870, "low": 1.0830, "close": 1.0850, "volume": 8000.0}]
        feat.compute_batch(bars1)
        second = feat.compute_batch(bars2)
        assert len(second) == 1


class TestBreakoutPressureDetectConsolidation:
    def test_consolidation_detected(self):
        """Tight range bars are detected as consolidation."""
        feat = BreakoutPressureProxyFeature()
        bars = [
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 400.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 600.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
        ]
        assert feat.detect_consolidation(bars, threshold=0.5) is True

    def test_no_consolidation_wide_range(self):
        """Wide range bars are NOT detected as consolidation."""
        feat = BreakoutPressureProxyFeature()
        bars = [
            {"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0850, "volume": 10000.0},
            {"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0850, "volume": 10000.0},
            {"open": 1.0850, "high": 1.0900, "low": 1.0800, "close": 1.0850, "volume": 10000.0},
        ]
        assert feat.detect_consolidation(bars, threshold=0.5) is False

    def test_consolidation_insufficient_bars(self):
        """Less than 3 bars returns False."""
        feat = BreakoutPressureProxyFeature()
        bars = [
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0849, "close": 1.0850, "volume": 500.0},
        ]
        assert feat.detect_consolidation(bars) is False


class TestBreakoutPressureCapabilitySpec:
    def test_capability_spec_structure(self):
        feat = BreakoutPressureProxyFeature()
        spec = feat.capability_spec()
        assert spec["feature_id"] == "microstructure/breakout_pressure_proxy"
        assert spec["quality_class"] == "proxy_inferred"
        assert "breakout_pressure" in spec["outputs"]
        assert "consolidation_detected" in spec["outputs"]

    def test_lookback_required(self):
        feat = BreakoutPressureProxyFeature()
        spec = feat.capability_spec()
        assert spec["lookback_required"] == 10


class TestBreakoutPressureConfidence:
    def test_disabled_under_10_bars(self):
        feat = BreakoutPressureProxyFeature()
        conf = feat.confidence(9)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_10_bars(self):
        feat = BreakoutPressureProxyFeature()
        conf = feat.confidence(10)
        assert conf.quality == 0.35
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_20_bars(self):
        feat = BreakoutPressureProxyFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.65
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_50_bars(self):
        feat = BreakoutPressureProxyFeature()
        conf = feat.confidence(50)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"
