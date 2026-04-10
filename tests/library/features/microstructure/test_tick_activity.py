"""
Self-Verification Tests for Packet 11A: TickActivityFeature.
15 tests covering init, compute, compute_batch, get_activity_level, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.tick_activity import TickActivityFeature


class TestTickActivityInit:
    def test_default_expected_ticks(self):
        feat = TickActivityFeature()
        assert feat._expected_ticks == 50

    def test_custom_expected_ticks(self):
        feat = TickActivityFeature(expected_ticks_per_bar=100)
        assert feat._expected_ticks == 100

    def test_feature_id(self):
        feat = TickActivityFeature()
        assert feat.config.feature_id == "microstructure/tick_activity"

    def test_quality_class(self):
        feat = TickActivityFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestTickActivityCompute:
    def test_normal_activity_with_tick_count(self):
        """tick_count == expected_ticks returns activity = 1.0 (NORMAL)."""
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bar = {"tick_count": 50, "volume": 1000.0}
        assert feat.compute(bar) == pytest.approx(1.0, abs=1e-6)

    def test_high_activity_with_tick_count(self):
        """tick_count > expected returns activity > 1.0 (HIGH)."""
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bar = {"tick_count": 120, "volume": 1000.0}
        result = feat.compute(bar)
        assert result > 1.0
        assert result == pytest.approx(2.4, abs=1e-6)

    def test_low_activity_with_tick_count(self):
        """tick_count < expected returns activity < 1.0 (LOW)."""
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bar = {"tick_count": 20, "volume": 1000.0}
        result = feat.compute(bar)
        assert 0.0 < result < 1.0
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_no_tick_count_fallback_to_volume(self):
        """Without tick_count, falls back to volume/expected_ticks."""
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bar = {"volume": 500.0}  # no tick_count
        result = feat.compute(bar)
        assert result == pytest.approx(10.0, abs=1e-6)

    def test_zero_volume_returns_zero(self):
        """Zero volume returns 0.0 activity."""
        feat = TickActivityFeature()
        bar = {"tick_count": 50, "volume": 0.0}
        assert feat.compute(bar) == 0.0

    def test_negative_clamped_to_zero(self):
        """Activity is clamped to 0.0 minimum."""
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bar = {"tick_count": 0, "volume": 0.0}
        assert feat.compute(bar) == 0.0


class TestTickActivityComputeBatch:
    def test_batch_returns_all_values(self):
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bars = [
            {"tick_count": 30, "volume": 100.0},
            {"tick_count": 60, "volume": 100.0},
            {"tick_count": 100, "volume": 100.0},
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 3
        assert results[0] == pytest.approx(0.6, abs=1e-6)
        assert results[1] == pytest.approx(1.2, abs=1e-6)
        assert results[2] == pytest.approx(2.0, abs=1e-6)

    def test_batch_empty_returns_empty(self):
        feat = TickActivityFeature()
        assert feat.compute_batch([]) == []

    def test_batch_populates_history(self):
        feat = TickActivityFeature(expected_ticks_per_bar=50)
        bars = [{"tick_count": 50, "volume": 100.0}]
        feat.compute_batch(bars)
        assert len(feat._history) == 1
        assert feat._history[0] == pytest.approx(1.0, abs=1e-6)


class TestTickActivityGetActivityLevel:
    def test_low_level(self):
        feat = TickActivityFeature()
        assert feat.get_activity_level(0.5) == "LOW"
        assert feat.get_activity_level(0.0) == "LOW"
        assert feat.get_activity_level(0.99) == "LOW"

    def test_normal_level(self):
        feat = TickActivityFeature()
        assert feat.get_activity_level(1.0) == "NORMAL"
        assert feat.get_activity_level(1.5) == "NORMAL"
        assert feat.get_activity_level(1.99) == "NORMAL"

    def test_high_level(self):
        feat = TickActivityFeature()
        assert feat.get_activity_level(2.0) == "HIGH"
        assert feat.get_activity_level(2.5) == "HIGH"
        assert feat.get_activity_level(2.99) == "HIGH"

    def test_extreme_level(self):
        feat = TickActivityFeature()
        assert feat.get_activity_level(3.0) == "EXTREME"
        assert feat.get_activity_level(5.0) == "EXTREME"
        assert feat.get_activity_level(10.0) == "EXTREME"


class TestTickActivityCapabilitySpec:
    def test_required_inputs(self):
        feat = TickActivityFeature()
        assert "tick_count" in feat.required_inputs
        assert "volume" in feat.required_inputs

    def test_output_keys(self):
        feat = TickActivityFeature()
        assert "activity_ratio" in feat.output_keys
        assert "activity_level" in feat.output_keys


class TestTickActivityConfidence:
    def test_disabled_under_10_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(9)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_10_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(10)
        assert conf.quality == 0.35
        assert conf.feed_quality_tag == "LOW"

    def test_low_at_19_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(19)
        assert conf.quality == 0.35
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_20_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.65
        assert conf.feed_quality_tag == "MEDIUM"

    def test_medium_at_49_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(49)
        assert conf.quality == 0.65
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_50_bars(self):
        feat = TickActivityFeature()
        conf = feat.confidence(50)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"
