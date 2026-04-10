"""
Self-Verification Tests for Packet 5D: Volume Feature Family
Tests RVOLFeature, MFIFeature, and VolumeProfileFeature.
"""
from __future__ import annotations

import pytest

from src.library.features.volume import RVOLFeature, MFIFeature, VolumeProfileFeature
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


# =============================================================================
# RVOLFeature Tests
# =============================================================================


class TestRVOLFeature:
    def test_rvol_london_session_high_activity(self):
        """RVOL during LONDON session with above-average volume."""
        feat = RVOLFeature()
        # volume=4.8 gives rvol = 4.8 / (1.0 * 2.0 * 1.2) = 2.0
        inputs = {"volume": 4.8, "session_id": "LONDON", "day_of_week": 2, "time_of_day_minutes": 600}
        fv = feat.compute(inputs)
        assert "rvol" in fv.features
        assert fv.features["rvol"] == 2.0
        assert fv.bot_id == "SYSTEM"

    def test_rvol_london_ny_overlap_session(self):
        """RVOL during LONDON_NY_OVERLAP session."""
        feat = RVOLFeature()
        inputs = {"volume": 2.0, "session_id": "LONDON_NY_OVERLAP", "day_of_week": 0, "time_of_day_minutes": 900}
        fv = feat.compute(inputs)
        assert fv.features["rvol"] == 1.0  # 2.0 / (1.0 * 2.0 * 1.0)

    def test_rvol_am_session_factor(self):
        """RVOL with LONDON_AM session gets 1.5 factor."""
        feat = RVOLFeature()
        inputs = {"volume": 3.0, "session_id": "LONDON_AM", "day_of_week": 1, "time_of_day_minutes": 480}
        fv = feat.compute(inputs)
        assert fv.features["rvol"] == 2.0  # 3.0 / (1.0 * 1.5 * 1.0)

    def test_rvol_wednesday_higher_factor(self):
        """RVOL on Wednesday gets 1.2 day factor."""
        feat = RVOLFeature()
        inputs = {"volume": 3.6, "session_id": "LONDON", "day_of_week": 2, "time_of_day_minutes": 600}
        fv = feat.compute(inputs)
        assert fv.features["rvol"] == 1.5  # 3.6 / (1.0 * 2.0 * 1.2)

    def test_rvol_category_high(self):
        """rvol >= 2.0 returns HIGH category."""
        feat = RVOLFeature()
        inputs = {"volume": 5.0, "session_id": "LONDON", "day_of_week": 0, "time_of_day_minutes": 600}
        fv = feat.compute(inputs)
        # rvol = 5.0 / (1.0 * 2.0 * 1.0) = 2.5 >= 2.0 -> HIGH
        assert fv.features["rvol"] == 2.5

    def test_rvol_no_data(self):
        """RVOL returns NO_DATA quality when volume is None."""
        feat = RVOLFeature()
        inputs = {"volume": None, "session_id": "LONDON", "day_of_week": 0, "time_of_day_minutes": 600}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["rvol"]
        assert conf.feed_quality_tag == "NO_DATA"
        assert conf.quality == 0.0

    def test_rvol_zero_volume(self):
        """RVOL returns NO_DATA quality when volume is zero."""
        feat = RVOLFeature()
        inputs = {"volume": 0.0, "session_id": "LONDON", "day_of_week": 0, "time_of_day_minutes": 600}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["rvol"]
        assert conf.feed_quality_tag == "NO_DATA"

    def test_rvol_unknown_session_fallback(self):
        """RVOL falls back to 1.0 session factor for unknown session."""
        feat = RVOLFeature()
        inputs = {"volume": 2.0, "session_id": "SYDNEY", "day_of_week": 0, "time_of_day_minutes": 120}
        fv = feat.compute(inputs)
        assert fv.features["rvol"] == 2.0  # 2.0 / (1.0 * 1.0 * 1.0)


# =============================================================================
# MFIFeature Tests
# =============================================================================


class TestMFIFeature:
    def test_mfi_insufficient_data(self):
        """MFI returns INSUFFICIENT_DATA when fewer than period+1 bars."""
        feat = MFIFeature(period=14)
        inputs = {
            "high_prices": [1.0900] * 10,
            "low_prices": [1.0800] * 10,
            "close_prices": [1.0850] * 10,
            "volumes": [1000.0] * 10,
        }
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["mfi_14"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"

    def test_mfi_neutral_zone(self):
        """MFI in 20-80 range returns NEUTRAL signal."""
        feat = MFIFeature(period=14)
        # Simple oscillating price series that should give neutral MFI
        highs = [1.0860 + i * 0.001 for i in range(20)]
        lows = [1.0840 + i * 0.001 for i in range(20)]
        closes = [1.0850 + i * 0.001 for i in range(20)]
        volumes = [1000.0] * 20
        inputs = {
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        }
        fv = feat.compute(inputs)
        assert "mfi_14" in fv.features
        assert 0.0 <= fv.features["mfi_14"] <= 100.0

    def test_mfi_overbought(self):
        """MFI >= 80 returns OVERBOUGHT signal."""
        feat = MFIFeature(period=5)
        # Rising prices with increasing volume -> high MFI
        highs = [1.0850 + i * 0.001 for i in range(15)]
        lows = [1.0840 + i * 0.001 for i in range(15)]
        closes = [1.0850 + i * 0.001 for i in range(15)]
        volumes = [1000.0 + i * 100.0 for i in range(15)]  # increasing volume
        inputs = {
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        }
        fv = feat.compute(inputs)
        mfi = fv.features["mfi_5"]
        # With strong uptrend + increasing volume, MFI should be high
        assert mfi > 50.0

    def test_mfi_period_parameter(self):
        """MFI with custom period uses correct period key."""
        feat = MFIFeature(period=7)
        assert feat.config.feature_id == "volume/mfi_7"
        assert "mfi_7" in feat.output_keys

    def test_mfi_bot_id_system(self):
        """MFI FeatureVector has bot_id=SYSTEM."""
        feat = MFIFeature(period=14)
        highs = [1.0860 + i * 0.001 for i in range(30)]
        lows = [1.0840 + i * 0.001 for i in range(30)]
        closes = [1.0850 + i * 0.001 for i in range(30)]
        volumes = [1000.0] * 30
        fv = feat.compute({
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        })
        assert fv.bot_id == "SYSTEM"


# =============================================================================
# VolumeProfileFeature Tests
# =============================================================================


class TestVolumeProfileFeature:
    def test_profile_insufficient_data(self):
        """Profile returns INSUFFICIENT_DATA when fewer than 20 bars."""
        feat = VolumeProfileFeature()
        inputs = {
            "high_prices": [1.0900] * 10,
            "low_prices": [1.0800] * 10,
            "close_prices": [1.0850] * 10,
            "volumes": [1000.0] * 10,
        }
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["poc_price"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"

    def test_profile_poc_identified(self):
        """POC is correctly identified from high-volume price level."""
        feat = VolumeProfileFeature()
        # 30 bars, concentrated volume around 1.0850
        highs = [1.0855] * 20 + [1.0870] * 10
        lows = [1.0845] * 20 + [1.0860] * 10
        closes = [1.0850] * 20 + [1.0865] * 10
        volumes = [1000.0] * 20 + [500.0] * 10  # 2x volume in POC zone
        inputs = {
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        }
        fv = feat.compute(inputs)
        assert "poc_price" in fv.features
        assert "poc_strength" in fv.features
        assert "value_area_high" in fv.features
        assert "value_area_low" in fv.features
        assert "poc_distance" in fv.features
        # POC should be in the 1.0845-1.0855 range
        assert 1.0840 < fv.features["poc_price"] < 1.0860
        # POC strength should be meaningful (> 0.05)
        assert fv.features["poc_strength"] > 0.0
        # Value area high > value area low
        assert fv.features["value_area_high"] > fv.features["value_area_low"]

    def test_profile_bot_id_system(self):
        """Profile FeatureVector has bot_id=SYSTEM."""
        feat = VolumeProfileFeature()
        highs = [1.0860 + i * 0.001 for i in range(25)]
        lows = [1.0840 + i * 0.001 for i in range(25)]
        closes = [1.0850 + i * 0.001 for i in range(25)]
        volumes = [1000.0] * 25
        fv = feat.compute({
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        })
        assert fv.bot_id == "SYSTEM"

    def test_profile_all_output_keys_present(self):
        """All 5 output keys are present in FeatureVector."""
        feat = VolumeProfileFeature()
        highs = [1.0860 + i * 0.001 for i in range(30)]
        lows = [1.0840 + i * 0.001 for i in range(30)]
        closes = [1.0850 + i * 0.001 for i in range(30)]
        volumes = [1000.0] * 30
        fv = feat.compute({
            "high_prices": highs,
            "low_prices": lows,
            "close_prices": closes,
            "volumes": volumes,
        })
        expected_keys = {"poc_price", "poc_distance", "value_area_high", "value_area_low", "poc_strength"}
        assert set(fv.features.keys()) == expected_keys

    def test_profile_custom_num_bins(self):
        """Profile with custom num_bins parameter works correctly."""
        feat = VolumeProfileFeature(num_bins=10)
        assert feat.config.feature_id == "volume/profile"
        assert feat.num_bins == 10
