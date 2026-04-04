"""
Tests for HMM WFA Calibrator
============================

Tests for the Walk-Forward Analysis window calibrator that dynamically
calculates optimal window size based on avg_regime_transition_interval.

Verifies:
- Dynamic window calculation: f(avg_regime_transition_interval)
- Window clamped between 7 and 30 days
- Rolling 1-month baseline for scalping variants
"""

import pytest
from src.router.hmm_wfa_calibrator import (
    WfaCalibrator,
    WfaWindowConfig
)


class TestWfaCalibrator:
    """Test cases for WfaCalibrator."""

    @pytest.fixture
    def calibrator(self):
        """Create a WfaCalibrator instance."""
        return WfaCalibrator()

    # =======================================================================
    # AC2: Dynamic Window Calculation
    # =======================================================================

    def test_fast_regime_3_days_window_12(self, calibrator):
        """
        Given avg_regime_interval = 3 days (fast regime switching),
        When calculate_window_sync is called,
        Then window_days = 12 (3 * 4, clamped min 7).
        """
        config = calibrator.calculate_window_sync(3.0)

        assert config.window_days == 12
        assert config.avg_regime_interval_used == 3.0
        assert config.baseline == "scalping_variants"

    def test_slow_regime_10_days_window_30(self, calibrator):
        """
        Given avg_regime_interval = 10 days (slow regime switching),
        When calculate_window_sync is called,
        Then window_days = 30 (10 * 4, clamped max 30).
        """
        config = calibrator.calculate_window_sync(10.0)

        assert config.window_days == 30
        assert config.avg_regime_interval_used == 10.0

    def test_normal_regime_7_days_window_28(self, calibrator):
        """
        Given avg_regime_interval = 7 days (normal regime),
        When calculate_window_sync is called,
        Then window_days = 28 (7 * 4).
        """
        config = calibrator.calculate_window_sync(7.0)

        assert config.window_days == 28
        assert config.avg_regime_interval_used == 7.0

    def test_edge_case_1_day_interval_min_window(self, calibrator):
        """
        Given avg_regime_interval = 1 day (very fast switching),
        When calculate_window_sync is called,
        Then window_days = 7 (clamped to minimum).
        """
        config = calibrator.calculate_window_sync(1.0)

        assert config.window_days == 7  # MIN_WINDOW_DAYS
        assert config.avg_regime_interval_used == 1.0

    def test_edge_case_8_days_interval_window_30(self, calibrator):
        """
        Given avg_regime_interval = 8 days,
        When calculate_window_sync is called,
        Then window_days = 30 (8 * 4 = 32, clamped to 30).
        """
        config = calibrator.calculate_window_sync(8.0)

        assert config.window_days == 30  # Clamped to MAX_WINDOW_DAYS

    # =======================================================================
    # AC2: Window Type Selection
    # =======================================================================

    def test_window_type_rolling_1month_when_28_plus_days(self, calibrator):
        """
        Given window_days >= 28 (rolling_1month threshold),
        When calculate_window_sync is called,
        Then window_type = "rolling_1month".
        """
        config = calibrator.calculate_window_sync(7.0)  # 7 * 4 = 28

        assert config.window_days == 28
        assert config.window_type == "rolling_1month"

    def test_window_type_rolling_adaptive_when_below_28(self, calibrator):
        """
        Given window_days < 28,
        When calculate_window_sync is called,
        Then window_type = "rolling_adaptive".
        """
        config = calibrator.calculate_window_sync(6.0)  # 6 * 4 = 24

        assert config.window_days == 24
        assert config.window_type == "rolling_adaptive"

    # =======================================================================
    # AC2: Baseline Configuration
    # =======================================================================

    def test_baseline_scalping_variants(self, calibrator):
        """Verify baseline is set to scalping_variants."""
        config = calibrator.calculate_window_sync(7.0)

        assert config.baseline == "scalping_variants"

    # =======================================================================
    # Window Calculation Formula
    # =======================================================================

    def test_window_calculation_formula(self, calibrator):
        """
        Verify window calculation follows: window_days = avg_interval * 4.
        """
        test_cases = [
            (5.0, 20),   # 5 * 4 = 20
            (7.0, 28),   # 7 * 4 = 28
            (7.5, 30),   # 7.5 * 4 = 30 (clamped)
            (3.5, 14),   # 3.5 * 4 = 14
            (2.0, 8),    # 2 * 4 = 8
        ]

        for avg_interval, expected_window in test_cases:
            config = calibrator.calculate_window_sync(avg_interval)
            assert config.window_days == expected_window, \
                f"Failed for avg_interval={avg_interval}: got {config.window_days}, expected {expected_window}"


class TestWfaWindowConfig:
    """Test cases for WfaWindowConfig dataclass."""

    def test_config_creation(self):
        """Test WfaWindowConfig can be created with all fields."""
        config = WfaWindowConfig(
            window_days=28,
            window_type="rolling_1month",
            baseline="scalping_variants",
            avg_regime_interval_used=7.0
        )

        assert config.window_days == 28
        assert config.window_type == "rolling_1month"
        assert config.baseline == "scalping_variants"
        assert config.avg_regime_interval_used == 7.0

    def test_config_with_different_window_types(self):
        """Test config with different window types."""
        config_adaptive = WfaWindowConfig(
            window_days=14,
            window_type="rolling_adaptive",
            baseline="scalping_variants",
            avg_regime_interval_used=3.5
        )

        assert config_adaptive.window_type == "rolling_adaptive"

        config_1month = WfaWindowConfig(
            window_days=30,
            window_type="rolling_1month",
            baseline="scalping_variants",
            avg_regime_interval_used=8.0
        )

        assert config_1month.window_type == "rolling_1month"


class TestWfaCalibratorConstants:
    """Test WfaCalibrator constant values."""

    def test_multiplier_is_4(self):
        """Verify window multiplier is 4."""
        assert WfaCalibrator.WINDOW_MULTIPLIER == 4.0

    def test_min_window_is_7(self):
        """Verify minimum window is 7 days."""
        assert WfaCalibrator.MIN_WINDOW_DAYS == 7

    def test_max_window_is_30(self):
        """Verify maximum window is 30 days."""
        assert WfaCalibrator.MAX_WINDOW_DAYS == 30

    def test_1month_threshold_is_28(self):
        """Verify rolling 1month threshold is 28 days."""
        assert WfaCalibrator.ROLLING_1MONTH_THRESHOLD == 28
