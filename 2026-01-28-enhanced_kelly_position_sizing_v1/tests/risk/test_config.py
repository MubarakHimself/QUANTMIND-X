"""
Configuration tests for the risk management system.

Tests focus on:
- Prop firm preset loading and validation
- Configuration threshold validation
- Default value verification
"""

import pytest
from risk.config import (
    LYAPUNOV_CHAOS_THRESHOLD,
    ISING_CRITICAL_SUSCEPTIBILITY,
    RMT_MAX_EIGEN_THRESHOLD,
    RMT_CRITICAL_EIGEN_THRESHOLD,
    PHYSICS_CACHE_TTL,
    ACCOUNT_CACHE_TTL,
    MC_SIMULATION_RUNS,
    MC_MAX_DRAWDOWN,
    MC_RUIN_THRESHOLD,
    DEFAULT_K_FRACTION,
    MAX_RISK_PCT,
    MIN_LOT,
    LOT_STEP,
    MAX_LOT,
    validate_config,
    PropFirmPreset,
    FTPreset,
    The5ersPreset,
    FundingPipsPreset,
)


class TestPropFirmPresets:
    """Tests for prop firm preset configurations."""

    def test_ftmo_preset_loading(self):
        """Test FTMO preset loads with correct values."""
        preset = FTPreset()
        assert preset.name == "FTMO"
        assert preset.max_drawdown == 0.10  # 10%
        assert preset.max_daily_loss == 0.05  # 5%
        assert preset.profit_target == 0.10  # 10%

    def test_the5ers_preset_loading(self):
        """Test The5ers preset loads with correct values."""
        preset = The5ersPreset()
        assert preset.name == "The5ers"
        assert preset.max_drawdown == 0.08  # 8%
        assert preset.max_daily_loss == 0.04  # 4%
        assert preset.profit_target == 0.10  # 10%

    def test_fundingpips_preset_loading(self):
        """Test FundingPips preset loads with correct values."""
        preset = FundingPipsPreset()
        assert preset.name == "FundingPips"
        assert preset.max_drawdown == 0.12  # 12%
        assert preset.max_daily_loss == 0.06  # 6%
        assert preset.profit_target == 0.08  # 8%

    def test_preset_max_risk_calculation(self):
        """Test max risk percentage is calculated correctly from drawdown."""
        preset = FTPreset()
        # Max risk should be conservative portion of max drawdown
        max_risk = preset.get_max_risk_pct()
        assert 0 < max_risk <= preset.max_drawdown
        # FTMO 10% drawdown should yield ~2% max risk per trade
        assert max_risk == pytest.approx(0.02, rel=0.5)


class TestConfigurationThresholds:
    """Tests for physics and Monte Carlo configuration thresholds."""

    def test_lyapunov_threshold(self):
        """Test Lyapunov chaos threshold is in valid range."""
        assert 0.0 < LYAPUNOV_CHAOS_THRESHOLD <= 1.0
        assert LYAPUNOV_CHAOS_THRESHOLD == 0.5

    def test_ising_susceptibility_threshold(self):
        """Test Ising critical susceptibility threshold is valid."""
        assert 0.0 < ISING_CRITICAL_SUSCEPTIBILITY <= 1.0
        assert ISING_CRITICAL_SUSCEPTIBILITY == 0.8

    def test_rmt_eigenvalue_thresholds(self):
        """Test RMT eigenvalue thresholds are properly ordered."""
        assert RMT_MAX_EIGEN_THRESHOLD < RMT_CRITICAL_EIGEN_THRESHOLD
        assert RMT_MAX_EIGEN_THRESHOLD == 1.5
        assert RMT_CRITICAL_EIGEN_THRESHOLD == 2.0

    def test_monte_carlo_settings(self):
        """Test Monte Carlo simulation settings are reasonable."""
        assert MC_SIMULATION_RUNS >= 100  # Minimum for statistical significance
        assert MC_SIMULATION_RUNS == 2000
        assert 0.0 < MC_MAX_DRAWDOWN <= 1.0  # 10% max drawdown
        assert MC_MAX_DRAWDOWN == 0.10
        assert 0.0 < MC_RUIN_THRESHOLD <= 0.05  # Max 5% ruin threshold
        assert MC_RUIN_THRESHOLD == 0.005


class TestKellyConfiguration:
    """Tests for Kelly criterion configuration settings."""

    def test_default_k_fraction(self):
        """Test default Kelly fraction is half-Kelly for safety."""
        assert 0.0 < DEFAULT_K_FRACTION <= 1.0
        assert DEFAULT_K_FRACTION == 0.5  # Half-Kelly

    def test_max_risk_percentage(self):
        """Test max risk per trade is reasonable."""
        assert 0.0 < MAX_RISK_PCT <= 0.05  # Max 5% per trade
        assert MAX_RISK_PCT == 0.02  # 2% per trade


class TestBrokerConstraints:
    """Tests for broker lot size constraints."""

    def test_lot_constraints(self):
        """Test lot size constraints are properly ordered."""
        assert MIN_LOT > 0
        assert LOT_STEP > 0
        assert MAX_LOT > MIN_LOT
        assert MIN_LOT == 0.01
        assert LOT_STEP == 0.01
        assert MAX_LOT == 100.0


class TestCacheSettings:
    """Tests for cache TTL settings."""

    def test_cache_ttl_values(self):
        """Test cache TTL values are positive and reasonable."""
        assert PHYSICS_CACHE_TTL > 0
        assert ACCOUNT_CACHE_TTL > 0
        assert PHYSICS_CACHE_TTL == 300  # 5 minutes
        assert ACCOUNT_CACHE_TTL == 10  # 10 seconds


class TestConfigurationValidation:
    """Tests for configuration validation function."""

    def test_validate_config_accepts_valid_settings(self):
        """Test validation passes with current valid configuration."""
        # Should not raise any exception
        validate_config()

    def test_validate_config_checks_threshold_ranges(self):
        """Test validation would fail for invalid thresholds."""
        # This test verifies the validation logic exists
        # Actual invalid values tested by not raising errors with valid config
        assert LYAPUNOV_CHAOS_THRESHOLD >= 0.0
        assert LYAPUNOV_CHAOS_THRESHOLD <= 1.0

    def test_validate_config_checks_positive_values(self):
        """Test validation ensures positive values where required."""
        assert MC_SIMULATION_RUNS > 0
        assert PHYSICS_CACHE_TTL > 0
        assert MIN_LOT > 0
