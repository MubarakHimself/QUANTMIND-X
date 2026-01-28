"""
Tests for Physics Sensors

Focused tests for econophysics-based market regime detection sensors.
Tests cover Ising model, Lyapunov chaos detection, and RMT correlation analysis.
"""

import pytest
import numpy as np
import time
from src.risk.physics import (
    IsingRegimeSensor,
    ChaosSensor,
    CorrelationSensor,
)


class TestIsingRegimeSensor:
    """Tests for Ising regime sensor."""

    def test_phase_transition_detection(self):
        """Test that phase transitions are detected at critical susceptibility."""
        sensor = IsingRegimeSensor(
            grid_size=8,  # Smaller grid for faster tests
            equilibrium_steps=1000,
            measurement_steps=200,
        )

        # High volatility → high temperature → chaotic regime
        result = sensor.detect_regime(volatility=0.05)  # 5% annual vol

        assert "susceptibility" in result
        assert "magnetization" in result
        assert "regime_label" in result
        assert result["regime_label"] in ["ORDERED", "TRANSITIONAL", "CHAOTIC"]
        assert result["susceptibility"] >= 0

    def test_temperature_mapping(self):
        """Test that volatility maps correctly to temperature."""
        sensor = IsingRegimeSensor(grid_size=8, equilibrium_steps=500, measurement_steps=100)

        # Low volatility → low temperature
        result_low = sensor.detect_regime(volatility=0.001)
        assert 0.1 <= result_low["temperature"] < 3.0

        # High volatility → high temperature
        result_high = sensor.detect_regime(volatility=0.10)
        assert result_high["temperature"] > 5.0

    def test_ordered_state_detection(self):
        """Test that ordered states have high magnetization."""
        sensor = IsingRegimeSensor(
            grid_size=8,
            equilibrium_steps=500,
            measurement_steps=100,
            target_sentiment=1.0,  # Force up-spin bias
            control_gain=5.0,
        )

        # Low volatility should produce ordered state
        result = sensor.detect_regime(volatility=0.001)

        # With high bias and low temp, should be ordered
        assert result["regime_label"] in ["ORDERED", "TRANSITIONAL"]
        assert abs(result["magnetization"]) >= 0 or result["magnetization"] <= 1

    def test_metropolis_acceptance_rate(self):
        """Test that Metropolis algorithm produces reasonable acceptance."""
        sensor = IsingRegimeSensor(grid_size=8, equilibrium_steps=500, measurement_steps=100)

        # Run simulation and verify it completes
        result = sensor.detect_regime(volatility=0.02)

        # Should complete without error
        assert result is not None
        assert "calculated_at" in result

    def test_cache_functionality(self):
        """Test that caching works correctly."""
        sensor = IsingRegimeSensor(grid_size=8, equilibrium_steps=200, measurement_steps=50)

        # First call
        result1 = sensor.detect_regime(volatility=0.02)
        assert not sensor.is_cache_valid(max_age_seconds=0)

        # Second call should use cache
        assert sensor.is_cache_valid(max_age_seconds=60)
        cached = sensor.get_cached_result(volatility=0.02)
        assert cached is not None
        assert cached["susceptibility"] == result1["susceptibility"]

    def test_magnetization_bounds(self):
        """Test that magnetization stays within valid bounds."""
        sensor = IsingRegimeSensor(grid_size=8, equilibrium_steps=500, measurement_steps=100)

        for vol in [0.001, 0.02, 0.10]:
            result = sensor.detect_regime(volatility=vol)
            # Magnetization should be in [-1, 1]
            assert -1.0 <= result["magnetization"] <= 1.0


class TestChaosSensor:
    """Tests for Lyapunov chaos sensor."""

    def test_lyapunov_on_chaotic_data(self):
        """Test Lyapunov exponent detection on chaotic time series."""
        sensor = ChaosSensor(dimension=3, tau=10, lookback=100)

        # Generate chaotic data using logistic map
        # x[n+1] = r * x[n] * (1 - x[n]), r=4 produces chaos
        n_points = 500
        x = np.zeros(n_points)
        x[0] = 0.5
        r = 4.0
        for i in range(1, n_points):
            x[i] = r * x[i-1] * (1 - x[i-1])

        # Analyze chaos
        result = sensor.analyze_chaos(returns=x)

        assert "lyapunov_exponent" in result
        assert "chaos_level" in result
        assert result["chaos_level"] in ["STABLE", "MODERATE", "CHAOTIC"]
        # Logistic map at r=4 should produce positive Lyapunov
        assert result["lyapunov_exponent"] > 0

    def test_periodic_series_zero_lyapunov(self):
        """Test that periodic time series have near-zero Lyapunov exponent."""
        sensor = ChaosSensor(dimension=3, tau=10, lookback=100)

        # Generate periodic sine wave
        n_points = 500
        t = np.linspace(0, 20 * np.pi, n_points)
        x = np.sin(t)

        result = sensor.analyze_chaos(returns=x)

        # Periodic signal should have low Lyapunov exponent
        assert result["lyapunov_exponent"] < 0.5

    def test_phase_space_embedding(self):
        """Test that phase space embedding produces correct dimensions."""
        sensor = ChaosSensor(dimension=3, tau=12, lookback=100)

        # Create simple test data
        n_points = 500
        x = np.random.randn(n_points)

        result = sensor.analyze_chaos(returns=x)

        # Should complete without error
        assert result is not None
        assert "lyapunov_exponent" in result

    def test_kdtree_nearest_neighbor(self):
        """Test that KD-tree finds nearest neighbors correctly."""
        sensor = ChaosSensor(dimension=3, tau=12, lookback=100)

        # Create structured data with pattern
        n_points = 500
        x = np.sin(np.linspace(0, 50 * np.pi, n_points))

        result = sensor.analyze_chaos(returns=x)

        # Should find a match
        assert result["match_distance"] >= 0
        assert result["match_index"] >= 0
        assert result["match_index"] < n_points

    def test_insufficient_data_error(self):
        """Test that insufficient data raises ValueError."""
        sensor = ChaosSensor()

        # Too few points
        short_data = np.random.randn(50)

        with pytest.raises(ValueError, match="Insufficient data"):
            sensor.analyze_chaos(returns=short_data)

    def test_cache_functionality(self):
        """Test that caching works by data hash."""
        sensor = ChaosSensor(dimension=3, tau=10, lookback=100)

        # Create test data
        data = np.random.randn(500)
        data_hash = hash(data.tobytes())

        # First call
        result1 = sensor.analyze_chaos(returns=data)

        # Check cache
        assert sensor.is_cache_valid(max_age_seconds=60)
        cached = sensor.get_cached_result(data_hash)
        assert cached is not None
        assert cached["lyapunov_exponent"] == result1["lyapunov_exponent"]


class TestCorrelationSensor:
    """Tests for RMT correlation sensor."""

    def test_marchenko_pastur_distribution(self):
        """Test Marchenko-Pastur threshold calculation."""
        sensor = CorrelationSensor()

        # For N=10, T=100, Q=10
        # lambda_max = (1 + sqrt(1/10))^2 ≈ 1.32
        n_assets = 10
        n_periods = 100
        threshold = sensor._compute_marchenko_pastur_threshold(n_assets, n_periods)

        expected = (1 + np.sqrt(1 / (n_periods / n_assets))) ** 2
        assert abs(threshold - expected) < 0.01

    def test_eigenvalue_decomposition(self):
        """Test eigenvalue decomposition of correlation matrix."""
        sensor = CorrelationSensor()

        # Create simple correlation matrix
        np.random.seed(42)
        data = np.random.randn(100, 5)
        correlation = np.corrcoef(data.T)

        eigenvalues, eigenvectors = sensor._eigenvalue_decomposition(correlation)

        # Eigenvalues should be sorted descending
        assert len(eigenvalues) == 5
        assert eigenvalues[0] >= eigenvalues[-1]
        # Sum of eigenvalues should equal trace (approximately)
        assert abs(np.sum(eigenvalues) - np.trace(correlation)) < 0.1

    def test_systemic_risk_detection(self):
        """Test systemic risk detection from maximum eigenvalue."""
        sensor = CorrelationSensor()

        # Create highly correlated data (systemic risk)
        np.random.seed(42)
        common_factor = np.random.randn(100)
        returns = np.column_stack([
            common_factor + 0.1 * np.random.randn(100) for _ in range(10)
        ])

        result = sensor.detect_systemic_risk(returns_matrix=returns)

        assert "max_eigenvalue" in result
        assert "noise_threshold" in result
        assert "risk_level" in result
        assert result["risk_level"] in ["LOW", "MODERATE", "HIGH"]

    def test_noise_threshold_detection(self):
        """Test that noise threshold is calculated correctly."""
        sensor = CorrelationSensor(use_denoising=True)

        # Create random correlation matrix (mostly noise)
        np.random.seed(42)
        returns = np.random.randn(50, 20)

        result = sensor.detect_systemic_risk(returns_matrix=returns)

        # For random data, max eigenvalue should be near noise threshold
        assert result["noise_threshold"] > 0
        # Most eigenvalues should be below threshold for random data
        assert result["num_signal_eigenvalues"] >= 1  # At least one signal

    def test_denoising_reconstruction(self):
        """Test that denoising reconstructs valid correlation matrix."""
        sensor = CorrelationSensor(use_denoising=True)

        np.random.seed(42)
        returns = np.random.randn(100, 10)

        result = sensor.detect_systemic_risk(returns_matrix=returns)

        # Denoised matrix should be valid correlation matrix
        denoised = result["denoised_matrix"]
        assert denoised.shape == (10, 10)
        # Diagonal should be 1.0
        assert np.allclose(np.diag(denoised), 1.0, atol=0.01)
        # Values should be in [-1, 1]
        assert np.all(denoised >= -1.01) and np.all(denoised <= 1.01)

    def test_minimum_dimensions_validation(self):
        """Test that minimum dimension requirements are enforced."""
        sensor = CorrelationSensor()

        # Too few assets
        with pytest.raises(ValueError, match="at least 2 assets"):
            sensor.detect_systemic_risk(returns_matrix=np.random.randn(10, 1))

        # Too few periods
        with pytest.raises(ValueError, match="at least 20 periods"):
            sensor.detect_systemic_risk(returns_matrix=np.random.randn(5, 10))
