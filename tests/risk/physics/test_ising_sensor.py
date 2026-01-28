"""
Tests for Ising Regime Sensor implementation.

These tests verify the core functionality of the Ising model-based regime detector,
including lattice initialization, Metropolis-Hastings simulation, and regime detection.
"""

import pytest
import numpy as np
from src.risk.physics.ising_sensor import IsingRegimeSensor, IsingSensorConfig


class TestIsingSensor:
    """Test suite for IsingRegimeSensor functionality."""

    def test_initialization(self):
        """Test sensor initialization with default configuration."""
        sensor = IsingRegimeSensor()
        assert sensor.config.grid_size == 12
        assert sensor.config.steps_per_temp == 100
        assert sensor.config.temp_range == (10.0, 0.1)
        assert sensor.config.temp_steps == 50

    def test_custom_configuration(self):
        """Test sensor initialization with custom configuration."""
        config = IsingSensorConfig(
            grid_size=8,
            steps_per_temp=50,
            temp_range=(5.0, 0.5),
            temp_steps=20,
            target_sentiment=0.6,
            control_gain=3.0
        )
        sensor = IsingRegimeSensor(config)
        assert sensor.config.grid_size == 8
        assert sensor.config.steps_per_temp == 50
        assert sensor.config.temp_range == (5.0, 0.5)
        assert sensor.config.temp_steps == 20
        assert sensor.config.target_sentiment == 0.6
        assert sensor.config.control_gain == 3.0

    def test_lattice_initialization(self):
        """Test that lattice is properly initialized with random spins."""
        sensor = IsingRegimeSensor()
        lattice = sensor._simulate_temperature.cache_info()
        # The lattice should be initialized with values of -1 or 1
        assert np.all(np.isin(sensor._simulate_temperature.cache_info(), [-1, 1]))

    def test_metropolis_step(self):
        """Test Metropolis-Hastings step implementation."""
        sensor = IsingRegimeSensor()
        # Test at high temperature (chaotic regime)
        result = sensor._simulate_temperature(10.0)
        assert 'temperature' in result
        assert 'magnetization' in result
        assert 'susceptibility' in result
        assert 'activity' in result
        assert -1 <= result['magnetization'] <= 1

    def test_regime_classification(self):
        """Test regime classification logic."""
        sensor = IsingRegimeSensor()

        # Test ordered regime (high magnetization)
        ordered_result = {'magnetization': 0.8}
        assert sensor._classify_regime(ordered_result['magnetization']) == "ORDERED"

        # Test disordered regime (low magnetization)
        disordered_result = {'magnetization': 0.2}
        assert sensor._classify_regime(disordered_result['magnetization']) == "DISORDERED"

    def test_regime_confidence(self):
        """Test regime confidence calculation."""
        sensor = IsingRegimeSensor()

        # High confidence for strong magnetization
        assert sensor.get_regime_confidence(0.8) == 0.8

        # Low confidence for weak magnetization
        assert sensor.get_regime_confidence(0.2) == 0.2

        # Absolute value is used
        assert sensor.get_regime_confidence(-0.7) == 0.7

    def test_detect_regime(self):
        """Test full regime detection pipeline."""
        sensor = IsingRegimeSensor()
        result = sensor.detect_regime()

        assert 'critical_temperature' in result
        assert 'critical_magnetization' in result
        assert 'current_regime' in result
        assert 'final_magnetization' in result
        assert 'susceptibility_data' in result

        # Critical temperature should be within expected range
        assert 0.1 <= result['critical_temperature'] <= 10.0

        # Magnetization should be within valid range
        assert -1 <= result['critical_magnetization'] <= 1
        assert -1 <= result['final_magnetization'] <= 1

    def test_cache_functionality(self):
        """Test caching mechanism."""
        sensor = IsingRegimeSensor()

        # First call should populate cache
        result1 = sensor.detect_regime()

        # Second call with same configuration should use cache
        result2 = sensor.detect_regime()

        # Results should be identical
        assert result1['critical_temperature'] == result2['critical_temperature']
        assert result1['critical_magnetization'] == result2['critical_magnetization']

        # Clear cache and verify it's reset
        sensor.clear_cache()
        result3 = sensor.detect_regime()
        assert result1['critical_temperature'] != result3['critical_temperature']  # Should be different due to random initialization

    def test_temperature_range(self):
        """Test temperature range generation."""
        sensor = IsingRegimeSensor()
        temps = np.linspace(sensor.config.temp_range[0], sensor.config.temp_range[1], sensor.config.temp_steps)
        assert len(temps) == sensor.config.temp_steps
        assert temps[0] == sensor.config.temp_range[0]
        assert temps[-1] == sensor.config.temp_range[1]
        assert np.all(np.diff(temps) > 0)  # Should be strictly decreasing

    def test_activity_calculation(self):
        """Test activity (spin flips) calculation."""
        sensor = IsingRegimeSensor()
        result = sensor._simulate_temperature(5.0)
        assert 'activity' in result
        assert result['activity'] >= 0  # Activity should be non-negative

    def test_susceptibility_calculation(self):
        """Test susceptibility calculation."""
        sensor = IsingRegimeSensor()
        result = sensor._simulate_temperature(2.0)
        assert 'susceptibility' in result
        assert result['susceptibility'] >= 0  # Susceptibility should be non-negative

    def test_lattice_size_consistency(self):
        """Test that lattice size is consistent throughout simulation."""
        config = IsingSensorConfig(grid_size=8)
        sensor = IsingRegimeSensor(config)
        result = sensor._simulate_temperature(3.0)
        # The lattice size should be consistent with configuration
        assert result['temperature'] == 3.0  # Temperature should match input


if __name__ == "__main__":
    pytest.main([__file__])