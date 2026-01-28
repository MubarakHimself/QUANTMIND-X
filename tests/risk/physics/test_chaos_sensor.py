"""
Tests for ChaosSensor class.

These tests verify the functionality of the ChaosSensor including:
- Phase space reconstruction
- Method of analogues implementation
- Lyapunov exponent calculation
- Input validation
- Chaos level determination
"""

import numpy as np
import pytest
from src.risk.physics.chaos_sensor import ChaosSensor, ChaosAnalysisResult

class TestChaosSensor:
    """Test suite for ChaosSensor functionality."""

    def setup_method(self):
        """Setup method for each test."""
        self.sensor = ChaosSensor(
            embedding_dimension=3,
            time_delay=12,
            lookback_points=300,
            k_steps=10
        )

    def test_init_default_parameters(self):
        """Test default initialization parameters."""
        sensor = ChaosSensor()
        assert sensor.embedding_dimension == 3
        assert sensor.time_delay == 12
        assert sensor.lookback_points == 300
        assert sensor.k_steps == 10

    def test_time_delay_embedding(self):
        """Test time delay embedding functionality."""
        # Create test data with sufficient length for embedding
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0])

        # Test with valid parameters
        embedded = self.sensor._embed_time_delay(series)
        assert embedded.shape == (6, 3)  # (30 - (3-1)*12) = 6 vectors, 3 dimensions

        # Test with insufficient data
        with pytest.raises(ValueError):
            self.sensor._embed_time_delay(np.array([1.0, 2.0, 3.0]))

    def test_method_of_analogues(self):
        """Test method of analogues functionality."""
        # Create test embedded vectors with sufficient length
        series = np.random.randn(1000)
        embedded = self.sensor._embed_time_delay(series)

        # Test with sufficient data
        match_dist, match_idx = self.sensor._perform_method_of_analogues(embedded)
        assert isinstance(match_dist, float)
        assert isinstance(match_idx, (int, np.integer))  # Allow numpy integer types
        assert match_dist >= 0

        # Test with insufficient data
        with pytest.raises(ValueError):
            self.sensor._perform_method_of_analogues(embedded[:100])

    def test_lyapunov_exponent_calculation(self):
        """Test Lyapunov exponent calculation."""
        # Create test data
        series = np.random.randn(500)
        match_idx = 100

        # Test calculation
        lyapunov = self.sensor._calculate_lyapunov_exponent(series, match_idx)
        assert isinstance(lyapunov, float)

        # Test with edge case
        sensor = ChaosSensor(k_steps=0)
        assert sensor._calculate_lyapunov_exponent(series, match_idx) == 0.0

    def test_chaos_level_determination(self):
        """Test chaos level determination."""
        assert self.sensor._determine_chaos_level(-0.1) == "STABLE"
        assert self.sensor._determine_chaos_level(0.3) == "MODERATE"
        assert self.sensor._determine_chaos_level(0.6) == "CHAOTIC"

    def test_analyze_chaos_with_valid_input(self):
        """Test analyze_chaos with valid input."""
        # Create test returns data with sufficient length (need at least 800 points)
        returns = np.random.randn(1000)  # More than minimum 300 points and enough for method of analogues

        result = self.sensor.analyze_chaos(returns)
        assert isinstance(result, ChaosAnalysisResult)
        assert isinstance(result.lyapunov_exponent, float)
        assert isinstance(result.match_distance, float)
        assert isinstance(result.match_index, (int, np.integer))  # Allow numpy integer types
        assert result.chaos_level in ["STABLE", "MODERATE", "CHAOTIC"]
        assert result.trajectory_length == 1000

    def test_analyze_chaos_with_invalid_input(self):
        """Test analyze_chaos with invalid input."""
        # Test with non-numpy array
        with pytest.raises(ValueError):
            self.sensor.analyze_chaos([1.0, 2.0, 3.0])

        # Test with insufficient length
        with pytest.raises(ValueError):
            self.sensor.analyze_chaos(np.random.randn(299))

        # Test with multi-dimensional array
        with pytest.raises(ValueError):
            self.sensor.analyze_chaos(np.random.randn(300, 2))

    def test_call_method(self):
        """Test __call__ convenience method."""
        returns = np.random.randn(900)
        result = self.sensor(returns)
        assert isinstance(result, ChaosAnalysisResult)

    def test_performance_target(self):
        """Test that analysis completes within target time (< 100ms)."""
        returns = np.random.randn(900)

        import time
        start_time = time.time()
        self.sensor.analyze_chaos(returns)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        assert elapsed_time < 100, f"Analysis took {elapsed_time:.2f}ms, exceeding 100ms target"

    def test_different_parameters(self):
        """Test sensor with different parameters."""
        # Test with different embedding dimension
        sensor_2d = ChaosSensor(embedding_dimension=2)
        returns = np.random.randn(900)  # Need more points for 2D embedding
        result = sensor_2d.analyze_chaos(returns)
        assert isinstance(result, ChaosAnalysisResult)

        # Test with different time delay
        sensor_delay_6 = ChaosSensor(time_delay=6)
        result = sensor_delay_6.analyze_chaos(returns)
        assert isinstance(result, ChaosAnalysisResult)