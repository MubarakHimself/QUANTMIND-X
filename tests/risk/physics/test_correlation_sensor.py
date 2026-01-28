"""
Tests for Correlation Sensor with RMT filtering.

These tests verify the functionality of the CorrelationSensor class, including:
- Input validation
- Marchenko-Pastur threshold calculation
- Eigenvalue decomposition
- Risk level classification
- Denoising capabilities
- Performance requirements
"""

import numpy as np
import pytest
from src.risk.physics.correlation_sensor import CorrelationSensor


class TestCorrelationSensor:
    """Test suite for CorrelationSensor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sensor = CorrelationSensor(cache_size=10)

    def test_initialization(self):
        """Test sensor initialization with default parameters."""
        sensor = CorrelationSensor()
        assert sensor.cache_size == 100
        assert sensor.normalize_returns is True

        sensor = CorrelationSensor(cache_size=50, normalize_returns=False)
        assert sensor.cache_size == 50
        assert sensor.normalize_returns is False

    def test_input_validation(self):
        """Test input validation for returns matrix."""
        # Valid input
        returns = np.random.randn(5, 100)  # 5 assets, 100 periods
        self.sensor._validate_input(returns)  # Should not raise

        # Invalid: not a numpy array
        with pytest.raises(ValueError):
            self.sensor._validate_input("not an array")

        # Invalid: wrong dimensions
        with pytest.raises(ValueError):
            self.sensor._validate_input(np.random.randn(5, 5, 5))

        # Invalid: too few assets
        with pytest.raises(ValueError):
            self.sensor._validate_input(np.random.randn(1, 100))

        # Invalid: too few periods
        with pytest.raises(ValueError):
            self.sensor._validate_input(np.random.randn(5, 19))  # 19 periods < 20

    def test_normalize_returns(self):
        """Test returns normalization."""
        returns = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        # Test with normalization enabled (default)
        normalized = self.sensor._normalize_returns(returns)
        assert np.allclose(np.mean(normalized, axis=1), 0)
        assert np.allclose(np.std(normalized, axis=1), 1)

        # Test with normalization disabled
        sensor_no_normalize = CorrelationSensor(normalize_returns=False)
        normalized_no = sensor_no_normalize._normalize_returns(returns)
        assert np.array_equal(normalized_no, returns)

    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        returns = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])

        normalized = self.sensor._normalize_returns(returns)
        corr_matrix = self.sensor._calculate_correlation_matrix(normalized)

        # Check properties of correlation matrix
        assert corr_matrix.shape == (3, 3)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
        assert np.allclose(corr_matrix, corr_matrix.T)  # Should be symmetric
        assert np.all(corr_matrix >= -1.0)  # Within valid range
        assert np.all(corr_matrix <= 1.0)  # Within valid range

    def test_marchenko_pastur_threshold(self):
        """Test Marchenko-Pastur threshold calculation."""
        # Test with different asset/period ratios
        thresholds = []
        for n_assets in [5, 10, 20]:
            for t_periods in [50, 100, 200]:
                threshold = self.sensor._calculate_marchenko_pastur_threshold(n_assets, t_periods)
                thresholds.append(threshold)
                assert threshold > 0  # Should be positive

        # Threshold should increase with more periods relative to assets
        assert thresholds[0] < thresholds[1]  # 5/50 < 5/100
        assert thresholds[1] < thresholds[2]  # 5/100 < 5/200

    def test_eigenvalue_decomposition(self):
        """Test eigenvalue decomposition."""
        # Create a symmetric matrix
        matrix = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.7],
            [0.3, 0.7, 1.0]
        ])

        eigenvalues, eigenvectors = self.sensor._perform_eigenvalue_decomposition(matrix)

        # Check properties
        assert len(eigenvalues) == 3
        assert eigenvalues.shape == (3,)
        assert eigenvectors.shape == (3, 3)

        # Eigenvalues should be sorted in descending order
        assert np.all(eigenvalues[1:] <= eigenvalues[:-1])

        # Reconstruct matrix to verify decomposition
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        assert np.allclose(matrix, reconstructed)

    def test_denoise_correlation_matrix(self):
        """Test denoising of correlation matrix."""
        # Create a correlation matrix with noise
        np.random.seed(42)
        n = 10
        matrix = np.eye(n)  # Identity matrix (no correlation)

        # Add some noise eigenvalues
        eigenvalues = np.ones(n)
        eigenvalues[0] = 3.0  # Signal eigenvalue
        eigenvalues[1] = 2.5  # Signal eigenvalue
        eigenvalues[2:] = np.random.uniform(0.1, 1.2, n-2)  # Noise eigenvalues

        # Create orthogonal matrix
        H = np.random.randn(n, n)
        Q, _ = np.linalg.qr(H)

        # Reconstruct matrix
        correlation = Q @ np.diag(eigenvalues) @ Q.T
        np.fill_diagonal(correlation, 1.0)

        # Calculate threshold
        threshold = self.sensor._calculate_marchenko_pastur_threshold(n, 100)

        # Denoise
        denoised = self.sensor._denoise_correlation_matrix(correlation, eigenvalues, Q, threshold)

        # Check that noise eigenvalues are replaced
        denoised_eigenvalues, _ = self.sensor._perform_eigenvalue_decomposition(denoised)
        assert np.all(denoised_eigenvalues[2:] <= threshold + 0.1)  # Noise eigenvalues should be capped

    def test_risk_level_classification(self):
        """Test risk level classification."""
        assert self.sensor._classify_risk_level(1.0) == "LOW"
        assert self.sensor._classify_risk_level(1.4) == "LOW"
        assert self.sensor._classify_risk_level(1.5) == "MODERATE"
        assert self.sensor._classify_risk_level(1.8) == "MODERATE"
        assert self.sensor._classify_risk_level(2.0) == "HIGH"
        assert self.sensor._classify_risk_level(3.0) == "HIGH"

    def test_detect_systemic_risk(self):
        """Test end-to-end systemic risk detection."""
        # Create synthetic returns data
        np.random.seed(42)
        n_assets = 10
        t_periods = 100

        # Generate random returns
        returns = np.random.randn(n_assets, t_periods)

        # Detect systemic risk
        result = self.sensor.detect_systemic_risk(returns)

        # Check result structure
        assert "max_eigenvalue" in result
        assert "noise_threshold" in result
        assert "denoised_matrix" in result
        assert "risk_level" in result
        assert "eigenvalues" in result
        assert "eigenvectors" in result

        # Check types
        assert isinstance(result["max_eigenvalue"], float)
        assert isinstance(result["noise_threshold"], float)
        assert isinstance(result["risk_level"], str)
        assert isinstance(result["eigenvalues"], np.ndarray)
        assert isinstance(result["eigenvectors"], np.ndarray)

        # Check values
        assert result["max_eigenvalue"] > 0
        assert result["noise_threshold"] > 0
        assert result["risk_level"] in ["LOW", "MODERATE", "HIGH"]

    def test_cache_functionality(self):
        """Test caching functionality."""
        returns = np.random.randn(5, 100)

        # First call - should compute
        result1 = self.sensor.detect_systemic_risk(returns)

        # Second call with same data - should use cache
        result2 = self.sensor.detect_systemic_risk(returns)

        # Results should be identical
        assert np.allclose(result1["max_eigenvalue"], result2["max_eigenvalue"])
        assert np.allclose(result1["noise_threshold"], result2["noise_threshold"])
        assert np.array_equal(result1["denoised_matrix"], result2["denoised_matrix"])
        assert result1["risk_level"] == result2["risk_level"]
        assert np.array_equal(result1["eigenvalues"], result2["eigenvalues"])
        assert np.array_equal(result1["eigenvectors"], result2["eigenvectors"])

    def test_clear_cache(self):
        """Test cache clearing."""
        returns = np.random.randn(5, 100)

        # Initial call
        result1 = self.sensor.detect_systemic_risk(returns)

        # Clear cache
        self.sensor.clear_cache()

        # Second call should recompute
        result2 = self.sensor.detect_systemic_risk(returns)

        # Results should be identical (same computation)
        assert np.allclose(result1["max_eigenvalue"], result2["max_eigenvalue"])

    def test_performance(self):
        """Test performance requirements (< 150ms)."""
        import time

        # Create larger dataset
        returns = np.random.randn(20, 200)  # 20 assets, 200 periods

        # Measure execution time
        start_time = time.time()
        self.sensor.detect_systemic_risk(returns)
        end_time = time.time()

        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        assert execution_time < 150, f"Performance test failed: {execution_time:.2f}ms > 150ms"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.sensor = CorrelationSensor()

    def test_minimum_assets(self):
        """Test with minimum number of assets."""
        returns = np.random.randn(2, 100)  # Minimum 2 assets
        result = self.sensor.detect_systemic_risk(returns)
        assert result["max_eigenvalue"] > 0

    def test_minimum_periods(self):
        """Test with minimum number of periods."""
        returns = np.random.randn(5, 20)  # Minimum 20 periods
        result = self.sensor.detect_systemic_risk(returns)
        assert result["max_eigenvalue"] > 0

    def test_singular_matrix(self):
        """Test with potentially singular correlation matrix."""
        # Create returns that might lead to singular correlation
        returns = np.ones((5, 100))  # All returns are 1 (highly correlated)
        result = self.sensor.detect_systemic_risk(returns)
        assert result["max_eigenvalue"] > 0

    def test_nan_values(self):
        """Test with NaN values in returns."""
        returns = np.random.randn(5, 100)
        returns[0, 0] = np.nan  # Add a NaN value

        # Should handle NaNs gracefully
        result = self.sensor.detect_systemic_risk(returns)
        assert result["max_eigenvalue"] > 0


if __name__ == "__main__":
    pytest.main([__file__])