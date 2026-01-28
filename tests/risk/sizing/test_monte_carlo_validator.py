"""
Tests for Monte Carlo Validator

These tests validate the Monte Carlo simulation engine and risk assessment functionality.
"""

import numpy as np
import pytest
from src.risk.sizing.monte_carlo_validator import MonteCarloValidator, MonteCarloResults


class TestMonteCarloValidator:
    """Test suite for Monte Carlo Validator."""

    def setup_method(self):
        """Set up test data."""
        # Create sample returns data
        np.random.seed(42)
        self.sample_returns = np.random.normal(0.001, 0.02, 252)  # 252 trading days
        self.validator = MonteCarloValidator(
            returns=self.sample_returns,
            simulations=500,  # Reduced for faster testing
            days=252,
            start_capital=10000.0,
            random_seed=42
        )

    def test_initialization(self):
        """Test validator initialization."""
        assert self.validator.returns is not None
        assert self.validator.simulations == 500
        assert self.validator.days == 252
        assert self.validator.start_capital == 10000.0

    def test_run_simulation_shape(self):
        """Test simulation output shapes."""
        results = self.validator.run_simulation()

        # Check shapes
        assert results.paths.shape == (253, 500)  # days + 1, simulations
        assert results.final_values.shape == (500,)
        assert results.mean_path.shape == (253,)
        assert results.upper_path.shape == (253,)
        assert results.lower_path.shape == (253,)

    def test_simulation_statistics(self):
        """Test simulation statistics calculation."""
        results = self.validator.run_simulation()

        # Check that statistics are calculated
        assert isinstance(results.median_profit, float)
        assert isinstance(results.expected_return, float)
        assert isinstance(results.risk_of_ruin, float)
        assert isinstance(results.var_95, float)
        assert isinstance(results.var_99, float)

        # Check reasonable values
        assert -1.0 <= results.expected_return <= 1.0
        assert 0.0 <= results.risk_of_ruin <= 1.0
        assert results.var_95 <= results.var_99  # 95% VaR should be less negative than 99% VaR

    def test_validate_risk(self):
        """Test risk validation logic."""
        is_valid, risk_metrics = self.validator.validate_risk()

        assert isinstance(is_valid, bool)
        assert isinstance(risk_metrics, dict)

        # Check risk metrics are present
        assert 'risk_of_ruin' in risk_metrics
        assert 'expected_return' in risk_metrics
        assert 'var_95' in risk_metrics

    def test_get_risk_summary(self):
        """Test risk summary generation."""
        risk_summary = self.validator.get_risk_summary()

        assert isinstance(risk_summary, dict)
        assert 'risk_of_ruin' in risk_summary
        assert 'expected_return' in risk_summary
        assert 'var_95' in risk_summary

    def test_halve_risk(self):
        """Test risk halving calculation."""
        current_position = 1000.0
        new_position = self.validator.halve_risk(current_position)

        assert new_position == 500.0  # Should be exactly half

    def test_vectorized_bootstrap(self):
        """Test vectorized bootstrap functionality."""
        bootstrap_results = self.validator.run_vectorized_bootstrap(n_bootstrap=100)

        assert bootstrap_results.shape == (100, 2)  # 100 samples, 2 stats (mean, std)
        assert bootstrap_results.dtype == np.float64

    def test_reproducibility(self):
        """Test that simulations are reproducible with seed."""
        validator1 = MonteCarloValidator(
            returns=self.sample_returns,
            simulations=100,
            days=252,
            start_capital=10000.0,
            random_seed=42
        )

        validator2 = MonteCarloValidator(
            returns=self.sample_returns,
            simulations=100,
            days=252,
            start_capital=10000.0,
            random_seed=42
        )

        results1 = validator1.run_simulation()
        results2 = validator2.run_simulation()

        # Check that final values are identical (due to same seed)
        np.testing.assert_array_equal(results1.final_values, results2.final_values)

    def test_empty_returns(self):
        """Test behavior with empty returns data."""
        empty_returns = np.array([])
        validator = MonteCarloValidator(
            returns=empty_returns,
            simulations=100,
            days=252,
            start_capital=10000.0,
            random_seed=42
        )

        # Should raise an error or handle gracefully
        with pytest.raises(ValueError):
            validator.run_simulation()

    def test_performance(self):
        """Test performance requirements (target < 500ms for 2000 runs)."""
        import time

        start_time = time.time()

        # Create a validator with 2000 simulations
        performance_validator = MonteCarloValidator(
            returns=self.sample_returns,
            simulations=2000,
            days=252,
            start_capital=10000.0,
            random_seed=42
        )

        results = performance_validator.run_simulation()
        duration = time.time() - start_time

        print(f"Performance test: {duration:.2f}s for 2000 simulations")
        assert duration < 0.5  # Should complete in less than 500ms


class TestMonteCarloResults:
    """Test suite for MonteCarloResults dataclass."""

    def test_results_initialization(self):
        """Test MonteCarloResults initialization."""
        # Create mock data
        paths = np.random.rand(10, 100)
        final_values = np.random.rand(100)
        mean_path = np.mean(paths, axis=1)
        upper_path = np.percentile(paths, 95, axis=1)
        lower_path = np.percentile(paths, 5, axis=1)

        results = MonteCarloResults(
            paths=paths,
            final_values=final_values,
            mean_path=mean_path,
            upper_path=upper_path,
            lower_path=lower_path,
            median_profit=100.0,
            expected_return=0.01,
            risk_of_ruin=0.05,
            var_95=9000.0,
            var_99=8000.0,
            ci_95_lower=9500.0,
            ci_95_upper=10500.0,
            ci_99_lower=9000.0,
            ci_99_upper=11000.0
        )

        assert results.paths is not None
        assert results.final_values is not None
        assert results.mean_path is not None
        assert results.upper_path is not None
        assert results.lower_path is not None
        assert results.median_profit == 100.0
        assert results.expected_return == 0.01
        assert results.risk_of_ruin == 0.05
        assert results.var_95 == 9000.0
        assert results.var_99 == 8000.0
        assert results.ci_95_lower == 9500.0
        assert results.ci_95_upper == 10500.0
        assert results.ci_99_lower == 9000.0
        assert results.ci_99_upper == 11000.0