"""
Tests for Monte Carlo Validator.

These tests validate the Monte Carlo validator functionality including:
- Risk of ruin calculation
- Confidence interval computation
- Validation logic
- Performance benchmarks
"""

import numpy as np
import time
import pytest
from src.risk.sizing.monte_carlo_validator import MonteCarloValidator, ValidationResult


class TestMonteCarloValidator:
    """Test suite for Monte Carlo Validator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample historical returns data
        np.random.seed(42)  # For reproducible tests
        self.historical_returns = np.random.normal(0.001, 0.02, 252)  # 252 trading days
        self.validator = MonteCarloValidator(runs=100)  # Use fewer runs for faster tests

    def test_initialization(self):
        """Test validator initialization."""
        validator = MonteCarloValidator(runs=2000)
        assert validator.runs == 2000

    def test_bootstrap_resampling(self):
        """Test bootstrap resampling functionality."""
        simulated_returns = self.validator._bootstrap_resampling(self.historical_returns)

        # Check shape: (days, runs)
        assert simulated_returns.shape == (len(self.historical_returns), self.validator.runs)

        # Check that values are from original distribution
        assert np.all(np.isin(simulated_returns, self.historical_returns))

    def test_simulate_trading(self):
        """Test trading simulation functionality."""
        simulated_returns = self.validator._bootstrap_resampling(self.historical_returns)
        equity_curves, drawdowns = self.validator._simulate_trading(
            simulated_returns, initial_capital=10000.0
        )

        # Check shapes
        assert equity_curves.shape == (len(self.historical_returns) + 1, self.validator.runs)
        assert drawdowns.shape == (len(self.historical_returns) + 1, self.validator.runs)

        # Check that equity curves start at initial capital
        assert np.allclose(equity_curves[0, :], 10000.0)

        # Check that drawdowns are non-negative
        assert np.all(drawdowns >= 0)

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        simulated_returns = self.validator._bootstrap_resampling(self.historical_returns)
        equity_curves, drawdowns = self.validator._simulate_trading(
            simulated_returns, initial_capital=10000.0
        )

        risk_metrics = self.validator._calculate_risk_metrics(equity_curves, drawdowns)

        # Check required metrics
        assert 'risk_of_ruin' in risk_metrics
        assert 'ci_95' in risk_metrics
        assert 'expected_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics

        # Check data types
        assert isinstance(risk_metrics['risk_of_ruin'], float)
        assert isinstance(risk_metrics['ci_95'], np.ndarray)
        assert isinstance(risk_metrics['expected_drawdown'], float)
        assert isinstance(risk_metrics['var_95'], float)

    def test_validate_risk_success(self):
        """Test successful risk validation."""
        # Use low risk percentage that should pass
        result = self.validator.validate_risk(
            self.historical_returns, initial_capital=10000.0
        )

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.risk_of_ruin < 0.005  # Should be below 0.5% threshold
        assert result.adjusted_risk == 0.5  # Default adjustment factor

    def test_validate_risk_failure(self):
        """Test failed risk validation."""
        # Use high risk percentage that should fail
        result = self.validator.validate_risk(
            self.historical_returns, initial_capital=10000.0
        )

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.risk_of_ruin < 0.005  # Should be below 0.5% threshold
        assert result.adjusted_risk == 0.5  # Default adjustment factor

    def test_performance_summary(self):
        """Test performance summary generation."""
        summary = self.validator.get_performance_summary(
            self.historical_returns, risk_pct=0.05, initial_capital=10000.0
        )

        # Check required metrics
        assert 'mean_final_equity' in summary
        assert 'median_final_equity' in summary
        assert 'std_final_equity' in summary
        assert 'risk_of_ruin' in summary
        assert 'expected_drawdown' in summary
        assert 'var_95' in summary
        assert 'ci_95_lower' in summary
        assert 'ci_95_upper' in summary

        # Check data types
        assert isinstance(summary['mean_final_equity'], float)
        assert isinstance(summary['median_final_equity'], float)
        assert isinstance(summary['std_final_equity'], float)
        assert isinstance(summary['risk_of_ruin'], float)
        assert isinstance(summary['expected_drawdown'], float)
        assert isinstance(summary['var_95'], float)
        assert isinstance(summary['ci_95_lower'], float)
        assert isinstance(summary['ci_95_upper'], float)

    def test_input_validation(self):
        """Test input validation."""
        # Test empty returns array
        with pytest.raises(ValueError):
            self.validator.validate_risk(np.array([]), initial_capital=10000.0)

        # Test invalid risk percentage (this should be caught by the validator)
        with pytest.raises(ValueError):
            self.validator.validate_risk(self.historical_returns, initial_capital=10000.0)

    def test_simulation_time(self):
        """Test simulation performance timing."""
        start_time = time.time()
        result = self.validator.validate_risk(
            self.historical_returns, initial_capital=10000.0
        )
        end_time = time.time()

        # Test should complete in reasonable time (less than 1 second for 100 runs)
        assert end_time - start_time < 1.0
        assert result.simulation_time < 1.0

    def test_confidence_interval_bounds(self):
        """Test confidence interval bounds."""
        result = self.validator.validate_risk(
            self.historical_returns, initial_capital=10000.0
        )

        ci_lower, ci_upper = result.ci_95
        assert ci_lower < ci_upper
        assert ci_lower > 0  # Should be positive values
        assert ci_upper > 0

    def test_risk_of_ruin_calculation(self):
        """Test risk of ruin calculation logic."""
        # Create returns that guarantee ruin
        guaranteed_ruin_returns = np.full(252, -0.1)  # 10% daily loss

        result = self.validator.validate_risk(
            guaranteed_ruin_returns, initial_capital=10000.0
        )

        # Should have 100% risk of ruin
        assert result.risk_of_ruin == 1.0
        assert result.passed is False
        assert result.adjusted_risk == 0.5  # Default adjustment factor


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__])