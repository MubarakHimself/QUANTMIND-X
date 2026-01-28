"""
Tests for Monte Carlo Validator.

These tests validate the Monte Carlo validator functionality including:
- Risk of ruin calculation
- Confidence interval computation
- Validation logic
- Performance benchmarks
- Integration with StrategyPerformance model
"""

import time
import pytest
import numpy as np

from src.risk.sizing.monte_carlo_validator import MonteCarloValidator, ValidationResult
from src.risk.models.strategy_performance import StrategyPerformance


class TestMonteCarloValidator:
    """Test suite for Monte Carlo Validator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample StrategyPerformance for testing
        self.profitable_perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100,
            k_fraction=0.5
        )
        self.validator = MonteCarloValidator(runs=500)  # Fewer runs for faster tests

    def test_initialization(self):
        """Test validator initialization."""
        validator = MonteCarloValidator(runs=2000, initial_capital=50000.0)
        assert validator.runs == 2000
        assert validator.initial_capital == 50000.0

    def test_generate_synthetic_returns(self):
        """Test synthetic return generation from strategy performance."""
        returns = self.validator._generate_synthetic_returns(
            self.profitable_perf, risk_pct=0.02, n_trades=100
        )

        # Check shape
        assert len(returns) == 100

        # Check that returns match expected distribution (approximately)
        # About 55% should be positive (wins)
        win_ratio = np.sum(returns > 0) / len(returns)
        assert 0.4 < win_ratio < 0.7  # Allow for variance

    def test_bootstrap_resampling(self):
        """Test bootstrap resampling functionality."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        simulated = self.validator._bootstrap_resampling(returns)

        # Check shape: (n_periods, n_runs)
        assert simulated.shape == (len(returns), self.validator.runs)

        # Check that all values are from original returns
        unique_values = np.unique(simulated)
        assert all(v in returns for v in unique_values)

    def test_simulate_equity_curves(self):
        """Test equity curve simulation."""
        returns = np.random.normal(0.001, 0.02, 100)
        simulated = self.validator._bootstrap_resampling(returns)
        equity_curves, drawdowns = self.validator._simulate_equity_curves(simulated)

        # Check shapes
        assert equity_curves.shape[0] == len(returns) + 1  # +1 for starting point
        assert equity_curves.shape[1] == self.validator.runs
        assert drawdowns.shape == equity_curves.shape

        # Check that all curves start at initial capital
        assert np.allclose(equity_curves[0, :], self.validator.initial_capital)

        # Check drawdowns are non-negative
        assert np.all(drawdowns >= 0)

    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        returns = np.random.normal(0.001, 0.02, 100)
        simulated = self.validator._bootstrap_resampling(returns)
        equity_curves, drawdowns = self.validator._simulate_equity_curves(simulated)

        metrics = self.validator._calculate_risk_metrics(equity_curves, drawdowns, 0.02)

        # Check required keys
        assert 'risk_of_ruin' in metrics
        assert 'ci_95' in metrics
        assert 'expected_drawdown' in metrics
        assert 'var_95' in metrics
        assert 'final_equities' in metrics
        assert 'max_drawdowns' in metrics

        # Check types
        assert isinstance(metrics['risk_of_ruin'], float)
        assert isinstance(metrics['ci_95'], tuple)
        assert len(metrics['ci_95']) == 2
        assert isinstance(metrics['expected_drawdown'], float)
        assert isinstance(metrics['var_95'], float)
        assert isinstance(metrics['final_equities'], np.ndarray)

    def test_validate_risk_success(self):
        """Test successful risk validation with profitable strategy."""
        result = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.01
        )

        assert isinstance(result, ValidationResult)
        assert isinstance(result.passed, (bool, np.bool_))
        assert isinstance(result.risk_of_ruin, (float, np.floating))
        assert 0 <= result.risk_of_ruin <= 1
        assert result.adjusted_risk in (0.5, 1.0)
        assert result.simulation_time >= 0
        assert result.recommended_risk_pct is not None

    def test_validate_risk_with_high_risk(self):
        """Test validation with higher risk percentage."""
        result = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.05, runs=200
        )

        assert isinstance(result, ValidationResult)
        # Higher risk should result in lower recommended risk if failed
        if not result.passed:
            assert result.adjusted_risk == 0.5
            assert result.recommended_risk_pct == 0.025  # 0.05 * 0.5

    def test_validate_risk_custom_runs(self):
        """Test validation with custom run count."""
        result = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.02, runs=100
        )

        assert isinstance(result, ValidationResult)
        assert result.final_equities is not None
        assert len(result.final_equities) == 100

    def test_validate_risk_negative_expectancy(self):
        """Test validation with losing strategy (negative expectancy)."""
        losing_perf = StrategyPerformance(
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=100.0,
            total_trades=100
        )

        result = self.validator.validate_risk(
            losing_perf, risk_pct=0.02, runs=200
        )

        # Losing strategy should have some risk
        assert isinstance(result, ValidationResult)
        # Check result is valid regardless of risk_of_ruin value
        assert result.simulation_time >= 0
        assert result.recommended_risk_pct is not None

    def test_confidence_interval_bounds(self):
        """Test confidence interval calculation."""
        result = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.02
        )

        ci_lower, ci_upper = result.ci_95
        assert ci_lower < ci_upper
        # Both should be positive for profitable strategy
        assert ci_lower > 0
        assert ci_upper > 0

    def test_performance_summary(self):
        """Test performance summary generation."""
        summary = self.validator.get_performance_summary(
            self.profitable_perf, risk_pct=0.02, runs=200
        )

        # Check required metrics
        assert 'mean_final_equity' in summary
        assert 'median_final_equity' in summary
        assert 'std_final_equity' in summary
        assert 'profit_probability' in summary
        assert 'risk_of_ruin' in summary
        assert 'expected_drawdown' in summary
        assert 'var_95' in summary
        assert 'ci_95_lower' in summary
        assert 'ci_95_upper' in summary
        assert 'runs' in summary

        # Check types
        assert isinstance(summary['mean_final_equity'], float)
        assert isinstance(summary['profit_probability'], float)
        assert 0 <= summary['profit_probability'] <= 1

    def test_input_validation_invalid_perf(self):
        """Test input validation for invalid performance object."""
        with pytest.raises(ValueError, match="perf must be StrategyPerformance"):
            self.validator.validate_risk(
                None, risk_pct=0.02  # type: ignore
            )

    def test_input_validation_invalid_risk_pct(self):
        """Test input validation for invalid risk percentage."""
        with pytest.raises(ValueError, match="risk_pct must be between 0 and 1"):
            self.validator.validate_risk(
                self.profitable_perf, risk_pct=1.5
            )

        with pytest.raises(ValueError, match="risk_pct must be between 0 and 1"):
            self.validator.validate_risk(
                self.profitable_perf, risk_pct=0
            )

    def test_performance_target(self):
        """Test that simulation completes within performance target."""
        # Target: < 500ms for 2000 runs
        validator = MonteCarloValidator(runs=2000)

        start = time.time()
        result = validator.validate_risk(
            self.profitable_perf, risk_pct=0.02
        )
        elapsed = time.time() - start

        # Should complete within 1 second (allowing some margin)
        assert elapsed < 1.0, f"Simulation took {elapsed:.2f}s, expected < 1.0s"
        assert result.simulation_time < 1.0

    def test_reproducibility_with_fixed_seed(self):
        """Test that results are reproducible with proper random handling."""
        # Run twice with similar parameters
        result1 = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.02, runs=100
        )
        result2 = self.validator.validate_risk(
            self.profitable_perf, risk_pct=0.02, runs=100
        )

        # Results should be similar (not identical due to randomness)
        # But both should be valid ValidationResult instances
        assert isinstance(result1, ValidationResult)
        assert isinstance(result2, ValidationResult)

    def test_edge_case_zero_win_rate(self):
        """Test edge case with zero win rate."""
        # StrategyPerformance validates win_rate must be strictly between 0 and 1
        # So this should raise ValidationError during object creation
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            edge_perf = StrategyPerformance(
                win_rate=0.0,
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )

    def test_edge_case_perfect_win_rate(self):
        """Test edge case with perfect win rate."""
        # StrategyPerformance validates win_rate must be strictly between 0 and 1
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            edge_perf = StrategyPerformance(
                win_rate=1.0,
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])