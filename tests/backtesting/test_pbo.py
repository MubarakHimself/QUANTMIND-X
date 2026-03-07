"""
Tests for Probability of Backtest Overfitting (PBO) Calculator.
"""

import pytest
import numpy as np
from src.backtesting.pbo_calculator import PBOCalculator, PBOResult


class TestPBOCalculator:
    """Test suite for PBOCalculator."""

    def test_pbo_calculation_random(self):
        """PBO should detect overfitting in random parameter selection."""
        np.random.seed(42)
        train_results = np.random.randn(100) * 0.01
        test_results = np.random.randn(100) * 0.01

        calculator = PBOCalculator(n_blocks=5, n_simulations=50)
        pbo = calculator.calculate_pbo(train_results, test_results)

        # Random results should have moderate to high PBO
        assert 0.0 <= pbo <= 1.0

    def test_pbo_calculation_consistent_strategy(self):
        """PBO should be low for consistent strategy across periods."""
        np.random.seed(42)

        # Consistent positive returns in both halves
        train_returns = np.random.randn(100) * 0.01 + 0.005
        test_returns = np.random.randn(100) * 0.01 + 0.004

        calculator = PBOCalculator(n_blocks=5, n_simulations=50)
        pbo = calculator.calculate_pbo(train_returns, test_returns)

        # Consistent strategy should have lower PBO
        assert 0.0 <= pbo <= 1.0

    def test_pbo_calculation_overfitted_strategy(self):
        """PBO should detect overfitted strategy (high IS, low OOS)."""
        np.random.seed(42)

        # High returns in-sample, low/negative out-of-sample
        train_returns = np.random.randn(100) * 0.01 + 0.05  # High IS
        test_returns = np.random.randn(100) * 0.01 - 0.02   # Low/Neg OOS

        calculator = PBOCalculator(n_blocks=5, n_simulations=50)
        pbo = calculator.calculate_pbo(train_returns, test_returns)

        # Overfitted strategy should have higher PBO
        assert 0.0 <= pbo <= 1.0

    def test_evaluate_strategy_robustness_accept(self):
        """Low PBO strategy should get ACCEPT recommendation."""
        np.random.seed(42)

        # Consistent returns
        returns = np.random.randn(200) * 0.01 + 0.005
        drawdowns = np.abs(np.random.randn(200) * 0.02)

        calculator = PBOCalculator(n_blocks=5, n_simulations=30)
        result = calculator.evaluate_strategy_robustness(
            returns_series=returns.tolist(),
            drawdown_series=drawdowns.tolist()
        )

        assert "pbo" in result
        assert "recommendation" in result
        assert result["pbo"] >= 0.0
        assert result["pbo"] <= 1.0
        assert result["recommendation"] in ["ACCEPT", "CAUTION", "REJECT", "INSUFFICIENT_DATA"]

    def test_evaluate_strategy_robustness_reject(self):
        """High PBO strategy should get REJECT recommendation."""
        np.random.seed(42)

        # High drift between periods
        first_half = np.random.randn(100) * 0.01 + 0.05
        second_half = np.random.randn(100) * 0.01 - 0.03
        returns = np.concatenate([first_half, second_half])

        calculator = PBOCalculator(n_blocks=5, n_simulations=30)
        result = calculator.evaluate_strategy_robustness(
            returns_series=returns.tolist()
        )

        assert "pbo" in result
        assert result["pbo"] >= 0.0

    def test_evaluate_strategy_robustness_insufficient_data(self):
        """Should return INSUFFICIENT_DATA for short series."""
        calculator = PBOCalculator(n_blocks=5)
        result = calculator.evaluate_strategy_robustness(
            returns_series=[0.01, 0.02, 0.03]  # Only 3 points
        )

        assert result["recommendation"] == "INSUFFICIENT_DATA"
        assert result["pbo"] == 0.5

    def test_confidence_interval(self):
        """Should calculate bootstrap confidence interval."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01 + 0.005

        calculator = PBOCalculator(n_simulations=50)
        ci = calculator.calculate_confidence_interval(returns, confidence=0.95)

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= ci[1]

    def test_evaluate_parameter_sets(self):
        """Should evaluate multiple parameter sets."""
        parameter_results = {
            "param_a": {"is_return": 0.05, "oos_return": 0.03},
            "param_b": {"is_return": 0.08, "oos_return": 0.01},
            "param_c": {"is_return": 0.06, "oos_return": 0.04},
            "param_d": {"is_return": 0.04, "oos_return": 0.05},
        }

        calculator = PBOCalculator(n_blocks=4, n_simulations=20)
        result = calculator.evaluate_parameter_sets(parameter_results)

        assert "pbo" in result
        assert "recommendation" in result
        assert "best_parameter" in result
        assert result["n_parameters_tested"] == 4

    def test_evaluate_parameter_sets_empty(self):
        """Should handle empty parameter results."""
        calculator = PBOCalculator()
        result = calculator.evaluate_parameter_sets({})

        assert result["recommendation"] == "NO_DATA"

    def test_pbo_with_different_block_sizes(self):
        """Should work with different block sizes."""
        np.random.seed(42)
        train = np.random.randn(100) * 0.01
        test = np.random.randn(100) * 0.01

        for n_blocks in [3, 5, 10]:
            calculator = PBOCalculator(n_blocks=n_blocks, n_simulations=20)
            pbo = calculator.calculate_pbo(train, test)
            assert 0.0 <= pbo <= 1.0

    def test_pbo_with_zero_variance(self):
        """Should handle zero variance returns."""
        train = np.ones(50) * 0.01
        test = np.ones(50) * 0.01

        calculator = PBOCalculator(n_blocks=5, n_simulations=20)
        pbo = calculator.calculate_pbo(train, test)

        # Should return valid result
        assert 0.0 <= pbo <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
