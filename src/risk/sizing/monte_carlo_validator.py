"""
Monte Carlo Validator for Risk Assessment

This module implements a Monte Carlo validator for position sizing risk assessment.
It uses bootstrap resampling to simulate trading performance and calculate risk metrics.

Key features:
- Vectorized bootstrap resampling for performance
- Risk of ruin calculation
- 95% confidence intervals
- Risk adjustment logic
- Comprehensive validation results
"""

import numpy as np
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import time

@dataclass
class ValidationResult:
    """Container for validation results."""
    passed: bool
    risk_of_ruin: float
    adjusted_risk: float
    ci_95: Tuple[float, float]
    expected_drawdown: float
    var_95: float
    simulation_time: float

class MonteCarloValidator:
    """
    Monte Carlo validator for risk assessment using bootstrap resampling.

    This class simulates trading performance using historical returns to estimate
    risk metrics including risk of ruin, drawdowns, and confidence intervals.
    """

    def __init__(self, runs: int = 2000):
        """
        Initialize the Monte Carlo validator.

        Args:
            runs: Number of simulation runs (default: 2000)
        """
        self.runs = runs
        self.start_time = None

    def _bootstrap_resampling(self, historical_returns: np.ndarray) -> np.ndarray:
        """
        Perform vectorized bootstrap resampling of historical returns.

        Args:
            historical_returns: Array of historical returns

        Returns:
            Array of simulated returns for all runs
        """
        n_days = len(historical_returns)
        # Vectorized sampling with replacement
        random_indices = np.random.randint(0, n_days, size=(n_days, self.runs))
        return historical_returns[random_indices]

    def _simulate_trading(self, simulated_returns: np.ndarray,
                        initial_capital: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trading performance.

        Args:
            simulated_returns: Simulated returns from bootstrap resampling
            initial_capital: Starting capital

        Returns:
            Tuple of (equity_curves, drawdowns)
        """
        # Calculate daily returns for each simulation
        daily_returns = 1 + simulated_returns

        # Calculate cumulative returns and equity curves
        cumulative_returns = np.cumprod(daily_returns, axis=0)
        equity_curves = initial_capital * cumulative_returns

        # Add starting point (Day 0) - ensure correct shape
        equity_curves = np.vstack([np.full((1, self.runs), initial_capital), equity_curves])

        # Calculate drawdowns
        peak_equity = np.maximum.accumulate(equity_curves, axis=0)
        drawdowns = (peak_equity - equity_curves) / peak_equity

        return equity_curves, drawdowns

    def _calculate_risk_metrics(self, equity_curves: np.ndarray,
                             drawdowns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate risk metrics from simulation results.

        Args:
            equity_curves: Array of equity curves from simulations
            drawdowns: Array of drawdowns from simulations

        Returns:
            Dictionary of risk metrics
        """
        final_equities = equity_curves[-1, :]
        max_drawdowns = np.max(drawdowns, axis=0)

        # Risk of ruin (equity drops to zero or below)
        # Use a tolerance for floating point comparisons
        tolerance = 1e-10
        ruin_count = np.sum(final_equities <= tolerance)
        risk_of_ruin = ruin_count / self.runs if self.runs > 0 else 0.0

        # For guaranteed ruin scenario, force risk_of_ruin to 1.0
        if np.all(final_equities <= tolerance):
            risk_of_ruin = 1.0
        elif np.any(final_equities <= tolerance):
            # If any simulation results in ruin, calculate proportion
            risk_of_ruin = np.mean(final_equities <= tolerance)

        # 95% confidence interval
        ci_95 = np.percentile(final_equities, [2.5, 97.5])

        # Expected drawdown
        expected_drawdown = np.mean(max_drawdowns)

        # Value at Risk (95%)
        var_95 = np.percentile(final_equities, 5)

        return {
            'risk_of_ruin': risk_of_ruin,
            'ci_95': ci_95,
            'expected_drawdown': expected_drawdown,
            'var_95': var_95,
            'final_equities': final_equities,
            'max_drawdowns': max_drawdowns
        }

    def validate_risk(self, historical_returns: np.ndarray,
                    initial_capital: float = 10000.0) -> ValidationResult:
        """
        Validate risk level using Monte Carlo simulation.

        Args:
            historical_returns: Array of historical returns
            initial_capital: Starting capital (default: 10000.0)

        Returns:
            ValidationResult with risk assessment
        """
        self.start_time = time.time()

        # Validate inputs
        if not isinstance(historical_returns, np.ndarray):
            raise ValueError("historical_returns must be a numpy array")
        if len(historical_returns) == 0:
            raise ValueError("historical_returns cannot be empty")

        # Perform bootstrap resampling
        simulated_returns = self._bootstrap_resampling(historical_returns)

        # Simulate trading performance
        equity_curves, drawdowns = self._simulate_trading(
            simulated_returns, initial_capital
        )

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curves, drawdowns)

        # Determine if risk is acceptable
        threshold = 0.005  # 0.5% risk of ruin threshold
        passed = risk_metrics['risk_of_ruin'] < threshold

        # Calculate adjusted risk if failed
        adjusted_risk = 0.5 if not passed else 1.0  # Default adjustment factor for failed validation

        # Create validation result
        result = ValidationResult(
            passed=passed,
            risk_of_ruin=risk_metrics['risk_of_ruin'],
            adjusted_risk=adjusted_risk,
            ci_95=risk_metrics['ci_95'],
            expected_drawdown=risk_metrics['expected_drawdown'],
            var_95=risk_metrics['var_95'],
            simulation_time=time.time() - self.start_time
        )

        return result

    def get_performance_summary(self, historical_returns: np.ndarray,
                             risk_pct: float, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Get performance summary without validation decision.

        Args:
            historical_returns: Array of historical returns
            risk_pct: Risk percentage to test
            initial_capital: Starting capital (default: 10000.0)

        Returns:
            Dictionary of performance metrics
        """
        # Perform bootstrap resampling
        simulated_returns = self._bootstrap_resampling(historical_returns)

        # Simulate trading performance
        equity_curves, drawdowns = self._simulate_trading(
            simulated_returns, initial_capital
        )

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(equity_curves, drawdowns)

        return {
            'mean_final_equity': np.mean(risk_metrics['final_equities']),
            'median_final_equity': np.median(risk_metrics['final_equities']),
            'std_final_equity': np.std(risk_metrics['final_equities']),
            'risk_of_ruin': risk_metrics['risk_of_ruin'],
            'expected_drawdown': risk_metrics['expected_drawdown'],
            'var_95': risk_metrics['var_95'],
            'ci_95_lower': risk_metrics['ci_95'][0],
            'ci_95_upper': risk_metrics['ci_95'][1]
        }