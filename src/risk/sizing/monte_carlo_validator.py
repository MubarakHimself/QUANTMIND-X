"""
Monte Carlo Validator for Risk Assessment

This module provides a Monte Carlo simulation engine for validating trading strategies
and calculating risk metrics including risk of ruin, Value at Risk (VaR), and
confidence intervals.

The implementation uses bootstrap resampling from historical returns to project
future portfolio performance paths.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import time
from dataclasses import dataclass


@dataclass
class MonteCarloResults:
    """Container for Monte Carlo simulation results."""
    paths: np.ndarray  # Shape: (days, simulations)
    final_values: np.ndarray  # Shape: (simulations,)
    mean_path: np.ndarray  # Shape: (days,)
    upper_path: np.ndarray  # Shape: (days,)
    lower_path: np.ndarray  # Shape: (days,)
    median_profit: float
    expected_return: float
    risk_of_ruin: float
    var_95: float
    var_99: float
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float


class MonteCarloValidator:
    """
    Monte Carlo validator for risk assessment and strategy validation.

    This class implements bootstrap resampling to simulate future portfolio performance
    and calculates key risk metrics including risk of ruin, VaR, and confidence intervals.

    Attributes:
        returns (np.ndarray): Historical returns data for bootstrap resampling
        simulations (int): Number of Monte Carlo simulations to run
        days (int): Number of days to project into the future
        start_capital (float): Initial capital for the simulation
        random_seed (Optional[int]): Seed for reproducibility
    """

    def __init__(
        self,
        returns: np.ndarray,
        simulations: int = 2000,
        days: int = 252,
        start_capital: float = 10000.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the Monte Carlo validator.

        Args:
            returns: Historical returns data (1D numpy array)
            simulations: Number of Monte Carlo simulations
            days: Number of days to project
            start_capital: Initial capital for the simulation
            random_seed: Optional seed for reproducibility
        """
        self.returns = returns
        self.simulations = simulations
        self.days = days
        self.start_capital = start_capital
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)

    def run_simulation(self) -> MonteCarloResults:
        """
        Run Monte Carlo simulation using bootstrap resampling.

        Returns:
            MonteCarloResults: Container with simulation results and metrics
        """
        start_time = time.time()

        # Vectorized Bootstrap Resampling
        random_idx = np.random.randint(0, len(self.returns), size=(self.days, self.simulations))
        sim_returns = self.returns[random_idx]

        # Calculate Cumulative Returns
        # Eq_t = Eq_0 * product(1 + r_t)
        sim_paths = self.start_capital * (1 + sim_returns).cumprod(axis=0)

        # Add start point (Day 0)
        sim_paths = np.vstack([np.full((1, self.simulations), self.start_capital), sim_paths])

        duration = time.time() - start_time
        print(f"[MonteCarloValidator] Completed {self.simulations} simulations in {duration:.2f}s")

        # Extract final values for analysis
        final_values = sim_paths[-1, :]

        # Calculate statistics
        mean_path = np.mean(sim_paths, axis=1)
        upper_path = np.percentile(sim_paths, 95, axis=1)  # 95th percentile (Best Case)
        lower_path = np.percentile(sim_paths, 5, axis=1)   # 5th percentile (VaR)

        median_profit = np.median(final_values) - self.start_capital
        expected_return = (np.mean(final_values) / self.start_capital) - 1

        # Calculate risk metrics
        risk_of_ruin = np.mean(final_values <= 0)  # Probability of losing all capital
        var_95 = np.percentile(final_values, 5)    # 95% VaR (5th percentile)
        var_99 = np.percentile(final_values, 1)     # 99% VaR (1st percentile)

        # Calculate confidence intervals
        ci_95_lower = np.percentile(final_values, 2.5)
        ci_95_upper = np.percentile(final_values, 97.5)
        ci_99_lower = np.percentile(final_values, 0.5)
        ci_99_upper = np.percentile(final_values, 99.5)

        return MonteCarloResults(
            paths=sim_paths,
            final_values=final_values,
            mean_path=mean_path,
            upper_path=upper_path,
            lower_path=lower_path,
            median_profit=median_profit,
            expected_return=expected_return,
            risk_of_ruin=risk_of_ruin,
            var_95=var_95,
            var_99=var_99,
            ci_95_lower=ci_95_lower,
            ci_95_upper=ci_95_upper,
            ci_99_lower=ci_99_lower,
            ci_99_upper=ci_99_upper
        )

    def validate_risk(
        self,
        max_risk_of_ruin: float = 0.05,
        min_expected_return: float = 0.0,
        max_var_95: float = -0.2
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate risk based on predefined criteria.

        Args:
            max_risk_of_ruin: Maximum acceptable risk of ruin (default: 5%)
            min_expected_return: Minimum acceptable expected return (default: 0%)
            max_var_95: Maximum acceptable 95% VaR (default: -20%)

        Returns:
            Tuple[bool, Dict[str, float]]: (is_valid, risk_metrics)
        """
        results = self.run_simulation()

        # Calculate risk metrics
        risk_metrics = {
            'risk_of_ruin': results.risk_of_ruin,
            'expected_return': results.expected_return,
            'var_95': results.var_95 / self.start_capital,  # Normalize to percentage
            'median_profit': results.median_profit,
            'ci_95_lower': results.ci_95_lower / self.start_capital,
            'ci_95_upper': results.ci_95_upper / self.start_capital,
            'ci_99_lower': results.ci_99_lower / self.start_capital,
            'ci_99_upper': results.ci_99_upper / self.start_capital
        }

        # Validation logic
        is_valid = (
            results.risk_of_ruin <= max_risk_of_ruin and
            results.expected_return >= min_expected_return and
            results.var_95 / self.start_capital >= max_var_95
        )

        return is_valid, risk_metrics

    def get_risk_summary(self) -> Dict[str, float]:
        """
        Get a summary of key risk metrics.

        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        results = self.run_simulation()

        return {
            'risk_of_ruin': results.risk_of_ruin,
            'expected_return': results.expected_return,
            'var_95': results.var_95 / self.start_capital,
            'var_99': results.var_99 / self.start_capital,
            'ci_95_lower': results.ci_95_lower / self.start_capital,
            'ci_95_upper': results.ci_95_upper / self.start_capital,
            'ci_99_lower': results.ci_99_lower / self.start_capital,
            'ci_99_upper': results.ci_99_upper / self.start_capital,
            'median_profit': results.median_profit
        }

    def halve_risk(self, current_position_size: float) -> float:
        """
        Calculate position size that would halve the current risk exposure.

        Args:
            current_position_size: Current position size

        Returns:
            float: New position size that halves the risk
        """
        results = self.run_simulation()
        current_risk = results.risk_of_ruin

        # Simple heuristic: halve the position size to roughly halve the risk
        # This assumes risk is roughly proportional to position size
        new_position_size = current_position_size * 0.5

        return new_position_size

    def run_vectorized_bootstrap(self, n_bootstrap: int = 1000) -> np.ndarray:
        """
        Run vectorized bootstrap resampling for performance testing.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            np.ndarray: Bootstrap sample statistics
        """
        bootstrap_samples = np.random.choice(
            self.returns,
            size=(n_bootstrap, len(self.returns)),
            replace=True
        )

        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        bootstrap_stds = np.std(bootstrap_samples, axis=1)

        return np.column_stack([bootstrap_means, bootstrap_stds])