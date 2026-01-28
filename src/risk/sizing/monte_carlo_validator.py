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

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
Source: Refactored from quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py
"""

import logging
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

from src.risk.models.strategy_performance import StrategyPerformance


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Container for Monte Carlo validation results.

    Attributes:
        passed: Whether risk level is acceptable (ruin < 0.5%)
        risk_of_ruin: Probability of account ruin (0 to 1)
        adjusted_risk: Recommended risk multiplier (0.5 if failed, 1.0 if passed)
        ci_95: 95% confidence interval as (lower, upper) tuple
        expected_drawdown: Expected maximum drawdown across simulations
        var_95: Value at Risk at 95% confidence level
        simulation_time: Time taken for simulation in seconds
        recommended_risk_pct: Recommended risk percentage (adjusted)
        final_equities: Array of final equity values from all runs
        max_drawdowns: Array of maximum drawdowns from all runs
    """
    passed: bool
    risk_of_ruin: float
    adjusted_risk: float
    ci_95: tuple[float, float]
    expected_drawdown: float
    var_95: float
    simulation_time: float
    recommended_risk_pct: Optional[float] = None
    final_equities: Optional[np.ndarray] = None
    max_drawdowns: Optional[np.ndarray] = None

class MonteCarloValidator:
    """
    Monte Carlo validator for risk assessment using bootstrap resampling.

    This class simulates trading performance using historical returns to estimate
    risk metrics including risk of ruin, drawdowns, and confidence intervals.

    The validator uses vectorized numpy operations for performance, targeting
    < 500ms for 2000 simulation runs.

    Example:
        >>> validator = MonteCarloValidator(runs=2000)
        >>> perf = StrategyPerformance(win_rate=0.55, avg_win=100, avg_loss=50, total_trades=100)
        >>> result = validator.validate_risk(perf, risk_pct=0.02)
        >>> if result.passed:
        ...     print(f"Risk validated: {result.risk_of_ruin:.3%} ruin probability")
        ... else:
        ...     print(f"Risk too high: use {result.adjusted_risk:.0%} of proposed size")
    """

    def __init__(self, runs: int = 2000, initial_capital: float = 10000.0):
        """
        Initialize the Monte Carlo validator.

        Args:
            runs: Number of simulation runs (default: 2000 for < 500ms target)
            initial_capital: Starting capital for simulations (default: $10,000)
        """
        self.runs = runs
        self.initial_capital = initial_capital
        self._start_time: Optional[float] = None

    def _generate_synthetic_returns(self, perf: StrategyPerformance, risk_pct: float, n_trades: int = 100) -> np.ndarray:
        """
        Generate synthetic trade returns based on strategy performance metrics.

        Creates a distribution of returns matching the win rate, average win,
        and average loss from the strategy performance.

        Args:
            perf: StrategyPerformance metrics
            risk_pct: Risk percentage per trade (as decimal, e.g., 0.02 for 2%)
            n_trades: Number of trades to simulate (default: 100)

        Returns:
            Array of trade returns as decimals (e.g., 0.02 for 2% gain, -0.01 for 1% loss)
        """
        np.random.seed(None)  # Ensure randomness

        # Calculate trade outcomes in percentage terms
        # avg_win and avg_loss are in currency, convert to percentage returns
        # Using risk_pct as the risk amount per trade

        # Win return: avg_win as percentage of account
        win_return_pct = (perf.avg_win / self.initial_capital) if self.initial_capital > 0 else 0

        # Loss return: risk_pct (or avg_loss as percentage, whichever is relevant)
        # For this simulation, we use the actual loss relative to account
        loss_return_pct = -(perf.avg_loss / self.initial_capital) if self.initial_capital > 0 else -risk_pct

        # Generate trade outcomes
        outcomes = np.random.random(n_trades)
        returns = np.where(
            outcomes < perf.win_rate,
            win_return_pct,  # Win
            loss_return_pct  # Loss
        )

        return returns

    def _bootstrap_resampling(self, returns: np.ndarray) -> np.ndarray:
        """
        Perform vectorized bootstrap resampling of historical returns.

        Uses numpy's random integer generation for efficient sampling with
        replacement across all simulation runs.

        Args:
            returns: Array of historical/synthetic returns

        Returns:
            Array of simulated returns with shape (n_periods, n_runs)
        """
        n_periods = len(returns)
        # Vectorized sampling with replacement - create (periods, runs) array of indices
        random_indices = np.random.randint(0, n_periods, size=(n_periods, self.runs))
        return returns[random_indices]

    def _simulate_equity_curves(self, simulated_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate equity curves and drawdowns from bootstrap resampled returns.

        Args:
            simulated_returns: Array of shape (n_periods, n_runs) with returns

        Returns:
            Tuple of (equity_curves, drawdowns) arrays
        """
        # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... * (1 + rn)
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=0)

        # Convert to equity curves
        equity_curves = self.initial_capital * cumulative_returns

        # Add starting point (Day 0 at initial capital)
        equity_curves = np.vstack([np.full((1, self.runs), self.initial_capital), equity_curves])

        # Calculate drawdowns: (peak - equity) / peak
        peak_equity = np.maximum.accumulate(equity_curves, axis=0)
        drawdowns = (peak_equity - equity_curves) / peak_equity

        return equity_curves, drawdowns

    def _calculate_risk_metrics(
        self,
        equity_curves: np.ndarray,
        drawdowns: np.ndarray,
        risk_pct: float
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics from simulation results.

        Args:
            equity_curves: Array of equity curves with shape (n_periods + 1, n_runs)
            drawdowns: Array of drawdowns with shape (n_periods + 1, n_runs)
            risk_pct: Original risk percentage used for simulation

        Returns:
            Dictionary containing all risk metrics
        """
        final_equities = equity_curves[-1, :]
        max_drawdowns = np.max(drawdowns, axis=0)

        # Risk of ruin: equity drops to 50% or less of initial capital
        # Using 50% drawdown as ruin threshold (standard trading definition)
        ruin_threshold = 0.5 * self.initial_capital
        ruin_count = np.sum(final_equities <= ruin_threshold)
        risk_of_ruin = ruin_count / self.runs if self.runs > 0 else 0.0

        # 95% Confidence Interval
        ci_95_lower = np.percentile(final_equities, 2.5)
        ci_95_upper = np.percentile(final_equities, 97.5)

        # Expected maximum drawdown
        expected_drawdown = np.mean(max_drawdowns)

        # Value at Risk at 95% confidence (5th percentile of final equities)
        var_95 = np.percentile(final_equities, 5)

        return {
            'risk_of_ruin': risk_of_ruin,
            'ci_95': (ci_95_lower, ci_95_upper),
            'expected_drawdown': expected_drawdown,
            'var_95': var_95,
            'final_equities': final_equities,
            'max_drawdowns': max_drawdowns
        }

    def validate_risk(
        self,
        perf: StrategyPerformance,
        risk_pct: float,
        runs: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate risk level using Monte Carlo simulation with bootstrap resampling.

        This method:
        1. Generates synthetic returns based on strategy performance
        2. Performs vectorized bootstrap resampling
        3. Simulates equity curves across all runs
        4. Calculates risk metrics (ruin, VaR, drawdowns, CI)
        5. Applies decision logic: pass if ruin < 0.5%, else recommend halving risk

        Args:
            perf: StrategyPerformance object with win_rate, avg_win, avg_loss
            risk_pct: Proposed risk percentage per trade (e.g., 0.02 for 2%)
            runs: Number of simulation runs (uses instance default if None)

        Returns:
            ValidationResult with comprehensive risk assessment

        Raises:
            ValueError: If perf is invalid or risk_pct is out of range
        """
        self._start_time = time.time()

        # Use provided runs or instance default
        n_runs = runs if runs is not None else self.runs
        original_runs = self.runs
        self.runs = n_runs

        try:
            # Input validation
            if not isinstance(perf, StrategyPerformance):
                raise ValueError(f"perf must be StrategyPerformance, got {type(perf)}")
            if not 0 < risk_pct <= 1:
                raise ValueError(f"risk_pct must be between 0 and 1, got {risk_pct}")
            if not perf.is_profitable():
                logger.warning(
                    "Strategy has negative expectancy",
                    extra={
                        "win_rate": perf.win_rate,
                        "avg_win": perf.avg_win,
                        "avg_loss": perf.avg_loss,
                        "expectancy": perf.expectancy()
                    }
                )

            # Generate synthetic returns based on strategy performance
            # Number of trades to simulate = total_trades from performance data
            n_trades = max(perf.total_trades, 100)
            synthetic_returns = self._generate_synthetic_returns(perf, risk_pct, n_trades)

            # Perform vectorized bootstrap resampling
            simulated_returns = self._bootstrap_resampling(synthetic_returns)

            # Simulate equity curves
            equity_curves, drawdowns = self._simulate_equity_curves(simulated_returns)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(equity_curves, drawdowns, risk_pct)

            # Decision logic: pass if ruin < 0.5%
            ruin_threshold = 0.005  # 0.5%
            passed = risk_metrics['risk_of_ruin'] < ruin_threshold

            # Calculate adjusted risk: halve if failed
            adjusted_risk = 0.5 if not passed else 1.0
            recommended_risk_pct = risk_pct * adjusted_risk

            # Log results
            logger.info(
                "Monte Carlo validation complete",
                extra={
                    "risk_pct": risk_pct,
                    "risk_of_ruin": risk_metrics['risk_of_ruin'],
                    "passed": passed,
                    "recommended_risk_pct": recommended_risk_pct,
                    "simulation_time": time.time() - self._start_time
                }
            )

            return ValidationResult(
                passed=passed,
                risk_of_ruin=risk_metrics['risk_of_ruin'],
                adjusted_risk=adjusted_risk,
                ci_95=risk_metrics['ci_95'],
                expected_drawdown=risk_metrics['expected_drawdown'],
                var_95=risk_metrics['var_95'],
                simulation_time=time.time() - self._start_time,
                recommended_risk_pct=recommended_risk_pct,
                final_equities=risk_metrics['final_equities'],
                max_drawdowns=risk_metrics['max_drawdowns']
            )
        finally:
            # Restore original runs count
            self.runs = original_runs

    def get_performance_summary(
        self,
        perf: StrategyPerformance,
        risk_pct: float,
        runs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed performance summary without validation decision.

        Args:
            perf: StrategyPerformance object with metrics
            risk_pct: Risk percentage to test
            runs: Number of simulation runs (uses instance default if None)

        Returns:
            Dictionary with detailed performance metrics including:
            - mean_final_equity: Average final equity across all runs
            - median_final_equity: Median final equity
            - std_final_equity: Standard deviation of final equities
            - profit_probability: Percentage of profitable runs
            - risk_of_ruin: Probability of 50% drawdown
            - expected_drawdown: Average maximum drawdown
            - var_95: Value at Risk at 95% confidence
            - ci_95: 95% confidence interval
        """
        n_runs = runs if runs is not None else self.runs
        original_runs = self.runs
        self.runs = n_runs

        try:
            # Generate synthetic returns
            n_trades = max(perf.total_trades, 100)
            synthetic_returns = self._generate_synthetic_returns(perf, risk_pct, n_trades)

            # Bootstrap resampling
            simulated_returns = self._bootstrap_resampling(synthetic_returns)

            # Simulate equity curves
            equity_curves, drawdowns = self._simulate_equity_curves(simulated_returns)

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(equity_curves, drawdowns, risk_pct)

            final_equities = risk_metrics['final_equities']
            profit_probability = np.sum(final_equities > self.initial_capital) / n_runs

            return {
                'mean_final_equity': float(np.mean(final_equities)),
                'median_final_equity': float(np.median(final_equities)),
                'std_final_equity': float(np.std(final_equities)),
                'profit_probability': float(profit_probability),
                'risk_of_ruin': risk_metrics['risk_of_ruin'],
                'expected_drawdown': risk_metrics['expected_drawdown'],
                'var_95': risk_metrics['var_95'],
                'ci_95_lower': risk_metrics['ci_95'][0],
                'ci_95_upper': risk_metrics['ci_95'][1],
                'runs': n_runs
            }
        finally:
            self.runs = original_runs