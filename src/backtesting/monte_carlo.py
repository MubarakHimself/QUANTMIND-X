"""
Monte Carlo Simulation - Distribution Metrics

Task Group 3.6: Create monte_carlo.py for MonteCarloSimulator

Implements Monte Carlo simulation with 1000+ iterations and confidence intervals.
Randomizes trade order while preserving returns to calculate distribution metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging

# Import existing components
from backtesting.mt5_engine import MT5BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result with distribution metrics."""
    num_simulations: int
    base_return_pct: float

    # Confidence intervals
    confidence_interval_5th: float
    confidence_interval_95th: float
    confidence_interval_99th: float

    # Value at Risk metrics
    value_at_risk_95: float  # VaR at 95% confidence
    expected_shortfall_95: float  # Expected shortfall at 95% (CVaR)

    # Distribution statistics
    mean_return: float
    std_return: float
    median_return: float

    # Best/worst case
    best_return: float
    worst_return: float

    # Probability metrics
    probability_profitable: float
    probability_target: float  # Probability of reaching target return
    target_return: float = 0.0

    # Full simulation results
    simulation_returns: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_simulations': self.num_simulations,
            'base_return_pct': self.base_return_pct,
            'confidence_interval_5th': self.confidence_interval_5th,
            'confidence_interval_95th': self.confidence_interval_95th,
            'confidence_interval_99th': self.confidence_interval_99th,
            'value_at_risk_95': self.value_at_risk_95,
            'expected_shortfall_95': self.expected_shortfall_95,
            'mean_return': self.mean_return,
            'std_return': self.std_return,
            'median_return': self.median_return,
            'best_return': self.best_return,
            'worst_return': self.worst_return,
            'probability_profitable': self.probability_profitable,
            'probability_target': self.probability_target,
            'target_return': self.target_return
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for backtest validation.

    Randomizes trade order while preserving returns to calculate:
    - Confidence intervals (5th, 95th, 99th percentile)
    - Value at Risk (VaR) and Expected Shortfall (CVaR)
    - Probability of profitability
    - Distribution statistics

    Example:
        >>> simulator = MonteCarloSimulator(num_simulations=1000)
        >>> result = simulator.simulate(base_result, data)
        >>> print(f"95% CI: [{result.confidence_interval_5th:.2%}, {result.confidence_interval_95th:.2%}]")
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None,
        target_return_pct: float = 10.0
    ):
        """Initialize Monte Carlo simulator.

        Args:
            num_simulations: Number of simulations to run (default 1000)
            random_seed: Random seed for reproducibility
            target_return_pct: Target return for probability calculation
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        self.target_return_pct = target_return_pct

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"MonteCarloSimulator initialized with {num_simulations} simulations")

    def simulate(
        self,
        base_result: MT5BacktestResult,
        data: Optional[pd.DataFrame] = None
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation on backtest result.

        Args:
            base_result: Original backtest result to simulate from
            data: Optional OHLCV data for more detailed simulation

        Returns:
            MonteCarloResult with distribution metrics
        """
        logger.info(f"Running {self.num_simulations} Monte Carlo simulations")

        # Extract trade returns from base result
        trade_returns = self._extract_trade_returns(base_result)

        if not trade_returns:
            logger.warning("No trade returns found, using equity curve")
            trade_returns = self._extract_returns_from_equity(base_result)

        if not trade_returns:
            logger.error("Cannot extract returns for simulation")
            return self._empty_result(base_result)

        # Run simulations
        simulation_returns = []

        for i in range(self.num_simulations):
            # Randomize trade order
            shuffled_returns = np.random.permutation(trade_returns)

            # Calculate cumulative return
            cumulative_return = np.sum(shuffled_returns)
            simulation_returns.append(cumulative_return)

            if (i + 1) % 100 == 0:
                logger.debug(f"Completed {i + 1}/{self.num_simulations} simulations")

        # Calculate metrics
        result = self._calculate_metrics(simulation_returns, base_result.return_pct)

        logger.info(
            f"Monte Carlo simulation completed: "
            f"95% CI [{result.confidence_interval_5th:.2%}, {result.confidence_interval_95th:.2%}]"
        )

        return result

    def _extract_trade_returns(self, result: MT5BacktestResult) -> List[float]:
        """Extract individual trade returns from backtest result.

        Args:
            result: Backtest result

        Returns:
            List of trade returns as percentages
        """
        trade_returns = []

        for trade in result.trade_history:
            profit = trade.get('profit', 0)
            entry_price = trade.get('entry_price', 1)
            volume = trade.get('volume', 0)

            if entry_price > 0 and volume > 0:
                # Calculate return percentage
                # Approximate: profit / (volume * 100000 * entry_price)
                capital = volume * 100000 * entry_price
                return_pct = (profit / capital) * 100 if capital > 0 else 0
                trade_returns.append(return_pct)

        return trade_returns

    def _extract_returns_from_equity(self, result: MT5BacktestResult) -> List[float]:
        """Extract returns from equity curve.

        Args:
            result: Backtest result

        Returns:
            List of periodic returns
        """
        equity_curve = result.equity_curve

        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i - 1]
            curr_equity = equity_curve[i]

            if prev_equity > 0:
                ret = (curr_equity - prev_equity) / prev_equity * 100
                returns.append(ret)

        return returns

    def _calculate_metrics(
        self,
        simulation_returns: List[float],
        base_return_pct: float
    ) -> MonteCarloResult:
        """Calculate distribution metrics from simulation returns.

        Args:
            simulation_returns: List of simulated total returns
            base_return_pct: Original backtest return percentage

        Returns:
            MonteCarloResult with calculated metrics
        """
        returns_array = np.array(simulation_returns)

        # Confidence intervals
        ci_5th = np.percentile(returns_array, 5)
        ci_95th = np.percentile(returns_array, 95)
        ci_99th = np.percentile(returns_array, 99)

        # Value at Risk (95%)
        var_95 = np.percentile(returns_array, 5)

        # Expected Shortfall (95%) - average of worst 5% returns
        worst_5_pct_threshold = np.percentile(returns_array, 5)
        worst_returns = returns_array[returns_array <= worst_5_pct_threshold]
        expected_shortfall_95 = np.mean(worst_returns) if len(worst_returns) > 0 else var_95

        # Distribution statistics
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        median_return = np.median(returns_array)

        # Best/worst case
        best_return = np.max(returns_array)
        worst_return = np.min(returns_array)

        # Probability metrics
        probability_profitable = np.sum(returns_array > 0) / len(returns_array)
        probability_target = np.sum(returns_array >= self.target_return_pct) / len(returns_array)

        return MonteCarloResult(
            num_simulations=self.num_simulations,
            base_return_pct=base_return_pct,
            confidence_interval_5th=float(ci_5th),
            confidence_interval_95th=float(ci_95th),
            confidence_interval_99th=float(ci_99th),
            value_at_risk_95=float(var_95),
            expected_shortfall_95=float(expected_shortfall_95),
            mean_return=float(mean_return),
            std_return=float(std_return),
            median_return=float(median_return),
            best_return=float(best_return),
            worst_return=float(worst_return),
            probability_profitable=float(probability_profitable),
            probability_target=float(probability_target),
            target_return=self.target_return_pct,
            simulation_returns=simulation_returns
        )

    def _empty_result(self, base_result: MT5BacktestResult) -> MonteCarloResult:
        """Create empty result when simulation fails.

        Args:
            base_result: Original backtest result

        Returns:
            MonteCarloResult with zeros
        """
        return MonteCarloResult(
            num_simulations=0,
            base_return_pct=base_result.return_pct,
            confidence_interval_5th=0.0,
            confidence_interval_95th=0.0,
            confidence_interval_99th=0.0,
            value_at_risk_95=0.0,
            expected_shortfall_95=0.0,
            mean_return=0.0,
            std_return=0.0,
            median_return=0.0,
            best_return=0.0,
            worst_return=0.0,
            probability_profitable=0.0,
            probability_target=0.0,
            target_return=self.target_return_pct,
            simulation_returns=[]
        )


__all__ = [
    'MonteCarloSimulator',
    'MonteCarloResult',
]
