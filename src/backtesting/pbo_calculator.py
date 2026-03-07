"""
Probability of Backtest Overfitting (PBO) Calculator.

Uses Combinatorially Symmetric Cross-Validation (CSCV) method to detect
whether a strategy is likely overfitted to historical data.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PBOResult:
    """Result of PBO calculation."""
    pbo: float
    recommendation: str
    reason: str
    first_half_return: float
    second_half_return: float
    return_drift: float
    confidence_interval: tuple


class PBOCalculator:
    """
    Calculates Probability of Backtest Overfitting using
    Combinatorially Symmetric Cross-Validation (CSCV).

    The CSCV method works by:
    1. Splitting the backtest data into N blocks
    2. For each combination of train/validation splits
    3. Checking if the best parameters on training data
       also perform well on validation data
    4. PBO = probability that best-in-sample beats median-out-of-sample
    """

    def __init__(self, n_blocks: int = 5, n_simulations: int = 100):
        """
        Initialize PBO calculator.

        Args:
            n_blocks: Number of blocks to split data into for CSCV
            n_simulations: Number of bootstrap simulations
        """
        self.n_blocks = n_blocks
        self.n_simulations = n_simulations
        self._rng = np.random.default_rng()

    def calculate_pbo(
        self,
        train_returns: np.ndarray,
        test_returns: np.ndarray
    ) -> float:
        """
        Calculate PBO using CSCV bootstrap simulation.

        Args:
            train_returns: In-sample (training) returns
            test_returns: Out-of-sample (testing) returns

        Returns:
            Probability of backtest overfitting (0-1)
            Higher values indicate higher probability of overfitting
        """
        n_train = len(train_returns)
        n_test = len(test_returns)

        if n_train < self.n_blocks or n_test < self.n_blocks:
            return 0.5  # Insufficient data

        # Split into blocks
        block_size_train = n_train // self.n_blocks
        block_size_test = n_test // self.n_blocks

        train_blocks = [
            train_returns[i * block_size_train:(i + 1) * block_size_train]
            for i in range(self.n_blocks)
        ]

        test_blocks = [
            test_returns[i * block_size_test:(i + 1) * block_size_test]
            for i in range(self.n_blocks)
        ]

        # Count how often best-in-sample beats median-out-of-sample
        overfit_count = 0
        n_select = int(0.6 * self.n_blocks)  # 60% train, 40% validation

        for _ in range(self.n_simulations):
            # Random subset selection for in-sample
            selected = self._rng.choice(self.n_blocks, n_select, replace=False)

            # Get performance on selected blocks (in-sample)
            in_sample = np.concatenate([train_blocks[i] for i in selected])
            is_mean = np.mean(in_sample)
            is_std = np.std(in_sample)

            # Get performance on remaining blocks (out-of-sample)
            remaining_train = [train_blocks[i] for i in range(self.n_blocks) if i not in selected]
            oos_train = np.concatenate(remaining_train) if remaining_train else np.array([0])
            oos_mean = np.mean(oos_train)
            oos_std = np.std(oos_train)

            if is_std == 0 or oos_std == 0:
                continue

            # Normalize returns for fair comparison
            is_normalized = (is_mean - np.mean(train_returns)) / (np.std(train_returns) + 1e-8)
            oos_normalized = (oos_mean - np.mean(test_returns)) / (np.std(test_returns) + 1e-8)

            # Overfitting: high in-sample but low out-of-sample
            # If IS is top 20% but OOS is bottom 50%, count as overfit
            is_percentile = (np.sum(train_returns >= is_mean) / len(train_returns))
            oos_percentile = (np.sum(test_returns >= oos_mean) / len(test_returns))

            if is_percentile > 0.8 and oos_percentile < 0.5:
                overfit_count += 1
            elif is_normalized > 1.0 and oos_normalized < 0.0:
                overfit_count += 1

        return overfit_count / self.n_simulations

    def calculate_confidence_interval(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> tuple:
        """
        Calculate bootstrap confidence interval.

        Args:
            returns: Returns series
            confidence: Confidence level (default 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(returns)
        bootstrap_means = []

        for _ in range(self.n_simulations):
            sample = self._rng.choice(returns, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (lower, upper)

    def evaluate_strategy_robustness(
        self,
        returns_series: List[float],
        drawdown_series: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if a strategy is robust or likely overfitted.

        Args:
            returns_series: List of period returns
            drawdown_series: Optional list of drawdowns

        Returns:
            Dict with pbo, recommendation, and details
        """
        returns = np.array(returns_series)

        if len(returns) < 10:
            return {
                "pbo": 0.5,
                "recommendation": "INSUFFICIENT_DATA",
                "reason": "Need at least 10 data points for PBO calculation",
                "first_half_return": 0.0,
                "second_half_return": 0.0,
                "return_drift": 0.0,
                "confidence_interval": (0.0, 0.0)
            }

        # Split into two halves for basic robustness check
        mid = len(returns) // 2
        first_half = returns[:mid]
        second_half = returns[mid:]

        pbo = self.calculate_pbo(first_half, second_half)

        # Calculate confidence interval
        ci = self.calculate_confidence_interval(returns)

        # Calculate drift
        first_half_return = np.mean(first_half)
        second_half_return = np.mean(second_half)
        return_drift = first_half_return - second_half_return

        # Recommendations based on PBO
        if pbo < 0.20:
            recommendation = "ACCEPT"
            reason = "Low probability of overfitting - strategy appears robust"
        elif pbo < 0.50:
            recommendation = "CAUTION"
            reason = "Moderate risk of overfitting - recommend additional validation"
        else:
            recommendation = "REJECT"
            reason = "High probability of overfitting - likely curve fitting"

        # Also check for return drift
        if abs(return_drift) > 0.1:
            if recommendation == "ACCEPT":
                recommendation = "CAUTION"
                reason = "Low PBO but high return drift between periods"
            elif recommendation == "CAUTION":
                reason += " and significant return drift detected"

        return {
            "pbo": round(pbo, 3),
            "recommendation": recommendation,
            "reason": reason,
            "first_half_return": round(first_half_return, 4),
            "second_half_return": round(second_half_return, 4),
            "return_drift": round(return_drift, 4),
            "confidence_interval": (round(ci[0], 4), round(ci[1], 4))
        }

    def evaluate_parameter_sets(
        self,
        parameter_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Evaluate robustness across multiple parameter sets.

        Args:
            parameter_results: Dict mapping parameter names to their
                              in-sample and out-of-sample performance

        Returns:
            Dict with PBO and best parameter set recommendation
        """
        if not parameter_results:
            return {
                "pbo": 0.5,
                "recommendation": "NO_DATA",
                "reason": "No parameter results provided"
            }

        # Extract IS and OOS returns
        is_returns = []
        oos_returns = []
        param_names = []

        for name, result in parameter_results.items():
            if "is_return" in result and "oos_return" in result:
                is_returns.append(result["is_return"])
                oos_returns.append(result["oos_return"])
                param_names.append(name)

        if not is_returns:
            return {
                "pbo": 0.5,
                "recommendation": "INSUFFICIENT_METRICS",
                "reason": "Missing return metrics in parameter results"
            }

        is_arr = np.array(is_returns)
        oos_arr = np.array(oos_returns)

        pbo = self.calculate_pbo(is_arr, oos_arr)

        # Find best OOS performer
        best_idx = np.argmax(oos_arr)
        best_param = param_names[best_idx] if best_idx < len(param_names) else "unknown"

        return {
            "pbo": round(pbo, 3),
            "recommendation": "ACCEPT" if pbo < 0.3 else "REJECT",
            "reason": f"Best parameter: {best_param}",
            "best_parameter": best_param,
            "best_oos_return": round(oos_arr[best_idx], 4) if best_idx < len(oos_arr) else 0.0,
            "n_parameters_tested": len(param_names)
        }


__all__ = ['PBOCalculator', 'PBOResult']
