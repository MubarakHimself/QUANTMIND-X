"""Utility functions for MS-GARCH model."""

import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def compute_vol_regimes(conditional_vol: np.ndarray, n_regimes: int = 3) -> np.ndarray:
    """
    Cluster conditional volatilities into regimes using quantile-based splitting.

    Args:
        conditional_vol: 1D array of conditional volatilities
        n_regimes: Number of regimes to create (default: 3)

    Returns:
        1D array of regime labels (0, 1, ..., n_regimes-1)
    """
    if len(conditional_vol) == 0:
        return np.array([], dtype=int)

    # Compute quantile boundaries
    quantiles = np.linspace(0, 1, n_regimes + 1)
    boundaries = np.quantile(conditional_vol, quantiles)

    # Assign regimes based on boundaries
    regimes = np.zeros(len(conditional_vol), dtype=int)
    for i in range(n_regimes):
        if i == n_regimes - 1:
            # Last regime includes the upper boundary
            mask = (conditional_vol >= boundaries[i]) & (conditional_vol <= boundaries[i + 1])
        else:
            mask = (conditional_vol >= boundaries[i]) & (conditional_vol < boundaries[i + 1])
        regimes[mask] = i

    return regimes


def map_vol_to_regime(vol_state: str, return_direction: float) -> str:
    """
    Map volatility state + return direction to RegimeType string.

    Args:
        vol_state: One of "LOW_VOL", "MED_VOL", "HIGH_VOL"
        return_direction: Positive for uptrend, negative for downtrend

    Returns:
        RegimeType string matching src.events.regime.RegimeType
    """
    is_uptrend = return_direction > 0.0

    if vol_state == "LOW_VOL":
        # Low volatility: stable regimes
        return "RANGE_STABLE"
    elif vol_state == "MED_VOL":
        # Medium volatility: trending regimes (direction-dependent)
        return "TREND_BULL" if is_uptrend else "TREND_BEAR"
    elif vol_state == "HIGH_VOL":
        # High volatility: volatile or chaotic regimes
        return "RANGE_VOLATILE"
    else:
        # Unknown state defaults to stable
        return "RANGE_STABLE"


def estimate_transition_matrix(regime_sequence: np.ndarray, n_regimes: int) -> np.ndarray:
    """
    Count state transitions to build empirical transition matrix.

    Args:
        regime_sequence: 1D array of regime labels (integers 0..n_regimes-1)
        n_regimes: Number of regimes

    Returns:
        (n_regimes, n_regimes) transition probability matrix
    """
    if len(regime_sequence) < 2:
        # Default uniform transition matrix
        return np.ones((n_regimes, n_regimes)) / n_regimes

    # Initialize transition count matrix
    trans_counts = np.zeros((n_regimes, n_regimes))

    # Count transitions
    for i in range(len(regime_sequence) - 1):
        from_regime = int(regime_sequence[i])
        to_regime = int(regime_sequence[i + 1])
        if 0 <= from_regime < n_regimes and 0 <= to_regime < n_regimes:
            trans_counts[from_regime, to_regime] += 1

    # Normalize to probability matrix
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    trans_matrix = trans_counts / row_sums

    return trans_matrix


__all__ = [
    "compute_vol_regimes",
    "map_vol_to_regime",
    "estimate_transition_matrix",
]
