"""
Correlation Sensor Module

Provides eigenvalue-based correlation analysis using Random Matrix Theory (RMT).
Computes correlation matrices for M5 and H1 timeframes and classifies regimes.

The module analyzes the largest eigenvalue against the RMT threshold to determine
whether the market is in a correlated, uncorrelated, or neutral regime.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Default symbols for correlation analysis
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "USDCAD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY"
]


@dataclass
class CorrelationSensorData:
    """Data structure for correlation sensor output."""

    max_eigenvalue: float
    rmt_threshold: float
    is_correlated: bool
    regime: str  # "CORRELATED" | "UNCORRELATED" | "NEUTRAL"
    m5_matrix: NDArray[np.float64]
    h1_matrix: NDArray[np.float64]
    eigenvalues: List[float]
    symbols: List[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_eigenvalue": float(self.max_eigenvalue),
            "rmt_threshold": float(self.rmt_threshold),
            "is_correlated": bool(self.is_correlated),
            "regime": self.regime,
            "m5_matrix": self.m5_matrix.tolist(),
            "h1_matrix": self.h1_matrix.tolist(),
            "eigenvalues": [float(e) for e in self.eigenvalues],
            "symbols": self.symbols,
            "timestamp": self.timestamp
        }


class CorrelationSensor:
    """
    Correlation sensor using Random Matrix Theory (RMT) analysis.

    Analyzes the eigenvalue distribution of correlation matrices to detect
    market regimes:
    - CORRELATED: max eigenvalue exceeds RMT threshold (market is synchronized)
    - UNCORRELATED: eigenvalues within random matrix bounds (market is noise-like)
    - NEUTRAL: borderline case
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        rmt_factor: float = 2.0,
        correlation_threshold: float = 0.7
    ):
        """
        Initialize the correlation sensor.

        Args:
            symbols: List of forex symbols to analyze
            rmt_factor: Multiplier for RMT eigenvalue threshold (default 2.0)
            correlation_threshold: Threshold for correlation significance (default 0.7)
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.rmt_factor = rmt_factor
        self.correlation_threshold = correlation_threshold
        self._last_data: Optional[CorrelationSensorData] = None

    def _compute_rmt_threshold(self, n: int, t: int) -> float:
        """
        Compute the RMT threshold using Marchenko-Pastur distribution.

        Args:
            n: Number of assets (matrix dimension)
            t: Number of observations (time steps)

        Returns:
            Eigenvalue threshold from RMT
        """
        # Marchenko-Pastur bounds for correlation matrix eigenvalues
        # lambda_max = (1 + sqrt(n/t))^2
        # lambda_min = (1 - sqrt(n/t))^2
        ratio = n / t if t > 0 else 1.0
        sqrt_ratio = np.sqrt(ratio)

        lambda_max = (1 + sqrt_ratio) ** 2
        lambda_min = max(0, (1 - sqrt_ratio) ** 2)

        # Expected maximum eigenvalue for random matrix
        # Scale by rmt_factor to create a detection threshold
        threshold = lambda_max * self.rmt_factor

        logger.debug(
            f"RMT threshold: n={n}, t={t}, lambda_max={lambda_max:.4f}, "
            f"threshold={threshold:.4f}"
        )

        return threshold

    def _compute_correlation_matrix(
        self,
        returns: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], List[float]]:
        """
        Compute correlation matrix and eigenvalues.

        Args:
            returns: Array of asset returns (shape: t x n)

        Returns:
            Tuple of (correlation matrix, eigenvalues)
        """
        # Compute covariance matrix
        cov_matrix = np.cov(returns, rowvar=False)

        # Standardize to correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_devs[std_devs == 0] = 1.0  # Avoid division by zero
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(corr_matrix, 1.0)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        return corr_matrix, eigenvalues.tolist()

    def _generate_synthetic_returns(
        self,
        n_symbols: int,
        t_m5: int = 100,
        t_h1: int = 100
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Generate synthetic returns for testing.

        In production, this would fetch real market data.

        Args:
            n_symbols: Number of symbols
            t_m5: Number of M5 observations
            t_h1: Number of H1 observations

        Returns:
            Tuple of (M5 returns, H1 returns)
        """
        np.random.seed(int(time.time()) % 2**32)

        # M5: Higher noise, lower correlation
        m5_returns = np.random.randn(t_m5, n_symbols) * 0.001

        # H1: More structure, potentially higher correlation
        h1_returns = np.random.randn(t_h1, n_symbols) * 0.002

        # Add some correlation structure to H1
        if t_h1 > n_symbols:
            # Create a common factor
            common_factor = np.random.randn(t_h1, 1) * 0.001
            h1_returns = h1_returns * 0.7 + common_factor * 0.3

        return m5_returns, h1_returns

    def compute(
        self,
        m5_returns: Optional[NDArray[np.float64]] = None,
        h1_returns: Optional[NDArray[np.float64]] = None
    ) -> CorrelationSensorData:
        """
        Compute correlation sensor data.

        Args:
            m5_returns: Optional M5 returns array (shape: t x n)
            h1_returns: Optional H1 returns array (shape: t x n)

        Returns:
            CorrelationSensorData with analysis results
        """
        n_symbols = len(self.symbols)

        # Use provided returns or generate synthetic
        if m5_returns is None or h1_returns is None:
            t_m5 = 100 if m5_returns is None else m5_returns.shape[0]
            t_h1 = 100 if h1_returns is None else h1_returns.shape[0]
            m5_returns, h1_returns = self._generate_synthetic_returns(
                n_symbols, t_m5, t_h1
            )

        # Compute M5 correlation matrix
        m5_matrix, m5_eigenvalues = self._compute_correlation_matrix(m5_returns)

        # Compute H1 correlation matrix
        h1_matrix, h1_eigenvalues = self._compute_correlation_matrix(h1_returns)

        # Get the maximum eigenvalue from H1 (more significant for regime detection)
        max_eigenvalue = float(max(m5_eigenvalues[0], h1_eigenvalues[0]))

        # Compute RMT threshold
        t = max(m5_returns.shape[0], h1_returns.shape[0])
        rmt_threshold = self._compute_rmt_threshold(n_symbols, t)

        # Classify regime based on max eigenvalue vs threshold
        if max_eigenvalue > rmt_threshold * 1.2:
            regime = "CORRELATED"
            is_correlated = True
        elif max_eigenvalue < rmt_threshold * 0.8:
            regime = "UNCORRELATED"
            is_correlated = False
        else:
            regime = "NEUTRAL"
            is_correlated = False

        # Combine eigenvalues from both timeframes
        all_eigenvalues = sorted(
            m5_eigenvalues[:5] + h1_eigenvalues[:5],
            reverse=True
        )[:10]

        data = CorrelationSensorData(
            max_eigenvalue=max_eigenvalue,
            rmt_threshold=rmt_threshold,
            is_correlated=is_correlated,
            regime=regime,
            m5_matrix=m5_matrix,
            h1_matrix=h1_matrix,
            eigenvalues=all_eigenvalues,
            symbols=self.symbols,
            timestamp=time.time()
        )

        self._last_data = data
        logger.info(
            f"Correlation sensor: regime={regime}, max_eigenvalue={max_eigenvalue:.4f}, "
            f"rmt_threshold={rmt_threshold:.4f}"
        )

        return data

    def get_last_data(self) -> Optional[CorrelationSensorData]:
        """Get the last computed data."""
        return self._last_data


# Global sensor instance
_sensor_instance: Optional[CorrelationSensor] = None


def get_correlation_sensor() -> CorrelationSensor:
    """Get or create the global correlation sensor instance."""
    global _sensor_instance
    if _sensor_instance is None:
        _sensor_instance = CorrelationSensor()
    return _sensor_instance


def compute_correlation_sensor(
    m5_returns: Optional[NDArray[np.float64]] = None,
    h1_returns: Optional[NDArray[np.float64]] = None
) -> Dict:
    """
    Compute correlation sensor and return as dictionary.

    This is the main entry point for the API endpoints.

    Args:
        m5_returns: Optional M5 returns array
        h1_returns: Optional H1 returns array

    Returns:
        Dictionary representation of CorrelationSensorData
    """
    sensor = get_correlation_sensor()
    data = sensor.compute(m5_returns, h1_returns)
    return data.to_dict()


def get_correlation_matrix(timeframe: str) -> Dict:
    """
    Get correlation matrix for a specific timeframe.

    Args:
        timeframe: "M5" or "H1"

    Returns:
        Dictionary with matrix and metadata
    """
    sensor = get_correlation_sensor()
    data = sensor.get_last_data()

    if data is None:
        # Compute if no data available
        data = sensor.compute()

    if timeframe.upper() == "M5":
        matrix = data.m5_matrix.tolist()
    elif timeframe.upper() == "H1":
        matrix = data.h1_matrix.tolist()
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use 'M5' or 'H1'.")

    return {
        "timeframe": timeframe.upper(),
        "matrix": matrix,
        "symbols": data.symbols,
        "timestamp": data.timestamp
    }
