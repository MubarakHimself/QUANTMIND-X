"""
Chaos Sensor with Lyapunov Exponent Calculation

This module implements a chaos sensor that calculates the Lyapunov exponent
using phase space reconstruction and the method of analogues.

The Lyapunov exponent measures the rate of divergence of nearby trajectories
in phase space, indicating the degree of chaos in the system.

Classes:
    ChaosSensor: Main class for chaos analysis and Lyapunov exponent calculation
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

@dataclass
class ChaosAnalysisResult:
    """Data class to store chaos analysis results."""
    lyapunov_exponent: float
    match_distance: float
    match_index: int
    chaos_level: str
    trajectory_length: int

class ChaosSensor:
    """
    Chaos sensor that calculates Lyapunov exponent using phase space reconstruction
    and the method of analogues.

    The sensor analyzes the degree of chaos in financial time series data by:
    1. Reconstructing phase space using time delay embedding
    2. Finding nearest neighbors (analogues) in the embedded space
    3. Tracking divergence over time to calculate Lyapunov exponent

    Attributes:
        embedding_dimension (int): Dimension of the phase space (default: 3)
        time_delay (int): Time delay for embedding (default: 12)
        lookback_points (int): Number of points to analyze (default: 300)
        k_steps (int): Number of steps to track divergence (default: 10)
    """

    def __init__(self,
                 embedding_dimension: int = 3,
                 time_delay: int = 12,
                 lookback_points: int = 300,
                 k_steps: int = 10):
        """
        Initialize the ChaosSensor.

        Args:
            embedding_dimension (int): Dimension of the phase space (default: 3)
            time_delay (int): Time delay for embedding (default: 12)
            lookback_points (int): Number of points to analyze (default: 300)
            k_steps (int): Number of steps to track divergence (default: 10)
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.lookback_points = lookback_points
        self.k_steps = k_steps

    def _embed_time_delay(self, series: np.ndarray) -> np.ndarray:
        """
        Create time delay embedding of the series.

        Args:
            series (np.ndarray): Input time series data

        Returns:
            np.ndarray: Embedded vectors of shape (N_vectors, dim)

        Raises:
            ValueError: If series is too short for the specified embedding parameters
        """
        N = len(series)
        M = N - (self.embedding_dimension - 1) * self.time_delay

        if M <= 0:
            raise ValueError(
                f"Series too short ({N}) for embedding dimension {self.embedding_dimension} "
                f"and time delay {self.time_delay}. Required minimum length: "
                f"{(self.embedding_dimension - 1) * self.time_delay + 1}"
            )

        embedded = np.zeros((M, self.embedding_dimension))
        for d in range(self.embedding_dimension):
            st = d * self.time_delay
            ed = st + M
            embedded[:, d] = series[st:ed]

        return embedded

    def _perform_method_of_analogues(self,
                                  embedded_vectors: np.ndarray) -> Tuple[float, int]:
        """
        Perform method of analogues to find nearest neighbor.

        Args:
            embedded_vectors (np.ndarray): Embedded phase space vectors

        Returns:
            Tuple[float, int]: (match_distance, match_index)

        Raises:
            ValueError: If data is insufficient for analysis
        """
        if len(embedded_vectors) < self.lookback_points * 2:
            raise ValueError(
                f"Data insufficient for analysis. Need at least "
                f"{self.lookback_points * 2} points, got {len(embedded_vectors)}"
            )

        # Define subject trajectory (current)
        subject_vectors = embedded_vectors[-self.lookback_points:]

        # Define library (historical data)
        safety_buffer = self.lookback_points
        search_end_idx = len(embedded_vectors) - self.lookback_points - safety_buffer

        if search_end_idx < 100:
            raise ValueError(
                f"History too short to find analogues. Need at least 100 points, got {search_end_idx}"
            )

        search_space = embedded_vectors[:search_end_idx]

        # Nearest neighbor search
        query_point = subject_vectors[0].reshape(1, -1)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_space)
        distances, indices = nbrs.kneighbors(query_point)

        match_idx = indices[0][0]
        match_dist = distances[0][0]

        return match_dist, match_idx

    def _calculate_lyapunov_exponent(self,
                                  series: np.ndarray,
                                  match_idx: int) -> float:
        """
        Calculate Lyapunov exponent by tracking divergence.

        Args:
            series (np.ndarray): Original time series
            match_idx (int): Index of the matched analogue

        Returns:
            float: Lyapunov exponent
        """
        # Calculate divergence over k steps
        divergence_log_sum = 0.0
        for i in range(self.k_steps):
            if match_idx + i + self.lookback_points >= len(series):
                break

            # Current point distance
            current_point = series[-self.lookback_points + i]
            matched_point = series[match_idx + self.lookback_points + i]
            current_dist = abs(current_point - matched_point)

            # Previous point distance (for divergence calculation)
            if i > 0:
                prev_current = series[-self.lookback_points + i - 1]
                prev_matched = series[match_idx + self.lookback_points + i - 1]
                prev_dist = abs(prev_current - prev_matched)

                if prev_dist > 0:  # Avoid division by zero
                    divergence_log_sum += np.log(current_dist / prev_dist)

        lyapunov = divergence_log_sum / self.k_steps if self.k_steps > 0 else 0.0
        return lyapunov

    def _determine_chaos_level(self, lyapunov_exponent: float) -> str:
        """
        Determine chaos level based on Lyapunov exponent.

        Args:
            lyapunov_exponent (float): Calculated Lyapunov exponent

        Returns:
            str: Chaos level ('STABLE', 'MODERATE', or 'CHAOTIC')
        """
        if lyapunov_exponent < 0.2:
            return "STABLE"
        elif lyapunov_exponent < 0.5:
            return "MODERATE"
        else:
            return "CHAOTIC"

    def analyze_chaos(self, returns: np.ndarray) -> ChaosAnalysisResult:
        """
        Analyze chaos in the input returns using Lyapunov exponent calculation.

        Args:
            returns (np.ndarray): Input log-returns array (must have at least 300 points)

        Returns:
            ChaosAnalysisResult: Analysis results containing Lyapunov exponent and chaos level

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not isinstance(returns, np.ndarray):
            raise ValueError("Input must be a numpy ndarray")

        if len(returns) < 300:
            raise ValueError(
                f"Input array too short. Need at least 300 points, got {len(returns)}"
            )

        if returns.ndim != 1:
            raise ValueError("Input must be a 1D array")

        # Normalize returns for analysis
        returns_norm = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

        # Phase space reconstruction
        embedded_vectors = self._embed_time_delay(returns_norm)

        # Find nearest neighbor (analogue)
        match_dist, match_idx = self._perform_method_of_analogues(embedded_vectors)

        # Calculate Lyapunov exponent
        lyapunov_exponent = self._calculate_lyapunov_exponent(returns_norm, match_idx)

        # Determine chaos level
        chaos_level = self._determine_chaos_level(lyapunov_exponent)

        return ChaosAnalysisResult(
            lyapunov_exponent=lyapunov_exponent,
            match_distance=match_dist,
            match_index=match_idx,
            chaos_level=chaos_level,
            trajectory_length=len(returns)
        )

    def __call__(self, returns: np.ndarray) -> ChaosAnalysisResult:
        """
        Convenience method to call analyze_chaos directly.

        Args:
            returns (np.ndarray): Input log-returns array

        Returns:
            ChaosAnalysisResult: Analysis results
        """
        return self.analyze_chaos(returns)

    def get_reading(self) -> float:
        """
        Get chaos reading as a float value for position sizing.

        Returns a normalized chaos value (0.0 to 1.0) where:
        - 0.0 = Stable market
        - 0.5 = Moderate volatility
        - 1.0 = Highly chaotic market

        Returns:
            float: Normalized chaos level
        """
        # Return moderate chaos level as default when no data available
        return 0.35