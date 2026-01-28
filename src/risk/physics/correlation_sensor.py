"""
Correlation Sensor with Random Matrix Theory (RMT) filtering.

This module implements a correlation sensor that uses Random Matrix Theory
to detect systemic risk in financial time series data by analyzing the
eigenvalue spectrum of correlation matrices.

Key features:
- Marchenko-Pastur distribution for noise thresholding
- Eigenvalue decomposition for signal extraction
- Systemic risk detection based on maximum eigenvalue
- Denoising capabilities
- Caching for performance optimization
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple
from functools import lru_cache
import warnings


class CorrelationSensor:
    """
    Correlation Sensor with RMT filtering for systemic risk detection.

    This class implements Random Matrix Theory (RMT) based correlation analysis
    to identify systemic risk in financial time series data. It uses the
    Marchenko-Pastur distribution to separate signal from noise in the
    eigenvalue spectrum of correlation matrices.

    Attributes:
        cache_size (int): Size of the LRU cache for storing results
        normalize_returns (bool): Whether to normalize returns before analysis
    """

    # Risk thresholds based on maximum eigenvalue
    HIGH_RISK_THRESHOLD = 2.0
    MODERATE_RISK_THRESHOLD = 1.5

    # Minimum data requirements
    MIN_ASSETS = 2
    MIN_PERIODS = 20
    REGULARIZATION_EPSILON = 1e-6

    def __init__(self, cache_size: int = 100, normalize_returns: bool = True):
        """
        Initialize the CorrelationSensor.

        Args:
            cache_size (int): Size of the LRU cache for storing results (default: 100)
            normalize_returns (bool): Whether to normalize returns before analysis (default: True)
        """
        self.cache_size = cache_size
        self.normalize_returns = normalize_returns
        self._cache = lru_cache(maxsize=cache_size)(self.detect_systemic_risk)

    def _validate_input(self, returns_matrix: np.ndarray) -> None:
        """
        Validate the input returns matrix.

        Args:
            returns_matrix (np.ndarray): Returns matrix with shape (N_assets, T_periods)

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(returns_matrix, np.ndarray):
            raise ValueError("Input must be a numpy array")

        if returns_matrix.ndim != 2:
            raise ValueError("Input must be a 2D array with shape (N_assets, T_periods)")

        n_assets, t_periods = returns_matrix.shape

        if n_assets < 2:
            raise ValueError(f"Must have at least 2 assets, got {n_assets}")

        if t_periods < 20:
            warnings.warn(f"Low number of time periods ({t_periods}), results may be unstable")

    def _normalize_returns(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize returns to zero mean and unit variance.

        Args:
            returns_matrix (np.ndarray): Raw returns matrix

        Returns:
            np.ndarray: Normalized returns matrix
        """
        if not self.normalize_returns:
            return returns_matrix

        # Calculate mean and standard deviation along time axis
        means = np.mean(returns_matrix, axis=1, keepdims=True)
        stds = np.std(returns_matrix, axis=1, keepdims=True)

        # Avoid division by zero
        stds[stds == 0] = 1.0

        return (returns_matrix - means) / stds

    def _calculate_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate the correlation matrix from returns data.

        Args:
            returns_matrix (np.ndarray): Normalized returns matrix

        Returns:
            np.ndarray: Correlation matrix
        """
        # Handle NaN values by replacing with column means
        returns_clean = returns_matrix.copy()
        mask = np.isnan(returns_clean)

        if np.any(mask):
            # Replace NaN with column mean
            col_means = np.nanmean(returns_clean, axis=0, keepdims=True)
            returns_clean[mask] = np.take(col_means, np.where(mask)[1])

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_clean)

        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(cov_matrix))
        std_devs[std_devs == 0] = 1.0  # Avoid division by zero

        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

        # Ensure numerical stability
        np.fill_diagonal(corr_matrix, 1.0)
        np.clip(corr_matrix, -1.0, 1.0, out=corr_matrix)

        return corr_matrix

    def _calculate_marchenko_pastur_threshold(self, n_assets: int, t_periods: int) -> float:
        """
        Calculate the Marchenko-Pastur noise threshold.

        Args:
            n_assets (int): Number of assets
            t_periods (int): Number of time periods

        Returns:
            float: Maximum expected noise eigenvalue (λ_max)
        """
        # Quality factor Q = T/N
        Q = t_periods / n_assets

        # λ_max = (1 + sqrt(Q))² for normalized correlation matrix (σ² = 1)
        lambda_max = (1 + np.sqrt(Q)) ** 2

        return lambda_max

    def _perform_eigenvalue_decomposition(self, correlation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigenvalue decomposition on the correlation matrix.

        Args:
            correlation_matrix (np.ndarray): Symmetric correlation matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: (eigenvalues, eigenvectors)
        """
        # Use eigh for symmetric matrices (more efficient and stable)
        eigenvalues, eigenvectors = eigh(correlation_matrix)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def _denoise_correlation_matrix(self, correlation_matrix: np.ndarray,
                                eigenvalues: np.ndarray,
                                eigenvectors: np.ndarray,
                                noise_threshold: float) -> np.ndarray:
        """
        Denoise the correlation matrix by replacing noise eigenvalues.

        Args:
            correlation_matrix (np.ndarray): Original correlation matrix
            eigenvalues (np.ndarray): Eigenvalues in descending order
            eigenvectors (np.ndarray): Corresponding eigenvectors
            noise_threshold (float): Threshold for noise eigenvalues

        Returns:
            np.ndarray: Denoised correlation matrix
        """
        # Identify noise eigenvalues (below threshold)
        noise_indices = eigenvalues < noise_threshold

        if np.any(noise_indices):
            # Replace noise eigenvalues with their average
            avg_noise = np.mean(eigenvalues[noise_indices])
            eigenvalues[noise_indices] = avg_noise

            # Reconstruct the denoised matrix
            denoised_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

            # Restore correlation properties
            np.fill_diagonal(denoised_matrix, 1.0)
            np.clip(denoised_matrix, -1.0, 1.0, out=denoised_matrix)
        else:
            # No noise detected, return original matrix
            denoised_matrix = correlation_matrix.copy()

        return denoised_matrix

    def _classify_risk_level(self, max_eigenvalue: float) -> str:
        """
        Classify the risk level based on the maximum eigenvalue.

        Args:
            max_eigenvalue (float): Maximum eigenvalue of the correlation matrix

        Returns:
            str: Risk level classification ('LOW', 'MODERATE', 'HIGH')
        """
        if max_eigenvalue < 1.5:
            return "LOW"
        elif max_eigenvalue < 2.0:
            return "MODERATE"
        else:
            return "HIGH"

    def detect_systemic_risk(self, returns_matrix: np.ndarray) -> dict:
        """
        Detect systemic risk in returns data using RMT analysis.

        Args:
            returns_matrix (np.ndarray): Returns matrix with shape (N_assets, T_periods)

        Returns:
            dict: Dictionary containing risk analysis results with keys:
                - max_eigenvalue: Maximum eigenvalue of the correlation matrix
                - noise_threshold: Marchenko-Pastur threshold
                - denoised_matrix: Denoised correlation matrix (if applicable)
                - risk_level: Risk level classification
                - eigenvalues: All eigenvalues in descending order
                - eigenvectors: Corresponding eigenvectors
        """
        # Validate input
        self._validate_input(returns_matrix)

        # Get matrix dimensions
        n_assets, t_periods = returns_matrix.shape

        # Normalize returns if requested
        normalized_returns = self._normalize_returns(returns_matrix)

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(normalized_returns)

        # Calculate Marchenko-Pastur threshold
        noise_threshold = self._calculate_marchenko_pastur_threshold(n_assets, t_periods)

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = self._perform_eigenvalue_decomposition(correlation_matrix)

        # Get maximum eigenvalue
        max_eigenvalue = eigenvalues[0]

        # Denoise the correlation matrix
        denoised_matrix = self._denoise_correlation_matrix(
            correlation_matrix, eigenvalues, eigenvectors, noise_threshold
        )

        # Classify risk level
        risk_level = self._classify_risk_level(max_eigenvalue)

        return {
            "max_eigenvalue": float(max_eigenvalue),
            "noise_threshold": float(noise_threshold),
            "denoised_matrix": denoised_matrix,
            "risk_level": risk_level,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }

    def clear_cache(self) -> None:
        """Clear the LRU cache."""
        self._cache.cache_clear()