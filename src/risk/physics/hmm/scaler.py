"""
Feature Scaling Module
======================

Provides feature scaling functionality for HMM feature arrays.
"""

import numpy as np
from typing import Dict, Optional, Any


class FeatureScaler:
    """
    Feature scaler with multiple scaling methods.

    Supports standard (z-score), min-max, and robust scaling.
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize the scaler.

        Args:
            method: Scaling method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.params: Optional[Dict[str, np.ndarray]] = None

    def fit(self, features: np.ndarray) -> 'FeatureScaler':
        """
        Fit the scaler to the data.

        Args:
            features: 2D array of features (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        if self.method == "standard":
            self.params = {
                'mean': np.nanmean(features, axis=0),
                'std': np.nanstd(features, axis=0)
            }
        elif self.method == "minmax":
            self.params = {
                'min': np.nanmin(features, axis=0),
                'max': np.nanmax(features, axis=0)
            }
        else:  # robust
            self.params = {
                'median': np.nanmedian(features, axis=0),
                'iqr': np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
            }

        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted parameters.

        Args:
            features: 2D array of features

        Returns:
            Scaled features
        """
        if self.params is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if self.method == "standard":
            scaled = (features - self.params['mean']) / (self.params['std'] + 1e-8)
        elif self.method == "minmax":
            scaled = (features - self.params['min']) / (self.params['max'] - self.params['min'] + 1e-8)
        else:  # robust
            scaled = (features - self.params['median']) / (self.params['iqr'] + 1e-8)

        # Replace NaN with 0
        return np.nan_to_num(scaled, nan=0.0)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            features: 2D array of features

        Returns:
            Scaled features
        """
        return self.fit(features).transform(features)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get scaler parameters."""
        return self.params.copy() if self.params else {}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set scaler parameters."""
        self.params = params.copy()
