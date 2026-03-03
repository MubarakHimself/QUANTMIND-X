"""
Base Sensor Classes

Abstract base classes for regime sensors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseRegimeSensor(ABC):
    """Abstract base class for regime sensors."""

    @abstractmethod
    def predict_regime(self, features: np.ndarray, cache_key: str = None) -> Any:
        """
        Predict regime from feature array.

        Args:
            features: Feature array
            cache_key: Optional key for caching

        Returns:
            Regime reading
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded
        """
        pass
