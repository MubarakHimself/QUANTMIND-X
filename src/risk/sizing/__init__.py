"""
Position Sizing Package

This package provides advanced position sizing algorithms including:
- Kelly criterion with enhanced risk parameters
- Monte Carlo validation for sizing recommendations
- Risk-adjusted position optimization
"""
from .kelly_engine import PhysicsAwareKellyEngine
from .monte_carlo_validator import MonteCarloValidator

__all__ = [
    "PhysicsAwareKellyEngine",
    "MonteCarloValidator",
]

# Version
__version__ = "1.0.0"