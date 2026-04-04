"""
Position Sizing Package

This package provides advanced position sizing algorithms including:
- Kelly criterion with enhanced risk parameters
- Monte Carlo validation for sizing recommendations
- Risk-adjusted position optimization
- Session-scoped Kelly modifiers (Story 4.10)
"""
from .kelly_engine import PhysicsAwareKellyEngine
from .monte_carlo_validator import MonteCarloValidator
from .session_kelly_modifiers import SessionKellyModifiers, SessionKellyState, PremiumSessionAssault

__all__ = [
    "PhysicsAwareKellyEngine",
    "MonteCarloValidator",
    "SessionKellyModifiers",
    "SessionKellyState",
    "PremiumSessionAssault",
]

# Version
__version__ = "1.0.0"