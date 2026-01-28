"""
Risk Management Package

This package provides position sizing and risk management tools with
econophysics-based market regime detection.
"""

# Placeholder for future RiskGovernor entry point
__version__ = "1.0.0"

# Import physics sensors
from .physics import (
    IsingRegimeSensor,
    ChaosSensor,
    CorrelationSensor,
)

__all__ = [
    "IsingRegimeSensor",
    "ChaosSensor",
    "CorrelationSensor",
]
