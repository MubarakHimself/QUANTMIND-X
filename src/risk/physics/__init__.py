"""
Physics Sensors for Market Regime Detection

This package provides econophysics-based sensors for detecting market regimes
and systemic risk. Each sensor implements a different theoretical framework:

- IsingRegimeSensor: Phase transition detection using ferromagnetic Ising model
- ChaosSensor: Lyapunov exponent analysis for chaos detection
- CorrelationSensor: Random Matrix Theory for systemic risk detection
"""

from .ising_sensor import IsingRegimeSensor
from .chaos_sensor import ChaosSensor
from .correlation_sensor import CorrelationSensor

__all__ = [
    "IsingRegimeSensor",
    "ChaosSensor",
    "CorrelationSensor",
]

# Version
__version__ = "1.0.0"
