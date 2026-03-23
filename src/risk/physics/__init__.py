"""
Physics Sensors for Market Regime Detection

This package provides econophysics-based sensors for detecting market regimes
and systemic risk. Each sensor implements a different theoretical framework:

- IsingRegimeSensor: Phase transition detection using ferromagnetic Ising model
- ChaosSensor: Lyapunov exponent analysis for chaos detection
- CorrelationSensor: Random Matrix Theory for systemic risk detection
"""

from .ising_sensor import IsingRegimeSensor, IsingSensorConfig, IsingSystem
from .chaos_sensor import ChaosSensor
from .correlation_sensor import CorrelationSensor

try:
    from .hmm_sensor import HMMRegimeSensor
except ImportError:
    HMMRegimeSensor = None

__all__ = [
    "IsingRegimeSensor",
    "IsingSensorConfig",
    "IsingSystem",
    "ChaosSensor",
    "CorrelationSensor",
]
if HMMRegimeSensor is not None:
    __all__.append('HMMRegimeSensor')

# Backward compatibility - HMM sensor (legacy module)
try:
    from .hmm_sensor import HMMRegimeSensor
    __all__.append("HMMRegimeSensor")
except ImportError:
    pass

# Version
__version__ = "1.0.0"
