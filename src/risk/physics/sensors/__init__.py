"""
HMM Sensor Module

Modular structure for HMM-based regime detection sensors.

Submodules:
- config: HMMSensorConfig dataclass
- models: HMMRegimeReading dataclass
- base: BaseRegimeSensor abstract class
- hmm: HMMRegimeSensor implementation
"""

from .config import HMMSensorConfig
from .models import HMMRegimeReading
from .base import BaseRegimeSensor
from .hmm import HMMRegimeSensor, create_hmm_sensor

__all__ = [
    'HMMSensorConfig',
    'HMMRegimeReading',
    'BaseRegimeSensor',
    'HMMRegimeSensor',
    'create_hmm_sensor',
]
