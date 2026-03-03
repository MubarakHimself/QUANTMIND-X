"""
HMM Sensor Configuration

Configuration dataclass for the HMM Regime Sensor.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class HMMSensorConfig:
    """Configuration for the HMM Regime Sensor."""
    model_path: str = os.environ.get('HMM_MODEL_DIR', '/data/hmm/models')
    model_version: Optional[str] = None  # None = latest
    cache_size: int = 100
    confidence_threshold: float = 0.6
    regime_mapping: Dict[int, str] = None

    def __post_init__(self):
        if self.regime_mapping is None:
            self.regime_mapping = {
                0: "TRENDING_LOW_VOL",
                1: "TRENDING_HIGH_VOL",
                2: "RANGING_LOW_VOL",
                3: "RANGING_HIGH_VOL"
            }
