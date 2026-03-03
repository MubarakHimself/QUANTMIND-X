"""
HMM Sensor Data Models

Data classes for HMM regime sensor readings.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class HMMRegimeReading:
    """HMM regime reading result."""
    state: int
    regime: str
    confidence: float
    state_probabilities: Dict[int, float]
    next_state_probabilities: Dict[int, float]
    timestamp: datetime
    model_version: str
    features_used: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'state': self.state,
            'regime': self.regime,
            'confidence': self.confidence,
            'state_probabilities': self.state_probabilities,
            'next_state_probabilities': self.next_state_probabilities,
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'features_used': self.features_used
        }
