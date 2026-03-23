"""
HMM Feature Extraction Package
==============================

Modular package for extracting features from market data and Ising Model outputs
for Hidden Markov Model training.

Modules:
- features: FeatureConfig dataclass
- indicators: TechnicalIndicators calculations
- scaler: FeatureScaler for normalization
- models: HMMFeatureExtractor class
- utils: Utility functions
- trainer: HMMTrainer for scheduled model retraining

Reference: docs/architecture/components.md
"""

from .features import FeatureConfig
from .indicators import TechnicalIndicators
from .scaler import FeatureScaler
from .models import HMMFeatureExtractor
from .utils import prepare_training_data, load_config_from_file
from .trainer import HMMTrainer

__all__ = [
    'FeatureConfig',
    'TechnicalIndicators',
    'FeatureScaler',
    'HMMFeatureExtractor',
    'prepare_training_data',
    'load_config_from_file',
    'HMMTrainer',
]
