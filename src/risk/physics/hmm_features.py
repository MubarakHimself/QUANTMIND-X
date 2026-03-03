"""
HMM Feature Extraction Module
==============================

Extracts features from Ising Model outputs and market data for HMM training.
Features are used to train Hidden Markov Models for regime detection.

Features extracted:
- Ising outputs: magnetization, susceptibility, energy, temperature
- Price features: log returns, rolling volatility, momentum
- Technical indicators: RSI, ATR, MACD

Reference: docs/architecture/components.md

Backward Compatibility:
This module is maintained for backward compatibility.
New code should import from src.risk.physics.hmm instead.
"""

# Re-export all public APIs from the new modular structure
from .hmm import (
    FeatureConfig,
    TechnicalIndicators,
    FeatureScaler,
    HMMFeatureExtractor,
    prepare_training_data,
    load_config_from_file,
)

__all__ = [
    'FeatureConfig',
    'TechnicalIndicators',
    'FeatureScaler',
    'HMMFeatureExtractor',
    'prepare_training_data',
    'load_config_from_file',
]
