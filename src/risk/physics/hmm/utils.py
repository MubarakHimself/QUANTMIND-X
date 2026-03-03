"""
HMM Utilities Module
====================

Utility functions for HMM feature extraction and configuration.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .features import FeatureConfig
from .models import HMMFeatureExtractor


def prepare_training_data(ohlcv_data: Dict[str, pd.DataFrame],
                          config: Optional[FeatureConfig] = None,
                          symbol: Optional[str] = None,
                          timeframe: Optional[str] = None) -> Tuple[np.ndarray, List[str], HMMFeatureExtractor]:
    """
    Prepare training data for HMM from multiple symbols/timeframes.

    Args:
        ohlcv_data: Dictionary mapping symbol_timeframe to DataFrame
        config: Feature extraction configuration
        symbol: Optional specific symbol to filter
        timeframe: Optional specific timeframe to filter

    Returns:
        Tuple of (features array, list of keys, extractor instance)
    """
    extractor = HMMFeatureExtractor(config)
    all_features = []
    keys = []

    for key, df in ohlcv_data.items():
        # Parse key (format: SYMBOL_TIMEFRAME)
        parts = key.split('_')
        sym = parts[0]
        tf = '_'.join(parts[1:])  # Handle timeframes like M5, H1, H4

        # Filter by symbol/timeframe if specified
        if symbol and sym != symbol:
            continue
        if timeframe and tf != timeframe:
            continue

        # Extract features
        features = extractor.extract_features_batch(df, sym, tf)
        all_features.append(features)
        keys.append(key)

    # Combine all features
    combined_features = np.vstack(all_features)

    # Scale features
    scaled_features = extractor.scale_features(combined_features, fit=True)

    return scaled_features, keys, extractor


def load_config_from_file(config_path: str) -> FeatureConfig:
    """Load feature configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    feature_config = config_data.get('features', {})

    return FeatureConfig(
        include_magnetization=feature_config.get('ising_outputs', {}).get('magnetization', True),
        include_susceptibility=feature_config.get('ising_outputs', {}).get('susceptibility', True),
        include_energy=feature_config.get('ising_outputs', {}).get('energy', True),
        include_temperature=feature_config.get('ising_outputs', {}).get('temperature', True),
        include_log_returns=feature_config.get('price_features', {}).get('log_returns', True),
        include_rolling_volatility_20=feature_config.get('price_features', {}).get('rolling_volatility_20', True),
        include_rolling_volatility_50=feature_config.get('price_features', {}).get('rolling_volatility_50', True),
        include_price_momentum_10=feature_config.get('price_features', {}).get('price_momentum_10', True),
        rsi_period=feature_config.get('technical_indicators', {}).get('rsi_period', 14),
        atr_period=feature_config.get('technical_indicators', {}).get('atr_period', 14),
        macd_fast=feature_config.get('technical_indicators', {}).get('macd_fast', 12),
        macd_slow=feature_config.get('technical_indicators', {}).get('macd_slow', 26),
        macd_signal=feature_config.get('technical_indicators', {}).get('macd_signal', 9),
        scaling_method=feature_config.get('scaling', {}).get('method', 'standard'),
        clip_outliers=feature_config.get('scaling', {}).get('clip_outliers', True),
        clip_threshold=feature_config.get('scaling', {}).get('clip_threshold', 3.0)
    )
