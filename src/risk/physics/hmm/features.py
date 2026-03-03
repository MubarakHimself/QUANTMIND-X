"""
Feature Configuration Module
============================

Defines the FeatureConfig dataclass for controlling feature extraction.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    # Ising features
    include_magnetization: bool = True
    include_susceptibility: bool = True
    include_energy: bool = True
    include_temperature: bool = True

    # Price features
    include_log_returns: bool = True
    include_rolling_volatility_20: bool = True
    include_rolling_volatility_50: bool = True
    include_price_momentum_10: bool = True

    # Technical indicators
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Scaling
    scaling_method: str = "standard"  # 'standard', 'minmax', 'robust'
    clip_outliers: bool = True
    clip_threshold: float = 3.0
