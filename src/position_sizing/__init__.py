"""
QuantMindX Position Sizing Module

Enhanced Kelly Criterion implementation with safety layers for prop firm trading.
"""

from .kelly_config import EnhancedKellyConfig, PropFirmPresets
from .enhanced_kelly import EnhancedKellyCalculator, enhanced_kelly_position_size
from .kelly_analyzer import (
    KellyStatisticsAnalyzer,
    KellyParameters,
    calculate_per_trade_ev,
    calculate_daily_ev,
    verify_baseline_mathematics
)
from .portfolio_kelly import PortfolioKellyScaler

__all__ = [
    "EnhancedKellyConfig",
    "PropFirmPresets",
    "EnhancedKellyCalculator",
    "enhanced_kelly_position_size",
    "KellyStatisticsAnalyzer",
    "KellyParameters",
    "PortfolioKellyScaler",
    "calculate_per_trade_ev",
    "calculate_daily_ev",
    "verify_baseline_mathematics",
]
