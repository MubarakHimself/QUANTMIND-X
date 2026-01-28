"""
Enhanced Kelly Configuration

Dataclass configurations for position sizing with prop firm presets.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnhancedKellyConfig:
    """Configuration for Enhanced Kelly position sizing.
    
    This configuration implements the 3-layer protection system:
    - Layer 1: Kelly Fraction (reduce full Kelly by a safety factor)
    - Layer 2: Hard Risk Cap (never exceed maximum per-trade risk)
    - Layer 3: Dynamic Volatility Adjustment (scale based on ATR)
    """

    # Layer 1: Kelly Fraction (Safety Multiplier)
    kelly_fraction: float = 0.50  # 50% of Kelly for safety (70-80% returns, 30-40% drawdown)

    # Layer 2: Hard Risk Cap
    max_risk_pct: float = 0.02  # Maximum 2% per trade (prop firm safe)

    # Layer 3: Dynamic Volatility Thresholds
    high_vol_threshold: float = 1.3   # ATR ratio above this = reduce size
    low_vol_threshold: float = 0.7    # ATR ratio below this = increase size
    low_vol_boost: float = 1.2        # Multiplier for calm markets (conservative)

    # Data Requirements
    min_trade_history: int = 30       # Minimum trades before using Kelly
    atr_period: int = 20              # ATR averaging period
    confidence_interval: float = 0.95 # For statistical significance (future use)

    # Broker Constraints
    min_lot_size: float = 0.01
    lot_step: float = 0.01
    max_lot_size: float = 100.0

    # Safety Overrides
    allow_zero_position: bool = False  # Return min lot if sizing fails
    fallback_risk_pct: float = 0.01    # Fallback when insufficient history

    # Portfolio Integration
    enable_portfolio_scaling: bool = True  # Scale down when multiple bots active
    max_portfolio_risk_pct: float = 0.03   # 3% total daily risk across all bots

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not 0 < self.kelly_fraction <= 1.0:
            raise ValueError("kelly_fraction must be between 0 and 1")
        if not 0 < self.max_risk_pct <= 0.10:
            raise ValueError("max_risk_pct must be between 0 and 10%")
        if self.high_vol_threshold <= self.low_vol_threshold:
            raise ValueError("high_vol_threshold must be > low_vol_threshold")
        if self.min_trade_history < 10:
            raise ValueError("min_trade_history should be at least 10")
        return True


class PropFirmPresets:
    """Pre-configured settings for popular prop firms."""

    @staticmethod
    def ftmo_challenge() -> EnhancedKellyConfig:
        """FTMO Challenge Phase - Ultra conservative."""
        return EnhancedKellyConfig(
            kelly_fraction=0.40,        # More conservative
            max_risk_pct=0.01,          # 1% max per trade
            high_vol_threshold=1.2,     # More sensitive to volatility
            low_vol_boost=1.1,          # Minimal boost in calm markets
            allow_zero_position=True,   # Skip trades if risk too high
        )

    @staticmethod
    def ftmo_funded() -> EnhancedKellyConfig:
        """FTMO Funded Phase - Slightly more aggressive."""
        return EnhancedKellyConfig(
            kelly_fraction=0.55,        # Closer to half Kelly
            max_risk_pct=0.015,         # 1.5% max per trade
            high_vol_threshold=1.3,
            low_vol_boost=1.2,
        )

    @staticmethod
    def the5ers() -> EnhancedKellyConfig:
        """The5%ers preset - Moderate risk."""
        return EnhancedKellyConfig(
            kelly_fraction=0.50,
            max_risk_pct=0.02,
            high_vol_threshold=1.3,
            max_portfolio_risk_pct=0.04,  # Slightly higher portfolio risk allowed
        )

    @staticmethod
    def personal_aggressive() -> EnhancedKellyConfig:
        """Personal account - More aggressive (experienced traders only)."""
        return EnhancedKellyConfig(
            kelly_fraction=0.60,
            max_risk_pct=0.025,         # 2.5% max
            high_vol_threshold=1.5,     # Less sensitive
            low_vol_boost=1.3,          # Bigger boost in calm markets
        )

    @staticmethod
    def paper_trading() -> EnhancedKellyConfig:
        """Paper trading - Test full Kelly response."""
        return EnhancedKellyConfig(
            kelly_fraction=0.70,        # Higher Kelly fraction
            max_risk_pct=0.03,          # 3% for testing
            high_vol_threshold=1.5,
            min_trade_history=10,       # Lower history requirement for testing
        )
