"""
Risk Management Configuration

Constants and presets for risk management including:
- Prop firm constraints (FTMO, The5ers, FundingPips)
- Position sizing parameters (lot sizes, risk percentages)
- Cache TTL settings
"""

from typing import Dict, Type, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Position Sizing Constants
# =============================================================================

MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = 100.0

MAX_RISK_PCT = 0.05  # Maximum 5% risk per trade
DEFAULT_RISK_PCT = 0.02  # Default 2% risk per trade

PHYSICS_CACHE_TTL = 60  # Cache physics readings for 60 seconds
ACCOUNT_CACHE_TTL = 300  # Cache account info for 5 minutes


# =============================================================================
# Prop Firm Presets
# =============================================================================

class PropFirmType(str, Enum):
    """Types of prop firms."""
    FTMO = "ftmo"
    THE5ERS = "the5ers"
    FUNDING_PIPS = "funding_pips"
    CUSTOM = "custom"


@dataclass
class PropFirmPreset:
    """
    Prop firm preset configuration.

    Attributes:
        name: Display name of the prop firm
        max_risk_pct: Maximum risk percentage per trade
        max_daily_loss_pct: Maximum daily loss percentage
        max_total_loss_pct: Maximum total drawdown percentage
        min_trading_days: Minimum trading days required
    """
    name: str
    max_risk_pct: float
    max_daily_loss_pct: float
    max_total_loss_pct: float
    min_trading_days: int = 5

    def get_max_risk_pct(self) -> float:
        """Get max risk percentage."""
        return self.max_risk_pct


# =============================================================================
# Prop Firm Preset Configurations
# =============================================================================

FTPreset = PropFirmPreset(
    name="FTMO",
    max_risk_pct=0.02,  # 2% max per trade
    max_daily_loss_pct=0.05,  # 5% daily loss limit
    max_total_loss_pct=0.10,  # 10% total drawdown limit
    min_trading_days=5
)

The5ersPreset = PropFirmPreset(
    name="The5ers",
    max_risk_pct=0.025,  # 2.5% max per trade
    max_daily_loss_pct=0.06,  # 6% daily loss limit
    max_total_loss_pct=0.12,  # 12% total drawdown limit
    min_trading_days=5
)

FundingPipsPreset = PropFirmPreset(
    name="FundingPips",
    max_risk_pct=0.03,  # 3% max per trade
    max_daily_loss_pct=0.08,  # 8% daily loss limit
    max_total_loss_pct=0.15,  # 15% total drawdown limit
    min_trading_days=3
)

# Registry of all presets
PROP_FIRM_PRESETS: Dict[str, PropFirmPreset] = {
    "ftmo": FTPreset,
    "the5ers": The5ersPreset,
    "funding_pips": FundingPipsPreset,
}


def get_preset(preset_name: str) -> Optional[PropFirmPreset]:
    """
    Get a prop firm preset by name.

    Args:
        preset_name: Name of the preset (e.g., "ftmo", "the5ers")

    Returns:
        PropFirmPreset if found, None otherwise
    """
    return PROP_FIRM_PRESETS.get(preset_name.lower())


# =============================================================================
# Risk Tier Configuration
# =============================================================================

class RiskTier(str, Enum):
    """Risk tiers for position sizing."""
    CONSERVATIVE = "conservative"  # 1% risk
    MODERATE = "moderate"  # 2% risk
    AGGRESSIVE = "aggressive"  # 3% risk
    HIGH_RISK = "high_risk"  # 5% risk


RISK_TIERS: Dict[RiskTier, float] = {
    RiskTier.CONSERVATIVE: 0.01,
    RiskTier.MODERATE: 0.02,
    RiskTier.AGGRESSIVE: 0.03,
    RiskTier.HIGH_RISK: 0.05,
}


# =============================================================================
# Kelly Criterion Configuration
# =============================================================================

KELLY_FRACTION_CAP = 0.25  # Never exceed 25% Kelly fraction
KELLY_MIN_WIN_RATE = 0.45  # Minimum win rate to use Kelly
KELLY_MIN_EXPECTANCY = 0.1  # Minimum positive expectancy

# Kelly fraction multipliers for different market conditions
KELLY_MULTIPLIERS = {
    "trend_stable": 1.0,
    "range_stable": 0.8,
    "high_chaos": 0.5,
    "phase_transition": 0.3,
    "unknown": 0.7,
}


# =============================================================================
# Physics Sensor Configuration
# =============================================================================

CHAOS_THRESHOLD = 0.5  # Threshold for high chaos regime
CORRELATION_THRESHOLD = 0.7  # Threshold for high correlation
ISING_THRESHOLD = 0.6  # Threshold for phase transition

# Weights for physics adjustments
CHAOS_WEIGHT = 0.75
CORRELATION_WEIGHT = 0.5
ISING_WEIGHT = 0.25


# =============================================================================
# Portfolio Scaling Configuration
# =============================================================================

MAX_PORTFOLIO_RISK = 0.15  # Maximum 15% total portfolio risk
PORTFOLIO_CORRELATION_THRESHOLD = 0.7  # Threshold for correlated positions
PORTFOLION_SCALE_FACTOR = 0.8  # Scale factor when correlation is high
