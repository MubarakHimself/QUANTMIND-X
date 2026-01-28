"""
Configuration module for the risk management system.

This module centralizes all configuration constants, thresholds, and
prop firm presets for the Enhanced Kelly Position Sizing system.

Constants are organized by category:
- Physics thresholds for sensor alerts
- Cache TTL settings for performance
- Monte Carlo simulation settings
- Kelly criterion settings
- Broker lot constraints
"""

from __future__ import annotations

from dataclasses import dataclass


# =============================================================================
# Physics Sensor Thresholds
# =============================================================================

# Lyapunov exponent threshold for chaos detection
# Values > 0.5 indicate chaotic market dynamics
LYAPUNOV_CHAOS_THRESHOLD: float = 0.5

# Ising magnetic susceptibility threshold for phase transitions
# Values > 0.8 indicate critical regime transition (order <-> chaos)
ISING_CRITICAL_SUSCEPTIBILITY: float = 0.8

# Random Matrix Theory eigenvalue thresholds for correlation risk
# Values > 1.5 indicate moderate systemic correlation
# Values > 2.0 indicate high systemic risk (market bubble/crash)
RMT_MAX_EIGEN_THRESHOLD: float = 1.5
RMT_CRITICAL_EIGEN_THRESHOLD: float = 2.0


# =============================================================================
# Cache Settings
# =============================================================================

# Time-to-live for physics sensor cache results (in seconds)
# Reduces redundant calculations for market state that changes slowly
PHYSICS_CACHE_TTL: int = 300  # 5 minutes

# Time-to-live for account info cache (in seconds)
# Account balance changes more frequently, use shorter cache
ACCOUNT_CACHE_TTL: int = 10  # 10 seconds


# =============================================================================
# Monte Carlo Simulation Settings
# =============================================================================

# Number of bootstrap resampling runs for risk validation
# Higher values = more statistical confidence, slower computation
MC_SIMULATION_RUNS: int = 2000

# Maximum drawdown limit for Monte Carlo validation
# Tracked as fraction of initial account balance
MC_MAX_DRAWDOWN: float = 0.10  # 10% (prop firm standard)

# Risk of ruin threshold for rejecting a position size
# If probability of hitting max drawdown > this threshold, risk is halved
MC_RUIN_THRESHOLD: float = 0.005  # 0.5%


# =============================================================================
# Kelly Criterion Settings
# =============================================================================

# Default Kelly fraction safety scalar
# 0.5 = half-Kelly (recommended for trading)
# 1.0 = full Kelly (theoretically optimal but aggressive)
DEFAULT_K_FRACTION: float = 0.5

# Maximum risk percentage per trade
# Cap on position size regardless of Kelly calculation
# Protects against overleveraging during favorable conditions
MAX_RISK_PCT: float = 0.02  # 2% of account balance


# =============================================================================
# Broker Lot Constraints
# =============================================================================

# Minimum lot size allowed by broker
MIN_LOT: float = 0.01

# Lot step size (increment) for position sizing
# Positions must be multiples of this value
LOT_STEP: float = 0.01

# Maximum lot size allowed by broker
MAX_LOT: float = 100.0


# =============================================================================
# Prop Firm Presets
# =============================================================================

@dataclass(frozen=True)
class PropFirmPreset:
    """
    Base class for prop firm trading challenge configurations.

    Prop firms have specific drawdown and profit targets that must
    be respected when calculating position sizes.

    Attributes
    ----------
    name : str
        Name of the prop firm (e.g., "FTMO", "The5ers")
    max_drawdown : float
        Maximum total drawdown allowed (as fraction, e.g., 0.10 = 10%)
    max_daily_loss : float
        Maximum daily loss limit (as fraction, e.g., 0.05 = 5%)
    profit_target : float
        Profit target to pass the challenge (as fraction)
    """

    name: str
    max_drawdown: float
    max_daily_loss: float
    profit_target: float

    def get_max_risk_pct(self) -> float:
        """
        Calculate maximum risk per trade based on drawdown limits.

        Uses a conservative approach: max risk is a fraction of the
        maximum drawdown to ensure multiple consecutive losses can
        be sustained without violating prop firm rules.

        Returns
        -------
        float
            Maximum risk percentage as fraction (e.g., 0.02 = 2%)
        """
        # Conservative: 20% of max drawdown per trade
        # This allows 5 consecutive losses at max risk before hitting limit
        return self.max_drawdown * 0.2

    def validate(self) -> None:
        """
        Validate preset configuration values.

        Raises
        ------
        ValueError
            If any preset values are outside reasonable ranges
        """
        if not 0 < self.max_drawdown <= 0.20:
            raise ValueError(
                f"max_drawdown must be in (0, 0.20], got {self.max_drawdown}"
            )
        if not 0 < self.max_daily_loss <= 0.10:
            raise ValueError(
                f"max_daily_loss must be in (0, 0.10], got {self.max_daily_loss}"
            )
        if not 0 < self.profit_target <= 0.25:
            raise ValueError(
                f"profit_target must be in (0, 0.25], got {self.profit_target}"
            )
        if self.max_daily_loss >= self.max_drawdown:
            raise ValueError(
                f"max_daily_loss ({self.max_daily_loss}) must be "
                f"less than max_drawdown ({self.max_drawdown})"
            )


@dataclass(frozen=True)
class FTPreset(PropFirmPreset):
    """
    FTMO prop firm challenge configuration.

    FTMO Rules (as of 2024):
    - Max drawdown: 10% of initial balance
    - Max daily loss: 5% of initial balance
    - Profit target: 10% of initial balance
    """

    def __init__(self):
        super().__init__(
            name="FTMO",
            max_drawdown=0.10,  # 10%
            max_daily_loss=0.05,  # 5%
            profit_target=0.10,  # 10%
        )


@dataclass(frozen=True)
class The5ersPreset(PropFirmPreset):
    """
    The5ers prop firm challenge configuration.

    The5ers Rules (as of 2024):
    - Max drawdown: 8% of initial balance (High-Stakes program)
    - Max daily loss: 4% of initial balance
    - Profit target: 10% of initial balance
    """

    def __init__(self):
        super().__init__(
            name="The5ers",
            max_drawdown=0.08,  # 8%
            max_daily_loss=0.04,  # 4%
            profit_target=0.10,  # 10%
        )


@dataclass(frozen=True)
class FundingPipsPreset(PropFirmPreset):
    """
    FundingPips prop firm challenge configuration.

    FundingPips Rules (as of 2024):
    - Max drawdown: 12% of initial balance
    - Max daily loss: 6% of initial balance
    - Profit target: 8% of initial balance
    """

    def __init__(self):
        super().__init__(
            name="FundingPips",
            max_drawdown=0.12,  # 12%
            max_daily_loss=0.06,  # 6%
            profit_target=0.08,  # 8%
        )


# Available prop firm presets registry
PROP_FIRM_PRESETS: dict[str, type[PropFirmPreset]] = {
    "ftmo": FTPreset,
    "the5ers": The5ersPreset,
    "fundingpips": FundingPipsPreset,
}


def get_preset(name: str) -> PropFirmPreset:
    """
    Get a prop firm preset by name (case-insensitive).

    Parameters
    ----------
    name : str
        Name of the preset (e.g., "ftmo", "FTMO", "FtMo")

    Returns
    -------
    PropFirmPreset
        Instantiated preset configuration

    Raises
    ------
    ValueError
        If preset name is not recognized
    """
    key = name.lower()
    if key not in PROP_FIRM_PRESETS:
        available = ", ".join(PROP_FIRM_PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available: {available}"
        )
    return PROP_FIRM_PRESETS[key]()


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config() -> None:
    """
    Validate all configuration constants are within acceptable ranges.

    This function is called on module import to ensure configuration
    integrity. Raises ValueError if any configuration is invalid.

    Raises
    ------
    ValueError
        If any configuration constant is outside valid range
    """
    errors: list[str] = []

    # Validate physics thresholds
    if not 0.0 <= LYAPUNOV_CHAOS_THRESHOLD <= 1.0:
        errors.append(
            f"LYAPUNOV_CHAOS_THRESHOLD must be in [0, 1], got {LYAPUNOV_CHAOS_THRESHOLD}"
        )
    if not 0.0 <= ISING_CRITICAL_SUSCEPTIBILITY <= 1.0:
        errors.append(
            f"ISING_CRITICAL_SUSCEPTIBILITY must be in [0, 1], "
            f"got {ISING_CRITICAL_SUSCEPTIBILITY}"
        )
    if RMT_MAX_EIGEN_THRESHOLD >= RMT_CRITICAL_EIGEN_THRESHOLD:
        errors.append(
            f"RMT_MAX_EIGEN_THRESHOLD ({RMT_MAX_EIGEN_THRESHOLD}) must be "
            f"less than RMT_CRITICAL_EIGEN_THRESHOLD ({RMT_CRITICAL_EIGEN_THRESHOLD})"
        )

    # Validate cache TTL values
    if PHYSICS_CACHE_TTL <= 0:
        errors.append(
            f"PHYSICS_CACHE_TTL must be positive, got {PHYSICS_CACHE_TTL}"
        )
    if ACCOUNT_CACHE_TTL <= 0:
        errors.append(
            f"ACCOUNT_CACHE_TTL must be positive, got {ACCOUNT_CACHE_TTL}"
        )

    # Validate Monte Carlo settings
    if MC_SIMULATION_RUNS < 100:
        errors.append(
            f"MC_SIMULATION_RUNS must be >= 100 for statistical significance, "
            f"got {MC_SIMULATION_RUNS}"
        )
    if not 0.0 < MC_MAX_DRAWDOWN <= 1.0:
        errors.append(
            f"MC_MAX_DRAWDOWN must be in (0, 1], got {MC_MAX_DRAWDOWN}"
        )
    if not 0.0 < MC_RUIN_THRESHOLD <= 0.05:
        errors.append(
            f"MC_RUIN_THRESHOLD must be in (0, 0.05], got {MC_RUIN_THRESHOLD}"
        )

    # Validate Kelly settings
    if not 0.0 < DEFAULT_K_FRACTION <= 1.0:
        errors.append(
            f"DEFAULT_K_FRACTION must be in (0, 1], got {DEFAULT_K_FRACTION}"
        )
    if not 0.0 < MAX_RISK_PCT <= 0.05:
        errors.append(
            f"MAX_RISK_PCT must be in (0, 0.05], got {MAX_RISK_PCT}"
        )

    # Validate broker constraints
    if MIN_LOT <= 0:
        errors.append(f"MIN_LOT must be positive, got {MIN_LOT}")
    if LOT_STEP <= 0:
        errors.append(f"LOT_STEP must be positive, got {LOT_STEP}")
    if MAX_LOT <= MIN_LOT:
        errors.append(
            f"MAX_LOT ({MAX_LOT}) must be greater than MIN_LOT ({MIN_LOT})"
        )

    # Validate prop firm presets
    for preset_class in PROP_FIRM_PRESETS.values():
        try:
            preset = preset_class()
            preset.validate()
        except ValueError as e:
            errors.append(f"Preset {preset_class.__name__}: {e}")

    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        raise ValueError(error_message)


# Validate configuration on module import
validate_config()
