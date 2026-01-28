"""
Risk management package for Enhanced Kelly Position Sizing.

This package provides econophysics-aware position sizing that integrates
Ising phase transitions, Lyapunov chaos detection, and RMT correlation
filtering with the Kelly Criterion for superior risk-adjusted returns.
"""

from src.risk.config import (
    ACCOUNT_CACHE_TTL,
    DEFAULT_K_FRACTION,
    ISING_CRITICAL_SUSCEPTIBILITY,
    LYAPUNOV_CHAOS_THRESHOLD,
    LOT_STEP,
    MAX_LOT,
    MAX_RISK_PCT,
    MC_MAX_DRAWDOWN,
    MC_RUIN_THRESHOLD,
    MC_SIMULATION_RUNS,
    MIN_LOT,
    PHYSICS_CACHE_TTL,
    PROP_FIRM_PRESETS,
    RMT_CRITICAL_EIGEN_THRESHOLD,
    RMT_MAX_EIGEN_THRESHOLD,
    FTPreset,
    FundingPipsPreset,
    PropFirmPreset,
    The5ersPreset,
    get_preset,
    validate_config,
)

__version__: str = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Configuration
    "LYAPUNOV_CHAOS_THRESHOLD",
    "ISING_CRITICAL_SUSCEPTIBILITY",
    "RMT_MAX_EIGEN_THRESHOLD",
    "RMT_CRITICAL_EIGEN_THRESHOLD",
    "PHYSICS_CACHE_TTL",
    "ACCOUNT_CACHE_TTL",
    "MC_SIMULATION_RUNS",
    "MC_MAX_DRAWDOWN",
    "MC_RUIN_THRESHOLD",
    "DEFAULT_K_FRACTION",
    "MAX_RISK_PCT",
    "MIN_LOT",
    "LOT_STEP",
    "MAX_LOT",
    # Prop Firm Presets
    "PropFirmPreset",
    "FTPreset",
    "The5ersPreset",
    "FundingPipsPreset",
    "PROP_FIRM_PRESETS",
    "get_preset",
    "validate_config",
]
