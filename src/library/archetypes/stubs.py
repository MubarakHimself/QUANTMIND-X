"""
QuantMindLib V1 — Structural Archetype Stubs (Packet 6E)

Four base archetypes with structural capability profiles.
These define WHAT each archetype IS — frozen spec surfaces for future deep
implementations. Phase 2+ will add concrete BaseArchetype subclasses.

Also provides get_full_registry() — a module-level singleton that
bootstraps all V1 archetypes (ORB, derived, and stubs) into a single registry.
"""

from __future__ import annotations

from typing import List, Optional

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.orb import ORB_ARCHETYPE
from src.library.archetypes.registry import ArchetypeRegistry

# ---------------------------------------------------------------------------
# Structural stubs — Phase 2+ deep implementations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BreakoutScalper — faster ORB variant with tighter stops
# ---------------------------------------------------------------------------

BREAKOUT_SCALPER_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="breakout_scalper",
    base_type="scalping",
    description=(
        "High-frequency breakout scalper. Faster than ORB, tighter stops, "
        "shorter range definition. Phase 2+ deep implementation."
    ),
    v1_priority=False,
    quality_class="native_supported",

    required_features=[
        "indicators/atr_14",
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "volume/rvol",
        "volume/mfi",
        "microstructure/spread",
        "microstructure/tob_pressure",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
    ],
    excluded_features=[],
    session_tags=[
        "london",
        "new_york",
        "london_newyork_overlap",
    ],
    constraints=[
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=1.0),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
    ],
    execution_profile="scalp_fast_v1",
    max_trades_per_hour=10,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "parameters.range_minutes",
        "parameters.breakout_pips",
        "features.optional",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
    ],
    base_archetype_id=None,
)


# ---------------------------------------------------------------------------
# PullbackScalper — trend-following with mean-reversion entries
# ---------------------------------------------------------------------------

PULLBACK_SCALPER_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="pullback_scalper",
    base_type="scalping",
    description=(
        "Scalp in the direction of the trend on pullbacks. "
        "Trend confirmation + mean-reversion entry. Phase 2+."
    ),
    v1_priority=False,
    quality_class="native_supported",

    required_features=[
        "indicators/atr_14",
        "indicators/macd",
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/rsi_14",
        "volume/rvol",
        "volume/mfi",
        "microstructure/spread",
        "microstructure/tob_pressure",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
    ],
    excluded_features=[],
    session_tags=[
        "london",
        "new_york",
        "london_newyork_overlap",
    ],
    constraints=[
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=1.5),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "trend_confirm",
    ],
    execution_profile="pullback_v1",
    max_trades_per_hour=5,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "features.optional",
        "constraints.spread_filter",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
    ],
    base_archetype_id=None,
)


# ---------------------------------------------------------------------------
# MeanReversion — trade deviations from mean
# ---------------------------------------------------------------------------

MEAN_REVERSION_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="mean_reversion",
    base_type="mean_reversion",
    description=(
        "Trade deviations from mean with bounded risk. "
        "Bollinger-style entry, trend confirmation as filter. Phase 2+."
    ),
    v1_priority=False,
    quality_class="native_supported",

    required_features=[
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "indicators/atr_14",
        "indicators/vwap",
        "volume/mfi",
        "volume/profile",
    ],
    excluded_features=[],
    session_tags=[
        "london",
        "new_york",
        "london_newyork_overlap",
        "tokyo",
        "sydney",
    ],
    constraints=[
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=2.0),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
    ],
    execution_profile="mean_reversion_v1",
    max_trades_per_hour=None,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "features.optional",
        "constraints.spread_filter",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
    ],
    base_archetype_id=None,
)


# ---------------------------------------------------------------------------
# SessionTransition — capture regime changes near session boundaries
# ---------------------------------------------------------------------------

SESSION_TRANSITION_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="session_transition",
    base_type="session_transition",
    description=(
        "Strategies that capture transitions between market regimes or sessions. "
        "Watches for regime changes near session boundaries. Phase 2+."
    ),
    v1_priority=False,
    quality_class="native_supported",

    required_features=[
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/atr_14",
        "indicators/rsi_14",
        "indicators/macd",
        "volume/rvol",
        "volume/mfi",
        "microstructure/spread",
    ],
    excluded_features=[],
    session_tags=[
        "london",
        "new_york",
        "london_newyork_overlap",
    ],
    constraints=[
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=2.0),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "regime_transition",
    ],
    execution_profile="session_transition_v1",
    max_trades_per_hour=None,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "features.optional",
        "constraints.spread_filter",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
    ],
    base_archetype_id=None,
)


# ---------------------------------------------------------------------------
# Registry bootstrap — register all V1 archetypes
# ---------------------------------------------------------------------------

_FULL_REGISTRY: Optional[ArchetypeRegistry] = None


def get_full_registry() -> ArchetypeRegistry:
    """
    Get or create a full ArchetypeRegistry with all V1 archetypes registered.

    The singleton is created on first call. Subsequent calls return the same
    instance with all archetypes pre-registered:

        - ORB (opening_range_breakout) — from orb.py
        - LondonORB (london_orb)       — derived from ORB
        - NYORB (ny_orb)               — derived from ORB
        - ScalperM1 (scalper_m1)       — derived from ORB
        - BreakoutScalper (breakout_scalper)     — structural stub
        - PullbackScalper (pullback_scalper)     — structural stub
        - MeanReversion (mean_reversion)         — structural stub
        - SessionTransition (session_transition)  — structural stub

    Returns:
        ArchetypeRegistry: singleton registry with 8 archetypes registered
    """
    global _FULL_REGISTRY
    if _FULL_REGISTRY is None:
        _FULL_REGISTRY = ArchetypeRegistry()

        # ORB — the base archetype
        _FULL_REGISTRY.register(ORB_ARCHETYPE)

        # Derived archetypes — from derived.py (imported lazily to avoid circular refs)
        from src.library.archetypes.derived import (
            LONDON_ORB_ARCHETYPE,
            NY_ORB_ARCHETYPE,
            SCALPER_M1_ARCHETYPE,
        )

        _FULL_REGISTRY.register(LONDON_ORB_ARCHETYPE)
        _FULL_REGISTRY.register(NY_ORB_ARCHETYPE)
        _FULL_REGISTRY.register(SCALPER_M1_ARCHETYPE)

        # Structural stubs — register all 4 stub archetypes
        _FULL_REGISTRY.register(BREAKOUT_SCALPER_ARCHETYPE)
        _FULL_REGISTRY.register(PULLBACK_SCALPER_ARCHETYPE)
        _FULL_REGISTRY.register(MEAN_REVERSION_ARCHETYPE)
        _FULL_REGISTRY.register(SESSION_TRANSITION_ARCHETYPE)

    return _FULL_REGISTRY


__all__ = [
    "BREAKOUT_SCALPER_ARCHETYPE",
    "PULLBACK_SCALPER_ARCHETYPE",
    "MEAN_REVERSION_ARCHETYPE",
    "SESSION_TRANSITION_ARCHETYPE",
    "get_full_registry",
]
