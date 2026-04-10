"""
QuantMindLib V1 — Derived Archetypes (Packet 6E)

Three ORB-derived archetypes with session-specific overrides:
    LONDON_ORB_ARCHETYPE  — ORB scoped to London session
    NY_ORB_ARCHOTYPE      — ORB scoped to New York session
    SCALPER_M1_ARCHETYPE  — High-frequency scalper on M1

These are frozen ArchetypeSpec constants. They extend ORB with tighter
spread filters, restricted session scope, and adjusted trade rate limits.
"""

from __future__ import annotations

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.orb import ORB_ARCHETYPE

# ---------------------------------------------------------------------------
# London ORB — ORB scoped to London session
# ---------------------------------------------------------------------------

LONDON_ORB_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="london_orb",
    base_type="breakout",
    description=(
        "Opening Range Breakout filtered for London session. "
        "Higher volume activity (Wed peak), tighter spread filter."
    ),
    v1_priority=True,
    quality_class="native_supported",

    required_features=[
        "indicators/atr_14",
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "indicators/vwap",
        "volume/rvol",
        "volume/profile",
        "volume/mfi",
        "microstructure/spread",
        "microstructure/tob_pressure",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
        "orderflow/depth_thinning",
    ],
    excluded_features=[
        "indicators/bollinger",
    ],
    session_tags=[
        "london",
        "london_newyork_overlap",
    ],
    constraints=[
        # Tighter spread filter for London (1.2 vs ORB's 1.5)
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=1.2),
        ConstraintSpec(constraint_id="min_volume_rvol", operator=">=", value=0.8),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "atr_filter",
    ],
    execution_profile="london_orb_v1",
    max_trades_per_hour=2,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "parameters.range_minutes",
        "parameters.breakout_pips",
        "features.optional",
        "constraints.spread_filter",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
        "confirmations",
        "session_tags",
    ],
    base_archetype_id="opening_range_breakout",
)

# ---------------------------------------------------------------------------
# New York ORB — ORB scoped to New York session
# ---------------------------------------------------------------------------

NY_ORB_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="ny_orb",
    base_type="breakout",
    description=(
        "Opening Range Breakout filtered for New York session. "
        "Volatile market conditions, tighter spread filter (1.0 pips)."
    ),
    v1_priority=True,
    quality_class="native_supported",

    required_features=[
        "indicators/atr_14",
        "session/detector",
        "session/blackout",
    ],
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "indicators/vwap",
        "volume/rvol",
        "volume/profile",
        "volume/mfi",
        "microstructure/spread",
        "microstructure/tob_pressure",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
        "orderflow/depth_thinning",
    ],
    excluded_features=[
        "indicators/bollinger",
    ],
    session_tags=[
        "new_york",
        "london_newyork_overlap",
    ],
    constraints=[
        # Tighter spread filter for NY (1.0 vs ORB's 1.5)
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=1.0),
        ConstraintSpec(constraint_id="min_volume_rvol", operator=">=", value=0.8),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
        # Slippage control specific to volatile NY session
        ConstraintSpec(constraint_id="max_slippage_ticks", operator="<=", value=2),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "atr_filter",
    ],
    execution_profile="ny_orb_v1",
    max_trades_per_hour=2,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "parameters.range_minutes",
        "parameters.breakout_pips",
        "features.optional",
        "constraints.spread_filter",
        "constraints.max_slippage_ticks",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
        "confirmations",
        "session_tags",
    ],
    base_archetype_id="opening_range_breakout",
)

# ---------------------------------------------------------------------------
# Scalper M1 — High-frequency scalper on M1 timeframe
# ---------------------------------------------------------------------------

SCALPER_M1_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="scalper_m1",
    base_type="scalping",
    description=(
        "High-frequency scalper on M1 timeframe. "
        "Tight stops, order flow confirmation required."
    ),
    v1_priority=True,
    quality_class="native_supported",

    required_features=[
        "indicators/atr_14",
        "session/detector",
        "session/blackout",
        "microstructure/spread",
        "microstructure/tob_pressure",
    ],
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "volume/mfi",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
        "orderflow/depth_thinning",
        "microstructure/aggression",
    ],
    excluded_features=[
        "indicators/bollinger",
    ],
    session_tags=[
        "london",
        "new_york",
        "london_newyork_overlap",
    ],
    constraints=[
        # Very tight spread for scalping (0.8 pips)
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=0.8),
        ConstraintSpec(constraint_id="max_trades_per_hour", operator="<=", value=10),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "tob_pressure",
    ],
    execution_profile="scalp_fast_v1",
    max_trades_per_hour=10,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "features.optional",
        "constraints.spread_filter",
        "constraints.max_trades_per_hour",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
        "confirmations",
        "session_tags",
    ],
    base_archetype_id=None,
)


__all__ = [
    "LONDON_ORB_ARCHETYPE",
    "NY_ORB_ARCHETYPE",
    "SCALPER_M1_ARCHETYPE",
]
