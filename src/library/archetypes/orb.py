"""
QuantMindLib V1 — OpeningRangeBreakout Archetype

Opening Range Breakout (ORB) — trade breakouts of the first N minutes of a session.
Classic ICT/Paul Dean approach. Filters for session, spread, and news state.

This module provides:
    ORB_ARCHETYPE:     Frozen ArchetypeSpec constant
    OpeningRangeBreakout: Concrete BaseArchetype subclass
    get_default_registry: Module-level singleton for the default ArchetypeRegistry
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.registry import ArchetypeRegistry

if TYPE_CHECKING:
    from src.library.core.domain.bot_spec import BotSpec


# ---------------------------------------------------------------------------
# Module-level constant: ORB_ARCHETYPE
# ---------------------------------------------------------------------------

ORB_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="opening_range_breakout",
    base_type="breakout",
    description=(
        "Opening Range Breakout — trade breakouts of the first N minutes of a session. "
        "Classic ICT/Paul Dean approach. Filters for session, spread, and news state."
    ),
    v1_priority=True,
    quality_class="native_supported",

    # ORB requires: ATR (volatility filter), Session Detector, Session Blackout
    required_features=[
        "indicators/atr_14",
        "session/detector",
        "session/blackout",
    ],
    # ORB benefits from: RVOL (volume confirmation), VWAP (value reference)
    optional_features=[
        "indicators/rsi_14",
        "indicators/macd",
        "indicators/vwap",
        "volume/rvol",
        "volume/mfi",
        "volume/profile",
        "microstructure/spread",
        "microstructure/tob_pressure",
        "orderflow/spread_behavior",
        "orderflow/dom_pressure",
        "orderflow/depth_thinning",
    ],
    # ORB specifically excludes mean reversion indicators
    excluded_features=[
        "indicators/bollinger",  # mean reversion — conflicts with momentum breakout
    ],
    # Session scope: London and NY are primary ORB sessions
    session_tags=[
        "london",
        "london_newyork_overlap",
        "new_york",
        "tokyo",
        "sydney",
    ],
    # ORB constraints
    constraints=[
        ConstraintSpec(constraint_id="spread_filter", operator="<", value=1.5),
        ConstraintSpec(constraint_id="min_range_minutes", operator=">=", value=10),
        ConstraintSpec(constraint_id="max_range_minutes", operator="<=", value=120),
        ConstraintSpec(constraint_id="news_clear", operator="==", value="clear"),
    ],
    # Safety confirmations required
    confirmations=[
        "spread_ok",
        "sentinel_allow",
        "session_active",
        "news_clear",
        "atr_filter",
    ],
    execution_profile="orb_v1",
    max_trades_per_hour=3,

    default_allowed_areas=[
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "parameters.range_minutes",
        "parameters.breakout_pips",
        "parameters.kelly_fraction",
        "features.optional",
        "constraints.spread_filter",
        "constraints.min_range_minutes",
        "constraints.max_range_minutes",
    ],
    default_locked_components=[
        "archetype",
        "features.required",
        "symbol_scope",
        "execution_profile",
        "confirmations",
    ],
    base_archetype_id=None,
)


# ---------------------------------------------------------------------------
# Concrete archetype implementation: OpeningRangeBreakout
# ---------------------------------------------------------------------------

class OpeningRangeBreakout(BaseArchetype):
    """
    Concrete archetype for Opening Range Breakout strategies.

    Extends the ORB ArchetypeSpec with ORB-specific validation and feature suggestions.

    Quality class: native_supported (computed from cTrader OHLCV + session data)
    """

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the ORB ArchetypeSpec."""
        return ORB_ARCHETYPE

    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        """
        ORB-specific validation beyond the generic CompositionValidator checks.

        Validates ORB-specific parameter expectations.
        Returns a list of error messages (empty = valid).
        """
        errors: List[str] = []

        # ORB-specific: must have a breakout or range-related feature
        has_momentum = any(
            "macd" in f or "rsi" in f or "atr" in f
            for f in bot_spec.features
        )
        if not has_momentum:
            errors.append(
                "ORBValidation: No momentum indicator found "
                "(RSI, MACD, or ATR recommended)"
            )

        # ORB-specific: session list should include london or new_york
        has_primary_session = any(
            s in bot_spec.sessions
            for s in ["london", "new_york", "london_newyork_overlap"]
        )
        if not has_primary_session:
            errors.append(
                "ORBValidation: No primary ORB session "
                "(london/new_york/london_ny_overlap)"
            )

        # ORB-specific: confirmations should include spread_ok
        if "spread_ok" not in bot_spec.confirmations:
            errors.append(
                "ORBValidation: spread_ok confirmation required for ORB"
            )

        return errors

    def suggest_features(
        self,
        context: Dict[str, object],
    ) -> List[str]:
        """
        Context-aware feature suggestions for ORB.

        Args:
            context: Dict with keys:
                - regime: str (e.g. "TREND_STABLE", "VOLATILE")
                - session: str (current session)
                - spread_pips: float (current spread in pips)
                - volume_regime: str ("high" | "normal" | "low")

        Returns:
            List of feature_ids to consider adding to the composition.
        """
        suggestions: List[str] = []

        regime = context.get("regime", "TREND_STABLE")
        session = str(context.get("session", ""))
        spread_pips = float(context.get("spread_pips", 0.0))
        volume_regime = str(context.get("volume_regime", "normal"))

        # Always suggest RVOL if volume regime is high or if it's a high-activity session
        if volume_regime == "high" or session in ("london", "london_newyork_overlap"):
            suggestions.append("volume/rvol")

        # In volatile regime, suggest ATR over RSI/MACD
        if regime == "VOLATILE":
            suggestions.append("indicators/atr_14")
        elif regime == "TREND_STABLE":
            suggestions.append("indicators/macd")

        # Tight spread session: suggest order flow features for entry timing
        if spread_pips < 1.0:
            suggestions.extend([
                "orderflow/spread_behavior",
                "microstructure/tob_pressure",
            ])

        # London session: suggest volume profile
        if session == "london":
            suggestions.append("volume/profile")

        return suggestions


# ---------------------------------------------------------------------------
# Registry bootstrap
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: Optional[ArchetypeRegistry] = None


def get_default_registry() -> ArchetypeRegistry:
    """
    Get or create the default ArchetypeRegistry with ORB pre-registered.

    The singleton is created on first call. Subsequent calls return the same instance.
    ORB registers itself when this module is imported, ensuring the default
    ArchetypeRegistry has ORB pre-loaded.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ArchetypeRegistry()
        _DEFAULT_REGISTRY.register(ORB_ARCHETYPE)
    return _DEFAULT_REGISTRY
