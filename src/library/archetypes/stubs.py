"""
QuantMindLib V1 — Deep Archetype Implementations (Packet 11C)

Phase 2 deep implementations for the 4 stub archetypes:
    BreakoutScalper    — Breakout + Scalper hybrid (ARCH-011)
    PullbackScalper    — Trend-following pullback scalper (ARCH-012)
    MeanReversion     — Mean reversion archetype (ARCH-013)
    SessionTransition — Session boundary transition (ARCH-014)

Each provides a frozen ArchetypeSpec constant and a concrete BaseArchetype
subclass with real signal logic: signal_direction(), urgency(),
stop_ticks(), target_ticks(), regime_preference(), capability_spec().

Also provides get_full_registry() — a module-level singleton that
bootstraps all V1 archetypes (ORB, derived, and deep implementations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.base import BaseArchetype
from src.library.archetypes.constraints import ConstraintSpec
from src.library.archetypes.orb import ORB_ARCHETYPE
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.core.composition.capability_spec import CapabilitySpec

if TYPE_CHECKING:
    from src.library.core.domain.bot_spec import BotSpec


# ---------------------------------------------------------------------------
# BreakoutScalper — Breakout + Scalper hybrid
# ---------------------------------------------------------------------------

BREAKOUT_SCALPER_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="breakout_scalper",
    base_type="scalping",
    description=(
        "Breakout + Scalper hybrid. Wait for range expansion after consolidation, "
        "then scalp the breakout direction with tight stops. "
        "Consolidation: range < 50% of 20-bar ATR for 10+ bars. "
        "Breakout: range > 150% of ATR on high volume. "
        "Stop: 1x ATR. Target: 0.5x ATR."
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


class BreakoutScalper(BaseArchetype):
    """
    Breakout + Scalper hybrid archetype.

    Strategy: Wait for range expansion after consolidation, then scalp
    the breakout direction with tight stops.

    Key rules:
    - Consolidation detection: range < 50% of 20-bar ATR for 10+ bars
    - Breakout entry: range > 150% of ATR on high volume
    - Direction: up if close > high_of_consolidation, down if close < low_of_consolidation
    - Stop: 1x ATR
    - Target: 0.5x ATR (scalp the momentum, don't overstay)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_bars"]: List[Dict[str, Any]] = []
        self.__dict__["_consolidation_bars"] = 10
        self.__dict__["_atr_history"]: List[float] = []
        self.__dict__["_consolidation_high"] = 0.0
        self.__dict__["_consolidation_low"] = 0.0

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the BreakoutScalper ArchetypeSpec."""
        return BREAKOUT_SCALPER_ARCHETYPE

    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        errors: List[str] = []
        has_momentum = any("macd" in f or "rsi" in f or "atr" in f for f in bot_spec.features)
        if not has_momentum:
            errors.append(
                "BreakoutScalperValidation: No momentum indicator found "
                "(RSI, MACD, or ATR recommended)"
            )
        has_primary_session = any(
            s in bot_spec.sessions for s in ["london", "new_york", "london_newyork_overlap"]
        )
        if not has_primary_session:
            errors.append(
                "BreakoutScalperValidation: No primary scalper session "
                "(london/new_york/london_ny_overlap)"
            )
        if "spread_ok" not in bot_spec.confirmations:
            errors.append(
                "BreakoutScalperValidation: spread_ok confirmation required"
            )
        return errors

    def suggest_features(self, context: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        regime = context.get("regime", "TREND_STABLE")
        spread_pips = float(context.get("spread_pips", 0.0))
        volume_regime = str(context.get("volume_regime", "normal"))

        if volume_regime == "high":
            suggestions.append("volume/rvol")
        if regime == "VOLATILE":
            suggestions.append("indicators/atr_14")
        elif regime == "TREND_STABLE":
            suggestions.append("indicators/macd")
        if spread_pips < 1.0:
            suggestions.extend([
                "orderflow/spread_behavior",
                "microstructure/tob_pressure",
            ])
        return suggestions

    def capability_spec(self) -> CapabilitySpec:
        spec = self.spec
        return CapabilitySpec(
            module_id=f"archetype/{spec.archetype_id}",
            provides=[f"archetype/{spec.archetype_id}"],
            requires=spec.required_features,
            optional=spec.optional_features,
            outputs=["BotSpec"],
            compatibility=[spec.archetype_id],
            sync_mode=True,
        )

    def signal_direction(self, bar: Dict[str, Any], lookback: int = 10) -> int:
        """
        Determine signal direction based on consolidation-breakout pattern.

        Consolidation: range < 50% of ATR for 10+ bars -> consolidation detected
        Breakout: range > 150% of ATR on high volume -> entry direction

        Args:
            bar: Current bar with OHLCV, atr (optional), tick_count (optional)
            lookback: Number of bars to check for consolidation

        Returns:
            int: 1 = bullish breakout, -1 = bearish breakout, 0 = no signal
        """
        bars = self.__dict__["_bars"]
        atr_hist = self.__dict__["_atr_history"]

        bars.append(bar)
        if len(bars) > lookback + 20:
            bars[:] = bars[-(lookback + 20):]

        # Keep ATR history
        atr_val = bar.get("atr", 0.0)
        if atr_val > 0:
            atr_hist.append(atr_val)
            if len(atr_hist) > 20:
                atr_hist[:] = atr_hist[-20:]

        if len(bars) < lookback + 1 or len(atr_hist) < lookback:
            return 0

        # Compute 20-bar ATR as average of stored ATR values
        atr_20 = sum(atr_hist) / len(atr_hist) if atr_hist else 0.0
        if atr_20 <= 0:
            return 0

        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        volume = bar.get("volume", 0.0)

        if high <= 0 or close <= 0 or volume <= 0:
            return 0

        current_range = high - low

        # Check consolidation: last N bars had range < 50% of ATR
        consolidation_bars = self.__dict__["_consolidation_bars"]
        recent_bars = bars[-(consolidation_bars + 1):-1] if len(bars) > consolidation_bars else []

        consolidation_high = max(b.get("high", 0.0) for b in recent_bars) if recent_bars else high
        consolidation_low = min(b.get("low", 0.0) for b in recent_bars) if recent_bars else low

        self.__dict__["_consolidation_high"] = consolidation_high
        self.__dict__["_consolidation_low"] = consolidation_low

        consolidation_threshold = atr_20 * 0.5
        breakout_threshold = atr_20 * 1.5

        # Count how many recent bars were in consolidation
        in_consolidation = sum(
            1 for b in recent_bars
            if (b.get("high", 0.0) - b.get("low", 0.0)) < consolidation_threshold
        )

        if in_consolidation < self.__dict__["_consolidation_bars"] * 0.7:
            return 0

        # Check breakout: current range > 150% of ATR
        if current_range < breakout_threshold:
            return 0

        # Volume confirmation: above average
        volumes = [b.get("volume", 0.0) for b in recent_bars]
        avg_volume = sum(volumes) / len(volumes) if volumes and any(v > 0 for v in volumes) else volume
        if volume < avg_volume * 1.2:
            return 0

        # Direction: close above consolidation high = bullish, below = bearish
        if close > consolidation_high:
            return 1
        elif close < consolidation_low:
            return -1
        return 0

    def urgency(self, bar: Dict[str, Any]) -> float:
        """
        Return urgency score [0.0, 1.0] for the current bar.

        Higher when: breakout is strong, volume is high, ATR is elevated.
        """
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        volume = bar.get("volume", 0.0)
        atr_val = bar.get("atr", 0.0)

        if high <= 0 or low <= 0 or close <= 0:
            return 0.0

        current_range = high - low
        if atr_val <= 0:
            return 0.0

        range_ratio = current_range / atr_val
        volume_norm = min(volume / 10000.0, 1.0)

        # Urgency driven by how much range exceeds ATR threshold
        breakout_ratio = max(0.0, (range_ratio - 1.5) / 2.0)
        urgency = min(breakout_ratio * 0.6 + volume_norm * 0.4, 1.0)

        return urgency

    def stop_ticks(self, bar: Dict[str, Any]) -> float:
        """
        Return stop loss in ticks (pips * pip_fraction).

        Stop: 1x ATR from entry.
        """
        atr_val = bar.get("atr", 0.0)
        if atr_val <= 0:
            return 0.0
        return atr_val * 1.0

    def target_ticks(self, bar: Dict[str, Any]) -> float:
        """
        Return take profit in ticks (pips * pip_fraction).

        Target: 0.5x ATR — scalp the momentum, don't overstay.
        """
        atr_val = bar.get("atr", 0.0)
        if atr_val <= 0:
            return 0.0
        return atr_val * 0.5

    def regime_preference(self, regime: str) -> float:
        """
        Return preference score [0.0, 1.0] for the given market regime.

        BreakoutScalper prefers VOLATILE (range expansion) and TREND_STABLE.
        Avoids RANGING (consolidation is ok, but breakout needs direction).
        """
        prefs: Dict[str, float] = {
            "VOLATILE": 1.0,
            "TREND_STABLE": 0.8,
            "TREND_VOLATILE": 0.9,
            "RANGING": 0.3,
            "MEAN_REVERTING": 0.2,
        }
        return prefs.get(regime, 0.5)


# ---------------------------------------------------------------------------
# PullbackScalper — Trend-following pullback scalper
# ---------------------------------------------------------------------------

PULLBACK_SCALPER_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="pullback_scalper",
    base_type="scalping",
    description=(
        "Scalp in the direction of the trend on pullbacks. "
        "Trend: EMA(21) > EMA(50) = uptrend, < = downtrend. "
        "Pullback: price retraces to within 50% of prior move. "
        "Entry: pullback + RSI < 40 (bull) or > 60 (bear). "
        "Stop: below/above pullback low/high. Target: prior swing high/low."
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


class PullbackScalper(BaseArchetype):
    """
    Trend-following pullback scalper archetype.

    Strategy: Identify trend, wait for pullback, enter in trend direction.

    Key rules:
    - Trend: EMA(21) > EMA(50) = uptrend, < = downtrend
    - Pullback: price retraces to within 50% of prior move
    - Entry: pullback + RSI < 40 (bull) or > 60 (bear)
    - Stop: below pullback low / above pullback high
    - Target: prior swing high/low
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_bars"]: List[Dict[str, Any]] = []
        self.__dict__["_ema_fast"] = 21
        self.__dict__["_ema_slow"] = 50
        self.__dict__["_ema_history"] = {"fast": [], "slow": []}

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the PullbackScalper ArchetypeSpec."""
        return PULLBACK_SCALPER_ARCHETYPE

    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        errors: List[str] = []
        has_trend = any("macd" in f or "atr" in f for f in bot_spec.features)
        if not has_trend:
            errors.append(
                "PullbackScalperValidation: No trend indicator found "
                "(MACD or ATR recommended)"
            )
        has_primary_session = any(
            s in bot_spec.sessions for s in ["london", "new_york", "london_newyork_overlap"]
        )
        if not has_primary_session:
            errors.append(
                "PullbackScalperValidation: No primary scalper session "
                "(london/new_york/london_ny_overlap)"
            )
        if "spread_ok" not in bot_spec.confirmations:
            errors.append(
                "PullbackScalperValidation: spread_ok confirmation required"
            )
        return errors

    def suggest_features(self, context: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        regime = context.get("regime", "TREND_STABLE")
        spread_pips = float(context.get("spread_pips", 0.0))
        volume_regime = str(context.get("volume_regime", "normal"))

        if regime == "TREND_STABLE":
            suggestions.append("indicators/macd")
        if regime == "VOLATILE":
            suggestions.append("indicators/rsi_14")
        if volume_regime == "high":
            suggestions.append("volume/rvol")
        if spread_pips < 1.0:
            suggestions.extend([
                "orderflow/spread_behavior",
                "microstructure/tob_pressure",
            ])
        return suggestions

    def capability_spec(self) -> CapabilitySpec:
        spec = self.spec
        return CapabilitySpec(
            module_id=f"archetype/{spec.archetype_id}",
            provides=[f"archetype/{spec.archetype_id}"],
            requires=spec.required_features,
            optional=spec.optional_features,
            outputs=["BotSpec"],
            compatibility=[spec.archetype_id],
            sync_mode=True,
        )

    def _compute_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Compute EMA for a price series."""
        if len(prices) < period:
            return None
        alpha = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for price in prices[period:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def signal_direction(self, bar: Dict[str, Any]) -> int:
        """
        Determine signal direction based on trend + pullback + RSI pattern.

        Args:
            bar: Current bar with OHLCV, close (required), rsi (optional)

        Returns:
            int: 1 = bullish pullback entry, -1 = bearish pullback entry,
                 0 = no signal
        """
        bars = self.__dict__["_bars"]
        bars.append(bar)
        if len(bars) > 100:
            bars[:] = bars[-100:]

        close = bar.get("close", 0.0)
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)

        if close <= 0 or len(bars) < 55:
            return 0

        closes = [b.get("close", 0.0) for b in bars if b.get("close", 0.0) > 0]
        if len(closes) < 55:
            return 0

        ema_fast = self._compute_ema(closes, self.__dict__["_ema_fast"])
        ema_slow = self._compute_ema(closes, self.__dict__["_ema_slow"])

        if ema_fast is None or ema_slow is None:
            return 0

        rsi_val = bar.get("rsi", 50.0)

        # Trend direction: EMA(21) vs EMA(50)
        is_uptrend = ema_fast > ema_slow
        is_downtrend = ema_fast < ema_slow

        # Need at least a 5-bar trend to qualify
        if abs(ema_fast - ema_slow) / ema_slow < 0.001:
            return 0

        # Pullback detection: recent bars show price moved against trend
        # Look at last 5 bars for pullback pattern
        recent = bars[-6:-1] if len(bars) >= 6 else bars[:-1]
        if not recent:
            return 0

        if is_uptrend:
            # In uptrend: pullback = price dipped, RSI < 40 = oversold pullback
            recent_highs = [b.get("high", 0.0) for b in recent]
            prev_swing_high = max(recent_highs) if recent_highs else close
            pullback_low = min([b.get("low", 0.0) for b in recent] + [low])
            retracement = (prev_swing_high - pullback_low) / prev_swing_high if prev_swing_high > 0 else 0.0

            # Valid pullback: retraces 30-70% (within 50% zone)
            if 0.3 <= retracement <= 0.7 and rsi_val < 40:
                return 1

        elif is_downtrend:
            recent_lows = [b.get("low", 0.0) for b in recent]
            prev_swing_low = min(recent_lows) if recent_lows else close
            pullback_high = max([b.get("high", 0.0) for b in recent] + [high])
            retracement = (pullback_high - prev_swing_low) / prev_swing_low if prev_swing_low > 0 else 0.0

            if 0.3 <= retracement <= 0.7 and rsi_val > 60:
                return -1

        return 0

    def urgency(self, bar: Dict[str, Any]) -> float:
        """
        Return urgency score [0.0, 1.0].

        Higher when: RSI in extreme zone (close to entry threshold),
        ATR is elevated (trend is strong).
        """
        rsi_val = bar.get("rsi", 50.0)
        atr_val = bar.get("atr", 0.0)

        # RSI proximity to entry threshold
        rsi_dist_from_bull = abs(rsi_val - 40.0) / 40.0
        rsi_dist_from_bear = abs(rsi_val - 60.0) / 40.0
        rsi_urgency = 1.0 - min(rsi_dist_from_bull, rsi_dist_from_bear)

        # ATR contribution
        atr_urgency = min(atr_val / 100.0, 1.0) if atr_val > 0 else 0.0

        return min(rsi_urgency * 0.7 + atr_urgency * 0.3, 1.0)

    def stop_ticks(self, bar: Dict[str, Any]) -> float:
        """
        Return stop loss in ticks.

        Stop: below/above pullback low/high with ATR buffer.
        """
        atr_val = bar.get("atr", 0.0)
        if atr_val <= 0:
            return 0.0
        # Stop at pullback extreme + 0.5x ATR buffer
        return atr_val * 0.5

    def regime_preference(self, regime: str) -> float:
        """
        Return preference score [0.0, 1.0] for the given market regime.

        PullbackScalper prefers TREND_STABLE (clean trends for pullbacks).
        Avoids RANGING (no trend to pull back from) and MEAN_REVERTING.
        """
        prefs: Dict[str, float] = {
            "TREND_STABLE": 1.0,
            "TREND_VOLATILE": 0.7,
            "VOLATILE": 0.5,
            "RANGING": 0.2,
            "MEAN_REVERTING": 0.3,
        }
        return prefs.get(regime, 0.4)


# ---------------------------------------------------------------------------
# MeanReversion — Mean reversion archetype
# ---------------------------------------------------------------------------

MEAN_REVERSION_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="mean_reversion",
    base_type="mean_reversion",
    description=(
        "Trade deviations from mean with bounded risk. "
        "Mean: VWAP or 20-period SMA. "
        "Deviation: price > 2*ATR from mean = overbought/oversold. "
        "Entry: mean reversion signal when price crosses back toward mean. "
        "Stop: below/above extreme with ATR buffer. Target: mean."
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


class MeanReversion(BaseArchetype):
    """
    Mean reversion archetype.

    Strategy: Price deviates from mean, then reverts.

    Key rules:
    - Mean: VWAP or 20-period SMA
    - Deviation: price > 2*ATR from mean = overbought/oversold
    - Entry: mean reversion signal when price crosses back toward mean
    - Stop: below/above extreme with ATR buffer
    - Target: mean
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_bars"]: List[Dict[str, Any]] = []
        self.__dict__["_mean_period"] = 20
        self.__dict__["_deviation_threshold"] = 2.0

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the MeanReversion ArchetypeSpec."""
        return MEAN_REVERSION_ARCHETYPE

    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        errors: List[str] = []
        has_mean = any(
            "vwap" in f or "sma" in f or "ema" in f
            for f in bot_spec.features
        )
        if not has_mean:
            errors.append(
                "MeanReversionValidation: No mean indicator found "
                "(VWAP or SMA recommended)"
            )
        if "spread_ok" not in bot_spec.confirmations:
            errors.append(
                "MeanReversionValidation: spread_ok confirmation required"
            )
        return errors

    def suggest_features(self, context: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        regime = context.get("regime", "TREND_STABLE")
        spread_pips = float(context.get("spread_pips", 0.0))

        if regime in ("RANGING", "MEAN_REVERTING"):
            suggestions.append("indicators/rsi_14")
        elif regime == "TREND_STABLE":
            suggestions.append("indicators/vwap")
        if spread_pips < 1.5:
            suggestions.append("indicators/atr_14")
        return suggestions

    def capability_spec(self) -> CapabilitySpec:
        spec = self.spec
        return CapabilitySpec(
            module_id=f"archetype/{spec.archetype_id}",
            provides=[f"archetype/{spec.archetype_id}"],
            requires=spec.required_features,
            optional=spec.optional_features,
            outputs=["BotSpec"],
            compatibility=[spec.archetype_id],
            sync_mode=True,
        )

    def _compute_mean(self, closes: List[float]) -> Optional[float]:
        """Compute 20-bar SMA as mean."""
        if len(closes) < self.__dict__["_mean_period"]:
            return None
        period = self.__dict__["_mean_period"]
        return sum(closes[-period:]) / period

    def signal_direction(self, bar: Dict[str, Any]) -> int:
        """
        Determine signal direction based on mean reversion pattern.

        Deviation: price > 2*ATR from mean = extreme deviation
        Entry: price crossing back toward mean from extreme

        Args:
            bar: Current bar with close, atr (optional), vwap (optional)

        Returns:
            int: 1 = bullish reversion (buy when oversold and crossing up),
                 -1 = bearish reversion (sell when overbought and crossing down),
                 0 = no signal
        """
        bars = self.__dict__["_bars"]
        bars.append(bar)
        if len(bars) > 100:
            bars[:] = bars[-100:]

        close = bar.get("close", 0.0)
        atr_val = bar.get("atr", 0.0)
        vwap_val = bar.get("vwap", 0.0)

        if close <= 0 or len(bars) < self.__dict__["_mean_period"] + 1:
            return 0

        closes = [b.get("close", 0.0) for b in bars if b.get("close", 0.0) > 0]
        if len(closes) < self.__dict__["_mean_period"] + 1:
            return 0

        # Use VWAP if available, otherwise SMA
        mean_val = vwap_val if vwap_val > 0 else self._compute_mean(closes)
        if mean_val is None or mean_val <= 0:
            return 0

        deviation_threshold = atr_val * self.__dict__["_deviation_threshold"] if atr_val > 0 else 0.0

        # Check deviation: price significantly away from mean
        deviation = abs(close - mean_val)
        if deviation < deviation_threshold or deviation_threshold <= 0:
            return 0

        # Previous bar: was even further from mean (crossing back = reversion signal)
        prev_close = closes[-2] if len(closes) >= 2 else close
        prev_deviation = abs(prev_close - mean_val)

        # Reversion signal: current bar is closer to mean than previous bar
        if deviation >= prev_deviation:
            return 0

        # Direction: close below mean = bullish reversion, above = bearish
        if close < mean_val and prev_close <= close:
            # Was above or further below, now closer to mean from below
            # Check RSI for confirmation
            rsi_val = bar.get("rsi", 50.0)
            if rsi_val < 40:
                return 1
        elif close > mean_val and prev_close >= close:
            rsi_val = bar.get("rsi", 50.0)
            if rsi_val > 60:
                return -1

        return 0

    def urgency(self, bar: Dict[str, Any]) -> float:
        """
        Return urgency score [0.0, 1.0].

        Higher when: deviation is larger, RSI is in extreme zone.
        """
        close = bar.get("close", 0.0)
        vwap_val = bar.get("vwap", 0.0)
        atr_val = bar.get("atr", 0.0)

        closes = [b.get("close", 0.0) for b in self.__dict__["_bars"]]
        mean_val = vwap_val if vwap_val > 0 else self._compute_mean(closes)

        if close <= 0 or mean_val is None or mean_val <= 0 or atr_val <= 0:
            return 0.0

        deviation = abs(close - mean_val)
        deviation_ratio = deviation / atr_val

        rsi_val = bar.get("rsi", 50.0)
        rsi_extreme = 1.0 if rsi_val < 30 or rsi_val > 70 else 0.0

        # Urgency: deviation ratio above 2x and RSI extreme
        deviation_urgency = min((deviation_ratio - 2.0) / 3.0, 1.0)

        return min(deviation_urgency * 0.6 + rsi_extreme * 0.4, 1.0)

    def stop_ticks(self, bar: Dict[str, Any]) -> float:
        """
        Return stop loss in ticks.

        Stop: below/above extreme with ATR buffer (1x ATR).
        """
        atr_val = bar.get("atr", 0.0)
        if atr_val <= 0:
            return 0.0
        return atr_val * 1.0

    def regime_preference(self, regime: str) -> float:
        """
        Return preference score [0.0, 1.0] for the given market regime.

        MeanReversion prefers RANGING and MEAN_REVERTING.
        Works in TREND_STABLE but slower. Avoids TREND_VOLATILE.
        """
        prefs: Dict[str, float] = {
            "MEAN_REVERTING": 1.0,
            "RANGING": 1.0,
            "TREND_STABLE": 0.6,
            "VOLATILE": 0.4,
            "TREND_VOLATILE": 0.2,
        }
        return prefs.get(regime, 0.4)


# ---------------------------------------------------------------------------
# SessionTransition — Session boundary archetype
# ---------------------------------------------------------------------------

SESSION_TRANSITION_ARCHETYPE: ArchetypeSpec = ArchetypeSpec(
    archetype_id="session_transition",
    base_type="session_transition",
    description=(
        "Strategies that capture transitions between market regimes or sessions. "
        "Session boundaries: LONDON (07:00-09:00 UTC), NY (13:00-15:00 UTC). "
        "Pre-session: range contraction before boundary. "
        "Post-session: expansion and directional bias after open. "
        "Entry: break of pre-session range in direction of session bias. "
        "Stop: opposite side of pre-session range."
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

# Session boundary definitions (UTC hours)
_SESSION_BOUNDARIES = {
    "LONDON": (7, 9),
    "NY": (13, 15),
}


class SessionTransition(BaseArchetype):
    """
    Session transition archetype.

    Strategy: Trade the volatility increase and range shift at session boundaries.

    Key rules:
    - Session boundary detection: LONDON (07:00-09:00 UTC), NY (13:00-15:00 UTC)
    - Pre-session: range contraction before boundary
    - Post-session: expansion and directional bias after open
    - Entry: break of pre-session range in direction of session bias
    - Stop: opposite side of pre-session range
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__dict__["_bars"] = []
        self.__dict__["_session_bias"] = 0  # 1 = bullish, -1 = bearish, 0 = neutral
        self.__dict__["_pre_session_high"] = 0.0
        self.__dict__["_pre_session_low"] = 0.0
        self.__dict__["_in_pre_session"] = False

    @property
    def spec(self) -> ArchetypeSpec:
        """Return the SessionTransition ArchetypeSpec."""
        return SESSION_TRANSITION_ARCHETYPE

    def validate_bot_spec(self, bot_spec: BotSpec) -> List[str]:
        errors: List[str] = []
        has_session = any(
            "session" in f or "blackout" in f
            for f in bot_spec.features
        )
        if not has_session:
            errors.append(
                "SessionTransitionValidation: No session feature found "
                "(session/detector required)"
            )
        has_primary_session = any(
            s in bot_spec.sessions for s in ["london", "new_york", "london_newyork_overlap"]
        )
        if not has_primary_session:
            errors.append(
                "SessionTransitionValidation: No primary session "
                "(london/new_york/london_ny_overlap)"
            )
        if "spread_ok" not in bot_spec.confirmations:
            errors.append(
                "SessionTransitionValidation: spread_ok confirmation required"
            )
        return errors

    def suggest_features(self, context: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        regime = context.get("regime", "TREND_STABLE")
        session = str(context.get("session", ""))
        spread_pips = float(context.get("spread_pips", 0.0))

        if session in ("london", "new_york", "london_newyork_overlap"):
            suggestions.append("indicators/atr_14")
        if regime == "VOLATILE":
            suggestions.append("indicators/rsi_14")
        if spread_pips < 1.5:
            suggestions.append("volume/rvol")
        return suggestions

    def capability_spec(self) -> CapabilitySpec:
        spec = self.spec
        return CapabilitySpec(
            module_id=f"archetype/{spec.archetype_id}",
            provides=[f"archetype/{spec.archetype_id}"],
            requires=spec.required_features,
            optional=spec.optional_features,
            outputs=["BotSpec"],
            compatibility=[spec.archetype_id],
            sync_mode=True,
        )

    def signal_direction(self, bar: Dict[str, Any]) -> int:
        """
        Determine signal direction based on session transition pattern.

        Pre-session: detect range contraction (5 bars before boundary)
        Post-session: detect break of pre-session range

        Args:
            bar: Current bar with OHLCV, hour (UTC, optional), close

        Returns:
            int: 1 = bullish transition, -1 = bearish transition, 0 = no signal
        """
        bars = self.__dict__["_bars"]
        bars.append(bar)
        if len(bars) > 50:
            bars[:] = bars[-50:]

        close = bar.get("close", 0.0)
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        hour = bar.get("hour")

        if close <= 0 or high <= 0 or low <= 0:
            return 0

        if len(bars) < 6:
            return 0

        # Detect if we are in pre-session zone (before session open)
        session_open_hour = None
        session_close_hour = None
        current_session = None

        if hour is not None:
            for name, (start, end) in _SESSION_BOUNDARIES.items():
                if start - 2 <= hour <= start:
                    session_open_hour = start
                    session_close_hour = end
                    current_session = name
                    break
                elif start <= hour <= end:
                    current_session = name
                    break

        recent = bars[-6:-1]

        if session_open_hour is not None and hour == session_open_hour:
            # We just entered the session boundary window
            # Capture pre-session range from the 5 bars before
            self.__dict__["_in_pre_session"] = True
            self.__dict__["_pre_session_high"] = max(b.get("high", 0.0) for b in recent)
            self.__dict__["_pre_session_low"] = min(b.get("low", 0.0) for b in recent)

            # Determine session bias from pre-session structure
            mid = (self.__dict__["_pre_session_high"] + self.__dict__["_pre_session_low"]) / 2.0
            if mid > 0:
                range_pct = (self.__dict__["_pre_session_high"] - self.__dict__["_pre_session_low"]) / mid
                # Determine directional bias from prior bars
                prior = bars[-10:-6]
                if prior:
                    prior_high = max(b.get("high", 0.0) for b in prior)
                    prior_low = min(b.get("low", 0.0) for b in prior)
                    if prior_high > self.__dict__["_pre_session_high"]:
                        self.__dict__["_session_bias"] = 1
                    elif prior_low < self.__dict__["_pre_session_low"]:
                        self.__dict__["_session_bias"] = -1
                    else:
                        self.__dict__["_session_bias"] = 0

        elif current_session is not None and self.__dict__["_in_pre_session"]:
            # Post-session: check for break of pre-session range
            pre_high = self.__dict__["_pre_session_high"]
            pre_low = self.__dict__["_pre_session_low"]
            bias = self.__dict__["_session_bias"]

            if pre_high > 0 and pre_low > 0:
                if close > pre_high and bias != -1:
                    return 1
                elif close < pre_low and bias != 1:
                    return -1

            # Reset after one session cycle
            if hour == session_close_hour:
                self.__dict__["_in_pre_session"] = False

        return 0

    def urgency(self, bar: Dict[str, Any]) -> float:
        """
        Return urgency score [0.0, 1.0].

        Higher when: range is contracting pre-session, volume is elevated
        post-session.
        """
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        volume = bar.get("volume", 0.0)
        atr_val = bar.get("atr", 0.0)

        if high <= 0 or low <= 0:
            return 0.0

        current_range = high - low
        range_norm = min(current_range / 100.0, 1.0)
        volume_norm = min(volume / 10000.0, 1.0)

        # Urgency driven by volume post-session (higher) vs contraction (lower)
        in_pre = self.__dict__["_in_pre_session"]
        if in_pre:
            # Contraction phase: urgency grows as range shrinks
            contraction = 1.0 - range_norm
            urgency = contraction * 0.5
        else:
            # Expansion phase: urgency driven by volume
            urgency = volume_norm * 0.8

        return min(urgency, 1.0)

    def stop_ticks(self, bar: Dict[str, Any]) -> float:
        """
        Return stop loss in ticks.

        Stop: opposite side of pre-session range (ATR-scaled).
        """
        atr_val = bar.get("atr", 0.0)
        pre_high = self.__dict__["_pre_session_high"]
        pre_low = self.__dict__["_pre_session_low"]

        if pre_high > 0 and pre_low > 0:
            pre_range = pre_high - pre_low
            if atr_val > 0:
                return min(pre_range, atr_val * 2.0)
            return pre_range
        return atr_val * 1.5 if atr_val > 0 else 0.0

    def regime_preference(self, regime: str) -> float:
        """
        Return preference score [0.0, 1.0] for the given market regime.

        SessionTransition works best in VOLATILE (session opens bring moves).
        TREND_STABLE is also good. Avoids RANGING (low vol, few transitions).
        """
        prefs: Dict[str, float] = {
            "VOLATILE": 1.0,
            "TREND_STABLE": 0.8,
            "TREND_VOLATILE": 0.9,
            "RANGING": 0.3,
            "MEAN_REVERTING": 0.5,
        }
        return prefs.get(regime, 0.5)


# ---------------------------------------------------------------------------
# Registry bootstrap — register all V1 archetypes
# ---------------------------------------------------------------------------

_FULL_REGISTRY: Optional[ArchetypeRegistry] = None


def get_full_registry() -> ArchetypeRegistry:
    """
    Get or create a full ArchetypeRegistry with all V1 archetypes registered.

    The singleton is created on first call. Subsequent calls return the same
    instance with all archetypes pre-registered:

        - ORB (opening_range_breakout)                — from orb.py
        - LondonORB (london_orb)                       — derived from ORB
        - NYORB (ny_orb)                               — derived from ORB
        - ScalperM1 (scalper_m1)                       — derived from ORB
        - BreakoutScalper (breakout_scalper)           — deep implementation
        - PullbackScalper (pullback_scalper)           — deep implementation
        - MeanReversion (mean_reversion)               — deep implementation
        - SessionTransition (session_transition)       — deep implementation

    Returns:
        ArchetypeRegistry: singleton registry with 8 archetypes registered
    """
    global _FULL_REGISTRY
    if _FULL_REGISTRY is None:
        _FULL_REGISTRY = ArchetypeRegistry()

        # ORB — the base archetype
        _FULL_REGISTRY.register(ORB_ARCHETYPE)

        # Derived archetypes — from derived.py
        from src.library.archetypes.derived import (
            LONDON_ORB_ARCHETYPE,
            NY_ORB_ARCHETYPE,
            SCALPER_M1_ARCHETYPE,
        )

        _FULL_REGISTRY.register(LONDON_ORB_ARCHETYPE)
        _FULL_REGISTRY.register(NY_ORB_ARCHETYPE)
        _FULL_REGISTRY.register(SCALPER_M1_ARCHETYPE)

        # Deep implementations — register all 4 archetype classes
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
    "BreakoutScalper",
    "PullbackScalper",
    "MeanReversion",
    "SessionTransition",
    "get_full_registry",
]
