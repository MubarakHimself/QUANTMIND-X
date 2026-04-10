"""
QuantMindLib V1 — StrategyCodeGenerator

Packet 8A: Converts BotSpec into executable Python strategy code
for FullBacktestPipeline (PythonStrategyTester pattern).

Generates Python code using the on_bar(tester) function signature
consumed by PythonStrategyTester.run().
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Tuple

from src.library.features.registry import FeatureRegistry

if TYPE_CHECKING:
    from src.library.core.domain.bot_spec import BotSpec


# ---------------------------------------------------------------------------
# Supported archetype identifiers
# ---------------------------------------------------------------------------

SUPPORTED_ARCHETYPES = {
    "orb",
    "scalper",
    "breakout_scalper",
    "pullback_scalper",
    "mean_reversion",
}


# ---------------------------------------------------------------------------
# Archetype → human-readable name
# ---------------------------------------------------------------------------

_ARCHETYPE_NAMES = {
    "orb": "OpeningRangeBreakout",
    "scalper": "Scalper",
    "breakout_scalper": "BreakoutScalper",
    "pullback_scalper": "PullbackScalper",
    "mean_reversion": "MeanReversion",
}


# ---------------------------------------------------------------------------
# Symbol → default timeframe hint (kept for reference; MQL5Timeframe is
# always derived from the timeframe string or passed explicitly).
# ---------------------------------------------------------------------------

_SYMBOL_TIMEFRAME = {
    "EURUSD": "M5",
    "GBPUSD": "M5",
    "USDJPY": "M5",
    "USDCHF": "M5",
    "AUDUSD": "M5",
    "USDCAD": "M5",
    "NZDUSD": "M5",
}


# ---------------------------------------------------------------------------
# Timeframe string → MQL5Timeframe constant (int)
# ---------------------------------------------------------------------------

_TIMEFRAME_MAP = {
    "M1": 16385,
    "M5": 16386,
    "M15": 16387,
    "M30": 16388,
    "H1": 16393,
    "H4": 16396,
    "D1": 16401,
    "W1": 16405,
    "MN1": 16408,
}


# ---------------------------------------------------------------------------
# Default timeframe when none is specified
# ---------------------------------------------------------------------------

_DEFAULT_TIMEFRAME = "M5"


# ---------------------------------------------------------------------------
# Session name → comment string embedded in generated code
# ---------------------------------------------------------------------------

_SESSION_COMMENTS = {
    "london": "London session (07:00-12:00 UTC)",
    "new_york": "New York session (12:00-17:00 UTC)",
    "london_newyork_overlap": "London/NY overlap (12:00-16:00 UTC)",
    "tokyo": "Tokyo session (00:00-06:00 UTC)",
    "sydney": "Sydney session (22:00-06:00 UTC)",
    "asian": "Asian session (00:00-09:00 UTC)",
    "ny_afternoon": "NY afternoon (14:00-17:00 UTC)",
}


# ---------------------------------------------------------------------------
# StrategyCodeGenerator
# ---------------------------------------------------------------------------

class StrategyCodeGenerator:
    """
    Converts a BotSpec into executable Python strategy code for
    FullBacktestPipeline.

    The generated code follows the PythonStrategyTester pattern:
        def on_bar(tester):
            ...

    The tester instance provides MQL5-style functions:
        tester.buy(symbol, volume)
        tester.sell(symbol, volume)
        tester.close_all_positions()
        tester.iClose(symbol, timeframe, shift)

    Supported archetypes:
        - orb             : Opening Range Breakout
        - scalper         : VWAP mean-reversion scalper
        - breakout_scalper : ORB breakout + scalper confirmations
        - pullback_scalper: Trend-following pullback entries
        - mean_reversion  : VWAP + RSI mean reversion

    Args:
        feature_registry: Optional FeatureRegistry used for
            generate_with_features() to resolve feature dependencies.
    """

    def __init__(self, feature_registry: Optional[FeatureRegistry] = None) -> None:
        self._registry = feature_registry

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate(self, bot_spec: BotSpec) -> str:
        """
        Generate Python strategy code from a BotSpec.

        Args:
            bot_spec: Bot specification to convert.

        Returns:
            Python source code string defining an ``on_bar`` function.

        The returned code:
            1. Reads bot_spec.archetype to determine signal logic
            2. Reads bot_spec.features for indicator references
            3. Uses bot_spec.symbol_scope[0] as the trading symbol
            4. Embeds bot_spec.sessions as a comment for session awareness
        """
        archetype = (bot_spec.archetype or "").lower()
        symbol = _first_or_default(bot_spec.symbol_scope, "EURUSD")
        sessions = list(bot_spec.sessions)
        features = list(bot_spec.features)

        if archetype == "orb":
            return self._generate_orb(symbol, sessions, features)
        elif archetype == "scalper":
            return self._generate_scalper(symbol, sessions, features)
        elif archetype == "breakout_scalper":
            return self._generate_breakout_scalper(symbol, sessions, features)
        elif archetype == "pullback_scalper":
            return self._generate_pullback_scalper(symbol, sessions, features)
        elif archetype == "mean_reversion":
            return self._generate_mean_reversion(symbol, sessions, features)
        else:
            return self._generate_unsupported(symbol, archetype)

    def generate_with_features(
        self,
        bot_spec: BotSpec,
        feature_registry: Optional[FeatureRegistry] = None,
    ) -> str:
        """
        Extended generation that includes feature-specific indicator calculations.

        Uses the FeatureRegistry to understand required indicators and injects
        their computation into the generated strategy.

        Args:
            bot_spec: Bot specification to convert.
            feature_registry: FeatureRegistry instance (falls back to the
                one passed at construction time).

        Returns:
            Python source code string with feature-specific indicators included.
        """
        registry = feature_registry or self._registry
        base_code = self.generate(bot_spec)

        if registry is None:
            return base_code

        # Enrich with feature-specific indicator calculations
        enriched = self._inject_feature_indicators(base_code, bot_spec, registry)
        return enriched

    def get_timeframe_for_symbol(self, symbol: str) -> int:
        """
        Map a symbol to an MQL5 timeframe constant.

        Most symbols default to M5 (16386). This method exists so callers
        can determine the right timeframe to pass to FullBacktestPipeline.

        Args:
            symbol: Trading symbol (e.g. "EURUSD").

        Returns:
            MQL5Timeframe integer constant.
        """
        hint = _SYMBOL_TIMEFRAME.get(symbol, _DEFAULT_TIMEFRAME)
        return _TIMEFRAME_MAP.get(hint, 16386)  # M5 = 16386

    def validate_bot_spec(self, bot_spec: BotSpec) -> Tuple[bool, List[str]]:
        """
        Validate that a BotSpec can be converted to strategy code.

        Checks:
            - archetype is supported
            - symbol_scope is non-empty
            - features are registered (if registry is available)

        Args:
            bot_spec: Bot specification to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
            Empty error list means valid.
        """
        errors: List[str] = []

        # Archetype check
        archetype = (bot_spec.archetype or "").lower()
        if not archetype:
            errors.append("BotSpec.archetype is required but is empty")
        elif archetype not in SUPPORTED_ARCHETYPES:
            errors.append(
                f"Unsupported archetype '{archetype}'. "
                f"Supported: {sorted(SUPPORTED_ARCHETYPES)}"
            )

        # Symbol scope check
        if not bot_spec.symbol_scope:
            errors.append("BotSpec.symbol_scope must have at least one symbol")
        else:
            symbol = bot_spec.symbol_scope[0]
            if not re.match(r"^[A-Z]{3}(?:USD|EUR|GBP|JPY|CHF|CAD|AUD|NZD)$", symbol):
                errors.append(
                    f"Invalid symbol '{symbol}'. Expected format: EURUSD, GBPUSD, etc."
                )

        # Feature registry validation (if available)
        if self._registry is not None:
            for fid in bot_spec.features:
                if self._registry.get(fid) is None:
                    errors.append(f"Feature '{fid}' not found in FeatureRegistry")

        return len(errors) == 0, errors

    # -------------------------------------------------------------------------
    # Archetype-specific generators
    # -------------------------------------------------------------------------

    def _generate_orb(
        self,
        symbol: str,
        sessions: List[str],
        features: List[str],
    ) -> str:
        """Generate Opening Range Breakout strategy code."""
        session_comment = self._session_comment(sessions)
        range_bars = self._detect_range_bars(features)

        code = f'''"""
Generated ORB Strategy — symbol={symbol}
{session_comment}
"""
# -------------------------------------------------------------------------
# Imports — MQL5-style functions provided by PythonStrategyTester
# -------------------------------------------------------------------------
# (tester object supplies: iClose, iHigh, iLow, iOpen, buy, sell, etc.)
# -------------------------------------------------------------------------

RANGE_BARS = {range_bars}  # Number of bars defining the opening range
SYMBOL = "{symbol}"

# State variables (persist across bars within a single backtest run)
_range_high = None
_range_low = None
_in_range = True        # Once out of range, stays False
_position = 0           # 1 = long, -1 = short, 0 = flat


def on_bar(tester):
    global _range_high, _range_low, _in_range, _position

    bar_idx = tester.current_bar
    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    high = tester.iHigh(SYMBOL, tester.timeframe, 0)
    low = tester.iLow(SYMBOL, tester.timeframe, 0)

    # ── Build opening range ──────────────────────────────────────────────
    if _range_high is None:
        # Initialise on the very first bar
        _range_high = high
        _range_low = low

    if bar_idx < RANGE_BARS:
        # Extend the range as bars come in
        _range_high = max(_range_high, high)
        _range_low = min(_range_low, low)

    # ── ORB signal logic ─────────────────────────────────────────────────
    if bar_idx >= RANGE_BARS and _in_range:
        # Check for breakout
        if close > _range_high and _position == 0:
            tester.buy(SYMBOL, 0.01)
            _position = 1
        elif close < _range_low and _position == 0:
            tester.sell(SYMBOL, 0.01)
            _position = -1

        # Once price leaves the range, stay out (do not re-enter range)
        if close > _range_high or close < _range_low:
            _in_range = False

    # ── Session end: flat all positions ────────────────────────────────
    # (FullBacktestPipeline handles end-of-day flattening separately)
'''
        return code

    def _generate_scalper(
        self,
        symbol: str,
        sessions: List[str],
        features: List[str],
    ) -> str:
        """Generate VWAP mean-reversion scalper strategy code."""
        session_comment = self._session_comment(sessions)
        has_vwap = any("vwap" in f.lower() for f in features)
        has_rsi = any("rsi" in f.lower() for f in features)
        has_atr = any("atr" in f.lower() for f in features)

        code = f'''"""
Generated Scalper Strategy — symbol={symbol}
{session_comment}
VWAP mean-reversion with RSI confirmation.
"""
# -------------------------------------------------------------------------

SYMBOL = "{symbol}"
POSITION_SIZE = 0.01   # default lot size

_position = 0           # 1 = long, -1 = short, 0 = flat
_vwap_sum = 0.0
_vwap_count = 0
_rsi_sum = 0.0
_rsi_count = 0


def _compute_vwap(tester, bars=20):
    """Running VWAP over the last `bars` closes."""
    total = 0.0
    count = 0
    for i in range(bars):
        c = tester.iClose(SYMBOL, tester.timeframe, i)
        if c is not None:
            total += c
            count += 1
    return total / count if count > 0 else None


def _compute_rsi(tester, period=14):
    """Simplified RSI using recent gains vs losses."""
    if period < 1:
        return 50.0
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        c0 = tester.iClose(SYMBOL, tester.timeframe, i)
        c1 = tester.iClose(SYMBOL, tester.timeframe, i + 1)
        if c0 is None or c1 is None:
            continue
        delta = c0 - c1
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    if gains + losses == 0:
        return 50.0
    rs = gains / losses if losses > 0 else 999
    return 100.0 - (100.0 / (1.0 + rs))


def on_bar(tester):
    global _position, _vwap_sum, _vwap_count

    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    if close is None:
        return

    vwap = _compute_vwap(tester, bars=20)
    rsi = _compute_rsi(tester, period=14)

    # ── Entry logic: mean reversion to VWAP ────────────────────────────
    if _position == 0:
        if vwap is not None and rsi is not None:
            # LONG when price below VWAP and RSI approaching oversold
            if close < vwap and rsi < 40:
                tester.buy(SYMBOL, POSITION_SIZE)
                _position = 1
            # SHORT when price above VWAP and RSI approaching overbought
            elif close > vwap and rsi > 60:
                tester.sell(SYMBOL, POSITION_SIZE)
                _position = -1

    # ── Exit logic ─────────────────────────────────────────────────────
    elif _position == 1:
        # Take profit: price reverted back to or above VWAP
        if vwap is not None and close >= vwap:
            tester.sell(SYMBOL, POSITION_SIZE)
            _position = 0
    elif _position == -1:
        # Take profit: price reverted back to or below VWAP
        if vwap is not None and close <= vwap:
            tester.buy(SYMBOL, POSITION_SIZE)
            _position = 0
'''
        return code

    def _generate_breakout_scalper(
        self,
        symbol: str,
        sessions: List[str],
        features: List[str],
    ) -> str:
        """Generate combined ORB + scalper strategy code."""
        session_comment = self._session_comment(sessions)
        range_bars = self._detect_range_bars(features)

        code = f'''"""
Generated Breakout-Scalper Strategy — symbol={symbol}
{session_comment}
Combines ORB breakout detection with VWAP/RSI scalper confirmations.
"""
# -------------------------------------------------------------------------

SYMBOL = "{symbol}"
RANGE_BARS = {range_bars}

_range_high = None
_range_low = None
_in_range = True
_position = 0
_session_active = False


def _compute_vwap(tester, bars=20):
    total = 0.0
    count = 0
    for i in range(bars):
        c = tester.iClose(SYMBOL, tester.timeframe, i)
        if c is not None:
            total += c
            count += 1
    return total / count if count > 0 else None


def _compute_rsi(tester, period=14):
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        c0 = tester.iClose(SYMBOL, tester.timeframe, i)
        c1 = tester.iClose(SYMBOL, tester.timeframe, i + 1)
        if c0 is None or c1 is None:
            continue
        delta = c0 - c1
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    if gains + losses == 0:
        return 50.0
    rs = gains / losses if losses > 0 else 999
    return 100.0 - (100.0 / (1.0 + rs))


def on_bar(tester):
    global _range_high, _range_low, _in_range, _position, _session_active

    bar_idx = tester.current_bar
    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    high = tester.iHigh(SYMBOL, tester.timeframe, 0)
    low = tester.iLow(SYMBOL, tester.timeframe, 0)

    # ── Build ORB range ──────────────────────────────────────────────────
    if _range_high is None:
        _range_high = high
        _range_low = low

    if bar_idx < RANGE_BARS:
        _range_high = max(_range_high, high)
        _range_low = min(_range_low, low)

    vwap = _compute_vwap(tester, bars=20)
    rsi = _compute_rsi(tester, period=14)

    # ── Breakout + scalper confirmation ──────────────────────────────────
    if bar_idx >= RANGE_BARS and _position == 0 and _in_range:
        if close > _range_high and vwap is not None and rsi is not None:
            # Bullish breakout: price above VWAP confirms momentum
            if close > vwap:
                tester.buy(SYMBOL, 0.01)
                _position = 1
        elif close < _range_low and vwap is not None and rsi is not None:
            # Bearish breakout: price below VWAP confirms momentum
            if close < vwap:
                tester.sell(SYMBOL, 0.01)
                _position = -1

        if close > _range_high or close < _range_low:
            _in_range = False

    # ── Exit on mean reversion signal ───────────────────────────────────
    elif _position == 1:
        if rsi is not None and rsi > 65:
            tester.sell(SYMBOL, 0.01)
            _position = 0
    elif _position == -1:
        if rsi is not None and rsi < 35:
            tester.buy(SYMBOL, 0.01)
            _position = 0
'''
        return code

    def _generate_pullback_scalper(
        self,
        symbol: str,
        sessions: List[str],
        features: List[str],
    ) -> str:
        """Generate trend-following pullback scalper strategy code."""
        session_comment = self._session_comment(sessions)

        code = f'''"""
Generated Pullback-Scalper Strategy — symbol={symbol}
{session_comment}
Trend-following with pullback entries using EMA + RSI confirmation.
"""
# -------------------------------------------------------------------------

SYMBOL = "{symbol}"
POSITION_SIZE = 0.01

_position = 0
_fast_ema = None
_slow_ema = None


def _ema(values, period, prev_ema=None):
    if not values or len(values) == 0:
        return prev_ema
    alpha = 2.0 / (period + 1)
    if prev_ema is None:
        return values[0]
    return alpha * values[0] + (1 - alpha) * prev_ema


def _compute_ema_pair(tester, fast=8, slow=21):
    fast_vals = []
    slow_vals = []
    for i in range(slow):
        c = tester.iClose(SYMBOL, tester.timeframe, i)
        if c is not None:
            fast_vals.append(c)
            slow_vals.append(c)
    fast_vals = fast_vals[:fast]
    return (
        _ema(fast_vals[::-1], fast),
        _ema(slow_vals[::-1], slow),
    )


def _compute_rsi(tester, period=14):
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        c0 = tester.iClose(SYMBOL, tester.timeframe, i)
        c1 = tester.iClose(SYMBOL, tester.timeframe, i + 1)
        if c0 is None or c1 is None:
            continue
        delta = c0 - c1
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    if gains + losses == 0:
        return 50.0
    rs = gains / losses if losses > 0 else 999
    return 100.0 - (100.0 / (1.0 + rs))


def on_bar(tester):
    global _position, _fast_ema, _slow_ema

    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    if close is None:
        return

    fast_ema, slow_ema = _compute_ema_pair(tester, fast=8, slow=21)
    rsi = _compute_rsi(tester, period=14)

    _fast_ema = fast_ema
    _slow_ema = slow_ema

    trend = 0  # 1 = uptrend, -1 = downtrend, 0 = unclear
    if fast_ema is not None and slow_ema is not None:
        if fast_ema > slow_ema:
            trend = 1
        elif fast_ema < slow_ema:
            trend = -1

    # ── LONG: uptrend + pullback to EMA with RSI confirmation ───────────
    if _position == 0 and trend == 1:
        if rsi is not None and rsi < 45 and fast_ema is not None:
            # Price pulled back to fast EMA
            if close <= fast_ema * 1.001:
                tester.buy(SYMBOL, POSITION_SIZE)
                _position = 1

    # ── SHORT: downtrend + pullback to EMA with RSI confirmation ─────────
    elif _position == 0 and trend == -1:
        if rsi is not None and rsi > 55 and fast_ema is not None:
            if close >= fast_ema * 0.999:
                tester.sell(SYMBOL, POSITION_SIZE)
                _position = -1

    # ── Exit on trend reversal ──────────────────────────────────────────
    elif _position == 1 and trend == -1:
        tester.sell(SYMBOL, POSITION_SIZE)
        _position = 0
    elif _position == -1 and trend == 1:
        tester.buy(SYMBOL, POSITION_SIZE)
        _position = 0
'''
        return code

    def _generate_mean_reversion(
        self,
        symbol: str,
        sessions: List[str],
        features: List[str],
    ) -> str:
        """Generate VWAP + RSI mean reversion strategy code."""
        session_comment = self._session_comment(sessions)

        code = f'''"""
Generated Mean-Reversion Strategy — symbol={symbol}
{session_comment}
VWAP mean-reversion with RSI overbought/oversold signals.
"""
# -------------------------------------------------------------------------

SYMBOL = "{symbol}"
POSITION_SIZE = 0.01

_position = 0


def _compute_vwap(tester, bars=20):
    total = 0.0
    count = 0
    for i in range(bars):
        c = tester.iClose(SYMBOL, tester.timeframe, i)
        if c is not None:
            total += c
            count += 1
    return total / count if count > 0 else None


def _compute_rsi(tester, period=14):
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        c0 = tester.iClose(SYMBOL, tester.timeframe, i)
        c1 = tester.iClose(SYMBOL, tester.timeframe, i + 1)
        if c0 is None or c1 is None:
            continue
        delta = c0 - c1
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    if gains + losses == 0:
        return 50.0
    rs = gains / losses if losses > 0 else 999
    return 100.0 - (100.0 / (1.0 + rs))


def on_bar(tester):
    global _position

    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    if close is None:
        return

    vwap = _compute_vwap(tester, bars=20)
    rsi = _compute_rsi(tester, period=14)

    # ── Mean reversion signals ───────────────────────────────────────────
    if _position == 0:
        if vwap is not None and rsi is not None:
            # LONG: price well below VWAP + RSI oversold
            if close < vwap * 0.998 and rsi < 35:
                tester.buy(SYMBOL, POSITION_SIZE)
                _position = 1
            # SHORT: price well above VWAP + RSI overbought
            elif close > vwap * 1.002 and rsi > 65:
                tester.sell(SYMBOL, POSITION_SIZE)
                _position = -1

    # ── Exit when mean reversion completes ───────────────────────────────
    elif _position == 1:
        if vwap is not None and close >= vwap:
            tester.sell(SYMBOL, POSITION_SIZE)
            _position = 0
        elif rsi is not None and rsi > 75:
            # Aggressive exit on overbought
            tester.sell(SYMBOL, POSITION_SIZE)
            _position = 0
    elif _position == -1:
        if vwap is not None and close <= vwap:
            tester.buy(SYMBOL, POSITION_SIZE)
            _position = 0
        elif rsi is not None and rsi < 25:
            tester.buy(SYMBOL, POSITION_SIZE)
            _position = 0
'''
        return code

    def _generate_unsupported(self, symbol: str, archetype: str) -> str:
        """Generate a placeholder for unsupported archetypes."""
        supported = sorted(SUPPORTED_ARCHETYPES)
        code = f'''"""
Generated Placeholder Strategy — symbol={symbol}
ARCHETYPE NOT YET SUPPORTED: {archetype!r}

Supported archetypes: {supported}

This is a minimal placeholder. Replace the on_bar function below
with the appropriate strategy logic for this archetype.
"""
# -------------------------------------------------------------------------

SYMBOL = "{symbol}"

_position = 0


def on_bar(tester):
    global _position

    close = tester.iClose(SYMBOL, tester.timeframe, 0)
    if close is None:
        return

    # NOTE: Archetype "{archetype}" is not yet supported by StrategyCodeGenerator.
    # Add your strategy logic here.
    pass
'''
        return code

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _session_comment(self, sessions: List[str]) -> str:
        """Build a multi-line comment listing active sessions."""
        if not sessions:
            return "# Sessions: (none specified)"
        lines = [f"# Sessions: {sessions[0]}"]
        for s in sessions[1:]:
            comment = _SESSION_COMMENTS.get(s, s)
            lines.append(f"#   - {s} ({comment})")
        return "\n".join(lines)

    def _detect_range_bars(self, features: List[str]) -> int:
        """Infer RANGE_BARS from features if a range_minutes hint is present."""
        for f in features:
            # e.g. "orb/range_minutes_10" or "orb_10" etc.
            m = re.search(r"range(?:_?minutes)?[_\-]?(\d+)", f.lower())
            if m:
                return int(m.group(1))
        return 5  # sensible ORB default

    def _inject_feature_indicators(
        self,
        code: str,
        bot_spec: BotSpec,
        registry: FeatureRegistry,
    ) -> str:
        """
        Enrich generated code with feature-specific indicator stubs.

        Looks up each feature_id in the registry and appends a comment block
        documenting what the indicator computes. Full indicator computation is
        left to the PythonStrategyTester data pipeline.
        """
        lines = [code.rstrip()]

        lines.append("\n# -------------------------------------------------------------------------")
        lines.append("# Feature indicators (resolved from FeatureRegistry)")
        lines.append("# -------------------------------------------------------------------------")

        for fid in bot_spec.features:
            feature = registry.get(fid)
            if feature is None:
                lines.append(f"# [UNREGISTERED] {fid}")
                continue

            lines.append(f"# Feature: {fid}")
            lines.append(f"#   Source  : {feature.source}")
            lines.append(f"#   Quality : {feature.quality_class}")

            # Document required inputs
            inputs = sorted(feature.required_inputs)
            if inputs:
                lines.append(f"#   Inputs  : {', '.join(inputs)}")

            # Document output keys
            outputs = sorted(feature.output_keys)
            if outputs:
                lines.append(f"#   Outputs : {', '.join(outputs)}")

            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _first_or_default(values: List[str], default: str) -> str:
    """Return the first non-empty value or the default."""
    return next((v for v in values if v), default)


__all__ = ["StrategyCodeGenerator"]
