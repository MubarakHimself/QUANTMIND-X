"""
QuantMindLib V1 — Packet 6D Self-Verification Tests

Tests OpeningRangeBreakout archetype: ORB_ARCHETYPE, OpeningRangeBreakout,
get_default_registry, and all ORB-specific behaviors.
18 tests covering: ArchetypeSpec frozen, ORB required/optional/excluded features,
OpeningRangeBreakout methods, context-aware suggestions, default registry singleton,
and ORB-specific validation.
"""

from __future__ import annotations

import sys
import os

_worktree_root = os.getcwd()
if _worktree_root not in sys.path:
    sys.path.insert(0, _worktree_root)

# ---------------------------------------------------------------------------
# Test Utilities
# ---------------------------------------------------------------------------

pass_count = 0
fail_count = 0


def record(passed: bool, description: str) -> None:
    global pass_count, fail_count
    if passed:
        pass_count += 1
        print(f"  PASS  {description}")
    else:
        fail_count += 1
        print(f"  FAIL  {description}")


def section(name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.library.archetypes.orb import (
    ORB_ARCHETYPE,
    OpeningRangeBreakout,
    get_default_registry,
)
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.core.domain.bot_spec import BotSpec, BotMutationProfile


# ---------------------------------------------------------------------------
# AF1-AF6: ArchetypeSpec frozen
# ---------------------------------------------------------------------------

section("ArchetypeSpec frozen")

# AF1: ORB_ARCHETYPE is an ArchetypeSpec instance
record(
    isinstance(ORB_ARCHETYPE, ArchetypeSpec),
    "AF1: ORB_ARCHETYPE is an ArchetypeSpec instance",
)

# AF2: ORB_ARCHETYPE is frozen
try:
    ORB_ARCHETYPE.archetype_id = "renamed"
    record(False, "AF2: ORB_ARCHETYPE is frozen (setattr raises)")
except Exception:
    record(True, "AF2: ORB_ARCHETYPE is frozen (setattr raises)")

# AF3: archetype_id is correct
record(
    ORB_ARCHETYPE.archetype_id == "opening_range_breakout",
    "AF3: archetype_id is 'opening_range_breakout'",
)

# AF4: base_type is breakout
record(
    ORB_ARCHETYPE.base_type == "breakout",
    "AF4: base_type is 'breakout'",
)

# AF5: v1_priority is True
record(
    ORB_ARCHETYPE.v1_priority is True,
    "AF5: v1_priority is True",
)

# AF6: quality_class is native_supported
record(
    ORB_ARCHETYPE.quality_class == "native_supported",
    "AF6: quality_class is 'native_supported'",
)


# ---------------------------------------------------------------------------
# RF1-RF4: ORB required_features
# ---------------------------------------------------------------------------

section("ORB required_features")

# RF1: All required features are present
required_ids = {"indicators/atr_14", "session/detector", "session/blackout"}
record(
    required_ids.issubset(set(ORB_ARCHETYPE.required_features)),
    "RF1: All 3 required features (atr_14, detector, blackout) present",
)

# RF2: is_feature_required() returns True for required features
for fid in required_ids:
    record(
        ORB_ARCHETYPE.is_feature_required(fid) is True,
        f"RF2: is_feature_required('{fid}') is True",
    )

# RF3: is_feature_required() returns False for non-required
record(
    ORB_ARCHETYPE.is_feature_required("indicators/rsi_14") is False,
    "RF3: is_feature_required('rsi_14') is False",
)

# RF4: No excluded feature is in required
for fid in ORB_ARCHETYPE.excluded_features:
    record(
        fid not in ORB_ARCHETYPE.required_features,
        f"RF4: excluded feature '{fid}' not in required_features",
    )


# ---------------------------------------------------------------------------
# OF1-OF3: ORB optional_features
# ---------------------------------------------------------------------------

section("ORB optional_features")

# OF1: optional_features contains expected ORB-compatible features
expected_optional = {
    "indicators/rsi_14",
    "indicators/macd",
    "indicators/vwap",
    "volume/rvol",
    "volume/mfi",
    "volume/profile",
    "microstructure/spread",
}
for fid in expected_optional:
    record(
        fid in ORB_ARCHETYPE.optional_features,
        f"OF1: optional_features contains '{fid}'",
    )

# OF2: is_compatible_with() returns True for optional features
record(
    ORB_ARCHETYPE.is_compatible_with("indicators/rsi_14") is True,
    "OF2: is_compatible_with('rsi_14') is True",
)

# OF3: is_compatible_with() returns True for required features
record(
    ORB_ARCHETYPE.is_compatible_with("indicators/atr_14") is True,
    "OF3: is_compatible_with('atr_14') is True",
)


# ---------------------------------------------------------------------------
# EF1-EF2: ORB excluded_features
# ---------------------------------------------------------------------------

section("ORB excluded_features")

# EF1: excluded_features contains bollinger
record(
    "indicators/bollinger" in ORB_ARCHETYPE.excluded_features,
    "EF1: excluded_features contains 'indicators/bollinger'",
)

# EF2: is_feature_excluded() returns True for bollinger
record(
    ORB_ARCHETYPE.is_feature_excluded("indicators/bollinger") is True,
    "EF2: is_feature_excluded('bollinger') is True",
)


# ---------------------------------------------------------------------------
# AM1-AM5: OpeningRangeBreakout methods
# ---------------------------------------------------------------------------

section("OpeningRangeBreakout methods")

orb = OpeningRangeBreakout()

# AM1: spec property returns ORB_ARCHETYPE
record(
    orb.spec == ORB_ARCHETYPE,
    "AM1: spec property returns ORB_ARCHETYPE",
)

# AM2: spec archetype_id matches
record(
    orb.spec.archetype_id == "opening_range_breakout",
    "AM2: spec archetype_id matches",
)

# AM3: capability_spec() returns a CapabilitySpec
cap = orb.capability_spec()
record(
    cap is not None,
    "AM3: capability_spec() returns non-None",
)
record(
    cap.module_id == "archetype/opening_range_breakout",
    "AM3: capability_spec module_id is archetype/opening_range_breakout",
)
record(
    set(cap.requires) == set(ORB_ARCHETYPE.required_features),
    "AM3: capability_spec requires matches required_features",
)

# AM4: validate_bot_spec() returns list (not raises)
errors = orb.validate_bot_spec(
    BotSpec(
        id="test-bot",
        archetype="opening_range_breakout",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/atr_14"],
        confirmations=["spread_ok"],
        execution_profile="orb_v1",
    )
)
record(
    isinstance(errors, list),
    "AM4: validate_bot_spec() returns a list",
)

# AM5: suggest_features() returns list (not raises)
suggestions = orb.suggest_features({"regime": "TREND_STABLE"})
record(
    isinstance(suggestions, list),
    "AM5: suggest_features() returns a list",
)


# ---------------------------------------------------------------------------
# CS1-CS4: Context-aware suggestions (4 contexts)
# ---------------------------------------------------------------------------

section("Context-aware suggestions (4 contexts)")

# CS1: VOLATILE regime -> ATR suggested
ctx_volatile = {
    "regime": "VOLATILE",
    "session": "new_york",
    "spread_pips": 1.2,
    "volume_regime": "normal",
}
sugs = orb.suggest_features(ctx_volatile)
record(
    "indicators/atr_14" in sugs,
    "CS1: VOLATILE regime -> atr_14 suggested",
)

# CS2: TREND_STABLE regime -> MACD suggested
ctx_stable = {
    "regime": "TREND_STABLE",
    "session": "tokyo",
    "spread_pips": 1.2,
    "volume_regime": "normal",
}
sugs = orb.suggest_features(ctx_stable)
record(
    "indicators/macd" in sugs,
    "CS2: TREND_STABLE regime -> macd suggested",
)

# CS3: tight spread (< 1.0) -> order flow features
ctx_tight = {
    "regime": "TREND_STABLE",
    "session": "sydney",
    "spread_pips": 0.8,
    "volume_regime": "normal",
}
sugs = orb.suggest_features(ctx_tight)
record(
    "orderflow/spread_behavior" in sugs,
    "CS3: tight spread -> spread_behavior suggested",
)
record(
    "microstructure/tob_pressure" in sugs,
    "CS3: tight spread -> tob_pressure suggested",
)

# CS4: high volume regime + london -> RVOL + profile
ctx_london_high = {
    "regime": "TREND_STABLE",
    "session": "london",
    "spread_pips": 1.0,
    "volume_regime": "high",
}
sugs = orb.suggest_features(ctx_london_high)
record(
    "volume/rvol" in sugs,
    "CS4: london high volume -> rvol suggested",
)
record(
    "volume/profile" in sugs,
    "CS4: london session -> volume/profile suggested",
)


# ---------------------------------------------------------------------------
# DR1-DR2: Default registry singleton
# ---------------------------------------------------------------------------

section("Default registry singleton")

# DR1: get_default_registry() returns an ArchetypeRegistry
reg = get_default_registry()
record(
    isinstance(reg, ArchetypeRegistry),
    "DR1: get_default_registry() returns ArchetypeRegistry",
)

# DR2: ORB is pre-registered in default registry
record(
    reg.get("opening_range_breakout") == ORB_ARCHETYPE,
    "DR2: ORB_ARCHETYPE is pre-registered in default registry",
)

# DR3: Singleton — multiple calls return the same instance
reg2 = get_default_registry()
record(
    reg is reg2,
    "DR3: get_default_registry() is a singleton (same instance returned)",
)

# DR4: Default registry also has V1 priority set
record(
    "opening_range_breakout" in reg.list_v1_priority(),
    "DR4: ORB is in V1 priority list",
)


# ---------------------------------------------------------------------------
# OV1-OV4: ORB-specific validation (valid + 3 invalid cases)
# ---------------------------------------------------------------------------

section("ORB-specific validation")

# OV1: Valid BotSpec -> no errors
valid_bot = BotSpec(
    id="orb-valid-001",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD", "GBPUSD"],
    sessions=["london", "new_york"],
    features=["indicators/atr_14", "indicators/macd", "indicators/rsi_14"],
    confirmations=["spread_ok", "sentinel_allow", "session_active", "news_clear", "atr_filter"],
    execution_profile="orb_v1",
)
errors = orb.validate_bot_spec(valid_bot)
record(
    errors == [],
    "OV1: Valid ORB BotSpec -> no errors returned",
)

# OV2: Missing momentum indicator -> error
no_momentum_bot = BotSpec(
    id="orb-no-momentum",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["session/detector", "session/blackout"],  # no RSI, MACD, or ATR
    confirmations=["spread_ok"],
    execution_profile="orb_v1",
)
errors = orb.validate_bot_spec(no_momentum_bot)
record(
    len(errors) > 0,
    "OV2: Missing momentum indicator -> returns errors",
)
record(
    any("ORBValidation" in e and "momentum" in e.lower() for e in errors),
    "OV2: Error mentions ORBValidation and momentum",
)

# OV3: Missing primary ORB session -> error
no_session_bot = BotSpec(
    id="orb-no-session",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD"],
    sessions=["asia"],  # not london, new_york, or london_ny_overlap
    features=["indicators/atr_14"],
    confirmations=["spread_ok"],
    execution_profile="orb_v1",
)
errors = orb.validate_bot_spec(no_session_bot)
record(
    len(errors) > 0,
    "OV3: Missing primary ORB session -> returns errors",
)
record(
    any("ORBValidation" in e and "session" in e.lower() for e in errors),
    "OV3: Error mentions ORBValidation and session",
)

# OV4: Missing spread_ok confirmation -> error
no_spread_bot = BotSpec(
    id="orb-no-spread",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/atr_14"],
    confirmations=["sentinel_allow"],  # spread_ok missing
    execution_profile="orb_v1",
)
errors = orb.validate_bot_spec(no_spread_bot)
record(
    len(errors) > 0,
    "OV4: Missing spread_ok confirmation -> returns errors",
)
record(
    any("spread_ok" in e for e in errors),
    "OV4: Error mentions spread_ok",
)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
print(f"  RESULTS: {pass_count} passed, {fail_count} failed")
print(f"{'=' * 60}")

if fail_count > 0:
    print(f"\nSOME TESTS FAILED -- review output above.")
    sys.exit(1)
else:
    print(f"\nALL {pass_count} TESTS PASSED")
    sys.exit(0)
