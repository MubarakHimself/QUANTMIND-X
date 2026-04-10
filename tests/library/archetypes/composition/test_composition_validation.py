"""
QuantMindLib V1 — Packet 6B Self-Verification Tests

Tests ValidationResult, RequirementResolver, and CompositionValidator.
20 tests minimum (5+ per class).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Set up the import path BEFORE any library imports
# The worktree root is the cwd when running tests directly.
# Adding '.' to sys.path enables `from src.library...` imports.
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
# Import all library modules upfront so they are cached
# ---------------------------------------------------------------------------

from src.library.archetypes.composition.validation import ValidationResult
from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.composition.validator import CompositionValidator
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.features.registry import FeatureRegistry
from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.domain.bot_spec import BotSpec


# ---------------------------------------------------------------------------
# ValidationResult Tests (8 tests)
# ---------------------------------------------------------------------------

section("ValidationResult")

# V1: Default construction is valid and clean
vr1 = ValidationResult()
record(vr1.valid is True, "V1: Default constructed result is valid")
record(vr1.is_clean is True, "V1: Default constructed result is clean")
record(vr1.errors == [], "V1: Default errors is empty list")
record(vr1.warnings == [], "V1: Default warnings is empty list")
record(vr1.suggestions == [], "V1: Default suggestions is empty list")

# V2: Errors make result invalid
vr2 = ValidationResult(valid=False, errors=["FeatureNotFound: foo"])
record(vr2.valid is False, "V2: Result with errors is invalid")
record(vr2.is_clean is False, "V2: Result with errors is not clean")

# V3: Warnings keep valid=True but is_clean=False
vr3 = ValidationResult(warnings=["SessionOutsideArchetype: london"])
record(vr3.valid is True, "V3: Result with warnings only is still valid")
record(vr3.is_clean is False, "V3: Result with warnings is not clean")

# V4: Suggestions do not affect valid or clean
vr4 = ValidationResult(suggestions=["Consider optional: bar"])
record(vr4.valid is True, "V4: Result with suggestions only is valid")
record(vr4.is_clean is True, "V4: Result with suggestions only is clean")

# V5: merge() — two valid results stay valid
a = ValidationResult(valid=True, warnings=["warn1"], suggestions=["sug1"])
b = ValidationResult(valid=True, warnings=["warn2"], suggestions=["sug2"])
merged = a.merge(b)
record(merged.valid is True, "V5: Merge two valid results stays valid")
record(merged.warnings == ["warn1", "warn2"], "V5: merge() accumulates warnings")
record(merged.suggestions == ["sug1", "sug2"], "V5: merge() accumulates suggestions")
record(merged.errors == [], "V5: merge() no errors when both valid")

# V6: merge() — invalid result contaminates merged result
a2 = ValidationResult(valid=True, errors=["e1"])
b2 = ValidationResult(valid=False, errors=["e2"])
merged2 = a2.merge(b2)
record(merged2.valid is False, "V6: Merge with one invalid makes result invalid")
record(merged2.errors == ["e1", "e2"], "V6: merge() accumulates errors from both")

# V7: merge() — both invalid, errors accumulate
a3 = ValidationResult(valid=False, errors=["errA"], warnings=["warnA"])
b3 = ValidationResult(valid=False, errors=["errB"], warnings=["warnB"])
merged3 = a3.merge(b3)
record(len(merged3.errors) == 2, "V7: Both invalid: errors accumulate")
record(len(merged3.warnings) == 2, "V7: Both invalid: warnings accumulate")
record(merged3.valid is False, "V7: Both invalid: final is invalid")

# V8: merge() — suggestions accumulate across all states
a4 = ValidationResult(suggestions=["a"])
b4 = ValidationResult(suggestions=["b"])
merged4 = a4.merge(b4)
record(len(merged4.suggestions) == 2, "V8: merge() accumulates suggestions")


# ---------------------------------------------------------------------------
# RequirementResolver Tests (7 tests)
# ---------------------------------------------------------------------------

section("RequirementResolver")


class DummyFeature(FeatureModule):
    """Minimal feature for testing."""

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(feature_id="test/foo", quality_class="native_supported")

    @property
    def required_inputs(self) -> set:
        return {"close_prices"}

    @property
    def output_keys(self) -> set:
        return {"foo_out"}

    def compute(self, inputs):
        pass


def make_registry(*feature_ids: str) -> FeatureRegistry:
    """Create a FeatureRegistry with named dummy features."""
    reg = FeatureRegistry()
    for fid in feature_ids:
        f = DummyFeature()
        f.feature_id = fid
        object.__setattr__(f, "_feature_id", fid)
        reg.register(f, family="test")
        cap = CapabilitySpec(
            module_id=fid,
            provides=[f"{fid}_out"],
            requires=["close_prices"],
            optional=[],
            compatibility=["scalping", "orb"],
            sync_mode=True,
        )
        reg._capability_specs[fid] = cap
    return reg


# R1: Resolver accepts FeatureRegistry
reg1 = make_registry("a/1", "b/2")
resolver1 = RequirementResolver(reg1)
record(True, "R1: RequirementResolver instantiates with FeatureRegistry")

# R2: resolve() — all features found = valid
result = resolver1.resolve(["a/1", "b/2"])
record(result.valid is True, "R2: resolve() with all features present is valid")
record(result.errors == [], "R2: resolve() has no errors for existing features")

# R3: resolve() — missing feature produces error
result = resolver1.resolve(["a/1", "nonexistent/42"])
record(result.valid is False, "R3: resolve() with missing feature is invalid")
record(any("FeatureNotFound" in e for e in result.errors), "R3: Missing feature produces FeatureNotFound error")

# R4: resolve() — all features missing
result = resolver1.resolve(["z/1", "z/2"])
record(result.valid is False, "R4: All missing features = invalid")
record(len(result.errors) == 2, "R4: Both missing features produce errors")

# R5: check_missing_inputs() — no missing when inputs available
reg5 = make_registry("a/1")
resolver5 = RequirementResolver(reg5)
missing = resolver5.check_missing_inputs(["a/1"], available_inputs={"close_prices"})
record(missing == [], "R5: No missing inputs when close_prices available")

# R6: check_missing_inputs() — missing when inputs not available
missing = resolver5.check_missing_inputs(["a/1"], available_inputs=set())
record(len(missing) == 1, "R6: Missing input detected when not in available_inputs")
record("a/1" in missing[0] and "close_prices" in missing[0], "R6: Error message contains feature_id and missing input")

# R7: resolve() — valid result with present features
result = resolver1.resolve(["a/1"])
record(result.valid is True, "R7: Valid result with existing feature")


# ---------------------------------------------------------------------------
# CompositionValidator Tests (10 tests)
# ---------------------------------------------------------------------------

section("CompositionValidator")


def make_arch_registry() -> ArchetypeRegistry:
    """Create a registry with two archetypes for testing."""
    reg = ArchetypeRegistry()

    # Scalping archetype
    scalping = ArchetypeSpec(
        archetype_id="scalping",
        base_type="scalping",
        description="High-frequency scalping strategy",
        v1_priority=True,
        quality_class="native_supported",
        required_features=["indicators/rsi", "microstructure/spread"],
        optional_features=["indicators/macd", "volume/rvol"],
        excluded_features=["news/blackout", "session/detector"],
        session_tags=["london", "nyse_morning"],
        constraints=[],
        confirmations=["risk_limit_set", "spread_confirmed"],
        execution_profile="scalping_hft",
        max_trades_per_hour=120,
        default_allowed_areas=["indicators", "execution"],
        default_locked_components=["news"],
    )
    reg.register(scalping)

    # ORB archetype
    orb = ArchetypeSpec(
        archetype_id="orb",
        base_type="orb",
        description="Opening Range Breakout",
        v1_priority=True,
        quality_class="native_supported",
        required_features=["indicators/rsi", "session/detector"],
        optional_features=["volume/rvol"],
        excluded_features=["microstructure/aggression"],
        session_tags=["london", "nyse_morning", "asia"],
        constraints=[],
        confirmations=["range_defined"],
        execution_profile="orb_standard",
        max_trades_per_hour=30,
        default_allowed_areas=["indicators"],
        default_locked_components=["risk_limit"],
    )
    reg.register(orb)

    return reg


def make_feat_registry() -> FeatureRegistry:
    """Create a FeatureRegistry with test features."""
    reg = FeatureRegistry()

    for fid, provides, requires in [
        ("indicators/rsi_14", ["rsi_14"], ["close_prices"]),
        ("indicators/rsi_28", ["rsi_28"], ["close_prices"]),
        ("microstructure/spread", ["spread_score"], ["bid", "ask"]),
        ("session/detector", ["session_active"], ["close_prices"]),
        ("volume/rvol", ["rvol_ratio"], ["volume"]),
        ("indicators/macd", ["macd_line", "signal_line"], ["close_prices"]),
    ]:
        f = DummyFeature()
        f.feature_id = fid
        object.__setattr__(f, "_feature_id", fid)
        reg.register(f, family="test")
        cap = CapabilitySpec(
            module_id=fid,
            provides=provides,
            requires=requires,
            optional=[],
            compatibility=["scalping", "orb"],
            sync_mode=True,
        )
        reg._capability_specs[fid] = cap

    return reg


arch_reg = make_arch_registry()
feat_reg = make_feat_registry()
resolver = RequirementResolver(feat_reg)
validator = CompositionValidator(arch_reg, feat_reg, resolver)

# CV1: Full validation — valid bot passes
bot = BotSpec(
    id="bot-001",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set", "spread_confirmed"],
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(bot)
record(result.valid is True, "CV1: Valid bot spec passes full validation")
record(len(result.errors) == 0, "CV1: No errors for valid bot spec")

# CV2: Unknown archetype fails immediately
unknown_bot = BotSpec(
    id="bot-002",
    archetype="nonexistent_archetype",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=[],
    confirmations=[],
    execution_profile="standard",
)
result = validator.validate_bot_spec(unknown_bot)
record(result.valid is False, "CV2: Unknown archetype makes bot invalid")
record(any("ArchetypeNotFound" in e for e in result.errors), "CV2: Error is ArchetypeNotFound")

# CV3: Missing required feature
missing_req_bot = BotSpec(
    id="bot-003",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14"],  # missing microstructure/spread
    confirmations=["risk_limit_set"],
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(missing_req_bot)
record(result.valid is False, "CV3: Missing required feature is invalid")
record(any("RequiredFeatureMissing" in e for e in result.errors), "CV3: Error is RequiredFeatureMissing")

# CV4: Excluded feature present
excluded_bot = BotSpec(
    id="bot-004",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread", "session/detector"],
    confirmations=["risk_limit_set", "spread_confirmed"],
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(excluded_bot)
record(result.valid is False, "CV4: Excluded feature present makes bot invalid")
record(any("ExcludedFeaturePresent" in e for e in result.errors), "CV4: Error is ExcludedFeaturePresent")

# CV5: Session outside archetype scope — warning only (valid)
wrong_session_bot = BotSpec(
    id="bot-005",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["asia"],  # scalping only allows london, nyse_morning
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set", "spread_confirmed"],
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(wrong_session_bot)
record(result.valid is True, "CV5: Wrong session is a warning, not error")
record(any("SessionOutsideArchetype" in w for w in result.warnings), "CV5: Warning is SessionOutsideArchetype")

# CV6: Confirmation missing — warning only (valid)
no_conf_bot = BotSpec(
    id="bot-006",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set"],  # missing spread_confirmed
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(no_conf_bot)
record(result.valid is True, "CV6: Missing confirmation is warning, not error")
record(any("ConfirmationMissing" in w for w in result.warnings), "CV6: Warning is ConfirmationMissing")

# CV7: validate_feature_set() — shorthand check (valid)
result = validator.validate_feature_set(
    ["indicators/rsi_14", "microstructure/spread"],
    "scalping",
)
record(result.valid is True, "CV7: feature_set validation with valid features passes")
record(result.errors == [], "CV7: No errors for valid feature set")

# CV8: validate_feature_set() — missing required feature
result = validator.validate_feature_set(
    ["indicators/rsi_14"],  # missing microstructure/spread
    "scalping",
)
record(result.valid is False, "CV8: feature_set missing required is invalid")
record(any("RequiredFeatureMissing" in e for e in result.errors), "CV8: Error is RequiredFeatureMissing")

# CV9: validate_feature_set() — unknown archetype
result = validator.validate_feature_set(
    ["indicators/rsi_14"],
    "unknown_archetype",
)
record(result.valid is False, "CV9: unknown archetype in feature_set is invalid")

# CV10: _find_feature() — exact match
found = validator._find_feature("indicators/rsi_14", ["indicators/rsi_14", "indicators/rsi_28"])
record(found == "indicators/rsi_14", "CV10: exact match returns the feature_id")

# CV11: _find_feature() — prefix match (pattern is base, list has suffix)
found = validator._find_feature("indicators/rsi", ["indicators/rsi_14", "indicators/rsi_28"])
record(found == "indicators/rsi_14", "CV11: prefix match (pattern is base) finds first match")

# CV12: _find_feature() — no match returns None
found = validator._find_feature("indicators/macd", ["indicators/rsi_14"])
record(found is None, "CV12: no match returns None")

# CV13: Suggestions for optional features not in set
bot13 = BotSpec(
    id="bot-013",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set", "spread_confirmed"],
    execution_profile="scalping_hft",
)
result = validator.validate_bot_spec(bot13)
record(len(result.suggestions) >= 1, "CV13: optional features produce suggestions")
record(any("indicators/macd" in s or "volume/rvol" in s for s in result.suggestions), "CV13: suggestions include optional features")

# CV14: Excluded feature in validate_feature_set
result = validator.validate_feature_set(
    ["indicators/rsi_14", "session/detector"],  # session/detector excluded from scalping
    "scalping",
)
record(result.valid is False, "CV14: excluded feature in set makes validation invalid")
record(any("ExcludedFeaturePresent" in e for e in result.errors), "CV14: error is ExcludedFeaturePresent")

# CV15: ORB archetype validation
orb_bot = BotSpec(
    id="bot-015",
    archetype="orb",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "session/detector"],
    confirmations=["range_defined"],
    execution_profile="orb_standard",
)
result = validator.validate_bot_spec(orb_bot)
record(result.valid is True, "CV15: ORB archetype bot with correct features is valid")
record(result.errors == [], "CV15: No errors for ORB bot")

# CV16: is_clean with warnings but no errors
r = ValidationResult(valid=True, warnings=["w"])
record(r.is_clean is False, "CV16: result with warnings is not clean")

# CV17: is_clean with errors
r = ValidationResult(valid=True, errors=["e"])
record(r.is_clean is False, "CV17: result with errors is not clean")

# CV18: is_clean with errors and warnings
r = ValidationResult(valid=False, errors=["e"], warnings=["w"])
record(r.is_clean is False, "CV18: result with errors and warnings is not clean")

# CV19: is_clean with suggestions only
r = ValidationResult(valid=True, suggestions=["s"])
record(r.is_clean is True, "CV19: result with suggestions only is clean")

# CV20: Full pipeline — merge multiple validation results
vr_a = ValidationResult(valid=True, warnings=["warn_a"])
vr_b = ValidationResult(valid=False, errors=["err_b"])
vr_c = ValidationResult(valid=True, suggestions=["sug_c"])
merged = vr_a.merge(vr_b).merge(vr_c)
record(merged.valid is False, "CV20: merging multiple results ends invalid when any invalid")
record(len(merged.errors) == 1, "CV20: merged errors accumulate correctly")
record(len(merged.warnings) == 1, "CV20: merged warnings accumulate correctly")
record(len(merged.suggestions) == 1, "CV20: merged suggestions accumulate correctly")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'=' * 60}")
print(f"  RESULTS: {pass_count} passed, {fail_count} failed")
print(f"{'=' * 60}")

if fail_count > 0:
    print(f"\nSOME TESTS FAILED — review output above.")
    sys.exit(1)
else:
    print(f"\nALL {pass_count} TESTS PASSED")
    sys.exit(0)