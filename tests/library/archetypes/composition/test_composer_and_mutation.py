"""
QuantMindLib V1 — Packet 6C Self-Verification Tests

Tests CompositionResult, Composer, MutationEngine, and MutationResult.
24 tests total.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from src.library.archetypes.composition.result import CompositionResult
from src.library.archetypes.composition.validation import ValidationResult
from src.library.archetypes.composition.resolver import RequirementResolver
from src.library.archetypes.composition.validator import CompositionValidator
from src.library.archetypes.composer import Composer
from src.library.archetypes.mutation.engine import MutationEngine, MutationResult
from src.library.archetypes.archetype_spec import ArchetypeSpec
from src.library.archetypes.registry import ArchetypeRegistry
from src.library.features.registry import FeatureRegistry
from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.composition.capability_spec import CapabilitySpec
from src.library.core.domain.bot_spec import BotSpec, BotMutationProfile


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


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


def make_feat_registry() -> FeatureRegistry:
    """Create a FeatureRegistry with test features."""
    reg = FeatureRegistry()
    for fid, provides, requires in [
        ("indicators/rsi_14", ["rsi_14"], ["close_prices"]),
        ("indicators/rsi", ["rsi"], ["close_prices"]),
        ("indicators/atr_14", ["atr_14"], ["close_prices"]),
        ("indicators/macd", ["macd_line"], ["close_prices"]),
        ("indicators/macd_12_26_9", ["macd_line"], ["close_prices"]),
        ("microstructure/spread", ["spread_score"], ["bid", "ask"]),
        ("session/detector", ["session_active"], ["close_prices"]),
        ("volume/rvol", ["rvol_ratio"], ["volume"]),
        ("news/blackout", ["news_blackout"], ["close_prices"]),
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


def make_arch_registry() -> ArchetypeRegistry:
    """Create a registry with test archetypes."""
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


# ---------------------------------------------------------------------------
# C1-C8: CompositionResult
# ---------------------------------------------------------------------------

section("CompositionResult")

# C1: Default construction with valid=True but bot_spec=None -> not succeeded
# Per spec: succeeded = bot_spec is not None and valid. bot_spec=None means no
# composition was produced, so succeeded=False.
cr1 = CompositionResult(validation=ValidationResult())
record(cr1.succeeded is False, "C1: bot_spec=None gives succeeded=False even with valid=True")
record(cr1.failed is True, "C1: bot_spec=None gives failed=True")
record(cr1.bot_spec is None, "C1: Default bot_spec is None")
record(len(cr1.composition_notes) == 0, "C1: Default composition_notes is empty")

# C2: Failed composition (bot_spec=None)
cr2 = CompositionResult(
    bot_spec=None,
    validation=ValidationResult(valid=False, errors=["ArchetypeNotFound: foo"]),
)
record(cr2.succeeded is False, "C2: bot_spec=None is not succeeded")
record(cr2.failed is True, "C2: bot_spec=None is failed")
record(len(cr2.composition_notes) == 0, "C2: Empty notes on failure")

# C3: composition_notes field works
cr3 = CompositionResult(
    validation=ValidationResult(),
    composition_notes=["Note 1", "Note 2"],
)
record(len(cr3.composition_notes) == 2, "C3: composition_notes stored correctly")

# C4: archetype field is Optional
cr4 = CompositionResult(validation=ValidationResult())
record(cr4.archetype is None, "C4: archetype defaults to None")

# C5: succeeded=False when bot_spec present but validation invalid
dummy_spec = ArchetypeSpec(
    archetype_id="test",
    base_type="test",
    description="Test",
    execution_profile="test",
)
dummy_bot = BotSpec(
    id="test",
    archetype="test",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14"],
    confirmations=[],
    execution_profile="test",
    mutation=BotMutationProfile(
        bot_id="test",
        allowed_mutation_areas=[],
        locked_components=[],
        lineage=[],
        parent_variant_id=None,
        generation=0,
        unsafe_areas=[],
    ),
)
cr5 = CompositionResult(
    bot_spec=dummy_bot,
    validation=ValidationResult(valid=False, errors=["Error"]),
)
record(cr5.succeeded is False, "C5: bot_spec present but invalid -> not succeeded")

# C6: succeeded=True when both bot_spec present and valid
cr6 = CompositionResult(
    bot_spec=dummy_bot,
    validation=ValidationResult(valid=True),
)
record(cr6.succeeded is True, "C6: bot_spec present and valid -> succeeded")

# C7: composition_notes on warning
cr7 = CompositionResult(
    validation=ValidationResult(warnings=["Some warning"]),
    composition_notes=["Warning note"],
)
record(cr7.validation.is_clean is False, "C7: warnings make is_clean=False")

# C8: ValidationResult attribute present
cr8 = CompositionResult(validation=ValidationResult(valid=True))
record(hasattr(cr8, "validation"), "C8: validation attribute exists")


# ---------------------------------------------------------------------------
# CM1-CM10: Composer
# ---------------------------------------------------------------------------

section("Composer")

feat_reg = make_feat_registry()
arch_reg = make_arch_registry()
composer = Composer(arch_reg, feat_reg)

# CM1: Composer instantiates with registries
record(
    composer.archetype_registry is arch_reg, "CM1: Composer stores archetype_registry"
)
record(
    composer.feature_registry is feat_reg, "CM1: Composer stores feature_registry"
)
record(
    composer.validator is not None, "CM1: Composer creates CompositionValidator"
)

# CM2: Compose with valid archetype - returns bot_spec
result = composer.compose(
    bot_id="bot-test-001",
    archetype_id="scalping",
    config={
        "symbol_scope": ["EURUSD", "GBPUSD"],
        "sessions": ["london"],
        "features": [],
        "confirmations": [],
    },
)
record(result.succeeded is True, "CM2: Valid archetype composition succeeds")
record(result.bot_spec is not None, "CM2: bot_spec is returned")
record(result.bot_spec.id == "bot-test-001", "CM2: bot_id matches")
record(result.bot_spec.archetype == "scalping", "CM2: archetype matches")
record(result.validation.valid is True, "CM2: validation is valid")
record(result.archetype is not None, "CM2: archetype is returned")

# CM3: Compose with unknown archetype - returns bot_spec=None
result = composer.compose(
    bot_id="bot-unknown",
    archetype_id="nonexistent_archetype",
    config={"symbol_scope": ["EURUSD"], "sessions": ["london"]},
)
record(result.succeeded is False, "CM3: Unknown archetype composition fails")
record(result.bot_spec is None, "CM3: bot_spec is None for unknown archetype")
record(result.archetype is None, "CM3: archetype is None for unknown archetype")
record(
    any("ArchetypeNotFound" in e for e in result.validation.errors),
    "CM3: Error contains ArchetypeNotFound",
)

# CM4: Feature list includes required + user-specified
result = composer.compose(
    bot_id="bot-feat-test",
    archetype_id="scalping",
    config={
        "symbol_scope": ["EURUSD"],
        "sessions": ["london"],
        "features": ["indicators/macd"],
    },
)
record(result.succeeded is True, "CM4: Composition with optional features succeeds")
record(
    "indicators/macd" in result.bot_spec.features,
    "CM4: User-specified features included",
)
record(
    "indicators/rsi" in result.bot_spec.features, "CM4: Required features included"
)

# CM5: Execution profile override
result = composer.compose(
    bot_id="bot-exec-test",
    archetype_id="scalping",
    config={
        "symbol_scope": ["EURUSD"],
        "sessions": ["london"],
        "execution_profile": "custom_exec_profile",
    },
)
record(
    result.bot_spec.execution_profile == "custom_exec_profile",
    "CM5: execution_profile override applied",
)

# CM6: Mutation profile attached by default
result = composer.compose(
    bot_id="bot-mutation-test",
    archetype_id="scalping",
    config={"symbol_scope": ["EURUSD"], "sessions": ["london"]},
)
record(result.bot_spec.mutation is not None, "CM6: mutation profile is attached")
record(
    result.bot_spec.mutation.bot_id == "bot-mutation-test",
    "CM6: mutation bot_id matches",
)
record(
    result.bot_spec.mutation.generation == 0, "CM6: generation defaults to 0"
)
record(
    "indicators" in result.bot_spec.mutation.allowed_mutation_areas,
    "CM6: allowed_areas from archetype defaults",
)
record(
    "news" in result.bot_spec.mutation.locked_components,
    "CM6: locked_components from archetype defaults",
)

# CM7: Mutation profile override
result = composer.compose(
    bot_id="bot-mut-override",
    archetype_id="scalping",
    config={
        "symbol_scope": ["EURUSD"],
        "sessions": ["london"],
        "mutation_profile": {
            "allowed_mutation_areas": ["custom_area"],
            "locked_components": ["another_lock"],
            "generation": 5,
        },
    },
)
record(
    "custom_area" in result.bot_spec.mutation.allowed_mutation_areas,
    "CM7: mutation_profile override applied to allowed_areas",
)
record(
    "another_lock" in result.bot_spec.mutation.locked_components,
    "CM7: mutation_profile override applied to locked_components",
)
record(
    result.bot_spec.mutation.generation == 5, "CM7: mutation_profile generation override"
)

# CM8: from_archetype_template convenience method
result = composer.from_archetype_template(
    bot_id="bot-template",
    archetype_id="orb",
    symbol_scope=["EURUSD", "USDJPY"],
    sessions=["london", "nyse_morning"],
    optional_features=["volume/rvol"],
)
record(result.succeeded is True, "CM8: from_archetype_template succeeds")
record(
    result.bot_spec.symbol_scope == ["EURUSD", "USDJPY"],
    "CM8: symbol_scope passed through",
)
record(
    result.bot_spec.sessions == ["london", "nyse_morning"],
    "CM8: sessions passed through",
)
record(
    "volume/rvol" in result.bot_spec.features, "CM8: optional_features included"
)
record(
    result.bot_spec.execution_profile == "orb_standard",
    "CM8: execution_profile from archetype defaults",
)

# CM9: Required features always included, no duplicates
result = composer.compose(
    bot_id="bot-dedup",
    archetype_id="orb",
    config={
        "symbol_scope": ["EURUSD"],
        "sessions": ["london"],
        "features": ["session/detector", "indicators/rsi"],
    },
)
record(
    result.bot_spec.features.count("session/detector") == 1,
    "CM9: No duplicate features",
)
record(
    result.bot_spec.features.count("indicators/rsi") == 1,
    "CM9: No duplicate required features",
)

# CM10: composition_notes populated on warnings
result = composer.compose(
    bot_id="bot-warn-test",
    archetype_id="orb",
    config={
        "symbol_scope": ["EURUSD"],
        "sessions": ["asia"],  # not in orb session tags - will warn
        "confirmations": [],  # not all recommended - will warn
    },
)
record(
    result.validation.valid is True, "CM10: Warnings don't make composition invalid"
)
record(
    len(result.composition_notes) > 0, "CM10: composition_notes populated for warnings"
)


# ---------------------------------------------------------------------------
# M1-M6: MutationEngine
# ---------------------------------------------------------------------------

section("MutationEngine")

mutation_engine = MutationEngine(composer, arch_reg)

# M1: Mutate without mutation profile returns error
no_mut_bot = BotSpec(
    id="bot-no-mutation",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set"],
    execution_profile="scalping_hft",
    mutation=None,
)
result = mutation_engine.mutate(no_mut_bot, {"features.optional": ["indicators/macd"]})
record(result.success is False, "M1: Mutate without mutation profile fails")
record(
    any("no mutation profile" in e for e in result.errors),
    "M1: Error message about missing mutation profile",
)
record(result.mutated_spec is None, "M1: mutated_spec is None on failure")

# M2: Mutate locked component returns error
locked_bot = BotSpec(
    id="bot-locked",
    archetype="scalping",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi_14", "microstructure/spread"],
    confirmations=["risk_limit_set"],
    execution_profile="scalping_hft",
    mutation=BotMutationProfile(
        bot_id="bot-locked",
        # Include features and confirmations to allow M4-M6 tests to pass
        allowed_mutation_areas=["indicators", "execution", "features", "confirmations"],
        locked_components=["news", "archetype"],
        lineage=[],
        parent_variant_id=None,
        generation=0,
        unsafe_areas=[],
    ),
)
result = mutation_engine.mutate(locked_bot, {"news": "some_value"})
record(result.success is False, "M2: Mutating locked component fails")
record(
    any("locked" in e.lower() for e in result.errors),
    "M2: Error mentions locked component",
)

# M3: Mutate with invalid area returns error
result = mutation_engine.mutate(locked_bot, {"invalid_area.xyz": "value"})
record(result.success is False, "M3: Mutating invalid area fails")
record(
    any("not in allowed_mutation_areas" in e for e in result.errors),
    "M3: Error mentions allowed_mutation_areas",
)

# M4: Valid mutation succeeds
result = mutation_engine.mutate(
    locked_bot, {"features.optional": ["indicators/macd_12_26_9"]}
)
record(result.success is True, "M4: Valid mutation succeeds")
record(result.mutated_spec is not None, "M4: mutated_spec is returned")
record(
    result.mutated_spec.id == "bot-locked_v1", "M4: New bot_id with generation suffix"
)
record(
    result.mutated_spec.mutation.generation == 1,
    "M4: Generation incremented",
)
record(
    result.mutated_spec.mutation.parent_variant_id == "bot-locked",
    "M4: Parent variant ID set",
)
record(
    "indicators/macd_12_26_9" in result.mutated_spec.features,
    "M4: New feature added",
)
record(
    result.mutated_spec.features == [
        "indicators/rsi_14",
        "microstructure/spread",
        "indicators/macd_12_26_9",
    ],
    "M4: All features preserved and new added",
)

# M5: Lineage chain tracked
result_v1 = mutation_engine.mutate(
    locked_bot, {"features.optional": ["indicators/macd_12_26_9"]}
)
result_v2 = mutation_engine.mutate(
    result_v1.mutated_spec, {"confirmations": ["spread_confirmed"]}
)
record(
    result_v2.mutated_spec.mutation.lineage == ["bot-locked"],
    "M5: Lineage contains parent ID",
)
record(
    result_v2.mutated_spec.mutation.parent_variant_id == "bot-locked_v1",
    "M5: Parent variant ID is v1 bot",
)
record(
    result_v2.mutated_spec.mutation.generation == 2, "M5: Generation is 2"
)
record(
    "spread_confirmed" in result_v2.mutated_spec.confirmations,
    "M5: Confirmation added in second mutation",
)

# M6: Invalid mutation fails validation of mutated spec
result = mutation_engine.mutate(
    locked_bot, {"features.optional": ["indicators/macd_12_26_9"]}
)
record(result.success is True, "M6: Valid features pass validation")
record(len(result.mutation_log) > 0, "M6: Mutation log is populated")
record(
    any("Added optional feature" in entry for entry in result.mutation_log),
    "M6: Mutation log contains feature addition",
)


# ---------------------------------------------------------------------------
# MU1-MU6: MutationEngine.can_mutate()
# ---------------------------------------------------------------------------

section("MutationEngine.can_mutate()")

# MU1: None mutation profile returns False
record(
    mutation_engine.can_mutate(no_mut_bot, "features.optional") is False,
    "MU1: None mutation profile -> can_mutate=False",
)

# MU2: Exact match in allowed returns True
record(
    mutation_engine.can_mutate(locked_bot, "indicators") is True,
    "MU2: area in allowed_areas -> can_mutate=True",
)

# MU3: In locked returns False
record(
    mutation_engine.can_mutate(locked_bot, "news") is False,
    "MU3: key in locked_components -> can_mutate=False",
)

# MU4: Nested key with allowed prefix returns True
record(
    mutation_engine.can_mutate(locked_bot, "indicators.macd") is True,
    "MU4: nested key with allowed prefix -> can_mutate=True",
)

# MU5: Nested key with locked prefix returns False
record(
    mutation_engine.can_mutate(locked_bot, "news.some_field") is False,
    "MU5: nested key with locked prefix -> can_mutate=False",
)

# MU6: Neither in allowed nor locked returns False
record(
    mutation_engine.can_mutate(locked_bot, "risk_limit.xyz") is False,
    "MU6: neither allowed nor locked -> can_mutate=False",
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
