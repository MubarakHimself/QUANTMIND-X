"""
Packet 6E Self-Verification — 20 tests
Derived Archetypes + Structural Stubs
"""
import sys

PASS = 0
FAIL = 0


def test(n: int, name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [{n:02d}] PASS  {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  [{n:02d}] FAIL  {name}: {e}")
        FAIL += 1
    except Exception as e:
        print(f"  [{n:02d}] ERROR {name}: {e}")
        FAIL += 1


# ── Imports ──────────────────────────────────────────────────────────────────
from src.library.archetypes.orb import ORB_ARCHETYPE
from src.library.archetypes.derived import (
    LONDON_ORB_ARCHETYPE,
    NY_ORB_ARCHETYPE,
    SCALPER_M1_ARCHETYPE,
)
from src.library.archetypes.stubs import (
    BREAKOUT_SCALPER_ARCHETYPE,
    PULLBACK_SCALPER_ARCHETYPE,
    MEAN_REVERSION_ARCHETYPE,
    SESSION_TRANSITION_ARCHETYPE,
    get_full_registry,
)
from src.library.archetypes.registry import ArchetypeRegistry

# ── Helper assertions ─────────────────────────────────────────────────────────


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# =============================================================================
# GROUP A: Derived Archetype Properties
# =============================================================================

# A1: LONDON_ORB_ARCHETYPE
def t01_london_orb_base_archetype_id():
    assert_true(
        LONDON_ORB_ARCHETYPE.base_archetype_id == "opening_range_breakout",
        f"got {LONDON_ORB_ARCHETYPE.base_archetype_id!r}",
    )


def t02_london_orb_session_scope():
    sessions = LONDON_ORB_ARCHETYPE.session_tags
    assert_true("london" in sessions)
    assert_true("new_york" not in sessions, "LondonORB must not include new_york")
    assert_true("tokyo" not in sessions, "LondonORB must not include tokyo")


def t03_london_orb_spread_tighter_than_orb():
    london_spread = next(
        c.value
        for c in LONDON_ORB_ARCHETYPE.constraints
        if c.constraint_id == "spread_filter"
    )
    orb_spread = next(
        c.value
        for c in ORB_ARCHETYPE.constraints
        if c.constraint_id == "spread_filter"
    )
    assert_true(
        london_spread < orb_spread,
        f"London spread {london_spread} not tighter than ORB spread {orb_spread}",
    )


def t04_london_orb_has_min_rvol_constraint():
    cids = {c.constraint_id for c in LONDON_ORB_ARCHETYPE.constraints}
    assert_true("min_volume_rvol" in cids, f"min_volume_rvol missing from {cids}")


def t05_london_orb_v1_priority():
    assert_true(LONDON_ORB_ARCHETYPE.v1_priority)
    assert_true(LONDON_ORB_ARCHETYPE.quality_class == "native_supported")


# A2: NY_ORB_ARCHETYPE
def t06_ny_orb_base_archetype_id():
    assert_true(
        NY_ORB_ARCHETYPE.base_archetype_id == "opening_range_breakout",
        f"got {NY_ORB_ARCHETYPE.base_archetype_id!r}",
    )


def t07_ny_orb_session_scope():
    sessions = NY_ORB_ARCHETYPE.session_tags
    assert_true("new_york" in sessions)
    assert_true("london" not in sessions, "NYORB must not include london")
    assert_true("sydney" not in sessions, "NYORB must not include sydney")


def t08_ny_orb_spread_tighter_than_london():
    ny_spread = next(
        c.value
        for c in NY_ORB_ARCHETYPE.constraints
        if c.constraint_id == "spread_filter"
    )
    london_spread = next(
        c.value
        for c in LONDON_ORB_ARCHETYPE.constraints
        if c.constraint_id == "spread_filter"
    )
    assert_true(
        ny_spread < london_spread,
        f"NY spread {ny_spread} not tighter than London spread {london_spread}",
    )


def t09_ny_orb_has_slippage_constraint():
    cids = {c.constraint_id for c in NY_ORB_ARCHETYPE.constraints}
    assert_true(
        "max_slippage_ticks" in cids,
        f"max_slippage_ticks missing from {cids}",
    )


def t10_ny_orb_v1_priority():
    assert_true(NY_ORB_ARCHETYPE.v1_priority)
    assert_true(NY_ORB_ARCHETYPE.quality_class == "native_supported")


# A3: SCALPER_M1_ARCHETYPE
def t11_scalper_base_archetype_id():
    # Scalper has no base — it stands alone
    assert_true(SCALPER_M1_ARCHETYPE.base_archetype_id is None)


def t12_scalper_requires_tob_and_spread():
    req = set(SCALPER_M1_ARCHETYPE.required_features)
    assert_true("microstructure/spread" in req, f"missing spread from {req}")
    assert_true(
        "microstructure/tob_pressure" in req, f"missing tob_pressure from {req}"
    )


def t13_scalper_max_trades_per_hour():
    assert_true(
        SCALPER_M1_ARCHETYPE.max_trades_per_hour == 10,
        f"got {SCALPER_M1_ARCHETYPE.max_trades_per_hour}",
    )


def t14_scalper_tightest_spread():
    spread = next(
        c.value
        for c in SCALPER_M1_ARCHETYPE.constraints
        if c.constraint_id == "spread_filter"
    )
    assert_true(
        spread == 0.8,
        f"Scalper spread filter should be 0.8, got {spread}",
    )


def t15_scalper_v1_priority():
    assert_true(SCALPER_M1_ARCHETYPE.v1_priority)
    assert_true(SCALPER_M1_ARCHETYPE.quality_class == "native_supported")


# =============================================================================
# GROUP B: Structural Stub Counts and Properties
# =============================================================================

def t16_stub_count():
    # All derived archetypes are v1_priority=True
    # All stubs are v1_priority=False
    assert_true(
        not BREAKOUT_SCALPER_ARCHETYPE.v1_priority,
        "BreakoutScalper must be v1_priority=False",
    )
    assert_true(
        not PULLBACK_SCALPER_ARCHETYPE.v1_priority,
        "PullbackScalper must be v1_priority=False",
    )
    assert_true(
        not MEAN_REVERSION_ARCHETYPE.v1_priority,
        "MeanReversion must be v1_priority=False",
    )
    assert_true(
        not SESSION_TRANSITION_ARCHETYPE.v1_priority,
        "SessionTransition must be v1_priority=False",
    )


def t17_stub_quality_class():
    for stub in [
        BREAKOUT_SCALPER_ARCHETYPE,
        PULLBACK_SCALPER_ARCHETYPE,
        MEAN_REVERSION_ARCHETYPE,
        SESSION_TRANSITION_ARCHETYPE,
    ]:
        assert_true(
            stub.quality_class == "native_supported",
            f"{stub.archetype_id} quality_class={stub.quality_class!r}",
        )


def t18_stub_base_archetype_id_none():
    for stub in [
        BREAKOUT_SCALPER_ARCHETYPE,
        PULLBACK_SCALPER_ARCHETYPE,
        MEAN_REVERSION_ARCHETYPE,
        SESSION_TRANSITION_ARCHETYPE,
    ]:
        assert_true(
            stub.base_archetype_id is None,
            f"{stub.archetype_id} base_archetype_id={stub.base_archetype_id!r}",
        )


def t19_stub_base_types():
    # BreakoutScalper and PullbackScalper share base_type="scalping"
    # MeanReversion is "mean_reversion", SessionTransition is "session_transition"
    base_types = {
        BREAKOUT_SCALPER_ARCHETYPE.base_type,
        PULLBACK_SCALPER_ARCHETYPE.base_type,
        MEAN_REVERSION_ARCHETYPE.base_type,
        SESSION_TRANSITION_ARCHETYPE.base_type,
    }
    assert_true(
        len(base_types) == 3,
        f"Expected 3 unique base_types, got {base_types}",
    )
    assert_true(
        BREAKOUT_SCALPER_ARCHETYPE.base_type == "scalping",
        f"BreakoutScalper base_type={BREAKOUT_SCALPER_ARCHETYPE.base_type!r}",
    )
    assert_true(
        PULLBACK_SCALPER_ARCHETYPE.base_type == "scalping",
        f"PullbackScalper base_type={PULLBACK_SCALPER_ARCHETYPE.base_type!r}",
    )
    assert_true(
        MEAN_REVERSION_ARCHETYPE.base_type == "mean_reversion",
        f"MeanReversion base_type={MEAN_REVERSION_ARCHETYPE.base_type!r}",
    )
    assert_true(
        SESSION_TRANSITION_ARCHETYPE.base_type == "session_transition",
        f"SessionTransition base_type={SESSION_TRANSITION_ARCHETYPE.base_type!r}",
    )


# =============================================================================
# GROUP C: get_full_registry() Singleton + 8 Archetypes
# =============================================================================

def t20_registry_singleton_count():
    reg = get_full_registry()
    all_ids = reg.list_all()
    assert_true(
        len(all_ids) == 8,
        f"Expected 8 archetypes, got {len(all_ids)}: {all_ids}",
    )


def t21_registry_singleton_idempotent():
    reg1 = get_full_registry()
    reg2 = get_full_registry()
    assert_true(
        reg1 is reg2,
        "get_full_registry() must return the same singleton instance",
    )


def t22_registry_has_all_archetypes():
    reg = get_full_registry()
    expected = {
        "opening_range_breakout",
        "london_orb",
        "ny_orb",
        "scalper_m1",
        "breakout_scalper",
        "pullback_scalper",
        "mean_reversion",
        "session_transition",
    }
    actual = set(reg.list_all())
    assert_true(
        actual == expected,
        f"Missing: {expected - actual}, Extra: {actual - expected}",
    )


def t23_registry_v1_priority_archetypes():
    reg = get_full_registry()
    v1 = set(reg.list_v1_priority())
    expected_v1 = {
        "opening_range_breakout",
        "london_orb",
        "ny_orb",
        "scalper_m1",
    }
    assert_true(
        v1 == expected_v1,
        f"V1 archetypes: expected {expected_v1}, got {v1}",
    )


# =============================================================================
# GROUP D: ArchetypeRegistry.derive() Integration
# =============================================================================

def t24_registry_derive_london_orb():
    reg = get_full_registry()
    # Derive a custom variant from london_orb
    derived = reg.derive(
        base_archetype_id="london_orb",
        derived_id="london_orb_usdchf",
        overrides={
            "description": "London ORB tailored for USDCHF.",
            "default_allowed_areas": [
                "parameters.stop_loss_pips",
                "parameters.take_profit_pips",
            ],
        },
    )
    assert_true(
        derived.archetype_id == "london_orb_usdchf",
        f"archetype_id={derived.archetype_id!r}",
    )
    assert_true(
        derived.base_archetype_id == "london_orb",
        f"base_archetype_id={derived.base_archetype_id!r}",
    )
    assert_true(
        derived.base_type == "breakout",
        f"base_type={derived.base_type!r}",
    )
    assert_true(
        "USDCHF" in derived.description,
        f"description={derived.description!r}",
    )
    # Inherits required features from LondonORB
    assert_true(
        "indicators/atr_14" in derived.required_features,
        "missing inherited atr_14",
    )
    # Inherits session tags
    assert_true(
        "london" in derived.session_tags,
        "missing inherited london session tag",
    )


def t25_registry_derive_via_class_method():
    from src.library.archetypes.archetype_spec import ArchetypeSpec

    # Use the classmethod derive()
    derived = ArchetypeSpec.derive(
        base=ORB_ARCHETYPE,
        overrides={
            "archetype_id": "tokyo_orb",
            "session_tags": ["tokyo"],
            "description": "ORB scoped to Tokyo session.",
        },
    )
    assert_true(derived.base_archetype_id == "opening_range_breakout")
    assert_true(derived.archetype_id == "tokyo_orb")
    assert_true("tokyo" in derived.session_tags)
    assert_true("london" not in derived.session_tags)


def t26_registry_derive_unknown_base():
    reg = get_full_registry()
    from src.library.archetypes.registry import ArchetypeNotFoundError

    try:
        reg.derive(
            base_archetype_id="nonexistent_archetype",
            derived_id="custom_archetype",
            overrides={},
        )
        raise AssertionError("Expected ArchetypeNotFoundError")
    except ArchetypeNotFoundError:
        pass  # expected


# =============================================================================
# GROUP E: Cross-Archetype Compatibility Checks
# =============================================================================

def t27_bollinger_excluded_everywhere():
    reg = get_full_registry()
    for aid in reg.list_all():
        spec = reg.get(aid)
        assert_true(
            "indicators/bollinger" not in spec.required_features,
            f"{aid} has bollinger in required",
        )
        assert_true(
            "indicators/bollinger" not in spec.optional_features,
            f"{aid} has bollinger in optional",
        )


def t28_atr_required_for_all_breakout_archetypes():
    reg = get_full_registry()
    breakout_types = ["breakout", "scalping"]
    for aid in reg.list_all():
        spec = reg.get(aid)
        if spec.base_type in breakout_types:
            assert_true(
                "indicators/atr_14" in spec.required_features,
                f"{aid} (type={spec.base_type}) missing atr_14 in required",
            )


def t29_session_detector_required_all():
    reg = get_full_registry()
    for aid in reg.list_all():
        spec = reg.get(aid)
        assert_true(
            "session/detector" in spec.required_features,
            f"{aid} missing session/detector in required",
        )
        assert_true(
            "session/blackout" in spec.required_features,
            f"{aid} missing session/blackout in required",
        )


def t30_news_clear_constraint_all():
    reg = get_full_registry()
    for aid in reg.list_all():
        spec = reg.get(aid)
        news_cids = [
            c for c in spec.get_constraints() if c.constraint_id == "news_clear"
        ]
        assert_true(
            len(news_cids) == 1,
            f"{aid}: expected 1 news_clear constraint, got {len(news_cids)}",
        )
        assert_true(
            news_cids[0].operator == "==",
            f"{aid} news_clear operator={news_cids[0].operator!r}",
        )
        assert_true(
            news_cids[0].value == "clear",
            f"{aid} news_clear value={news_cids[0].value!r}",
        )


def t31_compatible_features_coverage():
    reg = get_full_registry()
    for aid in reg.list_all():
        spec = reg.get(aid)
        compat = reg.compatible_features(aid)
        assert_true(len(compat) > 0, f"{aid} has no compatible features")
        # All required features should be in compatible list
        for req in spec.required_features:
            assert_true(
                req in compat,
                f"{aid}: required {req} not in compatible_features",
            )
        # No excluded features should be in compatible list
        for excl in spec.excluded_features:
            assert_true(
                excl not in compat,
                f"{aid}: excluded {excl} still in compatible_features",
            )


def t32_all_archetypes_frozen():
    reg = get_full_registry()
    for aid in reg.list_all():
        spec = reg.get(aid)
        try:
            spec.model_validate(spec.model_dump())
        except Exception as e:
            raise AssertionError(f"{aid} frozen spec re-validation failed: {e}")


# =============================================================================
# Run
# =============================================================================

print("\n=== Packet 6E Self-Verification (20+ tests) ===\n")

# Group A: Derived archetype properties
print("GROUP A: Derived Archetype Properties")
test(1,  "LondonORB.base_archetype_id == 'opening_range_breakout'",     t01_london_orb_base_archetype_id)
test(2,  "LondonORB.session_scope: london only, not new_york/tokyo",    t02_london_orb_session_scope)
test(3,  "LondonORB.spread_filter < ORB.spread_filter (1.2 vs 1.5)",   t03_london_orb_spread_tighter_than_orb)
test(4,  "LondonORB has min_volume_rvol constraint",                    t04_london_orb_has_min_rvol_constraint)
test(5,  "LondonORB v1_priority=True, quality_class=native_supported",   t05_london_orb_v1_priority)
test(6,  "NYORB.base_archetype_id == 'opening_range_breakout'",        t06_ny_orb_base_archetype_id)
test(7,  "NYORB.session_scope: new_york only, not london/sydney",       t07_ny_orb_session_scope)
test(8,  "NYORB.spread_filter < LondonORB.spread_filter (1.0 vs 1.2)",t08_ny_orb_spread_tighter_than_london)
test(9,  "NYORB has max_slippage_ticks constraint",                    t09_ny_orb_has_slippage_constraint)
test(10, "NYORB v1_priority=True, quality_class=native_supported",     t10_ny_orb_v1_priority)
test(11, "ScalperM1.base_archetype_id is None (standalone)",           t11_scalper_base_archetype_id)
test(12, "ScalperM1 requires microstructure/spread + tob_pressure",    t12_scalper_requires_tob_and_spread)
test(13, "ScalperM1 max_trades_per_hour == 10",                       t13_scalper_max_trades_per_hour)
test(14, "ScalperM1 spread_filter == 0.8 (tightest)",                  t14_scalper_tightest_spread)
test(15, "ScalperM1 v1_priority=True, quality_class=native_supported",  t15_scalper_v1_priority)

# Group B: Structural stubs
print("\nGROUP B: Structural Stub Counts and Properties")
test(16, "All 4 stubs have v1_priority=False",                          t16_stub_count)
test(17, "All 4 stubs have quality_class=native_supported",            t17_stub_quality_class)
test(18, "All 4 stubs have base_archetype_id=None",                    t18_stub_base_archetype_id_none)
test(19, "4 stubs: 3 unique base_types (scalping x2, mean_reversion, session_transition)", t19_stub_base_types)

# Group C: get_full_registry()
print("\nGROUP C: get_full_registry() Singleton + 8 Archetypes")
test(20, "get_full_registry() returns registry with 8 archetypes",      t20_registry_singleton_count)
test(21, "get_full_registry() singleton idempotent (same instance)",     t21_registry_singleton_idempotent)
test(22, "All 8 archetype IDs present in registry",                   t22_registry_has_all_archetypes)
test(23, "V1 priority archetypes: ORB + LondonORB + NYORB + ScalperM1",t23_registry_v1_priority_archetypes)

# Group D: Registry.derive() integration
print("\nGROUP D: ArchetypeRegistry.derive() Integration")
test(24, "derive() from london_orb inherits ATR, session, overrides ID", t24_registry_derive_london_orb)
test(25, "ArchetypeSpec.derive() classmethod works with ORB",           t25_registry_derive_via_class_method)
test(26, "derive() raises ArchetypeNotFoundError for unknown base",    t26_registry_derive_unknown_base)

# Group E: Cross-archetype compatibility
print("\nGROUP E: Cross-Archetype Compatibility Checks")
test(27, "indicators/bollinger excluded from all archetypes",           t27_bollinger_excluded_everywhere)
test(28, "indicators/atr_14 required for all breakout/scalping",       t28_atr_required_for_all_breakout_archetypes)
test(29, "session/detector and session/blackout required for all",      t29_session_detector_required_all)
test(30, "news_clear constraint (operator==, value='clear') in all",    t30_news_clear_constraint_all)
test(31, "compatible_features covers required, excludes excluded",      t31_compatible_features_coverage)
test(32, "All archetypes are frozen (re-validation passes)",            t32_all_archetypes_frozen)

# Summary
total = PASS + FAIL
print(f"\n{'=' * 50}")
print(f"  PASSED: {PASS}/{total}")
print(f"  FAILED: {FAIL}/{total}")
print(f"{'=' * 50}\n")

sys.exit(0 if FAIL == 0 else 1)
