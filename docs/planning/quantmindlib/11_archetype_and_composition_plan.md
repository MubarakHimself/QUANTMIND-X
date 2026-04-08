# 11 — Archetype and Composition Plan

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Define V1 archetypes, composition rules, and mutation safety boundaries

---

## 1. Archetype System Overview

### Design Principle
Archetypes are spec compositions, not code classes. The archetype is a structured `ArchetypeSpec` that declares required features, optional features, sessions, and constraints. The `Composer` uses this spec to build a bot.

### Base vs Derived Archetypes

**Base archetypes** are hand-defined, foundational strategy types. Each base archetype declares a fixed capability profile.

**Derived archetypes** are generated from base archetypes + feature families + constraints. New derived archetypes can be created without turning every random indicator combination into a first-class design concept.

---

## 2. V1 Base Archetypes

### A. OpeningRangeBreakout (ORB) — Phase 1 Priority

**Archetype ID:** `opening_range_breakout`
**Phase 1 active:** Yes
**Description:** Trade breakouts of the opening range (first N minutes of session). Classic ICT/Paul Dean approach.

**Spec:**
```python
@dataclass
class OpeningRangeBreakoutSpec:
    archetype_id: str = "opening_range_breakout"
    base_type: str = "breakout"
    session: str = "london"  # default, overrideable
    range_minutes: int = 30  # opening range definition period
    breakout_pips: float = 2.0  # breakout threshold
    session_tags: List[str] = ["london", "new_york"]
    required_features: List[str] = ["indicators/atr", "session/detector", "session/blackout"]
    optional_features: List[str] = ["volume/rvol", "orderflow/tick"]
    constraints: List[ConstraintSpec] = [
        ConstraintSpec(id="spread_filter", operator="<", value=1.5),
        ConstraintSpec(id="news_blackout", operator="==", value="clear"),
    ]
    confirmations: List[str] = ["spread_ok", "sentinel_allow", "session_active"]
    execution_profile: str = "orb_v1"
```

**BotType mapping:** ORB (from `src/router/bot_manifest.py`)
**60/40 split reference:** Per memory index, ORB is 40% of bot mix

### B. BreakoutScalper

**Archetype ID:** `breakout_scalper`
**Phase 1 active:** No (Phase 2)
**Description:** High-frequency breakout scalping. Faster than ORB, tighter stops.

**Spec:** Similar structure to ORB but:
- `range_minutes: int = 15`
- `breakout_pips: float = 1.0`
- `execution_profile: str = "scalp_fast_v1"`
- `required_features` includes `orderflow/tick`

### C. PullbackScalper

**Archetype ID:** `pullback_scalper`
**Phase 1 active:** No (Phase 2)
**Description:** Scalp in the direction of the trend on pullbacks.

### D. MeanReversion

**Archetype ID:** `mean_reversion`
**Phase 1 active:** No (Phase 2)
**Description:** Trade deviations from mean with bounded risk.

### E. SessionTransition

**Archetype ID:** `session_transition`
**Phase 1 active:** No (Phase 2)
**Description:** Strategies that capture transitions between market regimes/sessions.

---

## 3. V1 Derived Archetypes

### A. LondonORB

**Archetype ID:** `london_orb`
**Composition:** `OpeningRangeBreakout` + [`volume/rvol`, `indicators/atr`] + [`london`, `london_newyork_overlap`]
**Constraints:** `spread_filter < 1.5`, `min_volume > 0.8`

**Spec generation:**
```python
london_orb = ArchetypeSpec.derive(
    base="opening_range_breakout",
    features=["volume/rvol", "indicators/atr"],
    sessions=["london", "london_newyork_overlap"],
    constraints=[spread_filter, min_volume]
)
```

### B. NYORB (New York ORB)

**Archetype ID:** `ny_orb`
**Composition:** `OpeningRangeBreakout` + [`indicators/atr`, `volume/imbalance`] + [`new_york`]
**Constraints:** `spread_filter < 1.0` (NY is volatile, tighter filter)

### C. ScalperM1

**Archetype ID:** `scalper_m1`
**Composition:** `BreakoutScalper` + [`orderflow/tick`, `orderflow/spread`] + [`london`, `new_york`]
**Constraints:** `max_trades_per_hour = 10`

---

## 4. Composition Rules

### Rule 1: Spec-First Composition
Bots are composed from specs, not assembled from raw code. The composition pipeline:
```
BotSpec (archetype + features + sessions + constraints)
    │
    ▼
Composer.load(ArchetypeSpec)
    │
    ▼
Composer.validate_capabilities(BotSpec.capability_spec)
    │
    ▼
Composer.resolve_dependencies(BotSpec.features)
    │
    ▼
Composer.check_compatibility(BotSpec.archetype, BotSpec.features)
    │
    ▼
Composer.apply_constraints(BotSpec.constraints)
    │
    ▼
BotStateManager initialized with composed bot
```

### Rule 2: Capability Declaration Before Composition
Every module in a composition must declare:
- What it provides (capability IDs)
- What it requires (capability IDs)
- What it outputs (schema types)
- What archetypes it is compatible with

The Composer validates all of this before building the bot.

### Rule 3: Mutual Exclusivity via Compatibility Rules
Some features cannot coexist:
```python
CompatibilityRule(
    rule_id="no_conflicting_momentum",
    subject="indicators/rsi",
    predicate="EXCLUDES",
    target="indicators/macd",  # Example: these shouldn't coexist without explicit config
    condition=None
)
```
This is a design-time validation, not a runtime restriction.

### Rule 4: Session Scope Enforcement
Bots can only be active during declared sessions:
```python
BotSpec.sessions = ["london", "london_newyork_overlap"]
# Bot is inactive outside these sessions
```

---

## 5. Mutation Safety Boundaries

### BotMutationProfile: Locked vs Mutable Surfaces

The `BotMutationProfile` in `BotSpec` declares what can and cannot be mutated:

```python
@dataclass
class BotMutationProfile:
    bot_id: str
    allowed_mutation_areas: List[str] = [
        "parameters.stop_loss_pips",
        "parameters.take_profit_pips",
        "parameters.range_minutes",
        "features.optional",  # can add optional features
    ]
    locked_components: List[str] = [
        "archetype",          # cannot change strategy type
        "features.required",   # cannot remove required features
        "symbol_scope",        # cannot change trading symbols
        "execution_profile",   # cannot change execution mode
    ]
    lineage: List[str] = []      # ancestor bot IDs
    parent_variant_id: Optional[str] = None
    generation: int = 0
    unsafe_areas: List[str] = []
```

### Mutation Engine Workflow (WF2)
```
VariantRegistry → BotMutationProfile (loaded with BotSpec)
    │
    ▼
MutationEngine.apply_mutations(BotSpec, allowed_areas, locked_areas)
    │
    ├─► Validate: mutation targets only allowed_areas
    ├─► Validate: no changes to locked_components
    ├─► Validate: lineage continuity (parent_variant_id chain maintained)
    └─► Validate: no unsafe_area mutations
    │
    ▼
Mutated BotSpec → EvaluationBridge → FullBacktestPipeline
    │
    ▼
EvaluationResult → DPR score → promote/demote
```

### Example: Mutating ORB Parameters
```
Original: range_minutes=30, breakout_pips=2.0, stop_loss=15
Mutation allowed: range_minutes=25 (in allowed_areas)
Mutation allowed: breakout_pips=1.8 (in allowed_areas)
Mutation REJECTED: archetype change to "mean_reversion" (locked)
Mutation REJECTED: remove required feature "indicators/atr" (locked)
```

### Mutation Boundaries in VariantRegistry
`VariantRegistry` tracks lineage:
```python
BotVariant {
    variant_id: "orb_london_v2",
    parent_id: "orb_london_v1",    # locked chain
    lineage: ["orb_london_v1", "orb_london_prime"],
    generation: 2                 # increment per mutation
}
```

---

## 6. Freely Mutable Surfaces

These are the parameters agents can freely tune during backtest optimization:

| Parameter | Type | Mutable | Constraint |
|-----------|------|---------|------------|
| `parameters.stop_loss_pips` | float | Yes | >= 1.0, <= 50.0 |
| `parameters.take_profit_pips` | float | Yes | >= 1.0, <= 100.0 |
| `parameters.range_minutes` | int | Yes | 10-120 |
| `parameters.breakout_pips` | float | Yes | >= 0.5, <= 5.0 |
| `parameters.kelly_fraction` | float | Yes | 0.1-0.8 |
| `features.optional` | List[str] | Yes | Only from optional list |
| `constraints.spread_filter` | float | Yes | 0.5-5.0 |

---

## 7. Controlled Mutation Surfaces

These require explicit approval or are bounded by safety rules:

| Parameter | Type | Controlled | Rule |
|-----------|------|-----------|------|
| `features.required` | List[str] | Locked | Cannot remove required features |
| `archetype` | str | Locked | Cannot change strategy type |
| `symbol_scope` | List[str] | Locked | Cannot change trading symbols |
| `execution_profile` | str | Locked | Cannot change execution mode |
| `sessions` | List[str] | Bounded | Can add/remove sessions, not all (min 1) |
| `capability_spec` | CapabilitySpec | Locked | Cannot change requirements |

---

## 8. Locked/Out-of-Scope Surfaces

These are never mutable:

| Parameter | Reason |
|-----------|--------|
| `capability_spec` | Core contract; changing breaks composition validation |
| `archetype.base_type` | Strategy family classification; changing is a new bot |
| `dependencies` | Bridge dependencies; removing kills switch/risk integration |
| `confirmations` | Safety confirmations; removing bypasses safety checks |

---

## 9. Archetype Registry

```python
class ArchetypeRegistry:
    def get(self, archetype_id: str) -> Optional[ArchetypeSpec]
    def register(self, archetype: ArchetypeSpec)
    def derive(self, base: str, features: List[str], sessions: List[str], constraints: List[Constraint]) -> ArchetypeSpec
    def compatible_features(self, archetype_id: str) -> List[str]
    def compatible_dependencies(self, archetype_id: str) -> List[str]
```

---

## 10. Indexed References

| Archetype | Codebase Reference | Notes |
|-----------|-------------------|-------|
| SCALPER | `src/router/bot_manifest.py` (StrategyType enum) | 60% of bot mix |
| ORB | `src/router/bot_manifest.py` (StrategyType enum) | 40% of bot mix |
| TRDDocument | `src/trd/schema.py` | Strategy spec with parameters, sessions, confirmations |
| VariantRegistry | `src/router/variant_registry.py` | Lineage, parent_id, generation tracking |
| BotMutationProfile | Shared object model | Mutation profile structure |
| WF2 (ImprovementLoop) | `flows/improvement_loop_flow.py` | Mutation → backtest → eval loop |
| BotTagRegistry mutual exclusivity | `src/router/bot_tag_registry.py` | Reference pattern for compatibility rules |