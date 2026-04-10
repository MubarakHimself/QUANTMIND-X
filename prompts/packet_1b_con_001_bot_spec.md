# SUPER AGENT — Implementation Prompt: QuantMindLib V1 — Packet 1B
## CONTRACT-001: BotSpec Multi-Profile Contract

---

**Role:** You are implementing QuantMindLib V1 core domain schemas. You are NOT writing business logic, feature implementations, or adapter interfaces.

---

## MANDATORY SCAN PHASE FIRST

Before writing any code, do the following **in order**:

1. **Read `docs/planning/quantmindlib/07_shared_object_model.md`** — Section "Object Inventory" §1–4. Note:
   - BotSpec: required fields, optional fields, producers, consumers, sync/async
   - BotRuntimeProfile: all required fields, enum types used (ActivationState, BotHealth, BotTier)
   - BotEvaluationProfile: nested types (BacktestMetrics, MonteCarloMetrics, WalkForwardMetrics, SessionScore)
   - BotMutationProfile: all required fields including lineage and locked components

2. **Read `docs/planning/quantmindlib/15_short_implementation_specs.md`** — §SPEC-001. Note:
   - Why it exists (central contract problem)
   - Affected files (TRD schema, BotManifest, VariantRegistry, lifecycle_manager)
   - Frozen/imutable requirement for BotSpec
   - Dependencies (CONTRACT-017 done, CONTRACT-018 can be parallel)

3. **Read `docs/planning/quantmindlib/14_ticket_backlog.md`** — §CONTRACT-001. Note the acceptance criteria.

4. **Read `src/library/core/types/enums.py`** — This is the source of truth for all enum types. Do NOT modify. Note:
   - ActivationState: ACTIVE, PAUSED, STOPPED
   - BotHealth: HEALTHY, DEGRADED, FAILING, OFFLINE
   - BotTier: TIER_1, TIER_2, TIER_3
   - EvaluationMode: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C

5. **Read `src/library/core/domain/__init__.py`** — Check if domain/ already exists.

6. **Run `ls src/library/core/`** — Confirm current directory structure.

7. **Run `python3 -m pytest tests/library/test_enums.py -v`** — Confirm 19/19 pass before starting. If any fail, STOP and report.

---

## EXACT SCOPE

Create the minimum viable BotSpec multi-profile contract.

**Files to create:**

- `src/library/core/domain/__init__.py` — exports all classes from bot_spec
- `src/library/core/domain/bot_spec.py` — all 8 Pydantic v2 models

**8 classes to implement (exact field lists from `07_shared_object_model.md`):**

### 1. `SessionScore` (nested model)
```
session_id: str
sharpe: float
max_drawdown: float
win_rate: float
total_trades: int
expectancy: float
```

### 2. `WalkForwardMetrics` (nested model)
```
n_splits: int
avg_sharpe: float
avg_return: float
stability: float
```

### 3. `MonteCarloMetrics` (nested model)
```
n_simulations: int
percentile_5_return: float
percentile_95_return: float
max_drawdown_95: float
sharpe_confidence_width: float
```

### 4. `BacktestMetrics` (nested model)
```
sharpe_ratio: float
max_drawdown: float
total_return: float
win_rate: float
profit_factor: float
expectancy: float
avg_bars_held: float
total_trades: int
```

### 5. `BotEvaluationProfile` (mutable model)
```
bot_id: str
backtest: BacktestMetrics
monte_carlo: MonteCarloMetrics
walk_forward: WalkForwardMetrics
pbo_score: float
robustness_score: float
spread_sensitivity: float
session_scores: Dict[str, SessionScore]
```

### 6. `BotMutationProfile` (mutable model)
```
bot_id: str
allowed_mutation_areas: List[str]
locked_components: List[str]
lineage: List[str]
parent_variant_id: Optional[str]
generation: int
unsafe_areas: List[str]
```

### 7. `BotRuntimeProfile` (mutable model)
```
bot_id: str
activation_state: ActivationState  ← from enums.py
deployment_target: str  # "paper" / "demo" / "live"
health: BotHealth  ← from enums.py
session_eligibility: Dict[str, bool]
dpr_ranking: int
dpr_score: float
journal_id: Optional[str]
report_ids: List[str]
```

### 8. `BotSpec` (FROZEN model — the canonical static spec)
```
id: str
archetype: str
symbol_scope: List[str]
sessions: List[str]
features: List[str]
confirmations: List[str]
execution_profile: str
capability_spec: Optional[Any] = None  # deferred to CONTRACT-018; placeholder only
runtime: Optional[BotRuntimeProfile] = None
evaluation: Optional[BotEvaluationProfile] = None
mutation: Optional[BotMutationProfile] = None
```

**Implementation requirements:**

- Use `from __future__ import annotations` at the top
- Use Pydantic v2: `from pydantic import BaseModel, Field, ConfigDict`
- **BotSpec must be frozen** — set `model_config = ConfigDict(frozen=True)`
- **All other models must be mutable** — set `model_config = ConfigDict(frozen=False)`
- Import enums ONLY from `src.library.core.types.enums` — never recreate
- Use `model_dump()` for `to_dict()` serialization
- Use `model_validate()` for `from_dict()` deserialization
- Include `__all__` listing all 8 class names at module level
- No field defaults to `None` unless it is Optional
- No Pydantic validators, no field constraints beyond type

---

## EXACT NON-SCOPE

- No `CapabilitySpec`, `DependencySpec`, `CompatibilityRule`, or `OutputSpec` — those are CONTRACT-018–021
- No `MarketContext`, `TradeIntent`, `ExecutionDirective`, `RiskEnvelope`, `FeatureVector`, or any other domain objects
- No `SentinelState`, `OrderFlowSignal`, `PatternSignal`, `RegistryRecord`, or `BotPerformanceSnapshot`
- No `CapabilitySpec` stub beyond `Optional[Any] = None` in BotSpec
- No modifications to `src/library/core/types/enums.py`
- No modifications to `tests/library/test_enums.py`
- No new test files for this packet (tests come after structure confirmed)
- No git commits

---

## LOCKED ASSUMPTIONS

- Platform: cTrader — no MT5-specific field names
- Python 3.11+ (Pydantic v2 with ConfigDict)
- Pydantic v2 (no Pydantic v1 `class Config` patterns)
- Frozen dataclasses via `ConfigDict(frozen=True)` — not `@dataclass(frozen=True)`
- No circular imports — use forward references via `__future__ annotations`
- cTrader backtest result compatibility is out of scope — comes in SPEC-009

---

## FILES LIKELY TO BE CREATED

```
src/library/core/domain/__init__.py
src/library/core/domain/bot_spec.py
```

---

## REQUIRED VERIFICATION STEPS

Run in this exact order:

1. **`ls src/library/core/domain/`** — confirm files exist
2. **Import test**: `python3 -c "from src.library.core.domain.bot_spec import BotSpec, BotRuntimeProfile, BotEvaluationProfile, BotMutationProfile, BacktestMetrics, MonteCarloMetrics, WalkForwardMetrics, SessionScore; print('OK')"`
3. **Enum binding**: `python3 -c "from src.library.core.domain.bot_spec import BotRuntimeProfile; from src.library.core.types.enums import ActivationState, BotHealth; p = BotRuntimeProfile(bot_id='test', activation_state=ActivationState.ACTIVE, deployment_target='paper', health=BotHealth.HEALTHY, session_eligibility={}, dpr_ranking=1, dpr_score=85.0, report_ids=[]); print('Enum binding OK')"`
4. **Frozen test**: Attempt to assign to BotSpec field after init — must raise `ValidationError`
5. **Mutable test**: Assign to BotRuntimeProfile field after init — must succeed
6. **Serialization round-trip**: Create BotSpec, dump to dict, validate back, compare — must equal original
7. **Enum tests still pass**: `python3 -m pytest tests/library/test_enums.py -v` — must be 19/19

---

## ANTI-GOALS

- Do NOT create Pydantic validators that embed business logic
- Do NOT use `Field(default=...)` unless the field truly needs a default
- Do NOT add docstrings beyond one-line module-level descriptions
- Do NOT create separate files for nested models — they all go in bot_spec.py
- Do NOT add `model_rebuild()` calls — the models are self-contained
- Do NOT touch any file outside `src/library/core/domain/` or `tests/library/`

---

## COMPLETION CRITERIA

- `src/library/core/domain/__init__.py` exists and exports all 8 classes
- `src/library/core/domain/bot_spec.py` exists with all 8 classes
- All imports resolve without errors
- `BotSpec` is frozen (mutation raises ValidationError)
- `BotRuntimeProfile`, `BotEvaluationProfile`, `BotMutationProfile` are mutable
- Serialization round-trip produces identical result
- Enum fields in BotRuntimeProfile accept correct enum values
- `python3 -m pytest tests/library/test_enums.py -v` → 19/19 still pass
- No file outside `src/library/core/domain/` was modified

---

## OPERATOR CHECKLIST

- [ ] Read `07_shared_object_model.md` §Object Inventory §1–4
- [ ] Read `15_short_implementation_specs.md` §SPEC-001
- [ ] Read `14_ticket_backlog.md` §CONTRACT-001
- [ ] Read `src/library/core/types/enums.py`
- [ ] Run `pytest tests/library/test_enums.py -v` — confirm 19/19
- [ ] Create `src/library/core/domain/__init__.py`
- [ ] Create `src/library/core/domain/bot_spec.py` with all 8 classes
- [ ] Verify imports resolve
- [ ] Verify BotSpec is frozen
- [ ] Verify mutable profiles are mutable
- [ ] Verify serialization round-trip
- [ ] Verify enum binding works
- [ ] Verify 19/19 enum tests still pass
- [ ] Report: files created, verification results, any issues

---

Report: Are all 8 classes implemented correctly? Do all verification steps pass? What files were created?
