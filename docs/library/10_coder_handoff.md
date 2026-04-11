# QuantMindLib V1 — Coder Handoff

## How to Approach This Codebase

This document is the primary guide for a coding agent starting on QuantMindLib. Read it first.

## Required Pre-Reading

Before touching any code, read these files in order:

1. **`docs/library/00_index.md`** — Navigation. Maps everything.
2. **`docs/library/01_library_overview.md`** — What the library is, what it is not, design principles.
3. **`docs/library/02_current_architecture.md`** — Actual package layout. What is implemented vs not.
4. **`docs/library/10_coder_handoff.md`** — This file. How to work in this codebase.

Then, depending on what you're working on:

| Area | Read Next |
|------|-----------|
| Features | `docs/library/04_feature_system.md`, `docs/library/05_proxy_microstructure_layer.md` |
| Bridges | `docs/library/06_bridge_and_integration_boundaries.md` |
| Runtime | `docs/library/07_runtime_and_decision_flow.md` |
| Errors | `docs/library/08_error_logic.md` |
| Contracts | `docs/library/03_public_contracts_and_objects.md` |

## Important Boundaries Not to Violate

1. **Library owns contracts, not platform internals.** Do not modify DPR scoring, governor logic, kill switch internals, HMM/BOCPD/MS-GARCH, or backtest engine internals from within the library.

2. **Bridges translate, not absorb.** Every bridge method should convert between library objects and system-native objects. Do not add business logic to bridges.

3. **Proxy features must be labeled as proxy.** Every feature with `quality_class="proxy_inferred"` MUST include a `notes` field explaining the inference method. Do not remove or rename this requirement.

4. **ExecutionDirective.approved is the rejection mechanism.** Controlled rejections use `approved=False, rejection_reason=...`. Do NOT raise exceptions for expected rejections.

5. **Explicit bootstrap is correct.** The FeatureRegistry singleton uses explicit registration, not auto-discovery. Do NOT add auto-discovery.

6. **Two Governor classes exist.** Reference by module path: `src.router.governor` and `src.risk.governor`. Do NOT rename them.

7. **cTrader adapter is NOT implemented.** `src/library/ctrader/` is empty. Do NOT assume it exists.

8. **Prefect flows are stubs.** `stub_flows.py` contains explicit stubs. The real flows do not exist. Do NOT implement Prefect flow logic inside the library.

## Known Completed Areas

The following areas are fully implemented and should NOT be rewritten:

- **All domain schemas** — 14 files in `core/domain/`, all frozen/mutable as designed
- **All enums** — 12+ in `core/types/enums.py`
- **All 9 bridges** — DPRRedisPublisher, DPRConcernEmitter, SingapuraDualEngineRouter, SentinelBridge, DPRBridge, RegistryBridge, JournalBridge, LifecycleBridge, EvaluationBridge, RiskBridge, ExecutionBridge, SafetyHooks
- **All 16 feature modules** — across 6 families
- **FeatureModule ABC + FeatureRegistry** — singleton bootstrap
- **Archetype system** — ORB fully implemented, 4 deep archetypes, Composer, MutationEngine
- **Runtime orchestrator** — RuntimeOrchestrator, FeatureEvaluator, IntentEmitter, BotStateManager, SafetyHooks
- **Evaluation** — EvaluationOrchestrator, StrategyCodeGenerator
- **Workflow bridges** — WF1Bridge, WF2Bridge
- **V1 Error hierarchy** — ERR-001 through ERR-004

## Known Deferred Areas

These areas should NOT be implemented as part of V1:

- **cTrader real adapter** — adapter in `src/library/ctrader/`
- **Multi-broker framework** — abstract beyond cTrader
- **Pattern service** — `PatternSignal` is a placeholder
- **Chart rendering** — footprint/VWAP rendering
- **External order flow adapter** — true executed trade data
- **EnsembleVoter live wiring** — exists but not connected
- **Prefect flow implementations** — stubs only
- **ErrorSeverity wiring into exceptions** — enum exists but not used
- **AuditRecord emission** — schema exists but not emitted

## Tracker Drift

The task tracker (`14_ticket_backlog.md`) has **stale TODO labels** for Phases 7-10. These are actually fully implemented. Do NOT use the task tracker as a to-do list. Use the actual file listings in `docs/library/02_current_architecture.md` as ground truth.

## Integration Priorities

The next phase is **integration** — wiring library components into existing system components. Priority order:

1. **cTrader Market Adapter** — Without it, the library cannot run on live data
2. **DPR Redis verification** — Verify Redis write path actually works in production
3. **SafetyHooks wiring** — Wire kill switch events into `SafetyHooks`
4. **IntentEmitter wiring** — Connect to real tick processing path
5. **BotStateManager wiring** — Connect to real market data feed
6. **Registry bidirectional sync** — Verify with real BotRegistry
7. **JournalBridge** — Wire to real trade log
8. **Prefect flow stubs** — Replace with real Prefect calls

## Important Patterns

### Adding a new feature:
1. Create feature class extending `FeatureModule`
2. Implement `config`, `required_inputs`, `output_keys`, `compute`
3. If proxy/inferred: set `quality_class="proxy_inferred"` in config AND include `notes`
4. Register in family `__init__.py`: `reg.register(MyFeature(), family="indicators")`
5. Add test in appropriate `tests/library/features/` directory

### Adding a new bridge:
1. Create bridge class in `src/library/core/bridges/`
2. Bridges translate, not absorb — keep translation logic only
3. Use `BridgeUnavailableError` for unreachable systems
4. Use full module paths for existing system references

### Adding a new domain object:
1. Create in appropriate `core/domain/` file
2. Use Pydantic v2 BaseModel
3. Frozen for immutable objects (BotSpec), mutable for runtime objects
4. Export from `core/domain/__init__.py`

### Common pitfalls:
- **Do not** return `None` from a bridge when an error occurs — raise `BridgeUnavailableError`
- **Do not** add business logic to bridges — translate only
- **Do not** modify `ExecutionDirective` schema casually — it's a contract
- **Do not** auto-import features — use explicit registration

## Code Reading Order

1. `src/library/core/types/enums.py` — All enum values
2. `src/library/core/domain/` — Schema definitions
3. `src/library/features/base/feature_module.py` — FeatureModule ABC
4. `src/library/features/registry.py` — FeatureRegistry
5. `src/library/features/_registry.py` — Bootstrap
6. `src/library/runtime/orchestrator.py` — Runtime wiring
7. `src/library/core/bridges/` — Bridge implementations
8. `src/library/archetypes/` — Archetype system
9. `src/library/evaluation/` — Evaluation pipeline
10. `src/library/workflows/` — Workflow bridges

## Test Locations

| Area | Test Location |
|------|-------------|
| Domain objects | `tests/library/core/domain/` |
| Features | `tests/library/features/`, `tests/library/test_volume_features.py` |
| Feature evaluator | `tests/library/runtime/test_feature_evaluator.py` |
| Runtime | `tests/library/runtime/` |
| Evaluation | `tests/library/evaluation/` |
| Workflows | `tests/library/workflows/` |
| Bridges | `tests/library/core/bridges/` |
| Archetypes | `tests/library/archetypes/` |

Run tests with: `cd /home/mubarkahimself/Desktop/QUANTMINDX/.claude/worktrees/elegant-turing && python -m pytest tests/library/ -v --tb=short`

## Key Imports

```python
# Domain objects
from src.library.core.domain import BotSpec, FeatureVector, TradeIntent
from src.library.core.domain import ExecutionDirective, MarketContext
from src.library.core.domain import BotRuntimeProfile, BotEvaluationProfile

# Features
from src.library.features.registry import FeatureRegistry
from src.library.features._registry import get_default_registry
from src.library.features.base import FeatureModule, FeatureConfig

# Runtime
from src.library.runtime.orchestrator import RuntimeOrchestrator
from src.library.runtime.feature_evaluator import FeatureEvaluator
from src.library.runtime.state_manager import BotStateManager
from src.library.runtime.safety_hooks import SafetyHooks

# Bridges
from src.library.core.bridges.sentinel_dpr_bridges import SentinelBridge, DPRBridge
from src.library.core.bridges.registry_journal_bridges import RegistryBridge, JournalBridge
from src.library.core.bridges.dpr_redis_bridge import DPRRedisPublisher

# Errors
from src.library.core.errors import (
    LibraryError, LibraryConfigError, ContractValidationError,
    BridgeError, BridgeUnavailableError, DependencyMissingError,
)
from src.library.core.errors.audit import AuditRecord, ErrorSeverity
```
