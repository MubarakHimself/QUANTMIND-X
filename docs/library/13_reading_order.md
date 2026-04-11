# QuantMindLib V1 — Reading Order for Next Session

## Purpose

This document provides the exact reading sequence for the next coding session. Follow this order to understand the codebase with minimum re-discovery work.

## One-Time Setup

Before starting, ensure you have access to:

```
/home/mubarkahimself/Desktop/QUANTMINDX/.claude/worktrees/elegant-turing/
```

The library root is:
```
/home/mubarkahimself/Desktop/QUANTMINDX/.claude/worktrees/elegant-turing/src/library/
```

## Reading Sequence

### Phase 1: Orientation (Read These First)

1. **`docs/library/00_index.md`** — Start here. Master navigation.
2. **`docs/library/01_library_overview.md`** — What the library is, is not, and why. Design principles. Ownership boundaries.
3. **`docs/library/02_current_architecture.md`** — Package layout, file inventory, phase status.

### Phase 2: Contracts (Read Before Touching Any Schema)

4. **`docs/library/03_public_contracts_and_objects.md`** — Every domain object, enum, and schema. Code locations. Sync/async classification.

### Phase 3: Feature System (Read Before Adding Features)

5. **`docs/library/04_feature_system.md`** — Feature classification rule, 16 feature modules, FeatureRegistry bootstrap, FeatureEvaluator, quality propagation.
6. **`docs/library/05_proxy_microstructure_layer.md`** — Semantic honesty rules, proxy vs native vs external classification, data flow.

### Phase 4: Integration Points (Read Before Wiring)

7. **`docs/library/06_bridge_and_integration_boundaries.md`** — What each bridge does, ownership, DPR dual engine handling, Governor class paths.
8. **`docs/library/07_runtime_and_decision_flow.md`** — Runtime orchestration, rejection points, SafetyHooks integration.

### Phase 5: Supporting Docs (Read as Needed)

9. **`docs/library/08_error_logic.md`** — V1 error hierarchy, rejection vs exception, RiskMode vs exception.
10. **`docs/library/09_evaluation_and_workflows.md`** — Evaluation pipeline, lifecycle, workflow bridges.
11. **`docs/library/11_known_gaps_and_deferred_items.md`** — What's complete vs stale vs deferred.
12. **`docs/library/12_change_log_current_state.md`** — Implementation summary, corrections discovered.

### Phase 6: Code Scan (After Reading Docs)

After reading docs 1-5, scan the actual source:

```
src/library/core/types/enums.py           # All enum values
src/library/core/domain/                 # Schema definitions
src/library/features/base/feature_module.py  # FeatureModule ABC
src/library/features/registry.py       # FeatureRegistry
src/library/features/_registry.py      # Bootstrap
src/library/runtime/orchestrator.py     # Runtime wiring
src/library/core/bridges/               # All bridges
```

### Phase 7: Key Source Files

| File | Why Read It |
|------|-----------|
| `src/library/core/types/enums.py` | Every enum value used throughout the library |
| `src/library/core/domain/bot_spec.py` | Central BotSpec contract + all profiles |
| `src/library/features/base/feature_module.py` | FeatureModule ABC + model_post_init |
| `src/library/features/_registry.py` | Bootstrap singleton pattern |
| `src/library/runtime/feature_evaluator.py` | Dependency resolution algorithm |
| `src/library/runtime/orchestrator.py` | Central wiring point (line 58: registry) |
| `src/library/core/bridges/dpr_redis_bridge.py` | Redis write fix pattern |
| `src/library/core/errors/base.py` | Error hierarchy |
| `src/library/core/errors/audit.py` | BridgeError + AuditRecord |

## Assumptions That Are Locked

These assumptions were confirmed during reconciliation and should not be reopened:

1. **FeatureRegistry uses explicit registration, not auto-discovery.** The bootstrap in `_registry.py` is correct.

2. **`model_post_init()` correctly propagates quality_class.** The condition `if self.quality_class == "native_supported" and cfg.quality_class != "native_supported"` evaluates correctly for proxy features.

3. **Proxy features are correctly labeled.** All 4 proxy features carry `quality_class="proxy_inferred"` and include `notes` explaining the inference method.

4. **Phases 7-10 are complete.** Runtime, evaluation, workflows, and DPR/registry bridges are fully implemented. The task tracker labels are stale.

5. **`ExecutionDirective.approved=False`** is the controlled rejection mechanism, not an exception.

6. **Two Governor classes exist.** Reference by module path: `src.router.governor` and `src.risk.governor`.

7. **EnsembleVoter is not wired to live tick path.** This is a Phase 2 deferred task, not a library bug.

8. **cTrader adapter is not implemented.** `src/library/ctrader/` is empty.

## Integration Seam (Where to Start)

The most natural starting point for integration is the **cTrader Market Adapter**:

```
1. src/library/core/composition/adapter_contracts.py
   └─ IMarketDataAdapter, IExecutionAdapter (interfaces already defined)
           │
           ▼
2. src/library/ctrader/
   └─ Create: client.py, market_adapter.py, execution_adapter.py
           │
           ▼
3. src/library/runtime/orchestrator.py
   └─ Wire: CTraderMarketAdapter → BotStateManager
```

Or, if the adapter is blocked on BLOCKER-3, start with **SafetyHooks wiring**:

```
1. src/library/runtime/safety_hooks.py
   └─ Wire: kill switch events from src/events/ssl.py, src/events/dpr.py
           │
           ▼
2. src/events/
   └─ Find: event types for kill switch, DPR concern
```

## What NOT to Do

- Do NOT re-read planning docs (`docs/planning/quantmindlib/`) as ground truth. Use `docs/library/` instead.
- Do NOT reopen FeatureRegistry bootstrap — it's correct.
- Do NOT add auto-discovery to feature registration.
- Do NOT implement cTrader adapter before BLOCKER-3 is complete.
- Do NOT assume Prefect flows exist — they are stubs.

## Quick Reference

**Total library source files:** ~85
**Total library test files:** ~40+
**All 216 core tests pass**
**Proxy quality_class test passes**
**`ExecutionDirective.approved` + `rejection_reason` implemented**
**DPR Redis write gap fixed**
**DPR dual engine router implemented**
