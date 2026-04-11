# QuantMindLib V1 — Library Overview

## What QuantMindLib Is

QuantMindLib is a **thin, composable bot library** that provides:

1. **Canonical contracts** — unified schemas (BotSpec, FeatureVector, TradeIntent, ExecutionDirective, etc.) that all systems share
2. **Feature system** — reusable, parameterized feature modules (indicators, volume, microstructure, order flow, session, transforms) with a registry
3. **Archetype system** — spec-driven strategy composition with validation and constrained mutation
4. **Runtime engine** — tick-to-intent execution pipeline: FeatureEvaluator → IntentEmitter → SafetyHooks
5. **Bridge layer** — two-way translation between library objects and existing platform systems (Sentinel, DPR, Risk, Registry, Evaluation, Workflows)
6. **Evaluation orchestration** — wiring to the existing 6-mode backtest pipeline

The library is **platform-aware but not platform-bound**: it knows about cTrader but keeps all core logic in vendor-neutral contracts.

## What QuantMindLib Is Not

- **Not a trading system** — it does not own risk internals, DPR internals, kill switches, ML models, or backtest engine internals
- **Not a cTrader wrapper** — the cTrader adapter is a stub; the library bridges to it, not owns it
- **Not a replacement for existing systems** — it integrates and wraps them
- **Not a full strategy builder** — bots describe setups via specs; existing workflows (AlphaForgeFlow, ImprovementLoopFlow) provide the development pipeline
- **Not a chart pattern engine** — `PatternSignal` is a placeholder; pattern analysis is Phase 2+
- **Not a true order flow source** — microstructure features use proxy/inferred signals from depth/spread data, honestly labeled as such

## Why It Exists

Before QuantMindLib:
- Bots were described by separate objects (`BotManifest`, `TRDDocument`, DPR scores, `VariantRegistry`) with no unified schema
- Feature logic existed in SVSS but not as a registry
- Risk, DPR, registry, and evaluation were tightly coupled to specific system implementations
- No systematic way to validate bot composition before deployment

After QuantMindLib:
- `BotSpec` is the canonical bot description consumed by all systems
- Feature registry provides reusable, quality-tagged feature modules
- Bridges translate between library objects and system-native objects without absorbing system logic
- Capability declarations enable automatic composition validation

## Current Phase

**Implementation complete through Phase 10.** The library is ready for integration — wiring library components into existing system components.

Phase 11 (deferred items: cTrader adapter, multi-broker, pattern service, chart rendering) is explicitly deferred.

## Relationship to Existing System Components

```
QuantMindLib
    │
    ├─ bridges ──► src/router/sentinel.py      (regime: HMM, sensors)
    │             src/router/dpr_scoring_engine.py (DPR router layer)
    │             src/risk/dpr/scoring_engine.py  (DPR risk layer)
    │             src/router/governor.py          (risk authorization)
    │             src/risk/governor.py            (Tier 2 risk rules)
    │             src/router/bot_manifest.py      (bot registry)
    │             src/router/bot_tag_registry.py  (tag operations)
    │             src/router/variant_registry.py  (variant lineage)
    │             src/backtesting/full_backtest_pipeline.py (evaluation)
    │             flows/alpha_forge_flow.py       (WF1, stubs)
    │             flows/improvement_loop_flow.py   (WF2, stubs)
    │
    ├─ data ────── src/svss/indicators/         (SVSS wrapped features)
    │             src/router/session_detector.py (session detection)
    │             src/market/news_blackout.py     (news state)
    │
    └─ events ──── src/events/dpr.py             (DPR score events)
                   src/events/ssl.py              (SSL events)
                   src/events/regime.py            (regime events)
```

## Library Design Principles

1. **Thin bots, strong platform** — Bots describe setups via specs; platform owns risk, news, portfolio controls
2. **Specs drive composition** — BotSpec, FeatureSpec, ArchetypeSpec are serializable and versioned
3. **Platform-aware, not platform-bound** — Vendor-neutral contracts, adapter-specific code isolated
4. **Sync/async is explicit** — Every boundary declares sync or async; no informal mixing
5. **Capability declarations first** — Modules declare what they need and produce before building
6. **Bridge, don't replace** — Existing strong systems are integration targets, not replacements
7. **Semantic honesty** — Proxy features are labeled as proxy, not claimed as true data

## What the Library Owns vs What Stays Outside

### Library Owns
- Domain contracts (BotSpec, FeatureVector, TradeIntent, ExecutionDirective, MarketContext, etc.)
- FeatureModule ABC and all feature implementations (16 modules)
- FeatureRegistry singleton and bootstrap
- FeatureEvaluator, IntentEmitter, BotStateManager, SafetyHooks
- EvaluationOrchestrator, StrategyCodeGenerator
- All bridge implementations (translation, not absorption)
- ArchetypeRegistry, Composer, MutationEngine, CompositionValidator
- V1 error hierarchy (LibraryError, BridgeError, etc.)
- RuntimeOrchestrator (wires everything together)

### External Systems Own
- DPR scoring internals (router + risk engines)
- Governor risk computation (router + risk)
- Kill switch logic (KillSwitch, ProgressiveKillSwitch)
- Circuit breaker internals (SSLCircuitBreaker, BotCircuitBreaker)
- Sentinel regime classification (HMM, sensors, EnsembleVoter)
- Backtest engine internals (MT5, mode runner, MC, WFA, PBO)
- Prefect flow execution (AlphaForgeFlow, ImprovementLoopFlow — stubs exist)
- cTrader Open API client (not implemented)
- External order flow data (V1 deferred)

## Key Design Decisions (Locked)

| Decision | Resolution |
|----------|-----------|
| DPR dual engines | DPRBridge handles both; router engine canonical; `DPRDualEngineRouter` |
| Two Governor classes | Referenced by module path until Phase 2 rename |
| Order flow quality | HIGH only via cTrader DOM; if unavailable, features disabled (not degraded) |
| Proxy features | Must carry `quality="proxy_inferred"` + `notes` field; not labeled as true flow |
| Phase 1 scope | SCALPER + ORB only |
| Error logic | Rejection = controlled outcome, not failure; engineering failure = exception |
| Evaluation | BotSpec → FullBacktestPipeline (existing 6-mode system) |
| Workflows | Library bridges to Prefect flows (flows are stubs) |
