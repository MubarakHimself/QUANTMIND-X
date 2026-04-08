# 05 — Gap Analysis: Memos vs Codebase

**Analysis date:** 2026-04-08
**Method:** Cross-reference memo recommendations against actual codebase scan

---

## How to Read This Document

Each item is mapped from memo recommendation to codebase reality. Alignment statuses:

| Status | Meaning |
|--------|---------|
| **ALIGNED** | Memo concept matches existing code exactly |
| **PARTIALLY ALIGNED** | Concept exists but with gaps or differences |
| **BRIDGE-ONLY** | Concept exists as integration target, not library internal |
| **MISSING** | Concept not implemented; must be created |
| **REFACTOR-NEEDED** | Existing code conflicts with memo recommendation |
| **CONFLICTING** | Code and memo explicitly contradict each other |
| **OUT-OF-SCOPE** | Correctly outside V1 scope |

---

## Part 1: Shared Semantic Contracts (Memo §5, §11)

### G-1: Shared Object Model as Center

| Aspect | Memo Recommendation | Codebase Reality | Status | Action |
|--------|---------------------|-------------------|--------|--------|
| Central shared object model | All major systems communicate via shared schemas | No centralized shared model; systems communicate via ad-hoc interfaces | **MISSING** | Create shared object model (priority P1) |
| Bot specification multi-profile | BotSpec = static + runtime + evaluation + mutation profiles | BotManifest (static) + DPR scores (evaluation) + VariantRegistry (mutation) exist separately | **PARTIALLY ALIGNED** | Unify into BotSpec with 4 profiles |
| TradeIntent | Central output of bot decision-making | No TradeIntent dataclass; execution via Governor/Commander pipeline | **MISSING** | Define TradeIntent schema |
| RiskEnvelope | Risk system output translated for library | RiskMandate exists (router/governor.py) but not as RiskEnvelope | **BRIDGE-ONLY** | Bridge RiskMandate to RiskEnvelope |
| ExecutionDirective | Governor's decision on what to execute | Commander handles execution; no ExecutionDirective schema | **MISSING** | Define ExecutionDirective schema |

### G-2: BotSpec Multi-Profile Structure

| Field | Memo | Codebase | Status |
|-------|------|----------|--------|
| id, archetype, symbol_scope, sessions, features, confirmations, execution_profile, dependencies | BotSpec static profile | BotManifest (similar fields but no features/archetype) | **PARTIALLY ALIGNED** |
| activation_state, deployment_target, health, session_eligibility, DPR_ranking, journal_ids | BotRuntimeProfile | BotRegistry tracks mode/status; DPR tracked separately | **PARTIALLY ALIGNED** |
| backtest/MC/WFA metrics, robustness_score, spread_sensitivity | BotEvaluationProfile | StrategyPerformance DB model; not attached to BotSpec | **BRIDGE-ONLY** |
| allowed_mutation_areas, locked_components, lineage, parent_variants | BotMutationProfile | VariantRegistry (lineage, parameters); no mutation boundary declarations | **PARTIALLY ALIGNED** |

---

## Part 2: Bridge Layer (Memo §6, §7)

### G-3: Bidirectional Bridge Design

| Bridge Target | Memo (Inbound) | Memo (Outbound) | Codebase Reality | Status |
|--------------|---------------|-----------------|------------------|--------|
| **Sentinel/market intelligence** | session_status, news_state, regime_state, routing_state, market_health | feature_summaries, pattern_outputs, bot_requests, feedback_loops | Sentinel → RegimeReport (inbound); no outbound features back to sentinel | **PARTIALLY ALIGNED** |
| **Risk/position sizing** | risk_envelopes, allowed_exposure, sizing_responses, stop/target rules | trade_intents, required_metadata, execution_urgency, session/archetype_tags | Governor → RiskMandate (outbound); no TradeIntent (inbound) | **PARTIALLY ALIGNED** |
| **Registry/DPR/journal** | bot_metadata, versioning, runtime_state, evaluation_state, tagging, lineage, promotion_status | evaluation_results, profile_updates, promotion_recommendations | DPR → BotRegistry (one-way); no journal bridge | **PARTIALLY ALIGNED** |
| **Evaluation workflows** | standard_bot_inputs, parameter_surfaces, fitness/robustness_outputs, mutation_constraints | evaluation_results, normalized_outputs | 6-mode backtest → StrategyPerformance (one-way) | **PARTIALLY ALIGNED** |

### G-4: Sync/Async Distinction

| Memo Pattern | Codebase Reality | Status | Action |
|-------------|------------------|--------|--------|
| Async: tick streams, depth updates, event ingestion, news notifications, registry notifications, audit/log streams | Sentinel (async shadow), SSLCircuitBreaker (async Redis), SVSSEngine (async), Layer3KillSwitch (async) | **ALIGNED** (existing async patterns) | Document in bridge spec |
| Sync: validation calls, compatibility checks, static spec parsing, deterministic parameter loading, one-shot calculations, promotion decisions | RiskGovernor, PositionSizing, DPR (router layer sync), LifecycleManager, KillSwitch | **ALIGNED** (existing sync patterns) | Document in bridge spec |
| Mixed mode: async stream → cached state store → sync read at decision time | Sentinel tick → RegimeReport cache → Governor sync read | **ALIGNED** (existing pattern) | Document as canonical pattern |

---

## Part 3: Capability-Driven Composition (Memo §7)

### G-5: CapabilitySpec / DependencySpec / CompatibilityRule

| Component | Memo | Codebase | Status | Action |
|-----------|------|----------|--------|--------|
| CapabilitySpec | Module declares what it can do | No CapabilitySpec anywhere | **MISSING** | Create in library core |
| DependencySpec | Module declares what it requires | No DependencySpec; bot dependencies declared in TRDDocument as strings | **MISSING** | Create in library core |
| CompatibilityRule | What module can/cannot combine with | BotTagRegistry has mutual exclusivity; no compatibility between features | **MISSING** | Create in library core |
| Feature requirement declaration | `OrderFlowFeature.requires = [TickStream, VolumeStream]` | No feature requirement declarations | **MISSED** | Create feature registry with requirements |
| OutputSpec | Schemas a module emits | No OutputSpec system | **MISSING** | Create in library core |

### G-6: Feature Families

| Feature Family | Memo | Codebase | Status |
|----------------|------|----------|--------|
| indicators | Reusable, parameterized, stateless | FeatureConfig in HMM only; SVSS has VWAP/RVOL/VolumeProfile/MFI | **PARTIALLY ALIGNED** |
| volume | Volume distribution, imbalance, delta behavior, pressure at levels | SVSS VolumeProfileIndicator; no volume feature registry | **PARTIALLY ALIGNED** |
| orderflow | Order flow proxy signals | No order flow feature; approximated from tick activity | **MISSING** |
| session | Session-aware features | SVSS session context; SessionDetector; session_tags in TRD | **PARTIALLY ALIGNED** |
| transforms | Data transformations | No transform feature family defined | **MISSING** |

---

## Part 4: Feature Registry (Memo §8, §9)

### G-7: Footprint Logic vs Charts

| Aspect | Memo | Codebase | Status |
|--------|------|----------|--------|
| Footprint logic | Volume distribution, imbalance, delta, pressure at levels | SVSS VolumeProfileIndicator (POC, value area); no delta calculation | **PARTIALLY ALIGNED** |
| Footprint chart rendering | NOT required | No rendering code | **OUT-OF-SCOPE** (correctly) |
| Footprint-derived signals | Consumable by bots, workflows, ML systems | No signal output from volume profile | **MISSING** |

### G-8: Pattern Analysis Placement

| Aspect | Memo | Codebase | Status |
|--------|------|----------|--------|
| Chart pattern analysis | Not forced into bot code; reserved as heavier service | No pattern analysis service | **MISSING** (correctly deferred) |
| Pattern service → PatternSignal | Future integration path | No PatternSignal schema | **OUT-OF-SCOPE** (correctly) |
| PatternService consumable by bots AND sentinel | Both consumers use same outputs | Sentinel has no pattern service integration | **OUT-OF-SCOPE** (correctly) |

---

## Part 5: Strategy Archetypes (Memo §10)

### G-9: Base vs Derived Archetypes

| Aspect | Memo | Codebase | Status |
|--------|------|----------|--------|
| Base archetypes (hand-defined) | breakout_scalper, pullback_scalper, ORB, mean_reversion, session-transition | StrategyType enum: SCALPER, ORB (Phase 1 active); STRUCTURAL, SWING, HFT, NEWS, GRID (legacy) | **PARTIALLY ALIGNED** |
| Derived archetypes | London ORB + volume confirmation; overlap breakout + order-flow filter; choppy-session reversion | No derived archetype system; only flat enum | **MISSING** |
| Archetype composition from base + feature families + constraints | New archetypes without turning every indicator combo into first-class design | No composition system | **MISSING** |

---

## Part 6: cTrader Boundary (Memo §16)

### G-10: cTrader Adapter Requirements

| Requirement | Memo | Codebase | Status |
|-------------|------|----------|--------|
| cTrader contracts for bars, ticks, market depth | Adapter inputs from cTrader | MT5 contracts currently in use (mt5_engine.py, data/brokers/mt5_socket_adapter.py) | **REFACTOR-NEEDED** |
| Two-layer design: adapter outside, core vendor-neutral | Core contracts stay platform-agnostic | No platform adapter pattern yet | **MISSING** |
| Async and sync interfaces | cTrader supports both event-driven and direct API | Current code is sync-first (MT5 socket adapter) | **MISSING** |
| IC Markets Raw assumptions only in adapter, not domain | Deployment notes vs core domain separation | No such separation | **MISSING** |
| Platform-aware without becoming platform-bound | cTrader adapter target, keep core neutral | MT5 embedded throughout (backtesting engine simulates MQL5) | **REFACTOR-NEEDED** |

### G-11: Platform Migration Surface

| Current Platform | Target Platform | Migration Effort | Status |
|-----------------|-----------------|-------------------|--------|
| MT5 socket adapters | cTrader adapters | `src/data/brokers/`, `src/risk/integrations/`, `src/risk/pipeline/` | **REFACTOR-NEEDED** |
| MQL5 EA code generation | cTrader algo code | `src/mql5/` | **REFACTOR-NEEDED** |
| MT5 backtest engine | cTrader backtest engine | `src/backtesting/mt5_engine.py` | **REFACTOR-NEEDED** |
| MT5 trading node (T1 on Kamatera) | cTrader trading node | No platform abstraction yet | **MISSING** |

---

## Part 7: Workflow Alignment (Memo §13)

### G-12: WF1 Alignment

| Memo Requirement | Codebase Reality | Status |
|-----------------|-------------------|--------|
| Same contracts for source → TRD → BotSpec | TRD → BotManifest (not BotSpec yet) | **PARTIALLY ALIGNED** |
| Generate vanilla and spiced variants | 6-mode backtest (4 primary modes) | **ALIGNED** |
| Run normal backtests | FullBacktestPipeline with 4 variants | **ALIGNED** |
| Run Monte Carlo | MonteCarloSimulator | **ALIGNED** |
| Run walk-forward / out-of-sample | WalkForwardOptimizer | **ALIGNED** |
| Write structured reports into evaluation schemas | BacktestReportSubAgent generates markdown; StrategyPerformance stores results | **PARTIALLY ALIGNED** |

### G-13: WF2 Alignment

| Memo Requirement | Codebase Reality | Status |
|-----------------|-------------------|--------|
| Take surviving variants + specs | VariantRegistry + BotVariant | **ALIGNED** |
| Use mutation rules to create improved variants | No explicit mutation rule system; manual improvement | **PARTIALLY ALIGNED** |
| Re-run same evaluation boundary | Same 6-mode backtest pipeline | **ALIGNED** |
| Promote/demote/kill based on normalized evaluation outputs | LifecycleManager with DPR threshold | **ALIGNED** |
| Monte Carlo = evaluation, NOT mutation | MonteCarloSimulator correctly separates eval from mutation | **ALIGNED** |

---

## Part 8: What Library Must NOT Own (Memo §12)

### G-14: Existing Systems Protected from Library Absorption

| System | Memo | Codebase | Status |
|--------|------|----------|--------|
| Enhanced Kelly / house-of-money / fee monitoring | Don't absorb | `src/position_sizing/`, `src/router/house_money.py` | **ALIGNED** — correctly preserved |
| Kill switch internals | Don't absorb | `src/router/kill_switch.py`, `progressive_kill_switch.py` | **ALIGNED** — correctly preserved |
| Session routing internals | Don't absorb | `src/router/session_detector.py`, `inter_session_cooldown.py` | **ALIGNED** — correctly preserved |
| Market intelligence ML internals | Don't absorb | `src/risk/physics/` (Ising, HMM, BOCPD, MS-GARCH) | **ALIGNED** — correctly preserved |
| Backtest report generation internals | Don't absorb | `src/agents/departments/subagents/backtest_report_subagent.py` | **ALIGNED** — correctly preserved |
| DPR logic | Don't absorb | `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py` | **ALIGNED** — correctly preserved |
| Bot registry internals | Don't absorb | `src/router/bot_manifest.py`, `bot_tag_registry.py` | **ALIGNED** — correctly preserved |

---

## Part 9: Library Core Package Structure (Memo §15)

### G-15: Recommended Repository Boundary vs Reality

| Memo Structure | Codebase Reality | Status |
|----------------|-----------------|--------|
| `library/core/` (domain, contracts, schemas, registries, types) | `src/library/` exists (stub) | **REFACTOR-NEEDED** |
| `library/features/` (indicators, volume, orderflow, session, transforms, patterns) | SVSS (`src/svss/`) + FeatureConfig (HMM) + asset_library (`src/mcp_tools/`) | **BRIDGE-ONLY** |
| `library/archetypes/` | No archetype system | **MISSING** |
| `library/composition/` | No composition system | **MISSING** |
| `library/runtime/` | No runtime module | **MISSING** |
| `library/adapters/` (ctrader/) | No adapter layer | **MISSING** |
| `library/bridges/` (to_risk_engine, to_sentinel, to_registry, to_evaluation, to_journal, to_workflows) | No bridge layer | **MISSING** |
| `platform/` (all existing systems as integration targets) | Already exists as `src/router/`, `src/risk/`, `src/backtesting/`, etc. | **ALIGNED** |

---

## Part 10: Order Flow Quality-Aware Design (Memo §5 / trading_platform_decision_memo)

### G-16: Order Flow Quality Tagging

| Aspect | Memo | Codebase | Status |
|--------|------|----------|--------|
| Order flow as enhancer vs core edge distinction | Treat as enhancer; quality-aware feature tagging | No quality tagging system | **MISSING** |
| Feeds tagged by quality and feature confidence | Same strategy degrades gracefully when better data unavailable | No feed quality metadata | **MISSING** |
| Order-flow logic allowed; chart rendering not required | Confirmed | SVSS indicators exist; no order flow quality tagging | **PARTIALLY ALIGNED** |

---

## Part 11: Shared Event Stream (Memo §7 / trading_platform_decision_memo)

### G-17: One Shared Ingress vs Isolated Collectors

| Aspect | Memo | Codebase | Status |
|--------|------|----------|--------|
| Shared market data ingress, normalized, distributed to all consumers | One shared stream over isolated collectors | DataManager (`src/data/data_manager.py`) attempts this; Sentinel + SVSS both consume | **PARTIALLY ALIGNED** |
| Bot introspection vs execution authorization separation | Bots propose, routing/risk layer decides | Commander handles execution; Governor handles risk authorization | **ALIGNED** |

---

## Part 12: DPR Redis Gap (Not in Memos — Codebase Finding)

### G-18: DPR Redis Gap (Updated)

| Aspect | Codebase Reality | Impact | Status |
|--------|-----------------|--------|--------|
| Router DPR engine writes `dpr:score:{bot_id}` to Redis | `DprScoringEngine` (router) now has `_write_score_to_redis()` with 24h TTL | FIXED in uncommitted code | **ALIGNED** |
| Risk layer DPR engine writes `session_concern:{magic_number}` to Redis | `DPRScoringEngine` (risk) writes session concern counters — different key pattern | DPR scores still not published from risk layer engine | **PARTIALLY ALIGNED** |

**Note:** The router layer DPR Redis publish is already implemented in the uncommitted `dpr_scoring_engine.py`. The risk layer DPR engine does NOT write DPR composite scores to Redis — it only manages session concern counters. DPR bridge must still unify both engines.

---

## Part 13: Data Storage Architecture (DuckDB WARM Tier)

### G-19: DuckDB WARM Tier Market Data (Not in Memos — Codebase Finding)

| Aspect | Codebase Reality | Impact | Status |
|--------|-----------------|--------|--------|
| 3-tier market data storage | HOT: Redis tick cache (1h); WARM: DuckDB (30d); COLD: Parquet | Documents an existing 3-tier tiering strategy not captured in memos | **MISSING** (documentation) |
| DuckDB WARM tier | `src/database/duckdb/market_data.py` — `market_data` table with OHLCV + tick_volume + spread + is_synthetic | WARM tier bridges HOT and COLD; DuckDB vs SQLAlchemy distinction | **MISSING** (documentation) |
| No DuckDB tiering doc | Architecture memos don't document the 3-tier HOT/WARM/COLD strategy | Future agents may not understand the tiering rationale | **MISSING** (documentation) |

---

## Summary Gap Matrix

| Category | Aligned | Partially Aligned | Bridge-Only | Missing | Refactor-Needed | Out-of-Scope |
|----------|---------|------------------|-------------|---------|-----------------|-------------|
| Shared contracts | 3 | 4 | 2 | 4 | 0 | 0 |
| Bridge layer | 3 | 4 | 0 | 0 | 0 | 0 |
| Capability/composition | 0 | 0 | 0 | 5 | 0 | 0 |
| Feature families | 1 | 2 | 0 | 2 | 0 | 2 |
| Archetypes | 1 | 1 | 0 | 2 | 0 | 0 |
| cTrader boundary | 0 | 0 | 0 | 3 | 3 | 0 |
| Workflow alignment | 4 | 3 | 0 | 0 | 0 | 0 |
| Library protection | 7 | 0 | 0 | 0 | 0 | 0 |
| Library structure | 1 | 0 | 1 | 5 | 1 | 0 |
| Order flow | 1 | 1 | 0 | 1 | 0 | 1 |
| Shared event stream | 1 | 1 | 0 | 0 | 0 | 0 |
| DPR Redis | 1 | 1 | 0 | 0 | 0 | 0 |
| Data storage (DuckDB) | 0 | 0 | 0 | 1 | 0 | 0 |
| **TOTAL** | **24** | **17** | **3** | **24** | **4** | **3** |

### Priority Actions (by gap count)

| Priority | Gap Count | Focus |
|----------|-----------|-------|
| P1 (Critical) | 24 missing | Shared object model, capability system, feature families, archetype system, DPR Redis (router fixed, risk layer partial), cTrader adapters, DuckDB WARM tier doc |
| P2 (High) | 16 partial | BotSpec multi-profile unification, bridge completeness, workflow contract alignment |
| P3 (Medium) | 4 refactor | cTrader migration surface, library package restructuring |
| P4 (Low) | 3 out-of-scope | Pattern analysis deferred, footprint charts not required |

---

## Traceability Matrix

| Gap ID | Memo Reference | Codebase Reference | Recovery Note |
|--------|---------------|-------------------|---------------|
| G-1 | §5, §11 | No shared model found | - |
| G-2 | §11 | `BotManifest`, `TRDDocument`, `VariantRegistry` | Multi-profile unification needed |
| G-3 | §6 | `RegimeReport`, `RiskMandate`, `DprScore`, `StrategyPerformance` | Bridge adapters needed |
| G-4 | §6 | Sentinel, SSLCircuitBreaker, SVSS, Layer3KillSwitch | Already async; document pattern |
| G-5 | §7 | No capability system | New creation |
| G-6 | §8 | SVSS, FeatureConfig, asset_library | Bridge existing to new registry |
| G-7 | §8 | SVSS VolumeProfileIndicator | Add signal output |
| G-8 | §9 | No pattern service | Correctly deferred |
| G-9 | §10 | StrategyType enum | Archetype system needed |
| G-10 | §16 | MT5 adapters everywhere | Migration planning required |
| G-11 | §16 | MT5 throughout | Migration planning required |
| G-12 | §13 | FullBacktestPipeline, BacktestReportSubAgent | Align contracts |
| G-13 | §13 | VariantRegistry, LifecycleManager | Align mutation rules |
| G-14 | §12 | All existing systems | Correctly preserved |
| G-15 | §15 | `src/library/` (stub) | Full restructuring needed |
| G-16 | trading_platform_decision_memo §5 | No quality tagging | New creation |
| G-17 | trading_platform_decision_memo §7 | DataManager, Sentinel, SVSS | Already aligned |
| G-18 | Codebase finding | DPR engines (router + risk) | Router DPR Redis write fixed in uncommitted code; risk layer DPR still needs extension |
| G-19 | Codebase finding | `src/database/duckdb/market_data.py` | 3-tier HOT/WARM/COLD tiering documented; DuckDB WARM tier needs architecture doc |