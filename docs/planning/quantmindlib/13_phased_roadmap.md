# 13 — Phased Roadmap

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Phased implementation roadmap for QuantMindLib V1

---

## Phase Overview

| Phase | Name | Purpose | Dependencies |
|-------|------|---------|---------------|
| 0 | Repo Scan + Docs Recovery | Foundation: understand reality | None |
| 1 | Core Contracts / Schemas | Build the shared object model | Phase 0 |
| 2 | Capability / Compatibility Layer | Enable spec-driven composition | Phase 1 |
| 3 | Bridge Definitions | Design and specify all bridges | Phase 1 |
| 4 | cTrader Adapter Boundary | Define platform adapter interface | Phase 1, Phase 3 |
| 5 | Feature Registry V1 | Build 5 feature families | Phase 1, Phase 2 |
| 6 | Archetype System V1 | Build archetype registry and base archetypes | Phase 2, Phase 5 |
| 7 | Runtime Intent Flow | Implement bot runtime (emit → risk → exec) | Phase 3, Phase 5 |
| 8 | Evaluation Alignment | Wire evaluation pipeline | Phase 3, Phase 7 |
| 9 | Workflow Alignment | WF1 and WF2 integration | Phase 3, Phase 8 |
| 10 | Registry/DPR/Journal Alignment | DPR Redis fix + registry bridge | Phase 3, Phase 8 |
| 11 | Deferred Roadmap | Future phases and enhancements | Phase 1-10 |

---

## PHASE 0: Repo Scan + Documentation Recovery

**Duration:** Exploration only (no implementation)
**Purpose:** Understand actual codebase state; produce planning artifacts

### Deliverables
- 17 planning documents (this package)
- Codebase scan report (02_codebase_scan_report.md)
- Documentation status report (03_documentation_status_report.md)
- Recovered architecture notes (04_recovered_architecture_notes.md)
- Gap analysis (05_gap_analysis_against_memos.md)

### Acceptance Criteria
- All 17 planning documents written and committed
- No implementation code written
- Planning package is indexed and navigable
- Downstream agents can use planning docs to implement without re-scanning

### Risks
- Repo changes between scan and implementation → periodic re-scan needed
- Documentation gaps may require additional recovery during Phase 1

### Notes
- This phase is the worktree is currently executing
- All 17 documents must be committed before Phase 1 begins

---

## PHASE 1: Core Contracts and Schemas

**Duration:** TBD (depends on team size)
**Purpose:** Build the shared object model — the canonical contracts all systems will use

### Deliverables
- `src/library/core/domain/bot_spec.py` — BotSpec, BotRuntimeProfile, BotEvaluationProfile, BotMutationProfile
- `src/library/core/domain/market_context.py` — MarketContext, SessionContext, FeatureVector
- `src/library/core/domain/trade.py` — TradeIntent, ExecutionDirective, OrderFlowSignal
- `src/library/core/domain/evaluation.py` — EvaluationResult, EvaluationCriteria
- `src/library/core/domain/registry.py` — RegistryRecord, BotPerformanceSnapshot
- `src/library/core/schemas/` — Pydantic schemas for all domain objects
- `src/library/core/types/` — RegimeType, BotState, SignalStrength, FeatureConfidence enums
- `src/library/core/registries/spec_registry.py` — SpecRegistry for versioned specs

### Dependencies
- Phase 0 complete
- All 17 planning documents reviewed

### Risks
- Contract definitions may need revision as implementation reveals gaps
- BotSpec must be compatible with existing TRDDocument schema (preserve TRD → BotSpec conversion)
- Two Governor classes (C1 in architecture doc) may require resolution before RiskEnvelope is finalized

### Acceptance Criteria
- All domain objects have Pydantic schemas with validation
- BotSpec can be serialized to/from JSON and YAML
- Existing TRDDocument can be converted to BotSpec (loss-free)
- BotRegistry can be synchronized with BotSpec (bidirectional)
- All contracts are documented with producers, consumers, sync/async classification

### Indexed References
- Shared object model: `07_shared_object_model.md`
- Gap analysis: `05_gap_analysis_against_memos.md` (G-1, G-2, G-3)
- Existing equivalents: `src/trd/schema.py`, `src/router/bot_manifest.py`, `src/router/sentinel.py` (RegimeReport)

---

## PHASE 2: Capability and Compatibility System

**Duration:** TBD
**Purpose:** Enable spec-first composition validation — modules declare what they need and what they produce

### Deliverables
- `src/library/core/contracts/capability.py` — CapabilitySpec, DependencySpec, CompatibilityRule, OutputSpec
- `src/library/core/contracts/adapter.py` — IMarketDataAdapter, IExecutionAdapter, IRegistryAdapter
- `src/library/core/contracts/bridge.py` — IBridge, SyncBridge, AsyncBridge, HybridBridge
- `src/library/core/composition/validator.py` — CompositionValidator
- `src/library/core/composition/requirements.py` — RequirementResolver

### Dependencies
- Phase 1 complete (contracts needed first)

### Risks
- Overly strict validation could block valid compositions
- Capability IDs must be consistent across all modules (naming convention required)

### Acceptance Criteria
- Every module in library declares CapabilitySpec at registration
- Composer can validate BotSpec compatibility before building bot
- No module can be composed without all its dependencies satisfied
- Compatibility rules are enforced at composition time, not runtime

### Indexed References
- Memo §7 (capability-driven composition)
- Gap analysis: G-5 (CapabilitySpec missing), G-6 (feature families)
- Existing: `src/router/bot_tag_registry.py` (mutual exclusivity as reference pattern)

---

## PHASE 3: Bridge Definitions

**Duration:** TBD
**Purpose:** Design and specify all bridge adapters between library and existing systems

### Deliverables
- `src/library/bridges/to_sentinel/sentinel_bridge.py`
- `src/library/bridges/to_risk/risk_bridge.py`
- `src/library/bridges/to_registry/registry_bridge.py`
- `src/library/bridges/to_dpr/dpr_bridge.py` (with DPR Redis write fix)
- `src/library/bridges/to_evaluation/evaluation_bridge.py`
- `src/library/bridges/to_journal/journal_bridge.py`
- `src/library/bridges/to_workflows/wf1_bridge.py`, `wf2_bridge.py`
- `src/library/bridges/to_lifecycle/lifecycle_bridge.py`

### Dependencies
- Phase 1 complete (contracts needed for translation)
- Phase 2 complete (capability declarations needed for validation)

### Risks
- DPR Redis gap (G-18) must be fixed in this phase
- DPR dual engines (C2) may require resolution before DPR bridge is finalized
- Existing systems' interfaces must be preserved — bridges must adapt, not change

### Acceptance Criteria
- All 8 bridges have two-way translation (inbound + outbound)
- DPR bridge writes scores to Redis (fixes G-18)
- All bridges handle error cases gracefully (fallback behavior defined)
- Sync/async mode is explicit per bridge operation

### Indexed References
- `08_bridge_inventory.md` — full bridge specifications
- Gap analysis: G-3 (bridge design), G-18 (DPR Redis gap)
- Recovery notes: R-2 (DPR dual-engine), R-10 (Governor), R-13 (LifecycleManager)

---

## PHASE 4: cTrader Adapter Boundary

**Duration:** TBD
**Purpose:** Define platform adapter interface and implement cTrader adapter

### Deliverables
- `src/library/adapters/base.py` — PlatformAdapter ABC, AdapterConfig
- `src/library/adapters/ctrader/client.py` — CTraderClient wrapper
- `src/library/adapters/ctrader/market_adapter.py` — CTraderMarketAdapter
- `src/library/adapters/ctrader/execution_adapter.py` — CTraderExecutionAdapter
- `src/library/adapters/ctrader/types.py` — type converters
- `src/library/adapters/ctrader/converter.py` — OHLCV, tick, depth converters
- `src/library/adapters/ctrader/backtest_engine.py` — cTrader backtest engine (produces same schema as MT5BacktestResult)

### Dependencies
- Phase 1 complete (IMarketDataAdapter, IExecutionAdapter interfaces defined in Phase 2)
- cTrader Open API documentation reviewed

### Risks
- cTrader Open API may have limitations not documented
- Backtest engine must produce same schema as MT5BacktestResult — schema compatibility is critical
- MT5 migration surface is large (5 areas identified)
- T1 trading node runs Windows — adapter deployment must be Windows-compatible

### Acceptance Criteria
- IMarketDataAdapter interface is implemented by CTraderMarketAdapter
- IExecutionAdapter interface is implemented by CTraderExecutionAdapter
- cTrader backtest engine produces BacktestResult with same schema as MT5BacktestResult
- Evaluation pipeline doesn't need to change when switching from MT5 to cTrader backtest
- All adapter code is in `src/library/adapters/` — library core has no cTrader imports

### Indexed References
- `09_ctrader_boundary_plan.md` — full cTrader boundary specification
- Gap analysis: G-10, G-11 (cTrader migration surface)
- Official docs: links in memo Appendix A

---

## PHASE 5: Feature Registry V1

**Duration:** TBD
**Purpose:** Build the 5 feature families with capability declarations

### Deliverables
- `src/library/features/indicators/base.py` + RSI, ATR, MACD, VWAP
- `src/library/features/volume/base.py` + RVOL, Volume Profile, MFI, Imbalance
- `src/library/features/orderflow/base.py` + Tick Activity, Spread Behavior, Session Volume
- `src/library/features/session/base.py` + Session Detector, Session Blackout
- `src/library/features/transforms/base.py` + Normalize, Rolling, Resample
- `src/library/core/registries/feature_registry.py` — FeatureRegistry singleton

### Dependencies
- Phase 1 complete (domain objects needed)
- Phase 2 complete (CapabilitySpec needed for declarations)
- Phase 4 partial (market adapter needed for live data access)

### Risks
- SVSS indicators must be wrapped, not rewritten — preserve existing behavior
- OrderFlowFeature is entirely new development — no existing code to wrap
- Quality-aware tagging must be implemented (feed quality degradation)

### Acceptance Criteria
- All 13 feature modules are registered in FeatureRegistry
- Each module declares CapabilitySpec with requires/provides/outputs/compatibility
- CompositionValidator can validate BotSpec feature lists
- FeatureEvaluator can compute feature stack for a given bot

### Indexed References
- `10_feature_family_plan.md` — full feature family specifications
- Gap analysis: G-6 (feature families), G-7 (footprint logic)
- Existing: `src/svss/indicators/` (wrap these), `src/router/session_detector.py` (wrap), `src/market/news_blackout.py` (wrap)

---

## PHASE 6: Archetype System V1

**Duration:** TBD
**Purpose:** Build archetype registry and implement base + initial derived archetypes

### Deliverables
- `src/library/archetypes/base.py` — BaseArchetype ABC, ArchetypeSpec
- `src/library/archetypes/orb.py` — OpeningRangeBreakout (Phase 1 priority)
- `src/library/archetypes/breakout_scalper.py` — BreakoutScalper (Phase 2)
- `src/library/archetypes/pullback_scalper.py` — PullbackScalper (Phase 2)
- `src/library/archetypes/mean_reversion.py` — MeanReversion (Phase 2)
- `src/library/archetypes/session_transition.py` — SessionTransition (Phase 2)
- `src/library/archetypes/derived/` — LondonORB, NYORB, ScalperM1
- `src/library/core/registries/archetype_registry.py` — ArchetypeRegistry singleton
- `src/library/core/composition/composer.py` — Composer (builds bot from specs)
- `src/library/core/composition/mutation.py` — MutationEngine (constrained mutation)

### Dependencies
- Phase 2 complete (capability system needed)
- Phase 5 complete (feature registry needed for archetype validation)
- Phase 1 complete (domain objects needed)

### Risks
- Archetype definitions may need refinement as feature composition reveals gaps
- Phase 1 is ORB only — other archetypes are Phase 2 scope creep risk

### Acceptance Criteria
- OpeningRangeBreakout archetype is fully defined and validated
- Composer can build a working bot from BotSpec + ArchetypeSpec
- MutationEngine respects locked/allowed surfaces
- VariantRegistry lineage is maintained through mutation

### Indexed References
- `11_archetype_and_composition_plan.md` — full archetype and composition specifications
- Gap analysis: G-9 (archetypes)
- Existing: `src/router/bot_manifest.py` (StrategyType SCALPER/ORB), `src/router/variant_registry.py` (lineage)

---

## PHASE 7: Runtime Intent Flow

**Duration:** TBD
**Purpose:** Implement the runtime execution path: tick → feature → intent → risk → execution

### Deliverables
- `src/library/runtime/intent_emitter.py` — IntentEmitter (bot emits TradeIntent from spec)
- `src/library/runtime/feature_evaluator.py` — FeatureEvaluator (evaluates feature stack)
- `src/library/runtime/state_manager.py` — BotStateManager (manages runtime state, cached FeatureVector + MarketContext)
- `src/library/runtime/safety_hooks.py` — SafetyHooks (kill switch, circuit breaker integration)
- Runtime integration tests

### Dependencies
- Phase 3 complete (bridges needed for RiskBridge, SentinelBridge)
- Phase 5 complete (feature evaluator needs feature registry)
- Phase 7 needs all 8 bridges for end-to-end flow

### Risks
- Sync/async boundary violations — ensure sync reads from cached state, async writes from event streams
- Performance: cached state must be thread-safe for concurrent access
- Kill switch integration must be reliable (safety-critical)

### Acceptance Criteria
- Tick arrives via cTrader adapter → cached in BotStateManager
- FeatureEvaluator computes FeatureVector on-demand
- IntentEmitter emits TradeIntent
- RiskBridge calls Governor → returns RiskEnvelope
- ExecutionBridge sends to cTrader → returns ExecutionResult
- DPRBridge and JournalBridge record the cycle

### Indexed References
- `06_target_architecture_v1.md` (Section 8: Runtime Intent Flow)
- Recovery notes: R-1 (Sentinel), R-10 (Governor)

---

## PHASE 8: Evaluation Alignment

**Duration:** TBD
**Purpose:** Wire evaluation pipeline so library bots are evaluated through existing 6-mode system

### Deliverables
- EvaluationBridge integration with FullBacktestPipeline
- BotSpec → backtest input conversion
- 4-mode results → EvaluationResult collection
- MC/WFA results → BotEvaluationProfile attachment
- BacktestReportSubAgent integration
- PBO calculation integration

### Dependencies
- Phase 3 complete (EvaluationBridge defined)
- Phase 7 partial (IntentEmitter needed for strategy code generation)
- Phase 4 partial (cTrader backtest engine needed)

### Risks
- BotSpec → strategy code generation must be reliable (produces valid trading logic)
- Evaluation results must be identical whether triggered from library or directly from API
- PBO calculation must be accurate

### Acceptance Criteria
- BotSpec can be evaluated through FullBacktestPipeline
- EvaluationResult schema matches existing MT5BacktestResult
- BotEvaluationProfile is populated with all evaluation metrics
- BacktestReportSubAgent generates reports for library evaluations

### Indexed References
- `12_evaluation_workflow_alignment.md` — full evaluation alignment
- Existing: `src/backtesting/full_backtest_pipeline.py`, `src/agents/departments/subagents/backtest_report_subagent.py`

---

## PHASE 9: Workflow Alignment (WF1 + WF2)

**Duration:** TBD
**Purpose:** Integrate library BotSpec into Prefect workflows

### Deliverables
- WF1 bridge: TRD → BotSpec → AlphaForgeFlow
- WF1 bridge: evaluation results → paper trading handoff
- WF2 bridge: variant lineage → MutationEngine → ImprovementLoopFlow
- WF2 bridge: evaluation results → promotion/demotion/kill decision
- Workflow trigger integration (PrefectTriggerTool)

### Dependencies
- Phase 8 complete (evaluation alignment)
- Phase 6 complete (archetype + mutation system)

### Risks
- WF1/WF2 are production workflows — changes must not break existing operations
- Parallel execution: library should support WF1 and WF2 running simultaneously
- 3-day paper lag timing must be reliable

### Acceptance Criteria
- TRDDocument converts to BotSpec for WF1 input
- AlphaForgeFlow can consume BotSpec
- ImprovementLoopFlow can consume mutated BotSpec
- Library evaluation results flow back into workflow state

### Indexed References
- `12_evaluation_workflow_alignment.md` (Sections 6-7: WF1 and WF2 alignment)
- Existing: `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py`
- Memory: `WF1_SYSTEM_SCAN.md` (source of truth)

---

## PHASE 10: Registry/DPR/Journal Alignment

**Duration:** TBD
**Purpose:** Complete DPR Redis fix, integrate registry, and wire journaling

### Deliverables
- DPRBridge with Redis publisher (fixes G-18)
- RegistryBridge bidirectional sync (BotRegistry ↔ BotSpec)
- JournalBridge for audit logging
- SSLCircuitBreaker integration via SafetyHooks
- BotCircuitBreaker integration
- DPR concern event → @session_concern tag flow

### Dependencies
- Phase 3 complete (all bridges defined)
- Phase 7 partial (SafetyHooks integration)

### Risks
- DPR Redis gap fix must not break existing DPR engine behavior
- Dual DPR engines must be handled gracefully
- Registry sync must be idempotent (no double-registration)

### Acceptance Criteria
- DPR scores are written to Redis (`dpr:score:{bot_id}` keys populated)
- BotSpec registration syncs to BotRegistry
- Lifecycle decisions (promote/quarantine) flow through LifecycleBridge
- Journal entries are written for all trade events

### Indexed References
- Bridge inventory: `08_bridge_inventory.md` (BRIDGE-3, BRIDGE-4, BRIDGE-6)
- Recovery notes: R-2 (DPR dual-engine), R-5 (BotTagRegistry), R-6 (VariantRegistry), R-13 (LifecycleManager)

---

## PHASE 11: Deferred Roadmap

Items out of V1 scope but planned for future phases:

### Phase 2 (Future V2)
- **Pattern service:** Chart pattern analysis as structured service → PatternSignal
- **Multi-broker adapter framework:** Abstract beyond cTrader to support multiple brokers
- **Advanced derived archetypes:** LondonORB + volume confirmation, OverlapBreakout + orderflow filter
- **cTrader Network Access integration:** Economic calendar via cTrader instead of direct Finnhub
- **Chart rendering:** Optional footprint/VWAP rendering for debugging
- **External order flow adapter:** IExternalOrderFlowAdapter implementation for true executed trade-flow data (Category B features: VolumeImbalance, TickActivity). cTrader Open API does not provide buy/sell volume. External provider required.
- **Category B feature activation:** Once external order flow data source is connected, VolumeImbalanceFeature and TickActivityFeature become V1-active with HIGH quality.

### Phase 3+ (Future V3+)
- **ML feature integration:** HMM/BOCPD/MS-GARCH output consumption as features
- **Evolutionary layer:** Automated strategy generation from archetype specs
- **Cross-platform abstraction:** Full broker abstraction layer for broker switching
- **Real-time portfolio optimization:** Multi-bot portfolio Kelly scaling

---

## Phase Dependency Diagram

```
Phase 0 (scan) ──────────────────────────────────────────────────────────┐
                                                                        │
Phase 1 (contracts) ──────────────────────────────────────────┐          │
                                                               │          │
Phase 2 (capability) ─────────────────────────────────┐       │          │
                                                        │       │          │
Phase 3 (bridges) ─────────────────────────┐           │       │          │
                                           │           │       │          │
Phase 4 (cTrader adapter) ───┐            │           │       │          │
                              │            │           │       │          │
Phase 5 (features) ─────┐    │            │           │       │          │
                         │    │            │           │       │          │
Phase 6 (archetypes) ───┴────┤            │           │       │          │
                              │            │           │       │          │
Phase 7 (runtime) ────────────┴────────────┤           │       │          │
                                           │           │       │          │
Phase 8 (evaluation) ─────────────────────┴────────────┤       │          │
                                                       │       │          │
Phase 9 (workflows) ────────────────────────────────────┴───────┤          │
                                                               │          │
Phase 10 (registry/DPR/journal) ────────────────────────────────┴──────────┘
```

---

## Critical Path (Must be Sequential)

The minimum viable library that enables a bot to go from spec to live trading:

```
Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6 → Phase 4 → Phase 7 → Phase 8 → Phase 10
(contracts)  (capability) (bridges) (features) (archetypes) (adapter) (runtime) (eval)   (DPR)
```

**Note:** Phase 4 (cTrader adapter) can start in parallel with Phase 5-7 once Phase 2 is done, since adapter uses interfaces defined in Phase 2.