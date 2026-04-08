# 15 — Short Implementation Specs

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Handoff specs for downstream implementation agents

---

## SPEC-001: BotSpec Multi-Profile Contract

### Title
Define BotSpec as the canonical bot specification with 4 profiles

### Scope
Create `src/library/core/domain/bot_spec.py` with:
- `BotSpec` dataclass (static profile)
- `BotRuntimeProfile` dataclass
- `BotEvaluationProfile` dataclass
- `BotMutationProfile` dataclass

### Non-Scope
- Bridge adapters (separate spec)
- Feature registry (separate spec)
- Archetype system (separate spec)
- Runtime execution (separate spec)

### Why It Exists
BotSpec is the central contract. All systems (registry, DPR, evaluation, workflows, lifecycle) must share the same bot description format. Currently, bots are described by separate objects (BotManifest, TRDDocument, DPR scores, VariantRegistry) with no unified schema.

### Affected Files/Modules Likely Involved
- `src/trd/schema.py` — TRDDocument (input for BotSpec)
- `src/router/bot_manifest.py` — BotManifest (current partial equivalent)
- `src/router/variant_registry.py` — BotVariant (lineage tracking)
- `src/router/lifecycle_manager.py` — lifecycle decisions

### Contracts/Interfaces to Respect
- TRDDocument → BotSpec conversion must be loss-free
- BotSpec must serialize to JSON/YAML (versioned)
- BotManifest existing fields must map into BotSpec

### Dependencies
- CONTRACT-017 (type enums) must be done first
- CONTRACT-018-021 (capability system) can be done in parallel

### Implementation Notes
- Use Pydantic v2 for validation (consistent with existing codebase)
- BotSpec should be a frozen dataclass (immutable static profile)
- Runtime/evaluation/mutation profiles are mutable
- Include `to_dict()` and `from_dict()` for serialization

### Test Expectations
- TRDDocument round-trip: TRD → BotSpec → TRD (loss-free)
- JSON serialization round-trip: BotSpec → JSON → BotSpec
- All 4 profile fields populated from existing equivalents

### Risks
- BotSpec becomes too large → split into composable sub-objects
- TRD → BotSpec conversion may reveal undocumented TRD fields

### Review/Handoff Notes
- Check TRD schema carefully for fields not yet mapped
- Two Governor classes (C1) may affect RiskEnvelope field names

### Indexed References
- Memo §11 (BotSpec multi-profile)
- Gap: G-2 (BotSpec multi-profile)
- Existing: `src/trd/schema.py`, `src/router/bot_manifest.py`

---

## SPEC-002: CapabilitySpec / DependencySpec / CompatibilityRule System

### Title
Build the spec-first composition validation system

### Scope
Create in `src/library/core/contracts/`:
- `CapabilitySpec` — declares what module provides/requires/outputs
- `DependencySpec` — declares module dependencies
- `CompatibilityRule` — declares composition rules
- `OutputSpec` — declares output schemas

Create in `src/library/core/composition/`:
- `CompositionValidator` — validates BotSpec composition
- `RequirementResolver` — resolves dependencies

### Non-Scope
- Feature implementation (separate spec)
- Archetype implementation (separate spec)
- Runtime execution (separate spec)

### Why It Exists
The memo requires modules declare what they need and what they produce. This enables automatic composition validation — bots can't be composed if their dependencies aren't satisfied. Currently, no such system exists.

### Affected Files/Modules Likely Involved
- `src/router/bot_tag_registry.py` — mutual exclusivity (reference pattern)
- `src/router/bot_manifest.py` — BotManifest dependencies (string list, no structured spec)

### Contracts/Interfaces to Respect
- CapabilitySpec must be self-contained (no circular dependencies)
- CompatibilityRule must be declarative (not imperative)

### Dependencies
- CONTRACT-001 (BotSpec) must be done first
- CONTRACT-022-023 (adapter/bridge interfaces) can be done in parallel

### Implementation Notes
- Capability IDs should follow dotted notation: `data/tick_stream`, `indicators/rsi`, `orderflow/tick`
- Compatibility rules should support conditions: `COMBINES_WITH`, `EXCLUDES`, `REQUIRES`
- Validator should raise specific exceptions (not generic)

### Test Expectations
- Valid composition passes validation
- Missing dependency raises DependencyMissingError
- Incompatible feature+archetype raises CompatibilityError
- Circular dependency raises CircularDependencyError

### Risks
- Overly strict validation blocks valid compositions
- Naming convention inconsistency across modules

### Review/Handoff Notes
- Establish naming convention before implementing (dotted path vs simple names)
- Consider bot-builder agent needs when designing validation messages

### Indexed References
- Memo §7 (capability-driven composition)
- Gap: G-5 (CapabilitySpec missing)
- Existing: `src/router/bot_tag_registry.py` (mutual exclusivity pattern)

---

## SPEC-003: SentinelBridge

### Title
Bridge between Sentinel's RegimeReport and library's MarketContext

### Scope
Create `src/library/bridges/to_sentinel/`:
- `sentinel_bridge.py` — SentinelBridge class
- `types.py` — RegimeReport → MarketContext converter
- `config.py` — SentinelBridgeConfig

Bidirectional:
- **Inbound:** RegimeReport → MarketContext (cached), SentinelState (event)
- **Outbound:** Feature summaries → Sentinel (feedback), Bot requests → Sentinel

### Non-Scope
- Sentinel implementation changes
- MarketScanner integration
- CalendarGovernor integration

### Why It Exists
Sentinel is the primary source of market regime information. The library needs a normalized MarketContext derived from RegimeReport. Currently, RegimeReport is consumed directly by Governor/KillSwitch/DPR — no library bridge.

### Affected Files/Modules Likely Involved
- `src/router/sentinel.py` — RegimeReport (source)
- `src/router/multi_timeframe_sentinel.py` — MultiTimeframeSentinel
- `src/risk/physics/` — sensor outputs
- `src/market/news_blackout.py` — NewsSensor state

### Contracts/Interfaces to Respect
- RegimeReport dataclass fields (do not change sentinel)
- Async event-driven regime updates
- Sync cache reads at decision time

### Dependencies
- CONTRACT-002 (MarketContext) must be done first
- CONTRACT-001 (BotSpec) should be done (for feature summaries)

### Implementation Notes
- RegimeReport → MarketContext conversion must preserve all fields
- Cache must be thread-safe (concurrent tick processing)
- Fallback: if Sentinel unavailable, use UNCERTAIN regime with is_stale flag

### Test Expectations
- RegimeReport with TREND_STABLE → MarketContext with correct regime
- RegimeReport with HIGH_CHAOS → MarketContext with is_systemic_risk flag
- NewsSensor KILL_ZONE → MarketContext.news_state = KILL_ZONE
- HMM regime included when available

### Risks
- Sentinel unavailable → bot continues with stale regime (may be dangerous)
- Regime classification lag → MarketContext may lag real market

### Review/Handoff Notes
- Recovery note R-1 documents full Sentinel architecture
- MultiTimeframeSentinel voting logic must be preserved

### Indexed References
- Memo §6 (bridge design)
- Recovery: R-1 (Sentinel regime detection)
- Gap: G-3 (Sentinel bridge)

---

## SPEC-004: DPRBridge with Redis Publisher (DPR Redis Gap Fix)

### Title
Bridge DPR scoring engines to library + fix DPR Redis write gap

### Scope
Create `src/library/bridges/to_dpr/`:
- `dpr_bridge.py` — DPRBridge class
- `score_publisher.py` — DPR Redis publisher (CRITICAL: fixes G-18)
- `tier_mapper.py` — DPR tier → BotTier mapping

Bidirectional:
- **Inbound:** DprScore → BotRuntimeProfile.dpr_score, DPRComponentScores → BotEvaluationProfile
- **Outbound:** BotPerformanceSnapshot → DPRScoringEngine → **Redis publish**

### Non-Scope
- DPR scoring algorithm changes
- DPR dual engine resolution (handle both, don't resolve)

### Why It Exists
DPR drives bot lifecycle (promotion at >= 50, quarantine on concern). The library must translate DPR scores into BotRuntimeProfile. Critical: DPR Redis gap (G-18) means scores are computed but not written to Redis. This bridge MUST fix that.

### Affected Files/Modules Likely Involved
- `src/router/dpr_scoring_engine.py` — DprScoringEngine (router layer)
- `src/risk/dpr/scoring_engine.py` — DPRScoringEngine (risk layer)
- `src/events/dpr.py` — DPRScoreEvent, DPRConcernEvent
- Redis (dpr:score:{bot_id} keys)

### Contracts/Interfaces to Respect
- DprScore 0-100 composite, 4-component weighting (25/30/20/25)
- T1 >= 80, T2 >= 50, T3 < 50 thresholds
- DPRConcernEvent (>20 point WoW drop → @session_concern)

### Dependencies
- CONTRACT-008 (BotRuntimeProfile) must be done first
- CONTRACT-009 (BotEvaluationProfile) should be done in parallel
- DPR-001 (DPR Redis publisher) is part of this spec

### Implementation Notes
- DPR score publisher must write to Redis: `dpr:score:{bot_id}` with 24h TTL
- JSON payload: `{"composite_score": float, "components": dict, "tier": str, "timestamp": str}`
- Handle both DPR engines gracefully (router + risk layers)
- Redis write failures: retry 3 times, log and continue

### Test Expectations
- BotPerformanceSnapshot → DPR score computed → Redis written
- DPR concern detection (>20 point WoW drop) → DPRConcernEvent emitted
- DPR tier mapping: score >= 80 → TIER_1, etc.

### Risks
- DPR dual engines: both may try to write → idempotent writes needed
- Redis unavailable → scores not written → downstream Redis consumers fail

### Review/Handoff Notes
- Gap G-18 is the critical bug this fixes
- Recovery note R-2 documents both DPR engines
- Recovery note R-16 documents tier thresholds

### Indexed References
- Memo §6 (bridge design), §12 (preserve DPR internals)
- Gap: G-18 (DPR Redis gap), C2 (DPR dual engines)
- Recovery: R-2 (DPR dual-engine), R-16 (DPR tier thresholds)

---

## SPEC-005: CTrader Market Adapter

### Title
Implement IMarketDataAdapter for cTrader platform

### Scope
Create `src/library/adapters/ctrader/`:
- `client.py` — CTraderClient (Open API Python SDK wrapper)
- `market_adapter.py` — CTraderMarketAdapter (IMarketDataAdapter)
- `types.py` — cTrader type conversions
- `converter.py` — OHLCV, tick, depth converters
- `config.py` — CTraderAdapterConfig

### Non-Scope
- Execution adapter (separate spec)
- Backtest engine (separate spec)
- Kill switch adapter (separate spec)
- cTrader Network Access integration

### Why It Exists
cTrader is the target platform (IC Markets cTrader Raw). The library needs a market data adapter that provides ticks, bars, depth, and historical data from cTrader. Currently, MT5 adapters exist but must be replaced.

### Affected Files/Modules Likely Involved
- `src/data/brokers/mt5_socket_adapter.py` — reference for adapter pattern
- `src/data/data_manager.py` — DataManager (consumer)
- `src/router/sentinel.py` — Sentinel (tick consumer)

### Contracts/Interfaces to Respect
- IMarketDataAdapter interface (defined in CONTRACT-022)
- Async iterator pattern for tick_stream, bar_stream, depth_stream
- Sync get_historical() for historical data
- Error handling: reconnect with exponential backoff

### Dependencies
- CTRADER-001 (IMarketDataAdapter interface) must be done first
- CONTRACT-022 (adapter interface) must be done first

### Implementation Notes
- Use cTrader Open API Python SDK
- Network access: WebSocket for streaming, HTTP for requests
- Reconnection: exponential backoff, max 5 retries
- Config: IC Markets Raw specific (commission, pip values, min lot)

### Test Expectations
- tick_stream yields TickEvent for subscribed symbols
- bar_stream yields BarEvent for subscribed timeframes
- depth_stream yields DepthEvent
- get_historical returns bars for date range

### Risks
- cTrader API limitations not documented
- WebSocket reconnection may cause tick gap
- IC Markets Raw specific config may change

### Review/Handoff Notes
- Use official docs: https://help.ctrader.com/open-api/python-SDK/python-sdk-index/
- Use Context7 for cTrader API docs if available
- Backtest engine (SPEC-011) depends on this

### Indexed References
- Memo §16 (cTrader boundary)
- Gap: G-10 (cTrader migration surface)
- Official docs: links in memo Appendix A

---

## SPEC-006: FeatureRegistry and 13 Feature Modules

### Title
Build V1 feature families with capability declarations

### Scope
Create `src/library/features/` with all 13 feature modules:
- indicators: RSI, ATR, MACD, VWAP
- volume: RVOL, Volume Profile, MFI, Imbalance
- orderflow: Tick Activity, Spread Behavior, Session Volume
- session: Session Detector, Session Blackout
- transforms: Normalize, Rolling, Resample

Create `src/library/core/registries/feature_registry.py`.

Each module declares CapabilitySpec with provides/requires/outputs/compatibility.

### Non-Scope
- Pattern analysis (deferred)
- Chart rendering (not required)
- Advanced features (Phase 2+)

### Why It Exists
The library needs reusable, parameterized feature modules. Currently, features exist in SVSS (indicators) and HMM (FeatureConfig) but not as a unified registry. Order flow features are entirely new.

### Affected Files/Modules Likely Involved
- `src/svss/indicators/` — wrap VWAP, RVOL, Volume Profile, MFI
- `src/router/session_detector.py` — wrap SessionDetector
- `src/market/news_blackout.py` — wrap NewsBlackoutService
- `src/risk/physics/hmm/features.py` — FeatureConfig (reference)

### Contracts/Interfaces to Respect
- FeatureModule ABC interface (compute(context) → result)
- CapabilitySpec declarations for each module
- Quality-aware tagging for volume/order flow features

### Dependencies
- CONTRACT-018-021 (capability system) must be done first
- FEATURE-001 (FeatureRegistry) must be done first

### Implementation Notes
- Wrap SVSS indicators: do not rewrite (preserve behavior)
- VolumeImbalanceFeature is entirely new (no existing code)
- Order flow features must include FeatureConfidence tagging
- Degrade gracefully: cTrader native → proxy → approximation → disabled

### Test Expectations
- FeatureRegistry.get("indicators/rsi") returns RSIFeature
- FeatureRegistry.requires("volume/imbalance") returns ["data/tick_stream", "data/volume"]
- FeatureRegistry.compatible_with("orderflow/tick", "opening_range_breakout") returns True
- All features register with correct CapabilitySpec

### Risks
- SVSS indicators may need API changes to work with library
- VolumeImbalanceFeature is complex (new development)
- Quality tagging may be inconsistent across features

### Review/Handoff Notes
- SVSS indicators are clean and standalone — good for wrapping
- Recovery note R-11 documents SVSS indicator system
- Recovery note R-12 documents session detection

### Indexed References
- `10_feature_family_plan.md` — full feature family specifications
- Gap: G-6 (feature families), G-7 (footprint logic)
- Recovery: R-11 (SVSS), R-12 (session detection)
- Memo §8 (feature families)

---

## SPEC-007: EvaluationBridge and Backtest Schema Compatibility

### Title
Bridge evaluation pipeline so library bots use existing 6-mode backtest system

### Scope
Create `src/library/bridges/to_evaluation/`:
- `evaluation_bridge.py` — EvaluationBridge class
- `backtest_mapper.py` — BacktestResult → EvaluationResult mapper
- `report_mapper.py` — BacktestReportSubAgent ↔ EvaluationResult

Key requirements:
- BotSpec → backtest input conversion
- 4-mode results collection
- MC/WFA/PBO results attachment
- **cTrader backtest schema must match MT5BacktestResult** (critical)

### Non-Scope
- Backtest engine implementation (separate spec)
- Report generation (BacktestReportSubAgent already exists)

### Why It Exists
The existing 6-mode evaluation pipeline (FullBacktestPipeline) is the canonical evaluation system. The library must integrate with it — bots should be evaluated through the same pipeline. Critical: cTrader backtest engine must produce the same result schema as MT5BacktestResult so the pipeline doesn't need to change.

### Affected Files/Modules Likely Involved
- `src/backtesting/full_backtest_pipeline.py` — FullBacktestPipeline (target)
- `src/backtesting/mt5_engine.py` — MT5BacktestResult (schema to match)
- `src/backtesting/mode_runner.py` — BacktestMode, SentinelEnhancedTester
- `src/agents/departments/subagents/backtest_report_subagent.py` — report integration

### Contracts/Interfaces to Respect
- MT5BacktestResult schema: sharpe, return_pct, drawdown, trades, equity_curve, trade_history, win_rate, profit_factor, kelly_score
- BacktestMode enum: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C
- SIT gate: >= 4 of 6 modes pass, OOS degradation <= 15%

### Dependencies
- CONTRACT-007 (EvaluationResult) must be done first
- SPEC-005 (CTrader market adapter) partial (historical data needed)
- SPEC-011 (cTrader backtest engine) partial

### Implementation Notes
- BotSpec → strategy code generation must be reliable
- CTraderBacktestResult must match MT5BacktestResult schema exactly
- BacktestReportSubAgent must work with both MT5 and cTrader results

### Test Expectations
- BotSpec with ORB archetype evaluates through FullBacktestPipeline
- EvaluationResult for VANILLA matches MT5BacktestResult schema
- EvaluationResult for SPICED includes regime_distribution, filtered_trades
- MonteCarloMetrics attached to BotEvaluationProfile
- WalkForwardMetrics attached to BotEvaluationProfile
- PBO score attached to BotEvaluationProfile

### Risks
- BotSpec → strategy code generation may fail for complex specs
- Schema mismatch between cTrader and MT5 backtest results breaks pipeline
- SentinelEnhancedTester regime filtering may not work with cTrader data

### Review/Handoff Notes
- Recovery note R-8 documents full evaluation pipeline
- Recovery note R-7 documents 6-mode backtest evaluation
- Backtest schema compatibility is critical — test early

### Indexed References
- `12_evaluation_workflow_alignment.md` — full evaluation alignment
- Gap: G-12 (WF1 alignment), G-13 (WF2 alignment)
- Recovery: R-7 (6-mode evaluation), R-8 (full pipeline chain), R-18 (MC vs mutation boundary)

---

## SPEC-008: Composer and MutationEngine

### Title
Spec-driven bot composition and constrained mutation system

### Scope
Create `src/library/core/composition/`:
- `composer.py` — Composer (builds bot from BotSpec + ArchetypeSpec)
- `validator.py` — CompositionValidator
- `requirements.py` — RequirementResolver
- `mutation.py` — MutationEngine (applies constrained mutations)

Create `src/library/archetypes/`:
- `base.py` — BaseArchetype ABC, ArchetypeSpec
- `orb.py` — OpeningRangeBreakout (Phase 1 priority)
- `derived/` — LondonORB, NYORB (initial derived archetypes)

### Non-Scope
- Runtime execution (separate spec)
- Advanced derived archetypes (Phase 2+)

### Why It Exists
The memo requires bots to be composed from specs, not assembled from raw code. The Composer validates the spec and builds the bot. MutationEngine applies constrained mutations for WF2.

### Affected Files/Modules Likely Involved
- `src/router/variant_registry.py` — BotVariant lineage (reference)
- `src/router/bot_tag_registry.py` — mutual exclusivity (reference pattern)
- `flows/improvement_loop_flow.py` — WF2 (mutation consumer)

### Contracts/Interfaces to Respect
- BotSpec, ArchetypeSpec, CapabilitySpec, CompatibilityRule
- Locked/allowed mutation surfaces from BotMutationProfile
- VariantRegistry lineage chain (parent_id)

### Dependencies
- ARCH-001-004 (CompositionValidator, RequirementResolver, registries) must be done first
- SPEC-001 (BotSpec) must be done first
- SPEC-002 (capability system) must be done first

### Implementation Notes
- Composer must validate before building (fail fast)
- MutationEngine must respect locked surfaces
- Lineage must be maintained through mutations

### Test Expectations
- Valid BotSpec + ArchetypeSpec → Composer builds bot
- Invalid BotSpec (missing dependency) → CompositionError
- Mutation allowed → mutated BotSpec with lineage preserved
- Mutation locked → MutationRejectedError

### Risks
- Composer too strict → blocks valid compositions
- MutationEngine lineage tracking may break on complex trees

### Review/Handoff Notes
- Recovery note R-6 documents VariantRegistry genealogy
- Recovery note R-5 documents BotTagRegistry mutual exclusivity (reference pattern)
- Phase 1 only ORB — other archetypes are Phase 2

### Indexed References
- Memo §10 (archetypes), §11 (bot spec model), §13 (mutation vs evaluation)
- Gap: G-9 (archetypes)
- Recovery: R-5 (TagRegistry), R-6 (VariantRegistry), R-18 (MC vs mutation boundary)

---

## SPEC-009: Runtime Intent Flow

### Title
Implement bot runtime: tick → feature → intent → risk → execution

### Scope
Create `src/library/runtime/`:
- `state_manager.py` — BotStateManager (cached FeatureVector + MarketContext)
- `feature_evaluator.py` — FeatureEvaluator (evaluates feature stack)
- `intent_emitter.py` — IntentEmitter (emits TradeIntent from spec)
- `safety_hooks.py` — SafetyHooks (kill switch, circuit breaker integration)

Implement full runtime flow:
```
tick → market_adapter → state_manager (cache)
                          ↓
                    feature_evaluator → feature_vector
                          ↓
                    intent_emitter → trade_intent
                          ↓
                    risk_bridge → governor → risk_envelope
                          ↓
                    execution_bridge → ctrader → execution_result
                          ↓
                    dpr_bridge (score) + journal_bridge (audit)
```

### Non-Scope
- cTrader adapter implementation (separate spec)
- Bridge implementations (separate specs)

### Why It Exists
The runtime is where the library's value is realized: a bot spec becomes a live trading decision. This spec implements the canonical flow documented in architecture §8.

### Affected Files/Modules Likely Involved
- `src/router/governor.py` — Governor (risk authorization target)
- `src/router/commander.py` — Commander (execution target)
- `src/router/kill_switch.py` — KillSwitch (safety target)
- `src/router/progressive_kill_switch.py` — ProgressiveKillSwitch (safety target)
- `src/risk/ssl/circuit_breaker.py` — SSLCircuitBreaker (safety target)

### Contracts/Interfaces to Respect
- IMarketDataAdapter (tick source)
- IExecutionAdapter (execution target)
- RiskEnvelope (risk authorization result)
- TradeIntent (bot decision output)
- ExecutionDirective (authorized execution)

### Dependencies
- All bridges (BRIDGE-001-009) must be done first
- FEATURE-018 (FeatureEvaluator) must be done first
- SPEC-001 (BotSpec) must be done first

### Implementation Notes
- Sync/async boundary: async tick writes to cache, sync decision-time reads
- Thread safety: cached state must handle concurrent access
- Error propagation: risk rejection must stop execution
- Safety hooks: kill switch events must be handled

### Test Expectations
- Tick arrives → MarketContext cached → FeatureVector computed → TradeIntent emitted
- TradeIntent → RiskEnvelope (approved) → ExecutionDirective → ExecutionResult
- TradeIntent → RiskEnvelope (rejected/HALTED) → no execution
- Kill switch event → active positions closed

### Risks
- Sync/async boundary violations (blocking on async or vice versa)
- Thread safety issues in cached state
- Performance: decision time must be < 50ms

### Review/Handoff Notes
- Recovery note R-10 documents Governor risk logic
- Recovery notes R-3 (kill switches), R-4 (circuit breaker S3-11), R-9 (SSL)
- Critical: sync/async boundary is explicit in architecture

### Indexed References
- `06_target_architecture_v1.md` (Section 8: Runtime Intent Flow)
- Recovery: R-1 (Sentinel), R-3 (kill switches), R-4 (circuit breaker), R-9 (SSL), R-10 (Governor)

---

## SPEC-010: WF1 and WF2 Bridge Integration

### Title
Integrate library BotSpec into Prefect workflows (AlphaForgeFlow + ImprovementLoopFlow)

### Scope
Create `src/library/bridges/to_workflows/`:
- `wf1_bridge.py` — WF1Bridge (TRD → BotSpec → AlphaForgeFlow)
- `wf2_bridge.py` — WF2Bridge (variant lineage → ImprovementLoopFlow)

WF1 integration:
- TRDDocument → BotSpec conversion
- BotSpec → AlphaForgeFlow input
- Evaluation results → paper trading handoff

WF2 integration:
- Variant lineage → BotMutationProfile
- MutationEngine output → ImprovementLoopFlow input
- Evaluation results → promote/demote/kill decision
- 3-day paper lag → live promotion

### Non-Scope
- Prefect flow implementation changes
- Paper trading validation (already exists)

### Why It Exists
WF1 and WF2 are the canonical strategy development pipelines. The library must integrate so BotSpec flows through these workflows. Currently, workflows consume TRD + strategy code directly.

### Affected Files/Modules Likely Involved
- `flows/alpha_forge_flow.py` — AlphaForgeFlow (WF1)
- `flows/improvement_loop_flow.py` — ImprovementLoopFlow (WF2)
- `src/trd/schema.py` — TRDDocument (input)
- `src/api/prefect_workflow_endpoints.py` — workflow trigger API

### Contracts/Interfaces to Respect
- Prefect flow trigger interface
- AlphaForgeFlow input format
- ImprovementLoopFlow input format
- Workflow approval gate (>= 4 of 6 modes, OOS <= 15%)

### Dependencies
- SPEC-001 (BotSpec) must be done first
- SPEC-008 (Composer) must be done first
- SPEC-007 (EvaluationBridge) must be done first

### Implementation Notes
- TRD → BotSpec conversion must be loss-free
- Evaluation results must flow back into workflow state
- Paper trading handoff must use existing enhanced_paper_trading_deployer

### Test Expectations
- TRDDocument → BotSpec conversion produces valid BotSpec
- AlphaForgeFlow.trigger accepts BotSpec input
- ImprovementLoopFlow.trigger accepts mutated BotSpec
- Promotion/demotion/kill decisions flow through LifecycleBridge

### Risks
- Workflow changes may break existing operations
- TRD → BotSpec conversion may lose information

### Review/Handoff Notes
- Memory: WF1_SYSTEM_SCAN.md is source of truth
- Recovery note R-8 documents full evaluation pipeline
- WF1/WF2 are production — changes must be backward compatible

### Indexed References
- `12_evaluation_workflow_alignment.md` (Sections 6-7: WF1/WF2 alignment)
- Memo §13 (workflow alignment)
- Memory: WF1_SYSTEM_SCAN.md
- Recovery: R-8 (evaluation pipeline chain)

---

## SPEC-011: CTrader Backtest Engine (Schema Compatibility)

### Title
Implement cTrader backtest engine producing same schema as MT5BacktestResult

### Scope
Create `src/library/adapters/ctrader/backtest_engine.py`:
- CTraderBacktestEngine class
- Produces BacktestResult with same schema as MT5BacktestResult

Critical: schema compatibility. The evaluation pipeline (FullBacktestPipeline) must not need to change when switching from MT5 to cTrader.

### Non-Scope
- Market data adapter (separate spec)
- Execution adapter (separate spec)

### Why It Exists
MT5 backtest engine (mt5_engine.py) is embedded in the evaluation pipeline. cTrader replacement must produce identical result schemas. Currently, mt5_engine.py simulates MQL5 environment — cTrader engine must replicate this.

### Affected Files/Modules Likely Involved
- `src/backtesting/mt5_engine.py` — MT5BacktestResult schema (MUST MATCH)
- `src/backtesting/full_backtest_pipeline.py` — evaluation pipeline (consumer)
- `src/backtesting/mode_runner.py` — SentinelEnhancedTester (regime filtering)

### Contracts/Interfaces to Respect
- MT5BacktestResult schema exactly: sharpe, return_pct, drawdown, trades, equity_curve, trade_history, win_rate, profit_factor, kelly_score, initial_cash, final_cash
- BacktestMode compatibility: VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C
- SentinelEnhancedTester regime filtering (chaos > 0.6, HIGH_CHAOS/NEWS_EVENT banned)

### Dependencies
- CTRADER-004 (CTraderMarketAdapter) partial (needs historical data access)
- BacktestMode compatibility must be verified with mode_runner.py

### Implementation Notes
- Schema MUST match MT5BacktestResult exactly — test early
- Include regime analytics for SPICED mode
- Walk-Forward support required for VANILLA_FULL / SPICED_FULL

### Test Expectations
- CTraderBacktestResult can be passed to FullBacktestPipeline with no changes
- Regime filtering (SentinelEnhancedTester) works with cTrader data
- Walk-Forward analysis produces same results as MT5 engine
- Monte Carlo shuffles trade order and produces confidence intervals

### Risks
- Schema mismatch → evaluation pipeline breaks
- cTrader historical data may differ from MT5 historical data
- Regime classification may differ between MT5 and cTrader data

### Review/Handoff Notes
- Recovery note R-7 documents 6-mode evaluation
- Recovery note R-8 documents full backtest pipeline
- This is the most critical migration task — test schema compatibility first

### Indexed References
- Gap: G-10, G-11 (cTrader migration)
- Recovery: R-7 (6-mode evaluation), R-8 (full pipeline)
- Official docs: cTrader Open API + backtesting docs