# 08 — Bridge Inventory

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Two-way bridge design between QuantMindLib and existing platform systems

---

## Bridge Design Principles

1. **Bridges are adapters, not absorbers** — The bridge translates between library objects and system-native objects. It does NOT absorb system logic.

2. **Two-way by default** — Every bridge has inbound (system → library) and outbound (library → system) paths.

3. **Sync/async is explicit per operation** — Each operation within a bridge declares sync or async.

4. **Error handling is defined per bridge** — Each bridge specifies how it handles translation failures.

5. **Mandatory vs stub-able in V1** — Each bridge is marked mandatory (V1 required) or stub-able (can be stubbed for testing).

---

## BRIDGE-1: Sentinel Bridge

**Target:** `src/router/sentinel.py`, `src/router/multi_timeframe_sentinel.py`
**Boundary:** BRIDGE (async in, sync/async out)
**Mandatory:** P1

### Purpose
Bidirectional translation between Sentinel's `RegimeReport` and library's `MarketContext` / `SentinelState`. Also handles library → sentinel feedback (feature summaries, bot requests).

### Inbound (Sentinel → Library)
```
Sentinel.RegimeReport
    │
    ├─► RegimeReport → MarketContext conversion
    │         │ (regime, chaos_score, regime_quality, susceptibility, is_systemic_risk)
    │         ▼
    │    MarketContext (cached in BotStateManager)
    │
    ├─► RegimeReport → SentinelState (full state snapshot)
    │         │ (sensor states, HMM state, ensemble vote)
    │         ▼
    │    SentinelState (async event)
    │
    └─► NewsSensor state → SessionContext.news_state
              │ (news_state ACTIVE/KILL_ZONE/CLEAR)
              ▼
         SessionContext (updated by SessionBridge)
```

### Outbound (Library → Sentinel)
```
FeatureEvaluator results
    │
    ├─► Feature summaries → Sentinel (optional feedback loop)
    │         │ (which features are firing, confidence levels)
    │         ▼
    │    Sentinel.feedback(features) — if sentinel supports feedback
    │
    └─► Bot requests for context → Sentinel
              │ (request specific regime data, sensor readings)
              ▼
         Sentinel.query(request) — if supported
```

### Sync/Async Behavior
- **Inbound:** Async (regime events from sentinel are event-driven)
- **Outbound:** Sync (feature summaries read from cache)
- **Cache:** RegimeReport cached in `BotStateManager`, sync read at decision time

### Latency Sensitivity
- Regime classification: < 10ms from tick to cached update
- MarketContext read at decision: < 1ms (from cache)

### Error Handling
- Sentinel unavailable → use last known regime, mark `MarketContext.is_stale = True`
- Regime report malformed → log error, return UNCERTAIN regime as fallback
- HMM prediction fails → exclude from ensemble, log warning

### Implementation Location
- `src/library/bridges/to_sentinel/sentinel_bridge.py`
- `src/library/bridges/to_sentinel/types.py`
- `src/library/bridges/to_sentinel/config.py`

### Affected Repo Files
- `src/router/sentinel.py` — RegimeReport interface
- `src/router/multi_timeframe_sentinel.py` — MultiTimeframeSentinel
- `src/router/market_scanner.py` — MarketScanner (session context)
- `src/router/calendar_governor.py` — CalendarGovernor
- `src/market/news_blackout.py` — NewsBlackoutService (news state)

### Indexed References
- Memo §6 (bridge design), §7 (sentinel bridge details)
- Codebase: `src/router/sentinel.py` (RegimeReport dataclass)
- Recovery note: R-1 (Sentinel regime detection architecture)

---

## BRIDGE-2: Risk Bridge

**Target:** `src/router/governor.py`, `src/risk/governor.py`, `src/position_sizing/enhanced_kelly.py`
**Boundary:** BRIDGE (sync primary)
**Mandatory:** P1

### Purpose
Bidirectional translation between library's `TradeIntent` / `RiskEnvelope` and existing Governor / EnhancedKelly systems.

### Inbound (Risk → Library)
```
Governor.RiskMandate
    │
    ├─► RiskMandate → RiskEnvelope conversion
    │         │ (allocation_scalar, risk_mode, position_size, kelly_fraction, risk_amount)
    │         ▼
    │    RiskEnvelope (returned to IntentEmitter)
    │
    ├─► EnhancedKelly.KellyResult → position sizing bridge
    │         │ (position_size, kelly_f, risk_amount)
    │         ▼
    │    RiskEnvelope.position_size (approved lot size)
    │
    └─► Governor exposure cap check → library halt
              │ (5% concurrent exposure cap → HALTED mode)
              ▼
         TradeIntent approval/rejection
```

### Outbound (Library → Risk)
```
TradeIntent
    │
    ├─► TradeIntent → Governor input
    │         │ (bot_id, symbol, direction, size_requested, stop_loss_pips)
    │         ▼
    │    Governor.calculate_risk(bot_id, TradeIntent)
    │         │
    │         ▼
    │    RiskEnvelope (received back)
    │
    └─► MarketContext + FeatureVector → Governor context
              │ (regime, chaos_score, regime_quality)
              ▼
         Governor internal risk computation
```

### Sync/Async Behavior
- **Primary:** Sync (risk calculation at decision time)
- **Governor risk computation:** Sync (physics throttling, correlation check, 5% cap)
- **EnhancedKelly:** Sync (pure math)
- **Async path:** Governor can emit async events on tier escalation (for SafetyHooks)

### Latency Sensitivity
- Risk calculation: < 5ms from TradeIntent to RiskEnvelope
- EnhancedKelly: < 1ms (pure computation)

### Error Handling
- Governor unavailable → return HALTED RiskEnvelope with reason "governor_unavailable"
- EnhancedKelly returns zero → use conservative fallback (0.01 lot minimum)
- 5% cap exceeded → always return HALTED, never partial approval

### Implementation Location
- `src/library/bridges/to_risk/risk_bridge.py`
- `src/library/bridges/to_risk/governor_mapper.py`
- `src/library/bridges/to_risk/sizing_mapper.py`

### Affected Repo Files
- `src/router/governor.py` — RiskMandate, Governor
- `src/risk/governor.py` — Tier 2 risk rules
- `src/position_sizing/enhanced_kelly.py` — EnhancedKellyCalculator, KellyResult
- `src/risk/prop_firm_overlay.py` — PropFirmOverlay

### Indexed References
- Memo §6 (bridge design), §12 (what library must not own)
- Codebase: `src/router/governor.py` (RiskMandate), `src/position_sizing/enhanced_kelly.py` (KellyResult)
- Recovery note: R-10 (Governor risk authorization logic), C1 (two Governor classes)

---

## BRIDGE-3: Registry Bridge

**Target:** `src/router/bot_manifest.py`, `src/router/bot_tag_registry.py`, `src/router/variant_registry.py`
**Boundary:** BRIDGE (sync primary, async writes)
**Mandatory:** P1

### Purpose
Bidirectional translation between library's `BotSpec` / `RegistryRecord` and existing BotRegistry / BotTagRegistry / VariantRegistry.

### Inbound (Registry → Library)
```
BotRegistry.BotManifest
    │
    ├─► BotManifest → BotSpec.static conversion
    │         │ (bot_id, strategy_type → archetype, symbols, timeframes, sessions, tags)
    │         ▼
    │    BotSpec (static profile loaded)
    │
    ├─► BotTagRegistry → BotRuntimeProfile.activation_state
    │         │ (@live → ACTIVE, @quarantine → QUARANTINED, @dead → RETIRED)
    │         ▼
    │    BotRuntimeProfile.activation_state
    │
    ├─► VariantRegistry.BotVariant → BotMutationProfile
    │         │ (lineage, parent_id, generation, parameters, locked_areas)
    │         ▼
    │    BotMutationProfile
    │
    └─► BotLifecycleLog → library event
              │ (tag transitions, promotion, quarantine)
              ▼
         LifecycleBridge events
```

### Outbound (Library → Registry)
```
BotSpec
    │
    ├─► BotSpec → BotRegistry.register()
    │         │ (create BotManifest from BotSpec static)
    │         ▼
    │    BotRegistry.register(bot_id, BotManifest)
    │
    ├─► BotRuntimeProfile → BotTagRegistry operations
    │         │ (ACTIVE → @live, QUARANTINED → @quarantine, RETIRED → @dead)
    │         ▼
    │    BotTagRegistry.apply_tag(bot_id, tag, reason)
    │
    ├─► BotMutationProfile → VariantRegistry.register_variant()
    │         │ (variant_id, parent_id, lineage, parameters)
    │         ▼
    │    VariantRegistry.register_variant(bot_variant)
    │
    └─► Promotion decision → LifecycleManager
              │ (DPR >= 50 → promote, DPR < 50 → quarantine)
              ▼
         LifecycleManager.promote_to_live() / quarantine()
```

### Sync/Async Behavior
- **Reads:** Sync (BotManifest loaded at startup, tags checked at runtime)
- **Writes:** Async (tag updates, variant registration, lifecycle events can be async)
- **Promotion decision:** Sync (DPR threshold check is synchronous)

### Latency Sensitivity
- BotSpec loading: < 100ms
- Tag operations: < 50ms
- Lifecycle decisions: < 100ms

### Error Handling
- BotRegistry unavailable → raise RegistryUnavailableError, bot cannot run
- Tag operation fails → log warning, continue (non-fatal)
- Variant registration fails → log error, bot can run without variant tracking

### Implementation Location
- `src/library/bridges/to_registry/registry_bridge.py`
- `src/library/bridges/to_registry/tag_mapper.py`
- `src/library/bridges/to_registry/variant_mapper.py`

### Affected Repo Files
- `src/router/bot_manifest.py` — BotRegistry, BotManifest
- `src/router/bot_tag_registry.py` — BotTagRegistry, TagHistoryEntry
- `src/router/variant_registry.py` — VariantRegistry, BotVariant
- `src/router/lifecycle_manager.py` — LifecycleManager

### Indexed References
- Memo §6 (bridge design), §12 (what library must not own)
- Codebase: `src/router/bot_manifest.py`, `src/router/bot_tag_registry.py`, `src/router/variant_registry.py`
- Recovery notes: R-5 (BotTagRegistry mutual exclusivity), R-6 (VariantRegistry genealogy), R-13 (LifecycleManager transitions)

---

## BRIDGE-4: DPR Bridge

**Target:** `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py`
**Boundary:** BRIDGE (async primary, sync reads)
**Mandatory:** P1 (fixes G-18: DPR Redis gap)

### Purpose
Bidirectional translation between library's `BotEvaluationProfile` / `BotPerformanceSnapshot` and existing DPR scoring engines. **Critical: this bridge must write DPR scores to Redis (fixing the DPR Redis gap).**

### Inbound (DPR → Library)
```
DPRScoringEngine.DprScore
    │
    ├─► DprScore → BotRuntimeProfile.dpr_score / dpr_ranking
    │         │ (composite_score, component_scores, tier)
    │         ▼
    │    BotRuntimeProfile.dpr_score (0-100)
    │    BotRuntimeProfile.dpr_ranking (computed from all bot scores)
    │
    ├─► DPRComponentScores → BotEvaluationProfile.component_scores
    │         │ (win_rate 25%, pnl 30%, consistency 20%, ev_per_trade 25%)
    │         ▼
    │    BotEvaluationProfile (attached to BotSpec)
    │
    ├─► DPRConcernEvent → library concern flag
    │         │ (>20 point WoW drop → @session_concern tag)
    │         ▼
    │    RegistryBridge.apply_tag("@session_concern")
    │
    └─► DPR tier → BotTier mapping
              │ (T1 >= 80, T2 >= 50, T3 < 50)
              ▼
         BotRuntimeProfile.tier
```

### Outbound (Library → DPR)
```
BotPerformanceSnapshot (periodic)
    │
    ├─► BotPerformanceSnapshot → DPRScoringEngine.score()
    │         │ (session_wr, net_pnl, consistency, ev_per_trade)
    │         ▼
    │    DPRScoringEngine.compute_composite_score()
    │
    ├─► DPR score → Redis publish (CRITICAL - fixes G-18)
    │         │ key: `dpr:score:{bot_id}`, TTL: 24h
    │         ▼
    │    Redis.setex(f"dpr:score:{bot_id}", 86400, score_json)
    │
    └─► DPR concern flag → DPRScoringEngine.check_concern_flag()
              │ (>20 point drop → DPRConcernEvent)
              ▼
         DPRConcernEvent → RegistryBridge (@session_concern tag)
```

### Sync/Async Behavior
- **DPR computation:** Sync (fast, deterministic)
- **Redis publish:** Async (non-blocking)
- **Score read:** Sync (read from Redis cache)

### DPR Redis Gap Fix
The router-layer `DprScoringEngine.score_all_bots()` does NOT write scores to Redis. This bridge MUST include Redis publish:
```python
# In DPRBridge.outbound:
await redis.setex(f"dpr:score:{bot_id}", 86400, json.dumps({
    "composite_score": score,
    "components": component_scores,
    "tier": tier,
    "timestamp": datetime.utcnow().isoformat()
}))
```

### Latency Sensitivity
- DPR score computation: < 10ms
- Redis write: < 5ms (async, non-blocking)
- Redis read: < 2ms

### Error Handling
- DPR engine unavailable → use last known score, mark stale
- Redis write fails → log error, retry 3 times, then log and continue
- Redis read fails → fall back to in-memory DPR engine score

### Implementation Location
- `src/library/bridges/to_dpr/dpr_bridge.py`
- `src/library/bridges/to_dpr/score_publisher.py`
- `src/library/bridges/to_dpr/tier_mapper.py`

### Affected Repo Files
- `src/router/dpr_scoring_engine.py` — DprScoringEngine (router), DprScore, DprComponents
- `src/risk/dpr/scoring_engine.py` — DPRScoringEngine (risk), DPRScore
- `src/events/dpr.py` — DPRScoreEvent, DPRConcernEvent, DPRComponentScores

### Indexed References
- Memo §6 (bridge design), §12 (what library must not own)
- Codebase: `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py`
- Gap: G-18 (DPR Redis gap), C2 (DPR dual engines)
- Recovery notes: R-2 (DPR dual-engine architecture), R-16 (DPR tier thresholds)

---

## BRIDGE-5: Evaluation Bridge

**Target:** `src/backtesting/full_backtest_pipeline.py`, `src/agents/departments/subagents/backtest_report_subagent.py`
**Boundary:** EVALUATION (sync primary)
**Mandatory:** P1

### Purpose
Bidirectional translation between library's `EvaluationResult` and existing backtest/evaluation pipeline. WF1 and WF2 both use this bridge.

### Inbound (Evaluation → Library)
```
FullBacktestPipeline.BacktestComparison
    │
    ├─► 4-mode results → EvaluationResult (per mode)
    │         │ (sharpe, max_drawdown, win_rate, profit_factor, total_trades)
    │         ▼
    │    EvaluationResult (VANILLA, SPICED, VANILLA_FULL, SPICED_FULL)
    │
    ├─► MonteCarloResult → BotEvaluationProfile.monte_carlo
    │         │ (confidence_5th/95th, VaR_95, prob_profitable)
    │         ▼
    │    BotEvaluationProfile.monte_carlo (MonteCarloMetrics)
    │
    ├─► WalkForwardResult → BotEvaluationProfile.walk_forward
    │         │ (aggregate metrics, windows passed/total, WFA efficiency)
    │         ▼
    │    BotEvaluationProfile.walk_forward (WalkForwardMetrics)
    │
    ├─► PBOCalculator → BotEvaluationProfile.pbo_score
    │         │ (PBO < 0.20 = ACCEPT, 0.20-0.50 = CAUTION, > 0.50 = REJECT)
    │         ▼
    │    BotEvaluationProfile.pbo_score
    │
    └─► BacktestReportSubAgent → library report format
              │ (structured markdown → EvaluationReport)
              ▼
         EvaluationReport (attached to BotEvaluationProfile)
```

### Outbound (Library → Evaluation)
```
BotSpec
    │
    ├─► BotSpec → FullBacktestPipeline input
    │         │ (strategy_code, symbol, timeframe from BotSpec)
    │         ▼
    │    FullBacktestPipeline.run_all_variants(BotSpec → strategy code)
    │
    └─► BotMutationProfile → Evaluation input
              │ (variant lineage, parameter changes for WF2)
              ▼
         ImprovementLoopFlow input (WF2)
```

### Sync/Async Behavior
- **Backtest pipeline:** Sync (full pipeline runs synchronously per request)
- **Report generation:** Sync (BacktestReportSubAgent generates markdown)
- **WF2 mutation:** Sync (mutation → backtest → eval cycle)

### Latency Sensitivity
- Backtest pipeline: minutes (workflow-level)
- Report generation: seconds
- Not latency-sensitive (evaluation is offline)

### Error Handling
- Backtest fails → return EvaluationResult with `passes_gate = False`, error details
- Report generation fails → return partial report, flag incomplete
- PBO calculation fails → set PBO to None, log warning

### Implementation Location
- `src/library/bridges/to_evaluation/evaluation_bridge.py`
- `src/library/bridges/to_evaluation/backtest_mapper.py`
- `src/library/bridges/to_evaluation/report_mapper.py`

### Affected Repo Files
- `src/backtesting/full_backtest_pipeline.py` — FullBacktestPipeline, BacktestComparison
- `src/backtesting/mt5_engine.py` — MT5BacktestResult
- `src/backtesting/mode_runner.py` — BacktestMode, SpicedBacktestResult, SentinelEnhancedTester
- `src/backtesting/monte_carlo.py` — MonteCarloResult
- `src/backtesting/walk_forward.py` — WalkForwardResult
- `src/backtesting/pbo_calculator.py` — PBOCalculator
- `src/agents/departments/subagents/backtest_report_subagent.py` — BacktestReportSubAgent
- `src/database/models/performance.py` — StrategyPerformance

### Indexed References
- Memo §13 (workflow alignment)
- Codebase: `src/backtesting/full_backtest_pipeline.py`, `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py`
- Recovery notes: R-7 (6-mode backtest evaluation), R-8 (full backtest pipeline chain), R-18 (Monte Carlo vs mutation boundary)

---

## BRIDGE-6: Journal Bridge

**Target:** `src/database/models/trade_record.py`, `src/router/trade_logger.py`
**Boundary:** BRIDGE (async primary)
**Mandatory:** P2 (stub-able in V1)

### Purpose
Bridge library trade events to audit log and trade records. Two-way: library emits events that get journaled; historical journal data feeds evaluation.

### Inbound (Journal → Library)
```
TradeRecord (historical)
    │
    ├─► TradeRecord → BotEvaluationProfile.historical_trades
    │         │ (trade history for evaluation)
    │         ▼
    │    EvaluationBridge input
    │
    └─► BotLifecycleLog → library lifecycle events
              │ (tag transitions, promotion, quarantine)
              ▼
         LifecycleBridge events (via RegistryBridge)
```

### Outbound (Library → Journal)
```
TradeIntent (emitted)
    │
    ├─► TradeIntent → TradeRecord journal entry
    │         │ (bot_id, symbol, direction, size, rationale, timestamp)
    │         ▼
    │    TradeRecord (created on intent emit)
    │
    ├─► ExecutionDirective → TradeRecord update
    │         │ (approved/rejected, approved_size, constraints)
    │         ▼
    │    TradeRecord updated on execution
    │
    ├─► Trade outcome → TradeRecord completion
    │         │ (PnL, exit reason, fees, execution quality)
    │         ▼
    │    TradeRecord completed (for DPR scoring input)
    │
    └─► KillEvent → Layer3 forced exit journal
              │ (forced close reason, tickets affected, layer3_events)
              ▼
         TradeRecord.layer3_events JSON field update
```

### Sync/Async Behavior
- **Inbound:** Sync (historical reads for evaluation)
- **Outbound:** Async (journal writes are non-blocking, fire-and-forget with retry)

### Latency Sensitivity
- Journal writes: non-latency-sensitive (async, background)
- Historical reads: < 100ms

### Error Handling
- Journal write fails → queue for retry, don't block trading
- Journal read fails → return empty result, log error

### Implementation Location
- `src/library/bridges/to_journal/journal_bridge.py`

### Affected Repo Files
- `src/database/models/trade_record.py` — TradeRecord model with `layer3_events`
- `src/router/trade_logger.py` — TradeLogger
- `src/router/lifecycle_manager.py` — BotLifecycleLog

### Indexed References
- Memo §6 (bridge design)
- Recovery note: R-19 (Layer3KillSwitch Redis lock bypass)

---

## BRIDGE-7: Workflow Bridge (WF1 + WF2)

**Target:** `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py`
**Boundary:** WORKFLOW (sync)
**Mandatory:** P1

### Purpose
Bridge library BotSpec into Prefect workflows and bridge evaluation results back to workflow state.

### WF1 Bridge (AlphaForgeFlow)

**Inbound (Workflow → Library):**
```
TRDDocument
    │
    └─► TRD → BotSpec conversion
              │ (strategy_id, entry_conditions, exit_conditions, parameters)
              ▼
         BotSpec (static profile)
```

**Outbound (Library → Workflow):**
```
BotSpec (with evaluation attached)
    │
    ├─► BotSpec → AlphaForgeFlow.trigger()
    │         │ (strategy spec for development pipeline)
    │         ▼
    │    AlphaForgeFlow (research → TRD → dev → compile → backtest)
    │
    ├─► EvaluationResult → AlphaForgeFlow approval gate
    │         │ (>= 4 of 6 modes pass, OOS degradation <= 15%)
    │         ▼
    │    Workflow approval / rejection
    │
    └─► BotEvaluationProfile → paper trading deployment
              │ (robustness_score >= 0.8 → deploy)
              ▼
         paper_trading_deployer (via agents)
```

### WF2 Bridge (ImprovementLoopFlow)

**Inbound (Workflow → Library):**
```
Surviving variants from WF1
    │
    ├─► Variant lineage → BotMutationProfile
    │         │ (parent bot ID, parameter changes, generation)
    │         ▼
    │    BotMutationProfile
    │
    └─► Mutation constraints → Library
              │ (allowed/locked areas for this variant family)
              ▼
         CompositionValidator (validates mutation)
```

**Outbound (Library → Workflow):**
```
BotMutationProfile (with new mutation applied)
    │
    ├─► Mutated BotSpec → ImprovementLoopFlow
    │         │ (variant spec for re-backtest)
    │         ▼
    │    ImprovementLoopFlow (analyze → re-backtest → MC → WFA → paper)
    │
    ├─► EvaluationResult → promotion/demotion/kill decision
    │         │ (same 6-mode evaluation, DPR score)
    │         ▼
    │    LifecycleManager.promote_to_live() / quarantine()
    │
    └─► 3-day paper lag → live promotion
              │ (paper validation → live deployment)
              ▼
         LifecycleManager.promote_to_live()
```

### Sync/Async Behavior
- **WF1/WF2:** Both are Prefect flows — sync trigger, async execution
- **Bridge calls:** Sync at trigger points, async during flow execution

### Latency Sensitivity
- Not latency-sensitive (workflow-level, minutes to hours)

### Error Handling
- Workflow trigger fails → raise WorkflowError, bot spec not consumed
- Evaluation result unavailable → workflow waits, times out after configured period

### Implementation Location
- `src/library/bridges/to_workflows/wf1_bridge.py`
- `src/library/bridges/to_workflows/wf2_bridge.py`

### Affected Repo Files
- `flows/alpha_forge_flow.py` — AlphaForgeFlow
- `flows/improvement_loop_flow.py` — ImprovementLoopFlow
- `flows/research_synthesis_flow.py` — ResearchSynthesisFlow
- `src/api/prefect_workflow_endpoints.py` — workflow trigger endpoints

### Indexed References
- Memo §13 (workflow alignment)
- Codebase: `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py`
- Memory: `WF1_SYSTEM_SCAN.md` (source of truth)

---

## BRIDGE-8: Lifecycle Bridge

**Target:** `src/router/lifecycle_manager.py`, `src/router/bot_circuit_breaker.py`
**Boundary:** WORKFLOW (sync primary)
**Mandatory:** P2

### Purpose
Bridge library promotion/quarantine decisions to existing LifecycleManager. Not a separate bridge — this is the lifecycle integration point of the RegistryBridge + DPRBridge.

### Lifecycle Decision Flow
```
DPR score >= 50
    │
    ▼
DPRBridge → BotRuntimeProfile.dpr_score updated
    │
    ▼
RegistryBridge → LifecycleManager.promote_to_live()
    │
    ├─► Add @live tag
    ├─► Register with KillSwitch
    ├─► Log to BotLifecycleLog
    └─► Dispatch to FloorManager (if quarantine)

DPR score drops > 20 points WoW OR circuit breaker triggers
    │
    ▼
RegistryBridge → LifecycleManager.quarantine()
    │
    ├─► Add @quarantine tag
    ├─► Remove @live tag
    ├─► Downgrade to PAPER
    ├─► Unregister from KillSwitch
    ├─► Log to BotLifecycleLog
    └─► Dispatch BOT_QUARANTINE_REVIEW to FloorManager
```

### Implementation Location
- `src/library/bridges/to_lifecycle/lifecycle_bridge.py` (referenced from registry_bridge)

### Indexed References
- Recovery notes: R-13 (LifecycleManager transitions), R-4 (BotCircuitBreaker S3-11 rules)

---

## Bridge Summary Matrix

| Bridge | Target System | Boundary | Sync/Async | Mandatory | Priority | DPR Redis Gap |
|--------|-------------|----------|-----------|-----------|----------|---------------|
| **Sentinel** | Sentinel, MarketScanner | BRIDGE | Hybrid | Yes | P1 | — |
| **Risk** | Governor, EnhancedKelly | BRIDGE | Sync | Yes | P1 | — |
| **Registry** | BotRegistry, TagRegistry, VariantRegistry | BRIDGE | Hybrid | Yes | P1 | — |
| **DPR** | DPRScoringEngine | BRIDGE | Hybrid | Yes | P1 | **FIXES G-18** |
| **Evaluation** | FullBacktestPipeline, BacktestReportSubAgent | EVALUATION | Sync | Yes | P1 | — |
| **Journal** | TradeRecord, TradeLogger | BRIDGE | Async | No | P2 | — |
| **WF1** | AlphaForgeFlow | WORKFLOW | Sync | Yes | P1 | — |
| **WF2** | ImprovementLoopFlow | WORKFLOW | Sync | Yes | P1 | — |
| **Lifecycle** | LifecycleManager, CircuitBreaker | WORKFLOW | Sync | No | P2 | — |