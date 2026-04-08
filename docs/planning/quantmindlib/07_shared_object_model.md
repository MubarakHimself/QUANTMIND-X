# 07 — Shared Object Model and Contract Inventory

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Canonical schemas for QuantMindLib V1

---

## Object Inventory

### 1. BotSpec

**Purpose:** Primary spec object — the canonical bot description consumed by all systems.
**Boundary:** CORE

**Required fields:**
- `id: str` — unique bot identifier
- `archetype: str` — archetype ID (e.g., "opening_range_breakout")
- `symbol_scope: List[str]` — symbols this bot trades
- `sessions: List[str]` — session IDs bot is active during
- `features: List[str]` — feature family IDs this bot uses
- `confirmations: List[str]` — confirmation rule IDs
- `execution_profile: str` — execution profile ID
- `capability_spec: CapabilitySpec` — what this bot needs/produces

**Optional fields:**
- `runtime: Optional[BotRuntimeProfile]`
- `evaluation: Optional[BotEvaluationProfile]`
- `mutation: Optional[BotMutationProfile]`

**Producers:** TRD parser, WF1 workflow, manual creation
**Consumers:** Composer, FeatureEvaluator, RegistryBridge, EvaluationBridge, LifecycleBridge
**Sync/Async:** Sync (static spec loaded at startup)
**Evidence:** Memo §11, Codebase: `src/trd/schema.py`, `src/router/bot_manifest.py` (BotManifest as partial substitute)

---

### 2. BotRuntimeProfile

**Purpose:** Runtime state of a bot — activation, health, DPR rank, eligibility
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `activation_state: ActivationState` — ACTIVE / SUSPENDED / QUARANTINED / RETIRED
- `deployment_target: str` — paper / demo / live
- `health: BotHealth` — health status (HEALTHY / DEGRADED / CRITICAL)
- `session_eligibility: Dict[str, bool]` — session_id → eligible
- `dpr_ranking: int` — rank in DPR (1 = top)
- `dpr_score: float` — DPR composite score 0-100
- `journal_id: Optional[str]` — active journal record ID
- `report_ids: List[str]` — backtest/report record IDs

**Producers:** SentinelBridge, DPRBridge, RegistryBridge
**Consumers:** BotStateManager, LifecycleBridge, SafetyHooks
**Sync/Async:** Hybrid (sync read, async writes)
**Evidence:** Codebase: `src/router/bot_manifest.py` (BotManifest.live_stats/paper_stats), `src/router/dpr_scoring_engine.py` (DprScore)

---

### 3. BotEvaluationProfile

**Purpose:** Evaluation results across all evaluation modes
**Boundary:** EVALUATION

**Required fields:**
- `bot_id: str`
- `backtest: BacktestMetrics` — 4-mode backtest results
- `monte_carlo: MonteCarloMetrics` — MC confidence intervals
- `walk_forward: WalkForwardMetrics` — WFA aggregate results
- `pbo_score: float` — probability of backtest overfitting
- `robustness_score: float` — combined robustness 0.0-1.0
- `spread_sensitivity: float` — sensitivity to spread changes
- `session_scores: Dict[str, SessionScore]` — per-session performance

**Monte Carlo note:** MC is evaluation, not mutation. MC shuffles trade order and reports confidence intervals. It does NOT change the candidate.

**Producers:** EvaluationBridge (from FullBacktestPipeline)
**Consumers:** DPRBridge (for DPR scoring), LifecycleBridge (for promotion), RegistryBridge (for registry updates)
**Sync/Async:** Sync (written after evaluation, read on demand)
**Evidence:** Codebase: `src/backtesting/full_backtest_pipeline.py` (BacktestComparison), `src/database/models/performance.py` (StrategyPerformance)

---

### 4. BotMutationProfile

**Purpose:** Mutation lineage and mutation safety boundaries
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `allowed_mutation_areas: List[str]` — spec fields that can mutate
- `locked_components: List[str]` — spec fields that are fixed
- `lineage: List[str]` — ordered list of ancestor bot IDs
- `parent_variant_id: Optional[str]` — immediate parent variant
- `generation: int` — generation number in lineage
- `unsafe_areas: List[str]` — mutation areas flagged as unsafe

**Producers:** MutationEngine (WF2 improvement loop)
**Consumers:** Composer (validates mutations), EvaluationBridge (attaches results)
**Sync/Async:** Sync (loaded with BotSpec)
**Evidence:** Codebase: `src/router/variant_registry.py` (BotVariant with lineage tracking)

---

### 5. MarketContext

**Purpose:** Current market regime and health — the primary market state object
**Boundary:** CORE / BRIDGE

**Required fields:**
- `regime: RegimeType` — TREND_STABLE / RANGE_STABLE / HIGH_CHAOS / BREAKOUT_PRIME / NEWS_EVENT / UNCERTAIN
- `chaos_score: float` — 0.0-1.0 Lyapunov-derived chaos
- `regime_quality: float` — 0.0-1.0 scalar for sizing
- `susceptibility: float` — 0.0-1.0 Ising susceptibility
- `is_systemic_risk: bool`
- `news_state: NewsState` — ACTIVE / KILL_ZONE / CLEAR
- `symbol: str`
- `timeframe: str`
- `timestamp: datetime`

**Optional fields:**
- `hmm_regime: Optional[str]`
- `hmm_confidence: Optional[float]`
- `hmm_agreement: Optional[float]` — Ising vs HMM agreement ratio
- `sqs: Optional[float]` — spread quality score

**Multi-timeframe regime fields (Q-2 decision — support via bridge, not library wrapping):**
- `timeframe_regimes: Optional[Dict[str, str]]` — e.g., `{"M1": "trend", "M5": "trend", "M15": "compression_breakout_candidate"}`
- `timeframe_alignment_score: Optional[float]` — 0.0-1.0 agreement across timeframes
- `regime_confidence: Optional[float]` — 0.0-1.0 confidence in dominant regime
- `volatility_regime: Optional[str]` — e.g., "elevated", "normal", "compressed"
- `liquidity_regime: Optional[str]` — e.g., "healthy", "thin", "stressed"
- `produced_by: Optional[str]` — source system e.g., "MultiTimeframeSentinel"

**Design note (Q-2):** MultiTimeframeSentinel stays external. Library MarketContext carries the outputs via SentinelBridge. Do NOT wrap MultiTimeframeSentinel as a library feature module. Derived helpers (multi-timeframe agreement score, HTF trend alignment flag) are acceptable as library-level derived features.

**Producers:** SentinelBridge (from RegimeReport)
**Consumers:** FeatureEvaluator (sync read), RiskBridge, BotStateManager
**Sync/Async:** Hybrid (async updates from sentinel, sync reads at decision time)
**Evidence:** Codebase: `src/router/sentinel.py` (RegimeReport), `src/risk/physics/` (sensor outputs)

---

### 6. SessionContext

**Purpose:** Current trading session state
**Boundary:** CORE

**Required fields:**
- `session_id: str` — e.g., "london", "new_york", "london_newyork_overlap"
- `is_active: bool`
- `is_blackout: bool`
- `blackout_reason: Optional[str]`
- `minutes_to_event: Optional[int]` — for news blackouts
- `currencies: List[str]` — currencies exposed in this session

**Producers:** SessionBridge (from SessionDetector + NewsBlackout)
**Consumers:** FeatureEvaluator, BotStateManager, SafetyHooks
**Sync/Async:** Hybrid
**Evidence:** Codebase: `src/router/session_detector.py`, `src/market/news_blackout.py` (SessionBlackoutStatus)

---

### 7. FeatureVector

**Purpose:** Computed feature values at a point in time — the output of the feature evaluation pipeline
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `timestamp: datetime`
- `features: Dict[str, float]` — feature_id → computed value
- `feature_confidence: Dict[str, FeatureConfidence]` — feature_id → quality tag
- `market_context_snapshot: Optional[MarketContext]` — regime at computation time

**FeatureConfidence enum:**
```python
@dataclass
class FeatureConfidence:
    source: str                  # "broker_feed" / "proxied" / "approximated"
    quality: float               # 0.0-1.0 confidence in data
    latency_ms: float            # data freshness
    feed_quality_tag: str       # HIGH / MEDIUM / LOW
```

**Producers:** FeatureEvaluator
**Consumers:** IntentEmitter, RiskBridge, DPRBridge
**Sync/Async:** Sync (computed on-demand, cached)
**Evidence:** Codebase: SVSS `IndicatorResult` as partial substitute; `src/risk/physics/hmm/features.py` (FeatureConfig)

---

### 8. OrderFlowSignal

**Purpose:** Order flow proxy signal from tick/price/volume data
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `timestamp: datetime`
- `symbol: str`
- `direction: SignalDirection` — BUY / SELL / NEUTRAL
- `strength: float` — 0.0-1.0
- `source: OrderFlowSource` — TICK_ACTIVITY / SPREAD_BEHAVIOR / VOLUME_PRESSURE / SESSION_VOLUME
- `confidence: FeatureConfidence`
- `supporting_evidence: Dict[str, Any]` — raw metrics that produced this signal

**Note:** Order flow is quality-aware (per trading_platform_decision_memo §5). The `confidence` field carries feed quality tagging so bots can degrade gracefully.

**Producers:** OrderFlowFeature modules
**Consumers:** IntentEmitter (as confirmation input), SentinelBridge
**Sync/Async:** Sync (evaluated with feature stack)
**Evidence:** Codebase: No existing OrderFlowSignal schema; partial in SVSS indicators

---

### 9. PatternSignal (Detailed Placeholder — Q-3 Decision)

**Purpose:** Chart pattern detection output (out of V1 scope — internal engine deferred, external contract defined now)
**Boundary:** OUT-OF-SCOPE (internal) / BRIDGE (contract)

**Design note (Q-3):** Detailed placeholder schema now, internal pattern engine later. Define the external interface now so other library components can depend on it. Defer: pattern engine internals, full ontology of pattern types, ML model specifics.

**Required fields:**
- `signal_id: str` — unique pattern signal identifier
- `symbol: str`
- `timeframe: str`
- `pattern_type: str` — e.g., "range_breakout_candidate", "session_range_boundary"
- `direction: str` — "bullish", "bearish", "neutral"
- `confidence: float` — 0.0-1.0
- `source: str` — producer system, e.g., "pattern_service", "sentinel"
- `as_of: datetime`

**Strongly recommended:**
- `role: str` — "setup", "confirmation", "warning", "blocker"
- `strength: float` — 0.0-1.0 pattern strength metric
- `window_start: datetime`
- `window_end: datetime`
- `metadata: Dict[str, Any]` — producer version, notes
- `features: Dict[str, float]` — supporting metrics (compression_score, boundary_integrity, volume_support)

**Optional:**
- `invalidates_at: Optional[datetime]`
- `linked_context_id: Optional[str]`
- `session_scope: Optional[str]`

**Not to detail yet:** geometric detection internals, full candlestick taxonomy, image/chart-processing, neural-network outputs.

```python
@dataclass
class PatternSignal:
    signal_id: str
    symbol: str
    timeframe: str
    pattern_type: str
    direction: str
    confidence: float
    source: str
    as_of: datetime
    role: str = "setup"
    strength: float = 0.0
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    invalidates_at: Optional[datetime] = None
    linked_context_id: Optional[str] = None
    producer_version: str = "v1"
```

**Evidence:** Memo §9 — "deferred/placeholder as future service candidate"

---

### 10. SentinelState

**Purpose:** Full sentinel/market intelligence state snapshot
**Boundary:** BRIDGE

**Required fields:**
- `regime_report: RegimeReport` — current regime classification
- `sensor_states: Dict[str, SensorState]` — per-sensor states
- `hmm_state: Optional[HMMState]` — HMM regime if enabled
- `ensemble_vote: str` — dominant regime from ensemble

**Producers:** SentinelBridge
**Consumers:** RiskBridge, DPRBridge, SafetyHooks
**Sync/Async:** Async (event-driven updates)
**Evidence:** Codebase: `src/router/sentinel.py` (Sentinel internal state)

---

### 11. RiskEnvelope

**Purpose:** Risk system authorization output — the approved risk mandate for a trade
**Boundary:** BRIDGE

**Required fields:**
- `allocation_scalar: float` — 0.0-1.0, physics-based throttling
- `risk_mode: RiskMode` — STANDARD / CLAMPED / HALTED
- `position_size: float` — approved lot size
- `kelly_fraction: float`
- `risk_amount: float` — $ amount at risk
- `kelly_adjustments: List[str]` — audit trail
- `mode: str` — demo / live

**Optional fields:**
- `stop_model: Optional[str]` — recommended stop model (breakeven/trailing/immediate)
- `constraints: List[str]` — execution constraints
- `tier: Optional[str]` — progressive kill switch tier level

**Producers:** RiskBridge (from Governor → RiskMandate)
**Consumers:** Commander (execution authorization), ExecutionBridge
**Sync/Async:** Sync (computed at decision time)
**Evidence:** Codebase: `src/router/governor.py` (RiskMandate as existing substitute)

---

### 12. TradeIntent

**Purpose:** Bot's trade request — what the bot wants to do
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `symbol: str`
- `direction: TradeDirection` — LONG / SHORT
- `confidence: float` — 0.0-1.0
- `size_requested: float` — lot size requested
- `rationale: str` — human-readable rationale
- `feature_vector: FeatureVector` — what produced this intent
- `market_context_snapshot: MarketContext` — regime at decision time
- `capability_decl: str` — capability ID that produced this intent
- `timestamp: datetime`

**Optional fields:**
- `entry_price: Optional[float]`
- `stop_loss_pips: Optional[float]`
- `take_profit_pips: Optional[float]`

**Producers:** IntentEmitter
**Consumers:** RiskBridge (TradeIntent → RiskEnvelope validation), ExecutionBridge
**Sync/Async:** Sync (emitted at decision time)
**Evidence:** Codebase: No TradeIntent dataclass; execution via Governor/Commander pipeline directly

---

### 13. ExecutionDirective

**Purpose:** Governor's decision on what to actually execute — the authorized trade
**Boundary:** CORE

**Required fields:**
- `trade_intent: TradeIntent` — the original intent
- `risk_envelope: RiskEnvelope` — the risk authorization
- `approved: bool` — whether to proceed
- `approved_size: float` — lot size after risk processing
- `stop_loss_pips: float`
- `take_profit_pips: float`
- `execution_constraints: List[str]` — extra constraints from Governor
- `rejection_reason: Optional[str]` — if not approved
- `timestamp: datetime`

**Producers:** Governor (via RiskBridge translation of RiskMandate)
**Consumers:** Commander, ExecutionBridge, cTraderExecutionAdapter
**Sync/Async:** Sync
**Evidence:** Codebase: `src/router/governor.py` → `src/router/commander.py` direct pipeline; no explicit schema

---

### 14. EvaluationResult

**Purpose:** Standardized evaluation output across all evaluation modes
**Boundary:** EVALUATION

**Required fields:**
- `bot_id: str`
- `mode: EvaluationMode` — VANILLA / SPICED / VANILLA_FULL / SPICED_FULL / MODE_B / MODE_C
- `sharpe_ratio: float`
- `max_drawdown: float` — percentage
- `win_rate: float`
- `profit_factor: float`
- `total_trades: int`
- `return_pct: float`
- `kelly_score: float`
- `passes_gate: bool` — >= 4 of 6 modes threshold
- `regime_distribution: Optional[Dict[str, int]]` — for SPICED modes
- `filtered_trades: Optional[int]` — trades skipped by regime filter

**Producers:** EvaluationBridge (from FullBacktestPipeline)
**Consumers:** DPRBridge, BotEvaluationProfile builder, RegistryBridge
**Sync/Async:** Sync (written after evaluation)
**Evidence:** Codebase: `src/backtesting/mt5_engine.py` (MT5BacktestResult), `src/backtesting/mode_runner.py` (SpicedBacktestResult)

---

### 15. RegistryRecord

**Purpose:** Bot registration record in the library's own registry
**Boundary:** CORE

**Required fields:**
- `bot_id: str`
- `bot_spec: BotSpec`
- `registered_at: datetime`
- `status: RegistryStatus` — ACTIVE / SUSPENDED / RETIRED
- `parent_id: Optional[str]` — for variants

**Producers:** RegistryBridge (on BotSpec creation)
**Consumers:** Composer, FeatureEvaluator, LifecycleBridge
**Sync/Async:** Sync (registration)
**Evidence:** Codebase: `src/router/bot_manifest.py` (BotRegistry), `src/database/models/bot_registry.py` (BotRegistryModel)

---

### 16. BotPerformanceSnapshot

**Purpose:** Periodic performance snapshot for DPR and registry updates
**Boundary:** BRIDGE

**Required fields:**
- `bot_id: str`
- `snapshot_time: datetime`
- `session_wr: float`
- `net_pnl: float`
- `consistency: float`
- `ev_per_trade: float`
- `max_drawdown: float`
- `trade_count: int`
- `dpr_score: float`

**Producers:** DPRBridge (computed from trade journal)
**Consumers:** BotRegistry, LifecycleBridge, DPRBridge (Redis write)
**Sync/Async:** Async (periodic snapshots)
**Evidence:** Codebase: `src/router/dpr_scoring_engine.py` (BotSessionMetrics)

---

### 17. CapabilitySpec

**Purpose:** Declaration of what a module needs and what it produces — enables automatic composition validation
**Boundary:** CORE

**Required fields:**
- `module_id: str`
- `provides: List[str]` — capability IDs this module provides
- `requires: List[str]` — capability IDs this module needs (hard dependencies)
- `optional: List[str]` — optional capability IDs
- `outputs: List[Type[DomainObject]]` — schema types this module emits
- `compatibility: List[str]` — archetype IDs this works with
- `sync_mode: bool` — True = sync interface, False = async

**Producers:** Feature module registration, Archetype registration
**Consumers:** Composer (validates compositions), RequirementResolver
**Sync/Async:** Sync (loaded with spec)
**Evidence:** Codebase: No CapabilitySpec exists; memo §7 defines the concept

---

### 18. DependencySpec

**Purpose:** What a module requires to run — used for dependency resolution
**Boundary:** CORE

**Required fields:**
- `module_id: str`
- `requires: List[str]` — capability IDs needed
- `optional: List[str]` — optional capability IDs
- `conflict: List[str]` — incompatible capability IDs

**Producers:** Module registration
**Consumers:** RequirementResolver, CompositionValidator
**Sync/Async:** Sync
**Evidence:** Codebase: TRDDocument has `dependencies` as strings; no structured DependencySpec

---

### 19. CompatibilityRule

**Purpose:** Rules governing what modules can or cannot be combined
**Boundary:** CORE

**Required fields:**
- `rule_id: str`
- `subject: str` — capability/module ID
- `predicate: str` — COMBINES_WITH / EXCLUDES / REQUIRES
- `target: str` — other capability/module ID
- `condition: Optional[str]` — optional condition expression

**Producers:** Module registration, manual rules
**Consumers:** CompositionValidator
**Sync/Async:** Sync
**Evidence:** Codebase: `src/router/bot_tag_registry.py` has mutual exclusivity rules for tags; no system-level compatibility rule

---

## Schema Type Enums

| Enum | Values | Used In |
|------|--------|---------|
| `RegimeType` | TREND_STABLE, RANGE_STABLE, HIGH_CHAOS, BREAKOUT_PRIME, NEWS_EVENT, UNCERTAIN | MarketContext |
| `NewsState` | ACTIVE, KILL_ZONE, CLEAR | MarketContext |
| `TradeDirection` | LONG, SHORT | TradeIntent, ExecutionDirective |
| `SignalDirection` | BUY, SELL, NEUTRAL | OrderFlowSignal |
| `OrderFlowSource` | TICK_ACTIVITY, SPREAD_BEHAVIOR, VOLUME_PRESSURE, SESSION_VOLUME | OrderFlowSignal |
| `RiskMode` | STANDARD, CLAMPED, HALTED | RiskEnvelope |
| `ActivationState` | ACTIVE, SUSPENDED, QUARANTINED, RETIRED | BotRuntimeProfile |
| `BotHealth` | HEALTHY, DEGRADED, CRITICAL | BotRuntimeProfile |
| `EvaluationMode` | VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C | EvaluationResult |
| `RegistryStatus` | ACTIVE, SUSPENDED, RETIRED | RegistryRecord |
| `FeatureConfidenceLevel` | HIGH, MEDIUM, LOW | FeatureConfidence |
| `BotTier` | TIER_1, TIER_2 | BotRuntimeProfile, SSLCircuitBreaker |

---

## Sync/Async Classification Summary

| Object | Boundary | Sync/Async | Notes |
|--------|----------|-----------|-------|
| BotSpec | CORE | Sync | Loaded at startup |
| BotRuntimeProfile | CORE | Hybrid | Sync reads, async writes |
| BotEvaluationProfile | EVALUATION | Sync | Written post-evaluation |
| BotMutationProfile | CORE | Sync | Loaded with BotSpec |
| MarketContext | CORE/BRIDGE | Hybrid | Async updates, sync reads |
| SessionContext | CORE | Hybrid | |
| FeatureVector | CORE | Sync | Computed on-demand |
| OrderFlowSignal | CORE | Sync | |
| PatternSignal | OUT-OF-SCOPE | — | Deferred |
| SentinelState | BRIDGE | Async | Event-driven |
| RiskEnvelope | BRIDGE | Sync | Computed at decision time |
| TradeIntent | CORE | Sync | Emitted at decision time |
| ExecutionDirective | CORE | Sync | |
| EvaluationResult | EVALUATION | Sync | Written post-evaluation |
| RegistryRecord | CORE | Sync | Registration |
| BotPerformanceSnapshot | BRIDGE | Async | Periodic snapshots |
| CapabilitySpec | CORE | Sync | Loaded with spec |
| DependencySpec | CORE | Sync | |
| CompatibilityRule | CORE | Sync | |