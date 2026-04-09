# 06 — Recommended Target Architecture: QuantMindLib V1

**Architecture date:** 2026-04-09
**Based on:** Memo §1-§17, codebase scan, gap analysis, `quantmindx_v1_proxy_microstructure_addendum.md`

**Update 2026-04-09:** Added Capability Matrix, explicit Sync/Async/Hybrid boundaries, market intelligence scan cross-check, updated feature families to 6 with microstructure layer. Updated C2 and C4 contradictions per actual code scan.

---

## 1. Design Principles

1. **Thin bots, strong platform** — Bots describe setups, confirmations, feature usage, and entry requests. They do NOT own risk, news, portfolio-wide controls, or kill-switch behavior. Platform systems own these through bridges.

2. **Specs drive composition, not raw code** — Bot configuration lives in structured specs (BotSpec, FeatureSpec, ArchetypeSpec), not scattered across Python modules. Specs are serializable, versioned, and auditable.

3. **Platform-aware, not platform-bound** — The library knows about cTrader, but core domain logic uses vendor-neutral contracts. All broker-specific code lives in adapters.

4. **Sync/async is explicit** — Every module boundary declares sync vs async behavior. No informal mixing.

5. **Capability declarations first** — Before building any module, declare what it requires (DependencySpec) and what it produces (OutputSpec). This enables automatic composition validation.

6. **Bridge, don't replace** — Existing strong systems (risk governor, DPR, sentinel, registry, workflows) are integration targets, not replacement candidates. The library wraps them.

---

## 2. Package Architecture

```
src/library/                          # QuantMindLib root
├── __init__.py
├── pyproject.toml                    # Library package definition
│
├── core/                             # Core domain (vendor-neutral)
│   ├── __init__.py
│   ├── domain/
│   │   ├── bot_spec.py               # BotSpec, BotRuntimeProfile, BotEvaluationProfile, BotMutationProfile
│   │   ├── market_context.py         # MarketContext, SessionContext, FeatureVector
│   │   ├── trade.py                  # TradeIntent, ExecutionDirective, OrderFlowSignal
│   │   ├── evaluation.py             # EvaluationResult, EvaluationCriteria, TestCase
│   │   └── registry.py               # RegistryRecord, BotPerformanceSnapshot
│   ├── contracts/
│   │   ├── capability.py             # CapabilitySpec, DependencySpec, CompatibilityRule, OutputSpec
│   │   ├── adapter.py                # IMarketDataAdapter, IExecutionAdapter, IRegistryAdapter
│   │   └── bridge.py                 # IBridge, SyncBridge, AsyncBridge, HybridBridge
│   ├── schemas/
│   │   ├── bot_spec_schema.py        # Pydantic schemas for all bot specs
│   │   ├── feature_schema.py         # FeatureVector, FeatureSpec schemas
│   │   ├── evaluation_schema.py      # Evaluation schemas
│   │   └── trade_schema.py           # TradeIntent, ExecutionDirective schemas
│   ├── registries/
│   │   ├── feature_registry.py       # FeatureRegistry singleton
│   │   ├── archetype_registry.py     # ArchetypeRegistry singleton
│   │   └── spec_registry.py          # SpecRegistry for versioned specs
│   └── types/
│       ├── regime.py                 # RegimeType enum
│       ├── bot_state.py              # BotState, BotTier enums
│       ├── signal_strength.py        # SignalStrength (-1.0 to 1.0)
│       └── feature_confidence.py     # FeatureConfidence (quality-aware tagging)
│
├── features/                         # Reusable feature modules
│   ├── __init__.py
│   ├── indicators/                   # Technical indicators
│   │   ├── base.py                   # FeatureModule (ABC), FeatureResult
│   │   ├── rsi.py                    # RSIFeature
│   │   ├── atr.py                    # ATRFeature
│   │   ├── macd.py                   # MACDFeature
│   │   └── vwap.py                  # VWAPFeature (wraps SVSS VWAPIndicator)
│   ├── volume/                       # Volume-based features
│   │   ├── base.py                   # VolumeFeature (ABC)
│   │   ├── rvol.py                  # RVOLFeature (wraps SVSS RVOLIndicator)
│   │   ├── profile.py                # VolumeProfileFeature (wraps SVSS VolumeProfileIndicator)
│   │   ├── mfi.py                   # MFIFeature (wraps SVSS MFIIndicator)
│   │   └── imbalance.py             # VolumeImbalanceFeature (delta, pressure at levels) [V1 deferred]
│   ├── microstructure/               # Proxy market microstructure features (NEW in V1)
│   │   ├── base.py                   # MicrostructureFeature (ABC)
│   │   ├── spread_state.py          # SpreadStateFeature (native supported)
│   │   ├── top_of_book.py           # TopOfBookPressureFeature (native supported)
│   │   ├── multi_level_depth.py     # MultiLevelDepthFeature (native supported)
│   │   ├── aggression_proxy.py      # AggressionProxyFeature (proxy/inferred)
│   │   ├── absorption_proxy.py      # AbsorptionProxyFeature (proxy/inferred)
│   │   ├── breakout_pressure.py    # BreakoutPressureProxyFeature (proxy/inferred)
│   │   ├── liquidity_stress.py      # LiquidityStressProxyFeature (proxy/inferred)
│   │   └── context.py              # MicrostructureContext aggregation
│   ├── orderflow/                    # Order flow (executed trade-flow, V1 deferred)
│   │   ├── base.py                   # OrderFlowFeature (ABC)
│   │   ├── tick_activity.py         # TickActivityFeature (V1 deferred, external required)
│   │   ├── spread_behavior.py       # SpreadBehaviorFeature [superseded by microstructure/spread]
│   │   └── session_volume.py        # SessionVolumeFeature (partial, OHLCV available)
│   ├── session/                      # Session-aware features
│   │   ├── base.py                   # SessionFeature (ABC)
│   │   ├── session_detector.py      # SessionDetectorFeature
│   │   └── blackout.py              # SessionBlackoutFeature
│   └── transforms/                   # Data transformations
│       ├── base.py                   # Transform (ABC)
│       ├── normalize.py             # NormalizeTransform
│       ├── rolling.py               # RollingWindowTransform
│       └── resample.py             # ResampleTransform
│
├── archetypes/                      # Strategy archetypes
│   ├── __init__.py
│   ├── base.py                       # BaseArchetype (ABC), ArchetypeSpec
│   ├── breakout_scalper.py          # BreakoutScalper
│   ├── pullback_scalper.py          # PullbackScalper
│   ├── orb.py                        # OpeningRangeBreakout
│   ├── mean_reversion.py            # MeanReversion
│   └── session_transition.py        # SessionTransition
│
├── composition/                      # Spec-driven composition
│   ├── __init__.py
│   ├── composer.py                   # Composer — builds bot from specs
│   ├── validator.py                  # CompositionValidator — validates spec compatibility
│   ├── requirements.py               # RequirementResolver — resolves dependencies
│   └── mutation.py                   # MutationEngine — applies constrained mutations to specs
│
├── runtime/                          # Runtime execution
│   ├── __init__.py
│   ├── intent_emitter.py             # IntentEmitter — bot emits TradeIntent from spec
│   ├── feature_evaluator.py          # FeatureEvaluator — evaluates feature stack
│   ├── state_manager.py             # BotStateManager — manages runtime state
│   └── safety_hooks.py               # SafetyHooks — kill switch, circuit breaker integration
│
├── adapters/                         # Platform adapters (cTrader V1)
│   ├── __init__.py
│   ├── base.py                       # PlatformAdapter (ABC), AdapterConfig
│   ├── ctrader/
│   │   ├── __init__.py
│   │   ├── client.py                 # CTraderClient — cTrader Open API Python SDK wrapper
│   │   ├── market_adapter.py         # CTraderMarketAdapter (IMarketDataAdapter impl)
│   │   ├── execution_adapter.py     # CTraderExecutionAdapter (IExecutionAdapter impl)
│   │   ├── types.py                  # cTrader-specific type conversions
│   │   └── converter.py              # OHLCV, tick, depth converters
│   └── external/
│       ├── __init__.py
│       ├── order_flow_adapter.py     # IExternalOrderFlowAdapter (optional, V1 deferred)
│       └── data_source_registry.py   # Registry of external data sources
│
└── bridges/                          # Two-way bridges to existing systems
    ├── __init__.py
    ├── to_sentinel/
    │   ├── __init__.py
    │   ├── sentinel_bridge.py        # SentinelBridge — maps RegimeReport ↔ MarketContext
    │   ├── types.py                  # RegimeReport → MarketContext converter
    │   └── config.py                 # SentinelBridgeConfig
    ├── to_risk/
    │   ├── __init__.py
    │   ├── risk_bridge.py             # RiskBridge — maps TradeIntent ↔ RiskMandate
    │   ├── governor_mapper.py        # Governor ↔ RiskEnvelope mapper
    │   └── sizing_mapper.py          # EnhancedKelly ↔ position sizing bridge
    ├── to_registry/
    │   ├── __init__.py
    │   ├── registry_bridge.py         # RegistryBridge — BotSpec ↔ BotRegistry/BotTagRegistry
    │   ├── tag_mapper.py             # Tag operations mapper
    │   └── variant_mapper.py         # VariantRegistry ↔ BotMutationProfile
    ├── to_evaluation/
    │   ├── __init__.py
    │   ├── evaluation_bridge.py      # EvaluationBridge — library results ↔ StrategyPerformance
    │   ├── backtest_mapper.py       # BacktestResult → EvaluationResult mapper
    │   └── report_mapper.py         # BacktestReportSubAgent ↔ EvaluationResult
    ├── to_dpr/
    │   ├── __init__.py
    │   ├── dpr_bridge.py             # DPRBridge — BotEvaluationProfile ↔ DPRScore
    │   ├── score_publisher.py       # DPR Redis publisher (fixes G-18)
    │   └── tier_mapper.py            # DPR tier ↔ BotTier mapper
    ├── to_journal/
    │   ├── __init__.py
    │   └── journal_bridge.py          # JournalBridge — library events ↔ TradeRecord/audit log
    ├── to_workflows/
    │   ├── __init__.py
    │   ├── wf1_bridge.py              # WF1 bridge — BotSpec input, evaluation output
    │   └── wf2_bridge.py              # WF2 bridge — variant specs, mutation results
    └── to_lifecycle/
        ├── __init__.py
        └── lifecycle_bridge.py        # LifecycleBridge — promotion/quarantine decisions
```

---

## 3. Sync/Async/Hybrid Design

### Async Paths (event streams)
```
cTrader tick stream → CTraderMarketAdapter → normalized tick events
                                        ├──► Feature workers (async evaluation)
                                        ├──► SentinelBridge (regime updates)
                                        └──► StateManager (cache)

cTrader depth stream → CTraderMarketAdapter → market depth events
                                        └──► OrderFlowFeature (if enabled)

Sentinel regime events → SentinelBridge → MarketContext updates
                                        └──► FeatureEvaluator (sync read)

DPR score events → DPRBridge → Redis publish
                                        └──► BotRegistry updates

Kill switch events → SafetyHooks → position close directives
```

### Sync Paths (decision-time)
```
BotStateManager (sync read of cached FeatureVector + MarketContext)
    │
    ▼
FeatureEvaluator (sync compute on cached state)
    │
    ▼
IntentEmitter (sync emit TradeIntent)
    │
    ▼
RiskBridge (sync call Governor → RiskMandate)
    │
    ▼
ExecutionBridge (sync call Commander → ExecutionDirective)
```

### Mixed Mode Pattern (Canonical)
```
async tick stream → normalized event bus → cached state store
                                        └──► FeatureEvaluator (sync read)
                                        └──► IntentEmitter (sync read)
                                        └──► RegistryBridge (async write)
                                        └──► JournalBridge (async write)
```

---

## 4. Library Core Layer (Vendor-Neutral)

### BotSpec: The Central Contract

BotSpec is the primary interface between all systems. It is a multi-profile object:

```python
@dataclass
class BotSpec:
    # Static profile (immutable during runtime)
    id: str
    archetype: str                           # e.g., "opening_range_breakout"
    symbol_scope: List[str]                 # e.g., ["EURUSD"]
    sessions: List[str]                    # e.g., ["london", "london_newyork_overlap"]
    features: List[str]                     # feature family IDs
    confirmations: List[str]                # confirmation rule IDs
    execution_profile: str                  # e.g., "scalp_fast_v1"
    dependencies: List[str]                 # bridge dependency IDs
    capability_spec: CapabilitySpec         # what this bot needs/produces
    # Runtime profile (managed by BotStateManager)
    runtime: Optional[BotRuntimeProfile]   # activation, health, DPR rank
    # Evaluation profile (written by EvaluationBridge)
    evaluation: Optional[BotEvaluationProfile]  # backtest, MC, WFA results
    # Mutation profile (managed by Composition/MutationEngine)
    mutation: Optional[BotMutationProfile]  # lineage, locked areas, parents
```

### CapabilitySpec: Module Contract System

```python
@dataclass
class CapabilitySpec:
    provides: List[str]                      # capability IDs this module provides
    requires: List[str]                     # capability IDs this module needs
    optional: List[str]                     # optional capability IDs
    outputs: List[Type[DomainObject]]     # schemas this module emits
    compatibility: List[str]               # archetype IDs this works with
    sync_mode: bool                        # True = sync interface, False = async
```

This allows automatic validation: can this bot run with these features given these capabilities?

### TradeIntent: Bot Decision Output

```python
@dataclass
class TradeIntent:
    bot_id: str
    symbol: str
    direction: TradeDirection              # LONG / SHORT
    confidence: float                      # 0.0-1.0
    entry_price: Optional[float]
    stop_loss_pips: Optional[float]
    take_profit_pips: Optional[float]
    size_requested: float                  # lot size request
    rationale: str                         # human-readable rationale
    feature_vector: FeatureVector          # what features produced this
    market_context_snapshot: MarketContext # regime at time of decision
    capability_decl: str                   # capability ID that produced this
    timestamp: datetime
```

### RiskEnvelope: Risk System Output

```python
@dataclass
class RiskEnvelope:
    allocation_scalar: float                # 0.0-1.0
    risk_mode: RiskMode                    # STANDARD / CLAMPED / HALTED
    position_size: float                   # approved lot size
    kelly_fraction: float
    risk_amount: float                     # $ at risk
    kelly_adjustments: List[str]           # audit trail
    mode: str                              # demo / live
    stop_model: Optional[str]              # recommended stop model
    constraints: List[str]                  # execution constraints
```

### MarketContext: Regime Information

```python
@dataclass
class MarketContext:
    regime: RegimeType                      # TREND_STABLE / RANGE_STABLE / HIGH_CHAOS / BREAKOUT_PRIME / NEWS_EVENT / UNCERTAIN
    chaos_score: float                      # 0.0-1.0
    regime_quality: float                   # 0.0-1.0
    susceptibility: float                  # 0.0-1.0
    is_systemic_risk: bool
    news_state: NewsState                  # ACTIVE / KILL_ZONE / CLEAR
    hmm_regime: Optional[str]              # HMM regime if available
    hmm_confidence: Optional[float]
    hmm_agreement: Optional[float]         # Ising vs HMM agreement ratio
    symbol: str
    timeframe: str
    timestamp: datetime
```

---

## 5. Feature Registry V1

The feature registry is a singleton that tracks all available feature modules:

```python
class FeatureRegistry:
    def register(self, feature_id: str, feature: FeatureModule, spec: CapabilitySpec)
    def get(self, feature_id: str) -> Optional[FeatureModule]
    def requires(self, feature_id: str) -> List[str]    # dependencies
    def outputs(self, feature_id: str) -> List[Type]    # output schemas
    def compatible_with(self, feature_id: str, archetype: str) -> bool
    def find_by_capability(self, capability_id: str) -> List[str]
    def find_compatible(self, requires: List[str], excludes: List[str]) -> List[str]
```

V1 feature families (6 families, see `10_feature_family_plan.md` for full detail):
1. **indicators** — RSI, ATR, MACD, VWAP (Native Supported, cTrader)
2. **volume** — RVOL, Volume Profile, MFI, Imbalance (Native Supported, cTrader)
3. **microstructure** — Spread dynamics, depth pressure, aggression proxy, absorption proxy, breakout pressure proxy, liquidity stress proxy (Native Supported + Proxy/Inferred, cTrader)
4. **orderflow_cat_a** — Spread Behavior, DOM Pressure, Depth Thinning (Native Supported, cTrader)
5. **orderflow_cat_b** — Volume Imbalance, Tick Activity, Session Volume (Future External-Data, V1 deferred)
6. **session** — Session Detector, Session Blackout (Native Supported, cTrader)
7. **transforms** — Normalize, Rolling Window, Resample (Native Supported)

---

## 6. Archetype System V1

### Base Archetypes (hand-defined)
- `BreakoutScalper` — breakout-based scalping
- `PullbackScalper` — pullback-based scalping
- `OpeningRangeBreakout` — ORB specifically (Phase 1 priority)
- `MeanReversion` — mean reversion
- `SessionTransition` — session-boundary strategies

### Derived Archetypes (generated from base + features + constraints)
```
LondonORB = OpeningRangeBreakder + [volume_confirmation, spread_ok]
            + sessions=["london", "london_newyork_overlap"]
            + constraints=[spread_filter < 1.5]

OverlapBreakout = OpeningRangeBreakout + [orderflow_filter]
                  + sessions=["london_newyork_overlap"]
                  + constraints=[min_volume > 0.8]

ChoppyReversion = MeanReversion + [chaos_filter, session_chop]
                  + constraints=[regime_quality < 0.5]
```

### Composition Rule
Archetypes are defined as spec compositions:
```python
@dataclass
class ArchetypeSpec:
    id: str
    base_archetype: str                    # base archetype ID
    required_features: List[str]           # feature family IDs
    optional_features: List[str]
    sessions: List[str]
    constraints: List[ConstraintSpec]
    compatible_with: List[str]            # feature compatibility
```

---

## 7. cTrader Adapter Boundary + Data Source of Truth

### What is Platform-Agnostic (Library Core)
- BotSpec, TradeIntent, RiskEnvelope, MarketContext schemas
- FeatureRegistry, ArchetypeRegistry
- Composer, Validator, IntentEmitter
- All bridge adapters (define interfaces, not platform logic)
- IExternalOrderFlowAdapter interface (platform-neutral)

### What is cTrader-Specific (Adapter Layer)
- CTraderClient and SDK wrapper
- OHLCV, tick, depth converters from cTrader format
- cTrader account state, position, order type mappings
- cTrader execution directives (order types, stops, limits)
- cTrader-specific error handling and reconnection logic

### Data Source of Truth Matrix

| Domain | Source of Truth | cTrader Role | External Role |
|--------|----------------|-------------|---------------|
| **Live execution** | cTrader | Direct | N/A |
| **Live depth/liquidity state** | cTrader | Direct | N/A |
| **Live quotes/ticks** | cTrader | Direct | N/A |
| **Account / position state** | cTrader | Direct | N/A |
| **Historical research/backtesting** | External canonical | Integration point only | Authoritative |
| **True executed trade flow** | External only | Not available | Authoritative |
| **Economic calendar / news** | External machine-readable | Not in Open API | Authoritative |

Key principle: cTrader is the **live execution and market-state adapter**. It is NOT the authoritative source for historical data or true trade flow. These domains always route through external sources.

### What Must Be Hidden Behind Adapters
- Direct MT5 socket calls → must route through IMarketDataAdapter
- Direct MT5 order execution → must route through IExecutionAdapter
- MQL5 indicator calls → must route through feature registry
- All broker-specific configuration → must be adapter-config only

### Adapter Interface Contracts
```python
class IMarketDataAdapter(ABC):
    @async
    def tick_stream(self, symbols: List[str]) -> AsyncIterator[TickEvent]
    @async
    def bar_stream(self, symbols: List[str], timeframe: str) -> AsyncIterator[BarEvent]
    @async
    def depth_stream(self, symbol: str) -> AsyncIterator[DepthEvent]
    def get_historical(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> List[Bar]
    def get_account_state(self) -> AccountState

class IExecutionAdapter(ABC):
    def send_order(self, directive: ExecutionDirective) -> ExecutionResult
    def cancel_order(self, order_id: str) -> bool
    def get_positions(self, symbol: Optional[str]) -> List[Position]
    def get_orders(self, symbol: Optional[str]) -> List[Order]
```

---

## 8. Runtime Intent Flow

```
[1] cTrader tick arrives via CTraderMarketAdapter (async)
        │
        ▼
[2] MarketContext cached (updated by SentinelBridge async)
        │
        ▼
[3] FeatureEvaluator evaluates feature stack (sync, on-demand)
        │
        ▼
[4] FeatureVector produced (FeatureVector dataclass)
        │
        ▼
[5] BotStateManager reads cached FeatureVector + MarketContext (sync)
        │
        ▼
[6] IntentEmitter computes TradeIntent from BotSpec + FeatureVector (sync)
        │
        ▼
[7] TradeIntent emitted (sync publish)
        │
        ├─► RiskBridge → Governor → RiskEnvelope (sync)
        │        │
        │        ▼
        │    Commander → ExecutionDirective → cTraderExecutionAdapter → cTrader
        │
        ├─► RegistryBridge → BotRegistry update (async)
        │
        ├─► DPRBridge → DPR scoring + Redis publish (async)
        │
        └─► JournalBridge → TradeRecord audit log (async)
```

---

## 9. Bot Lifecycle Through Library

```
[TRDDocument] ──(TRD→BotSpec converter)──► [BotSpec.static]
                                          │
                                          ▼
                                    [FeatureRegistry]
                                    check capabilities
                                          │
                                          ▼
                                    [ArchetypeRegistry]
                                    validate archetype compatibility
                                          │
                                          ▼
                                    [Composer]
                                    compose bot from spec
                                          │
                                          ├─► [BotStateManager]
                                          │         │
                                          │         ▼
                                          │   [Runtime loop]
                                          │    tick → feature → intent → risk → execution
                                          │
                                          └─► [EvaluationBridge]
                                                    │
                                                    ▼
                                          [6-mode backtest + MC + WFA]
                                                    │
                                                    ▼
                                          [StrategyPerformance DB]
                                                    │
                                                    ▼
                                          [DPRBridge → Redis + BotRegistry]
                                                    │
                                                    ▼
                                          [LifecycleBridge → promote/quarantine]
```

---

## 10. Out-of-Scope for V1

Per memo §3, confirmed by gap analysis, and addendum:
- Chart-pattern analysis service (reserved as future service candidate)
- Footprint chart rendering
- Replacing existing ML internals (sentinel physics sensors, HMM, BOCPD, MS-GARCH)
- Replacing kill-switch internals
- Replacing risk-engine internals
- Replacing DPR internals
- Replacing backtest report generation internals
- Multi-broker adapter framework (cTrader only in V1)
- **External order flow data adapter** (Category B executed trade-flow features, V1 deferred)
- **EnsembleVoter live wiring** (Phase 2 — voter exists but not wired to live tick path)
- **Neural pattern service** (Phase 3+)

---

## 11. Contradictions and Calling Out

### C1: Two Governor Classes
The codebase has `src/router/governor.py` (risk authorization) and `src/risk/governor.py` (risk execution/Tier 2 rules). This naming conflict must be resolved. Library bridge should reference by module path, not class name, until resolved.

**Resolution needed:** Rename one of the classes or establish a canonical import path.

### C2: DPR Dual Engines
Two DPR engines exist (router + risk layers). Code scan confirms: router engine (`src/router/dpr_scoring_engine.py`) writes DPR composite scores to Redis (`dpr:score:{bot_id}`) via `_write_score_to_redis()` — fix is in uncommitted code. Risk engine (`src/risk/dpr/scoring_engine.py`) writes session concern counters only, NOT DPR composite scores.

**Resolution confirmed (BLOCKER-2, Option B):** DPRBridge handles both engines. Router engine is canonical for DPR composite scores. Risk engine session concern counters consumed as-is. No engine consolidation needed. Redis write path confirmed in router engine uncommitted fix.

### C3: MT5 Embedded Throughout
Backtesting engine (`mt5_engine.py`) simulates MQL5 environment. cTrader adapter must replace this without breaking the evaluation pipeline.

**Resolution needed:** Create CTrader backtest engine that produces same result schemas as `MT5BacktestResult`, so evaluation pipeline doesn't need to change.

### C4: EnsembleVoter Not Wired into Live Tick Path
`EnsembleVoter` (`src/risk/physics/ensemble/voter.py`) implements weighted HMM+MS-GARCH+BOCPD voting with `AdaptiveWeightManager`. However, code scan confirms it is NOT wired into the live Sentinel tick path. Live path uses `HMMRegimeSensor` directly.

**Implication:** The library SentinelBridge should work with current `HMMRegimeSensor` output. EnsembleVoter wiring is a future Phase 2 enhancement — not required for V1. Document this as Phase 2 deferred wiring task.

### C5: Phase 1 Strategy Type Cleanup
`StrategyType` enum has SCALPER + ORB (active) + 6 legacy types. Library V1 should only handle SCALPER + ORB.

**Resolution needed:** Library validation should reject non-SCALSER/ORB archetypes until Phase 2.

---

## 12. Capability Matrix

Per `quantmindx_v1_proxy_microstructure_addendum.md`: Every feature or service in the proxy microstructure layer must declare capability.

### CapabilitySpec Examples

```python
CapabilitySpec(
    name="depth_snapshot_stream",
    provider="ctrader_adapter",
    quality="native_supported",
    mode="async",
)

DependencySpec(
    consumer="MultiLevelDepthFeature",
    requires=["depth_snapshot_stream"],
    optional=[],
)

CompatibilityRule(
    subject="AggressionProxyFeature",
    allowed_when=["quote_tick_stream", "depth_update_stream", "spread_state_available"],
    forbidden_when=["historical_only_mode_without_depth_proxy"],
)
```

### Capability Declaration Table (Proxy Market Microstructure Layer)

| Module | Capability ID | Requires | Provides | Quality | Mode | Bot/Sentinel/Shared |
|--------|-------------|----------|---------|---------|------|---------------------|
| SpreadStateFeature | `microstructure/spread` | `data/tick_stream` | `microstructure/spread` | native_supported | sync | Both |
| TopOfBookPressureFeature | `microstructure/top_book` | `data/depth_stream` | `microstructure/top_book` | native_supported | sync | Both |
| MultiLevelDepthFeature | `microstructure/depth_levels` | `data/depth_stream` | `microstructure/depth_levels` | native_supported | sync | Both |
| AggressionProxyFeature | `microstructure/aggression` | `data/tick_stream`, `data/depth_stream`, `data/spread_state` | `microstructure/aggression` | proxy_inferred | sync | Both |
| AbsorptionProxyFeature | `microstructure/absorption` | `data/depth_stream`, `data/spread_state` | `microstructure/absorption` | proxy_inferred | sync | Both |
| BreakoutPressureProxyFeature | `microstructure/breakout_pressure` | `data/depth_stream`, `data/spread_state`, `data/bars` | `microstructure/breakout_pressure` | proxy_inferred | sync | Both |
| LiquidityStressProxyFeature | `microstructure/liquidity_stress` | `data/depth_stream` | `microstructure/liquidity_stress` | proxy_inferred | sync | Both |
| MicrostructureContext (aggregation) | `context/microstructure` | all microstructure outputs | `context/microstructure` | mixed | sync | Both |

### Core Library Capabilities

| Capability ID | Provider | Quality | Mode | Notes |
|---------------|----------|---------|------|-------|
| `data/tick_stream` | CTraderMarketAdapter | native_supported | async | Bid/ask quotes, no tick volume |
| `data/depth_stream` | CTraderMarketAdapter | native_supported | async | DOM delta events |
| `data/bar_stream` | CTraderMarketAdapter | native_supported | async | OHLCV bars |
| `data/historical` | External adapter | external | sync | Dukascopy / self-recorded |
| `data/external_order_flow` | IExternalOrderFlowAdapter | external | async/sync | V1-deferred |
| `regime/report` | SentinelBridge | native_supported | async | RegimeReport → MarketContext |
| `risk/envelope` | RiskBridge | native_supported | sync | Governor → RiskEnvelope |
| `registry/bot` | RegistryBridge | native_supported | async | BotRegistry ↔ BotSpec |
| `evaluation/results` | EvaluationBridge | native_supported | sync | EvaluationResult collection |

---

## 13. Market Intelligence Cross-Check (Updated 2026-04-09)

Per addendum §2 instruction: The proxy microstructure layer must NOT duplicate existing market-intelligence logic. Code scan confirmed current reality:

### Confirmed Existing ML Components

| Component | Location | Role | V1 Library Action |
|-----------|----------|------|-----------------|
| HMM (Gaussian, 10-feature) | `src/risk/physics/hmm/` | Regime classification from OHLCV+Ising | Bridge only; library does NOT wrap HMM |
| MS-GARCH | `src/risk/physics/msgarch/` | Conditional volatility, vol regime (LOW/MED/HIGH) | Bridge only; library does NOT wrap |
| BOCPD | `src/risk/physics/bocpd/` | Regime changepoint detection (NOT steady-state) | Bridge only; library does NOT wrap |
| EnsembleVoter | `src/risk/physics/ensemble/voter.py` | Weighted HMM+MS-GARCH+BOCPD voting | Exists but NOT wired into live Sentinel tick path |
| Ising features (magnetization, susceptibility, energy) | `src/risk/physics/hmm/features.py` | Ising-inspired from tick data | Do NOT duplicate in microstructure layer |
| MultiTimeframeSentinel | `src/router/multi_timeframe_sentinel.py` | Per-timeframe regime, voting aggregation | Bridge only |
| NewsSensor | `src/router/sensors/news.py` | Kill zone from Finnhub | Wrapped via SessionBlackoutFeature |
| CalendarGovernor | `src/router/calendar_governor.py` | Economic calendar integration | Gap: no direct calendar feed to NewsSensor |
| DPR Redis gap | `src/router/dpr_scoring_engine.py` | Scores computed but Redis write path gap | Router fix in uncommitted code |

### Key Findings from Code Scan

1. **HMM regime** (8 states) is computed from 10 features (OHLCV + Ising). The regime → StrategyType suitability table (`STRATEGY_REGIME_SUITABILITY`) maps regime → bot type. This already exists — library should NOT replicate it.

2. **EnsembleVoter exists but not wired:** The voter in `src/risk/physics/ensemble/voter.py` implements weighted HMM+MS-GARCH+BOCPD voting but is NOT currently connected to the live Sentinel tick path. The live tick path uses `HMMRegimeSensor` directly. This is a future wiring task (Phase 2+).

3. **BOCPD is a changepoint detector, not a steady-state classifier:** It fires when `is_changepoint=True` (transition detected). Library should consume `is_transition` flag from MarketContext, not re-implement BOCPD.

4. **Ising-inspired features (magnetization, susceptibility, energy)** are computed in `HMMFeatureExtractor`. These are conceptually similar to aggression proxies but are derived from price sign sequences, not depth/spread. Library microstructure features are orthogonal — they use DOM/depth data, not price sign sequences.

5. **No MicrostructureContext exists** in the current system. The proxy microstructure layer is net-new development.

### Architectural Principle

The library's proxy microstructure layer:
- Exposes reusable depth/spread/quotes-derived features as modules
- Bridges to existing Sentinel regime output via `MarketContext`
- Does NOT wrap HMM, BOCPD, MS-GARCH, Ising models
- Does NOT absorb the sentinel's regime classification logic
- Lets the sentinel aggregate MicrostructureContext if useful for its own decisions

This keeps the library thin and preserves the sentinel's existing ML investments.

---

## 14. Sync / Async / Hybrid Boundaries (Explicit)

Per `quantmindx_v1_proxy_microstructure_addendum.md` §4 and `quantmindx_library_v1_architecture_memo.docx`: Every boundary must be explicitly declared.

### Async by Default

| Path | Mode | Reason |
|------|------|--------|
| cTrader tick stream | Async | WebSocket push events |
| cTrader depth stream | Async | WebSocket push events |
| cTrader bar stream | Async | WebSocket push or polling |
| External order flow stream | Async | WebSocket or polling |
| DPR score events | Async | Event-driven writes |
| Registry notifications | Async | Event-driven writes |
| Journal / audit streams | Async | Non-blocking writes |
| News / economic calendar events | Async | Polling or WebSocket |

### Sync by Default

| Path | Mode | Reason |
|------|------|--------|
| Historical data requests | Sync | One-shot request/response |
| Order submission | Sync | Must confirm before proceeding |
| Position snapshot | Sync | One-shot snapshot |
| Feature validation | Sync | Config-time checks |
| Compatibility checks | Sync | Static analysis |
| Spec loading | Sync | Config-time only |
| BotStateManager (read) | Sync | Reads from cached state |

### Hybrid Pattern (Canonical Runtime)

```
async tick stream → normalized event bus → cached state store
                                              ├─► FeatureEvaluator (sync read)
                                              ├─► IntentEmitter (sync read)
                                              ├─► BotStateManager (sync read)
                                              ├─► RiskBridge (sync call → Governor → RiskEnvelope)
                                              ├─► RegistryBridge (async write)
                                              ├─► DPRBridge (async write)
                                              └─► JournalBridge (async write)
```

### Library → Platform Sync Boundary

```
Bot requests current context (sync read of cached MicrostructureContext + MarketContext)
    │
    ▼
Risk/sentinel validation occurs (sync call via bridges)
    │
    ▼
TradeIntent emitted (sync emit)
    │
    ├─► RiskBridge (sync) → Governor (sync) → RiskEnvelope
    │
    └─► ExecutionBridge (sync) → cTraderExecutionAdapter (sync) → ExecutionResult
```