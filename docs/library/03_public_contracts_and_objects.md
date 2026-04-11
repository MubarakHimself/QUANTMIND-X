# QuantMindLib V1 — Public Contracts and Objects

## BotSpec and Profiles

### BotSpec (`src/library/core/domain/bot_spec.py`)
**The central contract.** Frozen dataclass — immutable during runtime. Primary interface between all systems.

```python
class BotSpec(BaseModel):
    bot_id: str
    archetype: str                          # e.g. "opening_range_breakout"
    symbol_scope: List[str]                # e.g. ["EURUSD"]
    sessions: List[str]                     # e.g. ["london", "london_newyork_overlap"]
    features: List[str]                    # feature IDs e.g. ["indicators/rsi_14"]
    confirmations: List[str]
    execution_profile: str
    strategy_type: StrategyType             # SCALPER, ORB
    max_position_size: float = 0.01
    max_daily_loss: float = 100.0
    max_slippage_ticks: int = 2
    capability_spec: Optional[Any] = None   # Phase 2 capability system
    # Runtime profile (managed separately)
    runtime: Optional[BotRuntimeProfile] = None
    # Evaluation profile (written by EvaluationBridge)
    evaluation: Optional[BotEvaluationProfile] = None
    # Mutation profile (managed by Composition/MutationEngine)
    mutation: Optional[BotMutationProfile] = None
```

### BotRuntimeProfile (`src/library/core/domain/bot_spec.py`)
Mutable runtime state. Written by DPRBridge, RegistryBridge.

```python
class BotRuntimeProfile(BaseModel):
    bot_id: str
    activation_state: ActivationState       # ACTIVE / PAUSED / STOPPED
    deployment_target: str                  # paper / demo / live
    health: BotHealth                      # HEALTHY / DEGRADED / FAILING
    session_eligibility: Dict[str, bool]
    dpr_ranking: int = 0                   # rank in DPR (1 = top)
    dpr_score: float = 0.0                 # 0-100
    dpr_tier: Optional[DPRTier] = None     # ELITE / PERFORMING / STANDARD / AT_RISK / CIRCUIT_BROKEN
    journal_id: Optional[str] = None
    report_ids: List[str] = Field(default_factory=list)
    daily_loss_used: float = 0.0
    open_risk: float = 0.0
```

### BotEvaluationProfile (`src/library/core/domain/bot_spec.py`)
Evaluation results across all modes. Written by EvaluationOrchestrator.

```python
class BotEvaluationProfile(BaseModel):
    bot_id: str
    backtest: BacktestMetrics
    monte_carlo: Optional[MonteCarloMetrics] = None
    walk_forward: Optional[WalkForwardMetrics] = None
    pbo_score: Optional[float] = None      # probability of backtest overfitting
    robustness_score: Optional[float] = None
    spread_sensitivity: Optional[float] = None
    session_scores: Dict[str, SessionScore] = Field(default_factory=dict)
```

### BotMutationProfile (`src/library/core/domain/bot_spec.py`)
Mutation lineage and safety boundaries. Managed by MutationEngine (WF2).

```python
class BotMutationProfile(BaseModel):
    bot_id: str
    allowed_mutation_areas: List[str]       # spec fields that can mutate
    locked_components: List[str]           # spec fields that are fixed
    lineage: List[str]                     # ordered list of ancestor bot IDs
    parent_variant_id: Optional[str] = None
    generation: int = 0
    unsafe_areas: List[str] = Field(default_factory=list)
```

## Market Context

### MarketContext (`src/library/core/domain/market_context.py`)
Current market regime and health. Written by SentinelBridge (async). Read at decision time (sync).

```python
class MarketContext(BaseModel):
    regime: RegimeType                     # TREND_STABLE / RANGE_STABLE / HIGH_CHAOS /
                                         # BREAKOUT_PRIME / NEWS_EVENT / UNCERTAIN
    news_state: NewsState                  # ACTIVE / KILL_ZONE / CLEAR
    regime_confidence: float = Field(ge=0.0, le=1.0)
    session_id: Optional[str] = None
    is_stale: bool = False
    spread_state: Optional[str] = None
    depth_snapshot: Optional[Dict[str, Any]] = None
    last_update_ms: int = 0
    trend_strength: float = 0.0
    volatility_regime: Optional[str] = None
    is_fresh(threshold_ms=5000) -> bool   # method
```

### RegimeReport (`src/library/core/domain/market_context.py`)
Input holder from SentinelBridge. Converted to MarketContext.

```python
class RegimeReport(BaseModel):
    regime: RegimeType
    chaos_score: float
    regime_quality: float
    susceptibility: float
    is_systemic_risk: bool
    news_state: NewsState
    symbol: str
    timeframe: str
    timestamp: datetime
```

### SessionContext (`src/library/core/domain/session_context.py`)
Current trading session state.

```python
class SessionContext(BaseModel):
    session_id: str
    is_active: bool = False
    is_blackout: bool = False
    blackout_reason: Optional[str] = None
    minutes_to_event: Optional[int] = Field(default=None, ge=0)
    currencies: List[str] = Field(default_factory=list)
```

## Feature System

### FeatureVector (`src/library/core/domain/feature_vector.py`)
Computed feature values at a point in time. Output of FeatureEvaluator.

```python
class FeatureVector(BaseModel):
    timestamp: datetime
    bot_id: str
    features: Dict[str, float]            # feature_id → computed value
    feature_confidence: Dict[str, FeatureConfidence]
    market_context_snapshot: Optional[MarketContext] = None
    # Helper methods
    get(feature_key) -> Optional[float]
    merge(other: FeatureVector) -> FeatureVector
    weighted_value(feature_key, weight) -> float
    quality_score() -> float
```

### FeatureConfidence (`src/library/core/domain/feature_vector.py`)
Quality metadata per feature.

```python
class FeatureConfidence(BaseModel):
    source: str                            # e.g. "ctrader_native", "library_transform"
    quality: float                         # 0.0-1.0
    latency_ms: float = 0.0
    feed_quality_tag: str                 # HIGH / MEDIUM / LOW / DISABLED / INSUFFICIENT_DATA
```

### FeatureModule ABC (`src/library/features/base/feature_module.py`)
Base class for all feature modules.

```python
class FeatureModule(BaseModel):
    # ABC — use concrete implementations
    feature_id: str
    quality_class: str = "native_supported"  # overridden by config for proxy features
    source: str
    enabled: bool = True
    model_post_init()                      # copies quality_class from config property

    @property
    def config(self) -> FeatureConfig
    @property
    def required_inputs(self) -> Set[str]
    @property
    def output_keys(self) -> Set[str]
    def compute(self, inputs: Dict[str, Any]) -> FeatureVector
```

### FeatureConfig (`src/library/features/base/feature_module.py`)
Per-feature configuration.

```python
class FeatureConfig(BaseModel):
    feature_id: str
    quality_class: str                      # "native_supported" / "proxy_inferred" / "external"
    source: str
    notes: Optional[str] = None             # mandatory for proxy_inferred features
```

## Trade Intent and Execution

### TradeIntent (`src/library/core/domain/trade_intent.py`)
Bot's trade request. Produced by IntentEmitter. Consumed by RiskBridge.

```python
class TradeIntent(BaseModel):
    bot_id: str
    symbol: str
    direction: TradeDirection              # LONG / SHORT
    confidence: int = Field(ge=0, le=100)
    size_requested: float = 0.01
    rationale: str = ""
    entry_price: Optional[float] = None
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    urgency: Literal["IMMEDIATE", "HIGH", "NORMAL", "LOW"] = "NORMAL"
    timestamp_ms: int = 0
```

### TradeIntentBatch (`src/library/core/domain/trade_intent.py`)
Multiple intents from a single bot.

```python
class TradeIntentBatch(BaseModel):
    bot_id: str
    intents: List[TradeIntent]
    timestamp_ms: int
```

### ExecutionDirective (`src/library/core/domain/execution_directive.py`)
Governor's decision on what to actually execute. Produced by RiskBridge.

```python
class ExecutionDirective(BaseModel):
    bot_id: str
    direction: TradeDirection
    symbol: str
    quantity: float = Field(gt=0.0)         # must be > 0 when approved
    risk_mode: RiskMode                    # STANDARD / CLAMPED / HALTED
    max_slippage_ticks: int = Field(ge=0)
    stop_ticks: int = Field(gt=0)
    limit_ticks: Optional[int] = Field(default=None, gt=0)
    timestamp_ms: int
    authorization: str = "RUNTIME_ORCHESTRATOR"
    approved: bool = True                  # False when rejected
    rejection_reason: Optional[str] = None  # human-readable reason when not approved
```

### RiskEnvelope (`src/library/core/domain/risk_envelope.py`)
Risk system authorization output.

```python
class RiskEnvelope(BaseModel):
    bot_id: str
    max_position_size: float = Field(ge=0)
    max_daily_loss: float = Field(ge=0)
    current_drawdown: float = Field(ge=0, le=1)
    risk_mode: RiskMode
    daily_loss_used: float = Field(ge=0)
    max_slippage_ticks: int = Field(ge=0)
    stop_ticks: int = Field(gt=0)
    position_size: float = Field(ge=0)
    open_risk: float = Field(ge=0)
    last_check_ms: int = 0
```

## Evaluation

### EvaluationResult (`src/library/core/domain/evaluation_result.py`)
Standardized evaluation output across all modes.

```python
class EvaluationResult(BaseModel):
    bot_id: str
    mode: EvaluationMode                   # VANILLA / SPICED / VANILLA_FULL /
                                           # SPICED_FULL / MODE_B / MODE_C
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float                         # 0-1
    profit_factor: float
    total_trades: int = Field(ge=0)
    return_pct: float
    kelly_score: float
    passes_gate: bool
    regime_distribution: Optional[Dict[str, int]] = None
    filtered_trades: Optional[int] = None
```

## Registry

### RegistryRecord (`src/library/core/domain/registry_record.py`)
Bot registration record.

```python
class RegistryRecord(BaseModel):
    bot_id: str
    bot_spec_id: str
    status: RegistryStatus                 # ACTIVE / SUSPENDED / ARCHIVED
    tier: BotTier                           # TIER_1 / TIER_2 / TIER_3
    registered_at_ms: int
    last_updated_ms: int
    owner: str = ""
    variant_ids: List[str] = Field(default_factory=list)
    deployed_at: Optional[datetime] = None
```

### BotPerformanceSnapshot (`src/library/core/domain/bot_performance_snapshot.py`)
Periodic performance snapshot for DPR and registry.

```python
class BotPerformanceSnapshot(BaseModel):
    bot_id: str
    snapshot_id: str
    snapshot_at_ms: int
    current_metrics: Dict[str, Any]
    daily_pnl: float
    total_pnl: float
    active_days: int = Field(ge=0)
    current_session: str = ""
    is_in_session: bool = False
```

## Sentinel

### SentinelState (`src/library/core/domain/sentinel_state.py`)
Full sentinel/market intelligence snapshot.

```python
class SentinelState(BaseModel):
    regime_report: RegimeReport
    sensor_states: Dict[str, SensorState]
    hmm_state: Optional[HMMState] = None
    ensemble_vote: Optional[str] = None
```

### SensorState, HMMState (`src/library/core/domain/sentinel_state.py`)
Per-sensor and HMM regime state.

## Order Flow (Placeholder)

### OrderFlowSignal (`src/library/core/domain/order_flow_signal.py`)
Order flow proxy signal.

```python
class OrderFlowSignal(BaseModel):
    bot_id: str
    timestamp: datetime
    symbol: str
    direction: SignalDirection              # BULLISH / BEARISH / NEUTRAL
    strength: float                        # 0.0-1.0
    source: OrderFlowSource                 # CTRADER_NATIVE / PROXY_INFERRED /
                                             # EXTERNAL / APPROXIMATED / DISABLED
    confidence: FeatureConfidence
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
```

### PatternSignal (`src/library/core/domain/pattern_signal.py`)
Chart pattern signal. **V1 placeholder** — detailed schema defined, internal engine deferred.

```python
class PatternSignal(BaseModel):
    bot_id: str
    pattern_type: str                      # e.g. "range_breakout_candidate"
    direction: SignalDirection
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    is_defined: bool = False                # always False in V1
    note: str = "V1 placeholder — internal engine deferred to Phase 2"
```

## Capability System

### CapabilitySpec (`src/library/core/composition/capability_spec.py`)

```python
class CapabilitySpec(BaseModel):
    module_id: str
    provides: List[str]
    requires: List[str]
    optional: List[str]
    outputs: List[str]
    compatibility: List[str] = Field(default_factory=list)
    sync_mode: bool = True
    provides_capability(capability_id) -> bool
    requires_capability(capability_id) -> bool
    is_compatible_with(module_id) -> bool
```

### DependencySpec (`src/library/core/composition/dependency_spec.py`)

```python
class DependencySpec(BaseModel):
    module_id: str
    requires: List[str]
    optional: List[str]
    conflict: List[str] = Field(default_factory=list)
    has_required() -> bool
    has_optional() -> bool
    has_conflict() -> bool
    missing_required(available: Set[str]) -> List[str]
```

### CompatibilityRule (`src/library/core/composition/compatibility_rule.py`)

```python
class CompatibilityRule(BaseModel):
    rule_id: str
    predicate: Literal["COMBINES_WITH", "EXCLUDES", "REQUIRES"]
    subject: str
    target: str
    condition: Optional[str] = None
    matches(other: CompatibilityRule) -> bool
    applies_to(module_id: str) -> bool
```

### OutputSpec (`src/library/core/composition/output_spec.py`)

```python
class OutputSpec(BaseModel):
    output_id: str
    output_type: str
    schema_version: str
    description: str
    fields: List[str] = Field(default_factory=list)
    mime_type: str = "application/json"
    produces_field(field_id) -> bool
    produces_any_field(fields) -> bool
```

## Composition

### ConstraintSpec (`src/library/archetypes/constraints.py`)

```python
class ConstraintSpec(BaseModel):
    field: str
    operator: str                          # < / > / <= / >= / == / !=
    value: Any
    check(value) -> bool
```

### ArchetypeSpec (`src/library/archetypes/archetype_spec.py`)
Frozen spec contract for archetypes.

```python
class ArchetypeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    archetype_id: str
    base_type: str
    required_features: List[str]
    optional_features: List[str]
    sessions: List[str]
    constraints: List[ConstraintSpec] = Field(default_factory=list)
    compatibility: List[str] = Field(default_factory=list)
```

### ValidationResult (`src/library/archetypes/composition/validation.py`)
Result of BotSpec validation.

```python
class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    merge(other: ValidationResult) -> ValidationResult
    @property
    def is_clean(self) -> bool
```

### CompositionResult (`src/library/archetypes/composition/result.py`)
Result of composition attempt.

```python
class CompositionResult(BaseModel):
    succeeded: bool
    bot_spec: Optional[BotSpec] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    composition_notes: List[str] = Field(default_factory=list)
```

### MutationResult (`src/library/archetypes/mutation/engine.py`)

```python
class MutationResult(BaseModel):
    succeeded: bool
    mutated_spec: Optional[BotSpec] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
```

## Bridge-Level Contracts

### ISentinelBridge (`src/library/core/composition/bridge_contracts.py`)

```python
class ISentinelBridge(Protocol):
    def to_sentinel_state(self, market_context: MarketContext) -> SentinelState: ...
    def update_hmm_state(self, hmm_state: HMMState) -> None: ...
    def update_sensor(self, name: str, state: SensorState) -> None: ...
    def is_fresh(self, threshold_ms: int = 5000) -> bool: ...
```

### IRiskBridge (`src/library/core/composition/bridge_contracts.py`)

```python
class IRiskBridge(Protocol):
    def authorize(self, intent: TradeIntent, envelope: RiskEnvelope) -> ExecutionDirective: ...
    def check_circuit_breaker(self, bot_id: str) -> bool: ...
```

### IFeatureBridge (`src/library/core/composition/bridge_contracts.py`)

```python
class IFeatureBridge(Protocol):
    def evaluate(self, bot_id: str, feature_ids: List[str], inputs: Dict) -> FeatureVector: ...
    def can_evaluate(self, feature_ids: List[str], available_inputs: Set[str]) -> Tuple[bool, List[str]]: ...
```

## Enums

| Enum | Location | Values |
|------|----------|--------|
| `RegimeType` | `core/types/enums.py` | TREND_STABLE, RANGE_STABLE, HIGH_CHAOS, BREAKOUT_PRIME, NEWS_EVENT, UNCERTAIN |
| `TradeDirection` | `core/types/enums.py` | LONG, SHORT |
| `RiskMode` | `core/types/enums.py` | STANDARD, CLAMPED, HALTED |
| `NewsState` | `core/types/enums.py` | ACTIVE, KILL_ZONE, CLEAR |
| `SignalDirection` | `core/types/enums.py` | BULLISH, BEARISH, NEUTRAL |
| `OrderFlowSource` | `core/types/enums.py` | CTRADER_NATIVE, PROXY_INFERRED, EXTERNAL, APPROXIMATED, DISABLED |
| `ActivationState` | `core/types/enums.py` | ACTIVE, PAUSED, STOPPED |
| `BotHealth` | `core/types/enums.py` | HEALTHY, DEGRADED, FAILING |
| `EvaluationMode` | `core/types/enums.py` | VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C |
| `RegistryStatus` | `core/types/enums.py` | ACTIVE, SUSPENDED, ARCHIVED |
| `FeatureConfidenceLevel` | `core/types/enums.py` | HIGH, MEDIUM, LOW, DISABLED, INSUFFICIENT_DATA |
| `BotTier` | `core/types/enums.py` | TIER_1, TIER_2, TIER_3, AT_RISK, CIRCUIT_BROKEN |
| `DPRTier` | `core/types/enums.py` | ELITE, PERFORMING, STANDARD, AT_RISK, CIRCUIT_BROKEN |
| `ErrorSeverity` | `core/types/enums.py` | LOW, MEDIUM, HIGH, CRITICAL (optional — not wired into exceptions) |
| `StrategyType` | `core/types/enums.py` | SCALPER, ORB, MEAN_REVERSION, BREAKOUT, GRID_MARTINGALE, NEWS_TRADE, OTHER |
