# QuantMindLib V1 — Runtime and Decision Flow

## Current Runtime Components

### RuntimeOrchestrator (`src/library/runtime/orchestrator.py`)
The central wiring point. Wires FeatureEvaluator, IntentEmitter, SafetyHooks together.

```python
class RuntimeOrchestrator:
    _feature_evaluator: FeatureEvaluator  # registry=get_default_registry()
    _state_manager: BotStateManager
    _safety_hooks: SafetyHooks
    _intent_emitter: IntentEmitter

    def tick(self, bot_id: str, market_data: Dict[str, Any]) -> Optional[ExecutionDirective]
    def evaluate_features(self, bot_id: str, feature_ids: List[str], inputs: Dict) -> FeatureVector
    def emit_intent(self, bot_id: str, feature_vector: FeatureVector, market_context: MarketContext) -> TradeIntent
    def authorize_trade(self, intent: TradeIntent, envelope: RiskEnvelope) -> ExecutionDirective
```

The orchestrator wires `FeatureEvaluator` via `get_default_registry()` at line 58:
```python
self._feature_evaluator = FeatureEvaluator(registry=get_default_registry())
```

### FeatureEvaluator (`src/library/runtime/feature_evaluator.py`)

```python
class FeatureEvaluator:
    _registry: FeatureRegistry  # via PrivateAttr

    def evaluate(
        self,
        bot_id: str,
        feature_ids: List[str],
        inputs: Dict[str, Any],
    ) -> FeatureVector:
        """
        Iteratively evaluates features in dependency order.
        - Skips features whose required_inputs are not met
        - Uses max_iterations cap to prevent infinite loops
        - Aggregates FeatureVector from all computable modules
        """
```

Key behaviors:
- Iterative evaluation (not recursive)
- Max iterations cap prevents circular dependency infinite loops
- Partial evaluation: computes available features, skips unavailable ones
- Supports chained dependencies: output of one feature becomes input to next

### IntentEmitter (`src/library/runtime/intent_emitter.py`)

```python
class IntentEmitter:
    def emit(
        self,
        bot_id: str,
        bot_spec: BotSpec,
        feature_vector: FeatureVector,
        market_context: MarketContext,
    ) -> Optional[TradeIntent]:
        """
        Emits TradeIntent from BotSpec + FeatureVector + MarketContext.
        Returns None if no trade signal qualifies.
        """
```

### BotStateManager (`src/library/runtime/state_manager.py`)

```python
class BotStateManager:
    """
    Thread-safe cached FeatureVector + MarketContext per bot.
    Uses RLock for concurrent access.
    """

    def update_market_context(self, bot_id: str, context: MarketContext) -> None
    def update_feature_vector(self, bot_id: str, fv: FeatureVector) -> None
    def get_market_context(self, bot_id: str) -> Optional[MarketContext]
    def get_feature_vector(self, bot_id: str) -> Optional[FeatureVector]
    def clear(self, bot_id: str) -> None
    def clear_all(self) -> None
```

### SafetyHooks (`src/library/runtime/safety_hooks.py`)

```python
class SafetyHooks:
    def check_kill_switch(self, bot_id: str) -> bool
    def check_circuit_breaker(self, bot_id: str) -> bool
    def check_dpr_circuit(self, bot_id: str) -> bool
    def check_ssl_dpr_combined(self, bot_id: str) -> bool
    def is_trading_allowed(self, bot_id: str) -> bool
    # Integrates with:
    # - DPRCircuitBreakerMonitor
    # - SSLCircuitBreakerDPRMonitor
    # - KillSwitch events
```

## Decision Flow (As Implemented)

```
[1] Market data arrives (tick, depth, quotes)
         │
         ▼
[2] BotStateManager.update_market_context() — caches MarketContext
         │
         ▼
[3] RuntimeOrchestrator.evaluate_features() → FeatureEvaluator.evaluate()
         │
         ├─► Iterates through feature_ids
         ├─► Checks required_inputs against available inputs
         ├─► Evaluates computable features (chained if needed)
         └─► Aggregates FeatureVector
         │
         ▼
[4] BotStateManager.update_feature_vector() — caches FeatureVector
         │
         ▼
[5] RuntimeOrchestrator.emit_intent() → IntentEmitter.emit()
         │
         ├─► Reads cached FeatureVector + MarketContext
         ├─► Evaluates BotSpec confirmations against features
         ├─► Emits TradeIntent if signal qualifies
         └─► Returns None if no qualifying signal
         │
         ▼
[6] RuntimeOrchestrator.authorize_trade() → RiskBridge.authorize()
         │
         ├─► TradeIntent + RiskEnvelope → ExecutionDirective
         ├─► approved=True when position_size > 0
         └─► approved=False with rejection_reason when rejected
         │
         ▼
[7] ExecutionBridge.send_directive()
         │
         └─► cTrader adapter (NOT YET IMPLEMENTED)
```

## Where Rejection/Blocking Occurs

Rejection is a **controlled outcome**, not an engineering failure. It happens at two points:

### Point 1: IntentEmitter — No Signal

```python
# IntentEmitter.emit() returns None if:
# - No feature crosses the confirmation threshold
# - Market context is unfavorable (kill zone, stale regime)
# - Bot is not eligible for current session
return None  # No TradeIntent emitted
```

### Point 2: RiskBridge — Authorization Denied

```python
# RiskBridge.authorize() returns ExecutionDirective with:
# approved = False
# rejection_reason = "position_size <= 0, trade not authorized"
# quantity = 0.0
# risk_mode = HALTED or CLAMPED
```

### Point 3: SafetyHooks — Kill Switch Active

```python
# SafetyHooks.is_trading_allowed() returns False when:
# - Kill switch event is active
# - Circuit breaker is tripped
# - DPR tier is CIRCUIT_BROKEN
# - SSL + DPR combined check fails
```

## What Is Complete

| Component | Status | Notes |
|-----------|--------|-------|
| RuntimeOrchestrator | ✓ Complete | Central wiring point |
| FeatureEvaluator | ✓ Complete | Dependency resolution, chaining, max_iterations cap |
| IntentEmitter | ✓ Complete | BotSpec + FeatureVector + MarketContext → TradeIntent |
| BotStateManager | ✓ Complete | Thread-safe, per-bot cache |
| SafetyHooks | ✓ Complete | DPR + SSL circuit breaking, kill switch checks |
| FeatureEvaluator wiring | ✓ Complete | `get_default_registry()` in orchestrator |
| DPRRedisPublisher | ✓ Complete | Redis writes for DPR scores |
| DPRConcernEmitter | ✓ Complete | Tier change → concern tag flow |
| DPRDualEngineRouter | ✓ Complete | Handles both DPR engines |
| RegistryBridge | ✓ Complete | Bot registration, status mapping |
| JournalBridge | ✓ Complete | Append-only logging |
| LifecycleBridge | ✓ Complete | BACKTEST→PAPER→LIVE transitions |
| EvaluationBridge | ✓ Complete | 6-mode results → EvaluationResult |
| WorkflowBridge | ✓ Complete | WF1→WF2 readiness tracking |

## What Is NOT Wired (Integration Needed)

| Component | Status | Notes |
|-----------|--------|-------|
| cTrader Market Adapter | ✗ Not implemented | `src/library/ctrader/` is empty |
| cTrader tick stream → BotStateManager | ✗ Not wired | Depends on adapter |
| SentinelBridge → BotStateManager | ✗ Not wired | Regime events → cache |
| Kill switch event bus → SafetyHooks | ✗ Not wired | `src/events/ssl.py`, `src/events/dpr.py` |
| DPR Redis → DPRBridge read path | ✗ Not wired | Publisher exists, read path needs verification |
| Prefect flows | ✗ Stubs only | `stub_flows.py` — explicit stubs |

## ExecutionDirective Schema

The `ExecutionDirective` (`src/library/core/domain/execution_directive.py`) is the canonical execution authorization:

```python
class ExecutionDirective(BaseModel):
    bot_id: str
    direction: TradeDirection              # LONG / SHORT
    symbol: str
    quantity: float = Field(gt=0.0)        # must be > 0 when approved
    risk_mode: RiskMode                    # STANDARD / CLAMPED / HALTED
    max_slippage_ticks: int = Field(ge=0)
    stop_ticks: int = Field(gt=0)
    limit_ticks: Optional[int] = Field(default=None, gt=0)
    timestamp_ms: int
    authorization: str = "RUNTIME_ORCHESTRATOR"
    approved: bool = True                   # False when rejected
    rejection_reason: Optional[str] = None  # human-readable when not approved
```

The `approved=False, rejection_reason=...` fields were added in Phase 11K (ERR-004). This cleanly separates controlled rejection from engineering failure.
