# QuantMindLib V1 — Bridge and Integration Boundaries

## What Each Bridge Does

### DPR Bridge Stack (3 components)

**1. DPRRedisPublisher** (`src/library/core/bridges/dpr_redis_bridge.py`)
- Publishes DPR composite scores to Redis
- Key: `dpr:score:{bot_id}`, TTL: 24 hours
- JSON payload: `{composite_score, components, tier, timestamp}`
- Uses Redis pipeline for efficiency
- Fixes the DPR Redis write gap

**2. DPRConcernEmitter** (`src/library/core/bridges/dpr_concern_bridge.py`)
- Emits DPR concern tags on tier changes
- `emit_tier_change(bot_id, old_tier, new_tier, score)`
- `emit_concern_event(bot_id, concern_type, description)`
- DPR concern → `@session_concern` tag → RegistryBridge

**3. DPRDualEngineRouter** (`src/library/core/bridges/dpr_dual_engine.py`)
- Handles both DPR engines (router + risk layer)
- Router engine is canonical for DPR composite scores
- Risk engine session concern counters consumed as-is
- `compute_combined_score(bot_id)` → DPRScore with tier

### SentinelBridge (`src/library/core/bridges/sentinel_dpr_bridges.py`)

```python
class SentinelBridge:
    def to_sentinel_state(self, market_context: MarketContext) -> SentinelState
    def update_hmm_state(self, hmm_state: HMMState) -> None
    def update_sensor(self, name: str, state: SensorState) -> None
    def is_fresh(self, threshold_ms: int = 5000) -> bool
    # MarketContext → RegimeReport conversion
    # regime, chaos_score, regime_quality, susceptibility, is_systemic_risk, news_state
```

### DPRBridge (`src/library/core/bridges/sentinel_dpr_bridges.py`)

```python
class DPRBridge:
    def compute_scores(self, bot_id: str, metrics: BotPerformanceSnapshot) -> DPRScore
    def get_top_bots(self, n: int = 10) -> List[DPRScore]
    # DPRScore: composite (0-100), 4 components (win_rate/pnl/consistency/ev),
    # tier: ELITE/PERFORMING/STANDARD/AT_RISK/CIRCUIT_BROKEN
```

### RegistryBridge (`src/library/core/bridges/registry_journal_bridges.py`)

```python
class RegistryBridge:
    def register(self, bot_id: str, bot_spec: BotSpec) -> RegistryRecord
    def deactivate(self, bot_id: str) -> None
    def to_runtime_profile(self, record: RegistryRecord) -> BotRuntimeProfile
    # RegistryStatus → ActivationState, BotTier mapping
```

### JournalBridge (`src/library/core/bridges/registry_journal_bridges.py`)

```python
class JournalBridge:
    def log_fill(self, bot_id: str, fill: Dict[str, Any]) -> None
    def log_pnl(self, bot_id: str, pnl: Dict[str, Any]) -> None
    def log_session(self, bot_id: str, session: Dict[str, Any]) -> None
    # Append-only with _by_bot index
```

### LifecycleBridge + EvaluationBridge + WorkflowBridge (`src/library/core/bridges/lifecycle_eval_workflow_bridges.py`)

```python
class LifecycleBridge:
    def request_transition(self, bot_id: str, target: str) -> bool
    # BACKTEST → PAPER → LIVE state machine
    # min_paper_days guard

class EvaluationBridge:
    def from_backtest_result(self, result: Any) -> EvaluationResult
    def to_evaluation_profile(self, results: Dict[EvaluationMode, EvaluationResult]) -> BotEvaluationProfile
    # PBO: max(0.0, 1.0 - walk_forward_stability)
    # robustness: (wf_stability * 0.5) + (mc_quality * 0.3) + (min(1.0, sharpe/2.0) * 0.2)

class WorkflowBridge:
    def is_wf1_to_wf2_ready(self, bot_id: str) -> bool
    # WF1 → WF2 readiness check
```

### RiskBridge + ExecutionBridge (`src/library/core/bridges/risk_execution_bridges.py`)

```python
class RiskBridge:
    def authorize(self, intent: TradeIntent, envelope: RiskEnvelope) -> ExecutionDirective
    # Maps TradeIntent + RiskEnvelope → ExecutionDirective
    # approved=True when position_size > 0
    # approved=False with rejection_reason when rejected

class ExecutionBridge:
    def send_directive(self, directive: ExecutionDirective) -> bool
    def close_all(self, bot_id: str) -> None
```

### SSLCircuitBreakerDPRMonitor (`src/library/core/bridges/ssl_dpr_integration.py`)

```python
class SSLCircuitBreakerDPRMonitor:
    def set_ssl_halted(self, bot_id: str) -> None
    def set_ssl_active(self, bot_id: str) -> None
    def reset_ssl_state(self, bot_id: str) -> None
    def check_ssl_dpr_combined(self, bot_id: str) -> bool
    def get_ssl_dpr_summary(self) -> Dict[str, Any]
    # Tracks SSL HALTED/ACTIVE state per bot
    # Combined SSL + DPR kill-switch decisions
```

### DPRCircuitBreakerMonitor (`src/library/core/bridges/safety_integration.py`)

```python
class DPRCircuitBreakerMonitor:
    def check_concern_events(self, bot_id: str) -> bool
    def check_ssl_dpr_combined(self, bot_id: str) -> bool
    # DPR circuit breaking conditions:
    # - tier is CIRCUIT_BROKEN
    # - score < 0.3
    # - AT_RISK consecutive
    # - rank > 10 with AT_RISK
```

## What Stays Outside the Library

| System | Location | Ownership |
|--------|----------|-----------|
| DPR scoring internals | `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py` | Router + Risk layers |
| Governor risk computation | `src/router/governor.py`, `src/risk/governor.py` | Router + Risk layers |
| Kill switch logic | `src/router/kill_switch.py`, `src/router/progressive_kill_switch.py` | Router layer |
| Circuit breaker internals | `src/risk/ssl/circuit_breaker.py` | Risk layer |
| Sentinel regime classification | `src/router/sentinel.py`, `src/risk/physics/` | Router + Risk layers |
| HMM, BOCPD, MS-GARCH, Ising | `src/risk/physics/` | Risk layer |
| EnsembleVoter | `src/risk/physics/ensemble/voter.py` | Exists but NOT wired to live tick |
| Backtest engine internals | `src/backtesting/full_backtest_pipeline.py`, `src/backtesting/mt5_engine.py` | Backtesting layer |
| Prefect flow execution | `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py` | Stub only |
| cTrader Open API client | Not implemented | Library does not own |

## Ownership Boundaries

```
Library owns:                          Existing systems own:
- FeatureModule ABC + 16 modules       - DPR scoring internals
- FeatureRegistry singleton            - Governor risk computation
- BotSpec, FeatureVector schemas       - Kill switch logic
- TradeIntent, ExecutionDirective      - Circuit breaker internals
- RuntimeOrchestrator wiring           - Sentinel regime classification
- Bridge translation logic              - Backtest engine internals
- DPRRedisPublisher (writes to Redis)  - Prefect flow execution
- DPRConcernEmitter                    - cTrader Open API client
- DPRDualEngineRouter (coordination)
- RegistryBridge (translation)
- JournalBridge (append-only)
- LifecycleBridge (translation)
- EvaluationBridge (translation)
- WorkflowBridge (translation)
```

## How Future Integration Should Proceed

Integration means **wiring library components into existing system components**, not replacing existing logic. The integration points are the bridges.

### Integration Priority Order

1. **cTrader Market Adapter** — Required before live trading. Until implemented, library runs with mock/stub data.
2. **DPR Redis verification** — Verify `DPRRedisPublisher` actually writes to Redis. DPR dual engine router should be confirmed.
3. **SafetyHooks wiring** — Wire `SafetyHooks` into the kill switch event bus (`src/events/ssl.py`, `src/events/dpr.py`).
4. **IntentEmitter wiring** — Connect `IntentEmitter` to the tick processing path.
5. **BotStateManager wiring** — Connect to real market data feed once cTrader adapter exists.
6. **Registry bidirectional sync** — Verify `RegistryBridge` syncs correctly with `BotRegistry`.
7. **JournalBridge** — Wire append-only journal entries to real trade log.
8. **Prefect flow wiring** — Replace stubs in `stub_flows.py` with real Prefect flow calls.

### What NOT to Do

- **Do not wrap existing ML systems** (HMM, BOCPD, MS-GARCH, Ising) into library features
- **Do not replace DPR scoring internals** — bridge to them
- **Do not replace governor risk computation** — bridge to it
- **Do not claim proxy features as true order flow**
- **Do not implement cTrader adapter inside the library** — it belongs in `src/library/ctrader/` but should follow the adapter contracts
- **Do not auto-discover features** — the explicit bootstrap is correct

## DPR Dual Engine Handling

Two DPR engines exist. The library handles them via `DPRDualEngineRouter`:

```
Router DPR engine (src/router/dpr_scoring_engine.py)
    │ canonical for DPR composite scores
    │ writes to Redis: dpr:score:{bot_id}
    ▼
DPRDualEngineRouter
    │ coordinates both engines
    ▼
Risk DPR engine (src/risk/dpr/scoring_engine.py)
    │ session concern counters only
    ▼
DPRRedisPublisher → Redis
```

Router engine is the canonical source. Risk engine counters are consumed but do NOT override the router score.

## Two Governor Classes

`src/router/governor.py` (authorization) and `src/risk/governor.py` (Tier 2 rules) are both referenced by module path in bridges:

```python
from src.router.governor import Governor  # risk authorization
from src.risk.governor import Governor    # Tier 2 rules
```

No rename was made in V1. Library bridges use full import paths.
