# Risk Pipeline State Audit - Story 4-0

**Date:** 2026-03-18
**Story:** 4-0-risk-pipeline-state-audit
**Status:** ready-for-dev → in-progress

---

## Executive Summary

This audit documents the current state of the risk pipeline implementation in QUANTMINDX. The risk pipeline consists of multiple components across `src/risk/` and `src/router/` directories, including physics-based sensors (Ising, Lyapunov, HMM), risk governance (Governor, EnhancedGovernor), and execution routing (Sentinel, Commander, BotCircuitBreaker).

**Key Finding:** The majority of risk pipeline components are **production-ready** with well-defined APIs. The primary work for Epic 4 stories (4.1-4.7) will be wiring these existing components to the UI, not rebuilding backend functionality.

---

## Component Classification Matrix

| Component | Location | Status | Classification | Notes |
|-----------|----------|--------|----------------|-------|
| PhysicsAwareKellyEngine | `src/risk/sizing/kelly_engine.py` | ✅ | **production-ready** | Full API with Monte Carlo validation |
| IsingSensor | `src/risk/physics/ising_sensor.py` | ✅ | **production-ready** | Regime detection with caching |
| ChaosSensor (Lyapunov) | `src/risk/physics/chaos_sensor.py` | ✅ | **production-ready** | Phase space reconstruction |
| HMMRegimeSensor | `src/risk/physics/hmm_sensor.py` | ✅ | **production-ready** | Model-based regime detection |
| Governor | `src/router/governor.py` | ✅ | **production-ready** | Tier 2 risk rules with Kelly integration |
| EnhancedGovernor | `src/router/enhanced_governor.py` | ✅ | **production-ready** | Fee-aware Kelly sizing with House Money Effect |
| Sentinel | `src/router/sentinel.py` | ✅ | **production-ready** | Unified regime reporting |
| Commander | `src/router/commander.py` | ✅ | **production-ready** | Strategy auction with BotManifest |
| BotCircuitBreaker | `src/router/bot_circuit_breaker.py` | ✅ | **production-ready** | Per-bot quarantine on consecutive losses |
| PropFirmRiskOverlay | `src/risk/prop_firm_overlay.py` | ✅ | **production-ready** | Prop firm-specific risk limits |
| BrokerRegistry | `src/router/broker_registry.py` | ✅ | **production-ready** | Dynamic pip values and fee structures |

---

## Detailed Component Audits

### 1. PhysicsAwareKellyEngine

**File:** `src/risk/sizing/kelly_engine.py`

**API:**
```python
class PhysicsAwareKellyEngine:
    def __init__(self, kelly_fraction: float = 0.5, enable_monte_carlo: bool = True, monte_carlo_threshold: float = 0.005)
    def calculate_position_size(self, perf: StrategyPerformance, physics: MarketPhysics, validator: Optional[MonteCarloValidator] = None) -> SizingRecommendation
    def calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float
    def calculate_lyapunov_multiplier(self, lyapunov_exponent: float) -> float
    def calculate_ising_multiplier(self, ising_susceptibility: float) -> float
    def calculate_eigen_multiplier(self, max_eigenvalue: Optional[float]) -> float
```

**Inputs:**
- `StrategyPerformance`: win_rate, avg_win, avg_loss, total_trades, payoff_ratio
- `MarketPhysics`: lyapunov_exponent, ising_susceptibility, rmt_max_eigenvalue

**Outputs:**
- `SizingRecommendation`: raw_kelly, physics_multiplier, final_risk_pct, position_size_lots, constraint_source, validation_passed, adjustments_applied

**Classification:** ✅ **production-ready** - Well-documented with comprehensive formulas and Monte Carlo validation support.

---

### 2. IsingSensor (Physics Regime Detection)

**File:** `src/risk/physics/ising_sensor.py`

**API:**
```python
class IsingRegimeSensor:
    def __init__(self, config: Optional[IsingSensorConfig] = None)
    def detect_regime(self, market_volatility: Optional[float] = None) -> Dict
    def get_regime_confidence(self, magnetization: float) -> float
    def clear_cache(self) -> None
    def is_cache_valid(self, max_age_seconds: float = 300.0) -> bool
    def get_reading(self) -> float  # Returns 0.0-1.0 normalized value
```

**Outputs:**
- Dictionary with: temperature, magnetization, susceptibility, current_regime, volatility_context
- Regimes: "CHAOTIC", "TRANSITIONAL", "ORDERED"

**Classification:** ✅ **production-ready** - Metropolis-Hastings algorithm with LRU caching. 12x12x12 lattice.

---

### 3. ChaosSensor (Lyapunov Exponent)

**File:** `src/risk/physics/chaos_sensor.py`

**API:**
```python
class ChaosSensor:
    def __init__(self, embedding_dimension: int = 3, time_delay: int = 12, lookback_points: int = 300, k_steps: int = 10)
    def analyze_chaos(self, returns: np.ndarray) -> ChaosAnalysisResult
    def get_reading(self) -> float  # Returns normalized 0.0-1.0
```

**Outputs:**
- `ChaosAnalysisResult`: lyapunov_exponent, match_distance, match_index, chaos_level, trajectory_length
- Chaos levels: "STABLE", "MODERATE", "CHAOTIC"

**Classification:** ✅ **production-ready** - Method of analogues with phase space reconstruction. Requires 300+ data points.

---

### 4. HMMRegimeSensor

**File:** `src/risk/physics/hmm_sensor.py`

**API:**
```python
class HMMRegimeSensor(_HMMRegimeSensor):
    def __init__(self, config: Optional[HMMSensorConfig] = None, config_path: str = ...)
    def predict_regime(self, features: np.ndarray, cache_key: Optional[str] = None) -> HMMRegimeReading
    def predict_from_ohlcv(self, ohlcv_data: Dict, volatility: Optional[float] = None) -> HMMRegimeReading
    def get_model_info(self) -> Dict
    def reload_model(self) -> bool
    def is_model_loaded(self) -> bool
    def clear_cache(self) -> None
    def get_reading(self) -> float
```

**Outputs:**
- `HMMRegimeReading`: state, regime, confidence, state_probabilities, next_state_probabilities, timestamp, model_version, features_used

**Classification:** ✅ **production-ready** - Requires trained HMM model (hmmlearn). Falls back to default readings if no model loaded.

---

### 5. Governor (Base Risk Layer)

**File:** `src/router/governor.py`

**API:**
```python
class Governor:
    def __init__(self, settings: Optional[object] = None)
    def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict, account_balance: Optional[float] = None, broker_id: Optional[str] = None, account_id: Optional[str] = None, mode: str = "live", **kwargs) -> RiskMandate
    def check_swarm_cohesion(self, correlation_matrix: Dict) -> float
```

**Outputs:**
- `RiskMandate`: allocation_scalar, risk_mode (STANDARD/CLAMPED/HALTED), position_size, kelly_fraction, risk_amount, kelly_adjustments, mode, notes

**Classification:** ✅ **production-ready** - Integrates with EnhancedKellyCalculator for fee-aware position sizing. Supports demo/live mode distinction.

---

### 6. EnhancedGovernor

**File:** `src/router/enhanced_governor.py`

**API:**
```python
class EnhancedGovernor(Governor):
    def __init__(self, account_id: Optional[str] = None, config: Optional[EnhancedKellyConfig] = None, settings: Optional['RiskSettings'] = None)
    def calculate_risk(self, regime_report: RegimeReport, trade_proposal: Dict, ...) -> RiskMandate
    def reset_daily_state(self, start_balance: float) -> None
```

**Key Features:**
- Kelly Criterion with 3-layer protection
- House Money Effect (1.5x up, 0.5x down multipliers)
- Dynamic pip value calculation
- Fee-aware position sizing
- Prop firm rule enforcement

**Classification:** ✅ **production-ready** - Full implementation with database integration for state persistence.

---

### 7. Sentinel (Intelligence Layer)

**File:** `src/router/sentinel.py`

**API:**
```python
class Sentinel:
    def __init__(self, shadow_mode: bool = False, hmm_weight: float = 0.0)
    def on_tick(self, symbol: str, price: float, timeframe: str = "H1", high: Optional[float] = None, low: Optional[float] = None) -> RegimeReport
    def set_mode(self, shadow_mode: bool = False, hmm_weight: float = 0.0) -> None
```

**Outputs:**
- `RegimeReport`: regime, chaos_score, regime_quality, susceptibility, is_systemic_risk, news_state, timestamp, hmm_regime, hmm_confidence, hmm_agreement, decision_source

**Classification:** ✅ **production-ready** - Aggregates Chaos, Regime, Correlation, and News sensors. Supports HMM dual-model (shadow/hybrid/production modes).

---

### 8. Commander (Execution Layer)

**File:** `src/router/commander.py`

**API:**
```python
class Commander:
    def __init__(self, bot_registry: Optional["BotRegistry"] = None, governor: Optional["Governor"] = None)
    def select_bots_for_auction(self, bots: List["BotManifest"]) -> List["BotManifest"]
    def set_router_mode(self, mode: str) -> None  # 'auction', 'priority', 'round_robin'
    def check_drawdown_limits(self, account_book: str, prop_firm_name: Optional[str], current_drawdown: float) -> tuple[bool, str]
```

**Classification:** ✅ **production-ready** - Strategy auction with BotManifest integration. Only @primal tagged bots can participate.

---

### 9. BotCircuitBreaker

**File:** `src/router/bot_circuit_breaker.py`

**API:**
```python
class BotCircuitBreakerManager:
    def __init__(self, db_manager: Optional[DBManager] = None, account_id: Optional[str] = None, account_balance: Optional[float] = None, account_book: Optional[AccountBook] = None, account_book_str: Optional[str] = None)
    def get_or_create_state(self, bot_id: str) -> BotCircuitBreaker
    def check_allowed(self, bot_id: str) -> Tuple[bool, Optional[str]]
    def record_trade(self, bot_id: str, is_loss: bool, fee: float = 0.0, trade_date: Optional[date] = None) -> BotCircuitBreaker
```

**Thresholds:**
- Personal Book: 5 consecutive losses
- Prop Firm Book: 3 consecutive losses (tighter)
- Daily trade limit: 20 trades

**Classification:** ✅ **production-ready** - Database-backed state tracking with automatic quarantine.

---

### 10. PropFirmRiskOverlay

**File:** `src/risk/prop_firm_overlay.py`

**API:**
```python
class PropFirmRiskOverlay:
    def __init__(self, firm_name: Optional[str] = None, initial_balance: float = 100000.0)
    def should_block_trade(self, account_book: str, prop_firm_name: Optional[str], current_drawdown: float) -> tuple[bool, str]
    def apply_risk_limits(self, account_book: str, max_drawdown: float = 0.05, daily_loss: float = 0.03, profit_target: float = 0.15) -> Dict[str, Any]
    def calculate_p_pass(self, win_rate: float, avg_win: float, avg_loss: float, account_book: str) -> Dict[str, Any]
```

**Supported Firms:** FTMO, Topstep, FundedNext, FundingPips

**Classification:** ✅ **production-ready** - Drift-Diffusion model for P_pass calculation. Recovery mode after drawdown breach.

---

### 11. BrokerRegistry

**File:** `src/router/broker_registry.py`

**API:**
```python
class BrokerRegistryManager:
    def __init__(self, db_manager: Optional[DatabaseManager] = None)
    def get_broker(self, broker_id: str) -> Optional[BrokerRegistry]
    def create_broker(self, broker_id: str, broker_name: str, spread_avg: float = 0.0, commission_per_lot: float = 0.0, ...) -> BrokerRegistry
    def update_broker(self, broker_id: str, **kwargs) -> Optional[BrokerRegistry]
    def get_pip_value(self, symbol: str, broker_id: str) -> float
    def get_commission(self, broker_id: str) -> float
```

**Classification:** ✅ **production-ready** - Database-backed broker profiles with dynamic pip values.

---

## Risk API Endpoints

**Location:** `src/api/settings_endpoints.py`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings/risk` | Get risk management settings |
| POST | `/api/settings/risk` | Update risk management settings |

**Note:** Dedicated risk API endpoints are minimal. Stories 4.1-4.7 should expand API exposure for UI wiring.

---

## Dependency Graph

```
Sensors Layer (src/risk/physics/)
├── IsingSensor → Market Physics Output
├── ChaosSensor → Lyapunov Exponent
└── HMMRegimeSensor → Regime States

Intelligence Layer (src/router/)
└── Sentinel
    ├── Aggregates: Chaos, Regime, Correlation, News Sensors
    └── Outputs: RegimeReport

Governance Layer (src/router/)
├── Governor (base)
├── EnhancedGovernor
│   ├── EnhancedKellyCalculator
│   ├── House Money Effect
│   └── PropFirmRiskOverlay
└── BotCircuitBreakerManager

Execution Layer (src/router/)
├── Commander
│   ├── BotManifest System
│   ├── RoutingMatrix
│   └── DynamicBotLimiter
└── BrokerRegistryManager
```

---

## Recommendations for Epic 4 Stories

### Story 4.1: CalendarGovernor / News Blackout
- **Wiring Required:** Integrate existing `NewsSensor` (from Sentinel) with calendar-aware trading rules
- **Existing:** `src/router/sensors/news.py` - NewsSensor with kill zone detection

### Story 4.2: Risk Parameters / Prop Firm Registry APIs
- **Already Available:** `BrokerRegistry`, `PropFirmRiskOverlay`, `BotCircuitBreaker`
- **Wiring Needed:** Expose via REST API for UI

### Story 4.3: Strategy Router / Regime State APIs
- **Already Available:** `Sentinel`, `Commander`, `RoutingMatrix`
- **Wiring Needed:** Real-time regime state WebSocket

### Story 4.4: Backtest Results API
- **Integration Point:** Connect to existing backtest infrastructure

### Story 4.5-4.7: Risk Canvas / Physics Sensors / Compliance
- **Already Available:** All physics sensors, Governor, EnhancedGovernor
- **Wiring Needed:** UI components for dashboard display

---

## Files Audited (READ-ONLY)

| Category | Files |
|----------|-------|
| **Sizing** | `src/risk/sizing/kelly_engine.py` |
| **Physics Sensors** | `src/risk/physics/ising_sensor.py`, `src/risk/physics/chaos_sensor.py`, `src/risk/physics/hmm_sensor.py` |
| **Governance** | `src/router/governor.py`, `src/router/enhanced_governor.py` |
| **Intelligence** | `src/router/sentinel.py` |
| **Execution** | `src/router/commander.py`, `src/router/bot_circuit_breaker.py` |
| **Risk Overlays** | `src/risk/prop_firm_overlay.py`, `src/router/broker_registry.py` |
| **API** | `src/api/settings_endpoints.py` (risk endpoints) |

---

## Conclusion

The risk pipeline backend is **comprehensive and production-ready**. Epic 4 stories should focus on:

1. **API Exposure** - Expand REST/WebSocket endpoints for UI consumption
2. **UI Wiring** - Connect frontend components to existing backend services
3. **Calendar Integration** - Wire NewsSensor to trading rules
4. **Dashboard Display** - Visualize regime state, risk metrics, sensor readings

**No rebuild required** - all core components are functional and ready for integration.
