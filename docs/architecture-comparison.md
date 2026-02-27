# Architecture Comparison: QuantMindX vs Hummingbot

**Document Type:** Architecture Decision Record (ADR)
**Date:** 2024-02-23
**Status:** Analysis Complete
**Decision Required:** Choose platform direction

---

## Executive Summary

| Aspect | QuantMindX | Hummingbot | Recommendation |
|--------|------------|------------|----------------|
| **Primary Focus** | AI-powered strategy development | Crypto market making & execution | Hybrid |
| **Position Sizing** | Advanced (Physics-aware Kelly) | Basic (Fixed % or manual) | QuantMindX |
| **Risk Management** | Multi-layer, prop-firm focused | Simple SL/TP per executor | QuantMindX |
| **Exchange Support** | MT5 + Binance (extensible) | 50+ crypto exchanges | Hummingbot |
| **Strategy Architecture** | Agent-based with LLM routing | Controller/Executor pattern | Hummingbot |
| **Data Flow** | Real-time + Sentinel sensors | Market data provider | Both strong |
| **Backtesting** | Limited | Full backtesting engine | Hummingbot |
| **Production Ready** | 70% (missing execution layer) | 100% | Hummingbot |

**Final Recommendation: Hybrid Approach** - Use Hummingbot for execution/connectors, migrate QuantMindX's unique IP (Kelly engine, circuit breaker, regime sensors) as Hummingbot extensions.

---

## 1. Strategy Router vs Controller/Executor Pattern

### QuantMindX: Strategy Router Architecture

```
+------------------+
|  Router Agent    |  <- LLM-based task classification
+--------+---------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
+------+ +------+ +------+
|Analyst| |Quant | |Copilot|
+------+ +------+ +------+
    |         |        |
    v         v        v
+---------------------------------+
|      Strategy Router Engine      |
|  - BotManifest (passport)       |
|  - BrokerRegistry (MT5/Crypto)  |
|  - CircuitBreakerManager        |
+---------------------------------+
                |
                v
+---------------------------------+
|      Native Bridge (ZMQ)        |
|  REQ/REP: Commands              |
|  PUB/SUB: Tick Ingestion        |
+---------------------------------+
```

**Key Characteristics:**
- **LLM-First:** Uses Claude/LangGraph for task routing
- **Passport System:** BotManifest declares bot requirements
- **Multi-Broker:** MT5 (forex) + Binance (crypto) support
- **Agent Coordination:** Analyst, QuantCode, Copilot agents

**Code Example (Router Classification):**
```python
# src/agents/router.py
def classify_task_node(state: RouterState) -> Dict[str, Any]:
    messages = state.get('messages', [])
    last_message = messages[-1].get("content", "") if messages else ""

    # Keyword-based classification
    if any(keyword in last_message.lower() for keyword in ['research', 'analysis', 'market']):
        task_type = "research"
        target_agent = "analyst"
    elif any(keyword in last_message.lower() for keyword in ['strategy', 'backtest', 'code']):
        task_type = "strategy_development"
        target_agent = "quantcode"
    # ...
```

### Hummingbot: Controller/Executor Pattern

```
+------------------------+
|  StrategyV2Base        |
|  - MarketDataProvider  |
|  - ConnectorManager    |
+----------+-------------+
           |
    +------+------+------+
    |             |      |
    v             v      v
+--------+  +--------+  +--------+
|Controller| |Controller| |Controller|
| (Logic)  | | (Logic)  | | (Logic) |
+----+-----+ +----+-----+ +----+----+
     |            |            |
     v            v            v
+----------------------------------------+
|       ExecutorOrchestrator             |
+----------------------------------------+
     |            |            |
     v            v            v
+--------+  +--------+  +--------+
|Position|  | Grid   |  | TWAP   |
|Executor|  |Executor|  |Executor|
+--------+  +--------+  +--------+
     |
     v
+------------------------+
|  ConnectorBase         |
|  (Exchange API)        |
+------------------------+
```

**Key Characteristics:**
- **Clear Separation:** Controllers decide WHAT, Executors decide HOW
- **Executor Types:** Position, Grid, DCA, TWAP, Arbitrage, XEMM, Order
- **Action Queue:** Controllers send ExecutorActions to orchestrator
- **Event-Driven:** Executors subscribe to market events

**Code Example (Controller):**
```python
# controllers/directional_trading/bollinger_v1.py
class BollingerV1Controller(DirectionalTradingControllerBase):
    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(...)
        df.ta.bbands(length=self.config.bb_length, ...)

        # Generate signal
        df["signal"] = 0
        df.loc[bbp < self.config.bb_long_threshold, "signal"] = 1
        df.loc[bbp > self.config.bb_short_threshold, "signal"] = -1

    def determine_executor_actions(self) -> List[ExecutorAction]:
        # Returns list of CreateExecutorAction, StopExecutorAction, etc.
        ...
```

### Comparison Matrix

| Aspect | QuantMindX Router | Hummingbot Controller |
|--------|-------------------|----------------------|
| **Pattern** | Agent-based (LLM routing) | Traditional (inheritance) |
| **Flexibility** | High (dynamic) | Medium (compile-time) |
| **Overhead** | Higher (LLM calls) | Lower (pure Python) |
| **Testability** | Harder | Easier |
| **Execution** | Via Native Bridge | Direct to Exchange |
| **State Mgmt** | RouterState (memory) | ExecutorOrchestrator |

---

## 2. Risk Management Comparison

### QuantMindX: Multi-Layer Risk System

```
+------------------------------------------+
|          Risk Management Stack           |
+------------------------------------------+
| Layer 0: Account Size Throttling        |
|   - Micro (<$100): 0.5x Kelly           |
|   - Small ($100-500): 0.75x             |
|   - Standard ($1000+): 1.0x             |
+------------------------------------------+
| Layer 1: Kelly Fraction (50% of full)   |
|   f_base = f* * 0.5                     |
+------------------------------------------+
| Layer 2: Hard Risk Cap (2% max)         |
|   f_final = min(f_base, 0.02)           |
+------------------------------------------+
| Layer 3: Physics-Aware Adjustment       |
|   - Lyapunov multiplier                 |
|   - Ising susceptibility               |
|   - Eigenvalue multiplier               |
+------------------------------------------+
| Layer 4: Fee Kill Switch                |
|   - Block if fees >= avg_win            |
+------------------------------------------+
| Circuit Breaker:                         |
|   - 3 consecutive losses = quarantine   |
|   - Daily trade limit (20 default)      |
|   - Fee-based halt (5% of balance)      |
+------------------------------------------+
```

**Physics-Aware Kelly Formula:**
```
f* = (p*b - q) / b           # Base Kelly
f_base = f* * 0.5            # Half-Kelly
P_ = max(0, 1 - 2*)          # Lyapunov
P_X = 0.5 if X > 0.8 else 1.0  # Ising
P_E = min(1, 1.5/_max)       # Eigenvalue
M_physics = min(P_, P_X, P_E)  # Weakest link
final_risk = f_base * M_physics
```

**Code Example:**
```python
# src/position_sizing/enhanced_kelly.py
class EnhancedKellyCalculator:
    def calculate(self, account_balance, win_rate, avg_win, avg_loss, ...):
        # Fee-adjusted returns
        avg_win_net = avg_win - fee_per_lot
        avg_loss_net = avg_loss + fee_per_lot

        # Fee kill switch
        if fee_per_lot >= avg_win:
            return KellyResult(position_size=0.0, status='fee_blocked')

        # Layer 0: Account tier
        account_multiplier = self.calculate_account_size_multiplier(account_balance)

        # Layer 1: Half-Kelly
        kelly_f = base_kelly * self.kelly_fraction

        # Layer 2: Hard cap
        kelly_f = min(kelly_f, self.max_risk_pct)  # 2%

        # Layer 3: Physics scalars
        kelly_f = kelly_f * physics_scalar * vol_scalar
```

### Hummingbot: Triple Barrier Risk

```
+------------------------------------------+
|       Triple Barrier Config              |
+------------------------------------------+
| stop_loss: Decimal (e.g., 0.02 = 2%)     |
| take_profit: Decimal (e.g., 0.04 = 4%)   |
| time_limit: int (seconds)                |
| stop_loss_order_type: MARKET/LIMIT       |
| take_profit_order_type: MARKET/LIMIT     |
+------------------------------------------+
```

**Code Example:**
```python
# hummingbot/strategy_v2/executors/position_executor/data_types.py
@dataclass
class TrailingStop:
    activation_price: Decimal
    trailing_delta: Decimal

@dataclass
class TripleBarrierConfig:
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_limit: Optional[int] = None
    trailing_stop: Optional[TrailingStop] = None
    take_profit_order_type: OrderType = OrderType.MARKET
    stop_loss_order_type: OrderType = OrderType.MARKET
```

### Risk Management Comparison

| Feature | QuantMindX | Hummingbot |
|---------|------------|------------|
| **Kelly Criterion** | Full implementation | None |
| **Physics Indicators** | Lyapunov, Ising, RMT | None |
| **Position Sizing** | Dynamic (Kelly) | Manual/Fixed |
| **Stop Loss** | Via broker | Built-in (triple barrier) |
| **Take Profit** | Via broker | Built-in |
| **Circuit Breaker** | 3-loss quarantine | None |
| **Fee Awareness** | Kill switch | Fee estimation only |
| **Account Tiers** | Yes | No |
| **Prop Firm Safe** | Yes (configurable) | N/A |

---

## 3. Position Sizing Comparison

### QuantMindX Position Sizing

```python
# Example calculation
account_balance = 10000.0
win_rate = 0.55
avg_win = 400.0
avg_loss = 200.0
current_atr = 0.0012
average_atr = 0.0010
stop_loss_pips = 20.0

result = calculator.calculate(
    account_balance=account_balance,
    win_rate=win_rate,
    avg_win=avg_win,
    avg_loss=avg_loss,
    current_atr=current_atr,
    average_atr=average_atr,
    stop_loss_pips=stop_loss_pips,
    regime_quality=0.8  # From Sentinel
)

# Result:
# position_size: 0.85 lots
# kelly_f: 0.0167 (1.67% risk)
# adjustments: [
#   "R:R ratio = 2.00",
#   "Base Kelly = 0.3250 (32.50%)",
#   "Layer 1 (x0.5): 0.1625",
#   "Layer 2 (capped at 2.0%): 0.0200",
#   "Regime Quality = 0.80",
#   "ATR ratio = 1.20, Vol Scalar = 0.83",
#   "Final risk: 0.0167"
# ]
```

### Hummingbot Position Sizing

```python
# Configuration-based (no Kelly)
class DirectionalTradingControllerConfigBase(ControllerConfigBase):
    total_amount_quote: Decimal = Decimal("100")  # Fixed amount

    # Or per-position
    @property
    def position_size(self):
        return self.total_amount_quote / current_price
```

### Position Sizing Comparison

| Feature | QuantMindX | Hummingbot |
|---------|------------|------------|
| **Kelly Criterion** | Physics-aware | Not implemented |
| **Dynamic Sizing** | Yes (ATR, regime) | No |
| **Volatility Adjustment** | Yes (ATR ratio) | No |
| **Account Protection** | Tiered multipliers | None |
| **House of Money** | Yes (1.0-1.5x on profits) | None |
| **Monte Carlo Validation** | Yes (optional) | None |

---

## 4. Data Flow Differences

### QuantMindX Data Flow

```
+-----------------+      +------------------+
| MT5/Binance     |      | Contabo Server   |
| (Broker API)    |      | (Regime Data)    |
+--------+--------+      +--------+---------+
         |                        |
         v                        v
+------------------------------------------+
|       Native Bridge (ZMQ)                |
|  - REQ/REP: Commands (port 5555)         |
|  - PUB/SUB: Ticks (port 5556)            |
+------------------------------------------+
         |
         v
+------------------------------------------+
|       Multi-Timeframe Sentinel           |
|  - RegimeSensor (Ising model)            |
|  - ChaosSensor (Lyapunov)                |
|  - CorrelationSensor                     |
+------------------------------------------+
         |
         v
+------------------------------------------+
|       Strategy Router Engine             |
|  - Uses regime quality for sizing        |
|  - Feeds physics to Kelly engine         |
+------------------------------------------+
```

### Hummingbot Data Flow

```
+-----------------+
| Exchange API    |
| (50+ connectors)|
+--------+--------+
         |
         v
+------------------------------------------+
|       ConnectorBase                      |
|  - Order management                      |
|  - Balance tracking                      |
|  - Order book subscription               |
+------------------------------------------+
         |
         v
+------------------------------------------+
|       MarketDataProvider                 |
|  - CandlesFeed                           |
|  - OrderBookTracker                      |
|  - Real-time prices                      |
+------------------------------------------+
         |
         v
+------------------------------------------+
|       Controller                         |
|  - Processes candles with pandas_ta      |
|  - Generates signals                     |
+------------------------------------------+
         |
         v
+------------------------------------------+
|       ExecutorOrchestrator               |
|  - Creates/manages executors             |
|  - Tracks positions                      |
+------------------------------------------+
```

### Data Flow Comparison

| Aspect | QuantMindX | Hummingbot |
|--------|------------|------------|
| **Primary Transport** | ZMQ (low latency) | WebSocket/REST |
| **Data Provider** | Custom Sentinel | MarketDataProvider |
| **Indicators** | Custom (physics) | pandas_ta |
| **Timeframes** | Multi-timeframe sentinel | Single or multi-candle config |
| **Regime Detection** | Ising/Lyapunov | None built-in |

---

## 5. What QuantMindX Has That Hummingbot Lacks

### 5.1 Physics-Aware Kelly Engine
```python
# Unique to QuantMindX
class PhysicsAwareKellyEngine:
    def calculate_lyapunov_multiplier(self, lyapunov_exponent: float) -> float:
        """P_ = max(0, 1.0 - (2.0 x ))"""
        return max(0.0, 1.0 - (2.0 * lyapunov_exponent))

    def calculate_ising_multiplier(self, ising_susceptibility: float) -> float:
        """P_X = 0.5 if X > 0.8 else 1.0"""
        return 0.5 if ising_susceptibility > 0.8 else 1.0
```

### 5.2 Regime Sensors (Econophysics)
```python
# src/router/sensors/regime.py
class RegimeSensor:
    """Ising model for phase transitions"""
    def _calculate_ising(self) -> RegimeReport:
        # Magnetization: Trend direction
        M = np.mean(self.spins)
        # Susceptibility: Criticality
        chi = np.var(self.spins)
        # Energy: Frustration
        E = energy / len(arr)

        # States: ORDERED, CRITICAL, DISORDERED
        if abs(M) > 0.7: state = "ORDERED"
        if chi > 0.24: state = "CRITICAL"
```

### 5.3 Bot Circuit Breaker
```python
# src/router/bot_circuit_breaker.py
class BotCircuitBreakerManager:
    MAX_CONSECUTIVE_LOSSES = 3
    DEFAULT_DAILY_TRADE_LIMIT = 20

    def record_trade(self, bot_id, is_loss, fee):
        if is_loss:
            state.consecutive_losses += 1

        # Auto-quarantine
        if state.consecutive_losses >= 3:
            self.quarantine_bot(bot_id, reason="3 consecutive losses")

        # Fee kill switch
        if self.fee_monitor.should_halt_trading():
            self.quarantine_bot(bot_id, reason="FEE_KILL_SWITCH")
```

### 5.4 BotManifest (Passport System)
```python
# src/router/bot_manifest.py
@dataclass
class BotManifest:
    """Bot passport for automatic routing"""
    bot_id: str
    strategy_type: StrategyType  # SCALPER, STRUCTURAL, SWING, HFT
    frequency: TradeFrequency    # HFT, HIGH, MEDIUM, LOW
    min_capital_req: float       # Margin buffer
    preferred_broker_type: BrokerType  # RAW_ECN, STANDARD
    prop_firm_safe: bool         # Violates 1-min rule?
    trading_mode: TradingMode    # PAPER, LIVE
    preferred_conditions: PreferredConditions  # ICT-style filtering
```

### 5.5 Paper->Live Promotion Workflow
```python
def check_promotion_eligibility(self) -> Dict[str, Any]:
    """Human-in-the-loop promotion tracking"""
    criteria = {
        "min_trading_days": 30,
        "min_sharpe_ratio": 1.5,
        "min_win_rate": 0.55,
        "max_drawdown": 10.0,
    }
    # Returns eligibility + requires human approval
```

### 5.6 Fee Kill Switch
```python
# Blocks trades when fees exceed expected profit
if fee_per_lot >= avg_win:
    return KellyResult(
        position_size=0.0,
        status='fee_blocked',
        adjustments=["FEE KILL SWITCH - Fees exceed expected profit"]
    )
```

---

## 6. What Hummingbot Has That QuantMindX Lacks

### 6.1 50+ Exchange Connectors
```
Exchanges: Binance, Bybit, OKX, KuCoin, Gate.io, Bitget, MEXC,
           Kraken, Coinbase, Hyperliquid, dYdX, Injective, etc.

Derivatives: Binance Perpetual, Bybit Perpetual, OKX Perpetual,
             Hyperliquid Perpetual, dYdX v4, etc.

AMM/Dex: Uniswap, PancakeSwap, SushiSwap, etc. (via Gateway)
```

### 6.2 Executor Types
```python
_executor_mapping = {
    "position_executor": PositionExecutor,  # Single position with SL/TP
    "grid_executor": GridExecutor,          # Grid trading
    "dca_executor": DCAExecutor,            # Dollar-cost averaging
    "arbitrage_executor": ArbitrageExecutor, # Cross-exchange arb
    "twap_executor": TWAPExecutor,          # Time-weighted avg price
    "xemm_executor": XEMMExecutor,          # Cross-exchange market making
    "order_executor": OrderExecutor,        # Simple order
}
```

### 6.3 Backtesting Engine
```python
# hummingbot/strategy_v2/backtesting/
class ExecutorSimulatorBase:
    """Simulates executor behavior on historical data"""

# Simulators for each executor type:
- DCAExecutorSimulator
- PositionExecutorSimulator
```

### 6.4 Triple Barrier Execution
```python
@dataclass
class TripleBarrierConfig:
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    time_limit: Optional[int]
    trailing_stop: Optional[TrailingStop]
    stop_loss_order_type: OrderType
    take_profit_order_type: OrderType
```

### 6.5 Performance Tracking
```python
class PerformanceReport:
    realized_pnl_quote: Decimal
    unrealized_pnl_quote: Decimal
    volume_traded: Decimal
    global_pnl_quote: Decimal
    global_pnl_pct: Decimal
    close_type_counts: Dict[CloseType, int]
    positions_summary: List[PositionSummary]
```

### 6.6 Position Management
```python
class PositionHold:
    """Tracks position across multiple orders"""
    def get_position_summary(self, mid_price: Decimal):
        # Calculates:
        # - Breakeven price
        # - Realized PnL (matched volume)
        # - Unrealized PnL (remaining)
        # - Cumulative fees
```

### 6.7 Database Persistence
```python
# SQLite persistence for:
- Executors (hummingbot.model.executors)
- Positions (hummingbot.model.position)
- Controllers (hummingbot.model.controllers)
```

### 6.8 MQTT Integration
```python
# Built-in MQTT for:
- Performance reports
- Custom controller info
- Remote monitoring
```

---

## 7. Migration Complexity Estimate

### Effort Matrix

| Component | Migration Direction | Effort | Risk |
|-----------|---------------------|--------|------|
| **Kelly Engine** | QMX -> HB | 2 days | Low |
| **Regime Sensors** | QMX -> HB | 3 days | Low |
| **Circuit Breaker** | QMX -> HB | 2 days | Low |
| **BotManifest** | QMX -> HB | 3 days | Medium |
| **Exchange Connectors** | HB -> QMX | 30+ days | High |
| **Executor System** | HB -> QMX | 14 days | High |
| **Backtesting** | HB -> QMX | 7 days | Medium |

### Migration Paths

#### Path A: Switch to Hummingbot (Est. 10 days)
1. Port Kelly engine as Hummingbot controller
2. Port regime sensors as data feed
3. Implement circuit breaker as strategy wrapper
4. Rewrite strategies using HB patterns
5. **Risk:** Lose MT5 support, prop firm features

#### Path B: Keep QuantMindX (Est. 60+ days)
1. Implement 50+ exchange connectors
2. Build executor system from scratch
3. Add backtesting engine
4. **Risk:** Huge effort, reinventing the wheel

#### Path C: Hybrid Approach (Est. 14 days) **RECOMMENDED**
1. Use Hummingbot for execution layer (connectors, executors)
2. Port QuantMindX IP as Hummingbot extensions:
   - `hummingbot/strategy_v2/executors/kelly_executor/`
   - `hummingbot/data_feed/regime_feed/`
   - `hummingbot/strategy_v2/risk/circuit_breaker.py`
3. Create custom controller base that uses physics-aware sizing
4. Keep BotManifest for routing decisions

---

## 8. Recommendation: Hybrid Approach

### Architecture Proposal

```
+--------------------------------------------------+
|                 QuantMindX Layer                 |
|  - BotManifest routing                          |
|  - PhysicsAwareKellyEngine                      |
|  - CircuitBreakerManager                        |
|  - RegimeSensor (Ising, Lyapunov)               |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|            Hummingbot Integration                |
|  - Custom ControllerBase with Kelly sizing      |
|  - RegimeFeed data provider                     |
|  - CircuitBreakerExecutor wrapper               |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|           Hummingbot Core (Unchanged)            |
|  - ExecutorOrchestrator                         |
|  - MarketDataProvider                           |
|  - 50+ Exchange Connectors                      |
|  - Backtesting Engine                           |
+--------------------------------------------------+
```

### Implementation Plan

#### Phase 1: Core Extensions (5 days)
1. Create `KellyExecutorConfig` extending `ExecutorConfigBase`
2. Implement `PhysicsAwareKellySizer` as strategy utility
3. Add `RegimeSensor` as data feed

#### Phase 2: Risk Management (3 days)
1. Port `CircuitBreakerManager` as executor wrapper
2. Add fee kill switch to position executor
3. Implement account tier throttling

#### Phase 3: Strategy Layer (3 days)
1. Create `QuantMindControllerBase` extending `DirectionalTradingControllerBase`
2. Integrate Kelly sizing into controller
3. Add BotManifest for bot configuration

#### Phase 4: MT5 Bridge (3 days)
1. Create MT5 connector for Hummingbot (optional)
2. Or use existing forex brokers via FIX API

### Benefits of Hybrid

1. **Best of Both Worlds:**
   - Hummingbot's production-ready execution
   - QuantMindX's scientific risk management

2. **Immediate Exchange Support:**
   - Access to 50+ crypto exchanges
   - Battle-tested order management

3. **Preserved IP:**
   - Kelly engine remains core differentiator
   - Physics sensors unique to platform

4. **Reduced Risk:**
   - No need to build execution from scratch
   - Community support for Hummingbot

5. **Future-Proof:**
   - Hummingbot actively maintained
   - Easy to add new exchanges

---

## 9. Decision Record

**Decision:** Implement hybrid approach

**Rationale:**
1. Hummingbot provides production-ready execution (saves 60+ days)
2. QuantMindX IP (Kelly, sensors, circuit breaker) is unique value
3. Hybrid preserves both investments
4. Lower risk than full migration either direction

**Next Steps:**
1. Create POC with KellyExecutorConfig
2. Port regime sensors as data feed
3. Validate with backtesting
4. Deploy to paper trading

**Stakeholders:** Architecture team, trading team, risk management

---

## Appendix A: File References

### QuantMindX Key Files
```
/home/mubarkahimself/Desktop/QUANTMINDX/src/
  agents/router.py                 - LLM-based routing
  position_sizing/enhanced_kelly.py - Physics-aware Kelly
  risk/sizing/kelly_engine.py      - Core Kelly math
  router/bot_circuit_breaker.py    - Risk limits
  router/bot_manifest.py           - Bot passport system
  router/sensors/regime.py         - Ising model sensor
  router/sensors/chaos.py          - Lyapunov sensor
  router/interface.py              - Native bridge
  data/brokers/binance_adapter.py  - Crypto integration
```

### Hummingbot Key Files
```
/home/mubarkahimself/Desktop/hummingbot-research/
  hummingbot/strategy_v2/
    controllers/controller_base.py          - Base controller
    executors/executor_base.py              - Base executor
    executors/executor_orchestrator.py      - Executor management
    executors/position_executor/            - Position handling
    models/executor_actions.py              - Action types
  hummingbot/connector/
    connector_base.pyx                      - Exchange interface
  controllers/directional_trading/          - Example controllers
```
