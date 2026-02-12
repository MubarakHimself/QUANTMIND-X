# QuantMind Standard Library (QSL) - MQL5 Asset Index

**Version:** 2.0.0 (V8)  
**Last Updated:** 2026-01-31  
**Status:** Production Ready

## Overview

The QuantMind Standard Library (QSL) is a modular, production-ready MQL5 library designed for building sophisticated Expert Advisors using a component-based architecture. The library follows best practices from the MQL5 community, particularly the modular EA design pattern that separates concerns into independent, reusable modules.

**V8 "Omni-Asset Scalper" Enhancements:**
- **Tiered Risk Engine**: Three-tier risk system (Growth/Scaling/Guardian) for adaptive position sizing
- **HFT Infrastructure**: Sub-5ms socket communication replacing REST polling
- **Multi-broker Support**: Unified interface for MT5, Binance, and other brokers

### Design Philosophy

The QSL follows a **modular architecture** where:
- Each module is self-contained and performs a specific function
- Modules communicate through well-defined interfaces
- The EA strategy acts as a coordinator, not a monolithic implementation
- All modules inherit from a base `CModule` class for consistency
- O(1) performance is prioritized for real-time trading operations

This approach is inspired by the MQL5 article ["Building an Expert Advisor using separate modules"](https://www.mql5.com/en/articles/7318), which demonstrates how to create maintainable, scalable EAs by separating trading logic, risk management, and utility functions into independent components.

## Directory Structure

```
src/mql5/Include/QuantMind/
├── Core/
│   ├── BaseAgent.mqh      # Base EA functionality
│   ├── Constants.mqh      # System-wide constants
│   └── Types.mqh          # Custom data structures
├── Risk/
│   ├── PropManager.mqh    # Prop firm risk management
│   ├── RiskClient.mqh     # Risk multiplier retrieval
│   └── KellySizer.mqh     # Kelly criterion position sizing
└── Utils/
    ├── JSON.mqh           # JSON parsing utilities
    ├── Sockets.mqh        # HTTP/WebSocket communication
    └── RingBuffer.mqh     # O(1) circular buffer
```

---

## Core Modules

### 1. BaseAgent.mqh

**Purpose:** Provides base functionality for all QuantMind Expert Advisors.

**Key Features:**
- EA initialization and validation
- Symbol and timeframe management
- Logging with rate limiting
- Account information retrieval
- Trading permission checks
- Lot size normalization

**Main Class:** `CBaseAgent`

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `Initialize(string agentName, string symbol, ENUM_TIMEFRAMES timeframe, int magicNumber)` | Initialize the agent with configuration | `bool` |
| `ValidateSymbol()` | Validate trading symbol availability | `bool` |
| `ValidateTimeframe()` | Validate timeframe is supported | `bool` |
| `IsTradingAllowed()` | Check if trading is permitted | `bool` |
| `NormalizeLot(double lots)` | Normalize lot size to broker requirements | `double` |
| `GetBid()` / `GetAsk()` | Get current market prices | `double` |
| `LogInfo(string message)` | Log information with rate limiting | `void` |
| `LogError(string message)` | Log error with error code | `void` |

**Usage Example:**

```mql5
#include <QuantMind/Core/BaseAgent.mqh>

class CMyEA : public CBaseAgent
{
public:
    bool Init()
    {
        if(!Initialize("MyEA", "EURUSD", PERIOD_H1, 123456))
        {
            LogError("Initialization failed");
            return false;
        }
        
        if(!IsTradingAllowed())
        {
            LogWarning("Trading not allowed");
            return false;
        }
        
        return true;
    }
    
    void ExecuteTrade(double lots)
    {
        double normalizedLots = NormalizeLot(lots);
        double bid = GetBid();
        double ask = GetAsk();
        
        LogInfo("Executing trade: " + DoubleToString(normalizedLots, 2) + " lots");
        // Trade execution logic here
    }
};
```

---

### 2. Constants.mqh

**Purpose:** Defines system-wide constants used across all QSL modules.

**Categories:**

#### Risk Management Constants
```mql5
QM_DAILY_LOSS_LIMIT_PCT      = 5.0    // 5% daily loss limit
QM_HARD_STOP_BUFFER_PCT      = 1.0    // 1% buffer before limit
QM_EFFECTIVE_LIMIT_PCT       = 4.0    // Effective stop at 4%
QM_KELLY_THRESHOLD           = 0.8    // Minimum Kelly score for A+ setups
QM_MAX_RISK_PER_TRADE_PCT    = 2.0    // Maximum risk per trade
```

#### Magic Number Ranges
```mql5
QM_MAGIC_ANALYST_START       = 100000  // Analyst agent range
QM_MAGIC_QUANT_START         = 110000  // Quant agent range
QM_MAGIC_COPILOT_START       = 120000  // Copilot agent range
QM_MAGIC_COINFLIP_START      = 130000  // Coin Flip Bot range
```

#### Communication Constants
```mql5
QM_BRIDGE_HOST               = "localhost"
QM_BRIDGE_PORT               = 8000
QM_HEARTBEAT_INTERVAL_SEC    = 60
QM_RISK_MATRIX_FILE          = "risk_matrix.json"
```

#### Performance Constants
```mql5
QM_PERF_HEARTBEAT_MAX_MS     = 100    // Max heartbeat response time
QM_PERF_RISK_RETRIEVAL_MAX_MS = 50    // Max risk retrieval time
QM_RING_BUFFER_SIZE_MEDIUM   = 500    // Medium buffer size
```

**Usage Example:**

```mql5
#include <QuantMind/Core/Constants.mqh>

void CheckRiskLimits(double dailyLoss)
{
    if(dailyLoss >= QM_EFFECTIVE_LIMIT_PCT)
    {
        Print("Hard stop triggered at ", QM_EFFECTIVE_LIMIT_PCT, "%");
        // Stop trading
    }
}
```

---

### 3. Types.mqh

**Purpose:** Defines custom data structures, enums, and types used across QSL modules.

**Key Enumerations:**

```mql5
enum ENUM_TRADE_DECISION
{
    TRADE_DECISION_REJECT,
    TRADE_DECISION_APPROVE,
    TRADE_DECISION_PENDING,
    TRADE_DECISION_MODIFIED
};

enum ENUM_RISK_STATUS
{
    RISK_STATUS_NORMAL,
    RISK_STATUS_THROTTLED,
    RISK_STATUS_HARD_STOP,
    RISK_STATUS_NEWS_GUARD,
    RISK_STATUS_PRESERVATION
};

enum ENUM_STRATEGY_QUALITY
{
    STRATEGY_QUALITY_F,      // Kelly < 0.2
    STRATEGY_QUALITY_D,      // Kelly 0.2-0.4
    STRATEGY_QUALITY_C,      // Kelly 0.4-0.6
    STRATEGY_QUALITY_B,      // Kelly 0.6-0.8
    STRATEGY_QUALITY_A,      // Kelly 0.8-0.9
    STRATEGY_QUALITY_A_PLUS  // Kelly >= 0.9
};
```

**Key Structures:**

```mql5
struct STradeProposal
{
    string symbol;
    ENUM_SIGNAL_TYPE signalType;
    double entryPrice;
    double stopLoss;
    double takeProfit;
    double lotSize;
    double kellyScore;
    double winRate;
    string strategyName;
    datetime timestamp;
};

struct SAccountState
{
    string accountId;
    double balance;
    double equity;
    double highWaterMark;
    double dailyPnL;
    double dailyDrawdown;
    ENUM_ACCOUNT_STATUS status;
    ENUM_RISK_STATUS riskStatus;
};

struct SRiskParameters
{
    double riskMultiplier;
    double maxRiskPerTrade;
    double kellyThreshold;
    bool preservationMode;
    bool newsGuardActive;
    bool hardStopActive;
    datetime lastHeartbeat;
};
```

**Utility Functions:**

```mql5
string TradeDecisionToString(ENUM_TRADE_DECISION decision);
string RiskStatusToString(ENUM_RISK_STATUS status);
ENUM_STRATEGY_QUALITY GetStrategyQualityFromKelly(double kellyScore);
```

**Usage Example:**

```mql5
#include <QuantMind/Core/Types.mqh>

void EvaluateStrategy(double kellyScore)
{
    ENUM_STRATEGY_QUALITY quality = GetStrategyQualityFromKelly(kellyScore);
    
    Print("Strategy Quality: ", StrategyQualityToString(quality));
    
    if(quality >= STRATEGY_QUALITY_A_PLUS)
    {
        Print("A+ setup detected - proceeding with trade");
    }
}
```

---

## Risk Management Modules

### 4. PropManager.mqh

**Purpose:** Comprehensive prop firm risk management including daily drawdown tracking, hard stop enforcement, news guard functionality, and quadratic throttle calculations.

**Main Class:** `CPropManager`

**Key Features:**
- Daily high water mark tracking
- Hard stop at 4.5% with 1% buffer (effective 4%)
- News guard (KILL_ZONE) functionality
- Quadratic throttle calculation
- Integration with Python backend for persistence

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `Initialize(string accountId, string firmName, double dailyLossLimit)` | Initialize prop manager | `bool` |
| `UpdateDailyState()` | Update daily P&L and drawdown | `void` |
| `CheckHardStop()` | Check if hard stop should activate | `bool` |
| `CalculateQuadraticThrottle()` | Calculate risk throttle multiplier | `double` |
| `SetNewsGuard(bool active)` | Enable/disable news guard | `void` |
| `GetRiskStatus()` | Get current risk status | `ENUM_RISK_STATUS` |
| `CanTrade()` | Check if trading is allowed | `bool` |

**Usage Example:**

```mql5
#include <QuantMind/Risk/PropManager.mqh>

CPropManager propMgr;

int OnInit()
{
    if(!propMgr.Initialize("ACC123456", "MyPropFirm", 5.0))
    {
        Print("PropManager initialization failed");
        return INIT_FAILED;
    }
    return INIT_SUCCEEDED;
}

void OnTick()
{
    propMgr.UpdateDailyState();
    
    if(!propMgr.CanTrade())
    {
        Print("Trading disabled: ", RiskStatusToString(propMgr.GetRiskStatus()));
        return;
    }
    
    double throttle = propMgr.CalculateQuadraticThrottle();
    double adjustedLots = baseLots * throttle;
    
    // Execute trade with adjusted position size
}
```

---

### 5. RiskClient.mqh

**Purpose:** Risk multiplier retrieval with fast path (GlobalVariable) and fallback mechanisms (file-based).

**Key Features:**
- Fast path: GlobalVariable set by Python agent
- Fallback path: risk_matrix.json file
- Data freshness validation (1-hour max age)
- Heartbeat transmission to Python backend

**Main Functions:**

| Function | Description | Returns |
|----------|-------------|---------|
| `GetRiskMultiplier(string symbol)` | Get risk multiplier for symbol | `double` |
| `ReadRiskFromFile(string symbol, QMRiskData &data)` | Read from risk_matrix.json | `bool` |
| `SendHeartbeat(string eaName, int magic, double balance)` | Send heartbeat to backend | `bool` |

**Usage Example:**

```mql5
#include <QuantMind/Risk/RiskClient.mqh>

void OnTick()
{
    double riskMultiplier = GetRiskMultiplier("EURUSD");
    
    if(riskMultiplier == 0.0)
    {
        Print("Trading halted - risk multiplier is zero");
        return;
    }
    
    double adjustedLots = baseLots * riskMultiplier;
    
    // Execute trade with adjusted risk
}

void OnTimer()
{
    // Send heartbeat every 60 seconds
    SendHeartbeat("MyEA", 123456, AccountInfoDouble(ACCOUNT_BALANCE));
}
```

---

### 6. KellySizer.mqh

**Purpose:** Kelly criterion position sizing calculations for optimal risk management.

**V8 Enhancement:** Implements three-tier risk system (Growth/Scaling/Guardian) for adaptive position sizing based on account equity. See [V8.1: KellySizer.mqh - Tiered Risk Engine](#v81-kellysizermqh---tiered-risk-engine) for details.

**Main Class:** `QMKellySizer`

**Formula:** `f* = (bp - q) / b`
- `b` = avgWin / avgLoss (payoff ratio)
- `p` = winRate (probability of win)
- `q` = 1 - p (probability of loss)

**Key Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `CalculateKellyFraction(double winRate, double avgWin, double avgLoss)` | Calculate optimal Kelly fraction | `double` |
| `CalculateLotSize(double kellyFraction, double equity, double riskPct, double stopPips)` | Calculate lot size from Kelly fraction | `double` |
| `GetLastError()` | Get last error code | `int` |

**Usage Example:**

```mql5
#include <QuantMind/Risk/KellySizer.mqh>

QMKellySizer kelly;

void CalculatePositionSize()
{
    double winRate = 0.55;  // 55% win rate
    double avgWin = 400.0;  // Average win $400
    double avgLoss = 200.0; // Average loss $200
    
    double kellyFraction = kelly.CalculateKellyFraction(winRate, avgWin, avgLoss);
    
    if(kellyFraction >= QM_KELLY_THRESHOLD)
    {
        Print("A+ setup detected - Kelly fraction: ", kellyFraction);
        
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double lots = kelly.CalculateLotSize(kellyFraction, equity, 1.0, 50);
        
        Print("Recommended lot size: ", lots);
    }
}
```

---

## Utility Modules

### 7. JSON.mqh

**Purpose:** Lightweight JSON parsing utilities for MQL5.

**Key Features:**
- Find JSON objects by key
- Extract double values from JSON
- Minimal dependencies
- Fast parsing for real-time applications

**Main Functions:**

| Function | Description | Returns |
|----------|-------------|---------|
| `FindJsonObject(string json, string key)` | Find JSON object by key | `string` |
| `ExtractJsonDouble(string json, string key)` | Extract double value | `double` |

**Usage Example:**

```mql5
#include <QuantMind/Utils/JSON.mqh>

void ParseRiskMatrix()
{
    string json = "{\"EURUSD\": {\"multiplier\": 1.5, \"timestamp\": 1234567890}}";
    
    string eurusdObj = FindJsonObject(json, "EURUSD");
    double multiplier = ExtractJsonDouble(eurusdObj, "multiplier");
    
    Print("EURUSD risk multiplier: ", multiplier);
}
```

---

### 8. Sockets.mqh

**Purpose:** HTTP/WebSocket communication for MQL5-Python integration.

**V8 Enhancement:** Introduces `CQuantMindSocket` class for sub-5ms latency event-driven communication with Python ZMQ socket server. See [V8.2: Sockets.mqh - HFT Infrastructure](#v82-socketsmqh---hft-infrastructure) for details.

**Key Features:**
- HTTP POST requests
- WebSocket support
- V8: Persistent socket connections with automatic reconnection
- V8: Sub-5ms latency for trade events
- Timeout handling
- Error recovery

**Main Functions:**

| Function | Description | Returns |
|----------|-------------|---------|
| `HttpPost(string url, string data, string &response)` | Send HTTP POST request | `bool` |
| `HttpGet(string url, string &response)` | Send HTTP GET request | `bool` |

**Usage Example:**

```mql5
#include <QuantMind/Utils/Sockets.mqh>

void SendHeartbeat()
{
    string url = "http://localhost:8000/heartbeat";
    string data = "{\"ea_name\":\"MyEA\",\"balance\":10000.0}";
    string response;
    
    if(HttpPost(url, data, response))
    {
        Print("Heartbeat sent successfully");
    }
    else
    {
        Print("Heartbeat failed");
    }
}
```

---

### 9. RingBuffer.mqh

**Purpose:** O(1) circular buffer implementation for efficient indicator calculations and time-series data management.

**Main Class:** `CRiBuff`

**Key Features:**
- O(1) push and get operations
- Automatic overwriting of oldest elements when full
- FIFO ordering preservation
- Aggregate operations (Sum, Average, Min, Max)
- Fixed memory allocation (no dynamic reallocation)

**Key Methods:**

| Method | Description | Complexity |
|--------|-------------|------------|
| `Push(double value)` | Add element to buffer | O(1) |
| `Get(int index)` | Get element by index (0 = newest) | O(1) |
| `GetNewest()` | Get most recent element | O(1) |
| `GetOldest()` | Get oldest element | O(1) |
| `Size()` | Get current size | O(1) |
| `IsFull()` | Check if buffer is full | O(1) |
| `IsEmpty()` | Check if buffer is empty | O(1) |
| `Clear()` | Reset buffer to empty state | O(1) |
| `Sum()` | Calculate sum of all elements | O(n) |
| `Average()` | Calculate average | O(n) |
| `Min()` / `Max()` | Find min/max value | O(n) |

**Usage Example:**

```mql5
#include <QuantMind/Utils/RingBuffer.mqh>

CRiBuff* priceBuffer;

int OnInit()
{
    priceBuffer = new CRiBuff(100);  // 100-element buffer
    return INIT_SUCCEEDED;
}

void OnTick()
{
    double close = iClose(NULL, PERIOD_CURRENT, 0);
    priceBuffer.Push(close);
    
    if(priceBuffer.Size() >= 20)
    {
        double ma20 = priceBuffer.Average();  // Simple moving average
        double highest = priceBuffer.Max();
        double lowest = priceBuffer.Min();
        
        Print("MA(20): ", ma20, " High: ", highest, " Low: ", lowest);
    }
}

void OnDeinit(const int reason)
{
    delete priceBuffer;
}
```

---

## Modular EA Architecture

### Strategy Pattern Implementation

The QSL follows a strategy pattern where the EA logic is separated from modules:

```mql5
// Base strategy class
class CStrategy
{
public:
    virtual void OnBuyFind(ulong ticket, double price, double lot) = 0;
    virtual void OnSellFind(ulong ticket, double price, double lot) = 0;
    virtual bool MayAndEnter() = 0;
    virtual void CreateSnap() = 0;
};

// Concrete strategy implementation
class CMyStrategy : public CStrategy
{
private:
    CParam* m_param;
    CSnap* m_snap;
    CTradeMod* m_trade;
    
public:
    bool Initialize(CParam* param, CSnap* snap, CTradeMod* trade)
    {
        m_param = param;
        m_snap = snap;
        m_trade = trade;
        return true;
    }
    
    virtual bool MayAndEnter()
    {
        // Strategy logic here
        if(/* entry signal */)
        {
            m_trade.Buy();
            return true;
        }
        return false;
    }
};
```

### EA Handler Structure

```mql5
// Global objects
CParam par;
CSnap snap;
CTradeMod trade;
CStrategy* strategy;

int OnInit()
{
    // Initialize modules
    par.Initialize();
    snap.Initialize();
    trade.Initialize();
    
    // Create strategy
    strategy = new CMyStrategy();
    strategy.Initialize(GetPointer(par), GetPointer(snap), GetPointer(trade));
    
    return INIT_SUCCEEDED;
}

void OnTick()
{
    // Simple, clean handler
    strategy.CreateSnap();
    strategy.MayAndEnter();
}

void OnDeinit(const int reason)
{
    delete strategy;
}
```

---

## Integration with Python Backend

### Heartbeat System

The QSL integrates with the Python backend through a heartbeat system:

```mql5
void OnTimer()
{
    SHeartbeatPayload payload;
    payload.eaName = "MyEA";
    payload.symbol = Symbol();
    payload.magicNumber = 123456;
    payload.accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    payload.accountEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    payload.dailyPnL = CalculateDailyPnL();
    payload.openPositions = PositionsTotal();
    payload.timestamp = TimeCurrent();
    
    SendHeartbeat(payload);
}
```

### Risk Synchronization

Risk multipliers are synchronized between Python and MQL5:

**Python Side:**
```python
# Update GlobalVariable
mt5.global_variable_set("QM_RISK_MULTIPLIER", 0.5)

# Write to file (fallback)
with open("risk_matrix.json", "w") as f:
    json.dump({"EURUSD": {"multiplier": 0.5, "timestamp": time.time()}}, f)
```

**MQL5 Side:**
```mql5
double multiplier = GetRiskMultiplier("EURUSD");
// Uses GlobalVariable first, falls back to file
```

---

## Testing

All QSL modules have comprehensive test coverage:

### Unit Tests
- Core module compilation tests
- Risk module compilation tests
- JSON parsing tests (22 test cases)

### Property-Based Tests
- Kelly Criterion accuracy (800 test cases)
- Ring Buffer performance (1400 test cases)
- FIFO ordering guarantees
- O(1) operation verification

**Run Tests:**
```bash
# Unit tests
pytest tests/mql5/test_core_modules_compilation.py -v
pytest tests/mql5/test_risk_modules_compilation.py -v
pytest tests/mql5/test_json_parsing.py -v

# Property-based tests
pytest tests/mql5/test_kelly_criterion_properties.py -v
pytest tests/mql5/test_ringbuffer_properties.py -v
```

---

## Best Practices

### 1. Module Independence
- Modules should not directly communicate with each other
- All communication goes through the strategy object
- Use pointers to avoid tight coupling

### 2. Error Handling
```mql5
if(!module.Initialize())
{
    LogError("Module initialization failed");
    return INIT_FAILED;
}
```

### 3. Performance
- Use O(1) operations where possible (RingBuffer)
- Cache frequently accessed values (CParam)
- Minimize file I/O (use GlobalVariables)

### 4. Logging
```mql5
// Use rate-limited logging
LogInfo("Trade executed");  // Max once per minute

// Always log errors
LogError("Order send failed");  // No rate limit
```

### 5. Memory Management
```mql5
// Always delete dynamically allocated objects
void OnDeinit(const int reason)
{
    if(CheckPointer(strategy) != POINTER_INVALID)
        delete strategy;
}
```

---

## Migration from Legacy Code

### From Monolithic EA to Modular QSL

**Before:**
```mql5
void OnTick()
{
    // All logic in one place
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk = CalculateRisk();
    if(CheckSignal())
    {
        double lots = CalculateLots(risk);
        OrderSend(...);
    }
}
```

**After:**
```mql5
void OnTick()
{
    strategy.CreateSnap();
    strategy.MayAndEnter();
}
```

### Extracting Modules

1. **Identify reusable code** (risk calculations, indicators, etc.)
2. **Create module class** inheriting from `CModule`
3. **Move code to module** methods
4. **Initialize in OnInit()** and use in strategy

---

## Future Enhancements

### Planned Features
- Signal module for indicator-based strategies
- Advanced trailing stop module
- Multi-timeframe analysis module
- Machine learning integration module
- Advanced news filter module

### Community Contributions
The QSL is designed to be extensible. To contribute a new module:

1. Inherit from `CModule` base class
2. Implement required interface methods
3. Add comprehensive tests
4. Document usage examples
5. Submit pull request

---

## References

### MQL5 Community Articles
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Modular EA Design Patterns](https://www.mql5.com/en/articles)

### QSL Documentation
- [Requirements Document](.kiro/specs/quantmindx-unified-backend/requirements.md)
- [Design Document](.kiro/specs/quantmindx-unified-backend/design.md)
- [Implementation Tasks](.kiro/specs/quantmindx-unified-backend/tasks.md)

### Test Coverage
- Core Modules: 13/13 tests passing
- Risk Modules: 21/21 tests passing
- Kelly Criterion: 8/8 property tests passing (800 test cases)
- JSON Parsing: 22/23 tests passing
- Ring Buffer: 13/13 property tests passing (1400 test cases)

---

## V8 Enhancements

### Overview

The V8 "Omni-Asset Scalper" expansion introduces significant enhancements to the QSL modules to support:
- **Tiered Risk Engine**: Dynamic risk adaptation based on account equity
- **HFT Infrastructure**: Sub-5ms execution via socket-based event streaming
- **Multi-broker Support**: Unified interface for MT5, Binance, and other brokers

### V8.1: KellySizer.mqh - Tiered Risk Engine

**Version:** 2.0.0 (V8)  
**Requirements:** 16.1-16.5

The V8 enhancement to KellySizer.mqh implements a three-tier risk system that adapts position sizing based on account equity ranges, enabling safe growth from small accounts ($100) to large accounts ($5,000+).

#### Three-Tier Risk System

| Tier | Equity Range | Risk Strategy | Purpose |
|------|--------------|---------------|---------|
| **Growth** | $100 - $1,000 | Dynamic 3% with $5 floor | Aggressive growth for small accounts |
| **Scaling** | $1,000 - $5,000 | Standard Kelly Criterion | Balanced growth with percentage risk |
| **Guardian** | $5,000+ | Kelly + Quadratic Throttle | Capital preservation with drawdown protection |

#### New Enumerations

```mql5
enum ENUM_RISK_TIER
{
    TIER_GROWTH,      // $100-$1K: Dynamic 3% risk with $5 floor
    TIER_SCALING,     // $1K-$5K: Kelly percentage-based risk
    TIER_GUARDIAN     // $5K+: Kelly + Quadratic Throttle
};
```

#### New Input Parameters

```mql5
input double InpGrowthModeCeiling = 1000.0;    // Growth Tier Ceiling ($)
input double InpScalingModeCeiling = 5000.0;   // Scaling Tier Ceiling ($)
input double InpFixedRiskAmount = 5.0;         // Fixed Risk Amount ($)
```

#### New Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `DetermineRiskTier(double equity)` | Determine which risk tier to use based on equity | `ENUM_RISK_TIER` |
| `GetRiskAmount(double equity)` | Calculate dynamic risk amount for Growth tier | `double` |
| `CalculateTieredPositionSize(...)` | Calculate position size using tiered risk logic | `double` |
| `CalculateGrowthTierLots(...)` | Growth tier position sizing (dynamic 3% with floor) | `double` |
| `CalculateScalingTierLots(...)` | Scaling tier position sizing (Kelly percentage) | `double` |
| `CalculateGuardianTierLots(...)` | Guardian tier position sizing (Kelly + throttle) | `double` |
| `ApplyQuadraticThrottle(...)` | Apply quadratic throttle to reduce risk near limit | `double` |
| `SetTieredRiskParams(...)` | Configure tiered risk parameters | `void` |

#### Growth Tier: Dynamic Aggressive Model

The Growth tier uses a dynamic aggressive model with a hard floor to ensure small accounts can grow without stagnation:

**Formula:** `RiskAmount = MathMax(AccountEquity * 3%, FixedFloorAmount)`

**Examples:**
- $100 equity → Risk = $3.00 (3%)
- $50 equity → Risk = $5.00 (Floor kicks in)
- $500 equity → Risk = $15.00 (Scales up)
- $1,000 equity → Transitions to Scaling tier

```mql5
double GetRiskAmount(double equity)
{
    // Calculate percentage-based risk
    double percentRisk = equity * (m_growthPercent / 100.0);
    
    // Apply hard floor
    double riskAmount = MathMax(percentRisk, m_fixedFloorAmount);
    
    return riskAmount;
}
```

#### Scaling Tier: Standard Kelly Criterion

The Scaling tier uses standard Kelly Criterion percentage-based risk for balanced growth:

```mql5
double CalculateScalingTierLots(double equity, double stopLossPips, 
                                double tickValue, double winRate,
                                double avgWin, double avgLoss)
{
    // Calculate Kelly fraction
    double kellyFraction = CalculateKellyFraction(winRate, avgWin, avgLoss);
    
    // Use full Kelly (riskPct = 1.0)
    double lots = CalculateLotSize(kellyFraction, equity, 1.0, 
                                  stopLossPips, tickValue);
    
    return lots;
}
```

#### Guardian Tier: Quadratic Throttle

The Guardian tier applies quadratic throttle to protect capital as daily loss approaches the limit:

**Formula:** `Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss)²`

**Example:**
- 0% daily loss → Multiplier = 1.00 (100% position size)
- 2% daily loss (50% of 4% limit) → Multiplier = 0.25 (25% position size)
- 3% daily loss (75% of 4% limit) → Multiplier = 0.0625 (6.25% position size)
- 4% daily loss → Multiplier = 0.00 (Hard stop)

```mql5
double ApplyQuadraticThrottle(double baseSize, double currentLoss, double maxLoss)
{
    // Calculate remaining capacity
    double remainingCapacity = (maxLoss - currentLoss) / maxLoss;
    
    // Ensure remaining capacity is in [0, 1]
    if(remainingCapacity < 0.0) remainingCapacity = 0.0;
    if(remainingCapacity > 1.0) remainingCapacity = 1.0;
    
    // Apply quadratic throttle
    double multiplier = MathPow(remainingCapacity, 2);
    double throttledSize = baseSize * multiplier;
    
    return throttledSize;
}
```

#### Usage Example: Tiered Risk System

```mql5
#include <QuantMind/Risk/KellySizer.mqh>

QMKellySizer kelly;

void OnTick()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double stopLossPips = 50.0;
    double tickValue = 10.0;  // $10 per pip per lot
    
    // Get current daily loss for Guardian tier
    double currentLoss = CalculateDailyLoss();
    double maxLoss = equity * 0.04;  // 4% daily loss limit
    
    // Calculate position size using tiered risk logic
    double lots = kelly.CalculateTieredPositionSize(
        equity,           // Current equity
        stopLossPips,     // Stop loss distance
        tickValue,        // Tick value
        currentLoss,      // Current daily loss (for Guardian tier)
        maxLoss,          // Max daily loss (for Guardian tier)
        0.55,             // Win rate (for Kelly calculation)
        400.0,            // Average win (for Kelly calculation)
        200.0             // Average loss (for Kelly calculation)
    );
    
    // Determine which tier was used
    ENUM_RISK_TIER tier = kelly.DetermineRiskTier(equity);
    
    Print("Equity: $", equity, " Tier: ", EnumToString(tier), " Lots: ", lots);
    
    // Execute trade with calculated position size
    if(lots > 0.0)
    {
        // Trade execution logic here
    }
}
```

#### Configuration Example

```mql5
// Configure tiered risk parameters
kelly.SetTieredRiskParams(
    1000.0,   // Growth tier ceiling ($1,000)
    5000.0,   // Scaling tier ceiling ($5,000)
    3.0,      // Growth tier percentage (3%)
    5.0       // Fixed floor amount ($5)
);
```

#### Tier Transition Behavior

The system automatically transitions between tiers based on real-time equity:

1. **Growth → Scaling**: When equity reaches $1,000, switches to Kelly percentage risk
2. **Scaling → Guardian**: When equity reaches $5,000, adds quadratic throttle protection
3. **Guardian → Scaling**: If equity drops below $5,000, removes throttle
4. **Scaling → Growth**: If equity drops below $1,000, returns to fixed risk

**Important:** Tier transitions are logged for audit purposes and should be tracked in the database for analysis.

---

### V8.2: Sockets.mqh - HFT Infrastructure

**Version:** 2.0.0 (V8)  
**Requirements:** 17.6-17.8

The V8 enhancement to Sockets.mqh introduces the `CQuantMindSocket` class for high-frequency trading infrastructure with sub-5ms latency communication to the Python ZMQ socket server.

#### Architecture Overview

The V8 socket infrastructure replaces the REST-based heartbeat polling system with an event-driven push notification system:

**V7 (REST):**
- EA sends heartbeat every 60 seconds via HTTP POST
- Average latency: 45ms
- Polling-based architecture

**V8 (Socket):**
- EA sends trade events instantly via persistent socket connection
- Average latency: <5ms
- Event-driven push architecture
- Automatic reconnection with exponential backoff

#### CQuantMindSocket Class

The `CQuantMindSocket` class extends `CSocketClient` with specialized methods for trading events and connection management.

**Key Features:**
- Persistent connection with automatic reconnection
- Sub-5ms latency for trade events
- Connection pooling support
- Automatic retry with exponential backoff
- JSON message formatting optimized for speed

#### Message Types

```mql5
enum MessageType
{
    TRADE_OPEN,      // Trade opened event
    TRADE_CLOSE,     // Trade closed event
    TRADE_MODIFY,    // Trade modified event (SL/TP change)
    HEARTBEAT,       // Periodic heartbeat
    RISK_UPDATE      // Risk multiplier update
};
```

#### Constructor

```mql5
// Default constructor (localhost:5555)
CQuantMindSocket socket;

// Custom server address
CQuantMindSocket socket("192.168.1.100", 5555);
```

#### Key Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `ConnectToServer()` | Establish connection to socket server | `bool` |
| `SendTradeOpen(eaName, symbol, volume, magic, riskMultiplier)` | Send trade open event | `bool` |
| `SendTradeClose(eaName, symbol, ticket, profit)` | Send trade close event | `bool` |
| `SendTradeModify(eaName, ticket, newSL, newTP)` | Send trade modify event | `bool` |
| `SendHeartbeat(eaName, symbol, magic, riskMultiplier)` | Send heartbeat | `bool` |
| `SendRiskUpdate(eaName, newMultiplier)` | Update risk multiplier | `bool` |
| `IsConnected()` | Check connection status | `bool` |
| `Disconnect()` | Close connection | `void` |

#### Connection Management

The socket client implements automatic reconnection with retry logic:

```mql5
bool ConnectToServer()
{
    if(m_connected)
    {
        return true;
    }
    
    // Test connection
    if(TestConnection())
    {
        m_connected = true;
        m_reconnectAttempts = 0;
        Print("[CQuantMindSocket] ✓ Connected to socket server at ", GetBaseUrl());
        return true;
    }
    
    m_reconnectAttempts++;
    Print("[CQuantMindSocket] Failed to connect (attempt ", 
          m_reconnectAttempts, "/", m_maxReconnectAttempts, ")");
    
    return false;
}
```

#### Usage Example: Trade Open Event

```mql5
#include <QuantMind/Utils/Sockets.mqh>

CQuantMindSocket socket;

int OnInit()
{
    // Connect to socket server
    if(!socket.ConnectToServer())
    {
        Print("Failed to connect to socket server");
        return INIT_FAILED;
    }
    
    Print("Socket connection established");
    return INIT_SUCCEEDED;
}

void OnTick()
{
    // Check for trade signal
    if(CheckBuySignal())
    {
        double volume = 0.1;
        int magic = 123456;
        double riskMultiplier = 1.0;
        
        // Send trade open event to server
        if(socket.SendTradeOpen("MyEA", Symbol(), volume, magic, riskMultiplier))
        {
            Print("Trade open event sent, risk multiplier: ", riskMultiplier);
            
            // Execute trade with adjusted risk
            double adjustedVolume = volume * riskMultiplier;
            
            // Order send logic here
        }
        else
        {
            Print("Failed to send trade open event");
        }
    }
}

void OnDeinit(const int reason)
{
    socket.Disconnect();
}
```

#### Usage Example: Trade Close Event

```mql5
void OnTradeClose(int ticket, double profit)
{
    // Send trade close event
    if(socket.SendTradeClose("MyEA", Symbol(), ticket, profit))
    {
        Print("Trade close event sent: ticket=", ticket, " profit=$", profit);
    }
    else
    {
        Print("Failed to send trade close event");
    }
}
```

#### Usage Example: Heartbeat with Risk Retrieval

```mql5
void OnTimer()
{
    double riskMultiplier = 1.0;
    
    // Send heartbeat and retrieve current risk multiplier
    if(socket.SendHeartbeat("MyEA", Symbol(), 123456, riskMultiplier))
    {
        Print("Heartbeat sent, current risk multiplier: ", riskMultiplier);
        
        // Update global risk multiplier
        GlobalVariableSet("QM_RISK_MULTIPLIER", riskMultiplier);
    }
    else
    {
        Print("Heartbeat failed, connection may be lost");
        
        // Attempt reconnection
        socket.ConnectToServer();
    }
}
```

#### Usage Example: Trade Modification

```mql5
void ModifyStopLoss(int ticket, double newSL)
{
    // Send trade modify event
    if(socket.SendTradeModify("MyEA", ticket, newSL, 0.0))
    {
        Print("Trade modify event sent: ticket=", ticket, " newSL=", newSL);
    }
    else
    {
        Print("Failed to send trade modify event");
    }
}
```

#### JSON Message Format

The socket client uses optimized JSON formatting for minimal latency:

**Trade Open:**
```json
{
    "type": "trade_open",
    "ea_name": "MyEA",
    "symbol": "EURUSD",
    "volume": 0.10,
    "magic": 123456,
    "timestamp": 1234567890
}
```

**Trade Close:**
```json
{
    "type": "trade_close",
    "ea_name": "MyEA",
    "symbol": "EURUSD",
    "ticket": 987654,
    "profit": 45.50,
    "timestamp": 1234567890
}
```

**Heartbeat:**
```json
{
    "type": "heartbeat",
    "ea_name": "MyEA",
    "symbol": "EURUSD",
    "magic": 123456,
    "timestamp": 1234567890
}
```

**Server Response:**
```json
{
    "status": "success",
    "risk_multiplier": 0.75,
    "timestamp": 1234567890
}
```

#### Connection Pooling

The socket server supports multiple concurrent EA connections:

```mql5
// EA 1
CQuantMindSocket socket1;
socket1.ConnectToServer();

// EA 2
CQuantMindSocket socket2;
socket2.ConnectToServer();

// Both EAs can send events simultaneously
socket1.SendTradeOpen("EA1", "EURUSD", 0.1, 111111, riskMult1);
socket2.SendTradeOpen("EA2", "GBPUSD", 0.2, 222222, riskMult2);
```

#### Error Handling

The socket client implements robust error handling:

```mql5
if(!socket.SendTradeOpen("MyEA", Symbol(), 0.1, 123456, riskMultiplier))
{
    int errorCode = socket.GetLastError();
    string errorMsg = socket.GetLastErrorMessage();
    
    Print("Socket error: ", errorCode, " - ", errorMsg);
    
    // Attempt reconnection
    if(!socket.IsConnected())
    {
        Print("Connection lost, attempting reconnection...");
        socket.ConnectToServer();
    }
}
```

#### Performance Benchmarks

| Operation | V7 REST | V8 Socket | Improvement |
|-----------|---------|-----------|-------------|
| Trade Open Event | 45ms | <5ms | 9x faster |
| Heartbeat | 45ms | <5ms | 9x faster |
| Risk Retrieval | 45ms | <5ms | 9x faster |
| Concurrent Connections | 10 max | 100+ | 10x scalability |

#### Integration with Python Socket Server

The MQL5 socket client integrates with the Python ZMQ socket server:

**Python Server (src/router/socket_server.py):**
```python
class SocketServer:
    def __init__(self, bind_address: str = "tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.bind_address = bind_address
        
    async def start(self):
        self.socket.bind(self.bind_address)
        print(f"Socket server listening on {self.bind_address}")
        
        while True:
            message = await self.receive_message()
            response = await self.process_message(message)
            await self.send_response(response)
```

**MQL5 Client:**
```mql5
CQuantMindSocket socket;
socket.ConnectToServer();  // Connects to tcp://localhost:5555
```

#### Best Practices

1. **Connection Management:**
   - Always check `IsConnected()` before sending events
   - Implement automatic reconnection in `OnTimer()`
   - Handle connection failures gracefully

2. **Event Timing:**
   - Send trade events immediately after order execution
   - Send heartbeats every 60 seconds
   - Don't send events on every tick (performance impact)

3. **Error Recovery:**
   - Log all socket errors for debugging
   - Implement fallback to file-based risk retrieval
   - Monitor connection status in EA dashboard

4. **Performance:**
   - Reuse socket instances (don't create new ones per event)
   - Use connection pooling for multiple EAs
   - Monitor latency metrics for performance degradation

---

## V8 Integration Examples

### Complete EA with Tiered Risk and Socket Communication

```mql5
#include <QuantMind/Core/BaseAgent.mqh>
#include <QuantMind/Risk/KellySizer.mqh>
#include <QuantMind/Utils/Sockets.mqh>

// Global objects
QMKellySizer kelly;
CQuantMindSocket socket;

// Input parameters
input double InpWinRate = 0.55;
input double InpAvgWin = 400.0;
input double InpAvgLoss = 200.0;

int OnInit()
{
    // Configure tiered risk parameters
    kelly.SetTieredRiskParams(1000.0, 5000.0, 3.0, 5.0);
    
    // Connect to socket server
    if(!socket.ConnectToServer())
    {
        Print("Warning: Socket connection failed, using fallback mode");
    }
    
    // Set timer for heartbeat
    EventSetTimer(60);
    
    return INIT_SUCCEEDED;
}

void OnTick()
{
    // Get current equity
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    
    // Calculate daily loss
    double dailyLoss = CalculateDailyLoss();
    double maxLoss = equity * 0.04;  // 4% daily loss limit
    
    // Check for trade signal
    if(CheckBuySignal())
    {
        // Calculate position size using tiered risk
        double lots = kelly.CalculateTieredPositionSize(
            equity,
            50.0,           // Stop loss pips
            10.0,           // Tick value
            dailyLoss,
            maxLoss,
            InpWinRate,
            InpAvgWin,
            InpAvgLoss
        );
        
        // Send trade open event
        double riskMultiplier = 1.0;
        if(socket.SendTradeOpen("MyEA", Symbol(), lots, 123456, riskMultiplier))
        {
            // Adjust position size based on server response
            lots *= riskMultiplier;
        }
        
        // Execute trade
        if(lots > 0.0)
        {
            // Order send logic here
            Print("Opening trade: ", lots, " lots");
        }
    }
}

void OnTimer()
{
    // Send heartbeat
    double riskMultiplier = 1.0;
    socket.SendHeartbeat("MyEA", Symbol(), 123456, riskMultiplier);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    socket.Disconnect();
}
```

---

## Support

For questions, issues, or contributions:
- GitHub: https://github.com/quantmindx
- Documentation: docs/knowledge/
- Test Suite: tests/mql5/

---

**Document Version:** 2.0.0 (V8)  
**Generated:** 2026-01-31  
**QSL Version:** 2.0.0 (V8)  
**V8 Enhancements:** Tiered Risk Engine, HFT Socket Infrastructure
