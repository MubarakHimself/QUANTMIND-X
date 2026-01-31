# QuantMind Standard Library (QSL) - MQL5 Asset Index

**Version:** 1.0.0  
**Last Updated:** 2026-01-31  
**Status:** Production Ready

## Overview

The QuantMind Standard Library (QSL) is a modular, production-ready MQL5 library designed for building sophisticated Expert Advisors using a component-based architecture. The library follows best practices from the MQL5 community, particularly the modular EA design pattern that separates concerns into independent, reusable modules.

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

**Key Features:**
- HTTP POST requests
- WebSocket support (future)
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

## Support

For questions, issues, or contributions:
- GitHub: https://github.com/quantmindx
- Documentation: docs/knowledge/
- Test Suite: tests/mql5/

---

**Document Version:** 1.0.0  
**Generated:** 2026-01-31  
**QSL Version:** 1.0.0
