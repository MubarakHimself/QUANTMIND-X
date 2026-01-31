# QuantMind Risk Management Library (MQL5)

The QuantMind Risk Management Library provides synchronized risk multiplier management between Python agents and MQL5 Expert Advisors. This bridge enables Python-based risk governance while maintaining low-latency MQL5 execution.

## Installation

1. Copy the library to your MQL5 Include path:
   ```
   extensions/mql5_library/Include/QuantMind/QuantMind_Risk.mqh
   → MQL5/Include/QuantMind/QuantMind_Risk.mqh
   ```

2. Copy test EA to your Experts folder (optional):
   ```
   extensions/mql5_library/Experts/TestRisk.mq5
   → MQL5/Experts/TestRisk.mq5
   ```

3. Open MetaEditor and compile (F7) to verify installation.

## Quick Start

### Basic Usage

```mql5
#include <QuantMind/QuantMind_Risk.mqh>

int OnInit()
{
    RiskInit();
    return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
    RiskDeinit(reason);
}

void OnTick()
{
    double riskMultiplier = GetRiskMultiplier(_Symbol);

    // Use in position sizing
    double lotSize = NormalizeDouble(
        AccountInfoDouble(ACCOUNT_BALANCE) * 0.01 * riskMultiplier,
        2
    );

    // Use in trade execution
    if(ShouldOpenTrade())
    {
        OpenTrade(lotSize * riskMultiplier);
    }
}
```

### With Heartbeat

```mql5
#include <QuantMind/QuantMind_Risk.mqh>

input int HeartbeatIntervalSeconds = 60;
input string EA_NAME = "MyTradingEA";

int OnInit()
{
    RiskInit();
    return INIT_SUCCEEDED;
}

void OnTick()
{
    // Get current risk multiplier
    double riskMultiplier = GetRiskMultiplier(_Symbol);

    // Send heartbeat to Python backend
    SendHeartbeat(EA_NAME, _Symbol,魔术Number, riskMultiplier);

    // Use risk multiplier in trading logic
    ExecuteTradingLogic(riskMultiplier);
}
```

## API Reference

### `GetRiskMultiplier(string symbol)`

Retrieves the risk multiplier for a given symbol using two-tier fallback:

1. **Fast Path**: Reads from `GlobalVariableGet("QM_RISK_MULTIPLIER")` set by Python agent
2. **Fallback Path**: Parses `risk_matrix.json` from `MQL5/Files/` directory

**Parameters:**
- `symbol` - Trading symbol (e.g., "EURUSD", "GBPUSD")

**Returns:**
- `double` - Risk multiplier value (default 1.0 if not found)

**Example:**
```mql5
double multiplier = GetRiskMultiplier("EURUSD");
Print("EURUSD Risk Multiplier: ", multiplier);

double lotSize = baseLotSize * multiplier;
```

### `SendHeartbeat(string eaName, string symbol, int magicNumber, double riskMultiplier)`

Sends lifecycle heartbeat to Python backend for monitoring and health checking.

**Parameters:**
- `eaName` - Name of the Expert Advisor
- `symbol` - Current trading symbol
- `magicNumber` - EA magic number for identification
- `riskMultiplier` - Current risk multiplier being used

**Returns:**
- `bool` - `true` if heartbeat sent successfully, `false` otherwise

**Example:**
```mql5
bool success = SendHeartbeat("MyEA", "EURUSD", 12345, 1.5);
if(!success)
{
    Print("Heartbeat failed - Python backend may be down");
}
```

### `RiskInit()`

Initializes the Risk Management Library. Call from `OnInit()`.

**Returns:**
- `int` - `INIT_SUCCEEDED`

### `RiskDeinit(const int reason)`

Deinitializes the Risk Management Library. Call from `OnDeinit()`.

**Parameters:**
- `reason` - Deinitialization reason code

## Configuration

### Input Parameters

```mql5
input int HeartbeatIntervalSeconds = 60;  // Heartbeat interval (default: 60s)
input string HeartbeatURL = "http://localhost:8000/heartbeat";
```

### Risk Matrix JSON Format

The `risk_matrix.json` file (located in `MQL5/Files/`):

```json
{
  "EURUSD": {
    "multiplier": 1.5,
    "timestamp": 1704067200
  },
  "GBPUSD": {
    "multiplier": 1.25,
    "timestamp": 1704067300
  },
  "XAUUSD": {
    "multiplier": 0.8,
    "timestamp": 1704067400
  }
}
```

### Data Freshness

- Risk data older than 1 hour (3600 seconds) is considered stale
- Stale data triggers a warning and falls back to default multiplier (1.0)
- Timestamp validation prevents using outdated risk parameters

## Python Integration

### Python Side (Writing Risk Matrix)

The Python risk governor writes to:

```python
# Wine path for Linux MT5 installations
RISK_MATRIX_PATH = "~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/risk_matrix.json"

import json
from datetime import datetime

risk_matrix = {
    "EURUSD": {
        "multiplier": 1.5,
        "timestamp": int(datetime.now().timestamp())
    }
}

with open(RISK_MATRIX_PATH, 'w') as f:
    json.dump(risk_matrix, f, indent=2)
```

### Fast Path (GlobalVariable)

For real-time updates, Python can set GlobalVariables via MT5 API:

```python
import MetaTrader5 as mt5

mt5.initialize()
mt5.global_variable_set("QM_RISK_MULTIPLIER", 1.5)
```

## Testing

### Using TestRisk.mq5

1. Compile and attach `TestRisk.mq5` to any chart
2. Enable desired tests in input parameters:
   - `TestGlobalVariablePath` - Test fast path
   - `TestFileFallbackPath` - Test JSON file reading
   - `TestHeartbeat` - Test REST heartbeat

3. Monitor Experts log for test results

### Expected Output

```
========================================
QuantMind Risk Library Test EA Started
========================================
Test Cycle #1 at 2026.01.29 12:00:00
========================================
--- Test Fast Path (GlobalVariable) ---
Set GlobalVariable "QM_RISK_MULTIPLIER" = 1.5
Retrieved multiplier for EURUSD: 1.5
PASS: Fast path working correctly
--- Test Fallback Path (JSON File) ---
Created test risk_matrix.json with content:
PASS: Fallback path working correctly
--- Test REST Heartbeat ---
PASS: Heartbeat function executed successfully
========================================
```

## Wine Path (Linux)

For Linux MT5 installations via Wine, the file path is:

```
~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files/risk_matrix.json
```

Convert to Windows path in Python:

```python
from pathlib import Path
import os

wine_path = Path.home() / ".wine/drive_c/Program Files/MetaTrader 5/MQL5/Files"
os.makedirs(wine_path, exist_ok=True)
```

## Troubleshooting

### WebRequest Errors

If heartbeat fails with error 4060:
- Add `http://localhost:8000` to Tools > Options > Expert Advisors > Allow WebRequest
- Ensure Python backend is running

### File Not Found

If `risk_matrix.json` is not found:
- Verify file is in `MQL5/Files/` directory (not `MQL5/Include/`)
- Check file permissions

### Stale Data Warning

If you see stale data warnings:
- Update timestamp in `risk_matrix.json`
- Ensure Python governor is writing fresh data
- Check system time synchronization

## Best Practices

1. **Always call `RiskInit()` in `OnInit()`**
2. **Always call `RiskDeinit()` in `OnDeinit()`**
3. **Cache `GetRiskMultiplier()` result per tick** (avoid multiple calls)
4. **Handle default multiplier (1.0) appropriately** - may indicate missing config
5. **Monitor heartbeat failures** - may indicate Python backend issues
6. **Validate risk multiplier range** - reject values outside expected bounds

## License

QuantMindX - https://github.com/quantmindx
