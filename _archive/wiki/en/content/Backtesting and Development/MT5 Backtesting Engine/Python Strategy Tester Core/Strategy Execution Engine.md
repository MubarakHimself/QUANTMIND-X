# Strategy Execution Engine

<cite>
**Referenced Files in This Document**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py)
- [core_engine.py](file://src/backtesting/core_engine.py)
- [test_mt5_engine.py](file://tests/backtesting/test_mt5_engine.py)
- [test_backend_bridge_integration.py](file://tests/integration/test_backend_bridge_integration.py)
- [base_strategy.mq5](file://data/assets/templates/base_strategy.mq5)
- [PropManager.mqh](file://src/mql5/Include/QuantMind/Core/PropManager.mqh)
</cite>

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

## Introduction
This document describes the strategy execution engine that enables backtesting Python strategies written in an MQL5-style syntax. The engine simulates MQL5 built-in functions (e.g., iTime, iClose, iHigh, iLow, iVolume), executes strategies bar-by-bar, manages state, captures logs, and computes performance metrics such as Sharpe ratio, maximum drawdown, and total return. It supports both pure Python strategies via a dedicated tester and a hybrid approach compatible with existing MQL5 assets.

## Project Structure
The strategy execution engine spans two primary modules:
- Python backtesting engine with MQL5-style overloads
- A simpler engine leveraging the Backtrader framework for Python strategies

```mermaid
graph TB
subgraph "Backtesting Engines"
A["PythonStrategyTester<br/>run(strategy_code, data, symbol, timeframe)"]
B["QuantMindBacktester<br/>run(strategy_code, data, strategy_name)"]
end
subgraph "MQL5 Compatibility"
C["MQL5Functions<br/>iTime/iClose/iHigh/iLow/iVolume"]
D["MQL5Timeframe<br/>PERIOD_* constants"]
end
subgraph "Data and Assets"
E["OHLCV DataFrame<br/>datetime index"]
F["MQL5 Template Strategy<br/>base_strategy.mq5"]
end
A --> C
A --> D
A --> E
B --> E
F -. "Reference for MQL5-style patterns" .-> A
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L319-L781)
- [core_engine.py](file://src/backtesting/core_engine.py#L13-L83)
- [base_strategy.mq5](file://data/assets/templates/base_strategy.mq5#L1-L45)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L1-L991)
- [core_engine.py](file://src/backtesting/core_engine.py#L1-L83)
- [base_strategy.mq5](file://data/assets/templates/base_strategy.mq5#L1-L45)

## Core Components
- PythonStrategyTester: Executes Python strategies with an on_bar(tester) function, simulating MQL5 built-ins and managing trading state.
- MQL5Functions: Provides MQL5-style functions (iTime, iClose, iHigh, iLow, iVolume) for accessing OHLCV data.
- MQL5Timeframe: Defines MQL5 timeframe constants and conversion helpers.
- MT5BacktestResult: Encapsulates backtest results including metrics and logs.
- QuantMindBacktester: Alternative engine that compiles and runs Python strategies using Backtrader.

Key responsibilities:
- Data preparation and validation
- Strategy compilation and on_bar() discovery
- Bar-by-bar execution loop
- Equity and state updates
- Metrics computation (Sharpe, drawdown, return)
- Logging and error reporting

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L319-L991)
- [core_engine.py](file://src/backtesting/core_engine.py#L13-L83)

## Architecture Overview
The engine orchestrates strategy execution through a deterministic pipeline: reset state, prepare data, compile strategy, iterate bars, update state, compute final metrics.

```mermaid
sequenceDiagram
participant U as "User"
participant T as "PythonStrategyTester"
participant P as "Prepared Data"
participant S as "Strategy Namespace"
participant M as "Metrics"
U->>T : run(strategy_code, data, symbol, timeframe)
T->>T : _reset_state()
T->>P : _prepare_data(data)
T->>S : exec(strategy_code) -> locate on_bar
loop For each bar
T->>T : _update_equity()
T->>S : on_bar(tester)
end
T->>T : close_all_positions()
T->>M : _calculate_result()
M-->>U : MT5BacktestResult
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L695-L781)

## Detailed Component Analysis

### PythonStrategyTester.run()
Workflow highlights:
- Resets internal state and stores symbol/timeframe.
- Prepares OHLCV data ensuring datetime handling and required columns.
- Compiles strategy_code and validates presence of on_bar().
- Iterates over bars, updating equity and invoking on_bar(tester).
- Closes remaining positions and computes final metrics.

```mermaid
flowchart TD
Start(["run() entry"]) --> Reset["Reset state"]
Reset --> StoreParams["Store symbol and timeframe"]
StoreParams --> Prepare["Prepare data (_prepare_data)"]
Prepare --> |Valid| Exec["Compile strategy_code"]
Prepare --> |Invalid| ReturnErr["Return error result"]
Exec --> HasOnBar{"on_bar found?"}
HasOnBar --> |No| ReturnErr
HasOnBar --> |Yes| LoopBars["Loop bars"]
LoopBars --> UpdateEq["_update_equity()"]
UpdateEq --> CallOnBar["Call on_bar(tester)"]
CallOnBar --> NextBar["Next bar"]
NextBar --> |More bars| LoopBars
NextBar --> |Done| ClosePos["close_all_positions()"]
ClosePos --> Metrics["_calculate_result()"]
Metrics --> End(["Return MT5BacktestResult"])
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L695-L781)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L695-L781)

### Data Preparation in _prepare_data()
Responsibilities:
- Accepts raw OHLCV DataFrame.
- Ensures a time column exists and is timezone-aware.
- Adds missing required columns with sensible defaults.
- Normalizes column names and types.

```mermaid
flowchart TD
A["Input DataFrame"] --> B{"Has datetime index?"}
B --> |Yes| C["Set time = index"]
B --> |No| D{"Has 'time' column?"}
D --> |No| E["Create default time range"]
D --> |Yes| F["Keep existing time"]
C --> G["Ensure timezone-aware"]
E --> G
F --> G
G --> H["Ensure required columns:<br/>open, high, low, close, tick_volume"]
H --> I["Return prepared DataFrame"]
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L783-L820)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L783-L820)

### State Management During Execution
State maintained per backtest:
- Symbol, timeframe, current_bar index
- Cash balance and equity curve
- Positions and trades
- Ticket counter and logs
- Data cache per symbol

Reset on each run to avoid cross-test contamination.

```mermaid
classDiagram
class PythonStrategyTester {
+float initial_cash
+float commission
+float slippage
+string symbol
+int timeframe
+int current_bar
+float cash
+float[] equity
+Position[] positions
+Trade[] trades
+float position_size
+int _ticket_counter
+string[] _logs
+Dict~string,DataFrame~ _data_cache
+run(...)
+_prepare_data(...)
+_update_equity()
+_reset_state()
+_log(message)
+_calculate_result()
}
class Position {
+string symbol
+float volume
+float entry_price
+string direction
+datetime entry_time
+int ticket
}
class Trade {
+string symbol
+float volume
+float entry_price
+float exit_price
+string direction
+datetime entry_time
+datetime exit_time
+float profit
+float commission
+int ticket
}
PythonStrategyTester --> Position : "manages"
PythonStrategyTester --> Trade : "records"
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L320-L405)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L124-L148)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L320-L405)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L846-L860)

### MQL5 Built-in Function Overloading
MQL5Functions exposes iTime, iClose, iHigh, iLow, iOpen, iVolume mapped to DataFrame lookups. Module-level convenience functions delegate to a default tester instance.

```mermaid
classDiagram
class MQL5Functions {
-PythonStrategyTester _tester
+iTime(symbol, timeframe, shift) datetime?
+iClose(symbol, timeframe, shift) float?
+iHigh(symbol, timeframe, shift) float?
+iLow(symbol, timeframe, shift) float?
+iOpen(symbol, timeframe, shift) float?
+iVolume(symbol, timeframe, shift) int?
}
class PythonStrategyTester {
+MQL5Functions _mql5_funcs
+iTime(...)
+iClose(...)
+iHigh(...)
+iLow(...)
+iOpen(...)
+iVolume(...)
}
PythonStrategyTester --> MQL5Functions : "owns"
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L154-L314)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L541-L563)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L154-L314)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L541-L563)

### Result Calculation via _calculate_result()
Calculations:
- Returns vector from equity curve
- Sharpe ratio (annualized) using hourly assumption
- Maximum drawdown as percentage
- Total return percentage
- Aggregates logs, equity curve, and trade history

```mermaid
flowchart TD
A["Equity array"] --> B["Compute returns = diff(equity) / equity[:-1]"]
B --> C["Sharpe = mean(excess)/std(excess) * sqrt(252*4)"]
B --> D["Max DD = min((equity-runmax)/runmax)*100"]
B --> E["Total Return = (final - initial)/initial * 100"]
C --> F["Build MT5BacktestResult"]
D --> F
E --> F
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L865-L971)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L865-L971)

### Security Considerations for Arbitrary Code Execution
- Both engines execute arbitrary Python code:
  - PythonStrategyTester.run() uses exec(strategy_code, globals(), namespace)
  - QuantMindBacktester.run() uses exec(strategy_code)
- Production deployments must enforce sandboxing:
  - Resource limits (AS/Data/Stack)
  - IO redirection and stdin/stdout capture
  - Restricted built-ins and forbidden opcodes
  - Timeouts and memory caps
- Reference implementations demonstrate guardrails (reliability_guard, timeout, forbidden op checks) that should be integrated.

Security-related references:
- Sandbox guards and IO restrictions
- Forbidden operations and timeouts

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L715-L716)
- [core_engine.py](file://src/backtesting/core_engine.py#L21-L22)

### Logging Mechanisms
- Internal logging via _log() appends messages to an internal list.
- Final result includes a combined log string for downstream reporting.
- Strategy errors are captured and logged per bar to aid debugging.

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L857-L860)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L771-L772)

### Examples of Strategy Development Patterns
- Minimal buy-on-first-bar pattern
- Trend-following logic using iClose and iOpen
- Entry/exit conditions based on price comparisons

Example references:
- Simple buy strategy
- Entry/exit logic with moving average-like condition
- Integration test invoking run()

**Section sources**
- [test_mt5_engine.py](file://tests/backtesting/test_mt5_engine.py#L169-L196)
- [test_mt5_engine.py](file://tests/backtesting/test_mt5_engine.py#L197-L230)
- [test_backend_bridge_integration.py](file://tests/integration/test_backend_bridge_integration.py#L397-L403)

### Common Pitfalls and Debugging Techniques
Common pitfalls:
- Missing on_bar function in strategy code
- Invalid or empty data passed to run()
- Out-of-bounds shifts to MQL5Functions
- Insufficient equity to open positions
- Incorrect timeframe mapping impacting data retrieval

Debugging tips:
- Inspect returned log for per-bar errors
- Validate data shape and time alignment
- Use module-level convenience functions for quick checks
- Confirm timeframe constants align with data frequency

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L742-L749)
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L107-L110)
- [test_mt5_engine.py](file://tests/backtesting/test_mt5_engine.py#L101-L111)

## Dependency Analysis
Relationships:
- PythonStrategyTester depends on pandas/numpy for data handling and MQL5Functions for OHLCV access.
- MQL5Timeframe provides constants and conversions for timeframes.
- MT5BacktestResult aggregates metrics and logs.
- QuantMindBacktester integrates with Backtrader for analyzer-based metrics.

```mermaid
graph TB
T["PythonStrategyTester"] --> DF["pandas/numpy"]
T --> MF["MQL5Functions"]
T --> TF["MQL5Timeframe"]
T --> RR["MT5BacktestResult"]
Q["QuantMindBacktester"] --> BT["Backtrader"]
Q --> RR
```

**Diagram sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L17-L32)
- [core_engine.py](file://src/backtesting/core_engine.py#L1-L6)

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L17-L32)
- [core_engine.py](file://src/backtesting/core_engine.py#L1-L6)

## Performance Considerations
- Data preparation: Prefer aligned, timezone-aware timestamps and pre-normalized columns to minimize overhead.
- Strategy loops: Keep on_bar() logic vectorized where possible; avoid repeated DataFrame slicing.
- Metrics computation: Equity updates and returns calculation are O(N); ensure N is bounded by dataset length.
- Memory management: Reuse lists and arrays; avoid frequent allocations inside tight loops.
- Scalability: For large datasets, consider chunked execution or parallelization of independent backtests.

[No sources needed since this section provides general guidance]

## Troubleshooting Guide
- Strategy fails to compile:
  - Ensure strategy_code defines on_bar(tester) and uses supported MQL5-style functions.
- No trades recorded:
  - Verify entry conditions and that iClose/iOpen return valid values for current_bar.
- Unexpected drawdown:
  - Check commission and slippage settings; confirm equity curve reflects realized PnL.
- MT5 integration issues:
  - Confirm MT5 availability and connection; verify timeframe mapping.

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L738-L760)
- [test_mt5_engine.py](file://tests/backtesting/test_mt5_engine.py#L347-L376)

## Conclusion
The strategy execution engine provides a robust, MQL5-compatible pathway to backtest Python strategies. It offers deterministic bar-by-bar execution, comprehensive metrics, and flexible data handling. For production use, integrate strong sandboxing and monitoring to mitigate risks associated with arbitrary code execution.

[No sources needed since this section summarizes without analyzing specific files]

## Appendices

### API Summary: PythonStrategyTester
- run(strategy_code, data, symbol, timeframe) -> MT5BacktestResult
- iTime/iClose/iHigh/iLow/iOpen/iVolume
- buy/sell/close_all_positions
- _prepare_data, _update_equity, _calculate_result, _log

**Section sources**
- [mt5_engine.py](file://src/backtesting/mt5_engine.py#L695-L991)

### Related MQL5 Asset Reference
- Base template strategy demonstrates MQL5-style structure and lifecycle hooks.

**Section sources**
- [base_strategy.mq5](file://data/assets/templates/base_strategy.mq5#L1-L45)

### Additional Risk Metric Reference
- Daily drawdown calculation concept for equity monitoring.

**Section sources**
- [PropManager.mqh](file://src/mql5/Include/QuantMind/Core/PropManager.mqh#L83-L93)