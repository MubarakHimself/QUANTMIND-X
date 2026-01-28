# QuantMindX Coding Standards

## General Principles
1.  **Safety First:** No file system IO outside allowed directories. No external network calls unless via MCP.
2.  **Type Hints:** All Python code must use `typing`.
3.  **Documentation:** All functions must have docstrings.

## MQL5 Guidelines
1.  **Strict Compilation:** Code must compile with 0 warnings.
2.  **Risk Management:** Every order send must have SL/TP.
3.  **Input Parameters:** All strategy parameters must be `input` variables.

## Python Backtesting
1.  **Backtrader:** Use `Backtrader` for all python strategy logic.
2.  **Data Feeds:** flexible processing of CSV or Pandas DataFrames.
