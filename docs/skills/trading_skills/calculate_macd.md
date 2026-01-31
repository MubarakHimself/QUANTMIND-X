---
name: calculate_macd
category: trading_skills
description: Calculate the Moving Average Convergence Divergence (MACD) indicator for trend analysis
version: 1.0.0
dependencies:
  - fetch_historical_data
tags:
  - indicator
  - trend
  - momentum
---

# Calculate MACD Skill

## Description

Calculates the Moving Average Convergence Divergence (MACD) indicator, a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "fast_period": {
      "type": "integer",
      "description": "Fast EMA period (default: 12)",
      "default": 12,
      "minimum": 5,
      "maximum": 50
    },
    "slow_period": {
      "type": "integer",
      "description": "Slow EMA period (default: 26)",
      "default": 26,
      "minimum": 10,
      "maximum": 100
    },
    "signal_period": {
      "type": "integer",
      "description": "Signal line EMA period (default: 9)",
      "default": 9,
      "minimum": 5,
      "maximum": 20
    },
    "timeframe": {
      "type": "string",
      "description": "Timeframe for data (e.g., 'H1', 'D1')",
      "default": "H1"
    },
    "bars": {
      "type": "integer",
      "description": "Number of bars to fetch",
      "default": 100,
      "minimum": 50
    }
  },
  "required": ["symbol"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "macd_line": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "MACD line values (fast EMA - slow EMA)"
    },
    "signal_line": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Signal line EMA values"
    },
    "histogram": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "MACD histogram (MACD - Signal)"
    },
    "current_macd": {
      "type": "number",
      "description": "Most recent MACD value"
    },
    "current_signal": {
      "type": "number",
      "description": "Most recent signal line value"
    },
    "current_histogram": {
      "type": "number",
      "description": "Most recent histogram value"
    },
    "crossover_signal": {
      "type": "string",
      "enum": ["bullish", "bearish", "none"],
      "description": "Crossover signal if detected"
    },
    "timestamps": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Timestamps for each value"
    }
  }
}
```

## Code

```python
import numpy as np
from typing import Dict, Any, List
import MetaTrader5 as mt5
from datetime import datetime

def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        data: Array of price data
        period: EMA period

    Returns:
        Array of EMA values
    """
    ema = np.zeros_like(data)
    multiplier = 2 / (period + 1)
    ema[0] = data[0]

    for i in range(1, len(data)):
        ema[i] = (data[i] * multiplier) + (ema[i - 1] * (1 - multiplier))

    return ema

def calculate_macd(
    symbol: str,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    timeframe: str = "H1",
    bars: int = 100
) -> Dict[str, Any]:
    """
    Calculate Moving Average Convergence Divergence (MACD) indicator.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        timeframe: Timeframe for data (e.g., 'H1', 'D1')
        bars: Number of bars to fetch

    Returns:
        Dictionary containing MACD line, signal line, histogram, and crossover signal

    Raises:
        ValueError: If insufficient data or invalid parameters
        ConnectionError: If MT5 connection fails
    """
    # Validate parameters
    if fast_period >= slow_period:
        raise ValueError("fast_period must be less than slow_period")

    # Map timeframe string to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    if timeframe not in tf_map:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    # Fetch historical data from MT5
    mt5_timeframe = tf_map[timeframe]
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

    if rates is None or len(rates) < slow_period + signal_period:
        min_bars = slow_period + signal_period
        raise ValueError(f"Insufficient data for MACD calculation. Need at least {min_bars} bars")

    # Extract close prices
    closes = rates['close']

    # Calculate EMAs
    fast_ema = calculate_ema(closes, fast_period)
    slow_ema = calculate_ema(closes, slow_period)

    # Calculate MACD line
    macd_line = fast_ema - slow_ema

    # Calculate signal line (EMA of MACD)
    signal_line = calculate_ema(macd_line, signal_period)

    # Calculate histogram
    histogram = macd_line - signal_line

    # Align arrays to same length
    start_idx = slow_period + signal_period - 1
    macd_values = macd_line[start_idx:]
    signal_values = signal_line[start_idx:]
    histogram_values = histogram[start_idx:]

    # Get current values
    current_macd = float(macd_values[-1])
    current_signal = float(signal_values[-1])
    current_histogram = float(histogram_values[-1])

    # Detect crossover
    if len(histogram_values) >= 2:
        prev_hist = histogram_values[-2]
        curr_hist = histogram_values[-1]

        if prev_hist < 0 and curr_hist > 0:
            crossover_signal = "bullish"
        elif prev_hist > 0 and curr_hist < 0:
            crossover_signal = "bearish"
        else:
            crossover_signal = "none"
    else:
        crossover_signal = "none"

    # Generate timestamps
    timestamps = [
        datetime.fromtimestamp(rates['time'][i + start_idx]).isoformat()
        for i in range(len(macd_values))
    ]

    return {
        "macd_line": [float(v) for v in macd_values],
        "signal_line": [float(v) for v in signal_values],
        "histogram": [float(v) for v in histogram_values],
        "current_macd": round(current_macd, 5),
        "current_signal": round(current_signal, 5),
        "current_histogram": round(current_histogram, 5),
        "crossover_signal": crossover_signal,
        "timestamps": timestamps
    }
```

## Example Usage

```python
# Calculate MACD for EURUSD on H1 timeframe
result = calculate_macd(
    symbol="EURUSD",
    fast_period=12,
    slow_period=26,
    signal_period=9,
    timeframe="H1",
    bars=100
)

print(f"Current MACD: {result['current_macd']}")
print(f"Current Signal: {result['current_signal']}")
print(f"Current Histogram: {result['current_histogram']}")
print(f"Crossover Signal: {result['crossover_signal']}")

# Example output:
# Current MACD: 0.00015
# Current Signal: 0.00012
# Current Histogram: 0.00003
# Crossover Signal: bullish

# Use in trading logic
if result['crossover_signal'] == "bullish":
    print("Bullish crossover detected - consider long entry")
elif result['crossover_signal'] == "bearish":
    print("Bearish crossover detected - consider short entry")

# Also check histogram trend
if result['current_histogram'] > 0:
    print("Bullish momentum (histogram positive)")
else:
    print("Bearish momentum (histogram negative)")
```

## Notes

- MACD is best used in trending markets
- Bullish crossover (MACD crosses above signal) is a buy signal
- Bearish crossover (MACD crosses below signal) is a sell signal
- Histogram shows momentum strength and direction
- Divergence between price and MACD can signal potential reversals
- Default parameters (12, 26, 9) are standard but can be adjusted for different markets

## Dependencies

- `fetch_historical_data`: Required to get OHLC data for calculations
- MetaTrader5 Python API: Must be connected to MT5 terminal
