---
name: calculate_rsi
category: trading_skills
description: Calculate the Relative Strength Index (RSI) technical indicator for trading analysis
version: 1.0.0
dependencies:
  - fetch_historical_data
tags:
  - indicator
  - momentum
  - technical_analysis
---

# Calculate RSI Skill

## Description

Calculates the Relative Strength Index (RSI), a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100, with values above 70 indicating overbought conditions and values below 30 indicating oversold conditions.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "period": {
      "type": "integer",
      "description": "RSI calculation period (default: 14)",
      "default": 14,
      "minimum": 2,
      "maximum": 50
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
    "rsi_values": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of RSI values"
    },
    "current_rsi": {
      "type": "number",
      "description": "Most recent RSI value"
    },
    "signal": {
      "type": "string",
      "enum": ["overbought", "oversold", "neutral"],
      "description": "Trading signal based on RSI level"
    },
    "timestamps": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Timestamps for each RSI value"
    }
  }
}
```

## Code

```python
import numpy as np
from typing import Dict, Any, List
import MetaTrader5 as mt5
from datetime import datetime, timedelta

def calculate_rsi(
    symbol: str,
    period: int = 14,
    timeframe: str = "H1",
    bars: int = 100
) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI) for a given symbol.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        period: RSI calculation period (default: 14)
        timeframe: Timeframe for data (e.g., 'H1', 'D1')
        bars: Number of bars to fetch

    Returns:
        Dictionary containing RSI values, current RSI, signal, and timestamps

    Raises:
        ValueError: If insufficient data or invalid parameters
        ConnectionError: If MT5 connection fails
    """
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

    if rates is None or len(rates) < period + 1:
        raise ValueError(f"Insufficient data for RSI calculation. Need at least {period + 1} bars")

    # Extract close prices
    closes = rates['close']

    # Calculate price changes
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RSI using Wilder's smoothing method
    rsi_values = []
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi_values.append(rsi)

        # Update averages with Wilder's smoothing
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Get current RSI value
    current_rsi = rsi_values[-1] if rsi_values else 50.0

    # Determine trading signal
    if current_rsi >= 70:
        signal = "overbought"
    elif current_rsi <= 30:
        signal = "oversold"
    else:
        signal = "neutral"

    # Generate timestamps
    timestamps = [
        datetime.fromtimestamp(rates['time'][i + period]).isoformat()
        for i in range(len(rsi_values))
    ]

    return {
        "rsi_values": rsi_values,
        "current_rsi": round(current_rsi, 2),
        "signal": signal,
        "timestamps": timestamps
    }
```

## Example Usage

```python
# Calculate RSI for EURUSD on H1 timeframe
result = calculate_rsi(
    symbol="EURUSD",
    period=14,
    timeframe="H1",
    bars=100
)

print(f"Current RSI: {result['current_rsi']}")
print(f"Signal: {result['signal']}")

# Example output:
# Current RSI: 72.45
# Signal: overbought

# Use in trading logic
if result['signal'] == "oversold":
    print("Consider entering long position")
elif result['signal'] == "overbought":
    print("Consider entering short position or taking profits")
```

## Notes

- RSI is most effective in ranging markets
- Divergences between price and RSI can signal potential reversals
- Default period of 14 is standard, but can be adjusted for different market conditions
- Values above 70/below 30 are standard thresholds, but can be adjusted (e.g., 80/20 for stronger signals)

## Dependencies

- `fetch_historical_data`: Required to get OHLC data for calculations
- MetaTrader5 Python API: Must be connected to MT5 terminal
