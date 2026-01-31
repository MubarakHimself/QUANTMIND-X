---
name: calculate_rsi
category: trading_skills
version: 1.0.0
dependencies:
  - fetch_historical_data
---

# Calculate RSI Skill

## Description
Calculates the Relative Strength Index (RSI) technical indicator for a given symbol and period.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., EURUSD, XAUUSD)"
    },
    "period": {
      "type": "integer",
      "description": "RSI period (typically 14)",
      "default": 14,
      "minimum": 2,
      "maximum": 50
    },
    "prices": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of closing prices for calculation"
    }
  },
  "required": ["prices"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "rsi_value": {
      "type": "number",
      "description": "RSI value (0-100)"
    },
    "signal": {
      "type": "string",
      "enum": ["oversold", "neutral", "overbought"],
      "description": "Trading signal based on RSI level"
    }
  },
  "required": ["rsi_value", "signal"]
}
```

## Code
```python
import numpy as np
from typing import List, Dict, Any

def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)

    Returns:
        Dict with rsi_value and signal

    RSI Formula:
    RSI = 100 - (100 / (1 + RS))

    Where:
    RS = Average Gain / Average Loss

    Signal interpretation:
    - RSI < 30: oversold (potential buy)
    - RSI > 70: overbought (potential sell)
    - Otherwise: neutral
    """
    if len(prices) < period + 1:
        raise ValueError(f"Need at least {period + 1} prices for RSI calculation")

    # Convert to numpy array for efficient calculation
    prices_array = np.array(prices, dtype=np.float64)

    # Calculate price changes
    deltas = np.diff(prices_array)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Calculate initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RSI using Wilder's smoothing method
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RS and RSI
    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    # Generate signal
    if rsi < 30:
        signal = "oversold"
    elif rsi > 70:
        signal = "overbought"
    else:
        signal = "neutral"

    return {
        "rsi_value": round(float(rsi), 2),
        "signal": signal
    }
```

## Example Usage
```python
# Calculate RSI for EURUSD
prices = [1.1000, 1.1010, 1.1020, 1.1015, 1.1005, 1.0995, 1.0990, 1.0995, 1.1005, 1.1015,
          1.1025, 1.1020, 1.1010, 1.1000, 1.0995, 1.0990, 1.0985, 1.0980, 1.0985, 1.0995,
          1.1005, 1.1015, 1.1025, 1.1030, 1.1025, 1.1015, 1.1005, 1.0995, 1.0990, 1.0985]

result = calculate_rsi(prices, period=14)
# Output: {"rsi_value": 45.32, "signal": "neutral"}
```
