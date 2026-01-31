---
name: fetch_historical_data
category: data_skills
version: 1.0.0
dependencies: []
---

# Fetch Historical Data Skill

## Description
Fetches historical OHLCV (Open, High, Low, Close, Volume) data for a given symbol and timeframe.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., EURUSD, XAUUSD)"
    },
    "timeframe": {
      "type": "string",
      "description": "Timeframe (M1, M5, M15, M30, H1, H4, D1)",
      "enum": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    },
    "bars_count": {
      "type": "integer",
      "description": "Number of bars to fetch (default: 100)",
      "default": 100,
      "minimum": 10,
      "maximum": 10000
    }
  },
  "required": ["symbol", "timeframe"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string"
    },
    "timeframe": {
      "type": "string"
    },
    "data": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "time": {
            "type": "string",
            "description": "ISO format timestamp"
          },
          "open": {
            "type": "number"
          },
          "high": {
            "type": "number"
          },
          "low": {
            "type": "number"
          },
          "close": {
            "type": "number"
          },
          "volume": {
            "type": "integer"
          }
        }
      }
    },
    "bars_count": {
      "type": "integer"
    }
  },
  "required": ["symbol", "timeframe", "data", "bars_count"]
}
```

## Code
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

def fetch_historical_data(
    symbol: str,
    timeframe: str,
    bars_count: int = 100
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV data for a trading symbol.

    Note: This is a mock implementation. In production, this would connect
    to MT5 terminal or a data provider API.

    Args:
        symbol: Trading symbol (e.g., EURUSD, XAUUSD)
        timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
        bars_count: Number of bars to fetch

    Returns:
        Dict with symbol, timeframe, data (OHLCV array), and bars_count
    """
    # Map timeframe to minutes
    timeframe_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440
    }

    if timeframe not in timeframe_minutes:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    minutes = timeframe_minutes[timeframe]

    # Generate mock historical data
    # In production, this would use MetaTrader5 Python API
    base_prices = {
        "EURUSD": 1.1000,
        "GBPUSD": 1.3000,
        "USDJPY": 145.00,
        "XAUUSD": 2000.00,
        "XAGUSD": 25.00,
    }
    start_price = base_prices.get(symbol, 1.0)

    # Generate price data using random walk
    np.random.seed(hash(symbol) % (2**32))

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    current_time = datetime.now()
    current_price = start_price

    for i in range(bars_count):
        # Calculate timestamp for this bar
        bar_time = current_time - timedelta(minutes=minutes * (bars_count - i))
        timestamps.append(bar_time)

        # Generate price movement
        change = np.random.normal(0, 0.0002)  # Small random changes
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0001)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0001)))
        volume = np.random.randint(1000, 10000)

        opens.append(round(open_price, 5))
        highs.append(round(high_price, 5))
        lows.append(round(low_price, 5))
        closes.append(round(close_price, 5))
        volumes.append(volume)

        current_price = close_price

    # Format data as array of objects
    data = [
        {
            "time": timestamps[i].isoformat(),
            "open": opens[i],
            "high": highs[i],
            "low": lows[i],
            "close": closes[i],
            "volume": volumes[i]
        }
        for i in range(bars_count)
    ]

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": data,
        "bars_count": bars_count
    }
```

## Example Usage
```python
# Fetch historical data for EURUSD H1
result = fetch_historical_data(
    symbol="EURUSD",
    timeframe="H1",
    bars_count=100
)
# Output: {
#   "symbol": "EURUSD",
#   "timeframe": "H1",
#   "bars_count": 100,
#   "data": [
#     {"time": "2024-01-01T00:00:00", "open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.1002, "volume": 5432},
#     ...
#   ]
# }
```
