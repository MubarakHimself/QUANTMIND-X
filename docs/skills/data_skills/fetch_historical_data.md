---
name: fetch_historical_data
category: data_skills
description: Fetch historical OHLCV data from MT5 for backtesting and analysis
version: 1.0.0
dependencies: []
tags:
  - data
  - historical
  - ohlcv
---

# Fetch Historical Data Skill

## Description

Fetches historical Open, High, Low, Close, Volume (OHLCV) data from MetaTrader 5 for backtesting strategies, calculating indicators, and performing market analysis. Supports various timeframes and date ranges with flexible filtering options.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "timeframe": {
      "type": "string",
      "enum": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"],
      "description": "Data timeframe (default: H1)"
    },
    "start_date": {
      "type": "string",
      "description": "Start date in ISO format (YYYY-MM-DD) or datetime"
    },
    "end_date": {
      "type": "string",
      "description": "End date in ISO format (YYYY-MM-DD) or datetime (default: now)"
    },
    "count": {
      "type": "integer",
      "description": "Number of bars to fetch (alternative to date range)",
      "minimum": 1,
      "maximum": 100000
    },
    "include_weekends": {
      "type": "boolean",
      "description": "Include weekend data (default: false)"
    },
    "fill_gaps": {
      "type": "boolean",
      "description": "Fill missing bars with previous close (default: false)"
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
    "success": {
      "type": "boolean",
      "description": "Whether data fetch was successful"
    },
    "symbol": {
      "type": "string",
      "description": "Symbol fetched"
    },
    "timeframe": {
      "type": "string",
      "description": "Timeframe of data"
    },
    "data": {
      "type": "array",
      "description": "Array of OHLCV bars"
    },
    "count": {
      "type": "integer",
      "description": "Number of bars returned"
    },
    "date_range": {
      "type": "object",
      "properties": {
        "start": {
          "type": "string"
        },
        "end": {
          "type": "string"
        }
      }
    },
    "statistics": {
      "type": "object",
      "properties": {
        "highest_high": {
          "type": "number"
        },
        "lowest_low": {
          "type": "number"
        },
        "avg_volume": {
          "type": "number"
        }
      }
    },
    "error": {
      "type": "string",
      "description": "Error message if unsuccessful"
    }
  }
}
```

## Code

```python
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def fetch_historical_data(
    symbol: str,
    timeframe: str = "H1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    count: Optional[int] = None,
    include_weekends: bool = False,
    fill_gaps: bool = False
) -> Dict[str, Any]:
    """
    Fetch historical OHLCV data from MetaTrader 5.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
        timeframe: Data timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        start_date: Start date (ISO format or 'YYYY-MM-DD')
        end_date: End date (ISO format or 'YYYY-MM-DD', default: now)
        count: Number of bars to fetch (alternative to date range)
        include_weekends: Include weekend data in results
        fill_gaps: Fill missing bars with previous close

    Returns:
        Dictionary containing OHLCV data and statistics
    """
    # Map timeframe strings to MT5 constants
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }

    if timeframe not in tf_map:
        return {
            "success": False,
            "error": f"Invalid timeframe: {timeframe}"
        }

    mt5_timeframe = tf_map[timeframe]

    # Initialize MT5
    if not mt5.initialize():
        return {
            "success": False,
            "error": f"MT5 initialization failed: {mt5.last_error()}"
        }

    try:
        # Fetch data based on parameters
        rates = None

        if count is not None:
            # Fetch by count
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        elif start_date and end_date:
            # Fetch by date range
            dt_start = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            dt_end = datetime.fromisoformat(end_date) if isinstance(end_date, str) else end_date
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, dt_start, dt_end)
        elif start_date:
            # Fetch from date to present
            dt_start = datetime.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            rates = mt5.copy_rates_from(symbol, mt5_timeframe, dt_start)
        else:
            # Default: fetch last 100 bars
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 100)

        if rates is None or len(rates) == 0:
            return {
                "success": False,
                "error": f"No data retrieved for {symbol} {timeframe}"
            }

        # Convert to list of dictionaries
        data_list = []
        for rate in rates:
            bar = {
                "time": datetime.fromtimestamp(rate["time"]).isoformat(),
                "open": float(rate["open"]),
                "high": float(rate["high"]),
                "low": float(rate["low"]),
                "close": float(rate["close"]),
                "tick_volume": int(rate["tick_volume"]),
                "spread": int(rate["spread"]),
                "real_volume": int(rate["real_volume"])
            }
            data_list.append(bar)

        # Filter weekends if requested
        if not include_weekends:
            data_list = [
                bar for bar in data_list
                if datetime.fromisoformat(bar["time"]).weekday() < 5  # Mon-Fri
            ]

        # Fill gaps if requested
        if fill_gaps and len(data_list) > 1:
            data_list = _fill_gaps(data_list, timeframe)

        # Calculate statistics
        closes = [bar["close"] for bar in data_list]
        volumes = [bar["tick_volume"] for bar in data_list]

        date_range = {
            "start": data_list[0]["time"] if data_list else None,
            "end": data_list[-1]["time"] if data_list else None
        }

        statistics = {
            "highest_high": max([bar["high"] for bar in data_list]) if data_list else None,
            "lowest_low": min([bar["low"] for bar in data_list]) if data_list else None,
            "avg_volume": sum(volumes) / len(volumes) if volumes else None
        }

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data_list,
            "count": len(data_list),
            "date_range": date_range,
            "statistics": statistics
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Data fetch error: {str(e)}"
        }

    finally:
        mt5.shutdown()


def _fill_gaps(data: List[Dict], timeframe: str) -> List[Dict]:
    """
    Fill missing bars with previous close price.

    Args:
        data: Array of OHLCV bars
        timeframe: Timeframe string

    Returns:
        Array with gaps filled
    """
    if len(data) < 2:
        return data

    # Determine time delta for timeframe
    timedelta_map = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
        "W1": timedelta(weeks=1),
        "MN1": timedelta(days=30)
    }

    delta = timedelta_map.get(timeframe, timedelta(hours=1))

    filled_data = []
    current_time = datetime.fromisoformat(data[0]["time"])

    for i, bar in enumerate(data):
        bar_time = datetime.fromisoformat(bar["time"])

        # Fill gaps between bars
        while current_time < bar_time:
            # Get previous close (or use current bar's open if first)
            prev_close = filled_data[-1]["close"] if filled_data else bar["open"]

            filled_data.append({
                "time": current_time.isoformat(),
                "open": prev_close,
                "high": prev_close,
                "low": prev_close,
                "close": prev_close,
                "tick_volume": 0,
                "spread": 0,
                "real_volume": 0
            })

            current_time += delta

        # Add actual bar
        filled_data.append(bar)
        current_time = bar_time + delta

    return filled_data


def fetch_multiple_symbols(
    symbols: List[str],
    timeframe: str = "H1",
    count: int = 100
) -> Dict[str, Any]:
    """
    Fetch historical data for multiple symbols.

    Args:
        symbols: List of trading symbols
        timeframe: Data timeframe
        count: Number of bars per symbol

    Returns:
        Dictionary mapping symbol to its data
    """
    results = {}

    for symbol in symbols:
        result = fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            count=count
        )
        results[symbol] = result

    return {
        "success": True,
        "data": results,
        "symbols_count": len(symbols)
    }
```

## Example Usage

```python
# Example 1: Fetch last 100 H1 bars for EURUSD
result = fetch_historical_data(
    symbol="EURUSD",
    timeframe="H1",
    count=100
)

print(f"Fetched {result['count']} bars")
print(f"Date range: {result['date_range']['start']} to {result['date_range']['end']}")
print(f"Highest: {result['statistics']['highest_high']}")
print(f"Lowest: {result['statistics']['lowest_low']}")

# Example 2: Fetch data for specific date range
result = fetch_historical_data(
    symbol="GBPUSD",
    timeframe="D1",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"January data: {result['count']} daily bars")

# Example 3: Fetch data with gap filling
result = fetch_historical_data(
    symbol="USDJPY",
    timeframe="H1",
    start_date="2024-01-01",
    count=200,
    fill_gaps=True
)

# Example 4: Convert to pandas DataFrame
if result["success"]:
    df = pd.DataFrame(result["data"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Calculate returns
    df["returns"] = df["close"].pct_change()

    print(df.head())

# Example 5: Fetch multiple symbols
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
multi_result = fetch_multiple_symbols(symbols, timeframe="H4", count=50)

for symbol, data in multi_result["data"].items():
    if data["success"]:
        print(f"{symbol}: {data['count']} bars")

# Example 6: Use in indicator calculation
def calculate_sma(symbol, period=20, count=100):
    result = fetch_historical_data(symbol=symbol, timeframe="H1", count=count)

    if result["success"]:
        closes = [bar["close"] for bar in result["data"]]
        sma = pd.Series(closes).rolling(window=period).mean()
        return sma.tolist()

    return None

sma_20 = calculate_sma("EURUSD", period=20)
```

## Notes

- MT5 terminal must be running and connected to broker
- Data availability depends on broker's history depth
- Weekend data is typically empty for forex markets
- `fill_gaps` is useful for indicators that require continuous data
- Real volume may be zero for forex (use tick_volume instead)

## Timeframe Reference

| Code | Description | Typical Use |
|------|-------------|-------------|
| M1 | 1 minute | Scalping |
| M5 | 5 minutes | Day trading |
| M15 | 15 minutes | Short-term trading |
| M30 | 30 minutes | Intraday |
| H1 | 1 hour | Intraday/swing |
| H4 | 4 hours | Swing trading |
| D1 | Daily | Position trading |
| W1 | Weekly | Long-term analysis |
| MN1 | Monthly | Macro analysis |

## Dependencies

- MetaTrader5 Python API
- pandas for DataFrame conversion (optional)

## See Also

- `fetch_live_tick`: Get real-time tick data
- `resample_timeframe`: Convert data to different timeframe
- `clean_data_anomalies`: Handle data quality issues
- `calculate_rsi`, `calculate_macd`: Indicators that use historical data
