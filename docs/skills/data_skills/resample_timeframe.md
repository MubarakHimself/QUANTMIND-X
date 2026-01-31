---
name: resample_timeframe
category: data_skills
description: Resample OHLCV data to different timeframes for multi-timeframe analysis
version: 1.0.0
dependencies:
  - fetch_historical_data
tags:
  - data
  - resampling
  - multi_timeframe
---

# Resample Timeframe Skill

## Description

Resamples OHLCV data from one timeframe to another, enabling multi-timeframe analysis. Converts lower timeframe data (e.g., M1) to higher timeframes (e.g., H1, D1) by aggregating open, high, low, close, and volume values appropriately.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "data": {
      "type": "array",
      "description": "Array of OHLCV bars to resample"
    },
    "from_timeframe": {
      "type": "string",
      "description": "Source timeframe (e.g., 'M1', 'H1')"
    },
    "to_timeframe": {
      "type": "string",
      "description": "Target timeframe (e.g., 'H1', 'D1', 'W1')"
    },
    "method": {
      "type": "string",
      "enum": ["ohlc", "hlc", "hl2", "ohlc4"],
      "description": "Aggregation method (default: ohlc)"
    }
  },
  "required": ["data", "from_timeframe", "to_timeframe"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether resampling was successful"
    },
    "from_timeframe": {
      "type": "string",
      "description": "Source timeframe"
    },
    "to_timeframe": {
      "type": "string",
      "description": "Target timeframe"
    },
    "data": {
      "type": "array",
      "description": "Resampled OHLCV data"
    },
    "count": {
      "type": "integer",
      "description": "Number of resampled bars"
    },
    "compression_ratio": {
      "type": "number",
      "description": "Ratio of source bars per target bar"
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
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def resample_timeframe(
    data: List[Dict[str, Any]],
    from_timeframe: str,
    to_timeframe: str,
    method: str = "ohlc"
) -> Dict[str, Any]:
    """
    Resample OHLCV data from one timeframe to another.

    Args:
        data: Array of OHLCV bars (each with 'time', 'open', 'high', 'low', 'close', 'tick_volume')
        from_timeframe: Source timeframe (e.g., 'M1', 'H1')
        to_timeframe: Target timeframe (e.g., 'H1', 'D1', 'W1')
        method: Aggregation method ('ohlc', 'hlc', 'hl2', 'ohlc4')

    Returns:
        Dictionary containing resampled data and metadata
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # Calculate compression ratio
        ratio = _calculate_compression_ratio(from_timeframe, to_timeframe)

        # Determine resampling frequency
        freq_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "M30": "30min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
            "W1": "1w",
            "MN1": "1m"
        }

        freq = freq_map.get(to_timeframe)
        if not freq:
            return {
                "success": False,
                "error": f"Invalid target timeframe: {to_timeframe}"
            }

        # Resample based on method
        if method == "ohlc":
            # Standard OHLC aggregation
            resampled = df.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum"
            })
        elif method == "hlc":
            # HLC (ignore open)
            resampled = df.resample(freq).agg({
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum"
            })
            resampled["open"] = resampled["close"]  # Use close as open
        elif method == "hl2":
            # HL2 (average of high and low)
            resampled = df.resample(freq).agg({
                "high": "max",
                "low": "min",
                "tick_volume": "sum"
            })
            resampled["open"] = (resampled["high"] + resampled["low"]) / 2
            resampled["close"] = resampled["open"]
        elif method == "ohlc4":
            # OHLC4 (average of O, H, L, C)
            temp = df.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum"
            })
            resampled = temp.copy()
            resampled["close"] = (temp["open"] + temp["high"] + temp["low"] + temp["close"]) / 4
            resampled["open"] = resampled["close"]
        else:
            return {
                "success": False,
                "error": f"Unknown method: {method}"
            }

        # Drop rows with NaN (gaps in data)
        resampled = resampled.dropna()

        # Convert back to list of dictionaries
        resampled_list = []
        for idx, row in resampled.iterrows():
            bar = {
                "time": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "tick_volume": int(row["tick_volume"])
            }
            resampled_list.append(bar)

        return {
            "success": True,
            "from_timeframe": from_timeframe,
            "to_timeframe": to_timeframe,
            "data": resampled_list,
            "count": len(resampled_list),
            "compression_ratio": ratio
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Resampling error: {str(e)}"
        }


def _calculate_compression_ratio(from_tf: str, to_tf: str) -> float:
    """Calculate the compression ratio between timeframes."""
    # Convert to minutes
    to_minutes = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200
    }

    from_min = to_minutes.get(from_tf, 1)
    to_min = to_minutes.get(to_tf, 1)

    return to_min / from_min if from_min > 0 else 1


def resample_multiple_timeframes(
    data: List[Dict[str, Any]],
    base_timeframe: str,
    target_timeframes: List[str],
    method: str = "ohlc"
) -> Dict[str, Any]:
    """
    Resample data to multiple target timeframes.

    Args:
        data: Source OHLCV data
        base_timeframe: Source timeframe
        target_timeframes: List of target timeframes
        method: Aggregation method

    Returns:
        Dictionary mapping timeframe to resampled data
    """
    results = {}

    for tf in target_timeframes:
        result = resample_timeframe(
            data=data,
            from_timeframe=base_timeframe,
            to_timeframe=tf,
            method=method
        )
        results[tf] = result

    return {
        "success": True,
        "base_timeframe": base_timeframe,
        "timeframes": results,
        "count": len(target_timeframes)
    }


def align_multi_timeframe_data(
    data_dict: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Align data from multiple timeframes by timestamp.

    Args:
        data_dict: Dictionary mapping timeframe to data

    Returns:
        Array of aligned data points
    """
    # Convert all to DataFrames
    dfs = {}
    for tf, data in data_dict.items():
        if data:
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            dfs[tf] = df

    # Find common timestamps
    if not dfs:
        return []

    # Get the longest timeframe's index as base
    base_tf = max(dfs.keys(), key=lambda x: len(dfs[x]))
    aligned = dfs[base_tf].copy()

    # Add columns from other timeframes
    for tf, df in dfs.items():
        if tf != base_tf:
            # Forward fill to align timestamps
            aligned[f"{tf}_close"] = df["close"].reindex(aligned.index, method="ffill")

    # Convert to list
    aligned_list = []
    for idx, row in aligned.iterrows():
        point = {
            "time": idx.isoformat(),
            "close": float(row["close"])
        }

        for tf in dfs.keys():
            if tf != base_tf:
                col = f"{tf}_close"
                if col in row:
                    point[tf] = float(row[col])

        aligned_list.append(point)

    return aligned_list


def get_heikin_ashi_bars(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate Heikin Ashi bars from OHLC data.

    Args:
        data: Source OHLCV data

    Returns:
        Array of Heikin Ashi bars
    """
    if not data:
        return []

    ha_bars = []
    prev_ha_close = data[0]["close"]

    for i, bar in enumerate(data):
        if i == 0:
            # First bar
            ha_open = (bar["open"] + bar["close"]) / 2
        else:
            # Subsequent bars
            ha_open = (ha_bars[i-1]["open"] + ha_bars[i-1]["close"]) / 2

        ha_close = (bar["open"] + bar["high"] + bar["low"] + bar["close"]) / 4
        ha_high = max(bar["high"], ha_open, ha_close)
        ha_low = min(bar["low"], ha_open, ha_close)

        ha_bars.append({
            "time": bar["time"],
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "tick_volume": bar.get("tick_volume", 0)
        })

    return ha_bars
```

## Example Usage

```python
# Example 1: Resample M1 to H1
m1_data = fetch_historical_data(
    symbol="EURUSD",
    timeframe="M1",
    count=100
)

result = resample_timeframe(
    data=m1_data["data"],
    from_timeframe="M1",
    to_timeframe="H1"
)

print(f"Resampled {result['from_timeframe']} to {result['to_timeframe']}")
print(f"Bars: {m1_data['count']} -> {result['count']}")
print(f"Compression ratio: {result['compression_ratio']}:1")

# Example 2: Resample to multiple timeframes
result = resample_multiple_timeframes(
    data=m1_data["data"],
    base_timeframe="M1",
    target_timeframes=["M5", "M15", "H1", "H4", "D1"]
)

for tf, data in result["timeframes"].items():
    if data["success"]:
        print(f"{tf}: {data['count']} bars")

# Example 3: Compare price across timeframes
def analyze_trend_across_timeframes(symbol):
    """Analyze trend across multiple timeframes."""
    # Fetch M1 data
    m1_result = fetch_historical_data(
        symbol=symbol,
        timeframe="M1",
        count=1440  # 24 hours
    )

    # Resample to multiple timeframes
    multi_result = resample_multiple_timeframes(
        data=m1_result["data"],
        base_timeframe="M1",
        target_timeframes=["M15", "H1", "H4", "D1"]
    )

    # Analyze trend for each timeframe
    trends = {}
    for tf, result in multi_result["timeframes"].items():
        if result["success"] and result["data"]:
            latest = result["data"][-1]
            prev = result["data"][-2] if len(result["data"]) > 1 else latest

            if latest["close"] > prev["close"]:
                trends[tf] = "bullish"
            elif latest["close"] < prev["close"]:
                trends[tf] = "bearish"
            else:
                trends[tf] = "neutral"

    return trends

trends = analyze_trend_across_timeframes("EURUSD")
print("Multi-timeframe trend analysis:")
for tf, trend in trends.items():
    print(f"  {tf}: {trend}")

# Example 4: Create Heikin Ashi bars
m1_data = fetch_historical_data(symbol="EURUSD", timeframe="M1", count=100)
ha_bars = get_heikin_ashi_bars(m1_data["data"])

print(f"Heikin Ashi bars: {len(ha_bars)}")
print(f"Latest HA close: {ha_bars[-1]['close']}")

# Example 5: Use with pandas for analysis
h1_result = resample_timeframe(
    data=m1_data["data"],
    from_timeframe="M1",
    to_timeframe="H1"
)

if h1_result["success"]:
    df = pd.DataFrame(h1_result["data"])
    df["time"] = pd.to_datetime(df["time"])

    # Calculate indicators on resampled data
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["rsi"] = calculate_rsi(df["close"])

    print(df.head())

# Example 6: Align data for multi-timeframe strategy
def align_mtf_signals(m1_data):
    """Align signals from multiple timeframes."""
    multi_result = resample_multiple_timeframes(
        data=m1_data,
        base_timeframe="M1",
        target_timeframes=["M15", "H1", "H4"]
    )

    # Extract close prices from each timeframe
    data_dict = {}
    for tf, result in multi_result["timeframes"].items():
        if result["success"]:
            data_dict[tf] = result["data"]

    # Align by timestamp
    aligned = align_multi_timeframe_data(data_dict)

    return aligned

aligned_data = align_mtf_signals(m1_data["data"])
for point in aligned_data[-10:]:  # Last 10 aligned points
    print(f"{point['time']}: M1={point['close']:.5f}, H1={point.get('H1', 'N/A')}")
```

## Notes

- Resampling loses some detail (compression ratio)
- Lower timeframes provide more data points
- Higher timeframes show broader trends
- Volume is summed during resampling
- Choose appropriate method for your analysis:
  - `ohlc`: Standard method, preserves all information
  - `hlc`: Used when open is unreliable
  - `hl2`: Midpoint method
  - `ohlc4`: Average price method

## Resampling Methods

| Method | Open | High | Low | Close | Use Case |
|--------|------|------|-----|-------|----------|
| `ohlc` | First | Max | Min | Last | Standard resampling |
| `hlc` | Close | Max | Min | Last | When open unreliable |
| `hl2` | HL/2 | Max | Min | HL/2 | Midpoint analysis |
| `ohlc4` | Avg | Max | Min | Avg | Average price |

## Common Timeframe Conversions

| From | To | Ratio |
|------|-----|-------|
| M1 | M5 | 5:1 |
| M1 | M15 | 15:1 |
| M1 | H1 | 60:1 |
| M1 | H4 | 240:1 |
| M1 | D1 | 1440:1 |
| M5 | H1 | 12:1 |
| M15 | H1 | 4:1 |
| H1 | D1 | 24:1 |

## Dependencies

- `fetch_historical_data`: Source data for resampling
- pandas for DataFrame operations

## See Also

- `fetch_historical_data`: Get source data
- `calculate_rsi`, `calculate_macd`: Indicators on resampled data
- `clean_data_anomalies`: Handle data issues
