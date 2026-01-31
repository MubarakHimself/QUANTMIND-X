---
name: write_indicator_buffer
category: system_skills
description: Write indicator values to MT5 indicator buffers for custom indicator development
version: 1.0.0
dependencies:
  - read_mt5_data
tags:
  - mt5
  - indicator
  - buffer
---

# Write Indicator Buffer Skill

## Description

Writes calculated indicator values to MetaTrader 5 indicator buffers for custom indicator development and testing. This skill is useful for developing custom indicators in Python and testing them before porting to MQL5, or for real-time indicator calculation and display.

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
      "description": "Indicator timeframe (default: H1)"
    },
    "indicator_name": {
      "type": "string",
      "description": "Name of the indicator (for identification)"
    },
    "buffer_index": {
      "type": "integer",
      "description": "Buffer index (0-7 for MT5 indicators)",
      "minimum": 0,
      "maximum": 7
    },
    "values": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of indicator values to write"
    },
    "timestamps": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Corresponding timestamps (ISO format)"
    },
    "draw_type": {
      "type": "string",
      "enum": ["line", "histogram", "arrow", "dots"],
      "description": "How to draw the indicator (default: line)"
    },
    "line_color": {
      "type": "string",
      "description": "Line color in hex format (e.g., '#FF0000' for red)"
    },
    "line_width": {
      "type": "integer",
      "description": "Line width (1-5, default: 1)",
      "minimum": 1,
      "maximum": 5
    }
  },
  "required": ["symbol", "indicator_name", "buffer_index", "values"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether write operation was successful"
    },
    "indicator_name": {
      "type": "string",
      "description": "Name of the indicator"
    },
    "buffer_index": {
      "type": "integer",
      "description": "Buffer index written to"
    },
    "values_written": {
      "type": "integer",
      "description": "Number of values written"
    },
    "file_path": {
      "type": "string",
      "description": "Path to written file (for file-based storage)"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp of write operation"
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
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

# Directory for storing indicator buffers
INDICATOR_DATA_DIR = "/tmp/mt5_indicators"

def write_indicator_buffer(
    symbol: str,
    indicator_name: str,
    buffer_index: int,
    values: List[float],
    timeframe: str = "H1",
    timestamps: Optional[List[str]] = None,
    draw_type: str = "line",
    line_color: str = "#0066CC",
    line_width: int = 1
) -> Dict[str, Any]:
    """
    Write indicator values to buffer/file for MT5 indicator development.

    Note: This is a Python-side implementation. For actual MT5 indicator buffers,
    you would need to use MQL5 or the MT5 Python API with custom indicators.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        indicator_name: Name of the indicator
        buffer_index: Buffer index (0-7)
        values: Array of indicator values
        timeframe: Indicator timeframe
        timestamps: Optional timestamps for each value
        draw_type: How to draw the indicator
        line_color: Line color in hex format
        line_width: Line width (1-5)

    Returns:
        Dictionary containing write result and metadata
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "indicator_name": indicator_name,
        "buffer_index": buffer_index
    }

    try:
        # Validate inputs
        if not 0 <= buffer_index <= 7:
            raise ValueError("Buffer index must be between 0 and 7")

        if not values:
            raise ValueError("Values array cannot be empty")

        if timestamps and len(timestamps) != len(values):
            raise ValueError("Timestamps and values must have same length")

        # Create indicator data directory
        os.makedirs(INDICATOR_DATA_DIR, exist_ok=True)

        # Prepare indicator data
        indicator_data = {
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_name": indicator_name,
                "buffer_index": buffer_index,
                "draw_type": draw_type,
                "line_color": line_color,
                "line_width": line_width,
                "created_at": datetime.now().isoformat()
            },
            "data": []
        }

        # Combine values with timestamps if provided
        for i, value in enumerate(values):
            entry = {
                "index": i,
                "value": float(value)
            }

            if timestamps and i < len(timestamps):
                entry["timestamp"] = timestamps[i]

            indicator_data["data"].append(entry)

        # Write to file
        file_name = f"{symbol}_{timeframe}_{indicator_name}_buffer{buffer_index}.json"
        file_path = os.path.join(INDICATOR_DATA_DIR, file_name)

        with open(file_path, 'w') as f:
            json.dump(indicator_data, f, indent=2)

        # Update result
        result["success"] = True
        result["values_written"] = len(values)
        result["file_path"] = file_path

        # Also write in CSV format for easy import to MT5/Excel
        csv_file = file_path.replace('.json', '.csv')
        with open(csv_file, 'w') as f:
            # Header
            f.write("index,value,timestamp\n")

            # Data
            for entry in indicator_data["data"]:
                ts = entry.get("timestamp", "")
                f.write(f"{entry['index']},{entry['value']},{ts}\n")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["values_written"] = 0

    return result


def read_indicator_buffer(
    symbol: str,
    indicator_name: str,
    buffer_index: int,
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """
    Read previously written indicator buffer from file.

    Args:
        symbol: Trading symbol
        indicator_name: Name of the indicator
        buffer_index: Buffer index
        timeframe: Indicator timeframe

    Returns:
        Dictionary containing indicator data
    """
    file_name = f"{symbol}_{timeframe}_{indicator_name}_buffer{buffer_index}.json"
    file_path = os.path.join(INDICATOR_DATA_DIR, file_name)

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"Indicator buffer file not found: {file_path}"
        }

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        return {
            "success": True,
            "data": data
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def append_to_indicator_buffer(
    symbol: str,
    indicator_name: str,
    buffer_index: int,
    new_values: List[float],
    new_timestamps: Optional[List[str]] = None,
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """
    Append new values to existing indicator buffer.

    Args:
        symbol: Trading symbol
        indicator_name: Name of the indicator
        buffer_index: Buffer index
        new_values: New values to append
        new_timestamps: New timestamps for the values
        timeframe: Indicator timeframe

    Returns:
        Dictionary containing append result
    """
    # Read existing buffer
    existing = read_indicator_buffer(symbol, indicator_name, buffer_index, timeframe)

    if not existing["success"]:
        # Create new buffer if doesn't exist
        return write_indicator_buffer(
            symbol=symbol,
            indicator_name=indicator_name,
            buffer_index=buffer_index,
            values=new_values,
            timestamps=new_timestamps,
            timeframe=timeframe
        )

    # Append to existing data
    data = existing["data"]
    current_index = len(data["data"])

    for i, value in enumerate(new_values):
        entry = {
            "index": current_index + i,
            "value": float(value)
        }

        if new_timestamps and i < len(new_timestamps):
            entry["timestamp"] = new_timestamps[i]

        data["data"].append(entry)

    # Write back
    file_name = f"{symbol}_{timeframe}_{indicator_name}_buffer{buffer_index}.json"
    file_path = os.path.join(INDICATOR_DATA_DIR, file_name)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    return {
        "success": True,
        "values_written": len(new_values),
        "total_values": len(data["data"]),
        "file_path": file_path
    }


def calculate_and_write_indicator(
    symbol: str,
    indicator_name: str,
    calculation_function: callable,
    timeframe: str = "H1",
    bars: int = 100
) -> Dict[str, Any]:
    """
    Calculate indicator values and write to buffer.

    Convenience function that combines data reading, calculation, and writing.

    Args:
        symbol: Trading symbol
        indicator_name: Name of the indicator
        calculation_function: Function that takes OHLC data and returns indicator values
        timeframe: Timeframe for calculation
        bars: Number of bars to calculate

    Returns:
        Dictionary containing calculation and write result
    """
    # Import here to avoid circular dependency
    from src.agents.skills.skill_schema import read_mt5_data

    # Fetch OHLC data
    rates_result = read_mt5_data(
        data_type="rates",
        symbol=symbol,
        timeframe=timeframe,
        count=bars
    )

    if not rates_result["success"]:
        return {
            "success": False,
            "error": f"Failed to fetch rates: {rates_result['error']}"
        }

    # Extract OHLC data
    rates = rates_result["data"]["rates"]
    timestamps = [r["time"] for r in rates]

    # Prepare data for calculation function
    import pandas as pd
    df = pd.DataFrame(rates)

    # Calculate indicator values
    try:
        values = calculation_function(df)
    except Exception as e:
        return {
            "success": False,
            "error": f"Calculation failed: {str(e)}"
        }

    # Write to buffer
    write_result = write_indicator_buffer(
        symbol=symbol,
        indicator_name=indicator_name,
        buffer_index=0,
        values=values.tolist() if isinstance(values, np.ndarray) else list(values),
        timestamps=timestamps,
        timeframe=timeframe
    )

    return write_result
```

## Example Usage

```python
# Example 1: Write simple moving average values
import numpy as np

# Calculate SMA values
def calculate_sma(prices, period=14):
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(np.nan)
        else:
            sma.append(np.mean(prices[i-period+1:i+1]))
    return sma

# Get close prices and calculate SMA
# (assuming you have fetched rates data)
close_prices = [1.0850, 1.0855, 1.0860, ...]  # Your price data
sma_values = calculate_sma(close_prices, period=20)

# Write to buffer
result = write_indicator_buffer(
    symbol="EURUSD",
    indicator_name="SMA_20",
    buffer_index=0,
    values=sma_values,
    timeframe="H1",
    line_color="#FF0000",
    line_width=2
)

print(f"Written {result['values_written']} values to {result['file_path']}")

# Example 2: Calculate and write RSI using convenience function
def calculate_rsi_from_ohlc(df, period=14):
    """Calculate RSI from OHLC DataFrame."""
    closes = df['close'].values
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = []
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi_values.append(rsi)

        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    # Pad with NaN for initial values
    rsi_values = [np.nan] * period + rsi_values
    return np.array(rsi_values)

result = calculate_and_write_indicator(
    symbol="EURUSD",
    indicator_name="RSI_14",
    calculation_function=calculate_rsi_from_ohlc,
    timeframe="H1",
    bars=100
)

print(f"RSI calculation: {result['success']}")

# Example 3: Read existing buffer
buffer_data = read_indicator_buffer(
    symbol="EURUSD",
    indicator_name="RSI_14",
    buffer_index=0,
    timeframe="H1"
)

if buffer_data["success"]:
    data = buffer_data["data"]
    print(f"Indicator: {data['metadata']['indicator_name']}")
    print(f"Values: {len(data['data'])}")

# Example 4: Append new values (real-time update)
new_values = [72.5, 73.0, 71.8]  # New RSI values
new_timestamps = [
    "2024-01-15T14:00:00",
    "2024-01-15T15:00:00",
    "2024-01-15T16:00:00"
]

result = append_to_indicator_buffer(
    symbol="EURUSD",
    indicator_name="RSI_14",
    buffer_index=0,
    new_values=new_values,
    new_timestamps=new_timestamps,
    timeframe="H1"
)

print(f"Appended {result['values_written']} values. Total: {result['total_values']}")
```

## Notes

- This is a Python-side implementation for indicator development
- For actual MT5 indicators, use MQL5 or MT5's custom indicator API
- Files are stored in `/tmp/mt5_indicators/` by default
- JSON format stores metadata and data
- CSV format is also generated for easy import to Excel/MT5
- Buffer indices 0-7 match MT5's 8 indicator buffers

## Use Cases

1. **Indicator Development**: Test custom indicators in Python before MQL5
2. **Backtesting**: Store indicator values for strategy backtesting
3. **Real-time Updates**: Append new values as they're calculated
4. **Data Export**: Export to CSV for analysis in other tools

## Dependencies

- `read_mt5_data`: For fetching OHLC data to calculate indicators

## See Also

- `calculate_rsi`, `calculate_macd`: Indicator calculation skills
- `read_mt5_data`: Fetch market data for calculations
