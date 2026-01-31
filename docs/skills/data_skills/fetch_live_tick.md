---
name: fetch_live_tick
category: data_skills
description: Fetch real-time tick data from MT5 for live trading and monitoring
version: 1.0.0
dependencies: []
tags:
  - data
  - realtime
  - tick
---

# Fetch Live Tick Skill

## Description

Fetches real-time tick data from MetaTrader 5 including current bid/ask prices, spread, and volume information. Supports streaming ticks for continuous monitoring and single tick updates for on-demand price checks.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "mode": {
      "type": "string",
      "enum": ["single", "stream"],
      "description": "Fetch mode: single tick or continuous stream"
    },
    "duration": {
      "type": "integer",
      "description": "Streaming duration in seconds (for stream mode)",
      "minimum": 1,
      "maximum": 3600
    },
    "callback": {
      "type": "string",
      "description": "Callback function name for streaming (optional)"
    },
    "include_volume": {
      "type": "boolean",
      "description": "Include volume data (default: true)"
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
      "description": "Whether tick fetch was successful"
    },
    "symbol": {
      "type": "string",
      "description": "Symbol fetched"
    },
    "mode": {
      "type": "string",
      "description": "Fetch mode used"
    },
    "tick": {
      "type": "object",
      "description": "Tick data (for single mode)"
    },
    "ticks": {
      "type": "array",
      "description": "Array of ticks (for stream mode)"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp of tick"
    },
    "statistics": {
      "type": "object",
      "properties": {
        "avg_spread": {
          "type": "number"
        },
        "min_spread": {
          "type": "number"
        },
        "max_spread": {
          "type": "number"
        },
        "tick_count": {
          "type": "integer"
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
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import time


def fetch_live_tick(
    symbol: str,
    mode: str = "single",
    duration: int = 60,
    callback: Optional[str] = None,
    include_volume: bool = True
) -> Dict[str, Any]:
    """
    Fetch real-time tick data from MetaTrader 5.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
        mode: Fetch mode ('single' or 'stream')
        duration: Streaming duration in seconds (for stream mode)
        callback: Callback function name for streaming
        include_volume: Include volume data

    Returns:
        Dictionary containing tick data and statistics
    """
    # Initialize MT5
    if not mt5.initialize():
        return {
            "success": False,
            "error": f"MT5 initialization failed: {mt5.last_error()}"
        }

    try:
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {
                "success": False,
                "error": f"Symbol {symbol} not found"
            }

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return {
                    "success": False,
                    "error": f"Could not select symbol {symbol}"
                }

        if mode == "single":
            return _fetch_single_tick(symbol, include_volume)
        elif mode == "stream":
            return _stream_ticks(symbol, duration, callback, include_volume)
        else:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Tick fetch error: {str(e)}"
        }

    finally:
        # Only shutdown for single mode, keep connection for stream
        if mode == "single":
            mt5.shutdown()


def _fetch_single_tick(symbol: str, include_volume: bool) -> Dict[str, Any]:
    """Fetch a single tick update."""
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        return {
            "success": False,
            "error": f"Failed to get tick for {symbol}"
        }

    tick_data = {
        "symbol": symbol,
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": float(tick.ask - tick.bid),
        "spread_points": int((tick.ask - tick.bid) / mt5.symbol_info(symbol).point),
        "time": datetime.fromtimestamp(tick.time).isoformat(),
        "time_msc": tick.time_msc,
        "flags": tick.flags,
        "volume": float(tick.volume) if include_volume else None,
        "session_volume": float(tick.session_volume) if include_volume else None
    }

    # Calculate mid price
    tick_data["mid"] = (tick_data["bid"] + tick_data["ask"]) / 2

    # Calculate pip value
    tick_data["pip_value"] = (tick_data["ask"] - tick_data["bid"]) * 10000

    # Add tick direction
    if tick.volume > 0:
        tick_data["tick_direction"] = "up" if tick.flags & 64 else "down"
    else:
        tick_data["tick_direction"] = "unknown"

    return {
        "success": True,
        "symbol": symbol,
        "mode": "single",
        "tick": tick_data,
        "timestamp": datetime.now().isoformat()
    }


def _stream_ticks(
    symbol: str,
    duration: int,
    callback: Optional[str],
    include_volume: bool
) -> Dict[str, Any]:
    """Stream ticks for specified duration."""
    ticks = []
    spreads = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            tick = mt5.symbol_info_tick(symbol)

            if tick is not None:
                tick_data = {
                    "symbol": symbol,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "spread": float(tick.ask - tick.bid),
                    "time": datetime.fromtimestamp(tick.time).isoformat(),
                    "volume": float(tick.volume) if include_volume else None
                }

                ticks.append(tick_data)
                spreads.append(tick_data["spread"])

                # Call callback function if provided
                if callback:
                    # In a real implementation, you would call the callback here
                    # For now, we'll just note it
                    tick_data["callback"] = callback

            # Small delay to avoid excessive CPU usage
            time.sleep(0.1)

        # Calculate statistics
        avg_spread = sum(spreads) / len(spreads) if spreads else 0
        min_spread = min(spreads) if spreads else 0
        max_spread = max(spreads) if spreads else 0

        statistics = {
            "avg_spread": round(avg_spread, 5),
            "min_spread": round(min_spread, 5),
            "max_spread": round(max_spread, 5),
            "tick_count": len(ticks)
        }

        return {
            "success": True,
            "symbol": symbol,
            "mode": "stream",
            "ticks": ticks,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Streaming error: {str(e)}"
        }

    finally:
        mt5.shutdown()


def get_tick_stream_generator(symbol: str, include_volume: bool = True):
    """
    Generator function for continuous tick streaming.

    Args:
        symbol: Trading symbol
        include_volume: Include volume data

    Yields:
        Tick data dictionaries
    """
    if not mt5.initialize():
        raise ConnectionError("MT5 initialization failed")

    try:
        while True:
            tick = mt5.symbol_info_tick(symbol)

            if tick is not None:
                tick_data = {
                    "symbol": symbol,
                    "bid": float(tick.bid),
                    "ask": float(tick.ask),
                    "spread": float(tick.ask - tick.bid),
                    "time": datetime.fromtimestamp(tick.time).isoformat(),
                    "volume": float(tick.volume) if include_volume else None
                }

                yield tick_data

            time.sleep(0.1)

    finally:
        mt5.shutdown()


def subscribe_to_ticks(
    symbols: List[str],
    callback: Callable[[Dict], None],
    include_volume: bool = True
) -> None:
    """
    Subscribe to tick updates for multiple symbols.

    Args:
        symbols: List of trading symbols
        callback: Function to call for each tick update
        include_volume: Include volume data
    """
    if not mt5.initialize():
        raise ConnectionError("MT5 initialization failed")

    try:
        while True:
            for symbol in symbols:
                tick = mt5.symbol_info_tick(symbol)

                if tick is not None:
                    tick_data = {
                        "symbol": symbol,
                        "bid": float(tick.bid),
                        "ask": float(tick.ask),
                        "spread": float(tick.ask - tick.bid),
                        "time": datetime.fromtimestamp(tick.time).isoformat(),
                        "volume": float(tick.volume) if include_volume else None
                    }

                    callback(tick_data)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping tick subscription...")
    finally:
        mt5.shutdown()
```

## Example Usage

```python
# Example 1: Get single tick
result = fetch_live_tick(symbol="EURUSD", mode="single")

if result["success"]:
    tick = result["tick"]
    print(f"Symbol: {tick['symbol']}")
    print(f"Bid: {tick['bid']}")
    print(f"Ask: {tick['ask']}")
    print(f"Spread: {tick['spread']} ({tick['spread_points']} points)")
    print(f"Mid: {tick['mid']}")
    print(f"Time: {tick['time']}")

# Example 2: Stream ticks for 30 seconds
result = fetch_live_tick(
    symbol="GBPUSD",
    mode="stream",
    duration=30
)

if result["success"]:
    stats = result["statistics"]
    print(f"Received {stats['tick_count']} ticks")
    print(f"Avg spread: {stats['avg_spread']}")
    print(f"Min/Max spread: {stats['min_spread']} / {stats['max_spread']}")

# Example 3: Use generator for continuous streaming
try:
    for tick in get_tick_stream_generator("USDJPY"):
        print(f"{tick['time']}: {tick['bid']} / {tick['ask']}")

        # Your trading logic here
        if tick["spread"] > 0.002:  # Wide spread alert
            print("Warning: Wide spread detected")

        # Break after some condition
        # if some_condition:
        #     break

except KeyboardInterrupt:
    print("\nStopped streaming")

# Example 4: Subscribe to multiple symbols
def on_tick_update(tick_data):
    """Callback function for tick updates."""
    print(f"{tick_data['symbol']}: {tick_data['bid']} / {tick['ask']}")

subscribe_to_ticks(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    callback=on_tick_update
)

# Example 5: Check spread before trading
result = fetch_live_tick(symbol="EURUSD", mode="single")

if result["success"]:
    tick = result["tick"]
    max_spread = 0.0002  # 2 pips

    if tick["spread"] <= max_spread:
        print("Spread acceptable for trading")
        # Execute trade
    else:
        print(f"Spread too wide: {tick['spread']}")

# Example 6: Get ticks with volume analysis
result = fetch_live_tick(
    symbol="XAUUSD",  # Gold
    mode="single",
    include_volume=True
)

if result["success"]:
    tick = result["tick"]
    if tick["volume"] > 0:
        print(f"Last trade volume: {tick['volume']} lots")
        print(f"Session volume: {tick['session_volume']} lots")
```

## Notes

- MT5 terminal must be running and connected to broker
- Tick streaming consumes CPU and network resources
- Spread can vary significantly during news events
- Volume may be zero for some symbols (depends on broker)
- Time precision: tick.time_msc provides millisecond precision

## Tick Data Fields

| Field | Description |
|-------|-------------|
| bid | Current bid price |
| ask | Current ask price |
| spread | Difference between ask and bid |
| spread_points | Spread in points (depends on symbol digits) |
| mid | Mid price (bid + ask) / 2 |
| pip_value | Spread in pips (standardized) |
| time | Tick timestamp (seconds) |
| time_msc | Tick timestamp (milliseconds) |
| volume | Last trade volume |
| session_volume | Total session volume |
| flags | Tick flags (direction, etc.) |

## Use Cases

1. **Price Monitoring**: Real-time price tracking
2. **Spread Analysis**: Monitor spread for entry timing
3. **Volume Analysis**: Detect unusual trading activity
4. **Arbitrage**: Compare prices across brokers
5. **News Trading**: Monitor fast price movements

## Dependencies

- MetaTrader5 Python API
- No skill dependencies (lowest-level data access)

## See Also

- `fetch_historical_data`: Get historical OHLCV data
- `resample_timeframe`: Convert tick data to bars
- `read_mt5_data`: General MT5 data reading
