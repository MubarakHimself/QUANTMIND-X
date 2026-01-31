---
name: detect_support_resistance
category: trading_skills
description: Detect support and resistance levels using price action analysis and pivot point calculation
version: 1.0.0
dependencies:
  - fetch_historical_data
tags:
  - price_action
  - levels
  - technical_analysis
---

# Detect Support/Resistance Skill

## Description

Detects key support and resistance levels using historical price data. This skill implements multiple methods including pivot point analysis, local extrema detection, and volume profile analysis to identify significant price levels where price action may reverse or consolidate.

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
      "description": "Timeframe for analysis (e.g., 'H1', 'D1')",
      "default": "D1"
    },
    "bars": {
      "type": "integer",
      "description": "Number of bars to analyze",
      "default": 100,
      "minimum": 50
    },
    "lookback": {
      "type": "integer",
      "description": "Lookback period for local extrema (default: 5)",
      "default": 5,
      "minimum": 3,
      "maximum": 20
    },
    "tolerance": {
      "type": "number",
      "description": "Price tolerance for grouping levels (pips or %)",
      "default": 0.001
    },
    "min_touches": {
      "type": "integer",
      "description": "Minimum touches to consider a level significant",
      "default": 2,
      "minimum": 1
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
    "support_levels": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "price": {
            "type": "number"
          },
          "touches": {
            "type": "integer"
          },
          "strength": {
            "type": "string",
            "enum": ["weak", "moderate", "strong"]
          },
          "last_tested": {
            "type": "string"
          }
        }
      },
      "description": "List of support levels sorted by strength"
    },
    "resistance_levels": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "price": {
            "type": "number"
          },
          "touches": {
            "type": "integer"
          },
          "strength": {
            "type": "string",
            "enum": ["weak", "moderate", "strong"]
          },
          "last_tested": {
            "type": "string"
          }
        }
      },
      "description": "List of resistance levels sorted by strength"
    },
    "nearest_support": {
      "type": "number",
      "description": "Nearest support level below current price"
    },
    "nearest_resistance": {
      "type": "number",
      "description": "Nearest resistance level above current price"
    },
    "current_price": {
      "type": "number",
      "description": "Current market price"
    },
    "pivot_points": {
      "type": "object",
      "properties": {
        "pivot": {
          "type": "number"
        },
        "r1": {
          "type": "number"
        },
        "r2": {
          "type": "number"
        },
        "r3": {
          "type": "number"
        },
        "s1": {
          "type": "number"
        },
        "s2": {
          "type": "number"
        },
        "s3": {
          "type": "number"
        }
      },
      "description": "Classic pivot point levels"
    }
  }
}
```

## Code

```python
import numpy as np
from typing import Dict, Any, List, Tuple
import MetaTrader5 as mt5
from datetime import datetime
from collections import defaultdict

def detect_support_resistance(
    symbol: str,
    timeframe: str = "D1",
    bars: int = 100,
    lookback: int = 5,
    tolerance: float = 0.001,
    min_touches: int = 2
) -> Dict[str, Any]:
    """
    Detect support and resistance levels using multiple methods.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe for analysis (e.g., 'H1', 'D1')
        bars: Number of bars to analyze
        lookback: Lookback period for local extrema
        tolerance: Price tolerance for grouping levels
        min_touches: Minimum touches to consider level significant

    Returns:
        Dictionary containing support/resistance levels, nearest levels,
        current price, and pivot points

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

    if rates is None or len(rates) < lookback * 2:
        raise ValueError(f"Insufficient data for analysis. Need at least {lookback * 2} bars")

    # Extract OHLC data
    highs = rates['high']
    lows = rates['low']
    closes = rates['close']
    times = rates['time']

    current_price = float(closes[-1])

    # Calculate pivot points (classic method)
    pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
    r1 = 2 * pivot - lows[-1]
    s1 = 2 * pivot - highs[-1]
    r2 = pivot + (highs[-1] - lows[-1])
    s2 = pivot - (highs[-1] - lows[-1])
    r3 = highs[-1] + 2 * (pivot - lows[-1])
    s3 = lows[-1] - 2 * (highs[-1] - pivot)

    pivot_points = {
        "pivot": float(pivot),
        "r1": float(r1),
        "r2": float(r2),
        "r3": float(r3),
        "s1": float(s1),
        "s2": float(s2),
        "s3": float(s3),
    }

    # Detect local extrema (swing highs and lows)
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(highs) - lookback):
        # Check for swing high
        is_swing_high = all(highs[i] >= highs[i - j] for j in range(1, lookback + 1))
        is_swing_high = is_swing_high and all(highs[i] >= highs[i + j] for j in range(1, lookback + 1))

        if is_swing_high:
            swing_highs.append((i, float(highs[i]), times[i]))

        # Check for swing low
        is_swing_low = all(lows[i] <= lows[i - j] for j in range(1, lookback + 1))
        is_swing_low = is_swing_low and all(lows[i] <= lows[i + j] for j in range(1, lookback + 1))

        if is_swing_low:
            swing_lows.append((i, float(lows[i]), times[i]))

    # Group nearby levels using tolerance
    def group_levels(levels: List[Tuple[int, float, int]]) -> Dict[float, List[Tuple[int, int]]]:
        """Group levels that are within tolerance of each other."""
        groups = defaultdict(list)

        for idx, price, timestamp in sorted(levels, key=lambda x: x[1]):
            # Find existing group within tolerance
            found_group = False
            for group_price in groups:
                if abs(price - group_price) / group_price <= tolerance:
                    groups[group_price].append((idx, timestamp))
                    found_group = True
                    break

            if not found_group:
                groups[price].append((idx, timestamp))

        return dict(groups)

    # Group resistance levels (swing highs)
    resistance_groups = group_levels(swing_highs)

    # Group support levels (swing lows)
    support_groups = group_levels(swing_lows)

    # Build level details
    def build_level_details(groups: Dict[float, List], is_resistance: bool) -> List[Dict[str, Any]]:
        """Build level details with strength and touch information."""
        levels = []

        for price, touches in groups.items():
            if len(touches) >= min_touches:
                # Determine strength based on number of touches
                if len(touches) >= 4:
                    strength = "strong"
                elif len(touches) >= 3:
                    strength = "moderate"
                else:
                    strength = "weak"

                # Get most recent test
                last_tested_idx = max(t[0] for t in touches)
                last_tested = datetime.fromtimestamp(times[last_tested_idx]).isoformat()

                levels.append({
                    "price": round(price, 5),
                    "touches": len(touches),
                    "strength": strength,
                    "last_tested": last_tested
                })

        # Sort by strength (strongest first) then by touches
        strength_order = {"strong": 0, "moderate": 1, "weak": 2}
        levels.sort(key=lambda x: (strength_order[x["strength"]], -x["touches"]))

        return levels

    resistance_levels = build_level_details(resistance_groups, True)
    support_levels = build_level_details(support_groups, False)

    # Find nearest support and resistance to current price
    nearest_support = None
    nearest_resistance = None

    if support_levels:
        # Find nearest support below current price
        supports_below = [s["price"] for s in support_levels if s["price"] < current_price]
        if supports_below:
            nearest_support = max(supports_below)

    if resistance_levels:
        # Find nearest resistance above current price
        resistances_above = [r["price"] for r in resistance_levels if r["price"] > current_price]
        if resistances_above:
            nearest_resistance = min(resistances_above)

    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "current_price": current_price,
        "pivot_points": pivot_points
    }
```

## Example Usage

```python
# Detect support and resistance for EURUSD
result = detect_support_resistance(
    symbol="EURUSD",
    timeframe="D1",
    bars=100,
    lookback=5,
    tolerance=0.001,
    min_touches=2
)

print(f"Current Price: {result['current_price']}")
print(f"Nearest Support: {result['nearest_support']}")
print(f"Nearest Resistance: {result['nearest_resistance']}")

print("\nSupport Levels:")
for level in result['support_levels'][:3]:
    print(f"  {level['price']} - Touches: {level['touches']}, Strength: {level['strength']}")

print("\nResistance Levels:")
for level in result['resistance_levels'][:3]:
    print(f"  {level['price']} - Touches: {level['touches']}, Strength: {level['strength']}")

print("\nPivot Points:")
pp = result['pivot_points']
print(f"  Pivot: {pp['pivot']}")
print(f"  R1: {pp['r1']}, R2: {pp['r2']}, R3: {pp['r3']}")
print(f"  S1: {pp['s1']}, S2: {pp['s2']}, S3: {pp['s3']}")

# Example output:
# Current Price: 1.0850
# Nearest Support: 1.0820
# Nearest Resistance: 1.0880
#
# Support Levels:
#   1.0820 - Touches: 3, Strength: moderate
#   1.0750 - Touches: 2, Strength: weak
#
# Resistance Levels:
#   1.0880 - Touches: 4, Strength: strong
#   1.0920 - Touches: 2, Strength: weak

# Use in trading logic
if result['nearest_support']:
    distance_to_support = result['current_price'] - result['nearest_support']
    print(f"Distance to support: {distance_to_support:.5f}")

if result['nearest_resistance']:
    distance_to_resistance = result['nearest_resistance'] - result['current_price']
    print(f"Distance to resistance: {distance_to_resistance:.5f}")

# Look for setups near key levels
if distance_to_support < 0.0010:
    print("Price near support - watch for bullish reversal")
elif distance_to_resistance < 0.0010:
    print("Price near resistance - watch for bearish reversal")
```

## Notes

- Support and resistance levels are dynamic and should be recalculated regularly
- Stronger levels (more touches) are more significant and likely to hold
- Combine with other indicators for confirmation
- Pivot points provide additional reference levels for intraday trading
- Levels that align across multiple timeframes are more significant
- Broken resistance can become support and vice versa

## Dependencies

- `fetch_historical_data`: Required to get OHLC data for analysis
- MetaTrader5 Python API: Must be connected to MT5 terminal
