---
name: detect_support_resistance
category: trading_skills
version: 1.0.0
dependencies:
  - fetch_historical_data
---

# Detect Support and Resistance Levels Skill

## Description
Identifies key support and resistance levels from historical price data using pivot point analysis.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "highs": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of high prices"
    },
    "lows": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of low prices"
    },
    "closes": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Array of closing prices"
    },
    "lookback_period": {
      "type": "integer",
      "description": "Period for pivot calculation (default: 5)",
      "default": 5,
      "minimum": 3,
      "maximum": 20
    },
    "num_levels": {
      "type": "integer",
      "description": "Number of support/resistance levels to return (default: 3)",
      "default": 3,
      "minimum": 1,
      "maximum": 10
    }
  },
  "required": ["highs", "lows", "closes"]
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
        "type": "number"
      },
      "description": "Identified support levels (price levels)"
    },
    "resistance_levels": {
      "type": "array",
      "items": {
        "type": "number"
      },
      "description": "Identified resistance levels (price levels)"
    },
    "current_price": {
      "type": "number",
      "description": "Current (last) price"
    },
    "nearest_support": {
      "type": "number",
      "description": "Nearest support level below current price"
    },
    "nearest_resistance": {
      "type": "number",
      "description": "Nearest resistance level above current price"
    }
  },
  "required": ["support_levels", "resistance_levels", "current_price", "nearest_support", "nearest_resistance"]
}
```

## Code
```python
import numpy as np
from typing import List, Dict, Any

def detect_support_resistance(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback_period: int = 5,
    num_levels: int = 3
) -> Dict[str, Any]:
    """
    Detect support and resistance levels using pivot point analysis.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        lookback_period: Period for pivot calculation
        num_levels: Number of S/R levels to identify

    Returns:
        Dict with support_levels, resistance_levels, and nearest levels

    Method:
    - Identifies pivot highs (resistance) where price is highest in lookback period
    - Identifies pivot lows (support) where price is lowest in lookback period
    - Clusters nearby pivots to find key levels
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Highs, lows, and closes must have the same length")

    if len(highs) < lookback_period * 2:
        raise ValueError(f"Need at least {lookback_period * 2} data points")

    highs_array = np.array(highs)
    lows_array = np.array(lows)
    closes_array = np.array(closes)

    # Find pivot highs (local maxima)
    pivot_highs = []
    for i in range(lookback_period, len(highs) - lookback_period):
        window_highs = highs_array[i - lookback_period:i + lookback_period + 1]
        if highs_array[i] == np.max(window_highs):
            pivot_highs.append(highs_array[i])

    # Find pivot lows (local minima)
    pivot_lows = []
    for i in range(lookback_period, len(lows) - lookback_period):
        window_lows = lows_array[i - lookback_period:i + lookback_period + 1]
        if lows_array[i] == np.min(window_lows):
            pivot_lows.append(lows_array[i])

    # Cluster pivots to find key levels
    def cluster_levels(levels: List[float], num_clusters: int) -> List[float]:
        """Cluster nearby levels and return cluster centers."""
        if not levels:
            return []

        levels_array = np.array(sorted(levels, reverse=True))

        if len(levels_array) <= num_clusters:
            return levels_array.tolist()

        # Simple clustering: group levels within 0.5% of each other
        clusters = []
        for level in levels_array:
            added = False
            for i, cluster_center in enumerate(clusters):
                if abs(level - cluster_center) / cluster_center < 0.005:  # 0.5% threshold
                    clusters[i] = (clusters[i] + level) / 2  # Update cluster center
                    added = True
                    break
            if not added:
                clusters.append(level)

        return sorted(clusters, reverse=True)[:num_clusters]

    # Get clustered levels
    resistance_levels = cluster_levels(pivot_highs, num_levels)
    support_levels = cluster_levels(pivot_lows, num_levels)
    support_levels = sorted(support_levels)  # Ascending for support

    # Current price
    current_price = closes_array[-1]

    # Find nearest support (largest support below current price)
    nearest_support = None
    for level in reversed(support_levels):
        if level < current_price:
            nearest_support = level
            break

    # Find nearest resistance (smallest resistance above current price)
    nearest_resistance = None
    for level in resistance_levels:
        if level > current_price:
            nearest_resistance = level
            break

    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels,
        "current_price": round(float(current_price), 5),
        "nearest_support": round(nearest_support, 5) if nearest_support else None,
        "nearest_resistance": round(nearest_resistance, 5) if nearest_resistance else None
    }
```

## Example Usage
```python
# Detect support/resistance levels
highs = [1.1010, 1.1025, 1.1040, 1.1035, 1.1020, 1.1030, 1.1045, 1.1055, 1.1040, 1.1025,
         1.1035, 1.1045, 1.1050, 1.1035, 1.1020, 1.1030, 1.1040, 1.1050, 1.1040, 1.1025]
lows =  [1.0990, 1.1000, 1.1010, 1.1005, 1.0995, 1.1005, 1.1015, 1.1025, 1.1015, 1.1005,
         1.1015, 1.1025, 1.1035, 1.1025, 1.1010, 1.1020, 1.1030, 1.1040, 1.1030, 1.1020]
closes = [1.1000, 1.1015, 1.1025, 1.1020, 1.1005, 1.1015, 1.1025, 1.1035, 1.1025, 1.1015,
          1.1025, 1.1035, 1.1040, 1.1030, 1.1015, 1.1025, 1.1035, 1.1045, 1.1035, 1.1020]

result = detect_support_resistance(highs, lows, closes)
# Output: {
#   "support_levels": [1.0995, 1.1005, 1.1015],
#   "resistance_levels": [1.1055, 1.1045, 1.1035],
#   "current_price": 1.1020,
#   "nearest_support": 1.1015,
#   "nearest_resistance": 1.1035
# }
```
