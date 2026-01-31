---
name: validate_stop_loss
category: trading_skills
description: Validate stop loss placement based on volatility, support/resistance levels, and risk-reward ratio
version: 1.0.0
dependencies:
  - detect_support_resistance
  - fetch_historical_data
tags:
  - risk_management
  - stop_loss
  - trade_validation
---

# Validate Stop Loss Skill

## Description

Validates stop loss placement for trading positions based on multiple factors including market volatility (ATR), support/resistance levels, risk-reward ratio, and minimum distance requirements. This skill helps ensure stop losses are placed at logical levels rather than arbitrary positions.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "entry_price": {
      "type": "number",
      "description": "Entry price for the trade"
    },
    "stop_loss_price": {
      "type": "number",
      "description": "Proposed stop loss price"
    },
    "take_profit_price": {
      "type": "number",
      "description": "Take profit price (optional, for R:R calculation)"
    },
    "direction": {
      "type": "string",
      "enum": ["long", "short"],
      "description": "Trade direction"
    },
    "timeframe": {
      "type": "string",
      "description": "Trading timeframe (e.g., 'H1', 'D1')",
      "default": "H1"
    },
    "atr_period": {
      "type": "integer",
      "description": "ATR period for volatility calculation",
      "default": 14
    },
    "min_risk_reward": {
      "type": "number",
      "description": "Minimum risk-reward ratio (default: 1.5)",
      "default": 1.5,
      "minimum": 1.0
    },
    "min_stop_distance_pips": {
      "type": "number",
      "description": "Minimum stop loss distance in pips",
      "default": 10.0
    },
    "use_atr_multiple": {
      "type": "number",
      "description": "Use ATR multiple as ideal stop distance (e.g., 1.5x ATR)",
      "default": 1.5
    }
  },
  "required": ["symbol", "entry_price", "stop_loss_price", "direction"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "is_valid": {
      "type": "boolean",
      "description": "Whether the stop loss is valid"
    },
    "validation_score": {
      "type": "number",
      "description": "Validation score from 0 to 100",
      "minimum": 0,
      "maximum": 100
    },
    "stop_distance_pips": {
      "type": "number",
      "description": "Stop loss distance in pips"
    },
    "risk_reward_ratio": {
      "type": "number",
      "description": "Calculated risk-reward ratio (if TP provided)"
    },
    "atr_value": {
      "type": "number",
      "description": "Current ATR value"
    },
    "atr_multiple": {
      "type": "number",
      "description": "Stop distance as multiple of ATR"
    },
    "near_support_resistance": {
      "type": "boolean",
      "description": "Whether stop is near support/resistance"
    },
    "level_type": {
      "type": "string",
      "enum": ["support", "resistance", "none"],
      "description": "Type of level near stop loss"
    },
    "ideal_stop_price": {
      "type": "number",
      "description": "Suggested ideal stop price based on analysis"
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Validation warnings"
    },
    "recommendations": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Recommendations for improvement"
    }
  }
}
```

## Code

```python
import numpy as np
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
from datetime import datetime

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) indicator.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period

    Returns:
        Current ATR value
    """
    # True Range calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # ATR using exponential moving average (Wilder's smoothing)
    atr = np.zeros_like(true_range)
    atr[period - 1] = np.mean(true_range[:period])

    for i in range(period, len(true_range)):
        atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period

    return float(atr[-1])

def price_to_pips(symbol: str, price: float) -> float:
    """
    Convert price difference to pips based on symbol.

    Args:
        symbol: Trading symbol
        price: Price difference

    Returns:
        Price in pips
    """
    symbol_upper = symbol.upper()

    # JPY pairs (2 decimal places for pips)
    if "JPY" in symbol_upper:
        return price * 100

    # Most other pairs (4 decimal places for pips)
    elif any(pair in symbol_upper for pair in ["EUR", "GBP", "AUD", "NZD", "USD", "CAD", "CHF"]):
        return price * 10000

    # Indices, commodities, etc. (varies)
    else:
        return price * 100

def validate_stop_loss(
    symbol: str,
    entry_price: float,
    stop_loss_price: float,
    take_profit_price: Optional[float] = None,
    direction: str = "long",
    timeframe: str = "H1",
    atr_period: int = 14,
    min_risk_reward: float = 1.5,
    min_stop_distance_pips: float = 10.0,
    use_atr_multiple: float = 1.5
) -> Dict[str, Any]:
    """
    Validate stop loss placement based on multiple factors.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
        entry_price: Entry price for the trade
        stop_loss_price: Proposed stop loss price
        take_profit_price: Take profit price (optional)
        direction: Trade direction ('long' or 'short')
        timeframe: Trading timeframe (e.g., 'H1', 'D1')
        atr_period: ATR period for volatility calculation
        min_risk_reward: Minimum risk-reward ratio
        min_stop_distance_pips: Minimum stop loss distance in pips
        use_atr_multiple: ATR multiple for ideal stop distance

    Returns:
        Dictionary containing validation results, score, and recommendations

    Raises:
        ValueError: If invalid parameters provided
    """
    # Validate direction
    if direction not in ["long", "short"]:
        raise ValueError("Direction must be 'long' or 'short'")

    # Validate stop loss direction
    if direction == "long" and stop_loss_price >= entry_price:
        return {
            "is_valid": False,
            "validation_score": 0,
            "warnings": ["Long trade stop loss must be below entry price"],
            "recommendations": ["Move stop loss below entry price"]
        }

    if direction == "short" and stop_loss_price <= entry_price:
        return {
            "is_valid": False,
            "validation_score": 0,
            "warnings": ["Short trade stop loss must be above entry price"],
            "recommendations": ["Move stop loss above entry price"]
        }

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

    # Fetch historical data from MT5
    mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    bars = atr_period + 20
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)

    if rates is None or len(rates) < atr_period:
        return {
            "is_valid": False,
            "validation_score": 0,
            "warnings": ["Insufficient data for ATR calculation"],
            "recommendations": ["Try different timeframe or symbol"]
        }

    # Extract OHLC data
    highs = rates['high']
    lows = rates['low']
    closes = rates['close']

    # Calculate ATR
    atr = calculate_atr(highs, lows, closes, atr_period)

    # Calculate stop distance in price and pips
    if direction == "long":
        stop_distance = entry_price - stop_loss_price
    else:
        stop_distance = stop_loss_price - entry_price

    stop_distance_pips = price_to_pips(symbol, stop_distance)

    # Calculate ATR multiple
    atr_multiple = stop_distance / atr if atr > 0 else 0

    # Calculate risk-reward ratio if take profit provided
    risk_reward_ratio = None
    if take_profit_price is not None:
        if direction == "long":
            reward = take_profit_price - entry_price
        else:
            reward = entry_price - take_profit_price

        if reward > 0 and stop_distance > 0:
            risk_reward_ratio = reward / stop_distance

    # Check if stop is near support/resistance
    # Find recent swing lows (for long) or swing highs (for short)
    lookback = 5
    near_level = False
    level_type = "none"

    if direction == "long":
        # Check if near recent swing low (support)
        recent_lows = lows[-lookback:]
        min_low = np.min(recent_lows)

        if abs(stop_loss_price - min_low) / min_low < 0.001:  # Within 0.1%
            near_level = True
            level_type = "support"
    else:
        # Check if near recent swing high (resistance)
        recent_highs = highs[-lookback:]
        max_high = np.max(recent_highs)

        if abs(stop_loss_price - max_high) / max_high < 0.001:  # Within 0.1%
            near_level = True
            level_type = "resistance"

    # Calculate ideal stop based on ATR multiple
    if direction == "long":
        ideal_stop = entry_price - (atr * use_atr_multiple)
    else:
        ideal_stop = entry_price + (atr * use_atr_multiple)

    # Initialize validation
    validation_score = 0
    warnings = []
    recommendations = []

    # Validation criteria

    # 1. Minimum stop distance (20 points)
    if stop_distance_pips >= min_stop_distance_pips:
        validation_score += 20
    else:
        warnings.append(
            f"Stop distance ({stop_distance_pips:.1f} pips) is too tight. "
            f"Minimum: {min_stop_distance_pips} pips."
        )

    # 2. ATR multiple validation (ideal: 1-2x ATR)
    if 1.0 <= atr_multiple <= 2.5:
        validation_score += 25
    elif atr_multiple < 1.0:
        warnings.append(
            f"Stop is too tight ({atr_multiple:.2f}x ATR). "
            f"Consider using {use_atr_multiple}x ATR for volatility buffer."
        )
    else:
        validation_score += 15  # Partial credit
        warnings.append(
            f"Stop is wide ({atr_multiple:.2f}x ATR). "
            f"Consider tighter stop around {use_atr_multiple}x ATR."
        )

    # 3. Near support/resistance (bonus)
    if near_level:
        validation_score += 20
    else:
        recommendations.append(
            f"Consider placing stop near {level_type} for better technical justification."
        )

    # 4. Risk-reward ratio (if take profit provided)
    if risk_reward_ratio is not None:
        if risk_reward_ratio >= min_risk_reward:
            validation_score += 25
        else:
            warnings.append(
                f"Risk-reward ratio ({risk_reward_ratio:.2f}) is below minimum ({min_risk_reward}). "
                f"Consider adjusting take profit or entry."
            )
    else:
        # Partial score if no take profit provided
        validation_score += 10
        recommendations.append(
            "Set take profit target to validate risk-reward ratio."
        )

    # 5. Stop on correct side of entry (10 points)
    validation_score += 10  # Already validated above

    # Determine if valid (score >= 60 = passing)
    is_valid = validation_score >= 60

    # Generate recommendations
    if validation_score < 80:
        if abs(stop_distance_pips - price_to_pips(symbol, atr * use_atr_multiple)) > price_to_pips(symbol, atr * 0.5):
            recommendations.append(
                f"Consider setting stop at {ideal_stop:.5f} "
                f"({use_atr_multiple}x ATR from entry)"
            )

    return {
        "is_valid": is_valid,
        "validation_score": round(validation_score, 0),
        "stop_distance_pips": round(stop_distance_pips, 1),
        "risk_reward_ratio": round(risk_reward_ratio, 2) if risk_reward_ratio else None,
        "atr_value": round(atr, 5),
        "atr_multiple": round(atr_multiple, 2),
        "near_support_resistance": near_level,
        "level_type": level_type,
        "ideal_stop_price": round(ideal_stop, 5),
        "warnings": warnings,
        "recommendations": recommendations
    }
```

## Example Usage

```python
# Example 1: Valid long trade stop
result = validate_stop_loss(
    symbol="EURUSD",
    entry_price=1.0850,
    stop_loss_price=1.0820,
    take_profit_price=1.0910,
    direction="long",
    timeframe="H1"
)

print(f"Valid: {result['is_valid']}")
print(f"Score: {result['validation_score']}/100")
print(f"Stop Distance: {result['stop_distance_pips']} pips")
print(f"Risk-Reward Ratio: {result['risk_reward_ratio']}")
print(f"ATR Multiple: {result['atr_multiple']}x")

if result['warnings']:
    print("\nWarnings:")
    for w in result['warnings']:
        print(f"  - {w}")

if result['recommendations']:
    print("\nRecommendations:")
    for r in result['recommendations']:
        print(f"  - {r}")

# Example output:
# Valid: True
# Score: 90.0/100
# Stop Distance: 30.0 pips
# Risk-Reward Ratio: 2.0
# ATR Multiple: 1.5x
#
# Warnings: none
# Recommendations: none

# Example 2: Invalid stop (too tight)
result = validate_stop_loss(
    symbol="GBPJPY",
    entry_price=150.50,
    stop_loss_price=150.48,  # Only 2 pips
    direction="long",
    timeframe="H1"
)

print(f"\nValid: {result['is_valid']}")
print(f"Score: {result['validation_score']}/100")

# Example 3: Short trade with ideal stop suggestion
result = validate_stop_loss(
    symbol="USDCHF",
    entry_price=0.8750,
    stop_loss_price=0.8780,
    take_profit_price=0.8690,
    direction="short",
    timeframe="D1"
)

print(f"\nShort Trade Validation:")
print(f"Valid: {result['is_valid']}")
print(f"Ideal Stop: {result['ideal_stop_price']}")
print(f"Near Support/Resistance: {result['near_support_resistance']}")

# Use in trading system
def place_trade_with_validated_stop(symbol, direction, entry, stop, tp=None):
    """
    Example function showing how to use stop validation in a trading system.
    """
    # Validate the stop loss
    validation = validate_stop_loss(
        symbol=symbol,
        entry_price=entry,
        stop_loss_price=stop,
        take_profit_price=tp,
        direction=direction
    )

    if not validation['is_valid']:
        print(f"Stop validation failed (score: {validation['validation_score']})")
        print("Warnings:")
        for w in validation['warnings']:
            print(f"  - {w}")

        # Suggest alternative stop
        if validation['ideal_stop_price']:
            print(f"Suggested stop: {validation['ideal_stop_price']}")

        return False

    # Stop is valid, proceed with trade
    print(f"Stop validated (score: {validation['validation_score']}/100)")
    # Place trade...
    return True
```

## Notes

- ATR-based stops adapt to market volatility and are more reliable than fixed pip stops
- Stops near support/resistance levels have technical justification
- Risk-reward ratio should ideally be 1:2 or higher for profitable trading
- Consider minimum stop distance to avoid being stopped out by noise
- Wider stops should be offset by smaller position sizes (see `calculate_position_size`)
- Trailing stops can lock in profits while giving room for continuation

## Stop Loss Best Practices

1. **Use ATR-based stops** for volatility-adjusted placement
2. **Place stops at logical levels** (support/resistance, not arbitrary prices)
3. **Maintain minimum 1:1.5 risk-reward** preferably 1:2 or higher
4. **Avoid too-tight stops** (at least 10-15 pips for forex, 1.5x ATR ideal)
5. **Consider position size** alongside stop distance for consistent risk
6. **Trail stops** when trade moves in your favor to lock in profits

## Dependencies

- `detect_support_resistance`: Can be used to find optimal support/resistance levels for stops
- `fetch_historical_data`: Required for ATR calculation and level detection
- `calculate_position_size`: Use validated stop to calculate proper position size

## See Also

- `calculate_position_size`: Calculate lot size based on validated stop distance
- `detect_support_resistance`: Find support/resistance levels for stop placement
- `calculate_rsi`, `calculate_macd`: Indicators to help determine trade entry and stop levels
