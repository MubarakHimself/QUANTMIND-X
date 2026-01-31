---
name: calculate_position_size
category: trading_skills
description: Calculate optimal position size based on risk percentage, account balance, and stop loss distance
version: 1.0.0
dependencies:
  - validate_stop_loss
tags:
  - risk_management
  - position_sizing
  - money_management
---

# Calculate Position Size Skill

## Description

Calculates the optimal position size (lot size) for a trade based on account balance, risk percentage, and stop loss distance. This skill implements proper risk management principles to ensure consistent risk per trade and prevent over-leveraging.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "account_balance": {
      "type": "number",
      "description": "Current account balance in base currency",
      "minimum": 0
    },
    "risk_percent": {
      "type": "number",
      "description": "Risk percentage per trade (e.g., 1.0 = 1%)",
      "default": 1.0,
      "minimum": 0.1,
      "maximum": 10.0
    },
    "stop_loss_pips": {
      "type": "number",
      "description": "Stop loss distance in pips",
      "minimum": 1
    },
    "symbol": {
      "type": "string",
      "description": "Trading symbol (e.g., 'EURUSD', 'GBPJPY')"
    },
    "pip_value": {
      "type": "number",
      "description": "Value per pip per lot (optional, auto-calculated if not provided)"
    },
    "max_position_percent": {
      "type": "number",
      "description": "Maximum position as percentage of account (default: 30%)",
      "default": 30.0,
      "minimum": 1,
      "maximum": 100
    },
    "round_to_lot_size": {
      "type": "number",
      "description": "Round to this lot size increment (e.g., 0.01 for micro lots)",
      "default": 0.01
    }
  },
  "required": ["account_balance", "risk_percent", "stop_loss_pips", "symbol"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "lots": {
      "type": "number",
      "description": "Calculated lot size for the trade"
    },
    "risk_amount": {
      "type": "number",
      "description": "Amount of money at risk in base currency"
    },
    "risk_percent": {
      "type": "number",
      "description": "Actual risk percentage"
    },
    "pip_value_per_lot": {
      "type": "number",
      "description": "Value of 1 pip for 1 standard lot"
    },
    "max_loss_pips": {
      "type": "number",
      "description": "Maximum loss in pips if stop is hit"
    },
    "position_value": {
      "type": "number",
      "description": "Total position value (lots * contract size)"
    },
    "position_as_percent_of_account": {
      "type": "number",
      "description": "Position size as percentage of account"
    },
    "margin_required": {
      "type": "number",
      "description": "Estimated margin required for the position"
    },
    "warnings": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Any warnings about the position size"
    }
  }
}
```

## Code

```python
from typing import Dict, Any, List
import math

# Standard lot sizes and pip values for common forex pairs
FOREX_PIP_VALUES = {
    # Major pairs (quote currency = USD)
    "EURUSD": 10.0,
    "GBPUSD": 10.0,
    "AUDUSD": 10.0,
    "NZDUSD": 10.0,

    # USD as base currency
    "USDJPY": 9.09,  # Approximate, varies with exchange rate
    "USDCHF": 10.0,   # Approximate, varies with exchange rate
    "USDCAD": 10.0,   # Approximate, varies with exchange rate

    # Cross pairs (need exchange rate adjustment)
    "EURGBP": 13.51,  # Approximate, varies with GBP/USD rate
    "EURJPY": 9.09,   # Approximate, varies with USD/JPY rate
    "GBPJPY": 9.09,   # Approximate, varies with USD/JPY rate
    "EURCHF": 10.0,   # Approximate
}

# Standard contract sizes
STANDARD_LOT_UNITS = 100000
MINI_LOT_UNITS = 10000
MICRO_LOT_UNITS = 1000

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    symbol: str,
    pip_value: float = None,
    max_position_percent: float = 30.0,
    round_to_lot_size: float = 0.01
) -> Dict[str, Any]:
    """
    Calculate optimal position size based on risk management principles.

    Args:
        account_balance: Current account balance in base currency
        risk_percent: Risk percentage per trade (e.g., 1.0 = 1%)
        stop_loss_pips: Stop loss distance in pips
        symbol: Trading symbol (e.g., 'EURUSD', 'GBPJPY')
        pip_value: Value per pip per lot (auto-calculated if not provided)
        max_position_percent: Maximum position as percentage of account
        round_to_lot_size: Round to this lot size increment

    Returns:
        Dictionary containing position size details, risk amount, and warnings

    Raises:
        ValueError: If invalid parameters provided
    """
    # Validate inputs
    if account_balance <= 0:
        raise ValueError("Account balance must be greater than zero")

    if not (0.1 <= risk_percent <= 10.0):
        raise ValueError("Risk percent must be between 0.1 and 10.0")

    if stop_loss_pips <= 0:
        raise ValueError("Stop loss pips must be greater than zero")

    # Calculate risk amount in base currency
    risk_amount = account_balance * (risk_percent / 100.0)

    # Get pip value per lot
    if pip_value is None:
        pip_value_per_lot = FOREX_PIP_VALUES.get(symbol.upper(), 10.0)
    else:
        pip_value_per_lot = pip_value

    # Calculate position size based on risk
    # Formula: Lots = Risk Amount / (Stop Loss Pips * Pip Value per Lot)
    lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

    # Round to specified lot size
    lots = math.floor(lots / round_to_lot_size) * round_to_lot_size

    # Ensure minimum lot size
    if lots < round_to_lot_size:
        lots = round_to_lot_size

    # Calculate position details
    position_value = lots * STANDARD_LOT_UNITS
    position_as_percent = (position_value / account_balance) * 100 if account_balance > 0 else 0

    # Estimate margin required (assume 1:100 leverage for calculation)
    leverage = 100
    margin_required = position_value / leverage

    # Generate warnings
    warnings = []

    # Warning 1: Position too large relative to account
    if position_as_percent > max_position_percent:
        warnings.append(
            f"Position size ({position_as_percent:.1f}%) exceeds maximum "
            f"({max_position_percent}%). Consider reducing lot size."
        )

    # Warning 2: Risk percentage is high
    if risk_percent > 2.0:
        warnings.append(
            f"Risk of {risk_percent}% per trade is high. "
            f"Consider reducing to 1-2% for consistent risk management."
        )

    # Warning 3: Very tight stop loss
    if stop_loss_pips < 10:
        warnings.append(
            f"Stop loss of {stop_loss_pips} pips is very tight. "
            f"Consider widening to avoid being stopped out by noise."
        )

    # Warning 4: Position is very large (leveraged)
    if position_as_percent > 50:
        warnings.append(
            f"Position is {position_as_percent:.1f}x account size. "
            f"This represents high leverage. Consider reducing exposure."
        )

    # Warning 5: Check if symbol is recognized
    if pip_value is None and symbol.upper() not in FOREX_PIP_VALUES:
        warnings.append(
            f"Symbol '{symbol}' not in predefined list. Using default pip value of $10. "
            f"Verify pip value for accurate calculation."
        )

    return {
        "lots": round(lots, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_percent": risk_percent,
        "pip_value_per_lot": round(pip_value_per_lot, 2),
        "max_loss_pips": stop_loss_pips,
        "position_value": round(position_value, 2),
        "position_as_percent_of_account": round(position_as_percent, 2),
        "margin_required": round(margin_required, 2),
        "warnings": warnings
    }
```

## Example Usage

```python
# Example 1: Standard trade on EURUSD
result = calculate_position_size(
    account_balance=10000.0,
    risk_percent=1.0,
    stop_loss_pips=20,
    symbol="EURUSD"
)

print(f"Lots: {result['lots']}")
print(f"Risk Amount: ${result['risk_amount']:.2f}")
print(f"Position as % of Account: {result['position_as_percent_of_account']:.1f}%")

if result['warnings']:
    print("\nWarnings:")
    for warning in result['warnings']:
        print(f"  - {warning}")

# Example output:
# Lots: 0.5
# Risk Amount: $100.00
# Position as % of Account: 500.0%
#
# Warnings:
#   - Position is 500.0x account size. This represents high leverage. Consider reducing exposure.

# Example 2: Conservative trade on GBPJPY
result = calculate_position_size(
    account_balance=5000.0,
    risk_percent=0.5,
    stop_loss_pips=30,
    symbol="GBPJPY",
    max_position_percent=20.0
)

print(f"\nLots: {result['lots']}")
print(f"Risk Amount: ${result['risk_amount']:.2f}")
print(f"Margin Required: ${result['margin_required']:.2f}")

# Example 3: Custom pip value (for indices, commodities, etc.)
result = calculate_position_size(
    account_balance=25000.0,
    risk_percent=1.5,
    stop_loss_pips=50,
    symbol="XAUUSD",  # Gold
    pip_value=10.0,   # Custom pip value
    round_to_lot_size=0.01
)

print(f"\nGold Position Size: {result['lots']} lots")
print(f"Risk: ${result['risk_amount']:.2f} ({result['risk_percent']}%)")

# Use in trading system
def open_trade_with_risk_management(symbol, entry_price, stop_loss_pips):
    """
    Example function showing how to use position sizing in a trading system.
    """
    account_balance = get_account_balance()  # Your account balance function

    position = calculate_position_size(
        account_balance=account_balance,
        risk_percent=1.0,  # 1% risk per trade
        stop_loss_pips=stop_loss_pips,
        symbol=symbol
    )

    # Check for warnings
    if position['warnings']:
        log_warnings(position['warnings'])

    # Open trade with calculated lot size
    if position['lots'] > 0:
        open_trade(
            symbol=symbol,
            lots=position['lots'],
            entry_price=entry_price,
            stop_loss=entry_price - (stop_loss_pips * 0.0001)  # For 4-digit broker
        )

    return position
```

## Notes

- Standard lot (1.0) = 100,000 units of base currency
- Mini lot (0.1) = 10,000 units
- Micro lot (0.01) = 1,000 units
- Risk per trade should typically be 1-2% of account balance
- Never risk more than 5% on a single trade
- Consider account drawdown when calculating risk (use equity instead of balance if in drawdown)
- Adjust position size for correlation if trading multiple correlated pairs
- Different brokers have different lot size requirements (check minimum and maximum)
- Some brokers use different pip definitions (5-digit vs 4-digit)

## Risk Management Best Practices

1. **Never risk more than 1-2% per trade** on a single position
2. **Use consistent risk percentage** across all trades
3. **Adjust position size, not stop loss** to manage risk
4. **Consider correlation** when trading multiple positions
5. **Reduce position size** during losing streaks
6. **Increase position size gradually** as account grows (compound, don't add fixed amounts)

## Dependencies

- `validate_stop_loss`: Can be used to validate stop loss distance before calculating position size
- MetaTrader5 Python API: For getting real-time account balance and opening trades

## See Also

- `validate_stop_loss`: Validate stop loss placement
- `calculate_rsi`, `calculate_macd`: Indicators to help determine stop loss distance
