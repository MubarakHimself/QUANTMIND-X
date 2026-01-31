---
name: calculate_position_size
category: trading_skills
version: 1.0.0
dependencies: []
---

# Calculate Position Size Skill

## Description
Calculates optimal position size in lots based on risk percentage, account balance, and stop loss distance.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "account_balance": {
      "type": "number",
      "description": "Current account balance",
      "minimum": 0
    },
    "risk_percent": {
      "type": "number",
      "description": "Risk percentage per trade (e.g., 1.0 for 1%)",
      "default": 1.0,
      "minimum": 0.1,
      "maximum": 10.0
    },
    "stop_loss_pips": {
      "type": "number",
      "description": "Stop loss distance in pips",
      "minimum": 1
    },
    "pip_value": {
      "type": "number",
      "description": "Value per pip in base currency (default: 10 for standard lot)",
      "default": 10.0
    }
  },
  "required": ["account_balance", "risk_percent", "stop_loss_pips"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "position_size_lots": {
      "type": "number",
      "description": "Position size in lots"
    },
    "risk_amount": {
      "type": "number",
      "description": "Risk amount in base currency"
    },
    "max_loss_pips": {
      "type": "number",
      "description": "Maximum loss in pips if stop loss is hit"
    }
  },
  "required": ["position_size_lots", "risk_amount", "max_loss_pips"]
}
```

## Code
```python
from typing import Dict, Any

def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    stop_loss_pips: float,
    pip_value: float = 10.0
) -> Dict[str, Any]:
    """
    Calculate position size based on risk management rules.

    Args:
        account_balance: Current account balance
        risk_percent: Risk percentage per trade (e.g., 1.0 for 1%)
        stop_loss_pips: Stop loss distance in pips
        pip_value: Value per pip in base currency (default: 10 for standard lot)

    Returns:
        Dict with position_size_lots, risk_amount, and max_loss_pips

    Formula:
    Risk Amount = Account Balance * (Risk Percent / 100)
    Position Size = Risk Amount / (Stop Loss Pips * Pip Value)
    """
    # Calculate risk amount
    risk_amount = account_balance * (risk_percent / 100.0)

    # Calculate position size in lots
    position_size_lots = risk_amount / (stop_loss_pips * pip_value)

    # Round to 2 decimal places (standard lot precision)
    position_size_lots = round(position_size_lots, 2)

    # Ensure minimum position size
    if position_size_lots < 0.01:
        position_size_lots = 0.01

    # Maximum loss if stop is hit
    max_loss_pips = stop_loss_pips * position_size_lots * pip_value

    return {
        "position_size_lots": position_size_lots,
        "risk_amount": round(risk_amount, 2),
        "max_loss_pips": round(max_loss_pips, 2)
    }
```

## Example Usage
```python
# Calculate position size for 1% risk on $10,000 account with 20 pip stop
result = calculate_position_size(
    account_balance=10000.0,
    risk_percent=1.0,
    stop_loss_pips=20.0
)
# Output: {"position_size_lots": 0.5, "risk_amount": 100.0, "max_loss_pips": 100.0}
```
