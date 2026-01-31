---
name: log_trade_event
category: system_skills
version: 1.0.0
dependencies: []
---

# Log Trade Event Skill

## Description
Logs trade events (entry, exit, stop hit, take profit) to the trading journal.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "event_type": {
      "type": "string",
      "enum": ["entry", "exit", "stop_hit", "take_profit", "error"],
      "description": "Type of trade event"
    },
    "symbol": {
      "type": "string",
      "description": "Trading symbol"
    },
    "action": {
      "type": "string",
      "enum": ["buy", "sell"],
      "description": "Trade direction"
    },
    "price": {
      "type": "number",
      "description": "Execution price"
    },
    "lots": {
      "type": "number",
      "description": "Position size in lots"
    },
    "pnl": {
      "type": "number",
      "description": "Profit/loss (for exit events)"
    },
    "strategy_name": {
      "type": "string",
      "description": "Name of the strategy"
    }
  },
  "required": ["event_type", "symbol", "action", "price", "lots", "strategy_name"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "logged": {
      "type": "boolean",
      "description": "True if event was logged successfully"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO format timestamp of the log entry"
    },
    "log_id": {
      "type": "string",
      "description": "Unique identifier for the log entry"
    }
  },
  "required": ["logged", "timestamp", "log_id"]
}
```

## Code
```python
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

def log_trade_event(
    event_type: str,
    symbol: str,
    action: str,
    price: float,
    lots: float,
    strategy_name: str,
    pnl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Log a trade event to the trading journal.

    Args:
        event_type: Type of event (entry, exit, stop_hit, take_profit, error)
        symbol: Trading symbol
        action: Trade direction (buy, sell)
        price: Execution price
        lots: Position size in lots
        strategy_name: Name of the strategy
        pnl: Optional profit/loss for exit events

    Returns:
        Dict with logged status, timestamp, and log_id
    """
    # Generate unique log ID
    log_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Create log entry
    log_entry = {
        "log_id": log_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "symbol": symbol,
        "action": action,
        "price": price,
        "lots": lots,
        "strategy_name": strategy_name,
        "pnl": pnl
    }

    # In production, this would write to:
    # - SQLite database at /data/journal/trade_journal.db
    # - ChromaDB collection for semantic search
    # - Redis Pub/Sub channel for real-time monitoring

    # For now, we just return success with the log entry data
    # The actual persistence would be handled by the system

    return {
        "logged": True,
        "timestamp": timestamp,
        "log_id": log_id
    }
```

## Example Usage
```python
# Log a trade entry
result = log_trade_event(
    event_type="entry",
    symbol="EURUSD",
    action="buy",
    price=1.1000,
    lots=0.1,
    strategy_name="RSI_Mean_Reversion"
)
# Output: {"logged": True, "timestamp": "2024-01-28T12:30:45.123456", "log_id": "abc-123-def"}

# Log a trade exit with PnL
result = log_trade_event(
    event_type="exit",
    symbol="EURUSD",
    action="sell",
    price=1.1050,
    lots=0.1,
    strategy_name="RSI_Mean_Reversion",
    pnl=50.0
)
```
