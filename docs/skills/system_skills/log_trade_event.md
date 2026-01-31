---
name: log_trade_event
category: system_skills
description: Log trade events to trading journal with structured data storage and retrieval
version: 1.0.0
dependencies: []
tags:
  - logging
  - journal
  - trade_history
---

# Log Trade Event Skill

## Description

Logs trade events to a structured trading journal for analysis, compliance, and performance tracking. Supports various event types including entries, exits, modifications, and errors. Logs are stored in both file-based format (JSON) and can be exported to databases for long-term storage.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "event_type": {
      "type": "string",
      "enum": [
        "signal_generated",
        "order_placed",
        "order_filled",
        "order_modified",
        "order_cancelled",
        "position_opened",
        "position_closed",
        "stop_loss_hit",
        "take_profit_hit",
        "margin_call",
        "error",
        "system_event"
      ],
      "description": "Type of trade event"
    },
    "symbol": {
      "type": "string",
      "description": "Trading symbol"
    },
    "direction": {
      "type": "string",
      "enum": ["long", "short", "none"],
      "description": "Trade direction (if applicable)"
    },
    "lots": {
      "type": "number",
      "description": "Position size in lots"
    },
    "entry_price": {
      "type": "number",
      "description": "Entry price"
    },
    "exit_price": {
      "type": "number",
      "description": "Exit price (if closed)"
    },
    "stop_loss": {
      "type": "number",
      "description": "Stop loss price"
    },
    "take_profit": {
      "type": "number",
      "description": "Take profit price"
    },
    "profit": {
      "type": "number",
      "description": "Profit/loss in base currency"
    },
    "pips": {
      "type": "number",
      "description": "Profit/loss in pips"
    },
    "order_id": {
      "type": "string",
      "description": "Order/ticket ID"
    },
    "strategy": {
      "type": "string",
      "description": "Strategy name that generated the trade"
    },
    "reason": {
      "type": "string",
      "description": "Reason for the event"
    },
    "notes": {
      "type": "string",
      "description": "Additional notes"
    },
    "metadata": {
      "type": "object",
      "description": "Additional custom data"
    }
  },
  "required": ["event_type"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether logging was successful"
    },
    "event_id": {
      "type": "string",
      "description": "Unique ID for the logged event"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp of the event"
    },
    "log_file": {
      "type": "string",
      "description": "Path to log file"
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
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import sqlite3

# Directory for trade logs
TRADE_LOG_DIR = "/data/journal/trades"
DB_PATH = "/data/journal/trade_journal.db"

def log_trade_event(
    event_type: str,
    symbol: Optional[str] = None,
    direction: Optional[str] = None,
    lots: Optional[float] = None,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    profit: Optional[float] = None,
    pips: Optional[float] = None,
    order_id: Optional[str] = None,
    strategy: Optional[str] = None,
    reason: Optional[str] = None,
    notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Log a trade event to the trading journal.

    Args:
        event_type: Type of trade event
        symbol: Trading symbol
        direction: Trade direction (long/short)
        lots: Position size
        entry_price: Entry price
        exit_price: Exit price
        stop_loss: Stop loss price
        take_profit: Take profit price
        profit: Profit/loss amount
        pips: Profit/loss in pips
        order_id: Order/ticket ID
        strategy: Strategy name
        reason: Reason for event
        notes: Additional notes
        metadata: Custom data

    Returns:
        Dictionary containing log result with event_id
    """
    # Generate unique event ID
    event_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Create log entry
    log_entry = {
        "event_id": event_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "symbol": symbol,
        "direction": direction,
        "lots": lots,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "profit": profit,
        "pips": pips,
        "order_id": order_id,
        "strategy": strategy,
        "reason": reason,
        "notes": notes,
        "metadata": metadata or {}
    }

    try:
        # Create log directory
        os.makedirs(TRADE_LOG_DIR, exist_ok=True)

        # Write to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(TRADE_LOG_DIR, f"trades_{date_str}.jsonl")

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Also write to database for querying
        _write_to_database(log_entry)

        return {
            "success": True,
            "event_id": event_id,
            "timestamp": timestamp,
            "log_file": log_file
        }

    except Exception as e:
        return {
            "success": False,
            "event_id": event_id,
            "error": str(e)
        }


def _write_to_database(log_entry: Dict[str, Any]) -> bool:
    """
    Write log entry to SQLite database for long-term storage.

    Args:
        log_entry: Log entry dictionary

    Returns:
        True if successful
    """
    try:
        # Create database directory
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                symbol TEXT,
                direction TEXT,
                lots REAL,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                profit REAL,
                pips REAL,
                order_id TEXT,
                strategy TEXT,
                reason TEXT,
                notes TEXT,
                metadata TEXT
            )
        ''')

        # Insert log entry
        cursor.execute('''
            INSERT INTO trade_events (
                event_id, timestamp, event_type, symbol, direction, lots,
                entry_price, exit_price, stop_loss, take_profit, profit, pips,
                order_id, strategy, reason, notes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_entry["event_id"],
            log_entry["timestamp"],
            log_entry["event_type"],
            log_entry["symbol"],
            log_entry["direction"],
            log_entry["lots"],
            log_entry["entry_price"],
            log_entry["exit_price"],
            log_entry["stop_loss"],
            log_entry["take_profit"],
            log_entry["profit"],
            log_entry["pips"],
            log_entry["order_id"],
            log_entry["strategy"],
            log_entry["reason"],
            log_entry["notes"],
            json.dumps(log_entry["metadata"])
        ))

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print(f"Database write error: {e}")
        return False


def query_trade_events(
    symbol: Optional[str] = None,
    event_type: Optional[str] = None,
    strategy: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Query trade events from the database.

    Args:
        symbol: Filter by symbol
        event_type: Filter by event type
        strategy: Filter by strategy
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        limit: Maximum results

    Returns:
        List of trade event dictionaries
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM trade_events WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to dictionaries
        columns = [
            "event_id", "timestamp", "event_type", "symbol", "direction",
            "lots", "entry_price", "exit_price", "stop_loss", "take_profit",
            "profit", "pips", "order_id", "strategy", "reason", "notes", "metadata"
        ]

        results = []
        for row in rows:
            entry = dict(zip(columns, row))
            # Parse metadata JSON
            if entry["metadata"]:
                entry["metadata"] = json.loads(entry["metadata"])
            results.append(entry)

        conn.close()
        return results

    except Exception as e:
        print(f"Query error: {e}")
        return []


def get_trade_statistics(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate trading statistics from logged events.

    Args:
        symbol: Filter by symbol
        strategy: Filter by strategy
        start_date: Start date (ISO format)
        end_date: End date (ISO format)

    Returns:
        Dictionary with trading statistics
    """
    # Query closed positions
    events = query_trade_events(
        symbol=symbol,
        event_type="position_closed",
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        limit=10000
    )

    if not events:
        return {
            "total_trades": 0,
            "total_profit": 0,
            "win_rate": 0,
            "average_profit": 0,
            "largest_win": 0,
            "largest_loss": 0
        }

    # Calculate statistics
    profits = [e["profit"] for e in events if e["profit"] is not None]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    total_trades = len(profits)
    total_profit = sum(profits)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    average_profit = total_profit / total_trades if total_trades > 0 else 0
    largest_win = max(wins) if wins else 0
    largest_loss = min(losses) if losses else 0

    return {
        "total_trades": total_trades,
        "total_profit": round(total_profit, 2),
        "win_rate": round(win_rate, 2),
        "average_profit": round(average_profit, 2),
        "largest_win": round(largest_win, 2),
        "largest_loss": round(largest_loss, 2),
        "profit_factor": abs(sum(wins) / sum(losses)) if losses else 0
    }
```

## Example Usage

```python
# Example 1: Log a signal generation
result = log_trade_event(
    event_type="signal_generated",
    symbol="EURUSD",
    direction="long",
    strategy="RSI_Mean_Reversion",
    reason="RSI oversold (< 30)",
    notes="Potential long entry at 1.0850"
)

print(f"Logged event: {result['event_id']}")

# Example 2: Log position opened
result = log_trade_event(
    event_type="position_opened",
    symbol="EURUSD",
    direction="long",
    lots=0.1,
    entry_price=1.0850,
    stop_loss=1.0820,
    take_profit=1.0910,
    order_id="12345",
    strategy="RSI_Mean_Reversion"
)

# Example 3: Log position closed with profit
result = log_trade_event(
    event_type="position_closed",
    symbol="EURUSD",
    direction="long",
    lots=0.1,
    entry_price=1.0850,
    exit_price=1.0910,
    profit=100.0,
    pips=60,
    order_id="12345",
    strategy="RSI_Mean_Reversion",
    reason="Take profit hit",
    notes="Trade completed as planned"
)

# Example 4: Log an error
result = log_trade_event(
    event_type="error",
    symbol="GBPUSD",
    strategy="Breakout",
    reason="Order failed",
    notes="Insufficient margin",
    metadata={"error_code": "ERR_NO_MONEY", "required": 1000, "available": 500}
)

# Example 5: Query recent trades
recent_trades = query_trade_events(
    symbol="EURUSD",
    limit=10
)

for trade in recent_trades:
    print(f"{trade['timestamp']}: {trade['event_type']} - {trade['symbol']}")

# Example 6: Get trading statistics
stats = get_trade_statistics(
    symbol="EURUSD",
    strategy="RSI_Mean_Reversion"
)

print(f"Total Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']}%")
print(f"Total Profit: ${stats['total_profit']}")
print(f"Profit Factor: {stats['profit_factor']}")

# Example 7: Query by date range
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=30)

trades_last_month = query_trade_events(
    start_date=start_date.isoformat(),
    end_date=end_date.isoformat()
)

print(f"Trades in last 30 days: {len(trades_last_month)}")
```

## Notes

- Events are logged to both JSONL files (by date) and SQLite database
- JSONL files are human-readable and can be easily imported/processed
- SQLite database enables efficient querying and statistics
- Each event gets a unique UUID for tracking
- All timestamps are in ISO 8601 format
- Custom metadata can be added for flexibility

## Event Types

| Event Type | Description |
|------------|-------------|
| `signal_generated` | Strategy generated a trade signal |
| `order_placed` | Order submitted to broker |
| `order_filled` | Order filled by broker |
| `order_modified` | Order parameters modified |
| `order_cancelled` | Order cancelled |
| `position_opened` | Position opened (fill) |
| `position_closed` | Position closed |
| `stop_loss_hit` | Stop loss triggered |
| `take_profit_hit` | Take profit triggered |
| `margin_call` | Margin call occurred |
| `error` | Error occurred |
| `system_event` | System-level event |

## Use Cases

1. **Compliance**: Maintain audit trail for regulatory requirements
2. **Performance Analysis**: Calculate win rate, profit factor, etc.
3. **Strategy Review**: Review past trades to improve strategy
4. **Debugging**: Track order flow and identify issues

## Dependencies

- No skill dependencies (standalone logging system)
- SQLite3 for database storage

## See Also

- `read_mt5_data`: Fetch trade data from MT5 for logging
- `calculate_position_size`: Calculate position size before logging
