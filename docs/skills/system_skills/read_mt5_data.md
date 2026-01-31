---
name: read_mt5_data
category: system_skills
description: Read market data from MetaTrader 5 terminal including prices, indicators, and account information
version: 1.0.0
dependencies: []
tags:
  - mt5
  - data_feed
  - market_data
---

# Read MT5 Data Skill

## Description

Reads various types of market data from the MetaTrader 5 terminal including current prices, historical OHLCV data, account information, open positions, pending orders, and trading history. This skill serves as the primary data interface for MT5-integrated trading systems.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "data_type": {
      "type": "string",
      "enum": [
        "tick",
        "rates",
        "account_info",
        "positions",
        "orders",
        "history",
        "symbols"
      ],
      "description": "Type of data to retrieve"
    },
    "symbol": {
      "type": "string",
      "description": "Trading symbol (required for tick, rates)"
    },
    "timeframe": {
      "type": "string",
      "enum": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"],
      "description": "Timeframe for rate data (default: H1)"
    },
    "count": {
      "type": "integer",
      "description": "Number of bars/ticks to retrieve (default: 100)",
      "default": 100,
      "minimum": 1,
      "maximum": 10000
    },
    "start_pos": {
      "type": "integer",
      "description": "Starting position (0 = most recent, default: 0)",
      "default": 0,
      "minimum": 0
    },
    "date_from": {
      "type": "string",
      "description": "Start date for history (ISO format: 'YYYY-MM-DD')"
    },
    "date_to": {
      "type": "string",
      "description": "End date for history (ISO format: 'YYYY-MM-DD')"
    }
  },
  "required": ["data_type"]
}
```

## Output Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether data retrieval was successful"
    },
    "data_type": {
      "type": "string",
      "description": "Type of data returned"
    },
    "data": {
      "description": "Retrieved data (varies by data_type)"
    },
    "count": {
      "type": "integer",
      "description": "Number of records returned"
    },
    "timestamp": {
      "type": "string",
      "description": "ISO timestamp of data retrieval"
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
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

def read_mt5_data(
    data_type: str,
    symbol: Optional[str] = None,
    timeframe: str = "H1",
    count: int = 100,
    start_pos: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read market data from MetaTrader 5 terminal.

    Args:
        data_type: Type of data to retrieve
        symbol: Trading symbol (required for tick, rates)
        timeframe: Timeframe for rate data (default: H1)
        count: Number of bars/ticks to retrieve
        start_pos: Starting position (0 = most recent)
        date_from: Start date for history (ISO format)
        date_to: End date for history (ISO format)

    Returns:
        Dictionary containing retrieved data and metadata

    Raises:
        ValueError: If invalid parameters or MT5 not connected
    """
    # Check MT5 connection
    if not mt5.initialize():
        return {
            "success": False,
            "data_type": data_type,
            "error": f"MT5 initialization failed: {mt5.last_error()}",
            "timestamp": datetime.now().isoformat()
        }

    result = {
        "success": True,
        "data_type": data_type,
        "timestamp": datetime.now().isoformat()
    }

    try:
        if data_type == "tick":
            # Get current tick data
            if symbol is None:
                raise ValueError("Symbol is required for tick data")

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise ValueError(f"Failed to get tick for {symbol}")

            result["data"] = {
                "symbol": symbol,
                "bid": tick.bid,
                "ask": tick.ask,
                "spread": tick.ask - tick.bid,
                "time": datetime.fromtimestamp(tick.time).isoformat(),
                "volume": tick.volume,
                "flags": tick.flags
            }
            result["count"] = 1

        elif data_type == "rates":
            # Get OHLCV rate data
            if symbol is None:
                raise ValueError("Symbol is required for rates data")

            # Map timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1,
            }

            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_H1)

            # Fetch rates
            if date_from and date_to:
                # Date range query
                dt_from = datetime.fromisoformat(date_from)
                dt_to = datetime.fromisoformat(date_to)
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, dt_from, dt_to)
            elif date_from:
                # From date to present
                dt_from = datetime.fromisoformat(date_from)
                rates = mt5.copy_rates_from(symbol, mt5_timeframe, dt_from, count)
            else:
                # From position
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, start_pos, count)

            if rates is None or len(rates) == 0:
                raise ValueError(f"No rates retrieved for {symbol} {timeframe}")

            # Convert to list of dicts for JSON serialization
            rates_list = []
            for rate in rates:
                rates_list.append({
                    "time": datetime.fromtimestamp(rate["time"]).isoformat(),
                    "open": float(rate["open"]),
                    "high": float(rate["high"]),
                    "low": float(rate["low"]),
                    "close": float(rate["close"]),
                    "tick_volume": int(rate["tick_volume"]),
                    "spread": int(rate["spread"]),
                    "real_volume": int(rate["real_volume"])
                })

            result["data"] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "rates": rates_list
            }
            result["count"] = len(rates_list)

        elif data_type == "account_info":
            # Get account information
            account = mt5.account_info()
            if account is None:
                raise ValueError("Failed to get account info")

            result["data"] = {
                "login": account.login,
                "server": account.server,
                "currency": account.currency,
                "balance": float(account.balance),
                "equity": float(account.equity),
                "margin": float(account.margin),
                "margin_free": float(account.margin_free),
                "margin_level": float(account.margin_level),
                "profit": float(account.profit),
                "orders": int(account.orders),
                "positions": int(account.positions)
            }
            result["count"] = 1

        elif data_type == "positions":
            # Get open positions
            positions = mt5.positions_get()
            if positions is None:
                positions = []

            positions_list = []
            for pos in positions:
                positions_list.append({
                    "ticket": pos.ticket,
                    "time": datetime.fromtimestamp(pos.time).isoformat(),
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "lots": float(pos.volume),
                    "price_open": float(pos.price_open),
                    "price_current": float(pos.price_current),
                    "sl": float(pos.sl) if pos.sl > 0 else None,
                    "tp": float(pos.tp) if pos.tp > 0 else None,
                    "profit": float(pos.profit),
                    "magic": pos.magic,
                    "comment": pos.comment
                })

            result["data"] = positions_list
            result["count"] = len(positions_list)

        elif data_type == "orders":
            # Get pending orders
            orders = mt5.orders_get()
            if orders is None:
                orders = []

            orders_list = []
            for order in orders:
                orders_list.append({
                    "ticket": order.ticket,
                    "time_setup": datetime.fromtimestamp(order.time_setup).isoformat(),
                    "symbol": order.symbol,
                    "type": order.type,
                    "lots": float(order.volume_current),
                    "price_open": float(order.price_open),
                    "sl": float(order.sl) if order.sl > 0 else None,
                    "tp": float(order.tp) if order.tp > 0 else None,
                    "magic": order.magic,
                    "comment": order.comment
                })

            result["data"] = orders_list
            result["count"] = len(orders_list)

        elif data_type == "history":
            # Get trading history
            if date_from and date_to:
                dt_from = datetime.fromisoformat(date_from)
                dt_to = datetime.fromisoformat(date_to)
                history = mt5.history_deals_get(dt_from, dt_to)
            else:
                # Get last 100 deals
                dt_to = datetime.now()
                dt_from = dt_to - timedelta(days=30)
                history = mt5.history_deals_get(dt_from, dt_to)

            if history is None:
                history = []

            history_list = []
            for deal in history:
                history_list.append({
                    "ticket": deal.ticket,
                    "time": datetime.fromtimestamp(deal.time).isoformat(),
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "lots": float(deal.volume),
                    "price": float(deal.price),
                    "profit": float(deal.profit),
                    "commission": float(deal.commission),
                    "swap": float(deal.swap),
                    "magic": deal.magic,
                    "comment": deal.comment
                })

            result["data"] = history_list
            result["count"] = len(history_list)

        elif data_type == "symbols":
            # Get available symbols
            symbols = mt5.symbols_get()
            if symbols is None:
                symbols = []

            symbols_list = []
            for sym in symbols:
                symbols_list.append({
                    "name": sym.name,
                    "description": sym.description,
                    "currency_base": sym.currency_base,
                    "currency_profit": sym.currency_profit,
                    "currency_margin": sym.currency_margin,
                    "digits": int(sym.digits),
                    "point": float(sym.point),
                    "trade_contract_size": float(sym.trade_contract_size),
                    "visible": bool(sym.visible)
                })

            result["data"] = symbols_list
            result["count"] = len(symbols_list)

        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        result["data"] = None
        result["count"] = 0

    finally:
        mt5.shutdown()

    return result
```

## Example Usage

```python
# Example 1: Get current tick data
tick = read_mt5_data(data_type="tick", symbol="EURUSD")
print(f"Bid: {tick['data']['bid']}, Ask: {tick['data']['ask']}")
print(f"Spread: {tick['data']['spread']}")

# Example 2: Get OHLCV data
rates = read_mt5_data(
    data_type="rates",
    symbol="GBPUSD",
    timeframe="H1",
    count=50
)
print(f"Retrieved {rates['count']} bars")
print(f"Latest close: {rates['data']['rates'][-1]['close']}")

# Example 3: Get account information
account = read_mt5_data(data_type="account_info")
print(f"Balance: ${account['data']['balance']:.2f}")
print(f"Equity: ${account['data']['equity']:.2f}")
print(f"Open Positions: {account['data']['positions']}")

# Example 4: Get open positions
positions = read_mt5_data(data_type="positions")
for pos in positions['data']:
    print(f"{pos['symbol']} {pos['type']} {pos['lots']} lots @ {pos['price_open']}")

# Example 5: Get trading history for date range
history = read_mt5_data(
    data_type="history",
    date_from="2024-01-01",
    date_to="2024-01-31"
)
total_profit = sum(deal['profit'] for deal in history['data'])
print(f"Total profit for period: ${total_profit:.2f}")

# Example 6: Get available symbols
symbols = read_mt5_data(data_type="symbols")
print(f"Available symbols: {symbols['count']}")
for sym in symbols['data'][:10]:
    print(f"  {sym['name']}: {sym['description']}")
```

## Notes

- MT5 terminal must be running and connected to a broker
- Some data types may require authentication/authorization
- Historical data availability depends on broker and subscription
- Rate data is returned as list of dictionaries for easy JSON serialization
- All monetary values are returned as float for precision
- Timestamps are in ISO 8601 format for consistency

## Error Handling

- Returns success=False with error message if operation fails
- Check MT5 terminal connection before use
- Verify symbol exists and is visible in Market Watch
- Date range queries require valid ISO date formats

## Dependencies

- MetaTrader5 Python API: Must be installed and MT5 terminal available
- No skill dependencies (lowest-level data access)

## See Also

- `write_indicator_buffer`: Write calculated indicator values to MT5
- `fetch_historical_data`: Higher-level data fetching skill
- `log_trade_event`: Log MT5 trading events to journal
