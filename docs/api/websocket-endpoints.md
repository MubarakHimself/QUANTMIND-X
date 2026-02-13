# WebSocket API Endpoints

This document describes the WebSocket endpoints available in QuantMindX for real-time communication.

## Endpoint

```
ws://localhost:8000/ws
```

For production deployments, use `wss://` for secure connections.

## Connection

### Establish Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('Connected to WebSocket');
};
```

### Connection Response

The server accepts the connection immediately. No initial message is sent.

## Client Messages

### Subscribe to Topic

Subscribe to receive messages for a specific topic:

```json
{
  "action": "subscribe",
  "topic": "backtest"
}
```

**Available Topics:**
- `backtest` - Backtest progress, logs, and results
- `trading` - Live trading updates
- `logs` - System log entries

**Response:**

```json
{
  "type": "subscription_confirmed",
  "topic": "backtest"
}
```

### Ping (Keep-Alive)

Send a ping to keep the connection alive:

```json
{
  "action": "ping"
}
```

**Response:**

```json
{
  "type": "pong"
}
```

## Server Messages

### Backtest Events

#### backtest_start

Sent when a backtest begins:

```json
{
  "type": "backtest_start",
  "data": {
    "backtest_id": "550e8400-e29b-41d4-a716-446655440000",
    "variant": "spiced",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| backtest_id | string | Unique identifier for the backtest |
| variant | string | Backtest variant: vanilla, spiced, vanilla_full, spiced_full |
| symbol | string | Trading symbol |
| timeframe | string | Chart timeframe (M1, M5, M15, M30, H1, H4, D1) |
| start_date | string | Backtest start date (YYYY-MM-DD) |
| end_date | string | Backtest end date (YYYY-MM-DD) |
| timestamp | string | ISO 8601 timestamp |

#### backtest_progress

Sent during backtest execution:

```json
{
  "type": "backtest_progress",
  "data": {
    "backtest_id": "550e8400-e29b-41d4-a716-446655440000",
    "progress": 50.0,
    "status": "Processing bar 5000/10000",
    "bars_processed": 5000,
    "total_bars": 10000,
    "current_date": "2023-06-15",
    "trades_count": 25,
    "current_pnl": 500.00,
    "timestamp": "2024-01-15T10:30:30.000Z"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| progress | number | Progress percentage (0-100) |
| status | string | Human-readable status message |
| bars_processed | number | Number of bars processed |
| total_bars | number | Total bars to process |
| current_date | string | Current date being processed |
| trades_count | number | Number of trades executed so far |
| current_pnl | number | Current profit/loss |

#### log_entry

Sent for each log message:

```json
{
  "type": "log_entry",
  "data": {
    "backtest_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2024-01-15T10:30:31.000Z",
    "level": "INFO",
    "message": "Trade executed: BUY 0.1 lots EURUSD",
    "module": "mode_runner",
    "function": "execute_trade",
    "line": 245,
    "logger_name": "backtest.550e8400-e29b-41d4-a716-446655440000"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| level | string | Log level: INFO, WARNING, ERROR, DEBUG |
| message | string | Log message content |
| module | string | Source module name |
| function | string | Source function name |
| line | number | Source line number |

#### backtest_complete

Sent when backtest finishes successfully:

```json
{
  "type": "backtest_complete",
  "data": {
    "backtest_id": "550e8400-e29b-41d4-a716-446655440000",
    "final_balance": 12500.00,
    "total_trades": 45,
    "win_rate": 62.5,
    "sharpe_ratio": 1.85,
    "drawdown": 8.5,
    "return_pct": 25.0,
    "duration_seconds": 45.3,
    "timestamp": "2024-01-15T10:31:00.000Z",
    "results": {}
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| final_balance | number | Final account balance |
| total_trades | number | Total number of trades |
| win_rate | number | Win rate percentage |
| sharpe_ratio | number | Sharpe ratio |
| drawdown | number | Maximum drawdown percentage |
| return_pct | number | Return percentage |
| duration_seconds | number | Backtest duration in seconds |
| results | object | Additional results (regime analytics, etc.) |

#### backtest_error

Sent when backtest encounters an error:

```json
{
  "type": "backtest_error",
  "data": {
    "backtest_id": "550e8400-e29b-41d4-a716-446655440000",
    "error": "Strategy execution failed",
    "error_details": "ZeroDivisionError: division by zero at line 45",
    "timestamp": "2024-01-15T10:30:45.000Z"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| error | string | Error message |
| error_details | string | Detailed error information (stack trace, etc.) |

### Trading Events

#### trading_update

Sent for live trading status updates:

```json
{
  "type": "trading_update",
  "data": {
    "bot_id": "bot-001",
    "status": "running",
    "pnl": 150.25
  }
}
```

## Client Implementation Examples

### JavaScript/TypeScript

```typescript
import { WebSocketClient, createBacktestClient } from './ws-client';

// Create client with auto-subscribe
const client = await createBacktestClient('http://localhost:8000');

// Register event handlers
client.on('backtest_start', (message) => {
  console.log('Backtest started:', message.data);
});

client.on('backtest_progress', (message) => {
  console.log(`Progress: ${message.data.progress}%`);
});

client.on('backtest_complete', (message) => {
  console.log('Backtest complete:', message.data);
  client.disconnect();
});

client.on('backtest_error', (message) => {
  console.error('Backtest error:', message.data.error);
});

client.on('log_entry', (message) => {
  console.log(`[${message.data.level}] ${message.data.message}`);
});

// Disconnect when done
// client.disconnect();
```

### Raw WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  // Subscribe to backtest topic
  ws.send(JSON.stringify({
    action: 'subscribe',
    topic: 'backtest'
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'subscription_confirmed':
      console.log('Subscribed to:', message.topic);
      break;
    case 'backtest_start':
      console.log('Backtest started:', message.data);
      break;
    case 'backtest_progress':
      console.log(`Progress: ${message.data.progress}%`);
      break;
    case 'backtest_complete':
      console.log('Complete:', message.data);
      break;
    case 'backtest_error':
      console.error('Error:', message.data.error);
      break;
    case 'log_entry':
      console.log(`[${message.data.level}] ${message.data.message}`);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket closed');
};
```

### Python

```python
import asyncio
import websockets
import json

async def listen_backtest():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as ws:
        # Subscribe to backtest topic
        await ws.send(json.dumps({
            "action": "subscribe",
            "topic": "backtest"
        }))
        
        # Listen for messages
        while True:
            try:
                message = await ws.recv()
                data = json.loads(message)
                
                if data['type'] == 'subscription_confirmed':
                    print(f"Subscribed to: {data['topic']}")
                elif data['type'] == 'backtest_start':
                    print(f"Backtest started: {data['data']}")
                elif data['type'] == 'backtest_progress':
                    print(f"Progress: {data['data']['progress']}%")
                elif data['type'] == 'backtest_complete':
                    print(f"Complete: {data['data']}")
                    break
                elif data['type'] == 'backtest_error':
                    print(f"Error: {data['data']['error']}")
                    break
                elif data['type'] == 'log_entry':
                    print(f"[{data['data']['level']}] {data['data']['message']}")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break

asyncio.run(listen_backtest())
```

## Error Handling

### Connection Errors

```javascript
ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
  if (event.wasClean) {
    console.log(`Connection closed: ${event.code} ${event.reason}`);
  } else {
    console.log('Connection died unexpectedly');
  }
  
  // Attempt reconnection
  setTimeout(() => {
    console.log('Reconnecting...');
    // Reconnect logic here
  }, 5000);
};
```

### Reconnection Strategy

```javascript
class ResilientWebSocket {
  constructor(url, options = {}) {
    this.url = url;
    this.reconnectInterval = options.reconnectInterval || 5000;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    this.reconnectAttempts = 0;
    this.ws = null;
    this.connect();
  }
  
  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.onopen?.();
    };
    
    this.ws.onclose = (event) => {
      if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), this.reconnectInterval);
      }
      this.onclose?.(event);
    };
    
    this.ws.onmessage = (event) => {
      this.onmessage?.(event);
    };
    
    this.ws.onerror = (error) => {
      this.onerror?.(error);
    };
  }
  
  send(data) {
    this.ws?.send(data);
  }
}
```

## Troubleshooting

### Connection Refused

- Ensure the server is running
- Check the correct port (default: 8000)
- Verify firewall settings

### No Messages Received

- Confirm subscription to the correct topic
- Check that backtest was started via REST API
- Verify WebSocket connection is open

### Messages Stop Suddenly

- Check server logs for errors
- Verify network connectivity
- Implement reconnection logic

### High Latency

- Check network conditions
- Consider message batching for high-frequency updates
- Verify server resource usage

## Rate Limits

No rate limits are currently enforced on WebSocket connections. However, clients should:

- Limit reconnection attempts
- Avoid sending excessive ping messages
- Process messages efficiently to prevent backlog

## Security

Current implementation has no authentication. For production:

- Add JWT authentication
- Implement rate limiting
- Use WSS (TLS) for encrypted connections
- Validate all incoming messages
