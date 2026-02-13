# QuantMind-X API Documentation

## Overview

QuantMind-X provides a RESTful API for strategy management, market data, and system control, along with WebSocket endpoints for real-time updates.

**Base URL:** `http://localhost:8000`

## Authentication

Currently in development - API keys will be required for production.

## REST API Endpoints

### Strategy Management

#### Create Strategy
```http
POST /api/strategies/create
Content-Type: application/json

{
  "name": "RSI Mean Reversion",
  "description": "RSI-based mean reversion strategy",
  "code": "// MQL5 or Python strategy code",
  "parameters": {
    "rsi_period": 14,
    "oversold": 30,
    "overbought": 70
  },
  "risk_settings": {
    "max_position_size": 0.02,
    "stop_loss_pct": 0.02
  }
}
```

#### List Strategies
```http
GET /api/strategies/list
```

Response:
```json
{
  "strategies": [
    {
      "id": "str_123",
      "name": "RSI Mean Reversion",
      "status": "active",
      "created_at": "2026-02-13T14:00:00Z",
      "performance": {
        "win_rate": 0.65,
        "profit_factor": 1.8
      }
    }
  ]
}
```

#### Update Strategy
```http
PUT /api/strategies/{id}/update
Content-Type: application/json

{
  "parameters": {
    "rsi_period": 21
  }
}
```

#### Delete Strategy
```http
DELETE /api/strategies/{id}
```

### Market Data

#### Get Available Symbols
```http
GET /api/market/symbols
```

Response:
```json
{
  "symbols": [
    {
      "symbol": "EURUSD",
      "description": "Euro vs US Dollar",
      "digits": 5,
      "trade_mode": "full"
    }
  ]
}
```

#### Get Real-time Ticks
```http
GET /api/market/ticks/{symbol}?limit=100
```

Response:
```json
{
  "symbol": "EURUSD",
  "ticks": [
    {
      "time": "2026-02-13T14:00:00Z",
      "bid": 1.08567,
      "ask": 1.08570,
      "volume": 100
    }
  ]
}
```

#### Get Historical Data
```http
GET /api/market/history/{symbol}?timeframe=H1&start=2026-02-01&end=2026-02-13
```

### Backtesting

#### Run Backtest
```http
POST /api/backtest/run
Content-Type: application/json

{
  "strategy_id": "str_123",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "start_date": "2026-01-01",
  "end_date": "2026-02-01",
  "initial_balance": 10000
}
```

Response:
```json
{
  "backtest_id": "bt_456",
  "status": "running",
  "estimated_completion": "2026-02-13T14:30:00Z"
}
```

#### Get Backtest Results
```http
GET /api/backtest/results/{id}
```

Response:
```json
{
  "backtest_id": "bt_456",
  "status": "completed",
  "performance": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": 0.05,
    "win_rate": 0.65,
    "profit_factor": 1.8
  },
  "trades": [
    {
      "id": 1,
      "symbol": "EURUSD",
      "type": "BUY",
      "volume": 0.1,
      "open_time": "2026-01-01T00:00:00Z",
      "close_time": "2026-01-01T02:00:00Z",
      "open_price": 1.08500,
      "close_price": 1.08750,
      "profit": 25.0
    }
  ]
}
```

### Risk Management

#### Get Current Risk Status
```http
GET /api/risk/status
```

Response:
```json
{
  "chaos_score": 0.3,
  "risk_multiplier": 1.0,
  "daily_loss": 0.01,
  "daily_limit": 0.02,
  "open_positions": 3,
  "total_exposure": 0.15
}
```

#### Update Risk Parameters
```http
PUT /api/risk/parameters
Content-Type: application/json

{
  "max_daily_loss": 0.03,
  "max_position_size": 0.025,
  "chaos_threshold": 0.5
}
```

## WebSocket Endpoints

### Market Data Stream
```
ws://localhost:8000/ws/market
```

Subscribe to real-time market data:
```json
{
  "action": "subscribe",
  "symbols": ["EURUSD", "GBPUSD"]
}
```

Real-time updates:
```json
{
  "type": "tick",
  "symbol": "EURUSD",
  "time": "2026-02-13T14:00:00Z",
  "bid": 1.08567,
  "ask": 1.08570
}
```

### Strategy Execution Stream
```
ws://localhost:8000/ws/execution
```

Real-time execution updates:
```json
{
  "type": "trade_opened",
  "strategy_id": "str_123",
  "symbol": "EURUSD",
  "type": "BUY",
  "volume": 0.1,
  "price": 1.08570
}
```

### Agent Communication Stream
```
ws://localhost:8000/ws/agents
```

Agent status updates:
```json
{
  "type": "agent_message",
  "agent": "quantcode",
  "message": "Strategy compilation completed successfully",
  "timestamp": "2026-02-13T14:00:00Z"
}
```

## Error Responses

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized (when implemented)
- `404` - Not Found
- `500` - Internal Server Error

Error response format:
```json
{
  "error": {
    "code": "STRATEGY_NOT_FOUND",
    "message": "Strategy with ID 'str_123' not found"
  }
}
```

## Rate Limiting

Currently not implemented - will be added for production.

## SDK Examples

### Python
```python
import requests

# Get strategies
response = requests.get("http://localhost:8000/api/strategies/list")
strategies = response.json()

# Run backtest
backtest_data = {
    "strategy_id": "str_123",
    "symbol": "EURUSD",
    "timeframe": "H1",
    "start_date": "2026-01-01",
    "end_date": "2026-02-01"
}
response = requests.post("http://localhost:8000/api/backtest/run", json=backtest_data)
```

### JavaScript
```javascript
// WebSocket connection for market data
const ws = new WebSocket('ws://localhost:8000/ws/market');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    symbols: ['EURUSD', 'GBPUSD']
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Market update:', data);
};
```

## Development Notes

- API is currently in active development
- Authentication and rate limiting will be added
- Endpoints may change as the system evolves
- WebSocket connections are persistent and will auto-reconnect