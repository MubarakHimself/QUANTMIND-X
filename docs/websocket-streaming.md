# WebSocket Log Streaming Architecture

## Overview
Real-time backtest progress and log streaming via WebSocket.

## Architecture
```mermaid
sequenceDiagram
    participant UI as UI Client
    participant WS as WebSocket Server
    participant API as API Endpoint
    participant Engine as Backtest Engine
    participant Logger as WS Logger
    
    UI->>WS: Connect to /ws
    WS->>UI: Connection accepted
    UI->>WS: Subscribe to \"backtest\"
    WS->>UI: Subscription confirmed
    
    UI->>API: POST /api/v1/backtest/run
    API->>Engine: Start backtest
    API->>UI: {backtest_id, status: \"running\"}
    
    Engine->>Logger: setup_backtest_logging()
    Logger->>WS: Register handlers
    
    Engine->>WS: broadcast(backtest_start)
    WS->>UI: backtest_start event
    
    loop Every 100 bars
        Engine->>WS: broadcast(backtest_progress)
        WS->>UI: backtest_progress event
        UI->>UI: Update progress bar
    end
    
    loop On log events
        Engine->>Logger: logger.info()
        Logger->>WS: broadcast(log_entry)
        WS->>UI: log_entry event
        UI->>UI: Append to log container
    end
    
    Engine->>WS: broadcast(backtest_complete)
    WS->>UI: backtest_complete event
    UI->>UI: Display results
```

## Components
- WebSocketLogHandler: Custom Python logging handler
- BacktestProgressStreamer: Backtest lifecycle events
- WebSocket Client: TypeScript client with reconnection
- BacktestRunner UI: Real-time display component

## Usage

### Backend Integration
```python
from src.api.ws_logger import setup_backtest_logging

# Setup WebSocket logging
logger, progress_streamer = setup_backtest_logging(backtest_id)

# Use logger for real-time logs
logger.info(\"Backtest started\")

# Broadcast progress
await progress_streamer.update_progress(
    progress=50.0,
    status=\"Processing...\",
    trades_count=10
)
```

### Frontend Integration
```typescript
import { createBacktestClient } from '$lib/ws-client';

// Connect to WebSocket
const client = await createBacktestClient('http://localhost:8000');

// Handle events
client.on('backtest_progress', (message) => {
  console.log('Progress:', message.data.progress);
});
```

## Message Types
- `backtest_start`: Backtest parameters
- `backtest_progress`: Progress percentage, bars processed, trades, P&L
- `log_entry`: Real-time log messages with level, timestamp, module
- `backtest_complete`: Final metrics (balance, trades, Sharpe, drawdown)
- `backtest_error`: Error details

## Testing
- `tests/api/test_ws_logger.py`: Unit tests (>80% coverage)
- `tests/e2e/test_backtest_streaming_e2e.py`: E2E integration tests
- `quantmind-ide/src/lib/components/BacktestRunner.test.ts`: UI component tests

Run with: `pytest tests/api/test_ws_logger.py -v` and `pytest tests/e2e/ -v`
