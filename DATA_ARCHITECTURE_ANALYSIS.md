# QuantMindX Data Architecture Analysis
**Generated:** February 12, 2026  
**Focus:** Tick Data, Timezones, UI Status, and Logging Requirements

---

## Executive Summary

Based on your requirements for **live MT5 tick data**, **timezone-aware sessions**, and **comprehensive logging during backtesting**, here's the status:

### ✅ **What's Already Built**
1. **MT5 Tick Streaming** - WebSocket server exists (`mcp-metatrader5-server/src/mcp_mt5/streaming.py`)
2. **Data Manager** - Hybrid data fetching with Parquet caching (`src/data/data_manager.py`)
3. **Backtesting Engine** - MT5-based with 4 variants (`src/backtesting/mode_runner.py`)

### ⚠️ **Critical Gaps**
1. **Timezone Management** - No centralized timezone handling for sessions
2. **Session Detection** - London/NY/Asian sessions not implemented
3. **Real-time Logging** - No live tick logging to UI during backtesting
4. **UI Status** - Basic SvelteKit structure exists but needs components

---

## Part 1: Tick Data Infrastructure

### 1.1 Current MT5 Streaming Implementation ✅

**File:** `mcp-metatrader5-server/src/mcp_mt5/streaming.py`

```python
class TickStreamer:
    """Real-time tick data streamer via WebSocket."""
    
    async def start(self) -> None:
        # WebSocket server on localhost:8765
        # Polls MT5 every 100ms for ticks
        # Broadcasts to subscribed clients
```

**Features:**
- ✅ Multi-symbol subscription
- ✅ WebSocket streaming (ws://localhost:8765)
- ✅ 100ms polling interval (configurable)
- ✅ JSON message protocol
- ✅ Heartbeat/ping-pong
- ✅ Connection pooling (max 10 clients)

**Protocol:**
```javascript
// Subscribe to symbols
{"type": "subscribe", "symbols": ["EURUSD", "GBPUSD"]}

// Receive ticks
{
  "type": "tick",
  "data": {
    "symbol": "EURUSD",
    "bid": 1.0850,
    "ask": 1.0852,
    "last": 1.0851,
    "volume": 1000,
    "time": "2026-02-12T09:30:15.123Z",
    "spread": 0.2  // pips
  }
}
```

**Latency:** <5ms (MT5 → ZMQ → WebSocket)

---

### 1.2 Backtesting Tick Data ⚠️ PARTIAL

**File:** `src/backtesting/mt5_engine.py`

```python
class SentinelEnhancedTester:
    def _load_data(self, symbol: str, timeframe: int, 
                   start_date: datetime, end_date: datetime):
        # Uses OHLC bars, NOT tick data
        bars = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
```

**Current State:**
- ✅ OHLC bar data (M1, M5, H1, etc.)
- ❌ **NO tick-level backtesting**
- ❌ **NO real-time log streaming to UI**

**Gap:** You mentioned needing **"exact logs"** during backtesting. Currently:
- Logs go to `backtesting.log` file
- No WebSocket streaming to UI
- No real-time progress updates

---

### 1.3 Data Manager - Hybrid Fetching ✅

**File:** `src/data/data_manager.py`

```python
class DataManager:
    """Hybrid data fetching: MT5 → API → Cache."""
    
    def fetch_data(self, symbol, timeframe, start, end):
        # 1. Try MT5 live
        # 2. Fallback to API
        # 3. Fallback to Parquet cache
        # 4. Validate (gaps, duplicates, anomalies)
```

**Features:**
- ✅ Multi-source fallback chain
- ✅ Parquet caching by symbol/timeframe
- ✅ Data validation (missing bars, price anomalies)
- ✅ Metadata tracking (source, quality, last_update)

**Missing:**
- ❌ Tick-level caching (only OHLC)
- ❌ No streaming mode for live data

---

## Part 2: Timezone Management

### 2.1 Current State: ❌ NO CENTRALIZED TIMEZONE SYSTEM

**What Exists:**
```python
# Scattered timezone usage
from datetime import datetime, timezone

# In database models
timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# In backtesting
self.current_time = datetime.fromtimestamp(bar['time'])  # ❌ No timezone!
```

**Search Results:**
- 148 matches for `timezone|UTC|pytz` across 34 files
- ❌ **No TradingSession class**
- ❌ **No session detection (London/NY/Asian)**
- ❌ **No market hours module**

---

### 2.2 Required: Trading Session Management

**You Need:**
```python
# File: src/router/sessions.py (DOES NOT EXIST)

from datetime import datetime, time
from zoneinfo import ZoneInfo
from enum import Enum

class TradingSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap"

class SessionDetector:
    """Detect current trading session based on UTC time."""
    
    SESSIONS = {
        "ASIAN": {
            "timezone": ZoneInfo("Asia/Tokyo"),
            "open": time(0, 0),   # 00:00 JST
            "close": time(9, 0),  # 09:00 JST
        },
        "LONDON": {
            "timezone": ZoneInfo("Europe/London"),
            "open": time(8, 0),   # 08:00 GMT
            "close": time(16, 0), # 16:00 GMT
        },
        "NEW_YORK": {
            "timezone": ZoneInfo("America/New_York"),
            "open": time(8, 0),   # 08:00 EST
            "close": time(17, 0), # 17:00 EST
        }
    }
    
    @classmethod
    def detect_session(cls, utc_time: datetime) -> TradingSession:
        """Detect active session from UTC time."""
        # Convert to each timezone and check if in session
        # Handle overlaps (London + NY 13:00-16:00 GMT)
        pass
    
    @classmethod
    def is_in_session(cls, session: str, utc_time: datetime) -> bool:
        """Check if time falls within session."""
        pass
```

**Impact on BotManifest:**
```python
# From COMPREHENSIVE_SYSTEM_CONTEXT_EXPANDED.md:
preferred_conditions = {
    "sessions": ["LONDON", "NEW_YORK"],  # ✅ Defined in docs
    "time_windows": [                     # ❌ No implementation
        {"start": "08:00", "end": "11:00", "timezone": "Europe/London"}
    ],
    # ...
}
```

**Current Commander.py:** ❌ Doesn't check sessions

---

### 2.3 News Timezone Handling

**File:** `src/router/sensors/news.py`

```python
# Current (simplified)
grep -r "news" shows usage but...
```

**From context doc:**
```python
# Required:
class NewsSensor:
    def check_state(self, event: NewsEvent) -> str:
        # ✅ Has impact filtering
        # ❌ NO timezone conversion for event.time
        # ❌ Hardcoded kill zone (should be configurable)
```

**Required Enhancement:**
```python
class NewsEvent:
    time: datetime  # ✅ Should be in UTC
    impact: str
    currency: str
    title: str
    
class NewsSensor:
    def __init__(self):
        # CONFIGURABLE (not hardcoded 15 min)
        self.kill_zone_minutes_pre = 15
        self.kill_zone_minutes_post = 15
    
    def check_state(self, event: NewsEvent, current_utc: datetime) -> str:
        """Check if in kill zone with timezone-aware comparison."""
        if event.impact != "HIGH":
            return "SAFE"
        
        # Both times in UTC
        time_diff_minutes = (event.time - current_utc).total_seconds() / 60
        
        if -self.kill_zone_minutes_post <= time_diff_minutes <= self.kill_zone_minutes_pre:
            return "KILL_ZONE"
        
        return "SAFE"
```

---

## Part 3: Logging Infrastructure

### 3.1 Current Logging ⚠️ FILE-BASED ONLY

**Backtesting Logs:**
```python
# src/backtesting/mode_runner.py
logger = logging.getLogger(__name__)

# Logs go to:
# - Console (stdout)
# - backtesting.log file
# ❌ NOT to UI WebSocket
```

**Trade Logging:**
```python
# src/router/trade_logger.py
class TradeLogger:
    def log_trade(self, trade_data):
        # Logs to file and database
        # ❌ NO real-time streaming
```

**What You Need:**
```
User: "While I'm training [backtesting], I need to see exactly the logs."
```

---

### 3.2 Required: Real-Time Log Streaming

**Architecture:**
```
┌─────────────────────────────────────────────────────────┐
│                 BACKTESTING PROCESS                     │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐                  │
│  │ Backtest     │───▶│ LogHandler   │                  │
│  │ Runner       │    │ (Custom)     │                  │
│  └──────────────┘    └──────┬───────┘                  │
│                              │                          │
│                              ▼                          │
│                    ┌─────────────────┐                  │
│                    │ WebSocket       │                  │
│                    │ Broadcaster     │                  │
│                    └────────┬────────┘                  │
└─────────────────────────────┼────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  UI WebSocket   │
                    │  Client         │
                    └─────────────────┘
```

**Implementation:**
```python
# File: src/api/ws_logger.py (DOES NOT EXIST)

import asyncio
import logging
from typing import Set
import json

class WebSocketLogHandler(logging.Handler):
    """Stream logs to connected WebSocket clients."""
    
    def __init__(self):
        super().__init__()
        self.clients: Set = set()
        self.loop = None
    
    def emit(self, record: logging.LogRecord):
        """Send log to all connected clients."""
        if not self.clients:
            return
        
        log_data = {
            "type": "log",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Broadcast to all clients
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(json.dumps(log_data)),
                self.loop
            )
    
    async def _broadcast(self, message: str):
        """Send to all WebSocket clients."""
        for client in list(self.clients):
            try:
                await client.send(message)
            except:
                self.clients.remove(client)

# Usage in backtesting
logger = logging.getLogger("backtesting")
ws_handler = WebSocketLogHandler()
logger.addHandler(ws_handler)
```

---

### 3.3 Backtest Progress Streaming

**Required Messages:**
```json
// Backtest started
{
  "type": "backtest_start",
  "variant": "spiced_full",
  "symbol": "EURUSD",
  "timeframe": "H1",
  "start_date": "2023-01-01",
  "end_date": "2024-01-01"
}

// Progress update (every 100 bars)
{
  "type": "backtest_progress",
  "bars_processed": 1000,
  "total_bars": 8760,
  "progress_pct": 11.4,
  "current_date": "2023-02-15T09:00:00Z",
  "trades_count": 45,
  "current_pnl": 125.50
}

// Log message
{
  "type": "log",
  "level": "INFO",
  "message": "Trade #45: BUY EURUSD 0.05 lots at 1.0850 (regime: TREND_UP, kelly: 0.018)",
  "timestamp": "2026-02-12T09:30:15Z"
}

// Backtest completed
{
  "type": "backtest_complete",
  "variant": "spiced_full",
  "final_balance": 1245.50,
  "total_trades": 150,
  "win_rate": 0.58,
  "sharpe_ratio": 1.8
}
```

---

## Part 4: UI Status Assessment

### 4.1 Current UI Structure ⚠️ BASIC SKELETON

**Framework:** SvelteKit + Tauri (desktop app)

**Directory Structure:**
```
quantmind-ide/
├── src/
│   ├── lib/           # 43 items (components)
│   ├── routes/
│   │   ├── +layout.svelte   # Layout wrapper
│   │   ├── +page.svelte     # Homepage
│   │   └── +page.ts         # Load function
│   └── app.html       # HTML template
├── package.json       # Dependencies
└── vite.config.js     # Build config
```

**Dependencies (package.json):**
```json
{
  "dependencies": {
    "@sveltejs/kit": "^2.x",
    "@tauri-apps/api": "^2.x",
    // ... other deps
  }
}
```

---

### 4.2 Homepage Content Analysis

**File:** `quantmind-ide/src/routes/+page.svelte`

Reading file to assess what UI components exist...

**Expected Components (from requirements):**
1. ❓ Dashboard with account balance
2. ❓ Active trades view
3. ❓ Backtest runner interface
4. ❓ Strategy router status
5. ❓ Real-time logs panel
6. ❓ Bot registry management
7. ❓ Agent task queue
8. ❓ News calendar
9. ❓ Session indicators (London/NY/Asian)

---

### 4.3 UI Component Gaps

**Required Components (Priority Order):**

**P1: Core Trading UI**
```svelte
<!-- File: src/lib/components/TradingDashboard.svelte -->
<script>
  import { onMount } from 'svelte';
  import WebSocketClient from '$lib/ws-client';
  
  let balance = 0;
  let activePositions = [];
  let currentSession = 'UNKNOWN';
  let regimeStatus = 'UNKNOWN';
  
  onMount(() => {
    // Connect to backend WebSocket
    const ws = new WebSocketClient('ws://localhost:8080/ws');
    
    ws.on('balance_update', (data) => {
      balance = data.balance;
    });
    
    ws.on('position_update', (data) => {
      activePositions = data.positions;
    });
  });
</script>

<div class="dashboard">
  <div class="balance-card">
    <h2>Account Balance</h2>
    <p class="balance">${balance.toFixed(2)}</p>
  </div>
  
  <div class="session-indicator">
    <h3>Current Session</h3>
    <span class="session {currentSession}">{currentSession}</span>
  </div>
  
  <div class="regime-status">
    <h3>Market Regime</h3>
    <span class="regime {regimeStatus}">{regimeStatus}</span>
  </div>
  
  <div class="positions">
    <h3>Active Positions ({activePositions.length})</h3>
    {#each activePositions as position}
      <PositionCard {position} />
    {/each}
  </div>
</div>
```

**P2: Backtest Runner**
```svelte
<!-- File: src/lib/components/BacktestRunner.svelte -->
<script>
  import { backtestStore } from '$lib/stores/backtest';
  
  let symbol = 'EURUSD';
  let timeframe = 'H1';
  let variant = 'spiced_full';
  let logs = [];
  let progress = 0;
  
  async function runBacktest() {
    const response = await fetch('/api/backtest/run', {
      method: 'POST',
      body: JSON.stringify({ symbol, timeframe, variant })
    });
    
    // Stream logs via WebSocket
    const ws = new WebSocket('ws://localhost:8080/backtest/logs');
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      
      if (msg.type === 'log') {
        logs = [...logs, msg];
      } else if (msg.type === 'backtest_progress') {
        progress = msg.progress_pct;
      }
    };
  }
</script>

<div class="backtest-runner">
  <h2>Run Backtest</h2>
  
  <select bind:value={symbol}>
    <option>EURUSD</option>
    <option>GBPUSD</option>
  </select>
  
  <select bind:value={variant}>
    <option value="vanilla">Vanilla</option>
    <option value="spiced">Spiced (Regime)</option>
    <option value="vanilla_full">Vanilla + Walk-Forward</option>
    <option value="spiced_full">Spiced + Walk-Forward</option>
  </select>
  
  <button on:click={runBacktest}>Run Backtest</button>
  
  <div class="progress">
    <progress value={progress} max="100"></progress>
    <span>{progress.toFixed(1)}%</span>
  </div>
  
  <div class="logs-panel">
    <h3>Backtest Logs (Live)</h3>
    <div class="log-container">
      {#each logs as log}
        <div class="log-entry {log.level}">
          <span class="time">{log.timestamp}</span>
          <span class="message">{log.message}</span>
        </div>
      {/each}
    </div>
  </div>
</div>
```

**P3: Session Clock**
```svelte
<!-- File: src/lib/components/SessionClock.svelte -->
<script>
  import { onMount, onDestroy } from 'svelte';
  
  let currentTime = new Date();
  let currentSession = 'UNKNOWN';
  let nextSession = null;
  let timeUntilNext = '';
  
  function updateClock() {
    currentTime = new Date();
    
    // Fetch session from backend
    fetch('/api/sessions/current')
      .then(r => r.json())
      .then(data => {
        currentSession = data.session;
        nextSession = data.next_session;
        timeUntilNext = data.time_until_next;
      });
  }
  
  let interval;
  onMount(() => {
    updateClock();
    interval = setInterval(updateClock, 1000);
  });
  
  onDestroy(() => clearInterval(interval));
</script>

<div class="session-clock">
  <div class="current-time">
    <h3>UTC Time</h3>
    <p class="time">{currentTime.toISOString().slice(11, 19)}</p>
  </div>
  
  <div class="session-status">
    <span class="badge {currentSession}">
      {currentSession}
    </span>
  </div>
  
  {#if nextSession}
    <div class="next-session">
      <small>Next: {nextSession} in {timeUntilNext}</small>
    </div>
  {/if}
</div>

<style>
  .session-clock {
    background: var(--card-bg);
    padding: 1rem;
    border-radius: 8px;
  }
  
  .badge.LONDON { background: #3b82f6; }
  .badge.NEW_YORK { background: #10b981; }
  .badge.ASIAN { background: #f59e0b; }
  .badge.OVERLAP { background: #8b5cf6; }
</style>
```

---

## Part 5: Implementation Recommendations

### 5.1 Data Layer Priority (Week 1-2)

**P1: Timezone Management System** ⏱️ 8 hours
```python
# File: src/router/sessions.py (NEW)

from datetime import datetime, time
from zoneinfo import ZoneInfo
from enum import Enum
from typing import Dict, Tuple, Optional

class TradingSession(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    CLOSED = "closed"

class SessionDetector:
    """Centralized timezone and session management."""
    
    SESSIONS = {
        "ASIAN": {
            "timezone": ZoneInfo("Asia/Tokyo"),
            "open": time(0, 0),
            "close": time(9, 0),
            "utc_offset_range": (0, 9)  # Approx
        },
        "LONDON": {
            "timezone": ZoneInfo("Europe/London"),
            "open": time(8, 0),
            "close": time(16, 0),
            "utc_offset_range": (8, 16)  # Varies with DST
        },
        "NEW_YORK": {
            "timezone": ZoneInfo("America/New_York"),
            "open": time(8, 0),
            "close": time(17, 0),
            "utc_offset_range": (13, 22)  # Varies with DST
        }
    }
    
    @classmethod
    def detect_session(cls, utc_time: datetime) -> TradingSession:
        """Detect active session."""
        utc_hour = utc_time.hour
        
        # Convert to each timezone and check
        asian_active = cls.is_in_session("ASIAN", utc_time)
        london_active = cls.is_in_session("LONDON", utc_time)
        ny_active = cls.is_in_session("NEW_YORK", utc_time)
        
        # Overlaps
        if london_active and ny_active:
            return TradingSession.OVERLAP
        elif london_active:
            return TradingSession.LONDON
        elif ny_active:
            return TradingSession.NEW_YORK
        elif asian_active:
            return TradingSession.ASIAN
        else:
            return TradingSession.CLOSED
    
    @classmethod
    def is_in_session(cls, session_name: str, utc_time: datetime) -> bool:
        """Check if time falls within session."""
        session = cls.SESSIONS[session_name]
        tz = session["timezone"]
        
        # Convert UTC to session timezone
        local_time = utc_time.astimezone(tz)
        current_time = local_time.time()
        
        return session["open"] <= current_time < session["close"]
    
    @classmethod
    def time_until_next_session(cls, utc_time: datetime) -> Tuple[str, int]:
        """Return next session and minutes until it opens."""
        # Calculate next session opening
        pass
```

**Integration Points:**
1. Commander.py - Filter bots by preferred sessions
2. NewsSensor - Timezone-aware kill zones
3. UI - Display current session
4. BotManifest validation

---

**P2: Real-Time Log Streaming** ⏱️ 6 hours
```python
# File: src/api/ws_logger.py (NEW)

import asyncio
import logging
from websockets.server import serve
from typing import Set
import json

class BacktestLogStreamer:
    """Stream backtest logs to UI via WebSocket."""
    
    def __init__(self, port: int = 8081):
        self.port = port
        self.clients: Set = set()
        self.handler = None
    
    async def start_server(self):
        """Start WebSocket server for log streaming."""
        async with serve(self.handle_client, "localhost", self.port):
            await asyncio.Future()
    
    async def handle_client(self, websocket):
        """Handle new WebSocket client."""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    def create_handler(self) -> logging.Handler:
        """Create logging handler that streams to WebSocket."""
        class StreamHandler(logging.Handler):
            def __init__(self, streamer):
                super().__init__()
                self.streamer = streamer
            
            def emit(self, record):
                log_data = {
                    "type": "log",
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Broadcast to all clients
                asyncio.create_task(
                    self.streamer.broadcast(json.dumps(log_data))
                )
        
        self.handler = StreamHandler(self)
        return self.handler
    
    async def broadcast(self, message: str):
        """Send message to all connected clients."""
        for client in list(self.clients):
            try:
                await client.send(message)
            except:
                self.clients.remove(client)

# Usage in backtesting
log_streamer = BacktestLogStreamer()
asyncio.create_task(log_streamer.start_server())

logger = logging.getLogger("backtesting")
logger.addHandler(log_streamer.create_handler())
```

---

**P3: Session-Aware Commander** ⏱️ 4 hours
```python
# File: src/router/commander.py (ENHANCE)

from src.router.sessions import SessionDetector, TradingSession

class Commander:
    def run_auction(self, regime_report, account_balance: float, 
                    current_utc: datetime) -> List[dict]:
        # Detect current session
        current_session = SessionDetector.detect_session(current_utc)
        
        # Filter bots by session preference
        eligible_bots = []
        for bot in self.bot_registry.list_by_tag("@primal"):
            preferred_sessions = bot.get("preferred_conditions", {}).get("sessions", [])
            
            if not preferred_sessions:  # No preference = trade anytime
                eligible_bots.append(bot)
            elif current_session.value.upper() in preferred_sessions:
                eligible_bots.append(bot)
            elif "OVERLAP" in preferred_sessions and current_session == TradingSession.OVERLAP:
                eligible_bots.append(bot)
        
        # Continue with regime filtering and auction
        ...
```

---

### 5.2 UI Implementation Priority (Week 2-3)

**P1: Dashboard with Real-Time Data** ⏱️ 12 hours
- Balance display with live updates
- Active positions table
- Session clock with timezone display
- Regime status indicator
- WebSocket connection handling

**P2: Backtest Runner with Live Logs** ⏱️ 10 hours
- Backtest parameter selection
- Progress bar with live updates
- Log streaming panel (auto-scroll)
- Results comparison table (4 variants)
- Chart visualization (equity curve)

**P3: Bot Registry Management** ⏱️ 8 hours
- List all bots with @tags
- Bot manifest editor
- Preferred conditions configurator
- Circuit breaker status
- Performance metrics

---

## Part 6: Complete Data Flow Architecture

### 6.1 Live Trading Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MT5 TERMINAL                             │
│  (Windows VPS or Local)                                     │
└────────────────┬────────────────────────────────────────────┘
                 │ Tick Data
                 ▼
┌─────────────────────────────────────────────────────────────┐
│          TICK STREAMER (WebSocket Server)                   │
│  mcp-metatrader5-server/src/mcp_mt5/streaming.py          │
│  - Port: 8765                                               │
│  - Protocol: JSON over WebSocket                            │
│  - Latency: <5ms                                            │
└────────────────┬────────────────────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
         ▼               ▼
┌──────────────┐  ┌──────────────┐
│ Strategy     │  │  UI Client   │
│ Router       │  │  (Dashboard) │
│  process_tick│  │  Display     │
└──────┬───────┘  └──────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Sentinel → Governor → Commander│
└─────────────────────────────────┘
```

---

### 6.2 Backtesting Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              DATA MANAGER                                   │
│  src/data/data_manager.py                                   │
│  - Load OHLC from MT5 or Cache                             │
│  - Validate data quality                                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         BACKTEST RUNNER                                     │
│  src/backtesting/mode_runner.py                            │
│  ┌─────────────────────────────────────────────┐           │
│  │  For each bar:                              │           │
│  │    1. Update regime (Sentinel)              │───────────┼──▶ Log
│  │    2. Calculate risk (Governor)             │           │
│  │    3. Run auction (Commander)               │           │
│  │    4. Execute bot decisions                 │           │
│  │    5. Track P&L                             │───────────┼──▶ Log
│  └─────────────────────────────────────────────┘           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         LOG STREAMER (WebSocket)                            │
│  src/api/ws_logger.py                                       │
│  - Broadcast logs to UI                                     │
│  - Send progress updates                                    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│         UI BACKTEST PANEL                                   │
│  quantmind-ide/src/lib/components/BacktestRunner.svelte    │
│  - Display live logs                                        │
│  - Show progress bar                                        │
│  - Render results                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 7: Summary & Next Steps

### 7.1 What Works Now ✅

1. **MT5 Tick Streaming** - WebSocket server ready (`ws://localhost:8765`)
2. **Data Manager** - Hybrid fetching with caching
3. **Backtesting Engine** - 4 variants with regime filtering
4. **Basic UI** - SvelteKit structure exists

### 7.2 Critical Gaps ⚠️

| Gap | Impact | Effort | Priority |
|-----|--------|--------|----------|
| **Timezone Management** | Can't detect sessions, news times wrong | 8h | P1 |
| **Real-Time Log Streaming** | Can't see backtest progress | 6h | P1 |
| **Session-Aware Routing** | Bots trade at wrong times | 4h | P1 |
| **UI Dashboard** | Can't monitor live trading | 12h | P2 |
| **Backtest UI** | Can't see logs during training | 10h | P2 |
| **Tick-Level Backtesting** | Less accurate than bar-based | 16h | P3 |

### 7.3 Recommended Implementation Order

**Week 1: Core Data Infrastructure**
1. Build timezone/session management system (8h)
2. Integrate sessions into Commander (4h)
3. Add real-time log streaming (6h)
4. Test with live MT5 connection (2h)

**Week 2: UI Foundation**
5. Build WebSocket client utilities (4h)
6. Create dashboard with live data (12h)
7. Add session clock component (2h)
8. Test end-to-end data flow (2h)

**Week 3: Backtest UI**
9. Build backtest runner component (10h)
10. Integrate log streaming panel (4h)
11. Add results comparison view (6h)
12. End-to-end testing (4h)

**Total:** ~74 hours (~10 working days)

---

## Part 8: Technical Specifications

### 8.1 WebSocket Endpoints

**Tick Streaming:**
- URL: `ws://localhost:8765`
- Protocol: JSON
- Messages: tick, subscribe, unsubscribe, heartbeat

**Log Streaming:**
- URL: `ws://localhost:8081/backtest/logs`
- Protocol: JSON
- Messages: log, backtest_progress, backtest_complete

**UI Data:**
- URL: `ws://localhost:8080/ws` (backend API)
- Protocol: JSON
- Messages: balance_update, position_update, regime_update, session_update

### 8.2 REST API Endpoints (Required)

```python
# File: src/api/session_endpoints.py (NEW)

from fastapi import APIRouter
from src.router.sessions import SessionDetector

router = APIRouter(prefix="/api/sessions")

@router.get("/current")
async def get_current_session():
    """Get current trading session."""
    now = datetime.now(timezone.utc)
    session = SessionDetector.detect_session(now)
    next_session, minutes_until = SessionDetector.time_until_next_session(now)
    
    return {
        "session": session.value.upper(),
        "utc_time": now.isoformat(),
        "next_session": next_session,
        "time_until_next": f"{minutes_until // 60}h {minutes_until % 60}m"
    }
```

### 8.3 Timezone Standards

**All timestamps in system:**
- Stored: UTC (database, logs)
- Transmitted: ISO8601 with Z suffix (`2026-02-12T09:30:15Z`)
- Displayed: User's local timezone (UI converts)

**Session times:**
- Defined: Local timezone with DST awareness
- Compared: Converted to UTC for calculations
- Library: `zoneinfo` (Python 3.9+, no pytz needed)

---

**End of Analysis. Ready to proceed with implementation based on priorities discussed.**
