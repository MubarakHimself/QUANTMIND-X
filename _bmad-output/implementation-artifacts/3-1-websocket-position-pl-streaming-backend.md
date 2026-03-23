# Story 3.1: WebSocket Position P/L Streaming Backend

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader,
I want to receive live position updates and P&L data via WebSocket in real-time,
so that I can monitor my active bots, open positions, and daily P&L with ≤3 second lag.

## Acceptance Criteria

1. **Given** the MT5 bridge is connected and receiving tick data,
   **When** a position is opened, modified, or closed on any bot,
   **Then** a `position_update` event is broadcast via WebSocket within ≤3 seconds

2. **Given** the trading session is active,
   **When** P&L calculations are updated (on each tick or position change),
   **Then** a `pnl_update` event is broadcast via WebSocket within ≤3 seconds

3. **Given** a client connects to the WebSocket endpoint,
   **When** the connection is established,
   **Then** the client receives the last known state (positions, P&L, bot statuses) immediately upon connection (state replay per NFR-R3)

4. **Given** the WebSocket connection drops,
   **When** the client reconnects,
   **Then** the client receives full state replay before live updates resume (per NFR-R3)

5. **Given** the MT5 ZMQ connection drops,
   **When** the system detects disconnection (≤10s per NFR-R5),
   **Then** a `bridge_status` event is broadcast with connection state

6. **Given** the system is running on Cloudzy,
   **When** Contabo is unreachable,
   **Then** WebSocket streaming continues independently (per NFR-R4)

## Tasks / Subtasks

- [x] Task 1 (AC: 1, 2) - Position & P&L Event Streaming
  - [x] Subtask 1.1 - Implement position_update event emission
  - [x] Subtask 1.2 - Implement pnl_update event emission
  - [x] Subtask 1.3 - Wire tick_stream_handler to WebSocket broadcaster
- [x] Task 2 (AC: 3, 4) - State Replay on Reconnection
  - [x] Subtask 2.1 - Implement last-known-state caching
  - [x] Subtask 2.2 - Implement state replay on client connect/reconnect
  - [x] Subtask 2.3 - Handle edge cases (empty state, stale state)
- [x] Task 3 (AC: 5) - Bridge Status Events
  - [x] Subtask 3.1 - Monitor ZMQ connection state
  - [x] Subtask 3.2 - Emit bridge_status events on connection changes
- [x] Task 4 (AC: 6) - Cloudzy-Independent Operation
  - [x] Subtask 4.1 - Ensure WebSocket runs on Cloudzy only
  - [x] Subtask 4.2 - Verify no Contabo dependency for trading data
- [x] Task 5 - Testing & Validation
  - [x] Subtask 5.1 - Write unit tests for event emission
  - [x] Subtask 5.2 - Write integration tests for state replay
  - [x] Subtask 5.3 - Verify ≤3s latency under load

## Dev Notes

### Project Structure Notes

**Files to modify/create:**
- `src/api/websocket_endpoints.py` - Extend existing ConnectionManager
- `src/api/tick_stream_handler.py` - Wire to WebSocket broadcaster
- `quantmind-ide/src/lib/ws-client.ts` - Frontend WebSocket client (may need updates)
- `tests/api/test_websocket_streaming.py` - New test file

**Existing WebSocket infrastructure:**
- `src/api/websocket_endpoints.py` - ConnectionManager with topic subscription
- Topic-based broadcasting: pool connections by topic for efficient delivery

**Key classes:**
- `ConnectionManager` - Manages WebSocket connections and topic subscriptions
- `TickStreamHandler` - Receives ZMQ ticks from MT5 (existing)
- Need to create: `TradingDataBroadcaster` - Emits trading events

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine & Trading: lines 958-1051 (P&L streaming requirements)
  - §2.2 Data Streams: WebSocket for Cloudzy trading data (line 317)
  - §13 Integration: WebSocket direct from Cloudzy (line 1531)
  - NFR constraints: ≤3s lag (line 51), state replay (line 2931)
- Story 3-0 (audit): `_bmad-output/implementation-artifacts/3-0-live-trading-backend-mt5-bridge-audit.md`
- UX: `_bmad-output/planning-artifacts/ux-design-specification.md`

### Technical Requirements

1. **WebSocket endpoint**: `/ws/trading` on Cloudzy node
2. **Event types**:
   - `position_update` - position open/modify/close
   - `pnl_update` - live P&L calculation
   - `bot_status_change` - bot state changes
   - `bridge_status` - MT5 connection state
   - `regime_change` - regime classification updates

3. **Latency requirements**:
   - Event emission ≤3s from trigger (NFR-P2)
   - ZMQ reconnect ≤10s (NFR-R5)

4. **State management**:
   - Cache last known state in memory (positions, P&L, bot statuses)
   - Replay state on client connect (NFR-R3)
   - Handle empty/stale state edge cases

5. **Independence**:
   - Must work without Contabo reachable (NFR-R4)
   - All trading data stays on Cloudzy

### Implementation Patterns

**Event emission pattern:**
```python
# In tick_stream_handler.py or new broadcaster
from src.api.websocket_endpoints import get_connection_manager

async def emit_position_update(position: Position):
    conn_mgr = get_connection_manager()
    await conn_mgr.broadcast(
        topic="trading",
        event_type="position_update",
        data=position.to_dict()
    )
```

**State replay pattern:**
```python
# On client connect/reconnect
async def on_client_connect(websocket, pool):
    state = get_cached_trading_state()
    await websocket.send_json({
        "event_type": "state_snapshot",
        "data": state
    })
```

**ZMQ connection monitoring:**
```python
# In tick_stream_handler.py
class TickStreamHandler:
    async def check_connection(self):
        # Monitor ZMQ connection
        # Emit bridge_status events on state change
```

### Previous Story Learnings

Story 3-0 audit findings:
- WebSocket infrastructure exists but event types not explicitly implemented
- ConnectionManager supports topic-based broadcasting
- Story 3-1 is the first implementation story - builds on audit findings
- UTC normalization at ZMQ ingest needs verification (noted in 3-0 review follow-ups)

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
- `src/api/websocket_endpoints.py` [extend] - ConnectionManager with topic subscription, broadcast functions
- `src/api/tick_stream_handler.py` [modify] - Added broadcast_position_change, broadcast_pnl_change, report_bridge_status methods
- `tests/api/test_websocket_streaming.py` [new] - Unit and integration tests (19 tests)

### Change Log
- 2026-03-17: Implemented position_update and pnl_update event broadcasting (Task 1)
- 2026-03-17: Implemented state caching and replay on connect via /ws/trading endpoint (Task 2)
- 2026-03-17: Implemented bridge_status event emission for ZMQ connection monitoring (Task 3)
- 2026-03-17: WebSocket runs independently on Cloudzy - no Contabo dependencies (Task 4)
- 2026-03-17: Added comprehensive tests for all event types and state replay (Task 5)
- 2026-03-17: [AI-Review Fix] Wired tick_stream_handler to broadcast functions - added broadcast_position_change(), broadcast_pnl_change(), report_bridge_status() methods that integrate with the broadcasting system

### Completion Notes
- All acceptance criteria satisfied:
  - AC1: position_update events broadcast within ≤3s
  - AC2: pnl_update events broadcast within ≤3s
  - AC3: State snapshot sent immediately on client connect
  - AC4: State replay on reconnection via /ws/trading endpoint
  - AC5: bridge_status events emit on ZMQ connection changes
  - AC6: WebSocket operates independently on Cloudzy (no Contabo dependency)
- Latency validation: All broadcasts complete well under 3s threshold
- Tests: 19 unit/integration tests passing covering all new functionality including integration with TickStreamHandler
