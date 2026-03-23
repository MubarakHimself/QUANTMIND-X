# Story 3.0: Live Trading Backend & MT5 Bridge Audit

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer starting Epic 3,
I want a complete audit of the live trading backend, MT5 bridge, WebSocket infrastructure, and kill switch state,
So that stories 3.1–3.7 target real existing code rather than building blindly.

## Acceptance Criteria

1. **Given** the backend in `src/`,
   **When** the audit runs,
   **Then** a findings document covers:
   - (a) MT5 ZMQ bridge implementation state and connection management
   - (b) existing kill switch classes (ProgressiveKillSwitch, SmartKillSwitch, BotCircuitBreaker) and their API exposure
   - (c) WebSocket endpoint(s) for position/P&L streaming
   - (d) Sentinel → Governor → Commander pipeline wiring state
   - (e) session mask and Islamic compliance implementation
   - (f) existing `TradingFloorPanel.svelte` and live trading UI component state

2. **Notes:**
   - Scan targets: `src/router/`, `src/trading/`, `src/bridge/`, `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
   - Confirm: Cloudzy-only routers include kill switch, MT5 bridge, execution, sentinel
   - Read-only exploration — no code changes

## Tasks / Subtasks

- [x] Task 1 (AC: 1a) - MT5 ZMQ Bridge Implementation Audit
  - [x] Subtask 1.1 - Locate and document MT5 ZMQ bridge code in src/
  - [x] Subtask 1.2 - Document connection management and reconnection logic
  - [x] Subtask 1.3 - Note any existing MT5 tick normalization to UTC
- [x] Task 2 (AC: 1b) - Kill Switch Classes Audit
  - [x] Subtask 2.1 - Find ProgressiveKillSwitch, SmartKillSwitch, BotCircuitBreaker
  - [x] Subtask 2.2 - Document API endpoints exposing kill switch functionality
  - [x] Subtask 2.3 - Note tier implementations (Soft Stop, Strategy Pause, Emergency Close)
- [x] Task 3 (AC: 1c) - WebSocket Infrastructure Audit
  - [x] Subtask 3.1 - Locate WebSocket endpoint(s) for trading data
  - [x] Subtask 3.2 - Document event types (position_update, pnl_update, regime_change, bot_status_change)
  - [x] Subtask 3.3 - Note reconnection and state replay patterns
- [x] Task 4 (AC: 1d) - Sentinel → Governor → Commander Pipeline Audit
  - [x] Subtask 4.1 - Map pipeline flow and component interactions
  - [x] Subtask 4.2 - Document routing matrix integration
- [x] Task 5 (AC: 1e) - Session Mask & Islamic Compliance Audit
  - [x] Subtask 5.1 - Document session mask implementation
  - [x] Subtask 5.2 - Document Islamic compliance (force-close 21:45 UTC)
  - [x] Subtask 5.3 - Note daily loss cap enforcement mechanisms
- [x] Task 6 (AC: 1f) - Frontend Live Trading Components Audit
  - [x] Subtask 6.1 - Audit TradingFloorPanel.svelte state and patterns
  - [x] Subtask 6.2 - Document existing live trading UI component inventory

## Dev Notes

### Project Structure Notes

**Target scan directories:**
- `src/router/` - All routing and endpoint definitions
- `src/api/` - API endpoints (trading_endpoints.py, kill_switch_endpoints.py, websocket_endpoints.py)
- `src/risk/` - Risk engine components (Governor, Commander)
- `quantmind-ide/src/lib/components/` - Frontend components

**Cloudzy-only components (must work without Contabo):**
- Kill switch routers
- MT5 bridge
- Execution routers
- Sentinel → Governor → Commander pipeline

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine & Trading: lines 958-1051
  - §13 Kill Switch: lines 1723-1800
  - §7.4 Session Registry: lines 1003-1030
- Epics: `_bmad-output/planning-artifacts/epics.md` - Epic 3 lines 836-854
- Scan targets: `src/`, `quantmind-ide/src/`

### Key Architecture Patterns

1. **MT5 ZMQ Bridge** - MetaTrader 5 on Windows/Wine only; bridge runs on Cloudzy
   - Actual location: `src/api/tick_stream_handler.py` (not `src/bridge/`)
2. **ZMQ reconnect** - ≤10s target reconnection time (NFR-R5)
3. **UTC normalization** - Architecture requires MT5 tick timestamps normalized to UTC at ingest
   - **AUDIT FINDING**: Not explicitly verified in current implementation - needs verification in Story 3.1
4. **Islamic compliance** - force-close 21:45 GMT enforced at Commander level AND in every EA template
   - **AUDIT FINDING**: NOT IMPLEMENTED - no session_mask or 21:45 UTC force-close found
5. **Kill switch tiers**:
   - Tier 1: Soft Stop (no new entries)
   - Tier 2: Strategy Pause (pause specific strategies)
   - Tier 3: Emergency Close (close all positions)
6. **WebSocket events**: `position_update`, `pnl_update`, `regime_change`, `bot_status_change`, `bridge_status`

### Existing Code to Verify

| Component | Verified Location | Purpose |
|-----------|-------------------|----------|
| ProgressiveKillSwitch | `src/router/progressive_kill_switch.py` | 5-tier kill switch orchestrator |
| SmartKillSwitch | `src/router/kill_switch.py` | Regime-aware exits |
| BotCircuitBreaker | `src/router/bot_circuit_breaker.py` | Per-bot circuit breaker (3 losses) |
| MT5Bridge | `src/api/tick_stream_handler.py` | ZMQ tick streaming (NOT in src/bridge/) |
| TradingFloorPanel | `quantmind-ide/src/lib/components/TradingFloorPanel.svelte` | Live trading UI (partial) |
| WebSocket endpoint | `src/api/websocket_endpoints.py` | `/ws/trading` for streaming |

### Known NFR Constraints

- **NFR-P1**: Kill switch protocol executes in full, in order — never skip steps
- **NFR-P2**: ≤3s lag on live data
- **NFR-R3**: Last known state replayed before live updates resume
- **NFR-R4**: Cloudzy must trade without Contabo reachable
- **NFR-R5**: MT5 ZMQ reconnect ≤10s target
- **NFR-D2**: Immutable audit log for kill switch activations
- **NFR-S3**: API keys in `.env` only — never in source code

## Dev Agent Record

### Agent Model Used
- claude-opus-4-6 (for exploration/audit)

### Debug Log References

### Completion Notes List
**AUDIT FINDINGS:**

**Task 1: MT5 ZMQ Bridge Implementation Audit**
- Status: PARTIALLY IMPLEMENTED
- Finding: No `src/bridge/` directory exists
- ZMQ references found in:
  - `src/api/tick_stream_handler.py` - tick streaming via ZMQ
  - `src/router/socket_server.py` - socket server for EA communication
  - `src/mql5/Experts/TickStreamer.mq5` - MT5 side ZMQ publisher
  - `src/mql5/Include/QuantMind/Utils/Sockets.mqh` - MQL5 ZMQ wrapper
- Connection management logic exists in tick_stream_handler.py but no dedicated MT5Bridge class found

**Task 2: Kill Switch Classes Audit**
- Status: FULLY IMPLEMENTED
- Found components:
  - `src/router/progressive_kill_switch.py` - ProgressiveKillSwitch (5-tier system: Bot→Strategy→Account→Session→System)
  - `src/router/kill_switch.py` - SmartKillSwitch with regime-aware exits
  - `src/router/bot_circuit_breaker.py` - BotCircuitBreakerManager (3 consecutive losses quarantine)
- API endpoints: `src/api/kill_switch_endpoints.py` exposes:
  - GET /status, /tiers, /alerts, /alerts/history, /config
  - POST /reset-tier, /reactivate-family, /account/reset, /system/reset-shutdown, /config
- Tier implementations:
  - Tier 1 (Soft Stop): Bot circuit breaker
  - Tier 2 (Strategy Pause): Strategy family quarantine
  - Tier 3 (Emergency Close): Account-level daily/weekly limits

**Task 3: WebSocket Infrastructure Audit**
- Status: FULLY IMPLEMENTED
- Found: `src/api/websocket_endpoints.py`
- ConnectionManager class with:
  - Topic subscription system (pooled connections)
  - Heartbeat (30s interval)
  - Broadcasting by topic or global
- Event types not explicitly named but infrastructure supports: position_update, pnl_update, regime_change, bot_status_change

**Task 4: Sentinel → Governor → Commander Pipeline Audit**
- Status: FULLY IMPLEMENTED
- Pipeline flow:
  - `src/router/sentinel.py` - Intelligence layer, aggregates sensor data into RegimeReport
    - Sensors: Chaos, Regime, Correlation, News
    - Produces: regime, chaos_score, susceptibility, news_state
  - `src/router/governor.py` (class Governor line 47) - Decision layer
  - `src/router/commander.py` (class Commander line 40) - Execution layer
- Routing matrix integration via `src/router/engine.py`

**Task 5: Session Mask & Islamic Compliance Audit**
- Status: NOT CLEARLY IMPLEMENTED
- No explicit session mask implementation found
- No 21:45 UTC force-close found in current codebase
- Daily loss cap: Implemented in AccountMonitor (Tier 3 of ProgressiveKillSwitch)
- Note: Islamic compliance may need to be added as new implementation

**Task 6: Frontend Live Trading Components Audit**
- Status: PARTIALLY IMPLEMENTED
- Found: `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- Current state: Tab container with Floor Manager and Copilot tabs
- Note: Actual live trading display (bot status grid, position/P&L streaming UI) appears to not exist yet - may need implementation in stories 3.4+

### File List
- `src/router/progressive_kill_switch.py` - 5-tier kill switch orchestrator
- `src/router/kill_switch.py` - SmartKillSwitch with regime-aware exits
- `src/router/bot_circuit_breaker.py` - Per-bot circuit breaker
- `src/api/kill_switch_endpoints.py` - Kill switch API endpoints
- `src/api/websocket_endpoints.py` - WebSocket infrastructure
- `src/router/sentinel.py` - Sentinel intelligence layer
- `src/router/governor.py` - Governor decision layer
- `src/router/commander.py` - Commander execution layer
- `src/router/socket_server.py` - Socket server for EA communication
- `src/api/tick_stream_handler.py` - Tick streaming handler
- `src/mql5/Experts/TickStreamer.mq5` - MT5 ZMQ publisher
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte` - Frontend panel (partial)

### Change Log
- 2026-03-17: Audit completed - documented all findings
- 2026-03-17: Code review - fixed file path inaccuracies, clarified audit findings

## Review Follow-ups (from code review)
- [ ] [AI-Review][Medium] UTC normalization at ZMQ ingest not verified - add verification to Story 3.1
- [ ] [AI-Review][High] Islamic compliance (21:45 force-close) NOT IMPLEMENTED - Stories 3.3 should address this gap
- [ ] [AI-Review][Low] Session mask implementation not found - Stories 3.3 should implement

## Status: done
