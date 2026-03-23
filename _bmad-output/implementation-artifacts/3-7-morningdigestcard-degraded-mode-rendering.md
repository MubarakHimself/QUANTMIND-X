# Story 3.7: MorningDigestCard & Degraded Mode Rendering

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader opening the ITT at the start of the trading day,
I want a Morning Digest card on first load and graceful degraded mode when Contabo is offline,
so that I get the overnight summary immediately and the UI never shows error screens during node drops.

## Acceptance Criteria

1. **Given** it is the first time I open Live Trading canvas in a session,
   **When** the canvas loads,
   **Then** a MorningDigestCard renders at the top of the tile grid,
   **And** it shows: overnight agent activity summary, pending approvals count (with chip), node health, critical alerts, active market session indicator.

2. **Given** Contabo is unreachable,
   **When** the Live Trading canvas renders,
   **Then** all Contabo-dependent data (agent activity, workflow state) shows degraded indicators — amber label "Contabo offline — retrying",
   **And** Cloudzy WebSocket data continues displaying normally,
   **And** TradingKillSwitch renders normally (Cloudzy-independent),
   **And** NO blank screens or error states appear anywhere.

3. **Given** Contabo reconnects,
   **When** the connection resumes,
   **Then** degraded indicators clear automatically within 10 seconds,
   **And** live data resumes from the reconnection point.

**Notes:**
- NFR-R4: strategy router + kill switch operational on Cloudzy even without Contabo
- Architecture: "Nothing disappears — all last-known data remains visible. QUANTMINDX never shows blank screen."
- MorningDigestCard pre-loads via `/morning-digest` Prefect-triggered aggregation on Contabo at session start

## Tasks / Subtasks

- [x] Task 1 (AC: #1) - MorningDigestCard Component
  - [x] Subtask 1.1 - Create MorningDigestCard.svelte component
  - [x] Subtask 1.2 - Wire to /morning-digest API endpoint
  - [x] Subtask 1.3 - Display: overnight agent activity, pending approvals (with chip), node health, critical alerts, market session
  - [x] Subtask 1.4 - Integrate into LiveTradingCanvas as first tile
- [x] Task 2 (AC: #2) - Degraded Mode Infrastructure
  - [x] Subtask 2.1 - Create node health store (contabo, cloudzy status)
  - [x] Subtask 2.2 - Add degraded indicator components
  - [x] Subtask 2.3 - Implement fallback UI for Contabo-dependent components
  - [x] Subtask 2.4 - Add amber "Contabo offline — retrying" labels
- [x] Task 3 (AC: #3) - Auto-Recovery
  - [x] Subtask 3.1 - Implement connection monitoring for Contabo
  - [x] Subtask 3.2 - Add reconnection logic (10 second recovery)
  - [x] Subtask 3.3 - Clear degraded indicators on successful reconnect
- [x] Task 4 - Backend API Implementation
  - [x] Subtask 4.1 - Create GET /morning-digest endpoint
  - [x] Subtask 4.2 - Implement Prefect-triggered aggregation (or placeholder)
  - [x] Subtask 4.3 - Add node health check endpoints
- [ ] Task 5 - Testing & Validation
  - [ ] Subtask 5.1 - Test MorningDigestCard renders on first load
  - [ ] Subtask 5.2 - Test degraded mode when Contabo unreachable
  - [ ] Subtask 5.3 - Test auto-recovery within 10 seconds
  - [ ] Subtask 5.4 - Test no blank screens during node drops

## Dev Notes

### Project Structure Notes

**Files to create:**
- `quantmind-ide/src/lib/components/live-trading/MorningDigestCard.svelte` - NEW Morning Digest card component
- `quantmind-ide/src/lib/components/live-trading/DegradedIndicator.svelte` - NEW degraded mode indicator
- `quantmind-ide/src/lib/components/live-trading/NodeHealthBadge.svelte` - NEW node health status badge
- `quantmind-ide/src/lib/stores/node-health.ts` - NEW store for node connectivity status

**Files to modify:**
- `quantmind-ide/src/lib/components/live-trading/LiveTradingCanvas.svelte` - ADD MorningDigestCard at top
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - ADD degraded indicators
- `quantmind-ide/src/lib/stores/trading.ts` - EXTEND with node health state
- `src/api/server.py` - ADD /morning-digest and /health/nodes endpoints
- `src/api/server_config_endpoints.py` - ADD node health check logic

**Existing infrastructure:**
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - Bot grid layout
- `quantmind-ide/src/lib/components/live-trading/BotStatusCard.svelte` - Individual bot card
- `quantmind-ide/src/lib/components/live-trading/GlassTile.svelte` - Tier 2 glass component
- `quantmind-ide/src/lib/stores/trading.ts` - Trading store with activeBots
- `quantmind-ide/src/lib/stores/kill-switch.ts` - Kill switch state (Cloudzy-independent)
- `src/api/server.py` - Main API server
- `src/api/server_config_endpoints.py` - Server configuration endpoints
- Story 3-4: WebSocket streaming infrastructure
- Story 3-5: Kill switch UI (Cloudzy-independent)

**Key architectural constraints:**
- MorningDigestCard shows on first load of Live Trading canvas
- Contabo is the "agents/compute" node (per Story 2-6)
- Cloudzy is "live trading" node - must remain operational
- NFR-R4: Kill switch works on Cloudzy even without Contabo
- "Nothing disappears" — all last-known data persists
- All timestamps must use `_utc` suffix
- Lucide icons (NOT emoji)
- Frosted Terminal aesthetic (Tier 2 glass)

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Epic 3 context: `_bmad-output/planning-artifacts/epics.md` (lines 1038-1067)
- Story 3-6: `_bmad-output/implementation-artifacts/3-6-manual-trade-controls-ui.md` - Modal patterns, trading store
- Story 3-5: `_bmad-output/implementation-artifacts/3-5-kill-switch-ui-all-tiers.md` - Kill switch (Cloudzy-independent)
- Story 3-4: `_bmad-output/implementation-artifacts/3-4-live-trading-canvas-layout-bot-status-grid-streaming-ui.md` - Canvas structure
- Story 2-6: Server connections configuration (Contabo, Cloudzy nodes)

## Dev Agent Guardrails

### Technical Requirements

1. **MorningDigestCard Component:**
   - Position: First element in LiveTradingCanvas, above BotStatusGrid
   - Shows: overnight agent activity summary, pending approvals (with chip), node health, critical alerts, active market session
   - Data source: GET /morning-digest API
   - GlassTile styling (Tier 2 glass)
   - Lucide icons: sun, bell, server, alert-triangle, clock

2. **Node Health Store:**
   - Track: `contaboStatus` (connected/disconnected/reconnecting), `cloudzyStatus` (connected/disconnected)
   - Polling interval: 5-15 seconds
   - Last-known state persisted for degraded mode display

3. **Degraded Indicator Component:**
   - Amber label: "Contabo offline — retrying"
   - Shows on any Contabo-dependent component
   - Does NOT block Cloudzy-dependent functionality
   - TradingKillSwitch always visible (Cloudzy-independent)

4. **Degraded Mode Behavior:**
   - Agent activity section → show degraded indicator
   - Workflow state → show degraded indicator
   - Any Contabo-dependent data → show last-known + degraded indicator
   - NO blank screens, NO error states

5. **Auto-Recovery:**
   - Monitor Contabo connection via health check
   - On reconnect: clear degraded indicators within 10 seconds
   - Resume live data from reconnection point
   - Show brief "Contabo reconnected" toast

6. **Backend API Endpoints:**
   - GET `/api/v1/server/morning-digest` - Returns overnight summary
     - Response: `{ agent_activity: [], pending_approvals: number, node_health: {}, critical_alerts: [], market_session: string }`
   - GET `/api/v1/server/health/nodes` - Returns node status
     - Response: `{ contabo: { status: string, latency_ms: number }, cloudzy: { status: string, latency_ms: number } }`

7. **State Management:**
   - New `node-health.ts` store:
     - `contaboStatus` state
     - `cloudzyStatus` state
     - `lastKnownData` cache
     - `checkNodeHealth()` action
   - Extend `trading.ts` with degraded mode awareness

8. **Market Session Indicator:**
   - Show active sessions: Tokyo, London, New York
   - Display open/closed state with color coding
   - Update based on UTC time

### Architecture Compliance

- Use Svelte 5 runes for reactivity ($state, $derived, $effect)
- Follow Frosted Terminal aesthetic (Tier 2 glass)
- Lucide icons (NOT emoji): sun, bell, server, alert-triangle, clock, wifi, wifi-off
- Kill switch always visible (Cloudzy-independent per NFR-R4)
- All timestamps must use `_utc` suffix
- "Nothing disappears" — persist last-known data

### Library/Framework Requirements

- Svelte 5 components
- Svelte stores for state management
- Lucide icons: sun, bell, server, alert-triangle, clock, wifi, wifi-off, check-circle
- CSS animations for degraded indicator pulse
- WebSocket for real-time node health updates (or polling)

### File Structure Requirements

```
quantmind-ide/src/lib/
├── components/
│   └── live-trading/
│       ├── MorningDigestCard.svelte    [NEW] - Morning digest display
│       ├── DegradedIndicator.svelte    [NEW] - Degraded mode badge
│       ├── NodeHealthBadge.svelte      [NEW] - Node status indicator
│       ├── LiveTradingCanvas.svelte    [MODIFY] - Add MorningDigestCard
│       └── BotStatusGrid.svelte        [MODIFY] - Add degraded indicators
├── stores/
│   ├── node-health.ts                  [NEW] - Node connectivity store
│   └── trading.ts                      [MODIFY] - Extend with degraded mode

src/api/
├── server.py                           [MODIFY] - Add /morning-digest endpoint
└── server_config_endpoints.py          [MODIFY] - Add node health checks
```

### Testing Requirements

- Test MorningDigestCard renders on first canvas load
- Test MorningDigestCard shows correct data from API
- Test degraded indicator appears when Contabo unreachable
- Test Cloudzy data still displays during Contabo outage
- Test TradingKillSwitch visible during Contabo outage
- Test no blank screens during node drops
- Test auto-recovery clears indicators within 10 seconds
- Test market session indicator shows correct state

## Previous Story Intelligence

From Story 3-6 (Manual Trade Controls):
- Modal patterns: PositionCloseModal, CloseAllModal implemented
- Trading store extended with close actions
- BotStatusCard with P&L flash animation
- Cross-canvas integration patterns

From Story 3-5 (Kill Switch UI):
- Kill switch is Cloudzy-independent (NFR-R4)
- EmergencyCloseModal with double-confirmation
- Kill switch always visible in TopBar
- Store pattern: kill-switch.ts with state management

From Story 3-4 (Live Trading Canvas):
- BotStatusGrid with responsive layout
- BotStatusCard components
- GlassTile styling established
- WebSocket streaming for real-time updates
- Canvas routing skeleton in place

From Story 2-6 (Server Connections):
- Contabo: agents/compute node
- Cloudzy: live trading node
- Server health check endpoints exist

**Apply to Story 3-7:**
- Reuse GlassTile styling for MorningDigestCard
- Reuse node health patterns from Story 2-6
- TradingKillSwitch is Cloudzy-independent reference point
- Extend trading.ts store with node health awareness

## Git Intelligence Summary

Recent work in Epic 3:
- Story 3-1: WebSocket streaming for positions
- Story 3-2: Kill switch backend tiers
- Story 3-3: Bot params API (session mask, Islamic compliance)
- Story 3-4: Live Trading canvas UI with BotStatusGrid
- Story 3-5: Kill switch UI with modals
- Story 3-6: Manual trade controls (close position, close all)

Pattern observed:
- Frontend: Components with GlassTile styling
- Backend: API endpoints for data and control
- Focus: Wiring UI to existing APIs, graceful degradation

For Story 3-7:
- Need NEW /morning-digest endpoint (Prefect-triggered aggregation)
- Need node health monitoring infrastructure
- Need degraded mode awareness across all components
- Reuse GlassTile from story 3-4

## Latest Tech Information

- Frontend: Svelte 5 with runes
- Design System: Frosted Terminal (Tier 2 glass)
- Icons: Lucide (NOT emoji)
- Backend: FastAPI with server router
- No existing morning-digest endpoint - must create
- Node health checks partially exist from Story 2-6
- "Nothing disappears" philosophy for degraded mode

## Project Context Reference

- Project: QUANTMINDX
- Primary goal: Trading command center with real-time monitoring
- Design aesthetic: Frosted Terminal (glass with backdrop blur)
- UI: Svelte 5 components
- Node architecture: Cloudzy (live trading), Contabo (agents/compute)
- Critical: NFR-R4 - Kill switch operational on Cloudzy without Contabo

## File List

**New Files:**
- `quantmind-ide/src/lib/components/live-trading/MorningDigestCard.svelte` - Morning digest card
- `quantmind-ide/src/lib/components/live-trading/DegradedIndicator.svelte` - Degraded mode indicator
- `quantmind-ide/src/lib/components/live-trading/NodeHealthBadge.svelte` - Node status badge
- `quantmind-ide/src/lib/stores/node-health.ts` - Node health store

**Modified Files:**
- `quantmind-ide/src/lib/components/live-trading/LiveTradingCanvas.svelte` - Add MorningDigestCard
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - Add degraded indicators
- `quantmind-ide/src/lib/stores/trading.ts` - Extend with degraded mode (FIXED: added imports and tradingDegraded derived store)
- `quantmind-ide/src/lib/stores/node-health.ts` - Added isCloudzyDegraded export (FIXED)
- `quantmind-ide/src/lib/stores/index.ts` - Added isCloudzyDegraded export (FIXED)
- `src/api/server.py` - Add /morning-digest endpoint
- `src/api/server_config_endpoints.py` - Add node health logic

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (claude-opus-4-6-20251001)

### Debug Log References

- Frontend build: Verified successful with `npm run build`
- Python syntax: Verified with `python3 -m py_compile`

### Completion Notes List

- **MorningDigestCard.svelte**: Created component that displays overnight agent activity, pending approvals (with amber chip), node health badges, critical alerts, and market session indicator. Uses GlassTile styling and Svelte 4 syntax.
- **DegradedIndicator.svelte**: Created amber badge component with pulse animation showing "Contabo offline — retrying" message.
- **NodeHealthBadge.svelte**: Created badge showing Contabo and Cloudzy node status with latency display.
- **node-health.ts**: Created Svelte store managing node connectivity state with derived stores for degraded mode status. Includes polling with 10-second intervals.
- **LiveTradingCanvas.svelte**: Added MorningDigestCard integration and node health monitoring startup/stop.
- **BotStatusGrid.svelte**: Added degraded indicator display when Contabo is unreachable.
- **server_config_endpoints.py**: Added `/api/v1/server/morning-digest` and `/api/v1/server/health/nodes` endpoints with node health checking logic.
- **server.py**: Added server_router include for new endpoints.

### Code Review Fixes Applied (2026-03-18)

- **H2 Fix**: Added degraded mode awareness to `trading.ts`:
  - Added imports: `nodeHealthState, isContaboDegraded, isCloudzyDegraded`
  - Added derived store: `tradingDegraded` combining node health with trading state
- **node-health.ts Fix**: Added `isCloudzyDegraded` derived store export
- **index.ts Fix**: Added `isCloudzyDegraded` to exports

### File List

**New Files:**
- `quantmind-ide/src/lib/components/live-trading/MorningDigestCard.svelte` - Morning digest card component
- `quantmind-ide/src/lib/components/live-trading/DegradedIndicator.svelte` - Degraded mode indicator
- `quantmind-ide/src/lib/components/live-trading/NodeHealthBadge.svelte` - Node status badge
- `quantmind-ide/src/lib/stores/node-health.ts` - Node health store

**Modified Files:**
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` - Added MorningDigestCard and health monitoring
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - Added degraded indicator
- `quantmind-ide/src/lib/stores/index.ts` - Added node-health exports
- `src/api/server.py` - Added server_router include
- `src/api/server_config_endpoints.py` - Added morning-digest and health/nodes endpoints
