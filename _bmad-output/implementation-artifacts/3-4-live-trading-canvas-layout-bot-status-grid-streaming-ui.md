# Story 3.4: Live Trading Canvas — Layout, Bot Status Grid & Streaming UI

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader monitoring active bots,
I want the Live Trading canvas to display all active bot statuses, live P&L, and open positions streaming in real time,
so that I have complete operational awareness of every running EA at a glance.

## Acceptance Criteria

**Given** I navigate to the Live Trading canvas (default home canvas),
**When** the canvas loads,
**Then** a GlassTile grid renders with one tile per active EA,
**And** each tile shows: EA name, symbol, current P&L (colour-coded), open position count, current regime, session active/inactive,
**And** all tiles use Tier 2 glass (`rgba(8,13,20,0.35)`, `blur(16px)`).

**Given** the WebSocket stream is active,
**When** position or P&L data arrives,
**Then** the affected tiles update within ≤3s,
**And** P&L changes flash green (`#00c896`, 100ms) or red (`#ff3b3b`, 100ms).

**Given** I expand a bot tile (click),
**When** the sub-page opens,
**Then** BreadcrumbNav appears: `Live Trading > [EA Name]`,
**And** the sub-page shows: session mask (active sessions highlighted), force_close_hour, overnight_hold, daily loss cap bar, current equity exposure.

## Tasks / Subtasks

- [x] Task 1 (AC: GlassTile Grid) - Bot Status Grid Layout
  - [x] Subtask 1.1 - Create GlassTile component with Tier 2 glass styling
  - [x] Subtask 1.2 - Implement grid layout for multiple active EAs
  - [x] Subtask 1.3 - Add bot card showing: EA name, symbol, P&L, position count, regime, session status
- [x] Task 2 (AC: WebSocket Integration) - Real-time Streaming
  - [x] Subtask 2.1 - Connect to WebSocket `/ws/trading` endpoint
  - [x] Subtask 2.2 - Subscribe to position_update, pnl_update, regime_change, bot_status_change events
  - [x] Subtask 2.3 - Implement ≤3s update latency
  - [x] Subtask 2.4 - Add P&L flash animations (green/red, 100ms)
- [x] Task 3 (AC: Expanded Bot Detail) - Bot Detail Sub-page
  - [x] Subtask 3.1 - Add BreadcrumbNav component (`Live Trading > [EA Name]`)
  - [x] Subtask 3.2 - Display session mask with active sessions highlighted
  - [x] Subtask 3.3 - Show force_close_hour and overnight_hold
  - [x] Subtask 3.4 - Add daily loss cap progress bar
  - [x] Subtask 3.5 - Display current equity exposure
- [x] Task 1 (AC: GlassTile Grid) - Bot Status Grid Layout
- [x] Task 2 (AC: WebSocket Integration) - Real-time Streaming
- [x] Task 3 (AC: Expanded Bot Detail) - Bot Detail Sub-page

### Review Follow-ups (AI)

- [ ] [AI-Review][MEDIUM] Add unit tests for GlassTile component rendering
- [ ] [AI-Review][MEDIUM] Add WebSocket connection/reconnection tests
- [ ] [AI-Review][MEDIUM] Add P&L flash animation tests
- [ ] [AI-Review][LOW] Add integration tests for bot detail navigation

## Dev Notes

### Project Structure Notes

**Files to create:**
- `quantmind-ide/src/lib/components/live-trading/GlassTile.svelte` - NEW component for glass tile
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - NEW grid container
- `quantmind-ide/src/lib/components/live-trading/BotStatusCard.svelte` - NEW individual bot card
- `quantmind-ide/src/lib/components/live-trading/BotDetailPage.svelte` - NEW expanded view
- `quantmind-ide/src/lib/stores/trading.ts` - NEW store for live trading state

**Files to modify:**
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` - EXTEND with GlassTile grid
- `quantmind-ide/src/lib/stores/` - Add WebSocket connection logic

**Existing infrastructure:**
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` - Existing canvas (may need updates)
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte` - Legacy panel (reference only)
- API: `/api/v1/trading/bots` - Bot status endpoint (story 3-3)
- API: `/api/v1/trading/bots/{bot_id}/params` - Bot params endpoint (story 3-3)
- WebSocket: `/ws/trading` - Streaming endpoint (story 3-1)

**Key architectural constraints:**
- Tier 2 glass: `rgba(8,13,20,0.35)` with `blur(16px)`
- P&L flash colors: green `#00c896`, red `#ff3b3b` (100ms duration)
- Update latency: ≤3s (NFR-P2)
- BreadcrumbNav format: `Live Trading > [EA Name]`
- Default home canvas: Live Trading canvas
- All timestamps: UTC with `_utc` suffix

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine: real-time monitoring
  - Frosted Terminal design system (Tier 2 glass)
- Epic 3 context: `_bmad-output/planning-artifacts/epics.md` (lines 947-975)
- UX Design: `_bmad-output/planning-artifacts/ux-design-specification.md`
- Story 3-1: `_bmad-output/implementation-artifacts/3-1-websocket-position-pl-streaming-backend.md` - WebSocket streaming
- Story 3-2: `_bmad-output/implementation-artifacts/3-2-kill-switch-backend-all-tiers.md` - Kill switch
- Story 3-3: `_bmad-output/implementation-artifacts/3-3-session-mask-islamic-compliance-loss-cap-apis.md` - Bot params API

## Dev Agent Guardrails

### Technical Requirements

1. **GlassTile Component:**
   - Background: `rgba(8,13,20,0.35)`
   - Backdrop filter: `blur(16px)`
   - Border: 1px solid with subtle opacity
   - Structure: header + metric body + timestamp footer

2. **Bot Status Card Display:**
   - EA name (bot identifier)
   - Symbol being traded
   - Current P&L (colour-coded: green positive, red negative)
   - Open position count
   - Current regime (ASIAN, LONDON, NEW_YORK, OVERLAP, CLOSED)
   - Session active/inactive indicator

3. **WebSocket Integration:**
   - Connect to `/ws/trading`
   - Listen for: `position_update`, `pnl_update`, `regime_change`, `bot_status_change`
   - Include `timestamp_utc` in all events
   - Implement reconnection with exponential backoff (from story 3-1)
   - Replay last known state on reconnection (NFR-R3)

4. **Real-time Updates:**
   - Tile updates within ≤3s of data arrival
   - P&L flash animation: green `#00c896` for gains, red `#ff3b3b` for losses
   - Animation duration: 100ms

5. **Bot Detail Sub-page:**
   - BreadcrumbNav: `Live Trading > [EA Name]`
   - Session mask: highlight active sessions
   - force_close_hour: from bot params (story 3-3)
   - overnight_hold: from bot params
   - Daily loss cap bar: visual progress indicator
   - Current equity exposure

6. **Skeleton Loading:**
   - Show skeleton pulse until first data arrives
   - No jarring blank-then-populated transitions

### Architecture Compliance

- Use Svelte 5 runes for reactivity
- Follow Frosted Terminal aesthetic (Tier 2 glass)
- Connect to existing WebSocket from story 3-1
- Use bot params API from story 3-3
- All timestamps must use `_utc` suffix
- Design: Hyprland-style glass - near-transparent fills + heavy backdrop blur

### Library/Framework Requirements

- Svelte 5 components
- Svelte stores for state management
- WebSocket client for real-time data
- Lucide icons for status indicators (NOT emoji)
- CSS backdrop-filter for glass effect

### File Structure Requirements

```
quantmind-ide/src/lib/
├── components/
│   └── live-trading/
│       ├── GlassTile.svelte       [NEW] - Glass tile container component
│       ├── BotStatusGrid.svelte   [NEW] - Grid layout for bot tiles
│       ├── BotStatusCard.svelte   [NEW] - Individual bot status card
│       ├── BotDetailPage.svelte   [NEW] - Expanded bot detail view
│       └── BreadcrumbNav.svelte   [NEW] - Breadcrumb navigation
├── stores/
│   └── trading.ts                 [NEW] - Live trading state store
└── components/canvas/
    └── LiveTradingCanvas.svelte  [MODIFY] - Integrate GlassTile grid
```

### Testing Requirements

- Unit tests for GlassTile rendering
- Test WebSocket connection and reconnection
- Test P&L flash animations trigger correctly
- Test bot detail navigation
- Test skeleton loading states
- Verify ≤3s update latency

## Previous Story Intelligence

From Story 3-3 (Session Mask, Islamic Compliance & Loss Cap APIs):
- Bot params endpoint available at `GET /api/v1/trading/bots/{bot_id}/params`
- Returns: session_mask, force_close_hour, overnight_hold, daily_loss_cap, current_loss_pct, islamic_compliance, swap_free
- Force-close countdown available when within 60 min of 21:45 UTC
- Loss cap breach events broadcast via WebSocket

**Apply to Story 3-4:**
- Use bot params endpoint for detail page
- Session mask data available for display
- Loss cap progress bar data available
- Use existing WebSocket for real-time updates

From Story 3-1 (WebSocket Streaming):
- WebSocket endpoint at `/ws/trading`
- Events: position_update, pnl_update, regime_change, bot_status_change
- Reconnection with exponential backoff
- State replay on reconnection

**Apply to Story 3-4:**
- Connect to existing WebSocket
- Listen for position/P&L updates
- Update tiles in real-time

## Git Intelligence Summary

Recent work in Epic 3:
- Story 3-1: WebSocket streaming implementation
- Story 3-2: Kill switch tiers with audit logging
- Story 3-3: Bot params API for session/Islamic/loss cap

Pattern observed:
- Extend existing endpoints/APIs rather than create new files
- Use Svelte 5 with runes
- Follow Frosted Terminal aesthetic
- Backend APIs are in place - focus is on frontend UI

## Latest Tech Information

- Frontend: Svelte 5 with runes for reactivity
- Design System: Frosted Terminal (Tier 2 glass: `rgba(8,13,20,0.35)`, `blur(16px)`)
- Icons: Lucide (NOT emoji)
- WebSocket: Already implemented in story 3-1
- Bot params API: Already implemented in story 3-3
- LiveTradingCanvas.svelte exists - needs GlassTile integration

## Project Context Reference

- Project: QUANTMINDX
- Primary goal: Trading command center with real-time monitoring
- Design aesthetic: Frosted Terminal (glass with backdrop blur)
- UI: Svelte 5 components
- NFR-P2: ≤3s lag on live data
- Architecture: Cloudzy WebSocket works without Contabo reachable (NFR-R4)

## File List

### New Files
- `quantmind-ide/src/lib/stores/trading.ts` - Live trading state store with WebSocket integration
- `quantmind-ide/src/lib/components/live-trading/GlassTile.svelte` - Glass tile container component (Tier 2)
- `quantmind-ide/src/lib/components/live-trading/BotStatusGrid.svelte` - Grid layout for bot cards
- `quantmind-ide/src/lib/components/live-trading/BotStatusCard.svelte` - Individual bot status display
- `quantmind-ide/src/lib/components/live-trading/BotDetailPage.svelte` - Expanded bot detail view
- `quantmind-ide/src/lib/components/live-trading/BreadcrumbNav.svelte` - Breadcrumb navigation

### Modified Files
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` - Integrated GlassTile grid
- `quantmind-ide/src/lib/stores/index.ts` - Added trading store exports
- `quantmind-ide/src/lib/components/settings/index.ts` - Added ServersPanel export (fix)

## Dev Agent Record

### Agent Model Used
Claude Opus 4.6

### Debug Log References

### Completion Notes List

**Implementation Complete - Tasks 1-3:**

1. **Task 1: GlassTile Grid Layout**
   - Created GlassTile.svelte with Tier 2 glass styling (`rgba(8,13,20,0.35)`, `blur(16px)`)
   - Created BotStatusGrid.svelte with responsive grid layout and skeleton loading
   - Created BotStatusCard.svelte with all required metrics: EA name, symbol, P&L (color-coded), position count, regime, session status

2. **Task 2: WebSocket Integration**
   - Implemented trading.ts store with WebSocket connection to `/ws/trading`
   - Handles position_update, pnl_update, regime_change, bot_status_change events
   - Implements exponential backoff reconnection
   - P&L flash animations (100ms green/red) on value changes

3. **Task 3: Bot Detail Sub-page**
   - Created BreadcrumbNav.svelte with `Live Trading > [EA Name]` format
   - Created BotDetailPage.svelte with:
     - Session mask grid (24-hour display with active highlighting)
     - Force close hour and overnight hold parameters
     - Daily loss cap progress bar with warning thresholds
     - Equity exposure with current P&L

4. **Updated LiveTradingCanvas.svelte**
   - Integrated all new components
   - Added connection status indicators (Live/Connecting/Error)
   - Added refresh button
   - Toggle between grid view and detail view

**Build Status:** ✅ Successful - frontend builds without errors
