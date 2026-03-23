# Story 3.5: Kill Switch UI — All Tiers

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader managing risk,
I want all kill switch tiers accessible from the TopBar and Live Trading canvas,
so that I can stop any level of activity at any moment from any canvas.

## Acceptance Criteria

**Given** I am on any canvas,
**When** the TradingKillSwitch renders in TopBar,
**Then** it shows Lucide `shield-alert` icon in ready state (grey),
**And** clicking it arms the switch (red pulse, 2s countdown visible),
**And** a second click (or Enter does NOT work — must click Confirm button) opens the confirmation modal.

**Given** the confirm modal is open,
**When** I select a tier and confirm,
**Then** the appropriate kill switch tier API fires,
**And** the TopBar switch shows "FIRED" state (grey, disabled),
**And** recovery requires app restart (TradingKillSwitch scope: Cloudzy MT5 only).

**Given** Tier 3 (Emergency Close) is selected,
**When** the double-confirmation modal renders,
**Then** it shows current open positions, estimated exposure, and "This will close all positions" warning in red border.

## Tasks / Subtasks

- [x] Task 1 (AC: TopBar Kill Switch) - TradingKillSwitch Component
  - [x] Subtask 1.1 - Update TopBar TradingKillSwitch with tier selection
  - [x] Subtask 1.2 - Add arm/disarm states with red pulse animation
  - [x] Subtask 1.3 - Add 2s countdown timer visualization
  - [x] Subtask 1.4 - Add aria-label updates for accessibility
- [x] Task 2 (AC: Confirmation Modal) - Kill Switch Confirmation Modal
  - [x] Subtask 2.1 - Create modal with tier selection (Tier 1/2/3)
  - [x] Subtask 2.2 - Add tier descriptions for each option
  - [x] Subtask 2.3 - Implement confirm/cancel actions
- [x] Task 3 (AC: Tier 3 Double-Confirm) - Emergency Close Modal
  - [x] Subtask 3.1 - Create double-confirmation modal for Tier 3
  - [x] Subtask 3.2 - Display current open positions count
  - [x] Subtask 3.3 - Display estimated exposure amount
  - [x] Subtask 3.4 - Add red border warning
- [x] Task 4 (AC: API Integration) - Backend API Wiring
  - [x] Subtask 4.1 - Wire Tier 1 API call to `/api/kill-switch/trigger`
  - [x] Subtask 4.2 - Wire Tier 2 API call to `/api/kill-switch/trigger`
  - [x] Subtask 4.3 - Wire Tier 3 API call to `/api/kill-switch/trigger`
  - [x] Subtask 4.4 - Handle FIRed state and disable after activation
- [x] Task 5 - Testing & Validation
  - [x] Subtask 5.1 - Test kill switch arm/disarm flow
  - [x] Subtask 5.2 - Test tier selection and confirmation
  - [x] Subtask 5.3 - Test Tier 3 double-confirmation
  - [x] Subtask 5.4 - Test API error handling

## Dev Notes

### Project Structure Notes

**Files to create:**
- `quantmind-ide/src/lib/components/kill-switch/KillSwitchModal.svelte` - NEW confirmation modal
- `quantmind-ide/src/lib/components/kill-switch/EmergencyCloseModal.svelte` - NEW Tier 3 double-confirm
- `quantmind-ide/src/lib/stores/kill-switch.ts` - NEW store for kill switch state

**Files to modify:**
- `quantmind-ide/src/lib/components/TopBar.svelte` - EXTEND TradingKillSwitch with tier selection
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` - ADD kill switch access

**Existing infrastructure:**
- `quantmind-ide/src/lib/components/KillSwitchView.svelte` - Existing kill switch view
- `quantmind-ide/src/lib/components/TopBar.svelte` - Has basic kill switch button
- API: `/api/kill-switch/trigger` - Kill switch trigger endpoint (story 3-2)
- API: `/api/kill-switch/status` - Get current status (story 3-2)
- API: `/api/v1/trading/bots` - Get bot positions for emergency modal

**Key architectural constraints:**
- TradingKillSwitch is TopBar ONLY — never elsewhere (architecture hard rule)
- Distinct from Workflow Kill Switch (per-workflow in FlowForge, Epic 11)
- `aria-label` updates reactively: "Emergency stop — click to arm" → "Armed — click Confirm" → "Trading stopped"
- Recovery requires app restart after firing
- Tier 3 shows: open positions, estimated exposure, red border warning

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine: kill switch implementation
- Epic 3 context: `_bmad-output/planning-artifacts/epics.md` (lines 978-1005)
- Story 3-2: `_bmad-output/implementation-artifacts/3-2-kill-switch-backend-all-tiers.md` - Kill switch backend
- Story 3-4: `_bmad-output/implementation-artifacts/3-4-live-trading-canvas-layout-bot-status-grid-streaming-ui.md` - Live Trading canvas

## Dev Agent Guardrails

### Technical Requirements

1. **TradingKillSwitch Component (TopBar):**
   - Lucide `shield-alert` icon
   - States: ready (grey), armed (red pulse), fired (grey, disabled)
   - Click to arm → 2s countdown → opens confirmation modal
   - `aria-label` updates reactively

2. **Confirmation Modal:**
   - Three tier options with descriptions:
     - Tier 1: Soft Stop — no new positions
     - Tier 2: Strategy Pause — pause specific strategies
     - Tier 3: Emergency Close — close all positions
   - Confirm button fires API
   - Cancel dismisses modal

3. **Tier 3 Double-Confirmation:**
   - Shows current open positions count
   - Shows estimated total exposure
   - Red border warning: "This will close all positions"
   - Two-step confirmation required

4. **API Integration:**
   - GET `/api/kill-switch/status` - Current status
   - POST `/api/kill-switch/trigger` - Trigger any tier (body: {tier, strategy_ids, activator})
   - GET `/api/v1/trading/bots` - Fetch position data for emergency modal

5. **State Management:**
   - Track armed/fired status in store
   - Disable switch after firing
   - Recovery requires app restart

### Architecture Compliance

- Use Svelte 5 runes for reactivity
- Follow Frosted Terminal aesthetic (Tier 2 glass from story 3-4)
- Lucide icons (NOT emoji)
- Tier 3 modal requires double-confirmation
- All timestamps must use `_utc` suffix

### Library/Framework Requirements

- Svelte 5 components
- Svelte stores for state management
- Lucide icons: shield-alert
- CSS animations for pulse/countdown

### File Structure Requirements

```
quantmind-ide/src/lib/
├── components/
│   └── kill-switch/
│       ├── KillSwitchModal.svelte      [NEW] - Tier selection modal
│       └── EmergencyCloseModal.svelte   [NEW] - Tier 3 double-confirm
├── stores/
│   └── kill-switch.ts                  [NEW] - Kill switch state
└── components/
    └── TopBar.svelte                 [MODIFY] - Update TradingKillSwitch
```

### Testing Requirements

- Test kill switch arm flow (click → armed state)
- Test confirmation modal tier selection
- Test Tier 3 double-confirmation flow
- Test API error handling and disabled states
- Verify aria-label updates

## Previous Story Intelligence

From Story 3-4 (Live Trading Canvas):
- GlassTile component available with Tier 2 glass styling
- BotStatusGrid with skeleton loading
- Modal patterns can reuse GlassTile for consistency

From Story 3-2 (Kill Switch Backend):
- Backend API endpoints exist for all three tiers
- ProgressiveKillSwitch with 5 tiers (Tier 1-5) implemented
- Status endpoint available
- Audit logging in place

**Apply to Story 3-5:**
- Wire to existing backend endpoints
- Reuse modal patterns from story 3-4
- Use GlassTile for confirmation modals

## Git Intelligence Summary

Recent work in Epic 3:
- Story 3-1: WebSocket streaming
- Story 3-2: Kill switch backend tiers
- Story 3-3: Bot params API
- Story 3-4: Live Trading canvas UI

Pattern observed:
- Frontend components built in story 3-4
- Backend APIs in place from stories 3-2/3-3
- Focus is on wiring UI to existing APIs

## Latest Tech Information

- Frontend: Svelte 5 with runes
- Design System: Frosted Terminal (Tier 2 glass)
- Icons: Lucide (NOT emoji)
- Kill switch backend: Already implemented in story 3-2
- Modal component patterns: Available from story 3-4

## Project Context Reference

- Project: QUANTMINDX
- Primary goal: Trading command center with real-time monitoring
- Design aesthetic: Frosted Terminal (glass with backdrop blur)
- UI: Svelte 5 components
- TradingKillSwitch: TopBar ONLY (architecture hard rule)
- Recovery: Requires app restart after firing

## File List

**New Files:**
- `quantmind-ide/src/lib/stores/kill-switch.ts` - Kill switch state management store
- `quantmind-ide/src/lib/components/kill-switch/KillSwitchModal.svelte` - Tier selection modal component
- `quantmind-ide/src/lib/components/kill-switch/EmergencyCloseModal.svelte` - Tier 3 double-confirmation modal

**Modified Files:**
- `quantmind-ide/src/lib/components/TopBar.svelte` - Updated TradingKillSwitch with tier selection, countdown, and new state management
- `quantmind-ide/src/lib/stores/index.ts` - Added kill switch store exports

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (via Claude Code)

### Debug Log References

- TopBar component integration: Lines 1-77 (script), 91-143 (template), 238-290 (styles)
- KillSwitchModal: Full component with tier selection UI
- EmergencyCloseModal: Full component with double-confirmation flow
- kill-switch.ts store: State management with Svelte stores pattern

### Completion Notes List

**Code Review Fixes Applied (2026-03-17):**
- Fixed EmergencyCloseModal to fetch real position data from `/api/v1/trading/bots` instead of using mock values
- Added loading state while fetching position data
- Updated API endpoint documentation to match actual implementation

**Implementation Summary (2026-03-17):**

1. **Created kill-switch.ts store** with:
   - State management: `killSwitchState`, `killSwitchCountdown`, `killSwitchFired`
   - Actions: `armKillSwitch()`, `disarmKillSwitch()`, `confirmKillSwitch()`, `triggerKillSwitch()`
   - API integration: Wired to `/api/kill-switch/trigger` endpoint
   - Derived store: `killSwitchAriaLabel` for accessibility

2. **Created KillSwitchModal.svelte** with:
   - Three tier options (Tier 1: Soft Stop, Tier 2: Strategy Pause, Tier 3: Emergency Close)
   - Tier descriptions for each option
   - Loading and error states
   - Frosted Terminal aesthetic matching existing UI

3. **Created EmergencyCloseModal.svelte** with:
   - Double-confirmation flow (two clicks required)
   - Open positions count display
   - Estimated exposure display
   - Red border warning
   - Pulsing animation on final confirmation

4. **Updated TopBar.svelte** with:
   - Integrated new modal components
   - 2-second countdown timer visualization
   - Red pulse animation when armed
   - FIRed state (disabled after activation)
   - Reactive aria-label updates
   - Access to all three tiers

5. **Updated stores/index.ts** with exports for kill-switch store

**Acceptance Criteria Status:**
- ✅ TradingKillSwitch shows `shield-alert` icon in ready state (grey)
- ✅ Clicking arms the switch (red pulse, 2s countdown visible)
- ✅ Second click (after countdown) opens confirmation modal
- ✅ Tier selection modal with all three options
- ✅ Tier 3 shows double-confirmation with positions, exposure, and red warning
- ✅ API integration wired to `/api/kill-switch/trigger`
- ✅ TopBar shows FIRED state (grey, disabled) after activation
- ✅ Recovery requires app restart

**Architecture Compliance:**
- ✅ TradingKillSwitch is TopBar ONLY
- ✅ Uses Lucide icons (shield-alert, shield, shield-x)
- ✅ Svelte 5 syntax with $state and $derived
- ✅ Frosted Terminal aesthetic (Tier 2 glass styling)
- ✅ All timestamps follow _utc suffix convention
