# Story 3.3: Session Mask, Islamic Compliance & Loss Cap APIs

Status: done

## Story

As a developer wiring the live trading canvas,
I want API endpoints exposing EA session mask state, Islamic compliance status, and daily loss cap usage,
So that the canvas can display these parameters transparently without modifying EA behaviour.

## Acceptance Criteria

**Given** `GET /api/trading/bots/{id}/params` is called,
**When** processed,
**Then** it returns: `{ session_mask, force_close_hour, overnight_hold, daily_loss_cap, current_loss_pct, islamic_compliance, swap_free }`.

**Given** the current time is within 60 minutes of the force-close time (21:45 UTC for Islamic mode),
**When** the endpoint is polled,
**Then** the response includes `{ force_close_countdown_seconds }`.

**Given** a daily loss cap is breached,
**When** the event fires,
**Then** an audit log entry records the breach,
**And** a notification event fires to the frontend.

## Tasks / Subtasks

- [x] Task 1 (AC: Bot Params Endpoint) - Bot Parameters API
  - [x] Subtask 1.1 - Create GET /api/trading/bots/{id}/params endpoint
  - [x] Subtask 1.2 - Return session_mask, force_close_hour, overnight_hold, daily_loss_cap, current_loss_pct, islamic_compliance, swap_free
  - [x] Subtask 1.3 - Add bot registry integration for parameter retrieval
- [x] Task 2 (AC: Force Close Countdown) - Force Close Countdown
  - [x] Subtask 2.1 - Add countdown calculation within 60 min of force-close
  - [x] Subtask 2.2 - Return force_close_countdown_seconds when within window
  - [x] Subtask 2.3 - Handle Islamic vs non-Islamic mode
- [x] Task 3 (AC: Loss Cap Breach) - Loss Cap Breach Events
  - [x] Subtask 3.1 - Add audit logging for loss cap breaches
  - [x] Subtask 3.2 - Add WebSocket notification for loss cap breach
  - [x] Subtask 3.3 - Wire to existing event system
- [x] Task 4 - Testing & Validation
  - [x] Subtask 4.1 - Unit tests for bot params endpoint
  - [x] Subtask 4.2 - Test force close countdown calculation
  - [x] Subtask 4.3 - Test loss cap breach events

## Dev Notes

### Project Structure Notes

**Files to modify/create:**
- `src/api/trading_endpoints.py` - Add bot params endpoint (EXISTING - extend)
- `src/router/sessions.py` - Add Islamic compliance functions (EXISTING - extend)
- `tests/api/test_session_loss_cap.py` - New test file

**Existing infrastructure:**
- `src/router/sessions.py` - SessionDetector for trading sessions (ASIAN, LONDON, NEW_YORK, OVERLAP)
- `src/api/trading_endpoints.py` - Bot status endpoint at `/api/v1/trading/bots`
- Bot registry exists via `BotRegistry` in bot_manifest

**Key architectural constraints:**
- Force-close time: 21:45 UTC (hard-coded, not broker time)
- All timestamps must use `_utc` suffix
- EA parameters are read-only in ITT - no overrides (FR40)
- Daily loss cap is display-only - EAs enforce via MQL5 inputs (FR8)

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine: Islamic compliance (line 541)
  - Islamic force-close at 21:45 UTC (line 1047)
  - FR8: daily loss cap display only (line 940)
  - FR39: Islamic compliance display (line 941)
- Epic 3 context: `_bmad-output/planning-artifacts/epics.md` (lines 918-943)
- Story 3-2: `_bmad-output/implementation-artifacts/3-2-kill-switch-backend-all-tiers.md`

## Dev Agent Guardrails

### Technical Requirements

1. **API Endpoint to add:**
   - `GET /api/trading/bots/{bot_id}/params` - Get bot trading parameters
   - Response: `{ bot_id, session_mask, force_close_hour, overnight_hold, daily_loss_cap, current_loss_pct, islamic_compliance, swap_free, force_close_countdown_seconds? }`

2. **Session Mask:**
   - Returns which sessions are active (ASIAN, LONDON, NEW_YORK, OVERLAP, CLOSED)
   - Uses existing SessionDetector from sessions.py

3. **Islamic Compliance:**
   - Force-close at 21:45 UTC (hard-coded per architecture)
   - Returns `islamic_compliance: true/false` and `swap_free: true/false`
   - Add `is_past_islamic_cutoff()` to sessions.py if not exists

4. **Loss Cap:**
   - `daily_loss_cap` - configured cap percentage
   - `current_loss_pct` - current daily loss as percentage
   - Display only - EAs enforce via MQL5 inputs

5. **Force Close Countdown:**
   - Within 60 minutes of 21:45 UTC: return `force_close_countdown_seconds`
   - Calculate seconds until force-close

6. **Loss Cap Breach Events:**
   - Audit log entry when breach detected
   - WebSocket notification to frontend

### Architecture Compliance

- Use UTC timestamps with `_utc` suffix throughout
- Follow existing endpoint patterns in trading_endpoints.py
- Use WebSocket broadcasting from story 3-1 for notifications

### Library/Framework Requirements

- FastAPI for REST endpoints
- SessionDetector for session detection
- BotRegistry for bot parameter retrieval
- WebSocket for loss cap breach notifications

### File Structure Requirements

```
src/
├── api/
│   └── trading_endpoints.py      [MODIFY] - Add bot params endpoint
├── router/
│   └── sessions.py                [MODIFY] - Add Islamic compliance functions
tests/
└── api/
    └── test_session_loss_cap.py  [NEW] - Tests for new functionality
```

### Testing Requirements

- Unit tests for bot params endpoint
- Test Islamic compliance detection
- Test force close countdown calculation
- Test loss cap breach event firing

## Previous Story Intelligence

From Story 3-2 (Kill Switch Backend):
- WebSocket broadcasting available for real-time events
- ConnectionManager pattern for event distribution
- Audit logging pattern established

**Apply to Story 3-3:**
- Use WebSocket to broadcast loss_cap_breached events
- Follow audit logging pattern for breach records

## Git Intelligence Summary

Recent work in Epic 3:
- Story 3-1: WebSocket streaming implementation
- Story 3-2: Kill switch tiers with audit logging
- Pattern: Extend existing endpoints rather than create new files

## Latest Tech Information

- Sessions.py exists with SessionDetector class
- Trading endpoints already have bot status endpoint
- Need to add Islamic compliance methods to sessions.py

## Project Context Reference

- Project: QUANTMINDX
- Primary goal: Trading command center with real-time monitoring
- Design aesthetic: Frosted Terminal (glass with backdrop blur)
- UI: Svelte 5 components

## File List

- `src/api/trading/models.py` - Added BotParamsResponse model
- `src/api/trading/control.py` - Added get_bot_params method
- `src/api/trading/routes.py` - Added bot params and loss cap breach endpoints
- `src/router/sessions.py` - Added Islamic compliance and loss cap breach functions
- `tests/api/test_session_loss_cap.py` - New test file (18 tests)

## Dev Agent Record

### Agent Model Used

Claude MiniMax-M2.5

### Debug Log References

### Completion Notes List

- Story 3-3 implemented successfully
- Added bot params endpoint at GET /api/v1/trading/bots/{bot_id}/params
- Added Islamic compliance functions: is_past_islamic_cutoff(), get_force_close_countdown_seconds(), is_within_countdown_window()
- Added LossCapAuditLog class for immutable audit logging of breach events
- Added WebSocket broadcast for loss_cap_breached events via existing ConnectionManager
- Added loss cap breach endpoints: GET /api/v1/trading/loss-cap/breaches, GET /api/v1/trading/loss-cap/breaches/{bot_id}
- Created 18 unit tests covering session detection, Islamic compliance, loss cap breach detection
- All tests pass

### File List
