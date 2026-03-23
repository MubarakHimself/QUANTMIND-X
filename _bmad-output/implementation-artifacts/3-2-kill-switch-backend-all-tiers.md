# Story 3.2: Kill Switch Backend — All Tiers

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a risk manager,
I want all three kill switch tiers implemented as atomic, audited backend operations,
So that each tier executes completely and correctly (NFR-P1 — correctness over speed).

## Acceptance Criteria

**Given** Tier 1 (Soft Stop — no new entries) is activated,
**When** the command fires,
**Then** all EAs stop opening new positions immediately,
**And** existing positions remain open,
**And** an immutable audit log entry is written: `{ tier: 1, activated_at_utc, activator }` (NFR-D2).

**Given** Tier 2 (Strategy Pause — pause specific strategies) is activated with strategy IDs,
**When** the command fires,
**Then** only specified strategies pause,
**And** other strategies continue normally.

**Given** Tier 3 (Emergency Close — close all positions) is activated,
**When** the command fires,
**Then** close orders are sent for all open positions via MT5 bridge,
**And** results (filled/partial/rejected) are captured and returned,
**And** the audit log records each close attempt and outcome.

## Tasks / Subtasks

- [x] Task 1 (AC: Tier 1) - Soft Stop Implementation
  - [x] Subtask 1.1 - Extend KillSwitch API for Tier 1 (soft stop - no new entries)
  - [x] Subtask 1.2 - Add audit logging with NFR-D2 compliance
  - [x] Subtask 1.3 - Wire to MT5 bridge for trade blocking
- [x] Task 2 (AC: Tier 2) - Strategy Pause Implementation
  - [x] Subtask 2.1 - Implement strategy-specific pause endpoint
  - [x] Subtask 2.2 - Add strategy ID validation
  - [x] Subtask 2.3 - Test isolation (paused vs running strategies)
- [x] Task 3 (AC: Tier 3) - Emergency Close Implementation
  - [x] Subtask 3.1 - Implement close-all-positions via MT5 bridge
  - [x] Subtask 3.2 - Handle partial fills and rejections
  - [x] Subtask 3.3 - Add per-position close audit logging
- [x] Task 4 (AC: All) - Atomic Execution & Audit
  - [x] Subtask 4.1 - Ensure all tiers execute in full, in order (NFR-P1)
  - [x] Subtask 4.2 - Add immutable audit log entries per NFR-D2
  - [x] Subtask 4.3 - Verify Cloudzy-independence (NFR-R4)
- [x] Task 5 - Testing & Validation
  - [x] Subtask 5.1 - Unit tests for each tier
  - [x] Subtask 5.2 - Integration tests with MT5 bridge
  - [x] Subtask 5.3 - Verify atomic execution and audit logging

## Dev Notes

### Project Structure Notes

**Files to modify/create:**
- `src/api/kill_switch_endpoints.py` - Add Tier 1/2/3 trigger endpoints (EXISTING - extend)
- `src/router/kill_switch.py` - Add tier-specific trigger methods to KillSwitch class (EXISTING - extend)
- `src/router/progressive_kill_switch.py` - Check for existing progressive tier implementation (may need new tiers)
- `tests/api/test_kill_switch_tiers.py` - New test file

**Existing kill switch infrastructure:**
- `src/router/kill_switch.py` - KillSwitch and SmartKillSwitch classes exist
  - Basic KillSwitch with CLOSE_ALL command
  - SmartKillSwitch with regime-aware exits
  - HALT_NEW_TRADES command exists
- `src/api/kill_switch_endpoints.py` - ProgressiveKillSwitch endpoints exist
  - Uses ProgressiveKillSwitch with 5 tiers (Tier 1-5)
  - Status, alerts, reset endpoints exist

**Key architectural constraints:**
- Kill switch runs on Cloudzy only (must work without Contabo) - NFR-R4
- All timestamps must use `_utc` suffix (naming convention)
- Tier execution must be atomic and complete - NFR-P1
- Audit logs must be immutable - NFR-D2

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
  - §7 Risk Engine & Trading: Kill switch tiers (lines ~347, 542, 670)
  - §2.2 Data Streams: WebSocket for trading data (line 317)
  - NFR-P1: Kill switch executes in full, in order (line 912)
  - NFR-R4: Kill switch operational on Cloudzy without Contabo (line 1064)
- Epic 3 context: `_bmad-output/planning-artifacts/epics.md` (lines 886-914)
- Story 3-1: `_bmad-output/implementation-artifacts/3-1-websocket-position-pl-streaming-backend.md`
- Story 3-0 (audit): `_bmad-output/implementation-artifacts/3-0-live-trading-backend-mt5-bridge-audit.md`

## Dev Agent Guardrails

### Technical Requirements

1. **API Endpoints to add:**
   - `POST /api/kill-switch/trigger` - Trigger specific tier
   - Request body: `{ tier: 1|2|3, strategy_ids?: string[], activator: string }`
   - Response: `{ success: bool, tier: number, audit_log_id: string, results?: CloseResult[] }`

2. **Tier Implementation Details:**

   **Tier 1 (Soft Stop):**
   - Send `HALT_NEW_TRADES` command to all registered EAs
   - Existing positions remain open
   - Create audit log entry: `{ tier: 1, activated_at_utc, activator, action: "HALT_NEW_TRADES" }`

   **Tier 2 (Strategy Pause):**
   - Accept list of strategy IDs to pause
   - Send `PAUSE_STRATEGY` command with strategy IDs
   - Other strategies continue normally
   - Create audit log entry: `{ tier: 2, activated_at_utc, activator, strategy_ids: [] }`

   **Tier 3 (Emergency Close):**
   - Send `CLOSE_ALL` command to all connected EAs
   - Collect close results per position
   - Create audit log entry per close attempt: `{ tier: 3, position_id, action: "CLOSE", result: "filled|partial|rejected", pnl: number }`

3. **Atomic Execution (NFR-P1):**
   - Each tier must complete ALL steps before returning
   - No early returns on partial success
   - Audit log written AFTER all actions complete

4. **Audit Logging (NFR-D2):**
   - All entries immutable once written
   - Include: timestamp_utc, tier, activator, actions taken, results
   - Store in database with append-only semantics

5. **Cloudzy Independence (NFR-R4):**
   - All kill switch logic runs on Cloudzy node
   - No dependency on Contabo being reachable
   - Test by verifying no cross-node API calls

### Architecture Compliance

- Follow existing `KillSwitch` class patterns from `src/router/kill_switch.py`
- Use `KillReason` and `ExitStrategy` enums from existing code
- Maintain consistency with ProgressiveKillSwitch in kill_switch_endpoints.py
- Use UTC timestamps with `_utc` suffix throughout

### Library/Framework Requirements

- FastAPI for REST endpoints (already in use)
- SQLAlchemy or existing DB layer for audit logs
- WebSocket broadcast for real-time status (built in story 3-1)
- MT5 ZMQ bridge for trade commands (existing infrastructure)

### File Structure Requirements

```
src/
├── api/
│   └── kill_switch_endpoints.py  [MODIFY] - Add tier trigger endpoints
├── router/
│   └── kill_switch.py            [MODIFY] - Add tier-specific trigger methods
tests/
└── api/
    └── test_kill_switch_tiers.py [NEW] - Tier-specific tests
```

### Testing Requirements

- Unit tests for each tier trigger method
- Integration tests simulating MT5 bridge responses
- Test atomic execution (no partial completions)
- Test audit log immutability
- Test Cloudzy independence (mock Contabo unavailable)

## Previous Story Intelligence

From Story 3-1 (WebSocket Position & P/L Streaming):
- Created `/ws/trading` endpoint for real-time data
- ConnectionManager supports topic-based broadcasting
- TickStreamHandler integrated with WebSocket broadcaster
- Pattern established: broadcast events to "trading" topic

**Apply to Story 3-2:**
- Kill switch activation should broadcast `kill_switch_activated` event
- Use same ConnectionManager pattern for real-time status updates
- Verify no conflicts with existing WebSocket infrastructure

## Git Intelligence Summary

Recent commits in trading/backend area:
- Story 3-1: WebSocket streaming implementation
- Modified: src/api/websocket_endpoints.py, src/api/tick_stream_handler.py
- Pattern: Use async/await with proper error handling

## Latest Tech Information

- Kill switch classes already exist: KillSwitch, SmartKillSwitch, ProgressiveKillSwitch
- Task is to add TIER-SPECIFIC API endpoints that map to the 3-tier requirement
- ProgressiveKillSwitch has 5 tiers - need to ensure Story 3-2 maps to tiers 1-3 appropriately
- Verify no breaking changes to existing progressive kill switch behavior

## Project Context Reference

- Project: QUANTMINDX
- Primary goal: Trading command center with real-time monitoring
- User: Risk manager needing emergency stop capabilities
- Design aesthetic: Frosted Terminal (glass with backdrop blur)
- UI: Svelte 5 components

## File List

- `src/api/kill_switch_endpoints.py` - Extended with Tier 1/2/3 trigger endpoints and audit logging
- `tests/api/test_kill_switch_tiers.py` - New test file for tier-specific tests

> Note: `src/router/kill_switch.py` was NOT modified - existing KillSwitch class used as-is

## Dev Agent Record

### Implementation Plan

1. **Extended kill_switch_endpoints.py** with:
   - `KillSwitchTriggerRequest` model for API requests
   - `KillSwitchTriggerResponse` model for API responses
   - `CloseResult` model for Tier 3 close results
   - `KillSwitchAuditLog` class for immutable audit logging (NFR-D2)
   - `POST /api/kill-switch/trigger` endpoint for all 3 tiers
   - `GET /api/kill-switch/audit` endpoint for audit log retrieval
   - `GET /api/kill-switch/audit/{audit_id}` endpoint for single entry

2. **Tier Implementation**:
   - Tier 1 (Soft Stop): Sends HALT_NEW_TRADES, blocks new positions
   - Tier 2 (Strategy Pause): Accepts strategy_ids, sends PAUSE_STRATEGY
   - Tier 3 (Emergency Close): Sends CLOSE_ALL, captures results

3. **NFR Compliance**:
   - NFR-P1: Atomic execution - all steps complete before return
   - NFR-D2: Immutable audit logs with UTC timestamps
   - NFR-R4: Works without Contabo (Cloudzy-independent mode)

### Completion Notes

All three kill switch tiers have been implemented with:
- REST API endpoints at `POST /api/kill-switch/trigger`
- Audit logging at `GET /api/kill-switch/audit`
- Atomic execution (no early returns)
- Cloudzy independence (works without socket server)
- Unit tests covering all tiers

## Story Completion Status

**Status:** done

**Next Steps:**
1. Run `code-review` workflow for peer review
2. Story 3-5 (Kill Switch UI) depends on these backend APIs
