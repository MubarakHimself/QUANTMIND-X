# Story 5.4: FloorManager Agent Wiring

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer wiring the Copilot,
I want the Copilot to route messages through the actual FloorManager agent (Claude Opus via Claude Agent SDK),
So that user messages receive intelligent responses and department delegation begins working.

## Acceptance Criteria

**Given** a configured and active Opus-tier provider exists (from Epic 2),
**When** I send a message via Copilot,
**Then** the message passes to the FloorManager agent instance via Claude Agent SDK,
**And** the FloorManager responds using the Opus-tier model.

**Given** FloorManager receives a task belonging to a department (e.g., "run research on GBPUSD"),
**When** it routes the task,
**Then** the appropriate Department Head agent is invoked via Redis Streams task dispatch,
**And** a routing notification appears in the conversation: "Delegating to Research Department."

**Given** the agent SDK call fails (timeout, rate limit, provider error),
**When** the error occurs,
**Then** the conversation shows: "Agent error: [reason]. Retry?" with a retry button,
**And** the error is logged to the audit trail,
**And** the system degrades gracefully — never crashes (NFR-I1).

## Tasks / Subtasks

- [x] Task 1 (AC: #1 - FloorManager SDK Integration)
  - [x] Subtask 1.1: Wire CopilotPanel.svelte to `/api/floor-manager/chat` endpoint (NOT `/api/chat/send`)
  - [x] Subtask 1.2: Pass canvas_context metadata to FloorManager API
  - [x] Subtask 1.3: Verify Opus-tier provider integration via provider config
- [x] Task 2 (AC: #2 - Department Routing)
  - [x] Subtask 2.1: Ensure FloorManager.classify_task() routes to correct department - EXISTS with keyword matching
  - [x] Subtask 2.2: Implement Redis Streams task dispatch for department delegation - EXISTS via delegate_to_department()
  - [x] Subtask 2.3: Add routing notification messages in conversation - ADDED in frontend response handling
- [x] Task 3 (AC: #3 - Error Handling)
  - [x] Subtask 3.1: Add error handling for provider timeouts, rate limits - Added in FloorManager.chat()
  - [x] Subtask 3.2: Display "Agent error: [reason]. Retry?" with retry button in UI - Added in frontend
  - [x] Subtask 3.3: Log errors to audit trail - Added [AUDIT] logging in FloorManager.chat()
  - [x] Subtask 3.4: Implement graceful degradation (never crash) - Added try/catch in FloorManager.chat()

## Dev Notes

### Previous Story Intelligence (Story 5.3)

Story 5.3 completed the Canvas Context System:
- Created `CanvasContextTemplate` YAML files per canvas
- Created `canvasContextService.ts` in frontend
- Added canvas_context metadata to API calls in CopilotPanel.svelte
- Created backend API endpoints (`/api/canvas-context/*`)
- **Known Bug:** CopilotPanel.svelte sends to `/api/chat/send` (legacy) instead of `/api/floor-manager/chat` — **MUST FIX in this story**

### Architecture Prerequisites (CRITICAL)

- **FloorManager** exists at `src/agents/departments/floor_manager.py` — already has:
  - `chat()` method for non-streaming chat
  - `chat_stream()` method for streaming
  - `classify_task()` for department routing
  - `delegate()` for explicit delegation
  - WebSocket support via `/api/floor-manager/ws`
- **Department Heads** exist at `src/agents/departments/heads/`
- **DepartmentMailService** at `src/agents/departments/department_mail.py` — for inter-department messaging
- **Provider config** from Epic 2 — Opus-tier model configured

### FloorManager API Endpoints (Already Exist)

- `POST /api/floor-manager/chat` — Main chat endpoint (use THIS)
- `GET /api/floor-manager/ws` — WebSocket for streaming
- `POST /api/floor-manager/delegate` — Explicit delegation
- `GET /api/floor-manager/status` — FloorManager status
- `GET /api/floor-manager/departments` — List departments
- `POST /api/floor-manager/classify` — Classify task without executing

### Key Implementation Details

1. **Fix CopilotPanel.svelte bug**: Change API call from `/api/chat/send` to `/api/floor-manager/chat`
2. **Include conversation history**: Pass prior turns to FloorManager for context
3. **Department routing**: FloorManager uses keyword-based classification to determine department
4. **Three-tier priority**: HIGH/MEDIUM/LOW via session_id namespace
5. **Redis Streams**: Department Head agents invoked via Redis Streams task dispatch

### File Structure Requirements

```
quantmind-ide/src/lib/components/trading-floor/
├── CopilotPanel.svelte    [MODIFY] Fix /api/chat/send → /api/floor-manager/chat
└── AgentPanel.svelte      [EXTEND] Add FloorManager response handling

src/api/
├── floor_manager_endpoints.py  [EXISTS] Already wired
└── server.py                  [VERIFY] floor_manager_router included

# Error handling (NEW)
src/agents/departments/
├── error_handler.py           [NEW] Error handling utilities
└── audit_logger.py            [NEW] Audit trail logging
```

### Testing Standards

- Integration tests for `/api/floor-manager/chat` endpoint
- Error handling tests (timeout, rate limit, provider error)
- Department routing tests
- Frontend component tests for error display
- WebSocket streaming tests

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§10 Department System)
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.4)
- Previous stories:
  - `_bmad-output/implementation-artifacts/5-3-canvas-context-system-canvascontexttemplate-per-department.md`
- FloorManager: `src/agents/departments/floor_manager.py`
- API endpoints: `src/api/floor_manager_endpoints.py`
- Frontend: `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte`
- Project Context: `_bmad-output/project-context.md` (§2.1 Agent Paradigm - Department System is CANONICAL)

### Known Bug (MUST FIX)

**CopilotPanel.svelte** currently sends to `/api/chat/send` (legacy) instead of `/api/floor-manager/chat` (canonical). This story MUST fix this bug. The fix involves:
1. Change the API endpoint URL
2. Update the request/response handling for the new endpoint format

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (via Claude Code)

### Implementation Notes

**Task 1.1 & 1.2 - FIXED: CopilotPanel.svelte wired to /api/floor-manager/chat**
- Changed API endpoint from `/chat/workshop/message` and `/chat/floor-manager/message` to `/floor-manager/chat`
- Updated request body format to use `context` object with `canvas_context` and `session_id`
- Added `stream: false` to request
- Added error handling for agent errors with user-friendly message: "Agent error: [reason]. Retry?"
- Response handling now supports both `content` and `reply` fields

**Task 1.3 - Provider Verification**
- FloorManager at `src/agents/departments/floor_manager.py` uses `model_tier = "opus"` (line 92)
- Provider config at `src/database/models/provider_config.py` has `tier_assignment` field
- Epic 2 already configured provider tiers

**Task 2 - Department Routing**
- Added `chat()` method to FloorManager at line ~424
- Uses `classify_task()` for keyword-based department classification
- Uses `delegate_to_department()` for Redis Streams task dispatch via DepartmentMailService
- Frontend already handles delegation notifications in response

**Task 3 - Error Handling**
- Added error handling in FloorManager.chat() with try/catch
- Frontend displays "Agent error: [reason]. Retry?" message
- Errors logged to audit trail with [AUDIT] prefix
- Graceful degradation - never crashes, returns error in response

### Debug Log References

- Backend FloorManager: `src/agents/departments/floor_manager.py`
- API endpoints: `src/api/floor_manager_endpoints.py`
- Frontend CopilotPanel: `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte`
- Provider config: `src/agents/providers/` (Epic 2)

### Completion Notes List

### File List

- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` - Modified API endpoint and request/response handling
- `src/agents/departments/floor_manager.py` - Added chat() method with classification and delegation
- `src/api/floor_manager_endpoints.py` - Added history parameter to ChatRequest

### Code Review Fixes Applied

**HIGH Priority Fixes (Story 5.4):**
1. **Added Retry Button** - CopilotPanel.svelte now shows a clickable "Retry" button after agent errors
2. **Added Conversation History** - Messages now include conversation history (last 10 messages) in API requests

**Files Modified:**
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte`:
  - Added `lastFailedMessage` state variable for retry
  - Added `retry` field to message interface
  - Added `retryLastMessage()` function
  - Added retry button in error message UI
  - Added CSS for retry button
  - Added conversation history to API request body

- `src/api/floor_manager_endpoints.py`:
  - Added `history` parameter to ChatRequest model

- `src/agents/departments/floor_manager.py`:
  - Added `history` parameter to `chat()` method signature

