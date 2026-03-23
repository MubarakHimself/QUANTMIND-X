# Story 8.5: Human Approval Gates Backend

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader responsible for deployment decisions,
I want mandatory human approval gates at post-backtest and pre-live checkpoints,
So that no strategy deploys without my explicit sign-off (FR28, FR30).

## Acceptance Criteria

**Given** a strategy passes the backtest gate (≥4/6 modes),
**When** the PENDING_REVIEW state is reached,
**Then** an `ApprovalGateBadge` appears in the TopBar with the strategy name,
**And** Mubarak receives a Copilot notification with: strategy name, backtest summary, key metrics, "Approve / Reject / Request Revision" options.

**Given** Mubarak approves,
**When** approval is recorded,
**Then** the approval event is stored as an immutable audit record: `{ strategy_id, approver, approved_at_utc, gate_type, metrics_snapshot }` (NFR-D2).

**Given** Mubarak requests revision,
**When** the revision request is submitted,
**Then** the agent re-executes with Mubarak's feedback as context,
**And** a new approval gate is created — no manual editing of agent artifacts.

**Notes:**
- Approval gate max hold: 15 minutes before `PENDING_REVIEW → EXPIRED_REVIEW` (7-day timeout before considered non-blocking)
- FR28: human approval required before paper trading AND before live deployment
- ApprovalGateBadge pattern: batch-surfaced, accumulates within workflow run

## Tasks / Subtasks

- [x] Task 1: Backend Approval Gate API (AC: all)
  - [x] Subtask 1.1: Extend approval_gate.py for Alpha Forge workflow types
  - [x] Subtask 1.2: Implement strategy_metrics_snapshot in extra_data
  - [x] Subtask 1.3: Add PENDING_REVIEW → EXPIRED_REVIEW timeout logic
- [x] Task 2: Alpha Forge Integration (AC: 1, 2)
  - [x] Subtask 2.1: Wire backtest completion to create approval gate
  - [x] Subtask 2.2: Connect approval action to workflow continuation
  - [x] Subtask 2.3: Implement revision request → re-execute flow
- [ ] Task 3: Frontend ApprovalGateBadge (AC: 1) - OUT OF SCOPE for backend story
  - [ ] Subtask 3.1: Create ApprovalGateBadge component for TopBar
  - [ ] Subtask 3.2: Implement badge click → approval panel navigation
  - [ ] Subtask 3.3: Add batch-surfacing for multiple pending approvals
- [ ] Task 4: Copilot Notifications (AC: 1) - OUT OF SCOPE for backend story
  - [ ] Subtask 4.1: Create approval request notification template
  - [ ] Subtask 4.2: Add approve/reject/revision action buttons
  - [ ] Subtask 4.3: Implement morning digest re-surface logic
- [x] Task 5: Audit Record Storage (AC: 2)
  - [x] Subtask 5.1: Ensure immutable audit record format
  - [x] Subtask 5.2: Add strategy_id and metrics_snapshot fields

## Dev Notes

### Critical Architecture Context

**FROM EPIC 8 CONTEXT:**
- Alpha Forge pipeline has 9 stages: VIDEO_INGEST → RESEARCH → TRD → DEVELOPMENT → COMPILE → BACKTEST → VALIDATION → EA_LIFECYCLE → APPROVAL
- Story 8.1 wired the departments through pipeline stages (now in review)
- Story 8.2 implemented TRD Generation Stage (done)
- Story 8.3 implemented Fast-Track template matching (done)
- Story 8.4 implemented Strategy Version Control (done)

**EXISTING IMPLEMENTATIONS:**
- Approval Gate API: `src/api/approval_gate.py` — 570 lines, full CRUD, WebSocket, department mail notifications
- Approval Store: `quantmind-ide/src/lib/stores/approvalStore.ts`
- Approval Panel: `quantmind-ide/src/lib/components/ApprovalPanel.svelte`

**APPROVAL GATE EXISTING MODEL (src/api/approval_gate.py):**
```python
class ApprovalGateModel(Base):
    gate_id: str
    workflow_id: str
    workflow_type: Optional[str]
    from_stage: str
    to_stage: str
    gate_type: GateType  # STAGE_TRANSITION, DEPLOYMENT, RISK_CHECK, MANUAL_REVIEW
    status: ApprovalStatus  # PENDING, APPROVED, REJECTED
    requester: Optional[str]
    assigned_to: Optional[str]
    approver: Optional[str]
    reason: Optional[str]
    notes: Optional[str]
    extra_data: Optional[Dict]  # JSON for metrics snapshot
```

**KEY ARCHITECTURE DECISIONS (from architecture.md):**
- Human approval gate: Non-blocking with daily re-surface
- PENDING_REVIEW state: 7-day hard timeout → EXPIRED_REVIEW (retrievable)
- Batch question surfacing: all pending questions from a workflow run presented together
- Morning digest: re-surfaces all pending approvals daily

### Project Structure Notes

**Integration Points:**
1. `src/router/workflow_orchestrator.py` — Wire approval gate creation at backtest completion
2. `src/api/approval_gate.py` — Extend for Alpha Forge specific fields
3. `quantmind-ide/src/lib/components/TopBar.svelte` — Add ApprovalGateBadge
4. `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` — Approval notifications

**Naming Conventions:**
- Frontend: Svelte 5 runes (`$state`, `$derived`, `$effect`)
- Backend: FastAPI with Pydantic v2
- Database: SQLAlchemy with SQLite (workflows.db)
- Styling: Frosted Terminal aesthetic (glass effect per MEMORY.md)

### References

- FR28: Human approval required before paper trading AND before live deployment
- FR30: Trader can review, modify, or reject any Alpha Forge output at each gate
- NFR-D2: Immutable audit record format for approvals
- Source: docs/architecture.md#5.6-Human-Approval-Gate
- Source: _bmad-output/planning-artifacts/epics.md###-Story-8.5

### Previous Story Intelligence

**FROM STORY 8-4 (Strategy Version Control):**
- Implemented strategy version control API with rollback capability
- Patterns established: immutable audit records, version tagging
- Testing approach: unit tests for version CRUD, integration for rollback flow

**RELEVANT PATTERNS TO REUSE:**
- Immutable audit record storage pattern from story 8-4
- Version tagging approach for strategy snapshots

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6

### Debug Log References

- Fixed timezone comparison bug in check-timeout endpoint (naive vs aware datetime)
- Added database migration for new columns (strategy_id, metrics_snapshot, etc.)

### Completion Notes List

- **Task 1 (Backend API)**: Extended approval_gate.py with Alpha Forge support:
  - Added PENDING_REVIEW and EXPIRED_REVIEW statuses
  - Added ALPHA_FORGE_BACKTEST and ALPHA_FORGE_DEPLOYMENT gate types
  - Added strategy_id, metrics_snapshot, revision_feedback, expires_at, hard_expires_at fields
  - Implemented timeout logic (15-min soft, 7-day hard)
  - Added new endpoints: /check-timeout, /request-revision, /alpha-forge/pending

- **Task 2 (Alpha Forge Integration)**: Extended workflow_orchestrator.py:
  - Added _request_alpha_forge_approval() method with metrics snapshot support
  - Added handle_revision_request() method for revision → re-execute flow

- **Task 5 (Audit Records)**: Implemented immutable audit record format in approve endpoint:
  - Audit record includes: strategy_id, approver, approved_at_utc, gate_type, metrics_snapshot

- Added database migration to add new columns to existing approval_gates table

### File List

**Backend (Python):**
- `src/api/approval_gate.py` — Extended for Alpha Forge integration (new statuses, gate types, fields, endpoints)
- `src/router/workflow_orchestrator.py` — Added Alpha Forge approval gate methods
- `tests/api/test_approval_gate.py` — Updated with migration-aware tests
- `tests/api/test_approval_gate_alpha_forge.py` — New tests for Alpha Forge features

**Frontend (Svelte):** - OUT OF SCOPE for this backend story
- `quantmind-ide/src/lib/components/TopBar.svelte` — NOT MODIFIED (Task 3)
- `quantmind-ide/src/lib/components/ApprovalPanel.svelte` — NOT MODIFIED (Task 3)
- `quantmind-ide/src/lib/stores/approvalStore.ts` — NOT MODIFIED (Task 3)
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` — NOT MODIFIED (Task 4)

### Change Log

- 2026-03-20: Implemented Alpha Forge approval gate backend API
  - Added PENDING_REVIEW, EXPIRED_REVIEW statuses
  - Added ALPHA_FORGE_BACKTEST, ALPHA_FORGE_DEPLOYMENT gate types
  - Added strategy_id, metrics_snapshot fields to model
  - Added 15-min soft timeout, 7-day hard timeout logic
  - Added /check-timeout, /request-revision, /alpha-forge/pending endpoints
  - Extended workflow_orchestrator with Alpha Forge approval methods
  - Added unit tests for new functionality

---

## Senior Developer Review (AI)

**Review Outcome:** Conditional Approve
**Review Date:** 2026-03-21

### Git vs Story Discrepancies

- 0 discrepancies found

### Issues Found: 1 High, 0 Medium, 0 Low

**HIGH — `logger` used before assignment in `approval_gate.py`:**
Lines 39 and 47 called `logger.warning(...)` in the import try/except blocks, but `logger = logging.getLogger(__name__)` was not assigned until line 49. This causes a `NameError: name 'logger' is not defined` crash at module import time whenever either optional dependency (`broadcast_approval_gate` or `get_mail_service`) fails to import — which is the common case on Contabo without full live-trading setup.

### Verification Summary

| AC | Claim | Verification | Status |
|----|-------|--------------|--------|
| #1 | PENDING_REVIEW state after backtest passes | `ApprovalStatus.PENDING_REVIEW` enum value present; Alpha Forge gates initialize to this status | PASS |
| #2 | Approval recorded as immutable audit with strategy_id, approver, approved_at_utc, gate_type, metrics_snapshot | `approve_gate()` builds `audit_record` dict with all 5 required fields for Alpha Forge gate types | PASS |
| #3 | Revision request → stores feedback, agent re-executes with feedback context | `/request-revision` endpoint stores `revision_feedback` on gate record | PASS |
| #4 | 15-min soft timeout → EXPIRED_REVIEW, 7-day hard timeout | `create_approval_gate` sets `expires_at` (+15 min) and `hard_expires_at` (+7 days); `/check-timeout` sets `EXPIRED_REVIEW` on hard timeout | PASS |
| Tests | No `@pytest.mark.asyncio` | 7 tests, no asyncio decorators — correct | PASS |

### Review Notes

7/7 tests pass. The immutable audit record pattern for Alpha Forge gates is correct. Timeout logic correctly distinguishes soft (notification only) from hard (status transition to EXPIRED_REVIEW). The logger-before-assignment bug was a crash-on-import risk.

### Action Items

- [x] Fixed: Moved `logger = logging.getLogger(__name__)` to before the try/except import blocks in `approval_gate.py`