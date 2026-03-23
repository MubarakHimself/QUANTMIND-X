# Story 8.1: Alpha Forge Orchestrator — Wiring Departments Through Pipeline Stages

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer building the strategy factory,
I want the Alpha Forge Prefect workflow wired to call real department implementations,
So that the full pipeline operates end-to-end: source → paper trading.

## Acceptance Criteria

**Given** a YouTube URL is submitted to the Alpha Forge pipeline,
**When** Workflow 1 (Creation) runs,
**Then** VIDEO_INGEST → RESEARCH (Research Dept, now real) → TRD_GENERATION → DEVELOPMENT (Dev Dept, now real) → COMPILE → basic BACKTEST stages execute in sequence,
**And** each stage writes its result to Prefect `workflows.db` for durability.

**Given** a compiled EA passes basic backtest,
**When** Workflow 2 (Enhancement Loop) triggers,
**Then** full 6-mode backtest → SIT gate → paper trading → human approval → live deployment stages execute,
**And** the human approval gate (PENDING_REVIEW state) halts the pipeline until confirmed.

**Given** any stage fails,
**When** the failure occurs,
**Then** Prefect marks the workflow FAILED,
**And** Mubarak is notified via Copilot: "Alpha Forge stage [name] failed — [reason]. Review?"

**Notes:**
- FR23: complete Alpha Forge loop without mandatory manual intervention at intermediate stages
- Architecture: DOE methodology — Directive (what) + Orchestration (who) + Execution (how)
- First Version Rule: first EA must mirror source strategy; spiced variants in subsequent loops only

## Tasks / Subtasks

- [x] Task 1: Set up Prefect infrastructure and flow definitions (AC: all)
  - [x] Subtask 1.1: Create flows/ directory structure with Prefect workflow definitions
  - [x] Subtask 1.2: Configure Prefect SQLite database (workflows.db) for durability
  - [x] Subtask 1.3: Define VideoIngestFlow for Workflow 1 (Creation)
  - [x] Subtask 1.4: Define AlphaForgeFlow for Workflow 2 (Enhancement Loop)
- [x] Task 2: Wire VIDEO_INGEST → RESEARCH stage (AC: 1)
  - [x] Subtask 2.1: Connect video_ingest API to Prefect flow
  - [x] Subtask 2.2: Connect ResearchHead (research_head.py) as real department implementation
  - [x] Subtask 2.3: Add result persistence to workflows.db
- [x] Task 3: Wire RESEARCH → TRD_GENERATION → DEVELOPMENT stages (AC: 1)
  - [x] Subtask 3.1: Wire Research hypothesis output to TRD generator
  - [x] Subtask 3.2: Wire TRD output to DevelopmentHead (development_head.py)
  - [x] Subtask 3.3: Verify MQL5 code generation (real, not mocked)
- [x] Task 4: Wire DEVELOPMENT → COMPILE → basic BACKTEST (AC: 1)
  - [x] Subtask 4.1: Wire MQL5 code to compile endpoint
  - [x] Subtask 4.2: Connect basic backtest execution
  - [x] Subtask 4.3: Persist stage outputs to workflows.db
- [x] Task 5: Implement Workflow 2 (Enhancement Loop) (AC: 2)
  - [x] Subtask 5.1: Connect 6-mode full backtest runner
  - [x] Subtask 5.2: Wire SIT gate (risk validation)
  - [x] Subtask 5.3: Connect paper trading deployment
  - [x] Subtask 5.4: Implement PENDING_REVIEW human approval state
- [x] Task 6: Error handling and Copilot notifications (AC: 3)
  - [x] Subtask 6.1: Implement stage failure detection
  - [x] Subtask 6.2: Wire Copilot notification for failures
  - [x] Subtask 6.3: Test end-to-end failure scenarios

## Dev Notes

### Critical Architecture Context

**FROM STORY 8-0 AUDIT FINDINGS:**
- Prefect NOT yet deployed - this story must set it up first
- Workflow orchestrator exists at `src/router/workflow_orchestrator.py` but uses in-memory state (needs migration to Prefect)
- All 9 pipeline stage components exist but NOT wired through Prefect
- Department agents are now REAL implementations (Epic 7 completed):
  - ResearchHead: `src/agents/departments/heads/research_head.py`
  - DevelopmentHead: `src/agents/departments/heads/development_head.py`

**PIPELINE STAGE IMPLEMENTATION STATE (from 8-0 audit):**

| Stage | Status | Location |
|-------|--------|----------|
| VIDEO_INGEST | ✅ Complete | src/video_ingest/ |
| RESEARCH | ✅ Complete | src/agents/departments/heads/research_head.py |
| TRD_GENERATION | ✅ Complete | src/trd/ |
| DEVELOPMENT | ✅ Complete | src/agents/departments/heads/development_head.py |
| COMPILE | ✅ Complete | src/mql5/ |
| BACKTEST | ✅ Complete | src/backtesting/ |
| VALIDATION | ⚠️ Partial | src/risk/ (needs wiring) |
| EA_LIFECYCLE | ✅ Complete | src/agents/tools/ea_lifecycle.py |
| APPROVAL | ❌ Missing | Needs build (Story 8.5) |

### Key Constraints

1. **Prefect Setup Required:** Story 8.1 must install and configure Prefect before wiring flows
2. **DOE Methodology:** Directive (what) + Orchestration (who) + Execution (how)
3. **First Version Rule:** First EA must mirror source strategy; spiced variants only in subsequent loops
4. **Deployment Window:** Friday 22:00 – Sunday 22:00 UTC only (for live deployment)
5. **Department Types:** Research (RESEARCH), Development (DEVELOPMENT) - both connected via Redis streams

### Project Structure Notes

- **Backend root:** `src/`
- **Flows location:** `flows/` (new - create this directory)
- **Prefect DB:** `workflows.db` (SQLite - create in flows/)
- **Department agents:** `src/agents/departments/heads/` (research_head, development_head)
- **Video ingest API:** `src/video_ingest/api.py`
- **TRD components:** `src/trd/schema.py`, `src/trd/parser.py`, `src/trd/validator.py`
- **MQL5 generator:** `src/mql5/generator.py`, `src/mql5/templates/`
- **Backtest API:** `src/api/backtest_endpoints.py`
- **EA lifecycle:** `src/agents/tools/ea_lifecycle.py`

### Technical Requirements

1. **Prefect Setup:**
   - Install `prefect` package
   - Create `flows/` directory with __init__.py
   - Configure workflows.db SQLite storage
   - Register flows with Prefect server (or use Prefect Orion local)

2. **Flow Definitions:**
   - VideoIngestFlow: YouTube URL → Video → Transcript → Research
   - AlphaForgeFlow: Research → TRD → Development → Compile → Backtest → Validate → Deploy

3. **Redis Integration:**
   - Departments communicate via Redis streams
   - ResearchHead listens for video_ingest output
   - DevelopmentHead listens for TRD output

4. **Durability:**
   - Each stage must write result to workflows.db
   - Use Prefect task result handlers for persistence
   - Enable workflow resume from last successful stage

### Testing Standards

- Unit tests for each flow task
- Integration tests for department wiring
- End-to-end test with mock video URL
- Error scenario tests (stage failures, API timeouts)

### References

- Architecture: docs/architecture.md §20 (Alpha Forge Two-Workflow Architecture)
- PRD: docs/prd.md FR23–FR31, FR74–FR79
- Story 8-0 findings: _bmad-output/implementation-artifacts/8-0-alpha-forge-pipeline-audit-findings.md
- Epic 7 completion: stories 7-1 through 7-8 wired departments as real implementations

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-20250514

### Debug Log References

### Completion Notes List

- Implemented Prefect infrastructure with SQLite backend (workflows.db)
- Created VideoIngestFlow for Workflow 1 (Creation pipeline)
- Created AlphaForgeFlow for Workflow 2 (Enhancement loop)
- Wired all pipeline stages with real department implementations
- Implemented SIT gate validation and human approval workflow
- Added Copilot notification for stage failures

### File List

- flows/__init__.py - Module exports
- flows/config.py - Prefect configuration
- flows/database.py - SQLite database for workflow durability
- flows/video_ingest_flow.py - VideoIngestFlow (Workflow 1)
- flows/alpha_forge_flow.py - AlphaForgeFlow (Workflow 2)
- requirements.txt - Added prefect>=2.20.0
- tests/flows/test_alpha_forge_flows.py - Unit tests for flows

## Change Log

- 2026-03-20: Initial implementation - Added Prefect infrastructure with SQLite backend, created VideoIngestFlow and AlphaForgeFlow with full pipeline wiring (Date: 2026-03-20)

---

## Senior Developer Review (AI)

**Review Outcome:** Conditional Approve
**Review Date:** 2026-03-21

### Git vs Story Discrepancies

- 0 discrepancies found

### Issues Found: 1 High, 1 Medium, 0 Low

**HIGH — UTC timezone bug in `check_deployment_window` (flows/alpha_forge_flow.py):**
`datetime.now()` was used instead of `datetime.now(timezone.utc)`. The deployment window spec says "Friday 22:00 UTC" but the check was evaluated against local server time. On a server in a non-UTC timezone this would allow or block deployments at the wrong times.

**MEDIUM — Wrong import in `alpha_forge_flow.py` missing `timezone`:**
`datetime` was imported without `timezone` despite `timezone.utc` being needed for the window fix.

### Verification Summary

| AC | Claim | Verification | Status |
|----|-------|--------------|--------|
| #1 | VideoIngestFlow: VIDEO_INGEST → RESEARCH → TRD → DEVELOPMENT → COMPILE → BACKTEST, each stage writes to workflows.db | All 6 stages present in video_ingest_flow.py with persist_workflow_results calls | PASS |
| #2 | AlphaForgeFlow: 6-mode backtest → SIT gate → paper trading → PENDING_REVIEW → live deployment | All stages present with proper gating logic | PASS |
| #3 | Stage failure → Prefect marks FAILED → Copilot notification | notify_failure() called on exception, db status set to "failed" | PASS |
| No `@pytest.mark.asyncio` | Tests use class-based style | No asyncio decorators found — correct | PASS |

### Review Notes

13/13 tests pass. Both flows are structurally correct. The deployment window bug was critical because `check_deployment_window` is also imported and used by `ea_deployment_flow.py`, compounding the impact. Fixed by importing `timezone` and using `datetime.now(timezone.utc)`.

### Action Items

- [x] Fixed: `check_deployment_window` now uses `datetime.now(timezone.utc)` instead of `datetime.now()` for correct UTC window enforcement
- [x] Fixed: Added `timezone` to `from datetime import datetime, timezone` in alpha_forge_flow.py
