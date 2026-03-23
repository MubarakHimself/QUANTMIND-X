# Story 8.2: TRD Generation Stage

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a strategy developer,
I want the TRD Generation stage to produce structured, complete implementation specifications,
So that the Development department has an unambiguous spec to code from (FR24).

## Acceptance Criteria

**Given** a research hypothesis passes confidence threshold (>=0.75 per ResearchHead config),
**When** TRD_GENERATION stage fires,
**Then** the Research Department produces a TRD with: strategy name, symbol, timeframe, entry/exit conditions, position sizing rules, risk parameters, full EA input parameter spec (session_mask, force_close_hour, overnight_hold, daily_loss_cap, spread_filter, etc.).

**Given** the TRD is validated,
**When** the TRD validator runs,
**Then** all required MQL5 parameter fields are present,
**And** if any are missing, the TRD is rejected with a list of missing fields for Research to fill.

**Notes:**
- TRD is the contract between Research and Development — quality here prevents compile failures
- TRD stored in strategy version control with unique strategy ID
- Islamic compliance parameters (force_close_hour, overnight_hold) must always be present

## Tasks / Subtasks

- [x] Task 1: Implement TRD generation from research hypothesis (AC: 1)
  - [x] Subtask 1.1: Create trd_generator.py module that takes ResearchHead output and creates TRDDocument
  - [x] Subtask 1.2: Map research hypothesis fields to TRD schema fields
  - [x] Subtask 1.3: Auto-populate default values for Islamic compliance parameters
  - [x] Subtask 1.4: Generate unique strategy_id with symbol and timestamp
- [x] Task 2: Implement TRD validation integration (AC: 2)
  - [x] Subtask 2.1: Integrate TRDValidator with generated TRDs
  - [x] Subtask 2.2: Handle validation errors - return rejection with missing field list
  - [x] Subtask 2.3: Add validation to AlphaForgeFlow TRD_GENERATION stage
  - [x] Subtask 2.4: Add clarification request generation for FloorManager
- [x] Task 3: Implement TRD storage and versioning (AC: notes)
  - [x] Subtask 3.1: Create strategy version control storage for TRDs
  - [x] Subtask 3.2: Implement TRD retrieval by strategy_id
  - [x] Subtask 3.3: Add TRD history tracking
- [x] Task 4: Wire TRD into AlphaForgeFlow (from Story 8-1 context)
  - [x] Subtask 4.1: Ensure trd_data flows correctly through pipeline
  - [x] Subtask 4.2: Handle validation failures in flow

## Dev Notes

### Critical Architecture Context

**FROM EPICS (Story 8-2):**
- TRD Generation Stage is part of Alpha Forge pipeline (Story 8-1)
- ResearchHead has TRD_ESCALATION_THRESHOLD = 0.75 (src/agents/departments/heads/research_head.py:58)
- TRD is the contract between Research and Development departments

**FROM STORY 8-1 (review):**
- Story 8-1 has already wired TRD_GENERATION stage in AlphaForgeFlow
- TRD module exists at `src/trd/` with schema, parser, validator
- API endpoints exist at `src/api/trd_endpoints.py`

**EXISTING COMPONENTS:**
| Component | Location | Status |
|-----------|----------|--------|
| TRD Schema | src/trd/schema.py | Complete |
| TRD Parser | src/trd/parser.py | Complete |
| TRD Validator | src/trd/validator.py | Complete |
| TRD API | src/api/trd_endpoints.py | Partial (CRUD, needs generation) |
| AlphaForgeFlow | flows/alpha_forge_flow.py | Wired |

### Key Constraints

1. **Islamic Compliance:** force_close_hour and overnight_hold MUST always be present in generated TRDs
2. **TRD as Contract:** Quality here prevents compile failures in Development stage
3. **Strategy ID Format:** `{symbol}_{timestamp}_{uuid}` for uniqueness
4. **Validation Threshold:** All required MQL5 parameters must be present

### Project Structure Notes

- **Backend root:** `src/`
- **TRD components:** `src/trd/schema.py`, `src/trd/parser.py`, `src/trd/validator.py`
- **TRD API:** `src/api/trd_endpoints.py`
- **Alpha Forge Flow:** `flows/alpha_forge_flow.py`
- **Research Department:** `src/agents/departments/heads/research_head.py`
- **Development Department:** `src/agents/departments/heads/development_head.py`

### Technical Requirements

1. **TRD Generator Module:**
   - Create `src/trd/generator.py` - generates TRD from research hypothesis
   - Map research output fields to TRD schema
   - Auto-populate default Islamic compliance parameters
   - Generate unique strategy_id

2. **Validation Integration:**
   - Use existing TRDValidator from `src/trd/validator.py`
   - Return rejection with missing field list if validation fails
   - Handle requires_clarification() for FloorManager

3. **Storage:**
   - Integrate with existing trd_endpoints.py storage
   - Strategy version control with unique IDs

4. **Flow Integration:**
   - Ensure trd_data flows correctly through AlphaForgeFlow
   - Handle validation failures gracefully

### Testing Standards

- Unit tests for TRD generator
- Integration tests for hypothesis → TRD flow
- Validation tests for missing parameter detection
- End-to-end test with sample research hypothesis

### References

- Epic 8: _bmad-output/planning-artifacts/epics.md#Epic-8
- Story 8-1: _bmad-output/implementation-artifacts/8-1-alpha-forge-orchestrator-wiring-departments-through-pipeline-stages.md
- TRD Schema: src/trd/schema.py
- ResearchHead: src/agents/departments/heads/research_head.py (TRD_ESCALATION_THRESHOLD at line 58)
- AlphaForgeFlow: flows/alpha_forge_flow.py

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.5 (via Claude Code)

### Implementation Plan

1. **Created TRDGenerator** (`src/trd/generator.py`):
   - Takes ResearchHead Hypothesis dataclass as input
   - Maps hypothesis fields to TRD schema fields
   - Auto-populates Islamic compliance parameters (force_close_hour=22, overnight_hold=False, daily_loss_cap=2.0)
   - Generates unique strategy_id using `{symbol}_{timestamp}_{uuid}` format
   - Provides generate_and_validate() method for combined generation + validation

2. **Created TRDStorage** (`src/trd/storage.py`):
   - Provides version control for TRD documents
   - Saves to `./data/trd/` directory
   - Archives versions to `./data/trd/history/{strategy_id}/`
   - Supports retrieval by strategy_id and version history

3. **Created TRD Generation API** (`src/api/trd_generation_endpoints.py`):
   - POST `/api/trd/generation/generate` - Generate TRD from hypothesis
   - POST `/api/trd/generation/validate` - Validate existing TRD
   - GET `/api/trd/generation/clarification/{strategy_id}` - Get clarification request
   - Integrates with existing trd_endpoints.py router registration

4. **Wired AlphaForgeFlow** (`flows/alpha_forge_flow.py`):
   - Added research_hypothesis parameter to flow
   - Added Stage 0: TRD Generation task
   - Added Stage 0b: TRD Validation task
   - Handles validation failures with FloorManager notification
   - Handles needs_clarification status for incomplete TRDs

5. **Created Tests** (`tests/trd/test_generator.py`):
   - 12 tests covering all major functionality
   - All tests pass

### Debug Log References

- TRD generator uses ResearchHead.Hypothesis dataclass as input
- Strategy ID format ensures uniqueness across generations
- Islamic compliance parameters are always included with defaults
- Flow validates confidence threshold (0.75) before escalating

### Completion Notes List

- TRD generation from research hypothesis implemented
- Validation integrated with automatic rejection for missing fields
- Clarification requests generated for incomplete TRDs
- Storage with version control implemented
- AlphaForgeFlow updated to support research_hypothesis input
- All 12 unit tests pass

### File List

**New Files:**
- `src/trd/generator.py` - TRD generator module
- `src/trd/storage.py` - TRD storage and versioning
- `src/api/trd_generation_endpoints.py` - TRD generation API
- `tests/trd/test_generator.py` - Unit tests for generator

**Modified Files:**
- `src/trd/__init__.py` - Added exports for generator and storage
- `src/api/server.py` - Added trd_generation_router import and include
- `flows/alpha_forge_flow.py` - Added TRD generation stage tasks and integration
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated story status to in-progress (will be review)

### Change Log

- 2026-03-20: Implemented TRD generation stage with full pipeline integration
  - Created generator, storage, and API endpoints
  - Wired into AlphaForgeFlow
  - All tests pass

---

## Senior Developer Review (AI)

**Review Outcome:** Approve
**Review Date:** 2026-03-21

### Git vs Story Discrepancies

- 0 discrepancies found

### Issues Found: 0 High, 0 Medium, 0 Low

### Verification Summary

| AC | Claim | Verification | Status |
|----|-------|--------------|--------|
| #1 | Research hypothesis >= 0.75 → TRD with all required fields including Islamic params | `TRDGenerator._build_parameters` merges `DEFAULT_ISLAMIC_PARAMS` (force_close_hour, overnight_hold, daily_loss_cap) and `DEFAULT_EA_PARAMS` (session_mask, spread_filter) into every generated TRD | PASS |
| #2 | TRD validator rejects with missing field list | `TRDValidator` produces `ValidationError` objects per field; `requires_clarification()` path returns field list | PASS |
| #3 | Islamic compliance params always present | `DEFAULT_ISLAMIC_PARAMS` hardcoded in `TRDGenerator`, applied unconditionally in `_build_parameters` | PASS |
| Tests | No `@pytest.mark.asyncio` | 12 tests, no asyncio decorators — correct | PASS |

### Review Notes

12/12 tests pass. Generator correctly produces all mandatory EA input parameters. The confidence threshold (0.75) check is in `alpha_forge_flow.py` before calling the generator, consistent with `TRD_ESCALATION_THRESHOLD` in research_head.py. Islamic compliance fields (force_close_hour=22, overnight_hold=False, daily_loss_cap=2.0) are always included.

### Action Items

- [x] No fixes required