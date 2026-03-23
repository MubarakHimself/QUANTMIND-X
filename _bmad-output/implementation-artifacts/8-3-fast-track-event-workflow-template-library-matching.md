# Story 8.3: Fast-Track Event Workflow — Template Library & Matching

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader responding to market events,
I want a strategy template library and fast-track matching so that I can deploy a relevant strategy within 11 minutes of a hot news event,
So that I can act on event-driven opportunities without waiting for the full 72-hour Alpha Forge loop (Journey 21).

## Acceptance Criteria

**Given** the strategy template library contains at least 3 templates,
**When** `GET /api/alpha-forge/templates` is called,
**Then** all templates return with: name, strategy_type, applicable_events, risk_profile, avg_deployment_time.

**Given** a HIGH-impact news event fires (from Story 6.3),
**When** the template matching runs,
**Then** matching templates are ranked by `{ template_name, confidence_score, estimated_deployment_time }`,
**And** the Copilot surfaces: "Fast-track available — GBPUSD event strategy template matches. Deploy? [11 min]"

**Given** Mubarak approves fast-track deployment,
**When** the fast-track Prefect workflow runs,
**Then** the strategy is live-deployed within 15 minutes with: conservative lot sizing, auto-expiry tag, Islamic compliance parameters,
**And** the deployment is a full pipeline run (compile → SIT gate only — no full 6-mode backtest).

**Notes:**
- FR3 (Story 3.3), Journey 21 (The Template Pull)
- Fast-track requires SIT gate (MODE_C) pass minimum — not the full 4/6 mode threshold
- Template library initially seeded manually; Research dept adds templates over time

## Tasks / Subtasks

- [x] Task 1: Create strategy template library with 3+ templates (AC: 1)
  - [x] Subtask 1.1: Design template schema with all required fields
  - [x] Subtask 1.2: Create 3 initial templates (news-event breakout, range expansion, volatility spike)
  - [x] Subtask 1.3: Implement template storage and retrieval
- [x] Task 2: Implement template API endpoints (AC: 1)
  - [x] Subtask 2.1: Create GET /api/alpha-forge/templates endpoint
  - [x] Subtask 2.2: Add template CRUD operations (create, update, delete)
  - [x] Subtask 2.3: Add template search/filter by applicable_events
- [x] Task 3: Implement template matching service (AC: 2)
  - [x] Subtask 3.1: Create template matcher that scores templates against news events
  - [x] Subtask 3.2: Integrate with Story 6.3 news alert system (via webhook endpoint)
  - [x] Subtask 3.3: Add confidence scoring algorithm
  - [x] Subtask 3.4: Implement Copilot integration for surfacing fast-track options
- [x] Task 4: Implement fast-track deployment workflow (AC: 3)
  - [x] Subtask 4.1: Create fast-track Prefect workflow (compile → SIT gate only)
  - [x] Subtask 4.2: Add conservative lot sizing configuration
  - [x] Subtask 4.3: Add auto-expiry tag (24-hour expiry for event strategies)
  - [x] Subtask 4.4: Ensure Islamic compliance parameters are always included
  - [x] Subtask 4.5: Implement 15-minute deployment timeout handling

## Dev Notes

### Critical Architecture Context

**FROM EPICS (Story 8-3):**
- Template library replaces/supplements existing `src/mql5/templates/ea_base_template.py`
- Fast-track triggered by HIGH-impact news events from Story 6.3
- Deployment pipeline: compile → SIT gate (MODE_C) → live (no full backtest)
- Target deployment time: 11-15 minutes

**FROM PREVIOUS STORIES (8-1, 8-2):**
- Story 8-1: AlphaForgeFlow already wired with Prefect
- Story 8-2: TRD generation, storage, and versioning implemented
- AlphaForgeFlow location: `flows/alpha_forge_flow.py`
- TRD components: `src/trd/` (schema, parser, validator, generator, storage)

**FROM EPIC 8 AUDIT (8-0 findings):**
- Single template exists at `src/mql5/templates/ea_base_template.py` - needs expansion
- Fast-track template library needs to be built (Story 8-3)
- Template matching needs integration with news events

**FROM STORY 6-3 (News Feed):**
- `src/knowledge/news/geopolitical_subagent.py` - classifies events as HIGH/MEDIUM/LOW
- News alert endpoint: `POST /api/news/alert`
- WebSocket broadcast for news alerts
- Sub-90-second latency for HIGH-impact events

### Key Constraints

1. **SIT Gate Only:** Fast-track skips full 6-mode backtest - only requires MODE_C (SIT) pass
2. **Conservative Lot Sizing:** Reduced risk for event-driven strategies (default 0.5x normal)
3. **Auto-Expiry Tag:** Event strategies auto-expire after 24 hours (no overnight hold)
4. **Islamic Compliance:** Must include force_close_hour, overnight_hold=False
5. **15-Minute Timeout:** Fast-track workflow must complete within 15 minutes or fail gracefully

### Project Structure Notes

- **Backend root:** `src/`
- **Template storage:** `src/mql5/templates/` (expand from ea_base_template.py)
- **Template API:** `src/api/` - new alpha_forge_templates.py
- **News integration:** `src/knowledge/news/` - integrate with geopolitical_subagent.py
- **Fast-track flow:** `flows/alpha_forge_flow.py` - add fast-track branch
- **Copilot integration:** `src/agents/departments/floor_manager.py` - extend for fast-track suggestions

### Technical Requirements

1. **Template Schema:**
   ```python
   Template:
     - id: str (UUID)
     - name: str
     - strategy_type: str (news_event_breakout, range_expansion, volatility_spike)
     - applicable_events: List[str] (HIGH_IMPACT_NEWS, CENTRAL_BANK, GEOPOLITICAL)
     - risk_profile: str (conservative, moderate, aggressive)
     - avg_deployment_time: int (minutes)
     - parameters: dict (EA input parameters)
     - created_at: datetime
     - updated_at: datetime
   ```

2. **Template Matching Service:**
   - Score templates based on: event type, symbol match, regime match
   - Confidence score = weighted average of match factors
   - Return ranked list with estimated deployment time

3. **Fast-Track Workflow:**
   - Trigger: Copilot approval or API call
   - Steps: Generate TRD from template → Compile → SIT Gate → Deploy
   - Timeout: 15 minutes total
   - Fallback: If timeout, notify FloorManager and pause

4. **News Integration:**
   - Subscribe to HIGH-impact news events from Story 6.3
   - On event: Run template matching → surface to Copilot
   - Store event-template associations for learning

### Testing Standards

- Unit tests for template matching algorithm
- Integration tests for news event → template matching flow
- Fast-track workflow timeout tests
- Template CRUD operation tests
- End-to-end test with simulated HIGH-impact event

### References

- Epic 8: _bmad-output/planning-artifacts/epics.md#Epic-8
- Story 8-0 findings: _bmad-output/implementation-artifacts/8-0-alpha-forge-pipeline-audit-findings.md
- Story 8-1: _bmad-output/implementation-artifacts/8-1-alpha-forge-orchestrator-wiring-departments-through-pipeline-stages.md
- Story 8-2: _bmad-output/implementation-artifacts/8-2-trd-generation-stage.md
- Story 6-3: _bmad-output/planning-artifacts/epics.md#Story-6.3
- Template base: src/mql5/templates/ea_base_template.py
- News system: src/knowledge/news/geopolitical_subagent.py
- AlphaForgeFlow: flows/alpha_forge_flow.py

## Dev Agent Record

### Agent Model Used
claude-sonnet-4-20250514

### Debug Log References

### Completion Notes List

- Implemented StrategyTemplate schema with all required fields (id, name, strategy_type, applicable_events, risk_profile, avg_deployment_time, parameters)
- Created 3 default templates: News Event Breakout, Range Expansion Strategy, Volatility Spike Capture
- Implemented TemplateStorage with JSON file storage and index management
- Created GET /api/alpha-forge/templates endpoint with filtering by event_type and symbol
- Added CRUD operations for templates (create, update, delete)
- Implemented TemplateMatcher with weighted scoring algorithm (event_match: 40%, symbol_match: 30%, risk_alignment: 20%, deployment_time: 10%)
- Created FastTrackEventListener for news event integration
- Implemented FastTrackFlow Prefect workflow with compile → SIT gate → deploy pipeline
- Added conservative lot sizing (0.5x multiplier), auto-expiry (24 hours), Islamic compliance parameters
- All 13 unit tests pass

### File List
- src/mql5/templates/schema.py (NEW)
- src/mql5/templates/storage.py (NEW)
- src/mql5/templates/matcher.py (NEW)
- src/mql5/templates/news_integration.py (NEW)
- src/mql5/templates/__init__.py (MODIFIED - added exports)
- src/api/alpha_forge_templates.py (NEW)
- src/api/server.py (MODIFIED - added router)
- flows/fast_track_flow.py (NEW)
- tests/mql5/test_templates.py (NEW)

## Change Log

- 2026-03-20: Implemented strategy template library with 3 default templates
- 2026-03-20: Created template API endpoints (GET/POST/PUT/DELETE /api/alpha-forge/templates)
- 2026-03-20: Implemented template matching service with confidence scoring
- 2026-03-20: Created FastTrackFlow Prefect workflow for 11-15 minute deployments
- 2026-03-20: Added fast-track deployment endpoint (POST /api/alpha-forge/fast-track/deploy)
- 2026-03-20: Added unit tests for template functionality (13 tests pass)

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
| #1 | GET /api/alpha-forge/templates returns name, strategy_type, applicable_events, risk_profile, avg_deployment_time — at least 3 templates | `seed_default_templates()` creates 3 templates (News Event Breakout, Range Expansion, Volatility Spike) with all required fields | PASS |
| #2 | HIGH-impact news event → ranked `{template_name, confidence_score, estimated_deployment_time}` | `TemplateMatcher.match_event()` returns `TemplateMatchResult` with all three fields, sorted by confidence_score descending | PASS |
| #3 | Fast-track: conservative lot sizing (0.5x), auto-expiry (24h), Islamic compliance | All 3 default templates have `lot_sizing_multiplier=0.5`, `auto_expiry_hours=24`, `is_islamic_compliant=True`, and `force_close_hour`/`overnight_hold` in parameters | PASS |
| Scoring weights | event_match 40%, symbol_match 30%, risk_alignment 20%, deployment_time 10% | `TemplateMatcher.WEIGHTS` dict matches spec exactly | PASS |
| Tests | No `@pytest.mark.asyncio` | 13 tests, no asyncio decorators — correct | PASS |

### Review Notes

13/13 tests pass. The confidence scoring algorithm is correctly implemented with the 4-factor weighted calculation. All 3 default templates include Islamic compliance parameters (force_close_hour, overnight_hold=False) in their parameters dict.

### Action Items

- [x] No fixes required