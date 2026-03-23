---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-assessment', 'step-04-coverage-plan', 'step-05-generate-document']
lastStep: 'step-05-generate-document'
lastSaved: '2026-03-21'
---

# Test Design: Epic 5 - Unified Memory & Copilot Core

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft

---

## Executive Summary

**Scope:** Epic-level test design for Epic 5: Unified Memory & Copilot Core (Stories 5.0-5.9)

**Epic Status:** DONE (all 10 stories completed: 5.0 audit + 5.1-5.9 implementation)

**Risk Summary:**

- Total risks identified: 18
- High-priority risks (Score >= 6): 7
- Critical categories: DATA (session isolation), TECH (streaming, embedding quality), OPS (kill switch independence)

**Coverage Summary:**

- P0 scenarios: 12 tests (~18 hours)
- P1 scenarios: 18 tests (~18 hours)
- P2/P3 scenarios: 24 tests (~14 hours)
- **Total effort**: 54 tests, ~50 hours (~6.25 days)

---

## Not in Scope

| Item                                      | Reasoning                                                                 | Mitigation                              |
| ----------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------- |
| **External LLM provider APIs (Claude Opus)** | Tested by provider; Epic 2 covered provider configuration                 | Epic 2 provider config tests validate   |
| **Trading Kill Switch (separate epic)**   | Architecturally independent; Epic 3 covered trading kill switch             | Epic 3 tests cover trading kill switch   |
| **MT5/Live Trading broker integration**  | Epic 3 covered live trading backend; separate integration testing           | Epic 3 integration tests                |
| **ChromaDB vector database cluster**      | Infrastructure-level concern; assumed working per deployment docs            | DB team validation; smoke tests cover   |
| **Redis Streams (department mail)**        | Epic 7 covered Redis Streams; assumed working                               | Epic 7 integration tests                |
| **sentence-transformers model loading**   | Third-party ML library; assume correct behavior per upstream testing        | Verify model file integrity once        |
| **WebSocket infrastructure (SSE transport)** | Network layer assumed working; covered by streaming integration tests       | Network/infrastructure team owns        |
| **Department Head agent implementations**  | Epic 7 covered department system; each head tested in Epic 7               | Epic 7 department tests                 |

---

## Risk Assessment

### High-Priority Risks (Score >= 6)

| Risk ID | Category | Description                                                                       | Probability | Impact | Score | Mitigation                                      | Owner      | Timeline   |
| ------- | -------- | --------------------------------------------------------------------------------- | ----------- | ------ | ----- | ---------------------------------------------- | ---------- | ---------- |
| R-001   | DATA     | Graph memory session isolation failure: draft nodes visible to other sessions       | 3           | 3      | 9     | Unit tests verify draft/commit visibility rules | QA/Backend | Sprint 5.x |
| R-002   | TECH     | OPINION node orphaned from SUPPORTED_BY edge (mandatory constraint violation)     | 3           | 3      | 9     | DB constraint + integration test for edge creation | Backend | Sprint 5.x |
| R-003   | TECH     | Vector embedding quality degradation causing irrelevant semantic search results      | 2           | 3      | 6     | Threshold-based recall tests; periodic model health check | QA/ML | Sprint 5.x |
| R-004   | OPS      | Copilot kill switch activation inadvertently blocks trading operations              | 2           | 3      | 6     | Architectural isolation tests; separate activation paths | Backend | Sprint 5.x |
| R-005   | TECH     | Streaming response token loss/corruption in SSE transport                           | 2           | 3      | 6     | E2E streaming test with token count verification | QA/Backend | Sprint 5.x |
| R-006   | TECH     | CanvasContextTemplate token budget exceeded causing context truncation              | 2           | 3      | 6     | Unit test for token counting; max_identifiers enforcement | Backend | Sprint 5.x |
| R-007   | TECH     | FloorManager routing misclassification: tasks sent to wrong department              | 2           | 3      | 6     | Integration tests for each department routing  | QA/Backend | Sprint 5.x |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description                                                               | Probability | Impact | Score | Mitigation                                      | Owner      |
| ------- | -------- | ------------------------------------------------------------------------- | ----------- | ------ | ----- | ---------------------------------------------- | ---------- |
| R-008   | TECH     | ReflectionExecutor fails to promote draft nodes (promotion logic bug)     | 2           | 2      | 4     | Integration test for draft-to-commit promotion | Backend    |
| R-009   | TECH     | Session recovery returns stale or missing committed state after crash       | 2           | 2      | 4     | Session recovery integration tests             | Backend    |
| R-010   | DATA     | Stale draft cleanup removes valid in-progress session data                  | 1           | 3      | 3     | Configurable threshold; test cleanup boundary  | Backend    |
| R-011   | PERF     | Copilot first-token latency exceeds NFR-P3 threshold (5 seconds)           | 2           | 2      | 4     | Performance benchmarks; streaming timing tests  | QA/Backend |
| R-012   | TECH     | Intent classification confidence miscalibration triggers wrong flow          | 2           | 2      | 4     | Threshold tests for confidence scoring         | QA/Backend |
| R-013   | TECH     | Cross-canvas navigation fails to preserve context (entity lost in transit)  | 2           | 2      | 4     | E2E navigation tests per entity type           | QA/Frontend |
| R-014   | SEC      | Canvas context metadata leaks session IDs across unauthorized canvases        | 1           | 3      | 3     | Security tests for session isolation           | QA/Sec     |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description                                                               | Probability | Impact | Score | Action  |
| ------- | -------- | ------------------------------------------------------------------------- | ----------- | ------ | ----- | ------- |
| R-015   | PERF     | Suggestion chip dynamic update causes UI jank                              | 2           | 1      | 2     | Monitor |
| R-016   | BUS      | Morning digest auto-trigger fires incorrectly (duplicate digest)            | 1           | 2      | 2     | Monitor |
| R-017   | OPS      | Kill switch history not properly cleared on resume                          | 1           | 1      | 1     | Monitor |
| R-018   | TECH     | Memory explorer tree view renders slowly with large node counts              | 1           | 1      | 1     | Monitor |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] Graph memory database schema migration verified (session_status, embedding columns)
- [ ] Sentence-transformers model (all-MiniLM-L6-v2) downloaded and verified
- [ ] ChromaDB instance accessible and schema initialized
- [ ] All 9 CanvasContextTemplate YAML files present and valid
- [ ] Copilot kill switch service running independently from trading kill switch
- [ ] FloorManager Opus-tier provider configured and reachable
- [ ] Redis Streams department mail service operational
- [ ] Frontend build passes with no TypeScript errors
- [ ] Test environment provisioned with isolated database (graph_memory.db test instance)

---

## Exit Criteria

- [ ] All P0 tests passing (100% pass rate required)
- [ ] All P1 tests passing (>= 95% pass rate; waivers for known issues)
- [ ] No open high-priority / high-severity bugs (R-001 through R-007)
- [ ] Streaming latency within NFR-P3 (<= 5 seconds first token)
- [ ] Opinion node SUPPORTED_BY edge constraint verified in DB
- [ ] Session isolation verified (draft nodes invisible to other sessions)
- [ ] Copilot kill switch architecturally confirmed independent from trading kill switch
- [ ] Test coverage agreed as sufficient (>= 80% critical paths covered)

---

## Project Team (Optional)

| Name       | Role            | Testing Responsibilities                                  |
| ---------- | --------------- | -------------------------------------------------------- |
| Mubarak    | QA Lead         | Test design, P0/P1 execution, risk assessment            |
| (Dev Team) | Backend         | Unit tests for graph memory, FloorManager, kill switch  |
| (Dev Team) | Frontend        | Component tests for CopilotPanel, WorkshopCanvas        |
| (Dev Team) | Platform        | Environment setup, Redis/DB infrastructure validation   |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (>= 6) + No workaround

| Requirement                                                          | Test Level      | Risk Link | Test Count | Owner | Notes                                       |
| -------------------------------------------------------------------- | --------------- | --------- | ---------- | ----- | ------------------------------------------- |
| Graph memory session isolation: draft nodes invisible to other sessions | Integration     | R-001     | 3          | QA    | Test draft vs committed visibility boundary  |
| OPINION node mandatory SUPPORTED_BY edge constraint                   | DB/Integration | R-002     | 2          | QA    | Verify orphaned OPINION nodes rejected       |
| Vector embedding semantic search relevance threshold                   | Unit            | R-003     | 2          | QA    | Test cosine similarity threshold enforcement |
| Copilot kill switch architectural independence from trading kill switch | Architecture    | R-004     | 2          | QA    | Verify no shared state/imports              |
| SSE streaming token integrity (no loss/corruption)                   | Integration     | R-005     | 3          | QA    | Token count verification end-to-end         |
| CanvasContextTemplate token budget enforcement                        | Unit            | R-006     | 2          | QA    | Verify max_identifiers truncation           |
| FloorManager department routing accuracy                              | Integration     | R-007     | 3          | QA    | Test each department classification path    |
| ReflectionExecutor draft-to-commit promotion                          | Integration     | R-008     | 2          | QA    | Critical path for memory persistence        |
| Session recovery returns correct committed state                      | Integration     | R-009     | 2          | QA    | Simulate crash recovery scenario            |
| CopilotPanel streaming UI: token-by-token rendering                   | Component       | R-005     | 2          | QA    | Verify no dropped tokens in UI              |
| FloorManager error handling: provider timeout/rate limit              | Integration     | R-007     | 2          | QA    | Graceful degradation with retry             |
| Kill switch activation terminates running agent tasks                  | Integration     | R-004     | 2          | QA    | Verify task cancellation + state cleanup    |

**Total P0**: 27 tests, ~27 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement                                                     | Test Level      | Risk Link | Test Count | Owner | Notes                                        |
| -------------------------------------------------------------- | --------------- | --------- | ---------- | ----- | -------------------------------------------- |
| CanvasContextTemplate loader per canvas                        | Unit            | -         | 2          | DEV   | 9 templates, verify load per canvas_id       |
| CanvasContextService: context metadata passed to FloorManager   | Integration     | R-014     | 2          | QA    | Verify canvas_context field in API request  |
| Intent classification: STRATEGY_PAUSE/RESUME detection         | Unit            | R-012     | 3          | DEV   | Test pattern matching per command type       |
| Intent classification confidence scoring (threshold 0.7)         | Unit            | R-012     | 2          | DEV   | Verify clarification flow below threshold   |
| NL command confirmation flow (destructive commands)              | Integration     | R-012     | 3          | QA    | Test confirm/cancel flow                     |
| Canvas context binding: live trading data resolution             | Integration     | R-013     | 2          | QA    | On live_trading canvas, verify positions    |
| Canvas context binding: risk data resolution                    | Integration     | R-013     | 2          | QA    | On risk canvas, verify regime/drawdown      |
| Canvas context binding: portfolio data resolution                | Integration     | R-013     | 2          | QA    | On portfolio canvas, verify accounts        |
| SuggestionChipBar: chips load per canvas context                | Component       | R-013     | 3          | QA    | Verify chip list changes on canvas switch   |
| Cross-canvas navigation: BotStatusCard 3-dot menu               | Component       | R-013     | 2          | QA    | Verify menu options per entity type         |
| CopilotPanel streaming: cursor blink animation (600ms)           | Component       | -         | 2          | QA    | Verify cursor visibility cycle               |
| CopilotPanel streaming: auto-scroll pause on scroll up          | Component       | -         | 2          | QA    | Verify autoScroll flag toggles correctly    |
| Tool call UI: pulsing dot animation                             | Component       | -         | 2          | QA    | Verify dot animation during tool exec       |
| Kill switch status persisted across page refresh                  | Integration     | R-004     | 2          | QA    | Verify onMount syncs UI state               |
| Stale draft cleanup: configurable threshold                       | Integration     | R-010     | 2          | QA    | Verify cleanup at threshold boundary        |

**Total P1**: 33 tests, ~22 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement                                                     | Test Level      | Risk Link | Test Count | Owner | Notes                                        |
| -------------------------------------------------------------- | --------------- | --------- | ---------- | ----- | -------------------------------------------- |
| OPINION node schema validation (all required fields)            | Unit            | R-002     | 3          | DEV   | action, reasoning, confidence, alternatives  |
| Embedding generation pipeline (all-MiniLM-L6-v2)                 | Unit            | R-003     | 2          | DEV   | Verify BLOB stored correctly                 |
| Cosine similarity search with null embeddings                  | Unit            | R-003     | 2          | DEV   | Graceful handling of missing embeddings     |
| CanvasContextTemplate: max_identifiers truncation               | Unit            | R-006     | 2          | DEV   | Verify truncation at 50 identifiers          |
| CanvasContextTemplate YAML schema validation                    | Unit            | -         | 3          | DEV   | Each of 9 YAML files validates schema        |
| Session checkpoint interval configuration                       | Unit            | R-008     | 2          | DEV   | Default 5 min; verify configurable           |
| Milestone-based checkpoint trigger                             | Unit            | R-008     | 2          | DEV   | Verify trigger after significant actions    |
| Memory explorer: node detail expansion                          | Component       | -         | 2          | DEV   | Verify expand/collapse per node             |
| Memory explorer: filter by node type                           | Component       | -         | 3          | DEV   | OPINION, OBSERVATION, WORLD, DECISION, etc.  |
| Skill browser: skill list renders correctly                     | Component       | -         | 2          | DEV   | Verify name, slash cmd, usage count          |
| Skill invocation from sidebar click                             | Integration     | -         | 2          | QA    | Verify slash cmd populates input             |
| Workshop morning digest auto-trigger on first daily open         | Integration     | R-016     | 2          | QA    | Verify localStorage check triggers digest   |
| Workshop left sidebar navigation (New Chat, History, etc.)       | Component       | -         | 4          | DEV   | 5 sidebar items tested                      |
| Kill switch resume clears interrupted state                      | Integration     | R-017     | 2          | QA    | Verify terminated_tasks cleared             |
| Kill switch history: activation logged with UTC timestamp       | Unit            | -         | 2          | DEV   | Verify audit trail entries                  |

**Total P2**: 35 tests, ~17.5 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement                                                     | Test Level      | Test Count | Owner | Notes                                        |
| -------------------------------------------------------------- | --------------- | ---------- | ----- | -------------------------------------------- |
| Performance benchmark: Copilot first-token latency              | Performance     | 5          | QA    | NFR-P3: <= 5 seconds (measure 20 runs)      |
| Performance benchmark: Large node tree rendering (Memory Explorer) | Performance     | 3          | QA    | Verify UI responsive with 100+ nodes        |
| Suggestion chip dynamic update: live state integration           | Integration     | 2          | QA    | Deferred in 5.9 - requires WebSocket        |
| Cross-canvas navigation: entity preservation after navigation   | E2E             | 3          | QA    | Navigate away and back; verify state         |
| Conversation history limit: 10 messages enforcement              | Integration     | 2          | QA    | Verify oldest messages dropped at limit     |
| Error boundary: provider outage during streaming                 | Integration     | 2          | QA    | Simulate provider failure mid-stream        |
| Exploratory: Chat history persists across browser refresh        | E2E             | 2          | QA    | Manual validation recommended                |
| Exploratory: Multiple canvas sessions open simultaneously         | E2E             | 2          | QA    | Memory isolation across tabs               |

**Total P3**: 21 tests, ~5.25 hours

---

## Execution Order

### Smoke Tests (< 5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Graph memory DB migration: session_status and embedding columns exist (5s)
- [ ] CanvasContextTemplate loader: verify workshop.yaml loads without error (5s)
- [ ] FloorManager: health check endpoint returns 200 (5s)
- [ ] Copilot kill switch: status endpoint returns inactive (5s)
- [ ] CopilotPanel: component renders without console errors (10s)
- [ ] Intent classifier: classify "pause GBPUSD" returns non-null intent (5s)
- [ ] SuggestionChipBar: renders at least 1 chip on workshop canvas (5s)

**Total**: 7 scenarios

### P0 Tests (< 30 min)

**Purpose**: Critical path validation

- [ ] R-001: Graph memory draft nodes invisible to other sessions (E2E)
- [ ] R-001: Graph memory committed nodes visible to other sessions (E2E)
- [ ] R-001: Graph memory committed nodes visible within same session (E2E)
- [ ] R-002: OPINION node creation requires SUPPORTED_BY edge (DB)
- [ ] R-002: OPINION node without SUPPORTED_BY edge rejected (DB)
- [ ] R-003: Semantic search returns nodes above cosine threshold (Unit)
- [ ] R-003: Semantic search returns empty for irrelevant query (Unit)
- [ ] R-004: Copilot kill switch activate does NOT import trading kill switch (Architecture)
- [ ] R-004: Copilot kill switch activate does NOT affect trading endpoints (Integration)
- [ ] R-005: SSE streaming: verify token count matches (Integration)
- [ ] R-005: SSE streaming: verify no partial tokens (Integration)
- [ ] R-005: SSE streaming: verify stream completes (Integration)
- [ ] R-006: Token budget: verify max_identifiers enforcement (Unit)
- [ ] R-006: Token budget: verify truncation at 50 identifiers (Unit)
- [ ] R-007: FloorManager routes "run research" to Research dept (Integration)
- [ ] R-007: FloorManager routes "compile EA" to Development dept (Integration)
- [ ] R-007: FloorManager routes "show risk" to Risk dept (Integration)
- [ ] R-008: ReflectionExecutor promotes valid draft nodes (Integration)
- [ ] R-008: ReflectionExecutor rejects invalid draft nodes (Integration)
- [ ] R-009: Session recovery loads committed state correctly (Integration)
- [ ] R-009: Session recovery handles no-committed-state edge case (Integration)
- [ ] R-005: CopilotPanel token rendering: verify no dropped tokens (Component)
- [ ] R-005: CopilotPanel token rendering: verify token append order (Component)
- [ ] R-007: FloorManager timeout returns graceful error (Integration)
- [ ] R-007: FloorManager rate limit returns graceful error (Integration)
- [ ] R-004: Kill switch terminates chat_stream mid-execution (Integration)
- [ ] R-004: Kill switch clears active task registry (Integration)

**Total**: 27 scenarios

### P1 Tests (< 45 min)

**Purpose**: Important feature coverage

- [ ] CanvasContextTemplate loader: verify all 9 templates load (Unit)
- [ ] CanvasContextService: verify canvas_context in API body (Integration)
- [ ] Intent: STRATEGY_PAUSE pattern detection (Unit)
- [ ] Intent: STRATEGY_RESUME pattern detection (Unit)
- [ ] Intent: POSITION_CLOSE pattern detection (Unit)
- [ ] Intent: confidence below 0.7 triggers clarification (Unit)
- [ ] NL confirmation: destructive command prompts confirm (Integration)
- [ ] NL confirmation: cancel returns to conversation (Integration)
- [ ] NL confirmation: confirm executes action (Integration)
- [ ] Canvas binding: live_trading canvas includes positions (Integration)
- [ ] Canvas binding: risk canvas includes regime data (Integration)
- [ ] Canvas binding: portfolio canvas includes accounts (Integration)
- [ ] SuggestionChipBar: chips change on canvas switch (Component)
- [ ] SuggestionChipBar: chip click triggers command (Component)
- [ ] CrossCanvasMenu: BotStatusCard shows 3-dot menu (Component)
- [ ] CrossCanvasMenu: "View Code" navigates to Development (Integration)
- [ ] Cursor blink: 600ms animation cycle verified (Component)
- [ ] Auto-scroll: pauses when user scrolls up (Component)
- [ ] Auto-scroll: resumes when new message arrives (Component)
- [ ] Tool call UI: pulsing dot shows during execution (Component)
- [ ] Tool call UI: collapses to checkmark on complete (Component)
- [ ] Kill switch: status persists after page refresh (Integration)
- [ ] Stale draft: cleanup runs at threshold boundary (Integration)
- [ ] Stale draft: valid drafts above threshold preserved (Integration)

**Total**: 25 scenarios

### P2/P3 Tests (< 60 min)

**Purpose**: Full regression coverage

- [ ] OPINION schema: action field required (Unit)
- [ ] OPINION schema: reasoning field required (Unit)
- [ ] OPINION schema: confidence field required (Unit)
- [ ] Embedding pipeline: BLOB stored in DB (Unit)
- [ ] Embedding search: null embedding handled gracefully (Unit)
- [ ] Token budget: truncation deterministic (Unit)
- [ ] Template schema: all 9 YAML files validate (Unit)
- [ ] Checkpoint interval: configurable via env var (Unit)
- [ ] Milestone trigger: fires after significant action (Unit)
- [ ] Memory explorer: node expand/collapse (Component)
- [ ] Memory explorer: filter by OPINION type (Component)
- [ ] Memory explorer: filter by OBSERVATION type (Component)
- [ ] Memory explorer: filter by WORLD type (Component)
- [ ] Skill browser: skill list renders (Component)
- [ ] Skill browser: slash command populated on click (Integration)
- [ ] Morning digest: auto-trigger on first daily open (Integration)
- [ ] Morning digest: no duplicate on subsequent opens (Integration)
- [ ] Workshop sidebar: New Chat navigation (Component)
- [ ] Workshop sidebar: History navigation (Component)
- [ ] Workshop sidebar: Projects/Workflows navigation (Component)
- [ ] Workshop sidebar: Memory navigation (Component)
- [ ] Workshop sidebar: Skills navigation (Component)
- [ ] Kill switch resume: terminated_tasks cleared (Integration)
- [ ] Kill switch history: UTC timestamp logged (Unit)
- [ ] Kill switch history: activator recorded (Unit)
- [ ] Perf: first-token latency benchmark (Performance)
- [ ] Perf: large node tree render time (Performance)
- [ ] Suggestion chips: dynamic update deferred (Integration)
- [ ] Cross-canvas: entity preserved after nav (E2E)
- [ ] Conversation history: 10-message limit (Integration)
- [ ] Error boundary: provider outage mid-stream (Integration)
- [ ] Exploratory: chat history persists (E2E)
- [ ] Exploratory: multi-tab session isolation (E2E)

**Total**: 34 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes                         |
| -------- | ----- | ---------- | ----------- | ----------------------------- |
| P0       | 27    | 1.0        | 27          | Complex integration setup     |
| P1       | 25    | 0.75       | 18.75       | Standard coverage             |
| P2       | 25    | 0.5        | 12.5        | Unit/component tests          |
| P3       | 21    | 0.25       | 5.25        | Exploratory/performance       |
| **Total**| **98**| **-**      | **~63.5**   | **~8 days (1.5 weeks)**       |

### Prerequisites

**Test Data:**

- `GraphMemoryFactory` fixture: Creates test nodes with session_status='draft' and 'committed', auto-cleanup
- `CanvasContextTemplateFactory` fixture: Provides valid YAML templates for each canvas, temp directory
- `SessionCheckpointFactory` fixture: Creates test sessions with checkpoints, auto-cleanup
- `CopilotConversationFactory` fixture: Creates conversation history, auto-cleanup
- `IntentTestDataFactory`: Provides sample NL commands for classification testing

**Tooling:**

- `pytest` for backend Python tests (unit, integration)
- `pytest-asyncio` for async test support
- `playwright` for E2E frontend tests
- `pytest-cov` for coverage reporting
- `locust` for performance/load testing (optional)
- `faker` for synthetic data generation

**Environment:**

- PostgreSQL/SQLite test instance for graph_memory.db
- ChromaDB test instance (in-memory or temp dir)
- Redis test instance for Streams
- Mock LLM provider endpoint for deterministic testing
- Static adapter frontend test server

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: >= 95% (waivers required for failures)
- **P2/P3 pass rate**: >= 90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: >= 80%
- **Security scenarios (SEC category)**: 100%
- **Business logic**: >= 70%
- **Edge cases**: >= 50%
- **Graph memory operations**: >= 90%
- **Copilot streaming flow**: 100%

### Non-Negotiable Requirements

- [ ] All P0 tests pass
- [ ] No high-risk (>= 6) items unmitigated
- [ ] Security tests (SEC category, R-014) pass 100%
- [ ] Performance targets met (NFR-P3: first token <= 5s)
- [ ] OPINION node SUPPORTED_BY constraint verified
- [ ] Session isolation verified (R-001)
- [ ] Kill switch architectural independence confirmed (R-004)

---

## Mitigation Plans

### R-001: Graph Memory Session Isolation Failure (Score: 9)

**Mitigation Strategy:** Comprehensive integration tests covering:
1. Draft nodes not visible in other sessions' queries
2. Committed nodes visible across sessions
3. Same-session visibility for both draft and committed

**Owner:** QA Lead + Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run integration test `test_session_isolation_draft_committed` with 3 sessions simultaneously querying same graph namespace

### R-002: OPINION Node Orphaned from SUPPORTED_BY Edge (Score: 9)

**Mitigation Strategy:**
1. Database-level constraint: NOT NULL SUPPORTED_BY edge reference
2. Application-level enforcement in `create_opinion_node()` - check edge exists before returning
3. Unit test verifies orphaned OPINION cannot be created

**Owner:** Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run test `test_opinion_requires_supported_by_edge` - expect DB constraint rejection

### R-003: Vector Embedding Quality Degradation (Score: 6)

**Mitigation Strategy:**
1. Threshold-based recall tests: cosine similarity >= 0.7 for relevant queries
2. Periodic model health check (weekly batch job)
3. Fallback to keyword search if similarity score low

**Owner:** QA/ML
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Run `test_embedding_similarity_threshold` with known relevant and irrelevant queries

### R-004: Copilot Kill Switch Blocks Trading (Score: 6)

**Mitigation Strategy:**
1. Architectural isolation test: verify `src/router/copilot_kill_switch.py` does not import `src/router/kill_switch.py`
2. Integration test: activate copilot kill switch, verify trading endpoints unaffected
3. Code review gate: PR must include isolation verification test

**Owner:** Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run `test_copilot_kill_switch_independence` and `test_trading_endpoints_unaffected_by_copilot_kill`

### R-005: SSE Streaming Token Loss (Score: 6)

**Mitigation Strategy:**
1. Token count verification: count tokens sent vs tokens received
2. Partial line buffer handling test (already in code review fixes)
3. E2E test with known token sequence

**Owner:** QA/Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run `test_sse_token_integrity` with 100-token known sequence

### R-006: CanvasContextTemplate Token Budget Exceeded (Score: 6)

**Mitigation Strategy:**
1. Unit test for token counting logic
2. Verify max_identifiers truncation at configured limit (default 50)
3. Integration test with template exceeding budget

**Owner:** Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run `test_token_budget_enforcement` with template having 100 identifiers

### R-007: FloorManager Routing Misclassification (Score: 6)

**Mitigation Strategy:**
1. Integration tests for each department routing path (Research, Development, Risk, Trading, Portfolio)
2. Confidence scoring threshold tests
3. Fallback to general query for unknown intents

**Owner:** QA/Backend
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Run `test_department_routing_*` tests for all 5 departments

---

## Assumptions and Dependencies

### Assumptions

1. Graph memory database schema migration (adding session_status, embedding columns) will be applied before P0 test execution
2. Sentence-transformers model (all-MiniLM-L6-v2) will be downloaded and cached on test environment first run
3. ChromaDB test instance will be available as in-memory or temp directory (not affecting production)
4. Mock LLM provider endpoint will be configured for deterministic intent classification testing
5. All 9 CanvasContextTemplate YAML files will be present in `src/canvas_context/templates/`
6. Epic 2 provider configuration tests pass (Opus-tier model reachable)
7. Redis Streams department mail service is operational for integration tests

### Dependencies

1. **Epic 2 (Provider Config)**: Opus-tier model must be configured and reachable for FloorManager chat tests
2. **Epic 3 (Trading Kill Switch)**: Must remain architecturally separate from Copilot kill switch (no shared code)
3. **Epic 7 (Department System)**: Redis Streams must be working for department routing tests
4. **Story 5.1 (Graph Memory Completion)**: Database columns (session_status, embedding) must be migrated
5. **Story 5.3 (Canvas Context)**: All 9 YAML templates must be valid and loadable

### Risks to Plan

- **Risk**: ChromaDB in-memory mode behaves differently from persistent mode in embedding tests
  - **Impact**: Embedding tests pass locally but fail in CI with persistent ChromaDB
  - **Contingency**: Use Docker container with persistent volume for CI embedding tests

- **Risk**: Mock LLM provider not deterministic for intent classification confidence tests
  - **Impact**: Confidence threshold tests (R-012) may flap
  - **Contingency**: Use deterministic pattern-matching fallback for confidence tests

- **Risk**: Session checkpoint tests require real async behavior; mocked tests may miss timing bugs
  - **Impact**: Race conditions in checkpoint reflection trigger not caught
  - **Contingency**: Add integration test with real asyncio timer in test environment

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: _______________ Date: _______________
- [ ] Tech Lead: _______________ Date: _______________
- [ ] QA Lead: _______________ Date: _______________

**Comments:**

---

## Interworking & Regression

| Service/Component        | Impact                                           | Regression Scope                                  |
| ------------------------ | ------------------------------------------------ | ------------------------------------------------ |
| **Graph Memory (src/memory/graph/)** | Core memory backbone for all agent sessions; session isolation critical | Existing memory tests (tests/memory/) must pass; graph memory schema migration verified |
| **FloorManager (src/agents/departments/floor_manager.py)** | Central routing hub for all Copilot messages; kill switch integration | All floor manager endpoint tests must pass |
| **SessionCheckpointService (src/agents/memory/session_checkpoint_service.py)** | Persists agent session state; triggers ReflectionExecutor | tests/memory/test_session_checkpoint_flow.py, tests/memory/test_session_recovery.py |
| **CanvasContextTemplate (src/canvas_context/)** | Context provider for all 9 canvases; CAG+RAG pattern | tests/canvas_context/test_template_loader.py, test_context_integration.py |
| **CopilotPanel (quantmind-ide/...)** | Main UI for Copilot; streaming, conversation, kill switch | tests/api/test_workshop_copilot.py, tests/api/test_copilot_kill_switch.py |
| **Intent System (src/intent/)** | NL command classification; confirmation flows | tests/intent/test_classifier.py |
| **Kill Switch Router (src/router/copilot_kill_switch.py)** | Must remain independent from trading kill switch | tests/api/test_copilot_kill_switch.py |
| **Workshop Canvas (quantmind-ide/src/lib/components/canvas/)** | Full Copilot home UI; morning digest, skill browser, memory explorer | E2E tests for WorkshopCanvas.svelte |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- PRD: `_bmad-output/planning-artifacts/project-overview.md`
- Epic 5: `_bmad-output/implementation-artifacts/sprint-status.yaml` (epic-5: done)
- Stories 5.0-5.9: `_bmad-output/implementation-artifacts/5-*-*.md`
- Architecture: `_bmad-output/planning-artifacts/architecture.md` (6.1 Graph Memory, 6.2 Canvas Context System, 10 Department System)
- Tech Spec: `src/memory/graph/types.py`, `src/agents/departments/floor_manager.py`, `src/router/copilot_kill_switch.py`

### Existing Test Files (Epic 5 Coverage)

| Test File                                                    | Coverage                        |
| ------------------------------------------------------------ | ------------------------------- |
| `tests/memory/test_session_checkpoint_flow.py`               | Story 5.2: Checkpoint flow      |
| `tests/memory/test_session_recovery.py`                      | Story 5.2: Session recovery     |
| `tests/memory/graph/test_session_isolation.py`               | Story 5.1: Session isolation    |
| `tests/canvas_context/test_template_loader.py`               | Story 5.3: Template loading     |
| `tests/canvas_context/test_context_integration.py`           | Story 5.3: Context integration  |
| `tests/api/test_workshop_copilot.py`                         | Story 5.4-5.5: Copilot wiring   |
| `tests/api/test_copilot_kill_switch.py`                      | Story 5.6: Kill switch          |
| `tests/intent/test_classifier.py`                            | Story 5.7: Intent classification |
| `tests/api/services/test_chat_memory_integration.py`         | Cross-story: Chat + memory      |

### Gap Analysis

**Covered by existing tests:**
- Graph memory schema (columns, types)
- Session checkpoint flow
- Session recovery
- Canvas context template loading
- Copilot kill switch (basic activation)
- Intent classification (unit level)

**New tests required (P0):**
- Session isolation (draft vs committed cross-session)
- OPINION SUPPORTED_BY edge constraint
- SSE streaming token integrity
- Token budget enforcement
- Department routing accuracy
- Kill switch architectural independence

**New tests required (P1):**
- Canvas context binding per canvas
- NL confirmation flow
- Streaming UI (cursor, auto-scroll, tool call)
- Cross-canvas navigation

**New tests required (P2/P3):**
- Performance benchmarks
- Memory explorer UI
- Skill browser
- Morning digest auto-trigger

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
