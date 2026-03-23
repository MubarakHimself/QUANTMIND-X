---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 7
epic_title: 'Department Agent Platform'
---

# Test Design: Epic 7 - Department Agent Platform

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft
**Mode:** Epic-Level (sprint-status.yaml detected)

---

## Executive Summary

**Scope:** Full test design for Epic 7 (Department Agent Platform)

**Epic 7 Stories:**
- 7-0: Department System Audit (done)
- 7-1: Research Department Real Hypothesis Generation (done)
- 7-2: Development Department MQL5 EA Code Generation (done)
- 7-3: MQL5 Compilation Integration (done)
- 7-4: Skill Catalogue - Registry, Authoring & Skill Forge (review)
- 7-5: Session Workspace Isolation for Concurrent Sessions (done)
- 7-6: Department Mail Redis Streams Migration (review)
- 7-7: Concurrent Task Routing - 5 Simultaneous Tasks (review)
- 7-8: Risk, Trading & Portfolio Department Real Implementations (review)
- 7-9: Department Kanban Sub-page UI (done)

**Risk Summary:**
- Total risks identified: 9
- High-priority risks (Score >= 6): 4
- Critical categories: TECH (Redis migration, session isolation, concurrent routing)

**Coverage Summary:**
- P0 scenarios: 8 tests (~16 hours)
- P1 scenarios: 12 tests (~12 hours)
- P2/P3 scenarios: 15 tests (~8 hours)
- **Total effort**: ~36 hours (~5 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | -------------- | --------------------- |
| **Legacy SQLite DepartmentMailService** | Being replaced by Redis Streams in Story 7-6 | New code uses Redis; SQLite deprecated |
| **MT5 Live Trading Integration** | Compilation only, no live trading on Contabo | Story 3.x covers live trading separately |
| **Contabo Server Infrastructure** | Docker MT5 compiler is external service | Tested via API contract; infrastructure team owns |
| **Frontend Svelte 5 Component Internal Refactoring** | Story 7-9 UI complete; internal refactoring is cosmetic | Covered by visual regression tests |
| **Third-party Claude Agent SDK Behavior** | External Haiku-tier agent calls are out of scope | Mocked in unit tests; integration tests verify contract |

---

## Risk Assessment

### High-Priority Risks (Score >= 6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- | -------- |
| R-001 | TECH | Redis Streams migration breaks existing SQLite-based department mail causing message loss or ordering violations | 2 | 3 | 6 | Dual-write pattern during migration; rollback procedure; message replay capability | Dev Lead | Before 7-6 merge |
| R-002 | TECH | Session workspace isolation failure causes cross-session data contamination in graph memory | 2 | 3 | 6 | Strict session_id filtering in graph queries; integration tests with 2 concurrent sessions | QA Lead | Before 7-5 merge |
| R-003 | TECH | Concurrent task routing with HIGH priority preemption causes task state corruption or deadlock | 2 | 3 | 6 | Redis atomic operations; task state machine validation; deadlock detection | Dev Lead | Before 7-7 merge |
| R-004 | PERF | MQL5 auto-correction loop (max 2 iterations) exhausts retries and escalates too frequently, blocking pipeline | 2 | 2 | 4 | Exponential backoff on corrections; threshold tuning; manual escalation option | QA Lead | Before 7-3 merge |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- |
| R-005 | TECH | Skill Forge produces invalid skill.md with malformed YAML or missing required fields | 2 | 2 | 4 | JSON Schema validation on skill authoring; unit tests for schema validation | Dev |
| R-006 | TECH | Department Head implementations (Risk/Trading/Portfolio) have incomplete edge case handling | 2 | 2 | 4 | Comprehensive AC mapping; edge case test suite; code review gate | QA Lead |
| R-007 | PERF | SSE endpoint for Kanban real-time updates causes connection saturation with 100+ concurrent users | 1 | 3 | 3 | WebSocket upgrade path; connection pooling; load testing | Dev |
| R-008 | DATA | Portfolio report P&L attribution calculations have rounding errors across broker boundaries | 2 | 2 | 4 | Decimal precision validation; tolerance thresholds; reconciliation tests | QA |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------- |
| R-009 | BUS | Kanban UI 400ms cyan border animation causes visual jitter on low-end devices | 1 | 1 | 1 | Monitor |
| R-010 | OPS | Department Kanban SSE mock data not replaced with TaskRouter integration (follow-up item) | 2 | 1 | 2 | Monitor |
| R-011 | BUS | Duration timer format (Xm / Xh Ym / Xd) edge case for tasks running > 7 days | 1 | 1 | 1 | Monitor |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] Requirements and assumptions agreed upon by QA, Dev, PM
- [ ] Test environment provisioned with Redis, PostgreSQL, and Docker accessible
- [ ] Contabo server with Docker MT5 compiler reachable from test environment
- [ ] All 10 Epic 7 stories have implementation PRs opened
- [ ] Department Head base class tests passing
- [ ] Graph memory store accessible for session isolation tests

---

## Exit Criteria

- [ ] All P0 tests passing (100% required)
- [ ] All P1 tests passing (or failures triaged with waivers)
- [ ] No open high-priority / high-severity bugs (Score >= 6 unmitigated)
- [ ] Redis Streams migration verified: zero message loss, <=500ms latency
- [ ] Session isolation verified: zero cross-session contamination in 2 concurrent session test
- [ ] Concurrent task routing verified: 5 simultaneous tasks complete within 1.2x parallel overhead
- [ ] MQL5 compilation auto-correction: <= 2 iterations before escalation
- [ ] Test coverage agreed as sufficient (>= 80% critical paths)

---

## Project Team (Optional)

| Name | Role | Testing Responsibilities |
| ---- | -------- | ------------------------ |
| Mubarak | QA Lead | P0/P1 test ownership; risk assessment sign-off |
| Dev Team | Developers | Unit tests for all new code; integration tests for API endpoints |
| PM | Product Manager | Acceptance criteria validation; business logic review |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Core functionality + High risk (Score >= 6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| Redis Streams migration - message delivery | Integration | R-001 | 3 | QA | Dual-write, rollback, message replay |
| Session isolation - concurrent sessions | Integration | R-002 | 3 | QA | Cross-session contamination detection |
| Concurrent task routing - priority preemption | Integration | R-003 | 4 | QA | 5-task concurrent dispatch with HIGH preempt |
| MQL5 compilation auto-correction | Unit | R-004 | 2 | DEV | Max 2 iterations, escalation verification |
| Research hypothesis confidence scoring | Unit | - | 3 | DEV | Confidence >= 0.75 triggers TRD escalation |
| Development TRD parsing and EA generation | Unit | - | 4 | DEV | TRD validation, MQL5 syntax correctness |
| Skill Forge schema validation | Unit | R-005 | 2 | DEV | Invalid YAML detection |
| Portfolio P&L attribution accuracy | Unit | R-008 | 2 | QA | Decimal precision, broker reconciliation |

**Total P0**: 23 tests, ~16 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (Score 3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| Redis Streams consumer group recovery | Integration | R-001 | 3 | QA | Offline consumer replay, <=500ms latency |
| Session commit workflow | Unit | R-002 | 3 | DEV | Draft to committed state transition |
| Task dependency chain with timeout | Integration | R-003 | 2 | QA | Dependent task waits, timeout enforcement |
| Risk Department backtest 6-mode execution | Integration | R-006 | 4 | QA | All modes, pass/fail verdict validation |
| Trading Department paper trade monitoring | Unit | R-006 | 3 | DEV | P&L tracking, win/loss ratio, regime correlation |
| Portfolio Department report API | API | R-008 | 3 | QA | Total equity, attribution, drawdown endpoints |
| Skill catalogue 12 core skills registration | Unit | - | 3 | DEV | All skills registered with correct metadata |
| Department Kanban SSE real-time updates | Component | R-007 | 2 | DEV | 400ms cyan border flash, targeted DOM update |
| Skill execution with usage_count tracking | Unit | - | 2 | DEV | Execution timing, counter increment |
| Department Head base class extension | Unit | R-006 | 3 | DEV | All heads implement process_task correctly |

**Total P1**: 28 tests, ~14 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| MQL5 compilation success path | Unit | 2 | DEV | .ex5 stored, compile_status=success |
| MQL5 compilation failure path | Unit | 2 | DEV | Error parsing, auto-correct attempts |
| Opinion node writing after research | Integration | 2 | DEV | Memory graph integration |
| Sub-agent result merging | Unit | 2 | DEV | Parallel research, merge visibility |
| TaskRouter wall-clock overhead <= 1.2x | Integration | 2 | QA | Performance test for parallelism |
| Priority enum (HIGH/MEDIUM/LOW) | Unit | 2 | DEV | Preemptive scheduling logic |
| Session workspace isolation conflict detection | Integration | 2 | QA | Same strategy namespace conflict |
| Kanban duration timer format | Component | 3 | DEV | Xm / Xh Ym / Xd formats |
| Department mail broadcast capability | Integration | 2 | QA | Broadcast to all departments |
| Backtest pass/fail verdict (>= 4/6 modes) | Integration | 2 | QA | Sharpe >= 1.0, DD <= 15%, WR >= 50% |

**Total P2**: 21 tests, ~7 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| Kanban animation performance on low-end devices | Component | 1 | QA | Visual jitter detection |
| SSE connection saturation (>100 users) | Performance | 1 | QA | Load test |
| Long-duration task timer (>7 days) | Component | 1 | DEV | Edge case for duration format |
| Skill Forge edge cases (empty description, missing SOP) | Unit | 2 | DEV | Schema validation limits |
| Concurrent 10-task routing stress test | Performance | 1 | QA | Beyond 5-task baseline |

**Total P3**: 6 tests, ~2 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Department Head base class initializes correctly (1s)
- [ ] Skill Manager loads 12 core skills (2s)
- [ ] Redis Streams client connects (1s)
- [ ] Graph memory store responds to ping (1s)
- [ ] TaskRouter priority enum values correct (1s)

**Total**: 5 scenarios (~6s)

### P0 Tests (<30 min)

**Purpose**: Critical path validation

- [ ] Redis Streams message delivery (integration, 3 tests, 5min)
- [ ] Session isolation with 2 concurrent sessions (integration, 3 tests, 8min)
- [ ] Concurrent task routing 5 tasks + HIGH preemption (integration, 4 tests, 10min)
- [ ] MQL5 auto-correction (unit, 2 tests, 2min)
- [ ] Research hypothesis confidence scoring (unit, 3 tests, 3min)
- [ ] Development TRD parsing and EA generation (unit, 4 tests, 4min)

**Total**: 19 scenarios (~32min)

### P1 Tests (<45 min)

**Purpose**: Important feature coverage

- [ ] Consumer group recovery (integration, 3 tests, 6min)
- [ ] Session commit workflow (unit, 3 tests, 4min)
- [ ] Task dependency chain (integration, 2 tests, 5min)
- [ ] Risk backtest 6-mode execution (integration, 4 tests, 10min)
- [ ] Trading paper trade monitoring (unit, 3 tests, 5min)
- [ ] Portfolio report API (API, 3 tests, 6min)
- [ ] Remaining P1 tests (10 tests, ~12min)

**Total**: 28 scenarios (~48min)

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] P2 tests (21 tests, ~30min)
- [ ] P3 tests (6 tests, ~10min)
- [ ] Performance tests (3 tests, ~15min)

**Total**: 30 scenarios (~55min)

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| --------- | ----- | ---------- | ----------------- | ----------------------- |
| P0 | 23 | 0.7 | ~16 | Complex Redis/async integration |
| P1 | 28 | 0.5 | ~14 | Standard coverage |
| P2 | 21 | 0.33 | ~7 | Simple scenarios |
| P3 | 6 | 0.33 | ~2 | Exploratory |
| **Total** | **78** | **-** | **~39 hours** | **~5 days** |

### Prerequisites

**Test Data:**
- Mock TRD documents with all parameter variations (faker-based)
- Mock strategy objects for Risk/Trading/Portfolio departments
- Redis test instance with clean stream state
- Graph memory test store with isolated sessions

**Tooling:**
- pytest for Python backend tests
- Vitest for Svelte frontend tests
- Playwright for E2E component tests
- Docker for MT5 compiler integration tests

**Environment:**
- Redis 7.x accessible (for Streams)
- PostgreSQL for graph memory
- Docker daemon for MQL5 compilation tests
- Contabo server reachable (or mocked for CI)

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: >= 95% (waivers required for failures)
- **P2/P3 pass rate**: >= 90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: >= 80%
- **Redis Streams patterns**: 100% (message delivery, consumer groups, replay)
- **Session isolation**: 100% (cross-contamination detection)
- **Concurrent routing**: 100% (5-task dispatch, priority preemption)
- **Business logic**: >= 70%
- **Edge cases**: >= 50%

### Non-Negotiable Requirements

- [ ] All P0 tests pass (R-001 through R-004 mitigated)
- [ ] No high-risk (>= 6) items unmitigated
- [ ] Redis Streams tests: zero message loss verified
- [ ] Session isolation tests: zero cross-contamination verified
- [ ] Concurrent routing tests: <= 1.2x overhead verified

---

## Mitigation Plans

### R-001: Redis Streams Migration (Score: 6)

**Mitigation Strategy:** Implement dual-write pattern during migration period. SQLite writes continue until Redis consumer group catches up. Message replay from Redis on startup.

**Owner:** Dev Lead
**Timeline:** Before 7-6 merge
**Status:** Planned
**Verification:** Integration test with 1000 messages: zero loss, <=500ms delivery

### R-002: Session Workspace Isolation (Score: 6)

**Mitigation Strategy:** Strict session_id filtering in graph store queries. Each session's nodes tagged with session_id + session_status. Integration test with 2 concurrent sessions writing to same entity_id.

**Owner:** QA Lead
**Timeline:** Before 7-5 merge
**Status:** Planned
**Verification:** 2 concurrent session test: zero cross-contamination detected

### R-003: Concurrent Task Routing Priority Preemption (Score: 6)

**Mitigation Strategy:** Redis atomic operations for task state transitions. State machine validation before transitions. Deadlock detection timeout.

**Owner:** Dev Lead
**Timeline:** Before 7-7 merge
**Status:** Planned
**Verification:** 5-task concurrent dispatch with HIGH priority preemption: correct ordering, no deadlock

### R-004: MQL5 Auto-Correction Loop (Score: 4)

**Mitigation Strategy:** Exponential backoff on auto-correction attempts. Clear escalation path after 2 failed attempts. Logging of all correction attempts.

**Owner:** QA Lead
**Timeline:** Before 7-3 merge
**Status:** Planned
**Verification:** Unit test: max 2 corrections, then escalation triggered

---

## Assumptions and Dependencies

### Assumptions

1. Redis 7.x Streams API is available in the target deployment environment
2. Docker daemon is accessible from CI/CD pipeline for MQL5 compilation tests
3. Contabo server with MT5 Docker image is reachable (or mocked in CI)
4. Graph memory store supports session_id namespace isolation
5. Department Head implementations follow the base class pattern consistently

### Dependencies

1. **Story 7-6 (Redis Streams)**: Required for Story 7-7 concurrent routing tests
2. **Story 7-5 (Session Isolation)**: Required for Story 7-7 conflict detection tests
3. **Story 7-2 (MQL5 EA Generation)**: Required for Story 7-3 compilation tests
4. **Story 7-8 (Risk/Trading/Portfolio)**: Required for Kanban SSE integration
5. **Backend API endpoints**: Required for frontend component tests

### Risks to Plan

- **Risk**: Redis Streams consumer lag causes message processing delays
  - **Impact**: Department communication delayed, task routing stalls
  - **Contingency**: Implement consumer lag monitoring; alert if >1s behind

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run)
- Run `*automate` for broader coverage once implementation exists
- Run `*test-design` for Epic 8 (Alpha Forge) after Epic 7 is complete

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: _______________ Date: _______________
- [ ] Tech Lead: _______________ Date: _______________
- [ ] QA Lead: _______________ Date: _______________

**Comments:**

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| ---------------- | ------ | ---------------- |
| **FloorManager** | Central orchestrator - all departments route through it | All department dispatch tests must pass |
| **Graph Memory Store** | Session isolation depends on session_id tagging | Session isolation tests; commit workflow tests |
| **Redis Streams** | Department mail and task routing both use streams | Message delivery; consumer groups; replay |
| **Department Heads (base.py)** | All heads inherit from base; changes affect all | Base class tests; head initialization tests |
| **TaskRouter** | 5-department concurrent dispatch depends on it | Priority preemption; dependency chains |
| **SSE Endpoint** | Kanban real-time updates depend on task state stream | SSE connection; card movement animation |
| **SkillManager** | 12 core skills depend on registration | Skill Forge; usage_count tracking |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- Epic: `_bmad-output/implementation-artifacts/sprint-status.yaml` (epic-7: done)
- Story 7-0: `_bmad-output/implementation-artifacts/7-0-department-system-audit.md`
- Story 7-0 Findings: `_bmad-output/implementation-artifacts/7-0-department-system-audit-findings.md`
- Story 7-1: `_bmad-output/implementation-artifacts/7-1-research-department-real-hypothesis-generation.md`
- Story 7-2: `_bmad-output/implementation-artifacts/7-2-development-department-real-mql5-ea-code-generation.md`
- Story 7-3: `_bmad-output/implementation-artifacts/7-3-mql5-compilation-integration.md`
- Story 7-4: `_bmad-output/implementation-artifacts/7-4-skill-catalogue-registry-authoring-skill-forge.md`
- Story 7-5: `_bmad-output/implementation-artifacts/7-5-session-workspace-isolation-concurrent-sessions.md`
- Story 7-6: `_bmad-output/implementation-artifacts/7-6-department-mail-redis-streams-migration.md`
- Story 7-7: `_bmad-output/implementation-artifacts/7-7-concurrent-task-routing-5-simultaneous-tasks.md`
- Story 7-8: `_bmad-output/implementation-artifacts/7-8-risk-trading-portfolio-department-real-implementations.md`
- Story 7-9: `_bmad-output/implementation-artifacts/7-9-department-kanban-sub-page-ui.md`
- Architecture: `docs/architecture.md`
- Tech Spec: `_bmad-output/planning-artifacts/architecture.md`

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
