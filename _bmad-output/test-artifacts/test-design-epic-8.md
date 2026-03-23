---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 8
epic_title: 'Alpha Forge — Strategy Factory'
---

# Test Design: Epic 8 - Alpha Forge — Strategy Factory

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft

---

## Executive Summary

**Scope:** Full test design for Epic 8 - Alpha Forge Strategy Factory

The Alpha Forge is a comprehensive strategy factory pipeline that transforms YouTube videos and market events into deployed trading EAs. This test design covers 10 stories (8-0 through 8-9) implementing the full pipeline: VIDEO_INGEST → RESEARCH → TRD_GENERATION → DEVELOPMENT → COMPILE → BACKTEST → VALIDATION → EA_LIFECYCLE → APPROVAL.

**Risk Summary:**

- Total risks identified: 15
- High-priority risks (≥6): 6 (R-001 through R-006)
- Critical categories: OPS (4), BUS (3), TECH (4), SEC (1)

**Coverage Summary:**

- P0 scenarios: 16 tests (~30-45 hours)
- P1 scenarios: 16 tests (~25-40 hours)
- P2/P3 scenarios: 21 tests (~30-55 hours)
- **Total effort**: 53 tests (~85-140 hours (~17-28 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
|------|----------|------------|
| **MT5 Terminal hardware integration** | Requires live MT5 terminal on Cloudzy; not available in test environment | Simulated mode in tests; real integration tested in staging |
| **Cloudzy SSH connectivity** | External cloud dependency; SSH authentication issues are infrastructure concern | Mock SSH in unit tests; integration tests use simulated fallback |
| **Prefect server deployment** | Prefect Orion local used; production Prefect server on Contabo not tested | workflows.db SQLite tested; production deployment separate concern |
| **Monaco Editor full functionality** | Stub implementation used; real Monaco integration is frontend concern | Component renders correctly; full Monaco features verified manually |
| **Real YouTube video transcription** | External API dependency; video_ingest tested with mock responses | Mock HTTP responses in tests; real API tested in integration |
| **scipy.stats ttest_ind accuracy** | Third-party library; assumes scipy implementation is correct | Test our usage/calling of scipy, not scipy itself |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
|---------|----------|-------------|-------------|--------|-------|------------|-------|----------|
| R-001 | OPS | **EA Deployment Pipeline complete failure** - Cloudzy SSH transfer fails, MT5 registration errors, ZMQ stream fails to connect. Multiple external dependencies (Cloudzy, MT5 Terminal, ZMQ port 8888) | 3 | 3 | **9** | Comprehensive try/except blocks; simulated fallback modes; 60s health check timeout | QA | Sprint 1 |
| R-002 | OPS | **Trading during restricted deployment window** - EA deployed outside Friday 22:00 - Sunday 22:00 UTC with active positions | 2 | 3 | **6** | `check_deployment_window()` uses `datetime.now(timezone.utc)` after 8-1 UTC bug fix; deployment halts outside window | QA | Sprint 1 |
| R-003 | TECH | **Prefect workflows.db corruption or data loss** - SQLite database corruption loses all pipeline state and stage results | 2 | 3 | **6** | Stage results persisted individually; workflows.db backup strategy; corruption recovery tested | QA | Sprint 2 |
| R-004 | BUS | **TRD validation false rejection** - Valid TRDs rejected due to missing optional parameter detection, blocking pipeline | 2 | 3 | **6** | `requires_clarification()` path returns field list for Research to fill; FloorManager notification | QA | Sprint 1 |
| R-005 | OPS | **Approval gate timeout misconfiguration** - 15-min soft/7-day hard timeout miscalculated due to timezone issues | 2 | 3 | **6** | Timezone-aware datetime comparison; `/check-timeout` endpoint validates timeout calculations | QA | Sprint 1 |
| R-006 | SEC | **A/B statistical significance p-value miscalculation** - Wrong p-value leads to false positive (promoting losing variant) or false negative (blocking winning variant) | 2 | 3 | **6** | Uses `scipy.stats.ttest_ind`; minimum 50 trades required; amber crown indicator for transparency | QA | Sprint 1 |

### Medium-Priority Risks (Score 3-5)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|-------------|--------|-------|------------|-------|
| R-007 | OPS | **Fast-track deployment timeout** - 15-minute deployment SLA breached; event opportunity lost | 3 | 2 | **5** | 15-min timeout configured; graceful failure notification to FloorManager | QA | Sprint 2 |
| R-008 | BUS | **Template matching scoring inaccuracy** - Wrong template ranked first, suboptimal strategy deployed | 3 | 2 | **5** | Weighted 4-factor scoring (event 40%, symbol 30%, risk 20%, deployment 10%); human approval before fast-track | QA | Sprint 2 |
| R-009 | TECH | **Variant browser uses demo data in production** - VariantBrowser returns mock data instead of querying actual version storage | 2 | 2 | **4** | Story notes acknowledge demo data; production would wire to actual storage APIs | DEV | Sprint 3 |
| R-010 | PERF | **Pipeline status board polling overhead** - 5s polling on Contabo increases server load with many active runs | 2 | 2 | **4** | 5s interval is reasonable; no WebSocket upgrade path specified | DEV | Sprint 3 |
| R-011 | TECH | **Provenance chain incomplete** - EA provenance metadata (source URL, research score) not persisted after database migration | 2 | 2 | **4** | Story 8.9 Task 4.1 pending database migration; tracking in backlog | DEV | Sprint 4 |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
|---------|----------|-------------|-------------|--------|-------|--------|
| R-012 | TECH | **Monaco editor memory leak** - `subscribe()` in VariantBrowser not cleaned up via `onDestroy` | 1 | 2 | **2** | Monitor |
| R-013 | TECH | **EA deployment audit record not persisted** - `get_db_session` import bug caused silent failure | 1 | 2 | **2** | Document |
| R-014 | TECH | **logger used before assignment** - `approval_gate.py` logger crash on import when optional dependencies fail | 1 | 2 | **2** | Document |
| R-015 | BUS | **Islamic compliance parameters missing** - TRD generation may not include force_close_hour/overnight_hold | 1 | 3 | **3** | Monitor |

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
- [ ] Test environment provisioned and accessible (Contabo Cloudzy instance)
- [ ] Prefect workflows.db initialized and accessible
- [ ] MT5 terminal available for deployment testing (or simulated mode)
- [ ] Strategy template library seeded with 3 default templates
- [ ] All 10 Epic 8 stories implemented and code reviewed
- [ ] Department agents (ResearchHead, DevelopmentHead) wired and operational

## Exit Criteria

- [ ] All P0 tests passing (100%)
- [ ] All P1 tests passing (or failures triaged, >=95%)
- [ ] No open high-priority / high-severity bugs
- [ ] EA deployment pipeline tested end-to-end with simulated modes
- [ ] Statistical significance engine verified with known trade distributions
- [ ] Cross-strategy loss propagation tested with correlation matrix
- [ ] Test coverage agreed as sufficient (>=80% for critical paths)

---

## Project Team

| Name | Role | Testing Responsibilities |
|------|------|-------------------------|
| Mubarak | QA Lead | Test design, P0/P1 execution, risk assessment |
| Dev Team | Backend | Unit tests for flows, API endpoints, TRD system |
| Dev Team | Frontend | Component tests for UI, VariantBrowser, Monaco integration |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|------------|-----------|------------|-------|-------|
| AlphaForgeFlow 6-stage sequence with DB persistence | INTEGRATION | R-001, R-003 | 2 | QA | workflows.db persistence |
| Stage failure graceful handling + Copilot notification | INTEGRATION | R-001 | 2 | QA | Error path coverage |
| Deployment window UTC enforcement | UNIT | R-002 | 3 | QA | Timezone edge cases |
| TRD generation from hypothesis >= 0.75 | UNIT | R-004 | 2 | QA | Happy path |
| TRD validation rejects incomplete TRDs | UNIT | R-004 | 2 | QA | Negative path |
| Islamic compliance params always present | UNIT | R-004 | 1 | QA | Edge case |
| Approval gate PENDING_REVIEW state creation | INTEGRATION | R-005 | 1 | QA | State machine |
| Immutable approval audit record | INTEGRATION | R-005 | 2 | QA | Audit verification |
| Revision request re-execution flow | INTEGRATION | R-005 | 1 | QA | Workflow |
| Approval gate timeout (15-min, 7-day) | UNIT | R-005 | 2 | QA | Timeout edge cases |
| EA deployment 5-stage pipeline | INTEGRATION | R-001 | 2 | QA | End-to-end |
| A/B statistical significance p < 0.05 | UNIT | R-006 | 2 | QA | Algorithm |
| Cross-strategy loss propagation | INTEGRATION | R-006 | 2 | QA | Risk logic |
| Provenance chain traceability | API | R-006 | 1 | QA | Data integrity |

**Total P0**: 25 tests, 30-45 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|------------|-----------|------------|-------|-------|
| GET /api/alpha-forge/templates endpoint | API | - | 2 | QA | REST |
| Template matching with event scoring | API | R-008 | 2 | QA | Matching |
| Fast-track lot sizing (0.5x) and auto-expiry (24h) | INTEGRATION | R-008 | 2 | QA | Config |
| Strategy version creation with semantic versioning | UNIT | - | 2 | QA | Version |
| Rollback flow with SIT re-validation | INTEGRATION | R-003 | 2 | QA | Recovery |
| GET /api/strategies/{id}/versions | API | - | 1 | QA | REST |
| Pipeline status board UI rendering | COMPONENT | R-010 | 2 | QA | UI |
| Pipeline status polling mechanism | INTEGRATION | R-010 | 1 | QA | Polling |
| PENDING_REVIEW amber badge + TopBar | COMPONENT | R-005 | 1 | QA | UI |
| Variant browser grid display | COMPONENT | R-009 | 2 | QA | UI |
| Monaco MQL5 syntax highlighting | COMPONENT | R-012 | 1 | QA | UI |
| Version timeline display | COMPONENT | R-009 | 1 | QA | UI |
| Edit mode Save/Run/Diff actions | COMPONENT | R-012 | 1 | QA | UI |
| A/B comparison view rendering | COMPONENT | R-006 | 2 | QA | UI |
| Loss propagation audit logging | INTEGRATION | R-006 | 1 | QA | Audit |

**Total P1**: 23 tests, 25-40 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
|-------------|------------|-----------|------------|-------|-------|
| Template CRUD operations | API | - | 3 | QA | REST |
| Template search/filter by events | API | - | 2 | QA | REST |
| Fast-track 15-min timeout handling | INTEGRATION | R-007 | 1 | QA | Timeout |
| Version comparison endpoint | API | - | 1 | QA | REST |
| MT5 parameter injection validation | UNIT | - | 2 | QA | Param |
| SSH rollback on transfer failure | INTEGRATION | R-001 | 1 | QA | Error |
| First tick received verification | INTEGRATION | - | 1 | QA | Health |
| Stage animation (cyan pulse/check) | COMPONENT | - | 1 | QA | UI |
| Promotion status tracker display | COMPONENT | - | 1 | QA | UI |
| p-value calculation with scipy | UNIT | R-006 | 2 | QA | Algorithm |
| Copilot notification for significance | INTEGRATION | R-006 | 1 | QA | NL |
| Provenance timeline visualization | COMPONENT | R-011 | 1 | QA | UI |

**Total P2**: 17 tests, 20-35 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
|-------------|------------|------------|-------|-------|
| Template matching weight tuning | UNIT | 1 | DEV | Algorithm |
| Version archive retrieval | API | 1 | DEV | REST |
| Morning digest re-surface | INTEGRATION | 1 | DEV | Notification |
| ZMQ reconnect within 10s | INTEGRATION | 1 | DEV | Performance |
| Empty pipeline state UI | COMPONENT | 1 | DEV | Edge case |
| Monaco Diff view | COMPONENT | 1 | DEV | UI |
| Provenance NL query | INTEGRATION | 1 | DEV | NL interface |

**Total P3**: 7 tests, 10-20 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] AlphaForgeFlow database initialization (30s)
- [ ] VideoIngestFlow stage sequence (1 min)
- [ ] TRD generation happy path (30s)
- [ ] Approval gate state creation (45s)
- [ ] Deployment window UTC check (30s)

**Total**: 5 scenarios

### P0 Tests (<30 min)

**Purpose**: Critical path validation

- [ ] AlphaForgeFlow 6-stage sequence (2 min)
- [ ] Stage failure notification (1 min)
- [ ] Deployment window enforcement (2 min)
- [ ] TRD generation + validation (2 min)
- [ ] Islamic compliance params (1 min)
- [ ] Approval gate lifecycle (3 min)
- [ ] Approval timeout edge cases (3 min)
- [ ] EA deployment pipeline (5 min)
- [ ] A/B statistical significance (3 min)
- [ ] Loss propagation (2 min)
- [ ] Provenance chain (2 min)

**Total**: 11 scenarios (~26 min)

### P1 Tests (<45 min)

**Purpose**: Important feature coverage

- [ ] Template API endpoints (3 min)
- [ ] Template matching logic (2 min)
- [ ] Fast-track configuration (2 min)
- [ ] Version control APIs (3 min)
- [ ] Rollback integration (4 min)
- [ ] Pipeline status board (3 min)
- [ ] Variant browser UI (3 min)
- [ ] Monaco editor (2 min)
- [ ] A/B comparison UI (3 min)

**Total**: 9 scenarios (~25 min)

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] Template CRUD (3 min)
- [ ] Fast-track timeout (2 min)
- [ ] Version comparison (1 min)
- [ ] MT5 parameter validation (2 min)
- [ ] UI component tests (5 min)
- [ ] Algorithm tests (3 min)
- [ ] Integration tests (10 min)

**Total**: 7 scenarios (~26 min)

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
|----------|-------|------------|-------------|-------|
| P0 | 25 | 2.0 | 30-45 | Complex flows, error paths |
| P1 | 23 | 1.0 | 25-40 | Standard coverage |
| P2 | 17 | 0.5 | 20-35 | Simple scenarios |
| P3 | 7 | 0.25 | 10-20 | Exploratory |
| **Total** | **72** | **-** | **85-140** | **~17-28 days** |

### Prerequisites

**Test Data:**

- WorkflowFactory factory (Prefect flow mocks with auto-cleanup)
- TRDFactory factory (hypothesis data, auto-cleanup)
- TemplateFactory factory (seed data, auto-cleanup)
- StrategyVersionFactory factory (artifacts, auto-cleanup)

**Tooling:**

- pytest for Python backend tests
- pytest-asyncio for async tests (no @pytest.mark.asyncio decorators found)
- unittest.mock for API mocking
- tempfile for workflows.db isolation

**Environment:**

- Contabo Cloudzy instance (for real SSH/MT5 integration tests)
- SQLite workflows.db (local Prefect Orion)
- Svelte component tests via Vitest

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions - blocks release)
- **P1 pass rate**: >=95% (waivers required for failures)
- **P2/P3 pass rate**: >=90% (informational)
- **High-risk mitigations (R-001-R-006)**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths (P0)**: 100%
- **API endpoints**: >=80%
- **Integration tests**: >=70%
- **Business logic**: >=70%
- **Edge cases**: >=50%

### Non-Negotiable Requirements

- [ ] All P0 tests pass (25/25)
- [ ] No high-risk (>=6) items unmitigated
- [ ] EA deployment pipeline tested with simulated modes
- [ ] Statistical significance engine verified against known data
- [ ] Cross-strategy loss propagation tested with correlation matrix

---

## Mitigation Plans

### R-001: EA Deployment Pipeline Failure (Score: 9)

**Mitigation Strategy:** All deployment stages have comprehensive try/except blocks with simulated fallback modes. Tests verify graceful degradation when Cloudzy/MT5/ZMQ are unavailable.

**Owner:** QA Lead
**Timeline:** Sprint 1
**Status:** In Progress
**Verification:** Run deployment tests in simulated mode; verify audit records created even when MT5 unavailable

### R-002: Trading During Restricted Deployment Window (Score: 6)

**Mitigation Strategy:** `check_deployment_window()` uses `datetime.now(timezone.utc)` (fixed from naive datetime bug). Deployment window enforced at EA_LIFECYCLE stage before any trading begins.

**Owner:** QA Lead
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Test with mock UTC times at window boundaries (Friday 21:59, 22:00, Sunday 21:59, 22:00)

### R-003: Prefect workflows.db Corruption (Score: 6)

**Mitigation Strategy:** Stage results persisted individually with idempotent writes. Database backup strategy documented. Corruption recovery tested.

**Owner:** Dev Team
**Timeline:** Sprint 2
**Status:** Planned
**Verification:** Simulate DB corruption; verify workflow resumes from last successful stage

### R-004: TRD Validation False Rejection (Score: 6)

**Mitigation Strategy:** `requires_clarification()` path returns complete missing field list. FloorManager notified for Research to fill gaps.

**Owner:** QA Lead
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Test with intentionally incomplete TRDs; verify rejection list is actionable

### R-005: Approval Gate Timeout Misconfiguration (Score: 6)

**Mitigation Strategy:** Timezone-aware datetime. `/check-timeout` endpoint validates calculations. Timezone bug fixed in review.

**Owner:** QA Lead
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Test timeout at exactly 15 minutes, 7 days; test timezone transitions

### R-006: A/B Statistical Significance Miscalculation (Score: 6)

**Mitigation Strategy:** Uses `scipy.stats.ttest_ind`. Minimum 50 trades required. Amber crown provides transparency. Copilot notification for significant results.

**Owner:** QA Lead
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Test p-value calculation with known trade distributions (equal means = p>0.05, different means = p<0.05)

---

## Assumptions and Dependencies

### Assumptions

1. Prefect Orion local sufficient for development testing; production Prefect server deployment is separate concern
2. MT5 terminal simulation acceptable for CI; real MT5 testing requires Contabo Cloudzy instance
3. Cloudzy SSH accessible from test environment; authentication via SSH key configured
4. Statistical significance requires >=50 trades per variant to be meaningful
5. Cross-strategy loss propagation triggered by daily loss cap breach event (Story 3.3)

### Dependencies

1. **Contabo Cloudzy instance** - Required for EA deployment (SSH, MT5, ZMQ) - Week 1
2. **Prefect workflows.db** - SQLite database must be initialized - Week 1
3. **3 default templates** - Strategy template library seeded - Week 1
4. **Department agents** - ResearchHead, DevelopmentHead operational - Week 1
5. **scipy installed** - For `scipy.stats.ttest_ind` - Week 1

### Risks to Plan

- **Risk**: Cloudzy instance not available for integration testing
  - **Impact**: EA deployment tests must run in simulated mode only
  - **Contingency**: Verify simulated mode thoroughly; manual testing on Cloudzy before production

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests from acceptance criteria (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.
- Run `*test-design` for Epic 9 (Portfolio & Multi-Broker Management) after Epic 8 complete.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: _______________ Date: ___________
- [ ] Tech Lead: _______________ Date: ___________
- [ ] QA Lead: _______________ Date: ___________

**Comments:**

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
|-------------------|--------|------------------|
| **Prefect Orion** | AlphaForgeFlow and VideoIngestFlow depend on Prefect for workflow orchestration | All flow tests must pass |
| **workflows.db** | All stage results persisted; loss = pipeline state lost | Database persistence tests |
| **TRD Generator** | TRD is contract between Research and Development | TRD validation tests |
| **Approval Gate API** | Deployment halted until approved | Approval lifecycle tests |
| **EA Registry** | Tracks deployed EA lifecycle state | Deployment tests |
| **MT5 Client** | Terminal registration and health checks | Deployment pipeline tests |
| **ZMQ Tools** | Tick streaming for live positions | Streaming tests |
| **Correlation Sensor** | Loss propagation depends on correlation data | Risk tests |
| **Kelly Engine** | Position sizing adjustments | Risk tests |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology (1-9 scale)
- `test-levels-framework.md` - Test level selection (Unit/Integration/E2E)
- `test-priorities-matrix.md` - P0-P3 prioritization criteria

### Related Documents

- Epic 8 Stories: `_bmad-output/implementation-artifacts/8-*`
- Sprint Status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Architecture: `docs/architecture.md` (Alpha Forge section)
- PRD: `docs/prd.md` (FR23-FR31, FR74-FR79)

### Implementation Artifacts

| Story | File | Status |
|-------|------|--------|
| 8-0 Pipeline Audit | `8-0-alpha-forge-pipeline-audit.md` | done |
| 8-1 Orchestrator Wiring | `8-1-alpha-forge-orchestrator-wiring-departments-through-pipeline-stages.md` | review |
| 8-2 TRD Generation | `8-2-trd-generation-stage.md` | review |
| 8-3 Fast-Track Templates | `8-3-fast-track-event-workflow-template-library-matching.md` | review |
| 8-4 Version Control | `8-4-strategy-version-control-rollback-api.md` | review |
| 8-5 Approval Gates | `8-5-human-approval-gates-backend.md` | review |
| 8-6 EA Deployment | `8-6-ea-deployment-pipeline-mt5-registration.md` | review |
| 8-7 Pipeline Status Board | `8-7-alpha-forge-canvas-pipeline-status-board.md` | review |
| 8-8 Variant Browser | `8-8-development-canvas-ea-variant-browser-monaco-editor.md` | review |
| 8-9 A/B Race Board | `8-9-ab-race-board-cross-strategy-loss-propagation-provenance-chain.md` | done |

### Existing Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| `tests/flows/test_alpha_forge_flows.py` | 13 | Database, flows, tasks |
| `tests/api/test_alpha_forge_templates.py` | 13 | Template endpoints |
| `tests/api/test_approval_gate_alpha_forge.py` | 7 | Approval gate |
| `tests/flows/test_ea_deployment_flow.py` | 16 | Deployment pipeline |
| **Total** | **49** | Backend focus |

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
**Date**: 2026-03-21
