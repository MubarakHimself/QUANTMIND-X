---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 1
epic_title: "Platform Foundation & Global Shell"
---

# Test Design: Epic 1 - Platform Foundation & Global Shell

**Date:** 2026-03-21
**Author:** Mubarak (TEA Agent - Master Test Architect)
**Status:** Draft
**Mode:** Epic-Level (sprint-status.yaml detected)

---

## Executive Summary

**Scope:** Full test design for Epic 1 - Platform Foundation & Global Shell

Epic 1 comprises 7 stories (1-0 through 1-6-9) covering:
- Story 1-0: Platform Codebase Exploration & Audit (completed)
- Story 1-1: Security Hardening & Legacy Import Cleanup (completed)
- Story 1-2: Svelte 5 Migration (completed)
- Story 1-3: NODE_ROLE Backend Deployment Split (completed)
- Story 1-4: TopBar & ActivityBar Frosted Terminal Aesthetic (completed)
- Story 1-5: StatusBand Redesign Frosted Terminal Ticker (completed)
- Story 1-6-9: 9-Canvas Routing Skeleton (completed)

**Risk Summary:**

- Total risks identified: 10
- High-priority risks (≥6): 1 (R-005: Kill Switch confirmation)
- Critical categories: SEC (Kill Switch, API stubs), PERF (Canvas transitions)

**Coverage Summary:**

- P0 scenarios: 6 (18 hours)
- P1 scenarios: 12 (12 hours)
- P2/P3 scenarios: 15 (6 hours)
- **Total effort**: ~36 hours (~5 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | -------------- | --------------------- |
| **Epic 2+ AI Provider & Server Connection tests** | Separate epic with its own test design | Epic 2 test design will cover |
| **Epic 3+ Live Trading backend (MT5 bridge)** | Separate epic scope | Epic 3 test design will cover |
| **Epic 7+ Agent SDK migration (LangChain replacement)** | Future work, currently stubbed | Epic 7 test design will cover |
| **E2E Playwright tests for frontend canvases** | No Playwright infrastructure exists yet | Follow-up: *automate workflow to generate |
| **Backend API contract testing (Pact)** | No microservices detected, monolith architecture | N/A for Epic 1 |
| **Performance/load testing** | Infrastructure not in scope for foundation epic | Epic 3+ will cover live trading perf |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- | -------- |
| R-005 | SEC | Kill Switch two-step confirmation may not prevent accidental activation in high-stress trading scenarios | 2 | 3 | 6 | UI audit + manual verification of modal flow; add keyboard interrupt (Escape to cancel) | QA Lead | Epic 1 retrospective |
| R-001 | TECH | NODE_ROLE invalid value defaults to "local" silently - may expose dev routers in production | 2 | 2 | 4 | MONITOR - Add explicit startup warning log when defaulting | Dev | Epic 1 |
| R-002 | TECH | Svelte 5 migration may have missed edge cases in $: reactive declarations | 2 | 2 | 4 | MONITOR - Run npm build verification | Dev | Epic 1 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- |
| R-003 | PERF | Canvas transitions may exceed 200ms NFR on low-end devices | 2 | 2 | 4 | MONITOR - Measure during testing | QA |
| R-006 | TECH | NODE_ROLE Cloudzy/Contabo router classification errors (70+ routers) | 2 | 2 | 4 | Verify each router group loads correctly per NODE_ROLE | Dev |
| R-009 | DATA | Degraded mode shows stale data without clear [stale] indicator | 2 | 2 | 4 | Verify StatusBand shows [stale] label when Contabo unreachable | QA |
| R-010 | SEC | API endpoints return stub responses after LangChain removal (e.g., IDE chat) | 2 | 2 | 4 | Verify stub endpoints return appropriate error/not-implemented | QA |
| R-007 | TECH | StatusBand P&L flash animation causes performance issues | 2 | 1 | 2 | DOCUMENT - 100ms animation is minimal | N/A |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------- |
| R-004 | SEC | LangChain import residue in unexamined files | 1 | 3 | 3 | DOCUMENT - Grep verification done, residual risk minimal |
| R-008 | TECH | Glass aesthetic backdrop-filter causes GPU repaint on low-end devices | 2 | 1 | 2 | DOCUMENT - Known Chromium issue, graceful degradation |
| R-011 | TECH | Svelte 5 migration: Component files exceeding 500 line NFR | 1 | 1 | 1 | DOCUMENT - Code review finding |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] Story 1-1: LangChain imports verified removed via `grep -rn "from langchain\|import langgraph" src/`
- [ ] Story 1-2: `npm run build` passes with zero Svelte 4 deprecation warnings
- [ ] Story 1-3: NODE_ROLE environment variable tested for all 3 values (contabo, cloudzy, local)
- [ ] Story 1-4: TopBar and ActivityBar render with Frosted Terminal aesthetic
- [ ] Story 1-5: StatusBand displays at 32px fixed bottom with session clocks
- [ ] Story 1-6-9: All 9 canvas routes registered and accessible
- [ ] Test environment provisioned with all required env vars
- [ ] `.env.example` updated with NODE_ROLE documented

## Exit Criteria

- [ ] All P0 tests passing (100% pass rate required)
- [ ] All P1 tests passing (≥95% pass rate)
- [ ] No open high-priority / high-severity bugs (Risk ≥6 must be mitigated)
- [ ] npm build passes in quantmind-ide/
- [ ] uvicorn starts without ImportError in all NODE_ROLE modes
- [ ] Kill Switch two-step confirmation modal verified functional
- [ ] Canvas transitions verified ≤200ms (or documented acceptable deviation)

## Project Team (Optional)

**Include only if roles/names are known or responsibility mapping is needed; otherwise omit.**

| Name | Role | Testing Responsibilities |
| ---- | ---- | ------------------------ |
| Mubarak | QA Lead | P0/P1 test execution, risk assessment sign-off |
| Mubarak | Dev Lead | NODE_ROLE backend verification, build validation |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| Kill Switch two-step confirmation (ARMED state → confirm modal) | E2E | R-005 | 3 | QA | Verify ShieldAlert icon pulses, modal blocks Enter key, two-step required |
| Kill Switch confirmation cancels on Escape key | E2E | R-005 | 1 | QA | Regression: prevent accidental activation |
| NODE_ROLE=cloudzy loads ONLY trading routers | API | R-006 | 2 | QA | Verify trading endpoints available, agent endpoints 404 |
| NODE_ROLE=contabo loads ONLY agent routers | API | R-006 | 2 | QA | Verify agent endpoints available, trading endpoints 404 |
| NODE_ROLE=invalid logs warning and defaults to local | API | R-001 | 1 | QA | Verify startup logs contain warning |
| npm build passes without Svelte 4 warnings | Build | R-002 | 1 | CI | Mandatory gate for all PRs |

**Total P0**: 10 tests, ~18 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| TopBar renders at 48px with QUANTMINDX wordmark | Component | - | 1 | QA | Visual verification |
| ActivityBar shows 9 canvas icons, 56px collapsed | Component | - | 1 | QA | Verify all 9 icons present |
| ActivityBar expands to 200px on hover/click | Component | - | 1 | QA | Animation smoothness |
| Canvas switch via ActivityBar click ≤200ms | E2E | R-003 | 3 | QA | Measure each of 9 canvases |
| Canvas switch via keyboard shortcuts (1-9) | E2E | R-003 | 1 | QA | Keyboard navigation |
| StatusBand displays session clocks (Tokyo/London/NY) | Component | - | 1 | QA | Verify 3 sessions shown |
| StatusBand P&L flash animation (green/red) | Component | R-007 | 2 | QA | 100ms flash visible |
| StatusBand degraded mode: Contabo unreachable shows [stale] | Component | R-009 | 1 | QA | Node health dot red |
| StatusBand click navigation to canvases | E2E | - | 4 | QA | Session→LiveTrading, bots→Portfolio, etc. |
| Frosted Terminal glass: rgba(8,13,20,0.08) applied | Component | R-008 | 1 | QA | backdrop-filter: blur(24px) |
| API endpoint stub responses return appropriate errors | API | R-010 | 3 | QA | ide_chat, workflow_orchestrator |
| langchain import verification: zero results | Unit | R-004 | 1 | CI | grep command verification |

**Total P1**: 21 tests, ~14 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| Svelte component line count ≤500 NFR verification | Unit | 10 | CI | Automated check |
| BreadcrumbNav shows canvas name → sub-page pattern | Component | 3 | QA | One per canvas type |
| All Lucide icons present (no emoji in UI) | Visual | 5 | QA | Spot check across components |
| Session clock accuracy (open/closed state) | Component | 3 | QA | One per session timezone |
| NODE_ROLE=local loads ALL routers | API | 1 | QA | Verify both Cloudzy and Contabo routers |
| .env.example completeness verification | Unit | 1 | CI | Verify all required env vars documented |
| ApiKeysPanel: no hardcoded API keys | Security | 1 | QA | Code review + grep |

**Total P2**: 24 tests, ~8 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| Canvas transition timing on low-end device | Perf | 9 | QA | Real device testing |
| backdrop-filter GPU fallback on old hardware | Visual | 3 | QA | Graceful degradation check |
| StatusBand accessibility: aria-live announcements | A11y | 1 | QA | Screen reader testing |
| Keyboard-only navigation completeness | E2E | 2 | QA | Tab + arrow keys |

**Total P3**: 15 tests, ~4 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] npm run build passes (60s)
- [ ] NODE_ROLE=local uvicorn starts without ImportError (30s)
- [ ] LangChain grep returns zero results (10s)

**Total**: 3 scenarios

### P0 Tests (<10 min)

**Purpose**: Critical path validation

- [ ] Kill Switch: ARMED state shows pulsing ShieldAlert icon (E2E)
- [ ] Kill Switch: Two-step confirm modal blocks Enter key (E2E)
- [ ] Kill Switch: Escape cancels activation (E2E)
- [ ] NODE_ROLE=cloudzy: trading endpoints 200, agent endpoints 404 (API)
- [ ] NODE_ROLE=contabo: agent endpoints 200, trading endpoints 404 (API)
- [ ] NODE_ROLE=invalid: warning log + defaults to local (API)

**Total**: 6 scenarios

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] TopBar: 48px height, wordmark visible (Component)
- [ ] ActivityBar: 9 canvas icons, 56px collapsed (Component)
- [ ] ActivityBar: expand to 200px on interaction (Component)
- [ ] Canvas switch via click for all 9 canvases (E2E)
- [ ] Canvas switch via keyboard 1-9 (E2E)
- [ ] StatusBand: 32px fixed bottom, session clocks (Component)
- [ ] StatusBand: P&L flash green/red 100ms (Component)
- [ ] StatusBand: [stale] label when Contabo unreachable (Component)
- [ ] StatusBand click navigates to correct canvas (E2E)
- [ ] Frosted Terminal glass aesthetic visible (Visual)
- [ ] API stubs return appropriate errors (API)

**Total**: 11 scenarios

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] Svelte component line count verification (CI)
- [ ] BreadcrumbNav renders correctly (Component)
- [ ] Lucide icons present, no emoji (Visual)
- [ ] Session clock open/closed state (Component)
- [ ] NODE_ROLE=local loads all routers (API)
- [ ] .env.example completeness (CI)
- [ ] ApiKeysPanel no hardcoded keys (Security)
- [ ] Canvas transition timing measurements (Perf)
- [ ] backdrop-filter GPU fallback (Visual)
- [ ] aria-live announcements (A11y)
- [ ] Keyboard-only navigation (E2E)

**Total**: 11 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| --------- | ----------------- | ---------- | ----------------- | ----------------------- |
| P0 | 10 | 1.5 | 15 | Complex E2E with API verification |
| P1 | 21 | 0.67 | 14 | Mix of component and E2E |
| P2 | 24 | 0.33 | 8 | Automated checks + component tests |
| P3 | 15 | 0.25 | 4 | Exploratory + perf testing |
| **Total** | **70** | **-** | **~41 hours** | **~5-6 days** |

### Prerequisites

**Test Data:**

- `.env.example` populated with all required env vars (NODE_ROLE, API keys placeholders)
- Test accounts with valid API keys for provider endpoints (mocked in test env)
- Session time data: deterministic clock for Tokyo/London/NY session testing

**Tooling:**

- pytest for Python backend API tests
- Vitest for Svelte component tests (or manual component testing)
- Playwright for E2E tests (infrastructure setup required)
- grep/bash for LangChain import verification

**Environment:**

- Local dev environment (NODE_ROLE=local)
- Cloudzy mock environment (NODE_ROLE=cloudzy)
- Contabo mock environment (NODE_ROLE=contabo)

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers (R-005 must be mitigated)

### Coverage Targets

- **Critical paths**: ≥80% (Kill Switch, NODE_ROLE routing)
- **Security scenarios**: 100% (R-005 Kill Switch, R-004 LangChain removal)
- **Business logic**: ≥70% (canvas routing, StatusBand display)
- **Edge cases**: ≥50% (degraded mode, animation performance)

### Non-Negotiable Requirements

- [ ] All P0 tests pass
- [ ] No high-risk (≥6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] Performance targets met (≤200ms canvas transitions) or documented deviation
- [ ] npm build passes with zero Svelte 4 deprecation warnings

---

## Mitigation Plans

### R-005: Kill Switch Two-Step Confirmation (Score: 6)

**Mitigation Strategy:** Add keyboard interrupt (Escape to cancel) and verify modal cannot be confirmed via Enter key. Ensure two-step flow (ARMED → confirm modal → explicit button click) is enforced.

**Owner:** QA Lead
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:**
- Manual E2E test: Click ShieldAlert in ARMED state, verify modal appears
- Press Enter, verify action does NOT execute
- Press Escape, verify modal closes
- Click confirm button, verify action executes

### R-001: NODE_ROLE Invalid Value Default (Score: 4)

**Mitigation Strategy:** Add explicit startup log warning when invalid NODE_ROLE defaults to "local"

**Owner:** Dev Lead
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Start server with NODE_ROLE=invalid, verify logs contain "WARNING: Invalid NODE_ROLE 'invalid', defaulting to 'local'"

### R-002: Svelte 5 Migration Edge Cases (Score: 4)

**Mitigation Strategy:** Verify npm build passes with zero warnings; manual component spot-check

**Owner:** Dev
**Timeline:** 2026-03-21
**Status:** In Progress
**Verification:** Run `npm run build 2>&1 | grep -i "deprecat\|warn"` returns empty

---

## Assumptions and Dependencies

### Assumptions

1. Playwright E2E test infrastructure will be set up as part of Epic 1 or subsequent epics - no existing Playwright tests exist
2. All 9 canvas components are functional (not just placeholders) based on implementation artifacts
3. Kill Switch backend is already implemented and wired to frontend via existing API endpoints
4. StatusBand API calls (sessions, P&L, node health) exist and return valid data

### Dependencies

1. **Playwright test infrastructure setup** - Required by 2026-03-25 for E2E tests
2. **NODE_ROLE environment configuration** - Required for all backend router tests
3. **Mock Cloudzy/Contabo environments** - Required for router isolation testing
4. **Kill Switch API endpoints** - Required for two-step confirmation E2E tests

### Risks to Plan

- **Risk**: E2E test infrastructure (Playwright) not available in time
  - **Impact**: P0 and P1 E2E tests deferred to follow-up sprint
  - **Contingency**: Convert E2E tests to manual test procedures until automation ready

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.
- Run `*playwright-generate` to scaffold E2E tests for TopBar, ActivityBar, StatusBand, and Canvas routing.

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
| ----------------- | -------------- | ------------------------------- |
| **Kill Switch Backend** | UI wired to existing API | All kill switch tests must pass before any trading epic |
| **NODE_ROLE Router System** | Conditional router registration | Any backend change must test all 3 NODE_ROLE modes |
| **Svelte 5 Migration** | All 188 frontend components | npm build must pass before any frontend epic |
| **Canvas Routing (MainContent.svelte)** | All 9 canvases depend on this | Regression: canvas switch tests must pass for all epics |
| **StatusBand API calls** | Dashboard metrics display | If sessions/P&L API changes, update StatusBand tests |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- PRD: Epic 1 definition (`_bmad-output/planning-artifacts/epics.md#Epic-1`)
- Story 1-0 Audit: `_bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md`
- Story 1-1 Implementation: `_bmad-output/implementation-artifacts/1-1-security-hardening-legacy-import-cleanup.md`
- Story 1-2 Implementation: `_bmad-output/implementation-artifacts/1-2-svelte-5-migration.md`
- Story 1-3 Implementation: `_bmad-output/implementation-artifacts/1-3-node-role-backend-deployment-split.md`
- Story 1-4 Implementation: `_bmad-output/implementation-artifacts/1-4-topbar-activitybar-frosted-terminal-aesthetic.md`
- Story 1-5 Implementation: `_bmad-output/implementation-artifacts/1-5-statusband-redesign-frosted-terminal-ticker.md`
- Story 1-6-9 Implementation: `_bmad-output/implementation-artifacts/1-6-9-canvas-routing-skeleton.md`

### Test ID Format

`{EPIC}.{STORY}-{LEVEL}-{SEQ}`

Examples:

- `1.3-API-001` (NODE_ROLE=cloudzy trading routers)
- `1.4-E2E-001` (Kill Switch two-step confirmation)
- `1.5-CMP-001` (StatusBand session clocks)

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
**Date**: 2026-03-21
