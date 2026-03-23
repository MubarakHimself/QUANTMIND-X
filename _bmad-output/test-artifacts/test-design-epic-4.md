# Test Design: Epic 4 - Risk Management & Compliance

**Date:** 2026-03-21
**Author:** Master Test Architect
**Status:** Draft

---

## Executive Summary

**Scope:** Full test design for Epic 4 - Risk Management & Compliance

**Epic Context:**
- **Epic Number:** 4
- **Epic Title:** Risk Management & Compliance
- **Stories:** 7 implementation stories (4-0 through 4-6), all marked as done
- **Tech Stack:** Python/FastAPI backend + Svelte 5 frontend
- **Test Stack:** pytest (backend), Vitest + Playwright (frontend)

**Risk Summary:**

- Total risks identified: 8
- High-priority risks (≥6): 4
- Critical categories: SEC (Security), BUS (Business), PERF (Performance)

**Coverage Summary:**

- P0 scenarios: 15 tests (~30 hours)
- P1 scenarios: 20 tests (~20 hours)
- P2/P3 scenarios: 35 tests (~15 hours)
- **Total effort**: 65 tests (~65 hours (~9 working days)**

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | --------- | ---------- |
| **MT5 bridge integration** | Story 3-0 covered backend MT5 integration | Tested upstream in Epic 3 |
| **PhysicsAwareKellyEngine internals** | Story 4-0 audit classified as production-ready, read-only wiring | Unit tests for API output only |
| **Ising/Lyapunov/HMM sensor algorithms** | Story 4-0 classified as production-ready, shadow mode for HMM | API contract tests only |
| **Database migration for `prop_firm_accounts`** | Known issue from Story 4-2 - requires separate migration task | Track as tech debt |
| **Legacy circuit breaker components** | Story 4-0 audit confirmed production-ready | Regression tests only |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- | -------- |
| R-001 | SEC | **Kelly Fraction Validation Bypass** - kelly_fraction > 1.0 not rejected properly, allowing oversized positions | 3 | 3 | **9** | Pydantic validation at API boundary, 422 response enforced | QA/Dev | Sprint 4+1 |
| R-002 | SEC | **Islamic Compliance Countdown Failure** - force-close at 21:45 UTC may not trigger, causing account violation | 2 | 3 | **6** | Timer validation tests, UI countdown display verification | QA | Sprint 4+1 |
| R-003 | PERF | **5-Second Polling Cascade Failure** - NFR-R1 requires independent failure isolation per tile | 2 | 3 | **6** | Each tile wrapped in try/catch, individual error states | QA/Dev | Sprint 4+1 |
| R-004 | BUS | **CalendarGovernor Journey 45 Failure** - NFP Friday 4-phase transition (Thu 18:00, Fri 13:15, Fri 14:00, Fri 15:00) | 2 | 3 | **6** | Journey 45 scenario tests with time-based assertions | QA | Sprint 4+1 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- |
| R-005 | TECH | **BotCircuitBreaker State Desync** - circuit breaker state may diverge between UI and backend | 2 | 2 | **4** | Polling interval tests, state consistency assertions | QA |
| R-006 | DATA | **Prop Firm Registry DB Consistency** - CRUD operations may have race conditions | 2 | 2 | **4** | Database transaction tests, concurrent update scenarios | QA/Dev |
| R-007 | OPS | **Demo Data Masking Real Failures** - fallback demo data may hide actual API failures | 2 | 2 | **4** | Test environment detection, disable demo data in test mode | Dev |
| R-008 | TECH | **HMM Shadow Mode Alert Misleading** - HMM always shows "warning" alert in shadow mode | 1 | 3 | **3** | Clear "shadow mode" badge UI test, alert state verification | QA |

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
- [ ] Test environment provisioned and accessible (`/api/risk/*` endpoints reachable)
- [ ] Test database seeded with prop firm accounts and risk parameters
- [ ] Story 4-6 Risk Canvas deployed to test environment
- [ ] Islamic compliance test accounts available for countdown testing
- [ ] Calendar API accessible or mock data available for Journey 45 scenario

## Exit Criteria

- [ ] All P0 tests passing (100% pass rate required)
- [ ] All P1 tests passing (or failures triaged with waivers)
- [ ] No open high-priority / high-severity bugs in SEC/PERF categories
- [ ] Test coverage agreed as sufficient by QA Lead
- [ ] R-001 through R-004 mitigations verified and documented
- [ ] NFR-R1 cascade failure isolation validated per tile

---

## Project Team

| Name | Role | Testing Responsibilities |
| ---- | ---- | ------------------------ |
| QA Lead | Test Architect | Test design, coverage plan, quality gates |
| Dev Lead | Backend Owner | API implementation, risk parameter validation |
| Dev Lead | Frontend Owner | Svelte components, polling, alert states |
| PM | Business Owner | Journey 45 scenario, Islamic compliance requirements |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| Kelly fraction validation (kelly_fraction > 1.0 → 422) | API | R-001 | 4 | QA | Boundary value testing, edge cases |
| Islamic compliance countdown (60min warning, 30min critical, 21:45 UTC force-close) | E2E | R-002 | 4 | QA | Time-based, requires clock manipulation |
| CalendarGovernor Journey 45 NFP scenario (4 phase transitions) | Unit | R-004 | 5 | Dev | Time-based state machine tests |
| NFR-R1 cascade failure isolation (each tile independent) | Component | R-003 | 2 | QA | Verify one tile failure doesn't cascade |

**Total P0**: 15 tests, ~30 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| Risk params GET /api/risk/params/{account_tag} | API | - | 3 | QA | Response schema validation |
| Risk params PUT /api/risk/params/{account_tag} partial updates | API | - | 4 | QA | Field isolation, audit logging |
| Prop firm CRUD operations (GET/POST/PUT/DELETE) | API | R-006 | 5 | QA | Database consistency tests |
| Regime classification API GET /api/risk/regime | API | - | 3 | QA | Response fields, regime types |
| Physics sensor API GET /api/risk/physics | API | - | 3 | QA | Ising, Lyapunov, HMM outputs |
| Compliance tile BotCircuitBreaker state rendering | Component | R-005 | 2 | Dev | State display, alert border |

**Total P1**: 20 tests, ~20 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| Backtest results API GET /api/backtests | API | 3 | QA | Response schema, 6 modes |
| Backtest detail GET /api/backtests/{id} | API | 3 | QA | Equity curve, trade distribution |
| Running backtests GET /api/backtests/running | API | 2 | QA | Progress percentage, partial metrics |
| Calendar events API endpoints | API | 3 | QA | NewsItem model, blackout windows |
| Alert state rendering (#ff3b3b border, alert-triangle icon) | Component | 4 | Dev | Visual verification |
| Visualization variants (sparkline, bar, histogram) | Component | 3 | Dev | Data type rendering |
| Prop firm config panel UI CRUD | E2E | 4 | QA | Form validation, save/cancel |

**Total P2**: 22 tests, ~11 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| Glass aesthetic UI validation (0.35 opacity, blur) | Visual | 3 | QA | Exploratory testing |
| Lucide icons render correctly (not emoji) | Visual | 2 | QA | Icon verification |
| Svelte 5 runes ($state, $derived) reactivity | Unit | 4 | Dev | Reactive state tests |
| 5-second polling interval timing accuracy | Integration | 2 | QA | Polling performance |
| HMM shadow mode badge visibility | Component | 2 | Dev | Badge display |

**Total P3**: 13 tests, ~3.25 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Risk params API responds (GET /api/risk/params/default) (30s)
- [ ] Regime API responds (GET /api/risk/regime) (30s)
- [ ] Physics API responds (GET /api/risk/physics) (30s)
- [ ] Risk Canvas loads without JS errors (45s)

**Total**: 4 scenarios (~2 min)

### P0 Tests (<15 min)

**Purpose**: Critical path validation - 100% pass required

- [ ] Kelly fraction validation: 1.0 accepted (42x)
- [ ] Kelly fraction validation: 1.01 rejected with 422
- [ ] Kelly fraction validation: 0.0 rejected with 422
- [ ] Islamic countdown: 61min before shows no warning
- [ ] Islamic countdown: 59min before shows warning
- [ ] Islamic countdown: 31min before shows critical
- [ ] CalendarGovernor: Thu 18:00 phase transition (0.5x)
- [ ] CalendarGovernor: Fri 13:15 phase transition (pause)
- [ ] CalendarGovernor: Fri 14:00 regime-check reactivation
- [ ] CalendarGovernor: Fri 15:00 normal operation
- [ ] Tile failure isolation: one tile error doesn't affect others

**Total**: 11 scenarios (~12 min)

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] Risk params GET response schema validation
- [ ] Risk params PUT partial update (single field)
- [ ] Risk params PUT partial update (multiple fields)
- [ ] Risk params audit logging verification
- [ ] Prop firm CRUD: Create new firm
- [ ] Prop firm CRUD: Update existing firm
- [ ] Prop firm CRUD: Delete firm
- [ ] Regime API: all regime types returned
- [ ] Regime API: strategy counts (active/paused)
- [ ] Physics API: Ising, Lyapunov, HMM outputs
- [ ] Physics API: alert states per sensor
- [ ] Compliance tile: BotCircuitBreaker state display

**Total**: 12 scenarios (~25 min)

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] Backtest list API (all 6 modes)
- [ ] Backtest detail with equity curve
- [ ] Running backtests with progress
- [ ] Calendar events list
- [ ] Calendar blackout windows
- [ ] Alert state: normal rendering
- [ ] Alert state: warning rendering (#ff3b3b)
- [ ] Alert state: critical rendering
- [ ] Visualization: sparkline for time-series
- [ ] Visualization: bar chart for scalar
- [ ] Visualization: histogram for distribution
- [ ] Prop firm config form validation
- [ ] Glass aesthetic spot check
- [ ] Lucide icons verification
- [ ] $state reactivity test
- [ ] $derived store computation
- [ ] Polling interval accuracy
- [ ] HMM shadow mode badge

**Total**: 18 scenarios (~45 min)

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| -------- | ----- | ---------- | ----------- | ----- |
| P0 | 15 | 2.0 | 30 | Complex time-based scenarios, Islamic compliance |
| P1 | 20 | 1.0 | 20 | Standard API and component tests |
| P2 | 22 | 0.5 | 11 | Simple validation scenarios |
| P3 | 13 | 0.25 | 3.25 | Exploratory and visual tests |
| **Total** | **70** | **-** | **~64 hours** | **~9 working days** |

### Prerequisites

**Test Data:**
- `risk_params_factory` - faker-based, auto-cleanup for account_tag variations
- `prop_firm_factory` - CRUD test data with drawdown_limit_pct, daily_loss_limit_pct
- `calendar_events_factory` - NewsItem with impact levels (Tier 1, Tier 2)
- `backtest_results_factory` - 6-mode results (VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C)

**Tooling:**
- `pytest` for Python backend API tests
- `Vitest` for Svelte component tests
- `Playwright` for E2E tests (Islamic countdown, Journey 45 scenarios)
- `datetime-mocking` utility for time-based tests (Journey 45, Islamic countdown)

**Environment:**
- Backend: FastAPI dev server on `localhost:8000`
- Frontend: Vite dev server on `localhost:5173`
- Test database: SQLite with in-memory for unit tests, PostgreSQL for integration tests

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions - blocking)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Kelly fraction validation**: 100% (SEC category)
- **Islamic compliance countdown**: 100% (SEC category)
- **CalendarGovernor Journey 45**: 100% (BUS category)
- **NFR-R1 cascade failure**: 100% (PERF category)
- **API endpoints**: ≥80%
- **Svelte components**: ≥70%

### Non-Negotiable Requirements

- [ ] All P0 tests pass (R-001, R-002, R-003, R-004)
- [ ] No high-risk (≥6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] Performance targets met (NFR-R1 cascade failure isolation)
- [ ] Time-based tests (Journey 45, Islamic countdown) validated with mocked clocks

---

## Mitigation Plans

### R-001: Kelly Fraction Validation Bypass (Score: 9)

**Mitigation Strategy:** Implement defense-in-depth validation:
1. Pydantic model enforces `kelly_fraction > 0 and kelly_fraction <= 1.0` at API boundary
2. Backend service re-validates before risk evaluation cycle
3. UI prevents input beyond 1.0 with client-side validation

**Owner:** Dev Lead (Backend)
**Timeline:** Sprint 4+1
**Status:** Planned
**Verification:** API test `test_kelly_fraction_above_one_rejected` passes with 422 response

### R-002: Islamic Compliance Countdown Failure (Score: 6)

**Mitigation Strategy:**
1. Time-based tests use mocked clocks to simulate countdown scenarios
2. UI displays countdown timer prominently within 60-minute window
3. Force-close trigger validated in isolation

**Owner:** QA Lead
**Timeline:** Sprint 4+1
**Status:** Planned
**Verification:** E2E tests verify countdown display at 59min, 30min, and force-close at 21:45 UTC

### R-003: 5-Second Polling Cascade Failure (Score: 6)

**Mitigation Strategy:**
1. Each tile component wraps fetch in individual try/catch
2. Store uses independent subscriptions per tile
3. Error state managed locally per tile

**Owner:** Dev Lead (Frontend)
**Timeline:** Sprint 4+1
**Status:** Planned
**Verification:** Component test `test_tile_isolation_one_failure_doesnt_cascade`

### R-004: CalendarGovernor Journey 45 Failure (Score: 6)

**Mitigation Strategy:**
1. Journey 45 implemented as state machine with 4 discrete phases
2. Phase transitions tested with explicit time assertions
3. Audit logging at each phase transition

**Owner:** Dev Lead (Backend)
**Timeline:** Sprint 4+1
**Status:** Planned
**Verification:** Unit tests verify all 4 phase transitions with mocked datetime

---

## Assumptions and Dependencies

### Assumptions

1. Islamic compliance 21:45 UTC force-close time is configurable in production
2. CalendarGovernor EnhancedGovernor bug (kelly_adjustments type) is fixed before P0 test execution
3. Database migrations for `prop_firm_accounts` and `risk_params` tables are applied
4. Demo data fallback can be disabled via environment variable for test isolation
5. HMM sensors remain in shadow mode (outputs logged but not controlling router)

### Dependencies

1. **Database migrations** - Required before API tests can run
2. **Calendar API access** - For real news events or mock data
3. **Playwright CLI** - Available for E2E time-based tests
4. **Clock mocking utility** - For Islamic countdown and Journey 45 tests

### Risks to Plan

- **Risk**: Story 4-2 database schema issue (prop_firm_accounts missing account_type column)
  - **Impact**: Prop firm API tests will fail against actual database
  - **Contingency**: Use in-memory mock data for prop firm tests until migration is applied

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| ----------------- | ------ | ---------------- |
| **EnhancedGovernor** | CalendarGovernor extends it; bug fix from Story 4-1 | `test_calendar_governor_journey_45` must pass |
| **StrategyRouter** | Regime state endpoint wires to it | `test_regime_api_*` must pass |
| **PhysicsAwareKellyEngine** | Kelly tile wires to it | `test_kelly_tile_*` must pass |
| **BotCircuitBreaker** | Compliance tile displays state | `test_compliance_tile_*` must pass |
| **Backtest Engine** | Story 4-4 API wires to it (read-only) | `test_backtest_api_*` must pass |
| **Epic 3 Kill Switch** | Shared infrastructure | `test_kill_switch_tiers` regression |

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.
- Run `*test-design` for Epic 5 (Unified Memory & Copilot Core) after this epic is approved.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: _________________ Date: __________
- [ ] Tech Lead: _________________ Date: __________
- [ ] QA Lead: _________________ Date: __________

**Comments:**

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection (Unit/API/Component/E2E)
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- Epic 4 Stories: `_bmad-output/implementation-artifacts/4-*-*.md`
- Sprint Status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Risk Pipeline Audit: `_bmad-output/planning-artifacts/risk-pipeline-audit-4-0.md`
- CalendarGovernor: `src/router/calendar_governor.py`
- Risk Endpoints: `src/api/risk_endpoints.py`
- Physics Sensors: `src/risk/physics/ising_sensor.py`, `src/risk/physics/hmm_sensor.py`
- Risk Canvas: `quantmind-ide/src/lib/components/canvas/RiskCanvas.svelte`

### Risk Scoring Reference

| Score | Action | Gate Impact |
| ----- | ------ | ---------- |
| 1-3 | DOCUMENT | None |
| 4-5 | MONITOR | Watch closely |
| 6-8 | MITIGATE | CONCERNS at gate |
| 9 | BLOCK | Automatic FAIL |

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
**Mode**: Epic-Level (Phase 4)
