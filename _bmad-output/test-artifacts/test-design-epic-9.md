---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 9
epic_title: 'Portfolio & Multi-Broker Management'
inputDocuments:
  - '_bmad-output/implementation-artifacts/9-0-portfolio-broker-infrastructure-audit.md'
  - '_bmad-output/implementation-artifacts/9-1-broker-account-registry-routing-matrix-api.md'
  - '_bmad-output/implementation-artifacts/9-2-portfolio-metrics-attribution-api.md'
  - '_bmad-output/implementation-artifacts/9-3-portfolio-canvas-multi-account-dashboard-routing-ui.md'
  - '_bmad-output/implementation-artifacts/9-4-portfolio-canvas-attribution-correlation-matrix-performance.md'
  - '_bmad-output/implementation-artifacts/9-5-trading-journal-component.md'
  - '_bmad/tea/testarch/knowledge/risk-governance.md'
  - '_bmad/tea/testarch/knowledge/probability-impact.md'
  - '_bmad/tea/testarch/knowledge/test-levels-framework.md'
  - '_bmad/tea/testarch/knowledge/test-priorities-matrix.md'
---

# Test Design: Epic 9 - Portfolio & Multi-Broker Management

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft
**Mode:** Epic-Level (Phase 4)

---

## Executive Summary

**Scope:** Full test design for Epic 9 - Portfolio & Multi-Broker Management, covering broker account registry, routing matrix API, portfolio metrics/attribution API, portfolio canvas UI, and trading journal component.

**Risk Summary:**

- Total risks identified: 12
- High-priority risks (≥6): 4
- Critical categories: DATA, SEC, PERF

**Coverage Summary:**

- P0 scenarios: 8 (12-16 hours)
- P1 scenarios: 14 (14-20 hours)
- P2/P3 scenarios: 18 (8-14 hours)
- **Total effort**: 34-50 hours (~5-7 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | -------------- | --------------------- |
| **MT5 live trading integration** | Requires live MT5 terminal connection; tested via heartbeat endpoint mocking | Story 3.x (Live Trading Backend) covers MT5 bridge |
| **Real-time WebSocket tick streaming** | Requires live broker data; covered by Story 3.x streaming tests | Live trading canvas tests mock WebSocket |
| **Islamic compliance force-close scheduler (21:45 GMT)** | AC7 gap deferred - calendar/scheduler integration not implemented yet | Track as open issue; defer to Story 9.x follow-up |
| **Correlation matrix production computation** | Currently returns mock data; Story 9.2 confirmed upper-triangle matrix only | Story 9.x enhancement for real correlation engine |
| **Broker heartbeat auto-detection** | Covered by Story 9.1 tests; simulated via mock MT5 heartbeats | Integration with live MT5 pending |
| **External broker APIs (ICMarkets, OANDA, Pepperstone)** | Third-party systems; contract tested via Story 9.1 broker registry | Risk transferred to vendor monitoring |
| **Multi-node deployment (Cloudzy/Contabo split)** | Infrastructure-level concern; platform team covers with health checks | Covered by Story 11.x (Infrastructure Audit) |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ------------- | ----------- | ------ | ----- | ------------ | ------- | -------- |
| R-001 | DATA | **Routing rule NULL ambiguity**: RoutingRule query used OR-with-NULL logic causing wrong rule updates (account_tag="hft" matched NULL-tag rules). Fixed in 9.1. | 2 | 3 | 6 | Unit test + integration test covering NULL-tag routing rules | QA | 2026-03-21 |
| R-002 | DATA | **Attribution field mapping error**: `fetchAttribution()` called wrong endpoint `/api/portfolio/pnl/strategy` instead of `/api/portfolio/attribution`, and extracted non-existent `data.attribution` field. Fixed in 9.4. | 2 | 3 | 6 | API integration tests verifying response field shapes | QA | 2026-03-21 |
| R-003 | DATA | **Correlation field mapping error**: `fetchCorrelation()` extracted `data.correlations` but real response uses `data.matrix`. Fixed in 9.4. | 2 | 3 | 6 | API integration tests verifying correlation response structure | QA | 2026-03-21 |
| R-004 | SEC | **Soft-delete vs hard-delete ambiguity**: DELETE broker account marks inactive but code path may not preserve audit trail for regulatory compliance (NFR-R2). | 2 | 3 | 6 | Verify soft-delete preserves all account history; audit log tests | QA + Dev | 2026-03-22 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ------------- | ----------- | ------ | ----- | ------------ | ------- |
| R-005 | PERF | **@lru_cache singleton staleness**: PortfolioHead cached indefinitely; real data wiring will cause stale data issues. Low risk now (demo data). | 2 | 2 | 4 | Monitor; add cache invalidation when real data integrated | Dev |
| R-006 | BUS | **Drawdown alert notification not wired**: AC4 logs warning but does not trigger actual notification system. Partial implementation. | 2 | 2 | 4 | Integration test verifying notification system called; open issue | QA |
| R-007 | PERF | **PerformancePanel demo data fallback**: AC3 `/api/portfolio/summary` response never populates `data.performance`; always falls back to demo. | 1 | 2 | 2 | Future story for dedicated `/api/portfolio/performance` endpoint | Dev |
| R-008 | TECH | **Svelte 4 reactive declarations in components**: AttributionPanel and CorrelationMatrix used `$:` instead of `$state`/`$derived`. Fixed to Svelte 5 runes in 9.4. | 2 | 1 | 2 | Svelte 5 migration tests verify reactive state | QA |
| R-009 | TECH | **HTTP status code violation**: PUT /routing-rules always returned 201 regardless of create/update. Fixed in 9.1. | 1 | 2 | 2 | API tests verify correct status codes (200=update, 201=create) | QA |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ------------- | ----------- | ------ | ----- | ------- |
| R-010 | OPS | **Demo data fallback masking real issues**: Frontend always falls back to demo when backend unavailable; may mask integration issues | 1 | 2 | 2 | Monitor |
| R-011 | BUS | **Unused GlassTile import**: AttributionPanel and CorrelationMatrix imported GlassTile but never used it. Fixed in 9.4. | 1 | 1 | 1 | Monitor |
| R-012 | OPS | **Missing Svelte 5 runes migration**: PortfolioCanvas uses bare `let` for local state instead of `$state()`. Fixed in 9.4 review. | 1 | 1 | 1 | Monitor |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] All Epic 9 story implementations merged to feature branch
- [ ] Test environment provisioned with database migrations applied
- [ ] Test data factories available: BrokerAccount factory, TradeJournal factory, PortfolioSummary factory
- [ ] Backend API accessible at `/api/portfolio/*` and `/api/brokers/*`
- [ ] Frontend dev server running at `localhost:5173`
- [ ] Story 9.0 audit findings available for infrastructure context
- [ ] Svelte 5 migration verified (npm build succeeds)

## Exit Criteria

- [ ] All P0 tests passing (100%)
- [ ] All P1 tests passing or failures triaged with waivers
- [ ] No open HIGH or CRITICAL severity bugs
- [ ] Test coverage agreed as sufficient by QA Lead
- [ ] API response shapes match acceptance criteria contract
- [ ] All review-identified issues verified fixed

---

## Project Team

| Name | Role | Testing Responsibilities |
| ---- | -------- | ------------------------ |
| Mubarak | QA Lead | Test design, P0/P1 test execution, risk assessment |
| Dev Team | Backend Dev | API unit/integration tests, database model tests |
| Dev Team | Frontend Dev | Component tests, Svelte 5 rune compliance |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| Broker account registration (POST /api/portfolio/brokers) | API | R-001 | 3 | QA | Happy path, validation, duplicate detection |
| Routing rule CRUD with NULL-tag edge case | API | R-001 | 4 | QA | Critical: verify exact NULL match behavior |
| Attribution endpoint field mapping | API | R-002 | 3 | QA | Verify correct endpoint and response fields |
| Correlation endpoint field mapping | API | R-003 | 3 | QA | Verify data.matrix field extraction |
| Broker soft-delete preserves audit trail | API | R-004 | 2 | QA | Verify all account history retained |
| Portfolio canvas tab navigation renders | Component | - | 2 | DEV | Verify all 4 tabs load (Dashboard/Attribution/Correlation/Performance) |
| AttributionPanel strategy table renders | Component | R-002 | 2 | DEV | Verify 6-column table with correct fields |
| CorrelationMatrix heatmap renders | Component | R-003 | 2 | DEV | Verify NxN grid + |r|>=0.7 highlighting |

**Total P0**: 21 tests, 14-18 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ------------- | ---------- | --------- | ---------- | ----- | ------- |
| GET /api/portfolio/routing-matrix returns full matrix | API | R-001 | 3 | QA | Strategies × accounts grid |
| Islamic account swap_free flag auto-set | API | R-006 | 2 | QA | AC7: account_type="Islamic" → swap_free=True |
| Drawdown alert threshold (10%) triggers | API | R-006 | 2 | QA | Verify alert flag + notification call |
| Portfolio summary endpoint total_equity accuracy | API | R-005 | 3 | QA | Aggregation correctness |
| AccountTile component renders equity/drawdown/exposure | Component | - | 2 | DEV | GlassTile grid layout |
| DrawdownAlert banner shows at 10% threshold | Component | R-006 | 2 | DEV | Visual alert rendering |
| RoutingMatrix sub-page regime/strategy-type filters | Component | - | 3 | DEV | Filter dropdown functionality |
| Trading Journal trade log table renders | Component | - | 2 | DEV | Filterable table with columns |
| Trade annotation save and persist | API | - | 3 | QA | AC3: note, annotated_at persistence |
| CSV export endpoint returns valid CSV | API | - | 2 | QA | AC4: Download + content verification |

**Total P1**: 24 tests, 16-22 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| PUT /api/portfolio/brokers/{id} update | API | 3 | QA | Account detail updates |
| GET /api/portfolio/brokers active_only filter | API | 2 | QA | Filter by is_active flag |
| CorrelationMatrix tooltip on hover | Component | 2 | DEV | Strategy A, B, coefficient, period |
| AttributionPanel column sorting | Component | 3 | DEV | Sort by each column |
| PortfolioHead @lru_cache behavior | Unit | 2 | DEV | Verify cache invalidation needs |
| HTTP status codes (200/201/400/404/409) | API | 4 | QA | Error handling coverage |
| PerformancePanel metrics display | Component | 2 | DEV | Fallback demo data behavior |
| Trading Journal trade detail modal | Component | 2 | DEV | Entry/exit prices, slippage, notes |
| Annotation CRUD (create, read, update) | API | 3 | QA | Full lifecycle |
| Invalid inputs (invalid leverage, account_type) | API | 2 | QA | Validation error handling |

**Total P2**: 25 tests, 12-16 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ------------- | ---------- | ---------- | ----- | ------- |
| Visual regression: Frosted Terminal aesthetic | E2E | 2 | QA | GlassTile blur, amber color palette |
| Responsive layout (mobile/tablet breakpoints) | Component | 2 | DEV | Canvas responsiveness |
| CorrelationMatrix NxN reconstruction from upper-triangle | Unit | 2 | DEV | Symmetric matrix build |
| P&L percentage calculations edge cases | Unit | 3 | DEV | Zero division, negative values |
| Multiple broker accounts (4+) rendering | Component | 2 | DEV | Grid layout scaling |
| Regime filter: LONDON, NEW_YORK, ASIAN, OVERLAP, CLOSED | API | 2 | QA | Each regime value |
| Strategy-type filter: SCALPER, HFT, STRUCTURAL, SWING | API | 2 | QA | Each strategy type |
| Export CSV file encoding (UTF-8, special chars) | API | 1 | QA | Non-ASCII symbol names |

**Total P3**: 16 tests, 4-8 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Backend health: GET /api/portfolio/summary returns 200 (15s)
- [ ] Frontend build: npm run build succeeds (60s)
- [ ] Database migrations applied: broker_accounts, routing_rules tables exist (10s)
- [ ] API auth: Valid token grants access to all /api/portfolio/* endpoints (20s)

**Total**: 4 scenarios (~2 min)

### P0 Tests (<15 min)

**Purpose**: Critical path validation

- [ ] POST /api/portfolio/brokers creates account (API, 30s)
- [ ] GET /api/portfolio/routing-matrix returns matrix (API, 20s)
- [ ] PUT /api/portfolio/brokers/{id}/routing-rules creates rule (API, 20s)
- [ ] Routing rule NULL-tag edge case: exact match verification (API, 30s)
- [ ] GET /api/portfolio/attribution returns correct fields (API, 20s)
- [ ] GET /api/portfolio/correlation returns data.matrix (API, 20s)
- [ ] DELETE /api/portfolio/brokers/{id} preserves history (API, 30s)
- [ ] PortfolioCanvas all 4 tabs render (Component, 45s)
- [ ] AttributionPanel 6-column table renders (Component, 30s)
- [ ] CorrelationMatrix heatmap with highlighting renders (Component, 45s)

**Total**: 10 scenarios (~5 min)

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] Islamic account swap_free auto-set (API, 30s)
- [ ] Drawdown > 10% triggers alert (API, 30s)
- [ ] AccountTile equity/drawdown/exposure display (Component, 45s)
- [ ] DrawdownAlert banner visibility (Component, 30s)
- [ ] RoutingMatrix regime filter dropdown (Component, 45s)
- [ ] RoutingMatrix strategy-type filter dropdown (Component, 45s)
- [ ] TradingJournal trade log table renders (Component, 60s)
- [ ] Trade annotation POST and persistence (API, 45s)
- [ ] CSV export endpoint (API, 60s)

**Total**: 9 scenarios (~8 min)

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] All remaining API tests (22 scenarios, ~20 min)
- [ ] All remaining Component tests (14 scenarios, ~25 min)
- [ ] Visual regression suite (2 scenarios, ~10 min)

**Total**: 38 scenarios (~55 min)

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| --------- | ----------------- | ---------- | ----------------- | ----------------------- |
| P0 | 21 | 0.75 | 14-18 | Complex setup, API contracts, Svelte components |
| P1 | 24 | 0.70 | 16-22 | Standard coverage, UI integration |
| P2 | 25 | 0.50 | 12-16 | Simple scenarios, edge cases |
| P3 | 16 | 0.30 | 4-8 | Exploratory, visual |
| **Total** | **86** | **-** | **46-64 hours** | **~6-8 days** |

### Prerequisites

**Test Data:**

- `BrokerAccountFactory` - faker-based, auto-cleanup for broker registration tests
- `RoutingRuleFactory` - for routing matrix CRUD tests
- `TradeJournalFactory` - for trading journal annotation tests
- `PortfolioSummaryFactory` - for metrics attribution tests

**Tooling:**

- pytest with asyncio_mode=auto for Python backend tests
- Vitest for Svelte/TypeScript frontend component tests
- Playwright for E2E visual regression tests

**Environment:**

- Staging API at `localhost:8000` with SQLite database
- Frontend dev server at `localhost:5173`
- Both services running simultaneously for fullstack tests

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: ≥90% (broker registration, routing, attribution, correlation)
- **API endpoints**: 100% (all /api/portfolio/* and /api/brokers/* covered)
- **Component rendering**: ≥80% (all Svelte components)
- **Edge cases**: ≥60% (NULL-tag, Islamic accounts, correlation threshold)

### Non-Negotiable Requirements

- [ ] All P0 tests pass (R-001, R-002, R-003, R-004 fully covered)
- [ ] No HIGH risks (score ≥6) unmitigated
- [ ] API response shapes verified against AC contracts
- [ ] Svelte 5 runes compliance confirmed

---

## Mitigation Plans

### R-001: Routing rule NULL ambiguity (Score: 6)

**Mitigation Strategy:** Add integration test specifically covering routing rule with NULL account_tag. Verify that when updating a rule with `account_tag="hft"`, it does NOT match existing rules with `account_tag=NULL`.

**Owner:** QA
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Test `PUT /api/portfolio/brokers/{id}/routing-rules` with explicit tag verifies exact match

### R-002: Attribution field mapping error (Score: 6)

**Mitigation Strategy:** Add API integration test that calls `/api/portfolio/attribution` and verifies the response contains `by_strategy` array with `equity_contribution` field. Currently `fetchAttribution()` calls wrong endpoint.

**Owner:** QA
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Test verifies correct endpoint AND correct response field extraction

### R-003: Correlation field mapping error (Score: 6)

**Mitigation Strategy:** Add API integration test that calls `/api/portfolio/correlation` and verifies response contains `matrix` (not `correlations`). Response shape validation.

**Owner:** QA
**Timeline:** 2026-03-21
**Status:** Planned
**Verification:** Test verifies `data.matrix` exists with correct structure

### R-004: Soft-delete audit trail (Score: 6)

**Mitigation Strategy:** Add test verifying DELETE marks account `is_active=False` but all account data (balance history, trades) remains queryable for audit purposes.

**Owner:** QA + Dev
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** After DELETE, GET account returns 200 with `is_active=False` and historical data intact

---

## Assumptions and Dependencies

### Assumptions

1. Backend API runs on FastAPI with SQLite; no external database dependencies
2. Frontend Svelte 5 migration complete; `$state` and `$derived` runes used consistently
3. All story implementations (9.1-9.5) are available in feature branch
4. MT5 connection simulated via mock heartbeat data
5. Islamic compliance (21:45 GMT force-close) deferred to future story

### Dependencies

1. Story 9.1 implementation (broker registry, routing matrix API) - Required by 2026-03-22
2. Story 9.2 implementation (portfolio metrics API) - Required by 2026-03-22
3. Svelte 5 component migration complete - Required by 2026-03-21
4. Database migrations (`add_broker_account_tables.py`) applied - Required before test execution

### Risks to Plan

- **Risk**: Islamic compliance force-close not implemented (AC7 gap)
  - **Impact**: Non-compliant Islamic accounts may not auto-close at 21:45 GMT
  - **Contingency**: Document as known gap; require manual verification before Islamic account go-live

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests from acceptance criteria (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: {name} Date: {date}
- [ ] Tech Lead: {name} Date: {date}
- [ ] QA Lead: {name} Date: {date}

**Comments:**

---

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| ----------------- | -------------- | ------------------------------- |
| **src/api/portfolio_endpoints.py** | Core API for all portfolio queries | Existing `/api/portfolio/*` tests must pass |
| **src/api/portfolio_broker_endpoints.py** | Broker account CRUD | Existing broker API tests must pass |
| **src/api/journal_endpoints.py** | Trading journal and annotations | Existing journal tests must pass |
| **src/router/routing_matrix.py** | Strategy-to-account routing | Routing logic regression tests |
| **src/database/models/broker_account.py** | Persistent broker data | Database model tests |
| **quantmind-ide/src/lib/stores/portfolio.ts** | Frontend state management | Portfolio store tests |
| **quantmind-ide/src/lib/components/canvas/PortfolioCanvas.svelte** | Main portfolio UI | Canvas tab navigation tests |
| **src/agents/departments/heads/portfolio_head.py** | PortfolioHead singleton with @lru_cache | Cache behavior tests |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology (1-3 scale)
- `test-levels-framework.md` - Test level selection (Unit/Integration/E2E)
- `test-priorities-matrix.md` - P0-P3 prioritization criteria

### Related Documents

- Epic 9 Context: `_bmad-output/planning-artifacts/epics.md#Epic-9`
- Story 9.0 Audit: `_bmad-output/implementation-artifacts/9-0-portfolio-broker-infrastructure-audit.md`
- Story 9.1: `_bmad-output/implementation-artifacts/9-1-broker-account-registry-routing-matrix-api.md`
- Story 9.2: `_bmad-output/implementation-artifacts/9-2-portfolio-metrics-attribution-api.md`
- Story 9.3: `_bmad-output/implementation-artifacts/9-3-portfolio-canvas-multi-account-dashboard-routing-ui.md`
- Story 9.4: `_bmad-output/implementation-artifacts/9-4-portfolio-canvas-attribution-correlation-matrix-performance.md`
- Story 9.5: `_bmad-output/implementation-artifacts/9-5-trading-journal-component.md`
- Architecture: `_bmad-output/planning-artifacts/architecture.md`

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 5.0 (BMad v6)
