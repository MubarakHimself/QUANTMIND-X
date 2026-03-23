---
stepsCompleted: []
lastStep: ''
lastSaved: ''
workflowType: 'testarch-test-design'
inputDocuments: []
---

# Test Design for QA: QUANTMINDX Full System

**Purpose:** Test execution recipe for QA team. Defines what to test, how to test it, and what QA needs from other teams.

**Date:** 2026-03-21
**Author:** Mubarak (TEA Agent - Master Test Architect)
**Status:** Draft
**Project:** QUANTMINDX

**Related:** See Architecture doc (test-design-architecture.md) for testability concerns and architectural blockers.

---

## Executive Summary

**Scope:** Comprehensive test strategy for QUANTMINDX autonomous trading platform. Covers all 11 completed epics with integrated test scenarios.

**Risk Summary:**

- Total Risks: 45+
- Critical (Score ≥9): 1 (Kill Switch bypass)
- High-Priority (Score ≥6): 11
- Critical Categories: SEC (security), PERF (latency), TECH (memory)

**Coverage Summary:**

- P0 tests: ~150 (critical paths, security, trading)
- P1 tests: ~200 (integration, feature coverage)
- P2 tests: ~150 (edge cases, regression)
- P3 tests: ~50 (exploratory, performance)
- **Total**: ~550 tests (~10-14 weeks with 1 QA)

---

## Not in Scope

**Components or systems explicitly excluded from this test plan:**

| Item | Reasoning | Mitigation |
|------|-----------|-----------|
| **MT5 ZMQ Protocol Internals** | MetaTrader 5 is external proprietary system | Test via MT5 Bridge interface only |
| **MQL5 EA Code Generation** | Covered in Epic 7 department tests | Separate test design |
| **External LLM Provider APIs** | Tested upstream by provider teams | Use /test endpoints |
| **Grafana/Prometheus Internal Ops** | Infrastructure team owns | Verify metrics export only |
| **Tauri Desktop Packaging** | Release engineering owns | Manual smoke test only |
| **Historical Backtest Engine** | Covered in Epic 4 | Separate test design |

---

## Dependencies & Test Blockers

**CRITICAL:** QA cannot proceed without these items from other teams.

### Backend/Architecture Dependencies (Pre-Implementation)

**Source:** See Architecture doc "Quick Guide" for detailed mitigation plans

1. **MT5 Mock Server** - Dev Team - P0
   - What QA needs: HTTP endpoint returning deterministic position/P&L data
   - Why it blocks: Cannot test trading flows without MT5

2. **W3C Trace Context** - Dev Team - P1
   - What QA needs: Correlation IDs in all log entries
   - Why it blocks: Cannot debug cross-service issues

3. **Memory Validation API** - Dev Team - P1
   - What QA needs: `/api/memory/validate` endpoint
   - Why it blocks: Cannot verify 6 memory subsystems consistency

### QA Infrastructure Setup (Pre-Implementation)

1. **Playwright Test Infrastructure** - QA
   - SvelteKit E2E tests
   - API integration tests
   - Parallel execution with sharding

2. **Test Data Factories** - QA
   - Bot factory (faker-based, auto-cleanup)
   - Provider factory (encrypted API keys)
   - Position factory (deterministic P&L)

3. **Mock Servers** - Dev
   - MT5 mock (deterministic trading data)
   - Anthropic/OpenAI mock (LLM responses)

---

## Risk Assessment

**Note:** Full risk details in Architecture doc. This section summarizes risks relevant to QA test planning.

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Score | QA Test Coverage |
|---------|----------|-------------|-------|------------------|
| R-001 | SEC | Kill Switch bypass (Workshop UI) | **9** | E2E routing validation |
| R-002 | SEC | API key encryption exposure | **6** | Unit tests + log audit |
| R-003 | PERF | WebSocket latency >3s | **6** | k6 load test |
| R-004 | TECH | HMM regime accuracy | **6** | Regime validation tests |
| R-005 | TECH | Memory fragmentation | **6** | Memory consistency API |
| R-006 | SEC | Kill switch atomic execution | **6** | Atomic execution tests |
| R-007 | OPS | Node independence violation | **6** | Network isolation tests |
| R-008 | PERF | MT5 reconnection >10s | **6** | Connection timing tests |
| R-009 | DATA | RegimeFetcher poll silent failure | **6** | Alert + fallback tests |
| R-010 | SEC | API keys in HTTP logs | **6** | Log sanitization tests |

### Medium/Low-Priority Risks

| Risk ID | Category | Description | Score |
|---------|----------|-------------|-------|
| R-011 | BUS | Strategy Router auction ambiguity | 4 |
| R-012 | TECH | Migration runner startup delays | 4 |
| R-013 | OPS | LifecycleScheduler timing | 4 |

---

## Entry Criteria

**QA testing cannot begin until ALL of the following are met:**

- [ ] All 11 epic implementations complete and integrated
- [ ] Test environments provisioned (local, staging)
- [ ] Test data factories ready (bot, provider, position)
- [ ] MT5 mock server available (or Windows runner)
- [ ] Playwright infrastructure configured
- [ ] Feature flags set for test execution
- [ ] Baseline performance metrics established

---

## Exit Criteria

**Testing phase is complete when ALL of the following are met:**

- [ ] All P0 tests passing (100% pass required)
- [ ] All P1 tests passing (≥95% pass rate, waivers for failures)
- [ ] No open high-priority / high-severity bugs (Score ≥6)
- [ ] Kill switch E2E validation complete
- [ ] WebSocket latency <3s under load
- [ ] Memory consistency API returns healthy
- [ ] SEC category tests 100% pass rate
- [ ] Test coverage agreed sufficient by QA Lead and Dev Lead

---

## Test Coverage Plan

**IMPORTANT:** P0/P1/P2/P3 = priority and risk level (what to focus on if time-constrained), NOT execution timing. See "Execution Strategy" for when tests run.

### P0 (Critical)

**Criteria:** Blocks core functionality + High risk (≥6) + No workaround + Affects majority of users

| Test ID | Requirement | Test Level | Risk Link | Notes |
|---------|-------------|-----------|-----------|-------|
| P0-001 | Kill Switch two-step confirmation | E2E | R-001 | Verify modal, Escape cancels, Enter doesn't confirm |
| P0-002 | Workshop UI routes to `/api/floor-manager/chat` | E2E | R-001 | Verify agent routing through Department System |
| P0-003 | API keys masked in all responses | API | R-002, R-010 | GET /providers, errors, logs |
| P0-004 | Fernet encryption roundtrip | Unit | R-002 | encrypt → decrypt = original |
| P0-005 | Kill switch atomic execution | API | R-006 | All bots stop, audit logged |
| P0-006 | WebSocket position/P&L streaming <3s | API | R-003 | k6 load test, P99 <3000ms |
| P0-007 | MT5 reconnection ≤10s | API | R-008 | Connection recovery timing |
| P0-008 | RegimeFetcher alert on poll failure | API | R-009 | Alert + fallback regime |
| P0-009 | Node independence (Cloudzy without Contabo) | Integration | R-007 | Network isolation test |
| P0-010 | Bot circuit breaker quarantine | API | R-005 | Consecutive loss → quarantine |

**Total P0:** ~40 tests

---

### P1 (High)

**Criteria:** Important features + Medium risk (3-4) + Common workflows + Workaround exists but difficult

| Test ID | Requirement | Test Level | Risk Link | Notes |
|---------|-------------|-----------|-----------|-------|
| P1-001 | Provider CRUD with encrypted keys | API | R-002 | Create, read, update, delete providers |
| P1-002 | Provider routing primary/fallback | API | R-003 | Default selection, failure fallback |
| P1-003 | Hot-swap cache invalidation | API | R-004 | Manual refresh, TTL behavior |
| P1-004 | Strategy Router auction logic | Unit | R-011 | Priority vs round-robin |
| P1-005 | HMM regime classification accuracy | Unit | R-004 | Validate against ground truth |
| P1-006 | Department mail message bus | Integration | - | SQLite async messaging |
| P1-007 | Agent SSE stream events | E2E | - | Real-time agent events |
| P1-008 | Kelly position sizing calculation | Unit | - | Half-Kelly + physics multiplier |
| P1-009 | Prometheus metrics endpoint | API | - | RED metrics exposed |
| P1-010 | MorningDigestCard degraded mode | E2E | R-006 | Contabo unreachable fallback |

**Total P1:** ~80 tests

---

### P2 (Medium)

**Criteria:** Secondary features + Low risk (1-2) + Edge cases + Regression prevention

| Test ID | Requirement | Test Level | Notes |
|---------|-------------|-----------|-------|
| P2-001 | Bot lifecycle (start/stop/restart) | API | LifecycleScheduler |
| P2-002 | Canvas routing (9 canvases) | E2E | ActivityBar navigation |
| P2-003 | StatusBand session clocks | Component | Tokyo/London/NY |
| P2-004 | Knowledge graph queries | API | Semantic search |
| P2-005 | DuckDB analytics queries | Unit | OHLCV time-series |
| P2-006 | Risk Ising sensor calculation | Unit | Physics model |
| P2-007 | Risk HMM regime model | Unit | Hidden Markov Model |
| P2-008 | Portfolio multi-account routing | Integration | RoutingMatrix |
| P2-009 | Alpha Forge TRD generation | API | Strategy factory |
| P2-010 | MCP server tools | API | MT5, backtest, KB |

**Total P2:** ~150 tests

---

### P3 (Low)

**Criteria:** Nice-to-have + Exploratory + Performance benchmarks

| Test ID | Requirement | Test Level | Notes |
|---------|-------------|-----------|-------|
| P3-001 | Tauri desktop packaging | Manual | OS integration |
| P3-002 | Grafana dashboard validation | Manual | Metrics visualization |
| P3-003 | VPS latency benchmarking | Performance | Contabo/Cloudzy |
| P3-004 | Memory consistency stress test | Performance | 6 subsystems under load |

**Total P3:** ~50 tests

---

## Execution Strategy

**Philosophy:** Run everything in PRs unless there's significant infrastructure overhead. Playwright with parallelization is extremely fast.

### Every PR: Playwright + pytest Tests (~15-20 min)

**All functional tests** (from any priority level):

- All E2E, API, integration, unit tests using Playwright/pytest
- Parallelized across 4 shards
- Total: ~200 Playwright tests + ~100 pytest tests

### Nightly: k6 Performance Tests (~30-60 min)

**All performance tests:**

- WebSocket latency under load
- MT5 reconnection timing
- Memory subsystem throughput
- Total: ~20 k6 tests

### Weekly: Chaos & Long-Running (~hours)

**Special infrastructure tests:**

- Multi-node failover (Cloudzy/Contabo isolation)
- Disaster recovery (backup restore)
- Memory consistency validation
- Endurance tests (4+ hours)

---

## QA Effort Estimate

**QA test development effort only** (excludes DevOps, Backend work):

| Priority | Count | Effort Range | Notes |
|----------|-------|--------------|-------|
| P0 | ~40 | ~80-120 hours | Security, atomic execution, latency |
| P1 | ~80 | ~80-100 hours | Integration, routing, messaging |
| P2 | ~150 | ~75-100 hours | Edge cases, regression |
| P3 | ~50 | ~25-40 hours | Exploratory, performance |
| **Total** | ~320 | **~260-360 hours** | **~6-9 weeks for 1 QA** |

**Epic Breakdown:**

| Epic | P0 | P1 | P2 | P3 | Total |
|------|----|----|----|----|-------|
| 1 - Platform Foundation | 10 | 21 | 24 | 15 | 70 |
| 2 - AI Providers | 18 | 15 | 17 | 4 | 54 |
| 3 - Live Trading | 10 | 20 | 30 | 10 | 70 |
| 4 - Risk | 8 | 15 | 25 | 8 | 56 |
| 5 - Memory | 6 | 12 | 20 | 5 | 43 |
| 6-11 - Others | 28 | 37 | 34 | 8 | 107 |

---

## Tooling & Access

| Tool or Service | Purpose | Access Required | Status |
|-----------------|---------|-----------------|--------|
| Playwright | E2E testing | Read repo | Ready |
| pytest | API/unit testing | Read repo | Ready |
| k6 | Performance testing | Cloud subscription | Pending |
| Mock MT5 Bridge | Trading simulation | Dev to build | Pending |
| Grafana | Metrics validation | Read | Ready |
| SQLite Browser | Department mail debugging | Read | Ready |

**Access requests needed:**

- [ ] k6 Cloud subscription for performance testing
- [ ] Windows runner for MT5 integration tests

---

## Interworking & Regression

**Services and components impacted by this feature:**

| Service/Component | Impact | Regression Scope | Validation Steps |
|-------------------|--------|-------------------|------------------|
| **Kill Switch Backend** | UI wired to existing API | All trading epics | Kill switch tests must pass |
| **ProviderRouter** | All agent calls route through | Epic 2+ | Provider routing tests |
| **Strategy Router** | All trading decisions | Epic 3+ | Auction logic tests |
| **Memory Facade** | All agents use memory | Epic 5+ | Memory consistency tests |
| **MT5 Bridge** | Position/P&L source | Epic 3+ | Bridge integration tests |

**Regression test strategy:**

- All Epic 1 (Platform) tests must pass before frontend changes
- All Epic 2 (Providers) tests must pass before agent changes
- All Epic 3 (Live Trading) tests must pass before trading changes
- Kill Switch tests block any trading feature release

---

## Appendix A: Code Examples & Tagging

**Playwright Tags for Selective Execution:**

```typescript
import { test } from '@playwright/test';

// P0 critical test - runs on every commit
test('@P0 @Security @Trading kill switch two-step confirmation', async ({ page }) => {
  await page.goto('/live-trading');
  await page.click('[data-testid="shield-alert"]');
  await expect(page.locator('.confirm-modal')).toBeVisible();
  await page.keyboard.press('Enter');
  // Should NOT activate kill switch
});

// P1 integration test
test('@P1 @Providers provider routing primary selection', async ({ request }) => {
  const response = await request.get('/api/providers/route');
  expect(response.status()).toBe(200);
  const body = await response.json();
  expect(body.selected).toBe('is_primary');
});

// Run specific tags
// npx playwright test --grep "@P0|@P1"  # P0 + P1 only
// npx playwright test --grep "@Security"  # Security only
```

---

## Appendix B: Knowledge Base References

- **Risk Governance**: `risk-governance.md` - Risk scoring methodology
- **Test Priorities Matrix**: `test-priorities-matrix.md` - P0-P3 criteria
- **Test Levels Framework**: `test-levels-framework.md` - E2E vs API vs Unit
- **Test Quality**: `test-quality.md` - Definition of Done
- **ADR Quality Checklist**: `adr-quality-readiness-checklist.md` - NFR assessment

---

**Generated by:** BMad TEA Agent - Master Test Architect
**Workflow:** `_bmad/tea/testarch/test-design`
**Version:** 4.0 (BMad v6)
**Date:** 2026-03-21
