---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
inputDocuments:
  - '_bmad-output/implementation-artifacts/3-0-live-trading-backend-mt5-bridge-audit.md'
  - '_bmad-output/implementation-artifacts/3-1-websocket-position-pl-streaming-backend.md'
  - '_bmad-output/implementation-artifacts/3-2-kill-switch-backend-all-tiers.md'
  - '_bmad-output/implementation-artifacts/3-3-session-mask-islamic-compliance-loss-cap-apis.md'
  - '_bmad-output/implementation-artifacts/3-4-live-trading-canvas-layout-bot-status-grid-streaming-ui.md'
  - '_bmad-output/implementation-artifacts/3-5-kill-switch-ui-all-tiers.md'
  - '_bmad-output/implementation-artifacts/3-6-manual-trade-controls-ui.md'
  - '_bmad-output/implementation-artifacts/3-7-morningdigestcard-degraded-mode-rendering.md'
  - '_bmad/tea/testarch/knowledge/risk-governance.md'
  - '_bmad/tea/testarch/knowledge/probability-impact.md'
  - '_bmad/tea/testarch/knowledge/test-levels-framework.md'
  - '_bmad/tea/testarch/knowledge/test-priorities-matrix.md'
---

# Test Design: Epic 3 - Live Trading Command Center

**Date:** 2026-03-21
**Author:** Master Test Architect (TEA)
**Status:** Draft

---

## Executive Summary

**Scope:** Full test design for Epic 3 - Live Trading Command Center

**Epic Overview:**
Epic 3 delivers the Live Trading Command Center with real-time WebSocket streaming, multi-tier kill switch controls, Islamic compliance monitoring, manual trade controls, and graceful degraded mode handling across distributed Cloudzy/Contabo nodes.

**Risk Summary:**

- Total risks identified: 12
- High-priority risks (≥6): 5
- Critical categories: PERF (streaming latency), SEC (kill switch), OPS (node independence), TECH (atomic execution)

**Coverage Summary:**

- P0 scenarios: 8 (estimated 16 hours)
- P1 scenarios: 12 (estimated 12 hours)
- P2/P3 scenarios: 15 (estimated 8 hours)
- **Total effort**: 36 hours (~4.5 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | -------- | ---------- |
| **MT5 ZMQ Protocol Details** | MetaTrader 5 is external proprietary system; ZMQ protocol tested via bridge interface only | MT5 bridge integration tests with mocked ZMQ |
| **MQL5 EA Code Generation** | Covered in Epic 7 (Development Department) | Separate test design for Epic 7 |
| **Prefect Workflow Infrastructure** | Morning digest aggregation uses Prefect; backend tested but Prefect itself not tested | Verify API contract only |
| **Historical Backtest Results** | Epic 4 covers backtest infrastructure | Separate test design for Epic 4 |
| **Paper Trading Simulation** | Paper trading has separate Epic 6 canvas | Verify UI wiring only |
| **Contabo Agent Node internals** | Agents run on Contabo; tested via API contracts | Integration tests with mocked agents |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- | -------- |
| R-001 | PERF | WebSocket streaming latency exceeds 3s threshold under load | 3 | 2 | 6 | Performance monitoring in production, latency tests | QA | 2026-03-22 |
| R-002 | SEC | Kill switch tier activation fails to execute atomically - partial state left | 2 | 3 | 6 | Atomic execution tests, audit log verification | QA | 2026-03-22 |
| R-003 | OPS | Cloudzy trading breaks when Contabo is unreachable (NFR-R4 violation) | 2 | 3 | 6 | Node independence integration tests, failover scenarios | QA | 2026-03-23 |
| R-004 | TECH | MT5 ZMQ reconnection exceeds 10s target (NFR-R5) causing missed ticks | 2 | 3 | 6 | Reconnection time tests, connection monitoring | QA | 2026-03-23 |
| R-005 | SEC | Kill switch audit logs can be modified/deleted (NFR-D2 violation) | 1 | 3 | 3 | Immutability tests, database permission checks | DEV | 2026-03-24 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- |
| R-006 | PERF | MorningDigestCard causes canvas render jank on first load | 2 | 2 | 4 | Performance profiling, lazy loading | DEV |
| R-007 | TECH | State replay sends stale data to reconnecting WebSocket clients | 2 | 2 | 4 | State freshness validation, TTL enforcement | QA |
| R-008 | BUS | Islamic compliance force-close fires at wrong time (21:45 UTC) | 2 | 2 | 4 | Timezone validation, edge case tests | QA |
| R-009 | DATA | Loss cap breach events not propagated to frontend via WebSocket | 2 | 2 | 4 | Event broadcasting verification tests | QA |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------ |
| R-010 | BUS | P&L flash animation causes visual glitches on rapid updates | 2 | 1 | 2 | Monitor |
| R-011 | TECH | BotStatusGrid skeleton loading shows incorrectly on slow connection | 2 | 1 | 2 | Monitor |
| R-012 | OPS | Degraded mode indicators fail to clear after Contabo reconnects | 1 | 2 | 2 | Manual verification |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] Requirements and acceptance criteria reviewed by QA, Dev, PM
- [ ] Test environment provisioned with Cloudzy node accessible
- [ ] MT5 bridge mock/simulator available for testing
- [ ] WebSocket endpoint `/ws/trading` accessible
- [ ] Kill switch API endpoints deployed to test environment
- [ ] Bot registry seeded with test EA instances
- [ ] Islamic compliance test bot configured
- [ ] Loss cap test scenarios configured

## Exit Criteria

- [ ] All P0 tests passing
- [ ] All P1 tests passing (or failures triaged)
- [ ] No open high-priority / high-severity bugs
- [ ] NFR-P2 (≤3s latency) verified with performance tests
- [ ] NFR-R4 (Cloudzy independence) verified with Contabo unreachable
- [ ] NFR-D2 (audit immutability) verified
- [ ] Test coverage agreed as sufficient

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| WebSocket streaming ≤3s latency | API | R-001 | 3 | QA | Under load, position/P&L events |
| Kill switch Tier 1 atomic execution | API | R-002 | 3 | QA | Soft stop, no new entries |
| Kill switch Tier 2 atomic execution | API | R-002 | 3 | QA | Strategy pause, isolation |
| Kill switch Tier 3 atomic execution | API | R-002 | 4 | QA | Emergency close, partial fills |
| Cloudzy independence (no Contabo) | Integration | R-003 | 3 | QA | WebSocket, kill switch work |
| MT5 bridge reconnection ≤10s | Integration | R-004 | 2 | QA | ZMQ reconnect scenarios |
| Kill switch audit log immutability | API | R-005 | 2 | QA | Append-only verification |

**Total P0**: 20 tests, 16 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| State replay on WebSocket reconnect | API | R-007 | 3 | QA | NFR-R3 compliance |
| Islamic compliance force-close (21:45 UTC) | Unit | R-008 | 4 | QA | Timezone edge cases |
| Loss cap breach WebSocket notification | Integration | R-009 | 2 | QA | Event propagation |
| MorningDigestCard renders on first load | Component | R-006 | 2 | DEV | Performance check |
| Bot status grid updates within 3s | Component | R-001 | 3 | QA | E2E latency verification |
| Kill switch confirmation modal flow | E2E | R-002 | 2 | QA | UI + API integration |
| Manual close position flow | API | R-002 | 3 | QA | Close result display |
| Degraded mode (Contabo offline) | Integration | R-003 | 3 | QA | Amber indicators, no blanks |

**Total P1**: 22 tests, 12 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| P&L flash animation green/red | Component | 3 | DEV | Visual verification |
| BotStatusGrid skeleton loading | Component | 2 | DEV | Loading states |
| Bot detail page navigation | E2E | 3 | QA | BreadcrumbNav |
| Session mask display (ASIAN/LONDON/NY) | Component | 4 | QA | Active session highlighting |
| Force close countdown calculation | Unit | 3 | DEV | Within 60-min window |
| Emergency close double-confirm modal | E2E | 2 | QA | Positions + exposure display |
| Degraded mode auto-recovery (10s) | Integration | 2 | QA | Contabo reconnect |
| Node health badge display | Component | 2 | DEV | Cloudzy/Contabo status |

**Total P2**: 21 tests, 7 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| Cross-canvas 3-dot menu close | E2E | 2 | QA | Other canvases |
| Lucide icons (no emoji) verification | Visual | 1 | QA | Accessibility |
| Frosted Terminal aesthetic compliance | Visual | 1 | QA | GlassTile styling |
| Market session indicator (Tokyo/London/NY) | Component | 2 | DEV | Open/closed state |
| Accessibility (aria-labels on kill switch) | Accessibility | 2 | QA | Screen reader |

**Total P3**: 8 tests, 2 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Kill switch status endpoint returns valid response (30s)
- [ ] WebSocket connects and receives state snapshot (45s)
- [ ] Bot list endpoint returns expected structure (30s)
- [ ] Kill switch Tier 1 soft stop fires successfully (60s)

**Total**: 4 scenarios

### P0 Tests (<30 min)

**Purpose**: Critical path validation

- [ ] WebSocket position_update latency ≤3s (E2E)
- [ ] WebSocket pnl_update latency ≤3s (E2E)
- [ ] Kill switch Tier 1 atomic execution (API)
- [ ] Kill switch Tier 2 atomic execution (API)
- [ ] Kill switch Tier 3 atomic execution (API)
- [ ] Kill switch audit log immutability (API)
- [ ] Cloudzy independence (Contabo unreachable) (Integration)
- [ ] MT5 bridge reconnection ≤10s (Integration)

**Total**: 8 scenarios

### P1 Tests (<60 min)

**Purpose**: Important feature coverage

- [ ] State replay on reconnect (API)
- [ ] Islamic compliance 21:45 UTC force-close (Unit)
- [ ] Loss cap breach notification (Integration)
- [ ] Bot status grid real-time updates (E2E)
- [ ] Kill switch confirmation modal (E2E)
- [ ] Manual close position (API)
- [ ] Degraded mode Contabo offline (Integration)
- [ ] MorningDigestCard first load (Component)

**Total**: 8 scenarios

### P2/P3 Tests (<90 min)

**Purpose**: Full regression coverage

- [ ] P&L flash animation (Component)
- [ ] Session mask display (Component)
- [ ] Force close countdown (Unit)
- [ ] Emergency close double-confirm (E2E)
- [ ] Degraded mode auto-recovery (Integration)
- [ ] Bot detail navigation (E2E)
- [ ] Cross-canvas 3-dot menu (E2E)
- [ ] Accessibility verification (A11y)
- [ ] Market session indicator (Component)
- [ ] Visual aesthetic compliance (Visual)

**Total**: 10 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| -------- | ----- | ---------- | ----------- | ----- |
| P0 | 20 | 0.8 | 16 | Complex async, security tests |
| P1 | 22 | 0.55 | 12 | Standard coverage |
| P2 | 21 | 0.33 | 7 | Simple scenarios |
| P3 | 8 | 0.25 | 2 | Exploratory |
| **Total** | **71** | **-** | **37** | **~4.6 days** |

### Prerequisites

**Test Data:**

- `bot_factory` - Faker-based factory for EA instances with strategy_id, symbol, regime
- `position_factory` - Open positions with ticket, direction, lot_size, entry_price
- `kill_switch_audit_factory` - Immutable audit log entries
- `session_mask_fixture` - ASIAN/LONDON/NY/OVERLAP/CLOSED states

**Tooling:**

- `pytest` with `pytest-asyncio` for async API tests
- `playwright` for E2E and component tests
- `websockets` library for WebSocket client testing
- `faker` for data factories
- `freezegun` for timezone/time testing

**Environment:**

- Cloudzy node accessible (or mocked)
- MT5 bridge mock/simulator running
- Contabo node reachable or mocked for degraded mode tests
- Test database with immutable audit log schema

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths (kill switch, WebSocket)**: ≥90%
- **Security scenarios (SEC category)**: 100%
- **Business logic (Islamic compliance, loss cap)**: ≥80%
- **Edge cases (reconnection, degraded mode)**: ≥60%

### Non-Negotiable Requirements

- [ ] All P0 tests pass
- [ ] No high-risk (≥6) items unmitigated
- [ ] Kill switch atomic execution tests pass 100%
- [ ] Cloudzy independence verified (NFR-R4)
- [ ] WebSocket latency verified ≤3s (NFR-P2)

---

## Mitigation Plans

### R-001: WebSocket streaming latency exceeds 3s (Score: 6)

**Mitigation Strategy:** Implement performance monitoring, add latency assertions to tests, profile broadcaster
**Owner:** QA
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Automated latency tests under load (10 bots, 50 positions)

### R-002: Kill switch atomic execution failure (Score: 6)

**Mitigation Strategy:** Comprehensive atomic tests, verify audit log completeness after each tier
**Owner:** QA
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Each tier verified to complete ALL steps, audit log entries verified

### R-003: Cloudzy trading breaks without Contabo (Score: 6)

**Mitigation Strategy:** Integration tests with Contabo unreachable, verify all Cloudzy-dependent features
**Owner:** QA
**Timeline:** 2026-03-23
**Status:** Planned
**Verification:** Network isolation tests, degraded mode tests

### R-004: MT5 reconnection exceeds 10s (Score: 6)

**Mitigation Strategy:** Connection monitoring tests, timeout assertions
**Owner:** QA
**Timeline:** 2026-03-23
**Status:** Planned
**Verification:** Automated ZMQ reconnect timing tests

### R-005: Kill switch audit logs modifiable (Score: 3)

**Mitigation Strategy:** Database permission checks, append-only schema enforcement
**Owner:** DEV
**Timeline:** 2026-03-24
**Status:** Planned
**Verification:** Attempt DELETE/UPDATE on audit log entries, verify rejection

---

## Assumptions and Dependencies

### Assumptions

1. MT5 bridge can be mocked for automated testing (real MT5 not required)
2. Cloudzy and Contabo nodes can be independently accessed/mocked in test environment
3. WebSocket endpoint `/ws/trading` is stable and not undergoing concurrent development
4. Bot registry API `/api/v1/trading/bots` returns consistent data structure
5. Kill switch backend Tier 1/2/3 endpoints are functional

### Dependencies

1. **Epic 3-1 (WebSocket backend)** - `/ws/trading` endpoint must be deployed
2. **Epic 3-2 (Kill switch backend)** - Tier 1/2/3 trigger endpoints must be functional
3. **Epic 3-3 (Bot params API)** - `/api/v1/trading/bots/{id}/params` must be functional
4. **Epic 3-4 (Live Trading Canvas)** - Frontend components must be integrated
5. **Epic 3-5 (Kill switch UI)** - Frontend modals must be wired to backend

### Risks to Plan

- **Risk**: MT5 bridge mock instability causes flaky tests
  - **Impact**: P0 tests fail intermittently
  - **Contingency**: Add retry logic, mock stability monitoring
- **Risk**: Contabo node not accessible for degraded mode tests
  - **Impact**: NFR-R4 verification blocked
  - **Contingency**: Use network isolation tools (iptables) to simulate

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| ----------------- | ------ | ---------------- |
| **Epic 2 (Providers)** | Kill switch and trading use provider config | Provider config tests must pass |
| **Epic 4 (Risk)** | Risk parameters API used by loss cap | Risk params tests must pass |
| **Epic 5 (Memory)** | Session checkpoints for bot state | Session checkpoint tests must pass |
| **Epic 7 (Departments)** | DepartmentMail for kill switch notifications | Mail streaming tests should pass |
| **WebSocket /ws/trading** | All real-time features depend on this | WebSocket tests must pass |
| **MT5 Bridge (tick_stream_handler)** | Position/P&L data source | Bridge integration tests must pass |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- Epic 3 Context: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Story 3-1: `_bmad-output/implementation-artifacts/3-1-websocket-position-pl-streaming-backend.md`
- Story 3-2: `_bmad-output/implementation-artifacts/3-2-kill-switch-backend-all-tiers.md`
- Story 3-3: `_bmad-output/implementation-artifacts/3-3-session-mask-islamic-compliance-loss-cap-apis.md`
- Story 3-4: `_bmad-output/implementation-artifacts/3-4-live-trading-canvas-layout-bot-status-grid-streaming-ui.md`
- Story 3-5: `_bmad-output/implementation-artifacts/3-5-kill-switch-ui-all-tiers.md`
- Story 3-6: `_bmad-output/implementation-artifacts/3-6-manual-trade-controls-ui.md`
- Story 3-7: `_bmad-output/implementation-artifacts/3-7-morningdigestcard-degraded-mode-rendering.md`

### Existing Test Coverage

| Test File | Stories Covered | Lines |
| --------- | --------------- | ----- |
| `tests/api/test_websocket_streaming.py` | 3-1 | 661 |
| `tests/api/test_kill_switch_tiers.py` | 3-2 | 305 |
| `tests/api/test_session_loss_cap.py` | 3-3 | 296 |
| `tests/api/test_position_close.py` | 3-6 | 404 |

### NFR Compliance Matrix

| NFR | Requirement | Verification Method |
| --- | ----------- | ------------------- |
| NFR-P1 | Kill switch executes in full, in order | Atomic execution tests |
| NFR-P2 | ≤3s lag on live data | Latency profiling tests |
| NFR-R3 | State replay before live updates | Reconnection tests |
| NFR-R4 | Cloudzy trades without Contabo | Network isolation tests |
| NFR-R5 | MT5 ZMQ reconnect ≤10s | Reconnection timing tests |
| NFR-D2 | Immutable audit log | Permission/append-only tests |

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
