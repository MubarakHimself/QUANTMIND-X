# Test Design: Epic 10 - Audit, Monitoring & Notifications

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft

---

## Executive Summary

**Scope:** full test design for Epic 10

**Risk Summary:**

- Total risks identified: 8
- High-priority risks (>=6): 2
- Critical categories: SEC, DATA

**Coverage Summary:**

- P0 scenarios: 16 (~16 hours)
- P1 scenarios: 20 (~20 hours)
- P2/P3 scenarios: 22 (~9.25 hours)
- **Total effort**: ~45.25 hours (~1 sprint)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| --- | --- | --- |
| **Legacy audit_log.py migration** | Pre-Epic 10 audit system, replaced by new AuditLogEntry model | Covered by existing system tests |
| **Third-party notification providers** | AlertService is external; integration tested via contract | Monitor AlertService SLA |
| **Mobile/tablet UI variants** | Frosted Terminal is desktop-only per architecture | Manual testing on desktop only |
| **Multi-node cold storage sync race conditions** | Epic 11 handles node sync; this epic covers single Contabo storage | Coordinate with Epic 11 |

---

## Risk Assessment

### High-Priority Risks (Score >=6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-001 | SEC | Audit log immutability not enforced at database level | 2 | 3 | 6 | Unit tests verify no DELETE endpoint; DB-level constraints | Backend | Sprint 1 |
| R-002 | DATA | Cold storage integrity - SHA256 verification untested | 2 | 3 | 6 | Test checksum generation, verification, corruption detection | Backend | Sprint 1 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-003 | PERF | NL query degrades with large time ranges | 2 | 2 | 4 | Test pagination, time range limits, query timeouts | Backend |
| R-004 | TECH | Notification toggle race with concurrent delivery | 2 | 2 | 4 | Test concurrent toggle + delivery scenarios | Backend |
| R-005 | OPS | Server health polling may miss brief spikes | 2 | 2 | 4 | Test different polling intervals | DevOps |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| --- | --- | --- | --- | --- | --- | --- |
| R-006 | SEC | Opinion node department filtering untested | 1 | 3 | 3 | Test role-based filtering |
| R-007 | TECH | Frontend timeline rendering with 100+ events | 2 | 1 | 2 | Monitor |
| R-008 | PERF | Server health polling overhead | 1 | 2 | 2 | Monitor |

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
- [ ] Test environment provisioned and accessible
- [ ] Test data available or factories ready
- [ ] Feature deployed to test environment
- [ ] Database models migrated (AuditLogEntry, NotificationConfig)
- [ ] Backend API endpoints reachable at /api/audit/*, /api/notifications, /api/server/health/*

## Exit Criteria

- [ ] All P0 tests passing
- [ ] All P1 tests passing (or failures triaged)
- [ ] No open high-priority / high-severity bugs
- [ ] Test coverage agreed as sufficient
- [ ] Security tests (R-001, R-002, R-006) pass 100%
- [ ] Cold storage checksum verification tested

---

## Project Team

| Name | Role | Testing Responsibilities |
| --- | --- | --- |
| Mubarak | QA Lead | Test design, P0/P1 execution |
| Backend Team | Backend Dev | Unit tests for APIs, cold storage |
| Frontend Team | Frontend Dev | Component tests for panels |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (>=6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Audit log immutability - no DELETE endpoint | Unit | R-001 | 3 | Backend | Verify no delete routes exist |
| NL query returns causal chain for EA pause | API | R-003 | 5 | QA | Time resolution, entity mapping |
| Reasoning API returns OPINION nodes | API | R-002 | 4 | Backend | decision_id lookup, department query |
| Notification toggle - always-on events blocked | API | R-004 | 4 | Backend | kill_switch, loss_cap, system_critical |

**Total P0**: 16 tests, 16 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Cold storage checksum verification | Unit | R-002 | 4 | Backend | SHA256 generation, corruption detection |
| Server health metrics accuracy | API | R-005 | 6 | Backend | CPU, memory, disk, latency, uptime |
| NL query time resolution (yesterday, last week) | Unit | R-003 | 6 | Backend | Time parsing edge cases |
| Notification settings panel toggle UI | Component | - | 4 | Frontend | Toggle state, disabled state |

**Total P1**: 20 tests, 20 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- |
| Opinion node department filtering | API | 4 | Backend | Role-based access control |
| Timeline rendering with 100+ events | E2E | 3 | Frontend | Performance with large datasets |
| Server health threshold breach alerting | Integration | 4 | Frontend | Red indicator, Copilot notification |
| Notification delivery suppression verification | Integration | 4 | Backend | Toggle verified at delivery time |

**Total P2**: 15 tests, 7.5 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- |
| Copilot audit query via NL | E2E | 2 | QA | Full journey "Why was EA_X paused?" |
| Edge case: empty audit log query | Unit | 3 | Backend | Graceful handling |
| UI visual regression (panels) | Visual | 2 | Frontend | Screenshot comparison |

**Total P3**: 7 tests, 1.75 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Audit API health check (5s)
- [ ] Notification config GET endpoint (5s)
- [ ] Server health metrics endpoint (5s)
- [ ] Reasoning API health check (5s)

**Total**: 4 scenarios

### P0 Tests (<20 min)

**Purpose**: Critical path validation

- [ ] Audit log immutability verification - no DELETE endpoint (Unit)
- [ ] NL query with "Why was EA_X paused yesterday?" (API)
- [ ] Reasoning API - GET /api/audit/reasoning/{decision_id} (API)
- [ ] Notification toggle - always-on events blocked (API)

**Total**: 16 scenarios

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] Cold storage checksum verification (Unit)
- [ ] Server health metrics accuracy (API)
- [ ] NL query time resolution (Unit)
- [ ] Notification settings panel toggle (Component)

**Total**: 20 scenarios

### P2/P3 Tests (<30 min)

**Purpose**: Full regression coverage

- [ ] Opinion node department filtering (API)
- [ ] Timeline rendering with 100+ events (E2E)
- [ ] Server health threshold breach alerting (Integration)
- [ ] Copilot audit query via NL (E2E)

**Total**: 22 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| --- | --- | --- | --- | --- |
| P0 | 16 | 1.0 | 16 | Complex setup, security testing |
| P1 | 20 | 1.0 | 20 | Standard coverage |
| P2 | 15 | 0.5 | 7.5 | Simple scenarios |
| P3 | 7 | 0.25 | 1.75 | Exploratory |
| **Total** | **58** | **-** | **45.25** | **~1 sprint** |

### Prerequisites

**Test Data:**

- audit_log_factory (faker-based, auto-cleanup) - audit entries across all 5 layers
- notification_config_factory (preset event types)
- opinion_node_factory (mock OPINION nodes for reasoning API)
- server_metrics_factory (psutil mock data)

**Tooling:**

- pytest for Python backend tests
- vitest for Svelte frontend tests
- Playwright for E2E tests
- psutil for server health metrics mocking

**Environment:**

- Contabo backend: http://localhost:8000
- Cloudzy trading node: accessible for health metrics
- SQLite database with AuditLogEntry, NotificationConfig tables
- Mock cold storage directory for checksum tests

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: >=95% (waivers required for failures)
- **P2/P3 pass rate**: >=90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: >=80%
- **Security scenarios**: 100%
- **Business logic**: >=70%
- **Edge cases**: >=50%

### Non-Negotiable Requirements

- [ ] All P0 tests pass
- [ ] No high-risk (>=6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] Cold storage integrity verification tested

---

## Mitigation Plans

### R-001: Audit Log Immutability Not Enforced (Score: 6)

**Mitigation Strategy:** Add database-level constraints to prevent UPDATE/DELETE on audit_log table. Unit tests verify no DELETE endpoint exists in router.

**Owner:** Backend Team
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Run `grep -r "delete" audit_endpoints.py` - should find no DELETE routes

### R-002: Cold Storage Integrity Verification (Score: 6)

**Mitigation Strategy:** Test checksum generation matches input, verification detects tampered files, corruption triggers alert

**Owner:** Backend Team
**Timeline:** Sprint 1
**Status:** Planned
**Verification:** Unit tests with known checksums and intentionally corrupted files

---

## Assumptions and Dependencies

### Assumptions

1. Cold storage is accessible via local filesystem (Contabo)
2. Notification delivery happens synchronously (no queue)
3. Opinion nodes are stored in memory graph with proper indexes
4. Server health metrics are collected via psutil

### Dependencies

1. Story 10.1 (NL Query API) - Required for audit query tests
2. Story 10.2 (Reasoning API) - Required for opinion node tests
3. Story 10.3 (Notification Config) - Required for notification toggle tests
4. Story 10.5 (Server Health Panel) - Required for metrics tests

### Risks to Plan

- **Risk**: Cold storage path not accessible in test environment
  - **Impact**: Cold storage tests cannot run
  - **Contingency**: Use mock filesystem or skip cold storage tests

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: Date:
- [ ] Tech Lead: Date:
- [ ] QA Lead: Date:

**Comments:**

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| --- | --- | --- |
| **Memory Graph (Story 5.1)** | Stores OPINION nodes queried by reasoning API | test_reasoning_log.py must pass |
| **FloorManager (Story 5.4)** | Routes audit queries | Integration tests with FloorManager |
| **AlertManager (Story 3.2)** | Kill switch notifications must remain always-on | test_notification_config.py |
| **Prometheus/Grafana** | Server health metrics source | Coordinate with DevOps |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- PRD: Audit & Monitoring FR59-FR65
- Epic: Epic 10 - Audit, Monitoring & Notifications
- Architecture: `_bmad-output/planning-artifacts/architecture.md#11-Audit-Log-FR59-65`
- Tech Spec: Story implementation artifacts in `_bmad-output/implementation-artifacts/10-*.md`

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
