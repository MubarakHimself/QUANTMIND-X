---
title: 'TEA Test Design → BMAD Handoff Document'
version: '1.0'
workflowType: 'testarch-test-design-handoff'
inputDocuments:
  - _bmad-output/implementation-artifacts/sprint-status.yaml
  - _bmad/tea/testarch/knowledge/risk-governance.md
  - _bmad/tea/testarch/knowledge/probability-impact.md
  - _bmad/tea/testarch/knowledge/test-levels-framework.md
  - _bmad/tea/testarch/knowledge/test-priorities-matrix.md
sourceWorkflow: 'testarch-test-design'
generatedBy: 'TEA Master Test Architect'
generatedAt: '2026-03-22'
projectName: 'QUANTMINDX'
---

# TEA → BMAD Integration Handoff

## Purpose

This document bridges TEA's test design outputs with BMAD's epic/story decomposition workflow (`create-epics-and-stories`). It provides structured integration guidance so that quality requirements, risk assessments, and test strategies flow into implementation planning.

## TEA Artifacts Inventory

| Artifact | Path | BMAD Integration Point |
|---|---|---|
| System Architecture Test Design | `_bmad-output/test-artifacts/test-design-architecture.md` | Epic quality requirements, story acceptance criteria |
| System QA Test Design | `_bmad-output/test-artifacts/test-design-qa.md` | Epic quality requirements, story acceptance criteria |
| Epic 1 Test Design | `_bmad-output/test-artifacts/test-design-epic-1.md` | Platform foundation test coverage |
| Epic 2 Test Design | `_bmad-output/test-artifacts/test-design-epic-2.md` | AI Providers & Server Connections |
| Epic 3 Test Design | `_bmad-output/test-artifacts/test-design-epic-3.md` | Live Trading & MT5 Bridge |
| Epic 4 Test Design | `_bmad-output/test-artifacts/test-design-epic-4.md` | Risk Pipeline & Physics Models |
| Epic 5 Test Design | `_bmad-output/test-artifacts/test-design-epic-5.md` | Memory Architecture & Graph |
| Epic 6-11 Test Designs | `_bmad-output/test-artifacts/test-design-epic-6.md` through `test-design-epic-11.md` | Department System, Knowledge, Portfolio, Alpha Forge, Canvas Context, Workflow Orchestration |
| Risk Assessment | (embedded in architecture + QA docs) | Epic risk classification, story priority |
| Coverage Strategy | (embedded in architecture + QA docs) | Story test requirements |

---

## Epic-Level Integration Guidance

### Risk References

**P0/P1 risks that should appear as epic-level quality gates:**

| Risk ID | Category | Score | Epic | Mitigation Owner |
|---|---|---|---|---|
| R-001 | SEC | **9** | Epic 1 (Platform) | Dev Lead |
| R-002 | SEC | 6 | Epic 2 (Providers) | Security |
| R-003 | PERF | 6 | Epic 3 (Live Trading) | QA |
| R-004 | TECH | 6 | Epic 4 (Risk) | Research |
| R-005 | TECH | 6 | Epic 5 (Memory) | Backend |
| R-006 | SEC | 6 | Epic 1 (Platform) | QA |
| R-007 | OPS | 6 | Epic 1 (Platform) | QA |
| R-008 | PERF | 6 | Epic 3 (Live Trading) | QA |
| R-009 | DATA | 6 | Epic 4 (Risk) | Risk |
| R-010 | SEC | 6 | Epic 2 (Providers) | QA |

### Quality Gates

**Recommended quality gates per epic based on risk assessment:**

| Epic | Gate Criteria |
|---|---|
| **Epic 1** | Kill Switch E2E routing validated before any trading release; P0 security tests 100% pass |
| **Epic 2** | API key masking verified in all responses; Fernet encryption roundtrip confirmed |
| **Epic 3** | WebSocket latency <3s under load; MT5 reconnection ≤10s; bot circuit breaker validated |
| **Epic 4** | RegimeFetcher poll failure alerts; HMM regime accuracy validated against ground truth |
| **Epic 5** | Memory consistency API returns healthy; 6 subsystems validation |
| **Epic 6-11** | Department mail audit trail verified; Canvas routing validated |

---

## Story-Level Integration Guidance

### P0/P1 Test Scenarios → Story Acceptance Criteria

**Critical test scenarios that MUST be acceptance criteria:**

| Story | Test Scenario | Acceptance Criteria |
|---|---|---|
| Kill Switch | Two-step confirmation modal | Escape cancels, Enter does NOT confirm, explicit button required |
| Kill Switch | Workshop UI routing | Must route to `/api/floor-manager/chat`, NOT `/api/chat/send` |
| Kill Switch | Atomic execution | All bots stop atomically; audit log complete |
| Provider CRUD | API key masking | GET /providers returns masked keys (`***`); no plaintext in logs |
| Provider CRUD | Fernet encryption | encrypt → decrypt = original; machine-local key derivation |
| Provider Routing | Primary/fallback | Default selects `is_primary=true`; fallback activates on failure |
| Live Trading | WebSocket latency | Position/P&L streaming P99 <3000ms under 100+ concurrent connections |
| Live Trading | MT5 reconnection | Connection recovery ≤10s; no silent failures |
| Risk | RegimeFetcher failure | Alert fires on poll failure; fallback to last known regime |
| Memory | Consistency validation | `/api/memory/validate` endpoint returns healthy across 6 subsystems |

### Data-TestId Requirements

**Recommended data-testid attributes for testability:**

| Component | data-testid | Purpose |
|---|---|---|
| Kill Switch Button | `shield-alert` | E2E routing to kill switch modal |
| Kill Switch Modal | `confirm-modal` | Verify modal visibility |
| Confirm Button | `confirm-kill-switch` | Explicit confirmation required |
| Cancel Button | `cancel-kill-switch` | Escape key must cancel |
| Provider List | `providers-list` | API response validation |
| Provider Add | `add-provider-btn` | Modal trigger |
| Bot Status Grid | `bot-status-grid` | Live trading bot monitoring |
| Canvas Nav | `canvas-activity-bar` | 9-canvas navigation |
| Session Clock | `session-clock-{zone}` | Tokyo/London/NY clocks |

---

## Risk-to-Story Mapping

| Risk ID | Category | P×I | Score | Recommended Story/Epic | Test Level |
|---|---|---|---|---|---|
| R-001 | SEC | 3×3 | **9** | Epic 1 - Kill Switch | E2E |
| R-002 | SEC | 2×3 | 6 | Epic 2 - Provider Encryption | Unit + API |
| R-003 | PERF | 3×2 | 6 | Epic 3 - WebSocket Streaming | API + k6 |
| R-004 | TECH | 2×3 | 6 | Epic 4 - HMM Regime | Unit |
| R-005 | TECH | 2×3 | 6 | Epic 5 - Memory Consistency | API |
| R-006 | SEC | 2×3 | 6 | Epic 1 - Kill Switch Atomic | API |
| R-007 | OPS | 2×3 | 6 | Epic 1 - Node Independence | Integration |
| R-008 | PERF | 2×3 | 6 | Epic 3 - MT5 Reconnection | API |
| R-009 | DATA | 2×3 | 6 | Epic 4 - RegimeFetcher | API |
| R-010 | SEC | 2×3 | 6 | Epic 2 - Log Sanitization | API |
| R-011 | BUS | 2×2 | 4 | Epic 3 - Strategy Router | Unit |
| R-012 | TECH | 2×2 | 4 | Backend - Migration Runner | Integration |
| R-013 | OPS | 2×2 | 4 | Backend - LifecycleScheduler | Unit |

---

## Recommended BMAD → TEA Workflow Sequence

1. **TEA Test Design** (`TD`) → produces this handoff document
2. **BMAD Create Epics & Stories** → consumes this handoff, embeds quality requirements
3. **TEA ATDD** (`AT`) → generates acceptance tests per story
4. **BMAD Implementation** → developers implement with test-first guidance
5. **TEA Automate** (`TA`) → generates full test suite
6. **TEA Trace** (`TR`) → validates coverage completeness

---

## Phase Transition Quality Gates

| From Phase | To Phase | Gate Criteria |
|---|---|---|
| Test Design | Epic/Story Creation | All P0 risks have mitigation strategy (R-001 Score 9 addressed first) |
| Epic/Story Creation | ATDD | Stories have acceptance criteria from test design; Kill Switch routing validated |
| ATDD | Implementation | Failing acceptance tests exist for all P0/P1 scenarios (R-001 through R-010) |
| Implementation | Test Automation | All acceptance tests pass |
| Test Automation | Release | Trace matrix shows ≥80% coverage of P0/P1 requirements; SEC tests 100% pass |

---

## Pre-Implementation Blockers

**Critical items that must be resolved before QA can begin:**

| Blocker | Owner | Target Date | Blocks |
|---|---|---|---|
| MT5 Mock Server | Dev | 2026-03-22 | Epic 3 E2E tests |
| W3C Trace Context | Dev | 2026-03-23 | Cross-service debugging |
| Memory Validation API | Backend | 2026-03-23 | Epic 5 consistency tests |
| Kill Switch routing fix (R-001) | Dev Lead | **Immediate** | Trading release blocked |

---

**Generated by:** BMad TEA Agent - Master Test Architect
**Workflow:** `_bmad/tea/testarch/test-design`
**Version:** 4.0 (BMad v6)
**Date:** 2026-03-22
