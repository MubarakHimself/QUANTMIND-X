---
stepsCompleted: []
lastStep: ''
lastSaved: ''
workflowType: 'testarch-test-design'
inputDocuments: []
---

# Test Design for Architecture: QUANTMINDX Full System

**Purpose:** Architectural concerns, testability gaps, and NFR requirements for review by Architecture/Dev teams. Serves as a contract between QA and Engineering on what must be addressed before test development begins.

**Date:** 2026-03-21
**Author:** Mubarak (TEA Agent - Master Test Architect)
**Status:** Architecture Review Pending
**Project:** QUANTMINDX
**Epic Coverage:** All 11 Epics (1-11, all completed)

---

## Executive Summary

**Scope:** System-level test architecture for the QUANTMINDX autonomous algorithmic trading platform. All 11 epics are complete; this document provides an integrated testability review and NFR compliance assessment.

**Business Context:**

- **Revenue/Impact:** Real-time trading platform with live capital exposure
- **Problem:** Multi-agent autonomous trading system requiring 24/7 reliability, sub-second latency, and regulatory compliance
- **GA Launch:** System integrated; production validation ongoing

**Architecture Overview:**

- **Frontend:** SvelteKit + Tauri desktop app (quantmind-ide/)
- **Backend:** Python FastAPI + 50+ API routers (src/)
- **Trading Bridge:** MT5 Bridge on Windows (mt5-bridge/)
- **Agents:** Department-based system with FloorManager orchestration
- **Data:** SQLite + DuckDB + AgentDB + ChromaDB + Qdrant
- **External:** Contabo VPS (HMM training), Cloudzy VPS (trading), Grafana Cloud (monitoring)

**Risk Summary:**

- **Total risks**: 45+ (aggregated from all epics)
- **High-priority (≥6)**: 12 risks requiring immediate mitigation
- **Test effort**: ~500+ tests (~8-12 weeks for 1 QA)

---

## Quick Guide

### 🚨 BLOCKERS - Team Must Decide (Can't Proceed Without)

**Pre-Implementation Critical Path - These MUST be completed before production trading:**

1. **B-001: MT5 Bridge Testability** - Cannot create E2E tests without MT5 mock server (Windows-only dependency)
2. **B-002: W3C Trace Context Missing** - Cannot debug cross-service issues without correlation IDs
3. **B-003: Agent Memory Consistency** - 6 memory subsystems may drift without consistency validation

**What we need from team:** Complete these 3 items or trading validation is blocked.

---

### ⚠️ HIGH PRIORITY - Team Should Validate

1. **R-001 (Epic 1): Kill Switch bypass** - Workshop UI routes to wrong endpoint
2. **R-002 (Epic 2): API key encryption** - Fernet key exposure risk
3. **R-003 (Epic 3): WebSocket latency** - 3s threshold under load
4. **R-004 (Epic 4): Regime HMM accuracy** - Wrong regime = wrong trades
5. **R-005 (Epic 5): Memory fragmentation** - 6 subsystems out of sync

**What we need from team:** Review recommendations and approve or suggest changes.

---

### 📋 INFO ONLY - Solutions Provided (Review, No Decisions Needed)

1. **Test strategy**: Playwright for E2E, pytest for API, Vitest for components
2. **Tiered CI/CD**: PR tests (<15min), Nightly (<60min), Weekly (hours)
3. **Quality gates**: P0=100%, P1≥95%, SEC=100%
4. **Epic test designs**: 11 epic-level test designs already complete

**What we need from team:** Just review and acknowledge.

---

## For Architects and Devs - Open Topics 👷

### Risk Assessment

**Total risks identified**: 45+ (from 11 epics)

#### High-Priority Risks (Score ≥6) - IMMEDIATE ATTENTION

| Risk ID | Category | Description | Prob | Impact | Score | Mitigation | Owner |
|---------|----------|-------------|------|--------|-------|------------|-------|
| R-001 | SEC | Kill Switch bypass - Workshop UI routes to `/api/chat/send` instead of `/api/floor-manager/chat` | 3 | 3 | **9** | Fix CopilotPanel.svelte routing | Dev Lead |
| R-002 | SEC | API key encryption - Fernet key exposure in plaintext | 2 | 3 | **6** | Verify machine-local key derivation, no plaintext in logs | Security |
| R-003 | PERF | WebSocket streaming latency exceeds 3s under load | 3 | 2 | **6** | Latency monitoring, performance profiling | QA |
| R-004 | TECH | HMM regime classification accuracy - wrong regime = wrong trades | 2 | 3 | **6** | Regime validation against ground truth | Research |
| R-005 | TECH | Agent memory fragmentation across 6 subsystems | 2 | 3 | **6** | Consistency checks between memory systems | Backend |
| R-006 | SEC | Kill switch atomic execution - partial state on failure | 2 | 3 | **6** | Atomic execution tests, audit verification | QA |
| R-007 | OPS | Cloudzy/Contabo node independence violation | 2 | 3 | **6** | Network isolation integration tests | QA |
| R-008 | PERF | MT5 ZMQ reconnection exceeds 10s | 2 | 3 | **6** | Connection monitoring, reconnection tests | QA |
| R-009 | DATA | RegimeFetcher poll failure silent | 2 | 3 | **6** | Alert on failure, fallback to last known | Risk |
| R-010 | SEC | API keys appearing in HTTP logs | 2 | 3 | **6** | Mask keys in all responses and logs | QA |
| R-011 | BUS | Strategy Router auction ambiguity | 2 | 2 | **4** | Document exact routing logic in ADR | Research |
| R-012 | TECH | Database migration runner startup delays | 2 | 2 | **4** | Run migrations async, add timeout | Backend |

#### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

### Testability Concerns and Architectural Gaps

**🚨 ACTIONABLE CONCERNS - Architecture Team Must Address**

#### 1. Blockers to Fast Feedback

| Concern | Impact | What Architecture Must Provide | Owner | Timeline |
|---------|--------|-------------------------------|-------|----------|
| **No MT5 Mock Server** | Cannot create E2E trading tests | Create MT5 mock server for testing (Windows/Wine workaround) | Dev | P0 |
| **No W3C Trace Context** | Cannot debug cross-service issues | Add correlation IDs to all API logs | Dev | P1 |
| **Agent Memory Drift** | 6 subsystems may be inconsistent | Implement consistency validation API | Dev | P1 |

#### 2. Architectural Improvements Needed

1. **Memory Consistency Validation API**
   - Current problem: No way to verify AgentDB, Graph, ChromaDB, Qdrant are in sync
   - Required change: Add `/api/memory/validate` endpoint
   - Impact if not fixed: Agent decisions based on stale memory
   - Owner: Backend

2. **MT5 Bridge Mock for Testing**
   - Current problem: MT5 only runs on Windows, E2E tests blocked
   - Required change: Create mock MT5 bridge returning deterministic data
   - Impact if not fixed: Cannot test trading flows in CI
   - Owner: Dev

3. **Distributed Tracing Implementation**
   - Current problem: No W3C Trace Context, correlation IDs missing
   - Required change: Add trace ID to all log entries
   - Impact if not fixed: Debugging requires manual log correlation
   - Owner: Backend

---

### Testability Assessment Summary

**📊 CURRENT STATE - FYI**

#### What Works Well

- ✅ **API-first design** - All business logic accessible via REST API
- ✅ **Feature flags** - Enable/disable features without deploy
- ✅ **Bot circuit breakers** - Per-bot quarantine prevents cascade failures
- ✅ **Progressive kill switch** - Tiered circuit breakers
- ✅ **50-bot global limit** - Prevents over-exposure
- ✅ **Department mail audit trail** - SQLite message bus provides traceability
- ✅ **Prometheus metrics** - RED metrics on port 9090
- ✅ **JSON structured logging** - Promtail/Loki compatible

#### Accepted Trade-offs

For Phase 1, the following trade-offs are acceptable:

- **SQLite operational DB** - Single file may corrupt under extreme load, acceptable for <50 concurrent agents
- **No cross-DC failover** - Single-region deployment, disaster recovery tested separately
- **MT5 Windows-only** - Requires Windows runner for integration tests

---

### Risk Mitigation Plans (High-Priority Risks ≥6)

#### R-001: Kill Switch Bypass (Score: 9) - CRITICAL

**Mitigation Strategy:**
1. Fix CopilotPanel.svelte to route to `/api/floor-manager/chat`
2. Verify all agent interactions go through Department System
3. Add integration test validating routing path

**Owner:** Dev Lead
**Timeline:** Immediate
**Status:** Planned
**Verification:** E2E test that agent chat triggers department mail message

---

#### R-002: API Key Encryption Exposure (Score: 6)

**Mitigation Strategy:**
1. Verify Fernet key derivation from machine UUID
2. Test encryption roundtrip (encrypt → decrypt = original)
3. Scan logs for plaintext keys (grep for 32+ char strings)
4. Mask all API keys in responses using `***` pattern

**Owner:** Security Team
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Unit tests + log audit

---

#### R-003: WebSocket Latency (Score: 6)

**Mitigation Strategy:**
1. Implement latency monitoring in WebSocket handler
2. Profile under load (100+ concurrent connections)
3. Add P95/P99 latency alerting

**Owner:** QA
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** k6 load test measuring WebSocket latency

---

### Assumptions and Dependencies

#### Assumptions

1. MT5 Bridge will eventually have mock server for CI testing
2. Agent memory consistency can be validated via eventual consistency checks
3. All 11 epics are complete and integrated
4. Playwright E2E infrastructure will be set up

#### Dependencies

1. **MT5 Mock Server** - Required before E2E trading tests
2. **W3C Trace Context** - Required for production debugging
3. **Memory Validation API** - Required for agent memory consistency tests
4. **Playwright Setup** - Required for E2E tests

#### Risks to Plan

- **Risk**: MT5 Windows-only blocks full E2E testing
  - Impact: Cannot test trading flows in CI
  - Contingency: Create mock MT5 bridge, manual testing on Windows

---

**End of Architecture Document**

**Next Steps for Architecture Team:**

1. Review Quick Guide (🚨/⚠️/📋) and prioritize blockers
2. Assign owners and timelines for high-priority risks (≥6)
3. Validate assumptions and dependencies
4. Provide feedback to QA on testability gaps

**Next Steps for QA Team:**

1. Wait for pre-implementation blockers to be resolved
2. Refer to companion QA doc (test-design-qa.md) for test scenarios
3. Begin test infrastructure setup (factories, fixtures, environments)
4. Execute epic-level test designs once blockers resolved
