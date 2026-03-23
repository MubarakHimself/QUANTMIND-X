---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-and-testability', 'step-04-coverage-plan', 'step-05-generate-output']
lastStep: 'step-05-generate-output'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 2
epic_title: AI Providers & Server Connections
inputDocuments:
  - _bmad-output/implementation-artifacts/2-0-provider-infrastructure-audit.md
  - _bmad-output/implementation-artifacts/2-1-provider-configuration-storage-schema.md
  - _bmad-output/implementation-artifacts/2-2-providers-servers-api-endpoints.md
  - _bmad-output/implementation-artifacts/2-3-claude-agent-sdk-provider-routing.md
  - _bmad-output/implementation-artifacts/2-4-provider-hot-swap-without-restart.md
  - _bmad-output/implementation-artifacts/2-5-providerspanel-ui-add-edit-test-delete.md
  - _bmad-output/implementation-artifacts/2-6-server-connection-configuration-panel.md
  - _bmad/tea/testarch/knowledge/risk-governance.md
  - _bmad/tea/testarch/knowledge/probability-impact.md
  - _bmad/tea/testarch/knowledge/test-levels-framework.md
  - _bmad/tea/testarch/knowledge/test-priorities-matrix.md
---

# Test Design: Epic 2 - AI Providers & Server Connections

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft

---

## Executive Summary

**Scope:** Full test design for Epic 2 - AI Providers & Server Connections

**Epic 2 Stories:**
- 2-0: Provider Infrastructure Audit (done)
- 2-1: Provider Configuration Storage Schema (done) - Fernet encryption, new fields
- 2-2: Providers & Servers API Endpoints (done) - Full CRUD for providers/servers
- 2-3: Claude Agent SDK Provider Routing (done) - Primary/fallback routing
- 2-4: Provider Hot Swap Without Restart (done) - Cache TTL, manual refresh
- 2-5: ProvidersPanel UI Add/Edit/Test/Delete (done) - Provider management UI
- 2-6: Server Connection Configuration Panel (done) - Server management UI

**Risk Summary:**

- Total risks identified: 8
- High-priority risks (Score >= 6): 3
- Critical categories: SEC (Security), TECH (Technical)

**Coverage Summary:**

- P0 scenarios: 12 tests (~16 hours)
- P1 scenarios: 15 tests (~15 hours)
- P2/P3 scenarios: 10 tests (~5 hours)
- **Total effort**: 37 tests (~36 hours, ~5 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | --------- | ---------- |
| **MT5 Bridge Integration** | MT5 server connectivity tested in Epic 3 | Covered by Epic 3 test design |
| **Agent Runtime SDK Initialization** | Anthropic SDK tested separately in department tests | Covered by department agent tests |
| **Legacy .env Provider Config** | Replaced by database config per 2-1 | Migration verified in 2-1 AC |
| **Third-party Provider APIs (Anthropic/OpenAI)** | External APIs tested upstream | Use provider /test endpoint |

---

## Risk Assessment

### High-Priority Risks (Score >= 6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- | -------- |
| R-001 | SEC | API key encryption at rest - if Fernet key derivation fails, keys exposed in plaintext | 2 | 3 | 6 | Verify machine-local key generation; test encryption roundtrip; verify no plaintext in logs | Dev | 2026-03-22 |
| R-002 | SEC | API keys appear in HTTP logs or error messages - data exposure | 2 | 3 | 6 | Mask all API keys in responses (AC-1); verify logs exclude keys; test error responses | QA | 2026-03-22 |
| R-003 | TECH | Provider routing returns wrong provider - silent fallback to unintended provider | 2 | 3 | 6 | Test routing with multiple providers; verify primary/fallback selection; verify routing logs | QA | 2026-03-23 |

### Medium-Priority Risks (Score 4-5)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ---------- | ----- |
| R-004 | PERF | Hot-swap cache TTL (5 min default) causes stale provider config during rapid updates | 2 | 2 | 4 | Test cache invalidation on update; verify manual /providers/refresh works; test TTL behavior | QA |
| R-005 | TECH | Concurrent provider config updates cause race condition in thread-local storage | 2 | 2 | 4 | Test concurrent updates to same provider; verify thread-safe access; stress test with load | QA |
| R-006 | DATA | Server config malformed JSON causes API 500 error | 1 | 3 | 3 | Validate server config schema on save; test malformed input rejection; verify error messages | Dev |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------- |
| R-007 | OPS | UI not reflecting provider state after hot-swap - stale display | 2 | 1 | 2 | Monitor |
| R-008 | BUS | Provider dropdown shows disabled providers - poor UX | 1 | 1 | 1 | Monitor |

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
- [ ] Test environment provisioned with SQLite database
- [ ] ProviderConfig model migrated with new schema fields
- [ ] Fernet encryption key generation tested
- [ ] Test providers (Anthropic mock, OpenAI mock) available
- [ ] Backend server running on localhost:8000
- [ ] Frontend dev server accessible at localhost:5173

---

## Exit Criteria

- [ ] All P0 tests passing (encryption, API key masking, routing)
- [ ] All P1 tests passing (or failures triaged with waivers)
- [ ] No open high-priority / high-severity security bugs
- [ ] Test coverage agreed as sufficient by QA lead
- [ ] Hot-swap refresh endpoint tested and working

---

## Project Team

| Name | Role | Testing Responsibilities |
| ---- | ---- | ------------------------ |
| Mubarak | QA Lead | P0 security tests, API integration tests |
| Dev Team | Backend Dev | Unit tests for encryption, routing logic |
| Dev Team | Frontend Dev | Component tests for ProvidersPanel, ServersPanel |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (>=6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ---------- | ---------- | --------- | ---------- | ----- | ----- |
| Fernet encryption roundtrip - encrypt/decrypt API key | Unit | R-001 | 3 | DEV | Test with valid key, invalid key, empty key |
| API keys masked in GET /api/providers response | API | R-002 | 2 | QA | Verify no plaintext key exposure |
| API keys masked in error responses | API | R-002 | 2 | QA | Test provider with invalid key |
| Provider routing selects correct primary provider | API | R-003 | 3 | QA | Test default, explicit selection, no primary |
| Provider routing falls back to secondary on primary failure | API | R-003 | 2 | QA | Test automatic fallback behavior |
| DELETE active provider returns 409 Conflict | API | - | 2 | QA | AC-4 from 2-2 |
| DELETE inactive provider succeeds | API | - | 2 | QA | AC-4 from 2-2 |
| Machine-local Fernet key derivation from UUID | Unit | R-001 | 2 | DEV | Test on different machines |

**Total P0**: 18 tests, 16 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ---------- | ---------- | --------- | ---------- | ----- | ----- |
| POST /api/providers creates provider with encrypted key | API | - | 3 | QA | AC-2 from 2-2 |
| PUT /api/providers/{id} updates fields, preserves key if absent | API | - | 3 | QA | AC-3 from 2-2 |
| POST /api/providers/{id}/test returns success/failure with latency | API | - | 3 | QA | AC-5 from 2-2 |
| GET /api/providers/available returns configured providers | API | - | 2 | QA | Test dropdown population |
| Cache TTL invalidation on provider update | API | R-004 | 2 | QA | Test 5-min TTL behavior |
| Concurrent provider updates - no race condition | API | R-005 | 2 | QA | Stress test with parallel requests |

**Total P1**: 15 tests, 15 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ---------- | ---------- | ---------- | ----- | ----- |
| Server CRUD - POST /api/servers creates server | API | 2 | QA | AC-6 from 2-2 |
| Server CRUD - PUT /api/servers/{id} updates | API | 2 | QA | Verify metadata field rename |
| Server CRUD - DELETE returns 409 for primary | API | 2 | QA | Similar to provider 409 behavior |
| POST /api/servers/{id}/test connectivity | API | 2 | QA | Test latency and status |
| GET /api/servers returns all servers | API | 2 | QA | List endpoint |
| Provider config schema validation - invalid fields rejected | API | 2 | QA | Test with missing required fields |
| UI: ProvidersPanel loads and displays providers | E2E | 2 | QA | Full user flow |
| UI: ProvidersPanel add/edit/delete modal flows | E2E | 3 | QA | Test modal interactions |

**Total P2**: 17 tests, 8.5 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| ---------- | ---------- | ---------- | ----- | ----- |
| UI: Server panel server type icons display | E2E | 1 | QA | Visual verification |
| UI: ProvidersPanel test button shows latency | E2E | 1 | QA | Test success/failure states |
| Provider hot-swap manual refresh flow | E2E | 1 | QA | POST /api/providers/refresh |
| Stress test: 100 concurrent provider reads | API | 1 | QA | Performance benchmarking |

**Total P3**: 4 tests, 2 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] GET /api/providers returns 200 and valid JSON (30s)
- [ ] GET /api/servers returns 200 and valid JSON (30s)
- [ ] GET /api/providers/available returns list (30s)
- [ ] DELETE /api/providers/{id} on nonexistent returns 404 (30s)
- [ ] POST /api/providers with valid payload returns 201 (45s)

**Total**: 5 scenarios

### P0 Tests (<20 min)

**Purpose**: Critical path validation - security and core routing

- [ ] Fernet encryption roundtrip - valid key (Unit, 2min)
- [ ] Fernet encryption roundtrip - invalid key fails (Unit, 1min)
- [ ] Machine-local key derivation - different UUIDs (Unit, 2min)
- [ ] GET /api/providers - keys masked in response (API, 1min)
- [ ] GET /api/providers - keys masked in error responses (API, 1min)
- [ ] Provider routing - selects primary by default (API, 2min)
- [ ] Provider routing - explicit provider selection (API, 2min)
- [ ] Provider routing - fallback on primary failure (API, 3min)
- [ ] DELETE active provider returns 409 (API, 1min)
- [ ] DELETE inactive provider succeeds (API, 1min)
- [ ] POST /api/providers creates with encrypted key (API, 2min)
- [ ] PUT /api/providers preserves key if absent (API, 2min)

**Total**: 12 scenarios

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] POST /api/providers/{id}/test - success returns latency (API, 2min)
- [ ] POST /api/providers/{id}/test - failure returns error (API, 2min)
- [ ] PUT /api/providers updates display_name (API, 1min)
- [ ] PUT /api/providers updates is_active (API, 1min)
- [ ] GET /api/providers/available populates dropdown (API, 1min)
- [ ] Cache invalidation on provider update (API, 3min)
- [ ] Concurrent updates - no race (API, 5min)
- [ ] Server CRUD - create/list (API, 2min)
- [ ] Server CRUD - update (API, 2min)
- [ ] Server CRUD - delete inactive (API, 1min)
- [ ] Server connectivity test endpoint (API, 2min)

**Total**: 11 scenarios

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] All P2/P3 scenarios from coverage plan
- [ ] UI: ProvidersPanel full flow (E2E, 5min)
- [ ] UI: ServersPanel full flow (E2E, 5min)
- [ ] Stress test concurrent reads (API, 5min)

**Total**: 12+ scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| -------- | ----- | ---------- | ----------- | ----- |
| P0 | 12 | 1.5 | 18 | Complex security testing |
| P1 | 11 | 1.0 | 11 | Standard coverage |
| P2 | 10 | 0.5 | 5 | Simple API validation |
| P3 | 4 | 0.25 | 1 | Exploratory |
| **Total** | **37** | **-** | **35** | **~5 days** |

### Prerequisites

**Test Data:**
- ProviderConfig factory (faker-based, auto-cleanup)
- ServerConfig factory (faker-based, auto-cleanup)
- Mock Fernet key fixture for encryption tests

**Tooling:**
- pytest with asyncio_mode = auto for backend
- Vitest with @testing-library/svelte for frontend
- Playwright for E2E (ProvidersPanel, ServersPanel)

**Environment:**
- SQLite test database with fresh migrations
- Mock HTTP responses for provider API calls
- Localhost API and frontend servers

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions) - security critical
- **P1 pass rate**: >=95% (waivers required for failures)
- **P2/P3 pass rate**: >=90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: >=90% (encryption, routing, auth)
- **Security scenarios**: 100% (API key masking, encryption)
- **Business logic**: >=80% (CRUD operations)
- **Edge cases**: >=60% (concurrent updates, malformed data)

### Non-Negotiable Requirements

- [ ] All P0 tests pass - encryption and API key masking verified
- [ ] No high-risk (>=6) items unmitigated
- [ ] Security tests (SEC category) pass 100%
- [ ] DELETE 409 behavior verified for active providers

---

## Mitigation Plans

### R-001: API Key Encryption at Rest (Score: 6)

**Mitigation Strategy:** Test Fernet encryption roundtrip, verify machine-local key derivation, ensure no plaintext in any output path (logs, responses, errors)

**Owner:** Dev Team
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:**
- Unit test: encrypt("key") -> decrypt() returns original
- Unit test: different machine UUIDs produce different keys
- Integration test: verify no plaintext in all API responses
- Log audit: grep logs for any 32+ character strings that could be API keys

### R-002: API Keys Appearing in Logs (Score: 6)

**Mitigation Strategy:** Mask API keys in all responses using `***` or partial display; configure log sanitization middleware

**Owner:** QA Team
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:**
- API test: GET /api/providers - verify keys masked
- API test: POST with invalid key - verify error message has no key
- Log test: after API calls, verify no keys in application logs

### R-003: Provider Routing Returns Wrong Provider (Score: 6)

**Mitigation Strategy:** Test routing logic with multiple providers; verify primary selection and automatic fallback

**Owner:** QA Team
**Timeline:** 2026-03-23
**Status:** Planned
**Verification:**
- API test: default routing uses is_primary=true provider
- API test: explicit provider_id overrides default
- API test: if primary fails, fallback activates automatically
- Log test: routing decisions logged for debugging

---

## Assumptions and Dependencies

### Assumptions

1. Fernet key derivation from machine UUID is deterministic (same UUID = same key)
2. Provider /test endpoint returns within 5s timeout
3. SQLite database supports concurrent reads without locks
4. Frontend API calls use apiFetch wrapper (no hardcoded localhost)
5. Thread-local storage for provider config is thread-safe

### Dependencies

1. cryptography>=3.4.0 package installed - Required by 2026-03-22
2. SQLite database with ProviderConfig schema migrated - Required by 2026-03-22
3. Mock HTTP responses available for provider APIs - Required by 2026-03-23
4. Frontend dev server accessible at localhost:5173 - Required by 2026-03-24

### Risks to Plan

- **Risk**: Provider /test endpoint timeout causes false failures
  - **Impact**: P0 tests fail intermittently, blocking CI
  - **Contingency**: Increase timeout to 10s for /test endpoint; mark as flaky if persists

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests for encryption (separate workflow; not auto-run)
- Run `*automate` for broader coverage once implementation exists

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
| ----------------- | ------ | ---------------- |
| **FloorManager Agent** | Uses ProviderRouter for AI requests | Must test with mock providers; verify routing logs |
| **Department Heads** | Use tier_assignment from provider config | Must verify model_list field populated correctly |
| **SettingsView** | Integrates ProvidersPanel and ServersPanel | E2E test full settings flow |
| **ProviderConfig model** | Schema change affects all provider consumers | Unit test all model field access patterns |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework (TECH/SEC/PERF/DATA/BUS/OPS)
- `probability-impact.md` - Risk scoring (1-9 scale)
- `test-levels-framework.md` - Test level selection (Unit/Integration/E2E)
- `test-priorities-matrix.md` - P0-P3 prioritization criteria

### Related Documents

- Epic 2 Overview: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Story 2-1: `_bmad-output/implementation-artifacts/2-1-provider-configuration-storage-schema.md`
- Story 2-2: `_bmad-output/implementation-artifacts/2-2-providers-servers-api-endpoints.md`
- Story 2-3: `_bmad-output/implementation-artifacts/2-3-claude-agent-sdk-provider-routing.md`
- Story 2-4: `_bmad-output/implementation-artifacts/2-4-provider-hot-swap-without-restart.md`
- Story 2-5: `_bmad-output/implementation-artifacts/2-5-providerspanel-ui-add-edit-test-delete.md`
- Story 2-6: `_bmad-output/implementation-artifacts/2-6-server-connection-configuration-panel.md`
- Existing Tests: `tests/api/test_provider_config.py`

### Test ID Format

Following `{EPIC}.{STORY}-{LEVEL}-{SEQ}` format:

- `2.1-UNIT-001`: Fernet encryption valid key
- `2.1-UNIT-002`: Fernet encryption invalid key
- `2.1-UNIT-003`: Machine-local key derivation
- `2.2-API-001`: GET /api/providers masking
- `2.2-API-002`: DELETE active provider 409
- `2.3-API-001`: Provider routing primary
- `2.3-API-002`: Provider routing fallback
- `2.5-E2E-001`: ProvidersPanel add flow
- `2.6-E2E-001`: ServersPanel CRUD flow

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
