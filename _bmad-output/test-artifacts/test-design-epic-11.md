---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context']
lastStep: 'step-02-load-context'
lastSaved: '2026-03-21'
inputDocuments:
  - "_bmad-output/implementation-artifacts/11-0-infrastructure-system-state-audit.md"
  - "_bmad-output/implementation-artifacts/11-1-nightly-rsync-cron.md"
  - "_bmad-output/implementation-artifacts/11-2-weekend-compute-protocol-scheduled-background-tasks.md"
  - "_bmad-output/implementation-artifacts/11-3-node-sequential-update-automatic-rollback.md"
  - "_bmad-output/implementation-artifacts/11-4-itt-rebuild-portability-full-backup-restore.md"
  - "_bmad-output/implementation-artifacts/11-5-flowforge-canvas-prefect-kanban-node-graph.md"
  - "_bmad-output/implementation-artifacts/11-6-server-migration-multi-platform-build.md"
  - "_bmad-output/implementation-artifacts/11-7-theme-presets-wallpaper-system.md"
  - "_bmad/tea/testarch/knowledge/risk-governance.md"
  - "_bmad/tea/testarch/knowledge/probability-impact.md"
  - "_bmad/tea/testarch/knowledge/test-levels-framework.md"
  - "_bmad/tea/testarch/knowledge/test-priorities-matrix.md"
---

# Test Design: Epic 11 - System Management & Resilience

**Date:** 2026-03-21
**Author:** Mubarak
**Status:** Draft
**Mode:** Epic-Level (detected via sprint-status.yaml)

---

## Executive Summary

**Scope:** Full test design for Epic 11 - System Management & Resilience

Epic 11 delivers critical infrastructure for system reliability, including backup/restore capabilities, scheduled data synchronization, weekend compute protocols, 3-node sequential updates with rollback, workflow management canvas, multi-platform builds, and theme customization.

**Risk Summary:**

- Total risks identified: 12
- High-priority risks (≥6): 4
- Critical categories: OPS (sequential update failure), DATA (backup integrity), TECH (migration), OPS (rsync)

**Coverage Summary:**

- P0 scenarios: 8 (critical infrastructure, no workaround)
- P1 scenarios: 12 (important features, medium risk)
- P2/P3 scenarios: 15 (secondary features, edge cases)
- **Total effort**: ~55 hours (~7 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| ---- | -------- | ---------- |
| **MT5 live trading state** | MT5 owns live trade data - external system | Tested via MT5 bridge endpoints only |
| **External vendor API testing** | Cloudzy/Contabo APIs tested by upstream teams | Verify rsync success via exit codes |
| **HMM model accuracy** | Model quality validated by research dept | Test pipeline execution, not model outputs |
| **Grafana dashboard UI** | Already covered by Epic 10 observability | Manual visual check only |
| **Legacy TUI components** | Deprecated in favor of ITT | No new testing required |

---

## Risk Assessment

### High-Priority Risks (Score ≥6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- | -------- |
| R-001 | OPS | 3-node sequential update fails mid-sequence, leaving system inconsistent | 2 | 3 | 6 | Health checks before each transition; automatic rollback on failure | Dev/Ops | 2026-03-25 |
| R-002 | DATA | Rsync integrity check fails, corrupted data synced to backup | 2 | 3 | 6 | SHA256 checksum verification pre/post transfer; retry with exponential backoff | Dev/Ops | 2026-03-22 |
| R-003 | TECH | Server migration leaves strategies unable to resume on new host | 2 | 3 | 6 | Pre-migration backup; health check verification; manual resume protocol | Dev/Ops | 2026-03-26 |
| R-004 | OPS | Full backup/restore cycle fails, system unrecoverable | 2 | 3 | 6 | Incremental validation; test restore on staging; checksum verification | Dev/Ops | 2026-03-24 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------------ | ------- |
| R-005 | TECH | FlowForge workflow kill switch affects wrong workflow | 1 | 3 | 3 | Per-card confirmation modal; workflow ID validation; isolated cancellation | Dev |
| R-006 | OPS | Weekend compute retry storm overloads Contabo | 2 | 2 | 4 | Exponential backoff (300s); max 1 retry; monitoring alerts | Dev/Ops |
| R-007 | TECH | Multi-platform build produces platform-specific regressions | 2 | 2 | 4 | CI matrix builds; platform-specific test suites; artifact validation | Dev |
| R-008 | BUS | Theme switching causes visual regression in existing components | 2 | 2 | 4 | Visual regression tests per theme; CSS custom property validation | Dev |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| ------- | -------- | ----------- | ----------- | ------ | ----- | ------- |
| R-009 | TECH | FlowForge node graph SVG rendering fails on large workflows | 1 | 2 | 2 | Lazy loading; viewport limits; zoom/pan controls |
| R-010 | OPS | Cron job misses schedule due to system clock drift | 1 | 2 | 2 | UTC timezone enforcement; cron health monitoring |
| R-011 | TECH | Theme preset persistence fails across sessions | 1 | 1 | 1 | localStorage fallback; default theme validation |
| R-012 | BUS | Scan-line overlay causes accessibility issues | 1 | 1 | 1 | prefers-reduced-motion respected; toggle available |

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
- [ ] Test environment provisioned and accessible (Contabo, Cloudzy, Desktop VMs)
- [ ] Test data available or factories ready (sample trade records, DuckDB tick data)
- [ ] All 8 stories deployed to test environment
- [ ] SSH keys configured for rsync between Cloudzy and Contabo
- [ ] Prefect server running in test environment
- [ ] CI/CD pipeline configured for multi-platform builds
- [ ] Tauri build environment available for desktop testing

## Exit Criteria

- [ ] All P0 tests passing (8/8)
- [ ] All P1 tests passing (12/12) or failures triaged
- [ ] No open high-priority / high-severity bugs (score ≥6)
- [ ] Test coverage agreed as sufficient (≥80% critical paths)
- [ ] Backup/restore cycle verified end-to-end
- [ ] Sequential update rollback tested at least once
- [ ] Multi-platform build artifacts validated (Linux, Windows, macOS)

---

## Project Team (Optional)

**Include only if roles/names are known or responsibility mapping is needed; otherwise omit.**

| Name | Role | Testing Responsibilities |
| ---- | ---- | ----------------------- |
| Mubarak | QA Lead | Test design, P0/P1 execution, risk assessment |
| Dev Team | Developers | Unit tests, P2/P3 coverage, build verification |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (≥6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| 3-node sequential update with health checks | Integration | R-001 | 3 | QA | Contabo→Cloudzy→Desktop order; health check timeout |
| Automatic rollback on health check failure | Integration | R-001 | 2 | QA | Rollback verification; notification sent |
| Rsync integrity verification (SHA256) | Unit | R-002 | 3 | DEV | Pre/post checksum generation; mismatch detection |
| Full backup creation and restore | Integration | R-004 | 3 | QA | Configs, knowledge base, strategies, graph memory |
| Server migration NODE_ROLE transfer | Integration | R-003 | 2 | QA | Credential migration; health check verification |
| Rsync failure notification (morning digest) | Unit | R-002 | 1 | DEV | Log to audit trail; notification file creation |

**Total P0**: 14 tests, 28 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| ----------- | ---------- | --------- | ---------- | ----- | ----- |
| Weekend compute flow execution (4 tasks) | Integration | R-006 | 4 | QA | Monte Carlo, HMM, PageIndex, Correlation |
| Weekend compute retry with exponential backoff | Unit | R-006 | 2 | DEV | 300s delay; max 1 retry |
| FloorManager "What's running this weekend?" query | API | - | 2 | QA | Task list; progress; ETA calculation |
| FlowForge Kanban 6-column layout | Component | - | 1 | QA | PENDING, RUNNING, PENDING_REVIEW, DONE, CANCELLED, EXPIRED_REVIEW |
| FlowForge per-card workflow kill switch | Component | R-005 | 3 | QA | Confirmation modal; workflow isolation |
| FlowForge node graph SVG rendering | Component | R-009 | 2 | QA | Task boxes; directed edges; zoom/pan |
| Multi-platform build (Linux, Windows, macOS) | E2E | R-007 | 6 | CI | Build verification per platform |
| Theme preset switching (4 presets) | Component | R-008 | 4 | QA | CSS custom property validation |
| Theme persistence across sessions | Integration | R-011 | 1 | QA | localStorage; reload validation |

**Total P1**: 25 tests, 25 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| Rsync dry-run mode verification | Unit | 2 | DEV | Dry-run output validation |
| Backup artifact checksum verification | Unit | 2 | DEV | tar.gz checksum generation/validation |
| Restore validation per component | Integration | 4 | QA | Configs, knowledge, strategies, graph memory |
| Migration script dry-run mode | Unit | 2 | DEV | NODE_ROLE validation; credential check |
| CI/CD multi-platform workflow validation | E2E | 3 | CI | Artifact upload; build verification |
| Scan-line overlay toggle | Component | 2 | DEV | prefers-reduced-motion; manual toggle |
| Wallpaper URL input validation | Component | 2 | DEV | URL parsing; fallback on invalid |
| Theme CSS custom property atomic swap | Unit | 4 | DEV | No page reload; instant application |

**Total P2**: 21 tests, 10.5 hours

### P3 (Low) - Run on-demand

| Requirement | Test Level | Test Count | Owner | Notes |
| ----------- | ---------- | ---------- | ----- | ----- |
| Large workflow node graph performance | Performance | 1 | QA | >100 tasks; lazy loading |
| Accessibility audit per theme | Manual | 4 | QA | Contrast ratios; reduced motion |
| Cross-browser theme verification | E2E | 3 | QA | Chrome, Firefox, Safari |
| Backup/restore on minimal disk space | Stress | 1 | DEV | Edge case handling |

**Total P3**: 9 tests, 2.25 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Backup script syntax validation (30s)
- [ ] Restore script syntax validation (30s)
- [ ] Rsync script dry-run to test destination (1min)
- [ ] Migration script dry-run (45s)
- [ ] FlowForge canvas route loads (30s)
- [ ] Theme preset store initialization (30s)

**Total**: 6 scenarios

### P0 Tests (<30 min)

**Purpose**: Critical infrastructure validation

- [ ] 3-node sequential update: Contabo health check passes (E2E)
- [ ] 3-node sequential update: Cloudzy health check passes (E2E)
- [ ] 3-node sequential update: Desktop health check passes (E2E)
- [ ] 3-node sequential update: Rollback triggered on Cloudzy failure (Integration)
- [ ] 3-node sequential update: Notification sent on rollback (Integration)
- [ ] Rsync pre-transfer checksum generation (Unit)
- [ ] Rsync post-transfer checksum match verification (Unit)
- [ ] Rsync checksum mismatch detection and retry (Integration)
- [ ] Full backup creation to tar.gz (Integration)
- [ ] Full restore from backup tar.gz (Integration)
- [ ] Backup artifact SHA256 verification (Unit)
- [ ] Server migration: NODE_ROLE transfer verification (Integration)
- [ ] Server migration: Health check on new host (Integration)
- [ ] Rsync failure: Audit trail logging (Integration)

**Total**: 14 scenarios

### P1 Tests (<60 min)

**Purpose**: Important feature coverage

- [ ] Weekend compute: Monte Carlo task execution (Integration)
- [ ] Weekend compute: HMM retraining execution (Integration)
- [ ] Weekend compute: PageIndex semantic pass (Integration)
- [ ] Weekend compute: Correlation refresh (Integration)
- [ ] Weekend compute retry: Exponential backoff delay (Unit)
- [ ] Weekend compute retry: Max 1 retry enforced (Unit)
- [ ] FloorManager query: Task list response (API)
- [ ] FloorManager query: Progress percentage (API)
- [ ] FloorManager query: ETA calculation (API)
- [ ] FlowForge Kanban: 6 columns render correctly (Component)
- [ ] FlowForge kill switch: Confirmation modal appears (Component)
- [ ] FlowForge kill switch: Only target workflow cancelled (Integration)
- [ ] FlowForge node graph: SVG renders with tasks (Component)
- [ ] Multi-platform build: Linux artifact produced (CI)
- [ ] Multi-platform build: Windows artifact produced (CI)
- [ ] Multi-platform build: macOS artifact produced (CI)
- [ ] Theme: Frosted Terminal CSS properties (Component)
- [ ] Theme: Ghost Panel CSS properties (Component)
- [ ] Theme: Open Air CSS properties (Component)
- [ ] Theme: Breathing Space CSS properties (Component)
- [ ] Theme persistence: Reload maintains selection (Integration)

**Total**: 21 scenarios

### P2/P3 Tests (<90 min)

**Purpose**: Full regression coverage

- [ ] Rsync dry-run: Output matches expected files (Unit)
- [ ] Rsync dry-run: Excludes correctly applied (Unit)
- [ ] Backup checksums: tar.gz integrity verified (Unit)
- [ ] Restore validation: Configs restored correctly (Integration)
- [ ] Restore validation: Knowledge base restored (Integration)
- [ ] Restore validation: Strategy artifacts restored (Integration)
- [ ] Restore validation: Graph memory restored (Integration)
- [ ] Migration dry-run: NODE_ROLE validated (Unit)
- [ ] Migration dry-run: Credentials checked (Unit)
- [ ] CI/CD: Linux build artifact uploaded (CI)
- [ ] CI/CD: Windows build artifact uploaded (CI)
- [ ] CI/CD: macOS build artifact uploaded (CI)
- [ ] Scan-line: Overlay visible when enabled (Component)
- [ ] Scan-line: Hides via prefers-reduced-motion (Component)
- [ ] Wallpaper: Valid URL renders (Component)
- [ ] Wallpaper: Invalid URL shows fallback (Component)
- [ ] Theme CSS: No page reload on switch (Unit)
- [ ] Theme CSS: Properties swap atomically (Unit)
- [ ] Theme CSS: All 4 presets validated (Unit)
- [ ] Large workflow: Node graph lazy loads (Performance)
- [ ] Accessibility: Theme contrast ratios (Manual)
- [ ] Cross-browser: Theme consistency (E2E)
- [ ] Backup stress: Minimal disk space handling (Stress)

**Total**: 23 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| -------- | ----- | ---------- | ----------- | ----- |
| P0 | 14 | 2.0 | 28 | Complex multi-node integration; rollback scenarios |
| P1 | 25 | 1.0 | 25 | Standard component and API tests |
| P2 | 17 | 0.5 | 8.5 | Simple unit and component tests |
| P3 | 9 | 0.25 | 2.25 | Exploratory and performance tests |
| **Total** | **65** | **-** | **~63.75 hours** | **~8 days** |

### Prerequisites

**Test Data:**

- Sample SQLite trade records (`tests/fixtures/trade_records.db`)
- DuckDB tick data snapshot (`tests/fixtures/tick_data_warm/`)
- Sample config directory (`tests/fixtures/config/`)
- Sample strategy artifacts (`tests/fixtures/strategies/`)

**Tooling:**

- Prefect (for workflow testing)
- pytest, pytest-asyncio (Python backend tests)
- Vitest (Svelte component tests)
- Playwright (E2E tests)
- SHA256 checksum tools (integrity verification)

**Environment:**

- Contabo test instance (NODE_ROLE=test)
- Cloudzy test instance
- Desktop VM (Tauri test environment)
- Staging environment for backup/restore validation

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: ≥95% (waivers required for failures)
- **P2/P3 pass rate**: ≥90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: ≥80% (3-node update, backup/restore, rsync)
- **Security scenarios**: 100% (SSH key auth, credential handling)
- **Business logic**: ≥70% (workflow kill switch, theme switching)
- **Edge cases**: ≥50% (disk full, network timeout, checksum mismatch)

### Non-Negotiable Requirements

- [ ] All P0 tests pass
- [ ] No high-risk (≥6) items unmitigated
- [ ] Rsync integrity verification passes 100%
- [ ] Sequential update rollback tested successfully
- [ ] Multi-platform builds produce valid artifacts

---

## Mitigation Plans

### R-001: 3-Node Sequential Update Failure (Score: 6)

**Mitigation Strategy:** Implement health check before each node transition. On failure, automatically rollback the failed node to previous version. Notify Mubarak via Copilot with failure details.

**Owner:** Dev/Ops
**Timeline:** 2026-03-25
**Status:** Planned
**Verification:** Execute update scenario where Cloudzy health check fails; verify Contabo maintains new version, Cloudzy rolls back, notification sent.

### R-002: Rsync Data Corruption (Score: 6)

**Mitigation Strategy:** SHA256 checksum generated before transfer, verified after transfer. On mismatch, retry up to 3 times with exponential backoff. Failed transfers logged to audit trail.

**Owner:** Dev/Ops
**Timeline:** 2026-03-22
**Status:** Planned
**Verification:** Simulate corrupted transfer file; verify checksum mismatch detected, retry attempted, notification sent.

### R-003: Server Migration Failure (Score: 6)

**Mitigation Strategy:** Full backup before migration. NODE_ROLE and credentials transferred securely. Health check verification post-migration. Manual resume protocol documented.

**Owner:** Dev/Ops
**Timeline:** 2026-03-26
**Status:** Planned
**Verification:** Execute migration from staging to staging; verify all configs transferred, health checks pass, strategies resume.

### R-004: Backup/Restore Failure (Score: 6)

**Mitigation Strategy:** Incremental validation during restore. Checksums verified at each step. Test restore on staging before production.

**Owner:** Dev/Ops
**Timeline:** 2026-03-24
**Status:** Planned
**Verification:** Execute full backup; corrupt one file in archive; verify restore detects corruption and reports specific failure.

---

## Assumptions and Dependencies

### Assumptions

1. Contabo, Cloudzy, and Desktop VMs are accessible for integration testing
2. SSH key-based authentication is configured for rsync between Cloudzy and Contabo
3. Prefect server is available and accessible in test environment
4. Tauri build environment can be provisioned for multi-platform testing
5. Test data fixtures (trade records, tick data) can be seeded without affecting production

### Dependencies

1. **Epic 10 observability stack** - Required for health endpoint implementation | Required by: 2026-03-24
2. **Epic 5 FloorManager** - Required for weekend task query integration | Required by: 2026-03-23
3. **Epic 8 Prefect flows** - Required for workflow cancellation API | Required by: 2026-03-25
4. **Epic 1 canvas routing** - Required for FlowForge canvas navigation | Required by: 2026-03-22

### Risks to Plan

- **Risk**: CI/CD multi-platform matrix builds timeout on macOS
  - **Impact**: macOS artifact not validated before release
  - **Contingency**: Increase CI timeout; run macOS build as separate job with longer timeout

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: {name} Date: {date}
- [ ] Tech Lead: {name} Date: {date}
- [ ] QA Lead: {name} Date: {date}

**Comments:**

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| ----------------- | ------ | ---------------- |
| **Prefect server** | FlowForge, weekend compute, sequential update depend on Prefect | Preflight: Verify Prefect server healthy; postflight: Verify all flows in terminal state |
| **FloorManager** | Weekend task query relies on FloorManager intent handlers | Preflight: Verify FloorManager routes "What's running this weekend?" correctly |
| **NODE_ROLE system** | Sequential update, migration depend on correct NODE_ROLE detection | Preflight: Verify NODE_ROLE env var on all 3 nodes |
| **Graph memory store** | Backup/restore must serialize/deserialize graph memory | Preflight: Verify graph memory stores are accessible; postflight: Verify graph state intact |
| **Notification system** | Rsync failure, weekend compute failure, rollback notifications | Preflight: Verify notification config endpoints reachable |
| **TT政务I shell (tauri.conf.json)** | Theme presets depend on transparent window and CSS custom properties | Preflight: Verify Tauri window configured for transparency |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- PRD: Epic 11 - System Management & Resilience
- Epic: `_bmad-output/planning-artifacts/epics.md` (Epic 11)
- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Implementation: `_bmad-output/implementation-artifacts/11-*`

### Story Index

| Story | Title | Status |
|-------|-------|--------|
| 11.0 | Infrastructure & System State Audit | done |
| 11.1 | Nightly Rsync Cron | done |
| 11.2 | Weekend Compute Protocol | done |
| 11.3 | 3-Node Sequential Update & Automatic Rollback | done |
| 11.4 | ITT Rebuild Portability — Full Backup & Restore | done |
| 11.5 | FlowForge Canvas — Prefect Kanban & Node Graph | done |
| 11.6 | Server Migration & Multi-Platform Build | in-progress |
| 11.7 | Theme Presets & Wallpaper System | done |

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
