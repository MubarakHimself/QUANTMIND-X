---
stepsCompleted: ['step-01-preflight-and-context', 'step-02-generation-mode']
lastStep: 'step-02-generation-mode'
lastSaved: '2026-03-21'
epic_num: 10
epic_title: 'Audit, Monitoring & Notifications'
test_artifacts: '{project-root}/_bmad-output/test-artifacts'
inputDocuments:
  - '_bmad-output/test-artifacts/test-design-epic-10.md'
  - 'src/api/audit_endpoints.py'
  - 'src/api/reasoning_log_endpoints.py'
  - 'src/api/notification_config_endpoints.py'
  - 'src/monitoring/cold_storage_sync.py'
  - 'src/database/models/audit_log.py'
---

# ATDD Checklist - Epic 10: Audit, Monitoring & Notifications

**Generated:** 2026-03-21
**Author:** Claude (Master Test Architect)
**Workflow:** bmad-tea-testarch-atdd
**Mode:** YOLO (Fully Autonomous)

---

## Executive Summary

Epic 10 covers the Audit, Monitoring & Notifications system with 5-layer audit logging, natural language query, agent reasoning transparency, notification configuration, and cold storage integrity verification.

**P0 Test Status (TDD Red Phase):**
- **Total P0 Tests:** 16
- **Failing (Red):** 3 - Represent gaps in implementation requiring attention
- **Passing (Green):** 13 - Implementation exists and works correctly

**High-Priority Risks Covered:**
| Risk ID | Category | Description | Score | Test Coverage |
|---------|----------|-------------|-------|---------------|
| R-001 | SEC | Audit log immutability not enforced at DB level | 6 | 3 tests (2 failing) |
| R-002 | DATA | Cold storage integrity - SHA256 verification untested | 6 | 3 tests (1 failing) |
| R-003 | PERF | NL query degrades with large time ranges | 4 | 6 tests (passing) |
| R-004 | TECH | Notification toggle race with concurrent delivery | 4 | 2 tests (passing) |

---

## Test Execution Results

### TDD Red Phase - Failing Tests (REQUIRE IMPLEMENTATION)

These tests FAIL because the expected behavior is not yet implemented:

#### 1. Audit Log Immutability - DB Level Enforcement (R-001)

| Test | Status | Issue |
|------|--------|-------|
| `test_delete_audit_entry_raises_error` | FAIL | DELETE succeeds - DB-level constraint not enforced |
| `test_update_audit_entry_raises_error` | FAIL | UPDATE succeeds - DB-level constraint not enforced |
| `test_no_delete_method_on_model` | PASS | ORM-level protection exists |

**Gap:** SQLite database does not enforce immutability at the constraint level. DELETE and UPDATE operations succeed when they should be blocked by DB-level triggers or constraints.

**Required Implementation:**
- Add SQLite trigger to prevent DELETE on audit_log table
- Add SQLite trigger to prevent UPDATE on audit_log table
- Or: Use a database that supports row-level immutability (e.g., PostgreSQL with policies)

---

#### 2. Cold Storage Integrity - Tamper Detection (R-002)

| Test | Status | Issue |
|------|--------|-------|
| `test_sync_detects_tampered_file` | FAIL | Sync re-copies source without verifying existing cold storage |
| `test_checksum_file_created_after_sync` | PASS | Checksum files are created |
| `test_verify_integrity_on_restored_file` | PASS | verify_file_integrity() works correctly |

**Gap:** When syncing files to cold storage, the implementation copies the source file again without verifying the existing file in cold storage against its stored checksum. Tampered files are not detected.

**Required Implementation:**
- Before re-syncing, verify existing cold storage file against its checksum
- If checksum mismatch detected, move corrupted file to failed queue
- Log integrity violation alert

---

### TDD Green Phase - Passing Tests (IMPLEMENTATION EXISTS)

#### 3. NL Query Time Resolution (R-003)

| Test | Status |
|------|--------|
| `test_parse_last_week_time_reference` | PASS |
| `test_parse_yesterday_specific_time` | PASS |
| `test_parse_today_time_reference` | PASS |

**Implementation Verified:** NLQueryParser correctly handles time references.

---

#### 4. NL Query Entity Mapping (R-003)

| Test | Status |
|------|--------|
| `test_extract_single_ea_entity` | PASS |
| `test_extract_multiple_ea_entities` | PASS |
| `test_extract_symbol_entity` | PASS |

**Implementation Verified:** Entity extraction works for EA IDs, multiple EAs, and currency symbols.

---

#### 5. Notification Always-On Events (R-004)

| Test | Status |
|------|--------|
| `test_notification_toggle_loss_cap_always_on` | PASS |
| `test_notification_toggle_system_critical_always_on` | PASS |

**Implementation Verified:** Always-on events (kill_switch, loss_cap, system_critical) correctly return 400 when toggle attempted.

---

#### 6. Reasoning API Department Query (R-002/R-003)

| Test | Status |
|------|--------|
| `test_get_department_reasoning_research` | PASS |
| `test_get_reasoning_by_decision_id` | PASS |

**Implementation Verified:** Reasoning API endpoints return OPINION nodes with required fields.

---

## Implementation Checklist

### Must Fix (P0 - Before Sprint End)

- [ ] **R-001: Add DB-level DELETE prevention for audit_log**
  - Create SQLite trigger: `CREATE TRIGGER prevent_audit_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'Audit log is immutable'); END;`
  - Or implement application-level guard in SQLAlchemy

- [ ] **R-001: Add DB-level UPDATE prevention for audit_log**
  - Create SQLite trigger: `CREATE TRIGGER prevent_audit_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'Audit log is immutable'); END;`

- [ ] **R-002: Implement cold storage tamper detection**
  - Modify `ColdStorageSync.sync_logs()` to check existing cold storage files
  - Before copying, verify destination checksum matches stored checksum
  - If mismatch, mark as failed and do not overwrite

---

### Already Implemented (Verify in PR)

- [x] NL Query Parser - time reference resolution
- [x] NL Query Parser - entity extraction
- [x] Notification always-on enforcement (API level)
- [x] Reasoning API - department query
- [x] Reasoning API - decision_id lookup
- [x] Cold storage - checksum calculation
- [x] Cold storage - verify_file_integrity function

---

## Test File Locations

| Test Class | File | Lines |
|------------|------|-------|
| TestAuditLogImmutabilityDBLevel | `tests/api/test_epic10_p0_atdd.py` | 28-150 |
| TestColdStorageIntegrity | `tests/api/test_epic10_p0_atdd.py` | 155-240 |
| TestNLQueryTimeResolution | `tests/api/test_epic10_p0_atdd.py` | 245-310 |
| TestNLQueryEntityMapping | `tests/api/test_epic10_p0_atdd.py` | 315-360 |
| TestNotificationAlwaysOnEvents | `tests/api/test_epic10_p0_atdd.py` | 365-420 |
| TestReasoningAPIDepartmentQuery | `tests/api/test_epic10_p0_atdd.py` | 425-520 |

---

## Running the Tests

```bash
# Run all P0 tests
python3 -m pytest tests/api/test_epic10_p0_atdd.py -v

# Run only failing tests (TDD red)
python3 -m pytest tests/api/test_epic10_p0_atdd.py -v -k "delete_audit or update_audit or detects_tampered"

# Run with detailed output
python3 -m pytest tests/api/test_epic10_p0_atdd.py -vv --tb=long
```

---

## Notes

1. **TDD Cycle:** These tests follow Red-Green-Refactor cycle. Currently in RED phase with 3 failing tests.

2. **Database Constraint Limitation:** SQLite does not support CHECK constraints that can prevent UPDATE/DELETE at the row level without triggers. Consider migrating to PostgreSQL for production.

3. **Cold Storage Verification:** The current implementation creates checksum files but doesn't verify existing files on re-sync. This is the primary gap for R-002.

4. **Test Isolation:** Immutability tests use in-memory SQLite with created tables. This ensures tests don't affect production data.

---

**Generated by:** BMad TEA Agent - Test Architect Module
**Version:** 5.0 (BMad v6)
**Next Step:** Implement failing tests or hand off to development team
