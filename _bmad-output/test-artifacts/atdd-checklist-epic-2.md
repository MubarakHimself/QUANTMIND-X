---
stepsCompleted: ['step-01-preflight-and-context', 'step-02-generation-mode', 'step-03-test-strategy', 'step-04-generate-tests', 'step-04c-aggregate']
lastStep: 'step-04c-aggregate'
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
  - _bmad-output/test-artifacts/test-design-epic-2.md
---

# ATDD Checklist: Epic 2 - AI Providers & Server Connections

**Date:** 2026-03-21
**Author:** Master Test Architect (BMad TEA)
**Status:** TDD RED PHASE - Tests Generated

---

## TDD Red Phase (Current)

Failing tests have been generated using TDD red phase methodology:
- Tests describe **expected behavior** that is NOT yet fully implemented
- All critical tests are marked with `pytest.mark.skip`
- Tests will **FAIL** when run without the skip marker
- Once feature is implemented, remove `test.skip()` to enter green phase

---

## Test Summary

### Generated Files

| File | Type | Test Count | Skipped | Runnable |
|------|------|------------|---------|----------|
| `tests/api/test_provider_config_p0_atdd.py` | API/Unit | 12 | 11 | 1 |
| `tests/crypto/test_encryption.py` | Unit | 10 | 10 | 0 |
| **Total** | | **22** | **21** | **1** |

### Risk Coverage

| Risk ID | Description | Score | Test Count | File |
|---------|-------------|-------|------------|------|
| R-001 | API key encryption at rest | 6 | 12 | test_encryption.py, test_provider_config_p0_atdd.py |
| R-002 | API keys masked in responses | 6 | 2 | test_provider_config_p0_atdd.py |
| R-003 | Provider routing returns wrong provider | 6 | 4 | test_provider_config_p0_atdd.py |

---

## P0 Tests by Story

### Story 2.1: Provider Configuration Storage Schema

**Fernet Encryption Tests (Unit):**

| Test ID | Test Name | Status | Risk |
|---------|-----------|--------|------|
| 2.1-UNIT-001 | test_encrypt_decrypt_roundtrip_returns_original_value | SKIP | R-001 |
| 2.1-UNIT-002 | test_encrypt_none_returns_none | SKIP | R-001 |
| 2.1-UNIT-003 | test_encrypt_empty_string_returns_none | SKIP | R-001 |
| 2.1-UNIT-004 | test_decrypt_none_returns_none | SKIP | R-001 |
| 2.1-UNIT-005 | test_decrypt_invalid_ciphertext_returns_ciphertext | SKIP | R-001 |
| 2.1-UNIT-006 | test_machine_key_is_deterministic | SKIP | R-001 |
| 2.1-UNIT-007 | test_different_machines_produce_different_keys | SKIP | R-001 |
| 2.1-UNIT-008 | test_machine_key_file_has_restricted_permissions | SKIP | R-001 |
| 2.1-UNIT-009 | test_secure_storage_is_singleton | SKIP | R-001 |
| 2.1-UNIT-010 | test_secure_storage_handles_missing_cryptography | SKIP | R-001 |

### Story 2.2: Providers & Servers API Endpoints

**API Key Masking Tests (API):**

| Test ID | Test Name | Status | Risk |
|---------|-----------|--------|------|
| 2.2-API-001 | test_list_providers_never_exposes_plaintext_api_key | SKIP | R-002 |
| 2.2-API-002 | test_error_responses_never_expose_api_keys | SKIP | R-002 |
| 2.2-API-003 | test_delete_active_provider_returns_409 | RUN | - |
| 2.2-API-004 | test_delete_inactive_provider_succeeds | RUN | - |
| 2.2-API-005 | test_post_provider_encrypts_api_key | SKIP | R-001 |
| 2.2-API-006 | test_put_provider_preserves_key_if_not_provided | SKIP | - |

### Story 2.3: Claude Agent SDK Provider Routing

**Provider Routing Tests (API):**

| Test ID | Test Name | Status | Risk |
|---------|-----------|--------|------|
| 2.3-API-001 | test_router_selects_primary_provider_by_default | SKIP | R-003 |
| 2.3-API-002 | test_router_respects_explicit_provider_selection | SKIP | R-003 |
| 2.3-API-003 | test_router_falls_back_to_secondary_on_primary_failure | SKIP | R-003 |
| 2.3-API-004 | test_router_cache_invalidates_on_refresh | SKIP | - |

---

## Acceptance Criteria Coverage

### Epic 2 Acceptance Criteria

| Criterion | Test(s) Covering | Status |
|-----------|------------------|--------|
| AC-1: API keys masked in GET /api/providers | test_list_providers_never_exposes_plaintext_api_key | SKIP |
| AC-2: POST /api/providers creates with encrypted key | test_post_provider_encrypts_api_key | SKIP |
| AC-3: PUT /api/providers preserves key if absent | test_put_provider_preserves_key_if_not_provided | SKIP |
| AC-4: DELETE active provider returns 409 | test_delete_active_provider_returns_409 | RUN |
| AC-4: DELETE inactive provider succeeds | test_delete_inactive_provider_succeeds | RUN |
| R-001: Fernet encryption roundtrip | test_encrypt_decrypt_roundtrip_returns_original_value | SKIP |
| R-002: API keys never in plaintext | test_list_providers_never_exposes_plaintext_api_key, test_error_responses_never_expose_api_keys | SKIP |
| R-003: Primary provider routing | test_router_selects_primary_provider_by_default, test_router_respects_explicit_provider_selection | SKIP |
| R-003: Fallback on primary failure | test_router_falls_back_to_secondary_on_primary_failure | SKIP |

---

## Next Steps (TDD Green Phase)

After implementing the features:

### Step 1: Enable Tests
```bash
# Remove test.skip() markers - EDIT all test files
# OR use pytest to run only specific tests
pytest tests/api/test_provider_config_p0_atdd.py -v --run-skip
```

### Step 2: Run Tests
```bash
# Run encryption tests
pytest tests/crypto/test_encryption.py -v

# Run API tests
pytest tests/api/test_provider_config_p0_atdd.py -v
```

### Step 3: Verify Green Phase
- All tests should PASS
- If tests fail, either:
  - Fix implementation (bug in feature)
  - Fix test (bug in test)

### Step 4: Commit
```bash
git add tests/api/test_provider_config_p0_atdd.py tests/crypto/test_encryption.py
git commit -m "feat(tests): P0 ATDD tests for Epic 2 provider config"
```

---

## Implementation Guidance

### Endpoints to Implement

| Endpoint | File | Priority |
|----------|------|----------|
| Fernet encrypt/decrypt | src/database/encryption.py | P0 |
| Machine-local key derivation | src/database/encryption.py | P0 |
| API key masking middleware | src/api/provider_config_endpoints.py | P0 |
| Primary provider routing | src/agents/providers/router.py | P0 |
| Fallback routing | src/agents/providers/router.py | P0 |

### Components to Implement

| Component | File | Priority |
|-----------|------|----------|
| ProviderConfig.encrypt_api_key | src/database/models/provider_config.py | P0 |
| ProviderConfig.set_api_key | src/database/models/provider_config.py | P0 |
| ProviderRouter.primary | src/agents/providers/router.py | P0 |
| ProviderRouter.fallback | src/agents/providers/router.py | P0 |
| ProviderRouter.execute_with_fallback | src/agents/providers/router.py | P0 |

---

## Notes

- **Backend Focus:** Epic 2 is primarily backend-focused (Python/FastAPI)
- **E2E Tests:** E2E tests for ProvidersPanel UI are covered in Story 2-5
- **YOLO Mode:** These tests were generated in fully autonomous mode
- **TDD Phase:** All tests marked with `pytest.mark.skip` for TDD red phase

---

**Generated by:** BMad TEA Agent - Test Architect Module
**Workflow:** `_bmad/tea/testarch/atdd`
**Version:** Epic 2 ATDD Run (2026-03-21)
