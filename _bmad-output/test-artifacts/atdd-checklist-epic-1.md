---
stepsCompleted: ['step-01-preflight-and-context']
lastStep: 'step-01-preflight-and-context'
lastSaved: '2026-03-21'
mode: epic-level
epic_num: 1
epic_title: "Platform Foundation & Global Shell"
story_id: epic-1
---

# ATDD Checklist: Epic 1 - Platform Foundation & Global Shell

**Date:** 2026-03-21
**Author:** TEA Agent (YOLO Mode)
**Mode:** Epic-Level ATDD

---

## Story Context

**Epic 1** comprises 7 stories covering:
- Story 1-0: Platform Codebase Exploration & Audit
- Story 1-1: Security Hardening & Legacy Import Cleanup
- Story 1-2: Svelte 5 Migration
- Story 1-3: NODE_ROLE Backend Deployment Split
- Story 1-4: TopBar & ActivityBar Frosted Terminal Aesthetic
- Story 1-5: StatusBand Redesign
- Story 1-6-9: 9-Canvas Routing Skeleton

---

## P0 Test Scenarios (Critical Path)

### 1. Kill Switch Two-Step Confirmation (E2E) - 4 tests

**Risk:** R-005 (Score: 6) - SEC

**Tests:**
- `test_kill_switch_armed_state_shows_pulsing_shield_alert` - Verify ShieldAlert icon pulses when ARMED
- `test_kill_switch_confirm_modal_blocks_enter_key` - Two-step confirmation: Enter key does NOT execute
- `test_kill_switch_confirm_modal_escape_cancels` - Escape key cancels activation
- `test_kill_switch_explicit_button_click_executes` - Only explicit button click executes

**Test Level:** E2E (Playwright)
**Status:** REQUIRES PLAYWRIGHT INFRASTRUCTURE

### 2. NODE_ROLE Router Isolation (API) - 5 tests

**Risk:** R-001 (Score: 4), R-006 (Score: 4)

**Tests:**
- `test_node_role_cloudzy_trading_endpoints_200` - NODE_ROLE=cloudzy: trading endpoints return 200
- `test_node_role_cloudzy_agent_endpoints_404` - NODE_ROLE=cloudzy: agent endpoints return 404
- `test_node_role_contabo_agent_endpoints_200` - NODE_ROLE=contabo: agent endpoints return 200
- `test_node_role_contabo_trading_endpoints_404` - NODE_ROLE=contabo: trading endpoints return 404
- `test_node_role_invalid_defaults_to_local_with_warning` - Invalid NODE_ROLE logs warning

**Test Level:** API (pytest)
**Status:** READY FOR GENERATION

### 3. Build Verification (CI) - 1 test

**Risk:** R-002 (Score: 4)

**Tests:**
- `test_npm_build_passes_without_svelte_warnings` - npm build passes with zero Svelte 4 deprecation warnings

**Test Level:** Build/CI
**Status:** READY FOR GENERATION

---

## Generated Failing Tests

### API Tests (tests/api/test_node_role_routing.py)

| Test | Status | Risk |
|------|--------|------|
| `test_cloudzy_trading_endpoints_available` | PASS | - |
| `test_cloudzy_agent_endpoints_not_available` | PASS | - |
| `test_cloudzy_only_trading_routers_registered` | PASS | R-006 |
| `test_contabo_agent_endpoints_available` | **FAIL** | R-006 |
| `test_contabo_trading_endpoints_not_available` | **FAIL** | R-006 |
| `test_contabo_only_agent_routers_registered` | PASS | R-006 |
| `test_invalid_role_defaults_to_local_with_warning` | PASS | R-001 |
| `test_invalid_role_validation_logic` | PASS | R-001 |
| `test_local_includes_all_routers` | **FAIL** | R-006 |
| `test_local_default_when_not_set` | PASS | - |

**3 FAILING tests** - Router isolation not properly implemented for contabo/local modes

### E2E Tests (tests/e2e/test_kill_switch_confirmation.py)

| Test | Status | Notes |
|------|--------|-------|
| `test_kill_switch_armed_state_shows_pulsing_shield_alert` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_kill_switch_confirm_modal_blocks_enter_key` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_kill_switch_confirm_modal_escape_cancels` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_kill_switch_explicit_button_click_executes` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_modal_has_tier_selection_options` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_modal_has_cancel_and_confirm_buttons` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_enter_key_blocked_when_no_tier_selected` | REQUIRES SETUP | Playwright infrastructure needed |
| `test_escape_key_works_regardless_of_tier_selected` | REQUIRES SETUP | Playwright infrastructure needed |

**REQUIRES PLAYWRIGHT SETUP** - E2E tests cannot run without Playwright infrastructure

### Build Tests (tests/test_build_verification.py)

| Test | Status | Notes |
|------|--------|-------|
| `test_npm_build_passes` | PASS | Story 1-2 complete |
| `test_no_svelte_4_deprecation_warnings` | PASS | No Svelte 4 warnings |
| `test_build_output_no_errors` | PASS | Build succeeds |
| `test_typescript_compilation` | PASS | TS compilation OK |
| `test_legacy_reactive_syntax_detected_but_acceptable` | PASS | Informational only |

**Story 1-2 ALREADY COMPLETE** - Svelte 5 migration verified

---

## TDD Cycle Results

### RED Phase (Failing Tests Confirmed)

**API Tests - 3 Failing:**
```
FAILED test_contabo_agent_endpoints_available - AssertionError: Contabo mode should include agent routers
FAILED test_contabo_trading_endpoints_not_available - AssertionError: Contabo mode should exclude trading routers
FAILED test_local_includes_all_routers - AssertionError: Local mode should include contabo routers
```

**Root Cause:** NODE_ROLE router isolation not properly implemented for contabo/local modes.

### GREEN Phase (Next Steps for DEV)

1. Fix `INCLUDE_CONTABO` to be `True` when `NODE_ROLE=local`
2. Fix `INCLUDE_CLOUDZY` to be `False` when `NODE_ROLE=contabo`
3. Verify trading routers 404 when `NODE_ROLE=contabo`
4. Verify agent routers 404 when `NODE_ROLE=cloudzy`

---

## Test Files Created

1. `/home/mubarkahimself/Desktop/QUANTMINDX/tests/api/test_node_role_routing.py` (10 tests)
2. `/home/mubarkahimself/Desktop/QUANTMINDX/tests/e2e/test_kill_switch_confirmation.py` (8 tests, requires Playwright)
3. `/home/mubarkahimself/Desktop/QUANTMINDX/tests/test_build_verification.py` (5 tests)

---

**Generated by:** BMad TEA Agent - ATDD Workflow
**Workflow:** `_bmad/tea/testarch/atdd`
**Execution Mode:** YOLO (Fully Autonomous)
