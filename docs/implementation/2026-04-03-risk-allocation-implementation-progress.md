# Risk Allocation Implementation Progress

Date: 2026-04-03
Status: implementation complete on targeted backend/python ring, plus repo-wide import/startup blocker cleanup
Current scope: Batches 1-8 implemented in worktree, with backend/python regressions verified and frontend verification partially blocked by missing local dependencies

## Post-review fixes (2026-04-04)

- [x] Fixed manual market-lock resume contract mismatch (UI now sends `admin_key` as query parameter)
- [x] Normalized canonical backend `lock_state` shape in UI store hydration (`source` -> `lock_source`, inferred `hard_lock_active`)
- [x] Restored overlap-aware `/api/sessions/check/{session}` semantics via `SessionDetector.is_in_session(...)`
- [x] Preserved explicit `0.0` account balance in Commander legacy dynamic limit status
- [x] Guarded duplicate `race_router` registration when it aliases `ab_race_router`
- [x] Fixed backend startup wiring for `NewsBlackoutService` by importing WebSocket `manager` (plus `ws_manager` compatibility alias)

## Active worktree

- Branch: `feat/risk-allocation-batch-1-authority`
- Purpose: isolate risk-allocation authority, runtime-truth, and lock-model cleanup from unrelated dirty work on `main`

## Execution status

### Batch 1: Authority Map And Runtime Contracts
- [x] Create isolated worktree
- [x] Run pre-edit caller scan
- [x] Classify governor ownership and live callers
- [x] Write/update authority tests
- [x] Implement `engine.py` single-authority selection
- [x] Implement `commander.py` explicit compatibility mode
- [x] Verify targeted router ring

### Batch 2: Session And Queue Authority
- [x] Isolate legacy dynamic bot-limit selection behind Commander compatibility helpers
- [x] Expose legacy limit source in Commander/API status surfaces
- [x] Repair `dynamic_bot_limits` compatibility contract for existing callers/tests
- [x] Introduce canonical session snapshot helper on `session_detector.py`
- [x] Move runtime/API callers to `session_detector.py` import surface where safe
- [x] Fix `OVERLAP` transition logic in `sessions.py`
- [x] Shift `/api/router/bot-limits/status` toward Commander-owned compatibility state

### Batch 3: Drawdown, Pressure, And Lock Model
- [x] Add pressure-state and hard-lock semantics to `account_monitor.py`
- [x] Add sticky manual market lock and canonical lock-state payloads to `progressive_kill_switch.py`
- [x] Expose manual market lock controls and `lock_state` via `kill_switch_endpoints.py`
- [x] Verify targeted progressive-kill-switch ring

### Batch 4: Broker Adaptation And MT5 Boundary
- [x] Introduce canonical broker execution profile in `src/router/broker_registry.py`
- [x] Route `EnhancedGovernor` broker pip-value lookup through broker profile truth instead of direct ad hoc DB logic
- [x] Separate broker connection telemetry (`src/api/broker_endpoints.py`) from broker profile truth (`src/router/broker_registry.py`)
- [x] Rename API-side in-memory connection owner to `broker_connections`
- [x] Delete duplicate broker handler definitions from `src/api/trading_endpoints.py`
- [x] Add regression tests for broker profile resolution and config overlay

### Batch 5: Runtime Config And Backend Truth
- [x] Extend `RiskSettings` with runtime-risk fields for kill-switch thresholds and adaptive daily-budget configuration
- [x] Add normalized `RuntimeRiskConfig` and `load_runtime_risk_config()` to `src/api/settings_endpoints.py`
- [x] Add `/api/settings/risk/runtime` endpoint for backend/trading-node consumers
- [x] Overlay progressive kill-switch thresholds from backend settings truth instead of YAML-only truth

### Batch 6: UI Operationalization And Browser-Facing State
- [x] Extend `quantmind-ide/src/lib/stores/kill-switch.ts` with `lock_state` hydration and manual market lock actions
- [x] Surface manual market lock and hard account lock banners in `TradingFloorPanel.svelte`
- [x] Add store tests for lock-state hydration and manual market-lock API calls
- [ ] Run frontend test/check/build locally

### Batch 7: Journal And Review Loop
- [x] Add `build_review_summary()` to `src/api/journal_endpoints.py`
- [x] Add `/api/journal/review-summary` endpoint for weekly/agentic bot review context
- [x] Add targeted journal summary tests

### Batch 8: Cleanup And Delete Pass
- [x] Delete duplicate broker handler implementations from `src/api/trading_endpoints.py`
- [x] Remove obsolete API-side `broker_registry` connection-owner name in favor of `broker_connections`
- [x] Keep compatibility only where still required by live callers
- [x] Re-run targeted backend verification after cleanup

## Key implementation decisions locked in

- Live router authority remains `src/router/enhanced_governor.py`, selected by `src/router/engine.py`
- `Commander` owns temporary legacy funded-breadth compatibility behavior until the adaptive allocator replaces it
- `src/router/sessions.py` remains canonical session/window logic; `src/router/session_detector.py` is the stable import surface
- `src/router/account_monitor.py` and `src/router/progressive_kill_switch.py` now expose canonical pressure/lock state for backend and UI consumption
- Broker profile truth lives in `src/router/broker_registry.py`
- Broker connection telemetry lives in `src/api/broker_endpoints.py`
- Runtime kill-switch thresholds now overlay from backend settings truth via `load_runtime_risk_config()`
- Frontend remains display/operator-control only; it hydrates backend lock state and does not invent allocator truth

## Verification that passed

### Python compile
- `python3 -m compileall src/router/broker_registry.py src/router/enhanced_governor.py src/api/broker_endpoints.py src/api/ide_handlers_broker.py src/api/trading_endpoints.py src/api/settings_endpoints.py src/router/progressive_kill_switch.py src/api/journal_endpoints.py tests/router/test_batch4_runtime_truth.py tests/api/test_journal_review_summary.py`

### Python regression ring
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_asyncio.plugin tests/router/test_batch4_runtime_truth.py tests/api/test_journal_review_summary.py tests/router/test_progressive_kill_switch.py tests/router/test_batch1_authority_selection.py tests/router/test_batch2_legacy_limit_status.py tests/router/test_dynamic_bot_limits.py tests/api/test_router_endpoints_batch2.py tests/router/test_session_detector.py tests/router/test_session_detector_integration.py tests/router/test_sessions.py -q`
- Result: `135 passed`

### Repo-wide blocker cleanup
- Added real missing modules:
  - `src/database/models/trade_record.py`
  - `src/router/decline_recovery.py`
- Restored broken import contracts:
  - `create_mail_service()` compatibility factory in `src/agents/departments/department_mail.py`
  - `ensemble_router` compatibility surface in `src/api/hmm_inference_server.py`
  - module-level session compatibility surface in `src/router/market_scanner.py`
- Added API startup compatibility routers for missing modules imported by `src/api/server.py`:
  - `src/api/flowforge_workflow_proxy.py`
  - `src/api/cooldown_endpoints.py`
  - `src/api/copilot_endpoints.py`
  - `src/api/dead_zone_endpoints.py`
  - `src/api/dpr_endpoints.py`
  - `src/api/race_endpoints.py`
  - `src/api/session_kelly_endpoints.py`
  - `src/api/ssl_endpoints.py`
  - `src/api/svss_endpoints.py`
  - `src/api/trading_results_endpoints.py`
  - `src/api/trading_session_endpoints.py`
  - `src/api/trading_session_risk_endpoints.py`
  - `src/api/trading_tilt_endpoints.py`
  - `src/api/weekend_cycle_endpoints.py`
  - `src/api/workflow_templates_endpoints.py`
- Hardened `MarketScanner` so HOT-tier DB initialization falls back cleanly when PostgreSQL drivers are unavailable
- Removed obsolete test-side import stubs now that real modules exist

### Backend startup smoke
- `python3 - <<'PY' ... importlib.import_module(\"src.api.server\") ... PY`
- Result: `IMPORT_OK`

### Consolidated regression ring
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_asyncio.plugin tests/router/test_batch1_authority_selection.py tests/router/test_batch2_legacy_limit_status.py tests/router/test_dynamic_bot_limits.py tests/api/test_router_endpoints_batch2.py tests/router/test_session_detector.py tests/router/test_session_detector_integration.py tests/router/test_sessions.py tests/router/test_batch4_runtime_truth.py tests/api/test_journal_review_summary.py tests/router/test_progressive_kill_switch.py tests/router/test_market_scanner.py tests/router/test_multi_symbol_simulation.py tests/api/test_session_endpoints.py -q`
- Result: `209 passed`

## Frontend verification status

- `quantmind-ide/node_modules` is not present in this environment
- attempted command: `cd quantmind-ide && npm run test:run -- src/lib/stores/kill-switch.test.ts`
- result: blocked with `node_modules missing`

## Repo-wide blocker status

- missing import blockers for `src.database.models.trade_record` and `src.router.decline_recovery` are cleared
- API startup no longer fails on missing endpoint modules referenced by `src/api/server.py`
- broader backend/python blocker status is now reduced to optional environment dependencies, not missing repo files

## Files added in this phase

- `tests/router/test_batch4_runtime_truth.py`
- `tests/api/test_journal_review_summary.py`
- `src/database/models/trade_record.py`
- `src/router/decline_recovery.py`
- `src/api/flowforge_workflow_proxy.py`
- `src/api/cooldown_endpoints.py`
- `src/api/copilot_endpoints.py`
- `src/api/dead_zone_endpoints.py`
- `src/api/dpr_endpoints.py`
- `src/api/race_endpoints.py`
- `src/api/session_kelly_endpoints.py`
- `src/api/ssl_endpoints.py`
- `src/api/svss_endpoints.py`
- `src/api/trading_results_endpoints.py`
- `src/api/trading_session_endpoints.py`
- `src/api/trading_session_risk_endpoints.py`
- `src/api/trading_tilt_endpoints.py`
- `src/api/weekend_cycle_endpoints.py`
- `src/api/workflow_templates_endpoints.py`

## Files changed in this phase

- `src/router/broker_registry.py`
- `src/router/enhanced_governor.py`
- `src/api/broker_endpoints.py`
- `src/api/ide_handlers_broker.py`
- `src/api/trading_endpoints.py`
- `src/api/settings_endpoints.py`
- `src/router/progressive_kill_switch.py`
- `src/api/journal_endpoints.py`
- `src/agents/departments/department_mail.py`
- `src/router/market_scanner.py`
- `src/api/hmm_inference_server.py`
- `src/api/session_endpoints.py`
- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- `quantmind-ide/src/lib/stores/kill-switch.test.ts`
- `tests/router/test_market_scanner.py`
- `tests/api/test_session_endpoints.py`
- `tests/router/test_progressive_kill_switch.py`
- `tests/router/test_batch1_authority_selection.py`
- `tests/router/test_batch4_runtime_truth.py`

## Remaining follow-up outside this batch set

- install frontend dependencies and run the targeted vitest/svelte-check ring
- optionally exercise the UI against a running backend with Chrome devtools
- decide whether the compatibility endpoint routers should be replaced by real implementations or kept until the missing stories land
