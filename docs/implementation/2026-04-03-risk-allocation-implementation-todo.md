# Risk Allocation Implementation Todo

Date: 2026-04-03
Branch: `feat/risk-allocation-batch-1-authority`
Status: backend/python implementation and repo-wide blocker cleanup complete; remaining work is verification/integration follow-up

## Completed

- [x] Batch 1 authority cleanup
- [x] Batch 2 session/window and compatibility cleanup
- [x] Batch 3 pressure-state and lock-state model
- [x] Batch 4 broker profile truth vs connection telemetry split
- [x] Batch 5 runtime risk config normalization and settings overlay
- [x] Batch 6 UI lock-state hydration and manual market lock actions
- [x] Batch 7 journal review summary surface
- [x] Batch 8 cleanup/delete pass on duplicate broker handler code
- [x] Targeted backend verification ring (`135 passed`)
- [x] Add real repo modules for `trade_record` and `decline_recovery`
- [x] Repair API startup import surface for missing endpoint routers
- [x] Remove obsolete test-side import stubs after real modules existed
- [x] Consolidated regression ring (`209 passed`)
- [x] Fixed review findings: manual market-lock resume contract, lock-state normalization, overlap session check semantics, zero-balance legacy limit handling
- [x] Added duplicate-router include guard for `race_router` alias case
- [x] Fixed backend NewsBlackoutService startup wiring (`server.py` imports websocket `manager`), added `ws_manager` alias compatibility in `websocket_endpoints.py`

## Remaining

- [ ] Install frontend dependencies under `quantmind-ide/`
- [ ] Run `npm run test:run -- src/lib/stores/kill-switch.test.ts`
- [ ] Run `npm run check` for Svelte/TypeScript validation
- [ ] If backend is started, verify manual market lock and hard-lock banners with browser tools
- [ ] Run manual browser flow for `/api/kill-switch/market-lock` and `/api/kill-switch/market-lock/resume` once frontend deps are installed
- [ ] Install `finnhub-python` on runtime node to enable live calendar fetch (service now starts but logs fetch error when package is missing)
- [ ] Decide whether compatibility API routers added for missing server imports should remain temporary shims or be replaced by full implementations

## Review focus

- [ ] Confirm the broker boundary is acceptable:
  - broker profile truth in `src/router/broker_registry.py`
  - connection telemetry in `src/api/broker_endpoints.py`
- [ ] Confirm backend settings should remain the runtime truth for kill-switch tier-3 thresholds
- [ ] Confirm the new journal review summary shape is sufficient for weekly/agentic review workflows
- [ ] Confirm no additional legacy broker, server-import, or risk-policy owners need deletion before merge

## Rules

- Delete obsolete code after caller migration and verification
- Do not comment out dead code
- Keep runtime truth on backend/trading-node paths, not the frontend
- Record verification evidence before claiming completion
