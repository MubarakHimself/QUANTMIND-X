# Risk Allocation Batch Decomposition

Date: 2026-04-03
Status: pre-implementation decomposition
Purpose: break the redesign into executable, reviewable batches with file ownership, node boundaries, and verification expectations

## 1. Working rule

Implementation should not start from feature slices.
It should start from authority cleanup.

Why:
- allocator logic is already duplicated
- session truth is already duplicated
- lock logic is already duplicated
- broker/runtime config truth is already split across backend and UI

If we implement UI or sizing changes before authority ownership is fixed, the repo will drift further.

## 2. Logical ownership model

### 2.1 Trading node

Owns:
- funded-breadth decision
- concurrent open-slot decision
- pre-trade feasibility checks
- hot-path drawdown and pressure enforcement
- live lock enforcement
- position modification / staged exits
- broker execution handoff

Primary files in scope:
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/router/sessions.py`
- `src/router/account_monitor.py`
- `src/router/progressive_kill_switch.py`
- `src/router/kill_switch.py`
- `src/router/bot_circuit_breaker.py`
- `src/router/position_monitor.py`
- `src/router/position_monitor_engine.py`
- `src/router/commander.py`
- `src/router/engine.py`
- `src/router/interface.py`
- `src/data/brokers/mt5_socket_adapter.py`
- `src/router/socket_server.py`

### 2.2 Backend node

Owns:
- runtime config APIs
- journal ingestion and query surfaces
- review and reporting artifacts
- UI-facing state contracts
- deployment-level persistence of operator settings

Primary files in scope:
- `src/api/settings_endpoints.py`
- `src/api/kill_switch_endpoints.py`
- `src/api/journal_endpoints.py`
- `src/api/trading_session_risk_endpoints.py`
- `src/api/broker_endpoints.py`
- `src/api/economic_calendar_endpoints.py`
- `src/router/broker_registry.py`
- `config/settings/settings.json`
- `config/brokers.yaml`
- `config/trading_system.yaml`

### 2.3 Storage / cold paths

Owns:
- archived journals
- historical review artifacts
- cold storage sync
- long-horizon persistence only

Relevant surfaces:
- `src/monitoring/cold_storage_sync.py`
- database models/repositories touched by journal and broker persistence

### 2.4 Frontend

Owns display and operator interaction only.
It must not become allocator truth.

Primary files in scope:
- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/stores/risk.ts`
- `quantmind-ide/src/lib/components/KillSwitchView.svelte`
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
- `quantmind-ide/src/lib/components/SettingsView.svelte`
- `quantmind-ide/src/lib/components/BrokerManagement.svelte`
- `quantmind-ide/src/lib/components/NewsView.svelte`
- `quantmind-ide/src/lib/components/MarketOverview.svelte`

## 3. Batch sequence

## Batch 1: Authority Map And Runtime Contracts

### Goal
Select the single runtime authority for allocation and define the canonical state and API contracts before any behavior change.

### Why first
Current risk behavior is split across:
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/router/prop/governor.py`

No later batch is safe until ownership is explicit.

### Files to inspect first
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/router/prop/governor.py`
- `src/router/engine.py`
- `src/router/commander.py`
- `src/router/interface.py`

### Expected decisions
- which file becomes sole runtime allocation authority
- which files become wrappers/adapters only
- which files become delete candidates
- canonical runtime state shape for:
  - funded breadth
  - open slots
  - pressure state
  - lock source
  - feasibility vetoes

### Deliverables
- one ownership table
- one runtime state contract
- one backend payload contract for UI consumption
- delete-candidate list with live caller notes

### Review expectations
Review must confirm:
- exactly one authority owns allocation behavior
- every remaining owner is either wrapper or scheduled for deletion
- no prop-firm-only logic remains the hidden source of truth

## Batch 2: Session And Queue Authority

### Goal
Collapse session truth into one model and align queue/ranking paths with the funded-breadth and open-slot architecture.

### Files
- `src/router/sessions.py`
- `src/router/dpr_scoring_engine.py`
- `src/router/queue_reranker.py`
- `src/router/queue_remix.py`
- `src/router/session_performer.py`
- `src/router/inter_session_cooldown.py`
- `src/router/dead_zone_scheduler.py`
- `src/router/commander.py`
- `src/router/dynamic_bot_limits.py`

### Current conflict
`sessions.py` already has:
- `CANONICAL_WINDOWS`
- `SESSION_BOT_MIX`
- `PREMIUM_SESSIONS`

but also legacy `SESSIONS` / `SessionTemplate` with static concurrency.

`dynamic_bot_limits.py` hardcodes tier caps that conflict with adaptive breadth.

### Deliverables
- one canonical session/window authority
- one premium-session definition path
- one queue-stage contract:
  - eligible
  - ranked
  - funded
  - open-slot eligible
- explicit retirement plan for `dynamic_bot_limits.py`

### Review expectations
Review must confirm:
- no third session authority was introduced
- premium sessions are not duplicated across files
- correlation pressure acts before funding, not only after fills

## Batch 3: Drawdown, Pressure, And Lock Model

### Goal
Replace flat tiny-account stop assumptions with adaptive operating budget + hard account lock + pressure bands.

### Files
- `src/router/account_monitor.py`
- `src/router/progressive_kill_switch.py`
- `src/router/kill_switch.py`
- `src/router/bot_circuit_breaker.py`
- `src/router/sentinel.py`
- `src/api/kill_switch_endpoints.py`
- `src/api/trading_session_risk_endpoints.py`

### Required outcomes
- global hard account lock remains sovereign
- session pressure can tighten but not bypass hard lock
- sticky manual market lock exists and survives refresh/reconnect via backend truth
- stop reasons are externally visible

### Deliverables
- canonical lock-state model
- adaptive operating budget logic for small accounts
- pressure bands exposed in API payloads
- reason-code model for why the system is reduced or stopped

### Review expectations
Review must confirm:
- session opportunity cannot reopen trading after global hard lock
- manual market lock is sticky until explicit resume
- kill-switch and account-monitor semantics are no longer contradictory

## Batch 4: Broker Adaptation And MT5 Boundary

### Goal
Make broker adaptation explicit while preserving MT5 as execution boundary and keeping hot-path logic on the trading node.

### Files
- `src/router/broker_registry.py`
- `src/api/broker_endpoints.py`
- `src/position_sizing/enhanced_kelly.py`
- `src/position_sizing/edge_cases.py`
- `src/data/brokers/mt5_socket_adapter.py`
- `src/router/socket_server.py`
- `config/brokers.yaml`
- `src/mql5/Include/QuantMind/Risk/KellySizer.mqh`
- `src/mql5/Include/QuantMind/Core/Constants.mqh`
- `src/mql5/Include/QuantMind/Risk/PropManager.mqh`

### Current split
- `src/router/broker_registry.py` is sizing/profile oriented
- `src/api/broker_endpoints.py` is connection/discovery oriented

Both can stay only if boundary is explicit.

### Deliverables
- broker profile contract
- connected-account registry contract
- hot-path feasibility contract
- MQL reduce-to-safety-layer plan

### Review expectations
Review must confirm:
- frontend never talks directly to broker execution surfaces
- backend/trading node transport boundaries are explicit
- broker-specific assumptions are centralized
- MQL is not still acting as shadow risk-policy owner

## Batch 5: Runtime Config And Backend Truth

### Goal
Move runtime truth to deployment-level backend persistence and remove local-only settings as authoritative risk state.

### Files
- `src/api/settings_endpoints.py`
- `config/settings/settings.json`
- `config/trading_system.yaml`
- `src/api/kill_switch_endpoints.py`
- `src/api/journal_endpoints.py`
- database-backed persistence touched by these endpoints

### Current problem
`RiskSettings` still exposes legacy fields like:
- `dailyLossLimit`
- `maxDrawdown`
- `balanceZones`
- `propFirmPreset`

These leak outdated logic into runtime config.

### Deliverables
- deployment-safe runtime config contract
- separation between operator settings and hot-path computed state
- migration plan for legacy risk settings fields

### Review expectations
Review must confirm:
- backend persists truth
- trading node consumes synced truth
- frontend edits through backend only
- local dev defaults are not runtime truth in deployment

## Batch 6: UI Operationalization And Browser Verification

### Goal
Expose allocator truth in UI without fabricating it in stores or components.

### Files
- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/stores/risk.ts`
- `quantmind-ide/src/lib/components/KillSwitchView.svelte`
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
- `quantmind-ide/src/lib/components/SettingsView.svelte`
- `quantmind-ide/src/lib/components/BrokerManagement.svelte`
- `quantmind-ide/src/lib/components/NewsView.svelte`
- `quantmind-ide/src/lib/components/MarketOverview.svelte`

### Required UI outcomes
- show funded breadth
- show open-slot pressure state
- show lock source and reason
- show sticky manual market lock state
- show broker feasibility or fee vetoes where relevant

### Browser/devtools verification required
For each touched component:
- verify route/API used
- verify websocket payloads if applicable
- verify rendered state matches backend payload
- verify no local-only fallback is pretending to be runtime truth
- verify sticky manual market lock after refresh/reconnect

### Review expectations
A UI batch is not done until browser/devtools evidence exists.

## Batch 7: Journal And Review Loop

### Goal
Extend journal payloads so weekly review and agentic diagnosis can use real runtime context.

### Files
- `src/api/journal_endpoints.py`
- related database models/repositories used by journal writes and reads
- any backend aggregation layer introduced for reviews

### Runtime context that should be captured
- funding decision context
- session window
- pressure state at entry
- lock source if affected
- fee/spread veto data if skipped or reduced
- breaker / ejection reason
- broker feasibility details

### Review expectations
Review must confirm:
- journal additions are useful for diagnosis, not vanity fields
- Kelly refresh remains off hot path
- review artifacts can explain repeated failures by session/family/broker context

## Batch 8: Cleanup And Deletion Pass

### Goal
Delete superseded runtime branches only after caller migration is proven.

### Highest-probability delete candidates
- static tier logic in `src/router/dynamic_bot_limits.py`
- prop-firm-specific runtime remnants in hot-path allocator code
- duplicated session authorities in `src/router/sessions.py`
- shadow policy code in non-authoritative governors
- MQL prop-firm remnants that remain policy owners rather than safety rails

### Rule
Do not comment out dead code.
Delete it after migration verification.

### Review expectations
Review must confirm:
- migrated callers are known
- removed files no longer have live references
- deletion reduced policy duplication, not just moved it

## 4. First execution order

Start with only these:

1. Batch 1
2. Batch 2
3. Batch 3

Do not start UI implementation before Batch 1 ownership and Batch 3 lock semantics are settled.
Do not start broker-boundary cleanup before Batch 4 is formally scoped.

## 5. What code review should expect by batch

### For Batch 1-3
- architecture findings first
- line/file references for any surviving duplicate authority
- explicit note if a path still depends on prop-firm assumptions
- explicit note if hard lock can still be bypassed

### For Batch 4-5
- deployment/runtime truth findings first
- explicit node-ownership notes
- explicit note if local settings still leak into live runtime

### For Batch 6
- UI findings first
- browser verification evidence required
- exact component + route + payload mismatch if any

### For Batch 8
- deletion safety findings first
- explicit note of any remaining live callers

## 6. Execution checklist artifacts

The execution checklist artifacts for the first three batches are now:

- `docs/implementation/2026-04-03-risk-allocation-batch-1-execution-checklist.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-2-execution-checklist.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-3-execution-checklist.md`

Each checklist should remain:

- delete-aware
- deployment-aware
- explicit about review evidence
- explicit about UI/browser verification when a batch touches frontend surfaces
