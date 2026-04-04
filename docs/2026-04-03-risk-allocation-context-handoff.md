# QUANTMINDX Risk Allocation Context Handoff

Date: 2026-04-03
Purpose: Compact, tracked handoff for future sessions before implementation starts
Status: Canonical session memory artifact

## 1. Why this file exists

The detailed redesign plan and summary spec currently live under:

- `docs/superpowers/plans/2026-04-03-risk-allocation-redesign-implementation-plan.md`
- `docs/superpowers/specs/2026-04-03-risk-allocation-redesign-design.md`

Those are useful, but the `docs/superpowers/plans/` path is gitignored in this repo. This file exists so the core context survives outside ignored paths and can be used after compaction or in future implementation sessions.

## 2. Core architectural decisions already locked

### 2.1 What this subsystem is

The risk/position-sizing subsystem is a capital allocation and protection layer for already-validated EAs. It is not the primary signal-validation engine.

Validated EAs own strategy logic.
The risk system owns:

- funding eligibility
- ranked queue allocation
- funded breadth
- concurrent open-risk slots
- drawdown protection
- lock states
- broker-feasibility checks
- fee/spread-aware sizing

### 2.2 Runtime hierarchy

The runtime hierarchy is locked as:

1. `Variant Universe`
2. `Deployable Pool`
3. `Session-Eligible Queue`
4. `Ranked Session Queue`
5. `Funded Session Bots`
6. `Concurrent Open-Risk Slots`
7. `Open Positions`

Critical distinction:

- funded session bots are not the same thing as open-risk slots
- a large bot universe does not imply large simultaneous account exposure

### 2.3 Scaling rule

As equity grows:

- funded breadth grows fastest
- concurrent slots grow more slowly
- per-slot risk grows slowest

Growth should come mainly from:

- better selection
- more funded bots
- more recycled slots
- more session-aware deployment
- compounding

Not mainly from making each trade much larger.

## 3. The major correction made in this session

The earlier designs and docs blurred three different things:

- candidate bot count
- funded runtime participation
- actual open account risk

This caused the arithmetic drift.

The corrected model is:

- many variants can exist
- a smaller subset becomes session-eligible
- a smaller subset becomes funded for the session
- an even smaller subset consumes concurrent open-risk slots

## 4. Sessions and families

### 4.1 Sessions must stay first-class

The redesign must account for the full multi-session model, not just one simplified active window:

- Sydney
- Sydney-Tokyo overlap
- Tokyo
- Tokyo-London overlap
- London Open
- London Mid
- inter-session / cooldown
- London-NY overlap
- NY Wind Down
- dead/reset windows

Premium sessions remain important and should affect deployment permission, breadth, and ranking preference.

### 4.2 Families in scope

The main families in scope are:

- scalping
- ORB

Scalping and ORB share one allocation framework but not identical parameters.

Scalping should generally have:

- more funded bots
- more slot recycling
- lower per-slot risk
- stricter spread/fee gating

ORB should generally have:

- fewer funded bots
- fewer slots
- slightly more room per trade
- slower breaker escalation

## 5. Stops, pressure, and drawdown

### 5.1 Flat tiny-account daily stop is not enough

A naive flat `3%` daily stop is too coarse for tiny accounts.

The corrected model is:

- an adaptive operating budget
- a global hard account lock
- drawdown pressure bands before the hard lock

This allows the system to breathe while still preserving the account.

### 5.2 Session pressure is valid, but not sovereign

Session-based tightening is valid and should modulate:

- funded breadth
- open slots
- family permissions
- pressure state

But session logic must never override the global hard account lock.

### 5.3 Loss-pressure / tightening model

We moved away from vague “soft bleed” language.

The better model is explicit pressure states such as:

- `NORMAL`
- `CAUTION`
- `RESTRICTED`
- `STOPPED`

These should be driven by:

- drawdown pressure
- session conditions
- market conditions
- lock states

## 6. SL/TP and sizing relationship

The planning docs clarified the 3-layer exit structure:

- `Layer 1`: broker-resident hard safety SL/TP
- `Layer 2`: dynamic position modification
- `Layer 3`: system-level forced protection / kill behavior

The key design decision locked here:

- position sizing must use worst-case protective loss basis
- dynamic exit logic still manages the trade after entry

So sizing is based on:

- stop-distance loss
- spread cost
- round-trip commission
- slippage buffer
- lot feasibility

Journaled realized outcomes then feed later Kelly/DPR refresh.

## 7. Broker adaptation

Broker adaptation already partially exists in the codebase and should be extended, not reinvented.

Relevant surfaces found in the scan:

- `config/brokers.yaml`
- `src/router/broker_registry.py`
- `src/position_sizing/enhanced_kelly.py`
- `src/position_sizing/edge_cases.py`
- `quantmind-ide/src/lib/components/BrokerManagement.svelte`

Correct architecture:

- core allocator remains broker-agnostic
- broker adapter supplies feasibility and execution constraints
- MT5 remains the execution bridge

Broker-adaptation inputs should include:

- min lot
- lot step
- symbol specs
- pip value behavior
- commission model
- spread behavior
- execution mode

## 8. MT5 and execution boundary

Locked understanding:

1. QUANTMINDX decides:
- whether capital may be allocated
- how much risk may be used
- the lot size
- stop/modify instructions

2. MT5 handles:
- account session
- broker execution channel
- symbol/account values
- order send / modify / close

3. Broker applies:
- actual spread
- commission
- fills
- account updates

MQL must become a validation/execution-safety layer, not a hidden second policy owner.

## 9. Locks and kill-switch semantics

The following states must remain distinct:

- `News Lock`
- `Market Lock`
- `Loss Lock`
- `Kill Switch`

Manual market lock must be sticky until explicit resume.

Copilot kill switch remains separate from live trading kill behavior.

Kill switch should be directional and intentional:

- monitoring continues
- no new funding when appropriate
- position monitor manages staged exits/tightening
- operator can always see why trading is stopped

## 10. Journal and review loop

The journal is not just for display.
It is part of the improvement loop.

It must capture enough runtime context to support:

- later funding-score refresh
- weekly review
- bot failure diagnosis
- agentic improvement workflows

Important principle:

- Kelly refresh is periodic and off hot path
- not per tick
- not per trade

Preferred timing:

- end-of-day
- dead/reset windows

## 11. Existing code surfaces confirmed by scan

### 11.1 Risk and allocation path

- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/position_sizing/enhanced_kelly.py`
- `src/position_sizing/kelly_config.py`
- `src/position_sizing/edge_cases.py`

### 11.2 Session / queue / ranking

- `src/router/sessions.py`
- `src/router/dpr_scoring_engine.py`
- `src/router/queue_reranker.py`
- `src/router/queue_remix.py`
- `src/router/session_performer.py`
- `src/router/inter_session_cooldown.py`
- `src/router/dead_zone_scheduler.py`

### 11.3 Protection and locks

- `src/router/progressive_kill_switch.py`
- `src/router/kill_switch.py`
- `src/router/account_monitor.py`
- `src/router/bot_circuit_breaker.py`
- `src/router/position_monitor.py`
- `src/router/position_monitor_engine.py`

### 11.4 Sentinel and news

- `src/router/sentinel.py`
- `src/market/news_blackout.py`
- `src/api/economic_calendar_endpoints.py`
- `quantmind-ide/src/api/economic_calendar_endpoints.py`

### 11.5 UI / IDE surfaces

- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/stores/risk.ts`
- `quantmind-ide/src/lib/components/KillSwitchView.svelte`
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
- `quantmind-ide/src/lib/components/SettingsView.svelte`
- `quantmind-ide/src/lib/components/BrokerManagement.svelte`
- `quantmind-ide/src/lib/components/NewsView.svelte`
- `quantmind-ide/src/lib/components/MarketOverview.svelte`

### 11.6 Settings / deployment config

- `src/api/settings_endpoints.py`
- `config/settings/settings.json`
- `config/brokers.yaml`
- `config/trading_system.yaml`

## 12. Fresh scan conclusions

### 12.1 Multiple runtime policy owners already exist

The allocator/risk path is currently split across multiple files that can all shape runtime behavior:

- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/router/prop/governor.py`

This is the main implementation hazard. The redesign cannot succeed if these remain parallel policy owners after migration.

### 12.2 Sessions already have both canonical and legacy shapes in one file

`src/router/sessions.py` already contains useful modern structures:

- `CANONICAL_WINDOWS`
- `SESSION_BOT_MIX`
- `PREMIUM_SESSIONS`
- `TRADING_WINDOWS`

But it also still contains legacy `SESSIONS` / `SessionTemplate` structures with static `max_concurrent_bots`.

Implication:

- do not invent a third session authority
- promote one shape
- migrate callers
- delete the losing branch

### 12.3 Legacy static slot logic still exists and conflicts with the redesign

`src/router/dynamic_bot_limits.py` is explicitly tier-based and static:

- hardcoded account tiers
- hardcoded bot caps
- hardcoded 3% total risk split across bots

This directly conflicts with the adaptive funded-breadth/open-slot model discussed in this session.

Current likely live callers to audit first:

- `src/router/commander.py`
- `src/router/market_scanner.py`

### 12.4 Current stop/kill logic still embeds old flat thresholds

`src/router/account_monitor.py` still centers around flat daily/weekly percent stops.

`src/router/progressive_kill_switch.py` still embeds:

- 3/5/7 style account thresholds
- session-level trades-per-minute assumptions
- prop-firm-oriented kill language

This means the new adaptive daily operating budget and hard-lock model is not yet the runtime truth.

### 12.5 Broker handling is duplicated in two different roles

There are two distinct broker surfaces today:

1. `src/router/broker_registry.py`
- persistent broker fee/profile registry
- sizing-oriented

2. `src/api/broker_endpoints.py`
- in-memory broker/account discovery and active-account switching
- transport/ops-oriented

These are not the same responsibility, but they currently overlap in naming and operator expectations.

The redesign should keep both concepts only if their boundary is explicit:

- broker profile / feasibility registry
- live connected account registry

### 12.6 Settings are still carrying legacy prop-firm and tier assumptions

`src/api/settings_endpoints.py` exposes `RiskSettings` fields such as:

- `dailyLossLimit`
- `maxDrawdown`
- `balanceZones`
- `propFirmPreset`
- `hardStopBuffer`

Those are strong indicators that old prop-firm and static-tier assumptions still leak into runtime configuration.

This settings surface must be audited before new allocator truth is attached to it.

### 12.7 UI surfaces exist, but they do not yet expose allocator truth

Current UI state is fragmented:

- `quantmind-ide/src/lib/stores/kill-switch.ts`
  - mostly emergency/news state
  - still uses direct `fetch('/api/kill-switch/...')` calls
- `quantmind-ide/src/lib/stores/risk.ts`
  - mostly physics/compliance telemetry
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
  - currently shows news blackout banners, not allocator state
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
  - shows active bots/PnL/session clocks, but not funded breadth, pressure state, or lock-source clarity

So the UI work is not greenfield.
It is mostly alignment and exposure of runtime truth that backend does not yet publish cleanly.

## 13. Deployment and node topology

### 13.1 Current repo reality

The repo already reflects a distributed deployment model, but the responsibilities are not yet cleanly written down as one runtime contract.

Observed surfaces:

- `docs/deployment-guide.md`
  - documents a two-VPS split:
    - Contabo: training / regime API
    - Cloudzy: live API server / trading execution
- `src/data/brokers/mt5_socket_adapter.py`
  - Linux-side MT5 socket bridge client
- `src/router/socket_server.py`
  - low-latency ZMQ request/reply server
- `src/svss/ticker.py`
  - MT5 ZMQ tick subscription path
- `src/monitoring/cold_storage_sync.py`
  - cold-storage persistence/sync path
- `src/monitoring/tracing.py`
  - already carries `NODE_ROLE`

### 13.2 Target logical-node model for this redesign

For this redesign, the safest architecture to preserve is three logical nodes:

1. `Trading Node`
- queue evaluation
- funded-breadth and open-slot decisions
- live drawdown / pressure / lock enforcement
- session-state hot path
- broker feasibility checks
- MT5 bridge / ZMQ / execution handoff

2. `Backend Node`
- authoritative runtime config APIs
- journal ingestion and read models
- weekly review aggregation
- slower recalibration / analytics work
- UI-facing operational state

3. `Storage / Cold Node`
- archived logs
- persisted journals
- historical artifacts
- cold storage sync

MetaTrader / Windows bridge remains an execution boundary attached to the trading node.

### 13.3 Node responsibility rule

The redesign must preserve this split:

- hot path belongs on the trading node
- operator/config/review surfaces belong on the backend node
- archives and long-horizon persistence belong on cold/storage paths

Do not let IDE-local state become deployment truth.
Do not move latency-sensitive trading decisions onto backend-only slow paths.

### 13.4 Node connection shape

Expected connection shape:

- frontend UI -> backend APIs / websocket surfaces
- backend node -> trading node control/state surfaces
- trading node -> MT5 socket bridge / broker execution
- backend node + trading node -> shared persisted state
- hot logs / summaries -> cold storage sync

Any direct frontend-to-trading shortcut should be treated as suspicious unless the deployment reason is explicit.

## 14. Important cleanup rule

Deprecated or superseded code must not be left around as shadow logic.

Required behavior in future coding sessions:

- identify deprecated branches before editing nearby code
- confirm whether they still have live callers
- migrate remaining callers
- delete the deprecated branch once replacement is verified

Do not leave two policy owners alive if one is marked deprecated.

## 15. Code review expectations

### 15.1 What review should focus on

Actual code review for this redesign should default to runtime-safety review, not style review.

Primary findings to hunt:

- duplicate runtime policy owners still alive
- legacy static tier logic still reachable from live paths
- session logic bypassing global hard lock
- backend/runtime truth still leaking from local settings
- UI showing fabricated state instead of backend state
- trading-node hot path depending on backend-only slow work
- broker-specific assumptions hardcoded outside broker adaptation
- deprecated branches left commented out instead of removed

### 15.2 What counts as a credible batch

A batch is only credible if review can confirm:

- one authority owns the changed behavior
- replaced callers are known and migrated or explicitly deferred
- operator-facing state explains why trading is reduced or blocked
- deployment behavior is preserved outside local dev
- no local shortcut became production truth

### 15.3 Review evidence expected

Implementation review should leave evidence of:

- files that are now authoritative
- files that are wrappers only
- files that became delete candidates
- runtime/API payloads checked
- UI/runtime behavior actually verified

## 16. UI verification expectations

### 16.1 Browser verification is required

UI work for this redesign should be checked in a running app, not only by reading Svelte code.

Preferred path:

- run the IDE in development or deployment-like mode
- inspect behavior with Chrome devtools / browser devtools
- verify network calls, websocket updates, and rendered state

### 16.2 Components that must be checked when touched

- `quantmind-ide/src/lib/components/KillSwitchView.svelte`
- `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
- `quantmind-ide/src/lib/components/SettingsView.svelte`
- `quantmind-ide/src/lib/components/BrokerManagement.svelte`
- `quantmind-ide/src/lib/components/NewsView.svelte`
- `quantmind-ide/src/lib/components/MarketOverview.svelte`
- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/stores/risk.ts`

### 16.3 What to verify in devtools

- correct API route usage
- no stale local-only/localhost-only transport assumptions
- websocket messages update rendered UI state
- lock source and stop reason are visible
- funded breadth / pressure / session status are backend-driven, not fabricated in frontend
- sticky manual market lock survives refresh / reconnect if backend still reports it
- settings edits round-trip through backend persistence and come back as runtime truth

## 17. Current known risks

These are the main design hazards still visible in the codebase:

- multiple governor/risk-policy owners
- session duplication/conflict
- duplicate broker-registry concepts
- prop-firm remnants in config and MQL
- static dynamic-bot-limit tier logic still referenced in router paths
- local/settings-level risk truth instead of deployment/runtime truth
- fail-open SQS paths
- kill/lock states not yet unified in API/UI
- duplicate position-monitor ownership risk

## 18. What is still genuinely open

These are still open because they require calibration, not because the architecture is unresolved:

- exact equity-band numbers
- exact session/family budget percentages
- exact news-lock scope mapping
- exact catastrophic fail-safe stop distance

## 19. Immediate next steps after this handoff

The next working order should be:

1. use this handoff doc as the compact memory artifact
2. use the detailed redesign plan as the full decomposition reference, but prefer tracked docs for handoff
3. read the tracked batch decomposition:
   - `docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md`
4. read the Batch 1 authority plan:
   - `docs/implementation/2026-04-03-risk-allocation-batch-1-authority-plan.md`
5. read the execution checklists in order:
   - `docs/implementation/2026-04-03-risk-allocation-batch-1-execution-checklist.md`
   - `docs/implementation/2026-04-03-risk-allocation-batch-2-execution-checklist.md`
   - `docs/implementation/2026-04-03-risk-allocation-batch-3-execution-checklist.md`
6. audit the competing runtime authorities first:
   - governor family
   - sessions
   - account/kill lock path
   - broker registry path
7. only after authority ownership is decided should coding begin
8. delete deprecated branches after migration verification, not before and not by commenting them out

## 20. Files to read next

If a future agent resumes this work, it should read these first:

- `docs/2026-04-03-risk-allocation-context-handoff.md`
- `docs/2026-04-03-risk-allocation-todo.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-1-authority-plan.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-1-execution-checklist.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-2-execution-checklist.md`
- `docs/implementation/2026-04-03-risk-allocation-batch-3-execution-checklist.md`
- `docs/superpowers/specs/2026-04-03-risk-allocation-redesign-design.md`
- `docs/superpowers/plans/2026-04-03-risk-allocation-redesign-implementation-plan.md`
- `src/router/enhanced_governor.py`
- `src/router/governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/router/prop/governor.py`
- `src/router/sessions.py`
- `src/router/dynamic_bot_limits.py`
- `src/router/progressive_kill_switch.py`
- `src/router/bot_circuit_breaker.py`
- `src/router/account_monitor.py`
- `src/router/position_monitor.py`
- `src/router/broker_registry.py`
- `src/api/broker_endpoints.py`
- `src/api/settings_endpoints.py`
- `src/api/trading_session_risk_endpoints.py`
- `src/api/kill_switch_endpoints.py`
- `src/data/brokers/mt5_socket_adapter.py`
- `src/router/socket_server.py`
- `quantmind-ide/src/lib/stores/kill-switch.ts`
- `quantmind-ide/src/lib/stores/risk.ts`
- `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
