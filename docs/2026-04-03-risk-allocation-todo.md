# QUANTMINDX Risk Allocation Todo List

Date: 2026-04-03
Purpose: decomposed implementation todo list before coding begins

## Scan snapshot

- [x] Confirm tracked handoff artifact exists outside ignored plan paths
- [x] Confirm multiple runtime allocation owners exist:
  - [x] [governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/governor.py)
  - [x] [enhanced_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/enhanced_governor.py)
  - [x] [calendar_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/calendar_governor.py)
  - [x] [src/risk/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/governor.py)
  - [x] [prop/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/prop/governor.py)
- [x] Confirm session duplication inside [sessions.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sessions.py)
- [x] Confirm legacy static tier logic still exists in [dynamic_bot_limits.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dynamic_bot_limits.py)
- [x] Confirm account stop path still embeds flat thresholds in [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py)
- [x] Confirm progressive kill path still embeds old thresholds / prop-firm language in [progressive_kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/progressive_kill_switch.py)
- [x] Confirm broker handling is split between:
  - [x] [router broker registry](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/broker_registry.py)
  - [x] [broker_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/broker_endpoints.py)
- [x] Confirm runtime config still carries legacy prop/tier settings in [settings_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/settings_endpoints.py)
- [x] Confirm current UI surfaces expose news/physics/active-bot telemetry more than allocator truth

## Phase 0: Read and align

- [ ] Read [context handoff](/home/mubarkahimself/Desktop/QUANTMINDX/docs/2026-04-03-risk-allocation-context-handoff.md)
- [ ] Read [batch decomposition](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md)
- [ ] Read [Batch 1 authority plan](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-1-authority-plan.md)
- [ ] Read [Batch 1 execution checklist](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-1-execution-checklist.md)
- [ ] Read [Batch 2 execution checklist](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-2-execution-checklist.md)
- [ ] Read [Batch 3 execution checklist](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-3-execution-checklist.md)
- [ ] Read [detailed implementation plan](/home/mubarkahimself/Desktop/QUANTMINDX/docs/superpowers/plans/2026-04-03-risk-allocation-redesign-implementation-plan.md)
- [ ] Read [summary spec](/home/mubarkahimself/Desktop/QUANTMINDX/docs/superpowers/specs/2026-04-03-risk-allocation-redesign-design.md)
- [ ] Re-scan current runtime files before editing
- [ ] Confirm which deprecated paths still have live callers
- [ ] Confirm the tracked docs under `docs/` are the handoff truth, not the ignored plan path alone
- [ ] Confirm deployment topology assumptions against:
  - [ ] [deployment-guide.md](/home/mubarkahimself/Desktop/QUANTMINDX/docs/deployment-guide.md)
  - [ ] [mt5_socket_adapter.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/data/brokers/mt5_socket_adapter.py)
  - [ ] [socket_server.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/socket_server.py)

## Phase 1: Canonical state model

- [ ] Define canonical runtime types for:
  - [ ] lock states
  - [ ] session candidate
  - [ ] funded bot
  - [ ] slot allocation decision
  - [ ] pressure state
- [ ] Define one canonical session/window contract
- [ ] Define one canonical lock-state contract
- [ ] Define one canonical funded-bot versus open-slot model
- [ ] Review all existing DTOs for overlap with this model
- [ ] Decide where the canonical runtime state object lives
- [ ] Decide which API payload becomes the UI source of truth for:
  - [ ] funded breadth
  - [ ] concurrent open slots
  - [ ] pressure state
  - [ ] lock source
  - [ ] broker feasibility warnings
- [ ] Decide which node owns each state transition:
  - [ ] trading node
  - [ ] backend node
  - [ ] cold/storage path

## Phase 2: Allocation authority cleanup

- [ ] Audit [governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/governor.py)
- [ ] Audit [enhanced_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/enhanced_governor.py)
- [ ] Audit [calendar_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/calendar_governor.py)
- [ ] Audit [src/risk/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/governor.py)
- [ ] Audit [prop/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/prop/governor.py)
- [ ] Decide which file becomes the sole runtime allocation authority
- [ ] Decide which existing files become:
  - [ ] wrappers
  - [ ] adapters
  - [ ] delete candidates
- [ ] Remove shadow policy paths after migration
- [ ] Add explicit review note for each changed authority:
  - [ ] why it stays
  - [ ] why it goes
  - [ ] how callers migrate

## Phase 3: Session and queue model

- [ ] Clean [sessions.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sessions.py) into one authority
- [ ] Keep one session window model only:
  - [ ] canonical windows
  - [ ] premium session flags
  - [ ] family/session mix
  - [ ] dead/reset window
- [ ] Verify premium-session flags and canonical windows
- [ ] Align queue helpers:
  - [ ] [dpr_scoring_engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dpr_scoring_engine.py)
  - [ ] [queue_reranker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_reranker.py)
  - [ ] [queue_remix.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_remix.py)
  - [ ] [session_performer.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/session_performer.py)
- [ ] Make correlation pressure act before funding
- [ ] Define how funded breadth adapts by session quality and family mix
- [ ] Audit and retire [dynamic_bot_limits.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dynamic_bot_limits.py) if callers can move to the new allocator

## Phase 4: Small-account stop and pressure model

- [ ] Implement adaptive operating budget model
- [ ] Implement hard account lock model
- [ ] Add drawdown pressure bands before hard lock
- [ ] Ensure session pressure cannot bypass global hard lock
- [ ] Align [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py) with the new model
- [ ] Remove flat tiny-account stop assumptions from hot-path runtime logic
- [ ] Decide whether pressure state lives in:
  - [ ] account monitor
  - [ ] governor authority
  - [ ] a dedicated runtime state service

## Phase 5: Modifiers and breakers

- [ ] Refactor [session_kelly_modifiers.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/sizing/session_kelly_modifiers.py)
- [ ] Remove hardcoded threshold assumptions that are no longer authoritative
- [ ] Refactor [bot_circuit_breaker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/bot_circuit_breaker.py) for family-aware behavior
- [ ] Add weekly review escalation hooks
- [ ] Preserve journal-linked reason data
- [ ] Decide whether `personal` / `prop_firm` terminology stays runtime-relevant or is only metadata

## Phase 6: SQS and pre-trade quality gates

- [ ] Refactor [sqs_engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/sqs_engine.py)
- [ ] Remove fail-open hot-path defaults
- [ ] Define fail-small or block behavior for missing quality data
- [ ] Align quality gates with session and pressure states

## Phase 7: Position management and kill behavior

- [ ] Audit [position_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/position_monitor.py)
- [ ] Audit [position_monitor_engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/position_monitor_engine.py)
- [ ] Decide the single authoritative runtime manager
- [ ] Align kill-switch staging with position monitor delivery
- [ ] Preserve dynamic SL/TP centralization

## Phase 8: Broker adaptation and MT5 boundary

- [ ] Extend existing broker adaptation layer instead of replacing it
- [ ] Normalize broker profiles in:
  - [ ] [config/brokers.yaml](/home/mubarkahimself/Desktop/QUANTMINDX/config/brokers.yaml)
  - [ ] [broker_registry.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/broker_registry.py)
- [ ] Decide the boundary between:
  - [ ] broker profile / fee registry
  - [ ] connected broker/account discovery registry
- [ ] Confirm feasibility handling in:
  - [ ] [enhanced_kelly.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/position_sizing/enhanced_kelly.py)
  - [ ] [edge_cases.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/position_sizing/edge_cases.py)
- [ ] Extend [mt5_socket_adapter.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/data/brokers/mt5_socket_adapter.py) only where needed
- [ ] Verify node-role-safe transport assumptions:
  - [ ] backend -> trading
  - [ ] trading -> MT5 bridge
  - [ ] no frontend -> broker shortcut
- [ ] Reduce MQL policy ownership in:
  - [ ] [KellySizer.mqh](/home/mubarkahimself/Desktop/QUANTMINDX/src/mql5/Include/QuantMind/Risk/KellySizer.mqh)
  - [ ] [Constants.mqh](/home/mubarkahimself/Desktop/QUANTMINDX/src/mql5/Include/QuantMind/Core/Constants.mqh)

## Phase 9: Locks, news, and runtime config

- [ ] Unify lock-state semantics across:
  - [ ] [kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/kill_switch.py)
  - [ ] [progressive_kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/progressive_kill_switch.py)
  - [ ] [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py)
  - [ ] [sentinel.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sentinel.py)
- [ ] Refactor economic calendar / news endpoints
- [ ] Add sticky manual market lock and explicit resume
- [ ] Move authoritative runtime config ownership to deployment-level backend state
- [ ] Ensure IDE settings are editor surfaces, not runtime truth
- [ ] Remove or demote legacy `RiskSettings` fields that still encode prop-firm/static-tier behavior
- [ ] Verify news-lock state is session-aware, not a blunt global blackout unless intentionally escalated
- [ ] Define deployment-safe config flow:
  - [ ] backend persists truth
  - [ ] trading node consumes synced truth
  - [ ] frontend edits through backend only

## Phase 10: UI/API operationalization

- [ ] Refactor [kill_switch_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/kill_switch_endpoints.py)
- [ ] Extend [journal_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/journal_endpoints.py)
- [ ] Refactor [quantmind-ide store kill-switch](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/stores/kill-switch.ts)
- [ ] Review [quantmind-ide store risk](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/stores/risk.ts)
- [ ] Decide whether risk allocator state needs:
  - [ ] a new dedicated frontend store
  - [ ] an extension of existing `risk.ts`
  - [ ] or an extension of `kill-switch.ts`
- [ ] Update [KillSwitchView.svelte](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/components/KillSwitchView.svelte)
- [ ] Update [TradingFloorPanel.svelte](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/components/TradingFloorPanel.svelte)
- [ ] Update [LiveTradingCanvas.svelte](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte)
- [ ] Review [SettingsView.svelte](/home/mubarkahimself/Desktop/QUANTMINDX/quantmind-ide/src/lib/components/SettingsView.svelte) for deployment/runtime config separation
- [ ] Replace direct hardcoded `/api/...` transport calls where risk surfaces still bypass shared API helpers
- [ ] Verify touched UI with Chrome/browser devtools:
  - [ ] network requests
  - [ ] websocket payloads
  - [ ] rendered lock source / stop reason
  - [ ] sticky manual market lock behavior
  - [ ] backend-driven funded breadth / pressure display

## Phase 11: Journal and review loop

- [ ] Add missing journal fields for runtime context
- [ ] Add end-of-day or reset-window refresh hooks
- [ ] Define weekly review output artifacts
- [ ] Define what data feeds later agentic diagnosis
- [ ] Define review-worthy failure signatures:
  - [ ] repeated session ejections
  - [ ] repeated correlation conflicts
  - [ ] repeated min-lot infeasibility skips
  - [ ] repeated fee/spread vetoes
  - [ ] lock-trigger clusters around news/session transitions

## Phase 12: Cleanup

- [ ] Delete deprecated policy paths after migration
- [ ] Delete prop-firm-specific runtime remnants
- [ ] Delete duplicate session authorities
- [ ] Delete mock/fallback paths that still produce fail-open production behavior
- [ ] Do not comment out dead code

## Review gates before any implementation batch is marked done

- [ ] Review Gate A: ownership
  - [ ] one clear runtime authority selected
  - [ ] replacement/wrapper/delete map written down
- [ ] Review Gate B: safety
  - [ ] global hard lock still wins over session opportunity
  - [ ] pressure state is externally visible
  - [ ] operator can tell why trading is blocked
- [ ] Review Gate C: deployment
  - [ ] backend owns runtime truth
  - [ ] UI only edits and displays backend truth
  - [ ] local dev settings are not the live source of truth
- [ ] Review Gate D: cleanup
  - [ ] deprecated branches with migrated callers are removed
  - [ ] no shadow policy owners left behind
- [ ] Review Gate E: deployment and verification
  - [ ] node ownership is still coherent
  - [ ] no local-only config path became runtime truth
  - [ ] touched UI was checked in browser tools
  - [ ] review notes record what was actually verified

## Review checklist for each implementation batch

- [ ] One runtime allocation authority only
- [ ] Global hard account lock still cannot be bypassed
- [ ] Broker adaptation reused, not duplicated
- [ ] Deprecated branches removed where safe
- [ ] UI reflects deployment/runtime truth
- [ ] Operator can tell why trading is stopped
- [ ] Node ownership still matches deployment model
- [ ] Browser/devtools verification evidence recorded for touched UI
