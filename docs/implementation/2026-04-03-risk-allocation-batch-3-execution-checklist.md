# Risk Allocation Batch 3 Execution Checklist

Date: 2026-04-03
Status: ready for implementation session after Batch 2
Purpose: exact execution checklist for drawdown, pressure, and lock-model cleanup

## Batch 3 objective

Replace flat tiny-account stop logic and contradictory kill/lock ownership with one coherent runtime model:
- adaptive operating budget
- sovereign hard account lock
- externally visible pressure bands
- sticky manual market lock
- consistent reason codes across router and API surfaces

## Batch 3 success condition

Batch 3 is complete only if all of the following are true:
- one clear lock-state model exists across account, session, market, and manual controls
- global hard account lock cannot be bypassed by session opportunity or session reset logic
- pressure-state reduction is visible without pretending it is a full stop
- manual market lock is sticky until explicit backend reset
- old flat-threshold paths are either migrated, isolated as temporary compatibility paths, or deleted
- stop and reduction reasons are visible through API payloads, not only logs
- any deleted legacy branch is actually removed, not commented out

## 1. Read order

Read in this order before touching code:

1. [context handoff](/home/mubarkahimself/Desktop/QUANTMINDX/docs/2026-04-03-risk-allocation-context-handoff.md)
2. [batch decomposition](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md)
3. [Batch 1 execution checklist](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-1-execution-checklist.md)
4. [Batch 2 execution checklist](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-2-execution-checklist.md)
5. [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py)
6. [progressive_kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/progressive_kill_switch.py)
7. [kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/kill_switch.py)
8. [bot_circuit_breaker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/bot_circuit_breaker.py)
9. [sentinel.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sentinel.py)
10. [kill_switch_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/kill_switch_endpoints.py)
11. [trading_session_risk_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/trading_session_risk_endpoints.py)
12. [session_kelly_modifiers.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/sizing/session_kelly_modifiers.py)

## 2. Pre-edit caller and semantics scan

Run these before editing.

```bash
rg -n "daily_stop|weekly_stop|max_daily_loss_pct|max_weekly_loss_pct|daily_loss_pct|weekly_loss_pct|pressure|operating budget|hard lock|market lock|manual lock|resume" src/router src/api src/risk
rg -n "check_all_tiers|get_status|reset_tier|reset_account_stops|reset_system_shutdown|last_check_result" src/router/progressive_kill_switch.py src/api/kill_switch_endpoints.py src/router/account_monitor.py
rg -n "KillReason|ExitStrategy|trigger\(|panic\(|reset\(|_send_halt_new_trades|CLOSE_ALL" src/router/kill_switch.py src/router/progressive_kill_switch.py src/api/kill_switch_endpoints.py
rg -n "SessionKellyModifiers|SessionKellyState|get_current_state|session-risk-state" src/api/trading_session_risk_endpoints.py src/risk/sizing/session_kelly_modifiers.py
rg -n "BotCircuitBreakerManager|consecutive_losses|daily_trade_limit|quarantine|PROP_FIRM|PERSONAL" src/router/bot_circuit_breaker.py src/api/kill_switch_endpoints.py src/router/progressive_kill_switch.py
```

Expected findings:
- `account_monitor.py` still encodes flat daily/weekly percentage stops
- `progressive_kill_switch.py` still treats account-level stops as a tier inside a broader alert ladder
- `kill_switch.py` still mixes emergency stop semantics with reset semantics and smart-exit behavior
- `trading_session_risk_endpoints.py` still exposes old RHM/session-Kelly state rather than the broader pressure/lock model
- `bot_circuit_breaker.py` still uses personal vs prop-firm framing that may no longer be the canonical runtime model
- API payloads still expose pieces of truth, not one canonical reason/state model

Save these findings into implementation review notes before changing behavior.

## 3. Canonical state decision step

### Decision to enforce

Batch 3 should make the runtime distinction explicit between:
- `pressure state`
- `session restriction`
- `market lock`
- `manual market lock`
- `hard account lock`
- `system shutdown`

These are not interchangeable.

### Required model

At minimum, the runtime model should distinguish:
- `NORMAL`
- `CAUTION`
- `RESTRICTED`
- `STOPPED`

And should separately record lock source such as:
- `NONE`
- `SESSION_PRESSURE`
- `ACCOUNT_PRESSURE`
- `ACCOUNT_HARD_LOCK`
- `MARKET_LOCK`
- `MANUAL_MARKET_LOCK`
- `SYSTEM_SHUTDOWN`
- `NEWS_LOCK`
- `CHAOS_LOCK`

### Step checklist

- [ ] Decide where the canonical lock-state object lives in runtime code
- [ ] Decide whether pressure-state ownership belongs to `AccountMonitor`, `ProgressiveKillSwitch`, or a dedicated state service
- [ ] Decide one reason-code vocabulary for both router and API surfaces
- [ ] Confirm global hard account lock remains sovereign over session opportunity
- [ ] Confirm manual market lock is sticky and backend-owned
- [ ] Confirm session tightening is reduction logic, not silent full-stop logic unless escalated

### Review evidence required

Reviewer must be able to answer:
- what object now owns lock/pressure truth
- how the code distinguishes reduction from full stop
- how the operator can tell why the system is reduced or stopped

## 4. `account_monitor.py` execution steps

Target file:
- [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py)

### Current conflict

This file currently acts like a flat threshold tripwire:
- daily stop
- weekly stop
- warning thresholds at percentage-of-limit

That is too coarse for the new small-account model.

### What to do

- [ ] Identify the current state model and determine what fields survive
- [ ] Replace flat-threshold-only semantics with an adaptive structure that can represent:
  - [ ] operating budget consumed
  - [ ] hard lock reached
  - [ ] current pressure band
  - [ ] stop reason / lock source
- [ ] Preserve daily and weekly accounting, but stop treating them as the only truth
- [ ] Decide whether `daily_stop_triggered` and `weekly_stop_triggered` remain compatibility fields or are replaced
- [ ] Ensure DB persistence stores enough state for restart-safe lock behavior
- [ ] Ensure reset behavior does not silently clear a manual market lock or system shutdown

### What not to do yet

- [ ] do not redesign journal ingestion here
- [ ] do not redesign session ranking here
- [ ] do not move frontend code here

### Verification

```bash
rg -n "daily_stop_triggered|weekly_stop_triggered|max_daily_loss_pct|max_weekly_loss_pct|get_stop_status|reset_account_stops" src/router/account_monitor.py
```

Expected outcome:
- account monitor no longer looks like a flat percent-stop-only model
- account monitor can explain pressure vs hard lock
- reset semantics are narrower and safer

## 5. `progressive_kill_switch.py` execution steps

Target file:
- [progressive_kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/progressive_kill_switch.py)

### Current conflict

This file currently:
- checks all tiers in order
- escalates to kill behavior based on alert ladder
- treats account/session/system controls as different tiers but not one coherent lock contract
- still carries prop-firm-oriented assumptions in config and emergency paths

### What to do

- [ ] Identify the exact boundary between:
  - [ ] pressure evaluation
  - [ ] lock evaluation
  - [ ] kill execution
- [ ] Stop using tier semantics as a substitute for a canonical runtime lock-state model
- [ ] Ensure `check_all_tiers(...)` returns or records enough reason detail for API/UI display
- [ ] Ensure session restrictions can tighten deployment without silently acting like a global hard stop
- [ ] Ensure account hard lock always wins over session opportunity
- [ ] Ensure manual market lock and system shutdown are not overwritten by later lower-severity checks
- [ ] Audit prop-firm emergency handling and classify it as:
  - [ ] compatibility-only
  - [ ] still needed
  - [ ] delete candidate after migration

### What not to do yet

- [ ] do not redesign broker transport here
- [ ] do not redesign session/window authority here
- [ ] do not redesign queue-stage selection here

### Verification

```bash
rg -n "check_all_tiers|get_status|reset_tier|emergency_kill_prop_firm|_last_check_result|tier3_max_daily_loss_pct|tier3_max_weekly_loss_pct" src/router/progressive_kill_switch.py
```

Expected outcome:
- progressive kill switch reads as orchestration over canonical state, not contradictory policy owner
- prop-firm-specific branches are explicitly classified
- returned status surfaces reason and state, not just alert count

## 6. `kill_switch.py` execution steps

Target file:
- [kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/kill_switch.py)

### Current conflict

`kill_switch.py` still owns emergency-stop mechanics, reset semantics, and smart-exit behavior. That is acceptable only if policy ownership stays elsewhere.

### What to do

- [ ] Confirm this file remains execution-oriented, not policy-authoritative
- [ ] Confirm `KillReason` covers the canonical reasons actually needed after Batch 3
- [ ] Add or refactor reason mapping only if required by the new canonical state model
- [ ] Ensure reset semantics remain explicit and manual-review oriented
- [ ] Ensure this file does not silently infer account/session pressure policy on its own

### What not to do yet

- [ ] do not rewrite staged exit algorithms unless required by lock-semantics consistency
- [ ] do not redesign socket transport here

### Verification

```bash
rg -n "class KillReason|class ExitStrategy|def trigger|def reset|panic\(|NEWS_EVENT|DRAWDOWN_LIMIT|SYSTEM_ERROR" src/router/kill_switch.py
```

Expected outcome:
- kill switch stays an execution mechanism
- policy reasons are mapped cleanly, not invented ad hoc
- reset remains manual and explicit

## 7. `bot_circuit_breaker.py` execution steps

Target file:
- [bot_circuit_breaker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/bot_circuit_breaker.py)

### Current conflict

This file still frames thresholds in terms of `PERSONAL` vs `PROP_FIRM`, with hardcoded consecutive-loss assumptions and daily trade limits. That may conflict with the new family/session-aware breaker model.

### What to do

- [ ] Identify which fields/thresholds are still truly runtime-relevant
- [ ] Decide whether book-type terminology stays runtime-authoritative or becomes compatibility metadata
- [ ] Align circuit-breaker outcomes with Batch 3 reason codes
- [ ] Ensure bot/family breaker semantics can coexist with account pressure and hard lock without contradiction
- [ ] Preserve data needed for later review and journal diagnosis

### What not to do yet

- [ ] do not redesign full weekly journal workflow here
- [ ] do not redesign candidate/funded queue logic here

### Verification

```bash
rg -n "AccountBook|LOSS_THRESHOLDS|DEFAULT_DAILY_TRADE_LIMIT|check_allowed|record_trade|is_quarantined|quarantine_reason" src/router/bot_circuit_breaker.py
```

Expected outcome:
- breaker semantics are compatible with family/session-aware runtime controls
- outdated personal/prop-firm framing is either removed, narrowed, or explicitly temporary

## 8. `sentinel.py` and lock-source integration

Target file:
- [sentinel.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sentinel.py)

### What to check

- [ ] confirm what news/chaos state values are actually emitted
- [ ] confirm whether sentinel outputs can map cleanly to:
  - [ ] market lock
  - [ ] news lock
  - [ ] chaos lock
- [ ] document whether any translation layer is required between sentinel regime reports and the canonical lock-state model

### Verification

```bash
rg -n "news_state|KILL_ZONE|chaos_score|regime_quality|is_systemic_risk|get_current_regime" src/router/sentinel.py
```

Expected outcome:
- sentinel remains signal/context source
- sentinel is not mistaken for the lock-state owner

## 9. API surface cleanup steps

### `kill_switch_endpoints.py`

Target file:
- [kill_switch_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/kill_switch_endpoints.py)

### What to do

- [ ] identify endpoints that expose old tier-only or alert-only truth
- [ ] ensure status payload exposes:
  - [ ] current pressure state
  - [ ] current lock source
  - [ ] whether trading is reduced or fully stopped
  - [ ] whether manual resume is required
  - [ ] human-readable reason text
- [ ] ensure reset endpoints respect sticky manual-review semantics
- [ ] classify any in-memory-only config update path as compatibility-only or delete candidate if backend truth is being introduced elsewhere

### `trading_session_risk_endpoints.py`

Target file:
- [trading_session_risk_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/trading_session_risk_endpoints.py)

### What to do

- [ ] audit whether this endpoint still exposes only old RHM/session-Kelly state
- [ ] decide whether it becomes:
  - [ ] compatibility view over the new pressure model
  - [ ] merged into a broader allocator-state API later
  - [ ] delete candidate after migration
- [ ] document exact callers before deleting or narrowing it

### Verification

```bash
rg -n "session-risk-state|SessionKellyModifiers|SessionKellyState|get_status|reset_system_shutdown|reset_account_stops|last_check_result" src/api/kill_switch_endpoints.py src/api/trading_session_risk_endpoints.py
```

Expected outcome:
- API surfaces expose one coherent story about reduction/stops
- old RHM-only session risk is no longer pretending to be the whole runtime truth

## 10. Delete checkpoints

### Allowed deletions in Batch 3

Delete only if caller migration is complete and verified:
- obsolete flat-threshold branches in `account_monitor.py`
- obsolete tier-only branches in `progressive_kill_switch.py`
- obsolete reset/status payload paths in `kill_switch_endpoints.py`
- obsolete compatibility-only paths in `trading_session_risk_endpoints.py`
- obsolete personal/prop-firm runtime branches in `bot_circuit_breaker.py`

### Not-safe-to-delete-by-default in Batch 3

- [kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/kill_switch.py)
- [sentinel.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sentinel.py)
- [account_monitor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/account_monitor.py) as a whole file, unless a replacement is created in the same batch
- [progressive_kill_switch.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/progressive_kill_switch.py) as a whole file, unless orchestration ownership is fully replaced in the same batch

### Mandatory rule

- [ ] if obsolete code remains for compatibility, document exact remaining callers and expiration intent
- [ ] if obsolete code is replaced and callers are migrated, delete it
- [ ] never comment out dead lock/pressure logic instead of deleting it

## 11. Post-edit verification

Run all of these after edits:

```bash
rg -n "daily_stop|weekly_stop|max_daily_loss_pct|max_weekly_loss_pct|pressure|hard lock|market lock|manual lock|reason|lock_source|state" src/router src/api src/risk
rg -n "check_all_tiers|get_status|reset_tier|reset_account_stops|reset_system_shutdown|last_check_result" src/router/progressive_kill_switch.py src/api/kill_switch_endpoints.py src/router/account_monitor.py
rg -n "SessionKellyModifiers|SessionKellyState|session-risk-state" src/api/trading_session_risk_endpoints.py src/risk/sizing/session_kelly_modifiers.py
python3 -m compileall src/router/account_monitor.py src/router/progressive_kill_switch.py src/router/kill_switch.py src/router/bot_circuit_breaker.py src/router/sentinel.py src/api/kill_switch_endpoints.py src/api/trading_session_risk_endpoints.py
```

Expected outcome:
- pressure and lock language are coherent across runtime and API files
- hard account lock cannot be silently bypassed
- manual market lock/reset semantics are explicit
- edited Python files still compile

## 12. Review checklist

Review findings should be ordered by severity and include file references.

Reviewer must check:
- [ ] Can session opportunity ever reopen trading after global hard lock?
- [ ] Is pressure state distinct from full stop in runtime and API payloads?
- [ ] Is manual market lock sticky until explicit reset?
- [ ] Does `kill_switch.py` stay execution-oriented rather than policy-authoritative?
- [ ] Does any old flat-threshold-only logic still remain as hidden source of truth?
- [ ] Does any endpoint still expose an incomplete or misleading picture of runtime state?
- [ ] Were dead branches removed instead of commented out?
- [ ] Are deferred deletions documented with live callers?

## 13. Browser/UI follow-up note

Batch 3 is backend/router work first, but it creates mandatory UI follow-up checks later.

When UI work starts, browser/devtools verification must confirm:
- backend-reported lock source is rendered accurately
- backend-reported pressure state is rendered accurately
- manual market lock remains sticky after refresh/reconnect
- operator can tell whether the system is reduced, stopped, or awaiting manual resume

Do not implement those UI changes in Batch 3 unless the same session explicitly includes frontend work.

## 14. Batch 3 completion note to record

When Batch 3 finishes, record:
- canonical lock-state owner chosen
- canonical reason-code model chosen
- whether `AccountMonitor` still exists as the pressure/hard-lock owner or was narrowed
- whether `ProgressiveKillSwitch` remains orchestration only
- whether `trading_session_risk_endpoints.py` still exists and why
- any deleted branches or deleted files
- exact commands run for verification
