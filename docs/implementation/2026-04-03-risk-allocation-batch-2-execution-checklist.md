# Risk Allocation Batch 2 Execution Checklist

Date: 2026-04-03
Status: ready for implementation session after Batch 1
Purpose: exact execution checklist for session and queue authority cleanup

## Batch 2 objective

Collapse session truth into one model and align queue/ranking paths with the funded-breadth and open-slot architecture.

## Batch 2 success condition

Batch 2 is complete only if all of the following are true:
- one session/window authority exists
- one premium-session definition path exists
- queue stages are explicit:
  - eligible
  - ranked
  - funded
  - open-slot eligible
- correlation pressure is introduced before funding, not only after execution
- static tier bot-limit logic is either migrated away from live callers or explicitly isolated for later deletion

## 1. Read order

Read in this order before touching code:

1. [context handoff](/home/mubarkahimself/Desktop/QUANTMINDX/docs/2026-04-03-risk-allocation-context-handoff.md)
2. [batch decomposition](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md)
3. [sessions.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sessions.py)
4. [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)
5. [dynamic_bot_limits.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dynamic_bot_limits.py)
6. [dpr_scoring_engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dpr_scoring_engine.py)
7. [queue_reranker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_reranker.py)
8. [queue_remix.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_remix.py)
9. [session_performer.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/session_performer.py)
10. [trading_session_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/trading_session_endpoints.py)
11. [router_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/router_endpoints.py)

## 2. Pre-edit caller scan

Run these before editing.

```bash
rg -n "CANONICAL_WINDOWS|SESSION_BOT_MIX|SessionTemplate|SESSIONS =" src/router src/api
rg -n "DynamicBotLimiter" src/router src/api
rg -n "queue_reranker|queue_remix|dpr_scoring_engine|session_performer" src/router src/api
rg -n "run_auction\(" src/router src/api
```

Expected findings:
- `sessions.py` contains both canonical windows and legacy session templates
- `commander.py` still imports and uses `DynamicBotLimiter`
- `api/router_endpoints.py` still references `DynamicBotLimiter`
- `api/trading_session_endpoints.py` already prefers `CANONICAL_WINDOWS`
- dead-zone/session performer paths already depend on DPR/queue helpers

## 3. Session authority decision

### Decision to enforce

Use the canonical window model in [sessions.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/sessions.py) as the single session/window authority:
- `CANONICAL_WINDOWS`
- `SESSION_BOT_MIX`
- `PREMIUM_SESSIONS`
- `TRADING_WINDOWS`

### Step checklist

- [ ] Confirm canonical windows are the production authority
- [ ] Identify every remaining caller of legacy `SESSIONS` / `SessionTemplate`
- [ ] Decide whether legacy session templates are:
  - [ ] compatibility-only temporary path
  - [ ] delete candidate after caller migration
- [ ] Keep one premium-session definition path only

### Verification

```bash
rg -n "get_session_template|get_all_session_templates|SESSIONS\b|SessionTemplate\b" src/router src/api
```

## 4. Queue-stage contract step

### Required contract

Batch 2 should make these stages explicit in code and review notes:
- eligible queue
- ranked queue
- funded session bots
- open-slot eligible candidates

### Files to align

- [dpr_scoring_engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dpr_scoring_engine.py)
- [queue_reranker.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_reranker.py)
- [queue_remix.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/queue_remix.py)
- [session_performer.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/session_performer.py)
- [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)

### What to do

- [ ] Document where each stage currently lives
- [ ] Remove ambiguous stage overlap where ranking and funding are blurred together
- [ ] Define where correlation pressure is applied before funding
- [ ] Define how session quality and family mix enter funded breadth

### What not to do yet

- [ ] do not implement full pressure-band stop logic here
- [ ] do not implement full broker feasibility logic here
- [ ] do not implement UI display changes here

## 5. `commander.py` session/queue cleanup steps

Target file:
- [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)

### What to check

- [ ] how `run_auction(...)` currently mixes:
  - [ ] session filtering
  - [ ] bot type filtering
  - [ ] dynamic bot limits
  - [ ] governor sizing results
- [ ] whether session filtering is already tied to `SESSION_BOT_MIX`
- [ ] where top-N selection still depends on `DynamicBotLimiter`

### Required Batch 2 outcome

- [ ] session-aware selection is still rooted in canonical windows
- [ ] selection logic is expressed in terms of funded breadth/open-slot architecture, not static bot-cap tiers
- [ ] any leftover `DynamicBotLimiter` usage is explicitly temporary or removed

### Verification

```bash
rg -n "DynamicBotLimiter|SESSION_BOT_MIX|run_auction\(" src/router/commander.py
```

## 6. `dynamic_bot_limits.py` handling

Target file:
- [dynamic_bot_limits.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/dynamic_bot_limits.py)

### Current problem

This file hardcodes:
- account tiers
- max bot counts
- 3% total risk split assumptions

That conflicts with the adaptive funded-breadth design.

### Batch 2 rule

- [ ] Do not keep this as a hidden runtime authority
- [ ] Either migrate live callers away from it or mark it as explicitly temporary with remaining callers documented
- [ ] If all live callers are migrated in Batch 2, delete it

### Caller check

```bash
rg -n "DynamicBotLimiter" src/router src/api
```

Known callers to inspect:
- [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)
- [market_scanner.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/market_scanner.py)
- [router_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/router_endpoints.py)

## 7. API/session surface checks

### `trading_session_endpoints.py`

- [ ] confirm it already uses `CANONICAL_WINDOWS`
- [ ] verify payloads reflect canonical session truth

### `router_endpoints.py`

- [ ] inspect for legacy bot-limit exposure
- [ ] remove endpoint behavior that cements outdated static-tier assumptions if changed in same batch

### Verification

```bash
rg -n "CANONICAL_WINDOWS|DynamicBotLimiter" src/api/trading_session_endpoints.py src/api/router_endpoints.py
```

## 8. Delete checkpoints

### Allowed deletions in Batch 2

Delete only if caller migration is complete and verified:
- legacy session authority branches inside `sessions.py`
- `dynamic_bot_limits.py`

### Mandatory rule

- [ ] if obsolete session/bot-limit code remains, document exact remaining callers
- [ ] if migrated fully, delete the obsolete branch/file
- [ ] never comment out dead session or bot-limit logic instead of deleting it

## 9. Post-edit verification

Run all of these after edits:

```bash
rg -n "CANONICAL_WINDOWS|SESSION_BOT_MIX|SessionTemplate|SESSIONS =" src/router src/api
rg -n "DynamicBotLimiter" src/router src/api
rg -n "queue_reranker|queue_remix|dpr_scoring_engine|session_performer" src/router src/api
python3 -m compileall src/router/sessions.py src/router/commander.py src/router/dynamic_bot_limits.py src/router/dpr_scoring_engine.py src/router/queue_reranker.py src/router/queue_remix.py src/router/session_performer.py src/api/trading_session_endpoints.py src/api/router_endpoints.py
```

Expected outcome:
- one session/window authority is obvious
- static bot-limit path is either gone or explicitly temporary
- queue-stage ownership is clearer
- edited Python files still compile

## 10. Review checklist

Review findings should be ordered by severity and include file references.

Reviewer must check:
- [ ] Does one session/window authority now exist?
- [ ] Are premium sessions defined in one place only?
- [ ] Does any live path still depend on static bot-limit tiers?
- [ ] Is correlation pressure now accounted for before funding, or is the gap explicitly documented?
- [ ] Were dead session or tier branches removed instead of commented out?
- [ ] Are remaining deferred deletions documented with live callers?

## 11. Batch 2 completion note to record

When Batch 2 finishes, record:
- canonical session authority chosen
- remaining legacy session callers, if any
- whether `DynamicBotLimiter` still exists
- where queue stages now live
- exact commands run for verification
