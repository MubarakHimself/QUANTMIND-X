# Risk Allocation Batch 1 Execution Checklist

Date: 2026-04-03
Status: ready for implementation session
Purpose: exact execution checklist for allocation-authority cleanup

## Batch 1 objective

Establish a single runtime allocation authority for the trading path and remove hidden authority selection from surrounding components.

## Batch 1 success condition

Batch 1 is complete only if all of the following are true:
- one file is the runtime allocation authority for live router flow
- `StrategyRouter` is the only place selecting that authority
- `Commander` no longer silently invents a governor owner
- remaining governor-like files are explicitly classified as:
  - wrapper
  - support module
  - legacy path pending migration
  - delete candidate
- any deleted branch is actually removed, not commented out

## 1. Read order

Read in this order before touching code:

1. [context handoff](/home/mubarkahimself/Desktop/QUANTMINDX/docs/2026-04-03-risk-allocation-context-handoff.md)
2. [batch decomposition](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-decomposition.md)
3. [Batch 1 authority plan](/home/mubarkahimself/Desktop/QUANTMINDX/docs/implementation/2026-04-03-risk-allocation-batch-1-authority-plan.md)
4. [engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/engine.py)
5. [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)
6. [enhanced_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/enhanced_governor.py)
7. [governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/governor.py)
8. [calendar_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/calendar_governor.py)
9. [prop/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/prop/governor.py)
10. [src/risk/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/governor.py)
11. [settings_endpoints.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/api/settings_endpoints.py)

## 2. Pre-edit caller scan

Run these before editing.

```bash
rg -n "EnhancedGovernor|CalendarGovernor|PropGovernor|RiskGovernor|Governor\(" src
rg -n "calculate_risk\(" src/router src/api src/backtesting
rg -n "calculate_position_size\(" src/risk src/router src/backtesting
rg -n "load_risk_settings\(" src
```

Expected findings:
- `src/router/engine.py` selects among multiple governor implementations
- `src/router/commander.py` can still construct `EnhancedGovernor()` itself
- `src/api/heartbeat.py` and `src/router/socket_server.py` still reference `PropGovernor`
- `src/backtesting/mode_runner.py` still uses `EnhancedGovernor`
- `src/risk/governor.py` remains a parallel orchestration path

Save these findings into review notes before changing behavior.

## 3. Authority decision step

### Decision to enforce

Use [enhanced_governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/enhanced_governor.py) as the canonical trading-node allocation authority.

### Step checklist

- [ ] Confirm `EnhancedGovernor.calculate_risk(...)` is the runtime contract to keep
- [ ] Confirm `Governor` remains only a base helper, not a peer owner
- [ ] Confirm `CalendarGovernor` is not allowed to remain a separate top-level owner long-term
- [ ] Confirm `PropGovernor` does not remain a bypass around the canonical authority
- [ ] Confirm `src/risk/governor.py` is not part of live router authority

### Review evidence required

Reviewer must be able to answer:
- why `EnhancedGovernor` was chosen over the others
- what exact role each non-authority file now has

## 4. `engine.py` execution steps

Target file:
- [engine.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/engine.py)

### What to do

- [ ] Identify current authority-selection logic for:
  - [ ] prop-firm accounts
  - [ ] normal accounts
  - [ ] fallback base governor
- [ ] Rewrite selection so `engine.py` is the only runtime authority selector
- [ ] Remove any ambiguous branch that allows separate peer authority ownership without explicit classification
- [ ] Add code comments only where needed to mark temporary compatibility wrappers

### What not to do yet

- [ ] do not rewrite session logic here
- [ ] do not rewrite pressure math here
- [ ] do not rewrite broker calibration here

### Verification

```bash
rg -n "self\.governor = |PropGovernor|EnhancedGovernor|Governor\(" src/router/engine.py
```

Expected outcome:
- one clear selection rule
- no ambiguous “maybe this other peer governor is also authoritative” behavior

## 5. `commander.py` execution steps

Target file:
- [commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py)

### What to do

- [ ] Find the fallback constructor path that creates `EnhancedGovernor()` when none is provided
- [ ] Remove hidden authority selection from Commander
- [ ] Make Commander consume an already-supplied authority object from engine/router path
- [ ] If backward compatibility is temporarily needed, make it explicit and documented as compatibility-only

### What not to do yet

- [ ] do not remove `DynamicBotLimiter` in Batch 1 unless a caller migration is done in the same session
- [ ] do not rewrite queue/session math here yet

### Verification

```bash
rg -n "EnhancedGovernor\(|Governor\(|self\._governor" src/router/commander.py
```

Expected outcome:
- Commander no longer silently decides allocator ownership

## 6. Non-authority file classification steps

### `src/router/governor.py`
- [ ] classify as base helper only or document why not
- [ ] remove any docs/comments that imply peer ownership if changed

### `src/router/calendar_governor.py`
- [ ] classify as temporary wrapper/subclass or supporting modifier path
- [ ] do not let it remain undefined as a separate owner

### `src/router/prop/governor.py`
- [ ] identify all live callers
- [ ] decide whether it becomes:
  - [ ] compatibility wrapper
  - [ ] deferred migration path
  - [ ] future delete candidate

### `src/risk/governor.py`
- [ ] classify as:
  - [ ] analytical path
  - [ ] test/backtesting support
  - [ ] future merge/delete candidate

### Verification commands

```bash
rg -n "PropGovernor|CalendarGovernor|RiskGovernor" src
rg -n "calculate_position_size\(" src/risk src/router src/backtesting
```

## 7. Delete checkpoints

### Allowed deletions in Batch 1

Only delete a branch/file if both are true:
- all live callers were migrated in the same batch
- verification confirms no remaining references

### Not-safe-to-delete-by-default in Batch 1

- [prop/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/prop/governor.py)
- [src/risk/governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/risk/governor.py)
- [governor.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/governor.py)

### Mandatory rule

- [ ] If obsolete code remains for compatibility, mark it as temporary in docs/review notes
- [ ] If obsolete code is replaced and callers are migrated, delete it
- [ ] Never comment out dead code instead of deleting it

## 8. Post-edit verification

Run all of these after edits:

```bash
rg -n "EnhancedGovernor|CalendarGovernor|PropGovernor|RiskGovernor|Governor\(" src
rg -n "calculate_risk\(" src/router src/api src/backtesting
python3 -m compileall src/router/engine.py src/router/commander.py src/router/enhanced_governor.py src/router/governor.py src/router/calendar_governor.py src/router/prop/governor.py src/api/settings_endpoints.py
```

Expected outcome:
- one clear runtime authority in router flow
- remaining non-authority files are explainable by role
- edited Python files still compile

## 9. Review checklist

Review findings should be ordered by severity and include file references.

Reviewer must check:
- [ ] Does `engine.py` now own governor selection alone?
- [ ] Does `commander.py` still create its own authority?
- [ ] Can a prop-firm path still bypass the canonical authority?
- [ ] Is `src/risk/governor.py` still masquerading as a live router owner?
- [ ] Were any dead branches removed instead of commented out?
- [ ] Are deferred deletions explicitly tracked with remaining callers?

## 10. Batch 1 completion note to record

When Batch 1 finishes, record:
- the chosen authority file
- the list of wrapper/support/legacy files
- any remaining prop-firm-specific live callers
- any deleted files or deleted branches
- exact commands run for verification
