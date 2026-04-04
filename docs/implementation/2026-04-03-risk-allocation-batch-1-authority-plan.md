# Risk Allocation Batch 1 Authority Plan

Date: 2026-04-03
Status: decomposition only
Purpose: define the first implementation batch that resolves allocation-authority ownership before any runtime behavior rewrite

## Goal

Pick the single runtime allocation authority and map every currently-live governor-style path into one of four buckets:
- keep as authority
- keep as wrapper/adapter temporarily
- refactor into supporting module
- delete after migration

This batch is architecture-first.
It should not try to finish sessions, UI, or broker work.

## Why this batch exists

The repo currently has too many live policy owners:
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/risk/governor.py`
- `src/router/prop/governor.py`

And multiple entry points instantiate them differently:
- `src/router/engine.py`
- `src/router/commander.py`
- `src/api/heartbeat.py`
- `src/router/socket_server.py`
- backtesting paths under `src/backtesting/`

Until that is cleaned up, later implementation will keep leaking policy duplication.

## Batch boundary

### In scope

- authority ownership
- live caller mapping
- runtime state contract for allocator outputs
- wrapper/delete decisions
- review checklist for later behavior changes

### Out of scope

Do not change these behaviors in Batch 1 beyond what is strictly needed to establish ownership:
- session family math
- pressure-band formulas
- broker fee calibration
- UI rendering changes
- journal schema expansion
- MQL policy reduction

Those belong to later batches.

## Node ownership assumption for Batch 1

Batch 1 is mostly trading-node and backend-contract work.

### Trading node files
- `src/router/governor.py`
- `src/router/enhanced_governor.py`
- `src/router/calendar_governor.py`
- `src/router/prop/governor.py`
- `src/router/engine.py`
- `src/router/commander.py`
- `src/router/interface.py`

### Backend contract files
- `src/api/settings_endpoints.py`
- `src/api/kill_switch_endpoints.py`
- `src/api/trading_session_risk_endpoints.py`

### Read-only reference during Batch 1
- `src/risk/governor.py`
- `src/api/heartbeat.py`
- `src/router/socket_server.py`
- `src/backtesting/mode_runner.py`

## Current observed call graph

### Primary live runtime path

1. `src/router/engine.py`
- loads risk settings via `load_risk_settings()`
- chooses governor implementation based on account type
- passes governor into `Commander`
- later calls `self.governor.calculate_risk(...)`

2. `src/router/commander.py`
- if no governor supplied, creates `EnhancedGovernor()` itself
- directly calls `_governor.calculate_risk(...)`
- still imports and uses `DynamicBotLimiter`

This means there is already ambiguity about whether:
- `StrategyRouter.engine` owns governor choice, or
- `Commander` owns fallback governor choice

That ambiguity must be removed in Batch 1.

### Secondary and legacy paths

- `src/api/heartbeat.py`
  - instantiates `PropGovernor`
- `src/router/socket_server.py`
  - references `PropGovernor`
- `src/backtesting/mode_runner.py`
  - instantiates `EnhancedGovernor`
- `src/risk/governor.py`
  - maintains separate `RiskGovernor.calculate_position_size(...)` orchestration path

These may remain for now, but they cannot keep pretending to be equivalent runtime owners.

## Recommended authority decision

### Recommended runtime authority

Use `src/router/enhanced_governor.py` as the base runtime authority for the trading node.

Reason:
- it already matches the `calculate_risk(...)` contract expected by the router path
- it is already wired into `StrategyRouter`
- it already sits closer to live router usage than `src/risk/governor.py`
- it already contains account/session-aware sizing behavior and fee-aware Kelly integration

### Recommended supporting roles

- `src/router/governor.py`
  - keep as low-level base mandate / clamp helper only if still needed by `EnhancedGovernor`
  - not a top-level runtime owner after migration

- `src/router/calendar_governor.py`
  - likely refactor into a supporting calendar/news modifier mixed into or called by the authority path
  - not a separate authority class long-term

- `src/router/prop/governor.py`
  - likely becomes deprecated runtime branch or reduced compatibility wrapper
  - current prop-firm tier assumptions conflict with the new adaptive architecture

- `src/risk/governor.py`
  - keep temporarily as non-router analytical/orchestrator path only if tests/backtesting still depend on it
  - not a live trading-node authority

## File-by-file Batch 1 decisions

### 1. `src/router/engine.py`

Status in Batch 1:
- modify

Why:
- currently chooses between `PropGovernor`, `EnhancedGovernor`, and base `Governor`
- this file should become the single place that selects the trading-node runtime authority

Required Batch 1 outcome:
- one authority selection rule
- Commander no longer silently invents a different owner
- prop-firm special handling does not bypass the canonical authority decision

### 2. `src/router/commander.py`

Status in Batch 1:
- modify

Why:
- currently creates `EnhancedGovernor()` if no governor passed
- that fallback makes Commander a hidden authority selector

Required Batch 1 outcome:
- Commander should consume an already-selected authority
- if no governor is supplied, it should fail clearly or use an explicit compatibility stub, not pick silently
- note `DynamicBotLimiter` import as a later deletion/migration concern

### 3. `src/router/enhanced_governor.py`

Status in Batch 1:
- keep as candidate authority
- inspect deeply
- minimal refactor only if needed for authority cleanup

Why:
- this is the best current fit for the router runtime contract

Required Batch 1 outcome:
- confirm it is the single authority or document why not
- document what parts inside it are still legacy and belong to later batches

### 4. `src/router/governor.py`

Status in Batch 1:
- keep temporarily
- demote from top-level owner

Why:
- `EnhancedGovernor` inherits from it
- it may still be useful as a base clamp/mandate helper

Required Batch 1 outcome:
- make sure it is no longer treated as a peer runtime authority in docs or wiring

### 5. `src/router/calendar_governor.py`

Status in Batch 1:
- inspect
- likely wrapper/mixin/supporting module

Why:
- currently extends `EnhancedGovernor`
- but separate authority inheritance increases ownership ambiguity

Required Batch 1 outcome:
- decide whether this remains a subclass, becomes a delegated modifier, or is deferred with an explicit wrapper label

### 6. `src/router/prop/governor.py`

Status in Batch 1:
- inspect closely
- likely deprecate from hot-path ownership

Why:
- embeds old prop-firm tiering assumptions
- still used by runtime entry points
- likely biggest source of architecture drift versus new adaptive model

Required Batch 1 outcome:
- identify all live callers
- decide whether to:
  - wrap it behind canonical authority temporarily, or
  - stop selecting it from live router paths

Do not delete it in Batch 1 unless all live callers are migrated.

### 7. `src/risk/governor.py`

Status in Batch 1:
- inspect only
- classify as non-router path or legacy analytical path

Why:
- separate `calculate_position_size(...)` orchestration
- strong test surface exists
- likely not the correct place for live router authority

Required Batch 1 outcome:
- explicitly document whether it remains:
  - analytical support
  - test-only support
  - later merge target
  - or delete candidate

### 8. `src/api/settings_endpoints.py`

Status in Batch 1:
- inspect only
- minimal changes only if needed to stop authority ambiguity

Why:
- risk settings leak old concepts into runtime owner selection

Required Batch 1 outcome:
- document which fields are legacy and should not drive future authority design

### 9. `src/api/kill_switch_endpoints.py`

Status in Batch 1:
- inspect only

Why:
- needed to understand current external state contract
- not yet the place to rewrite behavior

### 10. `src/api/trading_session_risk_endpoints.py`

Status in Batch 1:
- inspect only

Why:
- currently exposes `SessionKellyModifiers` state
- useful for mapping existing API contract versus future allocator state contract

## Delete-aware guidance for Batch 1

### Do not delete yet

These should not be deleted during Batch 1 unless their live callers are explicitly migrated in the same batch:
- `src/router/prop/governor.py`
- `src/risk/governor.py`
- `src/router/governor.py`

### Likely later delete or heavy demotion candidates

- `src/router/prop/governor.py`
  - if canonical authority absorbs or replaces prop-specific logic
- parts of `src/router/calendar_governor.py`
  - if calendar behavior becomes a delegated modifier instead of subclass owner
- top-level ownership semantics of `src/router/governor.py`
  - once only `EnhancedGovernor`-style authority remains

### Explicit rule

If a file is superseded:
- migrate callers
- verify behavior
- delete obsolete branch
- do not leave it commented out

## Batch 1 deliverables

1. authority decision record
2. live-caller matrix
3. keep/refactor/wrap/delete table
4. canonical runtime allocator state contract
5. explicit “not yet touched” list for later batches

## Batch 1 review checklist

A reviewer should be able to answer these with evidence:

1. Which file is now the sole runtime allocation authority?
2. Does `Commander` still silently choose a governor?
3. Can `StrategyRouter` still select a parallel authority path that bypasses the canonical one?
4. Are prop-firm-specific assumptions still the hidden owner of live risk behavior?
5. Which files are now wrappers only?
6. Which files are marked for deletion later, and which callers still keep them alive?
7. Was any dead branch deleted cleanly instead of being commented out?

## Batch 1 verification expectations

Before calling Batch 1 complete:
- re-run caller search for governor classes and `calculate_risk(...)`
- confirm authority selection is singular in router path
- confirm docs/todo/handoff are updated with the decision
- if any file was demoted or deprecated, note exact remaining live callers

## Suggested next artifact after this one

The next useful artifact is a stricter execution checklist for Batch 1 with:
- exact edit order
- exact grep/verification commands
- exact review prompts
- exact deletion checkpoints
