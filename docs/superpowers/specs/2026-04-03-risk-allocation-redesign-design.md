# QUANTMINDX Risk Allocation, Position Sizing, and Runtime Protection Redesign

Date: 2026-04-03
Status: Draft for review
Scope: Capital allocation, funded-queue logic, session-aware deployment, broker/MT5 integration, kill-switch scope, runtime SL/TP ownership, and surgical codebase changes.

## 1. Purpose

This document replaces the earlier ambiguous redesign direction with a production-oriented design that matches the actual goal of QUANTMINDX:

- support a very large bot universe
- fund only the right subset at the right time
- let scalping and ORB play to their strengths
- compound small accounts aggressively but not stupidly
- protect the account from swarm-scale over-allocation
- keep hot-path logic on the trading node and heavy learning/ranking refresh off the hot path

This document reconciles four sources of truth:

- March 2026 planning document
- March 2026 addendum
- the later Opus redesign draft
- current code reality in `src/`

## 2. Core Decisions Locked In

### 2.1 System Role

The risk subsystem is not primarily a signal validator.
It is a capital allocation and capital protection system sitting above trusted EAs.

Validated EAs own:

- entry/exit logic
- family-specific setup logic
- session-specific participation rules at strategy level

The risk subsystem owns:

- funding eligibility
- funded queue ranking
- slot allocation
- account-level drawdown control
- correlation-aware budget control
- fee/spread-aware position sizing
- kill-switch and market-lock behavior

### 2.2 Correct Runtime Hierarchy

The runtime hierarchy is:

1. Variant Universe
2. Deployable Pool
3. Session-Eligible Queue
4. Ranked Session Queue
5. Funded Session Bots
6. Concurrent Open-Risk Slots
7. Open Positions

These levels must remain distinct in code, UI, and documentation.

### 2.3 Critical Distinction

The system may support 200+ variants without funding 200+ live positions.

The correct scaling rule is:

- equity increases funded breadth fastest
- equity increases concurrent slots more slowly
- equity increases per-slot risk slowest

This preserves swarm throughput while preventing a few oversized positions from dominating account risk.

### 2.4 Delete, Do Not Comment Out

Legacy code that is superseded by this redesign must be deleted, not commented out.
Dead branches, duplicate risk engines, and stale prop-firm logic increase ambiguity and future bugs.

## 3. Runtime Objects and Definitions

### 3.1 Variant Universe

All known bot variants in QUANTMINDX.
This includes many variants of the same strategy family.
A variant may be in research, paper, review, deployable, or live-admissible states.

### 3.2 Deployable Pool

Variants that are technically allowed to be considered for runtime funding.
In cold-start deployment this pool is driven mostly by:

- backtest robustness
- walk-forward robustness
- paper-trading quality
- family/session compatibility
- execution feasibility

Not by mature live performance, because live history does not yet exist.

### 3.3 Session-Eligible Queue

Deployable variants filtered for the current session by:

- session mask
- strategy family
- symbol affinity
- Sentinel regime state
- news lock status
- SQS / spread environment
- fee feasibility
- bot breaker state
- market lock status
- minimum-lot feasibility
- correlation-cluster pressure

### 3.4 Ranked Session Queue

The session-eligible queue sorted by a funding score.
Early deployment uses a bootstrap funding score.
Mature deployment transitions toward live DPR and journal-driven score weight.

### 3.5 Funded Session Bots

Bots that are armed for the current session and allowed to compete for open-risk slots.
They are not all open simultaneously.
They are the active queue for that session.

### 3.6 Concurrent Open-Risk Slots

The maximum number of simultaneous live positions that may exist at once.
This is smaller than the funded session bot count.
Slots recycle as scalpers and ORBs close.

## 4. Cold-Start Funding Logic

## 4.1 Why Live DPR Alone Is Wrong At Launch

At deployment start, QUANTMINDX will not have enough live history to rank bots mainly from live performance.
So the system must use a bootstrap funding score until sufficient live journal data exists.

## 4.2 Bootstrap Funding Score

The funding score should combine:

- paper trading performance quality
- backtest robustness
- walk-forward robustness
- family/session fit
- current spread/SQS fit
- fee fit
- correlation penalty
- symbol diversification bonus
- breaker health

This score should be computed outside the execution hot path and cached for session use.

## 4.3 Transition To Mature Funding

Over time, the funding score shifts weights toward:

- live journal performance
- live session fit
- realized slippage quality
- realized fee burn
- realized regime fit
- DPR tiering

The transition should be time- and sample-based, not manual.

## 5. Family-Specific Funding Model

A single funding framework should exist, but the parameters differ by family.

### 5.1 Scalping Family

Scalping should have:

- more funded bots per session
- lower per-slot risk
- faster slot recycling
- higher sensitivity to spread and fee drag
- stricter session-level breaker behavior
- more premium-session leverage through breadth, not reckless size

### 5.2 ORB Family

ORB should have:

- fewer funded bots per session
- fewer concurrent slots
- slightly higher per-slot allowance than scalping
- longer hold tolerance
- slower breaker escalation
- stronger dependence on session open conditions

### 5.3 Premium Session Principle

Premium sessions should primarily expand deployment permission and queue quality, not blindly inflate size.

Premium sessions may:

- increase funded session bot count
- increase queue turnover opportunity
- enable earlier house-money activation
- permit more aggressive recycling of freed slots

Premium sessions should not automatically override drawdown or correlation protections.

## 6. Equity Bands and Slot Growth

This table is a design starting point, not final calibrated math.
It is intentionally shaped so breadth grows faster than per-trade aggression.

| Equity Band | Funded Session Bots | Concurrent Open Slots | Scalp Slot Risk Cap | ORB Slot Risk Cap | Hard Open-Risk Ceiling |
|---|---:|---:|---:|---:|---:|
| $200-$349 | 6 | 3 | 0.50% | 0.75% | 2.0%-2.5% |
| $350-$699 | 10 | 4 | 0.50% | 0.75% | 2.5%-3.0% |
| $700-$1499 | 16 | 5 | 0.45%-0.50% | 0.80% | 3.0%-3.5% |
| $1500-$2999 | 24 | 6 | 0.40%-0.50% | 0.80%-1.00% | 3.5%-4.0% |
| $3000-$5000+ | 36-48 | 8-10 | 0.40%-0.50% | 0.80%-1.00% | 4.0%-5.0% |

### 6.1 Interpretation

A larger account should generally fund more bots and permit more simultaneous positions.
It should not mainly express growth by massively increasing per-slot percentage risk.

### 6.2 Dynamic Self-Adjustment

The allocator should re-evaluate equity bands automatically after session resets and end-of-day state updates.
Large positive jumps in equity should not instantly explode per-slot risk.
They may increase funded breadth and slots first.

## 7. Position Sizing and Loss Basis

### 7.1 Non-Negotiable Rule

Position sizing must use a worst-case protective loss basis.
Sizing may not assume that dynamic break-even logic will always save the trade.

### 7.2 Runtime SL/TP Ownership

Normal runtime management uses the dynamic stop/take-profit logic already envisioned in the planning docs.
However, every live trade still requires catastrophic fail-safe protection at broker level in case:

- Python crashes
- ZMQ disconnects
- position monitor fails
- MT5 relay fails

This broker-side protection exists as a fail-safe, not as the main strategic exit logic.

### 7.3 Sizing Basis

Allowed cash risk per trade must be compared against:

- stop-distance cash loss
- current spread cost
- round-trip commission cost
- slippage buffer
- broker minimum lot feasibility

If the correct lot size is below broker minimum and rounding up would over-risk the account, the trade must be skipped.

### 7.4 Kelly Inputs

Kelly should use realized rolling stats refreshed outside the hot path.
Do not recalculate learning inputs every tick or trade.
Refresh at session reset windows or end of day.

## 8. Drawdown, Session Pressure, and Breakers

## 8.1 Small-Account Philosophy

Small accounts should not use giant per-trade percentages just because scalping can compound quickly.
Compounding should come from:

- breadth
- queue quality
- slot recycling
- better family/session fit
- faster reuse of risk capacity

## 8.2 Daily and Weekly Stops

Default account controls remain conservative until proven otherwise:

- daily hard stop around 3%
- rolling weekly hard stop around 7%

The hard stop must freeze new funding regardless of active session.

## 8.3 Session Pressure States

The system should maintain clear session states:

- Normal
- Caution
- Restricted
- Stopped

These should affect:

- funded session bot count
- concurrent slots
- house-money permission
- SQS threshold
- family breadth
- whether only top-ranked bots are allowed

## 8.4 Bot Breaker

### Scalping

- 2 consecutive losses in a session: remove from live for rest of that session
- returns next session
- repeated session failures across the week escalate to review/paper

### ORB

- more failure room than scalping
- repeated failed sessions, not just quick consecutive losses, should drive escalation

### Weekly Escalation Concept

- 1 bad session: session-only demotion
- 2 bad sessions in 5 trading days: watchlist
- 3 bad sessions in 5 trading days: paper/review

This escalation must write to journal state, not just temporary memory.

## 9. Correlation and Queue Pressure

Correlation must act before funding, not only after positions are already open.

### 9.1 Before Funding

When multiple same-direction or cluster-correlated candidates become session-eligible together, the ranked queue should:

- reduce their funding score
- prefer diversified winners
- avoid funding all cluster members equally

### 9.2 After Funding

Open-risk ceiling checks must include correlation-adjusted exposure, not just naive slot count.

### 9.3 Cold Start

At cold start, correlation pressure should still operate using instrument/family heuristics and current correlation matrix, even before strong live DPR exists.

## 10. News, Economic Calendar, and Kill-Switch Scope

## 10.1 MT5 Economic Calendar

MT5/MQL5 economic calendar support should be treated as a primary source for trading-node news gating.
Trade-server-time handling must be explicit.
The backend may mirror this data for UI, planning, and reporting.

## 10.2 Session-Specific News Impact

News gating should not be treated as one flat global pause if the impact is clearly session- or market-specific.
The design should support:

- symbol-level impact
- currency-level impact
- session-level impact
- full-market lock when required

Example:

- high-impact USD news may aggressively affect New York and overlap sessions
- it should not automatically be assumed to shut down unrelated session logic forever
- however, if resulting market chaos trips global conditions, market lock can escalate beyond a local news block

## 10.3 Lock Types

The system must distinguish:

- News Lock: automatic, can be session/symbol scoped
- Loss Lock: automatic, account-level drawdown lock
- Market Lock: manual operator lock, sticky across sessions/days until manually resumed
- Kill Switch: emergency protective state controlling open positions and new funding

## 10.4 Sticky Manual Market Lock

If operator manually locks the market today, the system must remain locked tomorrow until explicit manual resume through UI/API.
This is required both for bad markets and for suspected system bugs.

## 10.5 Kill-Switch Scope

The addendum direction stands:

- Sentinel, SVSS, and other monitoring continue running
- Commander must stop new activations
- funded queue freezes
- position monitor becomes exit/tightening delivery mechanism
- house-money expansion is overridden
- recovery may raise SQS thresholds temporarily

Copilot kill switch remains separate from live trading kill switch.

## 11. Node Responsibilities

## 11.1 Trading Node (Cloudzy / MT5-proximate)

Hot path responsibilities:

- session queue evaluation
- funded queue selection
- slot allocation
- live PnL and drawdown tracking
- news lock and loss lock enforcement
- kill-switch enforcement
- fee/spread/SQS checks
- MT5 order and modify relay

## 11.2 Backend Node

Non-hot-path responsibilities:

- journal aggregation
- weekly review logic
- variant diagnostics
- paper/live comparison
- bootstrap funding score refresh
- DPR refresh
- Kelly rolling-stat refresh
- UI-facing summary and planning APIs

## 11.3 Storage / Shared Assets

Persistent responsibilities:

- trade journals
- bot reports
- shared assets
- historical ranking state
- config state
- manual lock state

## 12. UI Impact

The redesign touches UI intentionally. Required UI changes should be designed surgically.

### 12.1 Live Trading UI Needs

The live trading UI should expose:

- current session state
- funded session bot count
- concurrent open slots used / available
- account drawdown state
- news lock status
- market lock status
- kill-switch status
- manual resume control
- upcoming calendar events by session/currency

### 12.2 Kill-Switch UI Needs

Add distinct UI states for:

- manual market lock
- automatic loss lock
- news lock
- emergency kill state
- manual resume actions

### 12.3 Journal / Agentic UI Needs

Agentic and portfolio/development surfaces need access to:

- per-bot journal slices
- session failure summaries
- weekly review candidates
- reason codes for demotion/review

This should avoid forcing raw log reading as the only workflow.

## 13. Exness Raw Spread Implications

Using Exness Raw Spread means the allocator must explicitly consider:

- commission up to $3.50 each side per lot
- spreads that can widen materially around volatility/news/open-close transitions
- minimum lot size and symbol-specific pip value feasibility

This makes fee-aware position sizing mandatory, especially for scalpers trading near session opens.

## 14. Codebase Change Map

This is a design-level surgical map, not yet a complete implementation checklist.

### 14.1 Delete or Retire

Delete or fully retire duplicate or stale risk paths rather than leaving comment tombstones.
Particular scrutiny should be applied to:

- stale prop-firm-only code
- duplicate governor/risk engines
- obsolete risk constants and presets
- duplicated session authorities

### 14.2 Must Be Reworked

High-priority modules likely needing rework:

- `src/router/enhanced_governor.py`
- `src/router/governor.py`
- `src/risk/config.py`
- `src/router/progressive_kill_switch.py`
- `src/router/bot_circuit_breaker.py`
- `src/risk/sizing/session_kelly_modifiers.py`
- `src/router/sessions.py`
- `src/risk/sqs_engine.py`
- `src/router/position_monitor_engine.py`
- `src/router/broker_registry.py`
- relevant MT5 sizing / constant files under `src/mql5/Include/QuantMind/`

### 14.3 UI / API Surfaces Likely Touched

- `src/api/kill_switch_endpoints.py`
- `src/api/copilot_kill_switch_endpoints.py` only for keeping the distinction explicit
- `src/api/trading_floor_endpoints.py`
- `src/api/journal_endpoints.py`
- canvas/live-trading/shared-assets related UI surfaces in the IDE project

## 15. Implementation Principles

- no LLM in live execution hot path
- ranking refresh outside hot path
- cached session funding inputs on trading node
- no commented-out dead logic
- use one authoritative queue hierarchy
- use one authoritative session authority
- use one authoritative risk configuration source
- keep MT5 as execution bridge, not policy owner

## 16. Open Questions To Lock Next

These remain open and should be finalized before implementation begins:

1. exact funded session bot definition in code terms
2. exact open-slot scaling per equity band
3. premium-session breadth rules for scalping vs ORB
4. session drawdown pressure thresholds
5. weekly review thresholds by family
6. exact journal fields required for funding-score refresh
7. exact news-lock granularity: symbol, currency, session, global
8. exact fail-safe stop distance policy for broker-side catastrophic protection

## 17. Immediate Next Step

Before coding, produce one final implementation-ready design pass focused only on:

- queue hierarchy
- slot scaling
- session/premium funding modifiers
- news/lock state machine
- node responsibility boundaries
- delete/modify/keep file map

That pass should be treated as the canonical build document.
