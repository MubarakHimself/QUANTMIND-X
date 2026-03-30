# QUANTMINDX — News Calendar Kill Switch & Backtest Data Architecture
## Design Specification — 2026-03-30

**Status:** Approved for implementation
**Session:** ui-nav3 (2026-03-30)
**Next agent:** Resume from "Implementation" section — all research and design decisions are complete.

---

## 0. Context & Decision Trail

This document captures all design decisions, their sources, and the files involved so any future agent can resume work without needing to re-research.

### Source Documents Used
| Document | Location | Sections Referenced |
|---|---|---|
| QuantMindX Planning Document March 2026 | `claude-desktop-workfolder/QuantMindX_Planning_Document_March2026.docx` | Section 4 (bot lifecycle), VPS architecture, EA template |
| QuantMindX Planning Addendum Session2 March 2026 | `claude-desktop-workfolder/QuantMindX_Planning_Addendum_Session2_March2026.docx` | S2/Sections 7,10,11,12,13,14; S3/Sections 3,4,5,6,7,9,10,11,14,15; S4/Sections 3,4,5,8,12,14,16,17; S5/Sections 1,2,4,5,6 |
| Session Memory 2026-03-30 | `memory/session_2026_03_30_work.md` | H-MM-1 done, news kill pending |
| Session Memory 2026-03-29 | `memory/session-2026-03-29-bug-list.md` | Full bug list, open items |

### Research Conducted This Session
1. **Codebase exploration** — kill switch chain, NewsSensor, economic calendar endpoints, NewsFeedPoller location
2. **Web research** — MT5 Python calendar (unavailable), Finnhub economic calendar API (chosen), OANDA (rejected), Forex Factory (rejected — no API)
3. **Backtest analysis** — data sources, split logic, agent access, agent tools (currently mocked)
4. **Planning doc gap analysis** — 20 gaps identified (C1–C20), 5 sprint-blocking

---

## 1. Problem Statement

### 1.1 The Critical Gap
`NewsSensor.update_calendar()` exists and is wired into the T4 SessionMonitor kill zone check — but it is **never called**. The entire kill switch chain works; it has no data feeding it.

**Chain (already implemented, just not fed):**
```
[missing feed] → NewsSensor.update_calendar()
                      ↓
              NewsSensor.check_state() → "KILL_ZONE"
                      ↓ (called on every check_all_tiers())
              SessionMonitor.check_session_allowed() → False
                      ↓
              ProgressiveKillSwitch._execute_kill(tier=4, level=BLACK)
                      ↓
              kill_switch.trigger(KillReason) → MT5 halt
```

### 1.2 Multi-Session Complexity
Trading runs across three major sessions: **Tokyo**, **London**, **New York**. Each session has its own currency exposure. A USD NFP event should not necessarily kill the Tokyo session if it's only trading JPY pairs. A JPY BoJ decision should kill Tokyo but not necessarily NY (unless NY trades JPY pairs).

### 1.3 Autonomous Recovery
The kill switch is **fully autonomous on recovery** — no manual reset needed for news events. `check_all_tiers()` is called per trade attempt. Once `news_sensor.check_state()` returns `"SAFE"`, the next trade attempt passes through automatically. This is by design (confirmed from `progressive_kill_switch.py:262–320`).

---

## 2. Approved Architecture: Approach B — News Blackout Service

**Decision source:** User approval + Planning doc C2/C3 sprint-blocking gaps
**Rejected alternatives:**
- Approach A (minimal wire) — doesn't build Session Gateway required by planning docs (C2)
- Approach C (Redis pub/sub) — over-engineered, Redis on kill-switch critical path is fragile

### 2.1 New Files to Create

| File | Purpose | Planning Doc Reference |
|---|---|---|
| `src/market/news_blackout.py` | Shared service — owns calendar feed + session-currency mapping + kill zone state | S3-4.5, S4-16 |
| `src/session/session_gateway.py` | Unified session service — replaces scattered session logic | S3-4.1, S4-14, S4-16 |
| `src/market/__init__.py` | Package init | — |
| `src/session/__init__.py` | Package init | — |

### 2.2 Files to Modify

| File | Change | Why |
|---|---|---|
| `src/api/server.py` lines 739–751 (startup) | Remove NewsFeedPoller startup | Duplicate of existing TFO system; user directive |
| `src/api/server.py` lines 805–811 (shutdown) | Remove NewsFeedPoller shutdown | Same reason |
| `quantmind-ide/src/api/economic_calendar_endpoints.py` | Replace `_generate_mock_events_for_date()` with live Finnhub call | C3 gap — mock data in production |
| `src/router/progressive_kill_switch.py` | Wire `NewsBlackoutService` into `SessionMonitor` initialization | Completes the missing feed |
| `src/api/server.py` | Start `NewsBlackoutService` background scheduler on startup | Ensures calendar data always fresh |
| `quantmind-ide/src/lib/components/live-trading/` | Add news kill zone status indicator to Trading Floor UI | H-NEWS-KILL requirement |

### 2.3 Files to Keep Unchanged
- `src/router/sensors/news.py` — `NewsSensor` is correct, no changes needed
- `src/router/progressive_kill_switch.py` (logic) — kill chain is correct
- `src/router/session_monitor.py` — SessionMonitor is correct
- `src/router/calendar_governor.py` — already production-ready
- `src/risk/models/calendar.py` — data models are correct

---

## 3. Economic Calendar Data Source Decision

**Chosen: Finnhub `/calendar/economic`**
**Decision source:** Web research this session + existing `FinnhubProvider` in codebase

### Why Finnhub
- Already integrated (`src/knowledge/news/finnhub_provider.py` — follow its auth pattern)
- Free tier: 60 req/min, forward-looking up to 3 months
- Fields map directly to existing `EconomicEvent` Pydantic model
- `FINNHUB_API_KEY` already in environment

### Why NOT others
- **MT5 Python library** — calendar functions not available in Python, only MQL5 (confirmed MQL5 forum)
- **OANDA forexlabs** — lookback-only, undocumented, deprecated risk
- **Forex Factory** — no API, scraping violates ToS
- **Trading Economics** — no usable free tier

### Finnhub Calendar Endpoint
```
GET https://finnhub.io/api/v1/calendar/economic?from=YYYY-MM-DD&to=YYYY-MM-DD&token=TOKEN
```
Response field `impact`: `"high"` / `"medium"` / `"low"` (string)
Filter to `"high"` impact only for kill zone purposes.
**Fetch schedule:** Every 30 minutes, cache 7 days of upcoming events.
**Startup:** Fetch immediately on service start, then schedule.

### Asian Session Coverage (Open Question — Verify Before Implementation)
Finnhub covers USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD, CNY events.
**TO VERIFY:** Does Finnhub cover all Tokyo-session relevant events (BoJ, Tankan, Japan CPI, RBA, AU employment)?
If coverage is thin for Asian session, fallback option is Finnhub + MT5 MQL5 service script bridge (see research notes).

---

## 4. Session-Currency Mapping

The `NewsBlackoutService` must map sessions to their currency exposure to avoid killing sessions unnecessarily.

```python
SESSION_CURRENCY_MAP = {
    "TOKYO":    ["JPY", "AUD", "NZD", "CNY"],   # 23:00–08:00 UTC
    "SYDNEY":   ["AUD", "NZD"],                  # 21:00–06:00 UTC
    "LONDON":   ["EUR", "GBP", "CHF"],           # 07:00–16:00 UTC
    "NEW_YORK": ["USD", "CAD"],                  # 12:00–21:00 UTC
    "OVERLAP":  ["EUR", "GBP", "USD", "CHF"],   # 12:00–16:00 UTC (London-NY overlap)
}
```

**Rule:** A news event for currency X triggers kill zone ONLY for sessions that trade X.
**Pair-level check:** If an event affects USD and the active Tokyo session is only running USDJPY, it IS affected.
**Source:** `NewsItem.get_affected_pairs()` already implements this in `src/risk/models/calendar.py`.

---

## 5. Kill Zone Timing

**Current NewsSensor defaults (do not change):**
- Pre-event block: 15 minutes
- Post-event block: 15 minutes
- `PRE_NEWS` warning: 30 minutes before (2× pre window)
- `POST_NEWS` recovery window: 30 minutes after (2× post window)

**Source:** `src/router/sensors/news.py:44` — `kill_zone_minutes_pre=15, kill_zone_minutes_post=15`
**Planning doc reference:** S3-4.5 specifies ±15 min as minimum; CalendarRule defaults (`DEFAULT_BLACKOUT_MINUTES=30`) apply to CalendarGovernor lot scaling (different from kill zone).

---

## 6. Autonomous Recovery Mechanism

**How it works (already implemented, no changes needed):**
1. `check_all_tiers()` is called per trade attempt
2. Calls `session_monitor.check_session_allowed()`
3. Which calls `news_sensor.check_state()`
4. Once event time + 15 min passes → `check_state()` returns `"SAFE"`
5. `check_session_allowed()` returns `True`
6. New trade attempts proceed normally

**No persistent lock is set for news events.** The `reset_tier(4)` method exists for EOD flag reset — not needed for news recovery.

**UI notification required:** When state transitions `KILL_ZONE → POST_NEWS → SAFE`, the frontend must receive a WebSocket event showing the system has resumed. The status band and kill switch icon must update.

---

## 7. Sprint-Blocking Gaps to Fix (from Planning Doc Analysis)

These must be fixed before live trading. Ordered by criticality.

### C1 — EA Configuration Block Incomplete
**File:** `src/mql5/templates/ea_base_template.py`
**Missing:** `BOT_TYPE` input, ATR-based `SL_ATR_MULT`/`TP_ATR_MULT`, `SESSION_TAGS` canonical names, `NEWS_BLACKOUT` boolean field, Trade Journal reason codes (`SESSION_END_CLOSE`, `TILT_PROFIT_TAKE`, etc.), HMM signal receiver section, equity-based sizing (replace `InpFixedLot` with `RISK_PCT=0.02`).
**Planning doc:** S4-3, S4-12, S4-17

### C2 — Session Gateway as Shared Service Missing
**Create:** `src/session/session_gateway.py`
**Currently:** Session enforcement scattered across `sessions.py`, `session_template.py`, `session_detector.py`
**Needs:** Unified gateway that all EAs and session templates query for session time windows and force-close logic.
**Planning doc:** S4-14, S4-16

### C3 — Economic Calendar Not Wired as Shared Service
**This spec.** See Section 2.
**Planning doc:** S3-4.5, S4-16

### C4 — SQS Weekend Session-Gated Ingestion Missing
**File:** `src/risk/sqs_engine.py`
**Missing:** `is_active_session()` guard (no weekend data ingestion), Monday warm-up window (threshold 0.60 for first 15 min), Sunday cache flush.
**Planning doc:** S5-5

### C5 — Bot Circuit Breaker Threshold Conflict
**File:** `src/router/bot_circuit_breaker.py`
**Conflict:** Code has 3 consecutive losses for scalping personal book. Planning doc S3-6.5 says 2 for scalping, 3 for ORB.
**Verify:** Check `LOSS_THRESHOLDS` dict — if scalping threshold = 3, change to 2.
**Planning doc:** S3-6.5, S5-1

---

## 8. Backtest Data Architecture

### 8.1 Current State
- Data: MT5 → Dukascopy → Cache fallback
- Cache: `./data/historical/` (Parquet), `./data/real_m5_multi_year/`
- WFO: 50% train / 10% gap / 20% test (rolling windows) — good but no global holdout
- Monte Carlo: randomizes trades from ONE backtest run — no separate data pool
- Full pipeline: ALL modes run on full dataset — no explicit splits
- Agent tools: `src/agents/tools/backtest_tools.py` — **MOCKED** (returns simulated results, not real pipeline)

### 8.2 Proposed Data Split Architecture

**Master clean sample = IMMUTABLE.** No process writes to it. Agents and backtests get a copy.

```
MASTER_DATA/
  ├── EURUSD_M5_MASTER.parquet    ← never touched
  ├── GBPUSD_M5_MASTER.parquet    ← never touched
  └── [symbol]_[tf]_MASTER.parquet

Per backtest session (copied from master):
  ├── 30% → In-sample (vanilla/spiced backtest)
  ├── 40% → Monte Carlo pool (dedicated separate data, not reusing trade history)
  └── 30% → Hard OOS holdout (never touched until final deployment validation)
```

**DataSplitConfig parameter** to add to `FullBacktestPipeline`:
```python
@dataclass
class DataSplitConfig:
    insample_pct: float = 0.30
    monte_carlo_pct: float = 0.40
    holdout_pct: float = 0.30
    master_data_path: str = "data/master/"
    copy_on_access: bool = True  # always work on a copy
```

### 8.3 Scale Consideration (200+ bots)
- Each bot backtest copies the required data slice (not the full master)
- Templatize: one `BacktestTemplate` per bot type (ORB, Momentum, Mean Reversion, Trend Continuation) with pre-configured parameters
- Parallel execution: `FullBacktestPipeline` already supports async; use task queue for 200+ concurrent backtests
- Agent tool `src/agents/tools/backtest_tools.py` must be unwired from mock and connected to real `FullBacktestPipeline`

### 8.4 Agent Access Fix Required
**File:** `src/agents/tools/backtest_tools.py`
**Current state:** Returns mock/simulated results
**Fix:** Wire `run_backtest()` to call `POST /api/v1/backtest/run` (already exists in `src/api/ide_backtest.py`)

---

## 9. UI Requirements

### 9.1 News Kill Zone Status (H-NEWS-KILL)
- Location: Kill switch panel in top bar (existing panel, add news zone row)
- States to show: SAFE (green) / PRE_NEWS (amber, countdown) / KILL_ZONE (red, flashing) / POST_NEWS (amber, recovery countdown)
- Per-session: Show which sessions are affected (e.g., "NY: KILL ZONE — NFP in 8 min")
- Source: WebSocket topic `kill-switch` already exists; add `news_state` field

### 9.2 Trading Floor Update on News
- When kill zone fires: Banner on Trading Floor showing event name, affected sessions, time to clear
- When clear: Brief "RESUMED" flash with green indicator
- All bot cards: grey out during KILL_ZONE for affected sessions

### 9.3 Economic Calendar Panel (existing `EconomicCalendarPanel.svelte`)
- Replace mock data store with live data from `/api/risk/economic-calendar`
- Show upcoming HIGH-impact events with countdown timers
- Color-coded by session impact (which sessions are affected)

---

## 10. Implementation Priority Order

**For the implementing agent — do in this order:**

1. ~~Remove NewsFeedPoller from `src/api/server.py`~~ ✅ DONE (commit 94717bb)
2. ~~Replace mock data in `quantmind-ide/src/api/economic_calendar_endpoints.py` with Finnhub live call~~ ✅ DONE (commit 94717bb)
3. ~~Create `src/market/news_blackout.py`~~ ✅ DONE (commit 94717bb)
4. ~~Wire `NewsBlackoutService` into `ProgressiveKillSwitch`~~ ✅ DONE (commit 94717bb)
5. **Create `src/session/session_gateway.py`** — consolidate session enforcement
6. **Add WebSocket events** for news state transitions (KILL_ZONE / SAFE) — verify WS topic exists in kill-switch store
7. **Update UI** — kill zone status in kill switch panel, Trading Floor banner, EconomicCalendarPanel live data
8. **Fix C4** — SQS weekend session guard (`src/risk/sqs_engine.py` missing `is_active_session()`)
9. **Fix C5** — bot circuit breaker scalping threshold = 2 not 3 (`src/router/bot_circuit_breaker.py`)
10. **Fix backtest agent tools** — unwire mock (`src/agents/tools/backtest_tools.py`), connect to real `/api/v1/backtest/run`
11. **Add DataSplitConfig** to `FullBacktestPipeline`

**Do NOT tackle yet (separate sprint):**
- C1 (EA template) — large scope, separate spec needed
- C2 (Session Gateway) — created in step 5 above but full consolidation is ongoing
- C8 (bot genealogy) — separate spec needed
- H-MM-2/3/4 (HMM UI controls) — separate spec per session notes

---

## 11. Open Questions (Verify Before Coding)

1. **Finnhub Asian session coverage** — Does Finnhub `/calendar/economic` reliably cover BoJ, Tankan, RBA, AU employment events? Test with `from=today&to=today+7days` and check for JPY/AUD events.
2. **Kill switch top bar** — Verify Chrome navigation to `localhost:3001` and inspect the top bar kill switch icon to confirm what data it already shows and what WebSocket topic it subscribes to. (Chrome extension was not connected during this session.)
3. **C18 verification** — Does `src/risk/sizing/session_kelly_modifiers.py` implement the exact continuous scaling formula from S3-11.3: `multiplier = 1.0 + (pnl_pct / 0.10)` with ceiling 2.5x? Read the file before touching it.
4. **NewsFeedPoller removal safety** — Before removing, confirm no other code imports from `src/knowledge/news/poller.py`. Run: `grep -r "news/poller\|NewsFeedPoller" src/`

---

## 12. Files Reference Map

```
KILL SWITCH CHAIN (read these to understand the existing system):
  src/router/progressive_kill_switch.py     — orchestrator (T1-T5)
  src/router/session_monitor.py             — T4, calls check_state()
  src/router/sensors/news.py                — NewsSensor, update_calendar(), check_state()
  src/router/calendar_governor.py           — lot scaling mixin (separate from kill zone)
  src/risk/models/calendar.py               — NewsItem, CalendarRule, CalendarEventType

ECONOMIC CALENDAR (existing, modify):
  quantmind-ide/src/api/economic_calendar_endpoints.py  — has mock data, replace with Finnhub
  quantmind-ide/src/lib/stores/economicCalendarStore.ts — frontend store
  quantmind-ide/src/lib/components/risk/EconomicCalendarPanel.svelte — panel UI

FINNHUB INTEGRATION (follow this auth pattern):
  src/knowledge/news/finnhub_provider.py    — existing Finnhub auth + retry pattern

NEWS FEED POLLER (remove these lines from):
  src/api/server.py lines 739–751           — startup (conditional on NODE_ROLE + FINNHUB_API_KEY)
  src/api/server.py lines 805–811           — shutdown

KILL SWITCH UI (already exists, add news state to):
  quantmind-ide/src/lib/stores/kill-switch.ts           — state management
  quantmind-ide/src/lib/components/CopilotPanel.svelte  — has kill switch controls
  (verify exact component in top bar via Chrome)

BACKTEST SYSTEM:
  src/backtesting/full_backtest_pipeline.py — orchestrator (add DataSplitConfig here)
  src/backtesting/monte_carlo.py            — needs dedicated data pool (not trade history)
  src/agents/tools/backtest_tools.py        — MOCKED — wire to real API
  src/api/ide_backtest.py                   — real REST endpoint POST /api/v1/backtest/run

SPRINT-BLOCKING FIXES:
  src/mql5/templates/ea_base_template.py    — C1: missing config block fields
  src/risk/sqs_engine.py                    — C4: missing weekend guard
  src/router/bot_circuit_breaker.py         — C5: scalping threshold = 2 not 3
```
