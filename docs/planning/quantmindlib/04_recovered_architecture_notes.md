# 04 — Recovered Internal Architecture Notes

**Recovery date:** 2026-04-08
**Method:** Code inspection of undocumented/stale areas
**Purpose:** Preserve recovered architecture for downstream implementation agents

---

## R-1: Sentinel Regime Detection Architecture

**Status:** Undocumented (MISSING)
**Source files:** `src/router/sentinel.py`, `src/router/multi_timeframe_sentinel.py`

### Overview
Sentinel is the market intelligence engine that ingests ticks, runs 4+ sensors, classifies regime, and emits `RegimeReport`. It is the primary input to Governor, KillSwitch, DPR, and Backtester.

### Regime Types
```
TREND_STABLE      — low chaos, directional movement
RANGE_STABLE      — low chaos, non-directional
HIGH_CHAOS        — high chaos, no clear direction
BREAKOUT_PRIME    — stable start, breakout momentum
NEWS_EVENT        — news/calendar-driven volatility
UNCERTAIN         — insufficient data or conflicting signals
```

### Sensor Chain (4 mandatory + 2 optional)
1. **ChaosSensor** — Lyapunov exponent for chaos detection
2. **RegimeSensor** (Ising) — IsingModel for magnetization/susceptibility
3. **CorrelationSensor** — cross-bot correlation penalties
4. **NewsSensor** — economic calendar blackouts (Finnhub)
5. **HMMRegimeSensor** — Hidden Markov Model (optional, shadow mode available)
6. **EnsembleVoter** — weighted ensemble of all sensors

### HMM Dual Model Support
- **Shadow mode:** HMM runs parallel, predictions logged but NOT used for decisions
- **Hybrid mode:** HMM + Ising weighted combination via `hmm_weight` (0.0-1.0)
- **Production mode:** HMM fully integrated into regime decision
- Shadow predictions persisted to `HMMShadowLog` table

### Multi-Timeframe Support
- `MultiTimeframeSentinel` maintains separate Sentinel per timeframe (M1/M5/M15/H1/H4/D1)
- OHLC bars aggregated from ticks via `TickAggregator`
- Dominant regime via majority voting; ties broken by highest timeframe

### Regimes from Physics Sensors
The physics sensors output regime labels that feed into the ensemble. Each sensor has its own regime classification method. The final regime is determined by the `_classify()` method combining all sensor outputs.

### Key Method: `on_tick(symbol, price, timeframe, high, low)`
1. Update all sensors with tick data
2. Optionally run HMM prediction (if enabled)
3. Classify regime via `_classify()`
4. Log shadow prediction (if shadow mode)
5. Persist to DB (if shadow mode)
6. Return `RegimeReport`

### Integration Points
- **Governor:** `regime`, `chaos_score`, `regime_quality` for risk throttling
- **KillSwitch:** `regime=="NEWS_EVENT"` or `chaos_score > 0.6` for immediate exit
- **DPR:** `regime_quality` for scoring
- **Backtester:** `SentinelEnhancedTester` uses regime for trade filtering

---

## R-2: DPR Dual-Engine Architecture

**Status:** Undocumented (MISSING)
**Source files:** `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py`

### Overview
The codebase has two DPR engines. They both compute the same 4-component weighted score but with different capabilities and data sources.

### Engine Comparison

| Aspect | Router Layer | Risk Layer |
|--------|-------------|------------|
| Location | `src/router/dpr_scoring_engine.py` | `src/risk/dpr/scoring_engine.py` |
| Async | Yes (full async) | Partially async |
| DB access | No (uses BotRegistry in-memory) | Yes (full SQLAlchemy) |
| Tie-breaking | No | Yes (4-level cascade) |
| Specialist boost | No | Yes (+5 for SESSION_SPECIALIST) |
| Trade source | BotRegistry.live_stats/paper_stats | TradeJournal table |
| Redis writes | Intended but GAP (not working) | No |
| Concern flagging | No | Yes (>20 point WoW drop) |

### DPR Redis Gap (CONFIRMED)
- Router layer `DprScoringEngine.compute_composite_score()` does NOT write to Redis
- The `score_all_bots()` method exists but has no Redis publish call
- Risk layer `DPRScoringEngine` also does not write to Redis
- Keys `dpr:score:{bot_id}` are referenced in code but never populated
- This means downstream systems expecting Redis DPR scores will not receive them

### DPR Scoring Components
```
win_rate    — 25% weight
pnl         — 30% weight (PnL baseline = 500 for score 100)
consistency — 20% weight
ev_per_trade — 25% weight (EV baseline = 1.25 for score 100)
```

### Tier Assignment
- T1: composite_score >= 80.0
- T2: composite_score >= 50.0
- T3: composite_score < 50.0

### Concern Detection
- Score drops >20 points week-over-week → `DPRConcernEvent` emitted
- Consecutive negative EV tracked via `session_concern:{magic_number}` Redis key (7-day TTL)

### Trade Source: T1 → T2 Data Flow
- T2 node fetches trade results from T1 node's `node_trading` database
- `fetch_trade_results_from_node_trading()` in risk layer DPR engine

---

## R-3: Three-Layer Kill Switch Architecture

**Status:** Undocumented (MISSING)
**Source files:** `src/router/kill_switch.py`, `src/router/progressive_kill_switch.py`, `src/risk/pipeline/layer3_kill_switch.py`

### Layer 1: Base KillSwitch
- STOP file monitoring (every 100ms)
- Socket-based `CLOSE_ALL` to all registered EAs
- `panic()` — synchronous immediate activation
- `trigger()` — async activation
- `reset()` — re-enable after review
- Global singleton: `get_kill_switch()`

### Layer 2: SmartKillSwitch (Regime-Aware)
Extends base with context-sensitive exit strategy:
```
Chaos < 0.3 AND DD < 10%  → Wait for breakeven
Chaos 0.3-0.6             → Trailing stop to breakeven
Chaos > 0.6 OR DD > 10%   → Immediate close
NEWS_EVENT regime         → Immediate close
```
Exit commands: `CLOSE_ALL`, `HALT_NEW_TRADES`, `SET_TRAILING_STOP` (to BREKEVEN)

### Layer 3: ProgressiveKillSwitch (5-Tier)
Full hierarchy from monitoring to nuclear shutdown:

| Tier | Level | Trigger | Action |
|------|-------|---------|--------|
| 5 | BLACK | Chaos > 0.75 OR broker timeout > 30s | Nuclear shutdown |
| 4 | RED/BLACK | EOD (21:00 UTC) OR session kill zone | Close all, block new |
| 3 | RED | Daily loss > 3% OR weekly loss > 5% | Close all, block new |
| 2 | ORANGE | 3+ failed bots OR family loss > 20% | Halt new trades |
| 1 | YELLOW | Bot-level circuit breaker | Warning only |

Alert levels: GREEN → YELLOW → ORANGE → RED → BLACK

### Layer 4: Layer3KillSwitch (Latency-Critical)
Runs on Kamatera T1 (Windows) for latency-critical forced exits. Bypasses Redis locks.
```
LYAPUNOV > 0.95     → Flag ALL scalping positions for forced exit
RVOL < 0.5          → Block new entries, flag positions for Layer 2 review
```
Priority queue: `kill:pending:{ticket}` sorted set, process oldest first.

### Integration Chain
```
ProgressiveKillSwitch (tier logic)
    ├──► KillSwitch (base) for emergency close
    └──► Layer3KillSwitch (forced exit on chaos/RVOL triggers)
```

---

## R-4: BotCircuitBreaker S3-11 Rules

**Status:** Undocumented (MISSING)
**Source files:** `src/router/bot_circuit_breaker.py`

### Quarantine Thresholds (by BotType)
```
SCALPER (PERSONAL book) → 2 consecutive losses
ORB (PROP_FIRM book)    → 3 consecutive losses
```

### S3-11: 3-Loss-in-a-Row Circuit Breaker
1. If 3 consecutive losses occur on a single day → block rest of that day
2. If 3 separate days with 3-loss streaks occur in a week → quarantine for rest of week
3. `daily_loss_streak_days` field tracks streak-day count
4. `last_loss_streak_date` tracks most recent streak date

### Daily Trade Limit
- 20 trades per day (configurable)
- `daily_trade_count` counter on `BotCircuitBreaker` record

### Sync on Quarantine
When `BotCircuitBreakerManager.quarantine_bot()` fires:
1. Sync to BotRegistry → remove `@live`, add `@quarantine`
2. Unregister from KillSwitch via `KillSwitch.unregister_ea()`

---

## R-5: BotTagRegistry Mutual Exclusivity Rules

**Status:** Undocumented (MISSING)
**Source files:** `src/router/bot_tag_registry.py`

### Tag System
Tags are singular strings (not arrays) — each bot has one primary lifecycle tag.

### Mutual Exclusivity Rules
```python
@primal      excludes: @paper_only, @quarantine, @dead
@paper_only  excludes: @primal
@quarantine  excludes: @primal
@dead        excludes: @primal
```

### Applying `@primal`
When applying `@primal`:
1. Auto-removes any conflicting tags first
2. Logs removal reason in `TagHistoryEntry`
3. Then applies `@primal`

### Tag Lifecycle
```
@paper_only → @primal → @live    (promotion path)
@live       → @quarantine        (demotion path)
@live       → @dead              (retirement path)
```

---

## R-6: VariantRegistry Genealogy

**Status:** Undocumented (MISSING)
**Source files:** `src/router/variant_registry.py`

### Variant Status Lifecycle
```
ACTIVE → OPTIMIZING → PAPER → LIVE → RETIRED
```

### Variant Model
`BotVariant` dataclass fields:
- `variant_id` — unique variant ID
- `parent_id` — immediate parent variant ID
- `origin_bot_id` — original bot this lineage started from
- `parameters` — current parameters
- `parameter_changes` — diff from parent
- `monte_carlo_run_id` — associated MC run
- `generation` — generation number in lineage
- `status` — variant status
- `DPR_score` — latest DPR score
- `lineage_depth` — depth in genealogy tree

### Persistence
- JSON file at `data/variant_registry.json`
- Singleton pattern: `VariantRegistry()` global instance

### Genealogy Methods
- `register_variant()` — create new variant with parent lineage
- `get_descendants(variant_id)` — all children/grandchildren
- `get_ancestors(variant_id)` — all parents/grandparents
- `get_siblings(variant_id)` — variants with same parent
- `get_active_variant_for_bot(bot_id)` — most recent non-retired variant

---

## R-7: 6-Mode Backtest Evaluation

**Status:** Undocumented (MISSING)
**Source files:** `src/backtesting/mode_runner.py`

### Mode Definitions

| Mode | Parameters | Data | Regime Filter |
|------|-----------|------|---------------|
| **VANILLA** | default | in-sample | No |
| **SPICED** | optimized | in-sample | Yes (HIGH_CHAOS, NEWS_EVENT skipped) |
| **VANILLA_FULL** | default | full history | No |
| **SPICED_FULL** | optimized | full history | Yes |
| **MODE_B** | default | alternate symbol/timeframe | stress test |
| **MODE_C** | default | bear market / high-vol regime | stress test |

### SPICED Backtester: SentinelEnhancedTester
- `chaos_threshold` default: 0.6
- `banned_regimes`: `['HIGH_CHAOS', 'NEWS_EVENT']`
- `_should_filter_trade()` — skips trade if `chaos_score > threshold` or `regime in banned_regimes`
- `_get_regime_quality()` — scalar for position sizing
- `_regime_history`, `_chaos_scores`, `_regime_qualities` — per-bar tracking

### Pass Criteria
- >= 4 of 6 modes must pass
- Each mode: Sharpe >= 1.0 AND Max DD <= 15% AND Win Rate >= 50%

### OOS Degradation Gate
- OOS degradation <= 15% = PASSED
- OOS degradation > 15% = FAILED

---

## R-8: Full Backtest Pipeline Evaluation Chain

**Status:** Undocumented (MISSING)
**Source files:** `src/backtesting/full_backtest_pipeline.py`

### Pipeline Steps
```
1. Data split (train/test/gap: 30/40/30 default)
     │
     ├──► 4 backtest variants (PythonStrategyTester + SentinelEnhancedTester)
     │         │
     │         ├──► VANILLA (default params, in-sample)
     │         ├──► SPICED (optimized params + regime filter, in-sample)
     │         ├──► VANILLA_FULL (Walk-Forward, default params)
     │         └──► SPICED_FULL (Walk-Forward + regime filter, optimized params)
     │
     ├──► Monte Carlo on VANILLA and SPICED (1000 sims, shuffle trade order)
     │
     ├──► PBO calculation (CSCV bootstrap, 100 sims, 5 blocks)
     │
     └──► Robustness score = mean([regime_robustness, WFA_validity, MC_confidence, PBO_score])
```

### Robustness Score Thresholds
```
>= 0.8  → Deploy (Low risk)
0.5-0.8 → Caution (Medium risk)
< 0.5   → Do Not Deploy (High risk)
PBO > 0.5 → High overfitting, overrides to High risk
```

### Output Artifacts
- `StrategyPerformance` DB record (kelly_score, sharpe_ratio, max_drawdown, win_rate, profit_factor, variant, symbol, parent_id)
- WF1 artifact tree: `backtests/{variant}/backtest_result.json`
- Report: `reports/backtests/summary.md`
- MC: `reports/backtests/monte_carlo.json`
- WFA: `reports/backtests/walk_forward.json`

---

## R-9: SSLCircuitBreaker State Machine

**Status:** Undocumented (MISSING)
**Source files:** `src/risk/ssl/circuit_breaker.py`, `src/events/ssl.py`

### States
```
LIVE → PAPER → RECOVERY → RETIRED
```

### Transition Triggers
- `LIVE → PAPER`: 2 consecutive losses (scalping) or 3 (ORB), or daily loss > threshold
- `PAPER → RECOVERY`: Recovery win count reached threshold
- `PAPER → RETIRED`: Extended failure or manual retirement
- `RECOVERY → LIVE`: Sustained positive performance

### Tier System
- **TIER_1**: More conservative thresholds
- **TIER_2**: Standard thresholds

### Per-Bot Tracking
- `paper_entry_timestamp` — when bot entered paper
- `recovery_win_count` — wins in recovery phase
- `state` — current SSL state
- Redis pub/sub for state change notifications (async)

---

## R-10: Governor Risk Authorization Logic

**Status:** Undocumented (MISSING)
**Source files:** `src/router/governor.py`, `src/risk/governor.py` (two separate files)

### RiskMandate Output
```
allocation_scalar  — 0.0-1.0 (physics-based throttling)
risk_mode          — STANDARD / CLAMPED / HALTED
position_size      — lots (EnhancedKelly calculated)
kelly_fraction     — Kelly fraction used
risk_amount        — $ amount at risk
kelly_adjustments  — audit trail of all adjustments
mode               — demo / live
```

### Risk Logic Chain
1. **Account-level drawdown check** — Tier 3 of ProgressiveKillSwitch
2. **Physics throttling:**
   - `chaos_score > 0.6` → `allocation_scalar = 0.2`
   - `chaos_score > 0.3` → `allocation_scalar = 0.7`
3. **Systemic correlation check** — reduce allocation for correlated signals
4. **5% concurrent exposure cap** — hard cap, returns `HALTED` if exceeded
5. **EnhancedKelly position sizing** — broker-specific fees (pip values, commissions)

### Two Governor Classes (Naming Conflict)
The codebase has two separate `Governor` classes:
- `src/router/governor.py` — risk authorization (compliance layer)
- `src/risk/governor.py` — risk execution (Tier 2 rules)

Both are involved in the risk pipeline. This naming overlap is a potential source of confusion for implementation agents.

---

## R-11: SVSS Indicator System

**Status:** Undocumented (MISSING)
**Source files:** `src/svss/` (indicators/ subdirectory)

### Indicator Base
```python
class BaseIndicator(ABC):
    def compute(tick) -> IndicatorResult
    def reset()
```

### IndicatorResult Dataclass
```python
name: str        — e.g., "VWAP", "RVOL"
value: float     — computed value
timestamp: datetime
session_id: str
metadata: dict   — extra context (e.g., session_name for VWAP)
```

### Available Indicators

| Indicator | Purpose | Key Output |
|-----------|---------|------------|
| `VWAPIndicator` | Volume-weighted average price | Session VWAP value |
| `RVOLIndicator` | Relative volume vs session average | RVOL ratio |
| `VolumeProfileIndicator` | Volume distribution per price level | POC, value area |
| `MFIIndicator` | Money flow index | MFI value (0-100) |

### Session Context
Indicators maintain session context — VWAP resets at session boundaries, RVOL compares to session average.

---

## R-12: CalendarGovernor and Session Blackout

**Status:** Undocumented (MISSING)
**Source files:** `src/router/calendar_governor.py`, `src/market/news_blackout.py`

### Session → Currency Mapping
```python
TOKYO:    {JPY, AUD, NZD, CNY}
SYDNEY:   {AUD, NZD}
LONDON:   {EUR, GBP, CHF}
NEW_YORK: {USD, CAD}
OVERLAP:  {EUR, GBP, USD, CHF, CAD}
```

### News Blackout Service
- Source: Finnhub `/calendar/economic` (polled every 30 minutes)
- Lookahead: 7 days
- Kill zone: 15 min before + 15 min after HIGH-impact events
- Maps to trading sessions via currency exposure
- Feeds NewsSensor in Sentinel
- Guard: `is_news_blackout_configured()` — only active if Finnhub API key is set

### Gap Identified
No economic calendar data feeds NewsSensor programmatically beyond Finnhub polling. Economic calendar → NewsSensor integration is a gap (per memory session notes).

---

## R-13: LifecycleManager State Transitions

**Status:** Undocumented (MISSING)
**Source files:** `src/router/lifecycle_manager.py`

### Promotion Logic
```
promote_to_live(strategy_id, dpr_score):
  1. Check DPR score >= 50.0 (T2 minimum)
  2. Remove existing tags (@live, @quarantine, @dead)
  3. Add @live tag via BotTagRegistry
  4. Set TradingMode.LIVE
  5. Register EA with KillSwitch
  6. Log to BotLifecycleLog
```

### Quarantine Logic
```
quarantine(strategy_id, reason):
  1. Add @quarantine via BotTagRegistry
  2. Remove @live tag
  3. Downgrade to PAPER TradingMode
  4. Unregister from KillSwitch
  5. Log to BotLifecycleLog
  6. Dispatch BOT_QUARANTINE_REVIEW to FloorManager
```

### Tag Hierarchy
```
quarantine > dead (in severity)
```

---

## R-14: ProgressiveKillSwitch Alert Levels

**Status:** Undocumented (MISSING)
**Source files:** `src/router/progressive_kill_switch.py`

### Alert Level Progression
```
GREEN → YELLOW → ORANGE → RED → BLACK
```

### Per-Tier Alert Assignment
- **GREEN/YELLOW:** Tier 1 (bot-level circuit breaker) — informational only
- **ORANGE:** Tier 2 (strategy family failures) — halt new trades
- **RED:** Tier 3 (account loss limits) — close all, block new
- **RED/BLACK:** Tier 4 (EOD/session kill zone) — close all, block new
- **BLACK:** Tier 5 (nuclear) — full system shutdown

### Manual Override
- `activate_manual_market_lock()` — sticky lock, must be explicitly resumed
- `resume_manual_market_lock()` — releases sticky lock

### Prop Firm Emergency
- `emergency_kill_prop_firm(prop_firm_name)` — kills all accounts for a prop firm

---

## R-15: StrategyType Cleanup (Phase 1)

**Status:** Undocumented (MISSING)
**Source files:** `src/router/bot_manifest.py`

### Phase 1 Active Strategy Types
```
SCALPER  — High-frequency, many trades
ORB      — Opening Range Breakout
```

### Legacy/Deprecated Strategy Types
The enum contains additional legacy types not active in Phase 1:
- STRUCTURAL
- SWING
- HFT
- NEWS
- GRID
(Exact enum values confirmed from code scan; implementation agents should treat SCALPER+ORB as canonical for Phase 1)

### BotType Mix (from Autonomous System Fix Plan)
- 60% scalping bots
- 40% ORB bots
(Referenced from memory index: autonomous-system-fix-plan-2026-03-25.md)

---

## R-16: DPR Tier Thresholds

**Status:** Undocumented (MISSING)
**Source files:** `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py`

### Tier Assignment
```
T1  — composite_score >= 80.0
T2  — composite_score >= 50.0
T3  — composite_score < 50.0
```

### Promotion Threshold
- DPR score >= 50.0 (T2) required for promotion to live
- Enforced by LifecycleManager

### DPR Concern Threshold
- Score drop >20 points week-over-week triggers `@session_concern` tag
- Dispatches DPR_CONCERN_REVIEW to FloorManager

---

## R-17: BotManifest BotType Mix per Session

**Status:** Undocumented (MISSING)
**Source files:** `src/router/bot_manifest.py`

### Canonical Session Windows (10 windows, from architecture doc)
Session windows define active trading periods. The system has 10 canonical session windows for different market sessions.

### Bot-Type Session Eligibility
- SCALPER bots: active during high-frequency windows (London open, NY open)
- ORB bots: active during opening range definition windows (London open, NY open)
- Session eligibility tracked in `BotManifest.session_tags` (TRDDocument schema)

### 10 Canonical Session Windows (from architecture §8)
Referenced from `_bmad-output/planning-artifacts/architecture.md` — exact window definitions should be recovered from `src/router/sessions.py`.

---

## R-18: Monte Carlo vs Mutation Boundary

**Status:** Undocumented (MISSING)
**Source files:** `src/backtesting/monte_carlo.py`, `flows/improvement_loop_flow.py`

### Critical Distinction
- **Monte Carlo is EVALUATION, not mutation**
- Monte Carlo: shuffles trade order, runs 1000 simulations, calculates confidence intervals
- Mutation: changes the candidate (parameters, strategy logic)
- These must remain separate in the library design

### Monte Carlo in WF2
WF2 (ImprovementLoopFlow) uses Monte Carlo for evaluation AFTER mutation produces new variants:
```
Analyze → Re-backtest (20% validation) → Monte Carlo → WFA → Paper → 3-day lag → Live
```
Monte Carlo stress-tests the variant under randomized conditions. It does NOT change the variant.

### Walk-Forward Distinctness
- Walk-Forward Analysis (WFA) tests out-of-sample performance across rolling windows
- WFA is evaluation, not mutation
- WFA and Monte Carlo must remain distinct in library evaluation paths

---

## R-19: Layer3KillSwitch Redis Lock Bypass

**Status:** Undocumented (MISSING)
**Source files:** `src/risk/pipeline/layer3_kill_switch.py`

### Lock Bypass Mechanism
When Layer 3 kill switch needs to force-close a position:
1. `release_layer2_locks(ticket)` deletes Redis key `lock:modify:{ticket}`
2. Position flagged in `kill:pending:{ticket}` priority queue
3. Forced close executed via `mt5.force_close_by_ticket()`
4. Event logged to `TradeRecord.layer3_events` JSON field

### Why Bypass?
Layer 2 has Redis locks for position modification (to prevent concurrent modifications). During forced exit, we need to override these locks immediately. Layer 3 is designed for Kamatera T1 (latency-critical node) and bypasses the lock layer.

---

## R-20: SVSS vs Library Feature Family Relationship

**Status:** Undocumented (MISSING)
**Source files:** `src/svss/`, `src/mcp_tools/asset_library.py`

### SVSS Indicators → Library Feature Families
SVSS provides real-time session-aware indicators that map to library feature families:
- `VWAPIndicator` → library feature family: **volume** (session VWAP)
- `RVOLIndicator` → library feature family: **volume** (relative volume)
- `VolumeProfileIndicator` → library feature family: **volume** (profile/logic allowed; rendering not required)
- `MFIIndicator` → library feature family: **indicators** (money flow)

### Asset Library → Library Feature Families
Asset library provides MQL5/strategy assets that map to library composition sources:
- Indicators (RSI, MACD, Bollinger) → library feature family: **indicators**
- Strategies → library archetype composition sources
- Risk management assets → library risk bridge integration

### Feature Confidence Tagging
Per trading_platform_decision_memo: features should be tagged by quality/feed confidence so strategies can degrade gracefully when better data is unavailable. This quality-aware feature system is not yet implemented in SVSS or asset library.