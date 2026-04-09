# 02 — Codebase Scan Report: QUANTMINDX Subsystems

**Scan date:** 2026-04-08
**Scope:** `/home/mubarkahimself/Desktop/QUANTMINDX/src/` (19 major subsystems)
**Method:** 8 parallel Explore agents reading file structure, class/function signatures, and key patterns — no full content reads

---

## 1. Router (Trading Intelligence Core)

**Path:** `src/router/`
**Files:** 60+ Python files, ~8,000+ lines
**Purpose:** Strategy auction, bot manifest, kill switches, sentinel, session management, DPR, lifecycle, routing matrix, market scanning, cooldown, tilt detection, weekend cycle, HMM version control

### Key Modules and Responsibilities

| Module | File | Class/Function | Responsibility |
|--------|------|----------------|----------------|
| **Bot auction** | `bot_manifest.py` | `BotRegistry`, `BotManifest`, `StrategyType` (SCALPER/ORB only in Phase 1) | Bot passport system; in-memory registry persisted to SQLite |
| **Sentinel** | `sentinel.py` | `Sentinel`, `RegimeReport` | Regime detection via 4 sensors (Chaos, Regime, Correlation, News); HMM shadow mode; outputs `RegimeReport` |
| **MultiTF** | `multi_timeframe_sentinel.py` | `MultiTimeframeSentinel`, `Timeframe`, `OHLCBar` | Separate Sentinel per timeframe; voting aggregation |
| **Governor** | `governor.py` | `Governor` | Risk authorization via `RiskMandate`; physics throttling; 5% exposure cap; EnhancedKelly integration |
| **Commander** | `commander.py` | `Commander` | Execution layer; bot selection and dispatch |
| **DPR** | `dpr_scoring_engine.py` | `DprScoringEngine`, `DprScore` | Async DPR scorer; writes to Redis (`dpr:score:{bot_id}`); 4-component weighted score |
| **Lifecycle** | `lifecycle_manager.py` | `LifecycleManager` | Bot promotion (DPR >= 50), quarantine, retirement; syncs tags and kill switch |
| **KillSwitch** | `kill_switch.py` | `KillSwitch`, `SmartKillSwitch`, `KillReason` | STOP file monitoring; socket-based close-all; regime-aware smart exit |
| **ProgressiveKillSwitch** | `progressive_kill_switch.py` | `ProgressiveKillSwitch`, `ProgressiveConfig` | 5-tier protection hierarchy; alert levels GREEN→BLACK; tier 2 family failure, tier 3 account loss, tier 4 EOD, tier 5 nuclear |
| **BotCircuitBreaker** | `bot_circuit_breaker.py` | `BotCircuitBreakerManager` | Per-bot quarantine (2 losses scalping, 3 ORB); 3-loss-in-a-row S3-11 rule; sync to BotRegistry |
| **BotTagRegistry** | `bot_tag_registry.py` | `BotTagRegistry`, `TagHistoryEntry` | Tag management (`@live`, `@quarantine`, `@dead`); mutual exclusivity enforcement |
| **VariantRegistry** | `variant_registry.py` | `VariantRegistry`, `BotVariant` | Genealogy tracking; JSON persistence; `ACTIVE→OPTIMIZING→PAPER→LIVE→RETIRED` |
| **MarketScanner** | `market_scanner.py` | `MarketScanner` | Session breakout/volatility/news/ICT scan per schedule |
| **InterSessionCooldown** | `inter_session_cooldown.py` | `InterSessionCooldown` | Cooldown state machine between sessions |
| **Tilt** | `tilt.py` | `TiltState`, `TiltPhase` | Tilt detection and suspend/resume |
| **SessionDetector** | `session_detector.py` | `SessionDetector` | Active session detection |
| **CalendarGovernor** | `calendar_governor.py` | `CalendarGovernor` | Economic calendar integration |
| **Weekend cycle** | `weekend_cycle_scheduler.py` | Multiple service classes | Friday analysis → Saturday refinement → Sunday calibration → Monday morning |
| **HMM** | `hmm_version_control.py`, `hmm_deployment.py`, `hmm_retrain_trigger.py`, `hmm_wfa_calibrator.py` | HMM version/deployment management | Shadow/hybrid/production modes; WFA calibration |
| **Workflow orchestrator** | `workflow_orchestrator.py` | `WorkflowOrchestrator` | WF1/WF2 coordination |
| **Socket server** | `socket_server.py` | `SocketServer` | EA communication |

### Key Inbound Flows
- Tick data → Sentinel → RegimeReport → Governor, KillSwitch, DPR, bot selection
- Trade outcomes → DPR → score → BotRegistry tags → LifecycleManager → promotion/quarantine
- TRD input → BotRegistry registration → Backtest pipeline → DPR scoring → promotion

### Key Outbound Flows
- Governor `RiskMandate` → Commander → execution
- LifecycleManager `promote_to_live()` → KillSwitch.register_ea()
- DPR `BotSessionMetrics` → BotRegistry live_stats/paper_stats updates
- Sentinel `RegimeReport` → Backtester (regime filtering), KillSwitch (news/chaos triggers)

### Coupling Points
- `src/risk/governor.py` (Governor vs router/Governor — two separate classes)
- `src/position_sizing/enhanced_kelly.py` (Governor imports EnhancedKellyCalculator)
- `src/database/models/bots.py` (BotCircuitBreaker, BotLifecycleLog)
- `src/events/` (regime, chaos, dpr, ssl event types)
- `src/risk/dpr/scoring_engine.py` (dual DPR implementation — router + risk layers)
- Redis (DPR scores, RVOL, SQS, SSL state)

### Risk if Touched
**HIGH** — router/ is the central nervous system. Changing sentinel, governor, DPR, or kill-switch interfaces risks destabilizing the entire trading loop. Treat as stable integration targets.

### Relation to QuantMindLib
- Primary integration target for bridge layer
- BotManifest → maps to QuantMindLib BotSpec (static profile)
- RegimeReport → maps to MarketContext
- RiskMandate → maps to RiskEnvelope
- DPR scores → maps to BotEvaluationProfile
- KillSwitch → maps to library safety hooks

---

## 2. Risk Engine

**Path:** `src/risk/`
**Files:** 40+ Python files across sub-packages: `physics/`, `dpr/`, `sizing/`, `ssl/`, `pipeline/`, `integrations/`
**Purpose:** Physics-based regime detection, Kelly position sizing, DPR scoring, circuit breaker, SQS gate, SVSS

### Key Modules and Responsibilities

| Module | File | Class/Function | Responsibility |
|--------|------|----------------|----------------|
| **Governor** | `governor.py` | `RiskMandate`, `Governor` | Tier 2 risk rules; physics throttling; 5% cap; EnhancedKelly |
| **Physics sensors** | `physics/` | `IsingRegimeSensor`, `ChaosSensor`, `CorrelationSensor`, `HMMRegimeSensor`, `EnsembleVoter` | 6-model regime detection; Ising+HMM ensemble |
| **Enhanced Kelly** | `sizing/kelly_engine.py` | `KellyEngine` | Fee-aware Kelly sizing; 3-layer protection |
| **DPR risk layer** | `dpr/scoring_engine.py` | `DPRScoringEngine` (risk) | Full DPR with DB + tie-breaking + specialist boost |
| **SSL** | `ssl/circuit_breaker.py` | `SSLCircuitBreaker` | Per-bot state machine; Redis pub/sub async |
| **SQS** | `sqs_engine.py` | `SQSEngine` | Spread quality gate (threshold 0.50) |
| **SVSS** | `svss_engine.py` | `SVSSEngine` | Shared Volume Session Service; async |
| **RVOL consumer** | `svss_rvol_consumer.py` | RVOL consumer | Early warning via Redis |
| **Layer3KillSwitch** | `pipeline/layer3_kill_switch.py` | Forced exit | Bypasses Redis locks; Lyapunov + RVOL triggers |

### Key Data Flows
```
Sentinel (router/) → RegimeReport → Governor.calculate_risk()
  → RiskMandate (allocation_scalar, position_size, risk_mode)
  → Commander

Trade execution → DPR → Score → BotRegistry → LifecycleManager → kill switch
```

### Async Patterns
- SSLCircuitBreaker: `async def` state transitions, Redis pub/sub
- SVSSEngine: `async def run()`
- DPR SSL consumer: async Redis consumption
- Layer3KillSwitch: SVSS pub/sub subscription

### Risk if Touched
**HIGH** — risk/ is the safety layer. Changes here affect all live bots. Treat as stable integration target.

### Relation to QuantMindLib
- Primary RiskEnvelope producer
- Governor maps to library risk bridge
- Physics sensors map to library market intelligence bridge
- DPR maps to library evaluation profile bridge

---

## 3. Position Sizing

**Path:** `src/position_sizing/`
**Files:** 8 Python files
**Purpose:** Pure Enhanced Kelly Criterion with 3-layer protection; no async; no Redis; standalone calculation engine

### Key Classes
- `EnhancedKellyCalculator` — main calculator with 3-layer protection
- `enhanced_kelly_position_size()` — standalone entry point function
- `KellyStatisticsAnalyzer` — session-level EV calculations
- `PortfolioKellyScaler` — portfolio-wide Kelly scaling
- `PropFirmPresets` — factory methods for FTMO, 5ers, personal, paper
- `MonteCarloValidator` — Monte Carlo simulation for sizing validation

### Key Inputs/Outputs
- Inputs: `account_balance`, `win_rate`, `avg_win`, `avg_loss`, `current_atr`, `average_atr`, `stop_loss_pips`, `pip_value`, optional `regime_quality`
- Output: `KellyResult` dataclass with `position_size`, `kelly_f`, `base_kelly_f`, `risk_amount`, `adjustments`, `status`

### Sync/Async
**Pure sync only** — no `async def`. Standalone math module.

### Risk if Touched
**LOW** — stateless, pure math, no external dependencies. Clean interface, high testability.

### Relation to QuantMindLib
- Primary position sizing computation engine
- `KellyResult` maps to library position sizing response
- `PropFirmPresets` maps to library configuration layer

---

## 4. Sentinel / Market Intelligence

**Path:** `src/router/sentinel.py`, `src/router/multi_timeframe_sentinel.py`, `src/router/market_scanner.py`
**Purpose:** Ingest ticks, update sensors, classify regime, detect sessions, scan opportunities

### Key Classes
- `Sentinel` — regime detection loop; 4 sensors; HMM shadow/hybrid/production
- `RegimeReport` — unified output: `regime` (TREND_STABLE/RANGE_STABLE/HIGH_CHAOS/BREAKOUT_PRIME/NEWS_EVENT/UNCERTAIN), `chaos_score`, `regime_quality`, `susceptibility`, `is_systemic_risk`, `hmm_regime`, `hmm_confidence`, `hmm_agreement`
- `ShadowLogEntry` — shadow mode prediction logging
- `MultiTimeframeSentinel` — OHLC aggregation; per-timeframe Sentinels; majority voting
- `MarketScanner` — session breakout/volatility/news/ICT scanning on schedule

### Data Flows
- Input: tick data (symbol, price, timeframe, high, low)
- Processing: sensor chain → regime classification → HMM prediction (if enabled) → ShadowLog DB persistence (if shadow)
- Output: `RegimeReport` consumed by Governor, KillSwitch, Backtester, DPR

### Async Patterns
- `async def` appears in sentinel.py (shadow mode DB persistence)
- Sync: main regime classification (no async needed for fast-path)
- Async: shadow log writing, HMM prediction

### Risk if Touched
**HIGH** — Sentinel is the market intelligence core. Changes affect regime classification, risk decisions, kill switches, and backtest filtering.

### Relation to QuantMindLib
- `RegimeReport` maps to `MarketContext` in shared object model
- `NewsSensor` state maps to `SentinelState`
- HMM predictions map to library feature outputs

---

## 5. Backtesting

**Path:** `src/backtesting/`
**Files:** 8 core files + helpers
**Purpose:** Python-based backtest engine simulating MQL5 environment; 4-mode evaluation; Monte Carlo; Walk-Forward; PBO

### Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `PythonStrategyTester` | `mt5_engine.py` | Simulates MQL5 with overloaded `iTime`/`iClose`/`iHigh`/`iLow`/`iVolume`; bar-by-bar iteration |
| `MT5BacktestResult` | `mt5_engine.py` | Result dataclass: sharpe, return_pct, drawdown, trades, equity_curve, trade_history |
| `SentinelEnhancedTester` | `mode_runner.py` | Base tester + sentinel integration + regime filtering |
| `SpicedBacktestResult` | `mode_runner.py` | Extends MT5BacktestResult with regime analytics |
| `FullBacktestPipeline` | `full_backtest_pipeline.py` | Orchestrates all 4 variants + optional MC + PBO |
| `WalkForwardOptimizer` | `walk_forward.py` | Train/test/gap windows; aggregate metrics |
| `MonteCarloSimulator` | `monte_carlo.py` | Trade-order randomization; 1000 sims default |
| `PBOCalculator` | `pbo_calculator.py` | CSCV bootstrap for overfitting detection |

### Backtest Modes (6-mode system)
1. **VANILLA** — default params, in-sample
2. **SPICED** — optimized params, in-sample + regime filtering
3. **VANILLA_FULL** — default params, full history + Walk-Forward
4. **SPICED_FULL** — optimized params, full history + Walk-Forward + regime
5. **MODE_B** — alternate symbol/timeframe stress
6. **MODE_C** — bear market / high-vol regime stress

### Evaluation Gate (SIT)
- >= 4 of 6 modes must pass
- All three thresholds: Sharpe >= 1.0, Max DD <= 15%, Win Rate >= 50%
- OOS degradation <= 15% = PASSED

### Storage
- DB: `StrategyPerformance` table (kelly_score, sharpe_ratio, max_drawdown, win_rate, profit_factor, variant, symbol, parent_id)
- Artifacts: `data/shared_assets/strategies/{family}/{source}/{id}/backtests/{variant}/`

### Async Patterns
None in backtesting core — pure synchronous pipeline. Async only at API layer (FastAPI endpoints).

### Risk if Touched
**MEDIUM** — backtesting is evaluation, not live execution. Changes affect strategy selection but not active trades.

### Relation to QuantMindLib
- `MT5BacktestResult` → maps to `EvaluationResult` in shared model
- `SpicedBacktestResult` + regime analytics → maps to extended evaluation profile
- Backtest modes → align with library evaluation paths (vanilla vs spiced)
- `BacktestReportSubAgent` → report generation maps to library evaluation bridge

---

## 6. DPR / Daily Performance Ranking

**Path:** `src/router/dpr_scoring_engine.py` (router layer), `src/risk/dpr/scoring_engine.py` (risk layer), `src/events/dpr.py`
**Purpose:** Composite bot scoring (0-100) across win rate, PnL, consistency, EV/trade

### Architecture: Dual DPR Engines
The codebase has two DPR engines — a lightweight async router-layer scorer and a full DB-backed risk-layer scorer. Both compute the same 4-component composite score but with different capabilities.

| Component | Weight |
|-----------|--------|
| win_rate | 25% |
| pnl | 30% |
| consistency | 20% |
| ev_per_trade | 25% |

### Key Types
- `DPRScoreEvent` — audit log persisted to SQLite before API ack (NFR-D1)
- `DPRComponentScores` — 4-component dataclass
- `DPRConcernEvent` — emitted when score drops >20 points week-over-week
- `DprScore` — full score dataclass with bot_id, session_id, component_scores, trade_count, max_drawdown, specialist_boost_applied

### DPR Redis Gap (CONFIRMED)
- **Scores are computed but NOT written to Redis in the router layer**
- The `DprScoringEngine` has `score_all_bots()` but no Redis publish call in the router layer
- The risk layer `DPRScoringEngine` also does not write scores to Redis
- Redis keys `dpr:score:{bot_id}` exist but are not being populated
- This is a known gap per memory index and WF1_SYSTEM_SCAN.md

### Tie-Breaking
4-level cascade: session win rate → max drawdown → trade count → magic number

### Specialist Boost
+5 for `@session_specialist` tagged bots, capped at 100

### Risk if Touched
**HIGH** — DPR drives bot lifecycle (promotion at >= 50, quarantine on concern). Changes affect bot survival.

### Relation to QuantMindLib
- `DPRScore` → maps to `BotEvaluationProfile.composite_score`
- Component scores → maps to `BotEvaluationProfile.component_scores`
- DPR events → library evaluation bridge emits same events
- DPR Redis gap → library bridge should include Redis publisher

---

## 7. Bot Registry, Tag, Variant

**Path:** `src/router/bot_manifest.py` (registry), `src/router/bot_tag_registry.py` (tags), `src/router/variant_registry.py` (variants)
**Purpose:** Bot lifecycle management — registration, tagging, genealogy

### BotRegistry
- Singleton in-memory registry, persisted to SQLite via `BotRegistryModel`
- `BotManifest` dataclass: bot_id, strategy_type, symbols, timeframes, tags, trading_mode, live_stats/paper_stats, decline_state, parent_bot_id, improvement_variant_id
- Phase 1: SCALPER + ORB only; other strategy types deprecated

### BotTagRegistry
- Auditable tag management with mutual exclusivity
- Tags: `@primal`, `@live`, `@paper_only`, `@quarantine`, `@dead`, `@session_specialist`, `@session_concern`
- Mutual exclusivity: `@primal` excludes `@paper_only`, `@quarantine`, `@dead`

### VariantRegistry
- Singleton, JSON persistence to `data/variant_registry.json`
- `BotVariant`: variant_id, parent_id, origin_bot_id, parameters, parameter_changes, monte_carlo_run_id, generation, status, DPR_score, lineage_depth
- Status: ACTIVE → OPTIMIZING → PAPER → LIVE → RETIRED

### Risk if Touched
**MEDIUM** — registry changes affect bot routing and lifecycle. Tags drive promotion/quarantine decisions.

### Relation to QuantMindLib
- `BotManifest` → maps to `BotSpec` static profile
- `BotVariant` → maps to `BotMutationProfile`
- `BotTagRegistry` → maps to library registry bridge
- Genealogy (`parent_id`) → library variant tracking

---

## 8. Kill Switches

**Path:** `src/router/kill_switch.py`, `src/router/progressive_kill_switch.py`, `src/risk/pipeline/layer3_kill_switch.py`
**Purpose:** Emergency exit, regime-aware protection, tiered safety net

### Three-Layer System

| Layer | Class | Purpose | Trigger |
|-------|-------|---------|---------|
| **Layer 1** | `KillSwitch` | Socket-based EA close-all; STOP file monitoring | Manual, file, command |
| **Layer 2** | `SmartKillSwitch` | Regime-aware smart exit (breakeven/trailing/immediate) | Chaos > 0.6, NEWS_EVENT, DD > 10% |
| **Layer 3** | `ProgressiveKillSwitch` | 5-tier protection hierarchy (GREEN→BLACK) | Account/family/system-level failures |
| **Layer 4** | `Layer3KillSwitch` | Forced exit bypassing Redis locks | Lyapunov > 0.95, RVOL < 0.5 |

### SmartKillSwitch Exit Rules
- Chaos < 0.3, DD < 10%: Wait for breakeven
- Chaos 0.3-0.6: Trailing stop to breakeven
- Chaos > 0.6 or DD > 10%: Immediate close
- NEWS_EVENT regime: Immediate close

### ProgressiveKillSwitch Tiers
- Tier 1: Bot-level circuit breaker (warning only)
- Tier 2: Strategy family failures (halt new trades at 3 failed bots or 20% loss)
- Tier 3: Account daily/weekly loss (close all at 3%/5%)
- Tier 4: EOD enforcement (21:00 UTC close, 10 trades/min max)
- Tier 5: Nuclear — chaos > 0.75, broker timeout > 30s

### Risk if Touched
**CRITICAL** — kill switches are safety systems. Changes can cause uncontrolled exits or failed exits during emergencies.

### Relation to QuantMindLib
- Library exposes safety hooks that map to kill switch triggers
- Bot specs include kill-switch dependency declarations
- `KillEvent` → library event bridge
- Progressive tiers → library safety envelope levels

---

## 9. Events System

**Path:** `src/events/`
**Files:** regime.py, chaos.py, tilt.py, session_template.py, cooldown.py, dpr.py, ssl.py
**Purpose:** Type-safe event dispatch for regime shifts, chaos events, DPR scores, SSL state transitions

### Event Types (30+)
- `RegimeShiftEvent` — regime state changes
- `ChaosEvent` — chaos threshold crossings
- `TiltPhaseEvent` — tilt detection and transitions
- `SessionTemplateEvent` — session lifecycle
- `CooldownPhaseEvent` — inter-session cooldown
- `DPRScoreEvent` / `DPRConcernEvent` — DPR changes
- `SSLCircuitBreakerEvent` — SSL state transitions

### Pattern
All events are dataclasses with typed fields, no inheritance hierarchy. Async event dispatch via Python event system.

### Risk if Touched
**LOW** — events are data containers, not logic. Changing event types affects consumers but not core logic.

### Relation to QuantMindLib
- Events → library bridge emits events to match existing event system
- Event types → library shared object model event equivalents

---

## 10. Database Models

**Path:** `src/database/models/`
**Files:** 20+ model files across: bots.py, performance.py, trading.py, market.py, hmm.py, account.py, broker_account.py, activity.py, approval_request.py, audit_log.py, chat.py, news_items.py, provider_config.py, session_checkpoint.py, trade_record.py, notification_config.py, server_config.py, bot_registry.py, wiki.py
**Purpose:** SQLAlchemy ORM models for all persistent data

### Key Models for QuantMindLib

| Model | Table | Key Fields | Relation |
|-------|-------|------------|----------|
| `StrategyPerformance` | `strategy_performance` | kelly_score, sharpe_ratio, max_drawdown, win_rate, variant, symbol, parent_id | Backtest results, genealogy |
| `BotCircuitBreaker` | `bot_circuit_breaker` | consecutive_losses, is_quarantined, magic_number, tier, state | Quarantine tracking |
| `BotLifecycleLog` | `bot_lifecycle_log` | bot_id, from_tag, to_tag, reason, timestamp | Tag transition audit |
| `BotRegistryModel` | `bot_registries` | bot_id, manifest_data (JSON) | Bot manifest persistence |
| `TradeRecord` | `trade_records` | Full trade with `layer3_events` JSON field | Forced exit audit |
| `PaperTradingPerformance` | `paper_trading_performance` | agent_id, win_rate, total_pnl, sharpe, validation_status | Paper validation |
| `HouseMoneyState` | `house_money_state` | account_id, daily_pnl, risk_multiplier, is_preservation_mode | House money tracking |
| `StrategyFamilyState` | `strategy_family_state` | family, failed_bots, total_pnl, is_quarantined | Family-level quarantine |
| `BotManifest` | `bot_manifests` | EA routing and lifecycle metadata | EA routing |
| `BotCloneHistory` | `bot_clone_history` | original_bot_id, clone_bot_id | Cloning operations |
| `DailyFeeTracking` | `daily_fee_tracking` | Fee monitoring per account per day | Fee tracking |

### Risk if Touched
**HIGH** — DB models are persistence contracts. Changing fields affects all consumers of that data.

### Relation to QuantMindLib
- Library evaluation results → `StrategyPerformance` persistence
- Bot lifecycle → `BotLifecycleLog` persistence
- DPR scores → `StrategyPerformance` persistence
- Variant lineage → `StrategyPerformance.parent_id` genealogy chain

---

## 11. Agents / Department System

**Path:** `src/agents/`
**Purpose:** AI agent orchestration — 5 departments (Research, Development, Trading, Risk, Portfolio) + Floor Manager (Opus) + department heads (Sonnet) + workers (Haiku)

### Architecture Pattern
- **BaseAgent** — Claude Agent SDK runtime with `query()` and `query_stream()`
- **Department heads** — Sonnet model, department specialization
- **Floor Manager** — Opus model, single entry point, orchestrates departments
- **Sub-agents** — Haiku model, task-specific
- **DepartmentMail** — SQLite-based inter-department messaging
- **Global tools** — registered at session start (read_skill, write_memory, search_memory, request_tool, send_department_mail)

### Workflow Integration
- WF1/WF2 triggered via `PrefectTriggerTool` or direct Prefect API
- Agent evaluation: `TestCase` / `EvaluationResult` / `EvaluationReport` framework
- Hook system: PRE_TOOL_USE, POST_TOOL_USE, STOP, SUBAGENT_START/SUBAGENT_STOP

### Key Subsystem Areas
| Area | Path | Purpose |
|------|------|---------|
| Core | `agents/core/base_agent.py` | BaseAgent runtime, ToolDefinition, AgentHooks |
| Departments | `agents/departments/` | FloorManager, heads, subagents, mail |
| Skills | `agents/skills/` | SkillManager, department_skills.py (29 skills), core_skills.py |
| Memory | `agents/memory/` | Episodic + semantic (ChromaDB) + profile + global |
| Evaluation | `agents/evaluation/` | TestCase framework, criteria protocols |
| MCP | `agents/mcp/` | MCP tool integrations (backtest, mt5_compiler, page_index, context7, sequential_thinking) |
| Tools | `agents/tools/` | broker_tools, risk_tools, trading_tools, strategy_router, memory_tools, knowledge_tools |

### Risk if Touched
**HIGH** — agent system is the orchestration layer. Changes affect task routing, workflow execution, and inter-department communication.

### Relation to QuantMindLib
- Library should be usable by agents as a tool/knowledge source
- Bot specs should be readable by agents for strategy generation
- Evaluation results should be accessible to agents for decision-making

---

## 12. Flows (Prefect Workflows)

**Path:** `flows/`
**Purpose:** Orchestrated multi-step pipelines for strategy development, evaluation, and deployment

### Workflow Inventory

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| **AlphaForgeFlow (WF1)** | `alpha_forge_flow.py` | Manual | Source → TRD → Dev → Compile → Backtest (6-mode) → Validate → Deploy to paper |
| **ImprovementLoopFlow (WF2)** | `improvement_loop_flow.py` | Continuous | Analyze → Re-backtest (20% val) → Monte Carlo → WFA → Paper → 3-day lag → Live |
| **ResearchSynthesisFlow** | `research_synthesis_flow.py` | Cron: Fri 20:00 | Pull week trades → identify declining → dispatch ResearchHead → queue TRDs |
| **WeekendComputeFlow (WF4)** | `weekend_compute_flow.py` | Cron: Sat 00:00 | Monte Carlo + HMM retrain + PageIndex + 90-day correlation |
| **FastTrackFlow** | `fast_track_flow.py` | Event | Template → Compile MQL5 → SIT only → Live (11-15 min) |
| **NodeUpdateFlow** | `node_update_flow.py` | Cron: Fri 22:00-Sun 22:00 | Sequential Contabo → Cloudzy → Desktop with rollback |

### Prefect Config
- Backend: SQLite at `flows/workflows.db`
- API URL: `http://127.0.0.1:4200/api`
- Prefect home: `.prefect/`

### Risk if Touched
**MEDIUM** — workflow changes affect strategy development pipeline. Not live trading but affects evaluation quality.

### Relation to QuantMindLib
- Library is the contract layer WF1/WF2 consume
- Bot specs are the input to WF1
- Evaluation results are the output from WF1/WF2
- Library should make workflows cleaner, not replace them

---

## 13. TRD (Trading Requirements Document)

**Path:** `src/trd/`
**Files:** schema.py, parser.py, validator.py, generator.py, storage.py
**Purpose:** Structured strategy specification from research to development

### TRDDocument Schema
- Fields: strategy_id, strategy_name, symbol, timeframe, entry_conditions, exit_conditions, position_sizing, parameters, bot_type, session_tags, news_blackout, atr_stop_period, atr_stop_multiplier
- 30+ standard EA parameters: session_mask, force_close_hour, spread_filter, trailing_stop, martingale settings, etc.
- Event types: HIGH_IMPACT_NEWS, CENTRAL_BANK, GEOPOLITICAL, ECONOMIC_DATA, EARNINGS
- Strategy types: NEWS_EVENT_BREAKOUT, RANGE_EXPANSION, VOLATILITY_SPIKE

### Relationship to QuantMindLib
- TRDDocument is the **precursor** to BotSpec — TRD is research output, BotSpec is library input
- Library should be able to consume TRDDocument and produce BotSpec
- TRD is the spec that drives composition (per memo §11)

---

## 14. MQL5 / EA Code Generation

**Path:** `src/mql5/`
**Files:** generator.py, parser.py, compiler/service.py, templates/ (26KB base template), Include/, Experts/
**Purpose:** Python → MQL5 EA code generation from TRD specifications

### Key Classes
- `MQL5Generator` — main code generator
- `DockerCompiler` — compiles MQL5 in Docker container
- `Autocorrect` — MQL5 syntax autocorrection
- `ErrorParser` — compile error parsing

### Risk if Touched
**HIGH** — code generation affects EA quality and compilation success. Part of the cTrader migration (MT5 EAs must be replaced or adapted).

### Relation to QuantMindLib
- Library provides feature composition logic
- Library generates or guides MQL5/cTrader code generation
- Feature specs → EA parameter injection

---

## 15. Data / Market Data Management

**Path:** `src/data/`
**Files:** data_manager.py, adaptive_throttler.py, dukascopy_fetcher.py, brokers/ (MT5, Mock, Binance adapters)
**Purpose:** Market data ingestion, normalization, caching, broker adapters

### Key Classes
- `DataManager` — unified market data API; hybrid fetch (live/cached/historical)
- `DukascopyFetcher` — historical tick data import
- Broker adapters: MT5 socket, mock, Binance — registry pattern for extensibility
- `SymbolSubscriptionManager` — tracks active subscriptions per bot
- `TickStreamCircuitBreaker` — hot tier tick cache (1-hour retention)

### Risk if Touched
**HIGH** — data is the foundation. Changes affect all consumers of market data (sentinel, risk, bots, backtesting).

### Relation to QuantMindLib
- Library requires normalized market data stream
- cTrader adapter → library market data bridge
- Tick/bar/OHLC data → library MarketContext inputs

---

## 16. SVSS (Shared Volume Session Service)

**Path:** `src/svss/`
**Files:** generator.py, parser.py, schema.py, storage.py, indicators/ (VWAP, RVOL, Volume Profile, MFI)
**Purpose:** Real-time indicator computation for session-aware trading

### Indicators
- `BaseIndicator` (ABC) with `compute(tick) -> IndicatorResult`
- `VWAPIndicator` — volume-weighted average price
- `RVOLIndicator` — relative volume
- `VolumeProfileIndicator` — volume distribution per price level
- `MFIIndicator` — money flow index

### Risk if Touched
**LOW** — SVSS is a clean, standalone indicator package. Changes affect indicator values but not core logic.

### Relation to QuantMindLib
- SVSS indicators → library feature family (volume, indicators)
- `IndicatorResult` → maps to `FeatureVector`
- `SVSSGenerator` → library feature computation engine

---

## 17. Library Stub (Current State)

**Path:** `src/library/`
**Files:** `base_bot.py` only (1 file, no `__init__.py`)
**Purpose:** Minimal stub — abstract bot interface for the agent system

### Current State
- `BaseBot` ABC with `get_signal(data: Dict) -> float` (-1.0 to 1.0)
- `calculate_entry_size()` using `EnhancedKellyCalculator`
- `to_dict()` for Strategy Auction (score = win_rate * (avg_win/avg_loss))
- No package structure, no contracts, no bridge layer

### What Needs to Be Built
- Full package structure: `core/`, `features/`, `archetypes/`, `adapters/`, `bridges/`
- Shared object model (BotSpec, MarketContext, etc.)
- Capability/compatibility system
- Feature families and registry
- cTrader adapter
- Bridge adapters to all existing systems

---

## 18. Strategy / Template Storage

**Path:** `src/strategy/`
**Files:** template_service.py, output.py, config.py, session_manager.py
**Purpose:** Strategy output storage, template management

### Key Classes
- `TemplateStorage` — JSON-based storage at `data/strategy_templates/`
- `TemplateService` — CRUD + search + default template initialization
- `EAOutputStorage` — EA output artifact management

### Risk if Touched
**LOW** — storage layer, minimal logic. Changes affect template persistence but not trading logic.

### Relation to QuantMindLib
- Template system → library archetype composition source
- Strategy output → library evaluation input

---

## 19. Asset Library (MCP Tools)

**Path:** `src/mcp_tools/asset_library.py`
**Purpose:** Shared MQL5 assets (indicators, strategies) via MCP tool server

### Key Classes
- `AssetInfo` dataclass — id, name, version, file_path, description, tags, parameters, category
- `AssetLibraryManager` — registry at `data/assets/registry.json`, timeframe config at `data/assets/timeframes.yaml`

### Capabilities
- `search_assets(query, category)` — search by name/description/tags
- `load_indicator(indicator_id)` — load MQL5 indicator code
- `list_indicators()`, `list_strategies()`
- `get_timeframe_config(timeframe_id)`
- `get_multi_timeframe_set(set_name)`
- `get_strategy_recommendations(strategy_type)`
- `validate_asset(asset_path)`

### Risk if Touched
**LOW** — asset management, not trading logic. Changes affect indicator/strategy loading.

### Relation to QuantMindLib
- Asset library → library feature family (indicators, strategies)
- `AssetLibraryManager` → library feature registry (existing, must be extended)
- Asset registry → library composition source

---

## Cross-Subsystem Data Flow Summary

```
Tick Ingestion
    │
    ├──► Sentinel (regime detection)
    │         ├──► RegimeReport (chaos_score, regime_quality)
    │         ├──► DPR (session health scoring)
    │         ├──► KillSwitch (news/chaos triggers)
    │         └──► Backtester (regime filtering)
    │
    ├──► Data Manager (normalization)
    │         └──► SVSS (VWAP, RVOL, Volume Profile, MFI)
    │
    └──► Governor (risk authorization)
              ├──► Commander (execution dispatch)
              ├──► Position Sizing (Kelly calculation)
              └──► BotCircuitBreaker (quarantine check)

TRD Input
    │
    ├──► BotRegistry (registration)
    ├──► WF1 Flow (research → dev → backtest)
    │         └──► 6-mode backtest → StrategyPerformance DB
    ├──► DPR Scoring (from live trade outcomes)
    └──► LifecycleManager (promotion/quarantine)
              ├──► BotTagRegistry
              └──► KillSwitch

DPR Scores
    │
    ├──► BotRegistry (live_stats/paper_stats)
    ├──► LifecycleManager (promotion at >= 50)
    └──► ⚠️ NOT WRITTEN TO REDIS (DPR Redis gap)
```