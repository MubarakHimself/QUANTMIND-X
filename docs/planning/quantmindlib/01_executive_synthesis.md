# 01 — Executive Synthesis: QuantMindLib in QUANTMINDX

## What QuantMindLib Is

**QuantMindLib** is the library layer that standardizes how trading bots, features, workflows, registries, evaluation systems, and platform adapters communicate across QUANTMINDX. It is the **shared language** — a coordinated layer with three roles:

1. **Shared object model** — schemas and contracts that all major systems can understand
2. **Bot-construction and composition grammar** — feature families, archetypes, and composition rules
3. **Bridge layer** — two-way translation between library objects and existing platform components

QuantMindLib lives at `src/library/` (currently a stub — 1 file, `BaseBot`, no `__init__.py`). The target is a full library package under `src/library/` with `core/`, `features/`, `archetypes/`, `adapters/`, and `bridges/` sub-packages.

**Internal name:** QuantMindLib (consistent with `src/library/`, no internal name divergence detected).

---

## What QuantMindLib Is NOT

- NOT a total replacement for the QUANTMINDX platform
- NOT a bag of indicators
- NOT only a cTrader wrapper
- NOT a replacement for existing sentinel ML internals
- NOT a replacement for kill-switch internals
- NOT a replacement for risk-engine internals
- NOT a replacement for DPR logic or bot registry internals
- NOT forced chart-pattern/ML services as mandatory V1 dependencies

---

## Why It Exists

The QUANTMINDX codebase has strong individual systems — risk governor, position sizing, sentinel, DPR, bot registry, backtesting, workflows — but they evolved independently and communicate through informal, ad-hoc interfaces. The memos identified this as the primary architectural risk: **semantic drift across systems** as the codebase grows.

QuantMindLib resolves this by establishing:
- A canonical shared object model (BotSpec, TradeIntent, RiskEnvelope, etc.)
- A capability-driven composition system where modules declare what they need and what they produce
- Bridge adapters that translate between library objects and existing system-native types
- A spec-first composition model (bake specs into objects, not raw code into objects)

Additionally, the platform migration from MT5 to cTrader creates a clean inflection point: instead of bridging MT5-specific patterns, the library should define platform-agnostic contracts with cTrader adapters as the concrete implementation.

---

## What Already Exists and Should Be Preserved

### Existing Strong Systems (integration targets, NOT replacement candidates)

| System | Location | Key Interface | Notes |
|--------|----------|---------------|-------|
| **Risk Governor** | `src/risk/governor.py` | `RiskMandate` (allocation_scalar, position_size, risk_mode) | Physics-based throttling, 5% exposure cap |
| **Enhanced Kelly Sizing** | `src/position_sizing/enhanced_kelly.py` | `enhanced_kelly_position_size()` | Pure math, no I/O, deterministic |
| **Sentinel** | `src/router/sentinel.py` | `RegimeReport` (regime, chaos_score, regime_quality, is_systemic_risk) | Ising+HMM dual model, shadow mode |
| **DPR Scoring** | `src/router/dpr_scoring_engine.py` + `src/risk/dpr/scoring_engine.py` | `DprScore` (0-100 composite) | 4-component weighted scoring |
| **BotRegistry** | `src/router/bot_manifest.py` | `BotRegistry` (singleton), `BotManifest` dataclass | Persisted to SQLite |
| **BotTagRegistry** | `src/router/bot_tag_registry.py` | Tag system (`@live`, `@quarantine`, `@dead`, etc.) | Mutual exclusivity enforced |
| **VariantRegistry** | `src/router/variant_registry.py` | `BotVariant` dataclass, genealogy tracking | JSON persistence |
| **BotCircuitBreaker** | `src/router/bot_circuit_breaker.py` | Quarantine thresholds by bot type (2-3 losses) | Sync to BotRegistry |
| **KillSwitch** | `src/router/kill_switch.py` | `KillSwitch.trigger()`, `KillEvent` | Socket-based, immediate/breakeven/smart/trailing |
| **ProgressiveKillSwitch** | `src/router/progressive_kill_switch.py` | 5-tier protection hierarchy | RED/BLACK alarm levels |
| **Layer3KillSwitch** | `src/risk/pipeline/layer3_kill_switch.py` | Forced exit bypassing Redis locks | Lyapunov + RVOL triggers |
| **LifecycleManager** | `src/router/lifecycle_manager.py` | `promote_to_live()`, `quarantine()`, `retire()` | DPR >= 50 threshold for promotion |
| **TRD Schema** | `src/trd/schema.py` | `TRDDocument` (strategy spec with 30+ params) | Already structured spec model |
| **Backtest Engine** | `src/backtesting/mt5_engine.py` | `PythonStrategyTester`, `MT5BacktestResult` | Python-simulated MQL5 |
| **Full Backtest Pipeline** | `src/backtesting/full_backtest_pipeline.py` | `FullBacktestPipeline.run_all_variants()` | 4 modes + MC + PBO |
| **Walk-Forward** | `src/backtesting/walk_forward.py` | `WalkForwardOptimizer` | Train/test/gap windows |
| **Monte Carlo** | `src/backtesting/monte_carlo.py` | `MonteCarloSimulator` | Trade-order randomization |
| **PBO Calculator** | `src/backtesting/pbo_calculator.py` | `PBOCalculator` | Probability of backtest overfitting |
| **WF1 (AlphaForge)** | `flows/alpha_forge_flow.py` | Prefect flow: research→TRD→Dev→Compile→Backtest→Deploy | Manual trigger |
| **WF2 (ImprovementLoop)** | `flows/improvement_loop_flow.py` | Prefect flow: analyze→re-backtest→Monte Carlo→WFA→Paper→3-day lag→Live | Continuous loop |
| **Backtest Report SubAgent** | `src/agents/departments/subagents/backtest_report_subagent.py` | Structured markdown reports | Section 6 via Haiku LLM |
| **Physics Sensors** | `src/risk/physics/` | IsingRegimeSensor, ChaosSensor, CorrelationSensor, HMMRegimeSensor, EnsembleVoter | 6-sensor regime detection |
| **SQS Engine** | `src/risk/sqs_engine.py` | Spread quality gate (SQS >= 0.50) | Hard block below 0.50 |
| **SSLCircuitBreaker** | `src/risk/ssl/circuit_breaker.py` | Per-bot state machine (LIVE→PAPER→RECOVERY→RETIRED) | Redis pub/sub async |
| **Events system** | `src/events/` | 30+ event types (regime, chaos, tilt, dpr, ssl) | Async event dispatch |
| **Asset Library** | `src/mcp_tools/asset_library.py` | `AssetLibraryManager` (JSON registry) | MQL5 indicator/strategy assets |
| **BaseBot stub** | `src/library/base_bot.py` | `BaseBot` ABC with `get_signal()`, `calculate_entry_size()` | Uses EnhancedKellyCalculator |

### Existing Partial/Fragile Systems (recover/validate before bridging)

| System | Location | Status | Gap |
|--------|----------|--------|-----|
| **StrategyType enum** | `src/router/bot_manifest.py` | Partial | Only SCALPER + ORB active in Phase 1, but 8+ legacy types exist |
| **FeatureConfig** | `src/risk/physics/hmm/features.py` | Partial | Only used within HMM; no cross-system feature registry |
| **SVSS indicators** | `src/svss/` | Good | VWAP, RVOL, Volume Profile, MFI; standalone and clean |
| **NewsBlackout** | `src/market/news_blackout.py` | Partial | Only Finnhub; no economic calendar feeding NewsSensor |

---

## The Main Implementation Direction

### Phase 1: Foundation (contracts + compatibility + bridges)
1. Establish canonical shared object model (BotSpec, BotRuntimeProfile, BotEvaluationProfile, MarketContext, SessionContext, FeatureVector, TradeIntent, RiskEnvelope, ExecutionDirective, CapabilitySpec, DependencySpec, CompatibilityRule)
2. Define capability/compatibility system — let modules declare requirements and outputs
3. Design bridge adapters for all existing systems (sentinel, risk, sizing, registry, DPR, evaluation, workflows)
4. Define cTrader adapter boundary — what is platform-aware vs platform-agnostic

### Phase 2: Core library building blocks
5. Feature families and registry (indicators, volume, orderflow, session, transforms)
6. Archetype system with base and derived types
7. Composition rules driven by specs, not raw code
8. Runtime intent flow with sync/async distinction

### Phase 3: Integration and alignment
9. Evaluation alignment — vanilla/spiced/MC/WFA paths all use same contracts
10. Workflow alignment — WF1 and WF2 both use same contracts
11. Registry/DPR/journal alignment — DPR writes to Redis (fix gap), journal uses library objects

### Phase 4: Deferred roadmap
12. Pattern service (out of core scope, reserved as future service candidate)
13. Advanced archetype derivation
14. Cross-platform adapter framework

### Key Principle
**Bots are thin.** They describe setups, confirmations, feature usage, and entry requests. They do NOT own risk, news logic, portfolio-wide controls, or kill-switch behavior. Those remain platform responsibilities delivered through bridges.

---

## Traceability Index

| Statement | Source |
|-----------|--------|
| Library = contract + bridge + composition layer | Memo: quantmindx_library_v1_architecture_memo.docx §1 |
| Bots must be thin | Memo: quantmindx_library_v1_architecture_memo.docx §4 |
| Platform = cTrader, broker = IC Markets Raw | Memo: trading_platform_decision_memo.docx + architecture_memo §16 |
| 3 role split: shared object model, bot grammar, bridge layer | Memo: quantmindx_library_v1_architecture_memo.docx §1 |
| Sync/async distinction required | Memo: quantmindx_library_v1_architecture_memo.docx §6 |
| DPR Redis gap confirmed | Memory: WF1_SYSTEM_SCAN.md + code: `src/router/dpr_scoring_engine.py` |
| Phase 1 = SCALPER + ORB only | Memory: MARKET_INTELLIGENCE_AND_BOT_REGISTRY_SESSION.md |
| 6-mode backtest evaluation | Code: `flows/alpha_forge_flow.py`, `src/backtesting/mode_runner.py` |
| BaseBot stub exists at `src/library/base_bot.py` | Code: actual file inspection |
| No formal FeatureRegistry | Code: FeatureConfig only in HMM, no cross-system registry |
| No archetype system | Code: StrategyType enum only |
| WF1/WF2 are canonical evaluation paths | Code: `flows/` directory + `WF1_SYSTEM_SCAN.md` |