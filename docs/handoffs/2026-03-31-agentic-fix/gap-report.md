## Agentic Planning Gap Report

### Key Requirements (from planning docs)
- **Phase-1 thesis (Scalping + ORB, IC Markets, volume edge, 3/5/7 risk, no leverage).** The March 2026 planning document insists on a narrow strategy family, a single IC Markets ECN feed, and a new account-level risk framework (daily 3 %, weekly 5 %, circuit breakers, 3-loss rule) before any live trading starts. See `claude-desktop-workfolder/QuantMindX_Planning_Document_March2026.docx` §§2–7.
- **Two distinct workflows with dataset-level firewalls.** Workflow 1 (AlphaForge) must run on a frozen train copy; Workflow 2 must operate on validation/test copies, enforce paper-trading gates, and feed structured reports at every stage (Stage 1–5 report Q1‑Q20). See the same doc §§3‑6.
- **Agentic architecture extensions.** Live Monitor subagent, symbol affinity routing, regime-conditional strategy pools, shared indicator library (SVSS), Session-Scoped Kelly modifiers, SVSS → CorrelationSensor → Governor plumbing, Spread Quality Score (SQS), and Layer 2/3 safety layers must be wired into the system. See `QuantMindX_Planning_Addendum_Session2_March2026.docx` §§3–14.
- **Risk stack enhancements.** RVOL-weighted sizing, portfolio correlation penalties, session Kelly multipliers, economic-calendar gating, and SQS overrides belong inside the Governor/EnhancedGovernor to control trades. See Addendum §§10–13.
- **Shared knowledge assets.** The shared layer must expose SVSS data, Regime Playbook guidance, broker calibration numbers, and the bot lifecycle report archive so downstream agents and humans can learn from Stage 1‑5 outputs. See base doc §8 and Addendum references to Regime Playbook/SVSS/SQS.

### Implemented Paths
- **Live Monitor subagent comparing live vs backtest/paper metrics.** `src/agents/departments/subagents/live_monitor_subagent.py` reads performance history, flags WR deltas, and notifies DPR, satisfying the detection/reporting requirement from the planning doc §4.4.
- **Layer‑2 Position Monitor service and Layer‑3 kill switch hooks.** `src/router/position_monitor.py` was built for the Cloudzy-low-latency SL adjustments described in Addendum §3, and `RiskGovernor` wires it via `register_position_with_monitor`.
- **SVSS shared indicator service (VWAP/RVOL/Volume Profile/MFI).** `src/risk/svss_service.py` calculates indicators and publishes to Redis, giving downstream consumers the shared data demanded in Addendum §7.
- **RVOL-weighted sizing, correlation penalties, and portfolio guards.** `src/risk/governor.py` multiplies Kelly by RVOL, uses `CorrelationSensor`, and injects portfolio scaling; `src/risk/sizing/session_kelly_modifiers.py` and `src/router/enhanced_governor.py` provide session-scoped house-money behavior, matching Addendum §§10–12.
- **Spread Quality Score (SQS) system.** `src/risk/sqs_engine.py` (plus the corresponding API/store/front-end tiles) captures the historical spread profile and can block new entries as described in Addendum §13.
- **3‑day HMM feedback lag.** `src/router/hmm_lag_buffer.py` enforces the calendar lag requirement from Addendum §14.1.
- **IC Markets broker profile seeded.** `scripts/populate_broker_registry.py` creates/updates the `icmarkets_raw` entry with the required spreads, commissions, and pip values, satisfying §2.3 of the planning doc.
- **Floor Manager UI uses `/api/chat/floor-manager/message`.** The TradingFloorPanel/CopilotPanel combo now hits the canonical agent endpoint (see `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte#L520-L780`), so the documented routing bug is no longer present.

### Gap Summary
1. **Routing matrix still models the old swing/structural account mix (High).**  
   - Doc requirement: Phase 1 must run only scalping + ORB, and BREAKOUT_PRIME should route into ORB (planning doc §2.2 and Addendum §4.1).  
   - Current state: `Commander.regime_strategy_map` still maps `TREND_STABLE` to structural/HFT, and `RoutingMatrix`only exposes “Machine Gun” (RoboForex, scalpers) and “Sniper” (Exness, STRUCTURAL/SWING). No IC Markets account or ORB-specific strategy types exist (`src/router/routing_matrix.py#L40-L124`).  
   - Impact: Strategy selection still permits outdated families, and RouteMatrix cannot enforce the new session masks or symbol affinities described in Addendum §4.  
2. **Symbol affinity metadata is ignored because Commander expects legacy string values (Medium).**  
   - Doc requirement: `BotManifest.symbol_affinity` is now a dict (0.0‑1.0) and should gate activation based on the IC Markets scanner, not plain tags (Addendum §8.3).  
   - Current state: `BotManifest` defines the dict, but `Commander.run_auction` still filters/sorts using `'preferred'`/`'exclude'` strings (`src/router/commander.py#L884-L936`), so affinity scores never influence dispatch.  
   - Impact: The new manifest data cannot change routing, so symbol-aware pool gating cannot work.  
3. **No dataset-copy firewall enforces the train/validation/test split per workflow (High).**  
   - Doc requirement: Workflow 1 must operate on a frozen train copy, Workflow 2 must re-backtest on a separate validation copy, and final decisions must never touch the live dataset (planning doc §6.1).  
   - Current state: Backtesting scripts (e.g., `src/backtesting/walk_forward.py#L158-L170`) slice global tables via `data.iloc[...]` without creating time-bounded clones, and no central service handles dataset distillation for agents.  
   - Impact: Agents can accidentally reuse the same DuckDB dataset across workflows, opening a path for look-ahead bias and undermining the paper‑trading gate.  
4. **Bot lifecycle reports / Regime Playbook are still missing (Medium).**  
   - Doc requirement: Stage 1‑5 reports (Q1–Q20) plus a Regime Playbook must be accessible to trading/risk agents so knowledge persists across workflows (planning doc §4, Addendum §8).  
   - Current state: The only report infrastructure is the generic `report_writer` skill (`src/agents/skills/builtin_skills.py` and `types.py`)—no Stage 1‑5 schema or Regime Playbook artifacts exist, and search for Q1…Q20 returns nothing.  
   - Impact: Live Monitor disagnostics, Research, and Risk departments have no structured document to answer the mandated questions, hampering the decline/recovery loop.

### Recommended Fixes
- **Align RoutingMatrix & Commander with the scalping + ORB thesis.**  
  - Update `StrategyType`/`AccountType` to include `SCALPER_LONG/SHORT/NEUTRAL` and `ORB_LONG/SHORT/FALSE_BREAKOUT`, and map them to IC Markets-based accounts (Machine Gun = scalping, Sniper = ORB).  
  - Adjust `Commander.regime_strategy_map` so `BREAKOUT_PRIME` only enables ORB pools and `TREND_STABLE` no longer unlocks STRUCTURAL/SWING.  
  - Inject the session mask (00:00-16:00 active, block 16:00-22:00) described in Addendum §5 before auctioning bots.  
  - Reference: `[src/router/routing_matrix.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/routing_matrix.py#L40-L124)` and `[src/router/commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py#L32-L88)` for the existing layout.  
- **Interpret symbol affinity as the new dict.**  
  - Update `Commander._filter_bots_for_symbol` (or equivalent) to read `BotManifest.symbol_affinity` scores, treat affinity ≥ 0.7 as preferred, < 0.3 as excluded, and use the underlying numeric value for ranking.  
  - Ensure the manifest loader (`BotManifest.__post_init__`) and routing matrix cooperate on this field.  
  - Reference: `[src/router/bot_manifest.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/bot_manifest.py#L400-L520)` and `[src/router/commander.py](/home/mubarkahimself/Desktop/QUANTMINDX/src/router/commander.py#L884-L936)`.  
- **Build a dataset-copy guard for both workflows.**  
  - Introduce a `DatasetCopyManager` (DuckDB) that creates read-only snapshots per workflow stage, enforces train/validation/test time boundaries, and exposes APIs for AlphaForge vs Workflow 2 tools.  
  - Have every backtest/re-backtest command pull from this manager instead of slicing the shared dataset directly (`src/backtesting/walk_forward.py#L158-L170` as the current hotspot).  
  - Log and audit the copy metadata so the next improvement loop can prove no look-ahead leakage occurred.  
- **Ship the Stage 1–5 report schema + Regime Playbook into shared assets.**  
  - Define a structured `BotVariantReport` entity that answers Q1–Q20, persists per variant, and is queryable from the Live Monitor, Risk, and Research agents.  
  - Create the Regime Playbook document (regime → approved families/filters) inside the shared asset layer (ChromaDB/knowledge base) and surface it via a lightweight endpoint.  
  - Tie Live Monitor alerts and Improvement loops to these reports so the decline/recovery workflow can cite concrete Stage 2/3 deltas.  

