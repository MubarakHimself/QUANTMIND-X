# 16 — Risks, Assumptions, and Open Questions

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Document blockers, assumptions, outdated-doc risks, coupling risks, and areas requiring human review

---

## 1. Genuine Blockers

### BLOCKER-1: Two Governor Classes Naming Conflict

**Issue:** The codebase has two separate `Governor` classes:
- `src/router/governor.py` — risk authorization (compliance layer)
- `src/risk/governor.py` — risk execution (Tier 2 rules)

Both are involved in the risk pipeline. The library must reference one or both, but the naming conflict is a source of confusion.

**Impact:** Bridge implementations may reference the wrong Governor. RiskEnvelope field names may differ between the two.

**Resolution needed:** Rename one class or establish canonical import path before finalizing RiskEnvelope schema.

**Status:** Unresolved
**Reference:** Architecture doc §C1, Recovery note R-10

---

### BLOCKER-2: DPR Dual Engines (Router + Risk Layers)

**Issue:** Two DPR engines exist with different capabilities. Router DPR engine writes `dpr:score:{bot_id}` to Redis (fix is in uncommitted code). Risk DPR engine writes `session_concern:{magic_number}` counters but NOT DPR composite scores to Redis. Downstream systems expecting Redis DPR scores from the risk layer will fail.

**Impact:** DPR bridge may get inconsistent scores depending on which engine responds. Risk layer Redis gap means real-time DPR monitoring from risk layer doesn't work.

**Resolution needed:** Extend risk-layer DPR engine Redis publish OR establish router engine as the canonical DPR score publisher. DPR bridge must unify both paths.

**Status:** Partially resolved — router engine fix in uncommitted code; risk layer extension still needed.
**Reference:** Gap G-18, Recovery note R-2

---

### BLOCKER-3: cTrader Open API Capability Verification

**Issue:** The library design assumes cTrader Open API supports: tick streams, bar streams, depth streams, historical data, order execution, and account state. These assumptions are based on documentation, not verified implementation.

**Impact:** If cTrader Open API doesn't support all required operations, adapter design may need significant revision.

**Resolution needed:** Verify cTrader Open API Python SDK capabilities before Phase 4 implementation. Review official docs: https://help.ctrader.com/open-api/

**Status:** Unresolved — assumption based on docs
**Reference:** `09_ctrader_boundary_plan.md` (Section 1), Memo Appendix A

---

### BLOCKER-4: BotSpec → Strategy Code Generation Reliability

**Issue:** The evaluation bridge requires BotSpec → strategy code conversion. The generated strategy code must produce valid backtest results. This conversion is complex and may fail for edge cases.

**Impact:** Complex BotSpecs may not convert to working strategy code. Evaluation pipeline may produce unreliable results.

**Resolution needed:** Test BotSpec → strategy code generation extensively before Phase 8. Establish fallback (manual strategy code input).

**Status:** Unresolved
**Reference:** SPEC-007 (EvaluationBridge), SPEC-001 (BotSpec)

---

## 2. Assumptions to Verify

### ASSUMPTION-1: TRD → BotSpec Loss-Free Conversion

**Assumption:** TRDDocument can be converted to BotSpec without losing information.

**Evidence:** TRDDocument has 30+ standard parameters, session_tags, entry/exit conditions. BotSpec is designed to absorb all of this. But some TRD fields may map imperfectly.

**Verification needed:** Compare TRD schema fields to BotSpec fields. Identify any unmapped fields.

**Reference:** CONTRACT-025 (TRD → BotSpec ticket), `src/trd/schema.py`

---

### ASSUMPTION-2: cTrader Backtest Schema Compatibility

**Assumption:** CTrader backtest results can match MT5BacktestResult schema exactly.

**Evidence:** Both are Python dataclasses with same fields. But cTrader may report metrics differently (e.g., Sharpe calculation, equity curve granularity).

**Verification needed:** Test early with sample cTrader data. If schema mismatch exists, document delta and adjust evaluation pipeline.

**Reference:** SPEC-011 (CTrader backtest engine), `src/backtesting/mt5_engine.py`

---

### ASSUMPTION-3: Order Flow Data Quality from cTrader

**Assumption:** cTrader provides sufficient tick/depth data for quality-aware order flow features.

**Evidence:** cTrader Open API documentation shows tick, bar, and depth support. But buy/sell volume delta may not be directly available.

**Verification needed:** Confirm cTrader provides buy/sell volume data for VolumeImbalanceFeature. If not, quality tagging degrades to MEDIUM/LOW.

**Reference:** `09_ctrader_boundary_plan.md` (Section 8: Order Flow Data Availability)

---

### ASSUMPTION-4: SVSS Indicators Can Be Wrapped

**Assumption:** SVSS indicators (VWAP, RVOL, Volume Profile, MFI) can be wrapped as library feature modules without behavioral changes.

**Evidence:** SVSS indicators are standalone and clean (BaseIndicator ABC, compute(tick) → IndicatorResult). Wrapping should be straightforward.

**Verification needed:** Confirm SVSS indicators have no hard dependencies on MT5-specific data structures. Test wrapped versions match original outputs.

**Reference:** SPEC-006 (FeatureRegistry), Recovery note R-11

---

### ASSUMPTION-5: Phase 1 ORB-Only Scope

**Assumption:** V1 implementation focuses on OpeningRangeBreakout archetype only. Other archetypes (breakout_scalper, pullback_scalper, mean_reversion, session_transition) are Phase 2+.

**Evidence:** Phase 1 strategy types are SCALPER + ORB only per memory index. 60/40 bot mix (60% scalping, 40% ORB).

**Verification needed:** Confirm Phase 1 scope is ORB only. Ensure implementation agents don't implement other archetypes prematurely.

**Reference:** `06_target_architecture_v1.md` (Section 10: Out-of-Scope), Memory: MARKET_INTELLIGENCE_AND_BOT_REGISTRY_SESSION.md

---

## 3. Outdated Documentation Risks

### DOC-RISK-1: docs/architecture.md Points to _bmad-output/

**Issue:** `docs/architecture.md` explicitly states it is stale and points to `_bmad-output/planning-artifacts/architecture.md` as authoritative. Any agent reading `docs/architecture.md` will get outdated info.

**Impact:** Implementation agents may read stale docs and make wrong decisions.

**Recommendation:** Add redirect note at top of `docs/architecture.md`: "This document is stale. See `_bmad-output/planning-artifacts/architecture.md`." Or mark as deprecated and point to the new location.

**Reference:** `03_documentation_status_report.md`

---

### DOC-RISK-2: DATA_ARCHITECTURE_ANALYSIS.md Focuses on MT5

**Issue:** `DATA_ARCHITECTURE_ANALYSIS.md` (2026-02-12) documents MT5 streaming (100ms polling), DataManager hybrid fetch, tick-level backtesting gaps. This is from before the cTrader decision.

**Impact:** Any agent using this doc for data architecture will design for MT5, not cTrader.

**Recommendation:** Mark as deprecated or update for cTrader data model.

**Reference:** `03_documentation_status_report.md`

---

### DOC-RISK-3: IMPLEMENTATION_ORCHESTRATION.md Kanban Items

**Issue:** `IMPLEMENTATION_ORCHESTRATION.md` contains Kanban items (MUB-151+) from early March 2026. These are likely complete or outdated.

**Impact:** Implementation agents may attempt to work on completed items.

**Recommendation:** Update doc with completion status or archive completed items.

**Reference:** `03_documentation_status_report.md`

---

### DOC-RISK-4: No Formal Architecture Decision Records

**Issue:** The codebase has no formal ADR (Architecture Decision Record) format. Decisions are captured in `architecture.md` (§1-§21) and handoff docs. This makes it hard to trace why a decision was made.

**Impact:** Future agents may not understand why certain decisions were made.

**Recommendation:** Continue documenting decisions in `_bmad-output/planning-artifacts/architecture.md`. Consider formal ADR format in Phase 2.

**Reference:** `03_documentation_status_report.md` (ADR section)

---

## 4. Coupling Risks

### COUPLING-1: Sentinel Regime Classification Tightly Coupled

**Issue:** Regime classification is centralized in Sentinel. If Sentinel changes (new sensors, new ensemble voting), all consumers (Governor, KillSwitch, DPR, Backtester) may see different regime data.

**Impact:** Changes to Sentinel may cascade to risk decisions, kill switch behavior, and evaluation results.

**Mitigation:** Library's MarketContext is a snapshot at decision time. Bridge isolates changes. But regime classification changes must be tested against all consumers.

**Reference:** Recovery note R-1

---

### COUPLING-2: Governor Risk Logic Dependencies

**Issue:** Governor has many dependent calculations (physics throttling, correlation check, 5% exposure cap, EnhancedKelly). Changing any of these affects the entire risk pipeline.

**Impact:** Library RiskBridge must understand Governor internals to map correctly.

**Mitigation:** RiskBridge maps to RiskEnvelope (output contract), not Governor internals. But Governor behavior changes must be reflected in RiskBridge.

**Reference:** Recovery note R-10

---

### COUPLING-3: BotLifecycleLog Audit Trail

**Issue:** Tag transitions and lifecycle events are logged to BotLifecycleLog. Library lifecycle decisions (promote/quarantine) must log to the same table. If library lifecycle bridge uses different logging patterns, audit trail is inconsistent.

**Impact:** Audit trail may have gaps or inconsistent entries.

**Mitigation:** Library LifecycleBridge should use same logging patterns as existing LifecycleManager.

**Reference:** Recovery note R-13

---

### COUPLING-4: TRD Schema Drift

**Issue:** TRDDocument is the primary input to the library. If TRD schema changes (new fields, renamed fields), BotSpec conversion breaks.

**Impact:** WF1 may produce invalid BotSpecs without clear error.

**Mitigation:** TRD → BotSpec conversion should validate input and raise clear errors on unexpected fields. Version TRD schema.

**Reference:** CONTRACT-025 (TRD → BotSpec conversion spec)

---

## 5. Platform Migration Risks

### MIGRATION-1: MT5 → cTrader Surface Area

**Issue:** MT5 is embedded throughout: market data adapters, risk integrations, kill switch pipeline, backtest engine, MQL5 code generation. Migration surface is large (5 areas identified).

**Impact:** Migration effort is significant. Risk of breaking live trading during transition.

**Mitigation:** Adapter pattern isolates platform code. Library core doesn't change. Parallel running (MT5 + cTrader) during transition.

**Reference:** `09_ctrader_boundary_plan.md` (Section 9: MT5 Migration Surface), Gap G-10, G-11

---

### MIGRATION-2: MQL5 → cTrader Algo Code

**Issue:** `src/mql5/` generates MQL5 EA code from TRD. cTrader uses Python/C# for algos. The code generation pipeline must be replaced or adapted.

**Impact:** Strategy code generation is a core part of WF1. Migration must not break the workflow.

**Mitigation:** Create `src/library/adapters/ctrader/ea_generator.py` that generates cTrader-compatible code. TRD input remains the same.

**Reference:** Gap G-11 (cTrader migration)

---

### MIGRATION-3: T1 Node Windows Dependency

**Issue:** Trading node (T1 on Kamatera) runs Windows for MT5. cTrader also runs on Windows. The trading node OS doesn't change, but the terminal does.

**Impact:** Adapter deployment on T1 must be Windows-compatible. No Linux trading node for cTrader.

**Mitigation:** Adapter code must be Windows-compatible. Test adapter on Windows before deployment.

**Reference:** `09_ctrader_boundary_plan.md` (Section 9)

---

## 6. Areas Requiring Human Review

### REVIEW-1: DPR Dual Engine Resolution

**Decision needed:** Should the dual DPR engines be consolidated into one, or should the library bridge handle both paths?

**Options:**
- **A:** Consolidate into single DPR engine (higher effort, cleaner)
- **B:** Library bridge handles both (existing code untouched, bridge complexity higher)

**Current state:** Option B (bridge handles both) is the current plan. Human review needed to confirm.

**Reference:** Gap C2, Recovery note R-2

---

### REVIEW-2: Governor Naming Conflict Resolution

**Decision needed:** Should one of the two Governor classes be renamed to avoid confusion?

**Options:**
- **A:** Rename `src/router/governor.py` → `RouterGovernor` or `RiskAuthorizer`
- **B:** Rename `src/risk/governor.py` → `RiskGovernor` or `RiskCalculator`
- **C:** Keep names but document import paths explicitly in library

**Current state:** Option C (document import paths) is the current plan. Human review needed to confirm.

**Reference:** Architecture doc §C1, Recovery note R-10

---

### REVIEW-3: Economic Calendar Integration Path

**Decision needed:** Should economic calendar integration go through cTrader Network Access (future) or continue with direct Finnhub polling (current)?

**Options:**
- **A:** Continue Finnhub direct polling (current, works)
- **B:** Migrate to cTrader Network Access (Phase 2+)
- **C:** Hybrid (Finnhub for backup, cTrader Network Access for primary)

**Current state:** Option A (Finnhub direct) is the current plan for V1. Human review needed to confirm.

**Reference:** `09_ctrader_boundary_plan.md` (Section 11: Network Access Feature), Recovery note R-12

---

### REVIEW-4: TRD → BotSpec Conversion Scope

**Decision needed:** Should TRD → BotSpec conversion be a strict transformation (only fields that map cleanly) or a lenient transformation (all TRD fields → BotSpec, even if imperfectly)?

**Options:**
- **A:** Strict: only map fields that have clean 1:1 mapping. TRD fields without clear mapping are discarded.
- **B:** Lenient: map all TRD fields, even if mapping is approximate. Preserve all information.

**Current state:** Option B (lenient) is implied by "loss-free" requirement in contracts. Human review needed to confirm.

**Reference:** CONTRACT-025 (TRD → BotSpec), ASSUMPTION-1

---

### REVIEW-5: Order Flow Feature Quality Threshold

**Decision needed:** What minimum quality level should order flow features degrade to before being disabled?

**Options:**
- **A:** LOW quality enabled (use session-relative proxies)
- **B:** MEDIUM quality minimum (OHLCV approximation only)
- **C:** HIGH quality only (cTrader DOM required, else disabled)

**Current state:** Not defined. Human review needed to set threshold.

**Reference:** `09_ctrader_boundary_plan.md` (Section 8), Memo §5 (quality-aware design)

---

### REVIEW-6: Phase 1 Bot Count

**Decision needed:** How many bots should be targeted for Phase 1 (first library-integrated bots)?

**Options:**
- **A:** 1 bot (single ORB bot for testing)
- **B:** 3-5 bots (small set for validation)
- **C:** All active bots migrated

**Current state:** Not defined. Human review needed.

**Reference:** Phase 6 (archetype system) and Phase 7 (runtime) — implementation scope depends on this

---

## 7. Open Questions (No Decision Needed Yet)

### Q-1: Backtest Historical Data Source

**Question:** Where does cTrader backtest engine get historical data? From broker (cTrader) or from external source (Dukascopy, etc.)?

**Context:** Current backtest engine (`mt5_engine.py`) simulates MQL5 with Python. cTrader backtest engine needs historical OHLCV data.

**Reference:** SPEC-011, `src/data/dukascopy_fetcher.py`

---

### Q-2: Multi-Timeframe Sentinel in Library

**Question:** Should the library's MarketContext support multi-timeframe regime (current single-timeframe)? Or should MultiTimeframeSentinel be wrapped as a library feature?

**Context:** `MultiTimeframeSentinel` maintains separate Sentinels per timeframe with voting aggregation. The library's MarketContext currently assumes single timeframe.

**Reference:** Recovery note R-1 (MultiTimeframeSentinel)

---

### Q-3: PatternSignal Schema Completeness

**Question:** Is PatternSignal placeholder schema sufficient for future integration, or should it be more detailed now?

**Context:** PatternSignal is out of V1 scope but defined as placeholder in shared object model. Schema is minimal.

**Reference:** `07_shared_object_model.md` (PatternSignal), Memo §9

---

### Q-4: BotSpec Versioning Strategy

**Question:** Should BotSpec have explicit version fields (v1, v2) or implicit versioning (hash-based)?

**Context:** BotSpec is a structured spec that may evolve. Versioning strategy affects backward compatibility.

**Reference:** SPEC-001 (BotSpec), SPEC-024 (SpecRegistry)

---

### Q-5: Feature Confidence Threshold Default

**Question:** What should be the default `feature_confidence.quality` threshold for enabling a feature in a bot?

**Context:** Quality-aware tagging requires thresholds. Default must work for IC Markets Raw broker.

**Reference:** SPEC-006 (FeatureRegistry), `10_feature_family_plan.md`

---

## 8. Risk Summary Matrix

| Risk ID | Category | Description | Severity | Status |
|---------|----------|-------------|----------|--------|
| BLOCKER-1 | Naming conflict | Two Governor classes | HIGH | Unresolved |
| BLOCKER-2 | DPR architecture | Dual engines, risk layer no Redis write | HIGH | Partially resolved |
| BLOCKER-3 | Platform capability | cTrader API not verified | HIGH | Unresolved |
| BLOCKER-4 | Conversion reliability | BotSpec → strategy code | MEDIUM | Unresolved |
| ASSUMPTION-1 | Schema mapping | TRD → BotSpec loss-free | MEDIUM | Needs verification |
| ASSUMPTION-2 | Schema compatibility | cTrader vs MT5 backtest | HIGH | Needs verification |
| ASSUMPTION-3 | Data quality | cTrader order flow data | MEDIUM | Needs verification |
| ASSUMPTION-4 | Code wrapping | SVSS indicator wrapping | LOW | Needs verification |
| ASSUMPTION-5 | Scope | Phase 1 ORB-only | LOW | Needs confirmation |
| DOC-RISK-1 | Stale docs | docs/architecture.md | LOW | Needs fix |
| DOC-RISK-2 | Stale docs | DATA_ARCHITECTURE_ANALYSIS.md | MEDIUM | Needs fix |
| DOC-RISK-3 | Stale docs | IMPLEMENTATION_ORCHESTRATION.md | LOW | Needs fix |
| DOC-RISK-4 | Missing docs | No formal ADRs | LOW | Future |
| COUPLING-1 | Tight coupling | Sentinel regime classification | MEDIUM | Acceptable |
| COUPLING-2 | Complex logic | Governor risk calculations | MEDIUM | Acceptable |
| COUPLING-3 | Audit consistency | BotLifecycleLog logging | MEDIUM | Needs attention |
| COUPLING-4 | Schema drift | TRD schema changes | MEDIUM | Needs mitigation |
| MIGRATION-1 | Large surface | MT5 → cTrader migration | HIGH | Acceptable (phased) |
| MIGRATION-2 | Code generation | MQL5 → cTrader algo | HIGH | Acceptable (phased) |
| MIGRATION-3 | Platform | T1 Windows dependency | MEDIUM | Acceptable |

---

## 9. Pre-Implementation Checklist

Before Phase 1 begins, verify:
- [ ] DPR Redis gap confirmed with code inspection (router fix verified in uncommitted; risk layer gap confirmed)
- [ ] Two Governor classes mapped with correct import paths
- [ ] cTrader Open API Python SDK reviewed for capability completeness
- [ ] TRD schema fields mapped to BotSpec fields
- [ ] MT5 backtest result schema documented for cTrader compatibility
- [ ] Stale docs marked with redirect notes
- [ ] Phase 1 scope (ORB-only) confirmed with stakeholder
- [ ] Economic calendar integration path confirmed (Finnhub direct vs cTrader Network Access)
- [ ] DPR dual engine handling strategy confirmed