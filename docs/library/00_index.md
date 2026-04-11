# QuantMindLib V1 — Library Documentation

**Status:** Implementation complete through Phase 10
**Last reconciled:** 2026-04-11
**Branch:** `claude/elegant-turing` (synced with `main`)

---

## Documentation Pack Contents

| # | File | Purpose |
|---|------|---------|
| 1 | `01_library_overview.md` | What QuantMindLib is, is not, why it exists, current phase |
| 2 | `02_current_architecture.md` | Actual package layout, runtime boundaries, implementation status |
| 3 | `03_public_contracts_and_objects.md` | All domain objects, contracts, schemas with code locations |
| 4 | `04_feature_system.md` | Feature registry, families, classification, bootstrap |
| 5 | `05_proxy_microstructure_layer.md` | Proxy microstructure semantics, features, data flow |
| 6 | `06_bridge_and_integration_boundaries.md` | Bridge inventory, ownership, what stays outside |
| 7 | `07_runtime_and_decision_flow.md` | Runtime orchestration, feature evaluation, intent emission |
| 8 | `08_error_logic.md` | V1 error hierarchy, AuditRecord, rejection semantics |
| 9 | `09_evaluation_and_workflows.md` | Evaluation orchestrator, workflow bridges, stubs |
| 10 | `10_coder_handoff.md` | How a coding agent should approach the codebase |
| 11 | `11_known_gaps_and_deferred_items.md` | Real gaps vs stale tracker items vs deferred |
| 12 | `12_change_log_current_state.md` | What was completed, corrections discovered |
| 13 | `13_reading_order.md` | Exact reading order for next Codex session |

## Quick Status

```
Phase 0  (Repo Scan + Docs)      ✓ COMPLETE
Phase 1  (Core Contracts/Schemas) ✓ COMPLETE
Phase 2  (Capability/Compatibility) ✓ COMPLETE
Phase 3  (Bridge Definitions)     ✓ COMPLETE
Phase 4  (cTrader Adapter)        STUB (adapters not implemented)
Phase 5  (Feature Registry V1)    ✓ COMPLETE
Phase 6  (Archetype System V1)   ✓ COMPLETE
Phase 7  (Runtime Intent Flow)   ✓ COMPLETE (not TODO as tracker says)
Phase 8  (Evaluation Alignment)   ✓ COMPLETE (not TODO as tracker says)
Phase 9  (Workflow Alignment)     ✓ COMPLETE (not TODO as tracker says)
Phase 10 (Registry/DPR/Journal)  ✓ COMPLETE (not TODO as tracker says)
Phase 11 (Deferred Roadmap)      DEFERRED (as planned)
         Error Logic (ERR-001-004) ✓ COMPLETE
```

**Tracker drift:** The task tracker (`14_ticket_backlog.md`) marks Phases 7-10 as TODO. These phases are actually fully implemented. The tracker labels are stale.

## Key Directories

```
src/library/                          # Library root
├── core/
│   ├── domain/                       # 14 domain objects (BotSpec, MarketContext, etc.)
│   ├── composition/                  # CapabilitySpec, DependencySpec, compatibility
│   ├── bridges/                      # 9 bridge files (DPR, Sentinel, Risk, Registry, etc.)
│   ├── errors/                       # V1 error hierarchy (LibraryError, BridgeError, etc.)
│   └── types/                        # All enums (RegimeType, TradeDirection, etc.)
├── features/                         # 16 feature modules across 6 families
│   ├── base/                         # FeatureModule ABC
│   ├── indicators/                   # RSI, ATR, MACD, VWAP
│   ├── volume/                       # RVOL, MFI, VolumeProfile
│   ├── microstructure/               # Spread, TOB, Aggression, Absorption, etc.
│   ├── orderflow/                    # SpreadBehavior, DOMPressure, DepthThinning
│   ├── session/                      # SessionDetector, SessionBlackout
│   └── transforms/                   # Normalize, Rolling, Resample
├── archetypes/                       # Archetype system + Composer
├── runtime/                          # FeatureEvaluator, IntentEmitter, BotStateManager, SafetyHooks
├── evaluation/                       # EvaluationOrchestrator, StrategyCodeGenerator
├── workflows/                        # WF1Bridge, WF2Bridge, stub flows
└── ctrader/                          # EMPTY — adapter not yet implemented
```

## External System Boundaries

The library bridges to these existing systems:

| System | Location | Role |
|--------|----------|------|
| Sentinel | `src/router/sentinel.py` | Regime classification (HMM, sensors) |
| DPR | `src/router/dpr_scoring_engine.py`, `src/risk/dpr/` | DPR scoring engines (dual) |
| Risk/Governor | `src/router/governor.py`, `src/risk/governor.py` | Risk authorization |
| Registry | `src/router/bot_manifest.py`, `src/router/bot_tag_registry.py` | Bot lifecycle |
| DPR Redis | Redis | DPR score persistence |
| Backtesting | `src/backtesting/full_backtest_pipeline.py` | Evaluation pipeline |
| Prefect Flows | `flows/alpha_forge_flow.py` | WF1 (stubs) |
| Events | `src/events/` | DPR, SSL, regime events |

## Integration Notes

- **Library owns:** domain contracts, features, archetypes, runtime logic, bridges (translation), evaluation orchestration
- **Existing systems own:** DPR internals, governor logic, kill switches, MT5/cTrader platform adapters, backtest engine internals, ML models (HMM, BOCPD, MS-GARCH)
- **DPR dual engines:** router + risk layer, handled by `DPRDualEngineRouter`
- **Two Governor classes:** `src/router/governor.py` (authorization) and `src/risk/governor.py` (Tier 2 rules) — referenced by module path, not class name
- **EnsembleVoter:** exists but NOT wired into live tick path (Phase 2+ deferred)

## Deferred Items (Phase 11)

1. **cTrader real adapter** — `src/library/ctrader/` is empty; real client not implemented
2. **Multi-broker adapter framework**
3. **Pattern service** — `PatternSignal` schema is a placeholder
4. **Chart rendering** (footprint/VWAP)
5. **External order flow adapter** — cTrader does not provide buy/sell volume
6. **EnsembleVoter live wiring** — exists but not connected to live tick path
7. **cTrader Network Access** for economic calendar

## Planning Docs (Legacy)

The original planning package is at `docs/planning/quantmindlib/`. Those docs describe what was planned. This `docs/library/` directory describes what was actually built. Where they differ, this directory wins.

## Next Step

Read `10_coder_handoff.md` for how to approach the next implementation phase.
