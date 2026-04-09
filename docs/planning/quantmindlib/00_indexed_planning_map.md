# QUANTMINDLIB PLANNING PACKAGE

## Indexed Planning Map

**Version:** 1.0
**Date:** 2026-04-08
**Status:** SCAN + DOCUMENT + PLAN + DECOMPOSE — NO IMPLEMENTATION
**Worktree:** elegant-turing
**Memos reconciled:** `quantmindx_library_v1_architecture_memo.docx`, `trading_platform_decision_memo.docx`

---

## Package Contents

| ID | Document | Purpose |
|----|----------|---------|
| `00` | `00_indexed_planning_map.md` | **THIS FILE** — Navigation hub |
| `01` | `01_executive_synthesis.md` | What QuantMindLib is, isn't, and why |
| `02` | `02_codebase_scan_report.md` | 19-subsystem scan from actual repo |
| `03` | `03_documentation_status_report.md` | Doc currency assessment |
| `04` | `04_recovered_architecture_notes.md` | Undocumented/stale areas recovered from code |
| `05` | `05_gap_analysis_against_memos.md` | Memo vs codebase gap analysis |
| `06` | `06_target_architecture_v1.md` | Recommended V1 target architecture |
| `07` | `07_shared_object_model.md` | Contract inventory (BotSpec, etc.) |
| `08` | `08_bridge_inventory.md` | Bridge layer design |
| `09` | `09_ctrader_boundary_plan.md` | cTrader boundary definition |
| `10` | `10_feature_family_plan.md` | V1 feature families |
| `11` | `11_archetype_and_composition_plan.md` | Archetypes and composition |
| `12` | `12_evaluation_workflow_alignment.md` | Library-to-eval alignment |
| `13` | `13_phased_roadmap.md` | Phased implementation roadmap |
| `14` | `14_ticket_backlog.md` | Ticket-style backlog with IDs |
| `15` | `15_short_implementation_specs.md` | Handoff specs for downstream agents |
| `16` | `16_risks_and_open_questions.md` | Blockers, assumptions, risks |

---

## Cross-Reference Index

### Source Documents
- `quantmindx_library_v1_architecture_memo.docx` → converted memo, 17 sections + 2 appendices
- `trading_platform_decision_memo.docx` → converted memo, 9 sections + appendices
- Source: `/home/mubarkahimself/Desktop/QUANTMINDX/quantmindx_library_v1_architecture_memo.docx`
- Source: `/home/mubarkahimself/Desktop/QUANTMINDX/trading_platform_decision_memo.docx`

### Authoritative Internal Docs
- `_bmad-output/planning-artifacts/architecture.md` → §1-§21 current architecture decisions
- `WF1_SYSTEM_SCAN.md` (in memory/) → WF1 source of truth
- `docs/WIKI_INTEGRATION_ARCHITECTURE.md` → QuantWiki integration
- `docs/BACKTEST_REPORTS_DESIGN.md` → backtest report pipeline
- `docs/architecture.md` → stale system reference (points to `_bmad-output`)
- `src/memory/ARCHITECTURE.md` → memory system architecture

### Key Repo Files (by subsystem)
| Subsystem | Key Files |
|---|---|
| **Bot definition** | `src/router/bot_manifest.py`, `src/database/models/bots.py` |
| **Risk engine** | `src/risk/governor.py`, `src/risk/engine.py` |
| **Position sizing** | `src/position_sizing/enhanced_kelly.py` |
| **Sentinel** | `src/router/sentinel.py`, `src/router/multi_timeframe_sentinel.py` |
| **DPR** | `src/router/dpr_scoring_engine.py`, `src/risk/dpr/scoring_engine.py` |
| **Registry** | `src/router/bot_tag_registry.py`, `src/router/variant_registry.py` |
| **Backtesting** | `src/backtesting/full_backtest_pipeline.py`, `src/backtesting/mt5_engine.py` |
| **Kill switches** | `src/router/kill_switch.py`, `src/router/progressive_kill_switch.py` |
| **Lifecycle** | `src/router/lifecycle_manager.py` |
| **Governor** | `src/router/governor.py` (risk authorization) |
| **TRD schema** | `src/trd/schema.py` |
| **Asset library** | `src/mcp_tools/asset_library.py` |
| **Library (stub)** | `src/library/base_bot.py` |
| **Agents** | `src/agents/core/base_agent.py`, `src/agents/departments/` |
| **Flows** | `flows/alpha_forge_flow.py`, `flows/improvement_loop_flow.py` |
| **Strategy** | `src/strategy/template_service.py` |
| **Events** | `src/events/*.py` (regime, chaos, tilt, dpr, ssl) |
| **Database models** | `src/database/models/performance.py`, `src/database/models/bots.py` |

### External Docs (from memo Appendix A)
- cTrader Algo: https://help.ctrader.com/ctrader-algo/
- cTrader Open API: https://help.ctrader.com/open-api/
- cTrader Python SDK: https://help.ctrader.com/open-api/python-SDK/python-sdk-index/
- cTrader MarketData: https://help.ctrader.com/ctrader-algo/references/MarketData/
- cTrader Ticks: https://help.ctrader.com/ctrader-algo/references/MarketData/Ticks/Ticks/
- cTrader MarketDepth: https://help.ctrader.com/ctrader-algo/id/references/MarketData/MarketDepth/MarketDepth/
- cTrader DOM: https://help.ctrader.com/trading-with-ctrader/depth-of-market/
- cTrader Network Access: https://help.ctrader.com/ctrader-algo/guides/network-access/
- cTrader Web (Linux): https://help.ctrader.com/ctrader-web/
- IC Markets cTrader Raw: https://www.icmarkets.com/global/en/trading-accounts/ctrader-raw
- IC Markets start trading: https://www.icmarkets.com/global/en/start-forex-trading
- IC Markets swap-free: https://www.icmarkets.com/global/en/trading-accounts/swap-free-account
- IC Markets overview: https://www.icmarkets.com/global/en/trading-accounts/overview

### Locked Decisions (from memos + repo validation)
1. **Platform:** cTrader (not MT5) — confirmed by trading_platform_decision_memo
2. **First broker:** IC Markets cTrader Raw — confirmed by architecture memo
3. **Bots:** thin and composable — confirmed by base_bot.py + bot_manifest.py
4. **Library role:** contract layer + bridge layer + composition layer — not a full trading system
5. **Existing systems:** integrated by default (risk, position sizing, sentinel, DPR, registry, eval workflows)
6. **Order flow V1:** logic/features only, not chart visualization
7. **Footprint:** logic allowed, chart rendering not required
8. **Chart patterns:** not a V1 core dependency
9. **Sync/async:** boundaries matter and must be explicit

### Codebase Reality Flags
- `src/library/` is a stub package (1 file: `base_bot.py`, no `__init__.py`)
- No `BotSpec` model exists — `BotManifest` + `TRDDocument` are current equivalents
- No formal FeatureRegistry — `FeatureConfig` + SVSS package approximate this
- No archetype system — `StrategyType` enum is the current substitute
- DPR Redis gap confirmed: scores computed but not written to Redis
- MT5 platform code exists throughout but must be isolated for cTrader migration
- WF1/WF2 workflows are the canonical evaluation paths

### Output File Locations
All planning docs live at:
```
docs/planning/quantmindlib/00_indexed_planning_map.md  (this file)
docs/planning/quantmindlib/01_executive_synthesis.md
docs/planning/quantmindlib/02_codebase_scan_report.md
docs/planning/quantmindlib/03_documentation_status_report.md
docs/planning/quantmindlib/04_recovered_architecture_notes.md
docs/planning/quantmindlib/05_gap_analysis_against_memos.md
docs/planning/quantmindlib/06_target_architecture_v1.md
docs/planning/quantmindlib/07_shared_object_model.md
docs/planning/quantmindlib/08_bridge_inventory.md
docs/planning/quantmindlib/09_ctrader_boundary_plan.md
docs/planning/quantmindlib/10_feature_family_plan.md
docs/planning/quantmindlib/11_archetype_and_composition_plan.md
docs/planning/quantmindlib/12_evaluation_workflow_alignment.md
docs/planning/quantmindlib/13_phased_roadmap.md
docs/planning/quantmindlib/14_ticket_backlog.md
docs/planning/quantmindlib/15_short_implementation_specs.md
docs/planning/quantmindlib/16_risks_and_open_questions.md
```

### Ticket ID Prefix Reference
| Prefix | Area |
|--------|------|
| `SCAN` | Repo scanning / documentation recovery |
| `DOC` | Documentation |
| `CONTRACT` | Core contracts / schemas |
| `LIB` | Library core development |
| `ARCH` | Archetype / composition |
| `BRIDGE` | Bridge definitions |
| `CTRADER` | cTrader adapter boundary |
| `FEATURE` | Feature families / registry |
| `EVAL` | Evaluation alignment |
| `WF` | Workflow alignment |
| `DPR` | Registry/DPR/journal alignment |