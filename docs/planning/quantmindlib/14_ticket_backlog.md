# 14 — Ticket Backlog

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Implementation-ready ticket backlog for QuantMindLib V1

---

## Ticket Format

Each ticket includes:
- ID (prefix + number)
- Title
- Purpose
- Affected repo areas
- Inputs
- Outputs
- Dependencies
- Complexity (S/M/L/XL)
- Acceptance criteria
- Reference links

---

## SCAN Tickets (Documentation Recovery)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| SCAN-001 | Recover session definitions from code | P3 | S | 0 |
| SCAN-002 | Recover 10 canonical session windows | P3 | M | 0 |
| SCAN-003 | Document two Governor classes conflict | P2 | S | 0 |
| SCAN-004 | Audit MT5 usage surface across codebase | P2 | M | 0 |
| SCAN-005 | Document DPR dual-engine architecture | P1 | M | 0 |
| SCAN-006 | Recover kill switch 3-layer+progressive architecture | P2 | L | 0 |
| SCAN-007 | Recover SSLCircuitBreaker state machine | P3 | M | 0 |
| SCAN-008 | Map TRD → BotSpec conversion requirements | P2 | M | 0 |
| SCAN-009 | Audit feature registry gaps vs SVSS | P2 | S | 0 |
| SCAN-010 | Document cTrader Open API capabilities | P1 | L | 0 |

---

## DOC Tickets (Documentation)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| DOC-001 | Mark stale architecture docs with redirects | P3 | S | 0 |
| DOC-002 | Create risk system architecture doc | P3 | M | 3 |
| DOC-003 | Create kill switch architecture doc | P3 | M | 3 |
| DOC-004 | Create session management doc | P3 | S | 3 |
| DOC-005 | Create DB schema overview | P4 | M | 1 |
| DOC-006 | Document DuckDB WARM tier market data (G-19) | P2 | M | 1 |

---

## CONTRACT Tickets (Core Contracts/Schemas)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| CONTRACT-001 | Define BotSpec dataclass with all 4 profiles | P1 | M | 1 |
| CONTRACT-002 | Define MarketContext schema (RegimeReport equivalent) | P1 | S | 1 |
| CONTRACT-003 | Define TradeIntent schema | P1 | S | 1 |
| CONTRACT-004 | Define ExecutionDirective schema | P1 | S | 1 |
| CONTRACT-005 | Define RiskEnvelope schema (RiskMandate equivalent) | P1 | S | 1 |
| CONTRACT-006 | Define FeatureVector schema | P1 | S | 1 |
| CONTRACT-007 | Define EvaluationResult schema (MT5BacktestResult compatible) | P1 | M | 1 |
| CONTRACT-008 | Define BotRuntimeProfile schema | P2 | S | 1 |
| CONTRACT-009 | Define BotEvaluationProfile schema | P2 | M | 1 |
| CONTRACT-010 | Define BotMutationProfile schema | P2 | M | 1 |
| CONTRACT-011 | Define SessionContext schema | P2 | S | 1 |
| CONTRACT-012 | Define OrderFlowSignal schema | P2 | S | 1 |
| CONTRACT-013 | Define SentinelState schema | P2 | S | 1 |
| CONTRACT-014 | Define RegistryRecord schema | P3 | S | 1 |
| CONTRACT-015 | Define BotPerformanceSnapshot schema | P3 | S | 1 |
| CONTRACT-016 | Define PatternSignal placeholder schema | P4 | S | 1 |
| CONTRACT-017 | Create type enums (RegimeType, TradeDirection, RiskMode, etc.) | P1 | S | 1 |
| CONTRACT-018 | Define CapabilitySpec schema | P1 | S | 2 |
| CONTRACT-019 | Define DependencySpec schema | P1 | S | 2 |
| CONTRACT-020 | Define CompatibilityRule schema | P1 | S | 2 |
| CONTRACT-021 | Define OutputSpec schema | P1 | S | 2 |
| CONTRACT-022 | Create adapter interface contracts (IMarketDataAdapter, IExecutionAdapter) | P1 | M | 2 |
| CONTRACT-023 | Create bridge interface contracts (IBridge, SyncBridge, AsyncBridge, HybridBridge) | P1 | M | 2 |
| CONTRACT-024 | Create SpecRegistry for versioned specs | P2 | M | 1 |
| CONTRACT-025 | TRDDocument → BotSpec conversion spec | P1 | M | 1 |

---

## LIB Tickets (Library Core Development)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| LIB-001 | Create library package structure (core/, features/, archetypes/, adapters/, bridges/) | P1 | M | 1 |
| LIB-002 | Initialize pyproject.toml for library package | P1 | S | 1 |
| LIB-003 | Create FeatureModule ABC base class | P1 | S | 5 |
| LIB-004 | Create BaseArchetype ABC base class | P1 | S | 6 |
| LIB-005 | Create BotStateManager (cached FeatureVector + MarketContext) | P1 | M | 7 |
| LIB-006 | Implement SpecRegistry singleton | P2 | M | 1 |

---

## ARCH Tickets (Archetype and Composition)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| ARCH-001 | Implement CompositionValidator | P1 | M | 2 |
| ARCH-002 | Implement RequirementResolver | P1 | M | 2 |
| ARCH-003 | Implement ArchetypeRegistry singleton | P1 | S | 6 |
| ARCH-004 | Implement FeatureRegistry singleton | P1 | S | 5 |
| ARCH-005 | Implement Composer (build bot from specs) | P1 | M | 6 |
| ARCH-006 | Implement MutationEngine (constrained mutations) | P1 | M | 6 |
| ARCH-007 | Implement OpeningRangeBreakout archetype (Phase 1 priority) | P1 | M | 6 |
| ARCH-008 | Implement LondonORB derived archetype | P2 | M | 6 |
| ARCH-009 | Implement NYORB derived archetype | P2 | M | 6 |
| ARCH-010 | Implement ScalperM1 derived archetype | P2 | M | 6 |
| ARCH-011 | Implement BreakoutScalper base archetype | P3 | M | 6 |
| ARCH-012 | Implement PullbackScalper base archetype | P3 | M | 6 |
| ARCH-013 | Implement MeanReversion base archetype | P3 | M | 6 |
| ARCH-014 | Implement SessionTransition base archetype | P3 | M | 6 |

---

## BRIDGE Tickets (Bridge Definitions)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| BRIDGE-001 | Implement SentinelBridge (RegimeReport ↔ MarketContext) | P1 | M | 3 |
| BRIDGE-002 | Implement RiskBridge (TradeIntent ↔ RiskEnvelope) | P1 | M | 3 |
| BRIDGE-003 | Implement RegistryBridge (BotSpec ↔ BotRegistry/BotTagRegistry/VariantRegistry) | P1 | L | 3 |
| BRIDGE-004 | Implement DPRBridge (BotEvaluationProfile ↔ DPRScore + Redis write) | P1 | L | 3 |
| BRIDGE-005 | Implement EvaluationBridge (EvaluationResult ↔ FullBacktestPipeline) | P1 | M | 3 |
| BRIDGE-006 | Implement JournalBridge (TradeIntent → TradeRecord) | P3 | M | 3 |
| BRIDGE-007 | Implement WF1Bridge (TRD → BotSpec → AlphaForgeFlow) | P1 | M | 9 |
| BRIDGE-008 | Implement WF2Bridge (variant lineage → MutationEngine → ImprovementLoopFlow) | P1 | M | 9 |
| BRIDGE-009 | Implement LifecycleBridge (promotion/quarantine decisions) | P2 | M | 3 |

---

## CTRADER Tickets (cTrader Adapter)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| CTRADER-001 | Define IMarketDataAdapter interface | P1 | S | 2 |
| CTRADER-002 | Define IExecutionAdapter interface | P1 | S | 2 |
| CTRADER-003 | Implement CTraderClient wrapper (Open API SDK) | P1 | L | 4 |
| CTRADER-004 | Implement CTraderMarketAdapter (IMarketDataAdapter) | P1 | L | 4 |
| CTRADER-005 | Implement CTraderExecutionAdapter (IExecutionAdapter) | P1 | L | 4 |
| CTRADER-006 | Implement cTrader type converters | P1 | M | 4 |
| CTRADER-007 | Implement OHLCV, tick, depth converters | P1 | M | 4 |
| CTRADER-008 | Implement CTraderBacktestEngine (produces same schema as MT5BacktestResult) | P1 | L | 4 |
| CTRADER-009 | Implement CTrader kill switch adapter integration | P2 | M | 4 |
| CTRADER-010 | Order flow quality tagging for cTrader data sources | P1 | M | 5 |
| CTRADER-011 | cTrader adapter error handling and reconnection | P1 | M | 4 |
| CTRADER-012 | cTrader adapter config (IC Markets Raw specific) | P1 | S | 4 |
| CTRADER-013 | IExternalOrderFlowAdapter interface (Category B V1 deferred) | P4 | M | 11 |

---

## FEATURE Tickets (Feature Families)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| FEATURE-001 | Implement FeatureRegistry with registration | P1 | M | 5 |
| FEATURE-002 | Wrap SVSS VWAPIndicator as VWAPFeature | P1 | S | 5 |
| FEATURE-003 | Wrap SVSS RVOLIndicator as RVOLFeature | P1 | S | 5 |
| FEATURE-004 | Wrap SVSS VolumeProfileIndicator as VolumeProfileFeature | P1 | S | 5 |
| FEATURE-005 | Wrap SVSS MFIIndicator as MFIFeature | P1 | S | 5 |
| FEATURE-006 | Implement VolumeImbalanceFeature (delta, pressure, POC) | P1 | M | 5 |
| FEATURE-007 | Implement RSIFeature | P1 | S | 5 |
| FEATURE-008 | Implement ATRFeature | P1 | S | 5 |
| FEATURE-009 | Implement MACDFeature | P1 | S | 5 |
| FEATURE-010 | Implement TickActivityFeature (order flow) | P1 | M | 5 |
| FEATURE-011 | Implement SpreadBehaviorFeature (order flow) | P1 | M | 5 |
| FEATURE-012 | Implement SessionVolumeFeature (order flow) | P1 | M | 5 |
| FEATURE-013 | Wrap SessionDetector as SessionDetectorFeature | P1 | S | 5 |
| FEATURE-014 | Wrap NewsBlackoutService as SessionBlackoutFeature | P1 | S | 5 |
| FEATURE-015 | Implement NormalizeTransform | P2 | S | 5 |
| FEATURE-016 | Implement RollingWindowTransform | P2 | S | 5 |
| FEATURE-017 | Implement ResampleTransform | P2 | S | 5 |
| FEATURE-018 | Implement FeatureEvaluator (computes feature stack) | P1 | M | 7 |
| FEATURE-019 | Implement quality-aware feature confidence tagging | P1 | M | 5 |
| FEATURE-020 | Feature capability declarations for all 13 modules | P1 | M | 5 |

---

## EVAL Tickets (Evaluation Alignment)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| EVAL-001 | Implement BotSpec → backtest input conversion | P1 | M | 8 |
| EVAL-002 | Wire EvaluationBridge into FullBacktestPipeline | P1 | M | 8 |
| EVAL-003 | Collect 4-mode results → EvaluationResult collection | P1 | M | 8 |
| EVAL-004 | Attach MonteCarloMetrics to BotEvaluationProfile | P1 | M | 8 |
| EVAL-005 | Attach WalkForwardMetrics to BotEvaluationProfile | P1 | M | 8 |
| EVAL-006 | Attach PBO score to BotEvaluationProfile | P1 | S | 8 |
| EVAL-007 | Compute robustness score and attach | P1 | S | 8 |
| EVAL-008 | Integrate BacktestReportSubAgent with EvaluationBridge | P2 | M | 8 |
| EVAL-009 | Backtest schema compatibility test (MT5 vs cTrader) | P1 | M | 4 |

---

## WF Tickets (Workflow Alignment)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| WF-001 | TRD → BotSpec converter for WF1 | P1 | M | 9 |
| WF-002 | AlphaForgeFlow integration with library | P1 | M | 9 |
| WF-003 | ImprovementLoopFlow integration with library | P1 | M | 9 |
| WF-004 | Library → paper trading handoff | P1 | M | 9 |
| WF-005 | Library → live promotion path (3-day paper lag) | P1 | M | 9 |

---

## DPR Tickets (Registry/DPR/Journal)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| DPR-001 | DPR Redis publisher (fixes G-18 gap) | P1 | M | 10 |
| DPR-007 | BotRegistry SQLite table migration (uncommitted migration script) | P2 | S | 1 |
| DPR-002 | DPR concern event → @session_concern tag flow | P1 | M | 10 |
| DPR-003 | DPR tier → BotTier mapping | P1 | S | 10 |
| DPR-004 | BotCircuitBreaker integration via SafetyHooks | P2 | M | 10 |
| DPR-005 | SSLCircuitBreaker integration via SafetyHooks | P2 | M | 10 |
| DPR-006 | DPR dual engine handling (router + risk layers) | P2 | L | 10 |

---

## DEFERRED Tickets (Future Phases)

| ID | Title | Priority | Complexity | Phase |
|----|-------|----------|------------|-------|
| DEFER-001 | PatternSignal schema refinement | P4 | M | 11 |
| DEFER-002 | Pattern service (chart pattern analysis) | P4 | L | 11 |
| DEFER-003 | Multi-broker adapter framework | P4 | XL | 11 |
| DEFER-004 | cTrader Network Access integration (economic calendar) | P4 | M | 11 |
| DEFER-005 | Advanced derived archetypes | P3 | M | 11 |
| DEFER-006 | Chart rendering (footprint/VWAP for debugging) | P4 | L | 11 |
| DEFER-007 | IExternalOrderFlowAdapter for true executed trade-flow (Category B) | P3 | L | 11 |

---

## Ticket Dependency Graph

### Phase 1 Prerequisites
```
CONTRACT-001 → LIB-001 (package structure needed for contracts)
CONTRACT-025 → CONTRACT-001 (TRD→BotSpec needs BotSpec)
CONTRACT-017 → CONTRACT-001 (enums needed for schemas)
```

### Phase 2 Prerequisites
```
CONTRACT-018 → CONTRACT-001 (CapabilitySpec needs domain objects)
CONTRACT-019 → CONTRACT-001
CONTRACT-020 → CONTRACT-001
CONTRACT-021 → CONTRACT-001
CONTRACT-022 → CONTRACT-023
CONTRACT-023 → CONTRACT-022
```

### Phase 3 Prerequisites
```
BRIDGE-001 → CONTRACT-001 (MarketContext)
BRIDGE-002 → CONTRACT-001 (TradeIntent, RiskEnvelope)
BRIDGE-003 → CONTRACT-001 (BotSpec)
BRIDGE-004 → CONTRACT-001 (BotEvaluationProfile)
BRIDGE-005 → CONTRACT-001 (EvaluationResult)
BRIDGE-006 → CONTRACT-001 (TradeIntent)
BRIDGE-007 → CONTRACT-001
BRIDGE-008 → CONTRACT-001
BRIDGE-009 → CONTRACT-001
CTRADER-001 → CONTRACT-022 (IMarketDataAdapter interface)
CTRADER-002 → CONTRACT-022 (IExecutionAdapter interface)
```

### Phase 4 Prerequisites
```
CTRADER-003 → CTRADER-001 (needs interface)
CTRADER-004 → CTRADER-001
CTRADER-005 → CTRADER-002
CTRADER-006 → CTRADER-003
CTRADER-007 → CTRADER-003
CTRADER-008 → CTRADER-004 (backtest needs market adapter)
```

### Phase 5 Prerequisites
```
FEATURE-001 → CONTRACT-018 (CapabilitySpec needed)
FEATURE-002 → FEATURE-001
FEATURE-003 → FEATURE-001
FEATURE-004 → FEATURE-001
FEATURE-005 → FEATURE-001
FEATURE-006 → FEATURE-001
FEATURE-007 → FEATURE-001
FEATURE-008 → FEATURE-001
FEATURE-009 → FEATURE-001
FEATURE-010 → FEATURE-001
FEATURE-011 → FEATURE-001
FEATURE-012 → FEATURE-001
FEATURE-013 → FEATURE-001
FEATURE-014 → FEATURE-001
FEATURE-015 → FEATURE-001
FEATURE-016 → FEATURE-001
FEATURE-017 → FEATURE-001
FEATURE-020 → FEATURE-001
FEATURE-018 → FEATURE-001
```

### Phase 6 Prerequisites
```
ARCH-003 → FEATURE-001 (FeatureRegistry needed)
ARCH-004 → FEATURE-001
ARCH-007 → ARCH-003, ARCH-004
ARCH-005 → ARCH-001, ARCH-002
ARCH-006 → ARCH-005
```

### Phase 7 Prerequisites
```
LIB-005 → BRIDGE-001 (needs SentinelBridge for MarketContext)
LIB-005 → BRIDGE-002 (needs RiskBridge for RiskEnvelope)
LIB-005 → FEATURE-018 (FeatureEvaluator)
FEATURE-018 → FEATURE-020 (capability declarations)
```

---

## Priority Ordering (Implementation Start Order)

### Tier 1 (Start immediately after Phase 0)
1. CONTRACT-001: BotSpec dataclass
2. CONTRACT-017: Type enums
3. LIB-001: Package structure
4. CONTRACT-022: Adapter interface contracts
5. CONTRACT-023: Bridge interface contracts
6. CTRADER-001: IMarketDataAdapter interface
7. CTRADER-002: IExecutionAdapter interface

### Tier 2 (After Tier 1)
8. CONTRACT-002-016: Other domain schemas
9. CONTRACT-018-021: Capability system
10. BRIDGE-001-009: All bridges
11. FEATURE-001: FeatureRegistry
12. FEATURE-020: Capability declarations

### Tier 3 (After Tier 2)
13. CTRADER-003-012: cTrader adapter
14. FEATURE-002-019: All feature modules
15. ARCH-001-010: Archetypes and composition
16. LIB-005: BotStateManager
17. FEATURE-018: FeatureEvaluator
18. EVAL-001-009: Evaluation alignment

### Tier 4 (After Tier 3)
19. WF-001-005: Workflow integration
20. DPR-001-006: Registry/DPR/journal alignment