# QuantMindLib V1 — Known Gaps and Deferred Items

## COMPLETE — Confirmed Fully Implemented

These areas are done and should not be rewritten:

### Phase 1: Core Contracts/Schemas
- All 14 domain objects implemented
- All 12+ enums in `core/types/enums.py`
- Phase 1 complete (24 classes across 14 domain files)

### Phase 2: Capability/Compatibility System
- CapabilitySpec, DependencySpec, CompatibilityRule, OutputSpec
- IMarketDataAdapter, IExecutionAdapter, IRiskAdapter protocols
- ISentinelBridge, IExecutionBridge, IFeatureBridge, IRiskBridge protocols
- SpecRegistry
- TRDConverter
- Phase 2 complete (14 types across 9 files)

### Phase 3: Bridge Definitions
- DPRRedisPublisher, DPRConcernEmitter, SingapuraDualEngineRouter
- SentinelBridge, DPRBridge, RegistryBridge, JournalBridge
- LifecycleBridge, EvaluationBridge, WorkflowBridge
- RiskBridge, ExecutionBridge
- SafetyHooks, SSLCircuitBreakerDPRMonitor, DPRCircuitBreakerMonitor
- Phase 3 complete (9 bridge files)

### Phase 4: cTrader Adapter
- IMarketDataAdapter, IExecutionAdapter interfaces defined ✓
- CTraderMarketAdapter, CTraderExecutionAdapter stubs exist ✓
- Real client implementation ✗ DEFERRED

### Phase 5: Feature Registry V1
- All 16 feature modules across 6 families
- FeatureModule ABC + FeatureConfig
- FeatureRegistry singleton + bootstrap
- Phase 5 complete

### Phase 6: Archetype System V1
- ORB fully implemented
- LondonORB, NYORB, ScalperM1 deep implementations
- Composer, MutationEngine
- ArchetypeRegistry, CompositionValidator
- Phase 6 complete

### Phase 7: Runtime Intent Flow
- RuntimeOrchestrator ✓
- FeatureEvaluator ✓
- IntentEmitter ✓
- BotStateManager ✓
- SafetyHooks ✓
- Phase 7 complete (**tracker says TODO — stale**)

### Phase 8: Evaluation Alignment
- EvaluationOrchestrator ✓
- StrategyCodeGenerator ✓
- BacktestReportBridge ✓
- EvaluationBridge ✓
- cTraderBacktestSchema ✓
- Phase 8 complete (**tracker says TODO — stale**)

### Phase 9: Workflow Alignment
- WF1Bridge ✓
- WF2Bridge ✓
- Prefect flows: stubs ✓ (external, not implemented)
- Phase 9 complete (**tracker says TODO — stale**)

### Phase 10: Registry/DPR/Journal Alignment
- DPRRedisPublisher ✓
- DPRConcernEmitter ✓
- SingapuraDualEngineRouter ✓
- RegistryBridge ✓
- JournalBridge ✓
- LifecycleBridge ✓
- SafetyHooks (circuit breaking) ✓
- Phase 10 complete (**tracker says TODO — stale**)

### Error Logic (Phase 11K)
- LibraryError, LibraryConfigError, ContractValidationError ✓
- BridgeError, BridgeUnavailableError ✓
- Bridge-specific subclasses (SentinelBridgeError, RiskBridgeError, etc.) ✓
- DependencyMissingError ✓
- FeatureNotFoundError ✓
- AuditRecord ✓
- ErrorSeverity (enum defined) ✓
- ExecutionDirective.approved ✓
- ExecutionDirective.rejection_reason ✓

## INCOMPLETE — Non-Deferred

Only ONE genuinely incomplete non-deferred item:

### BLOCKER-3: cTrader Open API Capability Verification
- **What:** Verify tick streams, bar streams, depth streams, historical data, order execution, account state against cTrader Open API docs
- **Why:** Needed before cTrader adapter can be implemented
- **Action:** Fetch https://help.ctrader.com/open-api/ and verify each capability
- **Owner:** Research (web fetch, no code)

## DEFERRED — Explicitly Deferred

These items were deferred by design decision:

### Phase 11 Items
| Item | Reason Deferred |
|------|----------------|
| cTrader real client (`src/library/ctrader/`) | BLOCKER-3 must complete first |
| Multi-broker adapter framework | V1 scope: cTrader only |
| Pattern service (`PatternSignal` is placeholder) | Too heavy for V1 core |
| Chart rendering (footprint/VWAP) | Not required for V1 |
| External order flow adapter | cTrader doesn't provide buy/sell volume |
| cTrader Network Access for calendar | Finnhub direct polling works for V1 |

### Phase 2 Deferred
| Item | Reason Deferred |
|------|----------------|
| EnsembleVoter live wiring | Voter exists but not wired; Phase 2 priority |
| ErrorSeverity wiring into exceptions | Enum exists but not used; low priority |
| AuditRecord emission from bridges | Schema exists but not emitted; low priority |

## STALE TRACKER ITEMS

These task tracker items have incorrect labels:

| Tracker Item | Actual Status |
|-------------|--------------|
| Phase 7 Runtime Intent Flow: TODO | ✓ COMPLETE (5 source files + 7 test files) |
| Phase 8 Evaluation Alignment: TODO | ✓ COMPLETE (5 source files + 7 test files) |
| Phase 9 Workflow Alignment: TODO | ✓ COMPLETE (4 source files + 3 test files) |
| Phase 10 Registry/DPR/Journal: TODO | ✓ COMPLETE (9 bridge files + 9 test files) |
| ASSUMPTION-1: TRD→BotSpec verification | ✓ Resolved (TRDConverter implemented) |
| ASSUMPTION-4: SVSS MT5 dependency | ✓ Theoretical only |
| ASSUMPTION-5: SCALPER+ORB scope | ✓ Confirmed in task tracker |
| DPR Redis gap (G-18) | ✓ Fixed (router + risk layer) |
| DPR dual engines (C2) | ✓ Resolved (handled by SingapuraDualEngineRouter) |
| Two Governor classes (C1) | ✓ Resolved (referenced by module path) |

## Pre-Existing Issues (Not Library Bugs)

| Issue | Not Library |
|-------|------------|
| 3 archetype test files with `sys.exit(0)` at module level | Pre-existing test infrastructure |
| DPR Redis gap in risk layer | Was confirmed; now fixed via DPRRedisPublisher |
| `src/router/enhanced_governor.py` in uncommitted worktree | Router-layer enhancement, not library |

## What to Do Next

1. **Complete BLOCKER-3** — cTrader API capability verification (research task)
2. **Update task tracker** — Change Phases 7-10 from TODO to COMPLETE
3. **Begin integration** — Wire library components into existing system (cTrader adapter first)
4. **Replace stubs** — `stub_flows.py` + Prefect flow implementations (Phase 2)
