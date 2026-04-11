# QuantMindLib V1 — Current Architecture

## Package Layout (As Implemented)

```
src/library/
├── __init__.py
├── base_bot.py                      # Base bot class (minimal)
├── pyproject.toml
│
├── core/
│   ├── __init__.py
│   ├── types/
│   │   ├── __init__.py             # 12 StrEnums exported
│   │   └── enums.py                # RegimeType, TradeDirection, RiskMode, NewsState,
│   │                                 # SignalDirection, OrderFlowSource, ActivationState,
│   │                                 # BotHealth, EvaluationMode, RegistryStatus,
│   │                                 # FeatureConfidenceLevel, BotTier, DPRTier,
│   │                                 # ErrorSeverity (optional)
│   │
│   ├── domain/
│   │   ├── __init__.py             # 24 exports
│   │   ├── bot_spec.py              # BotSpec (frozen), BotRuntimeProfile, BotEvaluationProfile,
│   │   │                             # BotMutationProfile, BacktestMetrics, MonteCarloMetrics,
│   │   │                             # WalkForwardMetrics, SessionScore
│   │   ├── market_context.py        # MarketContext, RegimeReport
│   │   ├── feature_vector.py         # FeatureVector, FeatureConfidence
│   │   ├── trade_intent.py           # TradeIntent, TradeIntentBatch
│   │   ├── execution_directive.py    # ExecutionDirective (approved, rejection_reason fields)
│   │   ├── risk_envelope.py          # RiskEnvelope
│   │   ├── session_context.py        # SessionContext
│   │   ├── sentinel_state.py         # SentinelState, SensorState, HMMState
│   │   ├── order_flow_signal.py      # OrderFlowSignal
│   │   ├── pattern_signal.py         # PatternSignal (V1 placeholder)
│   │   ├── evaluation_result.py      # EvaluationResult
│   │   ├── registry_record.py        # RegistryRecord
│   │   └── bot_performance_snapshot.py # BotPerformanceSnapshot
│   │
│   ├── composition/
│   │   ├── __init__.py             # 14 exports
│   │   ├── capability_spec.py       # CapabilitySpec
│   │   ├── dependency_spec.py        # DependencySpec
│   │   ├── compatibility_rule.py    # CompatibilityRule
│   │   ├── output_spec.py            # OutputSpec
│   │   ├── adapter_contracts.py       # IMarketDataAdapter, IExecutionAdapter, IRiskAdapter
│   │   ├── bridge_contracts.py        # ISentinelBridge, IExecutionBridge, IFeatureBridge, IRiskBridge
│   │   ├── spec_registry.py          # SpecRegistry
│   │   └── trd_converter.py          # TRDConverter (TRD → BotSpec)
│   │
│   ├── bridges/
│   │   ├── __init__.py
│   │   ├── sentinel_dpr_bridges.py   # SentinelBridge + DPRBridge + DPRScore
│   │   ├── registry_journal_bridges.py # RegistryBridge + JournalBridge
│   │   ├── lifecycle_eval_workflow_bridges.py # LifecycleBridge + EvaluationBridge + WorkflowBridge
│   │   ├── risk_execution_bridges.py # RiskBridge + ExecutionBridge
│   │   ├── dpr_redis_bridge.py      # DPRRedisPublisher
│   │   ├── dpr_concern_bridge.py     # DPRConcernEmitter
│   │   ├── ssl_dpr_integration.py    # SSLCircuitBreakerDPRMonitor
│   │   ├── safety_integration.py     # DPRCircuitBreakerMonitor
│   │   └── dpr_dual_engine.py        # DPRDualEngineRouter
│   │
│   ├── errors/
│   │   ├── __init__.py             # All error types exported
│   │   ├── base.py                  # LibraryError (base), LibraryConfigError,
│   │   │                             # ContractValidationError
│   │   └── audit.py                  # BridgeError (base), BridgeUnavailableError,
│   │                                   # DependencyMissingError, FeatureNotFoundError,
│   │                                   # AuditRecord, ErrorSeverity (optional enum)
│   │
│   ├── ctrader/
│   │   └── (EMPTY — adapter not implemented)
│   │
│   └── migrations/
│       └── (SQL migrations)
│
├── features/
│   ├── __init__.py
│   ├── registry.py                   # FeatureRegistry (central registry)
│   ├── _registry.py                  # Bootstrap: get_default_registry()
│   ├── base/
│   │   ├── __init__.py
│   │   └── feature_module.py         # FeatureModule ABC, FeatureConfig
│   ├── indicators/
│   │   ├── __init__.py             # RSIFeature, ATRFeature, MACDFeature, VWAPFeature
│   │   ├── rsi.py
│   │   ├── atr.py
│   │   ├── macd.py
│   │   └── vwap.py
│   ├── volume/
│   │   ├── __init__.py             # RVOLFeature, MFIFeature, VolumeProfileFeature
│   │   ├── rvol.py
│   │   ├── mfi.py
│   │   └── profile.py
│   ├── microstructure/
│   │   ├── __init__.py
│   │   ├── microstructure_base.py   # MicrostructureFeature ABC
│   │   ├── spread.py                 # SpreadStateFeature
│   │   ├── tob_pressure.py          # TopOfBookPressureFeature
│   │   ├── depth.py                  # MultiLevelDepthFeature
│   │   ├── aggression.py             # AggressionProxyFeature
│   │   ├── absorption.py             # AbsorptionProxyFeature
│   │   ├── breakout_pressure.py      # BreakoutPressureProxyFeature
│   │   ├── liquidity_stress.py       # LiquidityStressProxyFeature
│   │   ├── tick_activity.py         # TickActivityFeature
│   │   ├── volume_imbalance.py      # VolumeImbalanceFeature
│   │   └── context.py               # MicrostructureContext (aggregation)
│   ├── orderflow/
│   │   ├── __init__.py             # SpreadBehaviorFeature, DOMPressureFeature, DepthThinningFeature
│   │   ├── spread_behavior.py
│   │   ├── dom_pressure.py
│   │   └── depth_thinning.py
│   ├── session/
│   │   ├── __init__.py             # SessionDetectorFeature, SessionBlackoutFeature
│   │   ├── detector.py
│   │   └── blackout.py
│   └── transforms/
│       ├── __init__.py             # NormalizeTransform, RollingWindowTransform, ResampleTransform
│       ├── normalize.py
│       ├── rolling.py
│       └── resample.py
│
├── archetypes/
│   ├── __init__.py
│   ├── base.py                      # BaseArchetype ABC, ArchetypeSpec
│   ├── registry.py                  # ArchetypeRegistry, get_default_registry()
│   ├── composer.py                  # Composer, CompositionResult
│   ├── constraints.py               # ConstraintSpec
│   ├── orb.py                       # ORB_ARCHETYPE, OpeningRangeBreakout
│   ├── derived.py                   # LondonORB, NYORB, ScalperM1 (deep implementations)
│   ├── stubs.py                     # 4 archetype stubs (BreakoutScalper, etc.)
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── validation.py             # ValidationResult
│   │   ├── resolver.py              # RequirementResolver
│   │   ├── validator.py             # CompositionValidator
│   │   └── result.py                # CompositionResult
│   └── mutation/
│       ├── __init__.py
│       └── engine.py                # MutationEngine, MutationResult
│
├── runtime/
│   ├── __init__.py
│   ├── orchestrator.py             # RuntimeOrchestrator (wires everything)
│   ├── feature_evaluator.py         # FeatureEvaluator
│   ├── intent_emitter.py            # IntentEmitter
│   ├── state_manager.py             # BotStateManager (thread-safe cache)
│   └── safety_hooks.py             # SafetyHooks
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluation_orchestrator.py    # EvaluationOrchestrator
│   ├── strategy_code_generator.py   # StrategyCodeGenerator
│   ├── report_bridge.py             # BacktestReportBridge
│   └── ctrader_backtest_schema.py  # CTraderBacktestSchema (schema compatibility)
│
└── workflows/
    ├── __init__.py
    ├── wf1_bridge.py                # WF1Bridge
    ├── wf2_bridge.py                # WF2Bridge
    └── stub_flows.py                # AlgoForgeFlowStub, ImprovementLoopFlowStub (EXPLICIT STUBS)
```

## Implementation Status by Area

| Area | Status | Notes |
|------|--------|-------|
| Domain objects (14 files) | ✓ Complete | All schemas implemented, frozen where appropriate |
| Enums (12+ types) | ✓ Complete | All StrEnums in `core/types/enums.py` |
| Composition (capability, dependency, compatibility) | ✓ Complete | 7 files in `core/composition/` |
| Bridge definitions | ✓ Complete | 9 bridge files, DPR Redis fully wired |
| Error hierarchy | ✓ Complete | ERR-001 through ERR-004 committed |
| Feature modules (16 features) | ✓ Complete | 6 families, all with FeatureModule ABC |
| Feature registry | ✓ Complete | Singleton bootstrap with `get_default_registry()` |
| Archetype system | ✓ Complete | ORB fully implemented, 4 deep archetypes |
| Composer + mutation | ✓ Complete | CompositionValidator, MutationEngine |
| Runtime (orchestrator, evaluator, emitter, state) | ✓ Complete | 5 files, all wired |
| Evaluation orchestration | ✓ Complete | Full pipeline integration |
| Workflow bridges | ✓ Complete | WF1Bridge, WF2Bridge; Prefect flows are stubs |
| cTrader adapter | ✗ Empty | `src/library/ctrader/` does not exist |
| Prefect flows | ○ Stubs | `stub_flows.py` — explicit stubs for external flows |
| ErrorSeverity wiring | ○ Partial | Enum defined, not wired into exception classes (out of V1 scope) |
| AuditRecord emission | ○ Partial | Schema defined, bridges do not emit (out of V1 scope) |

## Runtime Boundaries (Sync/Async)

### Sync Paths (Decision-Time)
```
BotStateManager (sync read cached FeatureVector + MarketContext)
    │
    ▼
FeatureEvaluator (sync compute on cached state)
    │
    ▼
IntentEmitter (sync emit TradeIntent)
    │
    ▼
RiskBridge (sync call via RuntimeOrchestrator → Governor)
    │
    ▼
ExecutionBridge (sync call → ExecutionDirective with approved/rejection_reason)
```

### Async Paths (Event Streams)
```
cTrader tick/depth stream → (not implemented)
    └──► Feature workers (async evaluation) — not wired yet

Sentinel regime events → SentinelBridge → MarketContext updates
    └──► BotStateManager cache

DPR score events → DPRBridge → Redis publish (via DPRRedisPublisher)
    └──► DPRRedisPublisher._write_score_to_redis()

Kill switch events → SafetyHooks → position close directives
    └──► SafetyHooks.handle_kill_event()

SSL circuit breaker → SSLCircuitBreakerDPRMonitor → combined kill switch
    └──► SSLCircuitBreakerDPRMonitor.check_ssl_dpr_combined()
```

## Phase Tracking (vs Planning Docs)

| Phase | Planning Doc Status | Actual Status |
|-------|---------------------|---------------|
| Phase 7 (Runtime) | TODO | ✓ COMPLETE |
| Phase 8 (Evaluation) | TODO | ✓ COMPLETE |
| Phase 9 (Workflows) | TODO | ✓ COMPLETE |
| Phase 10 (DPR/Registry) | TODO | ✓ COMPLETE |
| Phase 4 (cTrader adapter) | Partial (interfaces) | ✗ EMPTY (adapters not implemented) |
| Phase 11 (Deferred) | DEFERRED | DEFERRED (as planned) |

**The task tracker (`14_ticket_backlog.md`) has stale TODO labels for Phases 7-10.** These are fully implemented.
