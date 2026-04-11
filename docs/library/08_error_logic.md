# QuantMindLib V1 — Error Logic

## Design Principle

**Rejection = controlled outcome. Engineering failure = exception.**

A bot being denied by the governor is a **controlled outcome** — expected, expected, and communicated via `ExecutionDirective.approved=False` and `rejection_reason`. It is NOT an error. The system worked correctly.

An unexpected condition that prevents normal operation is an **engineering failure** — it should raise an exception from the V1 error hierarchy.

## V1 Error Hierarchy

```
LibraryError (base)
├── LibraryConfigError
│   # Invalid BotSpec, unknown archetype, malformed configuration
│   # NOT a runtime failure — a configuration problem
│   └── Triggered by: Composer, FeatureRegistry, ArchetypeRegistry
│
├── ContractValidationError
│   # Pydantic validation failure on library domain objects
│   # NOT a runtime failure — a contract violation
│   └── Triggered by: Domain object instantiation
│
└── BridgeError (base)
    ├── BridgeUnavailableError
    │   # A bridge cannot reach its target system
    │   # Examples: Redis down, DPR engine unreachable, Registry unavailable
    │   ├── Triggers: DPRRedisPublisher, DPRBridge, RegistryBridge
    │   └── Resolution: Retry with backoff, fallback to last known state
    │
    ├── DependencyMissingError
    │   # A required feature/dependency is not registered or available
    │   ├── Triggers: FeatureRegistry.validate_composition()
    │   ├── Extends: BridgeError (not a bridge-specific error)
    │   └── NOT the same as a failed composition — this is a missing dependency
    │
    ├── FeatureNotFoundError
    │   # A feature ID is not found in the registry
    │   ├── Triggers: FeatureRegistry.get()
    │   └── Extends: BridgeError
    │
    ├── SentinelBridgeError (subclass)
    │   # Sentinel system unavailable or returning malformed data
    │   ├── Extends: BridgeError
    │   └── Fallback: UNCERTAIN regime, is_stale=True
    │
    ├── RiskBridgeError (subclass)
    │   # Governor or risk system unavailable
    │   ├── Extends: BridgeError
    │   └── Fallback: HALTED RiskEnvelope
    │
    ├── DPRBridgeError (subclass)
    │   # DPR scoring system unavailable
    │   ├── Extends: BridgeError
    │   └── Fallback: last known score, marked stale
    │
    ├── RegistryBridgeError (subclass)
    │   # Bot registry unavailable
    │   ├── Extends: BridgeError
    │   └── Triggers: Registry cannot be reached
    │
    └── EvaluationBridgeError (subclass)
        # Evaluation pipeline unavailable
        ├── Extends: BridgeError
        └── Triggers: FullBacktestPipeline cannot be reached
```

## What Was NOT Built (Out of V1 Scope)

### ErrorSeverity Wiring
`ErrorSeverity` enum exists in `core/types/enums.py` with values `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, but it is **NOT wired into the exception classes**. This means exceptions don't carry severity metadata by default. Adding severity to exceptions was deemed out of V1 scope — it can be added in Phase 2 if needed.

### AuditRecord Emission
`AuditRecord` (`src/library/core/errors/audit.py`) is a minimal schema defined for diagnostic/journal entries:
```python
class AuditRecord(BaseModel):
    event_type: str
    timestamp: datetime
    component: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: Optional[ErrorSeverity] = None
```

Bridges do **NOT currently emit AuditRecord** instances. Adding systematic audit logging was deemed out of V1 scope — it can be added when bridges are integrated with the event bus.

## File Locations

```
src/library/core/errors/
├── base.py          # LibraryError, LibraryConfigError, ContractValidationError
└── audit.py         # BridgeError, BridgeUnavailableError, DependencyMissingError,
                     # FeatureNotFoundError, AuditRecord, ErrorSeverity (enum)
```

## Rejection vs Exception

| Scenario | Mechanism | Example |
|----------|-----------|---------|
| Governor denies trade | `ExecutionDirective.approved=False, rejection_reason=...` | `rejection_reason="position_size <= 0, trade not authorized"` |
| Feature inputs insufficient | `FeatureVector` with `quality=0.0, feed_quality_tag="INSUFFICIENT_DATA"` | RSI with fewer than period bars |
| DPR tier drops | Tag applied via `DPRConcernEmitter` | `@session_concern` tag |
| Kill switch active | `SafetyHooks.is_trading_allowed()=False` | Layer 3 kill |
| Bridge cannot reach system | `BridgeUnavailableError` | Redis down |
| Missing feature dependency | `DependencyMissingError` | Feature not registered |
| Bot spec invalid | `LibraryConfigError` | Unknown archetype |
| Contract validation failure | `ContractValidationError` | Malformed domain object |

## RiskMode vs Exception

`RiskMode` (STANDARD / CLAMPED / HALTED) communicates **how much risk the system is accepting**. It is NOT an exception mechanism.

- `STANDARD`: Normal operation, full position sizing
- `CLAMPED`: Reduced sizing (position_size < requested)
- `HALTED`: No new positions (but existing positions managed)

When `risk_mode=HALTED`, the `ExecutionDirective` should have `approved=False`. This is a controlled rejection, not an exception.

## DPRTier vs Exception

`DPRTier` (ELITE / PERFORMING / STANDARD / AT_RISK / CIRCUIT_BROKEN) communicates **bot health from the DPR scoring system**. It is NOT an exception mechanism.

- `CIRCUIT_BROKEN`: DPR circuit breaker has tripped — no new trades
- `AT_RISK`: Bot is underperforming — monitored closely
- `CIRCUIT_BROKEN` triggers `SafetyHooks.check_dpr_circuit()` to return False

When a bot is `CIRCUIT_BROKEN`, it is a **controlled state** managed via `SafetyHooks`, not an exception.

## BotHealth vs Exception

`BotHealth` (HEALTHY / DEGRADED / FAILING) communicates **bot operational health**. It is NOT an exception mechanism.

- `HEALTHY`: Normal operation
- `DEGRADED`: Some capabilities unavailable (partial data, degraded feed)
- `FAILING`: Critical failures — bot should be stopped

`FAILING` health should trigger `LibraryConfigError` or bridge-specific error, not runtime rejection.

## ERR-001 Through ERR-004 Completion Record

| Ticket | Description | Status |
|--------|-------------|--------|
| ERR-001 | Core error base (LibraryError, LibraryConfigError, ContractValidationError, BridgeError, BridgeUnavailableError) | ✓ Complete |
| ERR-002 | Bridge-specific subclasses + DependencyMissingError fix | ✓ Complete |
| ERR-003 | Minimal AuditRecord + optional ErrorSeverity | ✓ Complete |
| ERR-004 | ExecutionDirective additions (approved, rejection_reason — additive, safe defaults) | ✓ Complete |

All four tickets were committed across phases `c3fdd79` through `f9c279a`.
