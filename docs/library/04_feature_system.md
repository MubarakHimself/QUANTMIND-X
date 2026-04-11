# QuantMindLib V1 — Feature System

## Feature Registry Model

The FeatureRegistry (`src/library/features/registry.py`) is the central registry. It is a **singleton** bootstrapped by `get_default_registry()` in `src/library/features/_registry.py`.

```python
class FeatureRegistry:
    _features: Dict[str, FeatureModule]
    _by_family: Dict[str, List[str]]

    def register(self, feature: FeatureModule, family: str) -> None
    def get(self, feature_id: str) -> Optional[FeatureModule]
    def list_all(self) -> List[str]
    def list_by_family(self, family: str) -> List[str]
    def families(self) -> List[str]
    def validate_composition(self, feature_ids: List[str]) -> List[str]
    # raises DependencyMissingError on first missing feature
```

Bootstrap is **explicit and deterministic** — no import magic, no auto-discovery. Each family `__init__.py` imports its features and calls `reg.register()`.

```python
# src/library/features/_registry.py
def get_default_registry() -> FeatureRegistry:
    # Double-check locking with RLock for thread safety
    # _build_registry() imports all families and calls reg.register() explicitly
```

## Feature Classification Rule

Every feature must be classified. This is a mandatory semantic constraint.

| Class | Quality Label | Claim | V1 Status |
|-------|--------------|-------|-----------|
| **1. Native Supported** | `quality_class="native_supported"` | Directly from cTrader data, no inference | V1-active |
| **2. Proxy / Inferred** | `quality_class="proxy_inferred"` | Inferred from supported data via domain logic. Must include `notes` field. NOT true executed flow | V1-active |
| **3. Future External-Data** | `quality_class="external"` | Requires true executed trade flow — cTrader doesn't provide | V1-deferred |
| **4. Deferred** | N/A | Intentional postpone | Out-of-scope |

**Proxy/Inferred features MUST include `notes`** explaining: what is being inferred, the method, and what data supports it. Example:

```python
@property
def config(self) -> FeatureConfig:
    return FeatureConfig(
        feature_id=f"microstructure/aggression_{self.period}",
        quality_class="proxy_inferred",
        source="AggressionProxyFeature",
        notes=(
            "Inferred from quote/depth behavior; not true executed flow. "
            "Direction inferred from consecutive bid-pushes against thinning asks. "
            "spread_penalty reduces confidence when spread is unstable."
        ),
    )
```

## Implemented Feature Families (16 features)

### indicators/ (4 features)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| RSIFeature | `indicators/rsi.py` | native_supported | Wilder's RSI, period=14 |
| ATRFeature | `indicators/atr.py` | native_supported | True Range in FX pips, period=14 |
| MACDFeature | `indicators/macd.py` | native_supported | EMA-based MACD, fast=12/slow=26/signal=9 |
| VWAPFeature | `indicators/vwap.py` | native_supported | Typical-price weighted average |

### volume/ (3 features)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| RVOLFeature | `volume/rvol.py` | native_supported | Relative volume vs session+day factors |
| MFIFeature | `volume/mfi.py` | native_supported | Money Flow Index, period=14 |
| VolumeProfileFeature | `volume/profile.py` | native_supported | POC + value area, 20-bin histogram |

### microstructure/ (9 features — see `05_proxy_microstructure_layer.md`)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| SpreadStateFeature | `microstructure/spread.py` | native_supported | Spread dynamics from quotes |
| TopOfBookPressureFeature | `microstructure/tob_pressure.py` | native_supported | Bid/ask imbalance |
| MultiLevelDepthFeature | `microstructure/depth.py` | native_supported | Depth imbalance at multiple levels |
| AggressionProxyFeature | `microstructure/aggression.py` | proxy_inferred | Directional pressure estimate |
| AbsorptionProxyFeature | `microstructure/absorption.py` | proxy_inferred | Absorption detection |
| BreakoutPressureProxyFeature | `microstructure/breakout_pressure.py` | proxy_inferred | Breakout strength estimate |
| LiquidityStressProxyFeature | `microstructure/liquidity_stress.py` | proxy_inferred | Short-horizon liquidity stress |
| TickActivityFeature | `microstructure/tick_activity.py` | external | Tick frequency; V1-deferred |
| VolumeImbalanceFeature | `microstructure/volume_imbalance.py` | external | Buy/sell volume delta; V1-deferred |

### orderflow/ (3 features)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| SpreadBehaviorFeature | `orderflow/spread_behavior.py` | native_supported | Spread efficiency vs volume |
| DOMPressureFeature | `orderflow/dom_pressure.py` | native_supported | Multi-level depth pressure |
| DepthThinningFeature | `orderflow/depth_thinning.py` | native_supported | Book depth thinning |

### session/ (2 features)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| SessionDetectorFeature | `session/detector.py` | native_supported | 5 sessions: Sydney/Tokyo/London/NY/overlap |
| SessionBlackoutFeature | `session/blackout.py` | native_supported | News blackout + kill zone |

### transforms/ (3 features)
| Feature | File | Quality | Notes |
|---------|------|---------|-------|
| NormalizeTransform | `transforms/normalize.py` | native_supported | MinMax + Z-score normalization |
| RollingWindowTransform | `transforms/rolling.py` | native_supported | Rolling mean/std/min/max |
| ResampleTransform | `transforms/resample.py` | native_supported | M1→M5 timeframe resampling |

## FeatureModule ABC Contract

```python
class FeatureModule(BaseModel):
    feature_id: str
    quality_class: str = "native_supported"
    source: str
    enabled: bool = True

    @property
    def config(self) -> FeatureConfig:
        """Override to provide FeatureConfig with quality_class, source, notes."""

    @property
    def required_inputs(self) -> Set[str]:
        """Override: declare required input keys."""

    @property
    def output_keys(self) -> Set[str]:
        """Override: declare produced output keys."""

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        """Override: compute features from inputs."""
```

## MicrostructureFeature ABC (`src/library/features/microstructure/microstructure_base.py`)

Extends FeatureModule with `notes` field requirement for proxy features.

```python
class MicrostructureFeature(FeatureModule):
    # Extends FeatureModule
    # Subclasses override config to declare quality_class and notes
```

## FeatureEvaluator (`src/library/runtime/feature_evaluator.py`)

Evaluates a feature stack for a bot. Handles dependency resolution and chaining.

```python
class FeatureEvaluator:
    def __init__(self, registry: Optional[FeatureRegistry] = None)
    def evaluate(
        self,
        bot_id: str,
        feature_ids: List[str],
        inputs: Dict[str, Any],
    ) -> FeatureVector
    def can_evaluate(
        self,
        feature_ids: List[str],
        available_inputs: Set[str],
    ) -> Tuple[bool, List[str]]  # (can_evaluate, missing_inputs)
```

Key behaviors:
- Iterative evaluation with max_iterations cap to prevent infinite loops (circular dependencies)
- Only computes features whose `required_inputs` are satisfied by available inputs
- Supports chained dependencies: output of one feature becomes input to another
- Aggregates FeatureVector from all computable modules

## Registry Bootstrap

```python
# src/library/features/_registry.py
_registry: FeatureRegistry | None = None
_lock = threading.RLock()

def get_default_registry() -> FeatureRegistry:
    with _lock:
        if _registry is None:
            _registry = _build_registry()
        return _registry

def _build_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    # Explicit imports from each family __init__.py
    # Explicit register() calls for each feature
    return reg
```

## Quality Propagation

`FeatureModule.model_post_init()` copies `quality_class` from the `config` property to the instance when it differs from the class default:

```python
def model_post_init(self, __context) -> None:
    cfg = self.config
    if self.quality_class == "native_supported" and cfg.quality_class != "native_supported":
        object.__setattr__(self, "quality_class", cfg.quality_class)
```

This ensures proxy features (which declare `quality_class="proxy_inferred"` in their `config` property) get the correct quality_class at registration time. The condition is correct: it evaluates to True for proxy features.

## FeatureRegistry.validate_composition

```python
def validate_composition(self, feature_ids: List[str]) -> List[str]:
    """
    Check if all feature_ids are registered.
    Raises DependencyMissingError on first missing feature (not all collected).
    Returns [] if all present.
    """
```

## Notes on Wrappers vs Standalone

- **RSI, ATR, MACD, VWAP**: Standalone implementations (Wilder's RSI, True Range, etc.) — not wrapping SVSS indicators
- **RVOL, MFI, VolumeProfile**: Standalone implementations — not wrapping SVSS indicators
- **Microstructure features**: Standalone implementations using DOM/depth data
- **Session features**: Wrappers conceptually for session_detector and news_blackout, but implemented standalone
- **Transforms**: Standalone data transformations

The planning docs indicated wrapping SVSS indicators, but the actual implementation chose standalone implementations. Both approaches produce correct feature values; the choice doesn't affect the library's correctness.
