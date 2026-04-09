# 10 — Feature Family Plan

**Version:** 1.2
**Date:** 2026-04-09
**Purpose:** Define V1 feature families, their inputs/outputs, dependencies, and classification

**Update 2026-04-09:** Added formal Feature Classification Rule (4-class system), Proxy Market Microstructure Layer section, and proxy/inferred feature modules per `quantmindx_v1_proxy_microstructure_addendum.md`.

---

## Feature Family Overview

### V1 Scope: 6 Feature Families

| Family | Purpose | Live | Historical | Bot-Facing | Sentinel-Facing | Shared | V1 Data Source | Quality Class |
|--------|---------|------|------------|-----------|----------------|--------|----------------|----------------|
| **indicators** | Technical indicators (RSI, ATR, MACD, VWAP) | Yes | Yes | Yes | No | Yes | cTrader (native) | Native Supported |
| **volume** | Volume-based features (RVOL, Volume Profile, MFI, Imbalance) | Yes | Yes | Yes | Partial | Yes | cTrader (native) | Native Supported |
| **microstructure** | Proxy market microstructure (spread, depth, aggression, pressure) | Yes | Partial | Yes | Yes | Yes | cTrader (native + proxy) | Native + Proxy |
| **orderflow_cat_a** | Depth/liquidity-state features (Spread, DOM, top-of-book) | Yes | Yes | Yes | No | Yes | cTrader (native, V1 supported) | Native Supported |
| **orderflow_cat_b** | Executed trade-flow features (delta, tape-speed, footprint) | Yes | Yes | Yes | No | Yes | External only | Future External-Data |
| **session** | Session-aware features (Session Detector, Blackout) | Yes | Yes | Yes | Yes | Yes | cTrader (native) | Native Supported |
| **transforms** | Data transformations (Normalize, Rolling, Resample) | Yes | Yes | Yes | No | Yes | cTrader (native) | Native Supported |

### Footprint Logic vs Charts
- **Footprint logic:** ALLOWED. Volume distribution, imbalance, delta-style behavior, pressure at levels — these are `VolumeImbalanceFeature` and are in scope.
- **Footprint chart rendering:** NOT REQUIRED. No rendering code should be in V1 scope.

### Chart Pattern Analysis
- **Status:** OUT-OF-SCOPE for V1
- **Future path:** Pattern service → `PatternSignal` objects → bots and/or sentinel consume outputs
- **Placeholder:** `PatternSignal` schema defined in shared object model for future integration

---

## Feature Classification Rule (Formal, Required)

Per `quantmindx_v1_proxy_microstructure_addendum.md`, every V1 feature must be classified into one of four classes. This classification is mandatory — it prevents semantic drift during implementation.

### The Four Classes

| Class | Definition | Quality Claim | V1 Status |
|-------|------------|-------------|-----------|
| **1. Native Supported** | Directly available from cTrader/IC Markets data without inference | Can claim HIGH quality | V1-active |
| **2. Proxy / Inferred** | Not a direct measurement; inferred from supported data using domain logic | Must label as `quality="proxy_inferred"` — not true executed flow | V1-active |
| **3. Future External-Data** | Requires true executed-trade order flow data; cTrader does not provide this | Requires external data source | V1-deferred, interface designed |
| **4. Deferred** | Intentional postpone — too heavy, too uncertain, or wrong layer for V1 | N/A | Out-of-scope for V1 |

### Classification Example Table

| Feature | Class | Why |
|---------|-------|-----|
| Spread width | Native Supported | Directly from bid/ask quotes |
| Depth imbalance | Native Supported | Derived from depth data |
| Liquidity thinning | Native Supported | Derived from depth changes |
| Aggression proxy | Proxy / Inferred | Inferred from quote/depth behavior; not directly observed |
| Absorption proxy | Proxy / Inferred | Inferred, not directly observed |
| DOM Pressure | Native Supported | Derived from depth data |
| Depth Thinning | Native Supported | Derived from depth changes |
| Volume Imbalance (delta) | Future External-Data | Requires executed trade side/volume — cTrader does not provide |
| Buy/sell volume delta | Future External-Data | Requires true executed trade data |
| Footprint execution metrics | Future External-Data | Requires true trade prints |
| Heavy chart-pattern engine | Deferred | Too heavy for V1 core |
| Neural pattern service | Deferred | V2+ scope |

### Quality Labeling Contract
Every feature result must carry a quality label:

```python
@dataclass
class FeatureResult:
    feature_id: str
    value: Any
    quality: Literal["native_supported", "proxy_inferred", "external", "deferred"]
    source: str                      # Which source produced this
    notes: Optional[str]              # e.g., "Inferred from quote/depth behavior; not true executed flow"
```

Features classified as **Proxy / Inferred** MUST include `notes` explaining what is being inferred and the method used. Do not claim proxy features as true executed-flow signals.

### Classification Placement
- Every `CapabilitySpec` declaration must include a `quality_class` field
- Every feature module class docstring must state its classification
- FeatureRegistry queries can filter by quality class

---

## Proxy Market Microstructure Layer

**Family ID:** `microstructure`
**Path:** `src/library/features/microstructure/`
**Purpose:** Derive short-horizon market-structure signals from depth, spread, quotes, and bar data without claiming access to executed trade flow.

Per addendum: The objective is to improve market understanding, context sensitivity, execution timing, bot selectivity, confirmation quality, and adaptation to session and liquidity conditions — without requiring true order flow data.

### Design Principles (from addendum)
1. **Do not call these "true order flow"** — they are proxies and inferences
2. **cTrader is the data source** — depth, spread, quotes, bars, session context, volume regime
3. **The layer is honest** — explicitly labeled as proxy/inferred in all contracts
4. **The layer integrates with bots, sentinel, evaluation, journaling, DPR, and future ML consumers**

### V1 Modules

#### Native Supported (Class 1)

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| `SpreadStateFeature` | Native Supported | Spread width, change rate, expansion/compression | bid/ask ticks | spread_width, spread_change_rate, spread_volatility |
| `TopOfBookPressureFeature` | Native Supported | Top-of-book pressure, bid/ask ratio | depth top 1-3 levels | pressure_ratio, imbalance_score, book_velocity |
| `MultiLevelDepthFeature` | Native Supported | Depth imbalance at multiple levels | depth 5-10 levels | imbalance_per_level, total_depth, depth_weighted_pressure |

#### Proxy / Inferred (Class 2)

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| `AggressionProxyFeature` | Proxy / Inferred | Short-horizon directional pressure estimate | quote drift, depth withdrawal, spread stability | aggression_score, direction, confidence (labeled as proxy) |
| `AbsorptionProxyFeature` | Proxy / Inferred | Estimate of whether large orders are being absorbed | depth changes, spread behavior | absorption_score, likely_reversal |
| `BreakoutPressureProxyFeature` | Proxy / Inferred | Estimate breakout strength from depth and spread | depth, spread, bar momentum | breakout_pressure, fakeout_probability |
| `LiquidityStressProxyFeature` | Proxy / Inferred | Short-horizon liquidity stress | depth thinning, spread widening | stress_score, liquidity_regime |

### Raw Input Objects

```python
@dataclass
class QuoteTick:
    symbol: str
    bid: float
    ask: float
    timestamp: datetime

@dataclass
class DepthSnapshot:
    symbol: str
    bids: List[PriceLevel]   # price, size, level_index
    asks: List[PriceLevel]
    timestamp: datetime

@dataclass
class SpreadState:
    spread_pips: float
    spread_change_pips: float  # delta from previous
    volatility: float           # rolling spread volatility

@dataclass
class DepthUpdate:
    symbol: str
    is_bid: bool               # side that changed
    level_index: int
    new_size: float
    timestamp: datetime
```

### Output Contracts

```python
@dataclass
class MicrostructureSignal:
    feature_id: str
    direction: Literal["bullish", "bearish", "neutral"]
    score: float               # -1.0 to 1.0
    confidence: float          # 0.0 to 1.0
    quality: Literal["native_supported", "proxy_inferred"]
    source: str                # Which module produced this
    notes: Optional[str]       # Mandatory for proxy_inferred
    timestamp: datetime

@dataclass
class LiquidityState:
    thinning_score: float      # 0.0 = normal, 1.0 = thin
    concentration: float       # 0.0-1.0, how concentrated at key levels
    regime: Literal["normal", "stress", "abundant"]
    quality: Literal["native_supported", "proxy_inferred"]
    notes: Optional[str]
    timestamp: datetime

@dataclass
class SpreadPressureSignal:
    spread_pips: float
    spread_change_pips: float
    expansion_factor: float    # ratio to session average
    regime: Literal["compressed", "normal", "expanded", "volatile"]
    quality: Literal["native_supported"]
    timestamp: datetime

@dataclass
class MicrostructureContext:
    """Aggregated higher-level context object consumed by both bots and sentinel."""
    symbol: str
    timestamp: datetime
    spread_regime: str
    liquidity_regime: str
    depth_pressure: str
    aggression_proxy: str
    breakout_pressure: str
    quality_flags: List[str]
    source_components: List[str]
```

### Placement in System

```
Raw market/depth ingestion → cTrader adapter boundary
       ↓
Reusable feature computation → QuantMindLib microstructure family
       ↓
Heavier shared aggregation → sentinel or shared service
       ↓
Bot consumption → FeatureVector, MarketContext, or bridge outputs
```

### Sync/Async Boundaries

- **Async by default:** quote stream, depth updates, bar updates
- **Sync by default:** feature validation, compatibility checks, config loads, cached context reads
- **Hybrid:** bot requests current MicrostructureContext → sync read → risk/sentinel validation → TradeIntent decision

### Integration with Existing ML Systems

Per addendum instruction: Do NOT duplicate existing market-intelligence logic. The library should:
- Expose reusable microstructure features
- Expose contracts and bridge objects
- Let sentinel aggregate or enrich where appropriate
- NOT blindly absorb the market-intelligence system

Current overlap to preserve (from code scan):
- **Ising-inspired features** (magnetization, susceptibility, energy) — already computed in HMMFeatureExtractor; do NOT duplicate
- **HMM regime classification** — sentinel owns this; library MicrostructureContext bridges to it
- **BOCPD changepoint detection** — sentinel owns this; library respects `is_transition` flag
- **MS-GARCH volatility regime** — sentinel owns this; library uses `sigma_forecast` from MarketContext
- **NewsSensor state** — sentinel owns this via NewsBlackoutService; library SessionBlackoutFeature wraps it

### Relationship to SentinelBridge

The SentinelBridge already maps `RegimeReport` → `MarketContext`. The MicrostructureContext extends this:
```
Sentinel.RegimeReport
    │
    ├─► RegimeReport → MarketContext (existing)
    │
    └─► MicrostructureContext (new, library-produced)
              ├─► spread_regime (from SpreadStateFeature)
              ├─► liquidity_regime (from LiquidityStressProxyFeature)
              ├─► depth_pressure (from MultiLevelDepthFeature)
              ├─► aggression_proxy (from AggressionProxyFeature)
              ├─► breakout_pressure (from BreakoutPressureProxyFeature)
              └─► quality_flags (from feature quality labels)
```

### Feature Registry: Microstructure Family

```python
feature_registry.register("microstructure/spread", SpreadStateFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/top_book", TopOfBookPressureFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/depth_levels", MultiLevelDepthFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/aggression", AggressionProxyFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/absorption", AbsorptionProxyFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/breakout_pressure", BreakoutPressureProxyFeature(), CapabilitySpec(...))
feature_registry.register("microstructure/liquidity_stress", LiquidityStressProxyFeature(), CapabilitySpec(...))
```

### Indexed References

| Feature | Codebase Reference | Addendum Reference |
|---------|-------------------|---------------------|
| Ising features (magnetization, susceptibility, energy) | `src/risk/physics/hmm/features.py` (FeatureConfig) | Addendum §2B |
| HMM regime | `src/risk/physics/hmm_sensor.py` (HMMRegimeSensor) | Addendum §2A |
| BOCPD changepoint | `src/risk/physics/bocpd/detector.py` (BOCPDDetector) | Addendum §2C |
| MS-GARCH volatility | `src/risk/physics/msgarch/models.py` (MSGARCHSensor) | Addendum §2A |
| News state | `src/router/sensors/news.py` (NewsSensor) | Addendum §2A |
| Spread from cTrader | `09_ctrader_boundary_plan.md` §8 | Addendum §2A |
| Depth from cTrader | `09_ctrader_boundary_plan.md` §8 | Addendum §2A |

---

## FF-1: Indicators

**Family ID:** `indicators`
**Path:** `src/library/features/indicators/`
**Live/Historical:** Both

### Base Class
```python
class IndicatorFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> IndicatorResult
    # Required fields: requires=[TickStream], outputs=[IndicatorResult]
```

### V1 Modules

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| RSI | `RSIFeature` | Relative Strength Index | close prices | RSI value (0-100) |
| ATR | `ATRFeature` | Average True Range | high, low, close | ATR value in pips |
| MACD | `MACDFeature` | Moving Average Convergence Divergence | close prices | macd_line, signal_line, histogram |
| VWAP | `VWAPFeature` | Volume-Weighted Average Price | close, volume, session | VWAP value |

### Dependency Declaration (CapabilitySpec)
```python
RSIFeature.capability_spec = CapabilitySpec(
    module_id="indicators/rsi",
    provides=["indicators/rsi"],
    requires=["data/close_prices"],
    optional=[],
    outputs=[IndicatorResult],
    compatibility=["scalping", "breakout", "orb"],
    sync_mode=True
)
```

### Relationship to Existing Code
- **SVSS VWAPIndicator** → wrap as `VWAPFeature` (reuse, don't rewrite)
- **Asset library** (RSI, MACD, Bollinger) → existing MQL5 indicators, not Python
- **New implementation:** Python-based feature modules for library core

---

## FF-2: Volume

**Family ID:** `volume`
**Path:** `src/library/features/volume/`
**Live/Historical:** Both

### Base Class
```python
class VolumeFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> VolumeResult
```

### V1 Modules

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| RVOL | `RVOLFeature` | Relative Volume vs session average | volume, session_avg_volume | RVOL ratio |
| Volume Profile | `VolumeProfileFeature` | POC, value area, profile distribution | tick volume, price levels | POC price, VA high/low, profile |
| MFI | `MFIFeature` | Money Flow Index | high, low, close, volume | MFI value (0-100) |
| Imbalance | `VolumeImbalanceFeature` | Delta, buy/sell pressure at levels | tick volume, tick direction | imbalance_ratio, pressure_score | ⚠️ External required (Category B)

### Footprint Logic Detail
The `VolumeImbalanceFeature` provides footprint-derived logic:
- **Delta:** net buy/sell volume at each price level
- **Pressure:** directional volume imbalance
- **POC (Point of Control):** price level with highest volume
- **Value Area:** price range containing 70% of volume

These are logic outputs. No visual rendering.

### Dependency Declaration
```python
VolumeImbalanceFeature.capability_spec = CapabilitySpec(
    module_id="volume/imbalance",
    provides=["volume/imbalance", "volume/delta"],
    requires=["data/tick_stream", "data/volume"],
    optional=["data/depth_stream"],
    outputs=[VolumeResult],
    compatibility=["scalping", "orb"],
    sync_mode=True
)
```

### Quality-Aware Tagging
Volume features must tag outputs with source quality:
```python
@dataclass
class VolumeResult:
    feature_id: str
    value: float
    confidence: FeatureConfidence  # source: "ctrader_native" / "proxied" / "approximated"
    timestamp: datetime
    metadata: dict  # e.g., {"poc_price": 1.0850, "va_high": 1.0860}
```

### Relationship to Existing Code
- **SVSS RVOLIndicator** → wrap as `RVOLFeature`
- **SVSS VolumeProfileIndicator** → wrap as `VolumeProfileFeature`
- **SVSS MFIIndicator** → wrap as `MFIFeature`
- **New:** `VolumeImbalanceFeature` (not yet in SVSS, new V1 development)

---

## FF-3: Order Flow (Two Categories)

**Family ID:** `orderflow`
**Path:** `src/library/features/orderflow/`
**Live/Historical:** Live only (requires tick stream)

### Architecture Decision: Two-Category Order Flow Split

Per architecture decision (2026-04-08): Order flow is split into two distinct categories with different data sources and V1 support status. This supersedes the single monolith approach.

### Category A: Depth/Liquidity-State Features (V1 Supported — cTrader)

cTrader Open API provides real-time DOM and quote data that supports depth/liquidity-state features. These are V1 in scope.

**Base Class:**
```python
class DepthLiquidityFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> DepthLiquiditySignal
```

**V1 Modules (Category A — V1 Supported via cTrader):**

| Module | Class | Description | Inputs | Outputs | V1 Support |
|--------|-------|-------------|--------|---------|------------|
| Spread Behavior | `SpreadBehaviorFeature` | Spread dynamics, volatility of spread | bid/ask ticks | spread_score, spread_volatility | ✅ V1 Supported |
| DOM Pressure | `DOMPressureFeature` | Top-of-book pressure, depth imbalance | depth delta events | pressure_ratio, imbalance_score | ✅ V1 Supported |
| Depth Thinning | `DepthThinningFeature` | Depth thinning/thickening detection | depth stream | thinning_score, liquidity_state | ✅ V1 Supported |

**Category A Dependency Declaration:**
```python
SpreadBehaviorFeature.capability_spec = CapabilitySpec(
    module_id="orderflow/spread",
    provides=["orderflow/spread", "liquidity/spread"],
    requires=["data/tick_stream"],
    optional=[],
    outputs=[OrderFlowSignal],
    compatibility=["scalping", "orb"],
    sync_mode=True
)
```

**Quality Tagging (Category A):**
- cTrader DOM quote data → `FeatureConfidence.source = "ctrader_native"`, quality = HIGH
- These features are V1-supported per `09_ctrader_boundary_plan.md` §8

### Category B: Executed Trade-Flow Features (External Source Required)

cTrader Open API does NOT expose executed trade ticks, buy/sell volume delta, or true trade flow. These features require an external data source. Per architecture decision, Category B features are V1-deferred but the interface is designed now.

**Interface Design (V1 placeholder, not implementation):**
```python
class IExternalOrderFlowAdapter(ABC):
    """Optional adapter for external true-executed-trade order flow data."""
    @async
    def order_flow_stream(self, symbols: List[str]) -> AsyncIterator[OrderFlowTick]: ...
    def get_historical_flow(self, symbol: str, start: datetime, end: datetime) -> List[OrderFlowTick]: ...
    @property
    def source_name(self) -> str: ...

class ExecutedTradeFlowFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> ExecutedTradeFlowSignal
    # Requires IExternalOrderFlowAdapter — not available without external source
```

**V1 Modules (Category B — V1 Deferred, Interface Designed):**

| Module | Class | Description | Inputs | Outputs | V1 Support |
|--------|-------|-------------|--------|---------|------------|
| Volume Imbalance | `VolumeImbalanceFeature` | Delta, buy/sell pressure at levels | tick volume, tick direction | imbalance_ratio, pressure_score | ❌ Deferred (external required) |
| Tick Activity | `TickActivityFeature` | Tick rate, momentum, micro-trends | tick stream | SignalDirection, strength | ❌ Deferred (external required) |
| Session Volume | `SessionVolumeFeature` | Session-relative volume | volume, session_context | session_volume_ratio | ⚠️ Partial (cTrader volume available) |

**Category B Degradation Strategy:**
```
External true-order-flow available → HIGH quality → Feature ENABLED
External true-order-flow unavailable → Feature DISABLED (no degraded proxy)
```
Per REVIEW-5: HIGH quality required. If not available, features are disabled — not degraded.

**Note:** `SessionVolumeFeature` is partially available because cTrader provides OHLCV volume data (not tick-level buy/sell delta). Use `FeatureConfidence.source = "approximated"` with quality = MEDIUM when only OHLCV volume is available.

### Relationship to Existing Code
- **No existing true trade flow** in codebase — Category B requires external source
- **SVSS tick processing** provides tick data → Category A features build on this
- **cTrader DOM** provides best quality depth/liquidity-state data (Category A)

### External Order Flow Adapter Design Principle
- cTrader execution remains **independent** of whether external order-flow is connected
- Library feature modules consume `OrderFlowTick` regardless of source
- Adapter is optional — library functions without it (Category A features still active)
- cTrader adapter and external order-flow adapter operate independently
- Future extension: venue-native FX order book feeds, futures-based order-flow feeds

### V1 Supported Feature Subset (Order Flow)
Only these order flow features are V1-active with cTrader data:
- `SpreadBehaviorFeature` (bid/ask dynamics)
- `DOMPressureFeature` (depth imbalance)
- `DepthThinningFeature` (liquidity state)

All other order flow features require external data source (V1-deferred).

---

## FF-4: Session

**Family ID:** `session`
**Path:** `src/library/features/session/`
**Live/Historical:** Both

### Base Class
```python
class SessionFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> SessionResult
```

### V1 Modules

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| Session Detector | `SessionDetectorFeature` | Active session, session transitions | server_time | session_id, is_active |
| Session Blackout | `SessionBlackoutFeature` | News blackout status per session | news_state, calendar | is_blackout, blackout_reason |

### Session Detection Logic
Session definitions from codebase (R-12):
```python
SESSIONS = {
    "tokyo": {"currencies": ["JPY", "AUD", "NZD", "CNY"], "start": "00:00", "end": "09:00"},
    "sydney": {"currencies": ["AUD", "NZD"], "start": "22:00", "end": "07:00"},
    "london": {"currencies": ["EUR", "GBP", "CHF"], "start": "08:00", "end": "17:00"},
    "new_york": {"currencies": ["USD", "CAD"], "start": "13:00", "end": "22:00"},
    "london_newyork_overlap": {"currencies": ["EUR", "GBP", "USD", "CHF", "CAD"], "start": "13:00", "end": "17:00"},
}
```

### Dependency Declaration
```python
SessionDetectorFeature.capability_spec = CapabilitySpec(
    module_id="session/detector",
    provides=["session/detector"],
    requires=["data/server_time"],
    optional=[],
    outputs=[SessionResult],
    compatibility=["scalping", "orb", "mean_reversion"],
    sync_mode=True
)
```

### Relationship to Existing Code
- **SessionDetector** (`src/router/session_detector.py`) → wrap as `SessionDetectorFeature`
- **NewsBlackoutService** (`src/market/news_blackout.py`) → wrap as `SessionBlackoutFeature`
- **CalendarGovernor** (`src/router/calendar_governor.py`) → integrate via SessionContext

### Economic Calendar Gap
Currently: Finnhub polling only (every 30 min, 7-day lookahead, 15-min kill zone).
Gap: No direct economic calendar feeds NewsSensor programmatically beyond Finnhub.
V1 bridge should wrap existing NewsBlackoutService, not create a new calendar integration.

---

## FF-5: Transforms

**Family ID:** `transforms`
**Path:** `src/library/features/transforms/`
**Live/Historical:** Both

### Base Class
```python
class Transform(FeatureModule):
    # ABC with transform(data: pd.Series) -> pd.Series
```

### V1 Modules

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| Normalize | `NormalizeTransform` | Min-max or z-score normalization | numeric series | normalized series |
| Rolling Window | `RollingWindowTransform` | Rolling stats (mean, std, min, max) | numeric series | rolling stats |
| Resample | `ResampleTransform` | Timeframe resampling (M5→H1, etc.) | bar series | resampled bar series |

### Dependency Declaration
```python
NormalizeTransform.capability_spec = CapabilitySpec(
    module_id="transforms/normalize",
    provides=["transforms/normalize"],
    requires=["data/numeric_series"],
    optional=[],
    outputs=[TransformedSeries],
    compatibility=["scalping", "orb", "mean_reversion", "breakout"],
    sync_mode=True
)
```

---

## Feature Registry Integration

### Registration Pattern
```python
feature_registry = FeatureRegistry()

# Register each module with its capability spec
feature_registry.register("indicators/rsi", RSIFeature(), CapabilitySpec(...))
feature_registry.register("indicators/atr", ATRFeature(), CapabilitySpec(...))
feature_registry.register("volume/rvol", RVOLFeature(), CapabilitySpec(...))
feature_registry.register("volume/profile", VolumeProfileFeature(), CapabilitySpec(...))
feature_registry.register("volume/mfi", MFIFeature(), CapabilitySpec(...))
feature_registry.register("volume/imbalance", VolumeImbalanceFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/spread", SpreadBehaviorFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/dom_pressure", DOMPressureFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/depth_thinning", DepthThinningFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/volume_imbalance", VolumeImbalanceFeature(), CapabilitySpec(...))  # External required
feature_registry.register("orderflow/tick", TickActivityFeature(), CapabilitySpec(...))  # External required
feature_registry.register("orderflow/session_vol", SessionVolumeFeature(), CapabilitySpec(...))  # Partial - OHLCV available
feature_registry.register("session/detector", SessionDetectorFeature(), CapabilitySpec(...))
feature_registry.register("session/blackout", SessionBlackoutFeature(), CapabilitySpec(...))
feature_registry.register("transforms/normalize", NormalizeTransform(), CapabilitySpec(...))
feature_registry.register("transforms/rolling", RollingWindowTransform(), CapabilitySpec(...))
feature_registry.register("transforms/resample", ResampleTransform(), CapabilitySpec(...))
```

### Composition Validation
Before composing a bot from BotSpec:
```python
# Check all features in BotSpec are available
for feature_id in BotSpec.features:
    if not feature_registry.get(feature_id):
        raise FeatureNotFoundError(feature_id)

# Check all dependencies are satisfied
for feature_id in BotSpec.features:
    feature = feature_registry.get(feature_id)
    for dep in feature.capability_spec.requires:
        if not feature_registry.get(dep):
            raise DependencyMissingError(dep, feature_id)

# Check compatibility with archetype
archetype = archetype_registry.get(BotSpec.archetype)
for feature_id in BotSpec.features:
    feature = feature_registry.get(feature_id)
    if feature_id not in archetype.capability_spec.compatibility:
        raise CompatibilityError(feature_id, BotSpec.archetype)
```

---

## Feature Family Dependency Map

### 7th Family: Proxy Market Microstructure
```
data/tick_stream ────► SpreadStateFeature ────► MicrostructureSignal
data/depth_stream ───► TopOfBookPressureFeature ────► MicrostructureSignal
data/depth_stream ───► MultiLevelDepthFeature ────► MicrostructureSignal
data/depth_stream ───► DepthThinningFeature ────► LiquidityState
data/depth_stream ───► LiquidityStressProxyFeature ────► LiquidityState

quote + depth + spread ─► AggressionProxyFeature ───► MicrostructureSignal (proxy_inferred)
depth + spread + bar ─► BreakoutPressureProxyFeature ───► MicrostructureSignal (proxy_inferred)
depth + spread ─► AbsorptionProxyFeature ───► MicrostructureSignal (proxy_inferred)
```

### Category A: Depth/Liquidity-State (V1 Supported via cTrader)
```
data/tick_stream ────► SpreadBehaviorFeature ────► OrderFlowSignal (HIGH quality via cTrader)
                 └──► DOMPressureFeature ────► OrderFlowSignal (HIGH quality via cTrader)

data/depth_stream ───► DepthThinningFeature ────► OrderFlowSignal (HIGH quality via cTrader)
                 └──► DOMPressureFeature ────► OrderFlowSignal
```

### Category B: Executed Trade-Flow (External Source Required, V1 Deferred)
```
external_order_flow ──► VolumeImbalanceFeature ────► VolumeResult (HIGH quality via external)
external_order_flow ──► TickActivityFeature ────► OrderFlowSignal (HIGH quality via external)

OHLCV_volume ─────────► SessionVolumeFeature ────► OrderFlowSignal (MEDIUM quality via cTrader)
```

### Other Feature Families
```
data/tick_stream ────► RVOLFeature ────► VolumeResult
                 └──► MFIFeature ────► VolumeResult

data/volume ──────────► RVOLFeature ────► VolumeResult
                   ├──► VolumeProfileFeature ────► VolumeResult
                   └──► MFIFeature ────► VolumeResult

data/depth_stream ────► VolumeImbalanceFeature (optional, external required)

data/close_prices ───► RSIFeature ────► IndicatorResult
                   ├──► ATRFeature ────► IndicatorResult
                   └──► MACDFeature ────► IndicatorResult

data/server_time ────► SessionDetectorFeature ────► SessionResult
                   └──► SessionBlackoutFeature ────► SessionResult

news_state ──────────► SessionBlackoutFeature ────► SessionResult

indicator_results ──► NormalizeTransform ────► TransformedSeries
                  ├──► RollingWindowTransform ────► TransformedSeries
                  └──► ResampleTransform ────► TransformedSeries
```

---

## Indexed References

### Feature Classification References

| Class | Feature | V1 Status | Codebase Reference | Addendum § |
|-------|---------|-----------|-------------------|-------------|
| Native Supported | Spread dynamics | V1-active | cTrader DOM quote data | §2A |
| Native Supported | Depth imbalance | V1-active | cTrader DOM depth | §2A |
| Native Supported | Top-of-book pressure | V1-active | cTrader quote stream | §2A |
| Native Supported | Liquidity thinning | V1-active | cTrader DOM depth changes | §2A |
| Native Supported | Spread width/change rate | V1-active | cTrader quote stream | §2A |
| Native Supported | Multi-level depth imbalance | V1-active | cTrader DOM depth | §2A |
| Proxy / Inferred | Aggression proxy | V1-active | cTrader + domain logic | §2B |
| Proxy / Inferred | Absorption proxy | V1-active | cTrader + domain logic | §2B |
| Proxy / Inferred | Breakout pressure proxy | V1-active | cTrader + domain logic | §2B |
| Proxy / Inferred | Liquidity stress proxy | V1-active | cTrader + domain logic | §2B |
| Future External-Data | Volume imbalance (delta) | V1-deferred | — | §2C |
| Future External-Data | Buy/sell volume delta | V1-deferred | — | §2C |
| Future External-Data | Tape speed | V1-deferred | — | §2C |
| Future External-Data | Aggressive-flow metrics | V1-deferred | — | §2C |
| Future External-Data | Footprint execution metrics | V1-deferred | — | §2C |
| Deferred | Heavy chart-pattern engine | V1-out | — | §2D |
| Deferred | Neural pattern service | V2+ | — | §2D |
| Deferred | ML feature integration | V2+ | — | §2D |

### Code and Memo References

| Feature | Codebase Reference | Memo Reference | Recovery Note |
|---------|-------------------|----------------|---------------|
| VWAP | `src/svss/indicators/vwap.py` (VWAPIndicator) | §8 | R-11 |
| RVOL | `src/svss/indicators/rvol.py` (RVOLIndicator) | §8 | R-11 |
| Volume Profile | `src/svss/indicators/volume_profile.py` (VolumeProfileIndicator) | §8 | R-11 |
| MFI | `src/svss/indicators/mfi.py` (MFIIndicator) | §8 | R-11 |
| Session Detector | `src/router/session_detector.py` | §8 | R-12 |
| News Blackout | `src/market/news_blackout.py` (SessionBlackoutStatus) | §8 | R-12 |
| Economic Calendar | — (gap) | — | R-12 |
| FeatureConfig | `src/risk/physics/hmm/features.py` | §8 | — |
| Asset Library | `src/mcp_tools/asset_library.py` (AssetLibraryManager) | §8 | — |
| HMM regime | `src/risk/physics/hmm_sensor.py`, `src/risk/physics/ensemble/voter.py` | §2A | R-1 |
| MS-GARCH volatility | `src/risk/physics/msgarch/models.py` | §2A | R-1 |
| BOCPD changepoint | `src/risk/physics/bocpd/detector.py` | §2C | R-1 |
| EnsembleVoter | `src/risk/physics/ensemble/voter.py` | — | R-1 (not wired in tick path) |
| Ising features | `src/risk/physics/hmm/features.py` | §2B | R-1 |
| DPR Redis | `src/router/dpr_scoring_engine.py` | — | G-18 (uncommitted fix) |
| Agent SDK migration | `src/agents/providers/profile_runtime.py` | — | Session 2026-03-30 |