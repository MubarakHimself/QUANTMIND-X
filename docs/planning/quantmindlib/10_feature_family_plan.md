# 10 — Feature Family Plan

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Define V1 feature families, their inputs/outputs, and dependencies

---

## Feature Family Overview

### V1 Scope: 5 Feature Families

| Family | Purpose | Live | Historical | Bot-Facing | Sentinel-Facing | Shared |
|--------|---------|------|------------|-----------|----------------|--------|
| **indicators** | Technical indicators (RSI, ATR, MACD, VWAP) | Yes | Yes | Yes | No | Yes |
| **volume** | Volume-based features (RVOL, Volume Profile, MFI, Imbalance) | Yes | Yes | Yes | Partial | Yes |
| **orderflow** | Order flow proxies (Tick Activity, Spread Behavior, Session Volume) | Yes | Yes | Yes | No | Yes |
| **session** | Session-aware features (Session Detector, Blackout) | Yes | Yes | Yes | Yes | Yes |
| **transforms** | Data transformations (Normalize, Rolling, Resample) | Yes | Yes | Yes | No | Yes |

### Footprint Logic vs Charts
- **Footprint logic:** ALLOWED. Volume distribution, imbalance, delta-style behavior, pressure at levels — these are `VolumeImbalanceFeature` and are in scope.
- **Footprint chart rendering:** NOT REQUIRED. No rendering code should be in V1 scope.

### Chart Pattern Analysis
- **Status:** OUT-OF-SCOPE for V1
- **Future path:** Pattern service → `PatternSignal` objects → bots and/or sentinel consume outputs
- **Placeholder:** `PatternSignal` schema defined in shared object model for future integration

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
| Imbalance | `VolumeImbalanceFeature` | Delta, buy/sell pressure at levels | tick volume, tick direction | imbalance_ratio, pressure_score |

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

## FF-3: Order Flow

**Family ID:** `orderflow`
**Path:** `src/library/features/orderflow/`
**Live/Historical:** Live only (requires tick stream)

### Base Class
```python
class OrderFlowFeature(FeatureModule):
    # ABC with compute(context: FeatureContext) -> OrderFlowSignal
```

### V1 Modules

| Module | Class | Description | Inputs | Outputs |
|--------|-------|-------------|--------|---------|
| Tick Activity | `TickActivityFeature` | Tick rate, momentum, micro-trends | tick stream | SignalDirection, strength (0-1.0) |
| Spread Behavior | `SpreadBehaviorFeature` | Spread dynamics, volatility of spread | bid/ask ticks | spread_score, spread_volatility |
| Session Volume | `SessionVolumeFeature` | Session-relative volume | volume, session_context | session_volume_ratio |

### Order Flow Quality Tagging
Per trading_platform_decision_memo §5: order flow features must carry feed quality:
```python
OrderFlowFeature.capability_spec = CapabilitySpec(
    module_id="orderflow/tick_activity",
    provides=["orderflow/tick_activity"],
    requires=["data/tick_stream"],
    optional=[],
    outputs=[OrderFlowSignal],  # OrderFlowSignal includes FeatureConfidence
    compatibility=["scalping", "orb"],
    sync_mode=True
)
```

When cTrader DOM data is available: `FeatureConfidence.source = "ctrader_native"`, quality = HIGH.
When only OHLCV approximation: `FeatureConfidence.source = "approximated"`, quality = MEDIUM.

### Relationship to Existing Code
- **No existing order flow** in codebase — all new development
- **Partial:** SVSS tick processing provides tick data; order flow features build on top
- **cTrader DOM** provides best quality tick activity data

### Degradation Strategy
```
Available: cTrader DOM tick-level → HIGH quality order flow
Fallback: OHLCV tick approximation → MEDIUM quality
Fallback: Session-relative proxy → LOW quality
Fallback: Disabled → no order flow signal
```

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
feature_registry.register("orderflow/tick", TickActivityFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/spread", SpreadBehaviorFeature(), CapabilitySpec(...))
feature_registry.register("orderflow/session_vol", SessionVolumeFeature(), CapabilitySpec(...))
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

```
data/tick_stream ────► TickActivityFeature ────► OrderFlowSignal
                 ├──► SpreadBehaviorFeature ────► OrderFlowSignal
                 ├──► RVOLFeature ────► VolumeResult
                 ├──► MFIFeature ────► VolumeResult
                 └──► SessionVolumeFeature ────► OrderFlowSignal

data/volume ──────────► RVOLFeature ────► VolumeResult
                   ├──► VolumeProfileFeature ────► VolumeResult
                   ├──► VolumeImbalanceFeature ───► VolumeResult
                   └──► MFIFeature ────► VolumeResult

data/depth_stream ────► VolumeImbalanceFeature (optional)

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