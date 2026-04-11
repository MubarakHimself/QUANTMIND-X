# QuantMindLib V1 — Proxy Market Microstructure Layer

## Purpose

The proxy market microstructure layer provides **short-horizon market-structure signals** derived from depth, spread, and quote data available from cTrader. These signals help bots make better decisions about entry timing, liquidity acceptance, and breakout confirmation — **without claiming access to true executed trade flow**.

This is the honest layer: every feature is labeled correctly.

## Semantic Honesty Principle

**The proxy microstructure layer does NOT provide true order flow.** It provides:

- Signals from quote behavior (bid/ask drift, spread changes)
- Signals from depth behavior (concentration, thinning, imbalance at levels)
- Domain-logic inferences about aggression, absorption, breakout pressure

All of these are **inferred** from observable market data, not **observed** from executed trade data.

## What cTrader Provides (Class 1: Native Supported)

| Data Type | cTrader Source | Quality |
|-----------|---------------|---------|
| Bid/ask quotes (tick stream) | `tick_stream` via cTrader Open API | HIGH |
| Spread width and change | Computed from bid/ask | HIGH |
| Depth of market (DOM) | `depth_stream` via cTrader Open API | HIGH |
| Bid/ask volume at levels | Derived from DOM | HIGH |
| Multi-level depth imbalance | Computed from DOM levels | HIGH |

## What the Proxy Layer Computes (Class 2: Proxy/Inferred)

| Feature | Inference Method | NOT Claimed As |
|---------|-----------------|---------------|
| Aggression proxy | Consecutive bid-pushes against thinning asks | True aggression from tape |
| Absorption proxy | Large orders consuming depth without price impact | True absorption from trade prints |
| Breakout pressure | Depth withdrawal + spread expansion + momentum | True breakout confirmation from volume |
| Liquidity stress | Depth thinning rate, concentration at key levels | True liquidity from L2 data |

## Implemented Features

### Native Supported (Class 1)

#### SpreadStateFeature (`src/library/features/microstructure/spread.py`)
Spread dynamics from bid/ask quotes.

```python
output_keys: {f"spread_state_{symbol}"}
computed: spread_pips, spread_change_pips, expansion_factor, regime
confidence: 0.95, source="library_microstructure"
```

#### TopOfBookPressureFeature (`src/library/features/microstructure/tob_pressure.py`)
Top-of-book pressure, bid/ask imbalance.

```python
output_keys: {"tob_pressure", "tob_ratio"}
computed: bid_ask_ratio, imbalance_score
confidence: 0.95, source="library_microstructure"
```

#### MultiLevelDepthFeature (`src/library/features/microstructure/depth.py`)
Depth imbalance at multiple levels.

```python
output_keys: {"depth_imbalance", "depth_concentration", "depth_thinning"}
computed: imbalance at L1-L5, concentration score
confidence: 0.9, source="library_microstructure"
```

### Proxy/Inferred (Class 2)

All proxy features MUST return `FeatureConfidence` with `quality="proxy_inferred"` and MUST include `notes` in the output explaining the inference method.

#### AggressionProxyFeature (`src/library/features/microstructure/aggression.py`)

```python
quality_class: "proxy_inferred"
notes: "Inferred from quote/depth behavior; not true executed flow."
compute(inputs):
    quote_push = quote_state.bid_ask_drift_score
    depth_pull = depth_state.depth_withdrawal_score
    spread_penalty = spread_state.instability_score
    score = (quote_push + depth_pull) - spread_penalty
    direction = "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
```

#### AbsorptionProxyFeature (`src/library/features/microstructure/absorption.py`)

```python
quality_class: "proxy_inferred"
notes: "Inferred from depth consumption patterns; not true absorption from trade prints."
compute(inputs):
    large_orders_at_level = depth_state.large_order_size_at_levels
    price_impact = spread_state.price_impact_estimate
    absorption_score = large_orders / max(price_impact, epsilon)
```

#### BreakoutPressureProxyFeature (`src/library/features/microstructure/breakout_pressure.py`)

```python
quality_class: "proxy_inferred"
notes: "Inferred from depth withdrawal + spread expansion + momentum; not true breakout confirmation."
```

#### LiquidityStressProxyFeature (`src/library/features/microstructure/liquidity_stress.py`)

```python
quality_class: "proxy_inferred"
notes: "Short-horizon liquidity stress inferred from depth thinning; not true L2 liquidity."
```

### Deferred (Class 3: Future External-Data)

#### TickActivityFeature (`src/library/features/microstructure/tick_activity.py`)
Tick frequency and rhythm. **Requires external data source** — cTrader does not provide tick-level volume. Classified as `external`.

#### VolumeImbalanceFeature (`src/library/features/microstructure/volume_imbalance.py`)
Buy/sell volume delta. **Requires true executed trade data** — cTrader Open API does not provide buy/sell side volume. Classified as `external`.

## Data Flow: Normalized Market Input → MicrostructureContext

```
cTrader tick/depth stream (via CTraderMarketAdapter — NOT YET IMPLEMENTED)
    │
    ▼
FeatureEvaluator evaluates microstructure features
    │
    ├─► SpreadStateFeature ───┐
    ├─► TopOfBookPressureFeature ──┤
    ├─► MultiLevelDepthFeature ───┤
    ├─► AggressionProxyFeature ────┤
    ├─► AbsorptionProxyFeature ───┤
    ├─► BreakoutPressureProxy ────┤
    └─► LiquidityStressProxy ─────┘
    │
    ▼
MicrostructureContext aggregation
    │
    ├─► IntentEmitter (as confirmation input)
    ├─► SentinelBridge (optional feedback — NOT wired)
    └─► JournalBridge (journaled for debugging)
```

## MicrostructureContext (`src/library/features/microstructure/context.py`)

Aggregated higher-level context consumed by both bots and sentinel.

```python
class MicrostructureContext:
    symbol: str
    timestamp: datetime
    spread_regime: str           # "compressed" / "normal" / "expanded" / "volatile"
    liquidity_regime: str        # "normal" / "stress" / "abundant"
    depth_pressure: str          # "bullish" / "bearish" / "neutral"
    aggression_proxy: str        # "bullish" / "bearish" / "neutral"
    breakout_pressure: str       # "bullish" / "bearish" / "neutral"
    quality_flags: List[str]     # all quality labels from constituent features
    source_components: List[str] # which modules contributed
    thinning_score: Optional[float]
    absorption_score: Optional[float]
    concentration: Optional[float]
    is_transition: Optional[bool]  # from BOCPD changepoint signal
```

## Relationship to Sentinel / Market Intelligence

The proxy microstructure layer is **orthogonal** to existing ML systems:

| ML System | What It Does | Relationship |
|-----------|-------------|--------------|
| HMM (10-feature, OHLCV+Ising) | Regime classification | Library does NOT wrap HMM |
| MS-GARCH | Conditional volatility | Library does NOT wrap MS-GARCH |
| BOCPD | Changepoint detection | Library consumes `is_transition` from MarketContext |
| EnsembleVoter | HMM+MS-GARCH+BOCPD weighted voting | Exists but NOT wired into live tick path |
| Ising features (magnetization, susceptibility) | From price sign sequences | Orthogonal — library uses DOM/depth data |

**Library microstructure features use DOM/depth data. Existing ML systems use price/volume data. They are orthogonal, not redundant.**

The `MicrostructureContext` extends `MarketContext` — it does NOT replace sentinel regime classification.

## Quality Semantics Summary

| Feature | Quality Class | Data Source | Semantic Claim |
|---------|-------------|-------------|----------------|
| SpreadStateFeature | native_supported | cTrader quotes | True spread dynamics |
| TopOfBookPressureFeature | native_supported | cTrader DOM | True bid/ask imbalance |
| MultiLevelDepthFeature | native_supported | cTrader DOM | True depth imbalance |
| AggressionProxyFeature | proxy_inferred | Domain logic from quotes/depth | Inferred aggression signal |
| AbsorptionProxyFeature | proxy_inferred | Domain logic from depth | Inferred absorption signal |
| BreakoutPressureProxyFeature | proxy_inferred | Domain logic from depth/spread | Inferred breakout signal |
| LiquidityStressProxyFeature | proxy_inferred | Domain logic from depth | Inferred liquidity signal |
| TickActivityFeature | external | External source needed | N/A — V1 deferred |
| VolumeImbalanceFeature | external | True trade data needed | N/A — V1 deferred |

## NOT Duplicating Existing Systems

The proxy microstructure layer does NOT:
- Wrap HMM regime classification
- Wrap BOCPD changepoint detection
- Wrap Ising magnetization/susceptibility features
- Absorb the sentinel's regime classification logic
- Replace the EnsembleVoter (which exists but is not wired to live tick path)

It provides **short-horizon, broker-native market-structure signals** that complement, not replace, the existing ML-driven market intelligence layer.
