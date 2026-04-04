# QuantMindX — Multi-Model Regime Detection Roadmap

**Date:** 2026-03-26
**Status:** Planning
**Author:** Mubarak + Claude (collaborative design session)

---

## 1. The Vision

Three regime detection models working **together** — not independently — as a layered ensemble:

```
                    ┌─────────────────────────────┐
                    │   REGIME DECISION ENGINE     │
                    │  (Weighted Ensemble Voter)   │
                    └─────────┬───────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────┴─────┐  ┌─────┴─────┐  ┌──────┴──────┐
        │    HMM     │  │ MS-GARCH  │  │    BOCPD    │
        │  (States)  │  │  (Vol)    │  │ (Changepts) │
        └─────┬─────┘  └─────┬─────┘  └──────┬──────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   10-Feature Vec   │
                    │ (shared pipeline)  │
                    └───────────────────┘
```

Each model sees the **same feature stream** but answers a **different question**:

| Model    | Question It Answers                    | Output                      | Latency  |
|----------|----------------------------------------|-----------------------------|----------|
| HMM      | "What regime are we in?"               | 1 of 8 RegimeType labels    | <1ms     |
| MS-GARCH | "What volatility regime + forecast?"   | Vol regime + σ forecast     | <1ms     |
| BOCPD    | "Did the regime just change?"          | Changepoint probability     | <1ms     |

The ensemble voter combines them:
- **HMM** provides the base regime classification (TREND_BULL, RANGE_STABLE, etc.)
- **MS-GARCH** confirms/refines with volatility context (is this a low-vol trend or high-vol trend?)
- **BOCPD** signals transitions — when it fires, the router re-evaluates immediately instead of waiting for the next HMM cycle

---

## 2. Tiered Model Allocation

Premium sessions get the full ensemble. Non-premium get lighter compute.

```
PREMIUM SESSIONS (TOKYO_LDN, LONDON_OPEN, LDN_NY_OVERLAP):
  → Full ensemble: HMM + MS-GARCH + BOCPD
  → Session-specific trained models
  → Highest confidence thresholds
  → All 4 bot types active (MR, MOM, ORB, TC)

NON-PREMIUM SESSIONS (SYDNEY, TOKYO, NY_CLOSE, etc.):
  → HMM only (lightest compute)
  → Universal models (not session-specific)
  → Reduced position sizing
  → Limited bot types per SESSION_BOT_MIX
```

This maps directly to your existing `SessionKellyModifiers` — premium sessions already get
House Money Mode at +8%, so giving them better models is the natural extension.

---

## 3. Implementation Phases

### PHASE 1: MS-GARCH (Week 1-2)
**Why first:** Smallest code change, biggest impact. Uses the same feature pipeline, produces
a volatility regime + forecast that directly improves position sizing.

**What it does:**
- Regime-switching GARCH(1,1) with 2-3 volatility states (low/medium/high)
- Each state has its own GARCH parameters (α, β, ω)
- Produces: current vol state, 1-step-ahead σ forecast, state transition matrix
- Model size: ~1KB per symbol (comparable to HMM)

**Where it fits:**
```
src/risk/physics/
├── hmm/            ← exists
├── msgarch/        ← NEW
│   ├── __init__.py
│   ├── trainer.py      # MS-GARCH training (arch library)
│   ├── models.py       # MSGARCHSensor(BaseRegimeSensor)
│   └── utils.py
└── sensors/
    └── base.py     ← already has BaseRegimeSensor
```

**Integration points:**
- `MSGARCHSensor.predict_regime()` returns vol state + σ forecast
- σ forecast feeds into `SessionKellyModifiers` for dynamic position sizing
- Vol state cross-references with HMM regime for confidence boost

**Training:**
- Uses same M5 data we already have (2020-2024)
- `arch` library (pure Python, no GPU needed)
- Walk-forward: train on 3 years, validate on 1 year, test on 1 year
- Per-symbol + per-session training (same pattern as HMM)

**Dependencies:** `pip install arch` (well-maintained, ~500KB)

---

### PHASE 2: BOCPD (Week 2-3)
**Why second:** This is the one you're most excited about — and for good reason. BOCPD is
fundamentally different from HMM/GARCH because it's **online** and **doesn't need traditional
training**. It adapts in real-time.

**What it does:**
- Maintains a probability distribution over "run lengths" (time since last changepoint)
- When a new data point arrives, it updates the run-length posterior
- If P(changepoint) spikes above threshold → regime shift detected
- No offline training needed — it learns the hazard function parameters from data

**Where it fits:**
```
src/risk/physics/
├── hmm/
├── msgarch/
├── bocpd/          ← NEW
│   ├── __init__.py
│   ├── detector.py     # BOCPDDetector(BaseRegimeSensor)
│   ├── hazard.py       # Hazard functions (constant, logistic)
│   └── observation.py  # Observation models (Gaussian, Student-t)
└── sensors/
```

**Key insight for routing:** BOCPD doesn't tell you WHAT regime you're in — it tells you
WHEN the regime changed. That's why it's the perfect complement to HMM:

```
Normal operation:
  HMM says "TREND_BULL" → Commander routes to momentum/trend_follow bots

BOCPD fires (changepoint detected):
  → Commander IMMEDIATELY requests fresh HMM + MS-GARCH predictions
  → If HMM now says "RANGE_VOLATILE" → Commander re-routes to scalp/short_term
  → Faster reaction than waiting for next scheduled HMM cycle
```

**Training (or rather, calibration):**
- Calibrate hazard rate λ from historical data (how often do regimes actually change?)
- Calibrate observation model parameters from feature statistics
- Per-symbol calibration (XAUUSD changes more often than EURUSD)
- Can be done in seconds — no heavy compute needed

**Dependencies:** None — pure NumPy implementation (~200 lines)

---

### PHASE 3: Ensemble Voter (Week 3-4)
**Why third:** Can't build the voter until the components exist.

**What it does:**
- Takes outputs from all three models
- Weighted voting based on model confidence and historical accuracy
- Produces a single `RegimeType` + confidence score for the Commander

**Weighting strategy:**
```python
# Premium sessions (full ensemble)
weights = {
    "hmm": 0.45,      # Base regime classification
    "msgarch": 0.30,   # Volatility context
    "bocpd": 0.25,     # Transition detection (binary boost)
}

# Non-premium sessions (HMM only)
weights = {
    "hmm": 1.0,
    "msgarch": 0.0,
    "bocpd": 0.0,
}
```

When BOCPD fires a changepoint, the voter temporarily increases HMM weight to 0.60
and requests an immediate re-prediction (instead of waiting for the next cycle).

**Where it fits:**
```
src/risk/physics/
├── ensemble/       ← NEW
│   ├── __init__.py
│   ├── voter.py        # EnsembleVoter — the brain
│   ├── weights.py      # Adaptive weight management
│   └── metrics.py      # Ensemble accuracy tracking
```

---

### PHASE 4: Walk-Forward Validation + Purged CV (Week 4-5)
Upgrade the entire training pipeline with proper validation:

**Walk-Forward:**
```
Year 1-3: Train    → Year 4: Validate → Year 5: Test
Year 2-4: Train    → Year 5: Validate → Year 6: Test (with new data)
Year 3-5: Train    → Year 6: Validate → Year 7: Test
```
Each window slides forward. Model must pass ALL windows, not just one.

**Purged Cross-Validation:**
- 5-fold CV with embargo zones between train/test
- Embargo = 5 * average regime duration (prevents leakage)
- Combinatorial purged CV (de Prado, 2018) for small datasets

This applies to HMM AND MS-GARCH training. BOCPD uses online calibration instead.

---

### PHASE 5: More Data + More Pairs (Week 5-6)

**Extended data (back to 2010, possibly 2000):**
- histdata.com has data back to ~2000 for major pairs
- 20+ years of M1 data → aggregate to M5
- More data = more walk-forward windows = more robust models
- XAUUSD: combine Dukascopy + yfinance for maximum coverage

**Additional pairs (liquid, suited to scalper strengths):**

| Pair    | Why                                          | Best Scalper Types    |
|---------|----------------------------------------------|-----------------------|
| EURUSD  | ✅ Already have — highest liquidity          | MR, MOM, ORB, TC     |
| GBPUSD  | ✅ Already have — high vol, good for MOM     | MOM, ORB, TC          |
| USDJPY  | ✅ Already have — tight spreads              | MR, scalp             |
| AUDUSD  | ✅ Already have — good range behavior        | MR, range_trade       |
| XAUUSD  | ✅ Already have — high vol, momentum         | MOM, breakout         |
| USDCHF  | 🆕 Low spread, strong mean-reversion        | MR, range_trade       |
| EURJPY  | 🆕 Good vol, strong session patterns        | MOM, ORB              |
| GBPJPY  | 🆕 "Beast" — highest vol major cross        | MOM, breakout, TC     |
| EURGBP  | 🆕 Tight range — ideal MR pair             | MR, scalp             |
| NZDUSD  | 🆕 Already in secondary config — promote   | MR, range_trade       |

Selection criteria: liquidity (daily volume > $10B), spread < 2 pips during premium
sessions, clear session personality (trending vs ranging behavior).

---

### PHASE 6: Knowledge Distillation (Week 7-8)

**What you asked about — using "AI to distill":**

Knowledge distillation in ML means training a small fast model (student) to mimic a
large accurate model (teacher). For our case:

```
Teacher: Full ensemble (HMM + MS-GARCH + BOCPD) running on all features
Student: Tiny 2-layer neural net running on 4 features

Teacher produces: regime labels with high accuracy
Student learns: to approximate those labels with fraction of compute
```

**Why this matters for you:**
- Teacher runs during premium sessions (full compute budget)
- Student runs during non-premium sessions (minimal compute)
- Student can run on the trading server without the full model stack
- If the server loses the full models, student is the fallback

**Your Dell Precision 5550 (GPU training):**
- Quadro T2000 GPU — enough for the student network training
- Teacher doesn't need GPU (HMM + GARCH + BOCPD are all CPU-efficient)
- Student training: ~30 min on GPU vs ~4 hours on CPU

---

## 4. File Structure After All Phases

```
src/risk/physics/
├── hmm/                    ← EXISTS (Phase 0 — done)
│   ├── trainer.py          # HMM training with adaptive grid search
│   ├── models.py           # HMMFeatureExtractor
│   ├── features.py
│   ├── indicators.py
│   ├── scaler.py
│   └── utils.py
├── msgarch/                ← NEW (Phase 1)
│   ├── trainer.py          # MS-GARCH training
│   ├── models.py           # MSGARCHSensor(BaseRegimeSensor)
│   └── utils.py
├── bocpd/                  ← NEW (Phase 2)
│   ├── detector.py         # BOCPDDetector(BaseRegimeSensor)
│   ├── hazard.py
│   └── observation.py
├── ensemble/               ← NEW (Phase 3)
│   ├── voter.py            # EnsembleVoter
│   ├── weights.py
│   └── metrics.py
├── distillation/           ← NEW (Phase 6)
│   ├── teacher.py
│   ├── student.py
│   └── trainer.py
├── sensors/                ← EXISTS
│   ├── base.py             # BaseRegimeSensor ABC
│   ├── hmm.py
│   └── config.py
├── chaos_sensor.py
├── correlation_sensor.py
├── ising_sensor.py
└── hmm_sensor.py           ← backward compat wrapper
```

---

## 5. Inference Server Changes

The existing `hmm_inference_server.py` on port 8001 needs to be extended:

```
Current:  GET /api/hmm/regime/{symbol}  → HMM prediction only
After:    GET /api/regime/{symbol}      → Ensemble prediction
          GET /api/regime/{symbol}/hmm  → HMM only (fallback)
          GET /api/regime/{symbol}/vol  → MS-GARCH vol forecast
          GET /api/changepoint/{symbol} → BOCPD status
```

Model loading becomes:
```
/data/models/
├── hmm/
│   ├── hmm_per_symbol_EURUSD_v1.pkl
│   └── hmm_session_LONDON_OPEN_EURUSD_v1.pkl
├── msgarch/
│   ├── msgarch_EURUSD_v1.pkl
│   └── msgarch_session_LONDON_OPEN_EURUSD_v1.pkl
├── bocpd/
│   └── bocpd_EURUSD_calibration.json   (tiny — just parameters)
└── ensemble/
    └── ensemble_weights_v1.json
```

---

## 6. What Stays the Same

These parts of the system **don't need to change**:

- `RegimeType` enum (8 regimes) — all models map to the same 8 labels
- `REGIME_STRATEGY_MAP` — routing logic stays identical
- `Commander` — still receives a RegimeType + confidence, doesn't care which model produced it
- `SessionKellyModifiers` — still uses the same thresholds
- `SESSION_BOT_MIX` — still controls which bots trade which sessions
- `HMMDeploymentManager` state machine — extends to "ensemble" mode
- `HMMVersionControl` SSH sync — extends to sync all model types

---

## 7. Immediate Next Steps

1. **Implement MS-GARCH trainer** (extends the pattern from HMM trainer)
2. **Implement BOCPD detector** (pure NumPy, no training needed)
3. **Build ensemble voter** (combines all three)
4. **Add walk-forward validation** to training pipeline
5. **Download extended data** (back to 2010+)
6. **Add new pairs** (USDCHF, EURJPY, GBPJPY, EURGBP, NZDUSD)
7. **Update inference server** with ensemble endpoints
8. **Knowledge distillation** (once ensemble is stable)

---

## 8. Dependencies

```
# Already installed
hmmlearn          # HMM
numpy, pandas     # Core
scikit-learn      # Scaling, metrics

# New — Phase 1
arch              # MS-GARCH (pip install arch)

# New — Phase 6 (optional, only for distillation)
torch             # Student network training (uses your GPU)
```

BOCPD needs **no new dependencies** — pure NumPy.

---

## 9. Risk Mitigation

| Risk                              | Mitigation                                       |
|-----------------------------------|--------------------------------------------------|
| Ensemble disagrees (models fight) | Voter uses weighted majority; Commander falls back to HMM-only |
| BOCPD fires too often (noise)     | Calibrate hazard rate conservatively; require 2 consecutive signals |
| MS-GARCH overfits vol regimes     | Walk-forward validation; BIC model selection (same as HMM) |
| Server can't load all models      | Graceful degradation: ensemble → HMM-only → Ising-only |
| Extended data quality issues      | Validate data integrity per-year; reject years with >5% missing bars |
| New pairs behave differently      | Each pair trained independently; anti-overfit checks per pair |
