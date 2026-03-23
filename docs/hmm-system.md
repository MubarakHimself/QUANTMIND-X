# QUANTMINDX — HMM Regime Detection System

**Generated:** 2026-03-11

---

## Overview

The Hidden Markov Model (HMM) system provides probabilistic market regime classification as a complement to (and eventual replacement of) the Ising Model sensor. It operates in multiple deployment modes ranging from pure shadow observation to full production control of trading decisions.

---

## Architecture

```
Market Data (OHLCV)
        ↓
[HMMFeatureExtractor]  ← also consumes IsingRegimeSensor outputs
        ↓
  10-feature vector
        ↓
[Trained HMM Model]    ← loaded by HMMVersionControl
        ↓
  Regime Classification (TREND_STABLE / RANGE_STABLE / BREAKOUT_PRIME / HIGH_CHAOS)
        ↓
 [HMMRegimeSensor]     ← plugged into Sentinel
        ↓
  RegimeReport         ← used by Governor for risk mandate
```

---

## 1. Feature Extraction (`src/risk/physics/hmm/models.py`)

### `HMMFeatureExtractor`

Extracts a **10-dimensional feature vector** for each market data point.

#### Feature Groups

**Group 1 — Ising Model Outputs (4 features)**

| Feature | Source | Description |
|---------|--------|-------------|
| `magnetization` | `IsingRegimeSensor` | Weighted spin sum — measures market directional alignment |
| `susceptibility` | `IsingRegimeSensor` | Rate of change of magnetization — instability proxy |
| `energy` | derived: `-0.5 × mag²` | System energy approximation |
| `temperature` | `IsingRegimeSensor` | Thermal noise parameter (chaos proxy) |

**Group 2 — Price-based Features (4 features)**

| Feature | Calculation | Window |
|---------|-------------|--------|
| `log_returns` | `log(P_t / P_{t-1})` | 1 bar |
| `rolling_volatility_20` | `std(pct_change) × √252` | 20 bars |
| `rolling_volatility_50` | `std(pct_change) × √252` | 50 bars |
| `price_momentum_10` | `(P_t - P_{t-10}) / P_{t-10}` | 10 bars |

**Group 3 — Technical Indicators (2 features)**

| Feature | Implementation | Parameters |
|---------|---------------|------------|
| `rsi` | RSI (Relative Strength Index) | Period: 14 |
| `atr_normalized` | ATR / close price | Period: 14 |

> Note: MACD is implemented in `TechnicalIndicators` but not in the default 10-feature vector (available for extended configs).

#### Feature Configuration (`src/risk/physics/hmm/features.py`)

`FeatureConfig` dataclass controls which features are included and scaling behavior:

```python
@dataclass
class FeatureConfig:
    # Ising features (all enabled by default)
    include_magnetization: bool = True
    include_susceptibility: bool = True
    include_energy: bool = True
    include_temperature: bool = True

    # Price features (all enabled by default)
    include_log_returns: bool = True
    include_rolling_volatility_20: bool = True
    include_rolling_volatility_50: bool = True
    include_price_momentum_10: bool = True

    # Technical indicators
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Scaling (standard/minmax/robust)
    scaling_method: str = "standard"
    clip_outliers: bool = True
    clip_threshold: float = 3.0
```

#### Feature Scaling

Three scaling methods supported:

| Method | Formula | Notes |
|--------|---------|-------|
| `standard` | `(x - μ) / σ` | Default; z-score normalization |
| `minmax` | `(x - min) / (max - min)` | Bounds features to [0, 1] |
| `robust` | `(x - median) / IQR` | Outlier-resistant |

Outlier clipping applied after scaling: `clip(x, -3.0, 3.0)` by default. NaN values replaced with 0.

#### Batch Extraction

For training, `extract_features_batch()` iterates over a price DataFrame with a 50-bar warmup window (first 50 rows skipped).

---

## 2. The Ising Sensor (`src/risk/physics/ising_sensor.py`)

The HMM feature extractor calls `IsingRegimeSensor` internally to generate the 4 Ising features. The Ising sensor models price correlation as spin-lattice physics:

- **Magnetization** — weighted spin sum over the lattice; high |magnetization| = strong trend
- **Susceptibility** — ∂M/∂H; measures sensitivity to external "field" (news, volatility)
- **Energy** — `-0.5 × mag²` (approximation of Ising Hamiltonian)
- **Temperature** — set from market volatility; high temperature = disordered/chaotic market

**Key threshold** (used upstream by Governor):
- Susceptibility > 0.8 → Governor halves position sizing

---

## 3. Deployment Modes (`src/router/hmm_deployment.py`)

| Mode | ID | HMM Weight | Description |
|------|-----|-----------|-------------|
| `ISING_ONLY` | `ising_only` | 0% | Pure Ising sensor, HMM disabled (safe default) |
| `HMM_SHADOW` | `hmm_shadow` | 0% | HMM runs in parallel but does NOT affect trades; shadow log only |
| `HMM_HYBRID_20` | `hmm_hybrid_20` | 20% | Blended: 80% Ising + 20% HMM signal |
| `HMM_HYBRID_50` | `hmm_hybrid_50` | 50% | Balanced blend |
| `HMM_HYBRID_80` | `hmm_hybrid_80` | 80% | HMM-dominant blend |
| `HMM_ONLY` | `hmm_only` | 100% | Pure HMM regime (requires validated model) |

**Mode change requires an approval token** (generated via `POST /api/hmm/approval-token`) for transitions to higher HMM weights. This prevents accidental deployment of untested models.

---

## 4. Version Control (`src/router/hmm_version_control.py`)

### `HMMVersionControl`

Manages model synchronization between two VPS instances:
- **Contabo** — Training server (runs `scripts/train_hmm.py`)
- **Cloudzy** — Trading server (runs the live StrategyRouter)

#### Sync Flow

```
Contabo: Train new model → store versioned artifact
    ↓
HMMVersionControl.sync_to_cloudzy()
    ↓ (SSH/SFTP or HTTP)
    ↓ checksum verification
Cloudzy: Load new model version
    ↓
WebSocket streaming of sync progress
```

#### `SyncStatus` states

```
IDLE → IN_PROGRESS → SUCCESS / FAILED / VERSION_MISMATCH
```

#### `SyncProgress` fields
- `status` — current SyncStatus
- `progress` — float 0.0–100.0
- `message` — human-readable status
- `error` — error message if failed

---

## 5. Shadow Mode Logging (`src/router/sentinel.py`)

When HMM is in `HMM_SHADOW` mode, the Sentinel logs every prediction comparison:

```python
@dataclass
class ShadowLogEntry:
    symbol: str
    timeframe: str
    ising_regime: str      # What Ising model predicted
    hmm_regime: str        # What HMM predicted
    agreement: bool        # Do they agree?
    decision_source: str   # "ising" / "hmm" / "weighted"
    timestamp: datetime
```

Shadow logs are persisted in the `hmm_shadow_log` SQLite table and available via `GET /api/hmm/shadow-log`.

**Agreement metrics** are tracked and returned by `GET /api/hmm/status`:
- Agreement rate over last N predictions
- Per-regime breakdown of disagreements
- Used to decide if HMM is ready for promotion

---

## 6. Training Pipeline (`scripts/train_hmm.py`)

### Training Inputs

Data source: DuckDB `market_data` table (OHLCV per symbol/timeframe)

```bash
python3 scripts/train_hmm.py --n-components 4 --symbols EURUSD,GBPUSD
```

**Parameters:**
- `--n-components` — Number of hidden states (default: 4, maps to 4 regimes)
- `--symbols` — Comma-separated symbol list
- `--timeframe` — Timeframe to train on (default: H1)

### Training Steps

1. Load OHLCV data from DuckDB for specified symbols
2. `HMMFeatureExtractor.extract_features_batch()` → 2D feature matrix (skip 50-bar warmup)
3. `HMMFeatureExtractor.scale_features(fit=True)` → standardize features
4. Train `hmmlearn.GaussianHMM` with `n_components` states
5. Save model artifact (pickle) with version metadata to `src/router/hmm_version_control.py`
6. Store model metadata in `hmm_models` SQLite table

### Validation

```bash
python3 scripts/validate_hmm.py
```

Validates:
- Model loaded without error
- Prediction accuracy on held-out data
- Regime distribution (should not be degenerate)
- Agreement with Ising sensor on historical data

---

## 7. Training Scheduler (`scripts/schedule_hmm_training.py`)

Runs via systemd `hmm-training-scheduler.service` on the Contabo VPS. Triggers periodic retraining to adapt to market regime changes.

---

## 8. HMM Inference Server (`src/api/hmm_inference_server.py`)

Lightweight inference endpoint that can run standalone for regime prediction:
- Called by `RegimeFetcher` on Cloudzy if the HMM model is loaded locally
- Also powers the `GET /api/hmm/predict` endpoint for manual regime queries

---

## 9. API Endpoints (`src/api/hmm_endpoints.py`)

Base prefix: `/api/hmm`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/hmm/status` | GET | Full HMM system status (mode, weights, sync state, agreement metrics) |
| `/api/hmm/sync` | POST | Trigger model sync from Contabo to Cloudzy |
| `/api/hmm/sync/progress` | GET/WS | Stream sync progress |
| `/api/hmm/mode` | POST | Change deployment mode (requires approval token for restricted modes) |
| `/api/hmm/approval-token` | POST | Generate time-limited approval token for mode change |
| `/api/hmm/shadow-log` | GET | Retrieve shadow mode comparison log |
| `/api/hmm/models` | GET | List all versioned HMM models |
| `/api/hmm/train` | POST | Trigger training job (background task) |
| `/api/hmm/train/{job_id}` | GET | Get training job status |

### HMM Status Response Structure

```json
{
  "model_loaded": true,
  "model_version": "1.2.0",
  "deployment_mode": "hmm_shadow",
  "hmm_weight": 0.0,
  "shadow_mode_active": true,
  "contabo_version": "1.2.0",
  "cloudzy_version": "1.1.0",
  "version_mismatch": true,
  "agreement_metrics": {
    "total_predictions": 1000,
    "agreement_rate": 0.87,
    "regime_breakdown": {...}
  },
  "last_sync": "2026-03-10T14:00:00Z",
  "sync_status": "success"
}
```

---

## 10. RegimeFetcher (`src/router/engine.py`)

The `RegimeFetcher` class in the StrategyRouter polls for regime data every **5 minutes**:

**Fallback chain:**
1. **Contabo HMM API** — cached for up to 15 minutes if reachable
2. **Local HMM model** via `HMMVersionControl` — if Contabo is unreachable and a model is loaded
3. **ISING_ONLY mode** — last resort if no model available

```python
regime_fetcher = RegimeFetcher()
regime_fetcher._poll_contabo_regime()  # runs as asyncio background task
```

---

## 11. Database Schema

### `hmm_models` table (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | String UUID | UID for the model version |
| `version` | String | Semantic version (e.g. `1.2.0`) |
| `deployment_mode` | String | `shadow` / `hybrid` / `production` |
| `n_components` | Integer | Number of hidden states |
| `features` | JSON | List of feature names used |
| `trained_at` | DateTime | Training timestamp |
| `accuracy` | Float | Validation accuracy |

### `hmm_shadow_log` table (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | String | Trading symbol |
| `timeframe` | String | Timeframe |
| `ising_regime` | String | Ising model prediction |
| `hmm_regime` | String | HMM prediction |
| `agreement` | Boolean | Whether predictions agree |
| `decision_source` | String | Which model drove the decision |
| `timestamp` | DateTime | Prediction timestamp |

---

## 12. Integration with Risk System

The HMM regime output flows into the same risk pipeline as the Ising sensor:

```
HMMRegimeSensor → Sentinel.RegimeReport.regime
                               ↓
                          Governor
                               ↓
              risk_scalar (0.0 HALTED / 0.2 CLAMPED / 1.0 STANDARD)
                               ↓
                      PhysicsAwareKellyEngine
```

In `HMM_HYBRID` modes, the final regime is a weighted blend:
- `final_regime = (1 - hmm_weight) × ising_regime + hmm_weight × hmm_regime`

The blending is handled by the `HMMRegimeSensor` before emitting to `RegimeReport`.
