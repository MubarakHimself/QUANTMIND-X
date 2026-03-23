# QUANTMINDX — Risk Management & Position Sizing

**Generated:** 2026-03-11

---

## Overview

The risk system is a multi-layer stack that combines econophysics-based market sensing with Kelly criterion position sizing. It is designed for prop-firm compliance (FTMO, The5ers, FundingPips) with hard capital preservation rules.

```
Market Ticks
    ↓
[Sentinel]  ← ChaosSensor + RegimeSensor + CorrelationSensor + NewsSensor + HMM
    ↓
RegimeReport {chaos_score, regime, regime_quality, susceptibility, news_state}
    ↓
[Governor]  ← allocation_scalar calculation (STANDARD / CLAMPED / HALTED)
    ↓
RiskMandate {allocation_scalar, risk_mode, position_size, kelly_fraction}
    ↓
[Commander] ← strategy auction → filters bots by regime + risk mandate
    ↓
Trade Execution via MT5 Bridge
```

---

## Layer 1: Market Physics Sensors (`src/risk/physics/`)

### 1.1 Ising Model Sensor (`ising_sensor.py`)
Treats price correlation patterns as a magnetic spin system.

| Output | Meaning |
|--------|---------|
| `magnetization` | Directional trend strength (−1 to +1) |
| `susceptibility` | System instability — high = phase transition imminent |
| `energy` | Market order/disorder level |
| `temperature` | Simulated market temperature |

**Risk trigger:** If `ising_susceptibility > 0.8`, the Ising physics multiplier for Kelly is set to `P_χ = 0.5` (halves position size).

### 1.2 Lyapunov Exponent Sensor (`sensors/`)
Measures market predictability by tracking divergence of nearby trajectories.

**Risk trigger:** `P_λ = max(0, 1.0 − 2.0 × λ)`. A positive Lyapunov exponent reduces or eliminates the Kelly multiplier.

### 1.3 Random Matrix Theory (RMT / Eigenvalue Sensor)
Monitors correlation matrix eigenvalue spread.

**Risk trigger:** If `λ_max > 1.5`, the Eigen multiplier is `P_E = min(1.0, 1.5 / λ_max)` — large eigenvalue means correlated risk, reduces position.

### 1.4 HMM Regime Sensor (`src/risk/physics/hmm/`)
Hidden Markov Model with Ising outputs as features (10-feature vector):

**Features:** Ising magnetization, susceptibility, energy, temperature + price returns, volatility, momentum, RSI, ATR, MACD

**Regimes detected:** TREND_STABLE, RANGE_STABLE, BREAKOUT_PRIME, HIGH_CHAOS (+ sub-variants)

**Deployment modes (HMMDeploymentMode):**
- `ISING_ONLY` — HMM not used, Ising sensor only
- `SHADOW` — HMM runs in parallel, predictions logged but not used for trading
- `HYBRID` — Weighted combination of Ising and HMM
- `PRODUCTION` — HMM-only predictions

**Fallback chain:** Contabo API cache → local HMM model → ISING_ONLY

---

## Layer 2: The Sentinel (`src/router/sentinel.py`)

Aggregates all sensor outputs into a single `RegimeReport`.

```python
@dataclass
class RegimeReport:
    regime: str           # TREND_STABLE | RANGE_STABLE | BREAKOUT_PRIME | HIGH_CHAOS
    chaos_score: float    # 0.0 – 1.0
    regime_quality: float # 1.0 – chaos_score
    susceptibility: float # 0.0 – 1.0
    is_systemic_risk: bool
    news_state: str       # SAFE | KILL_ZONE
    timestamp: float
    hmm_regime: str       # HMM prediction (if active)
    hmm_confidence: float
    hmm_agreement: bool   # Ising and HMM agree?
    decision_source: str  # 'ising' | 'hmm' | 'weighted'
```

**Shadow logging:** In shadow mode, every regime prediction from Ising and HMM is logged to the database (`hmm_shadow_log` table) for agreement analysis and model calibration.

---

## Layer 3: The Governor (`src/router/governor.py`)

Calculates the `RiskMandate` — the compliance ticket that controls how much capital is at risk.

### Risk Scalar Logic

| Chaos Score | Mode | Scalar |
|-------------|------|--------|
| > 0.6 | `CLAMPED` | 0.2 × mode_multiplier |
| 0.3 – 0.6 | `CLAMPED` | 0.7 × mode_multiplier |
| < 0.3 | `STANDARD` | 1.0 × mode_multiplier |
| News = KILL_ZONE | `HALTED` | 0.0 |
| Susceptibility ≥ 1.0 | `HALTED` | 0.0 |

**Demo/Live mode distinction:**
- Live mode: base risk limits
- Demo mode: `demoRiskMultiplier` (default 1.5×) — more aggressive

### EnhancedGovernor (`src/router/enhanced_governor.py`)
Extends Governor with **fee-aware position sizing**. Uses `BrokerRegistryManager` to get broker-specific pip values and commission structures for precise lot size calculations.

```python
@dataclass
class RiskMandate:
    allocation_scalar: float  # 0.0 – 1.0
    risk_mode: str            # STANDARD | CLAMPED | HALTED
    notes: str
    position_size: float      # Calculated lots
    kelly_fraction: float     # Kelly fraction used
    risk_amount: float        # Dollar amount at risk
    kelly_adjustments: list   # Audit trail
    mode: str                 # 'demo' | 'live'
```

---

## Layer 4: Physics-Aware Kelly Engine (`src/risk/sizing/kelly_engine.py`)

**Class:** `PhysicsAwareKellyEngine`

The position sizing formula:

```
f*          = (p × b − q) / b           # Full Kelly
f_base      = f* × kelly_fraction       # Half-Kelly (default 0.5)
P_λ         = max(0, 1 − 2 × λ)        # Lyapunov multiplier
P_χ         = 0.5 if χ > 0.8 else 1.0  # Ising multiplier
P_E         = min(1, 1.5 / λ_max)       # Eigenvalue multiplier (if λ_max > 1.5)
M_physics   = min(P_λ, P_χ, P_E)        # Weakest-link aggregation
final_risk% = f_base × M_physics        # Max 10% hard cap
```

**Parameters:**
- `p` — win rate
- `b` — payoff ratio (avg_win / avg_loss)
- `q` — loss rate (1 − p)
- `kelly_fraction` — 0.5 (configurable)

**Output:** `SizingRecommendation` with `final_risk_pct`, `raw_kelly`, `physics_multiplier`, `constraint_source`, `validation_passed`

**Monte Carlo Validation:** If `final_risk_pct > 0.005` (0.5%), runs 1000-iteration MC simulation to check probability of ruin. Applies further adjustment if MC fails.

**Hard cap:** `final_risk_pct` is capped at 10% regardless of Kelly calculation.

### Lot Size Calculation (caller responsibility)
```
position_size_lots = (account_balance × final_risk_pct) / (stop_loss_pips × pip_value)
```

---

## Layer 5: Prop Firm Overlay (`src/risk/prop_firm_overlay.py`)

Applies prop firm-specific constraints on top of Kelly sizing:
- Maximum daily drawdown limits (FTMO: 5%, The5ers: 4%)
- Maximum total drawdown limits
- Lot size scaling to pass evaluation phase

---

## Layer 6: Kill Switches (`src/router/kill_switch.py`, `progressive_kill_switch.py`)

### SmartKillSwitch
- `soft_kill`: Halts new trades, lets existing close naturally
- `hard_kill`: Closes all positions immediately
- Regime-aware: activates based on `RegimeReport` chaos score + news state

### ProgressiveKillSwitch
Tiered loss-based shutdown:
- **Soft**: Reduces position sizing
- **Medium**: Halts new positions
- **Hard**: Full stop, all positions closed

---

## Layer 7: Bot Circuit Breaker (`src/router/bot_circuit_breaker.py`)

Per-bot circuit breaker stored in `bot_circuit_breaker` SQLite table:
- Tracks consecutive losses per bot
- Auto-quarantines bots on threshold breach
- Daily trade count limits
- Mode-aware (demo vs live limits differ)

---

## Layer 8: Portfolio Kelly (`src/risk/portfolio_kelly.py`)

Cross-bot correlation-aware Kelly calculation at portfolio level. Prevents over-concentration in correlated strategies.

---

## Key Config Parameters (from `src/risk/config.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `kelly_fraction` | 0.5 | Half-Kelly safety factor |
| `max_portfolio_risk` | 0.20 | 20% portfolio hard cap |
| `correlation_threshold` | 0.80 | Correlation correlation warning |
| `monte_carlo_threshold` | 0.005 | Trigger MC validation above 0.5% |
| `max_risk_cap` | 0.10 | 10% per-trade hard cap |

---

## Data Models for Risk

| Table | Purpose |
|-------|---------|
| `risk_tier_transitions` | Audit log of risk tier changes (growth/scaling/guardian) |
| `bot_circuit_breaker` | Per-bot quarantine state |
| `daily_fee_tracking` | Daily fee burn monitoring |
| `trade_proposals` | Proposed trades pending approval |

---

## Related Files

| File | Purpose |
|------|---------|
| `src/risk/physics/ising_sensor.py` | Ising spin-glass sensor |
| `src/risk/physics/hmm_sensor.py` | HMM regime sensor |
| `src/risk/physics/hmm/` | HMM feature extraction + training |
| `src/risk/sizing/kelly_engine.py` | Physics-Aware Kelly |
| `src/risk/sizing/monte_carlo_validator.py` | MC risk validation |
| `src/risk/governor.py` | Base risk governor |
| `src/router/enhanced_governor.py` | Fee-aware governor |
| `src/risk/prop_firm_overlay.py` | Prop firm constraints |
| `src/router/kill_switch.py` | Kill switches |
| `src/router/progressive_kill_switch.py` | Tiered kill switch |
| `src/router/bot_circuit_breaker.py` | Per-bot circuit breaker |
| `src/risk/portfolio_kelly.py` | Portfolio-level Kelly |
| `docs/api/enhanced_kelly_api.md` | Kelly API docs |
