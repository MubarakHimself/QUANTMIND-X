# Enhanced Kelly Position Sizing Ecosystem (Physics-Aware)

**Version:** 2.0 (Consolidated)  
**Date:** 2026-01-27  
**Target System:** `src/risk/position_sizing/`  
**Integration Goal:** Replace standard risk models with an "Econophysics-Governed" approach.

---

## 1. Executive Summary & Philosophy

This document defines the complete technical specification for the **Physics-Aware Kelly Criterion**. Unlike standard risk models that rely on lagging indicators (recent P&L), this system uses **leading physical indicators** of market structure to regulate position size.

**The Core Axiom:** 
> "Standard Kelly maximizes growth in **stable** systems ($P_{chaos} \approx 0$). In **critical** ($P_{critical} > 0$) or **chaotic** ($P_{chaos} > 0.5$) systems, standard Kelly leads to ruin. We must dampen 'f' (fraction) proportional to the entropy of the market."

---

## 2. The Equation

The master formula for Position Size ($S$) is:

$$ S = B \times f_{base} \times M_{physics} $$

Where:
*   $S$: Capital to risk (in $) or Position Size (in Lots/Units)
*   $B$: Account Balance (or Equity, depending on setting)
*   $f_{base}$: The base Fractional Kelly
*   $M_{physics}$: The composite Econophysics Multiplier ($0.0 \to 1.0$)

### 2.1 Base Kelly Component ($f_{base}$)

$$ f_{base} = \left( \frac{p \cdot b - (1-p)}{b} \right) \times K_{fraction} $$

*   $p$: **Win Probability** (Rolling 50-trade win rate, decaying weight)
*   $b$: **Payoff Ratio** (Average Win / Average Loss)
*   $K_{fraction}$: **Safety Scalar** (Default: 0.5 "Half-Kelly")

### 2.2 The Physics Multiplier ($M_{physics}$)

This is the innovation. It aggregates three distinct physical states of the market:

$$ M_{physics} = \min(P_{\lambda}, P_{\chi}, P_{E}) $$

*   $P_{\lambda}$ (Lyapunov Penalty): Predictability horizons
*   $P_{\chi}$ (Ising Penalty): Phase transition proximity
*   $P_{E}$ (Eigen Penalty): Systemic correlation risk

---

## 3. Component Specifications

### Module A: The Chaos Governor (Lyapunov)
**Source:** `quant-traderr-lab/Lyapunov Exponent/Lyapunov Pipeline.py`

*   **Concept:** Measures the rate at which similar price trajectories diverge.
*   **Input:** Log-returns of the asset (N=300 bars).
*   **Metric:** Max Lyapunov Exponent ($\lambda$).
*   **Logic:**
    *   $\lambda \le 0$: System is stable/periodic. **Penalty = 1.0 (No penalty)**.
    *   $\lambda > 0$: System is chaotic. Information decays exponentially.
    *   **Formula:** $P_{\lambda} = \max(0, 1.0 - (2.0 \times \lambda))$
    *   **Effect:** At $\lambda=0.5$ (High Chaos), size becomes 0.

### Module B: The Regime Sensor (Ising Model)
**Source:** `quant-traderr-lab/Ising Model/Ising Pipeline.py`

*   **Concept:** Models market participants as atomic spins ($+1/-1$). High magnetic susceptibility ($\chi$) indicates a "Critical Point" where a unified move (crash/rally) is imminent.
*   **Input:** Volatility stream mapped to "Temperature".
*   **Metric:** Magnetic Susceptibility ($\chi$).
*   **Logic:**
    *   $\chi < \chi_{crit}$ (e.g., 0.8): Normal noise. **Penalty = 1.0**.
    *   $\chi \ge \chi_{crit}$: Phase transition imminent. **Penalty = 0.5 (Halve Risk)**.
    *   **Effect:** Pre-emptive risk cutting *before* volatility spikes.

### Module C: The Correlation Guard (RMT)
**Source:** `quant-traderr-lab/RMT_Correlation_Filter/RMT_Pipeline.py`

*   **Concept:** Random Matrix Theory separates signal from noise in correlation matrices. The largest eigenvalue ($\lambda_{max}$) represents the "Market Mode."
*   **Input:** Portfolio-wide returns matrix.
*   **Metric:** $\lambda_{max}$ (Max Eigenvalue).
*   **Logic:**
    *   Normal Market: $\lambda_{max} \approx$ Low value ($< ~1.5$). **Penalty = 1.0**.
    *   Panic/Euphoria: $\lambda_{max}$ spikes ($> 2.0$). Everything moves together.
    *   **Formula:** $P_{E} = \min(1.0, \frac{1}{\lambda_{max} - \text{threshold}})$
    *   **Effect:** Prevents sizing up when "diversification" is mathematically impossible.

---

## 4. Implementation Architecture

### 4.1 Data Structures (Pydantic Models)

```python
from pydantic import BaseModel
from typing import Optional

class StrategyPerformance(BaseModel):
    win_rate: float        # e.g., 0.55
    avg_win: float         # e.g., 150.0
    avg_loss: float        # e.g., 100.0
    k_fraction: float = 0.5  # Half-Kelly

class MarketPhysics(BaseModel):
    lyapunov_exponent: float      # Chaos metric
    ising_susceptibility: float   # Phase transition metric
    rmt_max_eigenvalue: float     # Correlation metric
    
class SizingRecommendation(BaseModel):
    raw_kelly: float         # Pure Kelly %
    physics_multiplier: float # The aggregate penalty (0.0 - 1.0)
    final_risk_pct: float    # The executable risk %
    constraint_source: str   # Which physical law limited the size?
```

### 4.2 The Engine Class (`src/risk/physics_kelly.py`)

```python
class PhysicsAwareKellyEngine:
    def __init__(self, use_monte_carlo_validation: bool = True):
        self.validate = use_monte_carlo_validation

    def calculate_size(self, perf: StrategyPerformance, physics: MarketPhysics) -> SizingRecommendation:
        # 1. Base Calculations
        if perf.avg_loss == 0: return self._zero_risk("No Loss History")
        
        b = perf.avg_win / perf.avg_loss
        p = perf.win_rate
        q = 1 - p
        
        if b == 0: return self._zero_risk("Zero Payoff")
        
        # Kelly: f = (bp - q) / b
        f_raw = ((b * p) - q) / b
        f_safe = max(0.0, f_raw * perf.k_fraction)
        
        # 2. Physics Constraints
        
        # A. Chaos
        p_lambda = 1.0
        if physics.lyapunov_exponent > 0:
            # Linear decay: chaos 0.0 -> 1.0, chaos 0.5 -> 0.0
            p_lambda = max(0.0, 1.0 - (2.0 * physics.lyapunov_exponent))
            
        # B. Criticality (Binary cut)
        p_chi = 0.5 if physics.ising_susceptibility > 0.8 else 1.0
        
        # C. Correlation (Inverse scaling)
        p_eigen = 1.0
        if physics.rmt_max_eigenvalue > 1.5:
             # Example: Eigen 2.0 -> 0.75, Eigen 3.0 -> 0.5
            p_eigen = 1.5 / physics.rmt_max_eigenvalue
            
        # 3. Aggregation (The "Weakest Link" Principle)
        # We take the lowest scalar to be safe
        m_physics = min(p_lambda, p_chi, p_eigen)
        
        # Identify constraint source for logging
        constraint = "None"
        if m_physics == p_lambda: constraint = "Lyapunov Chaos"
        elif m_physics == p_chi: constraint = "Ising Criticality"
        elif m_physics == p_eigen: constraint = "RMT Correlation"
        
        final_risk = f_safe * m_physics
        
        # 4. Monte Carlo Validation (Optional)
        if self.validate and final_risk > 0.005: # Only validate meaningful sizes
             if not self._mc_validate(perf, final_risk):
                 final_risk *= 0.5 # Penalty for failing stress test
                 constraint += " + MC Failure"

        return SizingRecommendation(
            raw_kelly=f_raw,
            physics_multiplier=m_physics,
            final_risk_pct=round(final_risk, 4),
            constraint_source=constraint
        )

    def _mc_validate(self, perf, risk, runs=2000) -> bool:
        """
        Simulates 2000 future equity curves. 
        Returns False if Risk of Ruin (>30% DD) exceeds 1%.
        """
        # (Implementation details in Monte Carlo TRD)
        return True
```

---

## 5. Monte Carlo Validation Layer
**Source:** `quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py`

This is the final "Gatekeeper". Even if Kelly and Physics say "Bet 5%", the Monte Carlo engine runs the numbers:
1.  **Generate** 2,000 synthetic trade sequences based on user's Win Rate & Payoff distributions.
2.  **Apply** the proposed sizing (5%).
3.  **Count** how many sequences hit the "Max Drawdown" limit (e.g., 10%) defined by the Prop Firm.
4.  **Verdict:** If Probability of Ruin > 0.5%, the size is **rejected** and halved.

---

## 6. Integration Workflow

1.  **Market Data Ingestion:**
    *   Price feeds $\to$ [Econophysics Modules] $\to$ `MarketPhysics` Object (Daily).
2.  **Trade Journal Sync:**
    *   MT5 History $\to$ [Journal DB] $\to$ `StrategyPerformance` Object (Weekly/Per-Trade).
3.  **Risk Calculation:**
    *   Before *every* trade entry: Call `PhysicsAwareKellyEngine.calculate_size()`.
4.  **Execution:**
    *   Engine returns `0.012` (1.2% Risk).
    *   `RiskGovernor` (Phase 2 TRD) converts 1.2% $\to$ Lots (based on SL distance).
    *   Order sent to MT5.

---

## 7. Edge Case Handling

| Scenario | Physics State | Resulting Action |
|----------|---------------|------------------|
| **Flash Crash** | `Susceptibility` spikes | Risk halved *immediately*. |
| **New Strategy** | No history ($N < 50$) | `StrategyPerformance` returns default safe stats ($p=0.4, b=1.0$). Size is minimal. |
| **Data Outage** | Physics metrics stale | `M_physics` defaults to 0.5 (Safety Mode). |
| **HFT Noise** | `Lyapunov` implies random walk | Risk approaches 0%. System waits for signal. |

---

## 8. Development Checklist

- [ ] **Step 1:** Implement `src/risk/physics_kelly.py` (The Engine).
- [ ] **Step 2:** Integrate `src/risk/lyapunov.py` (The Sensor).
- [ ] **Step 3:** Integrate `src/risk/monte_carlo.py` (The Validator).
- [ ] **Step 4:** Unit Tests with `pytest` covering the physics penalty logic.
- [ ] **Step 5:** Backtest: Compare "Physics Kelly" vs "Standard Kelly" on 2020-2024 data (COVID crash handling).
