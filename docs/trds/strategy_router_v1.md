# Technical Requirements Document: QuantMindX Strategy Router

**System Version:** 1.0 (Final Architecture)
**Target Module:** `src/router/`
**Primary Objective:** Autonomous Trading Intelligence & Risk Governance

---

## 1. System Overview

The **Strategy Router** is the "Sentient Brain" that sits above the "Trading Limbs" (MT5). It implements a **Tri-Layer Control Hierarchy** to ensure that market physics (Intelligence) and account safety (Compliance) always override strategy impulses.

### 1.1 The "Sentient Loop"
The Router operates on a rigid cycle (Ticker Loop):
1.  **Observe (Sentinel):** "The market is Chaotic (0.7) and Correlated (0.9)."
2.  **Govern (Governor):** "Risk Budget is clamped to 20% due to Chaos."
3.  **Command (Commander):** "Deploy 'Bunker Strategy' (Hedge) within the 20% budget."
4.  **Execute (Interface):** Dispatch JSON command to MT5.

---

## 2. Component 1: The Sentinel (Intelligence Layer)

**Module:** `src/router/sentinel.py`
**Responsibility:** Market Diagnostics & Regime Classification.
**Dependencies:** `src/router/sensors/`

The Sentinel uses a "Sensor Array" to translate raw ticks into actionable Physics Metrics.

### 2.1 The Sensor Array (Detailed Specs)

#### A. ChaosSensor (The Turbulence Meter)
*   **File:** `src/router/sensors/chaos.py`
*   **Algorithm:** **Lyapunov Exponent** ($\lambda$).
*   **Input:** Rolling 300-tick window ($x_t$).
*   **Logic:** Calculates the rate of divergence for nearest neighbors in Phase Space.
    *   $\lambda < 0.1$: Laminar Flow (Stable Trend).
    *   $\lambda > 0.5$: Turbulent Flow (Chaos/Chop).
*   **Output:** `ChaosScore` (0.0 to 1.0).

#### B. RegimeSensor (The Phase Meter)
*   **File:** `src/router/sensors/regime.py`
*   **Algorithm:** **Ising Model** (Magnetism).
*   **Input:** Volatility Clustering & Order Flow.
*   **Logic:** Simulates "Spin Dynamics" via Metropolis-Hastings.
    *   High Susceptibility ($\chi$): Critical Point (Breakout Imminent).
    *   High Energy ($E$): Frustrated Market (Range).
*   **Output:** `RegimeState` (Enum: `ORDERED`, `CRITICAL`, `DISORDERED`).

#### C. CorrelationSensor (The Systemic Meter)
*   **File:** `src/router/sensors/correlation.py`
*   **Algorithm:** **Random Matrix Theory** (RMT).
*   **Input:** Real-time periodic correlation matrix of active pairs.
*   **Logic:** Computes Maximum Eigenvalue ($\lambda_{max}$).
    *   $\lambda_{max} \gg 2.0$: Swarm Coupling (Systemic Risk).
*   **Output:** `CorrelScore` (0.0 to 1.0).

#### D. NewsSensor (The Time Guardian)
*   **File:** `src/router/sensors/news.py`
*   **Source:** Economic Calendar API (ForexFactory/Econoday).
*   **Logic:** "Kill Zone" enforcement.
    *   *T-15 mins:* Classification -> `PRE_NEWS` (Block Entry).
    *   *T-0 mins:* Classification -> `NEWS_EVENT` (Hard Stop).
    *   *T+15 mins:* Classification -> `POST_NEWS` (Resume if Chaos < 0.3).
*   **Output:** `NewsState` Enum.

### 2.2 The Regime Classification Matrix

| Chaos Score | Ising State | News State | Regime Output | Authorized Bot Class |
|:-----------:|:-----------:|:----------:|:-------------:|:--------------------:|
| Low (<0.3) | `ORDERED` | `SAFE` | **TREND_STABLE** | Trend Followers |
| Low (<0.3) | `DISORDERED` | `SAFE` | **RANGE_STABLE** | Mean Reversion |
| Any | `CRITICAL` | `SAFE` | **BREAKOUT_PRIME** | Breakout/Snipers |
| High (>0.6) | Any | Any | **HIGH_CHAOS** | **NONE** (Bunker Mode) |
| Any | Any | `KILL_ZONE` | **NEWS_EVENT** | **NONE** (Flat) |

---

## 3. Component 2: The Governor (Compliance Layer)

**Module:** `src/router/governor.py`
**Responsibility:** Final Risk Authorization.

The Governor calculates the **Risk Scalar** ($0.0 - 1.0$) for every trade request.

### 3.1 The Risk Stacking Formula

The final authorized size for any trade is a product of three tiers:

$$ \text{FinalRisk} = \text{Tier1 (Bot)} \times \text{Tier2 (Swarm)} \times \text{Tier3 (Prop)} $$

1.  **Tier 1 (Bot/Kelly):** The Strategy's native edge calculation (see Component 6).
2.  **Tier 2 (Swarm/VaR):** **Portfolio Value-at-Risk**.
    *   If `PortfolioCorrelation > 0.8` (RMT), Tier 2 Scalar drops to 0.5 or lower.
3.  **Tier 3 (Prop/Throttle):** **Quadratic Survival Curve**.
    *   If `PropGovernor` is active (see Phase 5), it applies the "Distance to Ruin" throttle.

### 3.2 Native Bridge to Prop Module
The Governor is designed to be **Polymorphic**.
*   Standard Mode: Uses `BaseGovernor`.
*   Prop Mode: Uses `PropGovernor` (Phase 5 Extension), which inherits Tier 2 logic but wraps it with Prop Rules.

---

## 4. Component 3: The Commander (Decision Layer)

**Module:** `src/router/commander.py`
**Responsibility:** Bot Selection & Dispatch.

### 4.1 The Strategy Auction
When the Sentinel reports a Regime change, the Commander holds an "Auction":
1.  **Call for Bids:** All registered Bots return their suitability for the current Regime.
2.  **Rank:** Bots are ranked by `PerformanceScore` (Historical R:R).
3.  **Select:** The Top N bots are authorized.

### 4.2 The Dispatch Protocol
The Commander outputs instructions via the **Native Socket Interface**.

```json
// Example Dispatch Command
{
  "type": "COMMAND_DISPATCH",
  "target_magic": [1001, 1002],  // "Trend Bots"
  "action": "ENABLE",
  "params": {
    "risk_mode": "DYNAMIC",
    "regime_scalar": 1.0
  }
}
```

---

## 5. Component 4: The Interface (Native Bridge)

**Module:** `mcp-metatrader5-server` (Wrapper + Socket)
**Protocol:** ZeroMQ / TCP Sockets (JSON).

The Router does **not** touch files. It streams commands to the Bridge, which executes MQL5 orders.
*   **Port:** 5555 (Command), 5556 (Data Stream).
*   **Latency:** < 5ms.

---

## 6. Component 5: The Bot Architecture (Tier 1 Integration)

**Module:** `src/library/base_bot.py`
**Target:** All Strategy Implementations.

This is the **CRITICAL LINK** between Phase 4 (Router) and Phase 5 (Library). All bots must use the `PhysicsAwareKelly` engine.

### 6.1 The QuantMindBot Specification

```python
# MANDATORY IMPORT for all Bots
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

class QuantMindBot:
    def __init__(self, magic_number):
        self.magic_number = magic_number
        self.risk_engine = EnhancedKellyCalculator() # Tier 1 Engine

    def calculate_entry_size(self, signal_strength: float, sentinel_data: dict) -> float:
        """
        Calculates the Tier 1 position size.
        """
        # 1. Extract Physics Quality from Sentinel Report
        #    (Chaos 0.0 -> Quality 1.0)
        regime_quality = 1.0 - sentinel_data.get('chaos_score', 0.0)
        
        # 2. Ask Kelly for Size
        #    This is the "Bot's Opinion" before the Governor sees it.
        size_lots = self.risk_engine.calculate(
            account_balance=self.balance,
            win_rate=self.expected_win_rate,
            regime_quality=regime_quality,  # <--- PHYSICS INJECTION
            volatility_scalar=sentinel_data.get('volatility_ratio', 1.0)
        )
        return size_lots
```

---

## 7. Directory Structure (Implementation Map)

This is the canonical layout for the coding agent.

```
src/
├── router/
│   ├── __init__.py
│   ├── sentinel.py         # The Loop (Observer)
│   ├── governor.py         # The Brake (Risk Logic)
│   ├── commander.py        # The Hand (Dispatcher)
│   ├── state.py            # Redis State Manager
│   └── sensors/            # The Sensor Array
│       ├── __init__.py
│       ├── chaos.py        # Lyapunov Implementation
│       ├── regime.py       # Ising Implementation
│       ├── correlation.py  # RMT Implementation
│       └── news.py         # News Guard Implementation
├── library/
│   ├── base_bot.py         # QuantMindBot (Abstract Base)
│   └── strategies/
│       ├── orb_strategy.py
│       └── ...
└── position_sizing/
    ├── enhanced_kelly.py   # PhysicsAwareKelly Calculator
    └── kelly_config.py     # Risk Parameters
```

---

## 9. Implementation Status (2026-01-28)

The following components have been implemented on branch `feature/prop-module`:

### Core Router (`src/router/`)
*   **Sentinel:** Full loop implemented.
*   **Sensors:** 
    *   `chaos.py` (Lyapunov Proxy) - **DONE**
    *   `regime.py` (Ising Model) - **DONE**
    *   `correlation.py` (RMT Stub) - **DONE**
    *   `news.py` (Calendar Logic) - **DONE**
*   **Governor:** Base VaR logic implemented.
*   **Commander:** Base Auction logic implemented.
*   **State:** Shared Memory implemented.

### Prop Extension (`src/router/prop/`)
*   **PropGovernor:** Quadratic Throttle & News Guard - **DONE**
*   **PropCommander:** Preservation Mode & Coin Flip Bot - **DONE**
*   **PropState:** Persistence Layer - **DONE**

### Pending
*   **Kelly Linkage:** The connection to `enhanced_kelly.py` is currently a placeholder until the Kelly Upgrade (Physics Injection) is applied at a later verification stage.
