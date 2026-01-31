# Backend Implementation Plan v8: The Omni-Asset Scalper System

> **Status:** DRAFT FOR EXECUTION
> **Version:** 8.0 (Growth & Crypto Edition)
> **Location:** `docs/backend_implementation_plan_v8.md`
> **Prerequisites:** Completion of Plan v7.3 (Core Library & Database)

---

## 🏗️ 1. Architectural Evolution: Beyond Forex

**Context:**
Plan v7 established the "Hybrid Core" (Python Brain + MQL5 Body).
**Plan v8 expands this into an "Omni-Asset" system** capable of aggressive small-account growth and high-frequency crypto scalping.

### The "Why" (Research Synthesis)
1.  **Small Account Paradox:**
    *   *Problem:* Standard Kelly Criterion is too conservative for a $100 account, freezing growth (The "Kelly Halt"). Conversely, aggressive martingale leads to ruin (Ref: *LeoV's ATC Story* in `expert_advisors_based_on_popular_trading_systems...md`).
    *   *Solution:* A **Tiered Risk Engine** that switches modes based on equity.
2.  **Execution Speed:**
    *   *Requirement:* Scalping strategies (Order Flow/FVG) die if latency > 100ms.
    *   *Ref:* `wrtrading.com` (HFT Brokers) confirms Exness/Vantage are suitable if <50ms.
3.  **Crypto Architecture:**
    *   *Ref:* `hummingbot.org` (V2 Connectors).
    *   *Strategy:* We will adopt their "Connector" pattern to treat Binance Spot exactly like an MT5 Broker.

---

## 📈 2. The Tiered Risk Engine (Growth vs. Preservation)

**Goal:** Safely grow small accounts ($100) without blowing up, then lock in profits ($1,000+).

### 2.1 Logic Specification (For `KellySizer.mqh`)

| Tier | Equity Range | Risk Mode | Formula | Reason |
|------|--------------|-----------|---------|--------|
| **1. Growth** | $100 - $1,000 | **Fixed Fractional** | `Risk = $5` (Fixed) | Avoids stagnation. Linear growth. |
| **2. Scaling** | $1,000 - $5,000 | **Kelly Standard** | `Risk = Kelly %` | Introduction of geometric growth. |
| **3. Guardian** | $5,000+ | **Quadratic Throttle** | `Risk * ((Max - Loss)/Max)^2` | Preservation of capital. Prop Firm defense. |

**Instruction to Coding Agent:**
*   Modify `src/mql5/Include/QuantMind/Risk/KellySizer.mqh`.
*   Add `input double GrowthModeCeiling = 1000.0;`.
*   Add `input double FixedRiskAmount = 5.0;`.

---

## ⚡ 3. High-Frequency Execution (HFT Readiness)

**Goal:** Ensure the system can handle sub-second scalping on Volatility Indices (Deriv) and Raw FX (Exness).

### 3.1 The "Fast Path" Optimization
The current REST Heartbeat (1/min) is too slow for trade management.
We need a **Local Socket Stream**.

*   **Component:** `QuantMind/Utils/Sockets.mqh`.
*   **Logic:**
    1.  EAs open a persistent `ZMQ` (ZeroMQ) or `NamedPipe` connection to the local Python Agent.
    2.  *Trade Events* (Open/Close) are pushed **instantly** (<5ms), bypassing the DB polling loop.
*   *Why:* This allows the "Pilot Model" (future) to cancel a trade mid-execution if Order Flow shifts.

---

## 🤖 4. The Crypto Module (Hummingbot Pattern)

**Goal:** Trade Binance Spot/Futures using the same "Quant Agent" logic.

### 4.1 Architecture
We will NOT rewrite the wheel. We will build a **Connector** compatible with our Agent Workspaces.

*   **Directory:** `src/integrations/crypto/binance_connector.py`.
*   **Interface:** Must implement the standard `BrokerClient` interface (same as MT5).
    *   `get_balance()`
    *   `place_order(symbol, volume, direction)`
    *   `get_order_book()` (L2 Data for Scalping).

### 4.2 Spot Scalping Strategy
*   **Target:** `BTCUSDT`, `ETHUSDT`.
*   **Mechanism:** Limit Order grid (Market Making) vs. Taker Scalping.
*   *Status:* Phase 2 of this plan.

---

## 📝 5. Execution Checklist (Plan v8)

**Phase 1: Risk Engine Upgrade**
1.  [ ] **Modify QSL:** Update `KellySizer.mqh` with Tiered Risk Logic.
2.  [ ] **Update DB:** Add `risk_mode` column to `PropFirmAccounts` table.

**Phase 2: HFT Infrastructure**
1.  [ ] **Install lib:** Add `pyzmq` to `requirements.txt`.
2.  [ ] **Create Bridge:** Implement `socket_server.py` in `src/router/` for <5ms event listening.

**Phase 3: Crypto Connector**
1.  [ ] **Scaffold:** Create `src/integrations/crypto/`.
2.  [ ] **Implement:** Basic `BinanceClient` using `ccxt` (referenced in Session 2).

---

**Signed:** Antigravity (Planning Agent)
**Date:** Jan 30, 2026
