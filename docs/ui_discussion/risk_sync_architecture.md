# Risk Management & Sync Architecture

## The Challenge
Connect **QuantMind's Brain** (Python/Econophysics) to the **MT5 VPS's Muscle** (Execution EAs) to enable dynamic, "On-the-fly" position sizing.

---

## Approach 1: The "Real-Time Nervous System" (API Sync)
**Best for:** High-speed adjustments (Scalping), "Strategy Router" logic.

**How it works:**
1.  **QuantMind API** runs on the VPS (or accessible via tunnel).
2.  **EA Pre-Trade:** Before `OrderSend`, EA sends a `WebRequest` (GET `http://localhost:8000/risk/EURUSD`).
3.  **API Response:**
    ```json
    {
      "physics_multiplier": 0.5,
      "recommendation": "HALVE_RISK",
      "reason": "Lyapunov Chaos Detected",
      "max_lots": 1.5
    }
    ```
4.  **EA Action:** Modifies lot size instantly.

**Pros:** Instant reaction to market "Physics".
**Cons:** Adds 50-100ms latency.

---

## Approach 2: The "Updates Packet" (File Sync)
**Best for:** Swing trading, Portfolio Correlation updates.

**How it works:**
1.  **QuantMind Python** calculates Portfolio Correlations every 15 mins.
2.  **Write File:** Updates `C:\Users\MT5\Common\Files\risk_matrix.json`.
3.  **EA Logic:**
    *   `OnTick()`: Checks file timestamp.
    *   If changed, reloads global `Risk_Multiplier` variable.

**Pros:** Zero trading latency (pre-loaded). Robust if API fails.
**Cons:** "Laggy" (updates every X mins, not tick-by-tick).

---

## Approach 3: Internal "Reflex" Logic (EA Native)
**Best for:** Safety/Fallbacks (No Python needed).

**Concept:** "Logic Hedging" & "Safety Valves" embedded in MQL5.

1.  **The "Circuit Breaker" Function:**
    *   Tracks its own `DailyLoss`.
    *   If `DailyLoss > 2%` -> `ExpertRemove()` (Self-destruct for the day).
2.  **Logic Hedging:**
    *   **Inverse Correlation Check:** If `EurUsd_Long` exists, *block* `GbpUsd_Long` locally (using `iCorrelation` in MQL5, though less accurate than Python).
3.  **Native Kelly:**
    *   Maintain simple array of `Last_50_Trades`.
    *   Calc `WinRate` & `Payoff` inside `OnInit`.
    *   Apply standard Kelly fraction locally.

---

## Recommended Hybrid Architecture

1.  **Primary:** **Approach 2 (File Sync)** for the heavy "Econophysics" multipliers (updated every 5-15 mins).
2.  **Execution:** **Approach 3 (Internal)** for hard "Circuit Breakers" (Daily Loss limit).
3.  **Router:** **Approach 1 (API)** only for "We Begin" or "Switch Pair" commands.
