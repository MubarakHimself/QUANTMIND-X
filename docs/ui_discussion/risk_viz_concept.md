# Visualization Concept: Strategy Router & Kelly Criterion

## 🎯 The Problem
The **Strategy Router** determines *which* strategies run, and the **Kelly Criterion** determines *how much* they wager. Since these values are streamed "on the fly" but also have local safeguards, we need to visualize the **Flow of Authority**.

## 🧠 Proposed Architecture: The "Risk Hierarchy" Graph

Imagine a top-down Node Graph (like a genealogy tree or flowchart):

### Level 1: The Sovereign (Account)
*   **Node:** Global Risk Controller.
*   **Data:** Total Equity, Daily Drawdown Limit, Global Kelly Cap (e.g., "Max 5% Total Risk").
*   **Visual:** Large central node. Color shifts (Green -> Red) as Drawdown approaches limit.

### Level 2: The Routers (Strategy Groups)
*   **Nodes:** "Trending Pairs", "Mean Reversion", "News Scalpers".
*   **Data:** Allocation % (e.g., "Trending gets 40% of risk").
*   **Function:** This is the **Strategy Router**. It turns branches on/off based on market regime (Volatile vs Calm).

### Level 3: The Workers (Active EAs)
*   **Nodes:** Individual Bot Instances (e.g., `ORB-EURUSD-M15`).
*   **Visual Connection:**
    *   **Solid Line:** Router is Active.
    *   **Dotted Line:** Router is Paused.
    *   **Line Thickness:** Represents Position Size (Kelly Value).
*   **Inner Badges:**
    *   🛡️ **Safeguard:** Icon appears if Local Safeguard (Hard stop) overrides Global Kelly.
    *   📈 **Perf:** Tiny sparkline of P&L.

## 🎛️ Interaction Model

1.  **Click-to-Inspect:**
    *   Clicking a Worker Node shows the "Risk Calculation Trace":
        *   *Global Cap (2%) -> Router Weight (0.5) -> Local Kelly (1.2) = Final Size (0.6 Lots).*
2.  **Man-in-the-Loop Override:**
    *   Right-click a node to "Sever Connection" (Stop Trading) or "Manual Override" (Force Lot Size).

---

## 🧭 Context Integration for Agents

This visualization isn't just for you; it's the **Dashboard for the Agents**.

*   **Analyst Agent:** Uses this map to write specific Risk Clauses in the TRD (e.g., "This EA must respect the Level 2 Router's 'Low Volatility' signal").
*   **QuantCode:** When coding, it injects the specific listener code (Signal Slots) to subscribe to the correct Router Node.
