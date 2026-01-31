# Backend Implementation Specification
**Target Agent:** Implementation/Coding Agent
**Objective:** Build the missing "Hybrid Risk" and "Skill" assets required by the QuantMind X UI.

---

## 📦 Deliverable 1: QuantMind_Risk.mqh (The Shared Library)
**Path:** `extensions/includes/QuantMind_Risk.mqh` (Create if missing)

**Purpose:** A standardized MQL5 header file that EAs import to get "Physics-Aware" risk sizing.

**Requirements:**
1.  **Function:** `double QuantMind_GetRiskMultiplier(string symbol)`
2.  **Logic (Hybrid):**
    *   **Check 1 (File):** Read `Common/Files/risk_matrix.json` (Low latency).
    *   **Check 2 (API):** If file is stale (>15 mins), try `WebRequest("http://localhost:5555/risk/...")`.
    *   **Fallback:** Return `1.0` (Normal Risk) if both fail.
3.  **Circuit Breaker:** `bool QuantMind_CheckSafety()` – Returns `false` if Daily Loss > 2%.

**Code Sketch (MQL5):**
```cpp
//+------------------------------------------------------------------+
//| QuantMind_Risk.mqh - Universal Risk Interface                    |
//+------------------------------------------------------------------+
class CQuantMindRisk {
   public:
      static double GetMultiplier(string symbol) {
         // 1. Try Read File (fastest)
         double fileMult = ReadRiskFile(symbol); 
         if(fileMult != -1) return fileMult;
         
         // 2. Try API (fallback)
         return QueryAPI(symbol);
      }
};
```

---

## 🔄 Deliverable 2: Strategy Router Update (File Sync)
**Path:** `src/router/interface.py` & `src/router/syncer.py`

**Purpose:** The Python side of the "Hybrid" architecture. Must write the JSON file that the MQL5 library reads.

**Requirements:**
1.  **New Class:** `RiskSyncer`
2.  **Input:** Takes `MarketPhysics` objects (from `src.risk.physics`).
3.  **Output:** Writes to `MT5_COMMON_DIR/risk_matrix.json`.
4.  **Format:**
    ```json
    {
      "updated_at": 1715000000,
      "EURUSD": { "multiplier": 0.5, "reason": "Chaos" },
      "GBPUSD": { "multiplier": 1.0, "reason": "Stable" }
    }
    ```

---

## 🧠 Deliverable 3: Indicator Writer Skill
**Path:** `extensions/coder/skills/indicator_writer/SKILL.md`

**Purpose:** Gives the "QuantCode" agent the specific ability to write MQL5 Custom Indicators.

**Requirements (SKILL.md content):**
1.  **Role:** "The Technician".
2.  **Capabilities:**
    *   Create `.mq5` files in `Indicators/Custom/`.
    *   Use `OnCalculate` (not `OnTick`).
    *   Implement `SetIndexBuffer` correctly.
3.  **Template:** Include a robust "Blank Indicator" template in the skill instructions.

---
**Note to Agent:** Please verify `src/risk` exists before connecting the Syncer. The Physics engine is already built.
