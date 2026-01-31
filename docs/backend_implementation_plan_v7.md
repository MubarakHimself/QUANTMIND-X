# Backend Implementation Plan v7: The QuantMind Hybrid Core

> **Status:** APPROVED FOR EXECUTION
> **Version:** 7.3 (ChromaDB + PropFirm Edition)
> **Location:** `docs/backend_implementation_plan_v7.md`
> **Primary Consumer:** The Coding Agent (System Builder)

---

## 🏗️ Architectural Philosophy: The "Hybrid Brain & Body" Narrative

**Context:**
We are building a High-Frequency/Prop-Firm compatible trading system.
Previous iterations (v1-v6) focused on "Bridging the Gap", resulting in a monolithic library (`QuantMind_Risk.mqh`).
**Version 7 represents the "Hybrid Core" evolution.**
We are moving from that single file to a **Modular Standard Library (QSL)** supported by a robust Database Layer.

### The "Why" behind the Refactor (Context for Coding Agent)
You will find an existing file `QuantMind_Risk.mqh`. **Do not simply delete it.**
We are refactoring it because:
1.  **Separation of Concerns:** The "PropFirm Manager" (Drawdown) should not live with "JSON Parsers".
2.  **Agent Isolation:** Different Agents need specific subsets of the library.
3.  **Persistence:** We need a real database to track "Daily Drawdown" across restarts.

---

## 🏗️ 1. Agent Workspaces & Queues

**Concept:** Each Agent must have a dedicated workspace to operate without collision.
**Root:** `workspaces/` (New Directory)

### 1.1 Directory Structure
The Coding Agent must create this structure:

```text
workspaces/
├── analyst/            # (Analyst Agent)
│   ├── specs/          # Technical Requirement Docs (TRDs)
│   └── logs/           # Research logs & scraping dumps
│
├── quant/              # (Quant Agent)
│   ├── strategies/     # Python strategy drafts
│   └── backtests/      # Temporary backtest artifacts
│
└── executor/           # (Executor Agent)
    ├── deployment/     # Manifests & Compiled EAs
    └── heartbeat/      # Live status logs
```

### 1.2 Async Job Queues
To support "Cloud Code" style asynchronous tasks:
*   **Path:** `data/queues/`
*   **Usage:** Simple file-based FIFO queues (`quant_tasks.json`, `analyst_tasks.json`).

---

## 💾 2. The Database Layer (Phase 0.5)

**Concept:** A "Local-First" Persistence Stack that mimics Cloud Architecture.

### 2.1 Technical Specs
1.  **Relational (Structured):**
    *   **Engine:** SQLite (`data/quantmind.db`).
    *   **ORM:** SQLAlchemy (Python).
    *   **Purpose:** Storing PropFirm Accounts, Daily Snapshots, and Trade Logs.
2.  **Vector (Knowledge):**
    *   **Engine:** ChromaDB (`data/chromadb/`).
    *   **Purpose:** Storing Embeddings of Articles and Strategy DNA.
    *   **Note:** Replaces Qdrant.

### 2.2 Schema Requirements
*   **`PropFirmAccounts`:** `id`, `firm_name`, `daily_loss_limit` (e.g., 0.05).
*   **`DailySnapshots`:** `account_id`, `high_water_mark`, `current_equity`.
    *   *Critical:* This table drives the **Quadratic Throttle**.

---

## 📚 3. The Source of Truth (The Map)

**Strict Instruction to Coding Agent:**
Transplant logic from these Reference Files.

| Component | Logic Source | Absolute Path to Truth |
|-----------|--------------|------------------------|
| **Prop Limits (MQL5)** | Daily Drawdown & Hard Stop Logic | `file:///home/mubarkahimself/Desktop/QUANTMINDX/data/scraped_articles/trading/automated_risk_management_for_passing_prop_firm_challenges.md` |
| **Ring Buffer** | O(1) Indicator Memory | `file:///home/mubarkahimself/Desktop/QUANTMINDX/data/scraped_articles/optimization/ring_buffer_basics.md` |
| **Prop Logic (Py)** | Kelly Filter & Throttle | `file:///home/mubarkahimself/Desktop/QUANTMINDX/docs/trds/prop_module_v1.md` |
| **Socket Comms** | Low-Latency Bridge | `file:///home/mubarkahimself/Desktop/QUANTMINDX/data/scraped_articles/integration/websockets_for_metatrader_5.md` |

---

## 🏗️ 4. The QuantMind Standard Library (QSL)

**Concept:** A modular "Lego Block" system for Agents to discover and build with.
**Root:** `src/mql5/Include/QuantMind/`

### 4.1 Directory Structure
```text
QuantMind/
├── Core/               # (Base Dependencies)
│   ├── Object.mqh      # Base Entity class
│   └── Config.mqh      # Global Environment Loader
│
├── Risk/               # (Executor Agent's Domain)
│   ├── PropManager.mqh # Daily Drawdown / News Guard
│   ├── RiskClient.mqh  # Bridge to Python (JSON/REST)
│   └── KellySizer.mqh  # Position Sizing Logic
│
├── Signals/            # (Quant Agent's Domain)
│   ├── Indicators/     # Reusable logic (RingBuffer inside)
│
└── Utils/              # (Shared Helpers)
    ├── JSON.mqh        # CJAVal Parser
```

### 4.2 Module Requirements
1.  **Risk/PropManager.mqh**: Must implement `DailyDrawdownLock` (4.5% Hard Stop) and `NewsGuard`.
2.  **Signals/Indicators/*:** Must utilize `CRiBuff` (Ring Buffer).
3.  **Risk/RiskClient.mqh**: Must run a recursive file watcher on `risk_matrix.json`.

---

## 🧠 5. The PropFirm Module (Python Side)

**Concept:** The "Offensive Brain". It ensures we only risk capital on the best trades.

### 5.1 The Prop Commander
**File:** `src/router/prop/commander.py`
We found that the Prop Module *does* implement Kelly logic, but as a **Filter**.
*   **The "A+ Setup" Rule:** If `Mode=Preservation`, it **REJECTS** all trades where `KellyScore < 0.8`.

### 5.2 The Prop Governor
**File:** `src/router/prop/governor.py`
This component implements the **Quadratic Throttle**.
*   **Formula:** `Multiplier = ((MaxLoss - CurrentLoss) / MaxLoss) ^ 2`.

---

## 🏗️ 6. Migration Strategy (v6 -> v7)

**Current State (v6):**
*   `src/mql5/Include/QuantMind/QuantMind_Risk.mqh` exists (Monolith).

**Target State (v7):**
*   Modular QSL + Database.

**Refactoring Steps:**
1.  **Setup Phase:** Create `workspaces/`, `data/queues/`, and `data/chromadb/`.
2.  **Deconstruct:** Move `CJAVal` -> `Utils/JSON.mqh`, `Heartbeat` -> `Risk/RiskClient.mqh`.
3.  **Enhance:** Implement `PropManager.mqh` (New Logic).
4.  **Database:** Init `quantmind.db` and run migrations.

---

## 📝 Execution Checklist for Coding Agent

**Phase 0: Workspace & Database Setup**
1.  [ ] **Create Workspaces:** `workspaces/{analyst,quant,executor}`.
2.  [ ] **Init Database:** Set up SQLite and ChromaDB connections.
3.  [ ] **Install Deps:** Add `sqlalchemy`, `chromadb` to `requirements.txt`.

**Phase 1: The QuantMind Standard Library (QSL)**
1.  [ ] **Create Folder:** `src/mql5/Include/QuantMind/` and subfolders.
2.  [ ] **Implement Utils:** `JSON.mqh`, `Sockets.mqh`.
3.  [ ] **Implement Risk:** `PropManager.mqh` (PropFirm Logic), `RiskClient.mqh`.
4.  [ ] **Create Index:** `docs/knowledge/mql5_asset_index.md`.

**Phase 2: The Logic Upgrade (Python)**
1.  [ ] **Refactor Commander:** `src/router/prop/commander.py` -> Add `KellyScore > 0.8` logic.
2.  [ ] **Refactor Governor:** `src/router/prop/governor.py` -> Add `QuadraticThrottle`.

**Phase 3: The Pipeline**
1.  [ ] **Config:** Ensure `mcp-metatrader5-server` is active.

---

**Signed:** Antigravity (Planning Agent)
**Date:** Jan 30, 2026
