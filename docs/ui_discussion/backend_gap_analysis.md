# Backend Gap Analysis: Supporting "Session-Based" QuantMind X

## 🚨 Critical Missing Infrastructure

### 1. The "Session Manager" (Stateful Context)
**Current State:** Agents (QuantCode, Analyst) run as "One-Off" processes. Context is lost after execution.
**Requirement:** Persistent Chat Sessions linked to specific EA Manifests.
**Gap:**
*   Need a **Session Store** (SQLite/Redis) to save: `session_id`, `manifest_id`, `chat_history` (User + Agent messages).
*   Need a **Context Resolver**: A service that, upon "Resuming" a session, fetches:
    *   Current EA Performance (from MT5).
    *   Last Compiler Errors.
    *   Backtest History.
    *   Injects this as a "System Prompt" update.

### 2. The Strategy Router Broadcast
**Current State:** `PROP_FIRMS_AND_ROUTER.md` defines the *logic* (Regime classification, Correlation checks), but it's a static Python script.
**Requirement:** Live UI Visualization (Node Graph).
**Gap:**
*   Need a **WebSocket Publisher** in the Router Service.
*   Must broadcast events: `RouterNodeUpdate` (Active/Paused), `KellyUpdate` (Current Size), `GlobalRisk` (Current Drawdown).

### 3. MT5 Integration: Local vs Remote
**Current State:** `SESSION_SUMMARY.md` mentions a `mt5-bridge` (FastAPI) for *remote* VPS.
**Requirement:** User wants "Data Packet" methods and "Local Python API" (No DLLs).
**Gap:**
*   If running on Linux (User's OS), **MT5 is not native**.
*   **Solution:** The "Local" API actually needs to run inside the **Wine/Docker container** where MT5 is installed, *or* we stick to the VPS Bridge but make it "feel" local via the API Service (Transparent Proxy).
*   *Clarification Needed:* Is the user running MT5 on this specific Linux machine (via Wine), or just controlling a VPS? (The `mt5-bridge` folder suggests VPS).

### 4. Prop Firm Scraper Agent
**Current State:** No existing tooling for scraping specific Prop Firm T&Cs.
**Requirement:** `US-7.2` Scraper Agent.
**Gap:**
*   Need a **Firecrawl / Puppeteer** skill specifically tuned for Prop Firm Tables (Parsing "Max Loss", "Profit Target" from HTML tables).
*   Need a **Standardized Prop Node Schema** to store this data in the `docs/knowledge/prop_firms/` directory.

### 5. The "Manifest" File System Watcher
**Current State:** No automated "Manifest" creation.
**Requirement:** `US-5.1` Standardized Folders.
**Gap:**
*   Need a "FileSystem Watcher" or "Asset Service" that:
    *   Auto-creates `status.json` when a new Folder is detected.
    *   Updates `status.json` when key files (`code.mq5`) change.

---

## 🛠️ Recommended Backend Tasks (to add to task.md)

1.  [ ] **Implement Session Store:** Simple SQLite DB to track `chat_sessions`.
2.  [ ] **Create "Context Weavor" Service:** Python class to gather P&L + Backtest + Error Logs into a Context String.
3.  [ ] **Refactor Strategy Router:** Add `fastapi-websocket` to broadcast internal state.
4.  [ ] **Build Prop Firm Scraper Skill:** Create `skills/prop_scraper` with specialized selectors.
