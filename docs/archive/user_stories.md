# QuantMind X - User Stories & Workflow Definition (v3)

## 🌟 The Vision
A "Zero-to-Hero" Algorithmic Trading Factory. A unified IDE where raw ideas (videos/articles) enter, and profitable, risk-managed EAs exit to live markets or prop firm challenges.

---

## 🟢 Epic 1: Dashboard & Navigation (The Cockpit)
**Goal:** Global visibility and fluid navigation.

*   **US-1.1: The Top Bar:**
    *   Display **Global Account Balance** (Live sync via MT5 Python API).
    *   Display **Active Prop Firm Challenges** (Count/Status).
    *   Clicking these metrics navigates to specific detailed views.
*   **US-1.2: Refined Sidebar (Activity Bar):**
    *   **Library:** Merged "Knowledge Hub" (Articles) + "Assets" (EAs).
    *   **Prop Firms:** Dedicated section.
    *   **NPRD:** (Optional: can merge Output into Library).
    *   **Code Explorer:** Standard file tree (Permissions fixed).

## 📚 Epic 2: The Library (Knowledge + Assets)
**Goal:** Centralized intelligence and asset management.

*   **US-2.1: Knowledge Hub (Articles):**
    *   Render generic Markdown articles (Scripts/Strategies) with syntax highlighting.
    *   **Actionable Code:** "Add to Copilot" / "Add to Chat" buttons on code snippets.
    *   **Organization:** Drag-and-drop articles into specific "Agent Knowledge Bases".
*   **US-2.2: NPRD Integration ("The Librarian"):**
    *   NPRD outputs (Storylines) are auto-synced here.
    *   **Data Structure:** Must include Speaker Diarization, OCR Text, and Visual Descriptions (No interpretation).
    *   Display tags/status (e.g., "In Progress", "New").
*   **US-2.3: Assets (EAs):**
    *   List available EAs.
    *   Show Market/Pair configuration and Live Performance.
    *   **Actionable:** "Clone to Chat" / "Deploy to MT5".
*   **US-2.4: Shared Asset Library:**
    *   **Goal:** Allow Analyst and QuantCode to share reusable components (MQL5 Libraries, Indicators).
    *   **Mechanism:** "Assets/Include" folder is indexed. Analyst can "cite" a standardized `QuantMind_Risk.mqh`.
    *   **UI:** "Add to Context" button for Library items in Chat.

## 🤖 Epic 3: The Agent Delegation (Right Sidebar)
**Goal:** Specialized Agents for specific tasks.

*   **US-3.1: The Agent List:**
    *   Sidebar must explicitly list: **Analyst Agent**, **QuantCode Agent**, **QuantMind Copilot**.
    *   (Remove generic names like "Executor" or "Quant Agent").
*   **US-3.2: Interaction:**
    *   Toggling an agent switches the Chat Context and the Active Knowledge Base.

## 💬 Epic 4: Rich Chat Interface (The Director)
**Goal:** Human-in-the-loop orchestration.

*   **US-4.1: Input Capabilities:**
    *   **Slash Commands:** `/analyze`, `/code`, `/backtest`.
    *   **Attachments:** Button to upload/attach files to context.
    *   **Mode Switch:** Toggle between "Autonomous" vs "Interactive" execution.
    *   **Skills:** Dropdown to select "Agentic Skills" (e.g., "Interface Design", "Data Science").
*   **US-4.2: The Handoff:**
    *   User acts as the "Director", triggering agents via Chat.
    *   Major state changes ("We Begin") require user confirmation.

## 👨‍💻 Epic 5: Development & Strategy Gen (Session-Based)
**Goal:** Persistent, Context-Aware Autonomous Coding.

*   **US-5.1: The "We Begin" Trigger:**
    *   User clicks "Start Development" (or "Add to Chat") on a TRD in the Library.
    *   System creates a **Development Session** linked to that specific EA Manifest.
*   **US-5.2: Session Persistence:**
    *   **New Session:** Agent analyzes TRD, asks clarifying questions (if interactive), and codes V1.
    *   **Resume:** Clicking "Show Progress" opens the *existing* chat history for that EA. All previous context (decisions, loops) is preserved.
*   **US-5.3: Context Resolution (The Memory):**
    *   When resuming/iterating an EA, the Agent automatically scans:
        1.  **Active Instances:** Is it running? What is the current P&L/Drawdown?
        2.  **Backtest History:** What were the last results?
    *   This data acts as the "System Prompt" for the new session ("You are fixing a bug in ORB_EA_V1 which failed with 10% DD").
*   **US-5.4: The Build Loop:**
    *   Agent generates `.mq5` -> Sends to **MetaEditor MCP** -> Compiles.
    *   **Success:** Increments Version (V1 -> V2) and updates the Library Asset entry.
    *   **Error:** Agent self-corrects based on compiler logs (Autonomous Loop).
*   **US-5.5: EA Tagging System:**
    *   EAs are tagged by status (Incubation, Paper, Live, Archived).
    *   **Feedback Loop:** Failed Paper Trading EAs are moved to "Archived".

## 📊 Epic 6: Infrastructure & Viz
**Goal:** Visualizing the invisible engine.

*   **US-6.1: Strategy Router & Kelly:**
    *   Visual Node Graph showing how signals are routed.
    *   Live display of Kelly Criterion sizing calculations.
*   **US-6.2: MT5 Integration:**
    *   Use `MetaTrader5` Python package (No DLLs).
    *   Sync P&L and Balance in real-time.
*   **US-6.3: Adaptive Execution (KLE & Router):**
    *   **Pre-Trade Check:** Before any order, EA queries the **Kelly Logic Engine (KLE)**.
    *   **Dynamic Sizing:** KLE calculates `Size = Balance * Kelly% * PhysicsMultiplier` (Chaotic/Critical state dampening).
    *   **Auto-Deployment:** Router can re-assign an EA to a different pair/symbol if the current one becomes "Inefficient" (High Lyapunov Exponent).

## 🏆 Epic 7: Prop Firm Manager
**Goal:** Managing external funding constraints.

*   **US-7.1: Challenge Dashboard:**
    *   View Terms & Conditions (Max Daily Loss, Profit Target).
    *   Live tracking of current P&L vs Limits.
*   **US-7.2: Scraper:**
    *   Agent crawls Prop Firm sites to parse T&Cs into a "Prop Firm Node".

## ⚙️ Epic 8: Configuration & Logging
**Goal:** Stability and Debugging.

*   **US-8.1: Global Problems Panel:**
    *   VS Code-style bottom panel.
    *   Aggregates logs/errors from Python Agents, Docker Containers, and MT5.

## 🧪 Epic 9: Backtest Engine Management
**Goal:** Validating strategies before deployment.

*   **US-9.1: Live Runner Panel:**
    *   **Inputs:** Initial Cash, Commission, Date Range.
    *   **Control:** "Run Backtest" button (Handles blocking execution with spinner).
*   **US-9.2: Results Dashboard:**
    *   **Metrics:** Sharpe Ratio, Max Drawdown, Total Return, Trade Count.
    *   **Visuals:** Equity Curve Chart (requires backend upgrade to export series).
    *   **Logs:** Collapsible "Execution Log" to debug trade logic.
