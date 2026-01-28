# QuantMind IDE UI Specification

> **Vision:** "Visual Studio Code for Algo-Trading"
> **Aesthetic:** Dark "Mineral" Glassmorphism (Deep Cyan/Purple Accents)
> **Platform:** Next.js (Web) / Tauri (Desktop Wrapper)

![Mockup](/home/mubarkahimself/.gemini/antigravity/brain/b0d88891-4f13-44a1-aef7-5d074653333f/quantmind_ide_mockup_1769623886922.png)

## 1. Layout Architecture (The "Workbench")

The layout follows a standard IDE grid system (similar to VS Code):

### A. Activity Bar (Far Left)
Vertical icon strip for switching primary views:
1.  **Agents (Explorer):** Tree view of active agents, fleets, and deployed bots.
2.  **Assets (Knowledge):** Access to the Assets Hub (Skills, Templates, Articles).
3.  **Code (Editor):** File explorer for strategy development.
4.  **Backtests (Lab):** History of backtest runs and results.
5.  **Settings:** Global configuration.

### B. Side Bar (Left Panel)
Collapsible panel displaying content based on the Activity Bar selection.
*   *Example:* When "Agents" is selected, shows a tree:
    *   research-guild
    *   engineering-guild
    *   *User Bots*
        *   EURUSD_Scalper_v1 (`@perfect`)
        *   BTC_Trend_v2 (`@primal`)

### C. Main Editor Area (Center)
The core workspace. Supports tabs and split panes.
*   **Code Editor:** Monaco Editor (VS Code core) implementation.
    *   Syntax highlighting for Python and MQL5.
    *   "Run Backtest" CodeLens action above class definitions.
    *   IntelliSense powered by the Assets Hub language server.
*   **Chart View:** TradingView Lightweight Charts or specialized canvas chart.
    *   Displays Entry/Exit points from backtests.
    *   Real-time indicators.

### D. Primary Side Bar (Right - Collapsible)
Dedicated home for **QuantMind Co-pilot**.
*   **Chat Interface:** Persistent chat with context awareness of the open file/agent.
*   **Agentic Skills Palette:** Quick-access buttons for common skills (e.g., "Analyze this Strategy", "Optimize Parameters").

### E. Panel (Bottom)
Multi-tabbed output area:
1.  **Terminal:** System logs and direct CLI access to containers.
2.  **Account:** Real-time balance, equity, margin (connected to MT5).
3.  **Output:** Build logs from the Quant Code Agent.
4.  **Problems:** Linter errors and backtest warnings.

---

## 2. Component Stack

| UI Component | Library Choice |
| :--- | :--- |
| **Framework** | Next.js 14 (App Router) |
| **Styling** | Tailwind CSS + `clsx` |
| **Layout Manager** | `react-resizable-panels` (or `mosaic` for complex grids) |
| **Code Editor** | `@monaco-editor/react` |
| **Charts** | `lightweight-charts` (TradingView) |
| **Icons** | `lucide-react` (VS Code style icons) |
| **Tables** | `@tanstack/react-table` |

---

## 3. Key User Flows

### Flow 1: "Code & Verify"
1.  User opens `MyNewStrategy.py` in the Editor.
2.  User types logic. Co-pilot (Right Panel) suggests snippets from Assets Hub.
3.  User clicks "Run Backtest" (CodeLens).
4.  Bottom Panel switches to "Output" to show progress.
5.  Main Area splits: Top = Code, Bottom = Equity Curve Chart.

### Flow 2: "Agent Observation"
1.  User clicks "Agents" in Activity Bar.
2.  Selects `EURUSD_Scalper_v1`.
3.  Main Area opens "Agent Dashboard" tab.
    *   Live Price Chart with trade markers.
    *   "Heartbeat" log streaming in Bottom Panel.
    *   "Kill Switch" button in top toolbar.

---

## 4. Implementation Priority

**Phase 1: The Shell**
*   Set up Next.js project.
*   Implement `react-resizable-panels` layout.
*   Create the VS Code-like chrome (Activity Bar, Status Bar).

**Phase 2: The Editor**
*   Integrate Monaco Editor.
*   Connect File System (via simple API to read `src/` or `data/`).

**Phase 3: The Assets**
*   Connect Left Panel to `quantmindx-kb` MCP server (list skills/templates).
