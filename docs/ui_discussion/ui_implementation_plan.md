# QuantMind X - UI Implementation Plan

## 🎯 Objective
Refactor the `quantmind-ide-desktop` to become a "Zero-to-Hero" Algorithmic Trading Factory, implementing the v3 User Stories.

## 🛠️ Phase 1: Core Layout & Navigation
**Goal:** Establish the "Cockpit" structure.

### 1.1 Top Bar (`src/components/layout/TopBar.tsx`)
- **Left:** Global Account Balance (Live Widget).
- **Center:** Command Palette Trigger (Ctrl+P style).
- **Right:** Active Prop Challenges Ticker.

### 1.2 Activity Bar (Sidebar)
- **Remove:** `FileEdit` (Legacy).
- **Add/Update:**
    - `Library` (Book Icon) -> Combines Knowledge & Assets.
    - `PropFirms` (Trophy Icon).
    - `NPRD` (Brain Icon).
    - `Explorer` (Code Icon) -> Existing FileTree.
- **Fix:** `FileTree` permission error (Ensure `tauri.conf` allowlist includes target dirs).

### 1.3 Agent Sidebar (Right Panel)
- **Refactor:** `CopilotPanel` -> `AgentDirectorPanel`.
- **Tabs/Toggle:**
    1.  **Analyst Agent** (TRD Creator).
    2.  **QuantCode Agent** (Coder).
    3.  **QuantMind Copilot** (Director/General).
- **Context Switch:** Clicking an agent changes the active Chat Context.

## 💻 Phase 2: Feature Components

### 2.1 The Library (`src/components/library/LibraryView.tsx`)
- **Structure:** Split Pane (List on Left, Content on Right).
- **Tabs:** "Knowledge" (Articles), "Assets" (EAs), "NPRD" (Storylines).
- **Markdown Renderer:**
    - Use `react-markdown` with `Prism` syntax highlighting.
    - Inject "Add to Chat" button over code blocks.
- **Drag & Drop:** Allow dragging an Article from List -> Agent Sidebar (to "seed" the context).

### 2.2 Rich Chat Interface (`src/components/chat/ChatInterface.tsx`)
- **Input Area:**
    - `/` trigger opens Command Menu (`/analyze`, `/backtest`).
    - Paperclip icon -> File Picker.
    - "Skills" Dropdown (e.g., "Interface Design").
- **Message List:** Support Markdown, Code Blocks, and "Tool Call" visualization.

### 2.3 Prop Firm Manager (`src/components/propfirm/PropFirmView.tsx`)
- **Dashboard:** Cards for each active challenge.
- **Progress Bars:** Drawdown vs Limit (Red/Green zones).
- **Scraper Input:** URL Input -> Triggers "Prop Firm Node" creation agent.

### 2.4 Error Console (`src/components/console/ProblemsPanel.tsx`)
- Bottom panel (collapsible).
- Tabs: "Terminal", "Output", "Problems".
- Stream logs via WebSocket or File Watcher.

## 🔗 Phase 3: Data & State Integration
**Goal:** Connect UI to Reality.

### 3.1 Store (`src/store/index.ts`)
- Use **Zustand** for lightweight global state.
- Implement the `AppState` defined in `ui_data_models.md`.

### 3.2 APIs (`src/services/`)
- `mt5Api.ts`: Python Bridge for Balance/P&L.
- `fileSystem.ts`: Wrapper around Tauri FS for reading "Manifests".

## 🎨 Design System
- **Theme:** "Precision & Density" (VS Code Dark Modern).
- **Colors:** `#1e1e1e` (Bg), `#007acc` (Accent), `#cccccc` (Text).
- **Typography:** Monospace for data; Sans-serif for UI. 13px base size.
