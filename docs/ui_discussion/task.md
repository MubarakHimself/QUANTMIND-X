# QuantMind X - Implementation Tasks

## Phase 1: UI Foundation & Library
- [ ] **Refactor Shell/Sidebar**
    - [ ] Create Top Bar (Balance, Challenges).
    - [ ] Remove "File Edit", Merge "Assets/Knowledge" into "Library".
- [ ] **Implement Library View**
    - [ ] **Knowledge Tab**: Markdown Reader, Syntax Highlight, "Add to Chat" actions.
    - [ ] **Assets Tab**: EA List, Status/Performance Columns.
    - [ ] **Drag & Drop**: Logic for moving articles to Agent KBs.

## Phase 2: The Director (Chat Interface)
- [ ] **Enhanced Copilot Panel**
    - [ ] Add Slash Command Menu (`/`).
    - [ ] Add File Attachment UI.
    - [ ] Add "Agentic Skills" Selector.
- [ ] **Error Console**
    - [ ] Implement Bottom Panel ("Problems" view).
    - [ ] Connect to Log Stream (Redis/File).

## Phase 3: Backend & Data Structure
- [ ] **Define Data Packets ("QuantMind Manifest")**
    - [ ] Create Standard Folder Structure for EAs.
    - [ ] Implement `status.json` schema.
- [ ] **Refine Tagging System**
    - [ ] Rename "Bot Tags" to "EA Tags".
    - [ ] Implement "Archive" logic.
- [ ] **MT5 Integration (Python)**
    - [ ] Verify `MetaTrader5` package usage (No DLL).
    - [ ] Create Sync Service for Account Balance.

## Phase 4: Visualization & Prop Firms
- [ ] **Prop Firm Dashboard**
    - [ ] T&C Viewer.
    - [ ] P&L vs Drawdown Progress Bars.
- [ ] **Strategy/Risk Viz**
    - [ ] React Flow / Mermaid implementation for Strategy Router.
    - [ ] Visualizer for Kelly Criterion sizing.
