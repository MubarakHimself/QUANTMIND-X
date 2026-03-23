---
title: 'Epic 12 — UI Refactor: Shell Agent Panel, Design Tokens, Tile Grid, Trading Canvas, Portfolio Nav'
slug: 'epic-12-ui-refactor'
created: '2026-03-22'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack:
  - 'SvelteKit 2 + Svelte 5 (runes)'
  - 'TypeScript strict'
  - 'Tauri 2 (desktop shell)'
  - 'lucide-svelte (icons — no emoji)'
  - 'CSS custom properties (Frosted Terminal token system)'
  - 'Vite 5 (dev port 1420, static adapter)'
files_to_modify:
  - 'quantmind-ide/src/routes/+page.svelte'
  - 'quantmind-ide/src/app.css'
  - 'quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte'
  - 'quantmind-ide/src/lib/components/canvas/PortfolioCanvas.svelte'
  - 'quantmind-ide/src/lib/components/canvas/*.svelte (all 9)'
code_patterns:
  - 'Svelte 5 runes ($state, $derived, $props, $effect)'
  - 'Lucide icons only — named imports'
  - 'apiFetch<T>() wrapper for all API calls'
  - 'CSS custom property token system'
  - 'No SSR — static adapter, all data in onMount'
test_patterns:
  - 'Vitest + @testing-library/svelte'
  - 'Co-located test files (ComponentName.test.ts)'
---

# Tech-Spec: Epic 12 — UI Refactor

**Created:** 2026-03-22

---

## Overview

### Problem Statement

The current shell layout is a VS Code-style IDE with a `BottomPanel` (terminal/logs) at the bottom and no right-rail Agent Panel in the global grid. The UX spec mandates a "department canvas" model where each canvas has a collapsible **Agent Panel** (right rail, department-head-specific conversation) permanently in the shell — not a floating bubble and not hidden inside canvases individually.

Additionally: (1) the CSS token system has two competing parallel systems in `app.css` (old OKLCH vars + Frosted Terminal hex vars) with mismatched values across files; (2) the tile grid pattern is inconsistently applied across canvases — some are placeholders, some use legacy patterns; (3) `TradingCanvas.svelte` (canvas slot 5 — Paper Trading & Backtesting) is an empty placeholder pointing to the wrong epic; (4) Portfolio canvas and keyboard-based canvas navigation have routing alignment issues.

This epic brings the global shell into full spec compliance before the functional epics (5–11) add real content into it.

### Solution

Five targeted stories executed in dependency order:
1. **12-1** — Refactor `+page.svelte` grid to add Agent Panel right rail (320px collapsible) and remove BottomPanel. Wire up new `AgentPanel.svelte` (in `shell/`) with full chat management scaffold and dept-head-per-canvas wiring.
2. **12-2** — Consolidate `app.css` to a single Frosted Terminal token system. Remove competing OKLCH vars. Align hex values to UX spec + `ux-design-directions.html` reference. Apply "Balanced Terminal" (DIR-05) as default density config.
3. **12-3** — Extract `CanvasTileGrid.svelte` as a shared canvas layout wrapper. Migrate all 9 canvases to use it (placeholder-breaking ones get real skeleton tiles; live ones adopt the wrapper).
4. **12-4** — Replace `TradingCanvas.svelte` placeholder with a real tile grid covering paper trading monitoring and backtesting summary tiles (backend endpoints confirmed in `src/api/backtest_endpoints.py` and paper-trading data).
5. **12-5** — Fix Portfolio canvas breadcrumb navigation, align `canvasStore` keyboard shortcut wiring, fix legacy `activeView` prop threading from `MainContent` leaking into new canvas components.

### Scope

**In Scope:**
- Global shell grid restructure (`+page.svelte`)
- New `AgentPanel.svelte` in `components/shell/` (right rail, 320px, collapsible, dept-head-per-canvas, full chat management scaffold)
- `BottomPanel.svelte` removal from shell (file retained, not deleted — referenced for terminal content if ever needed in canvas sub-pages)
- Design token consolidation in `app.css` — Frosted Terminal canonical values
- `CanvasTileGrid.svelte` + `CanvasTile.svelte` shared components
- Migrate all 9 existing canvas `.svelte` files to use the shared tile grid wrapper
- `TradingCanvas.svelte` functional tile grid (paper trading + backtest tiles, real API wiring)
- `PortfolioCanvas.svelte` breadcrumb navigation fix
- Keyboard shortcuts (1–9) → canvas routing alignment
- `StatusBand.svelte` clickable segments (already partially built — verify all 5 click targets work)
- `ux-design-directions.html` referenced as visual design authority for all component styling

**Out of Scope:**
- Functional implementation of department heads (Epic 7)
- Agent Panel actual AI streaming (Epic 5)
- Full Trading canvas content beyond tile skeletons (Epic 7/8)
- Backtest viewer sub-page (Epic 8)
- Full Portfolio canvas content (Epic 9)
- Monaco editor integration (Epic 8)
- FlowForge canvas implementation (Epic 11)

---

## Context for Development

### Codebase Patterns

**Frontend stack:**
- Svelte 5 runes — all new components use `$state`, `$derived`, `$props`. No `let x = writable()`.
- Lucide icons: `import { PanelRight, X, MessageSquare } from 'lucide-svelte'`
- API calls: `import { apiFetch } from '$lib/api'` → `apiFetch<T>('/api/...')`
- Store imports: `import { someStore } from '$lib/stores'` (barrel at `$lib/stores/index.ts`)
- `onMount` for data fetching, `onDestroy` for cleanup — mandatory
- Static adapter: no `+page.server.ts`, no SSR

**Current shell grid (`+page.svelte` line 99–114):**
```css
.ide-layout {
  display: grid;
  grid-template-areas:
    "topbar topbar"
    "statusband statusband"
    "activity main"
    "activity bottom";
  grid-template-columns: var(--sidebar-width) 1fr;
  grid-template-rows: var(--header-height) auto 1fr auto;
  height: 100vh;
}
```

**Target shell grid (after 12-1):**
```css
.ide-layout {
  display: grid;
  grid-template-areas:
    "topbar topbar topbar"
    "statusband statusband statusband"
    "activity main agent";
  grid-template-columns: var(--sidebar-width) 1fr var(--agent-panel-width, 320px);
  grid-template-rows: var(--header-height) auto 1fr;
  height: 100vh;
}
/* Collapsed state: */
.ide-layout.agent-collapsed {
  grid-template-columns: var(--sidebar-width) 1fr 0px;
}
```

**Current `app.css` token conflict (lines 4–81):**
- Old OKLCH system: `--bg-primary: oklch(15% 0.02 260)`, `--accent-primary: oklch(65% 0.18 250)` — these are used in legacy components
- Frosted Terminal hex: `--color-bg-primary: #080d14`, `--color-accent-cyan: #00d4ff`, `--color-accent-amber: #f0a500` — these are the canonical UX spec values
- Design directions HTML uses: `--c-amber: #d4920e` (different from `#f0a500`), `--c-cyan: #00aacc` (different from `#00d4ff`)
- **Resolution (Story 12-2):** Trust `app.css` Frosted Terminal hex values as canonical — they match the UX spec exactly. The design directions HTML is a visual prototype with slight adjustments; the authoritative spec values are in UX spec + `app.css` current Frosted Terminal block. OKLCH tokens are removed and replaced with Frosted Terminal hex equivalents so components have a single token to reference.

**Agent Panel (reference: `ux-design-directions.html` lines 201–216):**
The HTML design directions shows the exact expected visual structure:
- Header (36px): dept badge + spacer + `[+]` `[⏱]` icon buttons
- Body: scrollable message area with agent messages, user messages, tool-use lines, suggestion chips
- Footer: input field
- Panel width: 320px (open), 0px (collapsed) — confirmed from UX spec §Global Shell Dimensions
- The deprecated `agent-panel/AgentPanel.svelte` is NOT used — it wired to old copilot/analyst/quantcode agent types. A new `AgentPanel.svelte` in `components/shell/` is built from scratch.

**Canvas context per dept head (from `canvasContextService.ts`):**
- `canvasContextService.loadCanvasContext(canvasId)` exists and is already called in `LiveTradingCanvas.svelte`
- Each canvas maps to a dept head: live-trading→(no dept head, Trading dept), research→Research Head, development→Development Head, risk→Risk Head, trading→Trading Head, portfolio→Portfolio Head, workshop→FloorManager, flowforge→FloorManager, shared-assets→(any)
- The new `AgentPanel.svelte` (shell/) receives `activeCanvas` prop and resolves the dept head label + API endpoint from a static map. Two session types visible: Interactive (chat input active) and Autonomous Workflow (status card only, no chat input).

**TradingCanvas current state (`quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte`):**
- Line 1–9: just renders `<CanvasPlaceholder canvasName="Trading" epicNumber={3} epicName="Live Trading Command Center" />`
- Wrong epic reference: Epic 3 is **Live Trading**, not the Trading Department (paper trading/backtesting) canvas
- Trading canvas (slot 5) is Epic 7+8 domain — paper trading monitoring, backtest results, strategy lifecycle
- Backend endpoints available: `src/api/backtest_endpoints.py` (`/api/backtest/*`), paper trading data via `/api/paper-trading/*` (see `src/api/trading/routes.py`)

**Portfolio canvas routing issue:**
- `PortfolioCanvas.svelte` exists but uses old `navigationStore` patterns mixed with new `activeCanvasStore`
- `canvasStore.ts` has `CANVASES` array with keyboard shortcuts wired (Story 1-6 work), but `ActivityBar.svelte` dispatches `viewChange` event which `+page.svelte` handles via `activeView` state — this conflicts with the new canvas routing system built in Story 1-6
- Fix: `+page.svelte` should derive active canvas from `activeCanvasStore` directly, not from `activeView` prop

**Balanced Terminal (DIR-05) from `ux-design-directions.html`:**
- Glass: heavy (`--glass-opacity: 0.08`, `--glass-content-opacity: 0.35`, `--glass-blur-radius: 20px`)
- StatusBand density: dense (8 segment items visible)
- Tile density: spacious (`--tile-min-width: 260px`, `--tile-gap: 20px`)
- This is Mubarak's preferred default — confirmed in memory file `project_design_direction_prefs.md`

### Files to Reference

| File | Purpose |
|------|---------|
| `quantmind-ide/src/routes/+page.svelte` | Global shell grid — modify for agent panel column |
| `quantmind-ide/src/app.css` | Token system — consolidate |
| `quantmind-ide/src/lib/components/TopBar.svelte` | TopBar (wired) |
| `quantmind-ide/src/lib/components/ActivityBar.svelte` | Canvas navigation dispatches |
| `quantmind-ide/src/lib/components/StatusBand.svelte` | Ambient status bar |
| `quantmind-ide/src/lib/components/BottomPanel.svelte` | Removed from shell (retained as file) |
| `quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte` | Replace placeholder |
| `quantmind-ide/src/lib/components/canvas/PortfolioCanvas.svelte` | Fix nav |
| `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte` | Reference for tile grid pattern |
| `quantmind-ide/src/lib/components/live-trading/GlassTile.svelte` | Existing glass tile (retained for live-trading; new canvases use `shared/TileCard.svelte`) |
| `quantmind-ide/src/lib/stores/canvas.ts` | Canvas store (CANVASES array, keyboard shortcuts) |
| `quantmind-ide/src/lib/services/canvasContextService.ts` | Context loading per canvas |
| `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte` | DEPRECATED — do NOT reuse |
| `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` | Reference for streaming pattern |
| `_bmad-output/planning-artifacts/ux-design-directions.html` | Visual authority — agent panel CSS, tile CSS, shell CSS |
| `_bmad-output/planning-artifacts/ux-design-specification.md` | Shell anatomy (lines 74–99), canvas structure |
| `_bmad-output/planning-artifacts/prd.md` | User journeys (52 total) |

### Technical Decisions

1. **Agent Panel lives in global shell, not per-canvas** — One `ShellAgentPanel.svelte` in `+page.svelte` grid, receives `activeCanvas` prop. Each canvas does NOT have its own agent panel DOM element. This matches the UX spec shell anatomy.

2. **New `AgentPanel.svelte` in `components/shell/` is NOT the deprecated `agent-panel/AgentPanel.svelte`** — Build new from scratch in `shell/`. The deprecated one wired to old agent types. New one is a UI scaffold with full chat management (new chat, session history, collapse), two session type views (interactive chat, autonomous status card), and dept-head label resolution. Actual AI streaming (SSE from `GET /api/agents/stream`) wired in Epic 5.

3. **BottomPanel is removed from shell only, file not deleted** — The BottomPanel has terminal UI that may be useful in future canvas sub-pages (e.g., Development canvas terminal). Remove from `+page.svelte` and grid; retain the `.svelte` file.

4. **Token consolidation strategy**: Keep Frosted Terminal hex vars (`--color-bg-primary`, `--color-accent-cyan`, etc.) as primary. Add missing semantic aliases. Remove competing OKLCH vars where they duplicate meaning. Where OKLCH vars are currently used in components, update those components to reference Frosted Terminal vars. Do NOT add an OKLCH↔hex bridge layer.

5. **`TileCard.svelte` is the canonical shared tile** — Architecture specifies `shared/TileCard.svelte` as the base glass tile. The existing `live-trading/GlassTile.svelte` stays for backward-compat; new canvas tiles use `TileCard`. A future cleanup will unify them (out of scope for Epic 12). `GlassSurface.svelte` is the shared backdrop-filter + scanline wrapper for non-tile glass surfaces.

6. **`CanvasTileGrid.svelte` layout wrapper** — Handles the grid CSS, canvas header (title + subtitle), and breadcrumb back-button state. Accepts `tiles` data slot or `{@render children}`. Canvases pass their tiles as children.

7. **Trading canvas (slot 5) tile content** — Four tiles for MVP: `PaperTradingStatusTile`, `BacktestResultsTile`, `StrategyPerformanceTile`, `StrategyLifecycleTile`. Each fetches real data from existing endpoints. Shows skeleton loaders when loading.

8. **Portfolio nav fix** — `PortfolioCanvas.svelte` replaces `navigationStore.navigateToView()` calls with `activeCanvasStore`-based sub-page state (`let subPage = $state('grid')`). Breadcrumb uses `subPage = 'grid'` to return.

---

## Implementation Plan

### Story Order (dependency-respecting)

```
12-1 (Shell Grid + Agent Panel) → 12-2 (Tokens) → 12-3 (Tile Grid + Shared Components) → 12-4 (Trading Canvas) + 12-6 (Dept Kanban) → 12-5 (Nav Fixes)
```

12-2 must come before 12-3 because 12-3 uses token vars. 12-4 and 12-6 can run in parallel after 12-3. 12-5 must come last (depends on shell + tile grid being in place).

---

### Story 12-1: Global Shell — Agent Panel Right Rail + Remove BottomPanel

**PRD Journey Mapping:**
- **Journey 1** (Morning Operator) — The Agent Panel is how Mubarak sees the Research dept notification without navigating away from Live Trading
- **Journey 11** (Overnight Research Cycle) — "8am: Research completed overnight cycle" notification surfaces in the Agent Panel right rail without leaving the current canvas
- **Journey 12** (Parameter Sweep) — Copilot notification "512 combinations tested" visible in Agent Panel
- **Journey 16** (Code Review) — Mubarak types feedback in Agent Panel on Development canvas → dept mail loop
- **Journey 17** (Physics Lens) — Risk canvas Agent Panel has Risk Dept Head loaded, can ask "what does the physics engine say?"

**FR Enablement:** FR20 (canvas-aware Copilot context), FR22 (direct Department Head chat — UI scaffolding)

#### Tasks

**Task 12-1-A: Remove BottomPanel from global shell**
- File: `quantmind-ide/src/routes/+page.svelte`
- Remove import: `import BottomPanel from "$lib/components/BottomPanel.svelte";`
- Remove: `<BottomPanel />` from template
- Update grid CSS (`.ide-layout`):
  - `grid-template-areas`: change to 3 columns (`"topbar topbar topbar"`, `"statusband statusband statusband"`, `"activity main agent"`)
  - `grid-template-columns`: `var(--sidebar-width) 1fr var(--agent-panel-width, 280px)`
  - `grid-template-rows`: `var(--header-height) auto 1fr` (remove `auto` bottom row)
- Add CSS var to `:root` in `app.css`: `--agent-panel-width: 320px;`
- Add collapsed state class: `.ide-layout.agent-panel-collapsed { --agent-panel-width: 0px; overflow: hidden; }`

**Task 12-1-B: Create `AgentPanel.svelte` in `components/shell/`**
- File: `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` (NEW — canonical location per architecture)
- This is NOT the deprecated `agent-panel/AgentPanel.svelte` — build fresh
- Props: `activeCanvas: string` (the current canvas ID)
- State:
  ```typescript
  let collapsed = $state(false);
  let sessions = $state<AgentSession[]>([]);  // list of chat sessions
  let activeSessionId = $state<string | null>(null);
  let messages = $derived(sessions.find(s => s.id === activeSessionId)?.messages ?? []);
  let inputValue = $state('');
  let showSessionHistory = $state(false);
  ```
- Session types: `interactive` (chat input visible) and `autonomous` (status card only, read-only)
- Chat management scaffold:
  - `[+]` button → creates new interactive session, sets `activeSessionId`
  - `[⏱]` button → toggles `showSessionHistory` (list of past sessions)
  - `[←]` button → sets `collapsed = true`
  - Session history panel: scrollable list of past sessions per dept head
  - Active autonomous workflow sessions shown as read-only status cards above the input
- Dept head map (static const inside component):
  ```typescript
  const CANVAS_DEPT_HEAD: Record<string, { label: string; color: 'cyan' | 'amber' | 'red' | 'green' | 'muted' }> = {
    'live-trading': { label: 'TRADING', color: 'green' },
    'research': { label: 'RESEARCH', color: 'amber' },
    'development': { label: 'DEVELOPMENT', color: 'cyan' },
    'risk': { label: 'RISK', color: 'red' },
    'trading': { label: 'TRADING', color: 'green' },
    'portfolio': { label: 'PORTFOLIO', color: 'cyan' },
    'shared-assets': { label: 'SHARED', color: 'muted' },
    'workshop': { label: 'FLOOR MGR', color: 'cyan' },
    'flowforge': { label: 'FLOOR MGR', color: 'cyan' },
  };
  ```
- Structure (matches `ux-design-directions.html` lines 201–216 + UX spec §7 AgentPanel):
  - `.ap-header` (36px): dept badge (`--color-accent-*` tinted), spacer, `[+]` (Plus icon), `[⏱]` (History icon), `[←]` (ChevronRight/collapse) button
  - `.ap-session-history` (conditional): scrollable list of past sessions — title, dept, date, status badge
  - `.ap-autonomous-status` (conditional, if autonomous sessions active): read-only workflow status cards
  - `.ap-body`: scrollable message area with agent messages, user messages, tool-use lines, suggestion chips
  - `.ap-footer`: input field + send button (only visible when an interactive session is active)
  - Empty state: "Ask the {deptHead} head anything about this canvas" + suggestion chip row (epic-7 wires real chips; here use placeholder chips)
- Collapse toggle: sets `collapsed = true`, parent reads via `bind:collapsed`
- NOTE: No AI wiring in this story — Epic 5 wires SSE to `GET /api/agents/stream`. Just UI scaffold + local echo.
- Width: 320px (non-collapsed). Transition: `transition: width 300ms ease`.
- Visual reference: `ux-design-directions.html` `.agent-panel`, `.ap-*` CSS classes (lines 201–216)

**Task 12-1-C: Wire `AgentPanel` into `+page.svelte`**
- Import `AgentPanel` from `$lib/components/shell/AgentPanel.svelte` in `+page.svelte`
- Import `activeCanvasStore` from `$lib/stores/canvas`
- Derive current canvas: `let currentCanvas = $derived($activeCanvasStore.id)`
- Add to template: `<AgentPanel activeCanvas={currentCanvas} />`
- Bind collapsed state:
  ```svelte
  let agentPanelCollapsed = $state(false);
  // In template:
  <div class="ide-layout" class:agent-panel-collapsed={agentPanelCollapsed}>
    ...
    <AgentPanel activeCanvas={currentCanvas} bind:collapsed={agentPanelCollapsed} />
  </div>
  ```
- Add `grid-area: agent;` style to AgentPanel's root element

**Task 12-1-D: Keyboard shortcut preservation**
- Ensure existing shortcuts (1–9 for canvas switching) still work after grid change
- Test: keyboard event listener is in `ActivityBar.svelte` — confirm it hasn't changed and keys 1–9 still switch canvases correctly

#### Acceptance Criteria

- **Given** the app is open, **when** the user looks at the shell, **then** a right-rail Agent Panel is visible (320px wide), docked between the canvas workspace and the right edge
- **Given** the Agent Panel is visible, **when** the user is on the Research canvas, **then** the Agent Panel header shows a "RESEARCH" dept badge (amber tint)
- **Given** the Agent Panel is visible, **when** the user is on the Risk canvas, **then** the header shows "RISK" (red tint)
- **Given** the Agent Panel collapse button is clicked, **when** the collapse animation completes (300ms), **then** the Agent Panel width is 0 and the canvas workspace fills the full remaining width
- **Given** the `[+]` button is clicked, **when** the click fires, **then** a new interactive session is created and the chat input area becomes active
- **Given** the `[⏱]` (history) button is clicked, **when** the click fires, **then** a session history panel slides open showing past sessions for this dept head
- **Given** the Agent Panel is open, **when** the user types a message and submits, **then** the message appears as a user bubble in the conversation area (local echo only — Epic 5 wires SSE)
- **Given** the BottomPanel previously existed at the bottom, **when** the app loads, **then** no BottomPanel is rendered, the full vertical space from StatusBand to bottom is used by the canvas + agent panel
- **Given** keyboard shortcuts 1–9 are pressed, **when** each shortcut fires, **then** the correct canvas loads and the Agent Panel dept badge updates accordingly

---

### Story 12-2: Design Token Consistency Pass

**PRD Journey Mapping:**
- **Journey 7** (New User Setup) — "Picked Frosted Terminal theme" — the token system must be coherent so theme application is consistent
- **Journey 1** (Morning Operator) — "StatusBand pulses: London session OPENING" — the amber/green/cyan must read correctly at a glance. Token mismatches cause visual noise.
- **All journeys** — The design token system is the foundation of every visual journey. Token consistency is a precondition for visual fidelity on every canvas.

**FR Enablement:** NFR-M4 (component cleanliness), foundational for all visual FRs

#### Tasks

**Task 12-2-A: Audit and consolidate `app.css`**
- File: `quantmind-ide/src/app.css`
- Remove old OKLCH vars that duplicate Frosted Terminal hex equivalents:
  - Remove `--bg-primary`, `--bg-secondary`, `--bg-tertiary`, `--bg-input`, `--bg-glass` (OKLCH based)
  - Remove `--accent-primary`, `--accent-secondary`, `--accent-finance`, `--accent-success`, `--accent-warning`, `--accent-danger` (OKLCH based)
  - Remove `--text-primary`, `--text-secondary`, `--text-muted`, `--text-accent` (OKLCH based)
  - Remove `--border-subtle`, `--border-medium`, `--border-strong`, `--border-accent` (OKLCH based)
  - Remove `--syntax-*` vars (these can be re-added when Monaco is integrated in Epic 8)
  - KEEP dimension vars (`--sidebar-width`, `--panel-width`, `--header-height`, `--status-height`)
  - KEEP font vars (`--font-family`, `--font-display`, `--font-nav`, `--font-mono`)
- Keep and EXPAND Frosted Terminal block — use UX spec canonical values:
  ```css
  :root {
    /* === FROSTED TERMINAL CANONICAL TOKEN SYSTEM (UX Spec Authority) === */

    /* Color — Background */
    --color-bg-base:         #080d14;                  /* Root background — deep space blue-black */
    --color-bg-surface:      rgba(8, 13, 20, 0.6);     /* Frosted glass panel surface */
    --color-bg-elevated:     rgba(16, 24, 36, 0.8);    /* Elevated card / modal surface */
    --color-border-subtle:   rgba(255, 255, 255, 0.06); /* Panel borders */
    --color-border-medium:   rgba(255, 255, 255, 0.09);

    /* Color — Accents (semantic) */
    --color-accent-cyan:     #00d4ff;   /* AI / Copilot / agent */
    --color-accent-amber:    #f0a500;   /* Live / active / running / kill switch armed */
    --color-accent-red:      #ff3b3b;   /* Kill / danger / alert */
    --color-accent-green:    #00c896;   /* Profit / success / workflow complete */

    /* Color — Text */
    --color-text-primary:    #e8edf5;   /* Primary readable text */
    --color-text-muted:      #5a6a80;   /* Secondary labels, ambient data */

    /* Glass System */
    --blur-glass:            blur(12px);
    --glass-shell-bg:        rgba(8, 13, 20, 0.08);    /* Shell tier — near-transparent */
    --glass-content-bg:      rgba(8, 13, 20, 0.35);    /* Content tier — heavier */

    /* Dimensions — Global Shell (fixed, non-negotiable) */
    --sidebar-width:         48px;
    --header-height:         40px;
    --statusband-height:     30px;
    --agent-panel-width:     320px;   /* UX spec §Global Shell Dimensions */

    /* Spacing (4px base unit) */
    --space-1:   4px;
    --space-2:   8px;
    --space-3:   12px;
    --space-4:   16px;
    --space-5:   20px;
    --space-6:   24px;
    --space-8:   32px;
    --space-10:  40px;
    --space-12:  48px;

    /* Typography */
    --font-data:    'JetBrains Mono', monospace;         /* All financial figures, P&L, risk, timestamps */
    --font-heading: 'Syne', sans-serif;                  /* Canvas titles, section headers */
    --font-body:    'Space Grotesk', 'IBM Plex Sans', sans-serif; /* Labels, body, agent prose */
    --font-ambient: 'Fragment Mono', 'Geist Mono', monospace;     /* StatusBand ticker, ambient data */

    /* Type scale */
    --text-xs:   11px;   /* Ambient, ticker, StatusBand */
    --text-sm:   12px;   /* Secondary labels, badges */
    --text-base: 13px;   /* Body default */
    --text-md:   14px;   /* Card content */
    --text-lg:   16px;   /* Sub-section heading */
    --text-xl:   20px;   /* Canvas title */
    --text-2xl:  24px;   /* Hero numbers (P&L hero tiles) */

    /* Tile Density — Dense (Balanced Terminal / default) — per UX spec responsive table */
    --tile-min-width:     280px;  /* Balanced Terminal = Dense tier */
    --tile-gap:           12px;   /* = var(--space-3) */
    --tile-padding:       var(--space-5);  /* 20px */
    --tile-border-radius: 6px;

    /* Wallpaper */
    --wallpaper-visible: 1;
  }
  ```
- Keep and update department accent variants:
  ```css
  /* Department accent overrides (applied per canvas via data-dept attribute) */
  [data-dept="live-trading"] { --dept-accent: var(--color-accent-green); }
  [data-dept="research"]     { --dept-accent: var(--color-accent-amber); }
  [data-dept="development"]  { --dept-accent: var(--color-accent-cyan); }
  [data-dept="risk"]         { --dept-accent: var(--color-accent-red); }
  [data-dept="trading"]      { --dept-accent: var(--color-accent-green); }
  [data-dept="portfolio"]    { --dept-accent: var(--color-accent-cyan); }
  [data-dept="workshop"]     { --dept-accent: var(--color-accent-cyan); }
  [data-dept="flowforge"]    { --dept-accent: var(--color-accent-cyan); }
  ```
- Add theme preset overrides for all 4 supported themes:
  ```css
  /* Theme: Ghost Panel (Kanagawa — Standard density) */
  [data-theme="ghost-panel"] {
    --tile-min-width: 320px;
    --tile-gap: 16px;
    --blur-glass: blur(16px);
  }
  /* Theme: Open Air (Tokyo Night — Spacious density) */
  [data-theme="open-air"] {
    --tile-min-width: 360px;
    --tile-gap: 20px;
    --blur-glass: blur(24px);
  }
  /* Theme: Breathing Space (Catppuccin Mocha) */
  [data-theme="breathing-space"] {
    --tile-min-width: 400px;
    --tile-gap: 24px;
    --blur-glass: blur(8px);
  }
  /* Default (Balanced Terminal / Frosted Terminal) = :root values above */
  ```

**Task 12-2-B: Update components using removed OKLCH vars**
- Search for any `.svelte` file referencing `var(--bg-primary)`, `var(--accent-primary)`, `var(--text-primary)`, etc.
- Run: `grep -r "var(--bg-primary\|var(--accent-primary\|var(--text-primary\|var(--border-subtle" quantmind-ide/src --include="*.svelte"`
- For each found usage, replace with Frosted Terminal equivalent:
  - `var(--bg-primary)` → `var(--color-bg-base)`
  - `var(--bg-secondary)` → `var(--color-bg-surface)`
  - `var(--bg-tertiary)` → `var(--color-bg-elevated)`
  - `var(--text-primary)` → `var(--color-text-primary)`
  - `var(--text-muted)` → `var(--color-text-muted)`
  - `var(--border-subtle)` → `var(--color-border-subtle)`
  - `var(--accent-primary)` → `var(--color-accent-cyan)` (default), context-specific
  - `var(--accent-danger)` / `var(--color-danger)` → `var(--color-accent-red)` (architecture canonical name)

**Task 12-2-C: Verify theme presets still work**
- `app.css` currently has `[data-theme="ghost-panel"]` and `[data-theme="open-air"]` blocks that override OKLCH vars — update these to override Frosted Terminal vars instead
- Ensure `AppearancePanel.svelte` (in `settings/`) still applies themes correctly after token rename

#### Acceptance Criteria

- **Given** the app loads with no theme override, **when** any canvas is visible, **then** the background is `#080d14` (deep space blue-black), cyan accents are `#00d4ff`, amber accents are `#f0a500`, green accents are `#00c896`, text is `#e8edf5`
- **Given** the Research canvas is active (data-dept="research"), **when** dept-accented elements render, **then** they use `--color-accent-amber` (#f0a500) as the dept accent
- **Given** the Risk canvas is active, **when** dept-accented elements render, **then** they use `--color-accent-red` (#ff3b3b)
- **Given** the theme is switched to "Ghost Panel" in Settings, **when** the theme applies, **then** `--tile-min-width` becomes 320px and `--tile-gap` becomes 16px
- **Given** the `app.css` is opened, **when** a developer searches for `oklch(`, **then** no results are found in the `:root` block (OKLCH vars replaced)
- **Given** all 9 canvas `.svelte` files are open, **when** they are inspected for `var(--bg-primary)` or similar old OKLCH vars, **then** none are found — all reference `--color-bg-*` or `--color-text-*` canonical token names
- **Given** the tile density tokens are set for Balanced Terminal default, **when** `--tile-min-width: 280px` and `--tile-gap: 12px` are applied, **then** the canvas tile grid renders at "dense" density (confirmed Balanced Terminal = Dense tier per UX spec)
- **Given** `app.css` is inspected, **when** a developer searches for `--color-danger`, **then** no results are found — the canonical token is `--color-accent-red`

---

### Story 12-3: Tile Grid Pattern — All 9 Canvases

**PRD Journey Mapping:**
- **Journey 1** (Morning Operator) — Live Trading canvas is home screen. The tile grid is Mubarak's morning glance view. "He doesn't have to click anything to see this." Every tile must load fast and be readable at a glance.
- **Journey 2** (Alpha Forge Trigger) — Workshop canvas tile grid shows morning digest + suggestion chips (the "opening view" before typing to Copilot)
- **Journey 6** (Portfolio Audit) — Portfolio canvas tile grid: monthly stats, allocation, correlation — all tiles on one screen before drilling into a sub-page
- **Journey 10** (Challenge Mode) — Risk canvas tile grid shows Challenge Mode progress, physics sensors, Kelly Engine state at a glance
- **Journey 12** (Parameter Sweep) — Development canvas tile grid shows EA variants, active pipeline stages, backtest queue
- **Journey 17** (Physics Lens) — Risk canvas tile grid surfaces Ising/HMM/Lyapunov sensor tiles immediately on canvas load
- **Journey 19** (Paper Trade Graduation) — Trading canvas (slot 5) tile grid shows paper trade monitoring tiles

**FR Enablement:** NFR-P4 (canvas transitions ≤200ms), foundational for FR1–FR9 (live trading display), FR20 (canvas-aware context), all canvas-level FRs

#### Tasks

#### Design Authority: CRM-Style Tile Pattern

From UX spec (§Micro-Emotions):
> "CRM-style canvas tiles: bounded cards, data in named sections, no overflow visible. The eye knows where to go. One click = sub-page depth. Nothing is lost — just organized beneath."

From UX spec (§Established Patterns):
> "Tile grid → sub-page → back (Amazon/CRM navigation)"

**What this means for `TileCard.svelte`:**
- Each tile is a **summary card** — it shows the most important 2–4 data points only. Never overflows. No "... and 12 more" within the tile itself.
- Data is in **named sections** — a header label (e.g., `"PAPER MONITORING"`) + key metrics with explicit field labels. No unlabeled numbers floating in space.
- The tile is always **read-ready** — even a first-time user should know what they're looking at from the label alone.
- Click = sub-page expansion. The full depth (tables, charts, history) lives on the sub-page. The tile is the teaser, not the full content.
- Empty states are neutral, not alarming — empty tile = "nothing yet" with a calm placeholder message.

**`TileCard.svelte` anatomy (CRM-pattern enforcement):**
```
┌─────────────────────────────────────────┐
│ SECTION LABEL (Fragment Mono, 10px caps) │  ← dept badge color
│                                          │
│  Primary metric   [value, JetBrains Mono]│  ← largest number
│  Secondary metric [value]                │  ← supporting data
│  Status indicator [dot + label]          │  ← state badge
│                                          │
│                          [→ view detail] │  ← only on hover
└─────────────────────────────────────────┘
```

**Task 12-3-A: Create `CanvasTileGrid.svelte`**
- File: `quantmind-ide/src/lib/components/shared/CanvasTileGrid.svelte` (NEW)
- Props:
  ```typescript
  interface Props {
    title: string;
    subtitle?: string;
    dept?: string; // for data-dept attribute on root
    showBackButton?: boolean;
    onBack?: () => void;
    children?: import('svelte').Snippet;
  }
  ```
- Structure:
  ```
  <div class="canvas-root" data-dept={dept}>
    <header class="canvas-header">
      {#if showBackButton}
        <button class="back-btn" onclick={onBack}><ArrowLeft /> Back</button>
        <span class="breadcrumb-sep">/</span>
      {/if}
      <h1 class="canvas-title">{title}</h1>
      {#if subtitle}<span class="canvas-sub">{subtitle}</span>{/if}
    </header>
    <div class="tile-grid">
      {@render children?.()}
    </div>
  </div>
  ```
- CSS:
  ```css
  .canvas-root { height: 100%; display: flex; flex-direction: column; overflow: hidden; }
  .canvas-header { padding: 14px 18px 8px; display: flex; align-items: baseline; gap: 10px; }
  .canvas-title { font-family: var(--font-display); font-weight: 800; font-size: 20px; color: var(--color-text-primary); }
  .canvas-sub { font-size: 10px; color: var(--color-text-muted); font-family: var(--font-data); }
  .tile-grid { flex: 1; overflow-y: auto; padding: var(--space-5); display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--tile-min-width, 280px), 1fr)); gap: var(--tile-gap, 12px); align-content: start; }
  .back-btn { display: flex; align-items: center; gap: 4px; font-family: var(--font-data); font-size: 10px; color: var(--color-text-muted); background: transparent; border: none; cursor: pointer; padding: 0; }
  .back-btn:hover { color: var(--color-text-primary); }
  ```

**Task 12-3-B: Create canonical `TileCard.svelte` in `shared/` and supporting shared components**
- Architecture canonical name: `shared/TileCard.svelte` (not `GlassTile.svelte`)
- Create `quantmind-ide/src/lib/components/shared/TileCard.svelte` — base glass tile (Frosted Terminal aesthetic)
  - Props: `title: string`, `value?: string`, `label?: string`, `size?: 'sm' | 'md' | 'lg' | 'xl'`, `span?: number`, `onclick?: () => void`, `loading?: boolean`, `dept?: string`
  - `size` maps to UX spec: sm=240px, md=320px (standard), lg=480px, xl=full-width
  - Glass CSS: `background: var(--color-bg-surface); backdrop-filter: var(--blur-glass);`
  - Hover: `border-color: rgba(255,255,255,0.13)`, subtle background lift
  - Financial values in `var(--font-data)`
- Also create:
  - `shared/GlassSurface.svelte` — backdrop-filter + scan-line `::before` pseudo-element wrapper
  - `shared/SkeletonLoader.svelte` — pulse animation using `--color-bg-elevated` (never white flash)
  - `shared/Breadcrumb.svelte` — tile → sub-page back navigation
- The existing `live-trading/GlassTile.svelte` is KEPT (live trading components reference it), but new canvases use `shared/TileCard.svelte`. The two will be unified in a future cleanup pass (out of scope here).

**Task 12-3-B2: Implement Workshop canvas left sidebar navigation**
- File: `quantmind-ide/src/lib/components/workshop/WorkshopCanvas.svelte`
- Implement the 3-column layout (left nav panel 200px + main conversation + no right rail)
- Left panel state:
  ```typescript
  type WorkshopView = 'new-chat' | 'history' | 'projects' | 'memory' | 'skills';
  let activeWorkshopView = $state<WorkshopView>('new-chat');
  let sessions = $state<WorkshopSession[]>([]);
  ```
- Left nav items (all Lucide icons):
  - `<Plus size={16} />` New Chat → `activeWorkshopView = 'new-chat'`
  - `<MessageSquare size={16} />` History → session list view
  - `<GitBranch size={16} />` Projects → link to FlowForge workflows (navigate to canvas 9)
  - `<Brain size={16} />` Memory → memory namespace browser (placeholder; Epic 6 fills)
  - `<Zap size={16} />` Skills → skill catalogue list (placeholder; Epic 7/9 fills)
- Active state: `class:active={activeWorkshopView === item.id}` — accent border on left edge
- Main area: renders `MorningDigest.svelte` on first load of day (check localStorage `lastOpenDate`), otherwise opens to centered FloorManager chat input
- The shell's right-rail AgentPanel should NOT render when Workshop is the active canvas — add `hidden={activeCanvas === 'workshop'}` to the AgentPanel in `+page.svelte`

**Task 12-3-B3: Settings view design refinement**
- File: `quantmind-ide/src/lib/components/SettingsView.svelte` (and sub-panels in `settings/`)
- Scope: design-only refinement — NOT a functional change. Just apply the Frosted Terminal token system consistently.
- Apply `CanvasTileGrid` header wrapper for the Settings canvas title
- Convert the Settings sections into TileCard-style panels: each settings group (Providers, Servers, Appearance, Notifications) gets its own TileCard
- The existing sub-panels (`ProvidersPanel`, `AppearancePanel`, `NotificationSettingsPanel`, `ServerHealthPanel`, `ServersPanel`) are preserved as-is; just the container layout and header are updated to match the tile grid visual language
- Theme switching in AppearancePanel stays as-is (it already works)

**Task 12-3-C: Migrate each canvas to use `CanvasTileGrid`**

For each of the 9 canvases, wrap existing content in `<CanvasTileGrid>`. Use `TileCard` (not the old `GlassTile`) for new tiles:

1. **`LiveTradingCanvas.svelte`** — Already has tile grid structure. Wrap in `<CanvasTileGrid title="Live Trading" dept="live-trading">`. Existing GlassTile instances left as-is (they use the live-trading path — unify later).

2. **`ResearchCanvas.svelte`** — `<CanvasTileGrid title="Research" dept="research">` + skeleton tiles per architecture: `AlphaForgeEntryTile` (YouTube URL input), `KnowledgeBaseTile`, `VideoIngestTile`, `HypothesisPipelineTile`.

3. **`DevelopmentCanvas.svelte`** — `<CanvasTileGrid title="Development" dept="development">` + skeleton tiles per architecture: `EALibraryTile`, `AlphaForgePipelineTile`, `BacktestQueueTile`.

4. **`RiskCanvas.svelte`** — `<CanvasTileGrid title="Risk" dept="risk">` + skeleton tiles per architecture: `KellyEngineTile`, `PhysicsSensorsTile` (shadow mode indicator), `PropFirmComplianceTile`, `ValidationQueueTile`.

5. **`TradingCanvas.svelte`** — Full implementation (Story 12-4).

6. **`PortfolioCanvas.svelte`** — `<CanvasTileGrid title="Portfolio" dept="portfolio">` + skeleton tiles per architecture: `LivePnLTile`, `AllocationTile`, `CorrelationMatrixTile`, `TradingJournalTile`. Nav fixes in Story 12-5.

7. **`SharedAssetsCanvas.svelte`** — `<CanvasTileGrid title="Shared Assets">` + skeleton tiles per architecture: `DocsLibraryTile`, `StrategyTemplatesTile`, `IndicatorsTile`, `SkillsTile`, `FlowComponentsTile`.

8. **`WorkshopCanvas.svelte`** — Workshop uses a 3-column layout (Claude.ai-inspired, architecture confirmed). NOT a tile grid:
   ```
   ┌──────────────┬─────────────────────────────┬─────────────────┐
   │  LEFT PANEL  │     MAIN CONVERSATION        │  (right rail =  │
   │  (200px)     │     (flex, centered)         │   AgentPanel    │
   │              │                              │   in shell)     │
   │  New Chat    │  ┌─────────────────────────┐ │                 │
   │  ─────────   │  │  Morning Digest         │ │                 │
   │  History     │  │  (first load of day)    │ │                 │
   │  ─────────   │  │  or "Good morning,      │ │                 │
   │  Projects    │  │   Mubarak" + centered   │ │                 │
   │  Memory      │  │   Copilot input         │ │                 │
   │  Skills      │  └─────────────────────────┘ │                 │
   └──────────────┴─────────────────────────────┴─────────────────┘
   ```
   - Left panel items (all Lucide icons, no emoji):
     - `[+] New Chat` — creates new FloorManager session
     - `History` — list of past sessions (scrollable, Fragment Mono timestamps)
     - `Projects` → links to active workflows in FlowForge
     - `Memory` → opens memory panel (graph memory namespace browser)
     - `Skills` → opens skill catalogue (list of available FloorManager skills)
   - Left panel is a navigation surface — clicking any item changes the main conversation area
   - The right-rail AgentPanel is NOT shown on Workshop canvas (Workshop IS the full-screen Copilot)
   - Use `CanvasTileGrid` for morning digest section header only; the rest is Workshop-specific layout.

9. **`FlowForgeCanvas.svelte`** — `CanvasTileGrid` header wrapper only. Content area = Prefect Kanban (`PrefectKanban.svelte`). Workflow Kill Switch lives in the Kanban row (per architecture — never a global button).

**Skeleton tiles for canvases still being implemented:**
- Tiles without real data use `SkeletonLoader` within a `TileCard`:
  ```svelte
  <TileCard title="Physics Sensors" dept="risk">
    <SkeletonLoader lines={3} />
    <span class="tile-badge">Epic 4</span>
  </TileCard>
  ```
- The `tile-badge` shows which epic owns the tile — gives Mubarak visibility into the build pipeline

#### Acceptance Criteria

- **Given** any of the 9 canvases is loaded, **when** the canvas renders at default view, **then** a tile grid is visible (no fullscreen blank placeholders, no raw "coming soon" text)
- **Given** the canvas title area is visible, **when** the user is on a non-home canvas, **then** the canvas name is displayed in Syne 800 font at 20px
- **Given** a tile is rendered, **when** it is hovered, **then** the border brightens (`rgba(255,255,255,0.13)`) and a subtle background lift occurs — matching `ux-design-directions.html` tile hover state
- **Given** the tile grid is rendered, **when** the viewport is at 1440px width with Balanced Terminal theme, **then** tiles render at minimum 280px wide with 12px gap (Dense/Balanced Terminal density)
- **Given** the Ghost Panel theme is applied, **when** the tile grid renders, **then** `--tile-min-width` is 320px and `--tile-gap` is 16px (Standard tier)
- **Given** a canvas with a sub-page open, **when** the user clicks the back button, **then** the tile grid view is restored and the back button disappears
- **Given** any of the 9 canvases is loaded, **when** a developer searches for financial number display, **then** all P&L, risk scores, and timestamps use `var(--font-data)` (JetBrains Mono)
- **Given** new tile components are inspected, **when** the import is checked, **then** they use `TileCard` from `$lib/components/shared/TileCard.svelte` (not `GlassTile`)
- **Given** the Workshop canvas (slot 8) is active, **when** it renders, **then** a left sidebar panel (200px) is visible with 5 navigation items: New Chat, History, Projects, Memory, Skills — all with Lucide icons, no emoji
- **Given** the Workshop canvas is active, **when** the shell is inspected, **then** the right-rail AgentPanel is hidden (Workshop is the full-screen Copilot — no double-agent panel)
- **Given** a TileCard is rendered with financial data, **when** the tile is inspected, **then** all numeric values are in `var(--font-data)` (JetBrains Mono) and all section labels are in `var(--font-ambient)` (Fragment Mono, 10px caps) — CRM named-section pattern enforced
- **Given** a TileCard is clicked, **when** data is more than the tile can show, **then** the overflow is NOT visible on the tile — full data is on the sub-page only (CRM "nothing is lost, just organized beneath")

---

### Story 12-4: Trading Canvas (Slot 5) — Paper Trading & Backtesting Content

**PRD Journey Mapping:**
- **Journey 19** (Paper Trade Graduation — primary for this canvas) — "EA_XAUUSD_Scalp_V2 completes 11 days of paper trading. A structured dossier arrives — win rate, avg R, max drawdown, Sharpe, regime-specific performance breakdown." The Trading canvas tile grid is where Mubarak sees paper trades in progress and can click into the dossier.
- **Journey 12** (Parameter Sweep) — "4 proceed to paper trading. Copilot notification: estimated paper trade review: 5 days." The Trading canvas shows these 4 in the Paper Trading Status tile.
- **Journey 3** (Geopolitical Setup) — After fast-track deployment, the new strategy enters paper monitoring on this canvas before any further escalation.
- **Journey 2** (Alpha Forge Trigger) — "72 hours later: EA_EURUSD_SD_V2 has been paper trading for 3 days." The Trading canvas tile shows its performance tile.

**FR Enablement:** FR29 (paper trading monitoring), FR27 (backtest matrix display — preview), FR5 (regime state visible per strategy)

#### Current State

`quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte` is:
```svelte
<CanvasPlaceholder canvasName="Trading" epicNumber={3} epicName="Live Trading Command Center" />
```
This is wrong on two counts: (1) it's a placeholder, and (2) it points to Epic 3 (Live Trading canvas) — the Trading Department canvas (slot 5) is NOT the same as Live Trading (slot 1). It maps to Epic 7 (department agents) + Epic 8 (Alpha Forge — paper trading gate).

#### Tasks

**Task 12-4-A: Replace TradingCanvas placeholder**
- File: `quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte`
- Replace entire file content with a real `CanvasTileGrid` implementation
- Import: `CanvasTileGrid`, `GlassTile` from shared, lucide icons (`PlayCircle`, `BarChart3`, `TrendingUp`, `Clock`, `CheckCircle`, `AlertCircle`)
- State:
  ```typescript
  let paperTrades = $state<PaperTradeEntry[]>([]);
  let recentBacktests = $state<BacktestSummary[]>([]);
  let loading = $state(true);
  ```
- `onMount`: fetch from `/api/paper-trading/active` and `/api/backtest/recent?limit=5`
- If API returns 404 or empty, render empty state tile (not error)

**Task 12-4-B: Implement 3 tile types (per architecture tile spec for Canvas 5)**

Architecture defines these tiles for `components/trading/tiles/`:

1. **`PaperTradingMonitorTile.svelte`** (spans 2 columns — `size="lg"`):
   - Architecture label: "EA performance monitoring, not agent trading" — EAs monitor themselves
   - Header badge: "PAPER MONITORING" (green)
   - Shows: table of active paper trade EAs (EA name, pair, session, days running, win rate, current P&L)
   - Numbers in `var(--font-data)`. Status dot per EA: running=cyan, paused=amber, failed=red
   - Empty state: "No EAs in paper monitoring phase — Alpha Forge feeds this when EAs reach the paper gate."
   - Click → sub-page `EAPerformanceDetail.svelte` (placeholder — Epic 7/8 fills content)

2. **`BacktestResultsTile.svelte`** (spans 1 column — `size="md"`):
   - Header badge: "RECENT BACKTESTS" (cyan)
   - Shows: last 5 backtest runs (EA name, pass/fail indicator, walk-forward Sharpe, date)
   - Data from: `GET /api/backtest/recent`
   - Empty state: "No backtest results yet."
   - Click → sub-page: Backtest Detail (placeholder — Epic 8 fills this)

3. **`EAPerformanceTile.svelte`** (spans full width — `size="xl"`):
   - Header badge: "ENHANCEMENT LOOP" (amber)
   - Shows: Alpha Forge Workflow 2 pipeline state (stages: Backtest → SIT Gate → Paper Monitoring → Approval → Live)
   - Each stage shown as a count badge: "4 Backtesting · 2 at SIT Gate · 3 Paper Monitoring · 1 Awaiting Approval"
   - State dot colors: RUNNING=cyan (Fragment Mono 11px caps per UX spec state badge pattern)
   - Data from: `GET /api/pipeline-status/stages` (exists in `src/api/pipeline_status_endpoints.py`)
   - Empty state: stages with 0 counts — neutral grey, not error state

**Task 12-4-C: Sub-page routing within Trading canvas**
```typescript
type TradingSubPage = 'grid' | 'ea-performance-detail' | 'backtest-detail';
let currentSubPage = $state<TradingSubPage>('grid');
let selectedEAId = $state<string | null>(null);
```
- Tile clicks set `currentSubPage` and optionally `selectedTradeId`
- `CanvasTileGrid` receives `showBackButton={currentSubPage !== 'grid'}` and `onBack={() => currentSubPage = 'grid'}`
- Sub-pages are placeholder components for now (Epic 7/8 fills content)

**Task 12-4-D: Update `CanvasPlaceholder.svelte` usage note**
- File: `quantmind-ide/src/lib/components/canvas/CanvasPlaceholder.svelte`
- No changes to the file itself; TradingCanvas no longer uses it

#### Acceptance Criteria

- **Given** the Trading canvas (slot 5) is loaded, **when** the canvas renders, **then** a tile grid with 4 tile sections is visible (no "Coming Soon" placeholder)
- **Given** the backend `/api/paper-trading/active` returns an empty list, **when** the Paper Trading Status tile renders, **then** it shows an informative empty state message (not an error)
- **Given** the backend `/api/paper-trading/active` returns active paper trades, **when** they render, **then** each entry shows EA name, pair, days running, and current P&L
- **Given** the Backtest Results tile is clicked, **when** the click fires, **then** the sub-page view replaces the tile grid and a back button appears in the canvas header
- **Given** the back button is clicked, **when** the click fires, **then** the tile grid view is restored
- **Given** the canvas is inspected, **when** the developer looks for `epicNumber={3}`, **then** no such prop exists (the wrong epic reference has been removed)
- **Given** the Strategy Lifecycle tile renders, **when** `/api/pipeline-status/stages` returns all zeros, **then** the tile renders stage indicators at 0 (neutral state — not a broken state)

---

### Story 12-5: Portfolio Canvas + Cross-Canvas Navigation Fixes

**PRD Journey Mapping:**
- **Journey 6** (Portfolio Audit) — "Portfolio canvas shows the monthly performance report: 18 profitable, 8 flat, 5 losing." This journey requires Portfolio canvas breadcrumb navigation to work correctly so Mubarak can drill into individual strategy performance and return to the tile grid.
- **Journey 20** (Dual Account) — Managing prop firm + personal capital requires navigating between Portfolio canvas sub-pages (account registry, routing matrix). Navigation must be clean.
- **Journey 27** (Performance Attribution — implied from Epic 9 Journeys 6, 20, 27) — Attribution drill-down requires reliable sub-page routing.
- **Journey 1** (Morning Operator) — "He clicks through to the Development canvas, sees the two variants listed. Returns to Live Trading. Total time: 8 minutes." Keyboard shortcuts 1–9 must reliably switch canvases.

**FR Enablement:** FR55 (portfolio metrics display), FR51 (broker account registry navigation), NFR-P4 (canvas transitions ≤200ms)

#### Tasks

**Task 12-5-A: Fix PortfolioCanvas sub-page routing**
- File: `quantmind-ide/src/lib/components/canvas/PortfolioCanvas.svelte`
- Read the file first to understand current sub-page state
- Replace any `navigationStore.navigateToView()` calls inside the canvas with local state
- Pattern (matching other canvases):
  ```typescript
  type PortfolioSubPage = 'grid' | 'account-detail' | 'attribution' | 'correlation';
  let currentSubPage = $state<PortfolioSubPage>('grid');
  let selectedAccountId = $state<string | null>(null);
  ```
- Pass to `CanvasTileGrid`: `showBackButton={currentSubPage !== 'grid'}` and `onBack={() => currentSubPage = 'grid'}`

**Task 12-5-B: Align canvas routing between ActivityBar and canvasStore**
- Current problem: `+page.svelte` uses `activeView = $state("live")` prop that gets updated by `ActivityBar`'s `on:viewChange` event, while `activeCanvasStore` in `canvas.ts` is a separate parallel system from Story 1-6
- Fix:
  1. In `+page.svelte`, derive active canvas from `activeCanvasStore` instead of local `activeView` state
  2. Update `ActivityBar.svelte` to dispatch to `activeCanvasStore.setCanvas()` instead of emitting `viewChange` event
  3. Remove `handleViewChange` function from `+page.svelte`
  4. `MainContent.svelte` currently reads `activeView` prop — replace with subscription to `activeCanvasStore`
  5. Ensure `CANVAS_SHORTCUTS` in `canvas.ts` (keyboard 1–9 binding) correctly fires `setCanvas()` and that the canvas component renders accordingly

**Task 12-5-C: Verify StatusBand click targets**
- File: `quantmind-ide/src/lib/components/StatusBand.svelte`
- Read the file to verify all 5 clickable segments work:
  1. Session clocks → navigate to Live Trading canvas
  2. Active bots count → navigate to Portfolio canvas
  3. Risk mode indicator → navigate to Risk canvas
  4. Router mode → navigate to Risk canvas
  5. Node health dots → open node status overlay (not a canvas nav — overlay)
- If any segment is missing click handling, add it using `activeCanvasStore.setCanvas(canvasId)`

**Task 12-5-D: Clean up legacy view prop threading**
- Remove `activeView` prop from `MainContent.svelte` interface (or mark deprecated)
- New canvas rendering in `+page.svelte` uses the canvas component registry pattern (from Story 1-6)
- Ensure no canvas component receives `activeView` as a prop — they each manage their own internal state

#### Acceptance Criteria

- **Given** the Portfolio canvas is loaded, **when** a portfolio sub-page tile is clicked, **then** the sub-page renders and a back button appears in the canvas header
- **Given** the Portfolio canvas sub-page is open, **when** the back button is clicked, **then** the tile grid view is restored (no stale navigation state in `navigationStore`)
- **Given** keyboard shortcut "6" is pressed, **when** the canvas switches, **then** the Portfolio canvas loads (slot 6 = Portfolio per the 9-canvas layout)
- **Given** keyboard shortcut "1" is pressed, **when** the canvas switches, **then** the Live Trading canvas loads as home
- **Given** the StatusBand active bots count is clicked, **when** the click fires, **then** the Portfolio canvas activates
- **Given** the StatusBand session clock is clicked, **when** the click fires, **then** the Live Trading canvas activates
- **Given** `+page.svelte` is inspected, **when** a developer searches for `activeView = $state`, **then** this state is removed (replaced by `activeCanvasStore` derived value)
- **Given** all 9 canvases are loaded sequentially, **when** each is loaded, **then** the correct canvas component renders without any `undefined` active view errors

---

### Story 12-6: Department Kanban Sub-Page (All Canvases Except Live Trading + Workshop)

**PRD Journey Mapping:**
- **Journey 5** (Weekly War Room) — "7 tasks in Development queue, 3 in Risk review" — The Department Kanban is where Mubarak sees what each dept is working on, blocked by, and has done this sprint.
- **Journey 8** (Silent Failure Alert) — "FloorManager detects Development has had no commits in 18h" — visible as a stale BLOCKED card in the Development Kanban.
- **Journey 16** (Code Review) — "Mubarak types feedback: 'Add more error handling'" in the Development canvas Agent Panel. The resulting task shows up in the Development Kanban as IN_PROGRESS.
- **Journey 12** (Parameter Sweep) — The Kanban shows task progress: "512 combinations tested" as a DONE card in the Development Kanban.

**FR Enablement:** FR10 (department task visibility), FR14 (department head task delegation), FR22 (human-in-loop status), foundational for all Epic 7 dept agent stories

#### Context

Architecture and the existing `_bmad-output/implementation-artifacts/7-9-department-kanban-sub-page-ui.md` confirm: every canvas except Live Trading and Workshop gets a **Department Kanban sub-page** accessible from a tile on the tile grid.

Kanban columns (per architecture, real-time SSE from `GET /api/departments/{dept}/tasks`):
```
TODO → IN_PROGRESS → BLOCKED → DONE
```

The `department-kanban/` component directory already exists in `quantmind-ide/src/lib/components/` (from a previous sprint). This story wires it into each canvas as a sub-page.

#### Tasks

**Task 12-6-A: Create `DeptKanbanTile.svelte` in `shared/`**
- File: `quantmind-ide/src/lib/components/shared/DeptKanbanTile.svelte` (NEW)
- A summary tile that lives on the canvas tile grid and shows: active task count, blocked count, done count (last 24h)
- Click → navigates to `currentSubPage = 'dept-kanban'` in the canvas
- Data from: `GET /api/departments/{dept}/tasks/summary`
- Empty state: "No active tasks — dept head is idle"

**Task 12-6-B: Wire sub-page routing in each canvas**
- For each canvas that has `currentSubPage` state (Research, Development, Risk, Trading, Portfolio, SharedAssets, FlowForge):
  - Add `'dept-kanban'` to the `SubPage` union type
  - Pass `showBackButton={currentSubPage !== 'grid'}` to `CanvasTileGrid`
  - Render `<DeptKanban dept={dept} />` when `currentSubPage === 'dept-kanban'`
- The `DeptKanban` component in `department-kanban/` accepts `dept` prop, renders 4 Kanban columns with real-time SSE subscription

**Task 12-6-C: Add `DeptKanbanTile` to skeleton canvases**
- For Research, Development, Risk, Trading, Portfolio, SharedAssets: add `<DeptKanbanTile dept={...} />` to their tile grid alongside the other skeleton tiles
- FlowForge already has its own Prefect Kanban — add a separate `DeptKanbanTile` for the department tasks (distinct from the workflow Kanban)

#### Acceptance Criteria

- **Given** the Development canvas tile grid is visible, **when** a "Development Tasks" summary tile is present, **then** it shows active task count, blocked count, and a click-able area
- **Given** the Development Kanban tile is clicked, **when** the sub-page renders, **then** a Kanban board with TODO / IN_PROGRESS / BLOCKED / DONE columns is shown
- **Given** the back button on the Kanban sub-page is clicked, **when** the click fires, **then** the canvas tile grid is restored
- **Given** the Live Trading canvas is open, **when** it is inspected, **then** NO `DeptKanbanTile` is present (Live Trading canvas is excluded per architecture)
- **Given** the Workshop canvas is open, **when** it is inspected, **then** NO `DeptKanbanTile` is present (Workshop is the FloorManager home, not a dept canvas)
- **Given** SSE is connected, **when** a task status changes, **then** the Kanban card moves to the correct column without a page refresh

---

## Additional Context

### Dependencies

- **12-1 depends on**: Story 1-6 (canvas routing skeleton — `activeCanvasStore` in `canvas.ts`), which is confirmed complete per git log
- **12-2 depends on**: None — can start in parallel with 12-1
- **12-3 depends on**: 12-2 (tokens must be in place before standardizing component styles)
- **12-4 depends on**: 12-3 (CanvasTileGrid + TileCard must exist)
- **12-5 depends on**: 12-1 (shell must have canvasStore as truth), 12-3 (PortfolioCanvas uses CanvasTileGrid)
- **12-6 depends on**: 12-3 (CanvasTileGrid + TileCard + sub-page routing pattern must exist)
- **Backend dependencies**:
  - `src/api/backtest_endpoints.py` exists ✅
  - `src/api/pipeline_status_endpoints.py` exists ✅
  - `src/api/trading/routes.py` includes paper trading ✅
  - `src/api/floor_manager_endpoints.py` (department tasks) exists ✅
  - Agent SSE stream: `GET /api/agents/stream` — wiring scoped to Epic 5, scaffold only in Epic 12

### PRD Journey Master Map for Epic 12

| Story | Primary Journeys | Enabling FRs |
|-------|-----------------|--------------|
| 12-1 (Agent Panel) | J1, J11, J12, J16, J17 | FR20, FR22 (UI scaffold); Epic 5 wires SSE |
| 12-2 (Design Tokens) | All journeys (visual foundation) | NFR-M4, visual baseline; Mubarak preferences: Ghost Panel, Balanced Terminal, Frosted Ghost |
| 12-3 (Tile Grid + Shared Components) | J1, J2, J5, J6, J10, J12, J17, J19 | NFR-P4 (≤200ms canvas switch), all canvas FRs |
| 12-4 (Trading Canvas — slot 5) | J2, J3, J12, J19 | FR29 (paper monitoring), FR27 preview |
| 12-5 (Nav Fixes) | J1, J6, J20 | FR55 (portfolio nav), NFR-P4 |
| 12-6 (Dept Kanban sub-page) | J5, J8, J12, J16 | FR10, FR14, FR22 (human-in-loop visibility) |

### How This Epic Maps to the UX Spec Shell Anatomy

The UX spec defines (lines 74–99 of `ux-design-specification.md`):
```
┌──────────────────────────────────────────────────────────────────────┐
│  TOPBAR  [Kill]  [Copilot]  [Notifications]              [Settings]  │ ← Already built (Epic 3)
├──────────────────────────────────────────────────────────────────────┤
│  STATUS BAND                                                          │ ← Already built (Epic 3)
├───┬──────────────────────────────────────────────┬───────────────────┤
│   │                                              │  AGENT PANEL      │
│ A │   CANVAS WORKSPACE                           │  ← Story 12-1    │
│ B │   ← Stories 12-3, 12-4, 12-6               │  (320px, shell/)  │
│ A │   Default: tile grid  ← Story 12-3          │  [+] [⏱] [←]     │
│ R │   On click: sub-page  ← 12-3/12-4/12-6     │  [Dept Head]      │
│   │   Kanban sub-page     ← Story 12-6          │  [sessions]       │
│   │   Breadcrumb nav      ← Story 12-5          │  [input]          │
└───┴──────────────────────────────────────────────┴───────────────────┘
  ↑ Already built (Epic 1)
```

Epic 12 completes the shell anatomy to full spec compliance. Agent Panel = **320px** (UX spec §Global Shell Dimensions, non-negotiable).

### `ux-design-directions.html` as Visual Reference

The file at `_bmad-output/planning-artifacts/ux-design-directions.html` is the authoritative **visual prototype** for this epic. Key CSS sections to reference:

- **Lines 11–25**: Color token reference — NOTE: HTML prototype values differ from UX spec. USE UX SPEC VALUES in `app.css` (green `#00c896`, text `#e8edf5`, muted `#5a6a80`). HTML prototype is for density/layout patterns only.
- **Lines 66–106**: TopBar, StatusBand visual structure
- **Lines 129–200**: Canvas workspace tile grid, tile CSS
- **Lines 201–216**: Agent Panel structure and CSS — **direct implementation reference for Story 12-1**
- **Lines 218–246**: Workshop canvas sidebar layout
- **Preferred themes**: Ghost Panel (DIR-03), Balanced Terminal (DIR-05), Frosted Ghost (DIR-06) — Mubarak's confirmed preferences. Direction 05 = default. These map to theme presets in the token system.

### Testing Strategy

- **Story 12-1**: Manual visual test — verify Agent Panel is exactly 320px, dept badge changes on canvas switch, collapse works, `[+]` creates session, `[⏱]` opens history panel
- **Story 12-2**: CSS inspection — verify no `oklch(` in `:root`, green = `#00c896`, text = `#e8edf5`, muted = `#5a6a80`, `--color-danger` does not exist (use `--color-accent-red`)
- **Story 12-3**: Visual test across all 9 canvases — no blank placeholders, all tiles render. Check tile grid uses `minmax(280px, 1fr)` with 12px gap for Balanced Terminal default.
- **Story 12-4**: Mock API responses (`PaperTradeEA[]`, `BacktestResult[]`) — verify empty state, populated state, sub-page routing, no `epicNumber={3}` reference
- **Story 12-5**: Keyboard shortcuts 1–9, StatusBand segment clicks, Portfolio back-button test
- **Story 12-6**: DeptKanban tile present on correct canvases (not Live Trading, not Workshop), Kanban columns render, back navigation works

Vitest component tests for: 12-1 (Agent Panel session state transitions), 12-5 (canvas routing — `activeCanvasStore` as source of truth), 12-6 (Kanban sub-page routing).

### Notes

- **Do NOT implement AI streaming in Epic 12** — Agent Panel in 12-1 is a UI scaffold. Agent SSE (`GET /api/agents/stream`) wired in Epic 5. Dept Head wiring in Epic 7.
- **Do NOT implement real physics sensor visualizations in 12-3** — Risk canvas tiles are skeletons with "Epic 4" badges. Epic 4 owns the sensor content.
- **Do NOT place a Workshop Copilot in the right-rail AgentPanel** — Workshop (Canvas 8) = FloorManager, full-screen Claude.ai-inspired. The right-rail AgentPanel = dept-head-level, contextual. Completely separate agents, separate surfaces. AgentPanel is hidden when Workshop is the active canvas.
- **CRM tile pattern is law** — Every TileCard follows: bounded card, named sections (Fragment Mono label + JetBrains Mono value), no visible overflow, one-click → sub-page depth. This is the "Amazon/CRM navigation" pattern from UX spec §Established Patterns. DO NOT put tables, charts, or full datasets in tiles — those belong on the sub-page.
- **BottomPanel.svelte is retained** — It has terminal UI for future Development canvas sub-page (Monaco + terminal side-by-side). Remove from shell grid only. Do NOT delete the file.
- **Svelte 5 runes required** — All new components: `$state`, `$derived`, `$props`, `$effect`. No Svelte 4 `writable()` patterns.
- **`agent-panel/AgentPanel.svelte` is DEPRECATED** — do not modify it. The new canonical component is `components/shell/AgentPanel.svelte`.
- **All financial numbers use `var(--font-data)`** — P&L, risk scores, timestamps in JetBrains Mono. StatusBand ticker and ambient labels use `var(--font-ambient)` (Fragment Mono). This is enforced at the component level.
- **Forbidden patterns (from architecture)**: raw `fetch()` in components (use `apiFetch<T>()`), hardcoded color/spacing (use CSS tokens), emoji (use Lucide), `export let` / `$:` (use runes), `--color-danger` (use `--color-accent-red`).
- **Session architecture in Agent Panel**: Interactive sessions = full chat UI. Autonomous workflow sessions = status card only (read-only). Epic 12 scaffolds both UI states; Epic 5 populates them.
