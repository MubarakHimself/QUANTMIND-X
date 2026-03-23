# Story 12.1: Global Shell — Agent Panel Right Rail + Remove BottomPanel

Status: done

## Story

As a **trader (Mubarak)**,
I want a **collapsible Agent Panel permanently docked in the right rail of the global shell** — displaying the active department head's identity, providing a full chat management scaffold (new session, history, message area, submit), and supporting both Interactive and Autonomous Workflow session modes,
So that I can **communicate with or monitor any Department Head from any canvas without navigating away** from my current work, and so that the BottomPanel no longer consumes vertical space that the canvas workspace and Agent Panel should own.

## Acceptance Criteria

**AC 12-1-1: Agent Panel renders in shell grid**
- **Given** the app is open
- **When** the shell renders
- **Then** a right-rail Agent Panel is visible at exactly 320px width, in `grid-area: agent`, docked between canvas workspace and right screen edge
- **And** the panel is a shell-level component — NOT rendered inside any canvas component

**AC 12-1-2: Dept head badge updates with canvas switch**
- **Given** the Agent Panel is visible
- **When** the user switches to Research canvas (keyboard or ActivityBar)
- **Then** the Agent Panel header badge shows "RESEARCH" tinted with `--color-accent-amber` (#f0a500)
- **And** switching to Risk canvas shows "RISK" tinted with `--color-accent-red` (#ff3b3b)
- **And** switching to Development canvas shows "DEVELOPMENT" tinted with `--color-accent-cyan` (#00d4ff)
- **And** switching to Live Trading shows "TRADING" tinted with `--color-accent-green` (#00c896)

**AC 12-1-3: Collapse and expand with 300ms animation**
- **Given** the Agent Panel is visible (320px)
- **When** the `[←]` collapse button is clicked
- **Then** the panel animates to 0px width over exactly 300ms (`transition: width 300ms ease`)
- **And** the canvas workspace fills the full remaining horizontal width
- **And** an expand trigger remains accessible to restore the panel

**AC 12-1-4: New interactive session created with `[+]`**
- **Given** the Agent Panel is open
- **When** the `[+]` button is clicked
- **Then** a new interactive session is created and added to the session list
- **And** the `.ap-footer` chat input area becomes visible and focused
- **And** `canvasContextService.loadCanvasContext(activeCanvasId)` is called to prime context

**AC 12-1-5: Session history panel opens with `[⏱]`**
- **Given** the Agent Panel is open
- **When** the `[⏱]` button is clicked
- **Then** a session history panel opens showing past sessions for the active dept head
- **And** each entry shows: session title, dept head label, date (UTC converted to local display), and status badge
- **And** clicking a past session loads its message history into the `.ap-body`

**AC 12-1-6: Interactive session — message echo**
- **Given** an interactive session is active
- **When** the user types a message and submits
- **Then** the message appears as a user bubble in `.ap-body` with correct typography (`var(--font-body)`)
- **And** no AI response is generated in Epic 12 (SSE wired in Epic 5)
- **And** the input clears after submission

**AC 12-1-7: Autonomous Workflow session mode — read-only**
- **Given** the Agent Panel is in Autonomous Workflow session mode
- **When** the session is displayed
- **Then** a read-only status card is shown (workflow name, current stage, elapsed time)
- **And** the `.ap-footer` chat input is NOT visible in this mode
- **And** the status card does NOT accept user input

**AC 12-1-8: BottomPanel removed from shell only**
- **Given** the app loads
- **When** the shell renders
- **Then** no BottomPanel is in the DOM
- **And** the full vertical space from StatusBand to screen bottom is occupied by canvas workspace + Agent Panel
- **And** `BottomPanel.svelte` file still exists on disk (not deleted — retained for future canvas sub-page use)

**AC 12-1-9: Workshop canvas hides Agent Panel**
- **Given** the Workshop canvas (slot 8) becomes the active canvas
- **When** the shell renders
- **Then** the right-rail Agent Panel is hidden (Workshop is the full-screen FloorManager Copilot — having two agent panels is forbidden)
- **And** the canvas workspace fills the full available width

**AC 12-1-10: Keyboard shortcuts preserved through grid change**
- **Given** keyboard shortcuts 1–9 are defined in `canvas.ts`
- **When** each key is pressed
- **Then** the correct canvas loads and the Agent Panel dept badge updates to match the new canvas

**AC 12-1-11: SSE-only wiring (no WebSocket)**
- **Given** the Agent Panel is inspected at the network level
- **When** the Agent Panel component mounts
- **Then** it references the SSE channel (`GET /api/agents/stream`) for agent events — never a WebSocket
- **And** the StatusBand WebSocket (live P&L, positions) is a completely separate connection unaffected by Agent Panel state

**AC 12-1-12: Three message types render with correct CSS classes**
- **Given** an Agent Panel session contains agent responses, user messages, and tool-call events
- **When** the `.ap-body` is inspected
- **Then** agent responses use `.ap-agent` (cyan-tinted: `rgba(0,170,204,0.05)` bg, `rgba(0,170,204,0.09)` border)
- **And** user messages use `.ap-user` (right-aligned, `rgba(255,255,255,0.03)` bg, max-width 88%)
- **And** tool-call lines use `.ap-tool` (`border-left: 2px solid rgba(0,170,204,0.2)`, `var(--font-data)`, 9px, `var(--color-text-muted)`)

**AC 12-1-13: OPINION node writes render as `.ap-tool` lines**
- **Given** a `tool_call` SSE event arrives with `tool: "write_memory"` and `args.node_type: "OPINION"`
- **When** the event is processed
- **Then** a `.ap-tool` line renders: `write_memory(OPINION · confidence=0.87 · action="...")`
- **And** clicking the line expands a detail view with full OPINION schema: `{action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role}`

**AC 12-1-14: Memory operation tool lines**
- **Given** an agent calls `write_memory(...)` or `search_memory(...)`
- **When** the SSE `tool_call` event arrives
- **Then** a `.ap-tool` line renders with tool name + abbreviated argument (truncated at 60 chars with ellipsis)
- **And** the line uses `var(--font-data)` font, 9px size, `var(--color-text-muted)` colour

**AC 12-1-15: MCP call lines**
- **Given** an agent uses an MCP server (`context7`, `sequential_thinking`, `web_fetch`, or internal RAG MCP)
- **When** the SSE `tool_call` event arrives
- **Then** a `.ap-tool` line renders, e.g.: `context7(query: "Kelly criterion sizing...")`
- **And** the rendering is visually identical to other `.ap-tool` lines — same class, same styling

**AC 12-1-16: Sub-agent status indicators in Autonomous Workflow mode**
- **Given** an Autonomous Workflow session is active
- **When** sub-agents are spawned
- **Then** a sub-agent status row is visible beneath the workflow stage label
- **And** each active sub-agent shows: role label + status badge (running=`--color-accent-cyan`, idle=`--color-text-muted`, blocked=`--color-accent-amber`)

**AC 12-1-17: RichRenderer renders structured content inside agent bubbles**
- **Given** an agent message contains a structured block (markdown table, code block, chart directive)
- **When** the `.ap-agent` bubble renders
- **Then** the structured block is rendered via `<RichRenderer>` component inline within the bubble
- **And** code blocks use `var(--font-data)` with a slightly elevated background

**AC 12-1-18: Canvas context primed on new Interactive session**
- **Given** the user clicks `[+]` to create a new Interactive session
- **When** session creation fires
- **Then** `canvasContextService.loadCanvasContext(activeCanvas)` is called
- **And** `GET /api/canvas-context/template/{canvas_name}` is called using the string canvas name (e.g. `template/risk`) — NOT a UUID
- **And** the returned `CanvasContextTemplate` is stored on the session object

**AC 12-1-19: SSE EventSource lifecycle**
- **Given** a new Interactive session is created
- **When** `createNewSession()` runs
- **Then** an `EventSource` is opened to `GET /api/agents/stream?session={sessionId}`
- **And** on component destroy (`onDestroy`), `eventSource.close()` is called — no dangling connections
- **And** the StatusBand WebSocket is completely unaffected

## Tasks / Subtasks

- [x] Task 1: Remove BottomPanel from global shell grid (AC: #8)
  - [x] 1.1: Remove `import BottomPanel` from `quantmind-ide/src/routes/+page.svelte`
  - [x] 1.2: Remove `<BottomPanel />` from template in `+page.svelte`
  - [x] 1.3: Update `.ide-layout` grid CSS to 3-column layout (see Dev Notes §Grid Target)
  - [x] 1.4: Add `--agent-panel-width: 320px` to `:root` in `app.css`
  - [x] 1.5: Add `.ide-layout.agent-panel-collapsed` CSS class for 0px collapsed state
  - [x] 1.6: Verify `BottomPanel.svelte` file is NOT deleted (retained on disk)

- [x] Task 2: Create `AgentPanel.svelte` in `components/shell/` (AC: #1, #2, #3, #4, #5, #6, #7, #9, #11, #12, #13, #14, #15, #16, #17, #18, #19)
  - [x] 2.1: Create `quantmind-ide/src/lib/components/shell/` directory
  - [x] 2.2: Create `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` (Svelte 5 runes)
  - [x] 2.3: Implement CANVAS_DEPT_HEAD static map (all 9 canvases)
  - [x] 2.4: Implement `.ap-header` with dept badge, `[+]` (Plus icon), `[⏱]` (History icon), `[←]` (ChevronRight icon)
  - [x] 2.5: Implement `.ap-session-history` conditional panel (scrollable session list)
  - [x] 2.6: Implement `.ap-body` with three message type renderers (`.ap-agent`, `.ap-user`, `.ap-tool`)
  - [x] 2.7: Implement `.ap-tool` line renderer — OPINION expand-on-click, MCP lines, memory op lines
  - [x] 2.8: Implement `.ap-autonomous-status` — workflow stage card + sub-agent status row
  - [x] 2.9: Implement `.ap-footer` — chat input (interactive session only)
  - [x] 2.10: Create `RichRenderer.svelte` in `components/shared/` — markdown tables, code blocks, chart directives
  - [x] 2.11: Wire SSE EventSource lifecycle (open on session create, close on destroy) — scaffold only
  - [x] 2.12: Wire `canvasContextService.getTemplate()` on `[+]` click (using getTemplate() per Dev Notes §9)
  - [x] 2.13: Implement Workshop canvas hidden state (no agent panel when activeCanvas === 'workshop')
  - [x] 2.14: Apply collapse transition `width 300ms ease`

- [x] Task 3: Wire AgentPanel into `+page.svelte` (AC: #1, #2, #9, #10)
  - [x] 3.1: Import `AgentPanel` from `$lib/components/shell/AgentPanel.svelte`
  - [x] 3.2: Import `activeCanvasStore` from `$lib/stores/canvasStore`
  - [x] 3.3: Derive `currentCanvas` from `activeCanvasStore` using `$derived`
  - [x] 3.4: Add `<AgentPanel activeCanvas={currentCanvas} bind:collapsed={agentPanelCollapsed} />` to template
  - [x] 3.5: Apply `.agent-panel-collapsed` class to `.ide-layout` div based on `agentPanelCollapsed` state
  - [x] 3.6: Verify keyboard shortcut handler (in ActivityBar or canvasStore) still fires canvas switches

- [x] Task 4: Write Vitest tests (AC: all)
  - [x] 4.1: `AgentPanel.test.ts` — dept badge updates on canvas prop change
  - [x] 4.2: Test collapse state class applied to shell
  - [x] 4.3: Test `[+]` creates new interactive session
  - [x] 4.4: Test user message echo to `.ap-body`
  - [x] 4.5: Test `.ap-tool` line renders for `tool_call` SSE event type
  - [x] 4.6: Test OPINION tool_call expansion
  - [x] 4.7: Test Workshop canvas hides panel (hidden class applied)
  - [x] 4.8: Test SSE EventSource opens on session create (mock EventSource)
  - [x] 4.9: Test SSE EventSource closes on component destroy

## Dev Notes

### CRITICAL ANTI-PATTERNS — DO NOT DO THESE

1. **DO NOT reuse `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte`** — that is the DEPRECATED component wired to old `analyst/copilot/quantcode` agent types. It must remain untouched on disk but is NOT used here. Build NEW in `components/shell/`.

2. **DO NOT add a WebSocket to AgentPanel** — the StatusBand already has the trading WebSocket (Cloudzy). Agent Panel uses SSE only (`GET /api/agents/stream`, Contabo). These two channels are NEVER merged in the same component. [Source: Arch-UI-4]

3. **DO NOT use `writable()` or `export let`** — Svelte 5 runes only: `$state`, `$derived`, `$props`, `$effect`. All new code in this story uses runes exclusively.

4. **DO NOT use raw `fetch()`** in Svelte components — use `apiFetch<T>()` from `$lib/api`. Exception: SSE `EventSource` is not a fetch call and does not use `apiFetch`.

5. **DO NOT delete `BottomPanel.svelte`** — retain on disk, only remove from `+page.svelte` imports/template.

6. **DO NOT render AgentPanel inside any canvas component** — it lives in the global shell `+page.svelte` only. Canvas components must NOT have their own agent panel instances.

7. **DO NOT use hardcoded colours** — CSS custom properties from the Frosted Terminal token system only.

8. **DO NOT use emoji** — Lucide icons only (`lucide-svelte` named imports).

9. **DO NOT use `/api/canvas-context/{UUID}`** — the implemented backend endpoint is `GET /api/canvas-context/template/{canvas_name}` where `canvas_name` is a string (e.g., `"risk"`, `"research"`). The `canvasContextService.loadCanvasContext()` method uses `POST /api/canvas-context/load` — check which to use. For template-only priming use `canvasContextService.getTemplate(canvasId)`. [Source: epic-12-stories.md §Backend Connections, §Canvas Context Endpoint Clarification]

---

### Grid Target — `+page.svelte` CSS Change

**Current grid (2-column, has bottom row):**
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
}
```

**Target grid (3-column, no bottom row):**
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
  width: 100vw;
  background: transparent;
  overflow: hidden;
  gap: 0;
  position: relative;
}

.ide-layout.agent-panel-collapsed {
  grid-template-columns: var(--sidebar-width) 1fr 0px;
  overflow: hidden;
}
```

**`app.css` addition to `:root`:**
```css
--agent-panel-width: 320px;
```

---

### `AgentPanel.svelte` — Required Structure

**File location:** `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` (NEW — directory must be created)

**Props interface:**
```typescript
interface Props {
  activeCanvas: string;
  collapsed?: boolean;
}
let { activeCanvas, collapsed = $bindable(false) }: Props = $props();
```

**State:**
```typescript
let sessions = $state<AgentSession[]>([]);
let activeSessionId = $state<string | null>(null);
let showSessionHistory = $state(false);
let inputValue = $state('');
let expandedToolLine = $state<string | null>(null);  // for OPINION expand-on-click

let activeSession = $derived(sessions.find(s => s.id === activeSessionId) ?? null);
let messages = $derived(activeSession?.messages ?? []);
let deptHead = $derived(CANVAS_DEPT_HEAD[activeCanvas] ?? CANVAS_DEPT_HEAD['workshop']);
let isWorkshop = $derived(activeCanvas === 'workshop' || activeCanvas === 'flowforge');
```

**Session type definitions:**
```typescript
interface AgentMessage {
  id: string;
  type: 'agent' | 'user' | 'tool';
  content: string;
  tool?: string;      // for type 'tool' — tool name
  args?: Record<string, unknown>;  // for type 'tool' — args
  timestamp: string;  // ISO UTC
}

interface SubAgentStatus {
  role: string;  // e.g. 'mql5_dev', 'backtester'
  status: 'running' | 'idle' | 'blocked';
}

interface AgentSession {
  id: string;
  type: 'interactive' | 'autonomous';
  deptHead: string;
  canvasId: string;
  canvasContext?: unknown;  // CanvasContextTemplate — Epic 5 injects into agent
  messages: AgentMessage[];
  workflowName?: string;    // autonomous mode
  workflowStage?: string;   // autonomous mode
  workflowElapsed?: number; // seconds
  subAgents?: SubAgentStatus[];
  createdAt: string;
  status: 'active' | 'completed' | 'error';
}
```

**CANVAS_DEPT_HEAD static map:**
```typescript
const CANVAS_DEPT_HEAD: Record<string, { label: string; color: 'cyan' | 'amber' | 'red' | 'green' | 'muted' }> = {
  'live-trading': { label: 'TRADING',     color: 'green' },
  'research':     { label: 'RESEARCH',    color: 'amber' },
  'development':  { label: 'DEVELOPMENT', color: 'cyan' },
  'risk':         { label: 'RISK',        color: 'red' },
  'trading':      { label: 'TRADING',     color: 'green' },
  'portfolio':    { label: 'PORTFOLIO',   color: 'cyan' },
  'shared-assets':{ label: 'SHARED',      color: 'muted' },
  'workshop':     { label: 'FLOOR MGR',   color: 'cyan' },
  'flowforge':    { label: 'FLOOR MGR',   color: 'cyan' },
};

const COLOR_MAP: Record<string, string> = {
  'cyan':  'var(--color-accent-cyan)',    // #00d4ff
  'amber': 'var(--color-accent-amber)',   // #f0a500
  'red':   'var(--color-accent-red)',     // #ff3b3b
  'green': 'var(--color-accent-green)',   // #00c896
  'muted': 'var(--color-text-muted)',
};
```

**`createNewSession()` function:**
```typescript
async function createNewSession() {
  // Canvas context priming — uses canvas_name string NOT UUID
  // canvasContextService.getTemplate() calls GET /api/canvas-context/template/{canvasId}
  const context = await canvasContextService.getTemplate(activeCanvas);

  const session: AgentSession = {
    id: crypto.randomUUID(),
    type: 'interactive',
    deptHead: deptHead.label,
    canvasId: activeCanvas,
    canvasContext: context,  // stored for Epic 5 to inject
    messages: [],
    createdAt: new Date().toISOString(),
    status: 'active',
  };
  sessions = [...sessions, session];
  activeSessionId = session.id;
  showSessionHistory = false;

  // SSE scaffold — no real events arrive in Epic 12
  openSSE(session.id);
}
```

**SSE EventSource scaffold (Contabo node):**
```typescript
let eventSource: EventSource | null = null;

function openSSE(sessionId: string) {
  // Close any existing connection
  eventSource?.close();
  // Open to Contabo SSE endpoint
  eventSource = new EventSource(`${API_CONFIG.CONTABO_BASE}/api/agents/stream?session=${sessionId}`);
  eventSource.onmessage = (e) => handleStreamEvent(JSON.parse(e.data));
  eventSource.onerror = () => {
    // In Epic 12 this silently retries — no real events expected
    console.warn('[AgentPanel] SSE connection error — will retry');
  };
}

function handleStreamEvent(event: AgentStreamEvent) {
  if (event.type === 'tool_call')       appendToolLine(event);
  if (event.type === 'agent_message')   appendAgentMessage(event);
  if (event.type === 'sub_agent_status') updateSubAgentStatus(event);
  if (event.type === 'task_status')     updateWorkflowStage(event);
}

// Cleanup on destroy — mandatory, prevents dangling connections
onDestroy(() => {
  eventSource?.close();
  eventSource = null;
});
```

**SSE Event types (for TypeScript interface):**
```typescript
type AgentStreamEvent =
  | { type: 'tool_call'; tool: string; args: Record<string, unknown> }
  | { type: 'agent_message'; content: string; role: 'assistant' }
  | { type: 'sub_agent_status'; agent_role: string; status: 'running' | 'idle' | 'blocked' }
  | { type: 'task_status'; workflow_id: string; stage: string; elapsed_s: number };
```

---

### AgentPanel DOM Structure

Matches `ux-design-directions.html` lines 201–216 (visual authority):

```
.agent-panel (grid-area: agent; 320px; flex-column; border-left: 1px solid var(--c-border))
  └── [hidden if isWorkshop]
  .ap-header (height: 36px; flex-row; align-items: center; padding: 0 8px; gap: 4px)
      .ap-dept-badge           — text label + color tint from CANVAS_DEPT_HEAD map
      .ap-spacer (flex: 1)
      button.ap-icon-btn [+]   — Plus icon; creates new interactive session
      button.ap-icon-btn [⏱]  — History icon; toggles showSessionHistory
      button.ap-icon-btn [←]  — ChevronRight icon; sets collapsed = true
  .ap-session-history (if showSessionHistory)
      — scrollable list of past sessions
      — each: title, dept label, date (local TZ from UTC), status badge
  .ap-body (flex: 1; overflow-y: auto; padding: 8px; gap: 7px; display: flex; flex-direction: column)
      .ap-autonomous-status (if activeSession?.type === 'autonomous')
          — workflow name, stage, elapsed time
          — sub-agent row: role label + status badge per sub-agent
      [for each message in messages]
      .ap-agent   — rgba(0,170,204,0.05) bg; border: 1px solid rgba(0,170,204,0.09); padding: 8px; border-radius: 4px
                    wraps content in <RichRenderer>
      .ap-user    — right-aligned; rgba(255,255,255,0.03) bg; max-width: 88%; padding: 8px
      .ap-tool    — border-left: 2px solid rgba(0,170,204,0.2); padding: 4px 6px; font: var(--font-data) 9px; color: var(--color-text-muted)
                    OPINION lines: expandable on click → shows {action,reasoning,confidence,alternatives_considered,constraints_applied,agent_role}
  .ap-footer (if activeSession?.type === 'interactive')
      textarea.ap-input        — resizable, sends on Enter (shift+Enter for newline)
      button.ap-send           — Send icon; submits message
```

**`.ap-tool` line format rules:**
- Generic: `{toolName}({firstArgKey}: "{firstArgValue}")`
- OPINION write: `write_memory(OPINION · confidence={conf} · action="{action truncated to 40 chars}")`
- search_memory: `search_memory(query: "{query truncated to 60 chars}")`
- context7: `context7(query: "{query truncated to 60 chars}")`
- sequential_thinking: `sequential_thinking(step {n}/{total} · {reasoning truncated})`
- web_fetch: `web_fetch(url: "{url truncated to 60 chars}")`

---

### RichRenderer Component

**File location:** `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` (NEW — shared/)

**Purpose:** Renders structured agent output blocks inline inside `.ap-agent` message bubbles.

**Props:**
```typescript
interface Props {
  content: string;  // raw agent message content — may contain markdown, code fences, chart directives
}
```

**Renders:**
- Markdown tables → styled `<table>` using `var(--font-data)` for numeric cells
- Code blocks (` ```lang ``` `) → `<pre><code>` with `var(--font-data)`, slightly elevated background (`var(--color-bg-elevated)`)
- Plain text → `<p>` with `var(--font-body)`
- Chart directive (`[CHART:bar:...]`) → placeholder `<div class="ap-chart-placeholder">` in Epic 12 (real chart in Epic 5)

---

### Backend Endpoints for Story 12-1

| Endpoint | Method | Purpose | Server | Status |
|---|---|---|---|---|
| `/api/canvas-context/template/{canvas_name}` | GET | Canvas context priming on new session | Contabo | Implemented — uses `canvas_name` string (e.g. `"risk"`) NOT UUID |
| `/api/agents/stream` | GET (SSE) | Agent event stream — scaffold only in Epic 12 | Contabo | Implemented |
| `/api/agents/sessions` | GET | Session history list for `[⏱]` panel | Contabo | Implemented (schema needs verification) |

**Canvas name → endpoint mapping:**
- `activeCanvasStore` returns a canvas ID string like `"live-trading"`, `"risk"`, `"research"`
- Pass this string directly to `canvasContextService.getTemplate(canvasId)` — the service handles the name-to-ID mapping internally via `getCanvasId()` method
- Do NOT manually strip or remap — `canvasContextService` already has the `nameToId` map

---

### Visual CSS Tokens Used in AgentPanel

All tokens must come from the Frosted Terminal system in `app.css` (Story 12-2 will expand this, but these already exist):

| Token | Value | Usage |
|---|---|---|
| `--color-accent-cyan` | `#00d4ff` | Development/Workshop dept badge, `.ap-tool` border, cyan status |
| `--color-accent-amber` | `#f0a500` | Research dept badge, blocked sub-agent |
| `--color-accent-red` | uses `--color-danger: #ff3b3b` in current app.css | Risk dept badge |
| `--color-accent-green` | `#00c896` | Trading/Live Trading dept badge, running sub-agent |
| `--color-text-muted` | OKLCH-based in current app.css: use `--text-muted` | Muted text, `.ap-tool` lines, idle sub-agent |
| `--color-bg-elevated` | OKLCH-based: use `--bg-tertiary` | Code block backgrounds |
| `--font-data` | Not yet defined — use `--font-mono` (`'JetBrains Mono', monospace`) | `.ap-tool` lines, financial numbers, code |
| `--font-body` | Not yet defined — use `--font-family` (`'Inter', system-ui`) | Agent/user message text |
| `--c-border` | Not defined — use `--border-subtle` | Agent Panel left border |
| `--glass-tier-2` | `rgba(8, 13, 20, 0.35)` | Agent Panel background (content tier glass) |
| `--glass-blur` | `blur(24px) saturate(160%)` | Agent Panel `backdrop-filter` |

**NOTE:** Story 12-2 will canonicalize these tokens. For 12-1, use the tokens that currently exist. Map as shown above. Do NOT skip the backdrop-filter — the glass aesthetic requires it.

**Agent Panel background (shell tier glass):**
```css
.agent-panel {
  background: var(--glass-tier-2);  /* rgba(8,13,20,0.35) */
  backdrop-filter: var(--glass-blur);  /* blur(24px) saturate(160%) */
  border-left: 1px solid var(--border-subtle);
  grid-area: agent;
  width: 320px;
  transition: width 300ms ease;
  overflow: hidden;
}
.agent-panel.collapsed {
  width: 0;
}
```

---

### `+page.svelte` Wiring

**Current state of `+page.svelte`:**
- Imports: `TopBar`, `ActivityBar`, `StatusBand`, `MainContent`, `BottomPanel`
- Grid: 2-column, 4-row (includes bottom row)
- `activeView` state variable drives canvas via `MainContent` prop (legacy pattern)
- Wallpaper system via `themeStore` is fully functional — do NOT touch

**Required changes:**
```typescript
// REMOVE:
import BottomPanel from "$lib/components/BottomPanel.svelte";

// ADD:
import AgentPanel from "$lib/components/shell/AgentPanel.svelte";
import { activeCanvasStore } from "$lib/stores/canvasStore";

// ADD state:
let agentPanelCollapsed = $state(false);

// ADD derived (Svelte 5 runes):
let currentCanvas = $derived($activeCanvasStore);
```

**Template change:**
```svelte
<!-- REMOVE: -->
<BottomPanel />

<!-- ADD (after MainContent): -->
<AgentPanel activeCanvas={currentCanvas} bind:collapsed={agentPanelCollapsed} />
```

**CSS grid class binding:**
```svelte
<div class="ide-layout" class:agent-panel-collapsed={agentPanelCollapsed}>
```

**Note:** The `activeView` legacy pattern (`handleViewChange`, `handleOpenFile`, `handleCloseTab`, `handleOpenSettings`) is left intact for this story — it's used by `MainContent.svelte`. Do not remove it in 12-1. Canvas routing cleanup is Story 12-5's scope.

---

### ActiveCanvasStore — How It Works

`activeCanvasStore` lives in `quantmind-ide/src/lib/stores/canvasStore.ts`.

```typescript
// canvasStore.ts exports:
export const activeCanvasStore = createCanvasStore();  // writable<string> defaulting to 'workshop'
export const CANVASES: Canvas[] = [...9 canvases...];
export const CANVAS_SHORTCUTS: Record<string, string> = { '1': 'live-trading', ..., '9': 'flowforge' };
```

In Svelte 5 components, subscribe via: `let canvas = $derived($activeCanvasStore);`

The `canvasContextStore` in `src/lib/stores/canvas.ts` is a DIFFERENT store used by NL commands system (Story 5-7). Do NOT confuse these two. `AgentPanel` uses `activeCanvasStore` from `canvasStore.ts`.

---

### Workshop Canvas Hidden State

Workshop (slot 8, `'workshop'`) and FlowForge (slot 9, `'flowforge'`) both trigger Agent Panel hiding because:
- Workshop IS the FloorManager Copilot — it is itself a full-screen agent interface
- Showing a right-rail Agent Panel on top of Workshop would create two agent interfaces simultaneously (forbidden per architecture)

Implementation:
```svelte
<!-- In AgentPanel.svelte template: -->
{#if !isWorkshop}
  <!-- full panel content -->
{/if}
```

The grid column `var(--agent-panel-width)` still exists in the CSS — the panel content is simply hidden. Optionally, the parent can also set `collapsed = true` when Workshop activates to reclaim the column width.

---

### File List — New Files to Create

| File | Action |
|---|---|
| `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` | CREATE NEW — canonical right-rail agent panel |
| `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` | CREATE NEW — structured content renderer for agent messages |
| `quantmind-ide/src/lib/components/shell/AgentPanel.test.ts` | CREATE NEW — Vitest tests |

| File | Action |
|---|---|
| `quantmind-ide/src/routes/+page.svelte` | MODIFY — grid restructure, add AgentPanel, remove BottomPanel |
| `quantmind-ide/src/app.css` | MODIFY — add `--agent-panel-width: 320px` to `:root` |

| File | Action |
|---|---|
| `quantmind-ide/src/lib/components/BottomPanel.svelte` | RETAIN — do NOT delete, do NOT modify |
| `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte` | RETAIN DEPRECATED — do NOT touch, do NOT import |

---

### Coding Standards

- **Svelte version:** Svelte 5 runes only — `$state`, `$derived`, `$props`, `$effect`, `$bindable`. No `export let`, no `$:` reactive, no `writable()`.
- **Icons:** Named imports from `lucide-svelte` only. Icons needed: `Plus`, `History`, `ChevronRight`, `Send`.
- **API calls:** `apiFetch<T>()` from `$lib/api`. Exception: `EventSource` is not `apiFetch` (it's native browser API).
- **File size:** Keep `AgentPanel.svelte` under 500 lines. Extract sub-components if needed.
- **Date display:** All timestamps stored as UTC ISO strings. Convert to local timezone only at display layer.
- **Financial numbers:** Any P&L, timing, or numeric data shown in Agent Panel uses `var(--font-data)` (JetBrains Mono). [Source: Arch-UI-9]
- **Testing:** Vitest + `@testing-library/svelte`. Co-locate test file as `AgentPanel.test.ts` in same `shell/` directory.

---

### Project Structure Notes

- `quantmind-ide/src/lib/components/shell/` does NOT yet exist — dev must create this directory.
- `quantmind-ide/src/lib/components/shared/` may or may not exist — check before creating. `GlassTile.svelte` lives in `live-trading/` not `shared/`.
- `canvasContextService` is a singleton from `$lib/services/canvasContextService` — import and call, do not instantiate.
- `API_CONFIG` (for Contabo base URL in SSE) — import from `$lib/config/api`. The Contabo node serves agent endpoints; Cloudzy serves trading data.

---

### Alignment with Existing Patterns

Reference these files for existing patterns before implementing:
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` — existing SSE+streaming UI pattern, Lucide icon usage, `canvasContextService` call pattern, `onMount`/`onDestroy` cleanup
- `quantmind-ide/src/lib/stores/canvasStore.ts` — `activeCanvasStore` subscription pattern
- `quantmind-ide/src/lib/services/canvasContextService.ts` — `getTemplate(canvasId)` vs `loadCanvasContext()` — use `getTemplate()` for lightweight session priming

### References

- [Source: _bmad-output/planning-artifacts/epic-12-stories.md §Story 12-1] — Full story definition, ACs, implementation approach
- [Source: _bmad-output/implementation-artifacts/tech-spec-epic-12-ui-refactor.md §Story 12-1] — Detailed tasks, grid CSS, AgentPanel structure, token conflict resolution
- [Source: _bmad-output/planning-artifacts/epic-12-stories.md §Arch-Agent-1 to Arch-Agent-8] — Architecture requirements for Agent Panel SSE, message types, OPINION nodes, MCP calls
- [Source: _bmad-output/planning-artifacts/epic-12-stories.md §Backend Connections Master Table] — Confirmed endpoints
- [Source: quantmind-ide/src/routes/+page.svelte lines 99–114] — Current grid CSS to modify
- [Source: quantmind-ide/src/lib/stores/canvasStore.ts] — activeCanvasStore, CANVASES, CANVAS_SHORTCUTS
- [Source: quantmind-ide/src/lib/services/canvasContextService.ts] — getTemplate(), loadCanvasContext(), nameToId map
- [Source: quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte] — Reference streaming/SSE pattern
- [Source: _bmad-output/planning-artifacts/ux-design-directions.html lines 201–216] — Visual authority for Agent Panel CSS and structure
- [Source: _bmad-output/planning-artifacts/prd.md §J1, J11, J12, J16, J17, J29, J32] — User journeys enabled by this story

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

No blocking issues encountered. Two test assertions required minor fixes during red-green cycle:
1. OPINION action string test used 43-char string (over 40-char truncate limit) — fixed test to use 25-char string matching actual truncation behavior.
2. CSS grid column count test used string.split(' ') which mis-split `var(--agent-panel-width, 320px)` at the comma-space — fixed to use content-based assertions instead.

### Completion Notes List

- Implemented `AgentPanel.svelte` (NEW) at `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` — Svelte 5 runes, 320px right-rail, full chat scaffold
- Implemented `RichRenderer.svelte` (NEW) at `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` — markdown tables, code fences, chart placeholders, plain text
- Removed `BottomPanel` from `+page.svelte` (import + template); `BottomPanel.svelte` file retained on disk untouched
- Updated `.ide-layout` CSS grid from 2-column/4-row to 3-column/3-row (added `agent` grid area)
- Added `--agent-panel-width: 320px` to `:root` in `app.css`
- Wired `activeCanvasStore` + `$derived` pattern in `+page.svelte` for reactive canvas prop pass-through
- SSE EventSource scaffold: opens to `CONTABO_HMM_API/api/agents/stream?session={id}` on session create, closes in `onDestroy` — no dangling connections
- Workshop + FlowForge canvases hide panel content via `isWorkshop` derived state
- Collapse animation: `transition: width 300ms ease` on `.agent-panel`; collapse trigger fixed-position button when collapsed
- All 42 new tests pass; no regressions in full suite (222 pass, 4 pre-existing skips)

### File List

- `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` — CREATED
- `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` — CREATED
- `quantmind-ide/src/lib/components/shell/AgentPanel.test.ts` — CREATED
- `quantmind-ide/src/routes/+page.svelte` — MODIFIED
- `quantmind-ide/src/app.css` — MODIFIED
- `quantmind-ide/src/lib/services/canvasContextService.ts` — MODIFIED (code review fix: double /api prefix in URLs)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — MODIFIED (story status updated)

### Senior Developer Review (AI)

**Reviewer:** claude-sonnet-4-6 on 2026-03-22

**Outcome:** APPROVED (after fixes applied)

**Issues Found and Fixed:**

1. **[HIGH] `--color-accent-green` missing from `app.css`** — `AgentPanel` uses `var(--color-accent-green)` for TRADING/Live Trading badge and running sub-agent status indicator, but this CSS variable was undefined. Fixed: added `--color-accent-green: #00c896` to `:root` in `app.css`.

2. **[HIGH] `canvasContextService.ts` double `/api` prefix** — `getTemplate()`, `loadCanvasContext()`, and `getAvailableCanvases()` all used `${API_CONFIG.API_BASE}/api/...`. Since `API_BASE` already returns `/api` suffix, this generated `http://host/api/api/...`. Fixed: removed the redundant `/api/` prefix from all three methods.

3. **[HIGH] `--color-accent-red` not defined in `app.css`** — AC 12-1-2 references `--color-accent-red` for the Risk canvas badge. Fixed: added `--color-accent-red: #ff3b3b` to `:root` in `app.css` alongside `--color-danger`. Updated `AgentPanel.svelte` `COLOR_MAP` to use `var(--color-accent-red)` instead of `var(--color-danger)`. Updated `AgentPanel.test.ts` assertions accordingly.

4. **[MEDIUM] SSE fallback URL used wrong port (8000 instead of 8001)** — `openSSE()` fell back to `API_CONFIG.LOCAL_API_URL` (port 8000, the main API) when Contabo isn't configured. The agent SSE endpoint runs on port 8001. Fixed: changed fallback to `http://localhost:8001`.

5. **[MEDIUM] Session history panel never called backend `/api/agents/sessions`** — AC 12-1-5 requires showing past sessions from the backend. The panel only showed in-memory (current-session) data, meaning all sessions from previous app loads were invisible. Fixed: added `loadSessionHistory()` function that fetches `GET /api/agents/sessions?dept_head={label}` on history panel open, merges with in-memory sessions (deduped by id). Wired to `[⏱]` button via `toggleSessionHistory()`.

6. **[MEDIUM] `.ap-tool` button missing `type="button"`** — `<button>` without explicit type defaults to `type="submit"` per HTML spec, which could cause accidental form submission if the panel is ever nested in a form. Fixed: added `type="button"` attribute.

7. **[MEDIUM] `.ap-tool` had no `:focus-visible` outline** — Keyboard users tabbing to tool-call lines had no visual focus indicator. Fixed: added `outline: none` with `:focus-visible` rule using `--color-accent-cyan` outline.

8. **[MEDIUM] `RichRenderer` used `JSON.stringify(block)` as each-block key** — Expensive serialization on every render, and semantically fragile if two identical blocks appear in sequence. Fixed: changed to index-based key `(blockIdx)`.

**All 42 tests pass after fixes. No regressions.**

### Change Log

- 2026-03-22: Story 12-1 implemented — Global Shell Agent Panel right rail + BottomPanel removal. Grid restructured from 2-column to 3-column. AgentPanel.svelte with full chat scaffold (interactive + autonomous modes, SSE lifecycle, OPINION expansion, RichRenderer integration) created in components/shell/. RichRenderer.svelte created in components/shared/. 42 new Vitest tests added, all passing.
- 2026-03-22: Code review (claude-sonnet-4-6) — 8 issues found and fixed: missing CSS tokens (--color-accent-green, --color-accent-red), double /api prefix in canvasContextService.ts, wrong SSE fallback port, session history missing backend API call, button type attribute, focus-visible outline, RichRenderer key stability. Story status → done.
