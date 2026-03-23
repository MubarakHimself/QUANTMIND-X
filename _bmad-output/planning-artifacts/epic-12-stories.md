---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics', 'step-03-create-stories']
status: ready-for-dev
inputDocuments:
  - '_bmad-output/implementation-artifacts/tech-spec-epic-12-ui-refactor.md'
  - '_bmad-output/planning-artifacts/prd.md'
  - '_bmad-output/planning-artifacts/architecture.md'
  - '_bmad-output/planning-artifacts/ux-design-specification.md'
  - '_bmad-output/planning-artifacts/ux-design-directions.html'
epicScope: 'Epic 12 — UI Refactor: Shell Agent Panel, Design Tokens, Tile Grid, Trading Canvas, Portfolio Nav'
createdDate: '2026-03-22'
mandate: >
  Write NEW code in correct directories. Make targeted surgical edits to legacy only where required.
  Do NOT extend, overwrite, or build on top of deprecated components.
  Settings panel and all sub-panels are preserved as-is.
---

# Epic 12 — Stories Breakdown

## Overview

Epic 12 brings the global shell layout, design token system, and all 9 canvas tile grids into full spec compliance. This is **structural and visual infrastructure** — no AI features, no trading logic, no Settings changes. All subsequent functional epics (4–11) build their content into the foundations established here.

**Architectural Mandate (Mubarak, 2026-03-22):**
> "We are not writing on top of the legacy or deprecated code. Write new code and make targeted edits. The legacy code is poorly designed and it's causing issues."

**Settings Mandate:** `ProvidersPanel`, `AppearancePanel`, `NotificationSettingsPanel`, `ServerHealthPanel`, `ServersPanel` — preserved as-is. No changes.

---

## Requirements Inventory

### Functional Requirements — PRD (Epic 12 Scope)

| ID | Requirement | Story |
|----|-------------|-------|
| FR5 | The trader can view the active market regime classification and strategy router state at all times | 12-4 |
| FR10 | The Copilot can receive natural language instructions and orchestrate the appropriate department, agent, or workflow response | 12-6 |
| FR14 | The system can maintain persistent agent memory across sessions — conversation context, strategy knowledge, and trader preferences | 12-1 |
| FR20 | The Copilot can operate context-aware on any canvas, with tools and commands appropriate to the active department | 12-1 |
| FR22 | The trader can directly converse with any individual Department Head via the Department Chat panel | 12-1 |
| FR27 | The system can run a full backtest matrix — Standard, Monte Carlo, Walk-Forward, and SIT — per EA variant | 12-4 |
| FR29 | The Trading Department can monitor paper trading performance of a new EA before live promotion | 12-4 |
| FR51 | The trader can register and configure at least 4 broker accounts simultaneously in the broker registry | 12-5 |
| FR55 | The trader can review portfolio-level performance metrics — total equity, drawdown, P&L attribution per strategy and per broker | 12-5 |

### Functional Requirements — Architecture.md (Epic 12 Scope)

These are requirements derived from explicit architectural decisions in `architecture.md` that directly constrain Epic 12 implementation:

| ID | Requirement | Story | Source |
|----|-------------|-------|--------|
| Arch-UI-1 | Agent Panel in `components/shell/` must implement TWO distinct session UI modes: Interactive (chat input visible, user can type) and Autonomous Workflow (status card only, read-only — no input) | 12-1 | Arch §6.2, §12 |
| Arch-UI-2 | Workshop canvas (slot 8) uses a Claude.ai-inspired 3-column layout — NOT a tile grid: 200px left nav panel + main conversation area + right rail is the global Agent Panel (hidden when Workshop active) | 12-3 | Arch §12 |
| Arch-UI-3 | Trading Kill Switch = TopBar ONLY; Workflow Kill Switch = FlowForge Kanban row ONLY. Neither appears on canvas tiles or inside canvas components | 12-3, 12-6 | Arch Forbidden Patterns |
| Arch-UI-4 | Agent Panel (shell) consumes SSE stream (`GET /api/agents/stream` — Contabo). StatusBand trading data consumes WebSocket (Cloudzy). These two channels are never merged in the same component | 12-1 | Arch §Communication Patterns |
| Arch-UI-5 | All canvas components that display dept-accented elements must carry a `data-dept` attribute on the root element, enabling the CSS token resolver to apply `--dept-accent` per canvas | 12-3 | Arch §Visual Design |
| Arch-UI-6 | `CanvasTileGrid` is the mandated shared layout wrapper for all canvases except Workshop. File location: `components/shared/CanvasTileGrid.svelte` | 12-3 | Arch §12 |
| Arch-UI-7 | Canvas-local state (sub-page routing, tile data) lives inside the canvas component. Global state (node health, StatusBand trading data) lives in `src/lib/stores/` | 12-5 | Arch §Communication Patterns |
| Arch-UI-8 | Canvas context loads via `canvasContextService.loadCanvasContext(canvasId)` when a new interactive Agent Panel session starts — existing service, existing call pattern | 12-1 | Arch §6.2 |
| Arch-UI-9 | All financial figures (P&L, balance, lot size, risk score, timestamps) must use `var(--font-data)` (JetBrains Mono) — never in a variable-width typeface | All | Arch §Visual Design |

### Architecture Requirements — Agent↔UI Bidirectional Context (Epic 12 Scope)

These requirements come from the agent identity, memory, MCP, and session architecture in `architecture.md` and directly constrain how the Agent Panel is built.

| ID | Requirement | Story | Source |
|----|-------------|-------|--------|
| Arch-Agent-1 | Agent Panel `.ap-body` renders THREE message types matching `ux-design-directions.html` line 209–211: `.ap-agent` (agent response — cyan-tinted bg), `.ap-user` (user input — right-aligned, white-tinted), `.ap-tool` (tool-call line — `border-left: 2px solid rgba(0,170,204,0.2)`, `var(--font-data)`, 9px, muted color). OPINION node writes, `write_memory`, `search_memory`, MCP calls all render as `.ap-tool` lines | 12-1 | UX HTML line 211, Arch §6.4 |
| Arch-Agent-2 | When an Autonomous Workflow session is active in the Agent Panel, a sub-agent status row is shown beneath the workflow title: one indicator per active sub-agent (sub-agent role label + status badge: running=cyan, idle=muted, blocked=amber). Sub-agent types include: `mql5_dev`, `backtester`, `data_researcher`, `trade_monitor`, `fill_tracker`, `performance_reporter` | 12-1 | Arch §14.3, §4.1 |
| Arch-Agent-3 | On new Interactive session creation (`[+]`), `canvasContextService.loadCanvasContext(canvasId)` is called — this hits `GET /api/canvas-context/{canvasId}`, assembles the `CanvasContextTemplate` (CAG stable identifiers + RAG live-state JIT), and primes the agent's session context. This call is wired in Epic 12; the context payload is consumed by the real agent in Epic 5 | 12-1 | Arch §6.2, §16.4 |
| Arch-Agent-4 | OPINION nodes written by agents after consequential actions surface in the Agent Panel as `.ap-tool` lines with format: `write_memory(OPINION · confidence=0.87 · action="...")` — expandable on click to show full schema: `{action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role}`. Default importance_score 0.7; approval-gate OPINION = 0.9 | 12-1 | Arch §6.4 |
| Arch-Agent-5 | MCP tool calls (`context7`, `sequential_thinking`, `web_fetch`, internal RAG/CAG MCP) surface as `.ap-tool` lines in the Agent Panel message stream, e.g. `context7(query: "Kelly criterion...")` or `sequential_thinking(step 2/5)`. These are rendered identically to memory operation lines — `.ap-tool` class, border-left cyan, data font | 12-1 | Arch §2.3, §4.3 |
| Arch-Agent-6 | Global tools pre-activated for every Interactive session: `read_skill`, `write_memory`, `search_memory`, `read_canvas_context`, `request_tool`, `send_department_mail`. Department-scoped tools activated on top (e.g., `mt5_bridge.read_positions` for Trading, `backtest.run` for Development). Session start tool activation is a backend concern; frontend must render the resulting tool-call events in the `.ap-tool` stream | 12-1 | Arch §16.3, §16.4 |
| Arch-Agent-7 | `RichRenderer.svelte` is a shared component (`components/shared/`) that renders structured agent output blocks inline inside `.ap-agent` message bubbles: markdown tables, code blocks (with syntax highlight), inline chart directives (bar/line via canvas), and file preview links. Agent Panel body wraps agent message content in `<RichRenderer>` | 12-1, 12-3 | Arch §12 |
| Arch-Agent-8 | The Agent Panel SSE scaffold: an `EventSource` connecting to `GET /api/agents/stream` (Contabo node) is initialized on `onMount` and closed on `onDestroy`. In Epic 12 it receives no live events (agent not yet wired). The connection lifecycle (open, error-retry, close) must be architecturally correct so Epic 5 can activate it by hooking into the existing event handlers | 12-1 | Arch §2.2, §Arch-UI-4 |

### Non-Functional Requirements (Epic 12 Scope)

| ID | Requirement | Source |
|----|-------------|--------|
| NFR-PERF-1 | Canvas switches (ActivityBar click): ≤200ms perceived | Architecture §Canvas Transition Budget, PRD |
| NFR-PERF-2 | Tile → sub-page: ≤200ms | Architecture §Canvas Transition Budget |
| NFR-PERF-3 | Loading states: skeleton pulse using `--color-bg-elevated` — never a white flash | Architecture §Canvas Transition Budget |
| NFR-MAINT-1 | All Svelte components under 500 lines — refactor at boundary | PRD Maintainability, Architecture |
| NFR-MAINT-2 | Svelte 5 runes only in new components — `$state`, `$derived`, `$props`, `$effect`. Never `export let`, `$:`, or `writable()` | Architecture §Communication Patterns |
| NFR-MAINT-3 | Lucide icons only (`lucide-svelte`) — never emoji, never other icon libraries | Architecture Forbidden Patterns |
| NFR-MAINT-4 | CSS custom property tokens only — never hardcoded colors, spacing, or dimensions in component styles | Architecture Forbidden Patterns |
| NFR-MAINT-5 | `apiFetch<T>()` for all API calls — never raw `fetch()` in Svelte components | Architecture Forbidden Patterns |
| NFR-MAINT-6 | All date/time values: UTC-aware; display layer converts to local timezone | Architecture §Date/Time |

### Additional Requirements — UX Spec + Visual Reference

| Requirement | Authority |
|------------|-----------|
| `ux-design-directions.html` lines 129–200 are the visual authority for tile grid CSS, tile hover states, and density layout | UX Spec + HTML prototype |
| `ux-design-directions.html` lines 201–216 are the visual authority for Agent Panel structure and CSS | UX Spec + HTML prototype |
| `ux-design-directions.html` lines 218–246 are the visual authority for Workshop canvas sidebar layout | UX Spec + HTML prototype |
| Token values follow UX spec (not HTML prototype values where they conflict). Green = `#00c896`, text = `#e8edf5`, muted = `#5a6a80` | UX Spec §Canonical Token Values |
| **Three primary UI inspirations** (per Mubarak preference 2026-03-22): **Breathing Space** (personal top pick — heavy glass, minimal 4-item StatusBand, 280px/18px tiles), **Ghost Panel** (near-transparent glass, 8-item band, 220px/10px tiles), **Balanced Terminal** (heavy glass, 8-item band, 280px/18px tiles — launched default for first-run) | Memory `project_design_direction_prefs.md` |
| Breathing Space token values: `glassOp:0.78`, `glassBlur:18px`, `tileMinWidth:280px`, `tileGap:18px`, StatusBand 4-item minimal | `ux-design-directions.html` DIR-2 |
| Ghost Panel token values: `glassOp:0.10`, `glassBlur:6px`, `tileMinWidth:220px`, `tileGap:10px`, StatusBand 8-item dense | `ux-design-directions.html` DIR-3 |
| Balanced Terminal token values: `glassOp:0.78`, `glassBlur:18px`, `tileMinWidth:280px`, `tileGap:18px`, StatusBand 8-item dense (launched default) | `ux-design-directions.html` DIR-5 |
| Agent Panel width = **320px** (non-negotiable per UX spec §Global Shell Dimensions) | UX Spec |
| Frosted Terminal glass: shell tier `rgba(8,13,20,0.08)`, content tier `rgba(8,13,20,0.35)`, `backdrop-filter: blur(12px)` | UX Spec + Memory `feedback_glass_aesthetic.md` |
| CRM tile pattern (UX spec §Established Patterns): bounded cards, 2–4 named data points, no overflow on tile face, one-click = sub-page depth | UX Spec |

### FR Coverage Map

| FR | Journey(s) | Story | Coverage Type |
|----|-----------|-------|---------------|
| FR5 | J12, J17, J18, J19 | 12-4 | UI scaffold — regime state visible on Trading canvas tile |
| FR10 | J5, J8, J12, J16 | 12-6 | UI scaffold — Dept Kanban visibility per canvas |
| FR14 | J11, J29, J38 | 12-1 | UI scaffold — Agent Panel session history persistence |
| FR20 | J1, J11, J12, J16, J17 | 12-1 | UI scaffold — Agent Panel canvas-aware dept head binding |
| FR22 | J16, J17, J32 | 12-1 | UI scaffold — interactive chat with Dept Head (SSE wired in Epic 5) |
| FR27 | J2, J12 | 12-4 | UI scaffold — BacktestResultsTile preview tile (data wired); J2 explicitly runs Monte Carlo + Walk-Forward |
| FR29 | J2, J12, J19 | 12-4 | Live data — PaperTradingMonitorTile (real API wired); J3 excluded — fast-track bypasses paper trading entirely |
| FR51 | J20, J27 | 12-5 | Bug fix — Portfolio canvas sub-page routing |
| FR55 | J6, J20, J27 | 12-5 | Bug fix — Portfolio canvas breadcrumb and keyboard nav |
| Arch-UI-1 | J2, J11, J31 | 12-1 | New — Interactive vs Autonomous session UI modes |
| Arch-UI-2 | J2, J5, J14, J31 | 12-3 | New — Workshop 3-column Claude.ai-inspired layout |
| Arch-UI-3 | J4, J23 | 12-3 | Enforcement — Kill Switch placement rules |
| Arch-UI-4 | J1, J11 | 12-1 | New — SSE/WebSocket stream separation |
| Arch-UI-5 | All canvas journeys | 12-3 | Enforcement — data-dept attribute on canvas roots |
| Arch-UI-6 | All canvas journeys | 12-3 | New — CanvasTileGrid shared wrapper |
| Arch-UI-7 | J1, J6, J20 | 12-5 | Bug fix — single source of truth for active canvas |
| Arch-UI-8 | J1, J11, J16, J17 | 12-1 | Integration — canvas context load on new session |
| Arch-UI-9 | All canvases | All | Enforcement — financial numbers always JetBrains Mono |
| NFR-PERF-1/2 | J1, J6, J17 | 12-3, 12-5 | Infrastructure — canvas and tile transition budgets |
| Arch-Agent-1 | J1, J11, J16, J17 | 12-1 | New — three message types in Agent Panel body (.ap-agent/.ap-user/.ap-tool) |
| Arch-Agent-2 | J2, J31 | 12-1 | New — sub-agent status indicators in Autonomous Workflow mode |
| Arch-Agent-3 | J1, J11, J16, J17 | 12-1 | New — canvas context priming on session start (CAG+RAG) |
| Arch-Agent-4 | J11, J12, J16 | 12-1 | New — OPINION node writes visible as .ap-tool lines |
| Arch-Agent-5 | J12, J16, J17 | 12-1 | New — MCP call lines visible in Agent Panel stream |
| Arch-Agent-6 | All agent journeys | 12-1 | New — global tool pre-activation per session |
| Arch-Agent-7 | J11, J12, J17 | 12-1, 12-3 | New — RichRenderer for inline structured output |
| Arch-Agent-8 | J1, J11 | 12-1 | New — SSE EventSource lifecycle scaffold |

---

## Agent↔UI Bidirectional Context Architecture

This section documents the architectural pattern for how agents GAIN CONTEXT from the UI and how they UPDATE the UI. Story 12-1 builds the frontend scaffold for this full loop. Epic 5 activates the live agent side.

### Direction 1: Agent Gains Context FROM the UI

When a new Interactive session starts (user clicks `[+]` in Agent Panel):

```
User clicks [+]
  → canvasContextService.loadCanvasContext(canvasId)
    → GET /api/canvas-context/{canvasId}
      → Backend assembles CanvasContextTemplate:
          base_descriptor: "Risk canvas — physics sensors, prop firm compliance, regime state"
          memory_scope: "risk_dept"
          workflow_namespaces: ["alpha_forge", "risk_pipeline"]
          skill_index: ["risk_assessment_skill", "kelly_engine_skill"]
          required_tools: ["risk_calculator.kelly_size", "risk_calculator.position_limit"]
      → CAG layer: stable canvas identifiers injected once at session start
      → RAG layer: live state (current regime, active positions, news blackout status) fetched JIT
  → Context payload stored in session object
  → On Epic 5 activation: payload injected into agent's system context before first token
```

This is why `read_canvas_context` is in the global activated tool set — every agent in every Interactive session can call it. The canvas context is what makes the same dept head behave differently on Risk canvas vs Research canvas.

### Direction 2: Agent Updates the UI

Once an agent is running (Epic 5), it pushes events to the frontend via SSE:

```
Agent executes tool call (e.g., search_memory, write_memory OPINION, context7 query)
  → SSE event published to GET /api/agents/stream (Contabo)
    → Agent Panel EventSource receives event
      → Event type "tool_call" → rendered as .ap-tool line in .ap-body
      → Event type "agent_message" → rendered as .ap-agent bubble in .ap-body
      → Event type "sub_agent_status" → updates sub-agent status indicator row
      → Event type "task_status" → updates Autonomous Workflow status card stage

In Epic 12: EventSource is opened and lifecycle-managed. No real events arrive.
In Epic 5: Real agent hooks into the existing event handlers — zero frontend re-architecture.
```

### OPINION Node Lifecycle in the UI

OPINION nodes are mandatory reasoning artifacts. Here is their full lifecycle as it relates to the Agent Panel:

```
Agent takes consequential action
  → Agent calls write_memory(node_type="OPINION", {
        action: "recommended reducing EUR exposure 30%",
        reasoning: "Ising model shows high clustering...",
        confidence: 0.87,
        alternatives_considered: ["hold position", "hedge with options"],
        constraints_applied: ["max_drawdown_limit", "correlation_threshold"],
        agent_role: "risk_head"
    })
  → SSE event: { type: "tool_call", tool: "write_memory", args: { node_type: "OPINION", ... } }
  → Agent Panel renders .ap-tool line:
      write_memory(OPINION · confidence=0.87 · action="recommended reducing EUR exposure...")
  → User can click the .ap-tool line to expand full OPINION schema
  → Backend: OPINION node stored in graph memory with SUPPORTED_BY edge to parent OBSERVATION node
  → importance_score: 0.7 default, 0.9 for approval-gate decisions
```

### MCP Stack Visibility

The following MCP servers are in the stack. Their calls all appear as `.ap-tool` lines in the Agent Panel:

| MCP Server | Example .ap-tool line rendered | Category |
|---|---|---|
| `context7` | `context7(query: "Kelly criterion sizing formula")` | External docs lookup |
| `sequential_thinking` | `sequential_thinking(step 2/5 · reasoning...)` | Step-by-step reasoning |
| `web_fetch` | `web_fetch(url: "...")` | Live web content |
| `RAG/CAG MCP (internal)` | `rag_search(query: "EUR correlation spike June 2024")` | Internal knowledge base |
| Internal MT5 compiler | `mt5_compiler.compile(ea: "EA_XAUUSD_V3.mq5")` | Development dept only |
| Internal backtest engine | `backtest.run(ea: "EA_XAUUSD_V3", mode: "monte_carlo")` | Development dept only |

### Memory Architecture in the Agent Panel Context

Graph memory workspace isolation ensures session safety:

| Memory Layer | What it holds | Agent Panel relevance |
|---|---|---|
| `session_status: draft` | All nodes/edges in current session before commit | Shown via tool-call lines as writes happen |
| `session_status: committed` | Committed nodes — visible to future sessions | Session history panel shows committed session summaries |
| `ReflectionExecutor` | Async post-session reflection (5-min debounce) | Not visible in Epic 12 (async background) |
| `all-MiniLM-L6-v2 embeddings` | 384-dim vector for semantic `search_memory` | `search_memory(...)` calls appear as .ap-tool lines |

Global tools available in every Interactive session: `read_skill`, `write_memory`, `search_memory`, `read_canvas_context`, `request_tool`, `send_department_mail`.

### Sub-Agent Hierarchy Visibility

Department hierarchy: `FloorManager (Opus)` → `Dept Head (Sonnet)` → `Sub-agents (Haiku)`

In the Agent Panel:
- **Interactive sessions** (human-initiated): show the Dept Head as the named conversational partner; sub-agents run silently; their status indicators appear if spawned
- **Autonomous Workflow sessions** (Prefect-initiated): show a status card; sub-agent status row shows each active sub-agent as `[role label] [running|idle|blocked badge]`
- Sub-agent types per dept: `mql5_dev`, `backtester`, `data_researcher`, `trade_monitor`, `fill_tracker`, `performance_reporter`

---

## Backend Connections Master Table

All endpoints consumed by Epic 12 frontend components. Verified against `src/api/` codebase.

| Story | Endpoint | Method | Purpose | Server Node | Implementation Status |
|-------|----------|--------|---------|-------------|----------------------|
| 12-1 | `/api/agents/stream` | GET (SSE) | Agent event stream — tool calls, agent messages, sub-agent status, workflow updates | Contabo | ✓ Implemented |
| 12-1 | `/api/canvas-context/template/{canvas_name}` | GET | Canvas context priming — `CanvasContextTemplate` (CAG + RAG) | Contabo | ⚠️ Implemented — but uses `canvas_name` (string: "research", "risk") NOT a UUID. `canvasContextService.loadCanvasContext()` must pass the string name, not the store's UUID |
| 12-1 | `/api/agents/sessions` | GET | Session history list for `[⏱]` panel — per dept head | Contabo | ✓ Router registered — schema needs verification |
| 12-4 | `/api/paper-trading/active` | GET | Active paper trading EAs — name, pair, days, win rate, P&L, status | Cloudzy | ✗ **NOT FOUND** — must be implemented before Story 12-4 |
| 12-4 | `/api/backtest/recent?limit=5` | GET | Recent backtest results — EA, pass/fail, walk-forward Sharpe, date | Contabo | ✓ Implemented in `backtest_endpoints.py` |
| 12-4 | `/api/pipeline-status/stages` | GET | Enhancement loop stage counts | Contabo | ✓ Implemented in `pipeline_status_endpoints.py` |
| 12-6 | `/api/floor-manager/departments/{dept}/tasks` | GET | Department task board — TODO/IN_PROGRESS/BLOCKED/DONE | Contabo | ? Unknown — needs verification in `floor_manager_endpoints.py` |
| 12-6 | `/api/agents/stream` | GET (SSE) | Real-time task status updates for Kanban | Contabo | ✓ Shared SSE channel |

### SSE Event Schema — `/api/agents/stream`

Agent Panel and Kanban tiles both consume this stream. All event types:

| Event Type | Schema | Rendered As |
|---|---|---|
| `tool_call` | `{ type, tool: "write_memory\|search_memory\|context7\|sequential_thinking\|web_fetch\|...", args: {...} }` | `.ap-tool` line (data font, 9px, cyan left-border) |
| `agent_message` | `{ type, content: "string", role: "assistant" }` | `.ap-agent` bubble |
| `user_message` | `{ type, content: "string", role: "user" }` | `.ap-user` bubble (right-aligned) |
| `sub_agent_status` | `{ type, agent_role: "mql5_dev\|backtester\|...", status: "running\|idle\|blocked" }` | Sub-agent status indicator row (Autonomous mode) |
| `task_status` | `{ type, workflow_id, stage: "string", elapsed_s: int }` | Workflow stage card update (Autonomous mode) |

OPINION node writes are `tool_call` events with `tool: "write_memory"` and `args.node_type: "OPINION"`:
```json
{
  "type": "tool_call",
  "tool": "write_memory",
  "args": {
    "node_type": "OPINION",
    "action": "recommended reducing EUR exposure 30%",
    "reasoning": "Ising model shows high clustering...",
    "confidence": 0.87,
    "alternatives_considered": ["hold position", "hedge with options"],
    "constraints_applied": ["max_drawdown_limit"],
    "agent_role": "risk_head",
    "importance_score": 0.7
  }
}
```

### Canvas Context — Endpoint Clarification

**Implemented path:** `GET /api/canvas-context/template/{canvas_name}` (string — e.g. `"research"`, `"risk"`)
**NOT:** `/api/canvas-context/{canvasId}` (UUID)

`canvasContextService.loadCanvasContext(canvasId)` in the frontend must map the store's `canvasId` to the `canvas_name` string before calling. The mapping is 1:1 (e.g., `"risk"` → `"risk"`, `"live-trading"` → `"live-trading"`).

### Pre-Implementation Requirement — Story 12-4

**`GET /api/paper-trading/active`** must exist before Story 12-4 can go to dev. Required response schema:
```json
{
  "items": [
    {
      "ea_name": "EA_XAUUSD_Scalp_V2",
      "pair": "XAUUSD",
      "days_running": 11,
      "win_rate": 0.71,
      "pnl_current": 1234.50,
      "status": "running|paused|failed",
      "started_at": "2026-03-12T10:30:00Z"
    }
  ]
}
```

**Owner:** Trading department backend. Route it via Cloudzy (live trade data path) using `apiFetch` with the trading base URL override.

---

## Epic 12: Global Shell Compliance

**Epic Goal:** Establish the shell grid, design token system, shared component library, and canvas tile grid pattern that all functional epics build upon. Agent Panel right rail, consolidated CSS tokens, `CanvasTileGrid`/`TileCard`/`SkeletonLoader` shared components, Trading canvas functional tiles, Portfolio canvas nav fix, and Department Kanban sub-pages across all department canvases.

**Story Dependency Order:**
```
12-1 (Shell Grid + Agent Panel)   → No blocker. Start immediately.
12-2 (Design Tokens)              → No blocker. Run in parallel with 12-1.
12-3 (Tile Grid + Shared Comps)   → Blocked on 12-2 (tokens must be consolidated first)
12-4 (Trading Canvas Content)     → Blocked on 12-3 (CanvasTileGrid + TileCard must exist)
12-6 (Dept Kanban Sub-Page)       → Blocked on 12-3 (sub-page routing pattern must exist)
12-5 (Portfolio + Nav Fixes)      → Blocked on 12-1 + 12-3
```

---

### Story 12-1: Global Shell — Agent Panel Right Rail + Remove BottomPanel

As a **trader (Mubarak)**,
I want a **collapsible Agent Panel permanently docked in the right rail of the global shell** — displaying the active department head's identity, providing a full chat management scaffold (new session, history, message area, submit), and supporting both Interactive and Autonomous Workflow session modes,
So that I can **communicate with or monitor any Department Head from any canvas without navigating away** from my current work, and so that the BottomPanel no longer consumes vertical space that the canvas workspace and Agent Panel should own.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J1 — The Morning Operator** | *"Mubarak launches QUANTMINDX. The Live Trading canvas loads. The Copilot Bubble shows a dim notification badge. He taps it: 'Research Dept completed overnight cycle — 2 new strategy variants generated.'"* — In Epic 12, the Agent Panel **right rail** replaces the floating Copilot Bubble for dept-head-level context. Notifications from the Research dept surface in the panel without leaving Live Trading. | FR20, FR22 |
| **J11 — Overnight Research Cycle** | *"At 8am: 'Research completed overnight cycle. 3 new hypotheses queued. 1 strategy flagged for Risk review. No action required.'"* — This notification surfaces in the Agent Panel right rail. Mubarak sees it on whatever canvas he's on — no click-away required. | FR20, FR14 |
| **J12 — The Parameter Sweep** | *"Copilot notification: '512 combinations tested. 4 survived correlation filter. Estimated paper trade review: 5 days.'"* — Surfaced in Agent Panel without leaving Development canvas. | FR20 |
| **J16 — The Code Review** | *"He types to Copilot: 'Lot sizing should use Kelly Engine output, not fixed lots.' Development receives the feedback via department mail and generates V4 in 8 minutes."* — This conversation happens in the Agent Panel on the Development canvas. Dept badge shows "DEVELOPMENT" (cyan). | FR22 |
| **J17 — The Physics Lens** | *"Mubarak asks: 'What does the physics engine say about risk right now?' Copilot synthesizes all three sensors: reduce EUR exposure 30% preemptively."* — Typed in the Agent Panel on Risk canvas. Dept badge shows "RISK" (red). | FR22, FR20 |
| **J29 — Dept Mail Morning Read** | *"StatusBand: mail badge 7 unread. Department Mail inbox tells the overnight story."* — The Agent Panel is the entry point for dept-head-level communication; mail badge on StatusBand links to Agent Panel session history. | FR14 |
| **J32 — The Department Council** | *"Mubarak: 'I want to hear from Research, Risk, Portfolio, and Development on whether we should scale to 35 active strategies.'"* — Multi-dept broadcast initiated from Agent Panel or Workshop; each dept head responds individually. Agent Panel must support switching between dept sessions. | FR22 |
| **J2 — Alpha Forge Trigger** | *"The Copilot confirms: 'Starting Alpha Forge Loop. Research Dept will extract the strategy logic. I'll notify you when Development has a draft.'"* — Autonomous Workflow session mode in Agent Panel (status card only, read-only) represents the running workflow; Interactive session is for human conversation. | Arch-UI-1 |

#### FR Enablement

- **FR20** (canvas-aware Copilot): Agent Panel resolves dept head from `activeCanvasStore.id` — different canvas = different dept badge, different context
- **FR22** (direct dept head chat): Interactive session UI — full chat input, message area, submission
- **FR14** (persistent memory scaffold): Session history panel (`[⏱]` button) — past sessions per dept head; Epic 5 wires real persistence
- **Arch-UI-1** (two session modes): Interactive (chat input active) vs Autonomous Workflow (read-only status card — used when a running workflow is showing progress)
- **Arch-UI-4** (SSE/WebSocket separation): Agent Panel is wired to SSE channel only; never touches WebSocket
- **Arch-UI-8** (canvas context load): When a new interactive session starts, `canvasContextService.loadCanvasContext(canvasId)` is called to prime the context
- **Arch-Agent-1** (three message types): `.ap-agent`, `.ap-user`, `.ap-tool` — all three rendered in `.ap-body`
- **Arch-Agent-2** (sub-agent indicators): Autonomous Workflow status card shows sub-agent status row
- **Arch-Agent-3** (canvas context priming): `[+]` → `loadCanvasContext(canvasId)` → `GET /api/canvas-context/{canvasId}`
- **Arch-Agent-4** (OPINION node display): OPINION writes surface as `.ap-tool` lines with expandable schema
- **Arch-Agent-5** (MCP call display): `context7`, `sequential_thinking`, `web_fetch`, internal RAG MCP all render as `.ap-tool` lines
- **Arch-Agent-6** (global tools): Session start tool activation is backend; frontend renders the resulting event stream
- **Arch-Agent-7** (RichRenderer): Agent messages with structured output blocks use `<RichRenderer>` inside `.ap-agent` bubble
- **Arch-Agent-8** (SSE lifecycle): `EventSource` to `GET /api/agents/stream` opened on mount, closed on destroy

#### Implementation Approach

**NEW files to create:**
- `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` — canonical; NOT the deprecated `agent-panel/AgentPanel.svelte`

**Agent Panel internal structure** (from `ux-design-directions.html` lines 201–216 + architecture):
```
.agent-panel (320px, flex column, border-left: 1px solid var(--c-border))
  .ap-header (36px)
    .ap-dept badge         — tinted by current canvas dept accent
    .ap-spacer
    .ap-icon-btn [+]       — new interactive session (Plus icon)
    .ap-icon-btn [⏱]      — session history toggle (History icon)
    .ap-icon-btn [←]       — collapse (ChevronRight icon)
  .ap-session-history (conditional: showSessionHistory)
    — scrollable list: title, dept label, date (local TZ), status badge
  .ap-autonomous-status (conditional: has autonomous sessions)
    — workflow name, current stage, elapsed time
    — sub-agent status row: [role] [running|idle|blocked] per active sub-agent
  .ap-body (flex: 1, scrollable, gap: 7px)
    .ap-agent messages     — rgba(0,170,204,0.05) bg, cyan border
    .ap-user messages      — right-aligned, white-tinted, max-width 88%
    .ap-tool lines         — border-left: 2px solid rgba(0,170,204,0.2), data font 9px
                            (OPINION writes, write_memory, search_memory, MCP calls)
    .ap-chips              — suggestion chips row
  .ap-footer (conditional: interactive session active)
    .ap-input              — chat input field
```

**SSE EventSource lifecycle:**
```typescript
// In AgentPanel.svelte onMount:
let eventSource: EventSource | null = null;

function openSSE(sessionId: string) {
  eventSource = new EventSource(`/api/agents/stream?session=${sessionId}`);
  eventSource.onmessage = (e) => handleStreamEvent(JSON.parse(e.data));
  eventSource.onerror = () => { /* retry logic */ };
}

function handleStreamEvent(event: AgentStreamEvent) {
  if (event.type === 'tool_call') appendToolLine(event);
  if (event.type === 'agent_message') appendAgentMessage(event);
  if (event.type === 'sub_agent_status') updateSubAgentStatus(event);
  if (event.type === 'task_status') updateWorkflowStage(event);
}

onDestroy(() => eventSource?.close());
```

**Canvas context priming on new session:**
```typescript
async function createNewSession() {
  const context = await canvasContextService.loadCanvasContext(activeCanvas);
  const session: AgentSession = {
    id: crypto.randomUUID(),
    type: 'interactive',
    deptHead: CANVAS_DEPT_HEAD[activeCanvas],
    canvasContext: context,  // Epic 5 injects this into agent
    messages: [],
    createdAt: new Date().toISOString()
  };
  sessions = [...sessions, session];
  activeSessionId = session.id;
  openSSE(session.id);
}
```

**Backend endpoints used by Story 12-1:**
| Endpoint | Method | Used for |
|---|---|---|
| `/api/canvas-context/{canvasId}` | GET | Canvas context priming on new session |
| `/api/agents/stream` | GET (SSE) | Agent event stream (scaffold in Epic 12, live in Epic 5) |
| `/api/agents/sessions` | GET | Session history list for `[⏱]` panel |

**TARGETED EDITS to existing files:**
- `quantmind-ide/src/routes/+page.svelte` — grid restructure (3 columns), import new AgentPanel, remove BottomPanel import + render, bind collapsed state
- `quantmind-ide/src/app.css` — add `--agent-panel-width: 320px` to `:root`, add `.agent-panel-collapsed` grid class

**DO NOT TOUCH:**
- `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte` — deprecated, leave untouched
- `quantmind-ide/src/lib/components/BottomPanel.svelte` — file retained on disk; only removed from shell grid
- All Settings components

#### Acceptance Criteria

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
- **And** all three match `ux-design-directions.html` lines 209–211 exactly

**AC 12-1-13: OPINION node writes render as `.ap-tool` lines**
- **Given** an agent writes an OPINION node to graph memory
- **When** the SSE event `{ type: "tool_call", tool: "write_memory", args: { node_type: "OPINION", confidence: 0.87, action: "..." } }` arrives
- **Then** a `.ap-tool` line renders: `write_memory(OPINION · confidence=0.87 · action="recommended reducing EUR exposure...")`
- **And** clicking the line expands a detail view showing full OPINION schema: `{action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role}`

**AC 12-1-14: Memory operation lines**
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
- **Given** an Autonomous Workflow session is displayed in the Agent Panel
- **When** sub-agents are spawned by the dept head
- **Then** a sub-agent status row is visible beneath the workflow stage label
- **And** each active sub-agent shows: role label (e.g., `mql5_dev`, `backtester`) + status badge
- **And** status badge colours: running = `--color-accent-cyan`, idle = `--color-text-muted`, blocked = `--color-accent-amber`

**AC 12-1-17: RichRenderer renders structured content inside agent bubbles**
- **Given** an agent message contains a structured block (markdown table, code block, chart directive)
- **When** the `.ap-agent` bubble renders
- **Then** the structured block is rendered via `<RichRenderer>` component inline within the bubble
- **And** code blocks use `var(--font-data)` with a slightly elevated background
- **And** tables use `var(--font-data)` for numeric cells, `var(--font-ambient)` for header labels

**AC 12-1-18: Canvas context primed on new Interactive session**
- **Given** the user clicks `[+]` to create a new Interactive session
- **When** session creation fires
- **Then** `canvasContextService.loadCanvasContext(activeCanvas)` is called
- **And** `GET /api/canvas-context/template/{canvas_name}` is called (e.g. `template/risk`, `template/research`) — note: endpoint uses `canvas_name` string, NOT a UUID
- **And** the returned `CanvasContextTemplate` object is stored on the session object (for Epic 5 to inject into agent)

**AC 12-1-19: SSE EventSource lifecycle — open on session create, close on destroy**
- **Given** a new Interactive session is created
- **When** `createNewSession()` runs
- **Then** an `EventSource` is opened to `GET /api/agents/stream?session={sessionId}` (Contabo URL)
- **And** in Epic 12, no meaningful events arrive (connection opens cleanly and stays idle)
- **And** on component destroy (`onDestroy`), `eventSource.close()` is called — no dangling connections
- **And** the WebSocket (Cloudzy — live trading P&L) is completely unaffected by Agent Panel SSE state

**Out of Scope for 12-1:** AI streaming response content (Epic 5), dept head routing logic (Epic 7), real session persistence to DB (Epic 5), suggestion chips (Epic 5/7)

---

### Story 12-2: Design Token Consistency Pass

As a **trader (Mubarak)**,
I want **all visual styles across every canvas and component to draw from a single, consistent Frosted Terminal token system** — eliminating the competing OKLCH variable layer and the mismatched colour values that cause visual noise,
So that every canvas renders with the correct colours, glass opacity, typography, and spacing that match the UX spec exactly, and so that any developer building future components has one authoritative token reference to follow.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J7 — New User Setup** | *"He sets risk defaults. Picks Frosted Terminal theme, anime wallpaper, lo-fi volume 30%."* — The token system must be coherent so that theme switching in Settings → Appearance applies consistently across every component. If two token systems exist simultaneously, half the UI ignores the theme change. | Visual foundation |
| **J1 — Morning Operator** | *"The StatusBand pulses: London session OPENING, regime TREND_STABLE, 12 active bots, daily P&L +$340."* — The amber/green/cyan status colours must be exactly as specified. Token mismatches cause incorrect colour rendering on the highest-frequency view in the product. | Visual foundation |
| **J10 — Challenge Mode** | *"StatusBand gains a challenge progress line."* — The challenge progress indicator relies on `--color-accent-amber` and `--color-accent-green` being precisely correct. | Visual foundation |
| **J17 — Physics Lens** | *"Risk canvas: Ising Model shows high EUR correlation clustering."* — Risk canvas sensor tiles use `--color-accent-red` for alert states. Token ambiguity between `--accent-danger` (old) and `--color-accent-red` (canonical) must be resolved. | Visual foundation |
| **J18 — Multi-Type Orchestration** | *"Strategy router shows a color-coded grid by type + state. Mubarak glances 30 seconds — understands the full system state."* — Colour-coded strategy types rely on a coherent token set. | Visual foundation |

#### FR Enablement

- **NFR-MAINT-4** (CSS token system): Single `:root` token block — zero competing systems
- **Arch-UI-9** (financial numbers): `--font-data` must be defined as canonical for all financial figures
- **NFR-PERF-3** (no white flash): `--color-bg-elevated` defined for skeleton loaders

#### Implementation Approach

**TARGETED EDITS only:**
- `quantmind-ide/src/app.css` — remove OKLCH vars; expand Frosted Terminal token block; add spacing scale, type scale, tile density defaults, wallpaper var, dept accent overrides, 4 theme preset blocks
- Any `.svelte` file discovered using removed OKLCH vars — replace with Frosted Terminal equivalents

**Token replacement map:**

| Old OKLCH token | New canonical Frosted Terminal token |
|---|---|
| `--bg-primary` | `--color-bg-base` |
| `--bg-secondary` | `--color-bg-surface` |
| `--bg-tertiary` | `--color-bg-elevated` |
| `--bg-input` | `--color-bg-elevated` |
| `--bg-glass` | `--glass-content-bg` |
| `--text-primary` | `--color-text-primary` |
| `--text-muted` | `--color-text-muted` |
| `--border-subtle` | `--color-border-subtle` |
| `--border-medium` | `--color-border-medium` |
| `--accent-primary` | `--color-accent-cyan` (default) |
| `--accent-danger` / `--color-danger` | `--color-accent-red` |
| `--accent-success` | `--color-accent-green` |
| `--accent-warning` | `--color-accent-amber` |

**DO NOT TOUCH:**
- Settings sub-panels (theme switching logic in `AppearancePanel.svelte` stays as-is)
- Any component not referencing the old OKLCH vars

#### Acceptance Criteria

**AC 12-2-1: No OKLCH vars in `:root`**
- **Given** `app.css` is opened
- **When** a developer searches for `oklch(`
- **Then** zero results are found in the `:root` block

**AC 12-2-2: Canonical colour values at default theme**
- **Given** the app loads with no theme override (Balanced Terminal default)
- **When** any canvas is rendered
- **Then** background resolves to `#080d14`, cyan = `#00d4ff`, amber = `#f0a500`, green = `#00c896`, red = `#ff3b3b`, text primary = `#e8edf5`, text muted = `#5a6a80`

**AC 12-2-3: Dept accent overrides resolve correctly**
- **Given** Research canvas is active (root element carries `data-dept="research"`)
- **When** dept-accented elements render
- **Then** `--dept-accent` resolves to `--color-accent-amber` (#f0a500)
- **And** `data-dept="risk"` resolves to `--color-accent-red`, `data-dept="development"` to `--color-accent-cyan`

**AC 12-2-4: Zero components using removed OKLCH vars**
- **Given** all `.svelte` files are scanned
- **When** a developer searches for `var(--bg-primary)`, `var(--accent-primary)`, `var(--text-primary)`, `var(--color-danger)`, `var(--accent-danger)`
- **Then** no results are found — all replaced with Frosted Terminal equivalents

**AC 12-2-5: Theme presets functional after token rename**
- **Given** the user switches to "Ghost Panel" in Settings → Appearance
- **When** the theme applies
- **Then** `--tile-min-width` = 220px, `--tile-gap` = 10px (compact tiles — per `ux-design-directions.html` DIR-3)
- **And** glass opacity drops to near-transparent (0.10) — wallpaper shows through
- **And** the AppearancePanel theme switch continues to work without error

**AC 12-2-6: Balanced Terminal default density (launched default)**
- **Given** no theme override is active (Balanced Terminal is the out-of-box default)
- **When** a tile grid renders
- **Then** `--tile-min-width` = 280px, `--tile-gap` = 18px — per `ux-design-directions.html` DIR-5 exact values
- **And** Breathing Space override (`data-theme="breathing-space"`) sets 4-item StatusBand density, same tile sizing (280px / 18px)
- **And** Ghost Panel override (`data-theme="ghost-panel"`) sets `--tile-min-width` = 220px, `--tile-gap` = 10px, near-transparent glass (`glassOp:0.10`)

**AC 12-2-7: No `--color-danger` token**
- **Given** `app.css` is inspected
- **When** searching for `--color-danger`
- **Then** no such property is defined — canonical token is `--color-accent-red`

**AC 12-2-8: Full spacing + typography scale defined**
- **Given** a developer needs spacing or type tokens
- **When** they check `app.css` `:root`
- **Then** `--space-1` through `--space-12`, `--text-xs` through `--text-2xl`, `--font-data`, `--font-heading`, `--font-body`, `--font-ambient` are all present and correct

**Out of Scope for 12-2:** Font loading (referenced by CSS vars; loaded via existing font import), Monaco syntax highlighting vars (deferred to Epic 8)

---

### Story 12-3: Tile Grid Pattern — Shared Components + All 9 Canvases

As a **trader (Mubarak)**,
I want **every canvas to display a structured, CRM-style tile grid** using the shared `CanvasTileGrid` layout wrapper and `TileCard` glass tiles — replacing all blank placeholder screens — with skeleton loaders for tiles whose data arrives in later epics, and a back-button breadcrumb for sub-page navigation,
So that I can **see organised, labelled information at a glance on every canvas** the moment I switch to it, with consistent visual language across all 9 canvases, and no canvas that looks "broken" or incomplete while it waits for its functional epic.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J1 — Morning Operator** | *"He doesn't have to click anything to see this."* — Live Trading is the home screen; the tile grid must be immediately readable on open. The pattern established here (CanvasTileGrid + TileCard) carries to all other canvases. | NFR-PERF-1 |
| **J2 — Alpha Forge Trigger** | *"He opens the Workshop canvas, types to the Copilot: 'I want to build a supply/demand scalping EA for this video.'"* — Workshop canvas is the FloorManager home — a Claude.ai-inspired 3-column layout, NOT a tile grid. Left nav panel (200px): New Chat, History, Projects, Memory, Skills. | Arch-UI-2 |
| **J5 — Weekly War Room** | *"Mubarak opens the Workshop canvas: 'Let's plan this week.'"* — Workshop's main conversation area with left navigation panel. | Arch-UI-2 |
| **J6 — Portfolio Audit** | *"Portfolio canvas shows the monthly performance report."* — Portfolio tile grid with LivePnL, Allocation, Correlation Matrix, and Trading Journal tiles visible at a glance before any sub-page drill. | FR55, NFR-PERF-2 |
| **J10 — Challenge Mode** | *"StatusBand gains a challenge progress line."* — Risk canvas tile grid shows the challenge progress tile, Kelly Engine state, physics sensor summary — Mubarak glances at Risk canvas to get the challenge picture instantly. | FR10 |
| **J12 — Parameter Sweep** | *"Copilot notification: '512 combinations tested.'"* — Development canvas tile grid surfaces backtest queue status, EA variants, Alpha Forge pipeline tiles at a glance. | NFR-PERF-2 |
| **J17 — Physics Lens** | *"Risk canvas: Ising Model shows high EUR correlation clustering. HMM shows 28% BREAKOUT probability rising. Lyapunov Exponent up 40% in 6 hours."* — These three sensors surface as tiles on the Risk canvas tile grid immediately on canvas load — no digging required. | Arch-UI-5 |
| **J19 — Paper Trade Graduation** | *"A structured dossier arrives — win rate, avg R, max drawdown, Sharpe, regime-specific performance breakdown."* — The Trading canvas tile grid (Story 12-4) is where Mubarak first sees the paper trading pipeline. | FR29 |
| **J4 / J23 — Crisis Recovery / Black Swan** | Kill Switch appears in **TopBar ONLY**. These journeys confirm the kill switch placement rule — never inside a canvas tile or on the canvas surface. | Arch-UI-3 |
| **J31 — Batch Trigger** | *"Workshop canvas shows a real-time workflow status panel with progress per workflow."* — Workshop canvas main area renders workflow status cards when autonomous workflows are active (visible via Autonomous Workflow session mode in Agent Panel, hidden when Workshop is active). | Arch-UI-2 |

#### FR Enablement

- **NFR-PERF-1** (≤200ms canvas switches): `CanvasTileGrid` renders skeleton immediately; no async blocking on mount
- **NFR-PERF-2** (instant skeleton state): `SkeletonLoader` with `--color-bg-elevated` pulse — zero white flash
- **NFR-MAINT-3** (Lucide icons): All tile header icons, nav icons, and action icons from `lucide-svelte`
- **Arch-UI-2** (Workshop 3-column): Workshop uses unique layout, not CanvasTileGrid
- **Arch-UI-3** (Kill Switch placement): No kill switch elements anywhere in canvas tiles or CanvasTileGrid
- **Arch-UI-5** (data-dept attribute): All canvas roots carry `data-dept` for CSS token dept accent resolution
- **Arch-UI-6** (CanvasTileGrid mandate): All 8 non-Workshop canvases use `CanvasTileGrid` as root layout

#### Implementation Approach

**NEW files to create:**
- `quantmind-ide/src/lib/components/shared/CanvasTileGrid.svelte`
- `quantmind-ide/src/lib/components/shared/TileCard.svelte`
- `quantmind-ide/src/lib/components/shared/GlassSurface.svelte`
- `quantmind-ide/src/lib/components/shared/SkeletonLoader.svelte`
- `quantmind-ide/src/lib/components/shared/Breadcrumb.svelte`
- `quantmind-ide/src/lib/components/shared/RichRenderer.svelte` — renders structured agent output inline (tables, code blocks, inline chart directives, file preview links) inside `.ap-agent` message bubbles and any canvas that surfaces agent-generated content
- `quantmind-ide/src/lib/components/shared/NotificationTray.svelte` — notification overlay for agent-generated alerts surfacing from the SSE stream (e.g., "Research cycle complete", "Risk threshold breached")
- `quantmind-ide/src/lib/components/shared/ConfirmModal.svelte` — confirmation modal for destructive actions (kill switch confirmation, session close with unsaved context, EA deletion)
- `quantmind-ide/src/lib/components/shared/FilePreviewOverlay.svelte` — overlay for agent-surfaced file references (EA source files, strategy documents, knowledge base articles linked in agent output)
- `quantmind-ide/src/lib/components/workshop/` directory (Workshop canvas left-nav layout)

**TARGETED EDITS to existing canvas files** (wrap in CanvasTileGrid, add skeleton TileCards):

| Canvas | Edit type | Tile content |
|--------|-----------|-------------|
| `LiveTradingCanvas.svelte` | Wrap in `<CanvasTileGrid>` + add `data-dept` | Existing GlassTile instances untouched |
| `ResearchCanvas.svelte` | Replace placeholder | `AlphaForgeEntryTile`, `KnowledgeBaseTile`, `VideoIngestTile`, `HypothesisPipelineTile` (skeletons) |
| `DevelopmentCanvas.svelte` | Replace placeholder | `EALibraryTile`, `AlphaForgePipelineTile`, `BacktestQueueTile` (skeletons) |
| `RiskCanvas.svelte` | Replace placeholder | `KellyEngineTile`, `PhysicsSensorsTile`, `PropFirmComplianceTile`, `ValidationQueueTile` (skeletons) |
| `TradingCanvas.svelte` | Full implementation (Story 12-4) | — |
| `PortfolioCanvas.svelte` | Wrap + add skeleton tiles | `LivePnLTile`, `AllocationTile`, `CorrelationMatrixTile`, `TradingJournalTile` (skeletons) |
| `SharedAssetsCanvas.svelte` | Replace placeholder | `DocsLibraryTile`, `StrategyTemplatesTile`, `IndicatorsTile`, `SkillsTile`, `FlowComponentsTile` (skeletons) |
| `WorkshopCanvas.svelte` | Replace with 3-column layout | Left nav panel + main conversation + MorningDigest on first load |
| `FlowForgeCanvas.svelte` | Wrap header only | Existing PrefectKanban as body content |

**DO NOT TOUCH:**
- `live-trading/GlassTile.svelte` (kept for backward compat — new canvases use `shared/TileCard.svelte`)
- `BottomPanel.svelte`
- Settings panel or sub-panels (12-3-B3 **excluded** per Mubarak direction)
- Any TopBar kill switch controls

#### Acceptance Criteria

**AC 12-3-1: No blank placeholder canvases**
- **Given** any of the 9 canvases is loaded
- **When** the canvas renders at default view
- **Then** structured content is visible — tile grid for 7 canvases, 3-column layout for Workshop, Prefect Kanban for FlowForge
- **And** no `<CanvasPlaceholder>` renders, no raw "Coming Soon" text exists on any canvas

**AC 12-3-2: Canvas title — Syne 800 at 20px**
- **Given** any canvas with a `CanvasTileGrid` header
- **When** the canvas title is inspected
- **Then** it uses `var(--font-heading)` (Syne), font-weight 800, `var(--text-xl)` (20px), colour `var(--color-text-primary)`

**AC 12-3-3: Tile hover state matches visual reference**
- **Given** a `TileCard` is rendered
- **When** it is hovered
- **Then** border transitions to `rgba(255,255,255,0.13)` and a subtle background lift occurs — matching `ux-design-directions.html` lines 129–200 tile hover pattern

**AC 12-3-4: Balanced Terminal density at default**
- **Given** no theme override is active, viewport is 1440px
- **When** a tile grid renders
- **Then** tiles are minimum 280px wide with 12px gap (Dense/Balanced Terminal per UX spec)

**AC 12-3-5: Ghost Panel density override**
- **Given** "Ghost Panel" theme is applied
- **When** the tile grid renders
- **Then** minimum tile width is 320px and gap is 16px (Standard tier)

**AC 12-3-6: CRM tile pattern — no overflow on tile face**
- **Given** a `TileCard` contains data that exceeds 4 named data points
- **When** the tile renders
- **Then** only 2–4 data points are visible on the tile face — overflow is never shown on the tile
- **And** a "→ view detail" hint appears on hover for tiles with sub-page depth

**AC 12-3-7: Financial typography enforced on TileCards**
- **Given** any `TileCard` displays numeric financial data
- **When** inspected
- **Then** all P&L values, risk scores, lot sizes, and timestamps use `var(--font-data)` (JetBrains Mono)
- **And** all section label headings use `var(--font-ambient)` (Fragment Mono, 10px uppercase) — CRM named-section pattern

**AC 12-3-8: New tiles import from `shared/TileCard`**
- **Given** any newly created canvas tile component
- **When** its import statement is checked
- **Then** it imports from `$lib/components/shared/TileCard.svelte`
- **And** `live-trading/GlassTile.svelte` is only imported by Live Trading canvas components

**AC 12-3-9: Sub-page breadcrumb navigation**
- **Given** a canvas with sub-page support is displaying a sub-page
- **When** the canvas header is inspected
- **Then** a back button is visible and correctly labelled
- **And** clicking back restores the tile grid in ≤200ms (NFR-PERF-2) with no stale sub-page state

**AC 12-3-10: data-dept attribute on all canvas roots**
- **Given** any of the 9 canvas root elements
- **When** the DOM is inspected
- **Then** a `data-dept` attribute is present matching the canvas's department (e.g., `data-dept="research"`, `data-dept="risk"`)
- **And** the CSS dept accent token `--dept-accent` resolves to the correct accent colour for that canvas

**AC 12-3-11: Skeleton tiles carry owning epic badge**
- **Given** a skeleton tile is rendered for content not yet built
- **When** the tile is viewed
- **Then** a badge showing the owning epic (e.g., "Epic 4", "Epic 7") is visible
- **And** `SkeletonLoader` uses `--color-bg-elevated` pulse animation — never a white flash

**AC 12-3-12: Workshop canvas — 3-column layout with Lucide nav**
- **Given** Workshop canvas (slot 8) is active
- **When** it renders
- **Then** a 200px left sidebar panel is visible with 5 items — each with a Lucide icon: New Chat (`Plus`), History (`MessageSquare`), Projects (`GitBranch`), Memory (`Brain`), Skills (`Zap`)
- **And** no emoji is present anywhere on the Workshop canvas
- **And** the right-rail Agent Panel is hidden while Workshop is active (Workshop IS the full-screen Copilot)

**AC 12-3-13: Kill Switch not present in any canvas tile**
- **Given** all 9 canvases are inspected
- **When** the source of each is checked
- **Then** no kill switch button, kill switch import, or kill switch state is present inside any canvas component or TileCard
- **And** the Trading Kill Switch exists only in `TopBar.svelte`

**Out of Scope for 12-3:** Settings panel refinement (excluded per Mubarak direction), functional tile data for skeleton canvases (each functional epic owns its tile data), Workshop morning digest data (MorningDigestCard component wrapped as-is; Epic 5/7 for real data)

---

### Story 12-4: Trading Canvas (Slot 5) — Paper Trading & Backtesting Content

As a **trader (Mubarak)**,
I want the **Trading canvas (slot 5) to display a real tile grid** — showing active paper trading EAs, recent backtest results, and the Alpha Forge enhancement loop pipeline stages — replacing the broken placeholder that incorrectly points to Epic 3 (Live Trading),
So that I can **see the paper trading pipeline at a glance**, know which EAs are in monitoring, click into backtest detail, and track the enhancement loop stage counts — without having to open a separate view or ask the Copilot.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J19 — Paper Trade Graduation** | *"EA_XAUUSD_Scalp_V2 completes 11 days of paper trading. A structured dossier arrives — win rate, avg R, max drawdown, Sharpe, regime-specific performance breakdown, spread sensitivity log, correlation to live portfolio, Risk Dept sign-off, Development Dept notes."* — The Trading canvas is where Mubarak sees this EA in the PaperTradingMonitorTile and clicks through to the dossier (Epic 7/8 fills dossier content). | FR29 |
| **J12 — Parameter Sweep** | *"512 combinations tested. 4 survived correlation filter. Estimated paper trade review: 5 days."* — The 4 surviving EAs enter paper monitoring. They appear in the PaperTradingMonitorTile on Trading canvas. The EAPerformanceTile shows the full pipeline count: "4 Backtesting · 2 at SIT Gate · 3 Paper Monitoring · 1 Awaiting Approval." | FR29, FR5 |
| **J2 — Alpha Forge Trigger** | *"72 hours later: EA_EURUSD_SD_V2 has been paper trading for 3 days — 71% win rate, 1.6R avg, max drawdown 0.8%."* — This EA is visible in the PaperTradingMonitorTile on Trading canvas: EA name, pair, days running, win rate, P&L. | FR29 |
| **J3 — Geopolitical Setup** | *"The fast-track workflow fires — no backtest, template-based, direct to live with a 7-day expiry tag."* — J3 is the fast-track path that **explicitly bypasses paper trading** — it goes directly to live. J3 is therefore **not** a FR29 journey. It is noted here as the contrast path: the Trading canvas `PaperTradingMonitorTile` would NOT show a J3-originated strategy because J3 skips the paper gate. | Architecture note — not FR29 |
| **J18 — Multi-Type Orchestration** | *"Strategy router shows a color-coded grid by type + state."* — Regime-per-strategy visibility on the EAPerformanceTile (pipeline stages include regime state per EA). | FR5 |

#### FR Enablement

- **FR29** (paper trading monitoring): `PaperTradingMonitorTile` fetches `GET /api/paper-trading/active` and displays live EA monitoring data
- **FR27** (backtest matrix display — preview): `BacktestResultsTile` fetches `GET /api/backtest/recent?limit=5` — shows last 5 results
- **FR5** (regime state per strategy): `EAPerformanceTile` fetches `GET /api/pipeline-status/stages` — shows enhancement loop stage counts
- **Arch-UI-6** (CanvasTileGrid mandate): Trading canvas uses `CanvasTileGrid` wrapper from 12-3

#### Implementation Approach

**FULL REPLACEMENT** (existing file is a one-line placeholder with wrong epic reference):
- `quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte` — replaced entirely with `CanvasTileGrid` implementation

**NEW files to create** in `components/trading/tiles/`:
- `PaperTradingMonitorTile.svelte` — `size="lg"` (spans 2 cols), fetches `/api/paper-trading/active`
- `BacktestResultsTile.svelte` — `size="md"`, fetches `/api/backtest/recent?limit=5`
- `EAPerformanceTile.svelte` — `size="xl"` (full width), fetches `/api/pipeline-status/stages`
- `EAPerformanceDetailPage.svelte` — sub-page placeholder (Epic 7/8 fills content)
- `BacktestDetailPage.svelte` — sub-page placeholder (Epic 8 fills content)

**Pre-condition for Story 12-4:** `GET /api/paper-trading/active` must exist before dev begins. It was not found in the codebase scan. If it does not exist, implement it in `src/api/trading/routes.py` or a new `src/api/paper_trading_endpoints.py` before writing any tile code. See Backend Connections Master Table for required response schema.

**DO NOT TOUCH:**
- `CanvasPlaceholder.svelte` (file kept; TradingCanvas simply stops using it)
- `backtest_endpoints.py` and `pipeline_status_endpoints.py` — consumed as-is; verify schema matches ACs before wiring

#### Acceptance Criteria

**AC 12-4-1: No placeholder — correct canvas identity**
- **Given** the Trading canvas (slot 5) is loaded via keyboard shortcut "5"
- **When** the canvas renders
- **Then** a `CanvasTileGrid` with title "Trading" (or "Trading Department") is visible
- **And** no `<CanvasPlaceholder>` exists in the rendered output
- **And** no `epicNumber={3}` prop reference exists anywhere in the component file (wrong-epic reference removed)

**AC 12-4-2: Paper Trading Monitor tile — empty state (calm)**
- **Given** `GET /api/paper-trading/active` returns an empty list or 404
- **When** the `PaperTradingMonitorTile` renders
- **Then** a calm neutral empty state is shown: "No EAs in paper monitoring phase — Alpha Forge feeds this when EAs reach the paper gate"
- **And** the tile renders correctly — not broken, not erroring

**AC 12-4-3: Paper Trading Monitor tile — populated state**
- **Given** `GET /api/paper-trading/active` returns active paper trade entries
- **When** the tile renders
- **Then** each EA entry shows: EA name, pair, days running, win rate, current P&L
- **And** all numeric values use `var(--font-data)` (JetBrains Mono)
- **And** status dots: running = `--color-accent-cyan`, paused = `--color-accent-amber`, failed = `--color-accent-red`

**AC 12-4-4: Backtest Results tile**
- **Given** `GET /api/backtest/recent?limit=5` returns results
- **When** the `BacktestResultsTile` renders
- **Then** up to 5 runs show: EA name, pass/fail indicator, walk-forward Sharpe, date (local timezone display)
- **And** clicking the tile sets `currentSubPage = 'backtest-detail'` and a back button appears

**AC 12-4-5: Enhancement Loop tile — all-zero state is neutral**
- **Given** `GET /api/pipeline-status/stages` returns all zero counts
- **When** the `EAPerformanceTile` renders
- **Then** each stage badge shows "0" in neutral grey (`--color-text-muted`) — not a broken or error state
- **And** stage badges use Fragment Mono (10px caps) per CRM label pattern

**AC 12-4-6: Enhancement Loop tile — populated state**
- **Given** `/api/pipeline-status/stages` returns non-zero counts
- **When** the tile renders
- **Then** counts display as: "N Backtesting · N at SIT Gate · N Paper Monitoring · N Awaiting Approval"
- **And** each running stage uses `--color-accent-cyan` for its count indicator

**AC 12-4-7: Sub-page routing**
- **Given** a tile on Trading canvas is clicked
- **When** the sub-page view activates
- **Then** `currentSubPage` changes from `'grid'` to the appropriate sub-page identifier
- **And** `CanvasTileGrid` `showBackButton` becomes true and back button appears in header
- **And** clicking back sets `currentSubPage = 'grid'` and tile grid is restored in ≤200ms

**AC 12-4-8: API errors handled as empty state**
- **Given** any of the three backend API calls returns 404 or network error
- **When** the relevant tile renders
- **Then** an appropriate empty state message is shown — never a thrown exception or broken DOM

**Out of Scope for 12-4:** EA performance dossier content (Epic 7/8), backtest viewer sub-page content (Epic 8), A/B race board (Epic 8), strategy lifecycle management, regime routing controls

---

### Story 12-5: Portfolio Canvas + Cross-Canvas Navigation Fixes

As a **trader (Mubarak)**,
I want **Portfolio canvas sub-page routing to work using local Svelte 5 state** (removing the legacy `navigationStore` dependency), and **all 9 canvas switches via keyboard shortcuts 1–9 and ActivityBar clicks to resolve from a single source of truth** (`activeCanvasStore`), with all 5 StatusBand click targets correctly wired,
So that I can **navigate between canvases and into Portfolio sub-pages without stale routing state, undefined errors, or two competing navigation systems** fighting each other.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J1 — Morning Operator** | *"He clicks through to the Development canvas, sees the two variants listed with their backtest summary scores. Approves one for paper trading. Returns to Live Trading. Total time: 8 minutes."* — This cross-canvas navigation (1 → 3 → 1) must complete in ≤200ms per transition. The current dual-system routing conflict causes hangs. | FR20, NFR-PERF-1 |
| **J6 — Portfolio Audit** | *"Portfolio canvas shows the monthly performance report: 18 profitable, 8 flat, 5 losing."* — Mubarak drills from the Portfolio tile grid into the performance sub-page and back. The legacy `navigationStore` call inside `PortfolioCanvas` leaves stale state that prevents clean back-navigation. | FR55 |
| **J20 — Dual Account** | *"FTMO Challenge ($100k) and Personal ($8k) run simultaneously. A strategy passing paper trade with 1.8% max drawdown gets routed to Personal only."* — Reviewing this routing decision requires navigating into Portfolio sub-pages (account registry, routing matrix). Sub-page navigation must be clean and reliable. | FR51 |
| **J27 — Client Layer** | *"Copilot generates a structured client report for each. Mubarak reviews, exports to PDF, sends. Total time: 8 minutes for both."* — Reviewing client account performance requires Portfolio canvas sub-page navigation to the attribution view. | FR55 |
| **J29 — Dept Mail Morning Read** | *"StatusBand: mail badge 7 unread."* — The five StatusBand click targets (session clock, active bots count, risk mode, router mode, node health) must navigate to the correct canvas. Clicking "active bots" must reliably route to Portfolio canvas without the routing conflict. | Arch-UI-7 |

#### FR Enablement

- **FR55** (portfolio metrics display + navigation): Portfolio sub-page routing works cleanly via local `$state`
- **FR51** (broker account registry navigation): Portfolio canvas correctly reaches account-detail sub-page
- **NFR-PERF-1** (≤200ms canvas switches): `activeCanvasStore` as single source of truth eliminates double-render
- **Arch-UI-7** (canvas-local vs global state): `+page.svelte` derives from `activeCanvasStore` — local `activeView` state removed entirely

#### Implementation Approach

**TARGETED EDITS only:**

| File | Change |
|------|--------|
| `PortfolioCanvas.svelte` | Replace `navigationStore.navigateToView()` calls with `let currentSubPage = $state<PortfolioSubPage>('grid')` local pattern |
| `+page.svelte` | Derive active canvas from `activeCanvasStore` — remove `activeView = $state(...)` and `handleViewChange` |
| `ActivityBar.svelte` | Replace `dispatch('viewChange', ...)` with `activeCanvasStore.setCanvas(canvasId)` |
| `MainContent.svelte` | Replace `activeView` prop subscription with `activeCanvasStore` subscription |
| `StatusBand.svelte` | Verify/add all 5 click targets using `activeCanvasStore.setCanvas()` |

**DO NOT TOUCH:**
- `canvas.ts` — `CANVASES` array and keyboard shortcuts are correct from Story 1-6; wire to them, don't rewrite
- Backend files, Settings

#### Acceptance Criteria

**AC 12-5-1: Portfolio sub-page — drill and back**
- **Given** the Portfolio tile grid is visible
- **When** a portfolio sub-page tile is clicked
- **Then** the sub-page view renders in ≤200ms and a back button appears in the canvas header

**AC 12-5-2: Portfolio sub-page routing — no navigationStore**
- **Given** `PortfolioCanvas.svelte` is inspected
- **When** a developer searches for `navigationStore`
- **Then** no calls to `navigationStore.navigateToView()` exist inside this component
- **And** sub-page state is managed with `let currentSubPage = $state<PortfolioSubPage>('grid')`

**AC 12-5-3: Back navigation — no stale state**
- **Given** the Portfolio sub-page is open
- **When** the back button is clicked
- **Then** `currentSubPage` resets to `'grid'`
- **And** the tile grid is restored with no residual sub-page state in `navigationStore`

**AC 12-5-4: Single source of truth — no `activeView` state in `+page.svelte`**
- **Given** `+page.svelte` is inspected
- **When** searching for `activeView = $state`
- **Then** no such declaration exists — active canvas is derived from `activeCanvasStore` only

**AC 12-5-5: ActivityBar dispatches to canvasStore only**
- **Given** `ActivityBar.svelte` is inspected
- **When** a canvas icon is clicked
- **Then** `activeCanvasStore.setCanvas(canvasId)` is called
- **And** no `dispatch('viewChange', ...)` event is emitted

**AC 12-5-6: All 9 keyboard shortcuts resolve correctly**
- **Given** the keyboard shortcuts are wired in `canvas.ts`
- **When** each key is pressed in sequence
- **Then** key 1 = Live Trading, 2 = Research, 3 = Development, 4 = Risk, 5 = Trading, 6 = Portfolio, 7 = SharedAssets, 8 = Workshop, 9 = FlowForge — each correct

**AC 12-5-7: All 5 StatusBand click targets wired**
- **Given** the StatusBand is rendered
- **When** each of the 5 clickable segments is clicked:
  - Session clocks → `activeCanvasStore.setCanvas('live-trading')`
  - Active bots count → `activeCanvasStore.setCanvas('portfolio')`
  - Risk mode indicator → `activeCanvasStore.setCanvas('risk')`
  - Router mode label → `activeCanvasStore.setCanvas('risk')`
  - Node health dots → opens node status overlay (NOT canvas navigation)
- **Then** each action fires correctly via `activeCanvasStore`

**AC 12-5-8: No undefined errors on canvas load**
- **Given** all 9 canvases are loaded sequentially
- **When** each is activated
- **Then** the correct canvas component renders
- **And** no console errors referencing `undefined` activeView, null canvas ID, or navigationStore conflict appear

**AC 12-5-9: Canvas transition budget met**
- **Given** `activeCanvasStore` is the single source of truth
- **When** any canvas switch is performed
- **Then** the new canvas is visible within 200ms — the dual-system latency is eliminated

**Out of Scope for 12-5:** Portfolio tile data content (Epic 9), broker account detail sub-pages (Epic 9), attribution and correlation matrix (Epic 9)

---

### Story 12-6: Department Kanban Sub-Page (All Canvases Except Live Trading + Workshop)

As a **trader (Mubarak)**,
I want **every department canvas** (Research, Development, Risk, Trading, Portfolio, SharedAssets, FlowForge) **to include a Dept Kanban summary tile** showing active/blocked/done task counts, clicking into a full Kanban board with TODO / IN_PROGRESS / BLOCKED / DONE columns — wired via SSE to `GET /api/departments/{dept}/tasks`,
So that I can **see what each department agent is working on, what is blocked, and what is done — directly from any department canvas** without asking the Copilot or switching to Workshop.

#### PRD Journey Linkage

| Journey | Excerpt | FR Enabled |
|---------|---------|-----------|
| **J5 — Weekly War Room** | *"Mubarak opens the Workshop canvas: 'Let's plan this week.' Copilot pulls Portfolio's performance summary, surfaces Research's recommendations."* — After planning, Mubarak needs to track that each dept received and is acting on its operating brief. The Kanban sub-page on each canvas confirms task receipt and progress. | FR10 |
| **J8 — Silent Failure** | *"3:14am. Contabo drops connection."* — More critically: *"FloorManager detects Development has had no commits in 18h."* The BLOCKED card sitting in the Development Kanban makes this visible to Mubarak the next morning without any Copilot prompt needed. | FR10 |
| **J12 — Parameter Sweep** | *"512 combinations tested. 4 survived correlation filter. Estimated paper trade review: 5 days."* — This entire computation is a series of department tasks. The Development Kanban shows: "512-combination sweep" as DONE, "4 EAs → paper queue" as IN_PROGRESS. | FR10, FR14 |
| **J16 — Code Review** | *"Mubarak types to Copilot: 'Lot sizing should use Kelly Engine output, not fixed lots.' Development receives the feedback via department mail and generates V4 in 8 minutes."* — After Mubarak submits this feedback in the Agent Panel on Development canvas, the task "V4 — Kelly-based lot sizing" appears in the Development Kanban as IN_PROGRESS, then DONE after 8 minutes. | FR22, FR14 |
| **J29 — Dept Mail Morning Read** | *"Department Mail inbox tells the overnight story — Research routing a new hypothesis to Development, Development completing a prototype, Risk flagging a correlation spike."* — The Kanban sub-pages give Mubarak the same overnight story in visual Kanban form when he clicks into each canvas. | FR10 |
| **J32 — Department Council** | *"Copilot broadcasts to all four simultaneously. Each responds within 3 minutes with domain-specific assessments."* — The council response tasks appear in each department's Kanban as IN_PROGRESS during response generation, then DONE when complete. Mubarak can check each canvas's Kanban to see the status. | FR22 |

#### FR Enablement

- **FR10** (dept task visibility): `DeptKanbanTile` summary on every dept canvas; `DeptKanban` full Kanban board on sub-page
- **FR14** (persistent agent memory / task delegation): Kanban is the UI surface showing what dept heads have been delegated and their status
- **FR22** (human-in-loop visibility): BLOCKED cards and human-approval tasks surface in the Kanban — Mubarak sees when a dept is waiting on him

#### Implementation Approach

**NEW files to create:**
- `quantmind-ide/src/lib/components/shared/DeptKanbanTile.svelte` — summary tile (active/blocked/done counts, click opens sub-page)

**TARGETED EDITS to 7 canvas files** (add `'dept-kanban'` to SubPage union + render `DeptKanbanTile` in grid):

| Canvas | Edit |
|--------|------|
| `ResearchCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="research" />` |
| `DevelopmentCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="development" />` |
| `RiskCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="risk" />` |
| `TradingCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="trading" />` |
| `PortfolioCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="portfolio" />` |
| `SharedAssetsCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="shared-assets" />` |
| `FlowForgeCanvas.svelte` | Add `'dept-kanban'` to SubPage type, add `<DeptKanbanTile dept="flowforge" />` (distinct from Prefect Workflow Kanban) |

**Backend endpoint for Kanban data:**
`GET /api/floor-manager/departments/{dept}/tasks` — status unknown from codebase scan. Check `floor_manager_endpoints.py`. If not present, it must be added before dev begins. Kanban SSE real-time updates come through the shared `/api/agents/stream` channel via `task_status` events.

**DO NOT TOUCH:**
- `LiveTradingCanvas.svelte` — excluded per architecture (Live Trading is not a department canvas)
- `WorkshopCanvas.svelte` — excluded per architecture (Workshop is the FloorManager home)
- The existing `department-kanban/` component directory (wired in; not rewritten)

#### Acceptance Criteria

**AC 12-6-1: Dept Kanban tile present on 7 correct canvases**
- **Given** any of: Research, Development, Risk, Trading, Portfolio, SharedAssets, or FlowForge canvas is loaded
- **When** the tile grid renders
- **Then** a `DeptKanbanTile` is visible showing: active task count, blocked count (using `--color-accent-amber`), done count (last 24h)

**AC 12-6-2: Dept Kanban tile absent from Live Trading + Workshop**
- **Given** the Live Trading canvas (slot 1) is loaded
- **When** it is inspected
- **Then** no `DeptKanbanTile` is present
- **And** Workshop canvas (slot 8) also has no `DeptKanbanTile`

**AC 12-6-3: Kanban sub-page — 4 columns render**
- **Given** the `DeptKanbanTile` on Development canvas is clicked
- **When** the Kanban sub-page renders
- **Then** a board with exactly 4 columns is visible: TODO / IN_PROGRESS / BLOCKED / DONE
- **And** the canvas header shows a back button

**AC 12-6-4: Kanban back navigation — clean**
- **Given** the Dept Kanban sub-page is open
- **When** the back button is clicked
- **Then** `currentSubPage` returns to `'grid'`
- **And** the canvas tile grid (including `DeptKanbanTile`) is visible again

**AC 12-6-5: SSE real-time task updates**
- **Given** the Kanban sub-page is open and SSE connects to `GET /api/departments/{dept}/tasks`
- **When** a task's status changes on the backend
- **Then** the Kanban card moves to the correct column without a page refresh

**AC 12-6-6: Empty state — neutral, not error**
- **Given** the department has no active tasks
- **When** the `DeptKanbanTile` renders
- **Then** a neutral message is shown: "No active tasks — dept head is idle"
- **And** the tile renders without any error state styling

**AC 12-6-7: BLOCKED cards visually distinct**
- **Given** tasks with BLOCKED status are in the Kanban board
- **When** the board renders
- **Then** BLOCKED column header uses `--color-accent-amber` and BLOCKED cards have a visible amber indicator distinguishing them from IN_PROGRESS cards

**AC 12-6-8: FlowForge — Dept Kanban distinct from Prefect Kanban**
- **Given** the FlowForge canvas is open
- **When** both Kanban-type views are visible
- **Then** the Prefect Workflow Kanban (workflow orchestration tasks) and the Dept Kanban (department agent tasks) are visually and structurally distinct
- **And** they are separately accessible via their own navigation paths

**Out of Scope for 12-6:** Task creation UI (Epic 7), task assignment to sub-agents (Epic 7), approval gate cards (Epic 8), skill execution task tracking (Epic 7/9)

---

## What Is Explicitly NOT in Epic 12

| Deferred Feature | Owning Epic |
|---|---|
| AI streaming in Agent Panel — live agent responses via SSE | Epic 5 |
| Real OPINION node writes from a live agent | Epic 5 (agent must exist first) |
| Real MCP call events in the tool stream | Epic 5 (agent must exist first) |
| Real memory commits to graph DB from agent actions | Epic 5 |
| Department Head routing / actual agent dispatch | Epic 7 |
| Real Copilot suggestion chips | Epic 5/7 |
| Real sub-agent spawning and task execution | Epic 7 |
| Physics sensor visualizations on Risk canvas tiles | Epic 4 |
| Full Portfolio canvas content (attribution, correlation matrix) | Epic 9 |
| EA backtest viewer sub-page content | Epic 8 |
| Monaco editor integration | Epic 8 |
| FlowForge canvas implementation | Epic 11 |
| Settings panel changes of any kind | Excluded (Mubarak direction) |
| A/B race board | Epic 8 |
| Knowledge base tiles with real data (Research, SharedAssets) | Epic 6 |
| Task creation UI in Dept Kanban | Epic 7 |
| Morning digest real data in Workshop | Epic 5/7 |

---

## Testing Strategy per Story

| Story | Test Method |
|---|---|
| **12-1** | Manual visual: 320px Agent Panel, dept badge per canvas, 300ms collapse, `[+]` creates session, `[⏱]` opens history; Workshop hides panel; Vitest: session state (Interactive vs Autonomous switching), three message type CSS classes, SSE EventSource open/close lifecycle, canvas context `GET /api/canvas-context/{canvasId}` called on `[+]`, `.ap-tool` line render from mock SSE events (OPINION write, `write_memory`, `context7`), sub-agent status indicator update from `sub_agent_status` event, RichRenderer table render inside `.ap-agent` bubble |
| **12-2** | CSS inspection: `oklch(` = 0 in `:root`; colour values exact; theme preset tile density override; `--color-danger` absent; AppearancePanel still functional |
| **12-3** | Visual test: all 9 canvases — no blank screens; Balanced Terminal = 280px/12px; Ghost Panel = 320px/16px; Workshop 200px sidebar + Lucide icons; no Kill Switch in canvas tiles; Vitest: TileCard import paths, data-dept attribute presence |
| **12-4** | API mock: empty list → calm empty state; populated list → EA table renders; `/stages` all-zeros → neutral; sub-page routing; no `epicNumber={3}` reference; Vitest: tile state on API responses |
| **12-5** | Keyboard shortcuts 1–9 all resolve; StatusBand 5 click targets; Portfolio back-button; `+page.svelte` zero `activeView = $state`; no navigationStore in PortfolioCanvas; Vitest: `activeCanvasStore` as sole source |
| **12-6** | DeptKanbanTile on 7 correct canvases; absent from Live Trading + Workshop; 4 Kanban columns; back navigation; BLOCKED amber styling; Vitest: sub-page routing for all 7 canvas edits |
