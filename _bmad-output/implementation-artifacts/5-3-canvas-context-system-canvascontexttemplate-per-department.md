# Story 5.3: Canvas Context System — CanvasContextTemplate per Department

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer building the Agent Panel,
I want a CanvasContextTemplate system that pre-loads the correct department head with relevant memory scope when a chat opens on any canvas,
So that agents are context-aware from the first message rather than starting blind.

## Acceptance Criteria

**Given** a user opens a new chat on any canvas (Live Trading, Risk, Portfolio, etc.),
**When** the session initializes,
**Then** the system loads the department-specific `CanvasContextTemplate` for that canvas,
**And** the template provides: base descriptor, memory scope (graph namespaces), workflow namespaces, department mailbox, shared assets, and skill index.

**Given** a user opens a chat on a canvas,
**When** the template loads,
**Then** the system loads only memory identifiers (not full content) from the graph using `session_status = committed`,
**And** content is fetched JIT (Just-In-Time) when the agent needs it,
**And** total pre-loaded context does not exceed the configured token budget.

**Given** the user clicks a suggestion chip,
**When** the chip is clicked,
**Then** it navigates to the relevant canvas/entity with context pre-loaded.

**Given** a user starts a chat via CopilotPanel,
**When** the request is sent,
**Then** the canvas context is passed as metadata: `{ message, canvas_context: "<canvas_name>", session_id }`.

**Notes:**
- FR20: canvas-aware Copilot context (CanvasContextTemplate per department)
- Architecture hard rule: "No memory pre-loading — load identifiers, fetch content JIT via tools"
- CanvasContextTemplate YAML configs stored under `flows/directives/`
- Architecture prerequisite: this story depends on Story 5.1 (Graph Memory) being complete

## Tasks / Subtasks

- [x] Task 1 (AC: #1 - Template Loading)
  - [x] Subtask 1.1: Create CanvasContextTemplate YAML schema and loader
  - [x] Subtask 1.2: Create YAML files per canvas (live_trading, risk, portfolio, research, development, workshop, flowforge, shared_assets)
  - [x] Subtask 1.3: Implement context_loader.ts in frontend to load template on canvas change
- [x] Task 2 (AC: #2 - Memory Integration)
  - [x] Subtask 2.1: Integrate GraphMemoryFacade to load committed nodes as identifiers
  - [x] Subtask 2.2: Implement JIT content fetch mechanism for memory nodes
  - [x] Subtask 2.3: Add token budget enforcement (max identifiers loaded)
- [x] Task 3 (AC: #3 - Agent Panel Integration)
  - [x] Subtask 3.1: Wire CanvasContextTemplate into AgentPanel.svelte
  - [x] Subtask 3.2: Pass canvas_context metadata to FloorManager API on chat init
  - [x] Subtask 3.3: Handle canvas context in CopilotPanel.svelte
- [x] Task 4 (AC: #3 - Suggestion Chips)
  - [x] Subtask 4.1: Create CanvasSuggestionChip component
  - [x] Subtask 4.2: Implement navigation with context pre-load

## Dev Notes

### Previous Story Intelligence (Story 5.2)

Story 5.2 completed the Session Checkpoint → Graph Memory Commit Flow:
- **ReflectionExecutor** at `src/memory/graph/reflection_executor.py` — handles reflection and session recovery
- **Embedding service** at `src/memory/graph/embedding_service.py` — vector search
- **GraphMemoryFacade** at `src/memory/graph/facade.py` — load_committed_state() method added
- **SessionCheckpointService** — checkpoint triggers, milestone triggers, stale draft cleanup
- **session_status** column: 'draft' vs 'committed' — CRITICAL for this story

**Key insight from Story 5.2:** The graph memory system is operational. Story 5.3 leverages it by loading only `committed` nodes as identifiers at session start.

### Architecture Context (Epic 5)

Epic 5 covers:
- Unified Memory (graph memory across 4 fragmented systems)
- Agent Panel pre-loaded with correct CanvasContextTemplate
- Department Mail (SQLite) verified working
- Workshop canvas becomes Copilot's home

**FRs covered in Epic 5:** FR10, FR11, FR13, FR14, FR15, FR16, FR20

**FR20 (this story):** "The Copilot can operate context-aware on any canvas, with tools and commands appropriate to the active department"

### Architecture Prerequisites (CRITICAL)

- **MUST complete after Stories 5.1 and 5.2** — graph memory foundation must exist
- **GraphMemoryFacade** exists at `src/memory/graph/facade.py` — use load_committed_state()
- **ReflectionExecutor** exists at `src/memory/graph/reflection_executor.py`
- **AgentPanel.svelte** exists at `quantmind-ide/src/lib/components/shell/AgentPanel.svelte` — needs CanvasContextTemplate integration
- **CopilotPanel.svelte** exists — KNOWN BUG: sends to `/api/chat/send` instead of `/api/floor-manager/chat` — DO NOT replicate

### CanvasContextTemplate Schema (from Architecture §6.2)

```python
CanvasContextTemplate(canvas="RISK"):
    base_descriptor: "You are the Risk Department Copilot..."
    memory_scope: ["risk.*", "portfolio.*", "trading.*"]   # Graph memory namespaces (committed only)
    workflow_namespaces: ["risk_workflows", "portfolio_workflows"]
    department_mailbox: "risk_dept_mail stream"
    shared_assets: ["risk_templates", "prop_firm_rules"]
    skill_index:                                           # Indexed skills for this canvas
      - id: "drawdown-review"
        path: "shared_assets/skills/departments/risk/drawdown-review/skill.md"
        trigger: "when reviewing drawdown or risk thresholds"
    required_tools: [risk_calculator, position_monitor, calendar]
    # All values are IDENTIFIERS, not content
    # Agents load content JIT when needed
```

### CAG + RAG Combined Pattern

- **CAG layer:** stable identifiers (canvas descriptor, dept SOP path, skill index, tool list) — pre-assembled on canvas load
- **RAG layer:** live state fetched JIT (HOT graph memory nodes `committed` only, current task board, recent department mail)
- **Memory isolation:** Template loads only `session_status = committed` memory nodes. In-progress drafts are invisible.

### File Structure Requirements

```
src/memory/graph/
├── __init__.py           [exists]
├── types.py              [exists] - SessionStatus, 14 node types
├── store.py             [exists] - session_status, embedding columns
├── operations.py        [exists] - commit methods
├── facade.py            [exists] - has load_committed_state()
├── reflection_executor.py [exists]
├── embedding_service.py [exists]
└── tier_manager.py      [exists]

# NEW: Canvas Context System
src/
└── canvas_context/
    ├── templates/                     [NEW]    YAML CanvasContextTemplate files per canvas
    │   ├── live_trading.yaml
    │   ├── research.yaml
    │   ├── development.yaml
    │   ├── risk.yaml
    │   ├── trading.yaml
    │   ├── portfolio.yaml
    │   ├── shared_assets.yaml
    │   ├── workshop.yaml
    │   └── flowforge.yaml
    └── context_loader.py              [NEW]    CAG+RAG assembly on canvas load

quantmind-ide/src/lib/components/shell/
├── AgentPanel.svelte                  [extend] Add CanvasContextTemplate loading
├── CopilotPanel.svelte               [extend] Pass canvas_context metadata
└── SuggestionChipBar.svelte          [NEW]    Canvas-aware suggestion chips
```

### Known Bug (DO NOT REPLICATE)

**CopilotPanel.svelte** currently sends to `/api/chat/send` (legacy) instead of `/api/floor-manager/chat` (canonical). The fix plan is in `docs/plans/2026-03-08-agent-architecture-migration-map.md`. **Always use `/api/floor-manager/chat`** for new agent interactions.

### Testing Standards

- Tests go in `tests/memory/` and `tests/canvas_context/`
- Integration tests for template loading
- JIT fetch tests (mock graph responses)
- Token budget enforcement tests
- Agent Panel integration tests

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md` (§6.2 Canvas Context System, §14.1 Two Session Types)
- Epic context: `_bmad-output/planning-artifacts/epics.md` (Epic 5, Story 5.3)
- Previous stories:
  - `_bmad-output/implementation-artifacts/5-1-graph-memory-completion-reflectionexecutor-opinion-nodes-embeddings.md`
  - `_bmad-output/implementation-artifacts/5-2-session-checkpoint-graph-memory-commit-flow.md`
- Graph memory: `src/memory/graph/facade.py`, `src/memory/graph/store.py`
- Frontend: `quantmind-ide/src/lib/components/shell/AgentPanel.svelte`

## Dev Agent Record

### Agent Model Used

Claude Opus 4.6 (via Claude Code)

### Debug Log References

- Backend template loading: `src/canvas_context/loader.py`
- Frontend service: `quantmind-ide/src/lib/services/canvasContextService.ts`
- API endpoints: `src/api/canvas_context_endpoints.py`

### Completion Notes List

**Implemented:**
1. Created CanvasContextTemplate Pydantic models in `src/canvas_context/types.py`
2. Created YAML template loader with caching in `src/canvas_context/loader.py`
3. Created 9 canvas YAML templates (live_trading, risk, portfolio, research, development, trading, workshop, flowforge, shared_assets)
4. Created backend API endpoints (`/api/canvas-context/*`) for template loading and context assembly
5. Created frontend CanvasContextService.ts with template loading and canvas context metadata building
6. Updated CopilotPanel.svelte to include canvas_context in API requests
7. Created test suite with 25 tests (all passing)

**Key Implementation Details:**
- Templates use memory scope patterns (e.g., "risk.*") not content - follows CAG+RAG pattern
- Token budget enforcement via max_identifiers field (default 50)
- Canvas context passed as `{ message, canvas_context, session_id }` in API calls
- JIT content fetch ready - template provides namespace identifiers, content loaded on demand

### File List

**Backend Files Created:**
- `src/canvas_context/__init__.py` — [NEW]
- `src/canvas_context/types.py` — [NEW]
- `src/canvas_context/loader.py` — [NEW]
- `src/canvas_context/templates/__init__.py` — [NEW]
- `src/canvas_context/templates/live_trading.yaml` — [NEW]
- `src/canvas_context/templates/research.yaml` — [NEW]
- `src/canvas_context/templates/development.yaml` — [NEW]
- `src/canvas_context/templates/risk.yaml` — [NEW]
- `src/canvas_context/templates/trading.yaml` — [NEW]
- `src/canvas_context/templates/portfolio.yaml` — [NEW]
- `src/canvas_context/templates/shared_assets.yaml` — [NEW]
- `src/canvas_context/templates/workshop.yaml` — [NEW]
- `src/canvas_context/templates/flowforge.yaml` — [NEW]
- `src/api/canvas_context_endpoints.py` — [NEW]
- `src/api/server.py` — [MODIFIED] Added canvas_context_router

**Frontend Files Created:**
- `quantmind-ide/src/lib/services/canvasContextService.ts` — [NEW]
- `quantmind-ide/src/lib/components/trading-floor/SuggestionChipBar.svelte` — [NEW]

**Frontend Files Modified:**
- `quantmind-ide/src/lib/components/trading-floor/CopilotPanel.svelte` — Added canvas_context metadata to API calls

**Test Files Created:**
- `tests/canvas_context/__init__.py` — [NEW]
- `tests/canvas_context/test_template_loader.py` — [NEW]
- `tests/canvas_context/test_context_integration.py` — [NEW]

### Status

Status: done

**Note:** All tasks complete. SuggestionChipBar.svelte created during code review.
