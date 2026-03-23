# Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

**As a** trader and developer managing workflows,
**I want** the FlowForge canvas to show all Prefect workflows in a Kanban view with per-workflow kill switches and a node graph viewer,
**So that** I can monitor, control, and understand all running workflows.

## Acceptance Criteria

1. **Given** I navigate to the FlowForge canvas (Canvas 9),
   **When** the canvas loads,
   **Then** it shows a PrefectKanbanCard layout with 6 columns: PENDING, RUNNING, PENDING_REVIEW, DONE, CANCELLED, EXPIRED_REVIEW.

2. **Given** a workflow is in RUNNING state,
   **When** the card renders,
   **Then** it shows: workflow name, dept, state badge (cyan pulse border), duration, step progress (X/Y), next step label,
   **And** a Workflow Kill Switch (Lucide `square` "Stop") appears on the card,
   **And** clicking it shows a two-step confirmation modal scoped to THAT workflow only.

3. **Given** I click a workflow card to view its node graph,
   **When** the FlowForgeNodeGraph opens,
   **Then** an SVG dependency graph renders with: task boxes (coloured by state), directed edges, zoom+pan, minimap,
   **And** selecting a node shows task detail tooltip.

4. **Given** the Workflow Kill Switch fires for one workflow,
   **When** it confirms,
   **Then** only that Prefect workflow is cancelled,
   **And** all other running workflows and live trading on Cloudzy continue unaffected.

## Tasks / Subtasks

- [x] Task 1: FlowForge Canvas Route (AC: #1)
  - [x] Task 1.1: Add FlowForge canvas route to canvas routing
  - [x] Task 1.2: Create FlowForgeCanvas.svelte layout with 6-column Kanban
- [x] Task 2: PrefectKanbanCard Component (AC: #1, #2)
  - [x] Task 2.1: Create PrefectKanbanCard component
  - [x] Task 2.2: Implement state badge with cyan pulse border
  - [x] Task 2.3: Add workflow kill switch with two-step confirmation
- [x] Task 3: FlowForgeNodeGraph Component (AC: #3)
  - [x] Task 3.1: Create FlowForgeNodeGraph component
  - [x] Task 3.2: Implement SVG dependency graph rendering
  - [x] Task 3.3: Add zoom+pan and minimap
  - [x] Task 3.4: Implement node selection tooltip
- [x] Task 4: Prefect API Integration (AC: #1, #2, #3, #4)
  - [x] Task 4.1: Add Prefect workflow status API endpoints
  - [x] Task 4.2: Implement workflow cancellation API
  - [x] Task 4.3: Wire to existing Prefect flows in flows/

## Dev Notes

### Key Architecture Context

**Workflow Kill Switch Rules:**
- Workflow Kill Switch is per-card in FlowForge ONLY — not a global button (architecture hard rule)
- Workflow Kill Switch ≠ TradingKillSwitch — these are architecturally independent
- Recovery: `/resume-workflow` command re-triggers from last completed Prefect step

**FlowForge Canvas:**
- Canvas 9 in the canvas routing system
- Connects to Prefect API on Contabo
- Shows all Prefect workflows across departments

**Technical Stack:**
- Frontend: Svelte 5 with SVG for node graph
- Backend: Prefect Python client
- API: FastAPI endpoints for workflow status/cancellation

### Files to Create/Modify

**NEW FILES:**
- `quantmind-ide/src/lib/components/canvas/FlowForgeCanvas.svelte` — Main canvas component
- `quantmind-ide/src/lib/components/flowforge/PrefectKanbanCard.svelte` — Kanban card component
- `quantmind-ide/src/lib/components/flowforge/FlowForgeNodeGraph.svelte` — Node graph viewer
- `quantmind-ide/src/lib/components/flowforge/WorkflowKillSwitchModal.svelte` — Confirmation modal
- `src/api/prefect_workflow_endpoints.py` — Prefect API integration

**MODIFY:**
- `quantmind-ide/src/lib/components/canvas/` — Add FlowForge to canvas routing
- `quantmind-ide/src/lib/stores/` — Add FlowForge store for workflow state

### Technical Specifications

**Prefect API Integration:**
```python
from prefect import get_client

@router.get("/api/prefect/workflows")
async def list_workflows():
    """List all Prefect workflows with status."""
    client = get_client()
    flows = await client.read_flows()
    # Return flow runs with state, duration, progress
```

**Workflow Cancellation:**
```python
@router.post("/api/prefect/workflows/{flow_id}/cancel")
async def cancel_workflow(flow_id: str):
    """Cancel a specific Prefect workflow."""
    client = get_client()
    await client.cancel_flow_run(flow_id)
```

**Kanban Columns:**
- PENDING: Flows not yet started
- RUNNING: Currently executing flows
- PENDING_REVIEW: Flows awaiting human approval
- DONE: Completed flows
- CANCELLED: Cancelled flows
- EXPIRED_REVIEW: Flows that timed out in review state

### Testing Standards

- Unit test: Kanban card rendering, node graph SVG generation
- Integration test: Prefect API connection, workflow cancellation
- Component test: Modal confirmation flow

### Project Structure Notes

- Epic 11: System Management & Resilience
- Canvas numbering: FlowForge is Canvas 9
- Uses existing canvas routing infrastructure from Epic 1

### Previous Story Intelligence

**From Story 11.2 (Weekend Compute Protocol):**
- Prefect flow patterns used
- Similar API integration patterns

**From Epic 5 (Copilot Core):**
- Canvas routing patterns established
- Modal confirmation patterns

### References

- Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Epics: `_bmad-output/planning-artifacts/epics.md` (Story 11.5)
- Canvas routing: Epic 1 Story 1.6
- Prefect: `flows/`, Prefect documentation

---

## Developer Implementation Guide

### What NOT to Do

1. **DO NOT** create a global kill switch — per-card only (architecture hard rule)
2. **DO NOT** confuse Workflow Kill Switch with Trading Kill Switch
3. **DO NOT** implement drag-to-add nodes — view only in this story
4. **DO NOT** affect live trading when cancelling workflows

### What TO Do

1. **DO** use Lucide icons (square for stop)
2. **DO** implement cyan pulse border for RUNNING state
3. **DO** show step progress (X/Y) on cards
4. **DO** add zoom+pan to node graph
5. **DO** implement two-step confirmation for workflow cancellation

### Code Patterns

**Kanban card pattern:**
```svelte
<script lang="ts">
  let { workflow } = $props<{ workflow: Workflow }>();
</script>

<div class="card {workflow.state}">
  {#if workflow.state === 'RUNNING'}
    <div class="pulse-border"></div>
  {/if}
  <span class="badge">{workflow.state}</span>
  <span class="progress">{workflow.completed_steps}/{workflow.total_steps}</span>
</script>
```

**Node graph pattern:**
```svelte
<svg viewBox="0 0 {width} {height}">
  {#each tasks as task}
    <g class="task" transform="translate({task.x}, {task.y})">
      <rect class="state-{task.state}" ... />
      {#each task.dependencies as dep}
        <path d="M{dep.x},{dep.y} L{task.x},{task.y}" ... />
      {/each}
    </g>
  {/each}
</svg>
```

---

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

### Completion Notes List

- Implemented FlowForgeCanvas with 6-column Prefect Kanban layout (PENDING, RUNNING, PENDING_REVIEW, DONE, CANCELLED, EXPIRED_REVIEW)
- Created PrefectKanbanCard with cyan pulse border animation for RUNNING state
- Implemented per-card workflow kill switch with two-step confirmation modal
- Created FlowForgeNodeGraph with SVG dependency graph, zoom+pan controls, minimap, and node selection tooltips
- Added FlowForge store (flowforge.ts) with API integration for workflow status/cancellation
- API endpoints already existed in prefect_workflow_endpoints.py - verified they're registered in server.py
- Canvas routing was already configured in canvasStore.ts and MainContent.svelte

### File List

- quantmind-ide/src/lib/components/canvas/FlowForgeCanvas.svelte (MODIFY - replaced placeholder with full Kanban implementation)
- quantmind-ide/src/lib/components/flowforge/PrefectKanbanCard.svelte (NEW)
- quantmind-ide/src/lib/components/flowforge/FlowForgeNodeGraph.svelte (NEW)
- quantmind-ide/src/lib/components/flowforge/WorkflowKillSwitchModal.svelte (NEW)
- quantmind-ide/src/lib/stores/flowforge.ts (NEW)
- quantmind-ide/src/lib/stores/index.ts (MODIFY - added flowforge store exports)
- src/api/prefect_workflow_endpoints.py (EXISTING - verified)
