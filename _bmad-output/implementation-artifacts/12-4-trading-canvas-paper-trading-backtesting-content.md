# Story 12.4: Trading Canvas (Slot 5) — Paper Trading & Backtesting Content

Status: done

## Story

As a **trader (Mubarak)**,
I want the **Trading canvas (slot 5) to display a real tile grid** — showing active paper trading EAs, recent backtest results, and the Alpha Forge enhancement loop pipeline stages — replacing the skeleton placeholder left by Story 12-3,
So that I can **see the paper trading pipeline at a glance**, know which EAs are in monitoring, click into backtest detail, and track the enhancement loop stage counts — without having to open a separate view or ask the Copilot.

## Acceptance Criteria

**AC 12-4-1: No placeholder — correct canvas identity**
- **Given** the Trading canvas (slot 5) is loaded via keyboard shortcut "5"
- **When** the canvas renders
- **Then** a `CanvasTileGrid` with title "Trading" (or "Trading Department") is visible
- **And** no `<CanvasPlaceholder>` exists in the rendered output
- **And** no `epicNumber={3}` prop reference exists anywhere in the component file (wrong-epic reference removed)
- **And** the skeleton `isLoading={true}` TileCards from 12-3 are fully replaced with live data tiles

**AC 12-4-2: Paper Trading Monitor tile — empty state (calm)**
- **Given** `GET /api/paper-trading/active` returns an empty list (`items: []`) or 404
- **When** the `PaperTradingMonitorTile` renders
- **Then** a calm neutral empty state is shown: "No EAs in paper monitoring phase — Alpha Forge feeds this when EAs reach the paper gate"
- **And** the tile renders correctly — not broken, not erroring

**AC 12-4-3: Paper Trading Monitor tile — populated state**
- **Given** `GET /api/paper-trading/active` returns active paper trade entries
- **When** the tile renders
- **Then** each EA entry shows: EA name, pair, days running, win rate, current P&L
- **And** all numeric values use `var(--font-data)` (JetBrains Mono) — class `.financial-value` or inline `font-family: var(--font-data)`
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
- **And** stage badges use Fragment Mono (10px caps) per CRM label pattern (class `.section-label`)

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
- **And** clicking back sets `currentSubPage = 'grid'` and tile grid is restored in ≤200ms (NFR-PERF-2)

**AC 12-4-8: API errors handled as empty state**
- **Given** any of the three backend API calls returns 404 or network error
- **When** the relevant tile renders
- **Then** an appropriate empty state message is shown — never a thrown exception or broken DOM

## Tasks / Subtasks

- [x] Task 1: Replace `TradingCanvas.svelte` with real tile grid (AC: #1, #7)
  - [x] 1.1: Remove all 3 skeleton `isLoading={true}` TileCards
  - [x] 1.2: Import `PaperTradingMonitorTile`, `BacktestResultsTile`, `EAPerformanceTile` from `components/trading/tiles/`
  - [x] 1.3: Wire `currentSubPage = $state<TradingSubPage>('grid')` local state with `showBackButton` and `onBack` wired to `CanvasTileGrid`
  - [x] 1.4: Pass `navigable={true}` and `onNavigate` callbacks to tiles that have sub-pages
  - [x] 1.5: Verify `data-dept="trading"` is still on the canvas root (from 12-3 — do not remove)

- [x] Task 2: Create `PaperTradingMonitorTile.svelte` in `quantmind-ide/src/lib/components/trading/tiles/` (AC: #2, #3)
  - [x] 2.1: Import `TileCard` from `$lib/components/shared/TileCard.svelte` — use `size="lg"` (spans 2 cols)
  - [x] 2.2: `onMount` — `apiFetch<ActiveAgentsResponse>('/api/paper-trading/active')` with try/catch
  - [x] 2.3: On 404 or empty `items`, render calm empty state text (see AC 12-4-2)
  - [x] 2.4: On populated `items`, render up to 3 items (CRM tile max — 4 data points): ea_name, pair, days_running, pnl_current; win_rate as status dot
  - [x] 2.5: Status dots: `status === 'running'` → `--color-accent-cyan`, `status === 'paused'` → `--color-accent-amber`, else → `--color-accent-red`
  - [x] 2.6: All numeric values (`pnl_current`, `win_rate`) wrapped in element with `class="financial-value"` (picks up `var(--font-data)` from TileCard CSS)
  - [x] 2.7: Wire `navigable` prop and `onNavigate` callback through to `TileCard`

- [x] Task 3: Create `BacktestResultsTile.svelte` in `quantmind-ide/src/lib/components/trading/tiles/` (AC: #4, #8)
  - [x] 3.1: Import `TileCard` from `$lib/components/shared/TileCard.svelte` — use `size="md"`
  - [x] 3.2: `onMount` — `apiFetch<BacktestSummary[]>('/api/backtests?limit=5')` (note: endpoint is `/api/backtests`, NOT `/api/backtest/recent` — see Dev Notes §Backend Contracts)
  - [x] 3.3: On error/empty, show neutral empty state: "No backtest results yet"
  - [x] 3.4: Render up to 5 rows: ea_name, pass/fail (win_rate ≥ 50% = pass), sharpe (`.financial-value`), run_at_utc formatted to local timezone
  - [x] 3.5: Pass/fail indicator: pass = `--color-accent-cyan`, fail = `--color-accent-red`
  - [x] 3.6: Wire `navigable` and `onNavigate` callback

- [x] Task 4: Create `EAPerformanceTile.svelte` in `quantmind-ide/src/lib/components/trading/tiles/` (AC: #5, #6)
  - [x] 4.1: Import `TileCard` from `$lib/components/shared/TileCard.svelte` — use `size="xl"` (full width)
  - [x] 4.2: `onMount` — `apiFetch<PipelineStatusResponse>('/api/pipeline/status')` and derive counts from `runs[].current_stage` (see Dev Notes §Backend Contracts)
  - [x] 4.3: On error, show neutral zero state for all 4 stage buckets
  - [x] 4.4: Derive stage counts from `PipelineRun[]` data: count by `current_stage` for Backtesting, SIT Gate (VALIDATION stage), Paper Monitoring (EA_LIFECYCLE), Awaiting Approval (APPROVAL)
  - [x] 4.5: Display as: "N Backtesting · N at SIT Gate · N Paper Monitoring · N Awaiting Approval"
  - [x] 4.6: Stage labels use `class="section-label"` (Fragment Mono 10px uppercase — from TileCard CSS)
  - [x] 4.7: Non-zero counts in running stages use `--color-accent-cyan`, zero counts use `--color-text-muted`

- [x] Task 5: Create sub-page placeholders (AC: #7)
  - [x] 5.1: Create `EAPerformanceDetailPage.svelte` in `quantmind-ide/src/lib/components/trading/tiles/` — placeholder only; heading "EA Performance Detail" + "Full detail view — Epic 7/8"; no API calls
  - [x] 5.2: Create `BacktestDetailPage.svelte` in `quantmind-ide/src/lib/components/trading/tiles/` — placeholder only; heading "Backtest Detail" + "Full backtest viewer — Epic 8"; no API calls

- [x] Task 6: Write Vitest tests (AC: all)
  - [x] 6.1: Test `TradingCanvas.svelte` — no skeleton `isLoading` TileCards, no `epicNumber` prop, `data-dept="trading"` present
  - [x] 6.2: Test `PaperTradingMonitorTile` — empty state text, financial-value class, status-dot colors, error handling
  - [x] 6.3: Test `BacktestResultsTile` — empty state, ea_name and sharpe structure, pass/fail colors, error handling
  - [x] 6.4: Test `EAPerformanceTile` — all-zero state, section-label class, stage label text, color tokens, BACKTEST/VALIDATION/EA_LIFECYCLE/APPROVAL stage mapping
  - [x] 6.5: Test sub-page routing — `currentSubPage` type with grid/backtest-detail/ea-performance-detail; onBack callback; showBackButton wiring

## Dev Notes

### CRITICAL: File Locations

All new tile components go in:
```
quantmind-ide/src/lib/components/trading/tiles/
  PaperTradingMonitorTile.svelte     ← NEW (size="lg")
  BacktestResultsTile.svelte         ← NEW (size="md")
  EAPerformanceTile.svelte           ← NEW (size="xl")
  EAPerformanceDetailPage.svelte     ← NEW (placeholder)
  BacktestDetailPage.svelte          ← NEW (placeholder)
```

The directory `components/trading/tiles/` does NOT yet exist — create it. `components/trading/` may not exist either — create both.

**EXISTING FILE to modify (full replacement of skeleton content):**
```
quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte
```
Current content is a 19-line skeleton with 3 `isLoading={true}` TileCards — replace with live tile imports and sub-page routing.

### CRITICAL ANTI-PATTERNS — DO NOT DO THESE

1. **DO NOT import `GlassTile`** from `live-trading/GlassTile.svelte` — that is Live Trading canvas only. All new tiles use `shared/TileCard.svelte` exclusively.

2. **DO NOT use raw `fetch()`** — all API calls must be `apiFetch<T>('/api/...')` imported from `$lib/api`.

3. **DO NOT use `export let`** or `$:` reactive declarations — Svelte 5 runes only: `$state`, `$derived`, `$props`, `$effect`.

4. **DO NOT hardcode colors** — use CSS tokens: `--color-accent-cyan`, `--color-accent-amber`, `--color-accent-red`, `--color-text-muted`, `--font-data`.

5. **DO NOT add kill switch anywhere** — Arch-UI-3 is absolute. No kill switch in canvas tiles or canvas components.

6. **DO NOT add Agent Panel** inside TradingCanvas — Agent Panel is shell-level only (`components/shell/AgentPanel.svelte`).

7. **DO NOT modify Settings sub-panels** — they are excluded from all Epic 12 work.

8. **DO NOT exceed 500 lines** per component (NFR-MAINT-1). Each tile should be 100-200 lines. If TradingCanvas exceeds 500 lines, extract sub-components.

9. **DO NOT import `CanvasPlaceholder`** — it is not used in any active canvas after 12-3.

### Backend Contracts (EXACT API signatures — verify before wiring)

#### `GET /api/paper-trading/active`
**Router prefix:** `APIRouter(prefix="/api/paper-trading")` → route is `@router.get("/active")`
**Full URL:** `/api/paper-trading/active`
**Response model:** `ActiveAgentsResponse`
```typescript
interface ActiveAgentItem {
  ea_name: string;
  pair: string;
  days_running: number;
  win_rate: number;
  pnl_current: number;
  status: string;       // "running" | "active" | "starting" | "validating" | others
  started_at: string;   // ISO datetime
}
interface ActiveAgentsResponse {
  items: ActiveAgentItem[];
}
```
**Empty response:** `{ items: [] }` — NOT a 404. Handle both empty `items` array and actual 404/500 errors as empty state.
**Source:** `src/api/paper_trading/routes.py` line 53, `src/api/paper_trading/models.py` lines 101-114

#### `GET /api/backtests?limit=5`
**Router prefix:** `APIRouter(prefix="/api/backtests")` → route is `@router.get("")`
**Full URL:** `/api/backtests?limit=5` (NOT `/api/backtest/recent`)
**Response model:** `BacktestSummary[]`
```typescript
interface BacktestSummary {
  id: string;
  ea_name: string;
  mode: string;             // "VANILLA" | "SPICED" | etc.
  run_at_utc: string;       // ISO datetime — MUST convert to local timezone for display
  net_pnl: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;         // 0-100
}
```
**Source:** `src/api/backtest_endpoints.py` line 431, `BacktestSummary` model lines 59-72

#### `GET /api/pipeline/stages`
**Router prefix:** `APIRouter(prefix="/api/pipeline")` → route is `@router.get("/stages")`
**Full URL:** `/api/pipeline/stages` (NOT `/api/pipeline-status/stages`)
**Response model:** `PipelineStagesResponse`
```typescript
interface PipelineStagesResponse {
  stages: string[];       // e.g. ["VIDEO_INGEST", "RESEARCH", ...]
  stage_order: string[];
  human_gates: string[];
}
```
**NOTE:** This endpoint returns stage *definitions*, not counts. To get per-stage counts, you need to call `GET /api/pipeline/status` (returns `PipelineStatusResponse { runs, total, active_count }`) and derive counts from the `runs[].current_stage` field. Show counts of active runs per stage bucket.
**Source:** `src/api/pipeline_status_endpoints.py` line 318

**For EAPerformanceTile counts:** Call `GET /api/pipeline/status` instead of `/stages`, then count `runs` by `current_stage`:
- "Backtesting" → count where `current_stage === "BACKTEST"`
- "at SIT Gate" → count where `current_stage === "VALIDATION"`
- "Paper Monitoring" → count where `current_stage === "EA_LIFECYCLE"`
- "Awaiting Approval" → count where `current_stage === "APPROVAL"`

### Project Structure Notes

**Shared components to USE (already exist from Story 12-3):**
- `quantmind-ide/src/lib/components/shared/CanvasTileGrid.svelte` — canvas wrapper
- `quantmind-ide/src/lib/components/shared/TileCard.svelte` — glass tile with size variants
- `quantmind-ide/src/lib/components/shared/SkeletonLoader.svelte` — pulse skeleton
- `quantmind-ide/src/lib/components/shared/Breadcrumb.svelte` — back button

**`TileCard` size props** (AC-critical):
- `size="lg"` → `grid-column: span 2` (Paper Trading Monitor)
- `size="md"` → default width (Backtest Results)
- `size="xl"` → `grid-column: 1 / -1` full width (Enhancement Loop)

**`CanvasTileGrid` props for sub-page routing:**
```svelte
<CanvasTileGrid
  title="Trading"
  dept="trading"
  showBackButton={currentSubPage !== 'grid'}
  onBack={() => currentSubPage = 'grid'}
>
```

**Sub-page type pattern** (matching other canvases):
```typescript
type TradingSubPage = 'grid' | 'backtest-detail' | 'ea-performance-detail';
let currentSubPage = $state<TradingSubPage>('grid');
```

**`apiFetch` import:**
```typescript
import { apiFetch } from '$lib/api';
```

**Svelte lifecycle imports:**
```typescript
import { onMount } from 'svelte';
```

### CSS Token Reference (from Story 12-2 — all active in app.css)

```css
/* Financial typography */
var(--font-data)          /* JetBrains Mono — all numeric values */
var(--font-ambient)       /* Fragment Mono — section labels, 10px uppercase */

/* Status colors */
var(--color-accent-cyan)  /* running/active/pass */
var(--color-accent-amber) /* paused/warning */
var(--color-accent-red)   /* failed/error */
var(--color-text-muted)   /* zero states, placeholder text */
var(--color-text-primary) /* primary content */

/* Glass */
var(--glass-content-bg)   /* TileCard base — rgba(8,13,20,0.35) */
var(--glass-blur)          /* blur(12px) saturate(160%) */
var(--color-border-subtle) /* tile border */
var(--color-bg-elevated)  /* SkeletonLoader pulse */
```

**CRM tile body pattern** — the `.tile-body` in `TileCard` already applies `font-family: var(--font-data)` to `.financial-value`, `.pnl`, `.lot-size`, `.risk-score`, `.timestamp` class selectors. Use these classes on numeric content inside tile slots.

Section labels use `.section-label` class → Fragment Mono 10px uppercase.

### Previous Story Intelligence (Story 12-3)

Story 12-3 (done) established all the patterns this story builds on:

1. **TradingCanvas.svelte current state** — 19-line file using `CanvasTileGrid` + 3 skeleton `TileCard` instances with `isLoading={true}` and `epicOwner="Epic 12-4"`. This is a FULL REPLACEMENT of those skeleton tiles with live data tiles.

2. **`data-dept="trading"` is already on the canvas** — the `CanvasTileGrid` `dept="trading"` prop from 12-3 handles this. Do NOT remove it.

3. **Shared components are confirmed working** — `CanvasTileGrid`, `TileCard`, `SkeletonLoader`, `Breadcrumb` are all tested and done from 12-3. Import from `$lib/components/shared/`.

4. **Kill switch compliance confirmed** — 12-3 verified zero kill switch imports in any canvas. 12-4 MUST maintain this.

5. **Svelte 5 runes are mandatory** for all NEW code — `TileCard` and `CanvasTileGrid` are Svelte 5 (`$props()`). New tile components must also be Svelte 5. Do not mix Svelte 4 patterns in new files.

6. **Component size discipline** — 12-3 created all shared components under 150 lines. Maintain this discipline. Tile components should stay under 200 lines each.

### Scope Boundaries (DO NOT IMPLEMENT)

- EA performance dossier content (Epic 7/8)
- Backtest viewer sub-page content (Epic 8 fills detail pages)
- A/B race board (Epic 8)
- Strategy lifecycle management
- Regime routing controls
- Any `live-trading/` canvas modifications
- FlowForge Kanban (Epic 11 — done)
- Workshop canvas (12-3 done, hands off)

### Testing Patterns (matching project test style)

Tests live co-located in the component directory or in `tests/component/`. Use Vitest + `@testing-library/svelte`. Mock `$lib/api` module with `vi.mock('$lib/api', ...)`. Assert DOM text, CSS classes, and attribute presence. Do not test implementation internals — test observable output.

Example mock pattern:
```typescript
import { vi } from 'vitest';
vi.mock('$lib/api', () => ({
  apiFetch: vi.fn()
}));
```

### References

- Story requirements: [Source: _bmad-output/planning-artifacts/epic-12-stories.md#Story 12-4]
- Tech spec: [Source: _bmad-output/implementation-artifacts/tech-spec-epic-12-ui-refactor.md]
- Backend — paper trading: [Source: src/api/paper_trading/routes.py#list_active_agents], [Source: src/api/paper_trading/models.py#ActiveAgentsResponse]
- Backend — backtests: [Source: src/api/backtest_endpoints.py#list_backtests, BacktestSummary]
- Backend — pipeline: [Source: src/api/pipeline_status_endpoints.py#get_pipeline_stages, PipelineRun]
- Shared components: [Source: quantmind-ide/src/lib/components/shared/TileCard.svelte]
- Canvas wrapper: [Source: quantmind-ide/src/lib/components/shared/CanvasTileGrid.svelte]
- Current TradingCanvas: [Source: quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte]
- API fetch wrapper: [Source: quantmind-ide/src/lib/api.ts#apiFetch]
- CSS tokens: [Source: quantmind-ide/src/app.css — Frosted Terminal token block]
- Arch constraints: [Source: _bmad-output/planning-artifacts/epic-12-stories.md#Architecture Requirements]
- Previous story: [Source: _bmad-output/implementation-artifacts/12-3-tile-grid-pattern-shared-components-all-9-canvases.md]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — no blocking issues encountered during implementation.

### Completion Notes List

- Replaced 19-line skeleton `TradingCanvas.svelte` with full live-data implementation using Svelte 5 `$state` rune for sub-page routing.
- Created `quantmind-ide/src/lib/components/trading/tiles/` directory (new, did not exist).
- `PaperTradingMonitorTile`: calls `/api/paper-trading/active`, empty state on empty items or any error, status dots using correct CSS tokens, financial-value class on pnl_current.
- `BacktestResultsTile`: calls `/api/backtests?limit=5` (correct endpoint per Dev Notes), local timezone date formatting via `toLocaleDateString`, pass/fail based on `win_rate >= 50`.
- `EAPerformanceTile`: calls `/api/pipeline/status` (not `/stages`) and derives 4 bucket counts from `runs[].current_stage`. All-zero state on error is neutral, not broken. section-label class on stage headings.
- Two placeholder detail pages created (no API calls, heading + Epic reference only).
- 87 new tests in `TradingCanvas.test.ts` — all pass using file-content assertion pattern (consistent with Story 12-3 `tile-grid.test.ts` pattern due to Svelte 5 + @testing-library/svelte render incompatibility).
- Updated 4 stale Story 12-3 tests in `tile-grid.test.ts` that asserted on the now-replaced skeleton state.
- Full regression: 431 pass, 4 skipped (pre-existing BacktestRunner.test.ts skip), 0 failures.
- All anti-patterns from Dev Notes verified absent: no GlassTile, no raw fetch(), no Svelte 4 patterns, no hardcoded colors, no kill switch, no CanvasPlaceholder.
- All components under 200 lines (NFR-MAINT-1 satisfied).

### File List

- `quantmind-ide/src/lib/components/canvas/TradingCanvas.svelte` — modified (full replacement of skeleton)
- `quantmind-ide/src/lib/components/trading/tiles/PaperTradingMonitorTile.svelte` — new; modified by review (win_rate display added, empty state init fixed)
- `quantmind-ide/src/lib/components/trading/tiles/BacktestResultsTile.svelte` — new; modified by review (Lucide CheckCircle/XCircle replacing ✓/✗)
- `quantmind-ide/src/lib/components/trading/tiles/EAPerformanceTile.svelte` — new
- `quantmind-ide/src/lib/components/trading/tiles/EAPerformanceDetailPage.svelte` — new
- `quantmind-ide/src/lib/components/trading/tiles/BacktestDetailPage.svelte` — new
- `quantmind-ide/src/lib/components/trading/tiles/TradingCanvas.test.ts` — new; modified by review (2 new test assertions added)
- `quantmind-ide/src/lib/components/shared/tile-grid.test.ts` — modified (4 stale Story 12-3 test assertions updated for 12-4 state)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` — modified (status: done)

### Senior Developer Review (AI)

**Reviewer:** Claude (Adversarial Code Review) on 2026-03-23
**Outcome:** Approved with fixes applied

**Issues Found and Fixed:**

- **[HIGH] AC 12-4-3 partial — `win_rate` not displayed in `PaperTradingMonitorTile`:** AC 12-4-3 explicitly requires "EA name, pair, days running, win rate, current P&L" — five data fields. The implementation displayed only four (win_rate was in the interface but never rendered). Fixed: added `<span class="financial-value win-rate">{item.win_rate.toFixed(1)}%</span>` to the populated-state row, plus `.win-rate` CSS rule.

- **[HIGH] Test gap — no assertion for `win_rate` display (AC 12-4-3):** `TradingCanvas.test.ts` had no test verifying `win_rate` appeared in populated output. Fixed: added test `'AC 12-4-3: renders win_rate as a displayed value'` asserting `win_rate.toFixed` appears in source.

- **[MEDIUM] Pass/fail indicators used raw `✓`/`✗` Unicode characters instead of Lucide icons:** Project memory (`feedback_icons_not_emoji.md`) mandates Lucide icons throughout the ITT UI. Fixed: replaced with `<CheckCircle size={12} />` / `<XCircle size={12} />` from `lucide-svelte`. Added test asserting `CheckCircle`/`XCircle` present and `'✓'`/`'✗'` absent.

- **[MEDIUM] Test coverage gap — Lucide icons not asserted in test suite:** Fixed: added test `'pass/fail uses Lucide icons (CheckCircle/XCircle) — not raw Unicode symbols'` to BacktestResultsTile describe block.

- **[LOW] `PaperTradingMonitorTile` `empty` state initialized to `false`:** Until `onMount` fires, component rendered empty `<ul>` instead of the empty-state message (brief blank flash). Fixed: `let empty = $state(true)` so empty-state message shows immediately until API responds with data.

**Post-fix test result:** 89/89 pass (2 new tests added). tile-grid.test.ts: 68/68 pass. Zero regressions.
