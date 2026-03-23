# Story 6.4: Research Canvas — Knowledge Query Interface

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader querying the knowledge base,
I want the Research canvas to provide a search interface over the full knowledge base,
so that I can directly query all indexed content without going through Copilot (FR43, FR49).

## Acceptance Criteria

1. [AC1] When the Research canvas loads, it renders a full-width search bar at the top, a source filter row (chips/tabs: All / Articles / Books / Logs / Personal), a hypothesis pipeline tile, a video ingest entry tile, and a knowledge base status tile showing source health from `GET /api/knowledge/sources`.

2. [AC2] When I enter a query and press Enter (or click Search), `POST /api/knowledge/search` is called with `{ query, sources, limit: 10 }` — results appear within ≤2 seconds. Each result card shows: title, source type badge (color-coded), relevance score (0.0–1.0), excerpt (first 300 chars), and a "View Full" button.

3. [AC3] While a search is in progress, a skeleton loader (pulsing placeholder tiles) replaces the result list. If no results are returned, a "No results found" empty state renders. If the API errors (network/offline PageIndex), an error banner shows with the warning text from the response `warnings[]` field.

4. [AC4] When I click "View Full" on a result, a document detail sub-page opens (replacing the tile grid), with BreadcrumbNav showing "Research > [Document Title]". The full excerpt renders with the source type badge, relevance score, provenance metadata (source_url, indexed_at_utc), and a "Send to Copilot" button.

5. [AC5] When I click "Send to Copilot", the document title + excerpt are prepended as context into the Copilot panel's input field via the `floor_manager/chat` endpoint, with `canvas_context: 'research'`. The Research canvas remains open (no navigation away).

6. [AC6] The source filter chips filter results client-side by `source_type` field. Selecting "Articles" shows only `source_type === 'articles'`, "Books" shows `source_type === 'books'`, "Logs" shows `source_type === 'logs'`, "Personal" shows `source_type === 'personal'`, "All" shows all results. Switching filters does NOT re-fetch from the API.

7. [AC7] The Hypothesis Pipeline tile and Video Ingest Entry tile are visible placeholder tiles (GlassTile) in the canvas layout with labels "Hypothesis Pipeline" and "Video Ingest" — they are non-functional stubs for Story 6.5/7.1 (do NOT implement pipeline logic here, just render the tiles). The hypothesis pipeline tile is the primary Alpha Forge workflow entry point per UX spec.

8. [AC8] The canvas loads the Research canvas context via `GET /api/canvas-context/research` on mount — same pattern as other canvases using `canvasContextService.loadCanvasContext('research')`.

9. [AC9] All icons use Lucide (`lucide-svelte`): Search, Filter, BookOpen, Newspaper, FlaskConical, ChevronRight, Home, Loader, AlertCircle, ExternalLink, Send, X. No emoji anywhere in the UI (per project aesthetic standard).

10. [AC10] The Research canvas file is `quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte`. It replaces the current `CanvasPlaceholder` stub entirely.

## Tasks / Subtasks

- [x] Task 1: Create Knowledge API client (`quantmind-ide/src/lib/api/knowledgeApi.ts`) (AC: 1, 2, 3)
  - [x] Create `quantmind-ide/src/lib/api/knowledgeApi.ts` — TypeScript API client
  - [x] Export `searchKnowledge(query, sources?, limit?)` → `POST /api/knowledge/search` returning `KnowledgeSearchResponse`
  - [x] Export `getKnowledgeSources()` → `GET /api/knowledge/sources` returning `KnowledgeSourceStatus[]`
  - [x] Define and export TypeScript interfaces: `KnowledgeSourceStatus`, `KnowledgeSearchResult`, `KnowledgeSearchResponse`
  - [x] Use `API_CONFIG.API_BASE` from `$lib/config/api` (NOT hardcoded `http://localhost:8000/api`)
  - [x] Mirror the same error-handling pattern as `$lib/api/skillsApi.ts` (apiFetch wrapper, throws on non-ok)

- [x] Task 2: Build ResearchCanvas.svelte — main layout and search (AC: 1, 2, 3, 6, 7, 8, 9, 10)
  - [x] Replace `quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte` entirely — remove CanvasPlaceholder
  - [x] `onMount`: call `canvasContextService.loadCanvasContext('research')` + `getKnowledgeSources()`
  - [x] Render search bar (full-width, prominent, Enter-to-search)
  - [x] Render source filter chip row: All / Articles / Books / Logs / Personal — stateful `activeFilter` variable
  - [x] Render tile row: Hypothesis Pipeline tile (GlassTile stub, FlaskConical icon), Video Ingest tile (GlassTile stub, Newspaper icon), Knowledge Base Status tile (GlassTile, shows source statuses)
  - [x] Render search results list (filtered client-side by `activeFilter`)
  - [x] Skeleton loader while `isSearching` is true (3 pulsing GlassTile-shaped placeholder divs)
  - [x] Empty state when `results.length === 0 && !isSearching && hasSearched`
  - [x] Error banner when `searchError` is set (show warning list from API response)
  - [x] Each result card: title, source badge, relevance score badge, excerpt (truncated to 300 chars), "View Full" button
  - [x] Use Lucide icons throughout — no emoji

- [x] Task 3: Build document detail sub-page (AC: 4, 5)
  - [x] Stateful `selectedResult: KnowledgeSearchResult | null` — when set, render detail sub-page instead of tile grid
  - [x] Detail sub-page: BreadcrumbNav ("Research > [title]"), full excerpt, source type badge, relevance score, provenance block (source_url, indexed_at_utc), "Send to Copilot" button, "← Back" via BreadcrumbNav click
  - [x] BreadcrumbNav "Research" crumb click → `selectedResult = null` (navigate back to tile grid)
  - [x] "Send to Copilot" button: build context string `"[Knowledge Context] Title: ${title}\n\nExcerpt: ${excerpt}"` and trigger `POST /floor-manager/chat` with `{ message: contextString, canvas_context: 'research', session_id: null }` — show a brief "Sent to Copilot" success toast (auto-dismiss 2s), do NOT navigate away

- [x] Task 4: Add source filter logic (AC: 6)
  - [x] `activeFilter: 'all' | 'articles' | 'books' | 'logs' | 'personal'` reactive variable (default: 'all')
  - [x] `filteredResults` derived: `activeFilter === 'all' ? results : results.filter(r => r.source_type === activeFilter)`
  - [x] Filter chips render `filteredResults.length` count in badge when filter is active
  - [x] Switching filters triggers no API call — purely client-side reactive filtering

- [x] Task 5: Styling — Frosted Terminal aesthetic (AC: 1, 9)
  - [x] Apply Frosted Terminal aesthetic matching project standard:
    - Shell-level: `rgba(10, 15, 26, 0.9)` with `backdrop-filter: blur(10px)` for canvas background
    - Content tiles (GlassTile): `rgba(8, 13, 20, 0.35)` with `blur(16px)` — use `GlassTile` component from `$lib/components/live-trading/GlassTile.svelte`
    - Search bar border: `rgba(0, 212, 255, 0.15)` default, `rgba(0, 212, 255, 0.3)` focus
    - Source type badge colors: articles=cyan `#00d4ff`, books=emerald `#00c896`, logs=amber `#f0a500`, personal=violet `#a78bfa`
    - Relevance score badge: green gradient `rgba(0, 200, 100, 0.15)` border
    - Font: `'JetBrains Mono', monospace` throughout
    - Active filter chip: `rgba(0, 212, 255, 0.15)` bg + `rgba(0, 212, 255, 0.3)` border + `#00d4ff` text
  - [x] Ensure canvas file stays under 500 lines (NFR-M3) — split into sub-components if needed

- [x] Task 6: Write tests in `quantmind-ide/src/lib/api/knowledgeApi.test.ts` (frontend) — optional but recommended
  - [x] Unit test `knowledgeApi.ts`: mock fetch, assert `POST /api/knowledge/search` called with correct body
  - [x] Unit test source filter: assert `SOURCE_FILTERS` and `SOURCE_BADGE_COLORS` are correctly defined
  - [x] Follow pattern from existing frontend test files (vitest, use mocks for fetch)

## Dev Notes

### Critical Constraints — Prevent Disasters

- **`ResearchCanvas.svelte` currently exists as a stub** — it imports `CanvasPlaceholder` from `./CanvasPlaceholder.svelte` and passes `canvasName="Research"`. You MUST replace its entire content. Do NOT add a new file — edit the existing `ResearchCanvas.svelte`.
- **Knowledge API already exists and is fully operational** — `POST /api/knowledge/search` and `GET /api/knowledge/sources` are live in `src/api/knowledge_endpoints.py`. The backend `KnowledgeSearchResponse` shape: `{ results: KnowledgeSearchResult[], total: int, query: str, warnings: str[] }`. Each `KnowledgeSearchResult`: `{ source_type: str, title: str, excerpt: str, relevance_score: float, provenance: { source_url, source_type, indexed_at_utc } }`. Do NOT modify backend files for this story.
- **Personal source type** — Story 6.2 added a `personal` PageIndex partition. However, the current `knowledge_endpoints.py` only fans out to `["articles", "books", "logs"]` (line 29: `KNOWN_COLLECTIONS`). The `personal` filter UI chip can exist but if filtered, the search may return 0 results from that source — this is expected behavior. Do NOT modify the backend collections list.
- **`GlassTile` component** — import from `$lib/components/live-trading/GlassTile.svelte`. It accepts `clickable: boolean` prop. Use it for all tile containers (hypothesis pipeline, video ingest, knowledge status, result cards).
- **`BreadcrumbNav` component** — existing component at `$lib/components/live-trading/BreadcrumbNav.svelte` is tightly coupled to Live Trading (imports `selectBot` from `trading` store). For Research canvas, do NOT use that component directly. Instead, implement inline breadcrumb markup in ResearchCanvas.svelte or create a new generic `ResearchBreadcrumb` in the same file — a simple `<nav>` with "Research" crumb and document title crumb.
- **"Send to Copilot" pattern** — there is NO shared "copilot input store". The correct pattern: call `POST ${API_BASE}/floor-manager/chat` directly with the context string as the `message`. The Copilot panel is a separate component — sending via the API is the correct approach. Optionally display a toast. Do NOT attempt to directly mutate the Workshop canvas message state.
- **Canvas context loading** — `canvasContextService.loadCanvasContext('research')` is the correct call. The `canvasContextService` is a singleton imported from `$lib/services/canvasContextService`. This is used by `WorkshopCanvas.svelte` (line 137) as the reference pattern.
- **API_BASE** — ALWAYS import from `API_CONFIG.API_BASE` (`$lib/config/api`). Do NOT hardcode `http://localhost:8000/api`. The API base resolves to CLOUDZY_API_URL in production.
- **Hypothesis Pipeline tile and Video Ingest tile are STUBS** — do NOT wire them to any API. They render as GlassTile components with an icon, label, and a subtle "Coming Soon" or "Story 6.5" label. The Hypothesis Pipeline tile is the future Alpha Forge Workflow 1 entry point per UX spec §186 and must be clearly labeled "Hypothesis Pipeline" with `FlaskConical` icon.
- **File size limit** — `ResearchCanvas.svelte` MUST stay under 500 lines (NFR-M3). If the component grows too large, extract result card rendering into a separate `ResearchResultCard.svelte` component in a new `quantmind-ide/src/lib/components/research/` directory.
- **No emoji** — memory file `feedback_icons_not_emoji.md` confirms: use Lucide icons only. No emoji in the UI.

### Architecture Mandates

Per architecture §1.3 (Knowledge & Vector Search Stack):
- Full-text search: PageIndex (3 Docker instances: articles, books, logs)
- Semantic search: ChromaDB + sentence-transformers (not directly exposed to UI — PageIndex handles fanout)
- The `POST /api/knowledge/search` endpoint fans out to ALL configured collections in parallel

Per architecture §FR43 (Semantic knowledge base query):
- Query interface is the Research canvas (this story)
- Results must include relevance score and provenance

Per architecture §FR49 (NL knowledge query via Copilot):
- Story 5.7 already handles NL query via Copilot's intent system
- This story adds DIRECT query (bypassing Copilot) per the story requirement

Per UX spec §Canvas 2 (Research):
- Primary role: Knowledge base, video ingest, hypothesis pipeline, prop firm research
- Alpha Forge pipeline entry: paste YouTube URL → pipeline initiates (Story 6.5 scope — tile only here)
- "View Full" → detail sub-page with BreadcrumbNav
- "Send to Copilot" passes document as context to active conversation

### Existing API Contract (POST /api/knowledge/search)

```typescript
// Request body
interface KnowledgeSearchRequest {
  query: string;          // min 1, max 2000 chars
  sources?: string[];     // subset of ['articles','books','logs']; null = all 3
  limit?: number;         // 1-100, default 5
}

// Response body
interface KnowledgeSearchResponse {
  results: KnowledgeSearchResult[];
  total: number;
  query: string;
  warnings: string[];     // offline instances reported here
}

interface KnowledgeSearchResult {
  source_type: string;    // 'articles' | 'books' | 'logs'
  title: string;          // filename (last path segment after '/')
  excerpt: string;        // first 300 chars of content
  relevance_score: number; // 0.0–1.0
  provenance: {
    source_url: string | null;
    source_type: string;
    indexed_at_utc: string | null;
  };
}
```

### Existing API Contract (GET /api/knowledge/sources)

```typescript
interface KnowledgeSourceStatus {
  id: string;             // 'articles' | 'books' | 'logs'
  type: string;
  status: string;         // 'online' | 'offline'
  document_count: number;
}
```

### New Frontend File: `quantmind-ide/src/lib/api/knowledgeApi.ts`

```typescript
import { API_CONFIG } from '$lib/config/api';

const API_BASE = API_CONFIG.API_BASE;

export interface KnowledgeSourceStatus { ... }
export interface KnowledgeSearchResult { ... }
export interface KnowledgeSearchResponse { ... }

export async function getKnowledgeSources(): Promise<KnowledgeSourceStatus[]> {
  const response = await fetch(`${API_BASE}/knowledge/sources`);
  if (!response.ok) throw new Error(`...`);
  return response.json();
}

export async function searchKnowledge(
  query: string,
  sources?: string[],
  limit = 10
): Promise<KnowledgeSearchResponse> {
  const response = await fetch(`${API_BASE}/knowledge/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, sources: sources ?? null, limit })
  });
  if (!response.ok) throw new Error(`...`);
  return response.json();
}
```

### ResearchCanvas Component Structure

```
ResearchCanvas.svelte
├── <header> — Canvas title + knowledge source status badges (online/offline counts)
├── <search-bar> — full-width input, Search button, Enter-to-search
├── <source-filter-row> — filter chips: All / Articles / Books / Logs / Personal
├── <tile-row> — 3 tiles in a row:
│   ├── GlassTile (Hypothesis Pipeline — FlaskConical — stub)
│   ├── GlassTile (Video Ingest — Newspaper — stub)
│   └── GlassTile (Knowledge Base — Database — source status)
├── <results-section>
│   ├── Skeleton loader (when isSearching)
│   ├── Error banner (when searchError)
│   ├── Empty state (when hasSearched && filteredResults.length === 0)
│   └── Result cards list (title, source badge, score, excerpt, "View Full")
└── <detail-sub-page> (when selectedResult !== null)
    ├── ResearchBreadcrumb (inline nav — "Research > [title]")
    ├── Document detail block
    └── "Send to Copilot" button
```

### Source Badge Color Mapping

```typescript
const SOURCE_BADGE_COLORS = {
  articles: '#00d4ff',   // cyan
  books:    '#00c896',   // emerald
  logs:     '#f0a500',   // amber
  personal: '#a78bfa',   // violet
};
```

### Send to Copilot Pattern

```typescript
async function sendTocopilot(result: KnowledgeSearchResult) {
  const contextMessage = `[Knowledge Context]\nTitle: ${result.title}\nSource: ${result.source_type}\n\nExcerpt:\n${result.excerpt}`;

  try {
    const response = await fetch(`${API_BASE}/floor-manager/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: contextMessage,
        canvas_context: 'research',
        session_id: null
      })
    });
    if (response.ok) {
      showToast = true;
      setTimeout(() => showToast = false, 2000);
    }
  } catch (e) {
    console.error('Failed to send to Copilot:', e);
  }
}
```

Note: The floor-manager chat endpoint is confirmed at `/api/floor-manager/chat` (router prefix is `/api/floor-manager` per `src/api/floor_manager_endpoints.py:27`). Use `${API_CONFIG.API_BASE}/floor-manager/chat` directly — `API_BASE` already includes `/api`.

### canvasContextService Pattern (from WorkshopCanvas.svelte)

```typescript
import { canvasContextService } from '$lib/services/canvasContextService';

onMount(async () => {
  try {
    await canvasContextService.loadCanvasContext('research');
  } catch (e) {
    console.error('Failed to load canvas context:', e);
  }
});
```

### Project Structure Notes

- Modified files:
  - `quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte` — replace CanvasPlaceholder stub with full implementation

- New files:
  - `quantmind-ide/src/lib/api/knowledgeApi.ts` — TypeScript API client for knowledge endpoints

- Possible new files (only if ResearchCanvas.svelte exceeds ~400 lines):
  - `quantmind-ide/src/lib/components/research/ResearchResultCard.svelte` — result card sub-component
  - `quantmind-ide/src/lib/components/research/ResearchDetailPage.svelte` — detail sub-page sub-component

- Do NOT modify:
  - `src/api/knowledge_endpoints.py` — backend is complete and tested
  - `src/api/knowledge_ingest_endpoints.py` — not in scope for this story
  - `quantmind-ide/src/lib/components/live-trading/BreadcrumbNav.svelte` — tightly coupled to Live Trading, implement inline in ResearchCanvas
  - `quantmind-ide/src/lib/components/canvas/index.ts` — `ResearchCanvas` already exported correctly

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 6.4 — Research Canvas Knowledge Query Interface]
- [Source: _bmad-output/planning-artifacts/architecture.md#1.3 — Knowledge & Vector Search Stack (FR43, FR49)]
- [Source: _bmad-output/planning-artifacts/ux-design-specification.md#Canvas 2 — Research (table §104, Alpha Forge entry §186)]
- [Source: src/api/knowledge_endpoints.py] — complete backend contract (POST /api/knowledge/search, GET /api/knowledge/sources)
- [Source: quantmind-ide/src/lib/components/canvas/WorkshopCanvas.svelte:29,137] — canvasContextService.loadCanvasContext pattern
- [Source: quantmind-ide/src/lib/api/skillsApi.ts] — apiFetch wrapper pattern for frontend API clients
- [Source: quantmind-ide/src/lib/components/live-trading/GlassTile.svelte] — GlassTile component (clickable prop, Tier 2 glass styling)
- [Source: quantmind-ide/src/lib/components/live-trading/BreadcrumbNav.svelte] — BreadcrumbNav design reference (do NOT import directly — implement inline)
- [Source: quantmind-ide/src/lib/config/api.ts] — API_CONFIG.API_BASE for all fetch calls
- [Source: quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte] — current stub to replace

### Previous Story Learnings (Story 6.3)

Story 6.3 (backend) established these patterns relevant to frontend integration:
- `POST /api/knowledge/search` endpoint confirmed live and registered under INCLUDE_CONTABO (Story 6.1)
- WebSocket topic `"news"` used for news feed — for Story 6.4, no WebSocket needed (REST only)
- `manager` singleton from `src.api.websocket_endpoints` — no frontend concern for this story
- Backend uses `KNOWN_COLLECTIONS = ["articles", "books", "logs"]` — Personal partition is NOT in this list; the filter chip is UI-only and will return 0 results from backend until KNOWN_COLLECTIONS is extended in a future story

Previous frontend stories (1.4, 1.5, 1.6-9) established:
- Svelte 4 reactive syntax (`$:`, `let`, `export let`) — project uses Svelte 4 (NOT Svelte 5 runes)
- `on:click` event handlers (Svelte 4 syntax) — NOT `onclick={}`
- Lucide icons from `lucide-svelte` with `size={N}` prop
- JetBrains Mono font throughout
- Frosted Terminal color palette: `#00d4ff` cyan, `#00c896` emerald, `#f0a500` amber, `#ff3b3b` red, `#e0e0e0` text

**CRITICAL Svelte 4 vs 5 Note:** The codebase uses **Svelte 4** syntax. Use `on:click`, `bind:value`, `$:` reactive declarations, NOT `onclick={}`, `$state()`, `$derived()`. `WorkshopCanvas.svelte` uses `onclick={}` which may be a Svelte 5 migration — verify the actual syntax used in recently-working canvas files (LiveTradingCanvas.svelte uses `on:click` — use this as the reference).

### Git Intelligence

Recent commits are all frontend canvas work (stories 1.4–1.6-9). The patterns to follow for frontend:
- New canvas component: replace existing stub, same file
- Imports from `$lib/` alias (SvelteKit path aliasing)
- No new routes or pages — Research canvas is already wired in `MainContent.svelte` line 1355
- Canvas is rendered inside `MainContent.svelte` `{:else if activeCanvas === 'research'}` block — no routing changes needed

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- Implemented Knowledge API client with TypeScript interfaces for search and source status
- Created full ResearchCanvas.svelte with Frosted Terminal aesthetic
- Search bar with Enter-to-search and loading states
- Source filter chips with client-side filtering (no API re-fetch)
- Three placeholder tiles: Hypothesis Pipeline, Video Ingest (stubs), Knowledge Base Status
- Document detail view with breadcrumb navigation and "Send to Copilot" functionality
- All acceptance criteria implemented: AC1-AC10
- Lucide icons throughout - no emoji
- Tests created using Vitest pattern
- Code review fixes applied: VideoIngestTile stub, 500-line limit met, scope creep removed

### File List

- `quantmind-ide/src/lib/api/knowledgeApi.ts` (new — TypeScript knowledge API client)
- `quantmind-ide/src/lib/api/knowledgeApi.test.ts` (new — Vitest unit tests for knowledge API)
- `quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte` (modified — replace CanvasPlaceholder stub with full implementation)
- `quantmind-ide/src/lib/components/research/ResearchResultCard.svelte` (new — result card sub-component)
- `quantmind-ide/src/lib/components/research/ResearchDetailPage.svelte` (new — detail view sub-component)
- `quantmind-ide/src/lib/components/research/ResearchSearchHeader.svelte` (new — search bar, filters, tiles sub-component)
- `quantmind-ide/src/lib/components/research/ResearchResultsSection.svelte` (new — results display sub-component)
- `quantmind-ide/src/lib/components/research/VideoIngestTile.svelte` (modified — converted to stub per AC7)
- `quantmind-ide/vitest.config.js` (modified — added $lib alias for tests)

### Review Fixes Applied

- **FIXED**: VideoIngestTile was fully functional but should be a stub per AC7. Converted to placeholder tile.
- **FIXED**: ResearchCanvas.svelte exceeded 500-line limit (687 lines). Refactored into 3 sub-components: ResearchSearchHeader, ResearchResultsSection, ResearchDetailPage. Now 284 lines.
- **FIXED**: Removed NewsView tab that was out of story scope (Story 6.6).
