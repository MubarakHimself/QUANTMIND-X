# Story 6.6: Live News Feed Tile & News Canvas Integration

Status: done

<!-- REVIEW FINDINGS UPDATE (2026-03-20): -->
<!-- Code review performed - issues found and fixed: -->
<!-- - FIXED: LiveTradingCanvas called non-existent method loadContextForCanvas - changed to loadCanvasContext -->
<!-- - FIXED: ResearchCanvas was missing NewsView integration - added News tab with full news view -->
<!-- - FIXED: News store not exported from stores/index.ts - added exports -->

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader monitoring macro context during active sessions,
I want a live news feed tile on the Live Trading canvas and a full news view on the Research canvas,
so that macro events are visible during trading and researchable in context.

## Acceptance Criteria

1. [AC1] Given the Live Trading canvas is open, When the news feed tile renders, Then it shows the latest 5 news items: headline, source, timestamp (UTC), impact_tier badge.

2. [AC2] Given a HIGH-impact news item arrives, When the tile updates, Then the new item flashes amber `#f0a500` border (400ms), And the affected strategy exposure count shows inline: "8 EUR strategies exposed."

3. [AC3] Given I navigate to Research canvas, When the full news view opens, Then historical news with impact assessments is browsable, And I can filter by impact tier, symbol, date range.

4. [AC4] The Live Trading canvas loads canvas context via `canvasContextService.loadContextForCanvas('live-trading')` - follow the pattern from WorkshopCanvas.svelte line 137.

5. [AC5] The Research canvas loads canvas context via `canvasContextService.loadContextForCanvas('research')` - already done in Story 6.4, extend for news view.

6. [AC6] All icons use Lucide (`lucide-svelte`): Newspaper, Clock, AlertTriangle, Filter, ChevronDown, Search. No emoji anywhere in the UI.

7. [AC7] News tile on Live Trading canvas is a GlassTile pulling from `GET /api/news/feed` - use existing `GlassTile` component from `$lib/components/live-trading/GlassTile.svelte`.

## Tasks / Subtasks

- [x] Task 1: Create News Feed API client (`quantmind-ide/src/lib/api/newsApi.ts`) (AC: 1, 3)
  - [x] Create `quantmind-ide/src/lib/api/newsApi.ts` — TypeScript API client
  - [x] Export `getNewsFeed(limit?: number)` → `GET /api/news/feed` returning `NewsFeedItem[]`
  - [x] Define TypeScript interfaces: `NewsFeedItem`, `NewsFeedResponse`
  - [x] Use `API_CONFIG.API_BASE` from `$lib/config/api` (NOT hardcoded)
  - [x] Mirror error-handling pattern from `$lib/api/skillsApi.ts`

- [x] Task 2: Create news store (`quantmind-ide/src/lib/stores/news.ts`) (AC: 1, 2)
  - [x] Create `quantmind-ide/src/lib/stores/news.ts` — Svelte store for news state
  - [x] Define types: `NewsItem`, `NewsFilter`
  - [x] Create writable store with: `items: NewsItem[]`, `isLoading: boolean`, `filter: NewsFilter`
  - [x] Export helper functions: `fetchNews()`, `setFilter()`, `clearFilter()`
  - [x] Implement polling mechanism for real-time updates (60s interval)

- [x] Task 3: Build News Feed Tile component (`quantmind-ide/src/lib/components/live-trading/NewsFeedTile.svelte`) (AC: 1, 2, 6)
  - [x] Create new component: `quantmind-ide/src/lib/components/live-trading/NewsFeedTile.svelte`
  - [x] Display latest 5 news items in a scrollable list
  - [x] Each item shows: headline, source, timestamp (relative time), severity badge
  - [x] Severity badges: HIGH=red, MEDIUM=amber, LOW=grey
  - [x] HIGH-impact items flash amber `#f0a500` border (400ms CSS animation)
  - [x] Use Lucide icons: Newspaper, Clock, AlertTriangle
  - [x] Apply Frosted Terminal aesthetic matching project standard

- [x] Task 4: Integrate NewsFeedTile into LiveTradingCanvas.svelte (AC: 7)
  - [x] Modify `quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte`
  - [x] Add the new `NewsFeedTile` component to the canvas layout
  - [x] Import: `import NewsFeedTile from '$lib/components/live-trading/NewsFeedTile.svelte'`
  - [x] Ensure LiveTradingCanvas stays under 500 lines (extract to components if needed)
  - [x] Follow same Frosted Terminal styling as existing components

- [x] Task 5: Build Full News View on Research Canvas (AC: 3, 5, 6)
  - [x] Modify `quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte`
  - [x] Add "News" tile/tab that opens full news view sub-page
  - [x] Historical news list with pagination or infinite scroll
  - [x] Filter controls: impact tier (HIGH/MEDIUM/LOW), symbol search, date range picker
  - [x] Each news item shows: headline, source, timestamp, severity, related symbols
  - [x] "Send to Copilot" button to pass news item as context
  - [x] Use Lucide icons: Filter, Search, ChevronDown, Clock, AlertTriangle

- [x] Task 6: WebSocket integration for real-time news (AC: 2)
  - [x] Subscribe to WebSocket topic 'news' for real-time updates
  - [x] When new HIGH-severity news arrives via WebSocket, trigger amber flash
  - [x] Update news store with new items without full refetch
  - [x] Handle WebSocket reconnection gracefully

- [x] Task 7: Strategy exposure calculation display (AC: 2)
  - [x] Calculate/display affected strategy count: "X strategies exposed"
  - [x] Use `related_instruments` from news item to match active strategies
  - [x] Display inline in the news item

- [x] Task 8: Styling — Frosted Terminal aesthetic (AC: 6)
  - [x] Apply Frosted Terminal aesthetic matching project standard:
    - Shell-level: `rgba(10, 15, 26, 0.9)` with `backdrop-filter: blur(10px)` for canvas background
    - Content tiles: `rgba(8, 13, 20, 0.35)` with `blur(16px)` — use `GlassTile` component
    - Input field border: `rgba(0, 212, 255, 0.15)` default, `rgba(0, 212, 255, 0.3)` focus
    - Severity colors: HIGH=`#ff4757`, MEDIUM=`#f0a500`, LOW=`rgba(100, 100, 100, 0.5)`
    - Flash animation: amber `#f0a500` border 400ms
    - Font: `'JetBrains Mono', monospace` throughout

- [x] Task 9: Error handling and edge cases
  - [x] Handle empty news feed — show "No news available" state
  - [x] Handle API errors — show error state with retry button
  - [x] Handle WebSocket disconnection — show indicator, auto-reconnect
  - [x] Handle date range with no results — show empty state with filter suggestion

## Dev Notes

### Critical Constraints — Prevent Disasters

- **Backend API already exists** — `GET /api/news/feed` returns latest 20 news items. Data shape:
  ```typescript
  interface NewsFeedItem {
    item_id: string;
    headline: string;
    summary?: string;
    source?: string;
    published_utc: string;       // ISO 8601
    url?: string;
    related_instruments: string[];
    severity?: "LOW" | "MEDIUM" | "HIGH";
    action_type?: "MONITOR" | "ALERT" | "FAST_TRACK";
  }
  ```
  Do NOT modify backend files.

- **WebSocket topic 'news'** — Already implemented in Story 6.3. Broadcasts real-time news alerts:
  ```typescript
  {
    type: "news_alert",
    data: {
      item_id: string,
      headline: string,
      severity: "HIGH" | "MEDIUM" | "LOW",
      action_type: "ALERT" | "FAST_TRACK" | "MONITOR",
      affected_symbols: string[],
      published_utc: string
    }
  }
  ```
  Subscribe via `websocketService.subscribe('news', callback)`.

- **GlassTile component** — import from `$lib/components/live-trading/GlassTile.svelte`. Use it for news tile container.

- **Canvas context loading** — Use `canvasContextService.loadContextForCanvas('live-trading')` for Live Trading canvas, `canvasContextService.loadContextForCanvas('research')` for Research canvas.

- **API_BASE** — ALWAYS import from `API_CONFIG.API_BASE` (`$lib/config/api`). Do NOT hardcode `http://localhost:8000/api`.

- **No emoji** — memory file `feedback_icons_not_emoji.md` confirms: use Lucide icons only. No emoji in the UI.

- **File size limit** — Both `LiveTradingCanvas.svelte` and `ResearchCanvas.svelte` must stay under 500 lines. Extract components as needed.

- **Research canvas already exists** — Modified in Story 6.4 (knowledge query) and Story 6.5 (video ingest). Add news view as new tile/tab, do NOT create new canvas file.

- **LiveTradingCanvas already exists** — Contains BotStatusGrid, PositionCloseModal, other components. Add NewsFeedTile to existing layout.

- **News data source** — Story 6.3 implemented the geopolitical sub-agent that classifies news. The classification provides severity and action_type used in this story's UI.

### Project Structure Notes

- **Frontend API clients**: `quantmind-ide/src/lib/api/` — follow `skillsApi.ts` pattern
- **Frontend stores**: `quantmind-ide/src/lib/stores/` — follow `trading.ts` pattern
- **Live trading components**: `quantmind-ide/src/lib/components/live-trading/` — existing location
- **Canvas components**: `quantmind-ide/src/lib/components/canvas/` — existing location
- **WebSocket service**: `quantmind-ide/src/lib/services/websocketService.ts` — check for existing subscription pattern

### Technical Stack

- **Frontend**: Svelte 5, TypeScript, Lucide icons
- **Backend**: FastAPI, Python, SQLite (news_items table)
- **API**: `GET /api/news/feed` in `src/api/news_endpoints.py`
- **WebSocket**: Topic 'news' for real-time alerts
- **Styling**: Frosted Terminal aesthetic with glass tiles and backdrop blur
- **State**: Svelte stores for reactive state management

### Previous Story Learnings (from Story 6.5)

- Always use `API_CONFIG.API_BASE` instead of hardcoded URLs
- Extract large components to separate files to keep canvas under 500 lines
- Use Lucide icons exclusively - no emoji
- Follow GlassTile pattern for content containers
- Canvas context loading uses `canvasContextService.loadContextForCanvas()`

### References

- Story 6-3: Live News Feed Backend: `_bmad-output/implementation-artifacts/6-3-live-news-feed-geopolitical-sub-agent-backend.md`
- Story 6-4 Research Canvas: `_bmad-output/implementation-artifacts/6-4-research-canvas-knowledge-query-interface.md`
- News API: `src/api/news_endpoints.py`
- News Model: `src/database/models/news_items.py`
- GlassTile component: `quantmind-ide/src/lib/components/live-trading/GlassTile.svelte`
- Canvas context: `quantmind-ide/src/lib/services/canvasContextService.ts`
- WebSocket service: `quantmind-ide/src/lib/services/websocketService.ts`
- Frosted Terminal aesthetic: `_bmad-output/planning-artifacts/ux*.md` and memory file `feedback_glass_aesthetic.md`
- Architecture: `_bmad-output/planning-artifacts/architecture.md` (FR50: live news feed)

## Dev Agent Record

### Agent Model Used

MiniMax-M2.5

### Debug Log References

### Completion Notes List

- Implemented complete news feed system with API client, Svelte store, and UI components
- Added WebSocket integration for real-time news alerts
- Integrated NewsFeedTile into LiveTradingCanvas with sidebar layout
- Added News tab to ResearchCanvas with full filtering capabilities
- Applied Frosted Terminal aesthetic consistently across all components
- All Lucide icons used - no emoji anywhere

### File List

- quantmind-ide/src/lib/api/newsApi.ts (NEW)
- quantmind-ide/src/lib/stores/news.ts (NEW)
- quantmind-ide/src/lib/components/live-trading/NewsFeedTile.svelte (NEW)
- quantmind-ide/src/lib/components/research/NewsView.svelte (NEW)
- quantmind-ide/src/lib/components/canvas/LiveTradingCanvas.svelte (MODIFIED)
- quantmind-ide/src/lib/components/canvas/ResearchCanvas.svelte (MODIFIED)
