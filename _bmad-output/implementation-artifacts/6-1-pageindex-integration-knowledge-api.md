# Story 6.1: PageIndex Integration & Knowledge API

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a research developer,
I want the 3 PageIndex Docker instances integrated with unified knowledge API endpoints,
so that agents and the Research canvas can query all knowledge sources in parallel.

## Acceptance Criteria

1. [AC1] `GET /api/knowledge/sources` returns all 3 PageIndex sources with `{ id, type, status, document_count }` for each.
2. [AC2] `POST /api/knowledge/search` fans out the query to all 3 PageIndex instances in parallel.
3. [AC3] Results from the search fanout are merged and ranked by relevance score.
4. [AC4] Each search result includes `{ source_type, title, excerpt, relevance_score, provenance }`.
5. [AC5] When a PageIndex instance is offline, it is skipped with a warning in the response and results from remaining instances are returned.
6. [AC6] All new endpoints are registered under `NODE_ROLE=contabo` (and `local`) in `src/api/server.py`.

## Tasks / Subtasks

- [x] Task 1: Create `src/api/knowledge_endpoints.py` with new unified knowledge API (AC: 1, 2, 3, 4, 5)
  - [x] Implement `GET /api/knowledge/sources` — queries `health_check()` + `get_stats()` on all 3 PageIndex instances
  - [x] Implement `POST /api/knowledge/search` with `KnowledgeSearchRequest` (query, sources?, limit?)
  - [x] Wire parallel fanout using `asyncio.gather()` with `return_exceptions=True` for graceful degradation
  - [x] Build result merger that sorts merged list by `relevance_score` descending
  - [x] Attach `provenance` block `{ source_url, indexed_at_utc, source_type }` to each result

- [x] Task 2: Extend `PageIndexClient` in `src/agents/knowledge/router.py` (AC: 2, 5)
  - [x] Add async `search_async()` method using `httpx.AsyncClient` (current `search()` is sync)
  - [x] Add async `health_check_async()` and `get_stats_async()` methods
  - [x] Timeout per-instance: 10s for search, 5s for health/stats — use existing `httpx.Timeout` pattern

- [x] Task 3: Register new router in `src/api/server.py` (AC: 6)
  - [x] Import `knowledge_unified_router` from `src/api/knowledge_endpoints.py`
  - [x] Include under `INCLUDE_CONTABO` block alongside existing `knowledge_router`

- [x] Task 4: Write tests in `tests/api/test_knowledge_sources.py` (AC: 1, 2, 5)
  - [x] Test `GET /api/knowledge/sources` — mock `PageIndexClient` health_check + get_stats
  - [x] Test `POST /api/knowledge/search` fanout — mock all 3 instances returning results
  - [x] Test graceful degradation — mock one instance raising `httpx.RequestError`, assert warning in response

## Dev Notes

### Critical Implementation Constraints

- **Do NOT modify `src/api/ide_knowledge.py`** — that file handles file upload/scraper sync. The new unified search API belongs in a new file `src/api/knowledge_endpoints.py`.
- **Existing `GET /api/knowledge`** in `ide_knowledge.py` lists local files — do not clash with `GET /api/knowledge/sources` (different paths, different concerns).
- **`KnowledgeRouter` is a synchronous singleton** (`kb_router` in `src/agents/knowledge/router.py`). For the new async API endpoints, add async variants to `PageIndexClient` directly using `httpx.AsyncClient`. Do not convert existing sync methods — other callers rely on them.
- **`search_all()` in `KnowledgeRouter`** calls each collection sequentially. The new `POST /api/knowledge/search` must use `asyncio.gather()` for true parallel fanout.
- **Provenance metadata**: PageIndex's `/search` response includes `source` and `score` fields. Map `source` → `provenance.source_url`, add `provenance.source_type` from the instance name (articles/books/logs), and `provenance.indexed_at_utc` is not available from PageIndex search results — omit or set to `null`.

### Existing Infrastructure to Reuse

- `PageIndexClient` in `src/agents/knowledge/router.py`: has `search()`, `health_check()`, `get_stats()` — base URLs from env vars `PAGEINDEX_ARTICLES_URL`, `PAGEINDEX_BOOKS_URL`, `PAGEINDEX_LOGS_URL` (defaults: `localhost:3000/3001/3002`)
- `KnowledgeRouter.search_all()` (same file): reference implementation of sequential fanout — the new async parallel version mirrors this logic
- `httpx>=0.24.0` is already in `requirements.txt` — use `httpx.AsyncClient` without adding deps
- `asyncio.gather(return_exceptions=True)` pattern — used in `src/api/websocket_endpoints.py` and other async handlers

### Node Role Gating

Knowledge endpoints belong to `contabo` + `local` nodes per architecture:

```python
# In src/api/server.py — INCLUDE_CONTABO block (line ~350+)
if INCLUDE_CONTABO:
    from src.api.knowledge_endpoints import router as knowledge_unified_router
    app.include_router(knowledge_unified_router)
```

### New File: `src/api/knowledge_endpoints.py`

Router prefix: `/api/knowledge` — tag: `knowledge-unified`

Endpoints to create:

```
GET  /api/knowledge/sources
POST /api/knowledge/search
```

Request/response models (Pydantic, define in the same file):

```python
class KnowledgeSourceStatus(BaseModel):
    id: str              # "articles" | "books" | "logs"
    type: str            # same as id for now
    status: str          # "online" | "offline"
    document_count: int  # from PageIndex /stats

class KnowledgeSearchRequest(BaseModel):
    query: str
    sources: Optional[List[str]] = None  # None = all 3
    limit: int = 5

class KnowledgeSearchResult(BaseModel):
    source_type: str
    title: str
    excerpt: str
    relevance_score: float
    provenance: dict     # { source_url, source_type, indexed_at_utc }

class KnowledgeSearchResponse(BaseModel):
    results: List[KnowledgeSearchResult]
    total: int
    query: str
    warnings: List[str]  # offline instances reported here
```

### Async PageIndex Search Pattern

Add to `PageIndexClient` in `src/agents/knowledge/router.py`:

```python
async def search_async(self, query: str, collection: str, limit: int = 5) -> List[Dict[str, Any]]:
    base_url = self._get_url_for_collection(collection)
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        try:
            response = await client.post(f"{base_url}/search", json={"query": query, "limit": limit})
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "content": item.get("content", ""),
                    "source": item.get("source", "unknown"),
                    "score": item.get("score", 0.0),
                    "collection": collection,
                })
            return results
        except Exception as e:
            logger.error(f"Async PageIndex search failed for {collection}: {e}")
            raise
```

### Fanout Pattern in Endpoint

```python
@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(request: KnowledgeSearchRequest):
    sources = request.sources or ["articles", "books", "logs"]
    client = PageIndexClient()
    tasks = [client.search_async(request.query, src, request.limit) for src in sources]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    merged = []
    warnings = []
    for source, result in zip(sources, raw_results):
        if isinstance(result, Exception):
            warnings.append(f"{source} instance offline: {str(result)[:100]}")
            continue
        for item in result:
            merged.append(KnowledgeSearchResult(
                source_type=source,
                title=item.get("source", "unknown"),
                excerpt=item.get("content", "")[:300],
                relevance_score=item.get("score", 0.0),
                provenance={"source_url": item.get("source"), "source_type": source, "indexed_at_utc": None}
            ))
    merged.sort(key=lambda r: r.relevance_score, reverse=True)
    return KnowledgeSearchResponse(results=merged, total=len(merged), query=request.query, warnings=warnings)
```

### Testing Standards

- Use `pytest-asyncio` (already in `requirements.txt`)
- Mock `httpx.AsyncClient` responses with `respx` or `unittest.mock.AsyncMock`
- Test file: `tests/api/test_knowledge_sources.py`
- Pattern: follows `tests/api/test_provider_config.py` — use `TestClient` from FastAPI for sync endpoint tests, `AsyncClient` for async

### Project Structure Notes

New files:
- `src/api/knowledge_endpoints.py` — new unified knowledge API router
- `tests/api/test_knowledge_sources.py` — tests for AC1, AC2, AC5

Modified files:
- `src/agents/knowledge/router.py` — add `search_async()`, `health_check_async()`, `get_stats_async()` to `PageIndexClient`
- `src/api/server.py` — register new router in `INCLUDE_CONTABO` block

Do NOT modify:
- `src/api/ide_knowledge.py` — file upload/scraper, different domain
- `src/agents/knowledge/retriever.py` — agent-facing tool, separate concern

### Architecture References

- [Source: _bmad-output/planning-artifacts/architecture.md#1.3 — Knowledge & Vector Search Stack]
- [Source: _bmad-output/planning-artifacts/architecture.md#Cross-Cutting Concerns — Knowledge provenance chain]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 6.1]
- PageIndex instance ports: articles=3000, books=3001, logs=3002 (env-overridable)
- NODE_ROLE gating: `INCLUDE_CONTABO = NODE_ROLE in ("contabo", "local")` [Source: src/api/server.py:104]

### Previous Story Learnings (Story 6.0)

Story 6.0 was an audit — no implementation. Key findings that apply:
- PageIndex client is already in `src/agents/knowledge/router.py` with correct URL env vars
- Three collections confirmed: `articles`, `books`, `logs`
- `health_check()` and `get_stats()` methods exist and work
- No existing `/sources` or unified `/search` endpoint — this story creates them
- `src/api/ide_knowledge.py` has `GET /api/knowledge` (file listing) — do not conflict

### Git Intelligence

Recent commits are all frontend/canvas work (Stories 1.4–1.6-9). Backend pattern to follow:
- New API files go in `src/api/` with `router = APIRouter(prefix="/api/...", tags=[...])`
- Models defined in same file or imported from `src/api/...models.py`
- Registered in `src/api/server.py` under the appropriate NODE_ROLE block
- Tests in `tests/api/` following `test_*.py` naming

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — implementation completed without blockers.

### Completion Notes List

- Created `src/api/knowledge_endpoints.py` with `GET /api/knowledge/sources` and `POST /api/knowledge/search` endpoints using Pydantic models `KnowledgeSourceStatus`, `KnowledgeSearchRequest`, `KnowledgeSearchResult`, `KnowledgeSearchResponse`.
- `GET /api/knowledge/sources` uses `asyncio.gather()` to check all 3 PageIndex instances in parallel; offline instances return `status="offline"` and `document_count=0` without error.
- `POST /api/knowledge/search` fans out using `asyncio.gather(return_exceptions=True)`; offline instances are skipped with a warning in the `warnings` list; results are merged and sorted by `relevance_score` descending.
- Added `search_async()` (10s timeout), `health_check_async()` (5s timeout), and `get_stats_async()` (5s timeout) to `PageIndexClient` using `httpx.AsyncClient`. Existing sync methods untouched.
- Registered `knowledge_unified_router` in `src/api/server.py` under the `INCLUDE_CONTABO` block alongside existing `knowledge_router`.
- 15 tests written and passing in `tests/api/test_knowledge_sources.py` covering AC1, AC2, AC3, AC4, AC5.
- Path `/api/knowledge/sources` and `/api/knowledge/search` do not conflict with existing `ide_knowledge.py` routes.

### File List

- `src/api/knowledge_endpoints.py` (new)
- `src/agents/knowledge/router.py` (modified — added `search_async`, `health_check_async`, `get_stats_async` to `PageIndexClient`)
- `src/api/server.py` (modified — registered `knowledge_unified_router` under `INCLUDE_CONTABO` block; added to MONITORED_ROUTERS)
- `tests/api/test_knowledge_sources.py` (new — 20 tests, all passing)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-19 | Implemented Story 6.1: created `knowledge_endpoints.py` (unified `/sources` + `/search` API), added async methods to `PageIndexClient`, registered router in `server.py`, added 15 tests — all passing | claude-sonnet-4-6 |
| 2026-03-19 | Code review (adversarial): Fixed 6 issues — (1) added `min_length=1/max_length=2000` on `query` field; (2) added `ge=1/le=100` constraint on `limit` field; (3) reuse `kb_router.client` singleton in endpoints instead of instantiating `PageIndexClient()` per request; (4) warn on unknown source names in sources filter instead of silently dropping; (5) improve `title` field to extract filename from source path rather than mapping raw path; (6) added `knowledge_unified_router` to `MONITORED_ROUTERS`; removed dead code from test; added 5 new tests for edge cases — 20 tests passing | claude-sonnet-4-6 |
