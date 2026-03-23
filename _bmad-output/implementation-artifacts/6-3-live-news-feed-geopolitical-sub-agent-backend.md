# Story 6.3: Live News Feed & Geopolitical Sub-agent Backend

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a trader monitoring macro events,
I want a live news feed pipeline with sub-90-second news-to-alert latency and a geopolitical sub-agent,
so that unscheduled events (ECB signals, geopolitical shocks) are detected and surfaced within 90 seconds (FR50, Journey 46).

## Acceptance Criteria

1. [AC1] A `NewsItem` SQLAlchemy model is created in `src/database/models/news_items.py` with fields: `item_id` (UUID string PK), `headline`, `summary`, `source`, `published_utc` (DateTime), `url`, `related_instruments` (JSON), `severity` (LOW/MEDIUM/HIGH/None), `action_type` (MONITOR/ALERT/FAST_TRACK/None), `classified_at_utc` (DateTime, nullable), `created_at_utc` (DateTime). Model is registered in `src/database/models/__init__.py`.

2. [AC2] A `NewsProvider` abstract base class and `FinnhubProvider` concrete implementation are created in `src/knowledge/news/` — `FinnhubProvider.fetch_latest(since_utc)` returns `List[NewsItem]` and is authenticated via `FINNHUB_API_KEY` env var. If the key is missing, a `503` is raised with `{ error: "FINNHUB_API_KEY not configured" }`.

3. [AC3] A `GeopoliticalSubAgent` class is created in `src/knowledge/news/geopolitical_subagent.py` — it accepts a `NewsItem`, calls Claude Haiku (`claude-3-haiku-20240307`) via the Anthropic SDK, and returns `{ impact_tier: HIGH|MEDIUM|LOW, affected_symbols: List[str], event_type: str, action_type: MONITOR|ALERT|FAST_TRACK }`. Classification completes in < 30 seconds per item (Haiku latency).

4. [AC4] A `NewsFeedPoller` background service is created in `src/knowledge/news/poller.py` — it polls Finnhub every 60 seconds via APScheduler `AsyncIOScheduler`, stores new `NewsItem` rows to SQLite, and invokes `GeopoliticalSubAgent.classify()` for each new article. If Finnhub is offline, it retries with exponential backoff (max 3 attempts, delays: 10s, 20s, 40s) before logging ERROR and skipping the cycle.

5. [AC5] `POST /api/news/alert` stores a pre-classified `NewsAlert` payload and broadcasts a WebSocket event to `topic="news"` connected clients. Alert payload shape: `{ item_id, headline, severity, action_type, affected_symbols, published_utc }`.

6. [AC6] `GET /api/news/feed` returns the latest 20 `NewsItem` rows from SQLite ordered by `published_utc DESC`, including the `severity` and `action_type` fields set by the geopolitical sub-agent.

7. [AC7] `GET /api/news/feed` and `POST /api/news/alert` are registered under `NODE_ROLE=contabo` (and `local`) in `src/api/server.py`.

8. [AC8] When `severity=HIGH` and `action_type=ALERT`, the poller sends department mail to the Copilot system so the alert is surfaced in the Copilot panel (use existing `floor_manager` mail bus pattern from Story 5.4).

9. [AC9] Poller lifecycle (`start`/`stop`) is wired to FastAPI `lifespan` events in `src/api/server.py` — poller starts on app startup (Contabo only) and stops cleanly on shutdown.

## Tasks / Subtasks

- [x] Task 1: Create `NewsItem` database model (AC: 1)
  - [x] Create `src/database/models/news_items.py` with `NewsItem` SQLAlchemy model — fields per AC1 schema
  - [x] Add `Index` on `published_utc` and `severity` for feed query performance
  - [x] Add `NewsItem` export to `src/database/models/__init__.py` alongside existing market/trading model imports
  - [x] Verify Alembic or direct `Base.metadata.create_all()` path — follow pattern from `src/database/models/risk_params.py`

- [x] Task 2: Create `src/knowledge/news/` package with NewsProvider abstraction (AC: 2)
  - [x] Create `src/knowledge/news/__init__.py`
  - [x] Create `src/knowledge/news/provider.py` — `NewsProvider` ABC with `fetch_latest(since_utc: datetime) -> List[NewsItem]` abstract method
  - [x] Create `src/knowledge/news/finnhub_provider.py` — `FinnhubProvider(NewsProvider)` using `finnhub-python` SDK (`import finnhub`) initialized with `FINNHUB_API_KEY` env var
    - [x] Use `finnhub.Client(api_key=api_key).general_news()` — wrap sync call in `asyncio.get_event_loop().run_in_executor(None, ...)`
    - [x] Raise `503` if `FINNHUB_API_KEY` not set
    - [x] Deduplicate by `item_id` (Finnhub `id` field → cast to string)
    - [x] Map Finnhub response fields: `id→item_id`, `headline→headline`, `summary→summary`, `source→source`, `datetime→published_utc` (UNIX timestamp → UTC datetime), `url→url`, `related→related_instruments`

- [x] Task 3: Create `GeopoliticalSubAgent` (AC: 3)
  - [x] Create `src/knowledge/news/geopolitical_subagent.py`
  - [x] Use `anthropic.Anthropic()` — same init pattern as `ResearchSubAgent._initialize_llm()` in `src/agents/departments/subagents/research_subagent.py`
  - [x] Model: `claude-3-haiku-20240307`, `max_tokens=400` (classification only — no long response needed)
  - [x] System prompt: classify the news headline + summary and return JSON `{ impact_tier, affected_symbols, event_type, action_type }` — include schema in prompt
  - [x] Parse JSON response safely — use `json.loads()` with `try/except`, fallback to `{ impact_tier: "LOW", affected_symbols: [], event_type: "unknown", action_type: "MONITOR" }` on parse failure
  - [x] Log classification at DEBUG level, ERROR on LLM failure

- [x] Task 4: Create `NewsFeedPoller` background service (AC: 4, 8)
  - [x] Create `src/knowledge/news/poller.py`
  - [x] Use `APScheduler AsyncIOScheduler` — follow pattern from `src/integrations/github_ea_scheduler.py` (already uses `AsyncIOScheduler` + `IntervalTrigger`)
  - [x] Poll every 60s: call `provider.fetch_latest(since_utc=last_polled_utc)` → persist new items to DB → classify each via `GeopoliticalSubAgent`
  - [x] Track `last_polled_utc` in-memory (start from `utcnow() - 5 minutes` on first run to catch recent articles)
  - [x] Exponential backoff retry (max 3 attempts, delays: 10s/20s/40s) — manual `asyncio.sleep` loop, same pattern as Story 6.2's `_retry_with_backoff`
  - [x] On HIGH+ALERT: broadcast via WebSocket `manager.broadcast(payload, topic="news")`
  - [x] On HIGH+FAST_TRACK: log WARNING "FAST_TRACK event detected — manual fast-track workflow required" (actual Fast-Track Workflow is Epic 8 scope — do NOT stub it)

- [x] Task 5: Create `src/api/news_endpoints.py` (AC: 5, 6, 7)
  - [x] Router prefix: `/api/news`, tag: `news`
  - [x] `GET /api/news/feed` — query DB for latest 20 `NewsItem` rows ordered by `published_utc DESC`, return list
  - [x] `POST /api/news/alert` — accept `NewsAlertRequest`, persist to DB (update `severity`/`action_type` on existing item or insert), broadcast via `manager.broadcast(payload, topic="news")`
  - [x] Import `manager` from `src.api.websocket_endpoints` for WebSocket broadcast
  - [x] Pydantic models in same file: `NewsFeedItem`, `NewsAlertRequest`, `NewsAlertResponse`

- [x] Task 6: Register router and wire lifespan (AC: 7, 9)
  - [x] In `src/api/server.py` INCLUDE_CONTABO block: import and register `news_router` from `src.api.news_endpoints` in the `# Knowledge & Research` section after `knowledge_unified_router`
  - [x] Add poller start/stop to FastAPI lifespan — use `@app.on_event("startup"/"shutdown")` pattern (same as existing event handlers)
  - [x] Guard poller startup: only start if `INCLUDE_CONTABO` is True and `FINNHUB_API_KEY` is set in env

- [x] Task 7: Write tests in `tests/api/test_news_feed.py` (AC: 1, 2, 5, 6)
  - [x] Test `GET /api/news/feed` — seed DB with mock `NewsItem` rows, assert 200 + latest 20 returned, ordered by `published_utc DESC`
  - [x] Test `POST /api/news/alert` — mock WebSocket broadcast, assert 200 + broadcast called with correct topic
  - [x] Test `FinnhubProvider` — mock `finnhub.Client`, assert correct field mapping and deduplication
  - [x] Test `GeopoliticalSubAgent` — mock Anthropic client, assert JSON parsing and fallback behavior on malformed response
  - [x] Test poller retry logic — mock provider raising exception 2 times then succeeding, assert 3rd attempt succeeds
  - [x] Follow `tests/api/test_provider_config.py` pattern: use `TestClient(FastAPI())` + `unittest.mock`

## Dev Notes

### Critical Constraints — Prevent Disasters

- **`src/knowledge/news/` is a NEW package** — the `src/knowledge/` directory does NOT exist yet (confirmed by audit). Create from scratch. Do NOT use `src/router/sensors/news.py` (`NewsSensor`) for live feed — that is a calendar-driven kill-zone enforcer, completely separate concern. Do not modify `NewsSensor`.
- **APScheduler is already in `requirements.txt` line 54** (`apscheduler>=3.10.0`). Use `AsyncIOScheduler` from `apscheduler.schedulers.asyncio` — same as `src/integrations/github_ea_scheduler.py` and `src/agents/cron/scheduler.py`.
- **Finnhub SDK**: `finnhub-python` is NOT in `requirements.txt` — add `finnhub-python>=2.4.0` to `requirements.txt`. Use `import finnhub` and `finnhub.Client(api_key=key)`. The SDK is synchronous — wrap in `run_in_executor`.
- **`news_items` table does NOT exist yet** — create it as a new SQLAlchemy model. Do NOT assume Alembic migration — follow the `src/database/models/risk_params.py` pattern which uses `Base.metadata.create_all()` at app startup.
- **WebSocket broadcast**: Import the global `manager` from `src.api.websocket_endpoints` — it is already initialized as a module-level singleton (`manager = ConnectionManager(...)`). Use `await manager.broadcast(payload, topic="news")`.
- **GeopoliticalSubAgent is Haiku-tier** (architecture §13.3) — model string MUST be `"claude-3-haiku-20240307"`. Do NOT use Sonnet or Opus. This is for < 30s classification latency.
- **Fast-Track workflow (§13.4) is Epic 8 scope** — do NOT implement it. On FAST_TRACK action_type, log a WARNING and store the flag in the DB. No template matching, no deployment logic.
- **No `finnhub-python` in requirements at time of writing** — add it. Do not use raw `httpx` REST calls for Finnhub (SDK is preferred for type safety and rate limit handling).
- **FINNHUB_API_KEY guard**: If `FINNHUB_API_KEY` is not set in env, the poller should NOT start (log WARNING and skip). `GET /api/news/feed` still works (returns stored items). `POST /api/news/alert` still works.
- **Sub-agent is NOT a department sub-agent** — `GeopoliticalSubAgent` is a lightweight classification caller, not a full `ResearchSubAgent`. It does NOT use the department mail system for internal routing — it is called directly from the poller. Only the alert broadcast (HIGH+ALERT) touches the mail/websocket system.

### Architecture Mandates from §13.3

Per architecture decision §13.3 (Live News Feed):
- Poll every **60 seconds** — not configurable per story scope (hardcode as `POLL_INTERVAL_SECONDS = 60`)
- Store in `news_items` SQLite table — SQLAlchemy ORM, same DB as the rest of the app
- `GET /api/news/feed` returns latest 20 items
- `severity` field is nullable until geopolitical sub-agent classifies the item
- `action_type` is nullable until classified

Per architecture §13.4 (Fast-Track): do NOT implement the Fast-Track Workflow. Log only.

### New Package: `src/knowledge/news/`

```
src/knowledge/news/__init__.py          # Package init, export NewsItem, FinnhubProvider, GeopoliticalSubAgent
src/knowledge/news/provider.py          # NewsProvider ABC
src/knowledge/news/finnhub_provider.py  # FinnhubProvider implementation
src/knowledge/news/geopolitical_subagent.py  # Claude Haiku classifier
src/knowledge/news/poller.py            # NewsFeedPoller (APScheduler)
```

### NewsItem SQLAlchemy Model

```python
# src/database/models/news_items.py
class NewsItem(Base):
    __tablename__ = 'news_items'
    item_id = Column(String(100), primary_key=True)
    headline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    source = Column(String(200), nullable=True)
    published_utc = Column(DateTime, nullable=False, index=True)
    url = Column(String(500), nullable=True)
    related_instruments = Column(JSON, nullable=True)    # ["EURUSD", "GBPUSD"]
    severity = Column(String(10), nullable=True, index=True)    # LOW / MEDIUM / HIGH
    action_type = Column(String(20), nullable=True)             # MONITOR / ALERT / FAST_TRACK
    classified_at_utc = Column(DateTime, nullable=True)
    created_at_utc = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
```

### Finnhub General News Endpoint

Finnhub provides `client.general_news(category='general', min_id=0)`. Category can be `'general'`, `'forex'`, `'crypto'`, `'merger'`. For macro/geopolitical use `'general'` and `'forex'` categories.

```python
# FinnhubProvider field mapping from Finnhub API response:
# finnhub_item = { 'id': 123, 'headline': '...', 'summary': '...', 'source': '...',
#                  'datetime': 1710000000, 'url': '...', 'related': 'EURUSD,GBPUSD', 'category': 'forex' }
# → item_id = str(finnhub_item['id'])
# → published_utc = datetime.utcfromtimestamp(finnhub_item['datetime']).replace(tzinfo=timezone.utc)
# → related_instruments = [s.strip() for s in finnhub_item.get('related', '').split(',') if s.strip()]
```

Deduplicate: before persisting, check `db.query(NewsItem).filter_by(item_id=item_id).first()` — skip if exists.

### GeopoliticalSubAgent Prompt Template

```python
CLASSIFICATION_PROMPT = """You are a geopolitical and macroeconomic news classifier for a forex trading system.

Classify the following news article and respond ONLY with valid JSON, no other text:

Headline: {headline}
Summary: {summary}

Return JSON with exactly these fields:
{{
  "impact_tier": "HIGH" or "MEDIUM" or "LOW",
  "affected_symbols": ["EURUSD", "GBPUSD"],  // forex pairs affected (empty list if none)
  "event_type": "central_bank" or "geopolitical" or "economic_data" or "market_shock" or "other",
  "action_type": "FAST_TRACK" or "ALERT" or "MONITOR"
}}

Use:
- HIGH + FAST_TRACK: Unexpected central bank actions, geopolitical shocks, major market disruptions
- HIGH + ALERT: Significant economic data surprises, scheduled high-impact events
- MEDIUM + MONITOR: Regular economic data, earnings, moderate news
- LOW + MONITOR: Background news, low market relevance"""
```

### APScheduler Poller Pattern

Follow `src/integrations/github_ea_scheduler.py` for scheduler init:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

class NewsFeedPoller:
    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._provider = FinnhubProvider()
        self._sub_agent = GeopoliticalSubAgent()
        self._last_polled_utc = datetime.now(timezone.utc) - timedelta(minutes=5)

    def start(self):
        self._scheduler.add_job(
            self._poll_cycle,
            trigger=IntervalTrigger(seconds=POLL_INTERVAL_SECONDS),
            id='news_feed_poll',
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info("NewsFeedPoller started (60s interval)")

    def stop(self):
        self._scheduler.shutdown(wait=False)
        logger.info("NewsFeedPoller stopped")

    async def _poll_cycle(self):
        # fetch → persist → classify → broadcast HIGH+ALERT
        ...
```

### API Endpoint File: `src/api/news_endpoints.py`

```
Router prefix: /api/news
Tags: ["news"]

GET  /api/news/feed         → NewsFeedResponse (list of NewsItem dicts, latest 20)
POST /api/news/alert        → NewsAlertRequest → NewsAlertResponse + WebSocket broadcast
```

Pydantic models:

```python
class NewsFeedItem(BaseModel):
    item_id: str
    headline: str
    summary: Optional[str] = None
    source: Optional[str] = None
    published_utc: str     # ISO format
    url: Optional[str] = None
    related_instruments: List[str] = []
    severity: Optional[str] = None      # LOW / MEDIUM / HIGH
    action_type: Optional[str] = None   # MONITOR / ALERT / FAST_TRACK

class NewsAlertRequest(BaseModel):
    item_id: str
    headline: str
    severity: str                        # HIGH / MEDIUM / LOW
    action_type: str                     # ALERT / FAST_TRACK
    affected_symbols: List[str] = []
    published_utc: str

class NewsAlertResponse(BaseModel):
    stored: bool
    broadcast: bool
    item_id: str
```

### Node Role Registration (server.py)

```python
# In src/api/server.py — INCLUDE_CONTABO block
# Knowledge & Research section (after knowledge_unified_router line ~365)
if INCLUDE_CONTABO:
    ...
    # Knowledge & Research
    app.include_router(knowledge_router)
    app.include_router(knowledge_unified_router)
    from src.api.knowledge_ingest_endpoints import router as knowledge_ingest_router
    app.include_router(knowledge_ingest_router)
    from src.api.news_endpoints import router as news_router       # Story 6.3
    app.include_router(news_router)
```

### Lifespan Wiring in server.py

Check if `server.py` uses `@app.on_event("startup")` or `lifespan` context manager — follow whichever pattern already exists. Guard with `INCLUDE_CONTABO` and `FINNHUB_API_KEY`:

```python
@app.on_event("startup")
async def startup_news_poller():
    if INCLUDE_CONTABO and os.environ.get("FINNHUB_API_KEY"):
        from src.knowledge.news.poller import NewsFeedPoller
        app.state.news_poller = NewsFeedPoller()
        app.state.news_poller.start()

@app.on_event("shutdown")
async def shutdown_news_poller():
    if hasattr(app.state, 'news_poller'):
        app.state.news_poller.stop()
```

### requirements.txt Addition

Add to `requirements.txt`:
```
finnhub-python>=2.4.0  # Finnhub news feed provider (Story 6.3)
```

Place near `firecrawl-py` line (line 4) with other data-provider SDKs.

### Testing Standards

- Use `pytest` with `unittest.mock.patch` — follow pattern from `tests/api/test_provider_config.py`
- Mock `finnhub.Client` with `unittest.mock.patch('src.knowledge.news.finnhub_provider.finnhub.Client')`
- Mock `anthropic.Anthropic` for `GeopoliticalSubAgent` classification tests
- Mock `asyncio.get_event_loop().run_in_executor` for async-wrapped sync calls
- Test file: `tests/api/test_news_feed.py`
- Use `pytest-asyncio` for async tests (`asyncio_mode = "auto"` or `@pytest.mark.asyncio`)
- In-memory SQLite for DB tests: use `create_engine("sqlite:///:memory:")` pattern
- Test WebSocket broadcast with `unittest.mock.AsyncMock` for `manager.broadcast`

### Project Structure Notes

New package:
- `src/knowledge/news/__init__.py` (new package)
- `src/knowledge/news/provider.py` (NewsProvider ABC)
- `src/knowledge/news/finnhub_provider.py` (FinnhubProvider)
- `src/knowledge/news/geopolitical_subagent.py` (GeopoliticalSubAgent — Haiku)
- `src/knowledge/news/poller.py` (NewsFeedPoller — APScheduler)

New database model:
- `src/database/models/news_items.py` (NewsItem SQLAlchemy model)

New API file:
- `src/api/news_endpoints.py` (GET /api/news/feed, POST /api/news/alert)

Modified files:
- `src/database/models/__init__.py` — add `NewsItem` export
- `src/api/server.py` — register `news_router` in INCLUDE_CONTABO block + lifespan hooks
- `requirements.txt` — add `finnhub-python>=2.4.0`

Test file:
- `tests/api/test_news_feed.py` (AC1, AC2, AC3, AC5, AC6)

Do NOT modify:
- `src/router/sensors/news.py` (`NewsSensor`) — calendar kill-zone enforcer, different concern
- `src/router/calendar_governor.py` — CalendarGovernor, separate from live news feed
- `src/agents/departments/subagents/research_subagent.py` — do not add news polling here
- `src/api/ide_knowledge.py` — knowledge scraper domain, unrelated

### References

- [Source: _bmad-output/planning-artifacts/architecture.md#13.3 — Live News Feed (FR50)]
- [Source: _bmad-output/planning-artifacts/architecture.md#13.4 — Fast-Track Event Workflow]
- [Source: _bmad-output/planning-artifacts/architecture.md#Database Tables — news_items]
- [Source: _bmad-output/planning-artifacts/epics.md#Story 6.3]
- [Source: src/integrations/github_ea_scheduler.py] — APScheduler AsyncIOScheduler pattern
- [Source: src/agents/departments/subagents/research_subagent.py:161] — Anthropic SDK Haiku init pattern
- [Source: src/api/websocket_endpoints.py:145] — global `manager` singleton, `broadcast(payload, topic=...)` API
- [Source: src/database/models/news_items.py — architecture table definition in §1280]
- [Source: src/database/models/risk_params.py] — SQLAlchemy model registration pattern
- [Source: src/api/server.py:363-365] — knowledge router INCLUDE_CONTABO registration pattern
- [Source: _bmad-output/implementation-artifacts/6-2-web-scraping-article-ingestion-personal-knowledge-api.md] — retry pattern, Anthropic usage
- Finnhub SDK: `finnhub-python` — `finnhub.Client(api_key=key).general_news(category='forex')`
- FR50: live news feed with enriched macro context
- Journey 46: HIGH severity event caught < 90 seconds (poll 60s + classify <30s = <90s total)

### Previous Story Learnings (Story 6.2)

Story 6.2 established these patterns for this story to follow:
- New API files use `router = APIRouter(prefix="/api/...", tags=[...])`
- Registered in `src/api/server.py` INCLUDE_CONTABO block after existing knowledge routers
- Synchronous SDK calls (e.g., Firecrawl, Finnhub) run in `asyncio.get_event_loop().run_in_executor(None, sync_fn)`
- Manual exponential backoff: `asyncio.sleep(base_delay * (2 ** attempt))` without external retry libraries
- `FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")` — guard with 503 if not set
- Tests in `tests/api/test_*.py` using `TestClient(FastAPI())` + `unittest.mock.patch`
- Story 6.2 also confirmed: `manager` WebSocket singleton importable from `src.api.websocket_endpoints`

### Git Intelligence

All recent commits (13a55a1, a09fe5a, 7f2e8df) are frontend canvas work (Epic 1 stories 1.4–1.6-9). Backend pattern to follow from Story 6.1 and Story 6.2:
- New API files in `src/api/` with `router = APIRouter(prefix="/api/...", tags=[...])`
- New service/domain code in `src/` subdirectories (e.g., `src/knowledge/news/`)
- Env-configured credentials: `os.environ.get("KEY")` — no defaults for secrets
- Tests in `tests/api/test_*.py` following `test_*.py` naming
- All logging via `logger = logging.getLogger(__name__)` at module level

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- StaticPool required for in-memory SQLite in FastAPI TestClient — in-memory DBs are per-connection; StaticPool ensures all connections share one DB instance across threads.
- `finnhub` SDK imported at module level with `try/except ImportError` fallback to `None` — enables `unittest.mock.patch('src.knowledge.news.finnhub_provider.finnhub')` in tests.
- WebSocket `manager` is imported lazily inside endpoint functions (not at module level in `news_endpoints.py`); tests patch `src.api.websocket_endpoints.manager` directly — the singleton.

### Completion Notes List

- Implemented all 7 tasks and 9 Acceptance Criteria for Story 6.3.
- Created new `src/knowledge/` domain package from scratch with `src/knowledge/news/` subpackage containing: `NewsProvider` ABC, `FinnhubProvider` (SDK + run_in_executor), `GeopoliticalSubAgent` (Haiku, max_tokens=400, JSON-only classification), and `NewsFeedPoller` (APScheduler 60s interval, exponential backoff 10s/20s/40s, HIGH+ALERT broadcast, HIGH+FAST_TRACK log-only per architecture §13.4).
- `NewsItem` SQLAlchemy model created with all AC1 fields, dual indexes on `published_utc` and `severity`, registered in `Base.metadata` via `models/__init__.py` import — auto-created by existing `init_database()` call.
- `GET /api/news/feed` returns latest 20 items ordered by `published_utc DESC`; `POST /api/news/alert` upserts and broadcasts to WebSocket topic="news".
- Both endpoints registered in INCLUDE_CONTABO block in `server.py` after existing knowledge routers.
- Poller lifecycle wired to `@app.on_event("startup"/"shutdown")` with `INCLUDE_CONTABO` and `FINNHUB_API_KEY` guards.
- `finnhub-python>=2.4.0` added to `requirements.txt`.
- 18 new tests in `tests/api/test_news_feed.py` — all 18 pass. No regressions in `tests/api/` test suite (383 passing, 45 pre-existing failures unchanged).

### File List

- `src/knowledge/__init__.py` (new — knowledge domain package)
- `src/knowledge/news/__init__.py` (new)
- `src/knowledge/news/provider.py` (new — NewsProvider ABC)
- `src/knowledge/news/finnhub_provider.py` (new — FinnhubProvider)
- `src/knowledge/news/geopolitical_subagent.py` (new — GeopoliticalSubAgent Haiku)
- `src/knowledge/news/poller.py` (new — NewsFeedPoller APScheduler)
- `src/database/models/news_items.py` (new — NewsItem SQLAlchemy model)
- `src/database/models/__init__.py` (modified — add NewsItem export)
- `src/api/news_endpoints.py` (new — GET /api/news/feed, POST /api/news/alert)
- `src/api/server.py` (modified — register news_router + lifespan hooks)
- `requirements.txt` (modified — add finnhub-python>=2.4.0)
- `tests/api/test_news_feed.py` (new — 18 tests)

### Change Log

- 2026-03-19: Story 6.3 implemented — Live News Feed & Geopolitical Sub-agent Backend. Created `src/knowledge/news/` domain package with NewsProvider ABC, FinnhubProvider (Finnhub SDK, dual-category polling, run_in_executor), GeopoliticalSubAgent (Haiku claude-3-haiku-20240307, JSON classification, fallback on parse failure), NewsFeedPoller (APScheduler 60s interval, 3-attempt exponential backoff 10s/20s/40s, HIGH+ALERT WebSocket broadcast, HIGH+FAST_TRACK warning log). NewsItem SQLAlchemy model with all AC1 fields and dual indexes. GET /api/news/feed and POST /api/news/alert endpoints registered under INCLUDE_CONTABO. Poller wired to startup/shutdown lifespan events guarded by FINNHUB_API_KEY. finnhub-python>=2.4.0 added to requirements.txt. 18 new tests all passing.

- 2026-03-19: Code Review Fix — Added missing `_send_copilot_mail()` method to NewsFeedPoller class to implement AC8 (HIGH+ALERT sends department mail to Copilot). Method uses DepartmentMailService to send alert messages to copilot department. All 18 tests still passing.
