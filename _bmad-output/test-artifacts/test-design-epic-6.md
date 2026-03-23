---
stepsCompleted: ['step-01-detect-mode', 'step-02-load-context', 'step-03-risk-assessment', 'step-04-coverage-plan', 'step-05-generate-document']
lastStep: 'step-05-generate-document'
lastSaved: '2026-03-21'
---

# Test Design: Epic 6 - Knowledge & Research Engine

**Date:** 2026-03-21
**Author:** Master Test Architect
**Status:** Draft

---

## Executive Summary

**Scope:** Full test design for Epic 6 - Knowledge & Research Engine

**Epic 6 Stories Completed:**
- Story 6.0: Knowledge Infrastructure Audit (done - audit only)
- Story 6.1: PageIndex Integration & Knowledge API (done)
- Story 6.2: Web Scraping, Article Ingestion & Personal Knowledge API (done)
- Story 6.3: Live News Feed & Geopolitical Sub-agent Backend (done - review status)
- Story 6.4: Research Canvas - Knowledge Query Interface (done)
- Story 6.5: YouTube Video Ingest UI & Pipeline Tracking (done)
- Story 6.6: Live News Feed Tile & News Canvas Integration (done)
- Story 6.7: Shared Assets Canvas (done)

**Risk Summary:**

- Total risks identified: 12
- High-priority risks (>=6): 4
- Critical categories: DATA (PageIndex availability), PERF (news latency), TECH (external APIs), SEC (API key exposure)

**Coverage Summary:**

- P0 scenarios: 8 (16 hours)
- P1 scenarios: 12 (12 hours)
- P2/P3 scenarios: 18 (9 hours)
- **Total effort**: 37 hours (~5 days)

---

## Not in Scope

| Item | Reasoning | Mitigation |
| --- | --- | --- |
| **PageIndex Docker container health monitoring** | Infrastructure concern handled by DevOps | Monitor via external uptime checks |
| **ChromaDB MCP server stress testing** | Separate epic (Epic 7/8) | Covered by Epic 8 performance tests |
| **Firecrawl SDK internal behavior** | Third-party vendor library | Trust vendor tests; focus on integration |
| **Finnhub API rate limit exhaustion** | Third-party vendor concern | Logged in Story 6.3; no recovery in scope |
| **MonacoEditorStub implementation** | Stub component, not functional | Covered by Story 6.7 integration tests |
| **Canvas routing logic** | Story 1.6-9 scope | Covered by Epic 1 regression tests |
| **Legacy `src/router/sensors/news.py` (NewsSensor)** | Calendar-driven kill-zone enforcer, separate domain | Separate test suite |

---

## Risk Assessment

### High-Priority Risks (Score >=6)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner | Timeline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R-001 | DATA | **PageIndex Docker containers offline** - All 3 instances (articles, books, logs) must be running for `GET /api/knowledge/sources` and `POST /api/knowledge/search` to return data. Currently NOT running per Story 6.0 audit. Search fanout degrades gracefully but returns 0 results. | 4 | 3 | 12 | Graceful degradation implemented (warnings in response). Knowledge API returns offline status per source. Pre-check containers in test environment setup. | QA | Sprint 6 |
| R-002 | PERF | **News feed latency exceeds 90-second SLA** - Architecture requires sub-90s news-to-alert latency (60s poll + 30s Haiku classification). Any degradation in Finnhub response or Haiku latency breaches SLA. | 3 | 3 | 9 | Monitor p95/p99 latency in production. Implement circuit breaker if Haiku latency >25s. Story 6.3 code review confirmed implementation. | QA | Sprint 6 |
| R-003 | TECH | **External API failures cascade** - Firecrawl (scraping), Finnhub (news), and Haiku (classification) are all external. Failures can cause data gaps or poller stalling. | 3 | 3 | 9 | Exponential backoff implemented (Story 6.2: 1s/2s/4s; Story 6.3: 10s/20s/40s). Error logging at ERROR level per NFR-I4. | QA | Sprint 6 |
| R-004 | SEC | **API key exposure via logs or error messages** - `FIRECRAWL_API_KEY` and `FINNHUB_API_KEY` handled via env vars but logged on errors. YAML front-matter injection was fixed in Story 6.2 code review. | 2 | 4 | 8 | Code review confirmed `_yaml_safe_str()` sanitization. Ensure test environment keys are test-only (not production). QA to audit log output for credential leakage. | QA | Sprint 6 |
| R-005 | DATA | **Personal knowledge file storage race conditions** - File-drop pattern to `data/knowledge_base/personal/` uses `job_id` for uniqueness. Concurrent writes to same path could collide. | 2 | 3 | 6 | Use `asyncio.Lock` for file write operations in test environment. Document as known limitation for high-concurrency scenarios. | DEV | Sprint 6 |
| R-006 | TECH | **NewsItem DB deduplication failures** - Story 6.3 uses `item_id` (Finnhub `id` cast to string) for deduplication. If Finnhub returns duplicate IDs with different content, the first write wins. | 2 | 3 | 6 | Implement `updated_at_utc` check in production - if incoming `published_utc` is newer, update existing row. Add uniqueness constraint on `(item_id, published_utc)` as safety net. | DEV | Sprint 6 |

### Medium-Priority Risks (Score 3-4)

| Risk ID | Category | Description | Probability | Impact | Score | Mitigation | Owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-007 | PERF | **Video ingest pipeline blocking** - `POST /api/video-ingest/process` is synchronous. Long-running video processing could block the FastAPI worker. | 2 | 2 | 4 | Backend `src/video_ingest/api.py` uses job queue with SQLite persistence. Document worker pool sizing requirements for production. | DEV |
| R-008 | DATA | **Knowledge search result freshness** - PageIndex is a Docker volume mount (read-only). Newly ingested articles via Firecrawl are written to `data/scraped_articles/` but PageIndex may not re-scan immediately. | 2 | 2 | 4 | File-drop pattern relies on PageIndex's volume scan interval. Document re-index latency (typically <60s). Test with time-bounded queries. | QA |
| R-009 | TECH | **WebSocket reconnection storms** - News tiles (Story 6.6) subscribe to `topic=news`. If Finnhub API fails repeatedly, the poller generates many HIGH alerts, potentially flooding WebSocket connections. | 2 | 2 | 4 | Implement WebSocket message throttling (max 1 broadcast per 5s per client). Story 6.6 review confirmed basic reconnection; enhance with exponential backoff on client side. | DEV |
| R-010 | BUS | **Research canvas "Send to Copilot" silently fails** - `POST /floor-manager/chat` may fail but the UI shows a 2s toast success regardless. If the endpoint returns error, user is not notified. | 2 | 2 | 4 | Add error handling in `sendToCopilot()` function. Show error toast if response is not ok. Story 6.4 review confirmed pattern; verify implementation. | QA |

### Low-Priority Risks (Score 1-2)

| Risk ID | Category | Description | Probability | Impact | Score | Action |
| --- | --- | --- | --- | --- | --- | --- |
| R-011 | OPS | **Poller lifecycle on fast restarts** - If FastAPI restarts within 60s of poller start, the APScheduler may not cleanly shutdown, leaving zombie poll cycles. | 1 | 2 | 2 | Use `shutdown(wait=False)` per Story 6.3 implementation. Document graceful shutdown procedure. Monitor for duplicate poll cycles via log analysis. |
| R-012 | DATA | **YouTube video ingest fails silently** - If `POST /api/video-ingest/process` returns 200 but processing fails async, the UI shows "Indexing" indefinitely (Story 6.5 review noted `isProcessing` stuck bug - fixed). | 1 | 2 | 2 | Verify bug fix in code review. Add job status polling timeout (max 10min for video processing). |
| R-013 | TECH | **Knowledge source badge colors hardcoded** - `SOURCE_BADGE_COLORS` in `knowledgeApi.ts` is a static map. Adding new source types requires code change. | 1 | 1 | 1 | Document as design limitation. If dynamic source types needed, migrate to API-driven color schema. |

### Risk Category Legend

- **TECH**: Technical/Architecture (flaws, integration, scalability)
- **SEC**: Security (access controls, auth, data exposure)
- **PERF**: Performance (SLA violations, degradation, resource limits)
- **DATA**: Data Integrity (loss, corruption, inconsistency)
- **BUS**: Business Impact (UX harm, logic errors, revenue)
- **OPS**: Operations (deployment, config, monitoring)

---

## Entry Criteria

- [ ] Requirements and assumptions agreed upon by QA, Dev, PM
- [ ] Test environment provisioned with PageIndex Docker containers running
- [ ] Test data: 1,806 scraped articles in `data/scraped_articles/`, ChromaDB collections populated
- [ ] `FIRECRAWL_API_KEY` and `FINNHUB_API_KEY` configured in test environment (test-tier keys, not production)
- [ ] `NODE_ROLE=contabo` set to enable knowledge and news endpoints
- [ ] Anthropic API key available for Haiku classification tests
- [ ] SQLite database initialized with `NewsItem` table (via `Base.metadata.create_all()`)
- [ ] Feature deployed to test environment (verify all 8 story implementations present)

## Exit Criteria

- [ ] All P0 tests passing (100% pass rate required)
- [ ] All P1 tests passing (>=95% pass rate, waivers for any failures)
- [ ] No open high-priority / high-severity bugs (R-001 through R-006)
- [ ] Test coverage agreed as sufficient (>=80% critical paths covered)
- [ ] API contract tests verify `POST /api/knowledge/search`, `GET /api/knowledge/sources`, `GET /api/news/feed`, `POST /api/news/alert` contract compliance
- [ ] WebSocket news topic subscription verified end-to-end
- [ ] Frontend canvas components render without console errors

---

## Project Team

**Include only if roles/names are known or responsibility mapping is needed; otherwise omit.**

| Name | Role | Testing Responsibilities |
| --- | --- | --- |
| TBD | QA Lead | P0/P1 test execution, risk assessment sign-off |
| TBD | Backend Dev | API contract tests, async poller tests |
| TBD | Frontend Dev | Canvas component tests, integration tests |
| TBD | DevOps | PageIndex container provisioning, environment setup |

---

## Test Coverage Plan

### P0 (Critical) - Run on every commit

**Criteria**: Blocks core journey + High risk (>=6) + No workaround

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Knowledge search fanout - parallel query to 3 PageIndex instances | API | R-001 | 3 | QA | Test all 3 online, 1 offline, 2 offline scenarios |
| Knowledge sources status endpoint returns correct document counts | API | R-001 | 2 | QA | Mock PageIndex responses; verify graceful offline handling |
| News feed API returns latest 20 items ordered by published_utc DESC | API | R-002 | 2 | QA | Verify SQLAlchemy query ordering and limit |
| News WebSocket broadcast on HIGH+ALERT | API | R-002 | 2 | QA | Mock poller; verify `manager.broadcast(topic="news")` called |
| Firecrawl URL ingestion with exponential backoff retry | API | R-003 | 3 | QA | Mock Firecrawl failing 1x, 2x, 3x then success |
| Personal knowledge note ingestion with provenance YAML front-matter | API | R-005 | 2 | QA | Verify front-matter sanitization, file-drop pattern |
| NewsItem deduplication by item_id | API | R-006 | 2 | QA | Mock duplicate Finnhub IDs; verify skip vs update |
| API key guard - FIRECRAWL_API_KEY / FINNHUB_API_KEY not configured | API | R-004 | 2 | QA | Verify 503 response with appropriate error message |

**Total P0**: 18 tests, 16 hours

### P1 (High) - Run on PR to main

**Criteria**: Important features + Medium risk (3-4) + Common workflows

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Research canvas loads and renders knowledge search interface | E2E | R-001 | 2 | QA | Verify canvas context load, search bar, filter chips |
| Knowledge search - client-side filter by source type | Component | - | 3 | DEV | Articles, Books, Logs, Personal, All filters |
| News feed tile on Live Trading canvas - latest 5 items display | E2E | R-002 | 2 | QA | Verify 5-item limit, severity badges, amber flash on HIGH |
| GeopoliticalSubAgent classification - Haiku JSON parsing | Unit | R-003 | 3 | DEV | Valid JSON, malformed JSON fallback, empty content |
| NewsFeedPoller exponential backoff - 10s/20s/40s delays | Unit | R-003 | 3 | DEV | Mock provider failing 1, 2, 3 times |
| Video ingest submission - POST /api/video-ingest/process | API | R-007 | 2 | QA | Valid YouTube URL, invalid URL rejection |
| Video ingest progress stages render correctly | Component | R-012 | 2 | DEV | Downloading, Transcribing, Chunking, Embedding, Indexing states |
| Knowledge canvas "Send to Copilot" sends correct payload | E2E | R-010 | 2 | QA | Verify `/floor-manager/chat` payload structure |
| PageIndex async methods - search_async, health_check_async | Unit | R-001 | 3 | DEV | Verify httpx.AsyncClient usage, timeouts (10s search, 5s health) |
| News WebSocket subscription reconnects on disconnect | E2E | R-009 | 2 | QA | Simulate disconnect; verify auto-reconnect with backoff |

**Total P1**: 24 tests, 12 hours

### P2 (Medium) - Run nightly/weekly

**Criteria**: Secondary features + Low risk (1-2) + Edge cases

| Requirement | Test Level | Risk Link | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Research canvas - empty state when no results | Component | - | 2 | DEV | Verify "No results found" message renders |
| Research canvas - error banner on API failure | Component | - | 2 | DEV | Verify warning text from response.warnings displayed |
| Research canvas - skeleton loader during search | Component | - | 1 | DEV | Verify 3 pulsing placeholder tiles |
| Knowledge search - limit constraint (1-100) | API | - | 2 | QA | Test limit=0 (should 422), limit=101 (should 422) |
| Knowledge search - query length constraint (1-2000 chars) | API | - | 2 | QA | Test empty query, query >2000 chars |
| Personal note ingestion - PDF and TXT file upload | API | - | 2 | QA | Verify file written to `data/knowledge_base/personal/` |
| News alert endpoint - upsert behavior | API | R-006 | 2 | QA | Alert on existing item updates; new item inserts |
| News feed filter - by severity (HIGH/MEDIUM/LOW) | API | - | 3 | QA | Verify filtered results match severity |
| Shared Assets canvas - asset type grid renders 6 categories | E2E | - | 2 | QA | Docs, Strategy Templates, Indicators, Skills, Flow Components, MCP Configs |
| Shared Assets canvas - asset detail view with MonacoEditor | E2E | - | 2 | QA | Verify read mode; Edit/Save mode transition |
| Video ingest - auth status check on mount | Component | - | 1 | DEV | Verify `GET /api/video-ingest/auth-status` called |
| YouTube URL validation - valid and invalid patterns | Unit | - | 4 | DEV | `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/playlist`, invalid |
| News exposure calculation - "X strategies exposed" | Unit | - | 2 | DEV | Mock related_instruments; verify count display |
| PageIndex file-drop - newly scraped article appears in search | Integration | R-008 | 2 | QA | Ingest URL, wait 60s, search confirms indexed |

**Total P2**: 29 tests, 14.5 hours

### P3 (Low) - Run on-demand

**Criteria**: Nice-to-have + Exploratory + Performance benchmarks

| Requirement | Test Level | Test Count | Owner | Notes |
| --- | --- | --- | --- | --- |
| Research canvas - keyboard navigation (Enter to search, Esc to clear) | E2E | 2 | QA | Accessibility check |
| Knowledge search - 50+ concurrent requests (load test) | Performance | 1 | QA | Measure p95 latency under load |
| News poller - 1000+ news items in DB (pagination/ordering) | Performance | 1 | QA | Verify `GET /api/news/feed` LIMIT 20 still fast |
| Shared Assets canvas - MonacoEditor with large file (>10k lines) | Exploration | 1 | DEV | Verify no UI freeze |
| Video ingest - playlist URL submission | E2E | 1 | QA | `youtube.com/playlist?list=` URL pattern |
| News feed - WebSocket message throttling under flood | Exploration | 1 | QA | 100 HIGH alerts in 10s; verify throttling kicks in |
| Personal knowledge - concurrent note uploads (10 simultaneous) | Integration | 1 | QA | Race condition detection on file storage |
| Knowledge API - PageIndex containers restart during active search | Chaos | 1 | QA | Simulate container restart mid-request |

**Total P3**: 9 tests, 2.25 hours

---

## Execution Order

### Smoke Tests (<5 min)

**Purpose**: Fast feedback, catch build-breaking issues

- [ ] Knowledge sources endpoint returns 200 with source list (30s)
- [ ] Knowledge search returns results for known query (45s)
- [ ] News feed endpoint returns 200 with item list (30s)
- [ ] Research canvas loads without console errors (60s)
- [ ] Video ingest auth status check passes (30s)

**Total**: 5 scenarios

### P0 Tests (<10 min)

**Purpose**: Critical path validation

- [ ] Knowledge search fanout - all 3 instances online (API)
- [ ] Knowledge search fanout - 1 instance offline with warning (API)
- [ ] Knowledge search fanout - 2 instances offline graceful degradation (API)
- [ ] Knowledge sources - offline instance returns status=offline (API)
- [ ] News feed - latest 20 items ordered correctly (API)
- [ ] News WebSocket - HIGH+ALERT triggers broadcast (API)
- [ ] Firecrawl - 3-attempt retry then success (API)
- [ ] Firecrawl - max retries exhausted returns failure (API)
- [ ] Personal note - provenance front-matter sanitization (API)
- [ ] NewsItem - duplicate item_id skipped (API)
- [ ] API key missing - FINNHUB_API_KEY returns 503 (API)
- [ ] API key missing - FIRECRAWL_API_KEY returns 503 (API)
- [ ] News alert - upsert on existing item (API)
- [ ] News alert - insert new item (API)

**Total**: 14 scenarios

### P1 Tests (<30 min)

**Purpose**: Important feature coverage

- [ ] Research canvas - search bar and filter chips render (E2E)
- [ ] Research canvas - client-side source filtering (Component)
- [ ] News tile - latest 5 items with severity badges (E2E)
- [ ] News tile - HIGH item amber flash animation (E2E)
- [ ] Haiku classification - valid JSON response (Unit)
- [ ] Haiku classification - malformed JSON fallback to LOW/MONITOR (Unit)
- [ ] Haiku classification - empty content handling (Unit)
- [ ] NewsFeedPoller backoff - 3 failures then success (Unit)
- [ ] Video ingest - valid YouTube URL accepted (API)
- [ ] Video ingest - invalid URL rejected (API)
- [ ] Send to Copilot - correct floor-manager/chat payload (E2E)
- [ ] PageIndex async methods - timeout enforcement (Unit)
- [ ] WebSocket - reconnection with backoff (E2E)

**Total**: 13 scenarios

### P2/P3 Tests (<60 min)

**Purpose**: Full regression coverage

- [ ] Research canvas - empty state (Component)
- [ ] Research canvas - error banner (Component)
- [ ] Research canvas - skeleton loader (Component)
- [ ] Knowledge search - limit boundary 1-100 (API)
- [ ] Knowledge search - query length validation (API)
- [ ] Personal note - PDF upload (API)
- [ ] Personal note - TXT upload (API)
- [ ] News feed - severity filter (API)
- [ ] Shared Assets - 6 category grid (E2E)
- [ ] Shared Assets - detail view MonacoEditor (E2E)
- [ ] Video ingest - auth status check (Component)
- [ ] YouTube URL validation patterns (Unit)
- [ ] News exposure count display (Unit)
- [ ] Keyboard navigation - accessibility (E2E)
- [ ] Knowledge search - 50 concurrent load (Performance)
- [ ] News feed - 1000 items pagination (Performance)
- [ ] MonacoEditor - large file (>10k lines) (Exploration)
- [ ] Video ingest - playlist URL (E2E)
- [ ] News WebSocket - message throttling (Exploration)
- [ ] Personal knowledge - concurrent uploads (Integration)
- [ ] Knowledge API - PageIndex restart mid-request (Chaos)

**Total**: 21 scenarios

---

## Resource Estimates

### Test Development Effort

| Priority | Count | Hours/Test | Total Hours | Notes |
| --- | --- | --- | --- | --- |
| P0 | 18 | 2.0 | 36 | Complex async mocking, WebSocket, external APIs |
| P1 | 24 | 1.0 | 24 | Standard API/component tests |
| P2 | 29 | 0.5 | 14.5 | Simple scenarios, edge cases |
| P3 | 9 | 0.25 | 2.25 | Exploratory, performance |
| **Total** | **80** | **-** | **76.75** | **~10 days** |

### Prerequisites

**Test Data:**

- `tests/api/conftest.py` - Pytest fixtures for API test client, mock PageIndex responses, mock Finnhub responses, mock Firecrawl responses
- `tests/api/test_knowledge_sources.py` - existing Story 6.1 tests (15 passing) - verify coverage adequacy
- `tests/api/test_knowledge_ingest.py` - existing Story 6.2 tests (14 passing) - verify coverage adequacy
- `tests/api/test_news_feed.py` - existing Story 6.3 tests (18 passing) - verify coverage adequacy
- `quantmind-ide/src/lib/api/knowledgeApi.test.ts` - existing Story 6.4 frontend tests
- `quantmind-ide/src/lib/api/newsApi.test.ts` - existing Story 6.6 frontend tests
- `quantmind-ide/src/lib/api/videoIngestApi.test.ts` - Story 6.5 frontend tests (not written per story spec)

**Tooling:**

- `pytest` + `pytest-asyncio` for backend API tests
- `unittest.mock.patch` + `respx` for HTTP mocking
- `vitest` for frontend component tests
- `httpx.AsyncClient` test client for async endpoint testing
- `faker` for test data generation (news items, knowledge articles)
- `playwright` for E2E canvas tests (verify installation)

**Environment:**

- PageIndex Docker containers running on `localhost:3000/3001/3002`
- SQLite database with `NewsItem` table auto-created on app startup
- Environment vars: `NODE_ROLE=contabo`, `FIRECRAWL_API_KEY=test_key`, `FINNHUB_API_KEY=test_key`, `ANTHROPIC_API_KEY=test_key`
- `asyncio.get_running_loop()` compatible Python 3.10+ runtime

---

## Quality Gate Criteria

### Pass/Fail Thresholds

- **P0 pass rate**: 100% (no exceptions)
- **P1 pass rate**: >=95% (waivers required for failures)
- **P2/P3 pass rate**: >=90% (informational)
- **High-risk mitigations**: 100% complete or approved waivers

### Coverage Targets

- **Critical paths**: >=80% - Knowledge search fanout, News feed + WebSocket, Firecrawl ingestion
- **Security scenarios**: 100% - API key guards, YAML sanitization, no credential leakage in logs
- **Business logic**: >=70% - Research canvas search/filter, Send to Copilot, News severity display
- **Edge cases**: >=50% - Empty states, malformed JSON, concurrent uploads

### Non-Negotiable Requirements

- [ ] All P0 tests pass (R-001 through R-006 covered)
- [ ] No high-risk (>=6) items unmitigated
- [ ] Security tests (SEC category R-004) pass 100%
- [ ] Performance targets met (PERF category R-002: news latency <90s)

---

## Mitigation Plans

### R-001: PageIndex Docker containers offline (Score: 12)

**Mitigation Strategy:** Knowledge API implements graceful degradation - offline instances return `status=offline`, `document_count=0`, and are skipped in search fanout with warnings in response. Test environment must provision running PageIndex containers. Production monitoring via health endpoint.

**Owner:** DevOps
**Timeline:** Before test execution
**Status:** Planned
**Verification:** `GET /api/knowledge/sources` returns all 3 sources with correct status; offline simulation via stopping containers

### R-002: News feed latency exceeds 90-second SLA (Score: 9)

**Mitigation Strategy:** Monitor p95/p99 latency in production. Implement circuit breaker if Haiku latency exceeds 25s (leaving 5s buffer for Finnhub). Story 6.3 confirmed 60s poll interval + Haiku classification.

**Owner:** QA
**Timeline:** Sprint 6 performance testing
**Status:** Planned
**Verification:** Load test with 100 news items; measure end-to-end latency from Finnhub response to WebSocket broadcast

### R-003: External API failures cascade (Score: 9)

**Mitigation Strategy:** Exponential backoff implemented for both Firecrawl (1s/2s/4s) and Finnhub (10s/20s/40s). Error logging at ERROR level per NFR-I4 compliance. Story 6.3 review confirmed retry logic.

**Owner:** QA
**Timeline:** Sprint 6
**Status:** Planned
**Verification:** Mock external APIs failing; verify correct delay sequence and final error after max retries

### R-004: API key exposure via logs or error messages (Score: 8)

**Mitigation Strategy:** Code review confirmed `_yaml_safe_str()` sanitization for YAML front-matter. Test environment uses test-only keys. QA audit logs for credential patterns (regex: `(?i)(api[_-]?key|token|secret)`).

**Owner:** QA
**Timeline:** Sprint 6 security audit
**Status:** Planned
**Verification:** Review test logs for credential patterns; verify 503 without key, not 500 with stack trace containing key

### R-005: Personal knowledge file storage race conditions (Score: 6)

**Mitigation Strategy:** Document as known limitation for high-concurrency scenarios. Use `asyncio.Lock` for file write operations in test. Production guidance: single-writer assumption for personal knowledge uploads.

**Owner:** DEV
**Timeline:** Post-Sprint 6
**Status:** Planned
**Verification:** Concurrent upload test (10 simultaneous); monitor for file collisions or data loss

### R-006: NewsItem DB deduplication failures (Score: 6)

**Mitigation Strategy:** Implement `updated_at_utc` check - if incoming `published_utc` is newer, update existing row. Add uniqueness constraint on `(item_id, published_utc)` as safety net in schema.

**Owner:** DEV
**Timeline:** Sprint 6
**Status:** Planned
**Verification:** Mock Finnhub returning duplicate ID with different `published_utc`; verify update occurs

---

## Assumptions and Dependencies

### Assumptions

1. PageIndex Docker containers will be running in test and production environments
2. Firecrawl free tier (500 pages/month) is sufficient for test data generation
3. Finnhub API key will be available in test environment (test tier)
4. Anthropic Haiku model (`claude-3-haiku-20240307`) will remain available and <30s latency
5. SQLite is acceptable for news_items storage (not requiring PostgreSQL/MySQL)
6. The frontend canvas components (ResearchCanvas, SharedAssetsCanvas) are already wired in the UI router

### Dependencies

1. **PageIndex Docker containers** - Required by Sprint 6 start - DevOps
2. **Anthropic API key (Haiku)** - Required for R-003/R-006 tests - Platform team
3. **Finnhub API key (test tier)** - Required by Sprint 6 start - Platform team
4. **Firecrawl API key (test tier)** - Required for R-003 tests - Platform team
5. **Vitest + Playwright setup** - Required for frontend tests - Frontend Dev
6. **Backend test fixtures (knowledgeApi, newsApi)** - Story 6.1/6.2/6.3 tests already exist, verify coverage

### Risks to Plan

- **Risk**: Finnhub API tier limits hit during testing (500 calls/day free tier)
  - **Impact**: News poller exhausts daily quota in <8 hours of continuous testing
  - **Contingency**: Use mock Finnhub responses for automated tests; manual testing only with live API

---

## Follow-on Workflows (Manual)

- Run `*atdd` to generate failing P0 tests from acceptance criteria (separate workflow; not auto-run).
- Run `*automate` for broader coverage once implementation exists.
- Run `*performance` for load testing Knowledge search and News feed under concurrent load.

---

## Approval

**Test Design Approved By:**

- [ ] Product Manager: TBD Date: TBD
- [ ] Tech Lead: TBD Date: TBD
- [ ] QA Lead: TBD Date: TBD

**Comments:**

---

---

---

## Interworking & Regression

| Service/Component | Impact | Regression Scope |
| --- | --- | --- |
| **PageIndex (articles, books, logs)** | Knowledge search and sources endpoints depend on all 3 instances | Story 6.1 tests: `tests/api/test_knowledge_sources.py` (15 tests) must pass |
| **ChromaDB MCP server** | Knowledge search uses ChromaDB for semantic search; ChromaDB outage does NOT break PageIndex fallback | Story 6.1/6.4 integration tests; ChromaDB covered separately in Epic 5 |
| **Finnhub API** | News feed backend (Story 6.3) depends on Finnhub for live news; API key required | Story 6.3 tests: `tests/api/test_news_feed.py` (18 tests) must pass |
| **Firecrawl** | Web scraping (Story 6.2) depends on Firecrawl for article ingestion | Story 6.2 tests: `tests/api/test_knowledge_ingest.py` (14 tests) must pass |
| **Anthropic Haiku** | GeopoliticalSubAgent (Story 6.3) uses Haiku for classification; latency affects news SLA | Story 6.3 Haiku unit tests must pass |
| **WebSocket manager** | News alerts broadcast via `topic=news`; WebSocket disconnect affects Live Trading news tile | Story 6.6 tests: `tests/api/test_news_feed.py` WebSocket tests must pass |
| **Floor Manager** | "Send to Copilot" posts to `/floor-manager/chat`; failure silently ignored | Story 6.4 E2E tests must cover error path |
| **Video Ingest API** | YouTube ingest (Story 6.5) uses `src/video_ingest/api.py` | Story 6.5 integration tests; backend API contract verified |
| **Canvas Router** | All canvases (Research, Shared Assets, Live Trading) routed via `canvas_router.py` | Epic 1 regression tests cover canvas routing |
| **NewsSensor (calendar)** | NOT modified - separate concern (kill-zone enforcement) | Excluded from Epic 6 scope |

---

## Appendix

### Knowledge Base References

- `risk-governance.md` - Risk classification framework
- `probability-impact.md` - Risk scoring methodology
- `test-levels-framework.md` - Test level selection
- `test-priorities-matrix.md` - P0-P3 prioritization

### Related Documents

- Epic 6 Overview: `_bmad-output/implementation-artifacts/sprint-status.yaml` (Epic 6 status: done)
- Story 6.0: `_bmad-output/implementation-artifacts/6-0-knowledge-infrastructure-audit.md`
- Story 6.1: `_bmad-output/implementation-artifacts/6-1-pageindex-integration-knowledge-api.md`
- Story 6.2: `_bmad-output/implementation-artifacts/6-2-web-scraping-article-ingestion-personal-knowledge-api.md`
- Story 6.3: `_bmad-output/implementation-artifacts/6-3-live-news-feed-geopolitical-sub-agent-backend.md`
- Story 6.4: `_bmad-output/implementation-artifacts/6-4-research-canvas-knowledge-query-interface.md`
- Story 6.5: `_bmad-output/implementation-artifacts/6-5-youtube-video-ingest-ui-pipeline-tracking.md`
- Story 6.6: `_bmad-output/implementation-artifacts/6-6-live-news-feed-tile-news-canvas-integration.md`
- Story 6.7: `_bmad-output/implementation-artifacts/6-7-shared-assets-canvas.md`

### Architecture References

- Knowledge & Vector Search Stack: `_bmad-output/planning-artifacts/architecture.md#1.3`
- Live News Feed (FR50): `_bmad-output/planning-artifacts/architecture.md#13.3`
- Fast-Track Event Workflow: `_bmad-output/planning-artifacts/architecture.md#13.4`
- News Items Database: `_bmad-output/planning-artifacts/architecture.md#1280`

### Existing Test Coverage

- `tests/api/test_knowledge_sources.py` - 15 tests (Story 6.1) - AC1, AC2, AC3, AC4, AC5 covered
- `tests/api/test_knowledge_ingest.py` - 14 tests (Story 6.2) - AC1, AC3, AC5, AC6 covered
- `tests/api/test_news_feed.py` - 18 tests (Story 6.3) - AC1, AC2, AC3, AC5, AC6 covered
- `quantmind-ide/src/lib/api/knowledgeApi.test.ts` - frontend (Story 6.4)
- `quantmind-ide/src/lib/api/newsApi.test.ts` - frontend (Story 6.6)

---

**Generated by**: BMad TEA Agent - Test Architect Module
**Workflow**: `_bmad/tea/testarch/test-design`
**Version**: 4.0 (BMad v6)
**Epic**: 6 - Knowledge & Research Engine
**Date**: 2026-03-21
