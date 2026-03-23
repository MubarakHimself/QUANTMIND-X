# QUANTMINDX — Data Pipeline & Knowledge System

**Generated:** 2026-03-11

---

## Overview

QUANTMINDX has a multi-stage data pipeline that ingests trading knowledge from external sources (MQL5 articles, YouTube trading videos, PDFs) and makes it available to AI agents via a semantic search knowledge base.

```
External Sources
    ├── MQL5.com articles  → Firecrawl scraper → data/scraped_articles/
    ├── YouTube videos     → Video Ingest pipeline → data/videos/
    └── PDF documents      → PDF indexing → data/pdf_documents/
                                    ↓
                         [PageIndex Docker services]
                                    ↓
                         Full-text search (REST API)
                                    ↓
                    [Agent Knowledge Tools] ← agents query KB
                    [MCP Knowledge Server]  ← mcp-servers/quantmindx-kb/
```

---

## 1. MQL5 Article Scraper

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/firecrawl_scraper.py` | Scrape MQL5 articles via Firecrawl API |
| `scripts/simple_scraper.py` | Lightweight HTML scraper (no API) |
| `scripts/scraper.py` | Full scraper with deduplication |
| `scripts/category_crawler.py` | Crawl by MQL5 category |
| `scripts/crawl_all_categories.py` | Batch crawl all categories |
| `scripts/filter_analyst_kb.py` | Filter articles for analyst knowledge base |
| `scripts/search_kb.py` | CLI search tool for knowledge base |

### Output
- `data/scraped_articles/` — JSON/Markdown articles
- `data/raw_articles/` — Raw scraped HTML
- `data/markdown/` — Converted markdown articles

### Usage
See `scripts/SCRAPER_USAGE.md` and `scripts/FIRECRAWL_USAGE.md` for instructions.
See `scripts/CATEGORY_CRAWLER_USAGE.md` for category crawl instructions.

---

## 2. Video Ingest Pipeline (`src/video_ingest/`)

Ingests YouTube trading videos and extracts trading strategies.

### Architecture
```
YouTube URL
    ↓
[video_ingest/downloader.py]  — yt-dlp download
    ↓
data/video_in/downloads/
    ↓
[video_ingest/processor.py]   — extract audio + frames
    ↓
video_analysis_temp/{audio/, frames/, video/}
    ↓
[video_ingest/extractors.py]  — LLM-based extraction
    ↓
Strategy: signals, timeframe, entry/exit rules
    ↓
data/video_in/strategies/     — saved strategy JSON
data/video_in/transcripts/    — saved transcripts
    ↓
[strategies_yt/]              — strategy database
```

### Key Files
| File | Purpose |
|------|---------|
| `src/video_ingest/downloader.py` | yt-dlp YouTube download |
| `src/video_ingest/processor.py` | Audio + frame extraction |
| `src/video_ingest/extractors.py` | LLM strategy extraction |
| `src/video_ingest/job_queue.py` | Async job queue |
| `src/video_ingest/models.py` | VideoJob, VideoStrategy data models |
| `src/video_ingest/api.py` | Internal API for video processing |
| `src/video_ingest/cache.py` | Caching layer |
| `src/video_ingest/rate_limiter.py` | API rate limiting |
| `src/video_ingest/cli.py` | CLI interface |
| `scripts/video_ingest_cli.py` | CLI wrapper script |

### Job States
```
PENDING → DOWNLOADING → PROCESSING → EXTRACTING → COMPLETED / FAILED
```

### Video-to-EA Workflow
`src/router/video_to_ea_workflow.py` — orchestrates the full pipeline:
1. Ingest video URL
2. Extract trading strategy via LLM
3. Pass strategy to Development agent
4. Generate EA code (MQL5 or PineScript)
5. Backtest and validate
6. Deploy to `data/trd/` as TRD (Trading Rule Document)

### Gemini Extension
`extensions/video-ingest-extension/` — Gemini CLI extension for video ingest:
- `scripts/process_video.py` — process video via Gemini Vision
- `scripts/qwen_runner.py` — alternative Qwen VL model runner

---

## 3. PageIndex Search Services (Docker)

Four Docker-based PageIndex search services:

| Service | Port | Data Source | Dockerfile |
|---------|------|-------------|-----------|
| `pageindex-articles` | 3000 | `data/scraped_articles/` | `docker/pageindex/Dockerfile.articles` |
| `pageindex-books` | 3001 | `data/knowledge_base/books/` | `docker/pageindex/Dockerfile.books` |
| `pageindex-logs` | 3002 | `data/logs/` | `docker/pageindex/Dockerfile.logs` |
| `kb-indexer` | — | Builds full index | `scripts/index_kb.dockerfile` |

Start with: `docker-compose up` (see `docker-compose.yml`)

**PageIndex REST API:** `GET /search?q=<query>` on each service port.

### Indexing Script
`scripts/index_to_pageindex.py` — pushes documents to PageIndex services.
`scripts/generate_document_index.py` — generates document index metadata.

---

## 4. Knowledge Base MCP Server (`mcp-servers/quantmindx-kb/`)

An MCP server exposing the knowledge base to AI agents.

| File | Purpose |
|------|---------|
| `server.py` | MCP server (ChromaDB backend) |
| `server_chroma.py` | ChromaDB-specific implementation |
| `start.sh` | Start the KB MCP server |
| `tests/test_retrieval_tools.py` | Retrieval tests |
| `tests/test_skills_integration.py` | Skills integration tests |

**Tools exposed to agents:**
- `search_knowledge` — semantic search over all articles
- `get_article` — retrieve full article by ID
- `list_categories` — browse article categories

**Vector database:** ChromaDB at `data/chromadb/`

---

## 5. PDF Document Processing (`src/api/pdf_endpoints.py`)

Upload and index PDF trading books for agent knowledge access:
- `uploads/pdfs/` — uploaded PDFs
- `data/pdf_documents/` — processed documents
- `data/pdf_jobs/` — processing job queue
- `src/agents/tools/knowledge/pdf_indexing.py` — indexing implementation
- `src/agents/tools/knowledge/mql5_book.py` — MQL5 book-specific queries

---

## 6. Strategy Database (`strategies-yt/`, `data/strategies/`)

### Directory Structure
```
strategies-yt/
├── src/                    # Strategy processing code
├── prompts/                # LLM prompts for strategy extraction
data/strategies/            # Saved strategy files
data/trd/                   # Trading Rule Documents (TRD format)
```

### TRD Format (Trading Rule Document)
A structured JSON/Markdown specification for a trading strategy:
- Entry conditions (indicator rules, pattern conditions)
- Exit conditions (TP/SL rules, time-based exits)
- Position sizing instructions
- Allowed symbols and timeframes
- Risk parameters

`src/router/trd_watcher.py` — watches `data/trd/` for new TRDs and automatically registers them as bots.

---

## 7. Knowledge Tools for Agents (`src/agents/tools/knowledge/`)

| File | Tools Exposed |
|------|---------------|
| `client.py` | HTTP client to PageIndex services |
| `knowledge_hub.py` | Unified KB query interface |
| `mql5_book.py` | MQL5 book-specific lookup |
| `pdf_indexing.py` | PDF document indexing |
| `registry.py` | Knowledge source registry |
| `strategies.py` | Strategy database queries |

---

## 8. Data Storage Layout

```
data/
├── agent_memory/          # Per-agent file-based memory
├── agent_queues/          # Agent task queues
├── assets/                # Shared trading assets
├── backtests/             # Backtest result files
├── chromadb/              # ChromaDB vector store (KB articles)
├── chromadb_test/         # Test ChromaDB instance
├── compliance_files/      # Prop firm compliance documents
├── db/                    # SQLite + DuckDB database files
├── departments/           # Department-level data
├── exports/               # Data exports
├── inputs/                # Input files for agents
├── knowledge/             # Curated knowledge base
├── logs/                  # Application logs (JSON for Promtail)
├── markdown/              # Converted markdown articles
├── nprd/                  # NPRD strategy documents
├── pdf_documents/         # Processed PDF content
├── pdf_jobs/              # PDF processing jobs
├── qdrant_db/             # Qdrant vector store
├── queues/                # Message queues
├── raw_articles/          # Raw scraped HTML
├── scraped_articles/      # Processed article JSON/Markdown
├── shared_assets/         # Assets shared between agents
├── strategies/            # Strategy definitions
├── trd/                   # Trading Rule Documents
├── videos/                # Downloaded videos
└── wiki/                  # Internal wiki content
```

---

## 9. HMM Training Pipeline

The HMM model must be trained before it can be used for regime detection.

**Training script:** `scripts/train_hmm.py`
**Validation script:** `scripts/validate_hmm.py`
**Scheduler:** `scripts/schedule_hmm_training.py` (runs via systemd `hmm-training-scheduler.service`)

**Training inputs:** DuckDB market data (`market_data` table)
**Feature extraction:** `src/risk/physics/hmm/features.py`
**Output:** Trained model saved to version control via `src/router/hmm_version_control.py`

---

## 10. Tiered Storage (`src/database/tiered_storage.py`)

Hot/warm/cold storage tiering for time-series data:

| Tier | Storage | Retention | Script |
|------|---------|-----------|--------|
| Hot | In-memory / DuckDB | Recent 30 days | — |
| Warm | DuckDB on disk | 30-180 days | `scripts/migrate_hot_to_warm.py` |
| Cold | Compressed archive | > 180 days | `scripts/archive_warm_to_cold.py` |

---

## 11. Monitoring & Observability

### Prometheus (`docker/prometheus/`)
- Metrics scraped from backend API (`/metrics` endpoint)
- MT5 Bridge metrics on port 9091
- Custom trading metrics via `prometheus_client`

### Grafana Dashboards (`monitoring/dashboards/`)
- Pre-built dashboards for trading metrics, agent activity, system health

### Loki + Promtail (`docker/promtail/`)
- JSON-structured logs from backend + MT5 bridge scraped by Promtail
- Log configuration: `src/monitoring/json_logging.py`
- Sent to Grafana Cloud Loki for centralized log search

### Log Files
```
logs/
├── api.json           # Backend API JSON logs
├── mt5-bridge.json    # MT5 Bridge logs
├── router.json        # Strategy router logs
```
