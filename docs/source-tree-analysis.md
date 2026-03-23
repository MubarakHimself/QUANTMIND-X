# QUANTMINDX — Annotated Source Tree

**Generated:** 2026-03-11

---

## Root Structure

```
QUANTMINDX/
├── src/                        # FastAPI backend (Python 3.12)
├── quantmind-ide/              # Desktop IDE (SvelteKit 4 + Tauri 2)
├── mt5-bridge/                 # MT5 REST bridge (FastAPI, Windows/Wine)
├── mcp-servers/                # MCP server extensions
├── mcp-metatrader5-server/     # MT5 MCP tools (pip installable)
├── feature-server/             # Experimental features (SvelteKit 5, port 3002)
├── strategies-yt/              # YouTube strategy extraction module
├── data/                       # Runtime data (NOT in git except .gitkeep)
├── docs/                       # Documentation
├── config/                     # App config and MCP configs
├── scripts/                    # Utility and automation scripts
├── tests/                      # All test files
├── docker/                     # Dockerfiles for services
├── monitoring/                 # Grafana dashboards
├── systemd/                    # Systemd service files
├── extensions/                 # CLI extensions (Gemini video ingest)
├── _bmad/                      # BMAD workflow tooling
└── .opencode/                  # OpenCode AI assistant config
```

---

## Backend: `src/`

```
src/
├── api/
│   ├── server.py                   ⭐ FastAPI entry point (50+ routers, startup hooks)
│   ├── ide_endpoints.py            — create_ide_api_app() factory
│   ├── chat_endpoints.py           — Legacy chat API (DEPRECATED path)
│   ├── floor_manager_endpoints.py  ⭐ Canonical agent entry point
│   ├── workshop_copilot_endpoints.py — Workshop copilot chat
│   ├── department_mail_endpoints.py — Department mail REST API
│   ├── hmm_endpoints.py            ⭐ HMM regime system management
│   ├── hmm_inference_server.py     — Standalone HMM inference endpoint
│   ├── router_endpoints.py         — StrategyRouter REST API
│   ├── trading_endpoints.py        — Trading operations
│   ├── trading_floor_endpoints.py  — Trading floor control
│   ├── broker_endpoints.py         — Broker management + WS stream
│   ├── kill_switch_endpoints.py    — Kill switch control
│   ├── lifecycle_scanner_endpoints.py — Bot lifecycle
│   ├── agent_management_endpoints.py — Agent CRUD
│   ├── agent_session_endpoints.py  — Session management
│   ├── agent_activity.py           — Activity feed
│   ├── agent_metrics.py            — Agent metrics
│   ├── agent_queue_endpoints.py    — Task queue management
│   ├── agent_tools.py              — Tool call proxy
│   ├── tool_call_endpoints.py      — Direct tool calls
│   ├── claude_agent_endpoints.py   — Claude SDK agent runner
│   ├── memory_endpoints.py         — Memory CRUD + dept + unified
│   ├── graph_memory_endpoints.py   — Graph memory queries
│   ├── mcp_endpoints.py            — MCP server tools proxy
│   ├── analytics_endpoints.py      — DuckDB analytics queries
│   ├── analytics_db.py             — Analytics DB helpers
│   ├── model_config_endpoints.py   — LLM model configuration
│   ├── provider_config_endpoints.py ⭐ Provider config API (recently fixed)
│   ├── settings_endpoints.py       — Application settings
│   ├── demo_mode_endpoints.py      — Demo mode toggle
│   ├── health_endpoints.py         — Health check
│   ├── version_endpoints.py        — Version info
│   ├── metrics_endpoints.py        — Prometheus + WS metrics
│   ├── heartbeat.py                — Server heartbeat
│   ├── approval_gate.py            — Trade approval gate
│   ├── pdf_endpoints.py            — PDF upload + indexing
│   ├── ide_knowledge.py            — Knowledge search for IDE
│   ├── ide_files.py                — File operations for IDE
│   ├── ide_ea.py                   — EA file management
│   ├── ide_assets.py               — Assets browser
│   ├── ide_strategies.py           — Strategy management
│   ├── ide_timeframes.py           — Timeframes config
│   ├── ide_mt5.py                  — MT5 connection via IDE
│   ├── ide_backtest.py             — Backtest runner
│   ├── ide_chat.py                 — IDE chat interface
│   ├── ide_video_ingest.py         — Video ingest for IDE
│   ├── ide_trading.py              — Trading from IDE
│   ├── ide_models.py               — IDE model selection
│   ├── ide_handlers.py             — IDE handler utilities
│   ├── ide_handlers_*.py           — Handler sub-modules (broker, assets, knowledge, strategy, trading, video_ingest)
│   ├── trd_endpoints.py            — TRD management
│   ├── ea_endpoints.py             — EA registry
│   ├── github_endpoints.py         — GitHub sync
│   ├── tradingview_endpoints.py    — TradingView webhook
│   ├── journal_endpoints.py        — Trade journal
│   ├── session_endpoints.py        — Chat session management
│   ├── session_checkpoint_endpoints.py — Agent checkpoints
│   ├── evaluation_endpoints.py     — Strategy evaluation
│   ├── workflow_endpoints.py       — Workflow execution
│   ├── batch_endpoints.py          — Batch operations
│   ├── paper_trading_endpoints.py  — Paper trading (MT5-dependent)
│   ├── paper_trading/              — Paper trading sub-module
│   ├── video_to_ea_endpoints.py    — Video → EA workflow
│   ├── monte_carlo_ws.py           — Monte Carlo WebSocket
│   ├── websocket_endpoints.py      — Generic WebSocket
│   ├── websocket_metrics.py        — Metrics WebSocket
│   ├── ws_logger.py                — WebSocket logger
│   ├── tick_stream_handler.py      — Tick stream WebSocket
│   ├── phase5_endpoints.py         — Phase 5 migration endpoints
│   ├── dependencies.py             — FastAPI dependency injection
│   ├── pagination.py               — Pagination utilities
│   ├── trading/                    — Trading sub-module (routes, broker, control, data, backtest)
│   ├── ide/                        — IDE sub-module (file, terminal, session, models)
│   └── services/
│       ├── chat_service.py         — Chat business logic
│       ├── chat_session_service.py — Session state management
│       └── workshop_copilot_service.py — Workshop copilot service
│
├── agents/
│   ├── core/
│   │   └── base_agent.py           ⚠️ DEPRECATED LangGraph BaseAgent
│   ├── departments/
│   │   ├── floor_manager.py        ⭐ FloorManager singleton (canonical orchestrator)
│   │   ├── types.py                ⭐ Department enum, DepartmentHeadConfig, SubAgentType
│   │   ├── heads/                  — Department head implementations (Research, Dev, Trading, Risk, Portfolio)
│   │   ├── subagents/              — Sub-agent worker implementations
│   │   └── schemas/                — Pydantic schemas for department API
│   ├── subagent/
│   │   ├── spawner.py              — AgentSpawner with CheckpointManager + HeartbeatManager
│   │   └── __init__.py
│   ├── skills/
│   │   ├── trading_skills/         — Trading-domain skills
│   │   ├── data_skills/            — Data-domain skills
│   │   └── system_skills/          — System-domain skills
│   ├── tools/
│   │   ├── knowledge/              — Knowledge base tools (PageIndex client, ChromaDB, PDF)
│   │   ├── strategies_yt/          — YouTube strategy tools
│   │   └── mcp/                    — MCP tool adapter
│   ├── memory/                     — Agent memory subsystems
│   ├── streaming/                  — SSE streaming (AgentStreamEventType)
│   ├── session/                    — Session state management
│   ├── evaluation/                 — Agent evaluation tools
│   ├── batch/                      — Batch agent operations
│   ├── files/                      — File handling for agents
│   ├── hooks/                      — Pre/post hook system
│   ├── cron/                       — Scheduled agent tasks
│   ├── integrations/               — External integrations
│   ├── mcp/                        — MCP client + adapter
│   ├── registry.py                 ⚠️ DEPRECATED factory registry
│   ├── claude_config.py            — Workspace-based agent config (6 agents, workspaces deleted)
│   └── langchain_adapter.py        ⚠️ DEPRECATED LangChain adapter
│
├── router/
│   ├── engine.py                   ⭐ StrategyRouter + RegimeFetcher (top-level coordinator)
│   ├── sentinel.py                 ⭐ Sentinel (regime aggregator → RegimeReport)
│   ├── governor.py                 ⭐ Governor → RiskMandate (risk scalar authorization)
│   ├── enhanced_governor.py        — EnhancedGovernor (fee-aware Kelly sizing)
│   ├── commander.py                ⭐ Commander (strategy auction → bot selection)
│   ├── bot_manifest.py             — BotManifest + BotRegistry
│   ├── routing_matrix.py           — Bot-to-account assignment
│   ├── broker_registry.py          — Broker fee profile management
│   ├── kill_switch.py              — SmartKillSwitch (regime-aware)
│   ├── progressive_kill_switch.py  — ProgressiveKillSwitch (tiered)
│   ├── bot_circuit_breaker.py      — Per-bot consecutive loss quarantine
│   ├── lifecycle_manager.py        — Bot lifecycle (paper → live promotion/demotion)
│   ├── hmm_version_control.py      ⭐ HMM model sync (Contabo ↔ Cloudzy)
│   ├── hmm_deployment.py           — HMM deployment mode management
│   ├── multi_timeframe_sentinel.py — Multi-timeframe regime aggregation
│   ├── market_scanner.py           — Market opportunity scanner
│   ├── scanners/
│   │   ├── symbol_scanner.py       — Per-symbol opportunity detection
│   │   └── trend_scanner.py        — Trend continuation signals
│   ├── trade_logger.py             — Enhanced trade context logging
│   ├── session_detector.py         — Forex session detection (London/NY/Tokyo/Sydney)
│   ├── trd_watcher.py              — Watch data/trd/ for new TRDs (auto-register bots)
│   ├── video_ingest_watcher.py     — Watch for new video ingest jobs
│   ├── video_to_ea_workflow.py     — Video → EA full conversion workflow
│   ├── workflow_orchestrator.py    — EA creation workflow orchestration
│   ├── promotion_manager.py        — Strategy promotion pipeline
│   ├── lifecycle_manager.py        — Bot lifecycle events
│   ├── bot_cloner.py               — Clone bot to new symbol
│   ├── dynamic_bot_limits.py       — Dynamic per-account bot count
│   ├── fee_monitor.py              — Daily fee burn monitoring
│   ├── house_money.py              — House money position scaling
│   ├── virtual_balance.py          — Virtual balance for paper trading
│   ├── github_sync.py              — EA file sync from GitHub
│   ├── ea_registry.py              — EA deployment registry
│   ├── account_monitor.py          — Account monitoring
│   ├── alert_manager.py            — Alert management
│   ├── strategy_monitor.py         — Strategy monitoring
│   ├── session_monitor.py          — Session monitoring
│   ├── system_monitor.py           — System monitoring
│   ├── interface.py                — Router interface definitions
│   ├── state.py                    — Router state management
│   ├── socket_server.py            — ZMQ socket server for tick data
│   ├── sync.py                     — Sync utilities
│   ├── sessions.py                 — Session tracking
│   ├── cli.py                      — Router CLI
│   ├── commands/                   — Router CLI commands
│   └── signals/                    — Signal processing utilities
│
├── risk/
│   ├── physics/
│   │   ├── ising_sensor.py         ⭐ Ising lattice model (spin magnetization)
│   │   ├── chaos_sensor.py         — Lyapunov exponent chaos detection
│   │   ├── hmm_sensor.py           — HMM → Sentinel adapter
│   │   ├── hmm/
│   │   │   ├── models.py           ⭐ HMMFeatureExtractor (10-feature vector)
│   │   │   ├── features.py         — FeatureConfig dataclass
│   │   │   ├── indicators.py       — TechnicalIndicators (RSI, ATR, MACD)
│   │   │   ├── scaler.py           — FeatureScaler
│   │   │   └── utils.py            — HMM utilities
│   │   └── sensors/                — Additional physics sensors
│   ├── sizing/
│   │   └── kelly_engine.py         ⭐ PhysicsAwareKellyEngine (f* formula with physics multipliers)
│   ├── integration/                — Risk system integration helpers
│   ├── models/                     — Risk data models
│   └── integrations/mt5/           — MT5-specific risk adapters
│
├── database/
│   ├── models/
│   │   ├── trading.py              — TradeProposal, RiskTierTransition, CryptoTrade
│   │   ├── bots.py                 — BotCircuitBreaker, BotCloneHistory, DailyFeeTracking
│   │   ├── account.py              — PropFirmAccount
│   │   ├── market.py               — MarketOpportunity
│   │   ├── agents.py               — AgentSession
│   │   ├── agent_session.py        — Session checkpoint data
│   │   ├── hmm.py                  — HMMModel, HMMShadowLog, HMMSyncStatus
│   │   ├── performance.py          — PerformanceSnapshot
│   │   ├── session_checkpoint.py   — Long-running operation checkpoint
│   │   └── provider_config.py      — LLM provider configuration
│   ├── duckdb/
│   │   ├── analytics.py            — DuckDB analytics tables
│   │   └── market_data.py          — market_data schema (OHLCV)
│   ├── migrations/
│   │   ├── migration_runner.py     — Auto-run migrations on startup
│   │   └── add_*.py                — 14 migration files
│   ├── repositories/               — Repository pattern wrappers
│   └── engine.py                   — SQLAlchemy engine + session factory
│
├── memory/
│   ├── graph/                      — DuckDB-backed graph memory
│   │   ├── __init__.py
│   │   └── integrations/
│   └── __init__.py                 — Unified Memory Facade
│
├── monitoring/
│   ├── json_logging.py             — JSON file logging for Promtail
│   └── __init__.py                 — Prometheus metrics + track_api_request
│
├── data/                           — Market data providers and transformers
│   └── brokers/                    — Broker data adapters
│
├── backtesting/                    — Backtest engine
│
├── integrations/
│   ├── crypto/                     — Binance + other crypto adapters
│   └── github_ea_scheduler.py      — GitHub EA file sync scheduler
│
├── queues/                         — Async task queues
├── mcp/                            — Internal MCP client utilities
├── mcp_tools/                      — MCP tool wrapper registry
├── position_sizing/                — Standalone position sizing module
├── tui/                            — Terminal UI (Textual library)
│   └── components/                 — TUI component widgets
└── video_ingest/
    ├── downloader.py               — yt-dlp YouTube download
    ├── processor.py                — Audio + frame extraction
    ├── extractors.py               — LLM-based strategy extraction
    ├── job_queue.py                — Async job queue
    ├── models.py                   — VideoJob, VideoStrategy models
    ├── api.py                      — Internal processing API
    ├── cache.py                    — Caching layer
    ├── rate_limiter.py             — API rate limiting
    └── cli.py                      — CLI interface
```

---

## Frontend: `quantmind-ide/`

```
quantmind-ide/
├── src/
│   ├── routes/
│   │   └── +page.svelte            ⭐ App entry point
│   ├── lib/
│   │   ├── components/
│   │   │   ├── ActivityBar.svelte  — Left icon bar (view switcher)
│   │   │   ├── StatusBand.svelte   — Bottom status bar
│   │   │   ├── BottomPanel.svelte  — Bottom terminal panel
│   │   │   ├── Breadcrumbs.svelte  — Navigation breadcrumbs
│   │   │   ├── WorkshopView.svelte ⭐ AI chat workshop (main view)
│   │   │   ├── TradingFloorPanel.svelte ⚠️ Floor Manager routing bug here
│   │   │   ├── BotsPage.svelte     — Bot management view
│   │   │   ├── DatabasePage.svelte — DB browser
│   │   │   ├── AlgoForgePanel.svelte — Algorithm development
│   │   │   ├── agent-panel/        — Chat interface components (12 + 9 settings)
│   │   │   ├── trading-floor/      — Trading floor components (12)
│   │   │   ├── bots/               — Bot management components (9)
│   │   │   ├── charts/             — Chart components (FanChart, MonteCarlo, etc.)
│   │   │   └── settings/
│   │   │       └── ProvidersPanel.svelte ⭐ LLM provider config (recently fixed)
│   │   ├── api/
│   │   │   ├── api.ts              — Base HTTP client
│   │   │   ├── chatApi.ts          — Chat endpoint client
│   │   │   └── agentSessions.ts    — Session management client
│   │   ├── agents/
│   │   │   ├── agentManager.ts     — Frontend agent lifecycle
│   │   │   ├── agentStreamStore.ts — SSE streaming store
│   │   │   ├── copilotAgent.ts     — Copilot agent config
│   │   │   ├── analystAgent.ts     — Analyst agent config
│   │   │   ├── quantcodeAgent.ts   — QuantCode agent config
│   │   │   ├── claudeCodeAgent.ts  — Claude Code integration
│   │   │   └── langchainAgent.ts   ⚠️ DEPRECATED LangChain agent
│   │   └── stores/                 — Svelte reactive stores
│   ├── src-tauri/
│   │   ├── tauri.conf.json         — Tauri app configuration
│   │   └── src/main.rs             — Tauri entry point (Rust)
│   ├── svelte.config.js            — Static adapter + Tauri config
│   ├── vite.config.ts              — Vite bundler config
│   └── package.json                — npm scripts + dependencies
```

---

## MT5 Bridge: `mt5-bridge/`

```
mt5-bridge/
├── server.py                   ⭐ FastAPI app (port 8001, /trade endpoint)
├── requirements.txt            — MetaTrader5 SDK + FastAPI
└── README.md
```

---

## MCP Servers: `mcp-servers/`

```
mcp-servers/
├── backtest-mcp-server/        — Backtest execution via MCP protocol
│   ├── server.py               — MCP server entry point
│   └── tools/                  — Backtest tool definitions
└── quantmindx-kb/              — Knowledge base MCP server
    ├── server.py               — ChromaDB-backed MCP server
    ├── server_chroma.py        — ChromaDB-specific implementation
    ├── start.sh                — Server start script
    └── tests/                  — KB retrieval + skills tests
```

---

## MCP MT5 Server: `mcp-metatrader5-server/`

```
mcp-metatrader5-server/
├── src/mcp_mt5/
│   ├── main.py                 — MCP server entry (stdio transport)
│   └── tools/                  — MT5 SDK tool definitions
└── setup.py / pyproject.toml   — pip installable package
```

---

## Scripts: `scripts/`

Key scripts organized by function:

```
scripts/
├── start_all.sh                — Start all services
├── install.sh                  — Full installation script
├── validate_env.sh             — Validate environment vars
│
├── # HMM Model Management
├── train_hmm.py                — Train HMM on DuckDB data
├── validate_hmm.py             — Validate trained model
├── schedule_hmm_training.py    — Scheduled HMM retraining (APScheduler)
│
├── # Market Data Scrapers
├── firecrawl_scraper.py        — Scrape MQL5 via Firecrawl API
├── simple_scraper.py           — Lightweight HTML scraper
├── scraper.py                  — Full scraper + deduplication
├── category_crawler.py         — Crawl by MQL5 category
├── crawl_all_categories.py     — Batch all-category crawl
├── filter_analyst_kb.py        — Filter articles for analyst KB
├── search_kb.py                — CLI knowledge base search
│
├── # Bot / Broker Management
├── populate_broker_registry.py — Seed broker data
├── validate_broker_registry.py — Validate broker data
├── schedule_lifecycle_check.py — Bot lifecycle daily check
│
├── # Data Pipeline
├── index_to_pageindex.py       — Push docs to PageIndex services
├── generate_document_index.py  — Generate document index metadata
├── video_ingest_cli.py         — Video ingest CLI wrapper
├── migrate_hot_to_warm.py      — Hot → warm tier migration
├── archive_warm_to_cold.py     — Warm → cold tier archiving
│
├── # Deployment
├── sync_config.sh              — Sync config to VPS
├── setup_contabo_crons.sh      — Set up cron jobs on Contabo VPS
├── quantmind_cli.sh            — Main CLI wrapper
│
└── # Database
    └── populate_broker_registry.py — Broker data seeding
```

---

## Tests: `tests/`

```
tests/
├── agents/         — Agent unit tests
├── api/            — API endpoint tests
├── backtesting/    — Backtest engine tests
├── benchmarks/     — Performance benchmarks
├── bots/           — Bot logic tests
├── brokers/        — Broker integration tests
├── cache/          — Cache tests
├── crypto/         — Crypto module tests
├── data/           — Data pipeline tests
├── database/       — DB model tests
├── e2e/            — End-to-end tests
├── integration/    — Integration tests
├── load/           — Load tests
├── mcp_servers/    — MCP server tests
├── mcp_tools/      — MCP tool tests
├── memory/         — Memory system tests
├── mql5/           — MQL5 tests
├── nprd/           — NPRD tests
├── position_sizing/— Position sizing tests
├── properties/     — Property-based tests (Hypothesis)
├── queues/         — Queue tests
├── risk/           — Risk engine tests
├── router/         — Router tests
├── tui/            — TUI tests
└── unit/           — Misc unit tests
```

---

## Config: `config/`

```
config/
├── settings/                   — Application settings (YAML/JSON)
├── mcp/                        — MCP server configs per agent type
└── agents/                     — Agent-specific configuration
```

---

## Data: `data/` (Runtime, not in git)

```
data/
├── db/
│   ├── quantmind.db            — SQLite operational database
│   └── analytics.duckdb        — DuckDB analytics database
├── chromadb/                   — ChromaDB vector store (KB articles)
├── qdrant_db/                  — Qdrant vector store
├── agent_memory/               — Per-agent file-based memory
├── agent_queues/               — Agent task queue files
├── scraped_articles/           — MQL5 scraped articles (JSON/MD)
├── knowledge_base/             — Curated knowledge base
├── trd/                        — Trading Rule Documents
├── strategies/                 — Strategy definitions
├── backtests/                  — Backtest results
├── logs/                       — JSON logs (api.json, router.json, etc.)
├── videos/                     — Downloaded YouTube videos
└── pdf_documents/              — Processed PDFs
```

---

## Docker: `docker/`

```
docker/
├── pageindex/
│   ├── Dockerfile.articles     — PageIndex for scraped articles (port 3000)
│   ├── Dockerfile.books        — PageIndex for knowledge books (port 3001)
│   └── Dockerfile.logs         — PageIndex for logs (port 3002)
├── prometheus/
│   └── prometheus.yml          — Prometheus scrape config
└── promtail/
    └── config.yml              — Promtail log scrape config
```

---

## Monitoring: `monitoring/`

```
monitoring/
└── dashboards/                 — Pre-built Grafana dashboard JSON
```

---

## Systemd: `systemd/`

```
systemd/
├── quantmind-api.service       — FastAPI backend (port 8000)
├── quantmind-tui.service       — TUI server (tmux session)
└── hmm-training-scheduler.service — HMM training cron
```

---

## Key File Relationships

| If you're working on... | Key files |
|------------------------|-----------|
| Agent routing bug | `CopilotPanel.svelte:123` → `floor_manager_endpoints.py` |
| HMM deployment | `hmm_version_control.py` + `hmm_deployment.py` + `hmm_endpoints.py` |
| Kelly position sizing | `kelly_engine.py` + `governor.py` + `enhanced_governor.py` |
| Bot lifecycle | `lifecycle_manager.py` + `bot_circuit_breaker.py` + `promotion_manager.py` |
| Adding a new API endpoint | `src/api/new_endpoints.py` → `server.py` (include router) |
| Adding a new DB model | `src/database/models/` + `migrations/add_*.py` |
| Knowledge base | `mcp-servers/quantmindx-kb/` + `src/agents/tools/knowledge/` |
| Department system | `floor_manager.py` + `types.py` + `heads/` |
