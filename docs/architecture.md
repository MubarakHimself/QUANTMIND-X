# QUANTMINDX — System Architecture

**Generated:** 2026-03-11

---

## System Overview

QUANTMINDX is a modular autonomous algorithmic trading platform organized as a Python/TypeScript monorepo. It combines an AI-agent orchestration layer, a physics-inspired market analysis engine, a real-time strategy router, and a desktop IDE frontend.

---

## Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUANTMINDX Monorepo                         │
├──────────────┬────────────────┬──────────────┬──────────────────┤
│  quantmind-  │  src/          │  mt5-bridge/ │  mcp-servers/    │
│  ide/        │  (FastAPI API) │  (MT5 Bridge)│  (Extensions)    │
│  (SvelteKit) │                │              │                  │
├──────────────┴────────────────┴──────────────┴──────────────────┤
│            data/ (SQLite + DuckDB + vector stores)              │
├─────────────────────────────────────────────────────────────────┤
│            Docker (PageIndex + Prometheus + Loki)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: FastAPI Backend (`src/`)

### Entry Point

`src/api/server.py` — Uvicorn FastAPI app on port 8000. At startup:
1. Loads `.env` environment variables
2. Configures JSON file logging (Promtail/Loki)
3. Includes 50+ API routers (see [API Contracts](./api-contracts.md))
4. Initializes `StrategyRouter` singleton
5. Starts `RegimeFetcher` background poll task
6. Starts `LifecycleScheduler` (daily bot lifecycle check at 3:00 AM UTC)
7. Starts `MarketScannerScheduler` (session-based scanning)
8. Starts GitHub EA Scheduler (`src/integrations/github_ea_scheduler.py`)
9. Starts Agent SSE Stream Handler
10. Starts Prometheus metrics server (port 9090)

### Backend Sub-domains

```
src/
├── api/           — 50+ FastAPI routers (REST + WebSocket + SSE)
├── agents/        — AI agent system (Department System + legacy)
├── backtesting/   — Backtest engine
├── data/          — Market data providers and transformers
├── database/      — SQLAlchemy models, DuckDB, migrations
├── integrations/  — External service integrations (crypto, GitHub)
├── memory/        — Unified memory + graph memory
├── monitoring/    — Prometheus metrics, JSON logging
├── mcp/           — Internal MCP client utilities
├── mcp_tools/     — MCP tool wrapper registry
├── position_sizing/ — Standalone position sizing module
├── queues/        — Async task queues
├── risk/          — Risk engine (physics sensors, Kelly, prop firm)
├── router/        — Strategy Router (the Sentient Loop)
├── tui/           — Terminal User Interface (Textual)
└── video_ingest/  — YouTube video strategy extraction pipeline
```

---

## Part 2: Agent Architecture

### Three Paradigms (Critical Context)

QUANTMINDX contains three agent paradigms in varying states of deprecation:

| Paradigm | Location | Status | Notes |
|----------|----------|--------|-------|
| **Legacy LangGraph** | `src/agents/core/base_agent.py` | **Deprecated** | Still imports `langgraph`, `langchain_openai`. Do not extend. |
| **Department System** | `src/agents/departments/` | **Active / Canonical** | `FloorManager` + 5 Department Heads |
| **Factory/Container** | `src/agents/registry.py` | **Deprecated** | Stub file with `logger.warning(...)`. Do not use. |

### Department System (Canonical)

```
FloorManager (Opus tier)
      ↓ keyword classify_task()
      ↓ dispatch() via DepartmentMailService
      ↓
  ┌──────────────────────────────────┐
  │  Department Heads (Sonnet tier)  │
  │  ● Research                      │
  │  ● Development                   │
  │  ● Trading                       │
  │  ● Risk                          │
  │  ● Portfolio                     │
  └──────────────────────────────────┘
             ↓ spawn SubAgents (Haiku tier)
  ┌─────────────────────────────────────────┐
  │  SubAgentType (15 types):               │
  │  analyst, quantcode, pinescript,        │
  │  risk_calculator, backtester,           │
  │  data_researcher, code_reviewer,        │
  │  strategy_optimizer, report_writer,     │
  │  market_analyst, portfolio_rebalancer,  │
  │  paper_trader, live_trader, copilot,    │
  │  executor                               │
  └─────────────────────────────────────────┘
```

### API Routing (Known Bug)

The Workshop UI in `CopilotPanel.svelte` currently sends to `/api/chat/send` (legacy endpoint) instead of `/api/floor-manager/chat` (canonical). This means UI agent interactions bypass the Department System. See `docs/plans/2026-03-08-agent-architecture-migration-map.md` for the fix.

### Agent Communication

- **Department Mail** — SQLite-backed async message bus at `.quantmind/department_mail.db`
- **Agent SSE Stream** — `GET /api/agents/stream` — real-time event streaming to UI
- **Agent Memory** — Unified Memory Facade → 6 memory subsystems (file-based, DB, graph, ChromaDB, Qdrant, AgentDB)

---

## Part 3: Strategy Router (`src/router/`)

The real-time trading orchestration engine. See [Strategy Router](./strategy-router.md) for full detail.

```
Tick → Sentinel → Governor → Commander → RoutingMatrix → MT5 Bridge
```

**Key components:**
- `Sentinel` — market regime classification (Ising + HMM sensors)
- `Governor` / `EnhancedGovernor` — risk scalar authorization
- `Commander` — strategy auction (auction / priority / round-robin)
- `RoutingMatrix` — bot-to-account assignment
- `ProgressiveKillSwitch` + `SmartKillSwitch` — tiered circuit breakers
- `BotCircuitBreaker` — per-bot consecutive loss quarantine

---

## Part 4: Risk Engine (`src/risk/`)

Physics-inspired multi-layer risk system. See [Risk Management](./risk-management.md) for full detail.

```
src/risk/
├── physics/
│   ├── ising_sensor.py       — Ising lattice spin model
│   ├── chaos_sensor.py       — Lyapunov exponent chaos detection
│   ├── hmm/                  — HMM regime model (see hmm-system.md)
│   └── hmm_sensor.py         — HMM → Sentinel adapter
├── sizing/
│   └── kelly_engine.py       — PhysicsAwareKellyEngine (f* formula)
├── integration/              — Risk integration utilities
├── models/                   — Risk model types
└── integrations/mt5/         — MT5-specific risk adapters
```

**Kelly formula:**
```
f* = (p×b − q) / b
f_base = f* × 0.5               (Half-Kelly)
M_physics = min(P_λ, P_χ, P_E) (weakest-link aggregation)
final_risk = f_base × M_physics (capped at 10%)
```

---

## Part 5: Database Layer (`src/database/`)

### Databases

| Database | File | ORM | Purpose |
|----------|------|-----|---------|
| SQLite | `data/db/quantmind.db` | SQLAlchemy | Operational: bots, trades, agents, accounts |
| DuckDB | `data/db/analytics.duckdb` | Native DuckDB | Time-series OHLCV, backtest results |
| AgentDB | `agentdb.db` + `agentdb.rvf` | Custom | Agent memory + vector search |
| Dept Mail | `.quantmind/department_mail.db` | SQLite raw | Inter-agent message bus |
| ChromaDB | `data/chromadb/` | ChromaDB | Knowledge base semantic search |
| Qdrant | `data/qdrant_db/` | Qdrant | Alternative vector store |

### Migration System

`src/database/migrations/migration_runner.py` — runs all pending migrations on server startup. 14 migration files, each adding a specific table or schema change.

---

## Part 6: Frontend IDE (`quantmind-ide/`)

SvelteKit 4 application, packaged as Tauri 2 desktop app or standalone web app.

**Key facts:**
- Port: 3001 (web dev mode)
- Backend: `http://localhost:8000`
- Svelte stores for reactive state (`src/lib/stores/`)
- 65+ Svelte components

**Layout:** VS Code-inspired with ActivityBar, panels, and modals.

**Views:**
- `WorkshopView.svelte` — AI agent chat (primary view)
- `TradingFloorPanel.svelte` — Department-based trading floor
- `BotsPage.svelte` — Bot management
- `DatabasePage.svelte` — Database browser
- `AlgoForgePanel.svelte` — Algorithm development

See [Component Inventory](./component-inventory.md) for full component list.

---

## Part 7: MT5 Bridge (`mt5-bridge/`)

Thin FastAPI service bridging QUANTMINDX to MetaTrader 5 SDK.

- **Port:** 8001
- **Endpoints:** `POST /trade`, `GET /positions`, `GET /account`, `GET /history`
- **Auth:** Bearer token (`MT5_BRIDGE_TOKEN` env var)
- **Metrics:** Prometheus metrics on port 9091
- **Platform:** Windows/Wine (MetaTrader 5 only runs on Windows)

---

## Part 8: MCP Servers (`mcp-servers/`)

Extensions exposing capabilities to AI agents via the Model Context Protocol.

| Server | Purpose | Transport |
|--------|---------|-----------|
| `mcp-metatrader5-server/` | MT5 SDK tools via MCP | stdio |
| `mcp-servers/backtest-mcp-server/` | Backtest execution via MCP | stdio/HTTP |
| `mcp-servers/quantmindx-kb/` | Knowledge base semantic search | stdio/HTTP |

---

## Inter-Service Communication

### Internal Communication

| From | To | Method |
|------|----|--------|
| SvelteKit IDE | FastAPI | HTTP REST (port 8000) |
| SvelteKit IDE | FastAPI | WebSocket (port 8000) |
| FastAPI | MT5 Bridge | HTTP REST (port 8001) |
| FastAPI | Contabo HMM | HTTP REST (VPS URL) |
| Department Heads | Floor Manager | SQLite message bus |
| Agents | StrategyRouter | In-process function calls |

### External Services

| Service | Purpose | Direction |
|---------|---------|-----------|
| Contabo VPS | HMM training + regime API | Outbound poll every 5 min |
| Cloudzy VPS | Live trading deployment | SSH/SFTP for model sync |
| Grafana Cloud | Log ingestion (Loki) | Promtail push |
| Grafana Cloud | Metrics ingestion | Prometheus remote_write |
| OpenRouter | LLM routing (optional) | Outbound |
| Anthropic API | Claude models (primary) | Outbound |
| OpenAI API | GPT models (optional) | Outbound |

---

## Startup Sequence

```
1. load_dotenv()
2. configure JSON logging (Promtail/Loki)
3. create FastAPI app (create_ide_api_app())
4. include 50+ routers
5. startup_event():
   a. start Prometheus metrics server (port 9090)
   b. initialize StrategyRouter (smart_kill + kelly_governor + multi_timeframe)
   c. start RegimeFetcher background poll (asyncio task)
   d. start GitHub EA Scheduler
   e. start LifecycleScheduler (APScheduler, 3 AM UTC daily)
   f. start MarketScannerScheduler (session-based)
   g. start Agent SSE StreamHandler
   h. start Metrics WebSocket broadcast task (1s interval)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Physics-inspired risk (Ising + HMM) | Traditional technical indicators fail in chaotic markets; physics models capture phase transitions |
| Half-Kelly sizing with physics multiplier | Full Kelly is too aggressive; physics multiplier provides additional safety margin |
| Department System over LangGraph | LangGraph was too opaque for trading operations; department system provides clear audit trail via mail |
| SQLite for operational data | Lightweight, no separate database server needed; appropriate for <50 concurrent agents |
| DuckDB for analytics | Columnar storage optimized for OHLCV time-series queries |
| Tauri for desktop | Native OS integration (file system, system tray) without Electron overhead |
| Contabo + Cloudzy split | Separation of training (compute-heavy) from trading (latency-sensitive) |
| 50-bot global limit | Prevents prop firm account over-exposure and portfolio correlation blow-up |
