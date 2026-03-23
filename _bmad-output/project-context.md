---
project_name: 'QUANTMINDX'
user_name: 'Mubarak'
date: '2026-03-11'
sections_completed: ['technology_stack', 'critical_implementation_rules', 'code_patterns', 'testing_rules', 'architecture_constraints']
existing_patterns_found: 47
---

# Project Context for AI Agents — QUANTMINDX

_Critical rules and patterns that AI agents must follow when implementing code in this project. Focus on unobvious details that prevent implementation mistakes._

---

## 1. Technology Stack & Versions

### Backend (Python)
| Technology | Version | Notes |
|-----------|---------|-------|
| Python | 3.12 | Required. No 3.10 or below. |
| FastAPI | latest | All REST + WebSocket endpoints |
| Uvicorn | latest | ASGI server, port 8000 |
| Pydantic | >=2.0.0 | v2 syntax only — `model_validate`, NOT `parse_obj` |
| SQLite | system | Operational data (bots, mail, sessions) |
| DuckDB | >=0.9.0 | Analytics only — never use for transactional writes |
| Anthropic SDK | >=0.5.0 | Primary AI. Models: claude-opus-4, claude-sonnet-4 |
| OpenAI SDK | >=1.0.0 | Legacy compat only — do not use for new features |
| pandas | >=2.1.0 | All DataFrame operations |
| numpy | >=1.24.0 | Numerical ops |
| python-dotenv | latest | `.env` loaded FIRST via `load_dotenv()` before other imports |
| pyzmq | >=25.1.2 | ZMQ sockets for HFT tick streams |
| aiohttp | >=3.9.1 | Async HTTP for WebSocket (crypto module) |
| Prometheus | 0.19.0 | Exact pin — metrics on port 9090 |

### Frontend (TypeScript/Svelte)
| Technology | Version | Notes |
|-----------|---------|-------|
| SvelteKit | ^2.0.0 | **Static adapter only** — no SSR |
| Svelte | ^4.0.0 | Svelte 4 syntax (not Svelte 5 runes) |
| Tauri | ^2.0.0 | Desktop shell. API: `@tauri-apps/api ^2.0.0` |
| TypeScript | ^5.0.0 | Strict mode enforced |
| Vite | ^5.0.0 | Build tool, dev port 1420 |
| Zod | ^3.22.0 | Runtime validation for API response shapes |
| Monaco Editor | ^0.45.0 | Code editor — loaded via `vite-plugin-monaco-editor` |
| lightweight-charts | ^4.1.0 | TradingView-style candle charts |
| Chart.js | ^4.4.1 | Metrics/analytics charts |
| D3 | ^7.8.5 | Custom visualizations |
| lucide-svelte | ^0.300.0 | Icon library — always import named icons |
| uuid | ^9.0.0 | UUID generation |

### Video Ingest Server (Node.js)
| Technology | Version | Notes |
|-----------|---------|-------|
| Express | ^5.2.1 | REST server for video ingestion |
| @google/generative-ai | ^0.24.1 | Gemini for video analysis |
| OpenAI | ^6.17.0 | Node OpenAI SDK |
| dotenv | ^17.2.3 | Env loading |

---

## 2. Critical Architecture Rules

### 2.1 — Agent Paradigm (MOST IMPORTANT)

There are **3 agent paradigms** — only ONE is canonical:

| Paradigm | Location | Status | Rule |
|----------|----------|--------|------|
| **Department System** | `src/agents/departments/` | ✅ CANONICAL | **Use this for all new agent work** |
| Legacy LangGraph | `src/agents/core/base_agent.py` | ❌ DEPRECATED | Do NOT extend. LangGraph/LangChain removed from requirements |
| Registry/Factory | `src/agents/registry.py` | ❌ DEPRECATED | Stub file only — logs a warning. Do NOT use. |

**Department hierarchy:**
- `FloorManager` (Opus tier) — classifies via `classify_task()` → dispatches via `DepartmentMailService`
- 5 Department Heads (Sonnet tier): Research, Development, Trading, Risk, Portfolio
- SubAgents (Haiku tier): 15 types including analyst, quantcode, pinescript, executor, backtester

### 2.2 — Known Routing Bug (Do Not Repeat)

**CopilotPanel.svelte** currently sends to `/api/chat/send` (legacy) instead of `/api/floor-manager/chat` (canonical). The fix plan is in `docs/plans/2026-03-08-agent-architecture-migration-map.md`. **Do not replicate this pattern in new components** — always use `/api/floor-manager/chat` for agent interactions.

### 2.3 — SvelteKit Static Adapter (No SSR)

The frontend uses `@sveltejs/adapter-static` with `strict: true`. This means:
- **NEVER** create `+layout.server.ts` or `+page.server.ts` — these are server-only and not supported
- **NEVER** use `load()` functions that fetch data server-side
- All data fetching must happen client-side (`onMount`, reactive statements)
- All routes must be prerenderable or use `fallback: 'index.html'`

### 2.4 — API Proxy Architecture

Vite dev server (port 1420) proxies all API calls:
- `/api/*` → `http://localhost:8000` (FastAPI backend)
- `/ws/*` → `ws://localhost:8000` (WebSocket)

**Frontend API calls MUST use relative paths** (e.g., `/api/trading/bots`) — never hardcode `localhost:8000` in Svelte components. Use the centralized `apiFetch` wrapper in `quantmind-ide/src/lib/api.ts`.

### 2.5 — Python Import Convention

All backend imports use the `src.` prefix from the project root:
```python
from src.api.ide_endpoints import create_ide_api_app
from src.agents.departments.floor_manager import FloorManager
from src.risk.sizing.kelly_engine import PhysicsAwareKellyEngine
```

**Never use relative imports** like `from .api import ...` in the main `src/` tree. The server must always be started from the project root (`/home/mubarkahimself/Desktop/QUANTMINDX`).

### 2.6 — Environment Variables

`.env` must be loaded FIRST via `load_dotenv()` before ANY other imports that read env vars:
```python
from dotenv import load_dotenv
load_dotenv()  # MUST be before all other application imports
```
Never hardcode API keys. Never commit `.env` files. Required env vars: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (legacy), `CORS_ALLOWED_ORIGINS` (optional).

---

## 3. Code Organization Patterns

### 3.1 — Frontend File Structure
```
quantmind-ide/src/
├── lib/
│   ├── api.ts              — Centralized API layer (single apiFetch wrapper)
│   ├── api/                — Sub-modules (e.g., api/memory.ts)
│   ├── components/         — PascalCase .svelte files
│   ├── config/             — api.ts config (API_CONFIG.API_URL)
│   ├── stores/             — TypeScript Svelte stores
│   │   └── index.ts        — Re-export barrel for all stores
│   ├── services/           — Non-reactive service classes (camelCase .ts)
│   └── ws-client.ts        — WebSocket client factory
├── routes/
│   ├── +layout.svelte      — Root layout
│   ├── +page.svelte        — Root page
│   └── [feature]/          — Feature-specific routes
```

### 3.2 — Component Naming Conventions
- Svelte components: **PascalCase** (`TradingFloorPanel.svelte`, `StatusBand.svelte`)
- Stores: **camelCase** with `Store` suffix (`chatStore.ts`, `settingsStore.ts`)
- Services: **camelCase** with `Service`/`Manager` suffix (`chatManager.ts`, `settingsManager.ts`)
- Test files: `ComponentName.test.ts` (co-located with component)

### 3.3 — Backend File Structure
```
src/
├── api/
│   ├── server.py           — FastAPI app entry point (50+ routers)
│   ├── ide_endpoints.py    — Main IDE app factory (create_ide_api_app)
│   ├── ide_*.py            — Modular endpoint groups (ide_trading.py, etc.)
│   └── *_endpoints.py      — Feature-specific routers
├── agents/departments/     — CANONICAL agent system
├── risk/physics/           — Ising, HMM, Chaos sensors
├── router/                 — StrategyRouter (Sentinel → Commander)
└── database/               — SQLAlchemy models
```

### 3.4 — FastAPI Router Pattern
Each endpoint file exports a `router = APIRouter()`:
```python
from fastapi import APIRouter
router = APIRouter()

@router.get("/endpoint")
async def handler():
    ...
```
Routers are included in `server.py` with a prefix: `app.include_router(router, prefix="/api")`.

### 3.5 — Svelte Component Pattern
All Svelte components use:
```svelte
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  // State declared as let (reactive by default)
  let loading = true;
  let data = [];

  onMount(async () => { /* fetch data */ });
  onDestroy(() => { /* cleanup intervals */ });
</script>
```
- Always clean up `setInterval` / `setInterval` in `onDestroy`
- Use `$lib/api` for API calls (maps to `quantmind-ide/src/lib/`)
- Import stores as `import { storeName } from '$lib/stores'`

---

## 4. Data Layer Rules

### 4.1 — Database Usage
- **SQLite** — operational: bots, sessions, department mail, memory entries
- **DuckDB** — analytics only (read-heavy, time-series, backtests). Do NOT write real-time operational data to DuckDB.
- Department mail bus: `.quantmind/department_mail.db` (SQLite)

### 4.2 — Pydantic v2 Syntax
Always use Pydantic v2 patterns:
```python
# CORRECT (v2)
class MyModel(BaseModel):
    field: str

instance = MyModel.model_validate(data)  # NOT parse_obj()
data = instance.model_dump()             # NOT .dict()

# WRONG (v1 - do not use)
instance = MyModel.parse_obj(data)
data = instance.dict()
```

### 4.3 — WebSocket Pattern
Frontend WebSocket clients are created via factory functions in `ws-client.ts`:
```typescript
import { createTradingClient, createWebSocketClient } from '$lib/api';
```
Backend WebSocket endpoints live in `src/api/websocket_endpoints.py`. Always handle reconnection logic client-side.

---

## 5. Testing Rules

### 5.1 — Python Tests
- File pattern: `test_*.py` in `/tests` directory
- Class pattern: `Test*`
- Function pattern: `test_*`
- Asyncio mode: `auto` (no need for `@pytest.mark.asyncio` decorator)
- Markers: `slow`, `integration`, `unit`, `load`, `property`
- Run from project root: `pytest` (uses `pytest.ini`)
- Coverage source: `src/` — exclude `tests/`, `venv/`, `__pycache__/`

### 5.2 — Frontend Tests
- Test framework: Vitest (not Jest)
- Co-locate test files with components: `BacktestRunner.test.ts` next to `BacktestRunner.svelte`
- Use `@testing-library/svelte` for component testing

---

## 6. Risk Engine Critical Rules

### 6.1 — Physics Sensor Architecture
The risk engine is **read-only for trading** — it classifies regime but does NOT send orders:
- `IsingModelSensor` — market correlation (spin magnetization)
- `HMMSensor` — regime: TREND / RANGE / BREAKOUT / CHAOS
- `LyapunovExponent` — chaos instability detection
- `PhysicsAwareKellyEngine` — Kelly fraction with physics multipliers

**Never bypass the Sentinel layer** when routing strategy signals. All trade signals must flow: `Tick → Sentinel → Governor → Commander → MT5Bridge`.

### 6.2 — Kill Switch Hierarchy
There are TWO kill switch implementations:
- `ProgressiveKillSwitch` — tiered, preferred for gradual shutdown
- `SmartKillSwitch` — pattern-based, for abnormal market detection
- `BotCircuitBreaker` — per-bot, quarantine on consecutive losses

**Always use the circuit breaker pattern** — never add a "force close all" without going through the kill switch hierarchy.

---

## 7. AI Model Usage Rules

Models are tiered by cost and capability:
- **FloorManager** = `claude-opus-4` (highest reasoning, task classification)
- **Department Heads** = `claude-sonnet-4` (balanced, most tasks)
- **SubAgents** = `claude-haiku-4` (fast, simple tasks, high volume)

**Never use claude-opus-4 for subagent tasks** — cost will spike. Use the three-tier hierarchy.

The Workshop/Copilot UI allows model selection via `AgentModelSelector.svelte`. Provider configuration is managed via `/api/settings/providers` endpoints.

---

## 8. File & Import Anti-Patterns

### DO NOT:
- Import from `src/agents/core/base_agent.py` (deprecated LangGraph)
- Import from `src/agents/registry.py` (deprecated stub)
- Use `langchain`, `langgraph`, `langchain_openai` (removed from requirements)
- Use SSR in SvelteKit (`+page.server.ts`, `+layout.server.ts`)
- Hardcode `localhost:8000` in Svelte components
- Use Pydantic v1 methods (`.dict()`, `.parse_obj()`)
- Save test files to project root (use `/tests`)
- Save markdown/docs to project root (use `/docs`)
- Commit `.env` or secrets

### DO:
- Use `DepartmentMailService` for agent-to-agent communication
- Use `apiFetch<T>()` wrapper for all frontend API calls
- Use `$lib/api` path alias in Svelte components
- Load `.env` before all other imports in Python entry points
- Run Python server from project root directory
- Use `asyncio_mode = auto` in pytest (no decorator needed)
- Wrap FastAPI startup tasks in `@app.on_event("startup")` or lifespan context

---

## 9. Deployment Context

- Backend: systemd service, runs as `uvicorn src.api.server:app --port 8000`
- Frontend: Tauri desktop app (`tauri:build`) or Vite static build
- Docker: Prometheus + Loki + Grafana via `docker-compose.yml`
- Prometheus metrics: port 9090
- Observability: JSON logs scrapped by Promtail → Loki → Grafana Cloud
- VPS: Contabo/Cloudzy, see `docs/deployment-guide.md`
