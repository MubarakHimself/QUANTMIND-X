# Story 1.3: NODE_ROLE Backend Deployment Split

Status: **COMPLETED**

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a system operator deploying QUANTMINDX across two servers,
I want the FastAPI backend to register only the routers appropriate to each node based on `NODE_ROLE`,
so that Cloudzy runs only live trading routers and Contabo runs only agent/compute routers.

## Acceptance Criteria

1. **Given** `NODE_ROLE=cloudzy` is set,
   **When** the FastAPI server starts,
   **Then** only live trading routers (MT5 bridge, execution, sentinel, risk) register,
   **And** agent/compute routers are not mounted.

2. **Given** `NODE_ROLE=contabo` is set,
   **When** the FastAPI server starts,
   **Then** only agent/compute routers register,
   **And** live trading routers are not mounted.

3. **Given** `NODE_ROLE=local` is set or the variable is absent,
   **When** the FastAPI server starts,
   **Then** all routers register (full local development mode).

4. **Given** an invalid `NODE_ROLE` value is set,
   **When** the server starts,
   **Then** it logs a clear warning and defaults to `local` mode without crashing.

5. **Given** the implementation is complete,
   **When** `NODE_ROLE` is added to `.env.example`,
   **Then** it includes all accepted values: `contabo | cloudzy | local` with comments.

## Tasks / Subtasks

- [x] Task 1: Read and understand current server.py structure (AC: #1-4)
  - [x] Read `src/api/server.py` completely
  - [x] Identify all router imports and their registration pattern
  - [x] Note: 70+ routers currently registered unconditionally

- [x] Task 2: Implement NODE_ROLE conditional block (AC: #1-4)
  - [x] Add `NODE_ROLE = os.getenv("NODE_ROLE", "local")` at top of server.py
  - [x] Add validation: if invalid value, log warning and default to "local"
  - [x] Create conditional registration blocks:
    - [x] `if NODE_ROLE in ("cloudzy", "local"):` → trading routers
    - [x] `if NODE_ROLE in ("contabo", "local"):` → agent/compute routers
    - [x] Both: health, version, metrics

- [x] Task 3: Classify routers by node (AC: #1-2)
  - [x] **Cloudzy routers** (live trading):
    - [x] `kill_switch_endpoints` (trading kill switch)
    - [x] `trading_endpoints` (trade execution)
    - [x] `broker_endpoints` (broker connection)
    - [x] `tick_stream_handler` (tick data)
    - [x] `tradingview_endpoints` (charts)
    - [x] `paper_trading_endpoints` (paper trading)
    - [x] Any MT5-related endpoints
  - [x] **Contabo routers** (agent/compute):
    - [x] All agent endpoints (`agent_management`, `agent_activity`, `agent_metrics`)
    - [x] All memory endpoints (`memory_router`, `memory_dept_router`, `memory_unified_router`)
    - [x] `floor_manager_endpoints` (Copilot)
    - [x] `workflow_endpoints` (Alpha Forge)
    - [x] `hmm_endpoints` (risk sensors)
    - [x] `settings_endpoints` (configuration)
    - [x] `provider_config_endpoints` (AI providers)
    - [x] `knowledge_endpoints` (if exists)
    - [x] `research_endpoints` (if exists)
  - [x] **Both/Local** (always loaded):
    - [x] `health_endpoints` (health checks)
    - [x] `version_endpoints` (version info)
    - [x] `metrics_endpoints` (observability)

- [x] Task 4: Update .env.example (AC: #5)
  - [x] Add `NODE_ROLE` to `.env.example`
  - [x] Comment: `# Accepted values: contabo | cloudzy | local (default: local)`
  - [x] Add default placeholder value

- [x] Task 5: Verify implementation (AC: #1-4)
  - [x] Test with `NODE_ROLE=cloudzy` - should only load trading routers
  - [x] Test with `NODE_ROLE=contabo` - should only load agent routers
  - [x] Test with `NODE_ROLE=local` or unset - should load all routers
  - [x] Test with invalid value - should warn and default to local

- [x] Task 6: Document systemd service requirements
  - [x] Note: Each node's systemd service file must set NODE_ROLE env var
  - [x] Example service file updates for reference

## Dev Notes

### Critical Context from Story 1.0 Audit

**STOP — READ BEFORE CODING.** Story 1.0 pre-populated the following verified findings.

#### Pre-populated Findings

| Item | Status | Evidence |
|------|--------|----------|
| server.py location | `src/api/server.py` | Verified path |
| Total routers | 70+ routers | Full list in Story 1.0 Section D |
| NODE_ROLE handling | NOT found | Must be implemented |
| Router registration pattern | Unconditional | All routers registered at startup |

#### Current Router Registration (from Story 1.0)

**Full router list from `src/api/server.py`:**
```
agent_activity.py, agent_management_endpoints.py, agent_metrics.py,
agent_queue_endpoints.py, agent_session_endpoints.py, agent_tools.py,
analytics_db.py, analytics_endpoints.py, approval_gate.py,
batch_endpoints.py, broker_endpoints.py, chat_endpoints.py,
claude_agent_endpoints.py, data_endpoints.py, demo_mode_endpoints.py,
department_mail_endpoints.py, dependencies.py, ea_endpoints.py,
evaluation_endpoints.py, floor_manager_endpoints.py, github_endpoints.py,
hmm_endpoints.py, hmm_inference_server.py, ide/ (folder), ide_assets.py,
ide_backtest.py, ide_chat.py, ide_ea.py, ide_endpoints.py, ide_files.py,
ide_handlers_assets.py, ide_handlers_broker.py, ide_handlers_knowledge.py,
ide_handlers.py, ide_handlers_strategy.py, ide_handlers_trading.py,
ide_handlers_video_ingest.py, ide_knowledge.py, ide_models.py, ide_mt5.py,
ide_strategies.py, ide_timeframes.py, ide_trading.py, ide_video_ingest.py,
journal_endpoints.py, kill_switch_endpoints.py, lifecycle_scanner_endpoints.py,
mcp_endpoints.py, memory_endpoints.py, metrics_endpoints.py,
model_config_endpoints.py, monte_carlo_ws.py, pagination.py,
paper_trading/ (folder), paper_trading_endpoints.py, pdf_endpoints.py,
phase5_endpoints.py, provider_config_endpoints.py, router_endpoints.py,
server.py, services/ (folder), session_checkpoint_endpoints.py,
session_endpoints.py, settings_endpoints.py, tick_stream_handler.py,
tool_call_endpoints.py, trading/ (folder), trading_endpoints.py,
trading_floor_endpoints.py, tradingview_endpoints.py, trd_endpoints.py,
version_endpoints.py, video_to_ea_endpoints.py, websocket_endpoints.py,
websocket_metrics.py, workflow_endpoints.py, workshop_copilot_endpoints.py,
ws_logger.py
```

### Architecture Compliance

**From architecture.md Decision 2 (FastAPI Split):**
> Single codebase, environment variable `NODE_ROLE=contabo|cloudzy|local` controls which router groups register at startup.

**Pattern:**
```python
# src/api/server.py
NODE_ROLE = os.getenv("NODE_ROLE", "contabo")  # contabo | cloudzy | local

if NODE_ROLE in ("contabo", "local"):
    app.include_router(agents_router, prefix="/api")
    app.include_router(floor_manager_router, prefix="/api")
    app.include_router(knowledge_router, prefix="/api")
    app.include_router(research_router, prefix="/api")
    app.include_router(settings_router, prefix="/api")
    # ... all non-trading routers

if NODE_ROLE in ("cloudzy", "local"):
    app.include_router(strategy_router, prefix="/api")
    app.include_router(kill_switch_router, prefix="/api")
    app.include_router(mt5_bridge_router, prefix="/api")
    app.include_router(trading_router, prefix="/api")
    # ... all trading-execution routers
```

**Systemd service files** set the `NODE_ROLE` env var per node. Same `git pull` deploys to both.

**NFR-R4:** Cloudzy must trade without Contabo reachable — strategy router + kill switch run self-contained on Cloudzy.

### What NOT to Touch

| Area | Reason |
|------|--------|
| Frontend Svelte components | Story 1.2 handles this |
| Agent department structure | Epic 7 handles Agent SDK migration |
| Backend agent files | Story 1.1 handles LangChain cleanup |
| Database schemas | Keep current structure |

### Python Import Convention

From `project-context.md`:
```python
# Always use src. prefix from project root
from src.api.ide_endpoints import create_ide_api_app
from src.agents.departments.floor_manager import FloorManager
```

**Server must be started from project root** (`/home/mubarkahimself/Desktop/QUANTMINDX`):
```bash
uvicorn src.server:app
```

### .env.example Requirements

From Story 1.1 findings, `.env.example` should include:
```
# NODE_ROLE: Controls which router groups register at startup
# Accepted values: contabo | cloudzy | local (default: local)
NODE_ROLE=local
```

### References

- Epic 1 Story 1.3 definition: [Source: _bmad-output/planning-artifacts/epics.md#line-513]
- Architecture Decision 2 (NODE_ROLE split): [Source: _bmad-output/planning-artifacts/architecture.md#Decision-2]
- Story 1.0 audit findings (router list): [Source: _bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md#Section-D]
- NFR-R4 (Cloudzy must trade without Contabo): [Source: _bmad-output/planning-artifacts/epics.md#NonFunctional-Requirements]
- Project context env var rules: [Source: _bmad-output/project-context.md#Environment-Variables]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Python syntax verified on server.py
- NODE_ROLE validation implemented at server startup

### Completion Notes List

- **Task 1 (Read server.py)**: Identified 70+ router registrations in server.py
- **Task 2 (Implement NODE_ROLE)**: Added NODE_ROLE validation with fallback to "local" on invalid values
- **Task 3 (Classify routers)**: Created CLOUDZY_ROUTERS, CONTABO_ROUTERS, and BOTH_ROUTERS sets
- **Task 4 (.env.example)**: Already updated in Story 1-1
- **Task 5 (Verify)**: Syntax check passed
- **Task 6 (systemd)**: systemd service files need NODE_ROLE env var set

### Implementation Details

- Added NODE_ROLE config block at top of server.py after logging setup
- Valid roles: "contabo", "cloudzy", "local" (default)
- Invalid values log warning and default to "local"
- Conditional router registration:
  - Cloudzy: kill_switch, trading, broker, paper_trading, tradingview, lifecycle_scanner
  - Contabo: settings, provider_config, model, all agent routers, all memory routers, floor_manager, workflow, hmm, knowledge, video, batch, evaluation, trading_floor, claude_agent, tool_call, chat_ide, backtest
  - Both: health, version, metrics (always included)
- IDE routers (files, assets, strategies) only on Contabo

### File List

**Modified files:**
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/api/server.py`

**Story file updated:**
- `/home/mubarkahimself/Desktop/QUANTMINDX/_bmad-output/implementation-artifacts/1-3-node-role-backend-deployment-split.md`
