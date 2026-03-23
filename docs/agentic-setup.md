# QUANTMINDX — Agentic Setup & Migration Status

**Generated:** 2026-03-11 | **Status:** ⚠️ Active Migration — Multiple Deprecated Systems

---

## ⚠️ Critical Situation Summary

The agentic system has **three overlapping paradigms** that need consolidation. The diagnosis (see `docs/plans/2026-03-08-agent-architecture-diagnosis.md`) identifies multiple deprecated layers still in active use, causing routing bugs and maintenance overhead.

**TL;DR:**
- The **Department System (Floor Manager)** is the canonical target architecture
- Legacy LangGraph/LangChain `BaseAgent`, named agents (`copilot.py`, `analyst.py`, `quantcode.py`), and factory runtime are deprecated but still active
- The Workshop UI has a **confirmed bug**: Floor Manager tab routes to the wrong API endpoint (`/api/chat/send` instead of `/api/floor-manager/chat`)
- Migration plan exists at `docs/plans/2026-03-08-agent-architecture-migration-map.md`

---

## 1. Three Agent Paradigms (Current State)

### Paradigm A: Legacy Named Agents (⚠️ ACTIVE but deprecated)

| Item | Detail |
|------|--------|
| **Files** | `src/agents/core/base_agent.py`, `src/agents/registry.py` |
| **Runtime** | LangGraph `create_react_agent` + LangChain wrappers |
| **Agents** | `copilot`, `analyst`, `quantcode`, `pinescript`, `router`, `executor` |
| **API** | `/api/chat/*` → `chat_service.py` → routes by agent type |
| **Status** | ⚠️ Still imported and used; hooks removed; registry deprecated |
| **Issue** | `base_agent.py` still imports `langchain_openai`, `langgraph`, `langmem` which may not be installed |

`registry.py` opens with:
```python
logger.warning("registry.py is deprecated. Use /api/floor-manager endpoints instead.")
```

### Paradigm B: Department System (✅ CANONICAL — should be the only one)

| Item | Detail |
|------|--------|
| **Orchestrator** | `src/agents/departments/floor_manager.py` (`FloorManager` singleton) |
| **Department Heads** | research, development, trading, risk, portfolio (+ legacy aliases: analysis, execution) |
| **Model Tiers** | Floor Manager → Opus; Department Heads → Sonnet; Workers → Haiku |
| **Communication** | SQLite mail service (`department_mail.py`) — async message queue |
| **API** | `/api/floor-manager/*` |
| **Worker Spawning** | `src/agents/subagent/spawner.py` via Anthropic Claude API |
| **Status** | ✅ Active and clean |

**Department routing** uses keyword scoring:
```
RESEARCH:    analyze, market, sentiment, signal, pattern, video, trading idea
DEVELOPMENT: ea, expert advisor, mql5, pinescript, code, implement
TRADING:     execute, order, buy, sell, trade, route, broker
RISK:        risk, position size, drawdown, var, limit, stop loss
PORTFOLIO:   portfolio, allocation, rebalance, performance
```

### Paradigm C: Factory/Container Runtime (❌ DEPRECATED)

| Item | Detail |
|------|--------|
| **Files** | `factory.py`, `di_container.py`, `compiled_agent.py` |
| **Status** | Deprecated — `registry.py` `CompiledAgent` class is a stub (`pass`) |

---

## 2. Workspace-Based Agent Config (claude_config.py)

Six agents are configured via `src/agents/claude_config.py` with workspace paths (`workspaces/`):

| Agent | Timeout | Purpose |
|-------|---------|---------|
| `analyst` | 600s | Market analysis, video ingest |
| `quantcode` | 900s | Strategy code generation + backtest |
| `copilot` | 300s | Orchestration |
| `pinescript` | 600s | Pine Script generation |
| `router` | 60s | Lightweight routing |
| `executor` | 120s | Execution tasks |

**Note:** The `workspaces/` directory and `CLAUDE.md` context files for these agents were deleted (visible in git status as `D workspaces/*/`). The workspace-based agents may be non-functional.

---

## 3. Confirmed Bug: Workshop Floor Manager Routing

**Location:** `quantmind-ide/src/lib/components/TradingFloorPanel.svelte`

**Problem:** The Workshop Floor Manager UI sends requests to `/api/chat/send` (legacy endpoint) instead of `/api/floor-manager/chat`.

```
BROKEN (current):
  Workshop Floor Manager Tab
      → POST /api/chat/send {agent: "floor_manager"}
      → chat_service.py: determine_target_agent()
      → Falls back to "copilot" (WRONG!)

CORRECT (should be):
  Workshop Floor Manager Tab
      → POST /api/floor-manager/chat
      → floor_manager_endpoints.py
      → FloorManager.process()
```

**Fix:** Change the API call endpoint in the UI component. See `docs/plans/2026-03-08-agent-architecture-migration-map.md` — Phase 1, Task 1.

---

## 4. Hook Systems (3 Conflicting — Needs Consolidation)

| Hook System | File | Status |
|-------------|------|--------|
| Legacy hooks | `src/agents/hooks.py` | ⚠️ Active |
| Core hook manager | `src/agents/core/hooks.py` | ⚠️ Active |
| Department hooks | `src/agents/hooks/department_hooks.py` | ✅ KEEP — canonical |

---

## 5. Memory Systems (6+ Overlapping)

| System | Location | Status |
|--------|----------|--------|
| `AgentMemory` | `src/agents/memory/agent_memory.py` | ✅ Keep (file-based storage) |
| `VectorMemory` | `src/agents/memory/vector_memory.py` | ✅ Keep (semantic search) |
| `DepartmentMemoryManager` | `src/agents/departments/memory_manager.py` | ✅ CANONICAL |
| `FloorManagerMemoryAccess` | `src/agents/departments/memory_access.py` | ✅ Keep (cross-dept read) |
| `UnifiedMemoryFacade` | `src/agents/memory/unified_memory_facade.py` | ✅ CANONICAL API |
| Graph Memory | `src/memory/graph/` | ✅ Keep (graph-based) |
| Legacy per-agent namespaces | various `copilot.py` etc. | ❌ Deprecate |

**Graph Memory System** (`src/memory/graph/`):
- `store.py` — DuckDB-backed graph storage
- `operations.py` — CRUD operations
- `facade.py` — public API
- `compaction.py` — memory pruning
- `migration.py` — schema migrations

---

## 6. Tool Systems (2 Conflicting)

| System | Location | Status |
|--------|----------|--------|
| Legacy `ToolRegistry` | `src/agents/tool_registry.py` | ❌ Deprecate |
| Department `ToolRegistry` | `src/agents/departments/tool_registry.py` | ✅ CANONICAL |
| `ToolAccessController` | `src/agents/departments/tool_access.py` | ✅ Keep |

---

## 7. Deprecated Skills (Old Agentic Setup)

The `docs/skills/` directory contains 12 skill definition markdown files from the old workspace-based agent system:

**Data Skills:** `clean_data_anomalies`, `fetch_historical_data`, `fetch_live_tick`, `resample_timeframe`
**System Skills:** `log_trade_event`, `read_mt5_data`, `send_alert_notification`, `write_indicator_buffer`
**Trading Skills:** `calculate_macd`, `calculate_position_size`, `calculate_rsi`, `detect_support_resistance`, `validate_stop_loss`

These skills were designed for the `ClaudeAgentConfig` workspace-based system. Their workspace directories (`workspaces/analyst/`, etc.) have been removed from git. **These skill files are effectively orphaned documentation.** The active skill system is in `src/agents/skills/`.

---

## 8. Active Agent Tools

The following tools are **currently active** (loaded by department agents):

| Category | File | Tools |
|----------|------|-------|
| Backtest | `src/agents/tools/backtest_tools.py` | Run backtests |
| Broker | `src/agents/tools/broker_tools.py` | Broker operations |
| EA Lifecycle | `src/agents/tools/ea_lifecycle.py` | EA deploy/stop |
| Knowledge | `src/agents/tools/knowledge/` | PageIndex, PDF, strategy search |
| MCP Tools | `src/agents/tools/mcp/` | MT5 compiler, Context7, backtest |
| Memory | `src/agents/tools/memory_tools.py` | Store/recall memory |
| MQL5 | `src/agents/tools/mql5_tools.py` | MQL5 generation/compile |
| PineScript | `src/agents/tools/pinescript_tools.py` | Pine Script generation |
| Risk | `src/agents/tools/risk_tools.py` | Position sizing, VaR |
| Trading | `src/agents/tools/trading_tools.py` | Trade proposals |
| Strategy-YT | `src/agents/tools/strategies_yt/` | Video strategy extraction |

---

## 9. Agent Streaming

`src/agents/streaming/agent_stream.py` — provides Server-Sent Events (SSE) streaming for agent responses to the frontend.

---

## 10. Migration Roadmap (From Plans)

Based on `docs/plans/2026-03-08-agent-architecture-migration-map.md`:

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Fix Workshop Floor Manager bug | ⏳ Pending | Route UI to `/api/floor-manager/chat` |
| Phase 2: New Workshop Copilot Service | ⏳ Pending | `src/api/services/workshop_copilot_service.py` |
| Phase 3: Consolidate workflow systems | ⏳ Pending | Merge 3 workflow orchestrators into 2 |
| Phase 4: Consolidate hook systems | ⏳ Pending | Single canonical hook system |
| Phase 5: Remove legacy endpoints | ⏳ Pending | Deprecate `/api/chat/*`, `/api/agent-management/*` |

---

## 11. Subagent Spawner

`src/agents/subagent/spawner.py` — spawns Claude-powered worker agents via Anthropic SDK:
- `CheckpointManager` — tracks long-running agent state
- `HeartbeatManager` — health monitoring
- `ProgressTracker` — ETA calculation
- Uses pool-key to avoid duplicate spawns

Many SDK config helpers are stubbed with "Deprecated — use department system instead" comments.

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `src/agents/departments/floor_manager.py` | ✅ Main orchestrator |
| `src/agents/departments/types.py` | Department configs + personalities |
| `src/agents/departments/department_mail.py` | SQLite async message bus |
| `src/agents/departments/heads/` | 5 Department Head implementations |
| `src/agents/departments/subagents/` | Worker agent implementations |
| `src/agents/claude_config.py` | Workspace agent configs (may be orphaned) |
| `src/agents/core/base_agent.py` | ❌ Legacy LangGraph base |
| `src/agents/registry.py` | ❌ Deprecated registry |
| `src/api/floor_manager_endpoints.py` | ✅ Canonical floor manager API |
| `src/api/chat_endpoints.py` | ⚠️ Legacy chat API (deprecate) |
