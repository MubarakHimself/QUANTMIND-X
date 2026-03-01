# QuantMindX Session Notes - 2026-03-01

> **Last Updated:** 2026-03-01
> **Branch:** feature/department-agent-framework

---

## Session Context

Session was reset. Rebuilding QuantMindX system from scratch.

---

## Decisions Made

### 1. Model Configuration (Backend)

| Setting | Value |
|---------|-------|
| Base | Direct Anthropic API (`https://api.anthropic.com/v1`) |
| Custom Providers | GLM (Zhipu), Minimax |
| Keep OpenRouter | For custom providers only |

**Minimax Details:**
- API Key: Get from https://platform.minimax.io/user-center/payment/coding-plan
- Models: M2.5, M2.5-highspeed, M2.1, M2
- Endpoint: Anthropic-compatible format

### 2. Agent Communication

**Status:** Broken - endpoints not registered in server.py

**Fix Required:**
- Register `floor_manager_router` from `floor_manager_endpoints.py`
- Register `chat_router` from `chat_endpoints.py`

**Mail System:** Keep as-is (SQLite-based)

### 3. Department Structure (Option B - 5 Departments)

| Department | Role | Tools (WRITE) |
|------------|------|---------------|
| Research | Analysis + Strategy dev | knowledge_tools, backtest_tools, memory_tools, mail |
| Development | EA/Bot building (Python, PineScript, MQL5) | mql5_tools, pinescript_tools, backtest_tools, knowledge_tools, memory_tools, mail |
| Trading | Order execution (READ-ONLY) | broker_tools (READ), memory_tools, mail |
| Risk | Risk management (READ-ONLY) | risk_tools (READ), memory_tools, mail |
| Portfolio | Allocation | memory_tools, mail |

**ALL departments get:** memory_tools, knowledge_tools, mail

### 4. Memory System

**Status:** Keep existing OpenClaw-inspired memory
- Location: `src/memory/memory_manager.py`
- SQLite-based
- No embeddings (due to low compute)
- Can add later if needed

**REMOVED:** LangMem (not compatible with Claude Agent SDK)

### 5. NotebookLM MCP

**Status:** Adding via notebooklm-mcp-cli
- Use for knowledge management
- Can create notebooks, add sources, query AI
- Integrates via MCP

---

## Items from Original Prompt (Deferred)

| Item | Status |
|------|--------|
| UI fixes (terminal, overflows) | Separate plan |
| MCP server config via UI | Separate plan |
| NPRD removal / integration | Separate plan |
| Paper trading / MT5 | Separate plan |
| Knowledge hub sync | Separate plan |

---

## Files Reference

### Critical Backend Files
| File | Purpose |
|------|---------|
| `src/agents/llm_provider.py` | LLM provider config |
| `src/agents/departments/types.py` | Department configs |
| `src/agents/departments/tool_access.py` | Tool permissions |
| `src/api/server.py` | Register routers |
| `src/memory/memory_manager.py` | Memory system |
| `src/agents/tools/memory_tools.py` | Memory tools for agents |

### Frontend Files
| File | Purpose |
|------|---------|
| `quantmind-ide/src/lib/components/AgentModelSelector.svelte` | Model dropdown |
| `quantmind-ide/src/lib/components/agent-panel/settings/APIKeysSettings.svelte` | API keys |

---

## Implementation Plan

See: `docs/plans/2026-03-01-backend-refactor.md`

### Backend Tasks (Tasks 1-6)
1. Remove LangMem
2. Add NotebookLM MCP
3. Configure GLM + Minimax
4. Refactor department structure
5. Fix agent communication endpoints
6. Update memory tools

### Frontend Tasks (Tasks 7-11)
7. UI/UX Improvements
   - Terminal view - don't open on start
   - Layout padding
   - Demo/Stop buttons
8. MCP Configuration via UI
9. NPRD Integration
10. Paper Trading / MT5
11. Knowledge Hub Enhancements

---

## Key File Locations (from exploration)

| Area | Key Files |
|------|----------|
| Workshop/Terminal | `WorkshopView.svelte`, `LogViewer.svelte` |
| MCP | `MCPSettings.svelte`, `mcp_endpoints.py`, `config/mcp/analyst-mcp.json` |
| NPRD | `src/nprd/`, `strategies-yt/` |
| Paper Trading | `PaperTradingPanel.svelte`, `mcp-metatrader5-server/` |
| Knowledge Hub | `KnowledgeHub.svelte`, `mcp-servers/quantmindx-kb/` |
| Scraped Articles | `data/scraped_articles/` |

---

## Agent Teams Capability

**Can use Claude Agent SDK for implementation:**
- Native `AgentDefinition` for multi-agent teams
- Subagent spawning via existing `src/agents/subagent/spawner.py`
- Can invoke specific agents by name from within a query

**Not yet in plan** - can be added for faster implementation

---

## Next Steps

After this plan is complete, we can work on:
- UI improvements
- NPRD integration
- Paper trading setup
- Agent teams implementation
