# Recent Codebase Issues - Resolution Plan

## Overview

This document outlines the current issues identified in the QuantMindX codebase and provides a plan to resolve them.

---

## Issue 1: UI Not Connected to Claude Code Native Backend

### Problem
The frontend still uses the old LangChain-based agents (`agentManager.ts`, `copilotAgent.ts`, `quantcodeAgent.ts`) instead of the new Claude Code native client (`claudeCodeAgent.ts`).

### Solution
See full plan in: `docs/UI_Claude_Code_Native_Integration_Plan.md`

### Quick Fix
Update UI components to use:
```typescript
import { streamToStore } from '$lib/agents/agentStreamStore';
const result = await streamToStore('copilot', messages, context);
```

---

## Issue 2: Backend v2 Router May Not Be Mounted

### Problem
The v2 agent API endpoints (`/api/v2/agents/*`) are defined in `src/api/claude_agent_endpoints.py` but may not be registered in the main FastAPI app.

### Files to Check
- `src/main.py` or `src/app.py`
- `src/api/__init__.py`

### Fix
Ensure the router is included:
```python
from src.api.claude_agent_endpoints import router as v2_router
app.include_router(v2_router)
```

---

## Issue 3: MCP Configuration Files Missing for Some Agents

### Problem
Some agent MCP configs may be missing or incomplete.

### Expected Files
```
config/mcp/
  analyst-mcp.json
  quantcode-mcp.json
  copilot-mcp.json
  pinescript-mcp.json
  router-mcp.json
  executor-mcp.json
```

### Fix
Create missing MCP configs. Example structure:
```json
{
  "mcpServers": {
    "mt5-compiler": {
      "command": "node",
      "args": ["path/to/mt5-compiler-server"]
    },
    "backtest-server": {
      "command": "python",
      "args": ["-m", "src.mcp.backtest_server"]
    }
  }
}
```

---

## Issue 4: Agent Hooks Module Not Implemented

### Problem
`claude_config.py` imports hooks from `src.agents.hooks` but this module may not exist or be incomplete.

### Expected Hooks
- `pre_analyst_hook`, `post_analyst_hook`
- `pre_quantcode_hook`, `post_quantcode_hook`
- `pre_copilot_hook`, `post_copilot_hook`

### Fix
Create `src/agents/hooks.py` with placeholder hooks if not implemented:
```python
async def pre_analyst_hook(task):
    return task

async def post_analyst_hook(task, result):
    return result

# Similar for other agents...
```

---

## Issue 5: Docker Build Issues (Contabo Deployment)

### Problems Fixed Previously
1. Missing `entrypoint.sh` - Copied from `docker/strategy-agent/`
2. `mcp-metatrader5-server` editable install failing - Commented out in `requirements.txt`
3. `prometheus-remote-write` requires Python 3.11+ - Commented out
4. Missing `fastapi` and `uvicorn` - Added to `requirements.txt`
5. Missing `scikit-learn` for HMM - Added to `requirements.txt`

### Outstanding
- MT5 initialization in entrypoint bypassed via `docker-compose.contabo.override.yml`

---

## Issue 6: Exploration Agent Failures

### Problem
Task agents (a725977, afca0dd) failed with: `classifyHandoffIfNeeded is not defined`

### Root Cause
Likely a bug in the agent spawning system or missing function definition.

### Fix
Investigate the agent framework code for the missing function.

---

## Implementation Priority

| Priority | Issue | Effort |
|----------|-------|--------|
| HIGH | Backend v2 router mounting | Low |
| HIGH | Agent hooks module | Medium |
| MEDIUM | MCP configs completion | Medium |
| MEDIUM | UI component updates | High |
| LOW | Exploration agent bug | Low |

---

## Verification Checklist

- [ ] `curl http://localhost:8000/api/v2/agents` returns agent list
- [ ] `src/agents/hooks.py` exists with all required hooks
- [ ] All MCP configs exist in `config/mcp/`
- [ ] UI Copilot chat uses `streamToStore()` or `streamAgent()`
- [ ] WebSocket streaming works for agent tasks

---

## Related Documentation

- `docs/UI_Claude_Code_Native_Integration_Plan.md` - Full UI migration plan
- `QuantMindX_Production_Deployment_Guide.md` - Deployment guide
- `workspaces/*/context/CLAUDE.md` - Agent system prompts
