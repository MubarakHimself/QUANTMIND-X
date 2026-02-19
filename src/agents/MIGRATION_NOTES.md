# Agent System Migration Notes

## Overview

This document describes the migration from LangGraph-based agents to Claude CLI-powered agents (v2 Agent Stack).

## What Changed

### Phase 1-3: Infrastructure and Core System
- Added Node.js 20.x and claude-cli to Docker image
- Created `src/agents/claude_config.py` - Agent configuration
- Created `src/agents/claude_orchestrator.py` - Subprocess orchestration
- Created `src/agents/hooks.py` - Pre/post execution hooks

### Phase 4-5: Tools and MCP
- Created bash tool scripts in `tools/` directory
- Created MCP configuration files in `config/mcp/` directory

### Phase 6: API Endpoints
- Created `src/api/claude_agent_endpoints.py` - New v2 API
- Updated `src/api/agent_management_endpoints.py` - Backward compatibility
- Registered v2 routes in `src/api/server.py`

### Phase 7: Frontend
- Created `quantmind-ide/src/lib/agents/claudeCodeAgent.ts`
- Created `quantmind-ide/src/lib/agents/agentStreamStore.ts`

### Phase 9: Cleanup
- Commented out LangChain/LangGraph dependencies in `requirements.txt`
- Removed LangChain dependencies from `quantmind-ide/package.json`

## Deprecated Files

The following files are **deprecated** and will be removed in a future version:

### Python Backend
- `src/agents/analyst.py` - Use Claude Orchestrator with agent_id='analyst'
- `src/agents/quantcode.py` - Use Claude Orchestrator with agent_id='quantcode'
- `src/agents/copilot.py` - Use Claude Orchestrator with agent_id='copilot'
- `src/agents/pinescript.py` - Use Claude Orchestrator with agent_id='pinescript'
- `src/agents/factory.py` - Replaced by `claude_orchestrator.py`
- `src/agents/compiled_agent.py` - No longer needed
- `src/agents/state.py` - No longer needed (state is managed per-task)

### TypeScript Frontend
- `quantmind-ide/src/lib/agents/langchainAgent.ts` - Use `claudeCodeAgent.ts`
- `quantmind-ide/src/lib/agents/agentManager.ts` - Use `claudeCodeAgent.ts`
- `quantmind-ide/src/lib/agents/analystAgent.ts` - Use `claudeCodeAgent.ts`
- `quantmind-ide/src/lib/agents/quantcodeAgent.ts` - Use `claudeCodeAgent.ts`
- `quantmind-ide/src/lib/agents/copilotAgent.ts` - Use `claudeCodeAgent.ts`

## Migration Guide

### Python API

**Old (LangGraph):**
```python
from src.agents.factory import create_agent_from_yaml
agent = create_agent_from_yaml("config/agents/analyst.yaml")
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

**New (Claude CLI):**
```python
from src.agents.claude_orchestrator import get_orchestrator

orchestrator = get_orchestrator()
task_id = await orchestrator.submit_task(
    agent_id="analyst",
    messages=[{"role": "user", "content": "..."}],
    context={}
)
result = await orchestrator.get_result("analyst", task_id)
```

### TypeScript Frontend

**Old (LangChain):**
```typescript
import { agentManager } from '$lib/agents';
const response = await agentManager.invoke('analyst', messages);
```

**New (Claude CLI):**
```typescript
import { runAgent, streamAgent } from '$lib/agents';

// Blocking call
const result = await runAgent('analyst', messages, context);

// Streaming
for await (const event of streamAgent('analyst', messages, context)) {
  console.log(event);
}
```

## API Endpoints

### New v2 Endpoints
- `GET /api/v2/agents` - List available agents
- `GET /api/v2/agents/{agent_id}/config` - Get agent config
- `POST /api/v2/agents/{agent_id}/run` - Run an agent
- `GET /api/v2/agents/{agent_id}/status/{task_id}` - Get task status
- `DELETE /api/v2/agents/{agent_id}/tasks/{task_id}` - Cancel task
- `WS /api/v2/agents/{agent_id}/stream/{task_id}` - Stream results

### Legacy Endpoints (Deprecated)
- `POST /api/agents/{agent_id}/invoke` - Use `/api/v2/agents/{agent_id}/run`

## Environment Variables

New variables required:
- `ANTHROPIC_API_KEY` - Claude API key
- `CLAUDE_MEM_PORT` - Memory service port (default: 37777)

## Workspace Structure

```
workspaces/
├── analyst/
│   ├── tasks/
│   ├── results/
│   ├── context/
│   │   └── CLAUDE.md
│   └── scratch/
├── quantcode/
├── copilot/
├── pinescript/
├── router/
└── executor/
```

## Testing

Run the test suite to verify the migration:
```bash
pytest tests/agents/test_claude_orchestrator.py -v