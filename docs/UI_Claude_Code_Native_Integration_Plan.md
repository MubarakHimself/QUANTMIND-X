# Plan: UI Integration with Claude Code Native Backend

## Server Information

**Contabo VPS:**
- IP: `155.133.27.86`
- HMM API: `http://155.133.27.86:8001`
- Grafana: `http://155.133.27.86:3001`
- Prometheus: `http://155.133.27.86:9090`

**Cloudzy VPS:**
- Trading API: Port 8000
- MT5 Bridge: Port 5005

## Context

The user wants to:
1. Verify Contabo server is ready per deployment guide
2. Replace agent sidebar in UI to use Claude Code native backend
3. Fix EA importing via GitHub in the UI (implementation exists but may not work)
4. Connect Cloudzy VPS to the GitHub EA sync service

## Exploration Findings

### 1. Contabo Deployment Status ✅ READY

**What's Working:**
- Repository cloned to `/opt/quantmindx`
- HMM Inference API running on port 8001
- Storage directories created (`/data/hmm`, `/data/cold_storage`)
- Self-hosted Grafana (port 3001) and Prometheus (port 9090)
- Docker services running with `docker-compose.contabo.override.yml`

**Verification Command:**
```bash
curl http://155.133.27.86:8001/health
```

### 2. UI Agent Panel - Current Implementation

**Key File:** `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte`

**Current invocation (line 126):**
```typescript
const response = await agentManager.invoke(currentAgent, message, {
  context: contextManager.serializeContext(context),
  model,
  provider
});
```

**Problem:** Uses old LangChain `agentManager` instead of new Claude Code native client.

**Available Agents (line 36-40):**
- `copilot` - General assistant
- `quantcode` - Code & debug
- `analyst` - Strategy analysis

**Streaming Status (line 187):**
```typescript
$: isStreaming = $activeStreams.length > 0;
```
Already has streaming support via `activeStreams` store.

### 3. EA GitHub Import - Already Implemented ✅

**Key Files:**
- `src/integrations/github_ea_sync.py` - Core sync logic
- `src/integrations/github_ea_scheduler.py` - APScheduler for scheduled syncs
- `src/api/github_endpoints.py` - REST API endpoints
- `quantmind-ide/src/lib/components/GitHubEASync.svelte` - UI component

**API Endpoints Available:**
- `GET /api/github/status` - Sync status
- `POST /api/github/sync` - Trigger manual sync
- `GET /api/github/eas` - List available EAs
- `POST /api/github/import` - Import selected EAs

**Required Environment Variables:**
```bash
GITHUB_EA_REPO_URL=https://github.com/your-org/quantmind-eas
GITHUB_EA_BRANCH=main
GITHUB_ACCESS_TOKEN=ghp_xxxx
EA_LIBRARY_PATH=/data/ea-library
```

**What's Missing:**
- Webhook integration for automatic sync triggers
- Repository management UI (add/remove repos)
- VPS connection configuration

### 4. Sidebar Clarification

**Two sidebars exist:**
1. `Sidebar.svelte` - Tree view for knowledge/EA/backtest navigation (NOT agent-related)
2. `ChatListSidebar.svelte` - Chat history sidebar (part of agent panel)

The "agent sidebar" refers to the `AgentPanel.svelte` component with its agent tabs.

---

## Implementation Plan

### Step 1: Update AgentPanel to Use Claude Code Native Backend

**File:** `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte`

**Changes Required:**

1. **Update imports (line 22):**
```typescript
// OLD
import { agentManager, activeStreams } from '../../agents';

// NEW
import {
  streamAgent,
  runAgent,
  agentStreamStore,
  type AgentMessage
} from '../../agents';
```

2. **Replace handleSendMessage function (lines 110-154):**
```typescript
async function handleSendMessage(event: CustomEvent<{
  message: string;
  model: string;
  provider: string;
  context: ChatContext;
}>) {
  const { message, model, provider, context } = event.detail;

  // Ensure a chat exists before sending
  const state = get(chatStore);
  if (!state.activeChatId) {
    chatStore.createChat(state.activeAgent);
  }

  // Add user message immediately
  chatStore.addMessage({
    role: 'user',
    content: message
  });

  try {
    // Build messages array for Claude Code API
    const messages: AgentMessage[] = [
      { role: 'user', content: message }
    ];

    // Build context for Claude Code
    const agentContext = {
      ...contextManager.serializeContext(context),
      model,
      provider
    };

    // Stream response using Claude Code native client
    for await (const event of streamAgent(currentAgent, messages, agentContext)) {
      if (event.type === 'progress') {
        // Update streaming indicator
        chatStore.updateStreamingStatus(true, event.data);
      } else if (event.type === 'tool_call') {
        // Show tool call in chat
        chatStore.addToolCall(event.data);
      } else if (event.type === 'completed') {
        // Add final response
        chatStore.addMessage({
          role: 'assistant',
          content: event.output,
          model,
          metadata: {
            provider: 'anthropic',
            taskId: event.task_id
          }
        });
      }
    }
  } catch (error) {
    console.error('Claude Code agent failed:', error);
    chatStore.addMessage({
      role: 'assistant',
      content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
    });
  }
}
```

3. **Update streaming indicator (line 187):**
```typescript
$: isStreaming = $agentStreamStore.isStreaming;
```

### Step 2: Ensure Backend v2 Router is Mounted

**File:** `src/main.py` or `src/app.py`

**Verify this line exists:**
```python
from src.api.claude_agent_endpoints import router as v2_router
app.include_router(v2_router)
```

**If missing, add it to the FastAPI app initialization.**

### Step 3: Configure GitHub EA Sync for Cloudzy VPS

**File:** `.env` on Cloudzy VPS

**Add required variables:**
```bash
GITHUB_EA_REPO_URL=https://github.com/your-org/quantmind-eas
GITHUB_EA_BRANCH=main
GITHUB_ACCESS_TOKEN=ghp_xxxx
EA_LIBRARY_PATH=/data/ea-library
GITHUB_EA_SYNC_INTERVAL_HOURS=24
```

**Create EA library directory:**
```bash
mkdir -p /data/ea-library
```

### Step 4: Test End-to-End Flow

1. **Backend v2 API:**
```bash
curl http://localhost:8000/api/v2/agents
# Expected: {"agents": ["analyst", "quantcode", "copilot", ...], "count": 6}
```

2. **GitHub EA Sync:**
```bash
curl http://localhost:8000/api/github/status
```

3. **UI Agent Chat:**
- Open QuantMind IDE
- Send message to Copilot
- Verify streaming response appears

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte` | Replace agentManager with streamAgent |
| `quantmind-ide/src/lib/components/Sidebar.svelte` | Remove getFallbackData(), API-only |
| `quantmind-ide/src/lib/components/MainContent.svelte` | Remove mock arrays |
| `quantmind-ide/src/lib/components/BacktestDashboard.svelte` | Remove mockResult |
| `quantmind-ide/src/lib/components/NewsView.svelte` | Connect to real API |
| `quantmind-ide/src/lib/components/DatabaseView.svelte` | Remove mockRows generation |
| `quantmind-ide/src/lib/components/SharedAssetsView.svelte` | Remove mockAsset creation |
| `quantmind-ide/src/lib/components/agent-panel/settings/MemoriesSettings.svelte` | Remove mockMemories |
| `src/main.py` | Verify v2 router mounted |
| `.env` (Cloudzy) | Add GitHub EA sync config |

## Files Already Implemented (No Changes)

| File | Purpose |
|------|---------|
| `src/api/claude_agent_endpoints.py` | v2 REST/WebSocket API |
| `src/agents/claude_orchestrator.py` | Claude CLI subprocess management |
| `quantmind-ide/src/lib/agents/claudeCodeAgent.ts` | Frontend client |
| `src/integrations/github_ea_sync.py` | EA sync logic |
| `src/api/github_endpoints.py` | GitHub REST API |

---

## Step 5: Remove Mock Data for Production Readiness

The user wants to remove mock/fallback data to prepare for live paper trading with imported EAs.

### Files with Mock Data to Remove

| File | Mock Data | Action |
|------|-----------|--------|
| `Sidebar.svelte` | `getFallbackData()` function | Remove, make API-only |
| `MainContent.svelte` | `mockNprdFiles`, `mockTrdFiles`, `mockEaFiles`, `mockBacktestFiles`, `mockAssets` | Remove all mock arrays |
| `MainEditor.svelte` | Demo content fallback | Remove demo content |
| `NewsView.svelte` | Mock news data | Connect to real API |
| `BacktestDashboard.svelte` | `mockResult` | Remove, use real results only |
| `DatabaseView.svelte` | `mockRows` generation | Remove mock data generation |
| `SharedAssetsView.svelte` | `mockAsset` creation | Remove mock creation |
| `MemoriesSettings.svelte` | `mockMemories` | Remove, use real memory API |

### Files to KEEP (Demo/Live Trading Mode - NOT Mock Data)

These are for paper trading functionality, not mock data:
- `demo_mode_endpoints.py` - Demo/live EA mode management
- `EAManagement.svelte` - Demo/live filtering
- `EnhancedPaperTradingPanel.svelte` - MT5 demo accounts
- `ModeIndicator.svelte` - Demo/live status indicator

### Implementation for Mock Data Removal

**1. Sidebar.svelte - Remove getFallbackData():**
```typescript
// REMOVE the entire getFallbackData() function (lines 190-444)
// UPDATE loadData() to only use API data, show empty state on error
async function loadData(view: string) {
  const config = viewConfig[view];
  if (!config || !config.endpoint) {
    treeData[view] = []; // Empty instead of fallback
    return;
  }

  loading = true;
  try {
    const response = await fetch(`${API_BASE}${config.endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // ... transform data ...
  } catch (e: any) {
    error = e.message;
    treeData[view] = []; // Empty instead of fallback
  } finally {
    loading = false;
  }
}
```

**2. MainContent.svelte - Remove mock arrays:**
```typescript
// DELETE these (lines 150-227):
// let mockNprdFiles = [...]
// let mockTrdFiles = [...]
// let mockEaFiles = [...]
// let mockBacktestFiles = [...]
// const mockAssets: Record<...>

// Update templates to use API data only
```

**3. BacktestDashboard.svelte - Remove mockResult:**
```typescript
// DELETE mockResult and mock data generation (lines 77-149)
// Use real API results only
```

### Paper Trading Optimization

For importing EAs that will paper trade:

1. **EA Import Flow:**
   - Import EA via GitHub sync
   - Auto-register as `demo` mode
   - Create virtual account with default balance
   - Deploy to MT5 demo account

2. **Configuration:**
   ```bash
   # .env for paper trading
   DEFAULT_EA_MODE=demo
   DEFAULT_VIRTUAL_BALANCE=10000.0
   PAPER_TRADING_ENABLED=true
   MT5_DEMO_SERVER=default
   ```

3. **EA Promotion Path:**
   ```
   Import → Demo (Virtual) → Paper Trade (MT5 Demo) → Validate → Live
   ```

---

## Verification

1. **Contabo HMM API:**
   ```bash
   curl http://155.133.27.86:8001/health
   ```

2. **Cloudzy Backend v2:**
   ```bash
   curl http://localhost:8000/api/v2/agents
   ```

3. **GitHub EA Sync:**
   ```bash
   curl -X POST http://localhost:8000/api/github/sync
   ```

4. **UI Streaming:**
   - Open Copilot chat in QuantMind IDE
   - Send a message
   - Verify streaming output appears in real-time

---

## Tasks Created

| # | Task | Status |
|---|------|--------|
| 7 | Update deployment guide with Claude Code native changes | pending |
| 8 | Replace agentManager with Claude Code native in AgentPanel | pending |
| 9 | Remove mock data from UI components for production | pending |
| 10 | Configure GitHub EA Sync on Cloudzy VPS | pending |
