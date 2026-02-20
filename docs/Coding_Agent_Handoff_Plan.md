# Coding Agent Handoff: UI Integration with Claude Code Native Backend

## Server Information

**Contabo VPS:**
- IP: `155.133.27.86`
- HMM API: `http://155.133.27.86:8001`
- Grafana: `http://155.133.27.86:3001`
- Prometheus: `http://155.133.27.86:9090`

**Cloudzy VPS:**
- Trading API: Port 8000
- MT5 Bridge: Port 5005

---

## Task 1: Replace AgentPanel with Claude Code Native Backend

**File:** `quantmind-ide/src/lib/components/agent-panel/AgentPanel.svelte`

**Current (line 22):**
```typescript
import { agentManager, activeStreams } from '../../agents';
```

**Replace with:**
```typescript
import {
  streamAgent,
  runAgent,
  agentStreamStore,
  type AgentMessage
} from '../../agents';
```

**Current (line 126):**
```typescript
const response = await agentManager.invoke(currentAgent, message, {
  context: contextManager.serializeContext(context),
  model,
  provider
});
```

**Replace handleSendMessage with:**
```typescript
async function handleSendMessage(event: CustomEvent<{
  message: string;
  model: string;
  provider: string;
  context: ChatContext;
}>) {
  const { message, model, provider, context } = event.detail;

  const state = get(chatStore);
  if (!state.activeChatId) {
    chatStore.createChat(state.activeAgent);
  }

  chatStore.addMessage({
    role: 'user',
    content: message
  });

  try {
    const messages: AgentMessage[] = [
      { role: 'user', content: message }
    ];

    const agentContext = {
      ...contextManager.serializeContext(context),
      model,
      provider
    };

    for await (const event of streamAgent(currentAgent, messages, agentContext)) {
      if (event.type === 'progress') {
        chatStore.updateStreamingStatus(true, event.data);
      } else if (event.type === 'tool_call') {
        chatStore.addToolCall(event.data);
      } else if (event.type === 'completed') {
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

**Update streaming indicator (line 187):**
```typescript
$: isStreaming = $agentStreamStore.isStreaming;
```

---

## Task 2: Remove Mock Data from UI Components

### 2.1 Sidebar.svelte

**Remove:** `getFallbackData()` function (lines 190-444)

**Update `loadData()` to show empty on error:**
```typescript
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
    // ... existing transform logic ...
  } catch (e: any) {
    error = e.message;
    treeData[view] = []; // Empty instead of fallback
  } finally {
    loading = false;
  }
}
```

### 2.2 MainContent.svelte

**Remove all mock arrays:**
- `mockNprdFiles` (line 150)
- `mockTrdFiles` (line 171)
- `mockEaFiles` (line 192)
- `mockBacktestFiles` (line 219)
- `mockAssets` (line 746)

### 2.3 BacktestDashboard.svelte

**Remove:** `mockResult` and mock data generation (lines 77-149)

### 2.4 NewsView.svelte

**Remove:** Mock news data (line 116) - connect to real API

### 2.5 DatabaseView.svelte

**Remove:** `mockRows` generation (line 755)

### 2.6 SharedAssetsView.svelte

**Remove:** `mockAsset` creation (lines 282-296)

### 2.7 MemoriesSettings.svelte

**Remove:** `mockMemories` array (line 22)

### 2.8 MainEditor.svelte

**Remove:** Demo content fallback (lines 57, 92, 106)

---

## Task 3: Configure GitHub EA Sync on Cloudzy VPS

**Add to `.env`:**
```bash
GITHUB_EA_REPO_URL=https://github.com/your-org/quantmind-eas
GITHUB_EA_BRANCH=main
GITHUB_ACCESS_TOKEN=ghp_xxxx
EA_LIBRARY_PATH=/data/ea-library
GITHUB_EA_SYNC_INTERVAL_HOURS=24
```

**Create directory:**
```bash
mkdir -p /data/ea-library
```

---

## Task 4: Verify Backend v2 Router

**File:** `src/main.py` or `src/app.py`

**Ensure this exists:**
```python
from src.api.claude_agent_endpoints import router as v2_router
app.include_router(v2_router)
```

---

## Verification Commands

```bash
# Contabo HMM API
curl http://155.133.27.86:8001/health

# Cloudzy Backend v2
curl http://localhost:8000/api/v2/agents

# GitHub EA Sync
curl http://localhost:8000/api/github/status
```

---

## Files to KEEP (Demo/Live Mode - NOT Mock)

These are for paper trading, NOT mock data:
- `src/api/demo_mode_endpoints.py`
- `quantmind-ide/src/lib/components/EAManagement.svelte`
- `quantmind-ide/src/lib/components/EnhancedPaperTradingPanel.svelte`
- `quantmind-ide/src/lib/components/ModeIndicator.svelte`

---

## Paper Trading Flow

```
Import EA → Demo Mode (Virtual) → Paper Trade (MT5 Demo) → Validate → Live
```

**.env for paper trading:**
```bash
DEFAULT_EA_MODE=demo
DEFAULT_VIRTUAL_BALANCE=10000.0
PAPER_TRADING_ENABLED=true
```
