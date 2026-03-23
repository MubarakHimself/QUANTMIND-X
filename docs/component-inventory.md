# QUANTMINDX — UI Component Inventory (quantmind-ide)

**Generated:** 2026-03-11 | **Framework:** SvelteKit 4 + Tauri 2 + TypeScript

---

## Architecture Overview

The IDE is a Tauri-wrapped SvelteKit 4 desktop application. It communicates with the FastAPI backend at `http://localhost:8000`. The UI is organized around a VS Code-inspired layout with an activity bar, panels, and modals.

**Entry point:** `quantmind-ide/src/app.html` → `src/routes/+page.svelte`
**Config:** `quantmind-ide/svelte.config.js` (static adapter + Tauri)

---

## Layout Components

| Component | Purpose |
|-----------|---------|
| `ActivityBar.svelte` | Left-side icon bar for switching between views |
| `BottomPanel.svelte` | Bottom status/terminal panel |
| `Breadcrumbs.svelte` | Navigation breadcrumbs |
| `StatusBand.svelte` | Status bar at bottom of IDE |

---

## Main Views / Pages

| Component | Route/View | Purpose |
|-----------|-----------|---------|
| `WorkshopView.svelte` | Workshop | AI agent workshop (main view) |
| `TradingFloorPanel.svelte` | Workshop → Trading Floor | Department-based trading floor |
| `BotsPage.svelte` | Bots | Bot management view |
| `DatabasePage.svelte` | Database | Database browser |
| `AlgoForgePanel.svelte` | AlgoForge | Algorithm development workspace |

---

## Agent Panel Components (`components/agent-panel/`)

The Agent Panel is the primary chat interface for interacting with AI agents.

| Component | Purpose |
|-----------|---------|
| `AgentPanel.svelte` | Main agent panel container |
| `AgentHeader.svelte` | Agent selection header + model selector |
| `ChatListSidebar.svelte` | List of past chat sessions |
| `ChatItem.svelte` | Single chat session row |
| `MessagesArea.svelte` | Chat message display area |
| `MessageBubble.svelte` | Individual message bubble (supports markdown) |
| `InputArea.svelte` | Message input + send controls |
| `ContextBar.svelte` | Context file/tool attachment bar |
| `ContextPicker.svelte` | Context selector modal |
| `ContextTag.svelte` | Individual context tag pill |
| `PineScriptPanel.svelte` | Pine Script code view within chat |
| `SkillMentionPalette.svelte` | @skill autocomplete palette |
| `SlashCommandPalette.svelte` | /command autocomplete palette |

### Agent Panel Settings (`agent-panel/settings/`)

| Component | Purpose |
|-----------|---------|
| `SettingsPanel.svelte` | Settings container |
| `GeneralSettings.svelte` | General preferences |
| `APIKeysSettings.svelte` | API key management |
| `MCPSettings.svelte` | MCP server configuration |
| `MemoriesSettings.svelte` | Agent memory viewer |
| `PermissionsSettings.svelte` | Tool permissions |
| `RulesSettings.svelte` | Agent rule management |
| `SkillsSettings.svelte` | Skills management |
| `WorkflowsSettings.svelte` | Workflow management |

---

## Trading Floor Components

| Component | Purpose |
|-----------|---------|
| `TradingFloorPanel.svelte` | ⚠️ Main floor panel — **has Floor Manager routing bug** |
| `CopilotPanel.svelte` | Copilot chat interface (legacy vs new path) |
| `DepartmentChatPanel.svelte` | Department chat interface |
| `DepartmentMailPanel.svelte` | Department mail viewer |
| `AgentDashboard.svelte` | Agent activity overview |
| `AgentActivityFeed.svelte` | Live agent activity feed |
| `AgentMetricsDashboard.svelte` | Agent performance metrics |
| `AgentModelSelector.svelte` | Model tier selector (Opus/Sonnet/Haiku) |
| `AgentQueuePanel.svelte` | Agent task queue viewer |
| `ApprovalPanel.svelte` | Trade approval gate UI |
| `AuctionQueue.svelte` | Strategy auction queue viewer |

---

## Bot Management Components

| Component | Purpose |
|-----------|---------|
| `BotsPage.svelte` | Bot list and management |
| `BotLifecyclePanel.svelte` | Bot lifecycle control (promote/demote/quarantine) |
| `BotLimitIndicator.svelte` | Displays current bot count vs. 50-bot limit |
| `BrokerConnectModal.svelte` | Add/edit broker connection |
| `BrokerManagement.svelte` | Broker accounts management |
| `BatchProcessingPanel.svelte` | Batch bot operations |
| `AlertsPanel.svelte` | Risk alerts and notifications |

---

## Trading & Backtesting Components

| Component | Purpose |
|-----------|---------|
| `BacktestDashboard.svelte` | Backtest overview dashboard |
| `BacktestRunner.svelte` | Run new backtests |
| `BacktestResultsView.svelte` | View backtest results |
| `CorrelationsTab.svelte` | Cross-strategy correlation analysis |

### Chart Components (`components/charts/`)

| Component | Purpose |
|-----------|---------|
| `FanChart.svelte` | Fan chart for probability cone visualization |
| `MonteCarloChart.svelte` | Monte Carlo simulation results |
| `MonteCarloHeatmap.svelte` | Heat map of MC simulation outcomes |
| `ProbabilityDistribution.svelte` | Probability distribution curve |
| `ResourceUsageChart.svelte` | System resource usage chart |
| `TickStreamChart.svelte` | Real-time tick stream chart |

---

## Strategy & Code Components

| Component | Purpose |
|-----------|---------|
| `CodeEditor.svelte` | Monaco-based code editor |
| `AssetsView.svelte` | Strategy assets browser |
| `ArticleViewer.svelte` | Knowledge base article reader |

---

## Settings Components (`components/settings/`)

| Component | Purpose |
|-----------|---------|
| `ProvidersPanel.svelte` | LLM provider configuration (updated: connects to provider config API) |
| `GeneralSettings.svelte` | General IDE settings |

---

## System Components

| Component | Purpose |
|-----------|---------|
| `AuthStatus.svelte` | Authentication state indicator |
| `CronJobManager.svelte` | Cron job management UI |
| `DatabaseHeader.svelte` | Database view header |
| `DatabaseStats.svelte` | Database statistics display |

---

## Frontend API Layer (`src/lib/api/`, `src/lib/api.ts`)

TypeScript API client modules:

| File | Purpose |
|------|---------|
| `api.ts` | Main API client (base URL config, fetch wrappers) |
| `chatApi.ts` | Chat endpoint client |
| `chatApi.test.ts` | Chat API tests |
| `agentSessions.ts` | Agent session management |
| `evaluation.ts` | Strategy evaluation API |
| `graphMemory.ts` | Graph memory API client |
| `memory.ts` | Memory API client |

---

## Agent Manager (`src/lib/agents/`)

Frontend agent orchestration:

| File | Purpose |
|------|---------|
| `agentManager.ts` | Manages agent lifecycle and routing |
| `agentStreamStore.ts` | Svelte store for streaming agent responses |
| `analystAgent.ts` | Analyst agent config |
| `copilotAgent.ts` | Copilot agent config |
| `quantcodeAgent.ts` | QuantCode agent config |
| `claudeCodeAgent.ts` | Claude Code agent integration |
| `langchainAgent.ts` | ⚠️ LangChain agent (legacy) |
| `memoryManager.ts` | Frontend memory management |
| `skills/analystSkills.ts` | Analyst skill definitions |
| `skills/copilotSkills.ts` | Copilot skill definitions |
| `skills/quantcodeSkills.ts` | QuantCode skill definitions |

---

## Agent Tools Backend (in quantmind-ide Python layer)

The IDE has a small Python layer (`quantmind-ide/src/`) for backend operations:

| File | Purpose |
|------|---------|
| `agents/tools/registry.py` | IDE-side tool registry |
| `agents/tools/file_operations.py` | File read/write for IDE |
| `agents/tools/workflow.py` | Workflow trigger tools |
| `agents/tools/mcp_integration.py` | MCP server integration |
| `agents/tools/analysis.py` | Analysis tools |
| `agents/tools/broker.py` | Broker tools |
| `agents/tools/deployment.py` | EA deployment tools |
| `agents/tools/video_ingest_trd.py` | Video ingest + TRD creation |
| `agents/mcp/client.py` | MCP client |
| `agents/mcp/adapter.py` | MCP tool adapter |
| `api/mcp_endpoints.py` | IDE-local MCP endpoints |
| `api/workflow_endpoints.py` | IDE-local workflow endpoints |
| `api/file_history_endpoints.py` | File history API |

---

## Feature Server (`feature-server/`)

A minimal SvelteKit 5 server running on port 3002. Purpose is likely experimental feature development separate from the main IDE. Uses `@sveltejs/adapter-node`.

---

## Component Count Summary

| Category | Count |
|----------|-------|
| Agent Panel components | 12 |
| Agent Panel Settings | 9 |
| Trading Floor | 12 |
| Bot Management | 9 |
| Backtesting / Charts | 9 |
| Strategy / Code | 4 |
| Settings | 3 |
| System utilities | 7 |
| **Total Svelte components** | **~65+** |

---

## Key Component State Management

Most components use Svelte 4 reactive stores (`$writable`, `$derived`) from `src/lib/stores/`.

WebSocket connections from:
- `src/lib/stores/agentStreamStore.ts` — agent SSE streaming
- Direct WebSocket in `TickStreamChart.svelte` for tick data
- `BrokerManagement.svelte` connects to `WS /ws/broker/{broker_id}`
