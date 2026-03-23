# QUANTMINDX — API Contracts Reference

**Generated:** 2026-03-11 | **Base URL:** `http://localhost:8000`

---

## Overview

The FastAPI backend exposes 50+ router modules. All endpoints return JSON. Authentication is currently token-based for MT5 Bridge only; the main API has no auth by default (intended for local/VPS deployment).

**API Docs (auto-generated):** `http://localhost:8000/docs`

---

## Health & System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Application health check |
| `/api/version` | GET | API version information |
| `/metrics` | GET | Prometheus metrics endpoint |
| `/api/metrics/system` | GET | System resource metrics |
| `/api/metrics/trading` | GET | Trading performance metrics |
| `/api/metrics/ws` | WS | WebSocket metrics stream (1s broadcast) |

---

## Agent & Chat

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat/send` | POST | Send message to legacy chat handler (deprecated path) |
| `/api/chat/sessions` | GET | List chat sessions |
| `/api/chat/sessions/{id}` | GET | Get session details |
| `/api/chat/sessions/{id}/messages` | GET | Get session messages |

### Floor Manager (Canonical)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/floor-manager/chat` | POST | Send message to FloorManager (canonical agent entry point) |
| `/api/floor-manager/status` | GET | Get FloorManager + department statuses |
| `/api/floor-manager/departments` | GET | List active departments |
| `/api/floor-manager/dispatch` | POST | Manual department dispatch |

### Workshop Copilot

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workshop-copilot/chat` | POST | Workshop copilot chat endpoint |
| `/api/workshop-copilot/sessions` | GET | List workshop sessions |

### Department Mail

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/department-mail/messages` | GET | List department messages |
| `/api/department-mail/messages/{id}` | GET | Get specific message |
| `/api/department-mail/send` | POST | Send inter-department message |
| `/api/department-mail/mark-read/{id}` | POST | Mark message as read |

---

## Agent Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents/stream` | GET (SSE) | Agent event SSE stream (filter by agent_id, task_id, event_type) |
| `/api/agents/sessions` | GET | List agent sessions |
| `/api/agents/sessions/{id}` | GET | Get session detail |
| `/api/agents/queue` | GET | Agent task queue |
| `/api/agents/queue/{id}` | DELETE | Remove queued task |
| `/api/agents/activity` | GET | Agent activity feed |
| `/api/agents/metrics` | GET | Agent performance metrics |
| `/api/agents/models` | GET | Available LLM model configs |
| `/api/agents/tools` | GET | Available agent tools |
| `/api/agents/tools/call` | POST | Call a specific agent tool |
| `/api/agents/checkpoints` | GET | Session checkpoints |
| `/api/agents/checkpoints/{id}` | GET | Get checkpoint detail |

### Claude Agent

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/claude-agent/run` | POST | Run Claude agent with specific tools |
| `/api/claude-agent/sessions` | GET | Claude agent session list |

---

## Provider / Model Config

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/providers` | GET | List LLM provider configurations |
| `/api/providers/{id}` | GET | Get provider by ID |
| `/api/providers/{id}` | PUT | Update provider config |
| `/api/providers/{id}/activate` | POST | Activate a provider |
| `/api/models` | GET | List available models per provider |

---

## Strategy Router

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/router/status` | GET | StrategyRouter engine status |
| `/api/router/tick` | POST | Submit tick data to router |
| `/api/router/bots` | GET | List registered bots |
| `/api/router/bots/{id}` | GET | Get bot manifest detail |
| `/api/router/regime` | GET | Current regime report |

---

## HMM System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hmm/status` | GET | HMM deployment status + agreement metrics |
| `/api/hmm/sync` | POST | Trigger model sync (Contabo → Cloudzy) |
| `/api/hmm/sync/progress` | GET | Sync progress |
| `/api/hmm/mode` | POST | Change deployment mode |
| `/api/hmm/approval-token` | POST | Generate approval token for mode change |
| `/api/hmm/shadow-log` | GET | Shadow mode comparison log |
| `/api/hmm/models` | GET | List versioned HMM models |
| `/api/hmm/train` | POST | Trigger training job |
| `/api/hmm/train/{job_id}` | GET | Training job status |
| `/api/hmm/predict` | POST | Run HMM inference on OHLCV data |

---

## Trading Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trading/proposals` | GET | List trade proposals |
| `/api/trading/proposals/{id}` | GET | Get proposal detail |
| `/api/trading/proposals/{id}/approve` | POST | Approve trade proposal |
| `/api/trading/proposals/{id}/reject` | POST | Reject trade proposal |
| `/api/trading/positions` | GET | Open positions |
| `/api/trading/history` | GET | Trade history |
| `/api/trading/accounts` | GET | Prop firm accounts |
| `/api/trading/accounts/{id}` | GET | Account detail |

### Paper Trading

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/paper-trading/accounts` | GET | Virtual accounts |
| `/api/paper-trading/deploy` | POST | Deploy bot to paper trading |
| `/api/paper-trading/promote/{bot_id}` | POST | Promote paper bot to live |

---

## Bot Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bots` | GET | List all bots |
| `/api/bots/{id}` | GET | Bot detail + manifest |
| `/api/bots/{id}/quarantine` | POST | Quarantine a bot |
| `/api/bots/{id}/promote` | POST | Promote bot lifecycle stage |
| `/api/bots/{id}/demote` | POST | Demote bot lifecycle stage |
| `/api/bots/clone` | POST | Clone a bot to a new symbol |
| `/api/bots/batch` | POST | Batch bot operations |
| `/api/lifecycle-scanner/check` | POST | Run lifecycle check |
| `/api/lifecycle-scanner/status` | GET | Lifecycle scanner status |

---

## Kill Switch

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/kill-switch/soft` | POST | Soft kill (halt new trades) |
| `/api/kill-switch/hard` | POST | Hard kill (close all positions) |
| `/api/kill-switch/status` | GET | Kill switch state |
| `/api/kill-switch/reset` | POST | Reset kill switch |
| `/api/kill-switch/progressive/status` | GET | Progressive kill switch tier status |

---

## Broker Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/brokers` | GET | List brokers |
| `/api/brokers/{id}` | GET | Broker detail (fees, pip values) |
| `/api/brokers/{id}` | PUT | Update broker config |
| `/api/brokers` | POST | Register new broker |
| `/api/brokers/{id}` | DELETE | Remove broker |
| `/api/brokers/ws` | WS | Broker WebSocket stream |

---

## Backtesting

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/backtest/run` | POST | Start backtest job |
| `/api/v1/backtest/{id}` | GET | Backtest result |
| `/api/v1/backtest/list` | GET | List backtest runs |
| `/api/monte-carlo/ws` | WS | Monte Carlo simulation WebSocket |

---

## Knowledge Base / IDE

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/knowledge/search` | GET | Semantic search over articles |
| `/api/knowledge/articles/{id}` | GET | Get full article |
| `/api/knowledge/categories` | GET | List knowledge categories |
| `/api/pdf/upload` | POST | Upload PDF for indexing |
| `/api/pdf/jobs` | GET | PDF processing jobs |

---

## Strategies & TRDs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trd` | GET | List Trading Rule Documents |
| `/api/trd/{id}` | GET | TRD detail |
| `/api/trd` | POST | Create TRD manually |
| `/api/strategies` | GET | List strategies |
| `/api/strategies/{id}` | GET | Strategy detail |
| `/api/strategies/{id}/evaluate` | POST | Evaluate strategy |

---

## EA Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/eas` | GET | List Expert Advisors |
| `/api/eas/{id}` | GET | EA detail |
| `/api/eas/deploy` | POST | Deploy EA to MT5 |
| `/api/github/sync` | POST | Sync EAs from GitHub repo |

---

## Video Ingest

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video-ingest/submit` | POST | Submit YouTube URL for processing |
| `/api/video-ingest/jobs` | GET | List video processing jobs |
| `/api/video-ingest/jobs/{id}` | GET | Job status + result |
| `/api/video-to-ea/run` | POST | Run full video → EA workflow |

---

## Analytics

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/performance` | GET | Performance metrics (win rate, Sharpe, etc.) |
| `/api/analytics/correlations` | GET | Cross-strategy correlations |
| `/api/analytics/market-data` | GET | OHLCV data query |
| `/api/analytics/regime-history` | GET | Historical regime classifications |

---

## Memory

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/memory/entries` | GET | List agent memory entries |
| `/api/memory/entries/{id}` | GET | Get memory entry |
| `/api/memory/search` | POST | Semantic memory search |
| `/api/memory/departments` | GET | Department-level memories |
| `/api/memory/unified` | GET | Cross-system memory view |
| `/api/graph-memory/nodes` | GET | Graph memory nodes |
| `/api/graph-memory/query` | POST | Graph memory query |

---

## Settings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/settings` | GET | All settings |
| `/api/settings/{key}` | PUT | Update setting |
| `/api/settings/demo-mode` | GET/POST | Demo mode toggle |

---

## MCP

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mcp/servers` | GET | List MCP server configs |
| `/api/mcp/tools` | GET | List available MCP tools |
| `/api/mcp/tools/call` | POST | Call MCP tool |

---

## Workflows & Batch

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workflows` | GET | List workflows |
| `/api/workflows/{id}/run` | POST | Execute workflow |
| `/api/batch/operations` | POST | Batch agent operations |

---

## Trading Floor

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/trading-floor/status` | GET | Floor-wide status |
| `/api/trading-floor/departments` | GET | Department statuses |
| `/api/trading-floor/control` | POST | Floor control commands |

---

## WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `WS /api/brokers/ws` | Broker data stream |
| `WS /api/monte-carlo/ws` | Monte Carlo simulation stream |
| `WS /ws/broker/{broker_id}` | Per-broker data stream |
| `WS /ws/monte-carlo` | Monte Carlo WebSocket (alias) |
| `WS /api/metrics/ws` | Metrics broadcast (1s interval) |

---

## SSE Endpoints

| Endpoint | Parameters | Description |
|----------|-----------|-------------|
| `GET /api/agents/stream` | `agent_id`, `task_id`, `event_type` | Agent event stream |

### SSE Event Types

| Event | Payload | Description |
|-------|---------|-------------|
| `agent_started` | `{agent_id, task_id}` | Agent task started |
| `tool_start` | `{tool_name, input}` | Tool call beginning |
| `tool_complete` | `{tool_name, output}` | Tool call completed |
| `thinking` | `{content}` | Agent reasoning |
| `text` | `{content}` | Text output |
| `agent_complete` | `{agent_id, result}` | Agent task finished |
| `error` | `{message}` | Error event |

---

## Common Request/Response Patterns

### Pagination

Query parameters: `?page=1&per_page=50`

### Error Response

```json
{
  "detail": "Error message here"
}
```

### Standard Success Response

Most endpoints return domain-specific JSON. No envelope wrapper.

---

## IDE-Specific Endpoints (`src/api/ide_endpoints.py`)

The `create_ide_api_app()` function mounts a sub-app at `/api/ide`:

| Endpoint | Purpose |
|----------|---------|
| `/api/ide/files` | File browser |
| `/api/ide/files/{path}` | Read/write file |
| `/api/ide/terminal` | Terminal session management |
| `/api/ide/session` | IDE session state |
| `/api/ide/assets` | Strategy assets browser |
| `/api/ide/ea` | EA file management |
| `/api/ide/strategies` | IDE strategy management |
| `/api/ide/timeframes` | Available timeframes |
| `/api/ide/mt5` | MT5 connection via IDE |
| `/api/ide/backtest` | IDE backtest runner |
| `/api/ide/knowledge` | IDE knowledge search |
| `/api/ide/chat` | IDE chat interface |
| `/api/ide/video-ingest` | Video ingest via IDE |
