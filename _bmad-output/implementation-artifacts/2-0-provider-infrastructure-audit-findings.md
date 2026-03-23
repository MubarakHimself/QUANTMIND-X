# Provider Infrastructure Audit — Story 2.0

**Date:** 2026-03-17
**Story:** 2-0-provider-infrastructure-audit
**Status:** Complete

---

## 1. Existing Provider Config Files/Classes

### Backend Agent Configuration

| File | Purpose | Key Findings |
|------|---------|--------------|
| `src/agents/config.py` | AgentConfig dataclass | Defines `llm_provider`, `llm_model`, `provider`, `model` fields. Default: `openrouter` provider, `claude-sonnet-4-20250514` model |
| `src/agents/llm_provider.py` | LLM Provider System | ProviderType enum with 5 providers: OPENROUTER, ZHIPU, ANTHROPIC, OPENAI, MINIMAX. Base URLs and API key env vars defined |
| `src/database/models/provider_config.py` | DB Model | ProviderConfig table with: id, name, api_key, base_url, enabled, timestamps |
| `src/api/provider_config_endpoints.py` | REST API | Full CRUD at `/api/providers`: list, create, update, delete, get, available |

### Provider Models Supported (from API)

- **Anthropic**: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5
- **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo
- **OpenRouter**: Claude, Gemini, Llama, Mistral routes
- **DeepSeek**: Chat, Coder
- **GLM (Zhipu)**: GLM-4, GLM-4 Flash, GLM-4 Plus
- **MiniMax**: M2.5, M2.1, M2
- **Google Gemini**: 2.0 Flash Exp, 1.5 Pro/Flash
- **Azure OpenAI**: GPT-4o, GPT-4
- **Cohere**: Command R+, Command R
- **Mistral**: Large, Small, Medium

---

## 2. Hardcoded API Keys or Model Names

### API Keys
**✅ No hardcoded API keys found**

All API keys are retrieved from environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `ZHIPU_API_KEY` / `GLM_API_KEY`
- `MINIMAX_API_KEY`
- `CONTABO_HMM_API_KEY`

### Model Names (Hardcoded but Expected)
These are defined in `src/agents/config.py` and `src/agents/llm_provider.py`:
- Default model: `claude-sonnet-4-20250514`
- Fallback models per agent type defined in `AGENT_MODELS` dict

### Base URLs (Hardcoded but Acceptable)
| Provider | Base URL |
|----------|----------|
| Anthropic | `https://api.anthropic.com/v1` |
| OpenAI | `https://api.openai.com/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| GLM (Zhipu) | `https://open.bigmodel.cn/api/paas/v4` |
| MiniMax | `https://api.minimax.chat/v1` |
| DeepSeek | `https://api.deepseek.com/v1` |

---

## 3. ProvidersPanel.svelte Implementation State

**Location:** `quantmind-ide/src/lib/components/settings/ProvidersPanel.svelte`

### Current Features
- ✅ Displays 6 provider cards: Anthropic, Zhipu (GLM), MiniMax, DeepSeek, OpenAI, OpenRouter
- ✅ Base URL configuration per provider
- ✅ API key input with show/hide toggle
- ✅ Save/Delete functionality via REST API
- ✅ Refresh button to reload provider data
- ✅ Status badges (Configured/Not configured)

### Current Gaps/Issues
1. **⚠️ Migration Errors**: Svelte directives error at top of file (migration-task)
2. **⚠️ Hardcoded localhost**: Uses `http://localhost:8000` in fetch calls (violates project rule)
3. **⚠️ Zhipu ID Mismatch**: Uses `zhipu` but backend expects `glm`
4. **⚠️ No Model Selection**: Doesn't allow selecting specific models per provider
5. **⚠️ No Test/Edit Functionality**: Missing from current implementation (story 2.5 will address)

---

## 4. Server Connection Configuration

### Contabo Server (Agent/Compute Node)
**Environment Variables:**
| Variable | Purpose |
|----------|---------|
| `CONTABO_HOST` | SSH host for Contabo |
| `CONTABO_USER` | SSH username |
| `CONTABO_SSH_KEY_PATH` | Path to SSH key |
| `CONTABO_HMM_API_URL` | HMM inference API URL (default: `http://localhost:8001`) |
| `CONTABO_HMM_API_KEY` | API key for HMM inference |
| `CONTABO_MODEL_PATH` | Path to HMM models |
| `CONTABO_METADATA_PATH` | Path to model metadata |

**Used in:**
- `src/router/engine.py` — Regime fetcher polling
- `src/router/hmm_version_control.py` — Model sync
- `src/api/hmm_endpoints.py` — Training jobs

### Cloudzy Server (Live Trading Node)
**Environment Variables:**
| Variable | Purpose |
|----------|---------|
| `MT5_DEMO_LOGIN` | MT5 demo login |
| `MT5_DEMO_PASSWORD` | MT5 demo password |
| `MT5_DEMO_SERVER` | MT5 demo server |
| `MT5_VPS_HOST` | MT5 VPS host |
| `MT5_VPS_PORT` | MT5 VPS port |
| `MT5_ACCOUNT_ID` | MT5 account ID |
| `MT5_CONNECTED` | Connection status flag |
| `CLOUDZY_HOT_DB_URL` | Hot database URL |

**Router Inclusion Control (from `src/api/server.py`):**
```python
INCLUDE_CLOUDZY = os.environ.get("INCLUDE_CLOUDZY", "true").lower() == "true"
INCLUDE_CONTABO = os.environ.get("INCLUDE_CONTABO", "true").lower() == "true"
```

---

## 5. Claude Agent SDK Initialization

### Current State
**Location:** `src/agents/subagent/spawner.py`

```python
from anthropic import Anthropic

SDK_AVAILABLE = False
try:
    from anthropic import Anthropic
    SDK_AVAILABLE = True
    logger.info("Anthropic SDK available for agent spawning")
except ImportError:
    logger.warning("Anthropic SDK not installed. Using fallback mode.")
```

### Usage Points
| File | Usage |
|------|-------|
| `src/agents/subagent/spawner.py` | AgentSpawner class for spawning sub-agents |
| `src/agents/departments/subagents/*.py` (6 files) | Lazy import in each subagent |

### Key Observation
The spawner uses the basic `Anthropic` client, not the newer "Claude Agent SDK" (the agent-specific SDK with tool use). Current implementation is simpler - just basic API calls.

---

## 6. Recommendations for Stories 2.1–2.6

### Story 2.1: Provider Configuration Storage Schema
- **Priority:** High
- The `ProviderConfig` table already exists in SQLite
- Consider adding encrypted storage for API keys (currently stored as plain text per comment)

### Story 2.2: Providers & Servers API Endpoints
- **Priority:** High
- Endpoints already exist at `/api/providers/*`
- May need additional endpoints for server connection config

### Story 2.3: Claude Agent SDK Provider Routing
- **Priority:** High
- Current `llm_provider.py` is marked DEPRECATED
- Need to wire up actual provider routing to use `ProviderConfig` from database

### Story 2.4: Provider Hot-Swap Without Restart
- **Priority:** Medium
- Current config is loaded at startup
- Need to implement dynamic reloading

### Story 2.5: ProvidersPanel UI Add/Edit/Test/Delete
- **Priority:** High
- Fix hardcoded `localhost:8000` — use `apiFetch` wrapper
- Fix Zhipu ID mismatch (`zhipu` → `glm`)
- Add model selection dropdown
- Fix Svelte migration errors

### Story 2.6: Server Connection Configuration Panel
- **Priority:** Medium
- Create new UI panel for Contabo/Cloudzy/MT5 server config
- Similar pattern to ProvidersPanel

---

## 7. Files Scanned (Read-Only)

| Path | Files Matched |
|------|---------------|
| `src/agents/` | 52 files referencing provider/model |
| `src/` (full) | 242 files matching search patterns |
| `quantmind-ide/src/lib/components/settings/` | ProvidersPanel.svelte |

---

## 8. Summary

| Aspect | Status |
|--------|--------|
| Provider Config | ✅ Exists (config.py, llm_provider.py) |
| Database Model | ✅ Exists (ProviderConfig) |
| REST API | ✅ Exists (/api/providers/*) |
| UI Component | ⚠️ Partial (ProvidersPanel.svelte with issues) |
| No Hardcoded Keys | ✅ Confirmed |
| Server Config | ⚠️ Env vars, no dedicated config UI |
| Agent SDK Init | ⚠️ Basic Anthropic client, not Agent SDK |
