# Story 1.1: Security Hardening & Legacy Import Cleanup

Status: **COMPLETED**

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer setting up QUANTMINDX for ongoing development,
I want all secrets removed from git tracking and all dead LangGraph/LangChain imports eliminated,
so that the codebase has no security vulnerabilities at the foundation layer and no import errors from removed packages.

## Acceptance Criteria

1. **Given** `.env` is currently tracked in git (confirmed by Story 1.0 audit),
   **When** the story tasks run,
   **Then** `.env` is added to `.gitignore`, removed from git tracking via `git rm --cached .env`, and a `.env.example` template with all required keys (no values) is committed in its place.
   > **⚠️ CORRECTED BY STORY 1.0 AUDIT:** `.env` is NOT tracked in git and `.gitignore` already has `.env` entries. `.env.example` already exists at project root (created 2026-03-06). This AC is **already satisfied**. Focus shifts to: (a) verifying `.env.example` completeness, (b) adding missing env vars discovered since initial creation.

2. **Given** LangGraph/LangChain imports exist in backend files (14 import lines confirmed in Story 1.0 audit),
   **When** each import is removed or stubbed,
   **Then** `from langchain` and `from langgraph` produce zero results in `grep -r` across `src/`,
   **And** `pip install -r requirements.txt` completes without errors on a clean environment.
   > **Note:** `requirements.txt` already has langchain/langgraph packages commented out. The import statements in source files are the remaining problem.

3. **Given** the backend runs after cleanup,
   **When** `uvicorn src.server:app` starts,
   **Then** no `ImportError` or `ModuleNotFoundError` appears in startup logs.

4. **Given** `ApiKeysPanel.svelte` exists at `quantmind-ide/src/lib/components/settings/ApiKeysPanel.svelte`,
   **When** the file is audited,
   **Then** no API key values are hardcoded in source — all keys are loaded via API calls to the backend `.env` configuration.

## Tasks / Subtasks

- [x] Task 1: Verify and expand `.env.example` completeness (AC: #1)
  - [x] Run `grep -rn "os.getenv\|os.environ" src/ --include="*.py"` to collect all env var names used in backend
  - [x] Compare collected env vars against existing `.env.example` — identify any missing entries
  - [x] Add missing env vars (confirmed missing: `NODE_ROLE`, `QWEN_API_KEY`, `GEMINI_API_KEY`, `VideoIngest_*` vars, `MCP_CONFIG_PATH`)
  - [x] `NODE_ROLE` must be added with comment: `# Accepted values: contabo | cloudzy | local (default: local)`
  - [x] All values must be placeholder strings (no real values) — see NFR-S3

- [x] Task 2: Remove/stub LangChain imports in `src/agents/core/base_agent.py` (AC: #2, #3) — **HIGHEST PRIORITY: this file cascades to all agents**
  - [x] Read file fully before editing
  - [x] Remove: `from langchain_openai import ChatOpenAI`
  - [x] Remove: `from langchain_anthropic import ChatAnthropic`
  - [x] Remove: `from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage`
  - [x] Remove: `from langchain_core.tools import tool, BaseTool`
  - [x] Remove: `from langgraph.prebuilt import create_react_agent`
  - [x] Remove: `from langgraph.checkpoint.memory import MemorySaver`
  - [x] Remove: `from langgraph.types import Command`
  - [x] Replace any usage of removed imports with minimal stubs or `pass` — do NOT delete classes/functions, just neutralize the langchain dependency
  - [x] Verify file stays under 500 lines (NFR-M3)

- [x] Task 3: Remove/stub LangChain imports in `src/agents/skills/` (AC: #2, #3)
  - [x] `src/agents/skills/base.py` — remove `from langchain_core.tools import BaseTool`, replace with `object` or a plain Python ABC stub
  - [x] `src/agents/skills/queuing.py` — remove `from langchain_core.tools import tool`, replace tool decorator usage with no-op stub or plain function
  - [x] `src/agents/skills/coding.py` — remove `from langchain_community.agent_toolkits import FileManagementToolkit` and `from langchain_community.tools import ShellTool`; stub out the classes they replaced

- [x] Task 4: Remove/stub LangChain imports in `src/agents/tools/` and `src/agents/knowledge/` (AC: #2, #3)
  - [x] `src/agents/tools/pinescript_tools.py` — remove `from langchain_core.tools import tool`
  - [x] `src/agents/knowledge/retriever.py` — remove `from langchain_core.tools import tool`

- [x] Task 5: Remove/stub LangChain/LangGraph imports in remaining files (AC: #2, #3)
  - [x] `src/integrations/pine_script_converter.py` — remove `from langgraph.graph import StateGraph, END`; stub out the StateGraph usage
  - [x] `src/agents/pinescript.py` — read file, remove langchain/langgraph imports, stub as needed
  - [x] `src/agents/queue_manager.py` — read file, remove langchain/langgraph imports, stub as needed
  - [x] `src/video_ingest/providers.py` — read file, remove langchain wrapper, keep base video logic
  - [x] `src/router/workflow_orchestrator.py` — read file, remove langchain imports, stub the orchestrator minimally
- [x] Task 6: Verify cleanup is complete (AC: #2, #3)
  - [x] Run: `grep -rn "from langchain\|import langchain\|from langgraph\|import langgraph" src/` — must return zero results
  - [x] Attempt: `python -c "import src.api.server"` or equivalent import check — no ImportError
  - [x] Run: `pip install -r requirements.txt` to confirm clean install

- [x] Task 7: Audit `ApiKeysPanel.svelte` (AC: #4)
  - [x] Read `quantmind-ide/src/lib/components/settings/ApiKeysPanel.svelte` fully
  - [x] Confirm no API key values are hardcoded (initial grep found none — verify manually)
  - [x] Document findings in Dev Agent Record below

## Dev Notes

### Critical Context from Story 1.0 Audit

**STOP — READ BEFORE CODING.** Story 1.0 pre-populated the following verified findings. Do not re-derive these; trust them unless a direct check contradicts them.

#### Pre-resolved Items (DO NOT REDO)

| Item | Status | Evidence |
|------|--------|----------|
| `.env` git tracking | ✅ Already resolved | `git ls-files --error-unmatch .env` → "pathspec did not match any file(s)" |
| `.gitignore` has `.env` entry | ✅ Already present | `.gitignore` contains `.env` and `.env.*` entries |
| `.env.example` exists | ✅ Exists at project root | Created 2026-03-06, 183 lines |
| `requirements.txt` langchain removed | ✅ Commented out | Lines 13-19 in `requirements.txt` are all `# REMOVED` comments |

#### Remaining Real Work: LangChain Import Cleanup

`requirements.txt` removed the packages, but **14 import lines** remain in source files. Python will raise `ModuleNotFoundError` on any of these when the file loads.

**Confirmed import lines (from `grep -rn "^from langchain\|^import langchain\|^from langgraph\|^import langgraph" src/`):**

```
src/agents/tools/pinescript_tools.py:14:  from langchain_core.tools import tool
src/agents/knowledge/retriever.py:8:    from langchain_core.tools import tool
src/agents/skills/queuing.py:10:        from langchain_core.tools import tool
src/agents/skills/base.py:7:           from langchain_core.tools import BaseTool
src/agents/skills/coding.py:6:         from langchain_community.agent_toolkits import FileManagementToolkit
src/agents/skills/coding.py:7:         from langchain_community.tools import ShellTool
src/agents/core/base_agent.py:11:      from langchain_openai import ChatOpenAI
src/agents/core/base_agent.py:12:      from langchain_anthropic import ChatAnthropic
src/agents/core/base_agent.py:13:      from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
src/agents/core/base_agent.py:14:      from langchain_core.tools import tool, BaseTool
src/agents/core/base_agent.py:15:      from langgraph.prebuilt import create_react_agent
src/agents/core/base_agent.py:16:      from langgraph.checkpoint.memory import MemorySaver
src/agents/core/base_agent.py:17:      from langgraph.types import Command
src/integrations/pine_script_converter.py:19: from langgraph.graph import StateGraph, END
```

**Additional files to check** (identified in Story 1.0 audit but not yet grep-confirmed with line numbers — read and fix):
- `src/agents/pinescript.py`
- `src/agents/queue_manager.py`
- `src/video_ingest/providers.py`
- `src/router/workflow_orchestrator.py`
- `src/api/ide_chat.py`

#### Approach for Each File Type

**Pattern: Agent core files (`base_agent.py`, `skills/base.py`, `skills/queuing.py`)**
- These will be **fully rebuilt in Epic 7** with Anthropic Agent SDK
- For now: Remove the langchain imports. If the class/function body uses the removed types, replace with:
  - `BaseTool` → `object` (or a minimal `class BaseTool: pass` stub at top of file)
  - `tool` decorator → no-op decorator: `def tool(f): return f`
  - `ChatOpenAI`, `ChatAnthropic` → remove and stub the method/class that used them with `raise NotImplementedError("Migrate to Anthropic Agent SDK in Epic 7")`
  - LangGraph types → remove and stub with comments pointing to Epic 7

**Pattern: API endpoint files (`ide_chat.py`)**
- Remove langchain chain calls
- Return a placeholder response so the endpoint stays functional:
  ```python
  return {"response": "Chat endpoint migration pending (Epic 7)", "streaming": False}
  ```

**Pattern: Integration files (`pine_script_converter.py`, `workflow_orchestrator.py`)**
- Remove LangGraph StateGraph — replace with a simple sequential function if used
- Keep the file structure intact — Epic 7 will rebuild the actual logic

#### `.env.example` Gap Analysis

The existing `.env.example` (183 lines) is comprehensive. The following env vars are used in source but missing from `.env.example`:

| Env Var | Source File | Required |
|---------|-------------|---------|
| `NODE_ROLE` | Story 1.3 target (not in source yet) | Add now — Story 1.3 depends on it |
| `QWEN_API_KEY` | `src/video_ingest/models.py:342` | Optional — video ingest feature |
| `QWEN_HEADLESS` | `src/video_ingest/models.py:343` | Optional |
| `QWEN_MODEL` | `src/video_ingest/models.py:344` | Optional |
| `GEMINI_API_KEY` | `src/video_ingest/models.py:346` | Optional — video ingest feature |
| `GEMINI_YOLO_MODE` | `src/video_ingest/models.py:347` | Optional |
| `QWEN_REQUESTS_PER_DAY` | `src/video_ingest/models.py:349` | Optional |
| `VideoIngest_CACHE_DIR` | `src/video_ingest/models.py:351` | Optional |
| `VideoIngest_CACHE_MAX_SIZE_GB` | `src/video_ingest/models.py:352` | Optional |
| `VideoIngest_CACHE_MAX_AGE_DAYS` | `src/video_ingest/models.py:353` | Optional |
| `VideoIngest_MAX_CONCURRENT_JOBS` | `src/video_ingest/models.py:355` | Optional |
| `VideoIngest_JOB_DB_PATH` | `src/video_ingest/models.py:356` | Optional |
| `VideoIngest_OUTPUT_DIR` | `src/video_ingest/models.py:358` | Optional |
| `VideoIngest_FRAME_INTERVAL` | `src/video_ingest/models.py:360` | Optional |
| `VideoIngest_AUDIO_BITRATE` | `src/video_ingest/models.py:361` | Optional |
| `VideoIngest_AUDIO_CHANNELS` | `src/video_ingest/models.py:362` | Optional |
| `VideoIngest_MAX_RETRY_ATTEMPTS` | `src/video_ingest/models.py:364` | Optional |
| `VideoIngest_BASE_RETRY_DELAY` | `src/video_ingest/models.py:365` | Optional |
| `VideoIngest_LOG_LEVEL` | `src/video_ingest/models.py:367` | Optional |
| `VideoIngest_LOG_FILE` | `src/video_ingest/models.py:368` | Optional |
| `MCP_CONFIG_PATH` | `src/mcp/client.py:459` | Optional |

#### Architecture Compliance

**From architecture.md Decision 3 (Agent SDK Migration):**
> "Remove LangGraph/LangChain from `requirements.txt`, migrate `src/agents/core/base_agent.py` and `src/agents/registry.py` to the Anthropic Agent SDK. Wire the existing `src/agents/departments/` structure to `claude-agent-sdk-python`."

**Story 1.1 scope is STEP 1 ONLY** — remove the dead imports. Story 7.x (Epic 7) does the real SDK migration.

**From architecture.md source tree (`requirements.txt` annotation):**
> `[extend] Remove langchain, langgraph, langchain_openai`

**Non-Functional Requirements directly applicable:**
- **NFR-S3:** All API keys and credentials in `.env` files only — never in source code or version control
- **NFR-M1:** Existing LangChain and LangGraph code is technical debt — migrate to Claude Agent SDK; no new LangChain/LangGraph code introduced
- **NFR-M3:** All Python backend files kept under 500 lines — refactor at boundary if any file grows during stubbing

#### What NOT to Touch

| Area | Reason |
|------|--------|
| `src/backtesting/` | DO NOT TOUCH — 6 modes confirmed working (Story 1.0 finding) |
| `src/memory/graph/` | DO NOT MODIFY — 80-90% done, extend only. Contains: `compaction.py`, `facade.py`, `migration.py`, `operations.py`, `store.py`, `tier_manager.py`, `tools.py`, `types.py` |
| `src/agents/departments/` | Keep structure intact — Epic 7 migrates it to Agent SDK |
| `.env` itself | Do NOT commit — `.gitignore` is correct |

#### Python File Locations Quick Reference

```
src/
  agents/
    core/
      base_agent.py               ← MODIFY (langchain cascade file)
    skills/
      base.py                     ← MODIFY (langchain_core.tools.BaseTool)
      queuing.py                  ← MODIFY (langchain_core.tools.tool)
      coding.py                   ← MODIFY (langchain_community toolkits)
    tools/
      pinescript_tools.py         ← MODIFY (langchain_core.tools.tool)
    knowledge/
      retriever.py                ← MODIFY (langchain_core.tools.tool)
    pinescript.py                 ← MODIFY (read first to confirm imports)
    queue_manager.py              ← MODIFY (read first to confirm imports)
  integrations/
    pine_script_converter.py      ← MODIFY (langgraph.graph.StateGraph)
  router/
    workflow_orchestrator.py      ← MODIFY (read first to confirm imports)
  api/
    ide_chat.py                   ← MODIFY (langchain chains → stub response)
  video_ingest/
    providers.py                  ← MODIFY (read first to confirm imports)
```

#### Git Intelligence (Recent Commits)

Last 5 commits are all frontend/settings changes (ProvidersPanel, workshop model dropdown, console log cleanup). No backend Python changes recently. This means:
- No risk of merge conflicts on Python files
- No prior langchain cleanup partially done — this is a clean starting point

### Project Structure Notes

- Alignment with unified project structure:
  - `src/` = Python FastAPI backend (monolith, all routers registered unconditionally until Story 1.3)
  - `quantmind-ide/` = Tauri 2 + SvelteKit 2 frontend (Svelte 4, migration pending Story 1.2)
  - `_bmad-output/` = planning artifacts — do not modify

- Detected variances relevant to this story:
  - 14 langchain/langgraph import lines in src/ files — this story removes them
  - `.env.example` exists but is missing ~20 env vars — this story adds them

### References

- Epic 1 story 1.1 definition: [Source: _bmad-output/planning-artifacts/epics.md#line-455]
- Epic 1 overview (work type, FRs): [Source: _bmad-output/planning-artifacts/epics.md#line-312]
- Architecture Decision 3 (Agent SDK migration steps): [Source: _bmad-output/planning-artifacts/architecture.md#line-185]
- Architecture source tree langchain annotation: [Source: _bmad-output/planning-artifacts/architecture.md#line-1891]
- NFR-S3 (API keys in .env only): [Source: _bmad-output/planning-artifacts/epics.md#line-147]
- NFR-M1 (no new LangChain): [Source: _bmad-output/planning-artifacts/epics.md#line-175]
- NFR-M3 (Python files under 500 lines): [Source: _bmad-output/planning-artifacts/epics.md#line-177]
- Story 1.0 audit findings (langchain file list): [Source: _bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md#section-B]
- Story 1.0 audit findings (.env status): [Source: _bmad-output/implementation-artifacts/1-0-platform-codebase-exploration-audit.md#section-C]
- Story 1.3 notes (NODE_ROLE .env.example requirement): [Source: _bmad-output/planning-artifacts/epics.md#line-542]

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- All langchain imports removed - verified with grep
- Python syntax verified on all modified files

### Completion Notes List

- **Task 1 (.env.example)**: Added NODE_ROLE, QWEN_*, GEMINI_*, VideoIngest_*, MCP_CONFIG_PATH to .env.example
- **Task 2 (base_agent.py)**: Removed all langchain imports, added stub classes (BaseMessage, HumanMessage, SystemMessage, BaseTool, MemorySaver, tool decorator), neutralized methods to raise NotImplementedError
- **Task 3 (skills/)**: Removed langchain imports from base.py, queuing.py, coding.py; replaced with stubs
- **Task 4 (tools/knowledge/)**: Removed tool decorator from pinescript_tools.py and retriever.py
- **Task 5 (remaining files)**: Removed langchain from pine_script_converter.py (added StateGraph stub), pinescript.py (complete stub), queue_manager.py (replaced HumanMessage with dict), providers.py (use direct API), workflow_orchestrator.py (replaced HumanMessage with dict), ide_chat.py (stub fallback)
- **Task 6 (verification)**: grep returns zero results for langchain/langgraph imports
- **Task 7 (ApiKeysPanel)**: Verified no API keys hardcoded - uses API calls to backend

### File List

**Modified files:**
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/.env.example`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/core/base_agent.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/skills/base.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/skills/queuing.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/skills/coding.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/tools/pinescript_tools.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/knowledge/retriever.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/integrations/pine_script_converter.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/pinescript.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/agents/queue_manager.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/video_ingest/providers.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/router/workflow_orchestrator.py`
- `/home/mubarkahimself/Desktop/quantmindx-story-1-backend/src/api/ide_chat.py`

**Story file updated:**
- `/home/mubarkahimself/Desktop/QUANTMINDX/_bmad-output/implementation-artifacts/1-1-security-hardening-legacy-import-cleanup.md`
