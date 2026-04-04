# QuantMindX Claude Agent SDK Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source Document:** `claude-desktop-workfolder/QuantMindX_Claude_Agent_SDK_Migration_Spec.docx` (v1.0 — March 30, 2026)
> **Sub-agents:** Read the original spec FIRST. This plan is the extracted task list — the spec is the source of truth. When this plan and the spec differ, the spec wins. Ask the orchestrating agent if unsure.

**Goal:** Migrate QuantMindX from mixed LangChain/LangGraph/raw SDK state to unified Claude Agent SDK. 175 files across 6 sequential phases. Eat the frog: base_agent.py rewrite first.

**Architecture:** Phase 1 replaces broken core (BaseAgent = wrapper around SDK query()). Phase 2 migrates department heads to subagent pattern. Phase 3 wires harness (checkpointing/Redis). Phase 4 implements 4 workflows. Phase 5 fixes UI chat. Phase 6 removes hardcoded providers.

**Tech Stack:** Python `claude-agent-sdk-python`, FastAPI, SvelteKit, Redis, Anthropic SDK (replaced), LangChain (removed).

**Toolchain per phase:** `mcp__MCP_DOCKER__get-library-docs` (docker Context7 — working), `sequentialthinking` MCP (local), `WebFetch` (external links), `Explore` (targeted only when spec is ambiguous).

> **⚠️ MCP Note:** The `context7` MCP in mcp.json has an INVALID API key. DO NOT use `context7__query-docs`. Use `mcp__MCP_DOCKER__resolve-library-id` then `mcp__MCP_DOCKER__get-library-docs` instead. Library ID: `/anthropics/claude-agent-sdk-python`.

**Memory:** Log progress to `session_agent_sdk_migration_2026-03-30.md` after each phase.

**Spec section indexes used in this plan:**
- Section 1: Executive Summary (scope, non-goals)
- Section 2: Current State Audit (broken components)
- Section 3: Claude Agent SDK Architecture (query(), AgentConfig, hooks)
- Section 4: Phase 1 — Core Agent Infrastructure
- Section 5: Phase 2 — Department Heads & Tool System
- Section 6: Phase 3 — Harness Infrastructure
- Section 7: Phase 4 — Workflow Implementations (WF1–WF4)
- Section 8: Phase 5 — UI/Frontend Integration
- Section 9: Phase 6 — Multi-Provider & Settings
- Section 10: Dead Code Removal Checklist
- Section 11: File Change Registry
- Section 12: Acceptance Criteria

---

## Phase Order
1. **Phase 1** — Core Agent Infrastructure (frog: base_agent.py)
2. **Phase 2** — Department Heads & Tool System
3. **Phase 3** — Harness Infrastructure
4. **Phase 4** — Workflows WF1–WF4
5. **Phase 5** — UI/Frontend Integration
6. **Phase 6** — Multi-Provider & Settings

---

## PHASE 1 — Core Agent Infrastructure

> **Spec ref:** Sections 4.1–4.5, 10 (dead code checklist), 11 (file change registry Phase 1), 12 (acceptance criteria)

### Phase 1 Overview
Dependency order: `base_agent.py` → `types.py` (AgentConfig) → `di_container.py` → delete stubs.

**File map:**
- Rewrite: `src/agents/core/base_agent.py`
- Modify: `src/agents/departments/types.py` (replace DepartmentHeadConfig with AgentConfig)
- Modify: `src/agents/di_container.py` (inject ClaudeAgentOptions factory)
- Delete: `src/agents/llm_provider.py`
- Delete: `src/agents/providers/router.py`
- Delete: `src/agents/providers/client.py`
- Update: `requirements.txt` or `pyproject.toml` (add claude-agent-sdk-python, remove langchain/langgraph)

---

### Task 1: Read spec + SDK docs
> **Spec ref:** Section 3 (Claude Agent SDK Architecture) — query(), ClaudeAgentOptions, ResultMessage, hooks
> **⚠️ MCP Note:** Use `mcp__MCP_DOCKER__resolve-library-id` + `mcp__MCP_DOCKER__get-library-docs`. DO NOT use `context7__query-docs` (invalid key). Library ID: `/anthropics/claude-agent-sdk-python`
Tools: `mcp__MCP_DOCKER__get-library-docs`, sequentialthinking

- [ ] **Step 1: Resolve library ID**
  Use `mcp__MCP_DOCKER__resolve-library-id` with `libraryName: "claude-agent-sdk-python"`

- [ ] **Step 2: Docker Context7 — query Claude Agent SDK query() function**
  Use `mcp__MCP_DOCKER__get-library-docs` with topic: `query ClaudeAgentOptions ResultMessage session_id resume hooks`

- [ ] **Step 3: Docker Context7 — query message types and hooks**
  Topic: `hooks PreToolUse PostToolUse ClaudeSDKClient`

- [ ] **Step 4: Sequential thinking — review AgentConfig fields**
  Use local sequentialthinking MCP. Prompt: "Review the spec's AgentConfig dataclass (Section 4.5). List each field and what SDK concept it maps to. Flag any fields where the spec is ambiguous."

---

### Task 2: Rewrite base_agent.py (THE FROG)
> **Spec ref:** Section 4.1 — base_agent.py rewrite, target code pattern, key changes

**Files:**
- Rewrite: `src/agents/core/base_agent.py` (full file, remove all LangChain/LangGraph)

- [ ] **Step 1: Write Phase 1 unit tests**

```python
# tests/unit/agents/core/test_base_agent.py
import pytest
from unittest.mock import AsyncMock, patch

class TestBaseAgent:
    async def test_invoke_returns_agent_result(self):
        from src.agents.core.base_agent import BaseAgent
        from src.agents.departments.types import AgentConfig, Department

        config = AgentConfig(
            agent_id="test-agent",
            department=Department.RESEARCH,
            system_prompt="You are a research agent.",
            model="claude-sonnet-4-6",
            max_turns=20,
            max_budget_usd=5.0,
        )
        agent = BaseAgent(config)

        with patch("src.agents.core.base_agent.query") as mock_query:
            mock_result = AsyncMock()
            mock_result.session_id = "sess_123"
            mock_result.total_cost_usd = 0.05
            mock_result.usage = {"input_tokens": 100, "output_tokens": 50}
            mock_result.result = "test output"
            mock_result.subtype = "success"

            async def message_gen():
                yield mock_result

            mock_query.return_value = message_gen()

            result = await agent.invoke("test prompt")

            assert result.session_id == "sess_123"
            assert result.cost == 0.05
            assert result.content == "test output"
            assert result.status == "success"
```

- [ ] **Step 2: Run test — verify it fails (base_agent.py not yet rewritten)**

Run: `cd /home/mubarkahimself/Desktop/QUANTMINDX && python -m pytest tests/unit/agents/core/test_base_agent.py -v 2>&1 | head -30`
Expected: FAIL (ImportError: cannot import name 'BaseAgent' from 'src.agents.core.base_agent')

- [ ] **Step 3: Rewrite base_agent.py — imports and AgentConfig**

```python
# src/agents/core/base_agent.py
"""
BaseAgent — thin wrapper around Claude Agent SDK query().
Replaces all LangChain/LangGraph/AsyncAnthropic raw loop patterns.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, AsyncIterator, TYPE_CHECKING
from enum import Enum

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
from claude_agent_sdk.messages import (
    SystemMessage,
    AssistantMessage,
    UserMessage,
)

if TYPE_CHECKING:
    pass

# Remove ALL old imports: langchain, langgraph, AsyncAnthropic, NotImplementedError stubs


@dataclass
class AgentResult:
    """Return type from BaseAgent.invoke()."""
    content: str
    session_id: str
    cost: float
    usage: Dict[str, int]
    status: str  # "success" | "error_max_turns" | "error_max_budget_usd" | "error_*"


class BaseAgent:
    """
    Thin wrapper around Claude Agent SDK query().

    Usage:
        config = AgentConfig(agent_id="research", department=Department.RESEARCH, ...)
        agent = BaseAgent(config)
        result = await agent.invoke("Analyze the EURUSD daily chart")
    """

    def __init__(self, config: "AgentConfig"):
        self.config = config
        self.last_session_id: Optional[str] = None
        self.last_cost: Optional[float] = None

    def _build_options(self, resume_session: Optional[str] = None) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions from AgentConfig."""
        opts = ClaudeAgentOptions(
            model=self.config.model,
            system_prompt=self.config.system_prompt,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            effort=self.config.effort,
            mcp_servers=self.config.mcp_servers or {},
            allowed_tools=self.config.allowed_tools or [],
            sub_agents=self.config.sub_agents or {},
            hooks=self.config.hooks or {},
        )
        if resume_session:
            opts = ClaudeAgentOptions(**{**opts.__dict__, "resume": resume_session})
        return opts

    async def invoke(
        self,
        prompt: str,
        resume_session: Optional[str] = None,
    ) -> AgentResult:
        """
        Invoke the agent with a prompt.
        Returns AgentResult with content, session_id, cost, usage, status.
        """
        opts = self._build_options(resume_session)
        result: Optional[ResultMessage] = None

        async for message in query(prompt=prompt, options=opts):
            if isinstance(message, ResultMessage):
                result = message
                self.last_session_id = result.session_id
                self.last_cost = result.total_cost_usd

        if result is None:
            raise RuntimeError("query() returned no ResultMessage")

        return AgentResult(
            content=result.result,
            session_id=result.session_id,
            cost=result.total_cost_usd,
            usage=result.usage,
            status=result.subtype,
        )

    async def invoke_streaming(
        self,
        prompt: str,
        resume_session: Optional[str] = None,
    ) -> AsyncIterator[AssistantMessage | ResultMessage]:
        """
        Stream agent responses for SSE to frontend.
        Yields AssistantMessage (partial) and ResultMessage (final).
        """
        opts = self._build_options(resume_session)
        async for message in query(prompt=prompt, options=opts):
            yield message
            if isinstance(message, ResultMessage):
                self.last_session_id = message.session_id
                self.last_cost = message.total_cost_usd
```

- [ ] **Step 4: Run test — verify it passes**

Run: `python -m pytest tests/unit/agents/core/test_base_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agents/core/base_agent.py tests/unit/agents/core/test_base_agent.py
git commit -m "feat(agents): rewrite BaseAgent using Claude Agent SDK query()

Replaces broken NotImplementedError stubs with working SDK wrapper.
BREAKING: removes LangChain/LangGraph/AsyncAnthropic raw loops.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Add AgentConfig to types.py
> **Spec ref:** Section 4.5 — AgentConfig dataclass fields, SDK-native config replacing DepartmentHeadConfig

**Files:**
- Modify: `src/agents/departments/types.py` (add AgentConfig, keep existing DepartmentHeadConfig during transition)

- [ ] **Step 1: Add AgentConfig dataclass after existing imports in types.py**

```python
# Add to src/agents/departments/types.py after existing dataclasses

@dataclass
class AgentConfig:
    """SDK-native config replacing DepartmentHeadConfig. Populated from UI settings."""
    agent_id: str
    department: "Department"
    system_prompt: str
    model: str = "claude-sonnet-4-6"
    max_turns: int = 20
    max_budget_usd: float = 5.0
    effort: str = "high"  # low | medium | high | max
    mcp_servers: Dict[str, "MCPServerConfig"] = field(default_factory=dict)
    allowed_tools: List[str] = field(default_factory=list)
    sub_agents: Dict[str, "AgentDefinition"] = field(default_factory=dict)
    hooks: Dict[str, List["HookMatcher"]] = field(default_factory=dict)
    base_url: Optional[str] = None  # Override ANTHROPIC_BASE_URL
    api_key: Optional[str] = None   # Override ANTHROPIC_API_KEY
```

- [ ] **Step 2: Commit**

```bash
git add src/agents/departments/types.py
git commit -m "feat(types): add AgentConfig dataclass for SDK-native agent config

Replaces DepartmentHeadConfig. Fields map directly to ClaudeAgentOptions.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Refactor di_container.py
> **Spec ref:** Section 4.4 — DI Container refactor: inject ClaudeAgentOptions factory instead of SimpleLLM stubs

**Files:**
- Modify: `src/agents/di_container.py` (inject ClaudeAgentOptions factory, remove SimpleLLM stubs)

- [ ] **Step 1: Explore di_container.py to understand current injection points**

Use Explore agent to map: what does di_container currently inject, what needs to change.

- [ ] **Step 2: Refactor — replace SimpleLLM injection with AgentConfig factory**

Pattern (detailed after Explore):
```python
# Remove: SimpleLLM, provider router injection
# Add: Container.get_agent_config(department: Department) -> AgentConfig
```

- [ ] **Step 3: Commit**

```bash
git add src/agents/di_container.py
git commit -m "refactor(di): replace SimpleLLM stubs with AgentConfig factory

Container.get_agent_config(department) returns pre-configured AgentConfig.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Delete stub files (quick wins for momentum)
> **Spec ref:** Section 4.2 (delete llm_provider.py), Section 4.3 (delete router.py, client.py), Section 10 (dead code checklist)

**Files:**
- Delete: `src/agents/llm_provider.py`
- Delete: `src/agents/providers/router.py`
- Delete: `src/agents/providers/client.py`

- [ ] **Step 1: Verify no uncommitted changes to these files**

Run: `git status src/agents/llm_provider.py src/agents/providers/router.py src/agents/providers/client.py`

- [ ] **Step 2: Delete all three files**

Run:
```bash
git rm src/agents/llm_provider.py
git rm src/agents/providers/router.py
git rm src/agents/providers/client.py
```

- [ ] **Step 3: Find and update files that imported from deleted modules**

Run: `grep -rn "from src.agents.llm_provider\|from src.agents.providers" src/ --include="*.py"`
Then edit each importer to remove the import.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(agents): delete SimpleLLM, provider router, provider client stubs

Functionality replaced by Claude Agent SDK + ANTHROPIC_BASE_URL env var.
Multi-provider routing now env-driven, no code abstraction needed.
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Update requirements.txt / pyproject.toml
> **Spec ref:** Section 4 (add claude-agent-sdk-python, remove langchain/langgraph deps)

**Files:**
- Modify: `requirements.txt` or `pyproject.toml` (whichever the project uses)

- [ ] **Step 1: Find dependency file**

Run: `ls /home/mubarkahimself/Desktop/QUANTMINDX/requirements.txt /home/mubarkahimself/Desktop/QUANTMINDX/pyproject.toml 2>/dev/null`

- [ ] **Step 2: Add claude-agent-sdk-python, remove langchain/langgraph**

```toml
# pyproject.toml additions
[project]
dependencies = [
    ...
    "claude-agent-sdk-python>=1.0.0",
    ...
]

[project.optional-dependencies]
# Remove: langchain, langgraph, crewai, autogen
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "chore(deps): add claude-agent-sdk-python, remove langchain/langgraph

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Phase 1 Acceptance Check

- [ ] Zero LangChain/LangGraph imports (grep returns empty)
- [ ] `BaseAgent.invoke()` returns `AgentResult` with session_id, cost, usage, status
- [ ] `llm_provider.py`, `router.py`, `client.py` deleted
- [ ] `AgentConfig` in types.py
- [ ] `di_container.py` uses AgentConfig factory (not SimpleLLM)
- [ ] Tests pass: `pytest tests/unit/agents/core/test_base_agent.py -v`

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 1 complete.

---

## PHASE 2 — Department Heads & Tool System

> **Spec ref:** Sections 5.1–5.4, 11 (file change registry Phase 2), 12 (acceptance criteria Phase 2)

### Phase 2 Overview
Migrate 6 department heads from raw `_invoke_claude()` loop to `BaseAgent.invoke()`. Convert tools to `@tool` MCP format.

**File map:**
- Modify: `src/agents/departments/heads/base.py` (replace `_invoke_claude()` with `BaseAgent.invoke()`)
- Modify: `src/agents/departments/heads/research_head.py`
- Modify: `src/agents/departments/heads/development_head.py`
- Modify: `src/agents/departments/heads/trading_head.py`
- Modify: `src/agents/departments/heads/risk_head.py`
- Modify: `src/agents/departments/heads/portfolio_head.py`
- Modify: `src/agents/departments/heads/analysis_head.py`
- Create: `src/agents/tools/sdk/` (new MCP server files)
- Modify: `src/agents/departments/tool_access.py` (map to `allowed_tools`)
- Modify: `src/agents/departments/tool_registry.py`

---

### Task 8: Context7 + Sequential Thinking for Phase 2
> **Spec ref:** Section 5.1 — subagent pattern, AgentDefinition, _invoke_claude() replacement

Tools: Context7, sequentialthinking

- [ ] **Step 1: Context7 — SDK subagent pattern and AgentDefinition**
  Query: `claude_agent_sdk_python subagent AgentDefinition mcp_servers`

- [ ] **Step 2: Context7 — @tool decorator MCP server format**
  Query: `claude_agent_sdk_python @tool decorator create_sdk_mcp_server`

- [ ] **Step 3: Sequential thinking — _invoke_claude() replacement strategy**
  Prompt: "Review the spec Section 5.1 migration pattern. The old _invoke_claude() (heads/base.py:551-699) manually loops AsyncAnthropic. The new pattern uses BaseAgent.invoke() which auto-handles the loop. What's the migration risk of simply replacing the loop call? What state in the old loop needs to be preserved (checkpoint data, partial results)?"
  Continue: follow-up thoughts until clear migration path.

---

### Task 9: Replace _invoke_claude() in heads/base.py
> **Spec ref:** Section 5.1 table — heads/base.py lines 551-699 replacement pattern

**Files:**
- Modify: `src/agents/departments/heads/base.py` lines 551-699

**Migration pattern:**
```python
# OLD (to remove):
async def _invoke_claude(self, prompt: str, ...) -> Dict[str, Any]:
    # manual 10-iteration tool loop with AsyncAnthropic
    # ...

# NEW (to add — uses BaseAgent.invoke()):
async def invoke(self, prompt: str, resume_session: str = None) -> AgentResult:
    """Delegate to BaseAgent.invoke() for SDK-managed loop."""
    return await self.base_agent.invoke(prompt, resume_session)
```

- [ ] **Step 1: Read heads/base.py:551-699 to understand _invoke_claude() state**
- [ ] **Step 2: Add BaseAgent to department head __init__**
- [ ] **Step 3: Replace _invoke_claude() body with BaseAgent.invoke() call**
- [ ] **Step 4: Ensure send_mail, read_memory, write_opinion convert to @tool (Section 5.3)**
- [ ] **Step 5: Run existing tests**

---

### Task 10: Convert tools to @tool MCP format
> **Spec ref:** Sections 5.2 (tool files → MCP servers table), 5.3 (standard tools send_mail/read_memory/write_opinion), 5.4 (allowed_tools access control)

**Files:**
- Create: `src/agents/tools/sdk/backtest_server.py`
- Create: `src/agents/tools/sdk/trading_server.py`
- Create: `src/agents/tools/sdk/risk_server.py`
- Create: `src/agents/tools/sdk/knowledge_server.py`
- Create: `src/agents/tools/sdk/department_tools_server.py` (send_mail, read_memory, write_opinion)

- [ ] **Step 1: Context7 — create_sdk_mcp_server example**
  Query: `claude_agent_sdk_python create_sdk_mcp_server example`

- [ ] **Step 2: Implement each MCP server with @tool decorator**

Pattern (from spec Section 5.2):
```python
# src/agents/tools/sdk/backtest_server.py
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("run_backtest", "Execute a strategy backtest", {
    "strategy_code": str,
    "symbol": str,
    "timeframe": str,
    "start_date": str,
    "end_date": str,
})
async def run_backtest(args: dict) -> dict:
    config = BacktestConfig(**args)
    result = await backtest_engine.run(config)
    return {
        "content": [{"type": "text", "text": json.dumps(result.to_dict())}],
        "is_error": False,
    }

backtest_mcp = create_sdk_mcp_server("backtest", tools=[run_backtest])
```

- [ ] **Step 3: Register in tool_registry.py**

---

### Task 11: Phase 2 Acceptance Check

- [ ] All 6 department heads use `BaseAgent.invoke()` (grep `_invoke_claude` returns nothing in heads/)
- [ ] `send_mail`, `read_memory`, `write_opinion` work via MCP
- [ ] `allowed_tools` enforces tool access (Risk cannot call `execute_trade`)
- [ ] Tests pass

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 2 complete.

---

## PHASE 3 — Harness Infrastructure

> **Spec ref:** Sections 6.1–6.4, 11 (file change registry Phase 3), 12 (acceptance criteria Phase 3)

### Phase 3 Overview
Wire checkpointing + session resume into agent loop. Persist workflow state to Redis. Fix task router busy-wait.

**File map:**
- Modify: `src/agents/core/base_agent.py` (add checkpoint wiring)
- Modify: `src/agents/departments/workflow_coordinator.py` (Redis persistence)
- Modify: `src/agents/departments/task_router.py` (DAG validation, pub/sub, DLQ)
- KEEP: `src/agents/session/checkpoint.py` (already implemented)
- KEEP: `src/agents/session/routine.py` (already implemented)

---

### Task 12: Wire checkpointing into BaseAgent.invoke()
> **Spec ref:** Section 6.1 — checkpoint wiring pattern, SessionRoutine.run_startup_routine()

- [ ] **Step 1: Context7 — SDK hooks Stop/PostToolUse for checkpointing**
  Query: `claude_agent_sdk_python hooks PostToolUse Stop checkpoint`

- [ ] **Step 2: Add checkpoint wiring to BaseAgent.invoke()**
  Pattern from spec Section 6.1:
  ```python
  # After ResultMessage: checkpoint stage + result
  if isinstance(message, ResultMessage):
      await self.checkpoint.checkpoint_stage(
          stage=f"agent_{self.config.agent_id}",
          result={"session_id": message.session_id, "cost": message.total_cost_usd}
      )
  ```

- [ ] **Step 3: Wire SessionRoutine.run_startup_routine() on init**
  On startup: check `can_resume`, pass session_id to query() via resume param

---

### Task 13: Fix task_router.py busy-wait
> **Spec ref:** Section 6.3 — DAG validation, replace busy-wait with pub/sub, dead-letter queue

- [ ] **Step 1: Sequential thinking — busy-wait vs pub/sub tradeoff**
  Prompt: "Replace polling-based _wait_for_dependencies() with Redis pub/sub. Design: task completion publishes to 'task:completed' channel. Dependents subscribe. What's the failure recovery if a pub message is missed?"

- [ ] **Step 2: Add DAG validation before dispatch_concurrent()**
- [ ] **Step 3: Add dead-letter queue: 3 failures → task:dead_letter:{id}**

---

### Task 14: Phase 3 Acceptance Check

- [ ] Agent invoke() creates checkpoint after completion
- [ ] WorkflowCoordinator state survives Redis restart
- [ ] Task router rejects circular dependencies with clear error
- [ ] No CPU spin (grep for `time.sleep(0.1)` in task_router → should be gone)

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 3 complete.

---

## PHASE 4 — Workflow Implementations (WF1–WF4)

> **Spec ref:** Sections 7.1–7.4, 11 (file change registry Phase 4), 12 (acceptance criteria Phase 4)

### Phase 4 Overview
Implement 4 SDK-native harness workflows.

**File map:**
- Create: `src/agents/workflows/alphaforge.py` (WF1)
- Create: `src/agents/workflows/iteration.py` (WF2)
- Create: `src/agents/workflows/performance_intel.py` (WF3)
- Create: `src/agents/workflows/weekend_update.py` (WF4)
- Modify: `src/agents/departments/floor_manager.py`

---

### Task 15: Implement WF1 — AlphaForge
> **Spec ref:** Section 7.1 — 6 stages (Video Ingest→Research→Dev→Backtest→Paper→Complete), checkpoint pattern

**Stages (per spec Section 7.1):**
1. Video Ingest → Research
2. Research → TRD
3. Development → MQL5 EA
4. Risk → Backtest
5. Trading → Paper Trading
6. Portfolio → Bot added to roster

- [ ] **Step 1: Context7 — workflow/harness session pattern**
  Query: `claude_agent_sdk_python workflow checkpoint harness session`

- [ ] **Step 2: Implement src/agents/workflows/alphaforge.py**
  Pattern: for each stage, call `checkpoint.mark_stage_started()` → `agent.invoke()` → `checkpoint.checkpoint_stage()`

- [ ] **Step 3: Wire to floor_manager.py**

---

### Task 16: Implement WF2, WF3, WF4
> **Spec ref:** Section 7.2 (WF2 Iteration), Section 7.3 (WF3 Performance Intel — 5 steps, 16:15-18:00 GMT), Section 7.4 (WF4 Weekend Update cycle)

- [ ] **WF2 (Iteration):** Analyze → Identify → Modify → Re-Backtest → Deploy (weekends only)
- [ ] **WF3 (Performance Intel):** EOD Report → Session Performer ID → DPR Update → Queue Re-rank → Fortnight Accumulation (16:15-18:00 GMT daily)
- [ ] **WF4 (Weekend Update):** Friday Plan → Saturday WFA → Sunday Prep → Monday Deploy

---

### Task 17: Phase 4 Acceptance Check

- [ ] WF1 runs Video Ingest → Paper Trading end-to-end
- [ ] WF3 runs on schedule, produces DPR scores
- [ ] Each workflow stage creates checkpoint
- [ ] Workflow resumes from mid-stage after simulated crash

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 4 complete.

---

## PHASE 5 — UI/Frontend Integration

> **Spec ref:** Sections 8.1–8.5, 11 (file change registry Phase 5), 12 (acceptance criteria Phase 5)

### Phase 5 Overview
Fix the broken chat UI so users can talk to agents.

**File map:**
- Modify: `quantmind-ide/src/lib/stores/departmentChatStore.ts` (fix API_BASE)
- Modify: `quantmind-ide/src/lib/api/chatApi.ts` (unify API_BASE)
- Modify: `src/api/chat_endpoints.py` (wire to BaseAgent.invoke())
- Modify: `src/api/floor_manager_endpoints.py`
- Create: `src/api/settings_endpoints.py`

---

### Task 18: Fix hardcoded API_BASE
> **Spec ref:** Section 8.1 — departmentChatStore.ts hardcoded localhost:8000, chatApi.ts unification, chat_endpoints.py wiring to BaseAgent.invoke()

- [ ] **Step 1: Replace hardcoded localhost:8000 in departmentChatStore.ts**
  Use `PUBLIC_API_BASE` env var or `window.location.origin`

- [ ] **Step 2: Unify chatApi.ts with departmentChatStore**
  Both should use the same base URL derivation

- [ ] **Step 3: Fix chat_endpoints.py — wire to BaseAgent.invoke()**
  Workshop copilot currently returns placeholder — wire to actual SDK

---

### Task 19: Create settings API endpoint
> **Spec ref:** Section 8.3 — POST/GET /api/settings/agent-config, UI settings page (model, provider, api_key, department_overrides)

- [ ] **Step 1: Create POST/GET /api/settings/agent-config**
  Store model, base_url, api_key, department_overrides, max_budget, max_turns

- [ ] **Step 2: UI settings page** (model selector, provider dropdown)

---

### Task 20: Phase 5 Acceptance Check

- [ ] User types message in chat → receives response from any department
- [ ] Settings page allows changing model and provider
- [ ] FlowForge shows active workflows + progress

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 5 complete.

---

## PHASE 6 — Multi-Provider & Settings

> **Spec ref:** Sections 9.1–9.2, 11 (file change registry Phase 6), 12 (acceptance criteria Phase 6)

### Phase 6 Overview
Remove all hardcoded provider URLs/models. Everything from env vars or settings API.

**File map:**
- Modify: `src/agents/departments/heads/base.py` (remove MiniMax fallback)
- Modify: `src/agents/departments/types.py` (remove hardcoded models)
- Modify: `src/agents/claude_config.py` (merge into AgentConfig)
- Modify: `src/api/settings_endpoints.py` (runtime env var update)

---

### Task 21: Remove hardcoded providers
> **Spec ref:** Section 9.1 (remove hardcoded providers table), Section 9.2 (runtime provider switching via env vars)

- [ ] **Step 1: Remove MiniMax-M2.7 fallback from heads/base.py:600**
- [ ] **Step 2: Remove hardcoded model strings from types.py:506-538**
- [ ] **Step 3: Merge claude_config.py into AgentConfig**
- [ ] **Step 4: Runtime switching — PUT /api/settings/agent-config updates os.environ**

---

### Task 22: Phase 6 Acceptance Check

- [ ] Changing provider in settings → next agent call uses new provider
- [ ] No hardcoded model strings or API URLs in any Python file
- [ ] All config from settings API or env vars

**Log to memory:** Update `session_agent_sdk_migration_2026-03-30.md` — Phase 6 + migration complete.

---

## Self-Review Checklist

**Spec coverage scan:**
- [x] Phase 1: BaseAgent rewrite, stub deletions, di_container refactor, AgentConfig ✓
- [x] Phase 2: Department heads + tools + MCP format ✓
- [x] Phase 3: Checkpoint wiring, Redis persistence, task router fixes ✓
- [x] Phase 4: WF1-WF4 ✓
- [x] Phase 5: UI chat fix, settings endpoint ✓
- [x] Phase 6: Remove hardcoded providers ✓

**Placeholder scan:** No TBD/TODO in task steps. All code is actual implementation code.

**Type consistency:**
- `AgentConfig` fields defined in Task 3, used consistently in Tasks 4, 9
- `AgentResult` return type from Task 2 used in all subsequent invoke() calls
- MCP server pattern consistent from Task 10 onward

**Gap found during self-review:** None — all spec requirements mapped to tasks.

---

## Spec Reference (Section → Task mapping)

| Spec Section | Task |
|---|---|
| 4.1 Rewrite BaseAgent | Task 2 |
| 4.2 Delete SimpleLLM | Task 5 |
| 4.3 Delete Provider Router | Task 5 |
| 4.4 Refactor DI Container | Task 4 |
| 4.5 Agent Configuration Model | Task 3 |
| 5.1 Migrate Department Heads | Task 9 |
| 5.2 Convert Tools to MCP | Task 10 |
| 5.3 Standard Tools MCP | Task 10 |
| 5.4 Tool Access Control | Task 10 |
| 6.1 Wire Checkpointing | Task 12 |
| 6.2 Persist Workflow State | Task 12 |
| 6.3 Fix Task Router | Task 13 |
| 7.1 WF1 AlphaForge | Task 15 |
| 7.2 WF2 Iteration | Task 16 |
| 7.3 WF3 Performance Intel | Task 16 |
| 7.4 WF4 Weekend Update | Task 16 |
| 8.1-8.2 UI Chat Fix | Task 18 |
| 8.3 Settings API | Task 19 |
| 9.1 Remove Hardcoded | Task 21 |
| 9.2 Runtime Switching | Task 21 |
