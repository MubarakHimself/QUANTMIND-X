# Claude-Flow Architecture Analysis for Department-Based Agent Systems

**Analysis Date:** 2025-02-26
**Analyst:** Claude Code Research Agent
**Source:** https://github.com/ruvnet/claude-flow

---

## Executive Summary

This document analyzes the claude-flow multi-agent orchestration framework and identifies key architectural patterns applicable to department-based agent systems like QuantMindX. The analysis focuses on patterns that can improve agent coordination, memory management, and hierarchical orchestration.

---

## Key Architectural Insights

### 1. Core Separation of Concerns

**claude-flow Pattern:**
```
claude-flow = LEDGER (orchestrator/state tracker)
Codex = EXECUTOR (worker that writes code/runs commands)
```

**Critical Behavioral Rule:**
- claude-flow **NEVER** writes code, runs commands, or creates files
- claude-flow **ONLY** coordinates agents, tracks state, and stores memory
- Codex (or similar) handles all execution

**Application to QUANTMINDX:**
Current implementation has `ClaudeOrchestrator` spawning subprocesses for execution, which creates a hybrid model. For department-based systems:
- Separate **Orchestrator Layer** (coordination only) from **Execution Layer** (worker agents)
- Orchestrator should manage queues, state, and memory - not execute tasks directly
- Worker agents handle all file operations, API calls, and command execution

**Code Pattern:**
```python
# Current (mixed concerns):
class ClaudeOrchestrator:
    async def _spawn_claude(self, ...):  # Orchestrator spawns processes
        process = await asyncio.create_subprocess_exec(...)

# Recommended (separation):
class DepartmentOrchestrator:
    """Only coordinates - never executes"""
    def __init__(self):
        self.state_tracker = StateTracker()
        self.memory_store = MemoryStore()
        self.task_queue = TaskQueue()

    async def delegate_task(self, task: Task) -> str:
        """Queue task for worker - don't execute"""
        return await self.task_queue.enqueue(task)

class WorkerAgent:
    """Only executes - doesn't coordinate"""
    async def execute_task(self, task: Task) -> Result:
        """Handle file operations, API calls, etc."""
        pass
```

---

### 2. Swarm Topologies for Department Coordination

**claude-flow Topologies:**

| Topology | Structure | Use Case |
|----------|-----------|----------|
| **hierarchical** | Tree structure with coordinator → leads → workers | Department-based systems (推荐) |
| **mesh** | All-to-all communication | Small teams needing peer review |
| **hierarchical-mesh** | Hierarchical with peer review at each level | Quality-critical workflows |
| **ring** | Circular handoff | Sequential processing pipelines |
| **star** | All workers report to coordinator | Simple command-and-control |
| **adaptive** | Dynamic topology selection | Complex, varying workflows |

**Application to QUANTMINDX Department Model:**

```
                         QUANTMINDX Orchestrator
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
    Research Dept            Development Dept            Operations Dept
         │                          │                          │
    ┌────┴────┐              ┌──────┴──────┐            ┌────┴────┐
    │         │              │             │            │         │
 Analyst  Researcher      QuantCode    Tester     Executor   Monitor
```

**Hierarchical Topology Implementation:**
```python
# Swarm initialization pattern from claude-flow
await swarm_init(
    topology="hierarchical",
    depth=2,  # Orchestrator → Dept Leads → Workers
    agents={
        "orchestrator": {
            "role": "coordinator",
            "capability": "coordination"
        },
        "research_lead": {
            "role": "lead",
            "department": "research",
            "spawnable_agents": ["analyst", "researcher"]
        },
        "dev_lead": {
            "role": "lead",
            "department": "development",
            "spawnable_agents": ["quantcode", "tester"]
        }
    }
)
```

---

### 3. Semantic Memory with Vector Search

**claude-flow Memory Architecture:**

```
AgentDB (Vector-based Semantic Memory)
├── Namespaced storage (patterns, decisions, errors)
├── Embedding-based search (score > 0.7 = strong match)
├── Memory decay and importance scoring
└── Cross-agent learning via shared namespaces
```

**MCP Tools for Memory:**
```bash
memory_search(query="task keywords", namespace="patterns")
memory_store(key="pattern-x", value="what worked", namespace="patterns")
```

**Application to QUANTMINDX:**

The current QUANTMINDX memory system (`src/memory/`) already implements vector search with sqlite-vec. Integration pattern:

```python
# Department-level memory namespaces
MEMORY_NAMESPACES = {
    "research": "research.findings",
    "development": "dev.patterns",
    "trading": "trading.decisions",
    "errors": "errors.learnings",
    "strategies": "strategies.library"
}

# Cross-department learning
async def learn_from_execution(
    task: Task,
    result: Result,
    department: str
):
    """Store learnings for other departments"""
    if result.success:
        # Pattern that worked
        await memory_store(
            key=f"{task.type}_pattern",
            value={
                "approach": result.approach,
                "department": department,
                "outcome": "success"
            },
            namespace=f"{department}.patterns"
        )
    else:
        # Error to avoid
        await memory_store(
            key=f"{task.type}_error",
            value={
                "error": result.error,
                "department": department,
                "remediation": result.remediation
            },
            namespace="errors.learnings"
        )

# Before starting a task
async def check_previous_learnings(task: Task) -> List[Memory]:
    """Search across all departments for relevant experience"""
    results = await memory_search(
        query=f"{task.type} {task.context}",
        namespaces=["*.patterns", "errors.learnings"],
        min_score=0.7
    )
    return results
```

---

### 4. Agent Handoff and Delegation Protocol

**claude-flow Handoff Pattern:**

```
Agent A → claude-flow:agent_spawn(agent_b, task_context)
claude-flow: Creates Agent B with overlay (task_id, files, spec)
Agent B: Executes and reports back via claude-flow:task_complete()
claude-flow: Notifies Agent A of completion
```

**Critical Feature:**
- Handoff is mediated by claude-flow (the LEDGER)
- No direct agent-to-agent communication
- All state transitions tracked in central ledger

**Application to QUANTMINDX:**

Current workflow_orchestrator.py implements linear pipeline. For department handoffs:

```python
class DepartmentHandoffProtocol:
    """Mediated handoff between departments"""

    def __init__(self, orchestrator: DepartmentOrchestrator):
        self.orchestrator = orchestrator
        self.ledger = TaskLedger()

    async def handoff(
        self,
        task: Task,
        from_dept: str,
        to_dept: str,
        context: HandoffContext
    ) -> str:
        """
        Hand off task between departments.

        This creates a ledger entry and notifies the target department.
        No direct agent-to-agent communication.
        """
        # Create handoff record in ledger
        handoff_id = await self.ledger.record_handoff(
            task_id=task.id,
            from_department=from_dept,
            to_department=to_dept,
            context=context,
            timestamp=datetime.utcnow()
        )

        # Queue for target department
        await self.orchestrator.queue_for_department(
            department=to_dept,
            task=task,
            handoff_id=handoff_id
        )

        return handoff_id

    async def complete_handoff(
        self,
        handoff_id: str,
        result: TaskResult
    ):
        """Mark handoff as complete and notify originator"""
        await self.ledger.update_handoff(
            handoff_id=handoff,
            status="completed",
            result=result
        )

# Usage example
handoff = DepartmentHandoffProtocol(orchestrator)

# Research → Development handoff
await handoff.handoff(
    task=trd_generation_task,
    from_dept="research",
    to_dept="development",
    context=HandoffContext(
        nprd_file="/path/to/nprd.json",
        trd_content="# Strategy Requirements...",
        priority="high"
    )
)
```

---

### 5. Self-Learning Workflow Pattern

**claude-flow Learning Cycle:**

```
LEARN (memory_search) → COORDINATE (swarm_init) → EXECUTE → REMEMBER (memory_store)
```

**Behavioral Rule:**
- ALWAYS search memory before starting a task
- ALWAYS store results after completion
- This creates continuous improvement across the swarm

**Application to QUANTMINDX:**

```python
class LearningWorkflowOrchestrator:
    """Orchestrator with built-in learning cycle"""

    async def execute_department_task(
        self,
        department: str,
        task: Task
    ) -> TaskResult:
        """
        Execute task following LEARN → COORDINATE → EXECUTE → REMEMBER cycle
        """
        # 1. LEARN: Search for relevant past experience
        learnings = await self._learn_from_history(
            task_type=task.type,
            department=department
        )

        # 2. COORDINATE: Select agents and plan based on learnings
        coordination_plan = await self._coordinate_with_learnings(
            task=task,
            learnings=learnings,
            department=department
        )

        # 3. EXECUTE: Delegate to worker agents
        result = await self._execute_with_coordination(
            plan=coordination_plan,
            task=task
        )

        # 4. REMEMBER: Store outcome for future learning
        await self._remember_outcome(
            task=task,
            result=result,
            department=department
        )

        return result

    async def _learn_from_history(
        self,
        task_type: str,
        department: str
    ) -> List[Learning]:
        """Search memory for relevant past experience"""
        # Search within department patterns
        dept_patterns = await memory_search(
            query=task_type,
            namespace=f"{department}.patterns",
            min_score=0.7
        )

        # Search error patterns to avoid
        errors = await memory_search(
            query=task_type,
            namespace="errors.learnings",
            min_score=0.7
        )

        # Search related departments
        related_patterns = await memory_search(
            query=task_type,
            namespace="*.patterns",  # Wildcard search
            min_score=0.6
        )

        return dept_patterns + errors + related_patterns

    async def _remember_outcome(
        self,
        task: Task,
        result: TaskResult,
        department: str
    ):
        """Store outcome in appropriate namespace"""
        if result.success:
            await memory_store(
                key=f"{task.type}_success_{datetime.utcnow().isoformat()}",
                value={
                    "task": task.dict(),
                    "approach": result.approach,
                    "metrics": result.metrics,
                    "department": department
                },
                namespace=f"{department}.patterns",
                importance=self._calculate_importance(result)
            )
        else:
            await memory_store(
                key=f"{task.type}_error_{datetime.utcnow().isoformat()}",
                value={
                    "task": task.dict(),
                    "error": result.error,
                    "remediation": result.remediation,
                    "department": department
                },
                namespace="errors.learnings",
                importance=0.9  # Errors are highly important
            )
```

---

### 6. MCP Integration for Tool Access

**claude-flow MCP Pattern:**

MCP (Model Context Protocol) provides:
- Tool discovery and registration
- Standardized tool invocation
- Cross-agent tool sharing
- Namespace-based tool organization

**Key MCP Tools in claude-flow:**

| Tool | Purpose | Namespace |
|------|---------|-----------|
| `memory_search` | Semantic search in AgentDB | coordination |
| `memory_store` | Store learnings | coordination |
| `swarm_init` | Initialize swarm topology | coordination |
| `agent_spawn` | Spawn new agent | coordination |
| `task_complete` | Report task completion | coordination |

**Application to QUANTMINDX:**

Current QUANTMINDX has MCP servers (filesystem, github, context7, etc.). Department-level MCP organization:

```python
# Department-specific MCP tool registries
DEPARTMENT_MCP_TOOLS = {
    "research": {
        "mcp_servers": [
            "brave-search",      # Market research
            "pageindex-articles", # Trading articles
            "context7",           # Documentation
        ],
        "tools": [
            "research_market_data",
            "search_trading_articles",
            "query_documentation"
        ]
    },
    "development": {
        "mcp_servers": [
            "mt5-compiler",      # MQL5 compilation
            "backtest-server",   # Strategy testing
            "context7",          # MQL5 docs
            "github"             # Code sync
        ],
        "tools": [
            "compile_mql5_code",
            "run_backtest",
            "sync_to_github",
            "query_mql5_docs"
        ]
    },
    "operations": {
        "mcp_servers": [
            "filesystem",        # Bot manifests
            "database",          # Trade journal
        ],
        "tools": [
            "get_bot_status",
            "execute_trade",
            "query_trade_journal"
        ]
    }
}

# Tool discovery across departments
class DepartmentMCPRegistry:
    """Registry for department MCP tools"""

    def __init__(self):
        self.registries = DEPARTMENT_MCP_TOOLS

    async def discover_tools(
        self,
        department: str,
        capability: str
    ) -> List[ToolInfo]:
        """Find tools for a specific capability"""
        dept_tools = self.registries.get(department, {})

        matching_tools = []
        for tool in dept_tools.get("tools", []):
            if capability in tool:
                matching_tools.append(ToolInfo(
                    name=tool,
                    department=department,
                    mcp_servers=dept_tools["mcp_servers"]
                ))

        return matching_tools

    async def call_tool(
        self,
        department: str,
        tool_name: str,
        **kwargs
    ) -> Any:
        """Call a tool from a specific department"""
        dept_config = self.registries[department]

        # Find the MCP server that hosts this tool
        # (implementation depends on MCP client setup)
        return await self._execute_via_mcp(
            servers=dept_config["mcp_servers"],
            tool=tool_name,
            kwargs=kwargs
        )
```

---

### 7. Hierarchical Depth Limits

**claude-flow Pattern:**

```
Depth 0: Coordinator (can only spawn leads)
Depth 1: Leads (can spawn workers)
Depth 2: Workers (leaf nodes, cannot spawn)
```

**Purpose:**
- Prevent runaway agent spawning
- Clear authority boundaries
- Predictable resource usage

**Application to QUANTMINDX:**

```python
from enum import Enum
from typing import Optional

class AgentDepth(int, Enum):
    COORDINATOR = 0
    DEPARTMENT_LEAD = 1
    WORKER = 2

class DepartmentAgent:
    """Base class for department agents with depth limits"""

    def __init__(
        self,
        name: str,
        role: str,
        depth: AgentDepth,
        spawnable_depth: Optional[AgentDepth] = None
    ):
        self.name = name
        self.role = role
        self.depth = depth
        self.spawnable_depth = spawnable_depth

        if depth == AgentDepth.WORKER:
            # Workers cannot spawn
            self.can_spawn = False
        elif depth == AgentDepth.DEPARTMENT_LEAD:
            # Leads can spawn workers
            self.can_spawn = True
            self.spawnable_depth = AgentDepth.WORKER
        elif depth == AgentDepth.COORDINATOR:
            # Coordinator can spawn leads
            self.can_spawn = True
            self.spawnable_depth = AgentDepth.DEPARTMENT_LEAD

    async def spawn_agent(
        self,
        agent_type: str,
        task: Task
    ) -> Optional[str]:
        """Spawn a sub-agent if allowed"""
        if not self.can_spawn:
            raise AgentSpawnError(
                f"{self.name} (depth {self.depth}) cannot spawn agents"
            )

        if self.spawnable_depth is None:
            raise AgentSpawnError(
                f"{self.name} has no spawnable depth configured"
            )

        # Verify the target agent type matches allowed depth
        target_depth = self._get_agent_depth(agent_type)
        if target_depth != self.spawnable_depth:
            raise AgentSpawnError(
                f"{self.name} can only spawn depth {self.spawnable_depth} agents, "
                f"not {target_depth} ({agent_type})"
            )

        # Proceed with spawn via orchestrator
        return await self._orchestrator.spawn_agent(
            agent_type=agent_type,
            task=task,
            parent=self.name,
            depth=target_depth
        )

# QUANTMINDX depth configuration
QUANTMINDX_HIERARCHY = {
    "orchestrator": {
        "depth": AgentDepth.COORDINATOR,
        "spawnable": ["research_lead", "dev_lead", "ops_lead"]
    },
    "research_lead": {
        "depth": AgentDepth.DEPARTMENT_LEAD,
        "spawnable": ["analyst", "researcher"]
    },
    "dev_lead": {
        "depth": AgentDepth.DEPARTMENT_LEAD,
        "spawnable": ["quantcode", "tester"]
    },
    "ops_lead": {
        "depth": AgentDepth.DEPARTMENT_LEAD,
        "spawnable": ["executor", "monitor"]
    },
    "analyst": {
        "depth": AgentDepth.WORKER,
        "spawnable": []
    },
    "quantcode": {
        "depth": AgentDepth.WORKER,
        "spawnable": []
    }
}
```

---

### 8. Agent Overlay Pattern

**claude-flow Overlay:**

Each agent has two instruction layers:
1. **Base Layer (agents/{name}.md)**: HOW to work (workflow, constraints)
2. **Overlay Layer (CLAUDE.md)**: WHAT to work (task_id, files, spec)

**Key Pattern:**
```python
overlay = {
    "task_id": "task-123",
    "files": ["src/main.py", "tests/test_main.py"],
    "spec_path": ".overstory/specs/task-123.md",
    "branch": "feature/task-123",
    "parent": "dev_lead",
    "depth": 2
}
```

**Application to QUANTMINDX:**

```python
class DepartmentAgentOverlay:
    """Dynamic task-specific configuration for agents"""

    def __init__(
        self,
        base_agent: DepartmentAgent,
        task: Task,
        handoff_context: Optional[HandoffContext] = None
    ):
        self.base_agent = base_agent
        self.task = task
        self.handoff_context = handoff_context

    def generate_overlay_instructions(self) -> str:
        """Generate task-specific instructions (WHAT to work on)"""
        instructions = f"""
# Task Overlay: {self.task.id}

## Task Information
- Task ID: {self.task.id}
- Type: {self.task.type}
- Priority: {self.task.priority}
- Deadline: {self.task.deadline}

## File Scope
{self._format_file_scope()}

## Handoff Context
{self._format_handoff_context()}

## Expected Outputs
{self._format_expected_outputs()}

## Constraints
{self._format_constraints()}
"""
        return instructions

    def _format_file_scope(self) -> str:
        """Format the files this agent can modify"""
        if self.task.file_scope:
            return "\n".join(f"- {f}" for f in self.task.file_scope)
        return "- All files in workspace"

    def _format_handoff_context(self) -> str:
        """Format context from previous departments"""
        if self.handoff_context:
            return f"""
Previous Department: {self.handoff_context.from_dept}
Handoff ID: {self.handoff_context.id}
Artifacts:
{self._format_artifacts()}
"""
        return "No previous context"

    def _format_artifacts(self) -> str:
        """Format artifacts from previous departments"""
        if not self.handoff_context or not self.handoff_context.artifacts:
            return "No artifacts"

        artifacts = []
        for artifact in self.handoff_context.artifacts:
            artifacts.append(f"- {artifact.type}: {artifact.path}")
        return "\n".join(artifacts)

# Base agent definitions (HOW to work)
# /home/mubarkahimself/Desktop/QUANTMINDX/src/agents/.analyst/agent.md
# /home/mubarkahimself/Desktop/QUANTMINDX/src/agents/.quantcode/agent.md

# When spawning a worker, combine base + overlay
async def spawn_with_overlay(
    base_agent_path: str,
    task: Task,
    handoff_context: Optional[HandoffContext]
) -> str:
    """Spawn agent with combined base + overlay instructions"""

    # Read base agent instructions
    base_instructions = Path(base_agent_path).read_text()

    # Generate overlay
    overlay = DepartmentAgentOverlay(
        base_agent=base_agent_path,
        task=task,
        handoff_context=handoff_context
    )
    overlay_instructions = overlay.generate_overlay_instructions()

    # Combine
    full_instructions = f"""
{base_instructions}

---

{overlay_instructions}
"""

    # Create worktree with combined CLAUDE.md
    worktree_path = await create_worktree(task.branch)
    claude_md = worktree_path / "CLAUDE.md"
    claude_md.write_text(full_instructions)

    # Spawn agent in worktree
    return await spawn_agent_in_worktree(worktree_path)
```

---

## Recommended Architecture for QUANTMINDX

### Department-Based Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    QUANTMINDX Orchestrator                   │
│                      (Depth 0 - Coordinator)                  │
│                                                               │
│  Responsibilities:                                           │
│  - Department coordination                                   │
│  - Task routing and queuing                                  │
│  - State tracking (ledger)                                   │
│  - Memory management (AgentDB)                               │
│  - DOES NOT execute tasks directly                           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Research Lead │   │  Dev Lead     │   │  Ops Lead     │
│   (Depth 1)   │   │   (Depth 1)   │   │   (Depth 1)   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
   ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
   │         │          │         │          │         │
   ▼         ▼          ▼         ▼          ▼         ▼
Analyst   Researcher  QuantCode  Tester  Executor  Monitor
(Depth 2) (Depth 2)   (Depth 2) (Depth 2) (Depth 2) (Depth 2)
```

### Implementation Roadmap

#### Phase 1: Orchestrator Separation
- Separate coordination from execution in `ClaudeOrchestrator`
- Create `DepartmentOrchestrator` that only queues tasks
- Move subprocess spawning to worker agents

#### Phase 2: Memory Integration
- Integrate existing `src/memory/` system with agent workflows
- Add department-specific namespaces
- Implement LEARN → COORDINATE → EXECUTE → REMEMBER cycle

#### Phase 3: Handoff Protocol
- Implement mediated handoff via orchestrator ledger
- Create artifact passing between departments
- Add handoff state tracking

#### Phase 4: Hierarchy Enforcement
- Add depth limits to agent spawning
- Create agent registry with spawn permissions
- Implement overlay pattern for task-specific instructions

#### Phase 5: MCP Tool Organization
- Organize MCP tools by department
- Create tool discovery registry
- Implement cross-department tool sharing

---

## Comparison: Current vs. Recommended

### Current Architecture (QUANTMINDX)

| Component | Pattern | Concern |
|-----------|---------|---------|
| `ClaudeOrchestrator` | Spawns subprocesses directly | Mixed coordination/execution |
| `SDKOrchestrator` | Direct API calls | Better but still mixed |
| `WorkflowOrchestrator` | Linear pipeline | No department handoffs |
| `BaseAgent` | LangGraph-based | Good foundation |
| `Memory` | Vector search with sqlite-vec | Excellent, needs integration |
| Agent definitions | Static `.md` files | No dynamic overlay |

### Recommended Architecture

| Component | Pattern | Benefit |
|-----------|---------|---------|
| `DepartmentOrchestrator` | Coordination only | Clear separation of concerns |
| `WorkerAgent` | Execution only | Isolated execution contexts |
| `HandoffProtocol` | Mediated via ledger | Trackable state transitions |
| `LearningWorkflow` | LEARN → EXECUTE → REMEMBER | Continuous improvement |
| `AgentOverlay` | Base + dynamic instructions | Task-specific context |
| `MCPRegistry` | Department-based tool discovery | Organized tool access |

---

## Key Takeaways

1. **Separation is Critical**: Orchestrators should NEVER execute - only coordinate
2. **Topology Matters**: Use hierarchical topology for department-based systems
3. **Memory is Learning**: Integrate vector search into every workflow step
4. **Handoffs are Handshakes**: All transitions should be tracked in a ledger
5. **Depth Limits Prevent Chaos**: Enforce hierarchy depth to prevent runaway spawning
6. **Overlays Provide Context**: Combine base instructions with task-specific context
7. **MCP Organizes Tools**: Department-based tool registries enable discovery

---

## References

- **Claude-Flow Repository**: https://github.com/ruvnet/claude-flow
- **Claude-Flow AGENTS.md**: Successfully retrieved via raw.githubusercontent.com
- **QUANTMINDX Memory Architecture**: `/home/mubarkahimself/Desktop/QUANTMINDX/src/memory/ARCHITECTURE.md`
- **QUANTMINDX Agent Analysis**: `/home/mubarkahimself/Desktop/QUANTMINDX/docs/agentic-needs-analysis.md`
