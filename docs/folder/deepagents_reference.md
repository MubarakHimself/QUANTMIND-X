# DeepAgents SDK Reference

> **IMPORTANT:** The coding agent MUST visit these links to gather actual code patterns and implementations.

---

## Official Documentation

| Topic | URL | What to Look For |
|-------|-----|------------------|
| **Quickstart** | https://docs.langchain.com/oss/python/deepagents/quickstart | Installation, basic agent creation |
| **Customization** | https://docs.langchain.com/oss/python/deepagents/customization | Custom prompts, tools, middleware |
| **Harness** | https://docs.langchain.com/oss/python/deepagents/harness | Filesystem tools, harness capabilities |
| **Backends** | https://docs.langchain.com/oss/python/deepagents/backends | StateBackend, StoreBackend, FilesystemBackend |
| **Long-term Memory** | https://docs.langchain.com/oss/python/deepagents/long-term-memory | Persistent memory, /memories/ path |
| **Subagents** | https://docs.langchain.com/oss/python/deepagents/subagents | task() tool, subagent spawning |
| **HITL** | https://docs.langchain.com/oss/python/deepagents/human-in-the-loop | interrupt_on, approval flows |
| **Middleware** | https://docs.langchain.com/oss/python/deepagents/middleware | before_model, after_model hooks |
| **Blog Post** | https://blog.langchain.com/deep-agents/ | Design philosophy, architecture patterns |

---

## Installation

```bash
pip install deepagents
```

**Note:** As of our codebase scan, `deepagents` is NOT installed in the ROI backend. Add to `requirements.txt`:

```
deepagents>=0.1.0
```

---

## Core Concepts

### 1. Creating a Deep Agent

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="You are a helpful assistant.",
    tools=[my_tool, another_tool],
)

# Invoke
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### 2. Backends (Memory/Filesystem)

DeepAgents use **backends** to handle filesystem operations:

| Backend | Purpose | Persistence |
|---------|---------|-------------|
| `StateBackend` | Default, ephemeral storage | Session only |
| `StoreBackend` | Persistent storage via LangGraph Store | Cross-session |
| `FilesystemBackend` | Access real filesystem | Persistent |
| `CompositeBackend` | Route paths to different backends | Mixed |

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

# Composite: /workspace/* = ephemeral, /memories/* = persistent
backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/memories/": StoreBackend(rt),
    }
)

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="...",
    tools=[],
    backend=backend,
    store=InMemoryStore(),  # Required for StoreBackend
)
```

### 3. Long-term Memory Pattern

DeepAgents recommend storing persistent learnings in `/memories/`:

```
/memories/
├── instructions.md    # Updated SOPs and procedures
├── lessons_learned.md # What worked and what didn't
└── personal_context.md # User preferences (for ROI)
```

The agent reads these on startup and updates them when learning.

### 4. Subagents (task tool)

DeepAgents can spawn subagents for specific tasks:

```python
from deepagents import create_deep_agent

# The agent automatically gets a "task" tool to spawn subagents
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="Break complex tasks into subtasks.",
    tools=[],
    enable_subagents=True,  # Enables task() tool
)

# When invoked, agent can call:
# task(description="Research topic X", tools=["web_search"])
```

### 5. Human-in-the-Loop (HITL)

Configure which tool calls require approval:

```python
agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="...",
    tools=[delete_file, send_email],
    interrupt_on={
        "delete_file": True,  # Always interrupt
        "send_email": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

When a tool call is interrupted:
1. Agent pauses execution
2. User is presented with the tool call for approval
3. User approves, edits, or rejects
4. Agent continues or handles rejection

### 6. Middleware

Add custom logic before/after model calls:

```python
from deepagents.middleware import before_model, after_model

@before_model
def add_context(state, runtime):
    """Add context before LLM call."""
    # Inject recent memories, preferences, etc.
    return {"additional_context": "..."}

@after_model
def analyze_response(state, runtime):
    """Analyze response after LLM call."""
    # Check for suggestions, extract intents, etc.
    return {"suggestion": "..."}

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="...",
    tools=[],
    middleware=[add_context, analyze_response],
)
```

---

## Harness Mode

Harness mode provides extended capabilities:

```python
from deepagents import create_deep_agent, HarnessConfig

harness_config = HarnessConfig(
    enable_todos=True,      # write_todos tool
    enable_subagents=True,  # task() tool
    max_subagents=5,
    extended_context=True,
)

agent = create_deep_agent(
    model="claude-sonnet-4-5-20250929",
    system_prompt="...",
    tools=[],
    harness_config=harness_config,
)
```

### write_todos Tool

In harness mode, agents can maintain a todo list:

```python
# Agent can call:
write_todos([
    {"id": 1, "task": "Research topic", "status": "in_progress"},
    {"id": 2, "task": "Summarize findings", "status": "pending"},
])
```

---

## D.O.E. → DeepAgents Mapping

| D.O.E. Concept | DeepAgents Equivalent |
|----------------|----------------------|
| Directive (SOP files) | `system_prompt` + `/memories/instructions.md` |
| Orchestration (LLM routing) | Main agent logic |
| Execution (Python scripts) | Tools |
| Self-Annealing | Update `/memories/lessons_learned.md` on error |
| Lessons Learned | Persistent `/memories/` via StoreBackend |
| Department Agent | Full DeepAgent instance |
| Subagents | `task()` tool spawns child agents |

---

## ROI-Specific Patterns

### ROI as Personal Assistant

```python
ROI_SYSTEM_PROMPT = """You are ROI, a personal AI assistant.

You delegate complex work to departments but you are NOT the orchestrator
for department internal work. Each department is a full DeepAgent.

Your responsibilities:
- Manage user's productivity (tasks, calendar, meetings)
- Route requests to appropriate departments
- Forward HITL requests from departments
- Provide status updates
"""
```

### Department as Independent DeepAgent

Each department has:
- Its own `create_deep_agent` instance
- Its own memory namespace (`/memories/` isolated)
- Its own subagents via `task()` tool
- Its own HITL rules

```python
def create_department_agent(name, model):
    return create_deep_agent(
        model=model,
        system_prompt=f"You are the {name} Department...",
        tools=load_department_tools(name),
        backend=department_backend(name),
        store=department_store(name),  # Isolated!
    )
```

### Knowledge Base Mounting

Mount read-only knowledge bases via FilesystemBackend:

```python
from deepagents.backends import FilesystemBackend

backend = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={
        "/knowledge/": FilesystemBackend(
            root_dir="/path/to/kb",
            virtual_mode=True,  # Sandboxed
        ),
    }
)
```

---

## Best Practices

1. **Use StoreBackend for persistent memory**: Critical learnings survive restarts
2. **Isolate department memory**: Each department has its own namespace
3. **Configure HITL for destructive actions**: Delete, send, modify external
4. **Use middleware for smart suggestions**: Proactive delegation hints
5. **Enable harness for deep research**: Extended capabilities when needed
6. **Update /memories/ on learning**: Self-annealing pattern
