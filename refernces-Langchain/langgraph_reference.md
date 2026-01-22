# LangGraph Reference for AI Agents

## Official Documentation Links

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Overview | https://docs.langchain.com/oss/python/langgraph/overview | Core benefits, ecosystem |
| Graph API | https://docs.langchain.com/oss/python/langgraph/graph-api | StateGraph, nodes, edges |
| Application Structure | https://docs.langchain.com/oss/python/langgraph/application-structure | File layout, langgraph.json |
| Workflows & Agents | https://docs.langchain.com/oss/python/langgraph/workflows-agents | Patterns: routing, orchestrator-worker |
| Memory | https://docs.langchain.com/oss/python/langgraph/add-memory | Short-term, long-term memory |
| Streaming | https://docs.langchain.com/oss/python/langgraph/streaming | Real-time responses |

---

## Patterns to Adopt

### 1. State Definition
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-accumulates messages
```

**Why**: The `add_messages` annotation automatically handles message accumulation.

### 2. Graph Construction
```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(AgentState)
builder.add_node("node_name", node_function)
builder.add_edge(START, "node_name")
builder.add_edge("node_name", END)
graph = builder.compile()
```

**Why**: Explicit graph construction is more maintainable than implicit chains.

### 3. Node Functions
```python
def my_node(state: AgentState) -> dict:
    # Process state
    return {"messages": [new_message]}  # Return partial state update
```

**Why**: Nodes return dicts that update specific state keys.

### 4. Conditional Edges
```python
def router(state: AgentState) -> str:
    if condition:
        return "node_a"
    return "node_b"

builder.add_conditional_edges("source", router, ["node_a", "node_b"])
```

**Why**: Enables dynamic routing based on state.

---

## Project Structure (Standard)

```
my-app/
├── my_agent/
│   ├── utils/
│   │   ├── tools.py      # Tool definitions
│   │   ├── nodes.py      # Node functions
│   │   └── state.py      # State definition
│   └── agent.py          # Graph construction
├── .env
├── pyproject.toml
└── langgraph.json
```

---

## langgraph.json Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent_name": "./path/to/agent.py:graph_variable"
  },
  "env": ".env"
}
```

---

## Development Commands

```bash
# Install CLI
pip install "langgraph-cli[inmem]"

# Start dev server
langgraph dev

# Access Studio
# Opens: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## Watch Out For

1. **State mutations**: Always return new state dicts, don't mutate in place
2. **Message format**: Use proper role/content structure
3. **Graph compilation**: Call `.compile()` after building
4. **Async vs sync**: Use `ainvoke` for async, `invoke` for sync
