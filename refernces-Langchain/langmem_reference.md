# LangMem Knowledge Reference

> **Source**: [LangMem Documentation](https://langchain-ai.github.io/langmem/)

## Overview

LangMem is LangChain's long-term memory SDK that enables agents to actively manage and learn from their interactions over time.

## Key Features

- 🧩 **Core memory API** - Works with any storage system
- 🧠 **Memory management tools** - Agents can record and search information during active conversations ("in the hot path")
- ⚙️ **Background memory manager** - Automatically extracts, consolidates, and updates agent knowledge
- ⚡ **Native LangGraph integration** - Works with LangGraph's Long-term Memory Store

## Installation

```bash
pip install -U langmem
```

---

## Memory Types

### 1. Semantic Memory (Facts & Knowledge)
Store structured facts, preferences, and relationships as triples.

```python
from langmem import create_memory_manager
from pydantic import BaseModel

class Triple(BaseModel):
    """Store facts, preferences, and relationships as triples."""
    subject: str
    predicate: str
    object: str
    context: str | None = None

manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Triple],
    instructions="Extract user preferences and useful information",
    enable_inserts=True,
    enable_deletes=True,
)
```

**When to use**: Building understanding over time, storing "what", "why", and "how" relationships.

---

### 2. Episodic Memory (Experiences & Learning)
Capture experiences with full chain of reasoning for adaptive learning.

```python
class Episode(BaseModel):
    """Write from agent's perspective with hindsight."""
    observation: str  # The context and setup - what happened
    thoughts: str     # Internal reasoning process "I ..."
    action: str       # What was done, how, in what format
    result: str       # Outcome and retrospective

manager = create_memory_manager(
    "anthropic:claude-3-5-sonnet-latest",
    schemas=[Episode],
    instructions="Extract successful explanations, capture full chain of reasoning.",
    enable_inserts=True,
)
```

**When to use**: 
- Adapt teaching style based on what worked
- Learn from successful problem-solving approaches
- Build library of proven techniques

---

### 3. Procedural Memory (Skills & Instructions)
Store and update agent instructions/prompts based on feedback.

```python
from langmem import create_prompt_optimizer

optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",
)

# Optimize based on conversations
optimized_prompt = optimizer.invoke({"trajectories": conversation_list})

# Store in LangGraph store
store.put(("instructions",), key="agent_instructions", value={"prompt": optimized_prompt})
```

**When to use**: Self-improvement of agent behavior through feedback loops.

---

## Memory Tools

### Basic Setup with Agent

```python
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# Set up storage with embeddings
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create agent with memory capabilities
agent = create_react_agent(
    "anthropic:claude-3-5-sonnet-latest",
    tools=[
        create_manage_memory_tool(namespace=("memories", "{user_id}")),
        create_search_memory_tool(namespace=("memories", "{user_id}")),
    ],
    store=store,
)
```

> **Production Tip**: Use `AsyncPostgresStore` instead of `InMemoryStore` for persistence.

---

## Namespace Patterns

Organize memories by user, organization, or feature:

```python
# Personal memories
namespace=("memories", "user-123")

# Shared team knowledge
namespace=("memories", "team-product")

# Project-specific memories
namespace=("memories", "project-x")

# Agent-specific write, team-wide read
agent_a_tools = [
    create_manage_memory_tool(namespace=("memories", "team_a", "agent_a")),  # Write
    create_search_memory_tool(namespace=("memories", "team_a")),              # Read
]
```

---

## Delayed Processing with ReflectionExecutor

Avoid processing memories on every message:

```python
from langmem import ReflectionExecutor, create_memory_store_manager

# Create memory manager
memory_manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("memories",),
)

# Wrap for deferred processing
executor = ReflectionExecutor(memory_manager)

# In your chat function
def chat(message: str):
    response = llm.invoke(message)
    
    to_process = {
        "messages": [{"role": "user", "content": message}] + [response]
    }
    
    # Wait 30+ minutes before processing
    # Cancels/reschedules if new messages arrive
    executor.submit(to_process, after_seconds=1800)
    
    return response.content
```

**Benefits**:
- Reduces redundant work
- Better context from complete conversations
- Lower token consumption

---

## With Storage (Production)

```python
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

manager = create_memory_store_manager(
    "anthropic:claude-3-5-sonnet-latest",
    namespace=("chat", "{user_id}", "triples"),
    schemas=[Triple],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)
```

---

## Key APIs

| Function | Purpose |
|----------|---------|
| `create_manage_memory_tool` | Tool for agents to store memories |
| `create_search_memory_tool` | Tool for agents to search memories |
| `create_memory_manager` | Extract memories without storage |
| `create_memory_store_manager` | Extract memories with LangGraph storage |
| `create_prompt_optimizer` | Optimize agent prompts from trajectories |
| `ReflectionExecutor` | Defer and deduplicate memory processing |

---

## Next Steps for Project

1. **Use multi-layer namespace strategy** for user, team, and project isolation
2. **Implement all three memory types**: Semantic (facts), Episodic (experiences), Procedural (skills)
3. **Use ReflectionExecutor** for background processing
4. **Integrate with PostgresStore** for production persistence
5. **Build UI** for memory management and visualization
