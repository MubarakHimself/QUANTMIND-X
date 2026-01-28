# LangChain Ecosystem Reference for AI Agents

> **IMPORTANT:** The LLM should visit each link below to gather actual code snippets and API patterns.

## Official Documentation Links

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Tools | https://docs.langchain.com/oss/python/langchain/tools | @tool decorator, schemas, ToolRuntime |
| Agents | https://docs.langchain.com/oss/python/langchain/agents | create_agent, ReAct, system prompts |
| Multi-Agent | https://docs.langchain.com/oss/python/langchain/multi-agent/index | Subagents, handoffs, skills, router |
| Subagents | https://docs.langchain.com/oss/python/langchain/multi-agent/subagents | Wrap subagent as tool, parallel execution |
| Skills Pattern | https://docs.langchain.com/oss/python/langchain/multi-agent/skills | load_skill tool, dynamic registration |
| RAG | https://docs.langchain.com/oss/python/langchain/rag | Indexing, retrieval, embeddings |
| Context Engineering | https://docs.langchain.com/oss/python/langchain/context-engineering | Model context, tool context, life-cycle |
| Agent Chat UI | https://docs.langchain.com/oss/python/langchain/ui | npx create-agent-chat-app |
| Studio | https://docs.langchain.com/oss/python/langchain/studio | langgraph dev, visual debugging |

---

## DeepAgents Documentation

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Overview | https://docs.langchain.com/oss/python/deepagents/overview | Core capabilities, planning patterns |
| Harness | https://docs.langchain.com/oss/python/deepagents/harness | Filesystem tools, storage backends, subagents |
| Customization | https://docs.langchain.com/oss/python/deepagents/customization | Custom prompts, tools, agents |
| Middleware | https://docs.langchain.com/oss/python/deepagents/middleware | TodoListMiddleware, FilesystemMiddleware |
| Blog Post | https://blog.langchain.com/deep-agents/ | Design philosophy, patterns |

---

## MCP (Model Context Protocol)

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Build Server | https://modelcontextprotocol.io/docs/develop/build-server | FastMCP, @mcp.tool(), logging |
| Build Client | https://modelcontextprotocol.io/docs/develop/build-client | ClientSession, StdioServerParameters |
| MCP Specification | https://modelcontextprotocol.io/docs/specification | Full protocol spec |

---

## Anthropic Skills

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Overview | https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview | SKILL.md format, progressive disclosure |
| Best Practices | https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices | Token budgets, naming, patterns |
| Skill Creator | https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md | Full skill creation workflow |

---

## Open Deep Research

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Blog Post | https://blog.langchain.com/open-deep-research/ | 3-phase architecture, multi-agent patterns |
| GitHub | https://github.com/langchain-ai/open_deep_research | Implementation, configuration |

---

## Additional Resources

| Topic | URL | What to Look For |
|-------|-----|------------------|
| Custom Agent + Plugin Retrieval | https://python.langchain.com.cn/docs/use_cases/agents/custom_agent_with_plugin_retrieval | Tool retrieval pattern, vector embeddings |
| Docker MCP Toolkit | (Search) | MCP Catalog, Gateway configuration |

---

## Prebuilt Components

| Component | Package | Use Case |
|-----------|---------|----------|
| `create_agent` | `langchain` | Single agent with tools |
| `create_deep_agent` | `deepagents` | Autonomous agent with filesystem |
| `create_react_agent` | `langgraph.prebuilt` | ReAct pattern agent |
| `create_manage_memory_tool` | `langmem` | Memory management |
| `create_search_memory_tool` | `langmem` | Memory retrieval |

---

## Agent Chat UI & Studio

### Agent Chat UI
**Purpose**: Ready-made chat interface.
```bash
npx create-agent-chat-app --project-name my-chat-ui
# Connect to: http://localhost:2024
# Graph ID: agent (from langgraph.json)
```

### LangSmith Studio
**Purpose**: Visual debugging.
```bash
langgraph dev
# Opens: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

---

## RAG Pipeline Example

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# Load documents
loader = WebBaseLoader(url)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)

# Store in pgvector
embeddings = OpenAIEmbeddings()
vectorstore = PGVector.from_documents(chunks, embeddings, connection=CONNECTION_STRING)

# Retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.get_relevant_documents(query)
```

---

## Agent Creation Pattern

```python
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Initialize model
model = init_chat_model("anthropic:claude-3-5-sonnet")

# Define tools
@tool
def search_database(query: str) -> str:
    """Search the database for records."""
    return f"Found results for '{query}'"

# Create agent
agent = create_react_agent(
    model=model,
    tools=[search_database],
    prompt="You are a helpful assistant..."
)

# Invoke
result = agent.invoke({"messages": [{"role": "user", "content": "Search for X"}]})
```

---

## Best Practices

1. **Tool Descriptions**: Clear, specific descriptions for LLM decision-making
2. **Schema Validation**: Use Pydantic models for tool inputs
3. **Error Handling**: Return actionable error messages
4. **Streaming**: Use `stream_mode="messages"` for token streaming
5. **Checkpointing**: Enable for multi-turn conversations
6. **Memory**: Use LangMem for cross-session persistence
