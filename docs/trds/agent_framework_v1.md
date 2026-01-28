# Technical Requirements Document: QuantMindX Agent Framework

**System Version:** 1.0 (Architecture Draft)
**Target Module:** `src/agents/`
**Goal:** Create a reusable, modular `BaseAgent` class that serves as the foundation for all QuantMind bots (Analyst, QuantCode, Sentinel-AI).

---

## 1. The Core Philosophy
Instead of writing ad-hoc scripts for each agent, we build a **Standardized Framework**.
Every agent in the system is an instance of `QuantMindAgent` (or a subclass), configured with specific **Skills** and **Knowledge Bases**.

### 1.1 The "Class" Structure
The user wants to import an agent anywhere.

```python
from src.agents.core import BaseAgent
from src.agents.skills import CodingSkill, ResearchSkill

# Spawn a coder
quant_coder = BaseAgent(
    name="QuantCode",
    model="google/gemini-2.0-flash-exp",
    skills=[CodingSkill(), MQL5DocsSkill()]
)

# Spawn an analyst
analyst = BaseAgent(
    name="MarketAnalyst",
    model="anthropic/claude-3-5-sonnet",
    skills=[ResearchSkill(), ChartReaderSkill()]
)
```

---

## 2. Key Components

### 2.1 The Agent Core (`src/agents/core/`)
*   **`BaseAgent`:** The wrapper around LangGraph/LangChain.
    *   **Input:** Chat History + User Command.
    *   **Brain:** LLM Router (decides Tool vs Response).
    *   **Output:** Streaming Token / Final Answer.
*   **`AgentState`:** Pydantic model tracking conversation history and "scratchpad".

### 2.2 The Skill System (`src/agents/skills/`)
A **Skill** is a collection of Tools + System Prompts.
*   *Example:* `MQL5Skill`
    *   **Tools:** `read_file`, `write_code`, `compile_mql5`.
    *   **Prompt:** "You are an expert MQL5 developer..."

### 2.3 The Knowledge Orchestrator (`src/agents/knowledge/`)
This is the "Brain" management system.
*   **`KnowledgeBase`:** A specific index (e.g., "Strategy Docs", "MQL5 API").
*   **`KnowledgeRouter`:** Decides *which* KB to query based on the question.
*   **Features:**
    *   **Ingestion:** Scrape -> Chunk -> Embed -> Store (Qdrant).
    *   **Retrieval:** Hybrid Search (Keyword + Vector).

### 2.4 MCP Integration
The Base Agent natively supports MCP Servers.
*   `agent.add_mcp_server("filesystem")`
*   `agent.add_mcp_server("brave-search")`

---

## 3. Directory Structure (Proposed)

```
src/
└── agents/
    ├── __init__.py
    ├── core/
    │   ├── base_agent.py      # The Main Class
    │   ├── memory.py          # Chat History (Redis/Sqlite)
    │   └── router.py          # LLM Decision Logic
    ├── skills/                # Reusable Skill Sets
    │   ├── coding.py
    │   ├── research.py
    │   └── data_analysis.py
    ├── knowledge/             # RAG/Knowledge Base Manager
    │   ├── ingestion.py       # Scraper/Chunker
    │   └── retriever.py       # Vector DB Client
    └── implementations/       # Specific Agent configs
        ├── analyst.py
        └── quant_code.py
```

---

## 4. Discussion Logic (CLI)
To satisfy the "Simple CLI" requirement:

`python -m src.agents.cli start --agent analyst`

This launches an interactive loops that:
1.  Loads the Agent config.
2.  Connects to OpenRouter.
3.  Enters a `while True:` loop for chat.
4.  Streams responses back to stdout.
