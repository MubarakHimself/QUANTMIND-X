# QuantMind Agent Factory System

This document describes the factory-based agent initialization system implemented in QuantMindX.

## Overview

The Agent Factory system provides a modern, configurable approach to creating and managing LangGraph agents with:

- **Configuration**: YAML-based agent configuration
- **Dependency Injection**: Centralized resource management
- **Observability**: Built-in metrics, logging, and lifecycle tracking
- **Hot Reload**: Configuration changes without restart

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Factory                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │   Config     │  │   Factory    │  │    Registry         │ │
│  │  (YAML)      │→ │              │→ │   (Singleton)       │ │
│  └──────────────┘  └──────────────┘  └──────────────────────┘ │
│                            │                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Dependency Container                           │ │
│  ├──────────┬──────────┬──────────┬──────────┬─────────────┤ │
│  │LLM Provider│Tool Reg │Checkpointer│Metrics   │ Observers  │ │
│  └──────────┴──────────┴──────────┴──────────┴─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Create an Agent from YAML

```python
from src.agents.factory import create_agent_from_yaml

# Create agent from configuration file
agent = create_agent_from_yaml("config/agents/analyst.yaml")

# Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze EURUSD"}]
})
```

### 2. Create an Agent Programmatically

```python
from src.agents.config import AgentConfig
from src.agents.factory import create_agent

config = AgentConfig(
    agent_id="my_analyst",
    agent_type="analyst",
    name="My Analyst",
    llm_model="anthropic/claude-sonnet-4",
    temperature=0.0,
)

agent = create_agent(config)
```

### 3. Use Convenience Functions

```python
from src.agents.analyst_v2 import create_analyst_agent
from src.agents.quantcode_v2 import create_quantcode_agent
from src.agents.copilot_v2 import create_copilot_agent
from src.agents.router import create_router_agent

# Create specific agent types
analyst = create_analyst_agent(agent_id="analyst_001")
quantcode = create_quantcode_agent(agent_id="quantcode_001")
```

## Configuration

### YAML Configuration

```yaml
agent_id: "analyst_001"
agent_type: "analyst"
name: "Analyst Agent"

# LLM Configuration
llm_provider: "openrouter"
llm_model: "anthropic/claude-sonnet-4"
temperature: 0.0
max_tokens: 4096

# Tool Configuration
tools:
  - research_market_data
  - parse_nprd
  - generate_trd

# State and Checkpointing
state_class: "MessagesState"
checkpointer_type: "memory"

# Observability
enable_tracing: true
enable_metrics: true
enable_logging: true
```

### Configuration Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_id` | string | required | Unique identifier |
| `agent_type` | string | required | Type: analyst, quantcode, copilot, router |
| `name` | string | required | Display name |
| `llm_provider` | string | "openrouter" | LLM provider |
| `llm_model` | string | agent default | Model identifier |
| `temperature` | float | 0.0 | LLM temperature |
| `max_tokens` | int | 4096 | Max tokens |
| `tools` | list | [] | Tool names |
| `state_class` | string | "MessagesState" | State class |
| `checkpointer_type` | string | "memory" | memory, postgres, redis |
| `enable_tracing` | bool | true | Enable LangSmith tracing |
| `enable_metrics` | bool | true | Enable Prometheus metrics |
| `enable_logging` | bool | true | Enable structured logging |

## Observability

### Metrics

The system tracks:

- `agent_creations_total` - Agent creation count
- `agent_invocations_total` - Invocation count by status
- `agent_invocation_duration_seconds` - Invocation latency
- `agent_tool_calls_total` - Tool call count

### Logging

Structured logging with event types:

- `agent_created` - Agent initialization
- `invocation_start/complete/error` - Invocation lifecycle
- `tool_call` - Tool execution
- `agent_error` - Error events

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents` | List all agents |
| POST | `/api/agents` | Create new agent |
| GET | `/api/agents/{id}` | Get agent details |
| GET | `/api/agents/{id}/metrics` | Get agent metrics |
| GET | `/api/agents/{id}/health` | Get health status |
| POST | `/api/agents/{id}/invoke` | Invoke agent |
| PUT | `/api/agents/{id}/config` | Update config |
| DELETE | `/api/agents/{id}` | Delete agent |
| GET | `/api/agents/health` | Health for all agents |

## Migration Guide

### Legacy → Factory Pattern

**Before (Legacy)**:
```python
from src.agents.analyst_v2 import compile_analyst_graph

graph = compile_analyst_graph()
result = graph.invoke(initial_state)
```

**After (Factory)**:
```python
from src.agents.analyst_v2 import create_analyst_agent

agent = create_analyst_agent()
result = agent.invoke({"messages": [...]})
```

The legacy functions remain available for backward compatibility.

## Testing

Run tests with:

```bash
pytest tests/agents/test_config.py -v
pytest tests/agents/ -v
```

## Dependencies

Required packages:
- `langchain-core`
- `langgraph`
- `prometheus-client`
- `pyyaml>=6.0`
- `watchdog>=3.0.0` (for hot reload)

## File Structure

```
src/agents/
├── config.py              # AgentConfig class
├── di_container.py        # Dependency injection
├── tool_registry.py       # Tool registration
├── factory.py             # Agent factory
├── compiled_agent.py     # Agent wrapper
├── observer.py            # Observer base class
├── metrics_collector.py  # Metrics tracking
├── registry.py            # Agent registry
├── health.py              # Health checks
├── config_watcher.py     # Hot reload
├── observers/
│   ├── prometheus_observer.py
│   ├── logging_observer.py
│   └── websocket_observer.py
├── analyst_v2.py         # Analyst agent
├── quantcode_v2.py       # QuantCode agent
├── copilot_v2.py         # Copilot agent
└── router.py             # Router agent

config/agents/
├── analyst.yaml
├── quantcode.yaml
├── copilot.yaml
└── router.yaml
```
