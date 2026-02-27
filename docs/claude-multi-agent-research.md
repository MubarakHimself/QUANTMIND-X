# Claude Multi-Agent Systems Research Report

**Research Date:** 2025-02-26
**Researcher:** Claude Agent (Research Specialist)
**Status:** Complete

## Executive Summary

This report synthesizes Anthropic's recommended patterns for multi-agent systems using Claude Agent SDK, Agent Teams, and Sub-agents. The research covers architectural patterns, orchestration strategies, context management, memory systems, and implementation best practices.

---

## 1. Agent Teams Architecture

### Overview
Agent Teams orchestrate multiple Claude Code instances with a team lead coordinating work across teammates with independent context windows.

### Key Components

#### Team Lead
- Coordinates work distribution
- Maintains shared task list
- Routes messages between teammates
- Makes orchestration decisions

#### Teammates
- Independent Claude Code instances
- Isolated context windows (no shared state)
- Specialized capabilities and tools
- Direct inter-agent messaging

#### Architecture Pattern
```
┌─────────────────────────────────────────────────┐
│                   Team Lead                     │
│  - Task Distribution                            │
│  - Message Routing                              │
│  - Orchestration Decisions                      │
└────────┬────────────────────────────────────────┘
         │
    ┌────┴────┬────────────┬────────────┐
    │         │            │            │
┌───▼───┐ ┌──▼───┐    ┌───▼───┐    ┌──▼────┐
│ Mate 1 │ │ Mate 2│    │ Mate 3│    │ Mate 4│
│ Code  │ │Test  │    │Review │    │Docs   │
└───────┘ └──────┘    └───────┘    └───────┘
```

### Configuration

**Environment Variable:**
```bash
CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

**Settings Configuration:**
```json
{
  "teammateMode": "in-process"
}
```

### Communication Patterns

1. **Shared Task List**: Team lead maintains prioritized task queue
2. **Mailbox Messaging**: Direct agent-to-agent communication
3. **Status Broadcasting**: Teammates broadcast availability and progress
4. **Escalation**: Failed tasks escalate to team lead for reassignment

---

## 2. Sub-Agents Implementation

### Definition
Sub-agents are specialized AI assistants with:
- Isolated context windows
- Custom system prompts
- Specific tool access control
- Independent permission modes
- Persistent memory scopes

### YAML Frontmatter Configuration

```yaml
---
name: code-reviewer
description: Expert code review specialist. Proactively reviews code for quality, security, and maintainability. Use immediately after writing or modifying code.
tools: Read, Grep, Glob, Bash
model: inherit
---

You are a senior code reviewer ensuring high standards of code quality and security.

When invoked:
1. Run git diff to see recent changes
2. Focus on modified files
3. Begin review immediately

Review checklist:
- Code is clear and readable
- Functions and variables are well-named
- No duplicated code
- Proper error handling
- No exposed secrets or API keys
- Input validation implemented
- Good test coverage
- Performance considerations addressed
```

### Programmatic Definition (Python SDK)

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async for message in query(
    prompt="Review the authentication module for security issues",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob", "Task"],
        agents={
            "code-reviewer": AgentDefinition(
                description="Expert code review specialist. Use for quality, security, and maintainability reviews.",
                prompt="""You are a code review specialist with expertise in security, performance, and best practices...""",
                tools=["Read", "Grep", "Glob"],
                model="sonnet",
            ),
            "test-runner": AgentDefinition(
                description="Runs and analyzes test suites. Use for test execution and coverage analysis.",
                tools=["Bash", "Read", "Grep"],
            ),
        },
    ),
):
    if hasattr(message, "result"):
        print(message.result)
```

### Permission Modes

1. **inherit**: Use parent agent's permissions (default)
2. **auto**: Subagent auto-approves its own tool use
3. **manual**: All tool use requires explicit approval

### Tool Access Control

Sub-agents can be restricted to specific tools:
```yaml
tools: Read, Grep, Glob  # Whitelist approach
```

This prevents unauthorized access to dangerous tools like Bash or Write.

---

## 3. Context Management

### Session Management

**Session ID Pattern:**
```python
session_id = None

# First query: capture the session ID
async for message in query(
    prompt="Read the authentication module",
    options=ClaudeAgentOptions(allowed_tools=["Read", "Glob"]),
):
    if hasattr(message, "subtype") and message.subtype == "init":
        session_id = message.session_id

# Resume with full context from the first query
async for message in query(
    prompt="Now find all places that call it",
    options=ClaudeAgentOptions(resume=session_id),
):
    if hasattr(message, "result"):
        print(message.result)
```

### Context Compaction

**Auto-Compaction Triggers:**
- Activates at ~95% of context window capacity
- Preserves recent conversation and critical system messages
- Summarizes earlier interactions
- Maintains tool definitions and subagent configurations

**Manual Compaction Boundaries:**
```python
options=ClaudeAgentOptions(
    compaction boundaries=["after_file_read", "before_analysis"]
)
```

### MaxTurns Limits

```python
options=ClaudeAgentOptions(
    max_turns=50  # Prevent infinite loops
)
```

**Stop Reasons:**
- `end_turn`: Agent voluntarily yielded control
- `tool_use`: Agent waiting for tool result
- `max_turns`: Turn limit reached

---

## 4. Hooks and Lifecycle Events

### Hook Types

| Hook Event | Description | Use Cases |
|------------|-------------|-----------|
| PreToolUse | Before tool execution | Security validation, logging |
| PostToolUse | After tool execution | Result processing, metrics |
| UserPromptSubmit | Before sending to Claude | Prompt enhancement, validation |
| SubagentStart | Subagent initialization | Setup, initialization |
| SubagentStop | Subagent termination | Cleanup, result aggregation |
| TeammateIdle | Teammate no work | Task reassignment |
| TaskCompleted | Task finished | Notification, cleanup |

### Security Hook Example

```python
async def validate_bash_command(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Validate and potentially block dangerous bash commands."""
    if input_data["tool_name"] == "Bash":
        command = input_data["tool_input"].get("command", "")
        dangerous_patterns = ["rm -rf /", "mkfs", "dd if=/dev/zero"]

        for pattern in dangerous_patterns:
            if pattern in command:
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Dangerous command blocked: {pattern}",
                    }
                }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(
                matcher="Bash",
                hooks=[validate_bash_command],
                timeout=120
            ),
        ],
    }
)
```

### Logging Hook Example

```python
async def log_tool_use(
    input_data: dict[str, Any],
    tool_use_id: str | None,
    context: HookContext
) -> dict[str, Any]:
    """Log all tool usage for analytics."""
    logger.info({
        "tool": input_data["tool_name"],
        "input": input_data["tool_input"],
        "session_id": context.session_id,
        "timestamp": datetime.now().isoformat(),
    })
    return {}
```

---

## 5. Tool Execution Patterns

### ToolRunner (Automatic Execution)

```typescript
import { betaZodTool } from '@anthropic-ai/sdk/helpers/beta/zod';
import { z } from 'zod';

const weatherTool = betaZdTool({
  name: 'get_weather',
  inputSchema: z.object({
    location: z.string().describe('City and state'),
    unit: z.enum(['celsius', 'fahrenheit']).default('fahrenheit'),
  }),
  run: async ({ location, unit }) => {
    const temp = unit === 'celsius' ? 22 : 72;
    return `The weather in ${location} is ${temp}°${unit === 'celsius' ? 'C' : 'F'}`;
  },
});

const finalMessage = await client.beta.messages.toolRunner({
  model: 'claude-sonnet-4-5-20250929',
  max_tokens: 1024,
  max_iterations: 10,
  tools: [weatherTool, calculatorTool],
  messages: [{ role: 'user', content: 'What is 25 * 4 and what is the weather in NYC?' }],
});
```

### Manual Tool Calling

```python
async def manual_tool_execution():
    """Manually execute tools with custom logic."""
    response = await client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        tools=[weather_tool, calculator_tool],
        messages=[{"role": "user", "content": "What is 25 * 4?"}]
    )

    while response.stop_reason == "tool_use":
        # Execute tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = await execute_tool(block.name, block.input)
                tool_results.append({
                    "tool_use_id": block.id,
                    "content": result
                })

        # Continue conversation
        response = await client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            tools=[weather_tool, calculator_tool],
            messages=[
                {"role": "user", "content": "What is 25 * 4?"},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results}
            ]
        )

    return response.content[-1].text
```

---

## 6. Memory Patterns

### Memory Scopes

| Scope | Description | Persistence |
|-------|-------------|-------------|
| user | Cross-project learning | Permanent |
| project | Project-specific context | Project lifetime |
| local | Task-specific | Task lifetime |

### MEMORY.md Pattern

```markdown
# Project Memory

## User Preferences
- Prefers TypeScript over JavaScript
- Uses tab indentation (4 spaces)
- Follows Airbnb style guide

## Project Conventions
- All API endpoints use `/api/v1/` prefix
- Error responses follow {error: string, details: object} format
- Database queries use ORM, not raw SQL

## Recent Decisions
- 2025-02-20: Migrated from Redux to Zustand for state management
- 2025-02-18: Adopted tRPC for type-safe APIs
```

### Persistent Memory Configuration

```yaml
---
memory:
  scopes: [user, project, local]
  user_path: ~/.claude/memory/user/
  project_path: .claude/memory/project/
  local_path: .claude/memory/local/
---
```

### Cross-Session Learning

Sub-agents can access persistent memory across sessions:

```python
options=ClaudeAgentOptions(
    memory=MemoryConfig(
        user_scope=True,
        project_scope=True,
        local_scope=True
    )
)
```

---

## 7. Orchestration Patterns

### Parallel Research

```
Team Lead
  ├── Research Agent A (Topic 1)
  ├── Research Agent B (Topic 2)
  └── Research Agent C (Topic 3)
```

**Use Case:** Investigate multiple aspects of a problem simultaneously.

### Competing Hypotheses

```
Team Lead
  ├── Proponent Agent (Argument A)
  └── Critic Agent (Argument B)
```

**Use Case:** Explore alternative solutions with adversarial review.

### Sequential Workflow

```
Agent A → Agent B → Agent C → Agent D
```

**Use Case:** Pipeline where each agent transforms output from the previous.

### Delegation Chain

```
Team Lead
  └── Sub-Lead
      ├── Specialist A
      ├── Specialist B
      └── Specialist C
```

**Use Case:** Hierarchical task breakdown and distribution.

---

## 8. Best Practices

### Architecture

1. **Single Responsibility**: Each agent has one clear purpose
2. **Interface Segregation**: Agents only access tools they need
3. **Loose Coupling**: Minimize dependencies between agents
4. **Clear Boundaries**: Well-defined interfaces and protocols

### Security

1. **Principle of Least Privilege**: Restrict tool access by default
2. **Hook Validation**: Use PreToolUse hooks for security checks
3. **Audit Logging**: Log all tool use and agent communications
4. **Permission Modes**: Use appropriate permission modes for each agent

### Performance

1. **Context Management**: Compact proactively, not reactively
2. **Parallel Execution**: Use Agent Teams for concurrent work
3. **Caching**: Cache expensive computations across sessions
4. **Resource Limits**: Set maxTurns and iteration limits

### Reliability

1. **Graceful Degradation**: Handle agent failures without cascade
2. **Idempotent Operations**: Design tool use to be retry-safe
3. **Health Monitoring**: Use hooks for agent health checks
4. **Checkpoint/Resume**: Support session resumption

---

## 9. Model Routing Strategy

Based on BMAD-METHOD configuration:

| Model | Use Cases | Cost |
|-------|-----------|------|
| **Opus** | Complex reasoning, architecture, security analysis, debugging | High |
| **Sonnet** | Implementation, feature development, refactoring, testing | Medium |
| **Haiku** | Quick fixes, simple tasks, boilerplate, file operations | Low |

### Agent-Model Mapping

```yaml
agents:
  coordinator:
    model: opus  # Complex orchestration
  architect:
    model: opus  # System design
  builder:
    model: sonnet  # Implementation
  tester:
    model: sonnet  # Testing
  security:
    model: opus  # Security analysis
  devops:
    model: sonnet  # Deployment
```

---

## 10. Integration Patterns

### Model Context Protocol (MCP)

Extend agent capabilities with external tools:

```typescript
import { Client } from '@modelcontextprotocol/sdk/client/index.js';

const mcpClient = new Client({
  name: "quantmind-agent",
  version: "1.0.0"
});

await mcpClient.connect(transport);
const tools = await mcpClient.listTools();
```

### Overstory Integration

QuantMindX uses Overstory for multi-agent orchestration:

```yaml
overstory:
  agents:
    scout: Read-only exploration
    builder: Implementation
    reviewer: Validation
    merger: Branch merge specialist
    lead: Team lead with sub-worker capability
    supervisor: Per-project supervision
    coordinator: Top-level orchestration
```

---

## 11. Code Examples

### Complete Agent Definition

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition, HookMatcher, MemoryConfig

# Security hook
async def security_validator(input_data, tool_use_id, context):
    if input_data.get("tool_name") == "Bash":
        command = input_data.get("tool_input", {}).get("command", "")
        if "rm -rf" in command and not command.startswith("rm -rf /tmp/"):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Dangerous command blocked"
                }
            }
    return {}

# Main agent configuration
async def run_security_review():
    async for message in query(
        prompt="Conduct a comprehensive security review of the authentication system",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Grep", "Glob", "Bash"],
            max_turns=100,
            memory=MemoryConfig(
                user_scope=True,
                project_scope=True
            ),
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="Bash",
                        hooks=[security_validator],
                        timeout=120
                    )
                ]
            },
            agents={
                "static-analyzer": AgentDefinition(
                    description="Static code analysis specialist",
                    prompt="You are a static analysis expert. Review code for security vulnerabilities...",
                    tools=["Read", "Grep", "Glob"],
                    model="sonnet"
                ),
                "test-runner": AgentDefinition(
                    description="Security test execution specialist",
                    prompt="You are a security testing expert. Run and analyze security tests...",
                    tools=["Bash", "Read"],
                    model="sonnet"
                ),
                "reporter": AgentDefinition(
                    description="Security report writer",
                    prompt="You are a technical writer specializing in security reports...",
                    tools=["Read", "Write"],
                    model="haiku"
                )
            }
        )
    ):
        if hasattr(message, "result"):
            print(message.result)
        elif hasattr(message, "content"):
            print(message.content)
```

---

## 12. Troubleshooting Guide

### Common Issues

**Issue:** Agent stops unexpectedly
- **Cause:** MaxTurns limit reached
- **Solution:** Increase maxTurns or break task into smaller steps

**Issue:** Context window overflow
- **Cause:** Too much conversation history
- **Solution:** Implement proactive compaction boundaries

**Issue:** Subagent can't access required tool
- **Cause:** Tool not in subagent's allowed list
- **Solution:** Add tool to subagent's tools configuration

**Issue:** Hook timeout
- **Cause:** Hook execution exceeded timeout
- **Solution:** Increase hook timeout or optimize hook logic

**Issue:** Memory not persisting
- **Cause:** Memory scopes not configured
- **Solution:** Enable user/project/local scopes in options

---

## 13. References and Resources

### Official Documentation
- [Agent Teams](https://code.claude.com/docs/en/agent-teams)
- [Sub-Agents](https://code.claude.com/docs/en/sub-agents)
- [Claude Agent SDK](https://code.claude.com/docs/en/claude-agent-sdk)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### SDK Libraries
- Python: `claude-agent-sdk`
- TypeScript: `@anthropic-ai/sdk`

### Related Projects
- [Overstory](https://github.com/jayminwest/overstory) - Multi-agent orchestration
- [Mulch](https://github.com/jayminwest/mulch) - Structured expertise management

---

## 14. Conclusion

Anthropic's multi-agent patterns provide a robust framework for building sophisticated AI systems. Key takeaways:

1. **Agent Teams** for parallel work with independent contexts
2. **Sub-agents** for specialized capabilities with tool restrictions
3. **Context Management** through session resumption and compaction
4. **Hooks** for security, logging, and lifecycle management
5. **Memory Scopes** for cross-session learning
6. **Orchestration Patterns** for different collaboration models

The QuantMindX project can leverage these patterns to build a comprehensive quantitative trading system with specialized agents for research, backtesting, execution, and risk management.

---

**Research Completed:** 2025-02-26
**Next Steps:** Apply these patterns to QuantMindX agent architecture
