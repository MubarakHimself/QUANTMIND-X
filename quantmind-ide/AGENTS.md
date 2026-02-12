# QuantMindX Agent Configuration

## Project Overview
QuantMindX is an AI-powered trading system with multi-agent architecture using LangChain/LangGraph.

## Development Environment
- **Package Manager**: npm
- **Framework**: SvelteKit + LangGraph.js
- **Language**: TypeScript (strict mode)

## Key Commands
```bash
npm run dev              # Start development server
npm test                 # Run tests
npm run lint             # Run ESLint
```

## AI Provider Configuration

### Supported Providers

QuantMindX agents support multiple AI providers via OpenRouter and direct provider APIs:

1. **OpenRouter** (Primary - Recommended)
   - Provides access to multiple models through a single API
   - Models: Claude (Anthropic), GPT-4, Gemini, Llama, DeepSeek, Qwen, etc.
   - API Key: Set `OPENROUTER_API_KEY` in environment
   - Base URL: `https://openrouter.ai/api/v1`

2. **Z.ai (Zhipu AI)**
   - Chinese AI provider with GLM models
   - Models: glm-4-flash, glm-4-plus, glm-4-0520, etc.
   - API Key: Set `ZHIPU_API_KEY` in environment
   - Base URL: `https://open.bigmodel.cn/api/paas/v4`

3. **Anthropic (Direct)**
   - Claude models directly from Anthropic
   - Models: claude-sonnet-4, claude-3.5-sonnet, etc.
   - API Key: Set `ANTHROPIC_API_KEY` in environment
   - Use for: Highest quality Claude model access

### Model Configuration by Agent

| Agent | Primary Provider | Model | Fallback |
|-------|-----------------|-------|----------|
| **Copilot** | OpenRouter | anthropic/claude-sonnet-4 | zhipu/glm-4-plus |
| **QuantCode** | OpenRouter | deepseek/deepseek-coder | zhipu/glm-4-flash |
| **Analyst** | OpenRouter | anthropic/claude-sonnet-4 | zhipu/glm-4-plus |

### Environment Variables

```bash
# OpenRouter (Primary)
OPENROUTER_API_KEY=your_openrouter_key

# Zhipu AI (Z.ai)
ZHIPU_API_KEY=your_zhipu_key

# Anthropic (Optional - for direct access)
ANTHROPIC_API_KEY=your_anthropic_key
```

### Provider Priority

The agent system will try providers in this order:
1. OpenRouter (most flexible, best routing)
2. Z.ai (cost-effective, fast)
3. Anthropic direct (fallback for Claude models)

## Coding Conventions
- Use TypeScript for all new code (.ts/.svelte)
- Place agents in `/src/lib/agents/`
- Each agent should export a `graph` and `config` object
- Use Zod for state schema validation

## Agent Instructions

### Copilot Agent
- General trading assistance
- Strategy analysis and optimization
- User guidance and workflow orchestration
- **Provider**: Use OpenRouter with Claude Sonnet 4 for best reasoning
- **Fallback**: Z.ai GLM-4 Plus for cost efficiency

### QuantCode Agent
- MQ5 code generation and debugging
- Strategy implementation
- Parameter optimization
- **Provider**: Use OpenRouter with DeepSeek Coder for code generation
- **Fallback**: Z.ai GLM-4 Flash for fast iterations

### Analyst Agent
- NPRD output analysis
- Trading pattern recognition
- Performance evaluation
- **Provider**: Use OpenRouter with Claude Sonnet 4 for analysis
- **Fallback**: Z.ai GLM-4 Plus for detailed reports

## Memory Management
- Use custom memory implementation (LangMem TS not available)
- Namespace memory by agent type
- Implement proper cleanup
- Store provider preference per conversation

## MCP Integration
- MCP tools in `/src/lib/mcp/`
- Each tool extends StructuredTool
- Register in agent configuration
- Include provider-specific tools when available
