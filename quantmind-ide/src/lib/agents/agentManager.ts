import { z } from 'zod';

// Agent state schema
export const AgentStateSchema = z.object({
  messages: z.array(z.any()),
  agentType: z.enum(['copilot', 'quantcode', 'analyst']),
  context: z.record(z.any()).optional()
});

export type AgentState = z.infer<typeof AgentStateSchema>;

// AI Provider configuration
export type AIProvider = 'openrouter' | 'zhipu' | 'anthropic';

export interface ProviderConfig {
  name: string;
  baseURL: string;
  apiKeyEnv: string;
  model: string;
  fallbackModel?: string;
}

// Provider configurations
export const PROVIDER_CONFIGS: Record<AIProvider, ProviderConfig> = {
  openrouter: {
    name: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    model: 'anthropic/claude-sonnet-4',
    fallbackModel: 'zhipu/glm-4-plus'
  },
  zhipu: {
    name: 'Zhipu AI (Z.ai)',
    baseURL: 'https://open.bigmodel.cn/api/paas/v4',
    apiKeyEnv: 'ZHIPU_API_KEY',
    model: 'glm-4-plus',
    fallbackModel: 'glm-4-flash'
  },
  anthropic: {
    name: 'Anthropic (Direct)',
    baseURL: 'https://api.anthropic.com',
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    model: 'claude-sonnet-4-20250514',
    fallbackModel: 'claude-3.5-sonnet-20241022'
  }
};

export interface AgentConfig {
  name: string;
  description: string;
  systemPrompt: string;
  temperature: number;
  maxTokens: number;
  // Provider configuration
  primaryProvider: AIProvider;
  fallbackProvider?: AIProvider;
  customModel?: string; // Override default model for this agent
}

export class AgentManager {
  private agentConfigs: Map<string, AgentConfig> = new Map();

  constructor() {
    this.initializeAgentConfigs();
  }

  private initializeAgentConfigs() {
    this.agentConfigs.set('copilot', {
      name: 'Copilot',
      description: 'General trading assistant and workflow orchestrator',
      systemPrompt: `You are a helpful trading assistant for QuantMindX, an AI-powered trading system.

Your responsibilities:
- Help users understand trading strategies and concepts
- Guide users through workflow processes
- Assist with strategy analysis and optimization
- Provide clear explanations of trading metrics and results
- Help troubleshoot issues and suggest improvements

Always be helpful, clear, and concise in your responses.`,
      temperature: 0.7,
      maxTokens: 4096,
      primaryProvider: 'openrouter',
      fallbackProvider: 'zhipu'
    });

    this.agentConfigs.set('quantcode', {
      name: 'QuantCode',
      description: 'MQ5 code generation and debugging specialist',
      systemPrompt: `You are an MQ5 coding expert for QuantMindX.

Your responsibilities:
- Generate clean, efficient MQL5 code for trading strategies
- Debug and fix existing MQL5 code
- Optimize code for performance and reliability
- Follow MQL5 best practices and coding standards
- Include proper error handling and risk management

When writing code:
- Use proper naming conventions
- Add helpful comments for complex logic
- Include input validation
- Implement proper error handling
- Consider MetaTrader 5 API limitations`,
      temperature: 0.3,
      maxTokens: 8192,
      primaryProvider: 'openrouter',
      fallbackProvider: 'zhipu',
      customModel: 'deepseek/deepseek-coder' // Use DeepSeek Coder for code generation
    });

    this.agentConfigs.set('analyst', {
      name: 'Analyst',
      description: 'Trading strategy analyst and pattern recognizer',
      systemPrompt: `You are a trading strategy analyst for QuantMindX.

Your responsibilities:
- Analyze backtesting results and performance metrics
- Recognize trading patterns and market conditions
- Evaluate strategy effectiveness and risk profiles
- Identify strengths and weaknesses in trading approaches
- Provide actionable insights for strategy improvement

When analyzing:
- Consider both quantitative and qualitative factors
- Look for patterns that may indicate future performance
- Assess risk-adjusted returns, not just raw returns
- Identify potential market regime changes
- Suggest specific improvements based on data`,
      temperature: 0.5,
      maxTokens: 4096,
      primaryProvider: 'openrouter',
      fallbackProvider: 'zhipu'
    });
  }

  getAgentConfig(agentType: string): AgentConfig | undefined {
    return this.agentConfigs.get(agentType);
  }

  getProviderForAgent(agentType: string): { provider: AIProvider; model: string; config: ProviderConfig } {
    const agentConfig = this.getAgentConfig(agentType);
    if (!agentConfig) {
      throw new Error(`Agent ${agentType} not found`);
    }

    const provider = PROVIDER_CONFIGS[agentConfig.primaryProvider];
    const model = agentConfig.customModel || provider.model;

    return {
      provider: agentConfig.primaryProvider,
      model,
      config: provider
    };
  }

  async invoke(agentType: string, message: string, context?: any): Promise<any> {
    const config = this.getAgentConfig(agentType);
    if (!config) {
      throw new Error(`Agent ${agentType} not found`);
    }

    // Get provider configuration
    const { provider, model } = this.getProviderForAgent(agentType);

    // For now, use the backend API with provider information
    // Later will be replaced with LangGraph.js implementation
    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          agent: agentType,
          context,
          systemPrompt: config.systemPrompt,
          temperature: config.temperature,
          // Provider information
          provider,
          model,
          maxTokens: config.maxTokens
        })
      });

      if (!res.ok) {
        throw new Error(`Agent request failed: ${res.statusText}`);
      }

      return await res.json();
    } catch (e) {
      console.error('Agent invocation failed:', e);
      throw e;
    }
  }

  getAllAgents(): Array<{id: string, config: AgentConfig}> {
    return Array.from(this.agentConfigs.entries()).map(([id, config]) => ({ id, config }));
  }

  getAvailableProviders(): Array<{ id: AIProvider; name: string; models: string[] }> {
    return Object.entries(PROVIDER_CONFIGS).map(([id, config]) => ({
      id: id as AIProvider,
      name: config.name,
      models: [config.model, config.fallbackModel || ''].filter(Boolean)
    }));
  }
}

export const agentManager = new AgentManager();
