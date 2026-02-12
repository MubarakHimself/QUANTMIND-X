import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import { BaseStore } from "@langchain/langgraph";

// ============================================================================
// MEMORY TYPES - LangMem-inspired patterns
// ============================================================================

/**
 * Semantic Memory (Facts & Knowledge)
 * Store structured facts, preferences, and relationships as triples.
 */
export interface SemanticMemory {
  subject: string;    // The entity (e.g., "user", "EURUSD", "strategy")
  predicate: string;  // The relationship (e.g., "prefers", "trend", "uses")
  object: string;     // The value (e.g., "scalping", "bullish", "Kelly")
  context?: string;   // Additional context or metadata
  timestamp?: number; // When this fact was recorded
}

/**
 * Episodic Memory (Experiences & Learning)
 * Capture experiences with full chain of reasoning for adaptive learning.
 */
export interface EpisodicMemory {
  observation: string;  // The context and setup - what happened
  thoughts: string;     // Internal reasoning process "I ..."
  action: string;       // What was done, how, in what format
  result: string;       // Outcome and retrospective
  timestamp?: number;   // When this episode occurred
}

/**
 * Procedural Memory (Skills & Instructions)
 * Store and update agent instructions/prompts based on feedback.
 */
export interface ProceduralMemory {
  name: string;         // Name of the skill or instruction
  instructions: string; // The actual prompt/instruction
  version: number;      // Version for tracking updates
  timestamp?: number;   // When this was last updated
}

// ============================================================================
// NAMESPACE STRATEGY - Memory isolation per agent and session
// ============================================================================

export type AgentType = 'copilot' | 'quantcode' | 'analyst';
export type MemoryNamespace = ["memories", AgentType] | ["memories", "session", string];

export const MemoryNamespaces = {
  // Per-agent namespaces
  copilot: ["memories", "copilot"] as MemoryNamespace,
  quantcode: ["memories", "quantcode"] as MemoryNamespace,
  analyst: ["memories", "analyst"] as MemoryNamespace,

  // Session-based namespace
  session: (sessionId: string): MemoryNamespace => ["memories", "session", sessionId],

  // Memory type suffixes
  semantic: "semantic",
  episodic: "episodic",
  procedural: "procedural",
} as const;

/**
 * Generate a full memory key for storage
 */
export function getMemoryKey(
  namespace: MemoryNamespace,
  memoryType: keyof typeof MemoryNamespaces,
  key: string
): string {
  return `${namespace.join("/")}/${memoryType}/${key}`;
}

// ============================================================================
// ENHANCED STATE GRAPH - With context and memory support
// ============================================================================

/**
 * Enhanced Agent State with memory and context support
 */
const StateAnnotation = Annotation.Root({
  // Message history with automatic accumulation
  messages: Annotation<{
    reducer: (a: any[], b: any[]) => a.concat(b);
    default: () => [];
  }>(),

  // Agent type for routing and context
  agentType: Annotation<AgentType | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Current session ID for memory isolation
  sessionId: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Memory namespace for this interaction
  memoryNamespace: Annotation<MemoryNamespace | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Semantic memories retrieved for this context
  semanticMemories: Annotation<SemanticMemory[]>({
    reducer: (_, b) => b,
    default: () => [],
  }),

  // Episodic memories (experiences) retrieved for this context
  episodicMemories: Annotation<EpisodicMemory[]>({
    reducer: (_, b) => b,
    default: () => [],
  }),

  // Procedural memories (instructions) for this agent
  proceduralMemories: Annotation<ProceduralMemory[]>({
    reducer: (_, b) => b,
    default: () => [],
  }),

  // Additional context metadata
  metadata: Annotation<Record<string, any>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
});

export type AgentState = typeof StateAnnotation.State;

// Provider configurations
export interface ProviderConfig {
  name: string;
  baseURL: string;
  apiKeyEnv: string;
  model: string;
  headers?: Record<string, string>;
}

export const PROVIDER_CONFIGS: Record<'openrouter' | 'zhipu' | 'anthropic', ProviderConfig> = {
  openrouter: {
    name: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    model: 'anthropic/claude-sonnet-4',
    headers: {
      'HTTP-Referer': 'https://quantmindx.local',
      'X-Title': 'QuantMindX Trading Platform',
    },
  },
  zhipu: {
    name: 'Zhipu AI',
    baseURL: 'https://openrouter.ai/api/v1',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    model: 'zhipu/glm-4-plus',
  },
  anthropic: {
    name: 'Anthropic',
    baseURL: 'https://api.anthropic.com',
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    model: 'claude-sonnet-4-20250514',
  },
};

// Create a model with the specified provider
export function createModel(
  provider: 'openrouter' | 'zhipu' | 'anthropic',
  modelOverride?: string,
  temperature = 0.7
) {
  const config = PROVIDER_CONFIGS[provider];
  const apiKey = typeof process !== 'undefined' ? process.env[config.apiKeyEnv] : '';

  return new ChatOpenAI(
    {
      model: modelOverride || config.model,
      temperature,
      apiKey,
      streaming: true,
    },
    {
      baseURL: config.baseURL,
      defaultHeaders: config.headers,
    }
  );
}

// Define tools for QuantMindX agents
export const createQuantMindXTools = () => {
  const getMarketData = tool(
    async ({ symbol, timeframe }: { symbol: string; timeframe: string }) => {
      // This would connect to the QuantMindX backend
      return `Market data for ${symbol} on ${timeframe}: Bid=1.0850, Ask=1.0852, Spread=2 pips`;
    },
    {
      name: 'get_market_data',
      description: 'Get current market data for a trading symbol. Returns bid, ask, and spread information.',
      schema: z.object({
        symbol: z.string().describe('Trading symbol (e.g., EURUSD, XAUUSD)'),
        timeframe: z.string().describe('Timeframe (e.g., M1, M5, H1, D1)'),
      }),
    }
  );

  const runBacktest = tool(
    async ({ strategy, symbol, period }: { strategy: string; symbol: string; period: string }) => {
      // This would trigger a backtest via the QuantMindX backend
      return `Backtest initiated for ${strategy} on ${symbol} for ${period}`;
    },
    {
      name: 'run_backtest',
      description: 'Run a backtest for a strategy on a symbol and time period.',
      schema: z.object({
        strategy: z.string().describe('Strategy name or ID'),
        symbol: z.string().describe('Trading symbol'),
        period: z.string().describe('Time period (e.g., 1M, 3M, 1Y)'),
      }),
    }
  );

  const getPositionSize = tool(
    async ({ balance, risk, stopLoss, takeProfit }: {
      balance: number;
      risk: number;
      stopLoss: number;
      takeProfit: number;
    }) => {
      // Kelly Criterion-based position sizing
      const kellyPercent = (risk * 0.8) / stopLoss;
      const positionSize = balance * kellyPercent;
      return `Position size: ${positionSize.toFixed(2)} (Kelly: ${(kellyPercent * 100).toFixed(2)}% of balance)`;
    },
    {
      name: 'get_position_size',
      description: 'Calculate optimal position size using Kelly Criterion for risk management.',
      schema: z.object({
        balance: z.number().describe('Account balance'),
        risk: z.number().describe('Risk per trade (0-1 as percentage)'),
        stopLoss: z.number().describe('Stop loss in pips'),
        takeProfit: z.number().describe('Take profit in pips'),
      }),
    }
  );

  return [getMarketData, runBacktest, getPositionSize];
};

// ============================================================================
// MEMORY TOOLS - LangMem-inspired memory management tools
// ============================================================================

/**
 * Store semantic memory (facts/triples)
 */
export const storeSemanticMemory = tool(
  async ({ subject, predicate, object, context, store, namespace }: {
    subject: string;
    predicate: string;
    object: string;
    context?: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    if (!store) {
      return JSON.stringify({ success: false, error: "No store provided" });
    }

    const memory: SemanticMemory = {
      subject,
      predicate,
      object,
      context,
      timestamp: Date.now(),
    };

    const key = getMemoryKey(
      namespace || ["memories", "copilot"],
      "semantic",
      `${subject}-${predicate}-${object}-${Date.now()}`
    );

    await store.put(namespace || ["memories", "copilot"], key, memory);

    return JSON.stringify({ success: true, key, memory });
  },
  {
    name: "store_semantic_memory",
    description: "Store a semantic memory (fact/triple) for later retrieval. Use this to remember user preferences, trading symbols, strategy details, etc.",
    schema: z.object({
      subject: z.string().describe("The entity (e.g., 'user', 'EURUSD', 'strategy')"),
      predicate: z.string().describe("The relationship (e.g., 'prefers', 'trend', 'uses')"),
      object: z.string().describe("The value (e.g., 'scalping', 'bullish', 'Kelly')"),
      context: z.string().optional().describe("Additional context or metadata"),
    }),
  }
);

/**
 * Search semantic memories
 */
export const searchSemanticMemories = tool(
  async ({ query, store, namespace, limit = 5 }: {
    query: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
    limit?: number;
  }) => {
    if (!store) {
      return JSON.stringify({ success: false, error: "No store provided", memories: [] });
    }

    // Note: In production, you'd use vector search with embeddings
    // For now, we'll do a simple prefix-based search
    const ns = namespace || ["memories", "copilot"];
    const items = await store.list(ns, { limit: limit * 10 });

    const memories: SemanticMemory[] = [];
    for await (const item of items) {
      if (item.value && memories.length < limit) {
        memories.push(item.value as SemanticMemory);
      }
    }

    return JSON.stringify({ success: true, memories, count: memories.length });
  },
  {
    name: "search_semantic_memories",
    description: "Search stored semantic memories (facts/triples). Use this to recall user preferences, trading history, etc.",
    schema: z.object({
      query: z.string().describe("Search query for finding relevant memories"),
      limit: z.number().optional().describe("Maximum number of memories to return (default: 5)"),
    }),
  }
);

/**
 * Store episodic memory (experience)
 */
export const storeEpisodicMemory = tool(
  async ({ observation, thoughts, action, result, store, namespace }: {
    observation: string;
    thoughts: string;
    action: string;
    result: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    if (!store) {
      return JSON.stringify({ success: false, error: "No store provided" });
    }

    const memory: EpisodicMemory = {
      observation,
      thoughts,
      action,
      result,
      timestamp: Date.now(),
    };

    const key = getMemoryKey(
      namespace || ["memories", "copilot"],
      "episodic",
      `episode-${Date.now()}`
    );

    await store.put(namespace || ["memories", "copilot"], key, memory);

    return JSON.stringify({ success: true, key, memory });
  },
  {
    name: "store_episodic_memory",
    description: "Store an episodic memory (experience) with full chain of reasoning. Use this to remember what worked, what didn't, and why.",
    schema: z.object({
      observation: z.string().describe("The context and setup - what happened"),
      thoughts: z.string().describe("Internal reasoning process (what you thought)"),
      action: z.string().describe("What was done, how, in what format"),
      result: z.string().describe("Outcome and retrospective"),
    }),
  }
);

/**
 * Update procedural memory (instructions)
 */
export const updateProceduralMemory = tool(
  async ({ name, instructions, version, store, namespace }: {
    name: string;
    instructions: string;
    version?: number;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    if (!store) {
      return JSON.stringify({ success: false, error: "No store provided" });
    }

    const memory: ProceduralMemory = {
      name,
      instructions,
      version: version || 1,
      timestamp: Date.now(),
    };

    const key = getMemoryKey(
      namespace || ["memories", "copilot"],
      "procedural",
      name
    );

    await store.put(namespace || ["memories", "copilot"], key, memory);

    return JSON.stringify({ success: true, key, memory });
  },
  {
    name: "update_procedural_memory",
    description: "Update procedural memory (instructions/prompts). Use this to improve agent behavior based on feedback.",
    schema: z.object({
      name: z.string().describe("Name of the skill or instruction"),
      instructions: z.string().describe("The actual prompt/instruction to store"),
      version: z.number().optional().describe("Version number (defaults to 1)"),
    }),
  }
);

/**
 * Create memory tools for an agent
 */
export function createMemoryTools(namespace: MemoryNamespace) {
  return [
    storeSemanticMemory,
    searchSemanticMemories,
    storeEpisodicMemory,
    updateProceduralMemory,
  ];
}

// ============================================================================
// MEMORY-AWARE AGENT NODE - Enhanced with memory retrieval
// ============================================================================

/**
 * Enhanced agent node that loads memories before processing
 */
async function createMemoryAwareAgentNode(
  model: any,
  systemPrompt: string,
  agentType: AgentType,
  store?: BaseStore
) {
  return async function agentNode(state: AgentState) {
    // Load relevant memories before processing
    let contextMemories = "";
    if (store && state.memoryNamespace) {
      try {
        // Search for relevant semantic memories
        const searchResult = await searchSemanticMemories.invoke({
          query: state.messages[state.messages.length - 1]?.content?.toString() || "",
          store,
          namespace: state.memoryNamespace,
          limit: 3,
        });

        const { memories } = JSON.parse(searchResult);
        if (memories && memories.length > 0) {
          contextMemories = "\n\nRelevant memories:\n" +
            memories.map((m: SemanticMemory) =>
              `- ${m.subject} ${m.predicate} ${m.object}${m.context ? ` (${m.context})` : ""}`
            ).join("\n");
        }
      } catch (error) {
        console.warn("Failed to load memories:", error);
      }
    }

    // Build messages with system prompt and memories
    const enhancedPrompt = contextMemories
      ? `${systemPrompt}\n\n${contextMemories}`
      : systemPrompt;

    const messages = [
      new SystemMessage(enhancedPrompt),
      ...state.messages
    ];

    const response = await model.invoke(messages);
    return { messages: [response] };
  };
}

// ============================================================================
// QUANTMINDX AGENT CREATION - Enhanced with memory support
// ============================================================================

/**
 * Create a QuantMindX agent graph with memory support
 */
export function createQuantMindXAgentWithMemory(
  agentType: AgentType,
  provider: 'openrouter' | 'zhipu' | 'anthropic' = 'openrouter',
  modelOverride?: string,
  options: {
    store?: BaseStore;
    sessionId?: string;
    enableMemory?: boolean;
  } = {}
) {
  const {
    store,
    sessionId = "default",
    enableMemory = true,
  } = options;

  // Determine memory namespace
  const memoryNamespace: MemoryNamespace = enableMemory
    ? (sessionId === "default"
        ? MemoryNamespaces[agentType]
        : MemoryNamespaces.session(sessionId))
    : ["memories", agentType];

  // Create tools
  const tools = [
    ...createQuantMindXTools(),
    ...(enableMemory && store ? createMemoryTools(memoryNamespace) : []),
  ];

  // Agent system prompts (enhanced with memory awareness)
  const systemPrompts = {
    copilot: `You are a helpful trading assistant for QuantMindX, an AI-powered trading system.

Your responsibilities:
- Help users understand trading strategies and concepts
- Guide users through workflow processes
- Assist with strategy analysis and optimization
- Provide clear explanations of trading metrics and results
- Help troubleshoot issues and suggest improvements

Memory capabilities:
- Store user preferences as semantic memories (triples: subject-predicate-object)
- Remember successful interactions as episodic memories (observation-thoughts-action-result)
- Learn and improve over time by updating procedural memories

When using tools, always explain what you're doing and why.
Store important information about the user, their preferences, and their trading goals.`,

    quantcode: `You are an MQ5 coding expert for QuantMindX.

Your responsibilities:
- Generate clean, efficient MQL5 code for trading strategies
- Debug and fix existing MQL5 code
- Optimize code for performance and reliability
- Follow MQL5 best practices and coding standards
- Include proper error handling and risk management

Memory capabilities:
- Store coding patterns and solutions as semantic memories
- Remember successful debugging approaches as episodic memories
- Maintain procedural memories for common coding tasks

When writing code:
- Use proper naming conventions
- Add helpful comments for complex logic
- Include input validation
- Implement proper error handling
- Consider MetaTrader 5 API limitations`,

    analyst: `You are a trading strategy analyst for QuantMindX.

Your responsibilities:
- Analyze backtesting results and performance metrics
- Recognize trading patterns and market conditions
- Evaluate strategy effectiveness and risk profiles
- Identify strengths and weaknesses in trading approaches
- Provide actionable insights for strategy improvement

Memory capabilities:
- Store market observations as semantic memories
- Remember successful analysis patterns as episodic memories
- Maintain procedural memories for analysis methodologies

When analyzing:
- Consider both quantitative and qualitative factors
- Look for patterns that may indicate future performance
- Assess risk-adjusted returns, not just raw returns
- Identify potential market regime changes
- Suggest specific improvements based on data`,
  };

  const systemPrompt = systemPrompts[agentType];

  // Temperature settings by agent type
  const temperatures = {
    copilot: 0.7,
    quantcode: 0.3,
    analyst: 0.5,
  };

  // Create model with provider
  const model = createModel(provider, modelOverride, temperatures[agentType]).bindTools(tools);

  // Create memory-aware agent node
  const agentNode = createMemoryAwareAgentNode(
    model,
    systemPrompt,
    agentType,
    store
  );

  // Build the graph with enhanced state
  const workflow = new StateGraph(StateAnnotation)
    .addNode('agent', agentNode)
    .addNode('tools', new ToolNode(tools))
    .addEdge('__start__', 'agent')
    .addConditionalEdges('agent', toolsCondition)
    .addEdge('tools', 'agent');

  const compiledGraph = workflow.compile();

  // Return graph with metadata
  return {
    graph: compiledGraph,
    agentType,
    memoryNamespace,
    sessionId,
  };
}

// ============================================================================
// BACKWARD COMPATIBLE WRAPPER - Original createQuantMindXAgent
// ============================================================================

/**
 * Legacy wrapper for backward compatibility
 * @deprecated Use createQuantMindXAgentWithMemory instead
 */
export function createQuantMindXAgent(
  agentType: 'copilot' | 'quantcode' | 'analyst',
  provider: 'openrouter' | 'zhipu' | 'anthropic' = 'openrouter',
  modelOverride?: string
) {
  const { graph } = createQuantMindXAgentWithMemory(agentType, provider, modelOverride, {
    enableMemory: false,
  });
  return graph;
}

// ============================================================================
// ENHANCED MULTI-PROVIDER AGENT MANAGER - With memory support
// ============================================================================

export interface MultiProviderAgentManagerOptions {
  store?: BaseStore;
  sessionId?: string;
  enableMemory?: boolean;
}

/**
 * Enhanced fallback manager for multi-provider support with memory
 */
export class MultiProviderAgentManager {
  private agents: Map<string, any> = new Map();
  private agentType: AgentType;
  private providerOrder: ('openrouter' | 'zhipu' | 'anthropic')[] = ['openrouter', 'zhipu', 'anthropic'];
  private options: MultiProviderAgentManagerOptions;

  constructor(
    agentType: AgentType,
    options: MultiProviderAgentManagerOptions = {}
  ) {
    this.agentType = agentType;
    this.options = options;

    // Initialize agents for each provider
    for (const provider of this.providerOrder) {
      try {
        const agentConfig = createQuantMindXAgentWithMemory(agentType, provider, undefined, options);
        this.agents.set(provider, agentConfig);
      } catch (error) {
        console.warn(`Failed to initialize ${provider} agent:`, error);
      }
    }
  }

  async invoke(message: string, context?: any) {
    const errors = [];

    for (const provider of this.providerOrder) {
      const agentConfig = this.agents.get(provider);
      if (!agentConfig) {
        continue;
      }

      try {
        const result = await agentConfig.graph.invoke(
          {
            messages: [new HumanMessage(message)],
            agentType: this.agentType,
            sessionId: context?.sessionId || this.options.sessionId || "default",
            memoryNamespace: agentConfig.memoryNamespace,
          },
          { configurable: { thread_id: context?.threadId || 'default' } }
        );
        return {
          response: result.messages[result.messages.length - 1].content,
          provider,
          memoryNamespace: agentConfig.memoryNamespace,
        };
      } catch (error) {
        console.warn(`${provider} failed:`, error);
        errors.push({ provider, error });
      }
    }

    throw new Error(`All providers failed: ${JSON.stringify(errors, null, 2)}`);
  }

  /**
   * Stream responses from the agent
   */
  async *stream(message: string, context?: any) {
    for (const provider of this.providerOrder) {
      const agentConfig = this.agents.get(provider);
      if (!agentConfig) {
        continue;
      }

      try {
        yield* agentConfig.graph.stream(
          {
            messages: [new HumanMessage(message)],
            agentType: this.agentType,
            sessionId: context?.sessionId || this.options.sessionId || "default",
            memoryNamespace: agentConfig.memoryNamespace,
          },
          { configurable: { thread_id: context?.threadId || 'default' } }
        );
        return;
      } catch (error) {
        console.warn(`${provider} stream failed:`, error);
      }
    }
  }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create an agent manager with fallback (legacy)
 * @deprecated Use createMemoryEnabledAgentManager instead
 */
export function createAgentManager(
  agentType: 'copilot' | 'quantcode' | 'analyst'
) {
  return new MultiProviderAgentManager(agentType);
}

/**
 * Create a memory-enabled agent manager
 */
export function createMemoryEnabledAgentManager(
  agentType: AgentType,
  options: MultiProviderAgentManagerOptions = {}
) {
  return new MultiProviderAgentManager(agentType, options);
}

/**
 * Create a memory store for production use
 * Note: In production, use AsyncPostgresStore or similar
 */
export function createInMemoryStore() {
  // This would be replaced with proper store in production
  // For now, we'll use a simple in-memory implementation
  return new Map<string, any>();
}

// ============================================================================
// MEMORY QUERY HELPERS
// ============================================================================

/**
 * Query semantic memories for a user/agent
 */
export async function querySemanticMemories(
  store: BaseStore,
  namespace: MemoryNamespace,
  filter?: (memory: SemanticMemory) => boolean
): Promise<SemanticMemory[]> {
  const memories: SemanticMemory[] = [];
  const items = store.list(namespace);

  for await (const item of items) {
    if (item.value) {
      const memory = item.value as SemanticMemory;
      if (!filter || filter(memory)) {
        memories.push(memory);
      }
    }
  }

  return memories;
}

/**
 * Get all episodic memories for a user/agent
 */
export async function getEpisodicMemories(
  store: BaseStore,
  namespace: MemoryNamespace,
  limit = 10
): Promise<EpisodicMemory[]> {
  const memories: EpisodicMemory[] = [];
  const items = store.list(namespace);

  for await (const item of items) {
    if (item.value && memories.length < limit) {
      const memory = item.value as any;
      if (memory.observation && memory.thoughts && memory.action && memory.result) {
        memories.push(memory as EpisodicMemory);
      }
    }
  }

  return memories.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
}

/**
 * Get procedural memories (instructions) for an agent
 */
export async function getProceduralMemories(
  store: BaseStore,
  namespace: MemoryNamespace
): Promise<ProceduralMemory[]> {
  const memories: ProceduralMemory[] = [];
  const items = store.list(namespace);

  for await (const item of items) {
    if (item.value) {
      const memory = item.value as any;
      if (memory.name && memory.instructions && memory.version) {
        memories.push(memory as ProceduralMemory);
      }
    }
  }

  return memories;
}

// ============================================================================
// AGENTS.MD CONFIGURATION SUPPORT
// ============================================================================

/**
 * Agent configuration from AGENTS.md
 */
export interface AgentConfig {
  name: string;
  role: string;
  provider: string;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  skills: Array<{
    id: string;
    name: string;
    description: string;
    enabled?: boolean;
  }>;
  tools: string[];
}

/**
 * Parse AGENTS.md file and extract agent configurations
 */
export function parseAgentsMd(content: string): Record<string, AgentConfig> {
  const agents: Record<string, AgentConfig> = {};
  const lines = content.split('\n');

  let currentAgent: Partial<AgentConfig> | null = null;
  let currentSection = '';
  let systemPromptBuffer: string[] = [];
  let skillsBuffer: Array<{ id: string; name: string; description: string }> = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Agent header (### agentname)
    const agentMatch = line.match(/^###\s+(\w+)$/);
    if (agentMatch) {
      // Save previous agent
      if (currentAgent && currentAgent.name) {
        if (systemPromptBuffer.length > 0) {
          currentAgent.systemPrompt = systemPromptBuffer.join('\n').trim();
          systemPromptBuffer = [];
        }
        if (currentAgent.name) {
          agents[currentAgent.name] = {
            name: currentAgent.name,
            role: currentAgent.role || '',
            provider: currentAgent.provider || 'openrouter',
            model: currentAgent.model || 'anthropic/claude-sonnet-4',
            temperature: currentAgent.temperature ?? 0.7,
            maxTokens: currentAgent.maxTokens ?? 4096,
            systemPrompt: currentAgent.systemPrompt || '',
            skills: skillsBuffer,
            tools: currentAgent.tools || []
          };
        }
        skillsBuffer = [];
      }

      currentAgent = { name: agentMatch[1].toLowerCase() };
      currentSection = 'header';
      continue;
    }

    if (!currentAgent) continue;

    // Section headers
    if (line.startsWith('**') && line.endsWith('**')) {
      const sectionName = line.replace(/\*\*/g, '').toLowerCase().replace(':', '').trim();

      if (sectionName === 'role') {
        currentSection = 'role';
        continue;
      } else if (sectionName === 'model configuration') {
        currentSection = 'config';
        continue;
      } else if (sectionName === 'system prompt') {
        currentSection = 'prompt';
        // Start collecting system prompt from next line
        if (lines[i + 1]?.startsWith('```')) {
          i++; // skip the ```
          i++; // move to first line of prompt
          while (i < lines.length && !lines[i].startsWith('```')) {
            systemPromptBuffer.push(lines[i]);
            i++;
          }
        }
        continue;
      } else if (sectionName === 'skills') {
        currentSection = 'skills';
        continue;
      } else if (sectionName === 'tools') {
        currentSection = 'tools';
        continue;
      }
    }

    // Parse role
    if (currentSection === 'role' && line.trim()) {
      currentAgent.role = line.trim();
    }

    // Parse model configuration
    if (currentSection === 'config') {
      const providerMatch = line.match(/Provider:\s*(.+)/);
      if (providerMatch) currentAgent.provider = providerMatch[1].trim();

      const modelMatch = line.match(/Model:\s*(.+)/);
      if (modelMatch) currentAgent.model = modelMatch[1].trim();

      const tempMatch = line.match(/Temperature:\s*(.+)/);
      if (tempMatch) currentAgent.temperature = parseFloat(tempMatch[1]);

      const tokensMatch = line.match(/Max Tokens:\s*(.+)/);
      if (tokensMatch) currentAgent.maxTokens = parseInt(tokensMatch[1]);
    }

    // Parse skills
    if (currentSection === 'skills') {
      const skillMatch = line.match(/-\s*`?(\w+-?\w*)`?:\s*(.+)/);
      if (skillMatch) {
        skillsBuffer.push({
          id: skillMatch[1],
          name: skillMatch[1].replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          description: skillMatch[2].trim()
        });
      }
    }

    // Parse tools
    if (currentSection === 'tools') {
      const toolMatch = line.match(/-\s*`?(\w+)`?:/);
      if (toolMatch && !currentAgent.tools) {
        currentAgent.tools = [];
      }
      if (toolMatch && currentAgent.tools) {
        currentAgent.tools.push(toolMatch[1]);
      }
    }
  }

  // Save last agent
  if (currentAgent && currentAgent.name) {
    if (systemPromptBuffer.length > 0) {
      currentAgent.systemPrompt = systemPromptBuffer.join('\n').trim();
    }
    agents[currentAgent.name] = {
      name: currentAgent.name,
      role: currentAgent.role || '',
      provider: currentAgent.provider || 'openrouter',
      model: currentAgent.model || 'anthropic/claude-sonnet-4',
      temperature: currentAgent.temperature ?? 0.7,
      maxTokens: currentAgent.maxTokens ?? 4096,
      systemPrompt: currentAgent.systemPrompt || '',
      skills: skillsBuffer,
      tools: currentAgent.tools || []
    };
  }

  return agents;
}

/**
 * Get default agent configurations (fallback)
 */
export function getDefaultAgentConfigs(): Record<string, AgentConfig> {
  return {
    copilot: {
      name: 'copilot',
      role: 'Trading Assistant & Workflow Guide',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.7,
      maxTokens: 4096,
      systemPrompt: `You are a helpful trading assistant for QuantMindX, an AI-powered trading system.

Your responsibilities:
- Help users understand trading strategies and concepts
- Guide users through workflow processes
- Assist with strategy analysis and optimization
- Provide clear explanations of trading metrics and results
- Help troubleshoot issues and suggest improvements

Memory capabilities:
- Store user preferences as semantic memories (triples: subject-predicate-object)
- Remember successful interactions as episodic memories (observation-thoughts-action-result)
- Learn and improve over time by updating procedural memories

When using tools, always explain what you're doing and why.
Store important information about the user, their preferences, and their trading goals.`,
      skills: [
        { id: 'market-analysis', name: 'Market Analysis', description: 'Analyze market conditions and trends' },
        { id: 'strategy-guidance', name: 'Strategy Guidance', description: 'Guide users through strategy development' },
        { id: 'troubleshooting', name: 'Troubleshooting', description: 'Identify and resolve common issues' },
        { id: 'metrics-explanation', name: 'Metrics Explanation', description: 'Explain performance metrics' }
      ],
      tools: ['get_market_data', 'run_backtest', 'get_position_size']
    },
    quantcode: {
      name: 'quantcode',
      role: 'MQL5 Code Expert',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.3,
      maxTokens: 8192,
      systemPrompt: `You are an MQ5 coding expert for QuantMindX.

Your responsibilities:
- Generate clean, efficient MQL5 code for trading strategies
- Debug and fix existing MQL5 code
- Optimize code for performance and reliability
- Follow MQL5 best practices and coding standards
- Include proper error handling and risk management

Memory capabilities:
- Store coding patterns and solutions as semantic memories
- Remember successful debugging approaches as episodic memories
- Maintain procedural memories for common coding tasks

When writing code:
- Use proper naming conventions
- Add helpful comments for complex logic
- Include input validation
- Implement proper error handling
- Consider MetaTrader 5 API limitations`,
      skills: [
        { id: 'code-generation', name: 'Code Generation', description: 'Generate MQL5 code from specifications' },
        { id: 'code-debugging', name: 'Code Debugging', description: 'Debug and fix MQL5 code issues' },
        { id: 'code-optimization', name: 'Code Optimization', description: 'Optimize code for performance' },
        { id: 'trd-to-ea', name: 'TRD to EA', description: 'Convert TRD to Expert Advisor' },
        { id: 'backtest-setup', name: 'Backtest Setup', description: 'Configure backtesting' }
      ],
      tools: ['get_market_data', 'run_backtest', 'get_position_size']
    },
    analyst: {
      name: 'analyst',
      role: 'Trading Strategy Analyst',
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4',
      temperature: 0.5,
      maxTokens: 6144,
      systemPrompt: `You are a trading strategy analyst for QuantMindX.

Your responsibilities:
- Analyze backtesting results and performance metrics
- Recognize trading patterns and market conditions
- Evaluate strategy effectiveness and risk profiles
- Identify strengths and weaknesses in trading approaches
- Provide actionable insights for strategy improvement

Memory capabilities:
- Store market observations as semantic memories
- Remember successful analysis patterns as episodic memories
- Maintain procedural memories for analysis methodologies

When analyzing:
- Consider both quantitative and qualitative factors
- Look for patterns that may indicate future performance
- Assess risk-adjusted returns, not just raw returns
- Identify potential market regime changes
- Suggest specific improvements based on data`,
      skills: [
        { id: 'backtest-analysis', name: 'Backtest Analysis', description: 'Analyze backtesting results in depth' },
        { id: 'pattern-recognition', name: 'Pattern Recognition', description: 'Identify trading patterns and setups' },
        { id: 'risk-assessment', name: 'Risk Assessment', description: 'Evaluate strategy risk profiles' },
        { id: 'performance-optimization', name: 'Performance Optimization', description: 'Suggest performance improvements' },
        { id: 'market-regime-detection', name: 'Market Regime Detection', description: 'Detect market changes' }
      ],
      tools: ['get_market_data', 'run_backtest', 'get_position_size']
    }
  };
}

// Global agent configs (loaded on startup)
let agentConfigs: Record<string, AgentConfig> = getDefaultAgentConfigs();

/**
 * Initialize agent configurations from AGENTS.md content
 */
export function initializeAgentConfigs(agentsMdContent?: string) {
  if (agentsMdContent) {
    try {
      agentConfigs = parseAgentsMd(agentsMdContent);
    } catch (error) {
      console.warn('Failed to parse AGENTS.md content, using defaults:', error);
      agentConfigs = getDefaultAgentConfigs();
    }
  } else {
    agentConfigs = getDefaultAgentConfigs();
  }
}

/**
 * Get all agent configurations
 */
export function getAgentConfigs(): Record<string, AgentConfig> {
  return agentConfigs;
}

/**
 * Get a specific agent configuration
 */
export function getAgentConfig(agentType: string): AgentConfig | undefined {
  return agentConfigs[agentType];
}

/**
 * Update agent configuration (for Settings UI)
 */
export function updateAgentConfig(agentType: string, updates: Partial<AgentConfig>): void {
  const config = agentConfigs[agentType];
  if (config) {
    agentConfigs[agentType] = { ...config, ...updates };
  }
}

/**
 * Generate AGENTS.md content from current configurations
 */
export function generateAgentsMd(): string {
  const lines: string[] = [];
  const configs = getAgentConfigs();

  lines.push('# AGENTS.md');
  lines.push('');
  lines.push('Agent configuration file for QuantMindX IDE. This file defines the behavior, prompts, and capabilities of all AI agents in the system.');
  lines.push('');

  // Global Settings
  lines.push('## Global Settings');
  lines.push('');
  lines.push('### Default Model');
  lines.push('All agents use OpenRouter as the default provider with Claude Sonnet 4 as the default model.');
  lines.push('');

  // Agent definitions
  lines.push('## Agent Definitions');
  lines.push('');

  for (const [key, agent] of Object.entries(configs)) {
    lines.push(`### ${agent.name}`);
    lines.push('');
    lines.push(`**Role**: ${agent.role}`);
    lines.push('');
    lines.push('**Model Configuration**:');
    lines.push(`- Provider: ${agent.provider}`);
    lines.push(`- Model: ${agent.model}`);
    lines.push(`- Temperature: ${agent.temperature}`);
    lines.push(`- Max Tokens: ${agent.maxTokens}`);
    lines.push('');
    lines.push('**System Prompt**:');
    lines.push('```');
    lines.push(agent.systemPrompt);
    lines.push('```');
    lines.push('');
    lines.push('**Skills**:');
    for (const skill of agent.skills) {
      lines.push(`- \`${skill.id}\`: ${skill.description}`);
    }
    lines.push('');
    lines.push('**Tools**:');
    for (const tool of agent.tools) {
      lines.push(`- ${tool}`);
    }
    lines.push('');
    lines.push('---');
    lines.push('');
  }

  return lines.join('\n');
}

// ============================================================================
// THREE-AGENT WORKFLOW COORDINATION
// ============================================================================

/**
 * Three-Agent Workflow State for coordinated operations
 */
export interface ThreeAgentWorkflowState {
  sessionId: string;
  currentStep: 'copilot' | 'analyst' | 'quantcode' | 'complete';
  strategyName?: string;
  nprdContent?: string;
  vanillaTRD?: string;
  enhancedTRD?: string;
  vanillaEA?: string;
  enhancedEA?: string;
  metadata: Record<string, any>;
}

/**
 * Three-Agent Workflow Coordinator
 * Orchestrates the complete workflow: Copilot → Analyst → QuantCode
 */
export class ThreeAgentWorkflowCoordinator {
  private copilotManager: any;
  private analystManager: any;
  private quantcodeManager: any;
  private store?: BaseStore;
  private sessionId: string;

  constructor(
    sessionId: string,
    options: {
      store?: BaseStore;
      enableMemory?: boolean;
      provider?: "openrouter" | "zhipu" | "anthropic";
    } = {}
  ) {
    this.sessionId = sessionId;
    this.store = options.store;
    const commonOptions = {
      sessionId,
      store: options.store,
      enableMemory: options.enableMemory ?? true,
      provider: options.provider,
    };

    // Create agent managers for the three specialized agents
    this.copilotManager = createMemoryEnabledAgentManager("copilot", commonOptions);
    this.analystManager = createMemoryEnabledAgentManager("analyst", commonOptions);
    this.quantcodeManager = createMemoryEnabledAgentManager("quantcode", commonOptions);
  }

  /**
   * Execute the complete workflow: YouTube → NPRD → TRD (Vanilla + Enhanced) → EA (Vanilla + Enhanced)
   */
  async executeFullWorkflow(input: {
    strategyName: string;
    description: string;
    sourceUrl?: string;
  }): Promise<ThreeAgentWorkflowState> {
    const state: ThreeAgentWorkflowState = {
      sessionId: this.sessionId,
      currentStep: "copilot",
      strategyName: input.strategyName,
      metadata: {
        startTime: new Date().toISOString(),
        sourceUrl: input.sourceUrl,
      },
    };

    try {
      // Step 1: Copilot - Initialize and store strategy info
      state.currentStep = "copilot";
      const copilotResult = await this.copilotManager.invoke(
        `User wants to create a new trading strategy called "${input.strategyName}". ` +
        `Description: ${input.description}. ` +
        `Please help initialize the strategy and prepare for Analyst agent to create an NPRD.`,
        { threadId: this.sessionId }
      );
      state.metadata.copilotResponse = copilotResult.response;

      // Step 2: Analyst - Create NPRD from strategy description
      state.currentStep = "analyst";
      const analystNPRDResult = await this.analystManager.invoke(
        `Create an NPRD for the trading strategy "${input.strategyName}". ` +
        `Description: ${input.description}. ` +
        `Source: ${input.sourceUrl || "manual input"}. ` +
        `Use the create_nprd tool to generate the NPRD structure.`,
        { threadId: this.sessionId }
      );

      // Step 3: Analyst - Convert NPRD to Vanilla TRD
      const analystVanillaTRDResult = await this.analystManager.invoke(
        `Convert the NPRD for "${input.strategyName}" to a Vanilla TRD. ` +
        `Use the convert_to_vanilla_trd tool.`,
        { threadId: this.sessionId }
      );
      state.metadata.vanillaTRDResponse = analystVanillaTRDResult.response;

      // Step 4: Analyst - Convert NPRD to Enhanced TRD
      const analystEnhancedTRDResult = await this.analystManager.invoke(
        `Convert the NPRD for "${input.strategyName}" to an Enhanced TRD. ` +
        `Use the convert_to_enhanced_trd tool. Include all QuantMindX features: ` +
        `Kelly Criterion, tiered risk management, house-money protection, circuit breaker.`,
        { threadId: this.sessionId }
      );
      state.metadata.enhancedTRDResponse = analystEnhancedTRDResult.response;

      // Step 5: QuantCode - Generate Vanilla EA from Vanilla TRD
      state.currentStep = "quantcode";
      const quantcodeVanillaEAResult = await this.quantcodeManager.invoke(
        `Generate a Vanilla MQL5 Expert Advisor from the Vanilla TRD for "${input.strategyName}". ` +
        `Use the generate_vanilla_ea tool. This should be a basic implementation without QuantMindX enhancements.`,
        { threadId: this.sessionId }
      );
      state.metadata.vanillaEAResponse = quantcodeVanillaEAResult.response;

      // Step 6: QuantCode - Generate Enhanced EA from Enhanced TRD
      const quantcodeEnhancedEAResult = await this.quantcodeManager.invoke(
        `Generate an Enhanced MQL5 Expert Advisor from the Enhanced TRD for "${input.strategyName}". ` +
        `Use the generate_enhanced_ea tool. Include all QuantMindX features: ` +
        `Kelly Criterion position sizing, tiered risk management, house-money protection, ` +
        `circuit breaker, router integration, and shared asset library usage.`,
        { threadId: this.sessionId }
      );
      state.metadata.enhancedEAResponse = quantcodeEnhancedEAResult.response;

      // Step 7: Complete - Return to Copilot for final summary
      state.currentStep = "complete";
      const copilotSummaryResult = await this.copilotManager.invoke(
        `The strategy "${input.strategyName}" has been fully processed. ` +
        `Both Vanilla and Enhanced versions have been created (TRD and EA). ` +
        `Please provide a summary of what was created and suggest next steps for the user.`,
        { threadId: this.sessionId }
      );
      state.metadata.summaryResponse = copilotSummaryResult.response;

      state.metadata.endTime = new Date().toISOString();
      state.metadata.duration = Date.now() - new Date(state.metadata.startTime).getTime();

      return state;
    } catch (error) {
      state.metadata.error = error instanceof Error ? error.message : String(error);
      state.metadata.errorStep = state.currentStep;
      throw error;
    }
  }

  /**
   * Execute partial workflow (e.g., NPRD → TRD only)
   */
  async executePartialWorkflow(
    startStep: "analyst" | "quantcode",
    input: any
  ): Promise<Partial<ThreeAgentWorkflowState>> {
    switch (startStep) {
      case "analyst":
        // Analyst workflow: Create NPRD → Convert to TRDs
        return this.executeAnalystWorkflow(input);

      case "quantcode":
        // QuantCode workflow: Generate EAs from TRDs
        return this.executeQuantCodeWorkflow(input);

      default:
        throw new Error(`Invalid start step: ${startStep}`);
    }
  }

  /**
   * Analyst workflow: NPRD → TRD (Vanilla + Enhanced)
   */
  private async executeAnalystWorkflow(input: {
    strategyName: string;
    description: string;
    sourceUrl?: string;
  }): Promise<Partial<ThreeAgentWorkflowState>> {
    const result: Partial<ThreeAgentWorkflowState> = {
      sessionId: this.sessionId,
      currentStep: "analyst",
      strategyName: input.strategyName,
      metadata: {},
    };

    // Create NPRD
    const nprdResult = await this.analystManager.invoke(
      `Create an NPRD for "${input.strategyName}". Description: ${input.description}`,
      { threadId: this.sessionId }
    );
    result.metadata.nprdResponse = nprdResult.response;

    // Convert to Vanilla TRD
    const vanillaTRDResult = await this.analystManager.invoke(
      `Convert the NPRD to a Vanilla TRD for "${input.strategyName}"`,
      { threadId: this.sessionId }
    );
    result.metadata.vanillaTRDResponse = vanillaTRDResult.response;

    // Convert to Enhanced TRD
    const enhancedTRDResult = await this.analystManager.invoke(
      `Convert the NPRD to an Enhanced TRD for "${input.strategyName}"`,
      { threadId: this.sessionId }
    );
    result.metadata.enhancedTRDResponse = enhancedTRDResult.response;

    return result;
  }

  /**
   * QuantCode workflow: TRD → EA (Vanilla + Enhanced)
   */
  private async executeQuantCodeWorkflow(input: {
    strategyName: string;
    vanillaTRD: string;
    enhancedTRD: string;
  }): Promise<Partial<ThreeAgentWorkflowState>> {
    const result: Partial<ThreeAgentWorkflowState> = {
      sessionId: this.sessionId,
      currentStep: "quantcode",
      strategyName: input.strategyName,
      metadata: {},
    };

    // Generate Vanilla EA
    const vanillaEAResult = await this.quantcodeManager.invoke(
      `Generate a Vanilla EA from the TRD for "${input.strategyName}". TRD: ${input.vanillaTRD}`,
      { threadId: this.sessionId }
    );
    result.metadata.vanillaEAResponse = vanillaEAResult.response;

    // Generate Enhanced EA
    const enhancedEAResult = await this.quantcodeManager.invoke(
      `Generate an Enhanced EA from the TRD for "${input.strategyName}". TRD: ${input.enhancedTRD}`,
      { threadId: this.sessionId }
    );
    result.metadata.enhancedEAResponse = enhancedEAResult.response;

    return result;
  }

  /**
   * Stream workflow progress
   */
  async *streamWorkflow(input: {
    strategyName: string;
    description: string;
    sourceUrl?: string;
  }): AsyncGenerator<{ step: string; data: any }, void, unknown> {
    yield { step: "copilot", data: { message: "Initializing workflow..." } };

    // Copilot step
    const copilotResult = await this.copilotManager.invoke(
      `Initialize strategy "${input.strategyName}"`,
      { threadId: this.sessionId }
    );
    yield { step: "copilot", data: copilotResult };

    // Analyst steps
    yield { step: "analyst", data: { message: "Creating NPRD..." } };
    const nprdResult = await this.analystManager.invoke(
      `Create NPRD for "${input.strategyName}"`,
      { threadId: this.sessionId }
    );
    yield { step: "analyst-nprd", data: nprdResult };

    yield { step: "analyst", data: { message: "Converting to Vanilla TRD..." } };
    const vanillaTRDResult = await this.analystManager.invoke(
      `Convert to Vanilla TRD`,
      { threadId: this.sessionId }
    );
    yield { step: "analyst-vanilla", data: vanillaTRDResult };

    yield { step: "analyst", data: { message: "Converting to Enhanced TRD..." } };
    const enhancedTRDResult = await this.analystManager.invoke(
      `Convert to Enhanced TRD`,
      { threadId: this.sessionId }
    );
    yield { step: "analyst-enhanced", data: enhancedTRDResult };

    // QuantCode steps
    yield { step: "quantcode", data: { message: "Generating Vanilla EA..." } };
    const vanillaEAResult = await this.quantcodeManager.invoke(
      `Generate Vanilla EA`,
      { threadId: this.sessionId }
    );
    yield { step: "quantcode-vanilla", data: vanillaEAResult };

    yield { step: "quantcode", data: { message: "Generating Enhanced EA..." } };
    const enhancedEAResult = await this.quantcodeManager.invoke(
      `Generate Enhanced EA`,
      { threadId: this.sessionId }
    );
    yield { step: "quantcode-enhanced", data: enhancedEAResult };

    yield { step: "complete", data: { message: "Workflow complete!" } };
  }
}

/**
 * Create a three-agent workflow coordinator
 */
export function createThreeAgentWorkflow(
  sessionId: string,
  options?: {
    store?: BaseStore;
    enableMemory?: boolean;
    provider?: "openrouter" | "zhipu" | "anthropic";
  }
) {
  return new ThreeAgentWorkflowCoordinator(sessionId, options);
}
