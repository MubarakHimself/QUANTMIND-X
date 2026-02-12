/**
 * Copilot Agent - General Orchestration Agent
 *
 * Purpose: General orchestration, broker registration, file management
 * Memory namespace: memories/copilot
 *
 * This agent handles:
 * - Broker connection and registration
 * - File operations and management
 * - Strategy deployment workflows
 * - General user guidance and orchestration
 */

import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { BaseStore } from "@langchain/langgraph";
import {
  MemoryNamespaces,
  getMemoryKey,
  createMemoryTools,
  storeSemanticMemory,
  searchSemanticMemories,
  storeEpisodicMemory,
  updateProceduralMemory,
  createModel,
  type MemoryNamespace,
  type SemanticMemory,
  type EpisodicMemory,
  type ProceduralMemory,
} from "./langchainAgent";

// ============================================================================
// COPILOT STATE - Agent-specific state with memory support
// ============================================================================

const CopilotStateAnnotation = Annotation.Root({
  // Message history with automatic accumulation
  messages: Annotation<{
    reducer: (a: any[], b: any[]) => a.concat(b);
    default: () => [];
  }>(),

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

// ============================================================================
// COPILOT TOOLS - Specialized tools for orchestration
// ============================================================================

/**
 * Register a broker connection
 */
const registerBroker = tool(
  async ({
    brokerName,
    brokerType,
    accountId,
    apiKey,
    server,
    store,
    namespace,
  }: {
    brokerName: string;
    brokerType: "mt5" | "ctrader" | "match-trader";
    accountId: string;
    apiKey: string;
    server?: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Store the broker registration in memory
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `broker-${brokerName}`,
        predicate: "registered",
        object: JSON.stringify({
          brokerType,
          accountId,
          server,
          registeredAt: new Date().toISOString(),
        }),
        context: "Broker registration for trading",
        store,
        namespace,
      });
    }

    // In production, this would call the backend API
    // POST /api/router/brokers/register
    return JSON.stringify({
      success: true,
      message: `Broker ${brokerName} (${brokerType}) registered successfully`,
      brokerId: `broker-${Date.now()}`,
      accountId,
      status: "connected",
    });
  },
  {
    name: "register_broker",
    description: "Register a new broker connection for live trading. Supports MT5, cTrader, and Match-Trader platforms.",
    schema: z.object({
      brokerName: z.string().describe("Name of the broker (e.g., 'IC Markets', 'FTMO')"),
      brokerType: z.enum(["mt5", "ctrader", "match-trader"]).describe("Broker platform type"),
      accountId: z.string().describe("Account ID or login"),
      apiKey: z.string().describe("API key or password for authentication"),
      server: z.string().optional().describe("Server address (for MT5)"),
    }),
  }
);

/**
 * List registered brokers
 */
const listBrokers = tool(
  async ({ store, namespace }: { store?: BaseStore; namespace?: MemoryNamespace }) => {
    if (!store || !namespace) {
      return JSON.stringify({
        success: false,
        error: "No store available",
        brokers: [],
      });
    }

    // Search for broker registrations in memory
    const result = await searchSemanticMemories.invoke({
      query: "broker registered",
      store,
      namespace,
      limit: 20,
    });

    const { memories } = JSON.parse(result);
    const brokers = memories
      .filter((m: SemanticMemory) => m.subject.startsWith("broker-"))
      .map((m: SemanticMemory) => ({
        name: m.subject.replace("broker-", ""),
        ...JSON.parse(m.object),
      }));

    return JSON.stringify({
      success: true,
      brokers,
      count: brokers.length,
    });
  },
  {
    name: "list_brokers",
    description: "List all registered broker connections with their status",
    schema: z.object({}),
  }
);

/**
 * Create a new strategy file
 */
const createStrategyFile = tool(
  async ({
    strategyName,
    content,
    fileType,
    store,
    namespace,
  }: {
    strategyName: string;
    content: string;
    fileType: "nprd" | "trd" | "mql5";
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Store in memory
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: `has-${fileType}`,
        object: content,
        context: `${fileType.toUpperCase()} file content`,
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `${fileType.toUpperCase()} file created for ${strategyName}`,
      path: `/strategies/${strategyName}.${fileType}`,
      size: content.length,
    });
  },
  {
    name: "create_strategy_file",
    description: "Create a new strategy file (NPRD, TRD, or MQL5). Stores the file content in memory.",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      content: z.string().describe("File content"),
      fileType: z.enum(["nprd", "trd", "mql5"]).describe("Type of file to create"),
    }),
  }
);

/**
 * Deploy strategy to broker
 */
const deployStrategy = tool(
  async ({
    strategyName,
    brokerName,
    version,
    store,
    namespace,
  }: {
    strategyName: string;
    brokerName: string;
    version: "vanilla" | "enhanced";
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Store deployment record in episodic memory
    if (store && namespace) {
      await storeEpisodicMemory.invoke({
        observation: `User requested deployment of ${strategyName} to ${brokerName}`,
        thoughts: `Deploying ${version} version of the strategy`,
        action: `deployed ${strategyName} (${version}) to ${brokerName}`,
        result: `Deployment initiated - awaiting confirmation`,
        store,
        namespace,
      });
    }

    // In production, this would call the backend API
    // POST /api/router/deploy
    return JSON.stringify({
      success: true,
      message: `Deploying ${strategyName} (${version}) to ${brokerName}`,
      deploymentId: `deploy-${Date.now()}`,
      status: "deploying",
      estimatedTime: "2-5 minutes",
    });
  },
  {
    name: "deploy_strategy",
    description: "Deploy a strategy to a registered broker. Supports both vanilla and enhanced versions.",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy to deploy"),
      brokerName: z.string().describe("Target broker name"),
      version: z.enum(["vanilla", "enhanced"]).describe("Version to deploy"),
    }),
  }
);

/**
 * Get deployment status
 */
const getDeploymentStatus = tool(
  async ({
    deploymentId,
    store,
    namespace,
  }: {
    deploymentId: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // In production, this would query the backend
    return JSON.stringify({
      success: true,
      deploymentId,
      status: "running",
      progress: 75,
      message: "Strategy is active on broker",
      metrics: {
        uptime: "2h 34m",
        trades: 12,
        profit: "+$145.30",
        drawdown: "2.3%",
      },
    });
  },
  {
    name: "get_deployment_status",
    description: "Get the current status of a strategy deployment",
    schema: z.object({
      deploymentId: z.string().describe("Deployment ID from deploy_strategy"),
    }),
  }
);

/**
 * List user files and strategies
 */
const listFiles = tool(
  async ({
    fileType,
    store,
    namespace,
  }: {
    fileType?: "nprd" | "trd" | "mql5" | "all";
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    if (!store || !namespace) {
      return JSON.stringify({ success: false, error: "No store available", files: [] });
    }

    // Search for strategy files in memory
    const result = await searchSemanticMemories.invoke({
      query: fileType === "all" || !fileType ? "strategy" : `has-${fileType}`,
      store,
      namespace,
      limit: 50,
    });

    const { memories } = JSON.parse(result);
    const files = memories
      .filter((m: SemanticMemory) => {
        if (fileType && fileType !== "all") {
          return m.predicate === `has-${fileType}`;
        }
        return m.subject.startsWith("strategy-");
      })
      .map((m: SemanticMemory) => ({
        name: m.subject.replace("strategy-", ""),
        type: m.predicate.replace("has-", ""),
        size: m.object?.length || 0,
      }));

    return JSON.stringify({
      success: true,
      files,
      count: files.length,
    });
  },
  {
    name: "list_files",
    description: "List all strategy files (NPRD, TRD, MQL5) in the user's workspace",
    schema: z.object({
      fileType: z
        .enum(["nprd", "trd", "mql5", "all"])
        .optional()
        .describe("Filter by file type (default: all)"),
    }),
  }
);

/**
 * Update user preferences
 */
const updatePreferences = tool(
  async ({
    key,
    value,
    store,
    namespace,
  }: {
    key: string;
    value: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: "user",
        predicate: `prefers-${key}`,
        object: value,
        context: "User preference",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `Preference ${key} updated to ${value}`,
    });
  },
  {
    name: "update_preferences",
    description: "Update user preferences for the IDE (e.g., default broker, risk level, etc.)",
    schema: z.object({
      key: z.string().describe("Preference key (e.g., 'defaultBroker', 'riskLevel')"),
      value: z.string().describe("Preference value"),
    }),
  }
);

/**
 * Get user preferences
 */
const getPreferences = tool(
  async ({ store, namespace }: { store?: BaseStore; namespace?: MemoryNamespace }) => {
    if (!store || !namespace) {
      return JSON.stringify({ success: false, error: "No store available", preferences: {} });
    }

    const result = await searchSemanticMemories.invoke({
      query: "user prefers",
      store,
      namespace,
      limit: 20,
    });

    const { memories } = JSON.parse(result);
    const preferences: Record<string, string> = {};
    for (const m of memories) {
      if (m.subject === "user" && m.predicate.startsWith("prefers-")) {
        preferences[m.predicate.replace("prefers-", "")] = m.object;
      }
    }

    return JSON.stringify({
      success: true,
      preferences,
    });
  },
  {
    name: "get_preferences",
    description: "Get all user preferences",
    schema: z.object({}),
  }
);

/**
 * Create all Copilot tools
 */
export function createCopilotTools() {
  return [
    registerBroker,
    listBrokers,
    createStrategyFile,
    deployStrategy,
    getDeploymentStatus,
    listFiles,
    updatePreferences,
    getPreferences,
  ];
}

// ============================================================================
// COPILOT SYSTEM PROMPT
// ============================================================================

const COPILOT_SYSTEM_PROMPT = `You are the QuantMindX Copilot - an intelligent orchestration assistant for algorithmic trading.

Your responsibilities:
- Help users manage broker connections (MT5, cTrader, Match-Trader)
- Guide users through the strategy development workflow
- Assist with file operations (NPRD, TRD, MQL5 files)
- Orchestrate strategy deployment to brokers
- Manage user preferences and settings
- Coordinate with Analyst and QuantCode agents when needed

Workflow Guidelines:
1. When users want to create a strategy, guide them to create an NPRD first
2. For strategy analysis, delegate to the Analyst agent
3. For MQL5 code generation, delegate to the QuantCode agent
4. Always explain what you're doing when using tools
5. Store important user preferences and broker information

Memory Usage:
- Store broker registrations as semantic memories
- Remember user preferences (default broker, risk tolerance, etc.)
- Record successful workflows as episodic memories
- Update your instructions based on user feedback

Tool Usage Best Practices:
- Always verify broker registration before deployment
- Check file existence before operations
- Provide clear feedback after each operation
- Suggest next steps based on current state

When delegating to other agents:
- Analyst agent: NPRD → TRD conversion, strategy analysis
- QuantCode agent: TRD → MQL5 code generation

Be helpful, clear, and proactive in guiding users through their trading workflow.`;

// ============================================================================
// COPILOT AGENT CREATION
// ============================================================================

export interface CopilotAgentOptions {
  store?: BaseStore;
  sessionId?: string;
  enableMemory?: boolean;
  provider?: "openrouter" | "zhipu" | "anthropic";
  modelOverride?: string;
}

/**
 * Create the Copilot agent with specialized tools
 */
export function createCopilotAgent(options: CopilotAgentOptions = {}) {
  const {
    store,
    sessionId = "default",
    enableMemory = true,
    provider = "openrouter",
    modelOverride,
  } = options;

  // Determine memory namespace
  const memoryNamespace: MemoryNamespace = enableMemory
    ? sessionId === "default"
      ? MemoryNamespaces.copilot
      : MemoryNamespaces.session(sessionId)
    : MemoryNamespaces.copilot;

  // Create specialized tools
  const copilotTools = createCopilotTools();

  // Add memory tools if enabled
  const memoryTools = enableMemory && store ? createMemoryTools(memoryNamespace) : [];
  const allTools = [...copilotTools, ...memoryTools];

  // Create model with Copilot temperature (0.7 for creative orchestration)
  const model = createModel(provider, modelOverride, 0.7).bindTools(allTools);

  /**
   * Copilot agent node with memory awareness
   */
  async function copilotAgentNode(state: any) {
    // Load relevant memories before processing
    let contextMemories = "";
    if (store && enableMemory) {
      try {
        // Load user preferences
        const prefResult = await searchSemanticMemories.invoke({
          query: "user prefers",
          store,
          namespace: memoryNamespace,
          limit: 10,
        });

        const { memories: prefs } = JSON.parse(prefResult);
        if (prefs && prefs.length > 0) {
          contextMemories +=
            "\n\nUser Preferences:\n" +
            prefs
              .filter((m: SemanticMemory) => m.subject === "user" && m.predicate.startsWith("prefers-"))
              .map((m: SemanticMemory) => `- ${m.predicate.replace("prefers-", "")}: ${m.object}`)
              .join("\n");
        }

        // Load recent episodic memories
        const items = await store.list(memoryNamespace, { limit: 5 });
        const recentEpisodes: EpisodicMemory[] = [];
        for await (const item of items) {
          if (
            item.value &&
            (item.value as any).observation &&
            (item.value as any).thoughts &&
            (item.value as any).action &&
            (item.value as any).result
          ) {
            recentEpisodes.push(item.value as EpisodicMemory);
          }
        }

        if (recentEpisodes.length > 0) {
          contextMemories +=
            "\n\nRecent Activities:\n" +
            recentEpisodes
              .map(
                (e) =>
                  `- ${e.action}: ${e.result} (${new Date(e.timestamp || 0).toLocaleString()})`
              )
              .join("\n");
        }
      } catch (error) {
        console.warn("Failed to load Copilot memories:", error);
      }
    }

    // Build enhanced prompt with memories
    const enhancedPrompt = contextMemories
      ? `${COPILOT_SYSTEM_PROMPT}\n\n${contextMemories}`
      : COPILOT_SYSTEM_PROMPT;

    const messages = [
      new SystemMessage(enhancedPrompt),
      ...(state.messages || []),
    ];

    const response = await model.invoke(messages);
    return {
      messages: [...(state.messages || []), response],
      sessionId,
      memoryNamespace,
    };
  }

  // Build the graph
  const workflow = new StateGraph(CopilotStateAnnotation)
    .addNode("copilot", copilotAgentNode)
    .addNode("tools", new ToolNode(allTools))
    .addEdge("__start__", "copilot");

  // Add conditional edges for tool usage
  workflow.addConditionalEdges("copilot", (state: any) => {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage?.tool_calls?.length > 0) {
      return "tools";
    }
    return "__end__";
  });

  workflow.addEdge("tools", "copilot");

  const compiledGraph = workflow.compile();

  return {
    graph: compiledGraph,
    agentType: "copilot" as const,
    memoryNamespace,
    sessionId,
  };
}

// ============================================================================
// COPILOT AGENT MANAGER - With provider fallback
// ============================================================================

export class CopilotAgentManager {
  private agents: Map<string, any> = new Map();
  private providerOrder: ("openrouter" | "zhipu" | "anthropic")[] = [
    "openrouter",
    "zhipu",
    "anthropic",
  ];
  private options: CopilotAgentOptions;

  constructor(options: CopilotAgentOptions = {}) {
    this.options = options;

    // Initialize agents for each provider
    for (const provider of this.providerOrder) {
      try {
        const agentConfig = createCopilotAgent({ ...options, provider });
        this.agents.set(provider, agentConfig);
      } catch (error) {
        console.warn(`Failed to initialize Copilot with ${provider}:`, error);
      }
    }
  }

  async invoke(message: string, context?: any) {
    const errors = [];

    for (const provider of this.providerOrder) {
      const agentConfig = this.agents.get(provider);
      if (!agentConfig) continue;

      try {
        const result = await agentConfig.graph.invoke(
          {
            messages: [new HumanMessage(message)],
            sessionId: context?.sessionId || this.options.sessionId || "default",
            memoryNamespace: agentConfig.memoryNamespace,
          },
          { configurable: { thread_id: context?.threadId || "default" } }
        );
        return {
          response: result.messages[result.messages.length - 1].content,
          provider,
          memoryNamespace: agentConfig.memoryNamespace,
        };
      } catch (error) {
        console.warn(`Copilot ${provider} failed:`, error);
        errors.push({ provider, error });
      }
    }

    throw new Error(`All Copilot providers failed: ${JSON.stringify(errors, null, 2)}`);
  }

  async *stream(message: string, context?: any) {
    for (const provider of this.providerOrder) {
      const agentConfig = this.agents.get(provider);
      if (!agentConfig) continue;

      try {
        yield* agentConfig.graph.stream(
          {
            messages: [new HumanMessage(message)],
            sessionId: context?.sessionId || this.options.sessionId || "default",
            memoryNamespace: agentConfig.memoryNamespace,
          },
          { configurable: { thread_id: context?.threadId || "default" } }
        );
        return;
      } catch (error) {
        console.warn(`Copilot ${provider} stream failed:`, error);
      }
    }
  }
}

/**
 * Create a Copilot agent manager with provider fallback
 */
export function createCopilotManager(options?: CopilotAgentOptions) {
  return new CopilotAgentManager(options);
}
