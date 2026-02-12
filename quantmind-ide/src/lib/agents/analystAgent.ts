/**
 * Analyst Agent - NPRD to TRD Conversion Agent
 *
 * Purpose: Convert NPRD (Natural Product Requirements Document) to TRD (Technical Requirements Document)
 * Memory namespace: memories/analyst
 *
 * This agent handles:
 * - YouTube strategy analysis → NPRD creation
 * - NPRD → TRD conversion (creates Vanilla + Enhanced versions)
 * - Strategy analysis and optimization recommendations
 * - Trading pattern recognition
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
// ANALYST STATE - Agent-specific state with memory support
// ============================================================================

const AnalystStateAnnotation = Annotation.Root({
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

  // NPRD content being processed
  nprdContent: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Generated TRD content
  trdContent: Annotation<Record<string, string>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),

  // Analysis results
  analysisResults: Annotation<Record<string, any>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
});

// ============================================================================
// ANALYST TOOLS - Specialized tools for NPRD to TRD conversion
// ============================================================================

/**
 * Create NPRD from YouTube strategy description
 */
const createNPRD = tool(
  async ({
    strategyName,
    description,
    sourceUrl,
    store,
    namespace,
  }: {
    strategyName: string;
    description: string;
    sourceUrl?: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Generate NPRD structure
    const nprd = {
      name: strategyName,
      version: "1.0.0",
      created: new Date().toISOString(),
      source: sourceUrl || "manual",
      sections: {
        overview: {
          name: strategyName,
          description: description,
          type: "algorithmic-trading-strategy",
        },
        entryConditions: {
          primary: [],
          secondary: [],
          filters: [],
        },
        exitConditions: {
          takeProfit: [],
          stopLoss: [],
          trailing: [],
        },
        riskManagement: {
          maxDrawdown: "20%",
          riskPerTrade: "1-2%",
          positionSizing: "kelly-criterion",
        },
        indicators: [],
        timeframe: "H1",
        symbols: ["EURUSD", "GBPUSD", "XAUUSD"],
      },
    };

    // Store NPRD in memory
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: "has-nprd",
        object: JSON.stringify(nprd),
        context: "NPRD created from strategy description",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `NPRD created for ${strategyName}`,
      nprd,
    });
  },
  {
    name: "create_nprd",
    description: "Create a Natural Product Requirements Document (NPRD) from a strategy description (e.g., from YouTube video).",
    schema: z.object({
      strategyName: z.string().describe("Name of the trading strategy"),
      description: z.string().describe("Detailed description of the strategy logic"),
      sourceUrl: z.string().optional().describe("Source URL (e.g., YouTube video)"),
    }),
  }
);

/**
 * Convert NPRD to TRD (Vanilla version)
 */
const convertToVanillaTRD = tool(
  async ({
    strategyName,
    nprdContent,
    store,
    namespace,
  }: {
    strategyName: string;
    nprdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Parse NPRD
    const nprd = JSON.parse(nprdContent);

    // Generate Vanilla TRD - basic implementation without enhancements
    const vanillaTRD = {
      name: strategyName,
      version: `${nprd.version}-vanilla`,
      variant: "vanilla",
      created: new Date().toISOString(),
      parentNPRD: nprd.name,
      sections: {
        implementation: {
          approach: "basic",
          features: [],
          exclusions: [
            "No Kelly Criterion position sizing",
            "No tiered risk management",
            "No adaptive parameters",
            "No house-money protection",
            "Basic stop loss and take profit only",
          ],
        },
        entryLogic: {
          type: "basic",
          conditions: nprd.sections?.entryConditions || {},
        },
        exitLogic: {
          type: "basic",
          conditions: nprd.sections?.exitConditions || {},
        },
        riskManagement: {
          type: "fixed-lot",
          lotSize: 0.01,
          stopLoss: nprd.sections?.riskManagement?.stopLoss || 50,
          takeProfit: nprd.sections?.riskManagement?.takeProfit || 100,
        },
        indicators: nprd.sections?.indicators || [],
        parameters: {
          magicNumber: Math.floor(Math.random() * 900000) + 100000,
          maxSlippage: 3,
        },
      },
    };

    // Store Vanilla TRD
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: "has-vanilla-trd",
        object: JSON.stringify(vanillaTRD),
        context: "Vanilla TRD - basic implementation",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `Vanilla TRD created for ${strategyName}`,
      trd: vanillaTRD,
    });
  },
  {
    name: "convert_to_vanilla_trd",
    description: "Convert an NPRD to a Vanilla TRD (basic implementation without QuantMindX enhancements).",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      nprdContent: z.string().describe("NPRD content as JSON string"),
    }),
  }
);

/**
 * Convert NPRD to Enhanced TRD (with QuantMindX features)
 */
const convertToEnhancedTRD = tool(
  async ({
    strategyName,
    nprdContent,
    store,
    namespace,
  }: {
    strategyName: string;
    nprdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    // Parse NPRD
    const nprd = JSON.parse(nprdContent);

    // Generate Enhanced TRD - with all QuantMindX features
    const enhancedTRD = {
      name: strategyName,
      version: `${nprd.version}-enhanced`,
      variant: "enhanced",
      created: new Date().toISOString(),
      parentNPRD: nprd.name,
      sections: {
        implementation: {
          approach: "enhanced",
          features: [
            "Kelly Criterion position sizing",
            "Tiered risk management (Conservative/Moderate/Aggressive)",
            "House-money protection (breakeven after profit threshold)",
            "Circuit breaker (pause trading after consecutive losses)",
            "Router integration for multi-symbol trading",
            "Shared asset library integration",
            "NPRD-compliant parameter naming",
          ],
        },
        entryLogic: {
          type: "enhanced",
          conditions: nprd.sections?.entryConditions || {},
          filters: [
            "Spread filter (max spread check)",
            "Session filter (trading hours)",
            "Volatility filter (ATR-based)",
          ],
        },
        exitLogic: {
          type: "enhanced",
          conditions: nprd.sections?.exitConditions || {},
          features: [
            "Trailing stop loss",
            "Breakeven trigger",
            "Partial take profit levels",
          ],
        },
        riskManagement: {
          type: "kelly-tiered",
          tiers: {
            conservative: {
              maxDrawdown: "10%",
              riskPerTrade: "0.5-1%",
              kellyFraction: 0.25,
            },
            moderate: {
              maxDrawdown: "15%",
              riskPerTrade: "1-2%",
              kellyFraction: 0.5,
            },
            aggressive: {
              maxDrawdown: "25%",
              riskPerTrade: "2-3%",
              kellyFraction: 0.75,
            },
          },
          houseMoney: {
            enableBreakeven: true,
            profitThreshold: "0.5R",
            trailingStart: "1R",
          },
          circuitBreaker: {
            enable: true,
            maxConsecutiveLosses: 3,
            pauseDuration: "1H",
          },
        },
        indicators: nprd.sections?.indicators || [],
        sharedAssets: {
          useKellySizer: true,
          useRiskGovernor: true,
          useRouter: true,
        },
        parameters: {
          magicNumber: Math.floor(Math.random() * 900000) + 100000,
          maxSlippage: 3,
          tieredRisk: true,
          kellyEnabled: true,
        },
      },
    };

    // Store Enhanced TRD
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: "has-enhanced-trd",
        object: JSON.stringify(enhancedTRD),
        context: "Enhanced TRD - with QuantMindX features",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `Enhanced TRD created for ${strategyName}`,
      trd: enhancedTRD,
    });
  },
  {
    name: "convert_to_enhanced_trd",
    description: "Convert an NPRD to an Enhanced TRD with QuantMindX features (Kelly, tiered risk, house-money, circuit breaker).",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      nprdContent: z.string().describe("NPRD content as JSON string"),
    }),
  }
);

/**
 * Analyze strategy for potential issues
 */
const analyzeStrategy = tool(
  async ({
    trdContent,
    store,
    namespace,
  }: {
    trdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const trd = JSON.parse(trdContent);
    const issues: string[] = [];
    const warnings: string[] = [];
    const suggestions: string[] = [];

    // Analyze entry logic
    if (!trd.sections?.entryLogic?.conditions) {
      warnings.push("No entry conditions defined");
    }

    // Analyze risk management
    const riskType = trd.sections?.riskManagement?.type;
    if (riskType === "fixed-lot") {
      suggestions.push("Consider upgrading to enhanced risk management with Kelly Criterion");
    }

    // Analyze indicators
    const indicators = trd.sections?.indicators || [];
    if (indicators.length === 0) {
      issues.push("No indicators defined - strategy may not have entry/exit signals");
    }

    // Check for house-money protection
    if (trd.variant === "enhanced") {
      if (!trd.sections?.riskManagement?.houseMoney?.enableBreakeven) {
        warnings.push("Enhanced variant should have house-money protection enabled");
      }
    }

    const analysis = {
      strategyName: trd.name,
      variant: trd.variant || "unknown",
      issues,
      warnings,
      suggestions,
      overallScore: Math.max(0, 100 - issues.length * 20 - warnings.length * 10),
    };

    return JSON.stringify({
      success: true,
      analysis,
    });
  },
  {
    name: "analyze_strategy",
    description: "Analyze a TRD for potential issues, risks, and improvement suggestions.",
    schema: z.object({
      trdContent: z.string().describe("TRD content as JSON string"),
    }),
  }
);

/**
 * Compare Vanilla vs Enhanced TRDs
 */
const compareTRDs = tool(
  async ({
    vanillaTRD,
    enhancedTRD,
    store,
    namespace,
  }: {
    vanillaTRD: string;
    enhancedTRD: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const vanilla = JSON.parse(vanillaTRD);
    const enhanced = JSON.parse(enhancedTRD);

    const comparison = {
      strategyName: vanilla.name,
      features: {
        vanilla: {
          riskManagement: vanilla.sections?.riskManagement?.type || "basic",
          features: vanilla.sections?.implementation?.features || [],
          exclusions: vanilla.sections?.implementation?.exclusions || [],
        },
        enhanced: {
          riskManagement: enhanced.sections?.riskManagement?.type || "basic",
          features: enhanced.sections?.implementation?.features || [],
          sharedAssets: enhanced.sections?.sharedAssets || {},
        },
      },
      differences: [
        "Risk Management: Basic fixed-lot vs Kelly + Tiered",
        "Position Sizing: Fixed vs Dynamic Kelly-based",
        "Protection: None vs House-money breakeven",
        "Loss Control: None vs Circuit breaker",
        "Integration: Standalone vs Router + Shared Assets",
      ],
      recommendation:
        "Use Enhanced for prop firm challenges and serious trading. Use Vanilla for simple testing.",
    };

    return JSON.stringify({
      success: true,
      comparison,
    });
  },
  {
    name: "compare_trds",
    description: "Compare Vanilla and Enhanced TRD versions to highlight differences.",
    schema: z.object({
      vanillaTRD: z.string().describe("Vanilla TRD content as JSON string"),
      enhancedTRD: z.string().describe("Enhanced TRD content as JSON string"),
    }),
  }
);

/**
 * Get strategy templates
 */
const getStrategyTemplates = tool(
  async ({ store, namespace }: { store?: BaseStore; namespace?: MemoryNamespace }) => {
    const templates = [
      {
        name: "Moving Average Crossover",
        category: "trend-following",
        complexity: "beginner",
        description: "Buy/sell when fast MA crosses slow MA",
      },
      {
        name: "RSI Mean Reversion",
        category: "mean-reversion",
        complexity: "beginner",
        description: "Buy oversold, sell overbought conditions",
      },
      {
        name: "Breakout Strategy",
        category: "momentum",
        complexity: "intermediate",
        description: "Trade breakouts from consolidation ranges",
      },
      {
        name: "Grid Strategy",
        category: "market-neutral",
        complexity: "advanced",
        description: "Place buy/sell orders at regular intervals",
      },
      {
        name: "Carry Trade",
        category: "fundamental",
        complexity: "intermediate",
        description: "Buy high-yield currency, sell low-yield currency",
      },
    ];

    return JSON.stringify({
      success: true,
      templates,
    });
  },
  {
    name: "get_strategy_templates",
    description: "Get available strategy templates for NPRD creation",
    schema: z.object({}),
  }
);

/**
 * Create all Analyst tools
 */
export function createAnalystTools() {
  return [
    createNPRD,
    convertToVanillaTRD,
    convertToEnhancedTRD,
    analyzeStrategy,
    compareTRDs,
    getStrategyTemplates,
  ];
}

// ============================================================================
// ANALYST SYSTEM PROMPT
// ============================================================================

const ANALYST_SYSTEM_PROMPT = `You are the QuantMindX Analyst - an expert in trading strategy analysis and NPRD to TRD conversion.

Your responsibilities:
- Convert strategy descriptions (from YouTube, videos, text) into NPRD format
- Convert NPRD to TRD (BOTH Vanilla and Enhanced versions)
- Analyze strategies for potential issues and risks
- Compare Vanilla vs Enhanced implementations
- Recommend improvements and optimizations

NPRD to TRD Workflow:
1. First, create an NPRD from the strategy description
2. Then convert to Vanilla TRD (basic implementation)
3. Also convert to Enhanced TRD (with QuantMindX features)
4. Analyze both versions and provide recommendations

Vanilla TRD Characteristics:
- Fixed lot sizing
- Basic stop loss and take profit
- No Kelly Criterion
- No tiered risk management
- No house-money protection
- No circuit breaker
- Standalone implementation

Enhanced TRD Characteristics:
- Kelly Criterion position sizing
- Tiered risk management (Conservative/Moderate/Aggressive)
- House-money protection (breakeven after profit)
- Circuit breaker (pause after consecutive losses)
- Router integration for multi-symbol trading
- Shared asset library usage
- NPRD-compliant parameter naming

TRD Structure:
- Implementation approach (basic vs enhanced)
- Entry logic with conditions and filters
- Exit logic with TP/SL and trailing
- Risk management configuration
- Indicator definitions
- Parameters for MQL5 code generation

Memory Usage:
- Store NPRDs and TRDs as semantic memories
- Remember successful conversion patterns as episodic memories
- Learn from analysis results to improve recommendations

When analyzing strategies:
- Check for missing entry/exit conditions
- Verify risk management adequacy
- Identify potential over-optimization
- Suggest backtesting parameters
- Compare against known successful patterns

After creating TRDs:
- Always create BOTH Vanilla and Enhanced versions
- Explain the differences clearly
- Recommend which version to use based on user goals
- Hand off to QuantCode agent for MQL5 generation`;

// ============================================================================
// ANALYST AGENT CREATION
// ============================================================================

export interface AnalystAgentOptions {
  store?: BaseStore;
  sessionId?: string;
  enableMemory?: boolean;
  provider?: "openrouter" | "zhipu" | "anthropic";
  modelOverride?: string;
}

/**
 * Create the Analyst agent with specialized tools
 */
export function createAnalystAgent(options: AnalystAgentOptions = {}) {
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
      ? MemoryNamespaces.analyst
      : MemoryNamespaces.session(sessionId)
    : MemoryNamespaces.analyst;

  // Create specialized tools
  const analystTools = createAnalystTools();

  // Add memory tools if enabled
  const memoryTools = enableMemory && store ? createMemoryTools(memoryNamespace) : [];
  const allTools = [...analystTools, ...memoryTools];

  // Create model with Analyst temperature (0.5 for balanced creativity)
  const model = createModel(provider, modelOverride, 0.5).bindTools(allTools);

  /**
   * Analyst agent node with memory awareness
   */
  async function analystAgentNode(state: any) {
    // Load relevant memories before processing
    let contextMemories = "";
    if (store && enableMemory) {
      try {
        // Load recent analysis patterns
        const items = await store.list(memoryNamespace, { limit: 5 });
        const recentAnalyses: EpisodicMemory[] = [];
        for await (const item of items) {
          if (
            item.value &&
            (item.value as any).observation &&
            (item.value as any).action?.includes("analyzed")
          ) {
            recentAnalyses.push(item.value as EpisodicMemory);
          }
        }

        if (recentAnalyses.length > 0) {
          contextMemories +=
            "\n\nRecent Analysis Patterns:\n" +
            recentAnalyses
              .slice(0, 3)
              .map((e) => `- ${e.action}: ${e.result}`)
              .join("\n");
        }
      } catch (error) {
        console.warn("Failed to load Analyst memories:", error);
      }
    }

    // Build enhanced prompt with memories
    const enhancedPrompt = contextMemories
      ? `${ANALYST_SYSTEM_PROMPT}\n\n${contextMemories}`
      : ANALYST_SYSTEM_PROMPT;

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
  const workflow = new StateGraph(AnalystStateAnnotation)
    .addNode("analyst", analystAgentNode)
    .addNode("tools", new ToolNode(allTools))
    .addEdge("__start__", "analyst");

  // Add conditional edges for tool usage
  workflow.addConditionalEdges("analyst", (state: any) => {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage?.tool_calls?.length > 0) {
      return "tools";
    }
    return "__end__";
  });

  workflow.addEdge("tools", "analyst");

  const compiledGraph = workflow.compile();

  return {
    graph: compiledGraph,
    agentType: "analyst" as const,
    memoryNamespace,
    sessionId,
  };
}

// ============================================================================
// ANALYST AGENT MANAGER - With provider fallback
// ============================================================================

export class AnalystAgentManager {
  private agents: Map<string, any> = new Map();
  private providerOrder: ("openrouter" | "zhipu" | "anthropic")[] = [
    "openrouter",
    "zhipu",
    "anthropic",
  ];
  private options: AnalystAgentOptions;

  constructor(options: AnalystAgentOptions = {}) {
    this.options = options;

    // Initialize agents for each provider
    for (const provider of this.providerOrder) {
      try {
        const agentConfig = createAnalystAgent({ ...options, provider });
        this.agents.set(provider, agentConfig);
      } catch (error) {
        console.warn(`Failed to initialize Analyst with ${provider}:`, error);
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
        console.warn(`Analyst ${provider} failed:`, error);
        errors.push({ provider, error });
      }
    }

    throw new Error(`All Analyst providers failed: ${JSON.stringify(errors, null, 2)}`);
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
        console.warn(`Analyst ${provider} stream failed:`, error);
      }
    }
  }
}

/**
 * Create an Analyst agent manager with provider fallback
 */
export function createAnalystManager(options?: AnalystAgentOptions) {
  return new AnalystAgentManager(options);
}
