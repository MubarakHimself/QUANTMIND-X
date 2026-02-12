# OpenRouter + LangChain + LangGraph TypeScript Integration Guide

## Research Summary

Comprehensive guide on integrating OpenRouter with LangChain and LangGraph.js for TypeScript applications, including multi-provider configurations, fallback strategies, and complete working examples.

---

## Table of Contents

1. [OpenRouter + LangChain TypeScript Integration](#1-openrouter--langchain-typescript-integration)
2. [LangGraph.js with Custom Providers](#2-langgraphjs-with-custom-providers)
3. [OpenRouter Model Naming Convention](#3-openrouter-model-naming-convention)
4. [Zhipu AI (Z.ai) Integration](#4-zhipu-ai-zai-integration)
5. [Provider Fallback Strategy](#5-provider-fallback-strategy)
6. [Complete Working Example](#6-complete-working-example)
7. [Best Practices](#7-best-practices)

---

## 1. OpenRouter + LangChain TypeScript Integration

### Overview

OpenRouter provides a unified API to access multiple LLM providers through a single endpoint. LangChain's `ChatOpenAI` class can be configured to use OpenRouter by customizing the `baseURL`.

### Basic Setup

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const chat = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4",
  temperature: 0.8,
  streaming: true,
  apiKey: process.env.OPENROUTER_API_KEY,
}, {
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://your-site.com", // Optional. Site URL for rankings on openrouter.ai
    "X-Title": "Your App Name", // Optional. Site title for rankings on openrouter.ai
  },
});

// Example usage
const response = await chat.invoke([
  new SystemMessage("You are a helpful assistant."),
  new HumanMessage("Hello, how are you?"),
]);
```

### Configuration Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `apiKey` | string | Yes | Your OpenRouter API key |
| `model` | string | Yes | Model identifier (e.g., `anthropic/claude-sonnet-4`) |
| `temperature` | number | No | Sampling temperature (0-1) |
| `streaming` | boolean | No | Enable streaming responses |
| `baseURL` | string | Yes | `https://openrouter.ai/api/v1` |
| `defaultHeaders` | object | No | Custom headers for OpenRouter rankings |

### Disabling Streaming Usage Metadata

Some proxies don't support the `stream_options` parameter:

```typescript
const llmWithoutStreamUsage = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4",
  temperature: 0.9,
  streamUsage: false,
  apiKey: process.env.OPENROUTER_API_KEY,
}, {
  baseURL: "https://openrouter.ai/api/v1",
});
```

---

## 2. LangGraph.js with Custom Providers

### StateGraph Configuration with OpenRouter

```typescript
import { StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { Annotation } from "@langchain/langgraph";

// Define state
const StateAnnotation = Annotation.Root({
  messages: Annotation({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

// Create model with OpenRouter
const model = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4",
  temperature: 0,
  apiKey: process.env.OPENROUTER_API_KEY,
}, {
  baseURL: "https://openrouter.ai/api/v1",
  defaultHeaders: {
    "HTTP-Referer": "https://your-app.com",
    "X-Title": "My AI Assistant",
  },
}).bindTools(tools);

// Define nodes
async function agentNode(state: typeof StateAnnotation.State) {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
}

const toolNode = new ToolNode(tools);

// Build graph
const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", agentNode)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges(
    "agent",
    (state) => {
      const lastMessage = state.messages[state.messages.length - 1];
      return lastMessage.tool_calls?.length > 0 ? "tools" : "__end__";
    },
    {
      tools: "tools",
      __end__: "__end__",
    }
  )
  .addEdge("tools", "agent");

export const graph = workflow.compile();
```

### Using initChatModel for Universal Provider Support

```typescript
import { initChatModel } from "langchain/chat_models/universal";

// Initialize with OpenRouter-compatible models
const claudeModel = await initChatModel("anthropic:claude-sonnet-4", {
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
  },
});

// Works with streaming
const stream = await claudeModel.stream([
  new HumanMessage("Tell me a joke")
]);

for await (const chunk of stream) {
  process.stdout.write(chunk.content);
}
```

---

## 3. OpenRouter Model Naming Convention

### Format

OpenRouter uses the format: **`provider/model-name`**

### Common Examples

| Provider | Model Name | OpenRouter ID |
|----------|------------|---------------|
| Anthropic | Claude Sonnet 4 | `anthropic/claude-sonnet-4` |
| Anthropic | Claude Opus 4.5 | `anthropic/claude-opus-4.5` |
| OpenAI | GPT-4o | `openai/gpt-4o` |
| OpenAI | GPT-4o Mini | `openai/gpt-4o-mini` |
| Google | Gemini 2.5 Flash | `google/gemini-2.5-flash-preview` |
| Meta | Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` |
| DeepSeek | DeepSeek V3 | `deepseek/deepseek-v3.2` |
| Mistral | Mixtral 8x7B | `mistralai/mixtral-8x7b-instruct` |

### Model Shortcuts

```typescript
// Nitro: prioritize throughput
const model = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4:nitro", // Same as provider.sort = "throughput"
});

// Floor: prioritize lowest price
const model = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4:floor", // Same as provider.sort = "price"
});
```

---

## 4. Zhipu AI (Z.ai) Integration

### Native LangChain Integration

```typescript
import { ChatZhipuAI } from "@langchain/community/chat_models/zhipuai";
import { HumanMessage } from "@langchain/core/messages";

// Default model is glm-3-turbo
const glm3turbo = new ChatZhipuAI({
  zhipuAIApiKey: process.env.ZHIPUAI_API_KEY,
});

// Use glm-4
const glm4 = new ChatZhipuAI({
  model: "glm-4",
  temperature: 1,
  zhipuAIApiKey: process.env.ZHIPUAI_API_KEY,
});

const messages = [new HumanMessage("Hello")];
const res = await glm4.invoke(messages);
```

### Using Zhipu AI via OpenRouter

```typescript
import { ChatOpenAI } from "@langchain/openai";

const glmModel = new ChatOpenAI({
  model: "zhipu/glm-4-flash", // Available via OpenRouter
  apiKey: process.env.OPENROUTER_API_KEY,
}, {
  baseURL: "https://openrouter.ai/api/v1",
});
```

---

## 5. Provider Fallback Strategy

### LangChain Built-in Fallback Middleware

```typescript
import { createAgent, modelFallbackMiddleware } from "langchain";

const agent = createAgent({
  model: "anthropic/claude-sonnet-4",
  tools: [],
  middleware: [
    modelFallbackMiddleware(
      "openai/gpt-4o",
      "google/gemini-2.0-flash-exp"
    ),
  ],
});
```

### Custom Fallback Implementation

```typescript
import { ChatOpenAI } from "@langchain/openai";

class FallbackModelManager {
  private models: ChatOpenAI[];
  private currentIndex = 0;

  constructor(configs: Array<{ model: string; baseURL?: string }>) {
    this.models = configs.map(
      (config) =>
        new ChatOpenAI({
          model: config.model,
          apiKey: process.env.OPENROUTER_API_KEY,
        },
        {
          baseURL: config.baseURL || "https://openrouter.ai/api/v1",
        })
    );
  }

  async invoke(messages: any[]): Promise<any> {
    const errors = [];

    for (let i = 0; i < this.models.length; i++) {
      const model = this.models[(this.currentIndex + i) % this.models.length];
      try {
        const response = await model.invoke(messages);
        this.currentIndex = (this.currentIndex + i) % this.models.length;
        return response;
      } catch (error) {
        errors.push({ model: model.modelName, error });
        console.warn(`Failed with ${model.modelName}:`, error);
      }
    }

    throw new Error(
      `All models failed: ${JSON.stringify(errors, null, 2)}`
    );
  }
}

// Usage
const fallbackManager = new FallbackModelManager([
  { model: "anthropic/claude-sonnet-4" },
  { model: "openai/gpt-4o" },
  { model: "google/gemini-2.0-flash-exp" },
]);

const response = await fallbackManager.invoke([new HumanMessage("Hello")]);
```

### Exponential Backoff Retry

```typescript
import { modelRetryMiddleware } from "langchain";

const agent = createAgent({
  model: "anthropic/claude-sonnet-4",
  tools: [],
  middleware: [
    modelRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2.0,
      initialDelayMs: 1000,
      retryOn: (error: Error) => {
        // Retry on rate limits and server errors
        if (error.name === "RateLimitError") return true;
        if (error.name === "HTTPError" && "statusCode" in error) {
          const statusCode = (error as any).statusCode;
          return statusCode === 429 || statusCode >= 500;
        }
        return false;
      },
      onFailure: "continue", // Return error message instead of throwing
    }),
  ],
});
```

---

## 6. Complete Working Example

### OpenRouter Agent with LangGraph.js

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";

// Define tools
const searchTool = tool(
  async ({ query }) => {
    // Simulated search
    return `Search results for: ${query}`;
  },
  {
    name: "search",
    description: "Search the web for information",
    schema: z.object({
      query: z.string(),
    }),
  }
);

const calculatorTool = tool(
  async ({ expression }) => {
    try {
      return `Result: ${eval(expression)}`;
    } catch {
      return "Invalid expression";
    }
  },
  {
    name: "calculator",
    description: "Evaluate mathematical expressions",
    schema: z.object({
      expression: z.string(),
    }),
  }
);

const tools = [searchTool, calculatorTool];

// Create model with OpenRouter
const createOpenRouterModel = (model: string) => {
  return new ChatOpenAI({
    model,
    temperature: 0,
    apiKey: process.env.OPENROUTER_API_KEY!,
    streaming: true,
  }, {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": "https://your-app.com",
      "X-Title": "OpenRouter Agent",
    },
  });
};

const primaryModel = createOpenRouterModel("anthropic/claude-sonnet-4")
  .bindTools(tools);

const fallbackModel = createOpenRouterModel("openai/gpt-4o")
  .bindTools(tools);

// Define state
const StateAnnotation = Annotation.Root({
  messages: Annotation({
    reducer: (a: any[], b: any[]) => a.concat(b),
    default: () => [],
  }),
});

// Define agent node with fallback
async function callModel(state: typeof StateAnnotation.State) {
  try {
    const response = await primaryModel.invoke(state.messages);
    return { messages: [response] };
  } catch (primaryError) {
    console.warn("Primary model failed, trying fallback:", primaryError);
    const response = await fallbackModel.invoke(state.messages);
    return { messages: [response] };
  }
}

// Build graph
const workflow = new StateGraph(StateAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", new ToolNode(tools))
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", toolsCondition)
  .addEdge("tools", "agent");

export const app = workflow.compile();

// Usage example
async function main() {
  const config = { configurable: { thread_id: "test-thread" } };

  const response = await app.invoke(
    {
      messages: [
        new HumanMessage(
          "What's the weather like today? Also, calculate 25 * 17"
        ),
      ],
    },
    config
  );

  console.log(response.messages[response.messages.length - 1].content);
}

main().catch(console.error);
```

### Multi-Provider Router with OpenRouter

```typescript
import { ChatOpenAI } from "@langchain/openai";

interface ProviderConfig {
  name: string;
  model: string;
  baseURL?: string;
  priority: number;
}

class MultiProviderRouter {
  private providers: Map<string, ChatOpenAI>;

  constructor(configs: ProviderConfig[]) {
    this.providers = new Map();

    // Sort by priority
    configs
      .sort((a, b) => a.priority - b.priority)
      .forEach((config) => {
        this.providers.set(
          config.name,
          new ChatOpenAI(
            {
              model: config.model,
              apiKey: process.env.OPENROUTER_API_KEY,
              temperature: 0,
            },
            {
              baseURL: config.baseURL || "https://openrouter.ai/api/v1",
            }
          )
        );
      });
  }

  async chat(
    providerName: string,
    messages: any[],
    options?: any
  ): Promise<any> {
    const model = this.providers.get(providerName);
    if (!model) {
      throw new Error(`Provider ${providerName} not found`);
    }

    try {
      return await model.invoke(messages, options);
    } catch (error) {
      console.error(`Error with ${providerName}:`, error);
      throw error;
    }
  }

  async chatWithFallback(
    providerOrder: string[],
    messages: any[],
    options?: any
  ): Promise<any> {
    const errors = [];

    for (const providerName of providerOrder) {
      try {
        return await this.chat(providerName, messages, options);
      } catch (error) {
        errors.push({ provider: providerName, error });
      }
    }

    throw new Error(
      `All providers failed: ${JSON.stringify(errors, null, 2)}`
    );
  }
}

// Usage
const router = new MultiProviderRouter([
  {
    name: "anthropic",
    model: "anthropic/claude-sonnet-4",
    priority: 1,
  },
  {
    name: "openai",
    model: "openai/gpt-4o",
    priority: 2,
  },
  {
    name: "google",
    model: "google/gemini-2.0-flash-exp",
    priority: 3,
  },
]);

// Try Anthropic, fall back to OpenAI, then Google
const response = await router.chatWithFallback(
  ["anthropic", "openai", "google"],
  [new HumanMessage("Hello!")]
);
```

---

## 7. Best Practices

### 1. Environment Configuration

```typescript
// config/providers.ts
export const providerConfigs = {
  openrouter: {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
      "HTTP-Referer": process.env.SITE_URL,
      "X-Title": process.env.APP_NAME,
    },
  },
  models: {
    primary: "anthropic/claude-sonnet-4",
    fallback: "openai/gpt-4o",
    budget: "google/gemini-2.0-flash-exp",
  },
};
```

### 2. Error Handling

```typescript
import { modelRetryMiddleware, toolRetryMiddleware } from "langchain";

const agent = createAgent({
  model: "anthropic/claude-sonnet-4",
  tools: [],
  middleware: [
    // Model-level retry
    modelRetryMiddleware({
      maxRetries: 3,
      backoffFactor: 2.0,
      onFailure: (error) => `Model error: ${error.message}`,
    }),
    // Tool-level retry
    toolRetryMiddleware({
      maxRetries: 2,
      retryOn: [NetworkError, TimeoutError],
    }),
  ],
});
```

### 3. Cost Optimization

```typescript
// Use cheaper models for simple tasks
const models = {
  reasoning: createOpenRouterModel("anthropic/claude-sonnet-4"),
  drafting: createOpenRouterModel("google/gemini-2.0-flash-exp"),
  summarization: createOpenRouterModel("openai/gpt-4o-mini"),
};

async function processTask(task: string, complexity: "low" | "medium" | "high") {
  const model = complexity === "low" ? models.drafting :
                complexity === "medium" ? models.summarization :
                models.reasoning;

  return await model.invoke([new HumanMessage(task)]);
}
```

### 4. Performance Thresholds (OpenRouter)

```typescript
const performanceModel = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4",
  apiKey: process.env.OPENROUTER_API_KEY,
});

// When streaming to OpenRouter directly, you can use provider preferences
const response = await performanceModel.invoke([new HumanMessage("Hello")], {
  // These would be passed as body parameters in raw API calls
  // provider: {
  //   preferred_max_latency: { p90: 3 },
  //   preferred_min_throughput: { p50: 100 },
  // }
});
```

### 5. Monitoring and Observability

```typescript
import { HumanMessage } from "@langchain/core/messages";

class MonitoredModel extends ChatOpenAI {
  private metrics = {
    totalCalls: 0,
    successfulCalls: 0,
    failedCalls: 0,
    totalLatency: 0,
  };

  async invoke(messages: any[], options?: any): Promise<any> {
    this.metrics.totalCalls++;
    const startTime = Date.now();

    try {
      const response = await super.invoke(messages, options);
      this.metrics.successfulCalls++;
      const latency = Date.now() - startTime;
      this.metrics.totalLatency += latency;
      return response;
    } catch (error) {
      this.metrics.failedCalls++;
      throw error;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      averageLatency: this.metrics.totalLatency / this.metrics.successfulCalls,
      successRate: this.metrics.successfulCalls / this.metrics.totalCalls,
    };
  }
}

const monitoredModel = new MonitoredModel({
  model: "anthropic/claude-sonnet-4",
  apiKey: process.env.OPENROUTER_API_KEY,
}, {
  baseURL: "https://openrouter.ai/api/v1",
});
```

### 6. Security Considerations

```typescript
// Never expose API keys in client-side code
// Always use environment variables
const model = new ChatOpenAI({
  model: "anthropic/claude-sonnet-4",
  apiKey: process.env.OPENROUTER_API_KEY, // Loaded from environment
}, {
  baseURL: "https://openrouter.ai/api/v1",
});

// For sensitive data, enable Zero Data Retention
// (configured via OpenRouter dashboard or API parameters)
```

---

## Sources

- [OpenRouter LangChain Integration](https://openrouter.ai/docs/guides/community/langchain)
- [LangChain ChatOpenAI Documentation](https://docs.langchain.com/oss/javascript/integrations/chat/openai)
- [OpenRouter Provider Selection Guide](https://openrouter.ai/docs/guides/routing/provider-selection)
- [LangChain Built-in Middleware](https://docs.langchain.com/oss/javascript/langchain/middleware/built-in)
- [LangChain.js Repository](https://github.com/langchain-ai/langchainjs)

---

## Key Takeaways

1. **OpenRouter works seamlessly with LangChain** via `ChatOpenAI` with custom `baseURL`
2. **Model naming follows `provider/model` format** (e.g., `anthropic/claude-sonnet-4`)
3. **LangGraph.js supports custom providers** through standard model initialization
4. **Built-in middleware provides fallback and retry** capabilities out of the box
5. **Zhipu AI has native LangChain support** but can also be accessed via OpenRouter
6. **Performance and cost optimization** can be achieved through strategic model selection and provider routing
