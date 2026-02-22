/**
 * TRD Agent - Technical Requirements Document Generator
 *
 * A LangChain/LangGraph-based agent that generates TRD.md and config JSON
 * from Truth Objects (parsed NPRD or strategy descriptions).
 *
 * Features:
 * - LangGraph state management for workflow orchestration
 * - ZMQ Strategy Router integration for EA registration
 * - Fee-aware Kelly position sizing parameters
 * - Config JSON generation for EA registration
 * - TRD.md generation with technical specifications
 *
 * @module strategies-yt/src/lib/server/agents/trdAgent
 */

import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, Annotation } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import { BaseStore } from "@langchain/langgraph";

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Truth Object - Input format for TRD generation
 * Represents a parsed NPRD or strategy description
 */
export interface TruthObject {
  // Core identification
  title: string;
  version: string;
  description: string;

  // Strategy classification
  strategy_type: "SCALPER" | "STRUCTURAL" | "SWING" | "HFT";
  frequency: "HFT" | "HIGH" | "MEDIUM" | "LOW";
  direction: "BOTH" | "LONG" | "SHORT";

  // Trading logic
  entry_conditions: string[];
  exit_conditions: string[];
  filters?: string[];

  // Risk parameters
  stop_loss_pips: number;
  take_profit_pips: number;
  kelly_fraction: number;

  // Trading preferences
  symbols: string[];
  timeframes: string[];
  sessions: ("ASIAN" | "LONDON" | "NEW_YORK" | "OVERLAP")[];
  trading_hours?: string;

  // Performance targets
  target_win_rate?: number;
  target_sharpe?: number;

  // Metadata
  author?: string;
  source_url?: string;
  tags?: string[];
}

/**
 * ZMQ Strategy Router configuration
 * Defines how the EA connects to the QuantMindX router
 */
export interface ZMQRouterConfig {
  enabled: boolean;
  endpoint: string;              // e.g., "tcp://localhost:5555"
  heartbeat_interval_ms: number;
  message_types: string[];
  subscription_topics: string[];
}

/**
 * Fee-aware Kelly position sizing parameters
 * Calculates optimal position size considering broker fees
 */
export interface FeeAwareKellyParams {
  // Kelly Criterion parameters
  kelly_fraction: number;         // Base Kelly fraction (0.10-0.40)
  win_rate: number;               // Expected win rate (0-1)
  avg_win: number;                // Average winning trade ($)
  avg_loss: number;               // Average losing trade ($)

  // Fee adjustments
  broker_fee_per_lot: number;     // Commission per lot traded
  spread_cost_pips: number;       // Average spread cost in pips
  swap_cost_per_day: number;      // Swap/holding cost per day

  // Risk limits
  max_risk_per_trade: number;     // Maximum risk as % of balance
  max_daily_loss: number;         // Maximum daily loss % (prop firm limit)
  max_drawdown: number;           // Maximum drawdown %

  // Position sizing
  base_lot: number;               // Starting lot size
  max_lot: number;                // Maximum lot cap
  scale_with_account: boolean;    // Scale lots with account growth
}

/**
 * Generated TRD output
 * Contains both TRD.md content and config JSON
 */
export interface GeneratedTRD {
  trd_markdown: string;
  config_json: string;
  metadata: {
    generated_at: string;
    source_truth_version: string;
    zmq_config: ZMQRouterConfig;
    kelly_params: FeeAwareKellyParams;
  };
}

// ============================================================================
// AGENT STATE - LangGraph State Management
// ============================================================================

/**
 * TRD Agent State for LangGraph workflow
 */
const TRDStateAnnotation = Annotation.Root({
  // Input truth object
  truthObject: Annotation<TruthObject | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Messages between nodes
  messages: Annotation<{
    reducer: (a: any[], b: any[]) => a.concat(b);
    default: () => [];
  }>(),

  // Generated outputs
  trdMarkdown: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  configJson: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // ZMQ Router configuration
  zmqConfig: Annotation<ZMQRouterConfig | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Kelly position sizing parameters
  kellyParams: Annotation<FeeAwareKellyParams | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Validation results
  validation: Annotation<Record<string, any>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),

  // Session tracking
  sessionId: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
});

export type TRDState = typeof TRDStateAnnotation.State;

// ============================================================================
// MODEL PROVIDER CONFIGURATION
// ============================================================================

export interface ProviderConfig {
  name: string;
  baseURL: string;
  apiKeyEnv: string;
  model: string;
}

export const PROVIDER_CONFIGS: Record<'openrouter' | 'anthropic', ProviderConfig> = {
  openrouter: {
    name: 'OpenRouter',
    baseURL: 'https://openrouter.ai/api/v1',
    apiKeyEnv: 'OPENROUTER_API_KEY',
    model: 'anthropic/claude-sonnet-4',
  },
  anthropic: {
    name: 'Anthropic',
    baseURL: 'https://api.anthropic.com',
    apiKeyEnv: 'ANTHROPIC_API_KEY',
    model: 'claude-sonnet-4-20250514',
  },
};

function createModel(
  provider: 'openrouter' | 'anthropic' = 'openrouter',
  temperature = 0.7
) {
  const config = PROVIDER_CONFIGS[provider];
  const apiKey = typeof process !== 'undefined' ? process.env[config.apiKeyEnv] : '';

  return new ChatOpenAI(
    {
      model: config.model,
      temperature,
      apiKey,
      streaming: true,
    },
    {
      baseURL: config.baseURL,
    }
  );
}

// ============================================================================
// ZMQ ROUTER INTEGRATION
// ============================================================================

/**
 * Default ZMQ Strategy Router configuration for QuantMindX
 */
export function createDefaultZMQConfig(): ZMQRouterConfig {
  return {
    enabled: true,
    endpoint: "tcp://localhost:5555",
    heartbeat_interval_ms: 5000,
    message_types: [
      "TRADE_OPEN",
      "TRADE_CLOSE",
      "TRADE_MODIFY",
      "HEARTBEAT",
      "RISK_UPDATE"
    ],
    subscription_topics: [
      "risk_multiplier",
      "regime_change",
      "circuit_breaker"
    ]
  };
}

/**
 * Generate ZMQ client code snippet for MQL5
 */
function generateZMQClientCode(config: ZMQRouterConfig): string {
  return `
// ZMQ Strategy Router Integration
// This EA connects to QuantMindX router at ${config.endpoint}

// ZMQ context and socket
#include <Zmq/Zmq.mqh>

input string ROUTER_ENDPOINT = "${config.endpoint}";
input int HEARTBEAT_INTERVAL = ${config.heartbeat_interval_ms};

Context context(${config.endpoint});
Socket socket(context, ZMQ_REQ);

// Initialize ZMQ connection in OnInit()
int OnInit() {
  socket.connect(ROUTER_ENDPOINT);
  SendHeartbeat();
  return INIT_SUCCEEDED;
}

// Send heartbeat to maintain connection
void SendHeartbeat() {
  ZmqMsg request;
  string json = "{\\"type\\": \\"heartbeat\\", \\"ea_name\\": \\"" + EA_Name + "\\"}";
  request.setData(json);
  socket.send(request);

  ZmqMsg response;
  socket.recv(response);
  // Process risk_multiplier from response
}

// OnTrade: Notify router of trade events
void OnTrade() {
  if (PositionsTotal() > 0) {
    NotifyTradeOpen();
  }
}
`;
}

// ============================================================================
// FEE-AWARE KELLY POSITION SIZING
// ============================================================================

/**
 * Calculate fee-aware Kelly position size
 *
 * Kelly Formula: f* = (bp - q) / b
 * Where:
 * - b = avg_win / avg_loss (win/loss ratio)
 * - p = win_rate (probability of win)
 * - q = 1 - p (probability of loss)
 *
 * Fee adjustment reduces effective b by transaction costs
 */
export function calculateFeeAwareKelly(params: FeeAwareKellyParams): {
  kelly_fraction: number;
  optimal_lot_size: number;
  risk_adjusted_fraction: number;
  fee_burn_pct: number;
} {
  // Calculate win/loss ratio
  const b = params.avg_win / params.avg_loss;
  const p = params.win_rate;
  const q = 1 - p;

  // Raw Kelly fraction
  const raw_kelly = (b * p - q) / b;

  // Adjust for fees (reduce effective win by transaction costs)
  const total_fee_per_trade = params.broker_fee_per_lot + params.spread_cost_pips;
  const fee_adjusted_win = params.avg_win - total_fee_per_trade;
  const fee_adjusted_b = fee_adjusted_win / params.avg_loss;

  // Fee-adjusted Kelly fraction
  const fee_adjusted_kelly = (fee_adjusted_b * p - q) / fee_adjusted_b;

  // Ensure Kelly is positive (skip trade if negative)
  const kelly_fraction = Math.max(0, fee_adjusted_kelly);

  // Cap Kelly fraction based on strategy type
  const capped_kelly = Math.min(kelly_fraction, params.kelly_fraction);

  // Calculate optimal lot size based on account balance
  // For $10K account with 2% risk and 50 pip SL:
  // Position size = (10000 * 0.02) / (50 * pip_value)
  const risk_amount = 10000 * params.max_risk_per_trade; // Using default balance
  const pip_value = 10; // Standard lot pip value
  const optimal_lot_size = (risk_amount * capped_kelly) / (params.stop_loss_pips * pip_value);

  // Calculate fee burn percentage
  const expected_trades_per_month = 100;
  const monthly_fee_burn = total_fee_per_trade * expected_trades_per_month;
  const fee_burn_pct = (monthly_fee_burn / 10000) * 100; // % of $10K account

  return {
    kelly_fraction: capped_kelly,
    optimal_lot_size: Math.max(params.base_lot, Math.min(optimal_lot_size, params.max_lot)),
    risk_adjusted_fraction: capped_kelly * 0.8, // Conservative 80% adjustment
    fee_burn_pct,
  };
}

/**
 * Generate Kelly position sizing code snippet for MQL5
 */
function generateKellyMQL5Code(params: FeeAwareKellyParams): string {
  const kelly = calculateFeeAwareKelly(params);

  return `
// Fee-Aware Kelly Position Sizing
// Kelly Fraction: ${kelly.kelly_fraction.toFixed(3)}
// Risk-Adjusted: ${kelly.risk_adjusted_fraction.toFixed(3)}

input double KELLY_FRACTION = ${kelly.kelly_fraction.toFixed(3)};
input double MAX_RISK_PER_TRADE = ${params.max_risk_per_trade};
input double BASE_LOT = ${params.base_lot};
input double MAX_LOT = ${params.max_lot};

// Broker fees for accurate position sizing
input double BROKER_FEE_PER_LOT = ${params.broker_fee_per_lot};
input double AVG_SPREAD_COST = ${params.spread_cost_pips};

/**
 * Calculate fee-aware Kelly position size
 * Adjusts for broker commission and spread costs
 */
double GetKellyLotSize(double balance, double slPips) {
  // Calculate risk amount
  double riskAmount = balance * MAX_RISK_PER_TRADE;

  // Adjust for fees
  double adjustedRisk = riskAmount - BROKER_FEE_PER_LOT - AVG_SPREAD_COST;

  // Apply Kelly fraction
  double kellyRisk = adjustedRisk * KELLY_FRACTION;

  // Convert to lots (assuming standard lot = 100k units, 1 pip = $10)
  double pipValue = 10.0; // For EURUSD standard lot
  double lotSize = kellyRisk / (slPips * pipValue);

  // Normalize and cap
  lotSize = NormalizeDouble(lotSize, 2);
  lotSize = MathMax(BASE_LOT, MathMin(lotSize, MAX_LOT));

  return lotSize;
}

// Example usage in OnTick
void OnTick() {
  double slPips = StopLossPips;
  double balance = AccountInfoDouble(ACCOUNT_BALANCE);
  double lotSize = GetKellyLotSize(balance, slPips);

  // Use lotSize in trade execution
}
`;
}

// ============================================================================
// TRD GENERATION TOOLS
// ============================================================================

/**
 * Tool: Validate Truth Object
 * Validates the input Truth Object structure and completeness
 */
const validateTruthObject = tool(
  async ({ truthObject }: { truthObject: TruthObject }) => {
    const issues: string[] = [];
    const warnings: string[] = [];

    // Required fields
    if (!truthObject.title) issues.push("Missing title");
    if (!truthObject.description) issues.push("Missing description");
    if (!truthObject.strategy_type) issues.push("Missing strategy_type");

    // Validate strategy type
    const validTypes = ["SCALPER", "STRUCTURAL", "SWING", "HFT"];
    if (!validTypes.includes(truthObject.strategy_type)) {
      issues.push(`Invalid strategy_type: ${truthObject.strategy_type}`);
    }

    // Validate entry/exit conditions
    if (!truthObject.entry_conditions?.length) {
      warnings.push("No entry conditions defined");
    }
    if (!truthObject.exit_conditions?.length) {
      warnings.push("No exit conditions defined");
    }

    // Validate Kelly fraction range
    if (truthObject.kelly_fraction < 0.1 || truthObject.kelly_fraction > 0.4) {
      warnings.push(`Kelly fraction ${truthObject.kelly_fraction} outside recommended range [0.1-0.4]`);
    }

    // Validate symbols
    if (!truthObject.symbols?.length) {
      issues.push("No trading symbols specified");
    }

    return {
      valid: issues.length === 0,
      issues,
      warnings,
      completeness: (1 - issues.length / 10) * 100, // Rough completeness score
    };
  },
  {
    name: "validate_truth_object",
    description: "Validate the Truth Object structure and completeness. Returns validation results with issues and warnings.",
    schema: z.object({
      truthObject: z.any().describe("Truth Object to validate"),
    }),
  }
);

/**
 * Tool: Generate TRD Markdown
 * Generates the TRD.md content from Truth Object
 */
const generateTRDMarkdown = tool(
  async ({ truthObject, kellyParams }: {
    truthObject: TruthObject;
    kellyParams?: FeeAwareKellyParams;
  }) => {
    const kelly = kellyParams ? calculateFeeAwareKelly(kellyParams) : null;

    const trd = `# Technical Requirements Document (TRD)

## ${truthObject.title}

**Version**: ${truthObject.version}
**Strategy Type**: ${truthObject.strategy_type}
**Frequency**: ${truthObject.frequency}
**Direction**: ${truthObject.direction}

---

## 1. Strategy Overview

${truthObject.description}

### Strategy Classification

| Parameter | Value |
|-----------|-------|
| Type | ${truthObject.strategy_type} |
| Frequency | ${truthObject.frequency} |
| Direction | ${truthObject.direction} |
| Kelly Fraction | ${truthObject.kelly_fraction} |

### Trading Instruments

**Symbols**: ${truthObject.symbols.join(", ")}
**Timeframes**: ${truthObject.timeframes.join(", ")}
**Sessions**: ${truthObject.sessions.join(", ")}
${truthObject.trading_hours ? `**Trading Hours**: ${truthObject.trading_hours}` : ""}

---

## 2. Entry Conditions

${truthObject.entry_conditions.map((cond, i) => `${i + 1}. ${cond}`).join("\n")}

---

## 3. Exit Conditions

${truthObject.exit_conditions.map((cond, i) => `${i + 1}. ${cond}`).join("\n")}

---

## 4. Risk Management

### Stop Loss & Take Profit

- **Stop Loss**: ${truthObject.stop_loss_pips} pips
- **Take Profit**: ${truthObject.take_profit_pips} pips
- **Risk/Reward Ratio**: ${(truthObject.take_profit_pips / truthObject.stop_loss_pips).toFixed(2)}

### Kelly Position Sizing

**Base Kelly Fraction**: ${truthObject.kelly_fraction}

${kelly ? `
**Fee-Adjusted Calculations**:
- Kelly Fraction (adjusted): ${kelly.kelly_fraction.toFixed(3)}
- Risk-Adjusted Fraction: ${kelly.risk_adjusted_fraction.toFixed(3)}
- Optimal Lot Size: ${kelly.optimal_lot_size.toFixed(2)}
- Monthly Fee Burn: ${kelly.fee_burn_pct.toFixed(2)}%
` : ""}

### Risk Limits

| Parameter | Value |
|-----------|-------|
| Max Risk Per Trade | ${kellyParams?.max_risk_per_trade || 0.02} (2%) |
| Max Daily Loss | ${kellyParams?.max_daily_loss || 0.05} (5%) |
| Max Drawdown | ${kellyParams?.max_drawdown || 0.15} (15%) |

---

## 5. Filters

${truthObject.filters && truthObject.filters.length > 0
      ? truthObject.filters.map((f, i) => `${i + 1}. ${f}`).join("\n")
      : "No additional filters defined."}

---

## 6. ZMQ Strategy Router Integration

The EA integrates with QuantMindX ZMQ Strategy Router for:

- **Real-time risk multiplier updates**
- **Circuit breaker notifications**
- **Regime change alerts**
- **Trade event logging**

### Connection Details

- **Endpoint**: tcp://localhost:5555
- **Heartbeat Interval**: 5000ms
- **Message Types**: TRADE_OPEN, TRADE_CLOSE, TRADE_MODIFY, HEARTBEAT, RISK_UPDATE

---

## 7. MQL5 Implementation Notes

### Required Inputs

\`\`\`mql5
input string EA_Name = "${truthObject.title}";
input int MagicNumber = 100001;
input double BaseLotSize = ${kelly?.optimal_lot_size || 0.01};
input double MaxLotSize = ${kellyParams?.max_lot || 0.5};
input double StopLossPips = ${truthObject.stop_loss_pips};
input double TakeProfitPips = ${truthObject.take_profit_pips};
input string PreferredSymbols = "${truthObject.symbols.join(",")}";
input ENUM_TIMEFRAMES PreferredTimeframe = PERIOD_${truthObject.timeframes[0] || "H1"};
\`\`\`

### Kelly Position Sizing Function

\`\`\`mql5
double GetKellyLotSize(double balance, double slPips) {
  double riskAmount = balance * ${(kellyParams?.max_risk_per_trade || 0.02)};
  double kellyRisk = riskAmount * ${(kelly?.kelly_fraction || truthObject.kelly_fraction).toFixed(3)};
  double lotSize = kellyRisk / (slPips * 10.0);
  return NormalizeDouble(MathMax(${kellyParams?.base_lot || 0.01}, MathMin(lotSize, ${kellyParams?.max_lot || 0.5})), 2);
}
\`\`\`

---

## 8. Backtest Expectations

${truthObject.target_win_rate ? `- **Target Win Rate**: >${truthObject.target_win_rate * 100}%` : ""}
${truthObject.target_sharpe ? `- **Target Sharpe Ratio**: >${truthObject.target_sharpe}` : ""}

### Validation Criteria (PAPER → LIVE)

- Minimum 30 days active
- Sharpe ratio > 1.5
- Win rate > 55%
- Maximum drawdown < 10%
- At least 50 trades executed

---

**Generated**: ${new Date().toISOString()}
**Author**: ${truthObject.author || "QuantMindX"}
${truthObject.source_url ? `**Source**: ${truthObject.source_url}` : ""}
**Tags**: ${truthObject.tags?.join(", ") || "@primal, demo"}
`;

    return { trd_markdown: trd };
  },
  {
    name: "generate_trd_markdown",
    description: "Generate TRD.md content from Truth Object. Includes strategy overview, entry/exit conditions, risk management, and ZMQ integration details.",
    schema: z.object({
      truthObject: z.any().describe("Truth Object with strategy details"),
      kellyParams: z.any().optional().describe("Optional fee-aware Kelly parameters"),
    }),
  }
);

/**
 * Tool: Generate Config JSON
 * Generates the EA config JSON for registration
 */
const generateConfigJSON = tool(
  async ({ truthObject, zmqConfig, kellyParams }: {
    truthObject: TruthObject;
    zmqConfig?: ZMQRouterConfig;
    kellyParams?: FeeAwareKellyParams;
  }) => {
    const kelly = kellyParams ? calculateFeeAwareKelly(kellyParams) : null;
    const safeId = truthObject.title.toLowerCase().replace(/[^a-z0-9]/g, "_");

    const config = {
      ea_id: safeId,
      name: truthObject.title,
      version: truthObject.version,
      description: truthObject.description,

      strategy: {
        type: truthObject.strategy_type,
        frequency: truthObject.frequency,
        direction: truthObject.direction,
      },

      symbols: {
        primary: truthObject.symbols,
        timeframes: truthObject.timeframes,
        symbol_groups: ["majors", "crosses"].filter(g =>
          g === "majors" ? truthObject.symbols.some(s => s.includes("USD")) :
          truthObject.symbols.some(s => !s.includes("USD"))
        ),
      },

      trading_conditions: {
        sessions: truthObject.sessions,
        timezone: "UTC",
        custom_windows: truthObject.trading_hours ? [{
          name: "Trading Window",
          start: truthObject.trading_hours.split("-")[0]?.trim() || "08:00",
          end: truthObject.trading_hours.split("-")[1]?.trim() || "17:00",
          days: ["MON", "TUE", "WED", "THU", "FRI"],
        }] : [],
        volatility: {
          min_atr: 0.0005,
          max_atr: 0.003,
          atr_period: 14,
        },
        preferred_regime: "TRENDING",
        min_spread_pips: 0,
        max_spread_pips: 3,
      },

      risk_parameters: {
        kelly_fraction: kelly?.kelly_fraction || truthObject.kelly_fraction,
        max_risk_per_trade: kellyParams?.max_risk_per_trade || 0.02,
        max_daily_loss: kellyParams?.max_daily_loss || 0.05,
        max_drawdown: kellyParams?.max_drawdown || 0.15,
        max_open_trades: 3,
        correlation_limit: 0.7,
      },

      position_sizing: {
        base_lot: kelly?.optimal_lot_size || kellyParams?.base_lot || 0.01,
        max_lot: kellyParams?.max_lot || 0.5,
        scale_with_account: kellyParams?.scale_with_account ?? true,
        respect_prop_limits: true,
      },

      broker_preferences: {
        preferred_type: "RAW_ECN",
        min_leverage: 100,
        allowed_brokers: [],
        excluded_brokers: [],
      },

      prop_firm: {
        compatible: true,
        supported_firms: ["FTMO", "The5ers", "FundingPips"],
        daily_loss_limit: kellyParams?.max_daily_loss || 0.05,
        max_trailing_drawdown: 0.10,
      },

      zmq_router: zmqConfig || createDefaultZMQConfig(),

      kelly_position_sizing: kellyParams ? {
        enabled: true,
        kelly_fraction: kelly.kelly_fraction,
        risk_adjusted_fraction: kelly.risk_adjusted_fraction,
        fee_burn_pct: kelly.fee_burn_pct,
        broker_fee_per_lot: kellyParams.broker_fee_per_lot,
        spread_cost_pips: kellyParams.spread_cost_pips,
      } : {
        enabled: true,
        kelly_fraction: truthObject.kelly_fraction,
      },

      tags: truthObject.tags || ["@primal", "demo"],
      author: truthObject.author || "QuantMindX",
      created_at: new Date().toISOString(),
    };

    return { config_json: JSON.stringify(config, null, 2) };
  },
  {
    name: "generate_config_json",
    description: "Generate EA config JSON for QuantMindX registration. Includes strategy, symbols, trading conditions, risk parameters, and ZMQ router configuration.",
    schema: z.object({
      truthObject: z.any().describe("Truth Object with strategy details"),
      zmqConfig: z.any().optional().describe("ZMQ Router configuration"),
      kellyParams: z.any().optional().describe("Fee-aware Kelly parameters"),
    }),
  }
);

// ============================================================================
// AGENT GRAPH NODES
// ============================================================================

/**
 * Node: Validate Input
 * Validates the Truth Object before processing
 */
async function validateInputNode(state: TRDState) {
  const result = await validateTruthObject.invoke({ truthObject: state.truthObject });
  const validation = JSON.parse(result);

  return {
    validation,
    messages: [
      new AIMessage(
        validation.valid
          ? "✓ Truth Object validated successfully"
          : `⚠ Validation found ${validation.issues.length} issues`
      ),
    ],
  };
}

/**
 * Node: Generate ZMQ Config
 * Creates ZMQ Strategy Router configuration
 */
async function generateZMQConfigNode(state: TRDState) {
  const zmqConfig = createDefaultZMQConfig();

  return {
    zmqConfig,
    messages: [
      new AIMessage(
        `✓ ZMQ Router config created: ${zmqConfig.endpoint}`
      ),
    ],
  };
}

/**
 * Node: Generate Kelly Parameters
 * Creates fee-aware Kelly position sizing parameters
 */
async function generateKellyParamsNode(state: TRDState) {
  const truth = state.truthObject!;

  // Estimate win rate from strategy type if not provided
  const defaultWinRates = {
    HFT: 0.55,
    SCALPER: 0.55,
    STRUCTURAL: 0.60,
    SWING: 0.65,
  };

  // Estimate avg win/loss from SL/TP ratio
  const winLossRatio = truth.take_profit_pips / truth.stop_loss_pips;
  const avgWin = truth.take_profit_pips * 10; // $ per pip standard lot
  const avgLoss = truth.stop_loss_pips * 10;

  const kellyParams: FeeAwareKellyParams = {
    kelly_fraction: truth.kelly_fraction,
    win_rate: truth.target_win_rate || defaultWinRates[truth.strategy_type],
    avg_win: avgWin,
    avg_loss: avgLoss,
    broker_fee_per_lot: 7, // Typical round-turn commission
    spread_cost_pips: 1.5, // Average spread
    swap_cost_per_day: 0.5,
    max_risk_per_trade: 0.02,
    max_daily_loss: 0.05,
    max_drawdown: 0.15,
    base_lot: 0.01,
    max_lot: 0.5,
    scale_with_account: true,
  };

  const kelly = calculateFeeAwareKelly(kellyParams);

  return {
    kellyParams,
    messages: [
      new AIMessage(
        `✓ Kelly params: ${kelly.kelly_fraction.toFixed(3)} fraction, ` +
        `${kelly.optimal_lot_size.toFixed(2)} lot size, ` +
        `${kelly.fee_burn_pct.toFixed(2)}% fee burn`
      ),
    ],
  };
}

/**
 * Node: Generate TRD Markdown
 * Creates the TRD.md content
 */
async function generateTRDMarkdownNode(state: TRDState) {
  const result = await generateTRDMarkdown.invoke({
    truthObject: state.truthObject,
    kellyParams: state.kellyParams!,
  });

  return {
    trdMarkdown: result.trd_markdown,
    messages: [
      new AIMessage("✓ TRD.md generated"),
    ],
  };
}

/**
 * Node: Generate Config JSON
 * Creates the config JSON content
 */
async function generateConfigJSONNode(state: TRDState) {
  const result = await generateConfigJSON.invoke({
    truthObject: state.truthObject,
    zmqConfig: state.zmqConfig!,
    kellyParams: state.kellyParams!,
  });

  return {
    configJson: result.config_json,
    messages: [
      new AIMessage("✓ config.json generated"),
    ],
  };
}

/**
 * Node: Final Summary
 * Provides completion summary
 */
async function finalSummaryNode(state: TRDState) {
  const truth = state.truthObject!;
  const summary = `
# TRD Generation Complete

**Strategy**: ${truth.title}
**Version**: ${truth.version}
**Type**: ${truth.strategy_type}

## Generated Files

1. **TRD.md** - Technical Requirements Document
2. **config.json** - EA registration config

## Key Parameters

- **Kelly Fraction**: ${state.kellyParams ? calculateFeeAwareKelly(state.kellyParams).kelly_fraction.toFixed(3) : truth.kelly_fraction}
- **ZMQ Endpoint**: ${state.zmqConfig?.endpoint || "tcp://localhost:5555"}
- **Position Size**: ${state.kellyParams ? calculateFeeAwareKelly(state.kellyParams).optimal_lot_size.toFixed(2) : "0.01"} lots

## Next Steps

1. Review generated TRD.md
2. Validate config.json settings
3. Generate MQL5 EA code
4. Deploy to PAPER trading for validation
`;

  return {
    messages: [
      new AIMessage(summary),
    ],
  };
}

// ============================================================================
// AGENT GRAPH BUILDER
// ============================================================================

/**
 * Create the TRD Agent workflow graph
 */
export function createTRDAgentGraph() {
  const tools = [
    validateTruthObject,
    generateTRDMarkdown,
    generateConfigJSON,
  ];

  const model = createModel().bindTools(tools);

  const workflow = new StateGraph(TRDStateAnnotation)
    // Validate input
    .addNode("validate_input", validateInputNode)

    // Generate configurations
    .addNode("generate_zmq_config", generateZMQConfigNode)
    .addNode("generate_kelly_params", generateKellyParamsNode)

    // Generate outputs
    .addNode("generate_trd_markdown", generateTRDMarkdownNode)
    .addNode("generate_config_json", generateConfigJSONNode)

    // Final summary
    .addNode("final_summary", finalSummaryNode)

    // Set entry point
    .addEdge("__start__", "validate_input")

    // Sequential workflow
    .addEdge("validate_input", "generate_zmq_config")
    .addEdge("generate_zmq_config", "generate_kelly_params")
    .addEdge("generate_kelly_params", "generate_trd_markdown")
    .addEdge("generate_trd_markdown", "generate_config_json")
    .addEdge("generate_config_json", "final_summary")
    .addEdge("final_summary", "__end__");

  return workflow.compile();
}

// ============================================================================
// MAIN AGENT CLASS
// ============================================================================

/**
 * TRD Agent - Main agent class for TRD generation
 *
 * @example
 * ```typescript
 * const agent = new TRDAgent();
 * const result = await agent.generate(truthObject);
 * console.log(result.trd_markdown);
 * console.log(result.config_json);
 * ```
 */
export class TRDAgent {
  private graph: ReturnType<typeof createTRDAgentGraph>;

  constructor() {
    this.graph = createTRDAgentGraph();
  }

  /**
   * Generate TRD.md and config.json from Truth Object
   *
   * @param truthObject - Input Truth Object with strategy details
   * @param sessionId - Optional session ID for tracking
   * @returns GeneratedTRD with both outputs
   */
  async generate(
    truthObject: TruthObject,
    sessionId?: string
  ): Promise<GeneratedTRD> {
    const initialState: Partial<TRDState> = {
      truthObject,
      sessionId: sessionId || `trd_${Date.now()}`,
      messages: [new HumanMessage(`Generate TRD for: ${truthObject.title}`)],
    };

    const finalState = await this.graph.invoke(initialState);

    return {
      trd_markdown: finalState.trdMarkdown || "",
      config_json: finalState.configJson || "",
      metadata: {
        generated_at: new Date().toISOString(),
        source_truth_version: truthObject.version,
        zmq_config: finalState.zmqConfig || createDefaultZMQConfig(),
        kelly_params: finalState.kellyParams || undefined,
      },
    };
  }

  /**
   * Stream TRD generation progress
   *
   * @param truthObject - Input Truth Object
   * @param sessionId - Optional session ID
   * @returns Async generator of progress updates
   */
  async *stream(
    truthObject: TruthObject,
    sessionId?: string
  ): AsyncGenerator<{ step: string; data: any }, void, unknown> {
    const initialState: Partial<TRDState> = {
      truthObject,
      sessionId: sessionId || `trd_${Date.now()}`,
      messages: [new HumanMessage(`Generate TRD for: ${truthObject.title}`)],
    };

    for await (const event of this.graph.stream(initialState)) {
      for (const [node, state] of Object.entries(event)) {
        if (typeof state === "object" && state !== null) {
          yield {
            step: node,
            data: state,
          };
        }
      }
    }
  }
}

// ============================================================================
// CONVENIENCE FUNCTIONS
// ============================================================================

/**
 * Quick TRD generation from a minimal Truth Object
 *
 * @param minimal - Minimal strategy details
 * @returns Generated TRD outputs
 */
export async function generateTRD(minimal: {
  title: string;
  description: string;
  strategy_type: TruthObject["strategy_type"];
  symbols: string[];
  stop_loss_pips: number;
  take_profit_pips: number;
}): Promise<GeneratedTRD> {
  const agent = new TRDAgent();

  const truthObject: TruthObject = {
    title: minimal.title,
    version: "1.0.0",
    description: minimal.description,
    strategy_type: minimal.strategy_type,
    frequency: "MEDIUM",
    direction: "BOTH",
    entry_conditions: ["Define entry conditions"],
    exit_conditions: ["Define exit conditions"],
    stop_loss_pips: minimal.stop_loss_pips,
    take_profit_pips: minimal.take_profit_pips,
    kelly_fraction: 0.25,
    symbols: minimal.symbols,
    timeframes: ["H1"],
    sessions: ["LONDON", "NEW_YORK"],
    tags: ["@primal", "demo"],
  };

  return agent.generate(truthObject);
}

/**
 * Create default fee-aware Kelly parameters
 */
export function createDefaultKellyParams(
  strategyType: TruthObject["strategy_type"]
): FeeAwareKellyParams {
  const kellyByType = {
    HFT: 0.10,
    SCALPER: 0.20,
    STRUCTURAL: 0.25,
    SWING: 0.30,
  };

  return {
    kelly_fraction: kellyByType[strategyType],
    win_rate: 0.60,
    avg_win: 100,
    avg_loss: 50,
    broker_fee_per_lot: 7,
    spread_cost_pips: 1.5,
    swap_cost_per_day: 0.5,
    max_risk_per_trade: 0.02,
    max_daily_loss: 0.05,
    max_drawdown: 0.15,
    base_lot: 0.01,
    max_lot: 0.5,
    scale_with_account: true,
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

export default TRDAgent;

// Re-export key types and utilities
export {
  validateTruthObject,
  generateTRDMarkdown,
  generateConfigJSON,
  calculateFeeAwareKelly,
  createDefaultZMQConfig,
  createDefaultKellyParams,
};
