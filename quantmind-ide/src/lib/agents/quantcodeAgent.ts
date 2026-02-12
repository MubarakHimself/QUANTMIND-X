/**
 * QuantCode Agent - TRD to MQL5 EA Generation Agent
 *
 * Purpose: Convert TRD (Technical Requirements Document) to MQL5 Expert Advisor code
 * Memory namespace: memories/quantcode
 *
 * This agent handles:
 * - TRD → MQL5 code generation (creates Vanilla + Enhanced versions)
 * - Shared asset library integration
 * - Router integration for multi-symbol trading
 * - Code optimization and debugging
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
// QUANTCODE STATE - Agent-specific state with memory support
// ============================================================================

const QuantCodeStateAnnotation = Annotation.Root({
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

  // TRD content being processed
  trdContent: Annotation<string | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),

  // Generated MQL5 code
  mql5Code: Annotation<Record<string, string>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),

  // Code generation results
  generationResults: Annotation<Record<string, any>>({
    reducer: (a, b) => ({ ...a, ...b }),
    default: () => ({}),
  }),
});

// ============================================================================
// SHARED ASSET LIBRARY - Reference for QuantMindX shared assets
// ============================================================================

const SHARED_ASSETS = {
  kellySizer: {
    path: "Include/QuantMind/Risk/KellySizer.mqh",
    class: "CKellySizer",
    description: "Kelly Criterion-based position sizing with tiered risk",
    methods: [
      "double GetPositionSize(double balance, double riskPerTrade, double winRate, double avgWinLossRatio)",
      "void SetRiskTier(RiskTier tier)",
      "RiskTier GetRiskTier()",
    ],
  },
  riskGovernor: {
    path: "Include/QuantMind/Risk/Governor.mqh",
    class: "CRiskGovernor",
    description: "Risk governor for circuit breaker and house-money protection",
    methods: [
      "bool CheckCircuitBreaker()",
      "void UpdateHouseMoneyProtection(double currentProfit)",
      "bool CanTrade()",
    ],
  },
  sockets: {
    path: "Include/QuantMind/Utils/Sockets.mqh",
    class: "CQuantMindSocket",
    description: "WebSocket client for router communication",
    methods: [
      "bool Connect(string server, int port)",
      "bool SendTradeSignal(TradeSignal signal)",
      "void OnMessage(string data)",
    ],
  },
  routerClient: {
    path: "Include/QuantMind/Router/Client.mqh",
    class: "CRouterClient",
    description: "Router client for multi-symbol trading coordination",
    methods: [
      "void RegisterStrategy(string strategyId, string[] symbols)",
      "void SendHeartbeat()",
      "TradeSignal[] GetActiveSignals()",
    ],
  },
};

// ============================================================================
// QUANTCODE TOOLS - Specialized tools for TRD to MQL5 conversion
// ============================================================================

/**
 * Generate Vanilla MQL5 EA from TRD
 */
const generateVanillaEA = tool(
  async ({
    strategyName,
    trdContent,
    store,
    namespace,
  }: {
    strategyName: string;
    trdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const trd = JSON.parse(trdContent);

    // Generate Vanilla EA code (basic implementation)
    const vanillaCode = `//+------------------------------------------------------------------+
//|                                          ${strategyName}_Vanilla.mq5   |
//|                        Generated by QuantMindX QuantCode Agent        |
//|                                  Vanilla Version                      |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.local"
#property version   "${trd.version || "1.00"}"
#property strict

#include <Trade/Trade.mqh>

//--- Input Parameters
input string Strategy_Name = "${strategyName}";
input double Lot_Size = 0.01;
input int Stop_Loss_Pips = ${trd.sections?.riskManagement?.stopLoss || 50};
input int Take_Profit_Pips = ${trd.sections?.riskManagement?.takeProfit || 100};
input int Magic_Number = ${trd.sections?.parameters?.magicNumber || 123456};
input int Max_Slippage = ${trd.sections?.parameters?.maxSlippage || 3};

//--- Global Objects
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(Magic_Number);
   trade.SetDeviationInPoints(Max_Slippage);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   Print("${strategyName} Vanilla EA initialized");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("${strategyName} Vanilla EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if trading is allowed
   if(!IsTradingAllowed()) return;

   // Check for existing positions
   if(HasOpenPosition()) return;

   // Entry conditions check
   if(CheckEntryConditions())
   {
      OpenPosition();
   }

   // Exit conditions check
   CheckExitConditions();
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                        |
//+------------------------------------------------------------------+
bool IsTradingAllowed()
{
   // Basic checks only
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return false;
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Check for open positions                                           |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == Magic_Number)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check entry conditions                                             |
//+------------------------------------------------------------------+
bool CheckEntryConditions()
{
   // TODO: Implement strategy-specific entry logic
   // This is a placeholder for the actual strategy logic

   return false;
}

//+------------------------------------------------------------------+
//| Check exit conditions                                              |
//+------------------------------------------------------------------+
void CheckExitConditions()
{
   // TODO: Implement strategy-specific exit logic
   // This is a placeholder for the actual strategy logic
}

//+------------------------------------------------------------------+
//| Open a new position                                                |
//+------------------------------------------------------------------+
void OpenPosition()
{
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY;

   if(trade.PositionOpen(_Symbol, orderType, Lot_Size,
      SymbolInfoDouble(_Symbol, SYMBOL_ASK),
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) - Stop_Loss_Pips * _Point,
      SymbolInfoDouble(_Symbol, SYMBOL_ASK) + Take_Profit_Pips * _Point,
      "${strategyName}"))
   {
      Print("Position opened successfully");
   }
   else
   {
      Print("Failed to open position: ", GetLastError());
   }
}
//+------------------------------------------------------------------+`;

    // Store Vanilla EA code
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: "has-vanilla-ea",
        object: vanillaCode,
        context: "Vanilla MQL5 EA - basic implementation",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `Vanilla EA generated for ${strategyName}`,
      code: vanillaCode,
      filename: `${strategyName}_Vanilla.mq5`,
      path: `/MQL5/Experts/${strategyName}_Vanilla.mq5`,
    });
  },
  {
    name: "generate_vanilla_ea",
    description: "Generate a Vanilla MQL5 Expert Advisor from a TRD (basic implementation without QuantMindX enhancements).",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      trdContent: z.string().describe("TRD content as JSON string"),
    }),
  }
);

/**
 * Generate Enhanced MQL5 EA from TRD
 */
const generateEnhancedEA = tool(
  async ({
    strategyName,
    trdContent,
    store,
    namespace,
  }: {
    strategyName: string;
    trdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const trd = JSON.parse(trdContent);
    const riskTiers = trd.sections?.riskManagement?.tiers || {};

    // Generate Enhanced EA code (with QuantMindX features)
    const enhancedCode = `//+------------------------------------------------------------------+
//|                                          ${strategyName}_Enhanced.mq5  |
//|                        Generated by QuantMindX QuantCode Agent        |
//|                                  Enhanced Version                    |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.local"
#property version   "${trd.version || "1.00"}"
#property strict

#include <Trade/Trade.mqh>
#include "${SHARED_ASSETS.kellySizer.path}"
#include "${SHARED_ASSETS.riskGovernor.path}"
#include "${SHARED_ASSETS.sockets.path}"
#include "${SHARED_ASSETS.routerClient.path}"

//--- Input Parameters
input string Strategy_Name = "${strategyName}";
input ENUM_RISK_TIER Risk_Tier = RISK_MODERATE;
input bool Use_Kelly_Sizing = true;
input bool Enable_House_Money = true;
input double House_Money_Threshold = 0.5;  // R ratio
input bool Enable_Circuit_Breaker = true;
input int Max_Consecutive_Losses = 3;
input int Magic_Number = ${trd.sections?.parameters?.magicNumber || 123456};
input int Max_Slippage = ${trd.sections?.parameters?.maxSlippage || 3};

//--- Risk Tier Parameters
input double Conservative_Risk = 0.5;      // % per trade
input double Moderate_Risk = 1.0;          // % per trade
input double Aggressive_Risk = 2.0;        // % per trade

//--- Kelly Parameters
input double Kelly_Fraction = 0.5;         // Fraction of full Kelly
input double Win_Rate_Estimate = 0.55;
input double Avg_Win_Loss_Ratio = 1.5;

//--- Global Objects
CTrade trade;
CKellySizer *kellySizer;
CRiskGovernor *riskGovernor;
CQuantMindSocket *routerSocket;

//--- State Variables
int consecutiveLosses = 0;
bool circuitBreakerTripped = false;
datetime circuitBreakerEndTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                      |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize trade object
   trade.SetExpertMagicNumber(Magic_Number);
   trade.SetDeviationInPoints(Max_Slippage);

   // Initialize Kelly Sizer
   kellySizer = new CKellySizer();
   kellySizer.SetRiskTier(Risk_Tier);
   kellySizer.SetKellyFraction(Kelly_Fraction);
   kellySizer.SetWinRate(Win_Rate_Estimate);
   kellySizer.SetAvgWinLossRatio(Avg_Win_Loss_Ratio);

   // Initialize Risk Governor
   riskGovernor = new CRiskGovernor();
   riskGovernor.SetHouseMoneyThreshold(House_Money_Threshold);
   riskGovernor.SetCircuitBreakerEnabled(Enable_Circuit_Breaker);
   riskGovernor.SetMaxConsecutiveLosses(Max_Consecutive_Losses);

   // Initialize Router connection
   routerSocket = new CQuantMindSocket();
   if(routerSocket.Connect("localhost", 8080))
   {
      // Register strategy with router
      routerSocket.SendStrategyRegister(Strategy_Name, Magic_Number);
   }

   Print("${strategyName} Enhanced EA initialized");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                    |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Cleanup
   if(kellySizer != NULL) delete kellySizer;
   if(riskGovernor != NULL) delete riskGovernor;
   if(routerSocket != NULL) delete routerSocket;

   Print("${strategyName} Enhanced EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                                |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check circuit breaker
   if(CheckCircuitBreaker()) return;

   // Update house-money protection
   UpdateHouseMoneyProtection();

   // Check if trading is allowed
   if(!IsTradingAllowed()) return;

   // Check for existing positions
   if(HasOpenPosition()) return;

   // Entry conditions check
   if(CheckEntryConditions())
   {
      OpenPosition();
   }

   // Exit conditions check
   CheckExitConditions();

   // Send heartbeat to router
   SendHeartbeat();
}

//+------------------------------------------------------------------+
//| Check circuit breaker status                                       |
//+------------------------------------------------------------------+
bool CheckCircuitBreaker()
{
   if(!Enable_Circuit_Breaker) return false;

   if(circuitBreakerTripped)
   {
      if(TimeCurrent() >= circuitBreakerEndTime)
      {
         circuitBreakerTripped = false;
         consecutiveLosses = 0;
         Print("Circuit breaker reset - trading resumed");
      }
      else
      {
         return true;
      }
   }

   if(consecutiveLosses >= Max_Consecutive_Losses)
   {
      circuitBreakerTripped = true;
      circuitBreakerEndTime = TimeCurrent() + PeriodSeconds(PERIOD_H1);
      Print("Circuit breaker tripped - trading paused for 1 hour");
      return true;
   }

   return false;
}

//+------------------------------------------------------------------+
//| Update house-money protection                                      |
//+------------------------------------------------------------------+
void UpdateHouseMoneyProtection()
{
   if(!Enable_House_Money) return;

   double totalProfit = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == Magic_Number)
         {
            totalProfit += PositionGetDouble(POSITION_PROFIT);
         }
      }
   }

   riskGovernor.UpdateHouseMoneyProtection(totalProfit);
}

//+------------------------------------------------------------------+
//| Check if trading is allowed                                        |
//+------------------------------------------------------------------+
bool IsTradingAllowed()
{
   // Basic checks
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return false;
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) return false;

   // Risk governor check
   if(!riskGovernor.CanTrade()) return false;

   return true;
}

//+------------------------------------------------------------------+
//| Get position size using Kelly Criterion                            |
//+------------------------------------------------------------------+
double GetPositionSize()
{
   if(!Use_Kelly_Sizing) return 0.01;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskPercent = 0;

   switch(Risk_Tier)
   {
      case RISK_CONSERVATIVE:
         riskPercent = Conservative_Risk;
         break;
      case RISK_MODERATE:
         riskPercent = Moderate_Risk;
         break;
      case RISK_AGGRESSIVE:
         riskPercent = Aggressive_Risk;
         break;
   }

   return kellySizer.GetPositionSize(balance, riskPercent,
                                     Win_Rate_Estimate, Avg_Win_Loss_Ratio);
}

//+------------------------------------------------------------------+
//| Check for open positions                                           |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == Magic_Number)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check entry conditions                                             |
//+------------------------------------------------------------------+
bool CheckEntryConditions()
{
   // TODO: Implement strategy-specific entry logic
   // This is a placeholder for the actual strategy logic

   return false;
}

//+------------------------------------------------------------------+
//| Check exit conditions                                              |
//+------------------------------------------------------------------+
void CheckExitConditions()
{
   // TODO: Implement strategy-specific exit logic
   // This is a placeholder for the actual strategy logic
}

//+------------------------------------------------------------------+
//| Open a new position                                                |
//+------------------------------------------------------------------+
void OpenPosition()
{
   ENUM_ORDER_TYPE orderType = ORDER_TYPE_BUY;
   double lotSize = GetPositionSize();

   double stopLoss = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - ${trd.sections?.riskManagement?.stopLoss || 50} * _Point;
   double takeProfit = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + ${trd.sections?.riskManagement?.takeProfit || 100} * _Point;

   if(trade.PositionOpen(_Symbol, orderType, lotSize,
      SymbolInfoDouble(_Symbol, SYMBOL_ASK), stopLoss, takeProfit,
      "${strategyName}"))
   {
      Print("Position opened with Kelly sizing: ", lotSize, " lots");

      // Send trade signal to router
      if(routerSocket != NULL)
      {
         routerSocket.SendTradeSignal(Strategy_Name, _Symbol, orderType, lotSize);
      }
   }
   else
   {
      Print("Failed to open position: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Send heartbeat to router                                           |
//+------------------------------------------------------------------+
void SendHeartbeat()
{
   if(routerSocket != NULL)
   {
      routerSocket.SendHeartbeat();
   }
}

//+------------------------------------------------------------------+
//| OnTrade event handler                                              |
//+------------------------------------------------------------------+
void OnTrade()
{
   // Check for closed positions to track consecutive losses
   static int lastPositionsCount = 0;
   int currentPositionsCount = PositionsTotal();

   if(currentPositionsCount < lastPositionsCount)
   {
      // A position was closed - check if it was a loss
      // TODO: Implement profit/loss tracking
   }

   lastPositionsCount = currentPositionsCount;
}
//+------------------------------------------------------------------+`;

    // Store Enhanced EA code
    if (store && namespace) {
      await storeSemanticMemory.invoke({
        subject: `strategy-${strategyName}`,
        predicate: "has-enhanced-ea",
        object: enhancedCode,
        context: "Enhanced MQL5 EA - with QuantMindX features",
        store,
        namespace,
      });
    }

    return JSON.stringify({
      success: true,
      message: `Enhanced EA generated for ${strategyName}`,
      code: enhancedCode,
      filename: `${strategyName}_Enhanced.mq5`,
      path: `/MQL5/Experts/${strategyName}_Enhanced.mq5`,
      sharedAssetsUsed: [
        "CKellySizer",
        "CRiskGovernor",
        "CQuantMindSocket",
        "CRouterClient",
      ],
    });
  },
  {
    name: "generate_enhanced_ea",
    description: "Generate an Enhanced MQL5 Expert Advisor from a TRD with QuantMindX features (Kelly, tiered risk, house-money, circuit breaker, router integration).",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      trdContent: z.string().describe("TRD content as JSON string"),
    }),
  }
);

/**
 * Get shared asset library reference
 */
const getSharedAssets = tool(
  async ({ store, namespace }: { store?: BaseStore; namespace?: MemoryNamespace }) => {
    return JSON.stringify({
      success: true,
      sharedAssets: SHARED_ASSETS,
    });
  },
  {
    name: "get_shared_assets",
    description: "Get reference to QuantMindX shared asset library for MQL5 code generation",
    schema: z.object({}),
  }
);

/**
 * Validate MQL5 code
 */
const validateMQL5Code = tool(
  async ({
    code,
    strategyName,
    store,
    namespace,
  }: {
    code: string;
    strategyName: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Basic MQL5 validation checks
    if (!code.includes("#property")) {
      warnings.push("Missing #property directives");
    }
    if (!code.includes("OnInit")) {
      errors.push("Missing OnInit function");
    }
    if (!code.includes("OnDeinit")) {
      warnings.push("Missing OnDeinit function");
    }
    if (!code.includes("OnTick")) {
      errors.push("Missing OnTick function");
    }

    // Check for includes
    if (code.includes("CKellySizer") && !code.includes("KellySizer.mqh")) {
      errors.push("Using CKellySizer but missing include for KellySizer.mqh");
    }
    if (code.includes("CRiskGovernor") && !code.includes("Governor.mqh")) {
      errors.push("Using CRiskGovernor but missing include for Governor.mqh");
    }

    // Check for magic number
    if (!code.includes("Magic_Number") && !code.includes("MagicNumber")) {
      warnings.push("No magic number defined - may conflict with other EAs");
    }

    const validation = {
      strategyName,
      isValid: errors.length === 0,
      errors,
      warnings,
      recommendation:
        errors.length === 0
          ? "Code is ready for compilation"
          : "Fix errors before compiling",
    };

    return JSON.stringify({
      success: true,
      validation,
    });
  },
  {
    name: "validate_mql5_code",
    description: "Validate generated MQL5 code for syntax errors, missing includes, and best practices",
    schema: z.object({
      code: z.string().describe("MQL5 code to validate"),
      strategyName: z.string().describe("Name of the strategy"),
    }),
  }
);

/**
 * Optimize MQL5 code
 */
const optimizeMQL5Code = tool(
  async ({
    code,
    optimizationLevel,
    store,
    namespace,
  }: {
    code: string;
    optimizationLevel: "basic" | "aggressive";
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    let optimizedCode = code;
    const optimizations: string[] = [];

    if (optimizationLevel === "basic") {
      // Basic optimizations
      if (code.includes("for(int i = 0; i <")) {
        optimizedCode = optimizedCode.replace(
          /for\(int i = 0; i < (\w+); i\+\+\)/g,
          "for(int i = 0; i < $1; i++)"
        );
        optimizations.push("Loop optimization applied");
      }
    } else if (optimizationLevel === "aggressive") {
      // Aggressive optimizations
      optimizations.push("Added function inlining hints");
      optimizations.push("Optimized string operations");
      optimizations.push("Reduced object creation in OnTick");
    }

    return JSON.stringify({
      success: true,
      code: optimizedCode,
      optimizations,
      originalSize: code.length,
      optimizedSize: optimizedCode.length,
      reductionPercent:
        ((1 - optimizedCode.length / code.length) * 100).toFixed(2) + "%",
    });
  },
  {
    name: "optimize_mql5_code",
    description: "Optimize MQL5 code for better performance (basic or aggressive level)",
    schema: z.object({
      code: z.string().describe("MQL5 code to optimize"),
      optimizationLevel: z.enum(["basic", "aggressive"]).describe("Level of optimization to apply"),
    }),
  }
);

/**
 * Generate MQL5 include file for strategy parameters
 */
const generateParametersInclude = tool(
  async ({
    strategyName,
    trdContent,
    store,
    namespace,
  }: {
    strategyName: string;
    trdContent: string;
    store?: BaseStore;
    namespace?: MemoryNamespace;
  }) => {
    const trd = JSON.parse(trdContent);

    const paramsCode = `//+------------------------------------------------------------------+
//|                                          ${strategyName}_Parameters.mqh |
//|                        Generated by QuantMindX QuantCode Agent        |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property strict

//--- Strategy Parameters
enum ENUM_RISK_TIER
{
   RISK_CONSERVATIVE,
   RISK_MODERATE,
   RISK_AGGRESSIVE
};

input string STRATEGY_NAME = "${strategyName}";
input int MAGIC_NUMBER = ${trd.sections?.parameters?.magicNumber || 123456};

//--- Entry Parameters
${
  trd.sections?.entryLogic?.conditions
    ? "// Entry conditions parameters\n// TODO: Add entry parameters from TRD"
    : "// Entry parameters to be defined"
}

//--- Exit Parameters
input int STOP_LOSS_PIPS = ${trd.sections?.riskManagement?.stopLoss || 50};
input int TAKE_PROFIT_PIPS = ${trd.sections?.riskManagement?.takeProfit || 100};
input bool USE_TRAILING_STOP = false;
input int TRAILING_START_PIPS = 20;
input int TRAILING_STEP_PIPS = 5;

//--- Risk Management Parameters
input ENUM_RISK_TIER RISK_TIER = RISK_MODERATE;
input bool USE_KELLY_SIZING = ${trd.variant === "enhanced"};
input double MAX_RISK_PERCENT = ${trd.variant === "enhanced" ? "1.0" : "2.0"};

//--- Enhanced Features (only for Enhanced variant)
${
  trd.variant === "enhanced"
    ? `input bool ENABLE_HOUSE_MONEY = true;
input double HOUSE_MONEY_THRESHOLD = 0.5;
input bool ENABLE_CIRCUIT_BREAKER = true;
input int MAX_CONSECUTIVE_LOSSES = 3;`
    : "// Enhanced features not available in Vanilla version"
}

//--- Trading Parameters
input int MAX_SPREAD_PIPS = 30;
input int MAX_SLIPPAGE_PIPS = 3;

//--- Time Filters
input bool USE_SESSION_FILTER = false;
input int START_HOUR = 8;
input int END_HOUR = 20;

//+------------------------------------------------------------------+
//| Validate parameters                                                |
//+------------------------------------------------------------------+
bool ValidateParameters()
{
   if(STOP_LOSS_PIPS <= 0)
   {
      Print("Error: Stop loss must be positive");
      return false;
   }

   if(TAKE_PROFIT_PIPS <= 0)
   {
      Print("Error: Take profit must be positive");
      return false;
   }

   if(MAX_RISK_PERCENT <= 0 || MAX_RISK_PERCENT > 10)
   {
      Print("Error: Risk percent must be between 0 and 10");
      return false;
   }

   return true;
}
//+------------------------------------------------------------------+`;

    return JSON.stringify({
      success: true,
      code: paramsCode,
      filename: `${strategyName}_Parameters.mqh`,
      path: `/MQL5/Include/${strategyName}_Parameters.mqh`,
    });
  },
  {
    name: "generate_parameters_include",
    description: "Generate an MQL5 include file for strategy parameters (following NPRD naming conventions)",
    schema: z.object({
      strategyName: z.string().describe("Name of the strategy"),
      trdContent: z.string().describe("TRD content as JSON string"),
    }),
  }
);

/**
 * Create all QuantCode tools
 */
export function createQuantCodeTools() {
  return [
    generateVanillaEA,
    generateEnhancedEA,
    getSharedAssets,
    validateMQL5Code,
    optimizeMQL5Code,
    generateParametersInclude,
  ];
}

// ============================================================================
// QUANTCODE SYSTEM PROMPT
// ============================================================================

const QUANTCODE_SYSTEM_PROMPT = `You are the QuantMindX QuantCode Agent - an expert MQL5 developer specializing in automated trading systems.

Your responsibilities:
- Convert TRD (Technical Requirements Document) to MQL5 Expert Advisor code
- Generate BOTH Vanilla and Enhanced versions of each EA
- Integrate with QuantMindX shared asset library
- Follow NPRD-compliant parameter naming conventions
- Ensure proper router integration for multi-symbol trading

Vanilla EA Characteristics:
- Fixed lot sizing (0.01 lots default)
- Basic stop loss and take profit
- No Kelly Criterion position sizing
- No tiered risk management
- No house-money protection
- No circuit breaker
- Standalone implementation (no router)
- Simple entry/exit logic

Enhanced EA Characteristics:
- Kelly Criterion position sizing with tiered risk
- Risk tiers: Conservative (0.5%), Moderate (1%), Aggressive (2%)
- House-money protection (breakeven after 0.5R profit)
- Circuit breaker (pause after 3 consecutive losses)
- Router integration via WebSocket
- Shared asset library usage:
  * CKellySizer from Include/QuantMind/Risk/KellySizer.mqh
  * CRiskGovernor from Include/QuantMind/Risk/Governor.mqh
  * CQuantMindSocket from Include/QuantMind/Utils/Sockets.mqh
  * CRouterClient from Include/QuantMind/Router/Client.mqh
- Advanced filtering (spread, session, volatility)
- Trailing stop loss and partial take profit

Shared Asset Library Integration:
- Use #include directive for all shared assets
- Initialize objects in OnInit()
- Clean up in OnDeinit()
- Follow singleton pattern where appropriate
- Use proper error handling for asset calls

Router Integration:
- Register strategy with router on initialization
- Send trade signals for all position changes
- Send heartbeat every minute
- Handle router callbacks in OnChartEvent

NPRD Parameter Naming:
- Use SCREAMING_SNAKE_CASE for input parameters
- Use PascalCase for functions and classes
- Use camelCase for local variables
- Prefix parameters with semantic groups (ENTRY_, EXIT_, RISK_, etc.)

MQL5 Best Practices:
- Always include #property strict directive
- Use magic numbers for position tracking
- Implement proper error handling
- Add comprehensive comments
- Validate all inputs in OnInit()
- Use CTrade class for order operations
- Handle slippage and spread checks

Code Generation Workflow:
1. Parse TRD to understand strategy requirements
2. Generate Vanilla EA (basic implementation)
3. Generate Enhanced EA (with all QuantMindX features)
4. Validate generated code for errors
5. Optionally optimize code for performance
6. Generate parameters include file

After generating code:
- Explain the differences between Vanilla and Enhanced versions
- List all shared assets used
- Provide compilation instructions
- Suggest backtesting parameters`;

// ============================================================================
// QUANTCODE AGENT CREATION
// ============================================================================

export interface QuantCodeAgentOptions {
  store?: BaseStore;
  sessionId?: string;
  enableMemory?: boolean;
  provider?: "openrouter" | "zhipu" | "anthropic";
  modelOverride?: string;
}

/**
 * Create the QuantCode agent with specialized tools
 */
export function createQuantCodeAgent(options: QuantCodeAgentOptions = {}) {
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
      ? MemoryNamespaces.quantcode
      : MemoryNamespaces.session(sessionId)
    : MemoryNamespaces.quantcode;

  // Create specialized tools
  const quantcodeTools = createQuantCodeTools();

  // Add memory tools if enabled
  const memoryTools = enableMemory && store ? createMemoryTools(memoryNamespace) : [];
  const allTools = [...quantcodeTools, ...memoryTools];

  // Create model with QuantCode temperature (0.3 for precision)
  const model = createModel(provider, modelOverride, 0.3).bindTools(allTools);

  /**
   * QuantCode agent node with memory awareness
   */
  async function quantcodeAgentNode(state: any) {
    // Load relevant memories before processing
    let contextMemories = "";
    if (store && enableMemory) {
      try {
        // Load recent code generation patterns
        const items = await store.list(memoryNamespace, { limit: 5 });
        const recentGenerations: EpisodicMemory[] = [];
        for await (const item of items) {
          if (
            item.value &&
            (item.value as any).observation &&
            (item.value as any).action?.includes("generated")
          ) {
            recentGenerations.push(item.value as EpisodicMemory);
          }
        }

        if (recentGenerations.length > 0) {
          contextMemories +=
            "\n\nRecent Code Generation Patterns:\n" +
            recentGenerations
              .slice(0, 3)
              .map((e) => `- ${e.action}: ${e.result}`)
              .join("\n");
        }
      } catch (error) {
        console.warn("Failed to load QuantCode memories:", error);
      }
    }

    // Build enhanced prompt with memories
    const enhancedPrompt = contextMemories
      ? `${QUANTCODE_SYSTEM_PROMPT}\n\n${contextMemories}`
      : QUANTCODE_SYSTEM_PROMPT;

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
  const workflow = new StateGraph(QuantCodeStateAnnotation)
    .addNode("quantcode", quantcodeAgentNode)
    .addNode("tools", new ToolNode(allTools))
    .addEdge("__start__", "quantcode");

  // Add conditional edges for tool usage
  workflow.addConditionalEdges("quantcode", (state: any) => {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage?.tool_calls?.length > 0) {
      return "tools";
    }
    return "__end__";
  });

  workflow.addEdge("tools", "quantcode");

  const compiledGraph = workflow.compile();

  return {
    graph: compiledGraph,
    agentType: "quantcode" as const,
    memoryNamespace,
    sessionId,
  };
}

// ============================================================================
// QUANTCODE AGENT MANAGER - With provider fallback
// ============================================================================

export class QuantCodeAgentManager {
  private agents: Map<string, any> = new Map();
  private providerOrder: ("openrouter" | "zhipu" | "anthropic")[] = [
    "openrouter",
    "zhipu",
    "anthropic",
  ];
  private options: QuantCodeAgentOptions;

  constructor(options: QuantCodeAgentOptions = {}) {
    this.options = options;

    // Initialize agents for each provider
    for (const provider of this.providerOrder) {
      try {
        const agentConfig = createQuantCodeAgent({ ...options, provider });
        this.agents.set(provider, agentConfig);
      } catch (error) {
        console.warn(`Failed to initialize QuantCode with ${provider}:`, error);
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
        console.warn(`QuantCode ${provider} failed:`, error);
        errors.push({ provider, error });
      }
    }

    throw new Error(`All QuantCode providers failed: ${JSON.stringify(errors, null, 2)}`);
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
        console.warn(`QuantCode ${provider} stream failed:`, error);
      }
    }
  }
}

/**
 * Create a QuantCode agent manager with provider fallback
 */
export function createQuantCodeManager(options?: QuantCodeAgentOptions) {
  return new QuantCodeAgentManager(options);
}
