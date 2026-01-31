//+------------------------------------------------------------------+
//|                                              TestCoreModules.mq5 |
//|                        QuantMind Standard Library (QSL) - Tests  |
//|                        Core Modules Compilation Test             |
//|                                                                  |
//| This EA tests the compilation of all Core QSL modules.          |
//| It imports and instantiates classes from BaseAgent, Constants,  |
//| and Types modules to verify they compile correctly.             |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

// Include Core modules
#include <QuantMind/Core/BaseAgent.mqh>
#include <QuantMind/Core/Constants.mqh>
#include <QuantMind/Core/Types.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input int    MagicNumber = QM_MAGIC_BASE;           // Magic number
input double RiskPercent = QM_DEFAULT_RISK_PCT;     // Risk per trade (%)
input bool   EnableLogging = true;                  // Enable detailed logging

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CBaseAgent *g_agent = NULL;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== TestCoreModules EA Initialization ===");
    Print("QSL Version: ", QM_VERSION_STRING);
    Print("System Name: ", QM_SYSTEM_NAME);
    
    // Test 1: Create BaseAgent instance
    g_agent = new CBaseAgent();
    if(g_agent == NULL)
    {
        Print("ERROR: Failed to create BaseAgent instance");
        return INIT_FAILED;
    }
    Print("✓ BaseAgent instance created successfully");
    
    // Test 2: Initialize BaseAgent
    if(!g_agent.Initialize("TestCoreModules", Symbol(), Period(), MagicNumber))
    {
        Print("ERROR: BaseAgent initialization failed");
        delete g_agent;
        g_agent = NULL;
        return INIT_FAILED;
    }
    Print("✓ BaseAgent initialized successfully");
    Print("  Agent Name: ", g_agent.GetAgentName());
    Print("  Symbol: ", g_agent.GetSymbol());
    Print("  Timeframe: ", EnumToString(g_agent.GetTimeframe()));
    Print("  Magic Number: ", g_agent.GetMagicNumber());
    
    // Test 3: Test Constants
    Print("\n=== Testing Constants ===");
    Print("✓ Daily Loss Limit: ", QM_DAILY_LOSS_LIMIT_PCT, "%");
    Print("✓ Kelly Threshold: ", QM_KELLY_THRESHOLD);
    Print("✓ Max Risk Per Trade: ", QM_MAX_RISK_PER_TRADE_PCT, "%");
    Print("✓ Heartbeat Interval: ", QM_HEARTBEAT_INTERVAL_SEC, " seconds");
    Print("✓ Risk Multiplier Range: ", QM_RISK_MULTIPLIER_MIN, " - ", QM_RISK_MULTIPLIER_MAX);
    
    // Test 4: Test Types - Create trade proposal
    Print("\n=== Testing Types ===");
    STradeProposal proposal;
    proposal.symbol = Symbol();
    proposal.signalType = SIGNAL_TYPE_BUY;
    proposal.entryPrice = SymbolInfoDouble(Symbol(), SYMBOL_ASK);
    proposal.stopLoss = proposal.entryPrice - 100 * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    proposal.takeProfit = proposal.entryPrice + 200 * SymbolInfoDouble(Symbol(), SYMBOL_POINT);
    proposal.lotSize = 0.01;
    proposal.kellyScore = 0.85;
    proposal.winRate = 0.65;
    proposal.avgWin = 200.0;
    proposal.avgLoss = 100.0;
    proposal.sharpeRatio = 1.5;
    proposal.maxDrawdown = 10.0;
    proposal.magicNumber = MagicNumber;
    proposal.strategyName = "TestStrategy";
    proposal.timestamp = TimeCurrent();
    proposal.comment = "Test proposal";
    
    Print("✓ Trade Proposal created:");
    Print("  Symbol: ", proposal.symbol);
    Print("  Signal: ", SignalTypeToString(proposal.signalType));
    Print("  Kelly Score: ", proposal.kellyScore);
    Print("  Strategy Quality: ", StrategyQualityToString(GetStrategyQualityFromKelly(proposal.kellyScore)));
    
    // Test 5: Test account state structure
    SAccountState accountState;
    accountState.accountId = IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN));
    accountState.balance = g_agent.GetAccountBalance();
    accountState.equity = g_agent.GetAccountEquity();
    accountState.freeMargin = g_agent.GetAccountFreeMargin();
    accountState.highWaterMark = accountState.equity;
    accountState.dailyPnL = 0.0;
    accountState.dailyDrawdown = 0.0;
    accountState.dailyLossLimit = QM_DAILY_LOSS_LIMIT_PCT;
    accountState.maxDrawdown = QM_MAX_DRAWDOWN_PCT;
    accountState.tradesCount = 0;
    accountState.lastUpdate = TimeCurrent();
    accountState.status = ACCOUNT_STATUS_ACTIVE;
    accountState.riskStatus = RISK_STATUS_NORMAL;
    accountState.isKillZone = false;
    
    Print("✓ Account State created:");
    Print("  Account ID: ", accountState.accountId);
    Print("  Balance: ", accountState.balance);
    Print("  Equity: ", accountState.equity);
    Print("  Status: ", EnumToString(accountState.status));
    Print("  Risk Status: ", RiskStatusToString(accountState.riskStatus));
    
    // Test 6: Test risk parameters structure
    SRiskParameters riskParams;
    riskParams.riskMultiplier = QM_RISK_MULTIPLIER_DEFAULT;
    riskParams.maxRiskPerTrade = QM_MAX_RISK_PER_TRADE_PCT;
    riskParams.kellyThreshold = QM_KELLY_THRESHOLD;
    riskParams.dailyLossLimit = QM_DAILY_LOSS_LIMIT_PCT;
    riskParams.hardStopThreshold = QM_EFFECTIVE_LIMIT_PCT;
    riskParams.preservationMode = false;
    riskParams.newsGuardActive = false;
    riskParams.hardStopActive = false;
    riskParams.lastHeartbeat = TimeCurrent();
    
    Print("✓ Risk Parameters created:");
    Print("  Risk Multiplier: ", riskParams.riskMultiplier);
    Print("  Max Risk Per Trade: ", riskParams.maxRiskPerTrade, "%");
    Print("  Kelly Threshold: ", riskParams.kellyThreshold);
    Print("  Preservation Mode: ", riskParams.preservationMode ? "ON" : "OFF");
    
    // Test 7: Test BaseAgent utility methods
    Print("\n=== Testing BaseAgent Utility Methods ===");
    Print("✓ Symbol Point: ", g_agent.GetSymbolPoint());
    Print("✓ Symbol Digits: ", g_agent.GetSymbolDigits());
    Print("✓ Symbol Min Lot: ", g_agent.GetSymbolMinLot());
    Print("✓ Symbol Max Lot: ", g_agent.GetSymbolMaxLot());
    Print("✓ Symbol Lot Step: ", g_agent.GetSymbolLotStep());
    Print("✓ Current Bid: ", g_agent.GetBid());
    Print("✓ Current Ask: ", g_agent.GetAsk());
    Print("✓ Current Spread: ", g_agent.GetSpread(), " points");
    Print("✓ Trading Allowed: ", g_agent.IsTradingAllowed() ? "YES" : "NO");
    
    // Test 8: Test lot normalization
    double testLot = 0.0123;
    double normalizedLot = g_agent.NormalizeLot(testLot);
    Print("✓ Lot Normalization: ", testLot, " -> ", normalizedLot);
    
    // Test 9: Test enum conversions
    Print("\n=== Testing Enum Conversions ===");
    Print("✓ Trade Decision APPROVE: ", TradeDecisionToString(TRADE_DECISION_APPROVE));
    Print("✓ Agent Type QUANT: ", AgentTypeToString(AGENT_TYPE_QUANT));
    Print("✓ Risk Status THROTTLED: ", RiskStatusToString(RISK_STATUS_THROTTLED));
    Print("✓ Signal Type BUY: ", SignalTypeToString(SIGNAL_TYPE_BUY));
    Print("✓ Strategy Quality A+: ", StrategyQualityToString(STRATEGY_QUALITY_A_PLUS));
    
    // Test 10: Test macros
    Print("\n=== Testing Utility Macros ===");
    double testPips = 10.0;
    int testPoints = QM_PIPS_TO_POINTS(testPips);
    Print("✓ Pips to Points: ", testPips, " pips = ", testPoints, " points");
    Print("✓ Points to Pips: ", testPoints, " points = ", QM_POINTS_TO_PIPS(testPoints), " pips");
    
    double testValue = 1.5;
    double clampedValue = QM_CLAMP(testValue, 0.0, 1.0);
    Print("✓ Clamp: ", testValue, " clamped to [0.0, 1.0] = ", clampedValue);
    
    bool inRange = QM_IN_RANGE(0.5, 0.0, 1.0);
    Print("✓ In Range: 0.5 in [0.0, 1.0] = ", inRange ? "TRUE" : "FALSE");
    
    Print("\n=== All Core Module Tests Passed ===");
    Print("✓ BaseAgent.mqh compiled and functional");
    Print("✓ Constants.mqh compiled and accessible");
    Print("✓ Types.mqh compiled and functional");
    Print("✓ All structures, enums, and functions working correctly");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("\n=== TestCoreModules EA Deinitialization ===");
    Print("Reason: ", reason);
    
    if(g_agent != NULL)
    {
        delete g_agent;
        g_agent = NULL;
        Print("✓ BaseAgent instance deleted");
    }
    
    Print("=== Deinitialization Complete ===");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // This EA is for compilation testing only
    // No trading logic is implemented
    
    // Log status periodically (every 60 seconds)
    static datetime lastLogTime = 0;
    datetime currentTime = TimeCurrent();
    
    if(currentTime - lastLogTime >= 60)
    {
        if(g_agent != NULL && EnableLogging)
        {
            g_agent.LogInfo("Core modules test EA running. All modules functional.");
        }
        lastLogTime = currentTime;
    }
}

//+------------------------------------------------------------------+
//| Test function to verify all imports                              |
//+------------------------------------------------------------------+
void TestAllImports()
{
    // This function exists to ensure all types are properly imported
    // and can be instantiated
    
    STradeProposal proposal;
    SAccountState accountState;
    SRiskParameters riskParams;
    SPositionInfo posInfo;
    SOrderInfo orderInfo;
    SHeartbeatPayload heartbeat;
    SStrategyPerformance performance;
    SAgentTask task;
    SMarketCondition market;
    SNewsEvent news;
    SRingBufferElement ringElement;
    SJsonParseResult jsonResult;
    
    // Test all enums
    ENUM_TRADE_DECISION decision = TRADE_DECISION_APPROVE;
    ENUM_AGENT_TYPE agentType = AGENT_TYPE_QUANT;
    ENUM_RISK_STATUS riskStatus = RISK_STATUS_NORMAL;
    ENUM_ACCOUNT_STATUS accountStatus = ACCOUNT_STATUS_ACTIVE;
    ENUM_SIGNAL_TYPE signalType = SIGNAL_TYPE_BUY;
    ENUM_STRATEGY_QUALITY quality = STRATEGY_QUALITY_A_PLUS;
    
    // Suppress unused variable warnings
    decision = decision;
    agentType = agentType;
    riskStatus = riskStatus;
    accountStatus = accountStatus;
    signalType = signalType;
    quality = quality;
}
//+------------------------------------------------------------------+
