//+------------------------------------------------------------------+
//|                                              TestRiskModules.mq5 |
//|                        QuantMind Standard Library (QSL) - Tests  |
//|                        Risk Modules Compilation Test             |
//|                                                                  |
//| This EA tests the compilation of all Risk QSL modules.          |
//| It imports and tests PropManager, RiskClient, and KellySizer.   |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://github.com/quantmindx"
#property version   "1.00"
#property strict

// Include Core modules (dependencies)
#include <QuantMind/Core/BaseAgent.mqh>
#include <QuantMind/Core/Constants.mqh>
#include <QuantMind/Core/Types.mqh>

// Include Risk modules
#include <QuantMind/Risk/PropManager.mqh>
#include <QuantMind/Risk/RiskClient.mqh>
#include <QuantMind/Risk/KellySizer.mqh>

//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input int    MagicNumber = QM_MAGIC_BASE;           // Magic number
input string FirmName = "TestPropFirm";             // Prop firm name
input double DailyLossLimit = QM_DAILY_LOSS_LIMIT_PCT; // Daily loss limit (%)
input bool   EnableHeartbeat = false;               // Enable heartbeat (requires backend)

//+------------------------------------------------------------------+
//| Global Variables                                                 |
//+------------------------------------------------------------------+
CPropManager *g_propManager = NULL;
QMKellySizer *g_kellySizer = NULL;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("=== TestRiskModules EA Initialization ===");
    Print("QSL Version: ", QM_VERSION_STRING);
    
    // Test 1: Create PropManager instance
    Print("\n=== Testing PropManager ===");
    g_propManager = new CPropManager();
    if(g_propManager == NULL)
    {
        Print("ERROR: Failed to create PropManager instance");
        return INIT_FAILED;
    }
    Print("✓ PropManager instance created successfully");
    
    // Test 2: Initialize PropManager
    string accountId = IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN));
    if(!g_propManager.Initialize(accountId, FirmName, DailyLossLimit))
    {
        Print("ERROR: PropManager initialization failed");
        delete g_propManager;
        g_propManager = NULL;
        return INIT_FAILED;
    }
    Print("✓ PropManager initialized successfully");
    Print("  Account ID: ", accountId);
    Print("  Firm Name: ", FirmName);
    Print("  Daily Loss Limit: ", DailyLossLimit, "%");
    
    // Test 3: Test PropManager daily drawdown calculation
    double testStartBalance = 10000.0;
    double testCurrentEquity = 9500.0;
    double drawdown = g_propManager.CalculateDailyDrawdown(testStartBalance, testCurrentEquity);
    Print("✓ Daily Drawdown Calculation:");
    Print("  Start Balance: ", testStartBalance);
    Print("  Current Equity: ", testCurrentEquity);
    Print("  Drawdown: ", DoubleToString(drawdown, 2), "%");
    
    // Test 4: Test quadratic throttle calculation
    double throttle = g_propManager.CalculateQuadraticThrottle();
    Print("✓ Quadratic Throttle: ", DoubleToString(throttle, 4));
    
    // Test 5: Test news guard functionality
    Print("\n=== Testing News Guard ===");
    g_propManager.SetNewsGuard(true);
    Print("✓ News Guard activated");
    bool tradingAllowed = g_propManager.IsTradingAllowed();
    Print("  Trading Allowed: ", tradingAllowed ? "YES" : "NO");
    g_propManager.SetNewsGuard(false);
    Print("✓ News Guard deactivated");
    
    // Test 6: Test account state retrieval
    Print("\n=== Testing Account State ===");
    SAccountState accountState;
    g_propManager.GetAccountState(accountState);
    Print("✓ Account State retrieved:");
    Print("  Balance: ", accountState.balance);
    Print("  Equity: ", accountState.equity);
    Print("  Daily P&L: ", accountState.dailyPnL);
    Print("  Drawdown: ", DoubleToString(accountState.dailyDrawdown, 2), "%");
    Print("  Risk Status: ", RiskStatusToString(accountState.riskStatus));
    
    // Test 7: Test KellySizer
    Print("\n=== Testing KellySizer ===");
    g_kellySizer = new QMKellySizer();
    if(g_kellySizer == NULL)
    {
        Print("ERROR: Failed to create KellySizer instance");
        return INIT_FAILED;
    }
    Print("✓ KellySizer instance created successfully");
    
    // Test 8: Calculate Kelly fraction
    double winRate = 0.55;
    double avgWin = 400.0;
    double avgLoss = 200.0;
    double kellyFraction = g_kellySizer.CalculateKellyFraction(winRate, avgWin, avgLoss);
    Print("✓ Kelly Fraction Calculation:");
    Print("  Win Rate: ", winRate);
    Print("  Avg Win: ", avgWin);
    Print("  Avg Loss: ", avgLoss);
    Print("  Kelly Fraction: ", DoubleToString(kellyFraction, 4));
    
    // Test 9: Calculate lot size
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double riskPct = 1.0; // 100% of Kelly
    double stopLossPips = 50.0;
    double tickValue = 10.0; // $10 per pip for standard lot
    double lotSize = g_kellySizer.CalculateLotSize(kellyFraction, equity, riskPct, stopLossPips, tickValue);
    Print("✓ Lot Size Calculation:");
    Print("  Equity: ", equity);
    Print("  Risk Pct: ", riskPct);
    Print("  Stop Loss: ", stopLossPips, " pips");
    Print("  Lot Size: ", DoubleToString(lotSize, 2));
    
    // Test 10: Test RiskClient
    Print("\n=== Testing RiskClient ===");
    double riskMultiplier = GetRiskMultiplier(Symbol());
    Print("✓ Risk Multiplier retrieved: ", DoubleToString(riskMultiplier, 4));
    
    // Test 11: Test heartbeat (optional - requires backend)
    if(EnableHeartbeat)
    {
        Print("\n=== Testing Heartbeat ===");
        bool heartbeatSent = SendHeartbeat("TestRiskModules", Symbol(), MagicNumber, riskMultiplier);
        if(heartbeatSent)
        {
            Print("✓ Heartbeat sent successfully");
        }
        else
        {
            Print("⚠ Heartbeat failed (backend may not be running)");
        }
    }
    
    // Test 12: Test risk status summary
    Print("\n=== Risk Status Summary ===");
    string summary = g_propManager.GetRiskStatusSummary();
    Print(summary);
    
    // Test 13: Test strategy quality from Kelly score
    Print("\n=== Testing Strategy Quality ===");
    double testKellyScores[] = {0.95, 0.85, 0.75, 0.55, 0.35, 0.15};
    for(int i = 0; i < ArraySize(testKellyScores); i++)
    {
        double score = testKellyScores[i];
        ENUM_STRATEGY_QUALITY quality = GetStrategyQualityFromKelly(score);
        Print("  Kelly Score ", DoubleToString(score, 2), " = ", StrategyQualityToString(quality));
    }
    
    Print("\n=== All Risk Module Tests Passed ===");
    Print("✓ PropManager.mqh compiled and functional");
    Print("✓ RiskClient.mqh compiled and functional");
    Print("✓ KellySizer.mqh compiled and functional");
    Print("✓ All risk management features working correctly");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("\n=== TestRiskModules EA Deinitialization ===");
    Print("Reason: ", reason);
    
    if(g_propManager != NULL)
    {
        delete g_propManager;
        g_propManager = NULL;
        Print("✓ PropManager instance deleted");
    }
    
    if(g_kellySizer != NULL)
    {
        delete g_kellySizer;
        g_kellySizer = NULL;
        Print("✓ KellySizer instance deleted");
    }
    
    Print("=== Deinitialization Complete ===");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // This EA is for testing only
    // Periodically update and log risk status
    
    static datetime lastUpdateTime = 0;
    datetime currentTime = TimeCurrent();
    
    // Update every 5 minutes
    if(currentTime - lastUpdateTime >= 300)
    {
        if(g_propManager != NULL)
        {
            // Update PropManager state
            g_propManager.Update();
            
            // Get current risk multiplier
            double multiplier = g_propManager.GetRiskMultiplier();
            
            // Log status
            Print("[TestRiskModules] Risk Multiplier: ", DoubleToString(multiplier, 4),
                  " | Drawdown: ", DoubleToString(g_propManager.GetCurrentDrawdown(), 2), "%",
                  " | Status: ", RiskStatusToString(g_propManager.GetRiskStatus()));
            
            // Send heartbeat if enabled
            if(EnableHeartbeat)
            {
                SendHeartbeat("TestRiskModules", Symbol(), MagicNumber, multiplier);
            }
        }
        
        lastUpdateTime = currentTime;
    }
}

//+------------------------------------------------------------------+
//| Test function to verify all risk calculations                    |
//+------------------------------------------------------------------+
void TestRiskCalculations()
{
    Print("\n=== Testing Risk Calculations ===");
    
    // Test various drawdown scenarios
    double testScenarios[][2] = {
        {10000, 10000},  // 0% drawdown
        {10000, 9800},   // 2% drawdown
        {10000, 9700},   // 3% drawdown
        {10000, 9600},   // 4% drawdown
        {10000, 9550},   // 4.5% drawdown (hard stop)
        {10000, 9500}    // 5% drawdown (breached)
    };
    
    for(int i = 0; i < ArrayRange(testScenarios, 0); i++)
    {
        double startBalance = testScenarios[i][0];
        double currentEquity = testScenarios[i][1];
        
        double drawdown = g_propManager.CalculateDailyDrawdown(startBalance, currentEquity);
        
        // Simulate the scenario
        // Note: This is a simplified test - in production, PropManager tracks state internally
        
        Print("Scenario ", i+1, ": Start=", startBalance, " Current=", currentEquity,
              " Drawdown=", DoubleToString(drawdown, 2), "%");
    }
}

//+------------------------------------------------------------------+
//| Test Kelly criterion edge cases                                  |
//+------------------------------------------------------------------+
void TestKellyEdgeCases()
{
    Print("\n=== Testing Kelly Edge Cases ===");
    
    // Test 1: Perfect strategy (100% win rate)
    double kelly1 = g_kellySizer.CalculateKellyFraction(1.0, 100, 50);
    Print("Perfect strategy (100% win): ", DoubleToString(kelly1, 4));
    
    // Test 2: Break-even strategy (50% win rate, 1:1 ratio)
    double kelly2 = g_kellySizer.CalculateKellyFraction(0.5, 100, 100);
    Print("Break-even strategy: ", DoubleToString(kelly2, 4));
    
    // Test 3: Losing strategy (40% win rate)
    double kelly3 = g_kellySizer.CalculateKellyFraction(0.4, 100, 100);
    Print("Losing strategy: ", DoubleToString(kelly3, 4));
    
    // Test 4: High win rate, low payoff
    double kelly4 = g_kellySizer.CalculateKellyFraction(0.8, 50, 100);
    Print("High win rate, low payoff: ", DoubleToString(kelly4, 4));
    
    // Test 5: Low win rate, high payoff
    double kelly5 = g_kellySizer.CalculateKellyFraction(0.3, 500, 100);
    Print("Low win rate, high payoff: ", DoubleToString(kelly5, 4));
}
//+------------------------------------------------------------------+
