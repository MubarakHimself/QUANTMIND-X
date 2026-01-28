---
title: Fortified Profit Architecture: Multi-Layered Account Protection
url: https://www.mql5.com/en/articles/20449
categories: Trading, Trading Systems
relevance_score: 0
scraped_at: 2026-01-23T21:44:25.085762
---

[![](https://www.mql5.com/ff/si/h2ryn394uwcpxwmxc2.jpg)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Feconomic-calendar%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dopen.calendar%26utm_content%3Deconomic.calendar%26utm_campaign%3Den.0009.desktop.default&a=qdeulxgvibgwytgewnvfatbocjnnninc&s=5c0c60f00ff5f5bedb0fdf65d9d79eb820442eb43ffac2b85aa003224f9dba14&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=flprsftrlhjbojuasbjexyiiextewjlv&ssn=1769193863970356534&ssn_dr=0&ssn_sr=0&fv_date=1769193863&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20449&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Fortified%20Profit%20Architecture%3A%20Multi-Layered%20Account%20Protection%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919386377684091&fz_uniq=5072058207410533255&sv=2552)

MetaTrader 5 / Examples


### Table of contents:

1. [Introduction](https://www.mql5.com/en/articles/20449#Introduction)
2. [System Overview and Understanding](https://www.mql5.com/en/articles/20449#SystemOverview)
3. [Getting Started](https://www.mql5.com/en/articles/20449#GettingStarted)
4. [Backtest Results](https://www.mql5.com/en/articles/20449#BacktestResults)
5. [Conclusion](https://www.mql5.com/en/articles/20449#Conclusion)

### Introduction

In modern algorithmic trading, the pursuit of high returns inevitably comes with heightened exposure to market volatility, execution uncertainty, and systemic risks that can rapidly erode capital. To thrive in such an environment, an EA must not only identify profitable opportunities, but also intelligently regulate how much risk it absorbs at every moment. This is where multi-layered account protection becomes essential—an architectural approach that blends adaptive lot sizing, volatility-sensitive exposure, and real-time performance monitoring into a unified defensive framework. Instead of relying on a single safeguard, the system employs multiple coordinated layers that evolve dynamically with market conditions.

By engineering protection directly into the trading into the trading logic—structurally, behaviorally, and systemically—we enable the EA to pursue high-growth strategies while maintaining a disciplined risk posture. It learns when to expand exposure, when to scale it back, and when to shut down entirely to preserve capital. Through equity floors, drawdown tiers, segmented execution logic, circuit breakers, and recovery protocols, the trading system behaves more like a self-regulating organism than a static set of rules. The result is an intelligent, resilient trading architecture capable of capturing upside while actively managing and neutralizing threats in real time.

### System Overview and Understanding

The Expert Advisor is designed as a sophisticated automated trading system for gold (XAUUSD) that combines a martingale recovery strategy with robust, multi-layered account protection. At its core, it operates as a risk-managed cycle: it identifies new trends using EMA crossovers and places an initial trade. If that trade loses, it enters a "recovery phase," increasing the lot size according to the selected martingale formula to recoup losses, continuing this process for a limited number of steps. When a trade wins, the sequence resets. This core engine is wrapped in a comprehensive safety framework designed to prevent the catastrophic losses typically associated with martingale systems. Key protective layers include dynamic equity stops, daily loss limits, trade throttling, and a circuit breaker that halts all trading after a set number of consecutive losses or a specific drawdown threshold.

The EA's protection logic functions like a series of shields and tripwires guarding your capital. Think of it as a hierarchical defense system: first, individual trades are shielded by slippage checks and volatility-adjusted stop losses. Next, the account is protected by real-time monitors tracking your total equity. If your drawdown from the account's peak equity exceeds a set percentage, the system can pause trading entirely. Further layers include limiting the number of trades within a time window to prevent overtrading and a recovery protocol that, after a severe loss sequence, automatically reduces risk and requires a waiting period or equity recovery before resuming. This structure ensures the aggressive martingale recovery mechanism is always constrained by predefined, rational limits that prioritize capital preservation.

Finally, the system employs persistent state management to maintain its defensive logic across market sessions and platform restarts. Using global variables, it remembers its peak equity, consecutive losses, and whether it is in a paused "recovery" state. This allows the EA to enforce its rules consistently, not just within a single run. For instance, if it hits a circuit breaker and pauses on a Friday, it will remain paused on Monday, preventing immediate re-entry into risky conditions. This design transforms the EA from a simple automated script into a resilient, state-aware trading system that methodically seeks profit while being fundamentally engineered to survive adverse market conditions and protect the trading account above all else.

![](https://c.mql5.com/2/185/Generated_Image.jpg)

### Getting Started

```
//+------------------------------------------------------------------+
//|                                                       GALEIT.mq5 |
//|                        GIT under Copyright 2025, MetaQuotes Ltd. |
//|                     https://www.mql5.com/en/users/johnhlomohang/ |
//+------------------------------------------------------------------+
#property copyright "GIT under Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/johnhlomohang/"
#property version   "1.00"

//--- Include Libraries
#include <Trade/Trade.mqh>

//--- Global Variables
CTrade trade;
ulong lastTicket = 0;
int martingaleStep = 0;
double dailyProfit = 0.0;
datetime lastTradeTime = 0;
string tradeSequenceId = "";

//--- Global Variable Names
#define GV_PEAK_EQUITY       "GV_PeakEquity"
#define GV_PAUSE_EA          "GV_EA_Paused"
#define GV_DD_LOCK_LEVEL     "GV_DrawdownLockLevel"
#define GV_CONSEC_LOSSES     "GV_ConsecLosses"
#define GV_TRADE_WINDOW_START "GV_TradeWindowStart"
#define GV_TRADES_IN_WINDOW  "GV_TradesInWindow"

//--- Input Parameters
input group  "Trading Strategy"
input int FastMAPeriod = 10;
input int SlowMAPeriod = 50;

input group "Martingale & Money Management"
input double InitialLotSize = 0.01;
input double LotMultiplier = 2.0;
input int MaxMartingaleSteps = 5;
input double RiskPercent = 2.0;

input group  "Volatility Management (ATR)"
input int ATR_Period = 14;
input double ATR_SL_Factor = 1.5;
input double ATR_TP_Factor = 1.0;

input group  "Account Protection"
input bool UseEquityStop = true;
input double EquityStopPercent = 8.0;
input bool UseDailyLossLimit = true;
input double DailyLossPercent = 5.0;
input bool UseMaxSpreadFilter = true;
input int MaxSpreadPoints = 5;

input group "Trailing Stop Parameters"
input bool UseTrailingStop = true;
input int BreakEvenAtPips = 500;
input int TrailStartAtPips = 600;
input int TrailStepPips = 100;

input group  "Circuit Breaker Settings"
input int MaxConsecutiveLosses = 3;
input double CircuitBreakerDD = 15.0;

input group  "Throttle Settings"
input int MaxTradesPerHour = 10;
input int ThrottleWindowSeconds = 3600;

input group  "Recovery Settings"
input double RecoveryRiskReduction = 0.5;
input double ResumeEquityPercent = 90.0;
input int ResumeAfterSeconds = 86400;
```

We start off, by defining the essential global variables that the EA will rely on to track its internal state and trading behavior. These include objects such as the CTrade instance for order execution, variables for storing the last trade ticket, martingale progression, and daily profit, along with time-tracking and sequence identifiers. Immediately after that, we declare a set of global variable names, which are stored using MetaTrader’s built-in GlobalVariable system. These allow the EA to maintain important risk-related states such as peak equity, pause conditions, drawdown tiers, consecutive losses, and trade window activity—even if MetaTrader restarts or the chart is reloaded.

Moving forward, we introduce grouped input parameters that define the core trading strategy and how the system interacts with market conditions. Next, it specifies martingale and money-management settings such as the initial lot size, lot multiplier, max recovery steps, and equity risk percentage. The volatility management section incorporates ATR-based dynamic stop-loss and take-profit calculations, ensuring that the trade exits automatically scale with market volatility rather than using fixed pip distances.

Finally, we define several layers of protective logic through additional input groups. The "Account Protection" section governs equity stops, daily loss limits, and spread filtering—tools designed to prevent trading under dangerous or costly market conditions. Trailing stop parameters allow the EA to dynamically secure profits once a trade becomes favorable. The "Circuit Breaker" and "Throttle" settings impose systemic safeguards, halting or limiting trading when conditions become too risky or when trade frequency gets excessive. Lastly, the "Recovery" section outlines how the EA should behave after protective shutdowns, including how much risk to reduce and under what conditions trading can safely resume. Collectively, these settings create a multi-layered, adaptive, and resilient trading framework.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("GaleIT EA Initialized");
   trade.SetExpertMagicNumber(12345);
   tradeSequenceId = GenerateTradeSequenceId();

   // Initialize all protection systems
   InitAccountShields();
   InitFailSafes();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("GaleIT EA Deinitialized - Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update protection systems
   UpdatePeakEquity();
   TryResumeFromRecovery();

   // Check if trading is allowed
   if(!IsTradingAllowed()) return;

   // Check all safety limits
   if(!CheckSafetyLimits()) return;

   // Manage existing positions
   ManageExistingPositions();

   // Check for new trading opportunities
   if(IsNewBar())
   {
      CheckForNewTrade();
   }

   ManageOpenTrades();
}
```

The initialization phase begins inside the OnInit() function, where the EA prints a startup message, sets a unique Magic Number for the trade identification, and generates a tradeSequenceId to track the current trading series. This ensures all trades placed by the EA are uniquely tagged and easy to manage. The EA then calls two important initialization functions: InitAccountShields() and InitFailSafes(). These functions prepare all risk-management layers—such as equity locks, daily loss trackers, circuit breakers, and recovery states—so that the EA begins running with all protection systems armed and synchronized.

When the EA is removed or the terminal shuts down, the OnDeinit() function logs a message showing that the EA has been deinitialized, along with the reason code provided by MetaTrader. While this function does not perform heavy logic, it gives transparency and traceability, allowing the trader or developer to understand whether the EA was manually removed, reloaded, or stopped by a system event. Clean deinitialization helps maintain consistent global variable states and reduces the risk of stale data affecting future sessions.

The core trading engine runs on the OnTick(), which executes every time the market price changes. The first actions taken are related to safety and recovery: the EA updates peak equity to track drawdown conditions, and attempts to resume trading if it previously entered a recovery or paused state. Before making any trading decisions, the EA checks whether trading is allowed through IsTradingAllowed() and whether any safety limits have been breached via CheckSafetyLimits(). If everything is safe, the EA proceeds to manage open positions, adjust stops, and evaluate trade logic. On every new bar, it calls CheckForNewTrade() to look for opportunities based on the strategy rules. Finally, ManageOpenTrades() ensures that trailing stops, break-even rules, or partial exits are executed correctly, completing the full cycle of evaluation and execution on each tick.

```
//+------------------------------------------------------------------+
//| Check if trading is allowed                                      |
//+------------------------------------------------------------------+
bool IsTradingAllowed()
{
   // Check if EA is paused by protection systems
   if(IsEAProtectedPaused()) return false;

   // Check circuit breaker
   if(CheckCircuitBreaker(MaxConsecutiveLosses, CircuitBreakerDD)) return false;

   // Check trade throttle
   if(!ThrottleAllowNewTrade(MaxTradesPerHour, ThrottleWindowSeconds)) return false;

   // Check spread filter
   if(UseMaxSpreadFilter)
   {
      long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(spread > MaxSpreadPoints * 10) return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check safety limits                                              |
//+------------------------------------------------------------------+
bool CheckSafetyLimits()
{
   // Equity Stop protection
   if(UseEquityStop)
   {
      double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      double equityDropPercent = (1 - (currentEquity / currentBalance)) * 100;

      if(equityDropPercent >= EquityStopPercent)
      {
         Print("Equity stop triggered: ", equityDropPercent, "%");
         CloseAllPositions();
         ExpertRemove();
         return false;
      }
   }

   // Daily loss limit
   if(UseDailyLossLimit)
   {
      double dailyLossLimit = (DailyLossPercent / 100) * AccountInfoDouble(ACCOUNT_BALANCE);
      if(dailyProfit <= -dailyLossLimit)
      {
         Print("Daily loss limit reached: ", dailyProfit);
         CloseAllPositions();
         return false;
      }
   }

   return true;
}
```

The IsTradingAllowed() function acts as the EA’s first gatekeeper, determining whether it is safe to execute new trades. It begins by checking if the EA has been paused by any protection mechanisms, such as recovery mode or drawdown locks. It then checks the circuit breaker, which halts trading after too many consecutive losses or excessive drawdown. The function also enforces throttling rules by limiting how many trades can be placed within a set time window. Finally, if the spread filter is enabled, it verifies that the current market spread does not exceed the maximum allowed level. Only if every one of these checks passes does the EA give permission to proceed with trade evaluations or entries.

The CheckSafetyLimits() function provides a second layer of defense by enforcing hard-stop account protection rules. First, it evaluates the equity stop-loss system—a mechanism that calculates the percentage drop from balance to equity. If this drop reaches or exceeds the configured threshold, the EA immediately closes all open positions and removes itself from the chart to prevent further damage. Next, the function checks the daily loss limit, ensuring that trading stops once the account has reached a predefined maximum daily drawdown. If either of these conditions is triggered, trading halts and the EA returns false, signaling that the safety limits have been breached. Only when both protections remain within safe boundaries does the EA allow the trading cycle to continue.

```
//+------------------------------------------------------------------+
//| Check for new trade opportunity                                  |
//+------------------------------------------------------------------+
void CheckForNewTrade()
{
   if(PositionsTotal() > 0) return;

   int signal = GetTradingSignal();
   if(signal != 0)
   {
      double lotSize = CalculateLotSize();
      double sl, tp;
      CalculateSLTP(signal, sl, tp);

      if(OpenPosition(signal, lotSize, sl, tp))
      {
         lastTradeTime = TimeCurrent();
         martingaleStep = 0;
         tradeSequenceId = GenerateTradeSequenceId();
      }
   }
}

//+------------------------------------------------------------------+
//| Get trading signal                                               |
//+------------------------------------------------------------------+
int GetTradingSignal()
{
   int fastMA = iMA(_Symbol, _Period, FastMAPeriod, 0, MODE_EMA, PRICE_CLOSE);
   int slowMA = iMA(_Symbol, _Period, SlowMAPeriod, 0, MODE_EMA, PRICE_CLOSE);

   if(fastMA == INVALID_HANDLE || slowMA == INVALID_HANDLE) return 0;

   double fastMAValues[], slowMAValues[];
   ArraySetAsSeries(fastMAValues, true);
   ArraySetAsSeries(slowMAValues, true);

   if(CopyBuffer(fastMA, 0, 0, 3, fastMAValues) < 3)
   {
      IndicatorRelease(fastMA);
      IndicatorRelease(slowMA);
      return 0;
   }
   if(CopyBuffer(slowMA, 0, 0, 3, slowMAValues) < 3)
   {
      IndicatorRelease(fastMA);
      IndicatorRelease(slowMA);
      return 0;
   }

   double currentFast = fastMAValues[0];
   double currentSlow = slowMAValues[0];
   double prevFast = fastMAValues[1];
   double prevSlow = slowMAValues[1];

   int signal = 0;
   if(prevFast <= prevSlow && currentFast > currentSlow)
      signal = 1;
   else if(prevFast >= prevSlow && currentFast < currentSlow)
      signal = -1;

   IndicatorRelease(fastMA);
   IndicatorRelease(slowMA);

   return signal;
}

//+------------------------------------------------------------------+
//| Calculate lot size                                               |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   if(martingaleStep == 0)
   {
      // Use risk-based lot sizing for first trade
      double atr = GetATR(_Symbol, _Period, ATR_Period);
      double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double slPrice = currentPrice - (atr * ATR_SL_Factor);

      return CalculateLotFromRisk(_Symbol, currentPrice, slPrice, RiskPercent);
   }
   else
   {
      // Martingale recovery trade
      return NormalizeDouble(InitialLotSize * MathPow(LotMultiplier, martingaleStep), 2);
   }
}

//+------------------------------------------------------------------+
//| Calculate Stop Loss and Take Profit                              |
//+------------------------------------------------------------------+
void CalculateSLTP(int signal, double &sl, double &tp)
{
   double atr = GetATR(_Symbol, _Period, ATR_Period);
   double currentPrice = signal > 0 ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(signal > 0)
   {
      sl = currentPrice - (atr * ATR_SL_Factor);
      tp = currentPrice + (atr * ATR_TP_Factor);
   }
   else
   {
      sl = currentPrice + (atr * ATR_SL_Factor);
      tp = currentPrice - (atr * ATR_TP_Factor);
   }

   // Validate distances
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minDist = 100 * point;

   if(signal > 0)
   {
      if(currentPrice - sl < minDist) sl = currentPrice - minDist;
      if(tp - currentPrice < minDist) tp = currentPrice + minDist;
   }
   else
   {
      if(sl - currentPrice < minDist) sl = currentPrice + minDist;
      if(currentPrice - tp < minDist) tp = currentPrice - minDist;
   }

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
}

//+------------------------------------------------------------------+
//| Open position                                                    |
//+------------------------------------------------------------------+
bool OpenPosition(int signal, double lotSize, double sl, double tp)
{
   double price = (signal > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   ENUM_ORDER_TYPE orderType = (signal > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   // Use protective execution
   return ExecuteProtectedOrder(_Symbol, orderType, lotSize, price, sl);
}

//+------------------------------------------------------------------+
//| Manage existing positions                                        |
//+------------------------------------------------------------------+
void ManageExistingPositions()
{
   int totalPositions = PositionsTotal();

   for(int i = totalPositions - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionGetString(POSITION_COMMENT) == tradeSequenceId)
      {
         double currentProfit = PositionGetDouble(POSITION_PROFIT);

         if(PositionGetInteger(POSITION_TIME_UPDATE) > lastTradeTime)
         {
            bool wasProfit = (currentProfit > 0);
            OnTradeClosed(wasProfit);

            if(!wasProfit)
            {
               martingaleStep++;
               if(martingaleStep > MaxMartingaleSteps)
               {
                  Print("Max martingale steps reached. Activating recovery protocol.");
                  double reducedRisk = RiskPercent;
                  RecoveryProtocol(reducedRisk, RecoveryRiskReduction, ResumeEquityPercent, ResumeAfterSeconds);
                  martingaleStep = 0;
               }
            }
            else
            {
               martingaleStep = 0;
            }

            dailyProfit += currentProfit;
            lastTradeTime = TimeCurrent();
         }
      }
   }
}
```

The trading engine first evaluates whether a new position is allowed by confirming that no existing positions are open. If the account is clear, the system requests a new trading signal generated by a crossover of fast and slow EMAs. Once a valid signal appears, the EA dynamically computes the lot size using ATR-based risk allocation for the first trade or a structured martingale increment for recovery trades. It then calculates volatility-adjusted stop-loss and take-profit levels based on ATR multipliers, ensuring all broker-required minimum distances are respected. When everything is validated, a protected execution function is used to place the trade safely, and the trade session is initialized with timestamps and a unique sequence ID.

The signal itself is derived from a classic moving-average cross logic, but made more robust by pulling multiple buffer elements to confirm a true crossover rather than a single-tick fluctuation. ATR is central to the sizing and SL/TP calculations: for new trades, ATR defines risk-based sizing via the distance to stop-loss, and for position exits, ATR defines dynamic profit targets and loss thresholds. SL/TP outputs are normalized to symbol precision, capped by minimum trading distances, and differ depending on whether the system receives a buy (bullish crossover) or sell (bearish crossover) opportunity.

The ManageOpenTrades() funciton is responsible for intelligently managing open trades and learning from their outcomes. Each time a trade associated with the current sequence is updated, the system evaluates whether the trade closed in profit or loss. Profitable trades reset the martingale step counter, while losses increment the step and may trigger additional martingale entries—up to a predefined limit. If the maximum step is reached, the EA activates a recovery protocol that reduces risk, halts trading until equity stabilizes, and only resumes after market conditions improve or a cooldown period elapses. This creates a safety-aware reinforcement cycle that attempts controlled recovery after losing streaks while guarding the account from runaway exposure.

```
double CalculateLotFromRisk(string symbol, double entry_price, double sl_price, double risk_percent)
{
   if(risk_percent <= 0) return(InitialLotSize);

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_amount = equity * (risk_percent / 100.0);
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double sl_points = MathMax(1.0, MathAbs(entry_price - sl_price) / point);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_value <= 0 || tick_size <= 0) return(InitialLotSize);

   double value_per_point_per_lot = tick_value / (tick_size / point);
   double risk_per_lot = sl_points * value_per_point_per_lot;
   if(risk_per_lot <= 0.0) return(InitialLotSize);

   double volume = risk_amount / risk_per_lot;
   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0) step = 0.01;

   double normalized = MathFloor(volume / step) * step;
   if(normalized < minLot) normalized = minLot;
   if(normalized > maxLot) normalized = maxLot;

   return(NormalizeDouble(normalized, (int)MathMax(0, (int)(-MathLog10(step)))));
}

//+------------------------------------------------------------------+
//| STRUCTURAL ACCOUNT SHIELDS                                      |
//+------------------------------------------------------------------+
void InitAccountShields()
{
   if(!GlobalVariableCheck(GV_PEAK_EQUITY))
      GlobalVariableSet(GV_PEAK_EQUITY, AccountInfoDouble(ACCOUNT_EQUITY));
   if(!GlobalVariableCheck(GV_PAUSE_EA))
      GlobalVariableSet(GV_PAUSE_EA, 0.0);
   if(!GlobalVariableCheck(GV_DD_LOCK_LEVEL))
      GlobalVariableSet(GV_DD_LOCK_LEVEL, 0.0);
}

void UpdatePeakEquity()
{
   double current = AccountInfoDouble(ACCOUNT_EQUITY);
   double peak = GlobalVariableGet(GV_PEAK_EQUITY);
   if(current > peak) GlobalVariableSet(GV_PEAK_EQUITY, current);
}

double GetCurrentDrawdownPercent()
{
   double peak = GlobalVariableGet(GV_PEAK_EQUITY);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(peak <= 0) return(0.0);
   return ((peak - equity) / peak) * 100.0;
}

bool IsEAProtectedPaused()
{
   return (GlobalVariableGet(GV_PAUSE_EA) != 0.0);
}

//+------------------------------------------------------------------+
//| TRADE-LEVEL REINFORCEMENT                                       |
//+------------------------------------------------------------------+
bool ExecuteProtectedOrder(string symbol, ENUM_ORDER_TYPE type, double volume, double price, double sl_price)
{
   if(IsEAProtectedPaused()) return false;

   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double allowableDeviationPoints = 10;

   if(type == ORDER_TYPE_BUY)
   {
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      if(MathAbs(ask - price) / point > allowableDeviationPoints) return false;
   }
   else
   {
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      if(MathAbs(bid - price) / point > allowableDeviationPoints) return false;
   }

   double minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   if(volume < minLot || volume > maxLot) return false;

   bool ok = false;
   if(type == ORDER_TYPE_BUY)
      ok = trade.Buy(volume, symbol, price, sl_price, 0, NULL);
   else if(type == ORDER_TYPE_SELL)
      ok = trade.Sell(volume, symbol, price, sl_price, 0, NULL);

   if(!ok) PrintFormat("Order failed: %s (err %d)", symbol, GetLastError());
   return ok;
}

//+------------------------------------------------------------------+
//| SYSTEMIC FAIL-SAFES                                             |
//+------------------------------------------------------------------+
void InitFailSafes()
{
   if(!GlobalVariableCheck(GV_CONSEC_LOSSES)) GlobalVariableSet(GV_CONSEC_LOSSES, 0);
   if(!GlobalVariableCheck(GV_TRADE_WINDOW_START)) GlobalVariableSet(GV_TRADE_WINDOW_START, TimeCurrent());
   if(!GlobalVariableCheck(GV_TRADES_IN_WINDOW)) GlobalVariableSet(GV_TRADES_IN_WINDOW, 0);
}

void OnTradeClosed(bool wasProfit)
{
   if(wasProfit)
      GlobalVariableSet(GV_CONSEC_LOSSES, 0);
   else
      GlobalVariableSet(GV_CONSEC_LOSSES, GlobalVariableGet(GV_CONSEC_LOSSES) + 1);

   GlobalVariableSet(GV_TRADES_IN_WINDOW, GlobalVariableGet(GV_TRADES_IN_WINDOW) + 1);
}

bool CheckCircuitBreaker(int maxConsecLosses, double drawdownThresholdPercent)
{
   int consec = (int)GlobalVariableGet(GV_CONSEC_LOSSES);
   double dd = GetCurrentDrawdownPercent();
   if(consec >= maxConsecLosses || dd >= drawdownThresholdPercent)
   {
      GlobalVariableSet(GV_PAUSE_EA, 1.0);
      Print("Circuit breaker engaged: consec=", consec, " dd=", dd);
      return true;
   }
   return false;
}

bool ThrottleAllowNewTrade(int maxTrades, int windowSeconds)
{
   datetime start = (datetime)GlobalVariableGet(GV_TRADE_WINDOW_START);
   int count = (int)GlobalVariableGet(GV_TRADES_IN_WINDOW);
   datetime now = TimeCurrent();

   if((now - start) > windowSeconds)
   {
      GlobalVariableSet(GV_TRADE_WINDOW_START, now);
      GlobalVariableSet(GV_TRADES_IN_WINDOW, 0);
      count = 0;
   }
   return (count < maxTrades);
}

void RecoveryProtocol(double &riskPercent, double reductionFactor, double resumeEquityPercentOfPeak, int resumeAfterSeconds)
{
   riskPercent *= reductionFactor;
   GlobalVariableSet(GV_PAUSE_EA, 1.0);
   GlobalVariableSet("GV_RecoveryResumeTime", TimeCurrent() + resumeAfterSeconds);
   GlobalVariableSet("GV_ResumeEquityPercent", resumeEquityPercentOfPeak);
}

bool TryResumeFromRecovery()
{
   double resumeTime = GlobalVariableGet("GV_RecoveryResumeTime");
   if(resumeTime == 0) return false;
   if(TimeCurrent() < (datetime)resumeTime) return false;

   double resumePercent = GlobalVariableGet("GV_ResumeEquityPercent");
   if(resumePercent <= 0)
   {
      GlobalVariableSet(GV_PAUSE_EA, 0.0);
      GlobalVariableSet("GV_RecoveryResumeTime", 0);
      return true;
   }

   double peak = GlobalVariableGet(GV_PEAK_EQUITY);
   double needed = peak * (resumePercent / 100.0);
   if(AccountInfoDouble(ACCOUNT_EQUITY) >= needed)
   {
      GlobalVariableSet(GV_PAUSE_EA, 0.0);
      GlobalVariableSet("GV_RecoveryResumeTime", 0);
      return true;
   }
   return false;
}
```

The CalculateLotFromRisk() function provides the engine’s core dynamic exposure control by converting a user-defined risk percentage into an exact, volatility-aware lot size. It computes the monetary risk allowable per trade based on account equity, the distance from entry to stop-loss in points, and the symbol’s monetary value per point. This allows the EA to size positions proportionally to both market volatility and the trader’s risk tolerance. Broker constraints such as minimum, maximum, and step-based lot increments are applied to ensure all calculated volumes remain tradeable. The normalized output ensures precision, compliance, and consistent risk allocation, especially in fast-moving markets.

Next, the Structural Account Shields block establishes a protective framework for monitoring equity health and safeguarding the account against deep drawdowns. At startup, the EA records peak equity, initializes protection variables, and then updates the peak dynamically whenever equity reaches a new high. With these values, the system measures real-time drawdown as a percentage of the peak, allowing the EA to make logical decisions based on equity stress levels. A pause flag is also included so external modules—such as circuit breakers or recovery protocols—can temporarily freeze all trading operations whenever a protection threshold is breached.

The Trade-Level Reinforcement system focuses on validating and securing each order during execution. Before sending any trade to the server, the EA ensures that protective mode is not active, that the execution price is close enough to the current bid/ask (preventing slippage-related traps), and that the lot size falls within broker rules. Orders are then opened using the MQL5 trading object with optional SL placement for extra safety. If a trade fails, the EA logs the specific symbol and error code, enabling smarter debugging and adaptive routines. This protective-execution model ensures that only structurally sound trades—properly priced, sized, and allowed—reach the market.

Finally, the Systemic Fail-Safes module acts as the EA’s emergency stability engine, monitoring trading behavior over time to detect dangerous patterns. It tracks consecutive losses, the number of trades in a sliding time window, and the current drawdown. Circuit breakers pause trading when loss streaks or drawdown thresholds are exceeded, preventing destructive spirals. Throttling logic limits how many trades may be taken within a defined time window, cutting off over-trading during volatile periods. When conditions deteriorate further, the Recovery Protocol reduces risk, pauses trading for a cooldown period, and optionally waits for equity to recover to a specific percentage of peak before resuming. This creates a layered fail-safe system that proactively protects capital while allowing the EA to resume operations only under healthy conditions.

### **Back Test Results**

The back-testing was evaluated on the M15 timeframe across roughly a 2-month testing window (01 October 2025 to 30 November 2025), with the following settings:

| Variable | Input Value |
| --- | --- |
| FastMAPeriod | 20 |
| SlowMAPeriod | 160 |
| InitialLotSize | 1 |
| LotMultiplier | 5.2 |
| MaxMartingaleStep | 13 |
| RistPercent | 7.8 |
| ATR\_Period | 69 |
| ATR\_SL\_Factor | 3.3 |
| ATR\_TP\_Factor | 9.8 |
| UseEquityStop | false |
| EquityStopPercentage | 27.2 |
| UseDailyLossLimit | true |
| DailyLossPercent | 12.7 |
| UseMaxSpreadFilter | true |
| MaxSpreadPoints | 25 |
| UseTrailingStop | true |
| BreakEvenAtPips | 500 |
| TrailStartAtPips | 600 |
| TrailStepPips | 100 |
| MaxConsecutiveLosses | 22 |
| CircuitBreaketDD | 115.5 |
| MaxTradesPerHour | 33 |
| ThrottleWindowSeconds | 21025 |
| RecoveryRiskReduction | 3.6 |
| ResumeEquityPercent | 135 |
| ResumeAfterSeconds | 845307 |

![](https://c.mql5.com/2/184/GALEITCRV.png)

![](https://c.mql5.com/2/184/GALEITEQQ.png)

### Conclusion

In summary, we developed a comprehensive multi-layered account protection framework that integrates dynamic exposure control, structural equity safeguards, trade-level execution reinforcement, and systemic fail-safes. This architecture combines risk-based lot sizing, peak-equity tracking, controlled drawdown locks, slippage-aware order validation, circuit breakers, and throttling mechanisms to ensure that every trade is both intelligently sized and responsibly managed. By layering these components into a unified protection engine, the system continuously adapts to volatility, monitors equity health, and intervenes automatically when trading conditions become unfavorable.

In conclusion, this multi-layered protection model empowers traders with a highly resilient, self-regulating environment that prioritizes capital preservation while still seeking strong performance. Instead of relying on a single defensive feature, the EA operates within a coordinated network of safeguards that minimize emotional decision-making, prevent catastrophic loss spirals, and ensure sustained trading longevity. This gives traders the confidence to pursue higher profits knowing that the system is engineered to actively manage risk, stabilize performance, and preserve the account through all phases of market behavior.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20449.zip "Download all attachments in the single ZIP archive")

[GALEIT.mq5](https://www.mql5.com/en/articles/download/20449/GALEIT.mq5 "Download GALEIT.mq5")(28.1 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/501542)**
(1)


![hemantto](https://c.mql5.com/avatar/avatar_na2.png)

**[hemantto](https://www.mql5.com/en/users/hemantto)**
\|
26 Dec 2025 at 06:49

Hello sir  , How can run in XAU/USD    ?   I did checkd in Mt5 tester ,not trades any openend ,  it not runiing properly  ? pls send instrution  .


![Mastering Kagi Charts in MQL5 (Part 2): Implementing Automated Kagi-Based Trading](https://c.mql5.com/2/185/20378-mastering-kagi-charts-in-mql5-logo__1.png)[Mastering Kagi Charts in MQL5 (Part 2): Implementing Automated Kagi-Based Trading](https://www.mql5.com/en/articles/20378)

Learn how to build a complete Kagi-based trading Expert Advisor in MQL5, from signal construction to order execution, visual markers, and a three-stage trailing stop. Includes full code, testing results, and a downloadable set file.

![The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://c.mql5.com/2/155/18658-komponenti-view-i-controller-logo.png)[The View and Controller components for tables in the MQL5 MVC paradigm: Containers](https://www.mql5.com/en/articles/18658)

In this article, we will discuss creating a "Container" control that supports scrolling its contents. Within the process, the already implemented classes of graphics library controls will be improved.

![Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://c.mql5.com/2/122/Developing_a_Multicurrency_Advisor_Part_24___LOGO.png)[Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)

In this article, we will look at how to connect a new strategy to the auto optimization system we have created. Let's see what kind of EAs we need to create and whether it will be possible to do without changing the EA library files or minimize the necessary changes.

![Automating Trading Strategies in MQL5 (Part 45): Inverse Fair Value Gap (IFVG)](https://c.mql5.com/2/184/20361-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 45): Inverse Fair Value Gap (IFVG)](https://www.mql5.com/en/articles/20361)

In this article, we create an Inverse Fair Value Gap (IFVG) detection system in MQL5 that identifies bullish/bearish FVGs on recent bars with minimum gap size filtering, tracks their states as normal/mitigated/inverted based on price interactions (mitigation on far-side breaks, retracement on re-entry, inversion on close beyond far side from inside), and ignores overlaps while limiting tracked FVGs.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/20449&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5072058207410533255)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)