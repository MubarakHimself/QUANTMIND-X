---
title: Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching
url: https://www.mql5.com/en/articles/20414
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:44:24.562438
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/20414&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083077379800175750)

MetaTrader 5 / Examples


### Table of contents:

1. [Introduction](https://www.mql5.com/en/articles/20414#Introduction)
2. [Strategy Overview](https://www.mql5.com/en/articles/20414#StrategyOverview)
3. [Getting Started](https://www.mql5.com/en/articles/20414#GettingStarted)
4. [Backtest Results](https://www.mql5.com/en/articles/20414#BacktestResults)
5. [Conclusion](https://www.mql5.com/en/articles/20414#Conclusion)

### Introduction

Over the last phases of development, we successfully combined Order Blocks (OB), Fair Value Gaps (FVG), and Break of Structure (BOS) into one unified [Smart Money Concepts](https://www.mql5.com/en/articles/16340) engine—a powerful framework capable of reading institutional price action with precision. Building on that foundation, we also designed a [custom market sentiment indicator](https://www.mql5.com/en/articles/19422) that interprets trend strength, volatility posture, and momentum alignment to classify the market into bullish, bearish, or neutral states. These two advancements now give us the perfect opportunity to merge structural price action with real-time sentiment into a single adaptive trading system.

This next stage focuses on creating an intelligent EA that dynamically switches strategies based on what the market is currently doing. Instead of relying on a fixed pattern, the EA will leverage our sentiment indicator to determine the dominant regime and activate the appropriate OB, BOS, or FVG strategy accordingly. In bullish conditions it may favor BOS continuation setups; during reversals it may prioritize OB-based rejections; and in compression environments it may shift to FVG mean-reversion logic. By blending structural logic with sentiment-driven adaptability, we move closer to building a trading system that reacts, evolves, and positions itself exactly the way a top-tier institutional trader would.

### Strategy Overview: Market Sentiment-Driven SMC Trading System

This intelligent trading system integrates three core Smart Money Concept (SMC) strategies—Order Blocks (OB), Fair Value Gaps (FVG), and Break of Structure (BOS)—with real-time market sentiment analysis to create an adaptive, context-aware trading algorithm. The system operates on a multi-timeframe framework, analyzing price action across higher timeframes (H4), medium timeframes (H1), and lower timeframes (M30) to determine overall market bias and sentiment. Based on calculated sentiment readings ranging from Strong Bullish to Strong Bearish, Risk-On to Risk-Off, and Neutral conditions, the EA dynamically selects and prioritizes the most appropriate SMC strategy, ensuring that trading decisions align with prevailing market conditions rather than applying a one-size-fits-all approach.

Neutral Market Sentiment + Order Blocks Strategy

During neutral market conditions—characterized by price consolidation near moving averages and the absence of clear higher timeframe bias—the system activates the Order Blocks (OB) strategy as its primary trading method. Neutral sentiment typically indicates range-bound markets where price oscillates between established support and resistance levels without strong directional momentum. In this environment, Order Blocks become highly effective as they identify areas where institutional orders were previously placed, creating natural reaction zones. The EA specifically looks for bullish OBs (bear candle followed by strong bull candle) and bearish OBs (bull candle followed by strong bear candle) that demonstrate significant order flow imbalance. When price retraces to these identified OB zones, the system executes trades with the expectation of mean reversion, capitalizing on the market's tendency to respect previous order accumulation areas during consolidation phases.

![](https://c.mql5.com/2/185/NeutralOB.png)

Trending (Bullish/Bearish) Market Sentiment + Break of Structure Strategy

When the system detects Strong Bullish or Strong Bearish sentiment—indicating clear directional trends confirmed across multiple timeframes—it prioritizes the Break of Structure (BOS) strategy to capitalize on momentum continuation. In trending markets characterized by higher highs and higher lows (bullish) or lower highs and lower lows (bearish), BOS identifies key swing points where market structure has been broken, signaling potential acceleration in the prevailing direction. The EA monitors for price breaking above recent swing highs in bullish trends or below recent swing lows in bearish trends, interpreting these breaks as liquidity runs that often precede sustained directional moves. This strategy aligns perfectly with trending conditions, as it focuses on entering in the direction of the established trend following structural confirmation, avoiding counter-trend trades that would have a lower probability of success during strong directional moves.

![](https://c.mql5.com/2/185/Bullishbos.png)

Risk-On/Risk-Off Sentiment + Fair Value Gaps Strategy

During Risk-On or Risk-Off market environments—characterized by breakouts from established ranges combined with higher timeframe bias—the system employs the Fair Value Gaps (FVG) strategy. These conditions typically occur when price has been consolidating but suddenly experiences momentum expansion, creating gaps between consecutive candles where minimal trading occurred. The EA identifies these imbalance zones where buying or selling pressure created significant price voids, anticipating that price will eventually return to "fair value" within these gaps. The system specifically trades at the 50% equilibrium level of these gaps, providing optimal risk-reward entry points during volatile breakout periods. This approach capitalizes on the market's tendency to fill price inefficiencies while respecting the overall directional bias, making it ideal for transitional market phases where momentum is building but the trend may not yet be fully established.

![](https://c.mql5.com/2/185/Riskfvg.png)

### Getting Started

```
//+------------------------------------------------------------------+
//|                                                     SMC_Sent.mq5 |
//|                        GIT under Copyright 2025, MetaQuotes Ltd. |
//|                     https://www.mql5.com/en/users/johnhlomohang/ |
//+------------------------------------------------------------------+
#property copyright "GIT under Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/johnhlomohang/"
#property version   "2.00"
#property description "Intelligent SMC EA with Market Sentiment-Based Strategy Switching"
#property copyright "Based on MARKSENT and SMCALL"

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include <Arrays/ArrayObj.mqh>

// Market Sentiment Component (from MARKSENT)
input group "=== Market Sentiment Settings ==="
input ENUM_TIMEFRAMES HigherTF = PERIOD_H4;
input ENUM_TIMEFRAMES LowerTF1 = PERIOD_H1;
input ENUM_TIMEFRAMES LowerTF2 = PERIOD_M30;
input int MAPeriod = 200;
input int SwingLookback = 5;
input double ATRThreshold = 0.002;

input group "=== Trading Settings ==="
input double LotSize = 0.02;
input double StopLoss = 500;
input double TakeProfit = 1500;
input long MagicNumber = 76543;
input bool EnableTrading = true;
input bool DrawAllObjects = true; // Draw all trading objects

input group "=== Strategy Selection ==="
input bool UseSentimentFilter = true; // Use sentiment to choose strategies
input bool AllowBOS = true;           // Allow Break of Structure strategy
input bool AllowOB = true;            // Allow Order Blocks strategy
input bool AllowFVG = true;           // Allow Fair Value Gaps strategy

input group "=== Visual Settings ==="
input int PanelCorner = 0;           // Top-left corner
input int PanelX = 10;
input int PanelY = 10;
input string FontFace = "Arial";
input int FontSize = 10;

// Color scheme based on market sentiment
input color BullishColor = clrLimeGreen;
input color BearishColor = clrRed;
input color RiskOnColor = clrDodgerBlue;
input color RiskOffColor = clrOrangeRed;
input color NeutralColor = clrGold;

// Strategy drawing colors
input color OB_BullColor = clrLime;
input color OB_BearColor = clrRed;
input color FVG_BullColor = clrPaleGreen;
input color FVG_BearColor = clrMistyRose;
input color BOS_BullColor = clrDodgerBlue;
input color BOS_BearColor = clrTomato;

// Cleanup settings
input bool RemoveObjectsAfterTradeClose = true;
input int RemoveObjectsAfterBars = 10; // Remove objects after X bars

// Global variables
CTrade trade;
CPositionInfo poss;

// Market Sentiment Handles
int higherTFHandle, lowerTF1Handle, lowerTF2Handle;
double higherTFMA[], lowerTF1MA[], lowerTF2MA[];
datetime lastSentimentUpdate = 0;
int currentSentiment = 0; // -2:RiskOff, -1:Bearish, 0:Neutral, 1:Bullish, 2:RiskOn
string currentSentimentText = "Neutral";

// SMC Trading Variables
datetime lastBarTime = 0;
double Bid, Ask;
string currentStrategy = "ALL";
```

We begin by including the necessary MQL5 libraries for order execution, position information, and array management, forming the foundation for both the trading operations and data storage required by the EA. The first section defines all inputs for the Market Sentiment Component, which reads and processes data across multiple timeframes (H4, H1, M30) using tools such as a 200-period moving average, swing lookbacks, and ATR thresholds. These parameters belong to the custom sentiment engine (MARKSENT), and they allow the EA to classify the market environment into states such as bullish, bearish, risk-on, risk-off, or neutral. By analyzing these different timeframes together, the EA builds a robust understanding of the market’s current direction and volatility characteristics.

In the next section, we define the Trading Settings and Strategy Selection, which give the EA flexibility in how it operates. Inputs such as lot size, stop loss, take profit, magic number, and object-drawing controls determine how orders are executed and visualized. More importantly, the Strategy Selection block allows the user to enable or disable responses to market sentiment and selectively activate BOS, OB, and FVG logic. This means the EA can be configured to trade exclusively based on sentiment, exclusively based on SMC patterns, or operate as a hybrid model that intelligently switches strategies depending on the sentiment state detected by the indicator. Combined with customizable visual parameters, this section ensures both functionality and clarity on the chart.

We then set up all the global variables and buffers needed for real-time calculations, including handles for higher- and lower-timeframe MAs, sentiment state tracking, and SMC strategy activity. Variables like currentSentiment, currentSentimentText, and currentStrategy serve as decision-making anchors: they store the current market mood and the active strategy to be used on the next signal. The EA continuously updates Bid/Ask prices, identifies new bars, and determines when to remove old SMC objects from the chart for effective visualization. Together, these global components create the core framework that allows the EA to merge market sentiment with Smart Money Concepts, enabling an adaptive trading system that adjusts its logic as the market evolves.

```
// Trade information structure
struct TradeInfo
{
   long ticket;
   string symbol;
   datetime openTime;
   double openPrice;
   ENUM_ORDER_TYPE type;
   string strategy;
   long magic;

   TradeInfo() : ticket(-1), symbol(""), openTime(0), openPrice(0), type(WRONG_VALUE), strategy(""), magic(0) {}
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(MagicNumber);
    indicatorName = "IntelligentSMC_" + IntegerToString(ChartID());

    // Initialize market sentiment indicators
    higherTFHandle = iMA(_Symbol, HigherTF, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);
    lowerTF1Handle = iMA(_Symbol, LowerTF1, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);
    lowerTF2Handle = iMA(_Symbol, LowerTF2, MAPeriod, 0, MODE_EMA, PRICE_CLOSE);

    ArraySetAsSeries(higherTFMA, true);
    ArraySetAsSeries(lowerTF1MA, true);
    ArraySetAsSeries(lowerTF2MA, true);

    // Initialize arrays
    detectedFVGs = new CArrayObj();
    detectedBOSZones = new CArrayObj();
    tradedOBs = new CArrayObj();
    activeTrades = new CArrayObj();

    CreateControlPanel();

    Print("Intelligent SMC EA Started - Strategy Switching & Clean Drawing Enabled");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    IndicatorRelease(higherTFHandle);
    IndicatorRelease(lowerTF1Handle);
    IndicatorRelease(lowerTF2Handle);

    // Clean up all objects
    CleanupAllObjects();

    if(currentOB != NULL)
    {
        delete currentOB;
        currentOB = NULL;
    }

    // Clean up FVGs
    if(detectedFVGs != NULL)
    {
        for(int i = detectedFVGs.Total()-1; i >= 0; i--)
        {
            CFVG* fvg = (CFVG*)detectedFVGs.At(i);
            if(fvg != NULL) delete fvg;
        }
        delete detectedFVGs;
    }

    // Clean up BOS zones
    if(detectedBOSZones != NULL)
    {
        for(int i = detectedBOSZones.Total()-1; i >= 0; i--)
        {
            CBOSZone* bos = (CBOSZone*)detectedBOSZones.At(i);
            if(bos != NULL) delete bos;
        }
        delete detectedBOSZones;
    }

    // Clean up traded OBs
    if(tradedOBs != NULL)
    {
        for(int i = tradedOBs.Total()-1; i >= 0; i--)
        {
            COrderBlock* ob = (COrderBlock*)tradedOBs.At(i);
            if(ob != NULL) delete ob;
        }
        delete tradedOBs;
    }

    // Clean up active trades
    if(activeTrades != NULL)
    {
        delete activeTrades;
    }

    ObjectsDeleteAll(0, indicatorName);
    Comment("");
}

//+------------------------------------------------------------------+
//| Cleanup all graphical objects                                    |
//+------------------------------------------------------------------+
void CleanupAllObjects()
{
    // Clean up old objects
    string prefix = "";
    int total = ObjectsTotal(0);

    for(int i = total-1; i >= 0; i--)
    {
        string name = ObjectName(0, i);
        if(StringFind(name, "OB_", 0) == 0 ||
           StringFind(name, "FVG_", 0) == 0 ||
           StringFind(name, "BOS_", 0) == 0 ||
           StringFind(name, "Trade_", 0) == 0)
        {
            ObjectDelete(0, name);
        }
    }
}
```

This section defines the TradeInfo structure, which stores all essential information about trades opened by the EA—ticket, symbol, entry time, price, order type, strategy used, and magic number. The default constructor initializes every field to a safe “empty” state, ensuring no uninitialized values cause logic errors when the EA processes or stores trade data. This structure becomes especially important for an adaptive, multi-strategy system like this one, because it allows the EA to track which strategy (OB, FVG, or BOS) triggered each trade, enabling features such as performance tracking, cleanup, or strategy-specific trade management.

The OnInit() and OnDeinit() functions handle the setup and teardown of the entire EA. On initialization, the EA sets the magic number, prepares unique indicator names, and creates all moving average handles used by the market sentiment engine across multiple timeframes. It also configures arrays as series and allocates object containers for detected SMC structures—FVGs, BOS zones, and traded OBs—as well as active trade tracking. A control panel is created for on-chart monitoring, and the EA announces that dynamic SMC and strategy-switching logic are ready. Conversely, OnDeinit() ensures a clean shutdown by releasing indicator handles, deleting all dynamically allocated SMC objects, clearing trade arrays, and removing every OB/FVG/BOS drawing from the chart. The dedicated CleanupAllObjects() function helps maintain a tidy chart environment by systematically deleting any graphical object that matches SMC naming conventions. Together, these routines ensure that the EA initializes cleanly, manages memory responsibly, and leaves no clutter behind when removed.

```
// Market Sentiment Calculation Component
int CalculateMarketSentiment()
{
    if(TimeCurrent() - lastSentimentUpdate < 5)
        return currentSentiment;

    lastSentimentUpdate = TimeCurrent();

    // Get MA values from multiple timeframes
    CopyBuffer(higherTFHandle, 0, 0, 3, higherTFMA);
    CopyBuffer(lowerTF1Handle, 0, 0, 3, lowerTF1MA);
    CopyBuffer(lowerTF2Handle, 0, 0, 3, lowerTF2MA);

    double higherTFPrice = iClose(_Symbol, HigherTF, 0);
    double lowerTF1Price = iClose(_Symbol, LowerTF1, 0);
    double lowerTF2Price = iClose(_Symbol, LowerTF2, 0);

    // Calculate biases across timeframes
    int higherTFBias = GetHigherTFBias(higherTFPrice, higherTFMA[0]);
    bool lowerTF1Bullish = IsBullishStructure(LowerTF1, SwingLookback);
    bool lowerTF1Bearish = IsBearishStructure(LowerTF1, SwingLookback);
    bool lowerTF2Bullish = IsBullishStructure(LowerTF2, SwingLookback);
    bool lowerTF2Bearish = IsBearishStructure(LowerTF2, SwingLookback);
    bool lowerTF1Breakout = HasBreakout(LowerTF1, SwingLookback, higherTFBias);
    bool lowerTF2Breakout = HasBreakout(LowerTF2, SwingLookback, higherTFBias);

    // Determine final sentiment based on multi-timeframe analysis
    currentSentiment = DetermineSentiment(higherTFBias,
        lowerTF1Bullish, lowerTF1Bearish, lowerTF1Breakout,
        lowerTF2Bullish, lowerTF2Bearish, lowerTF2Breakout);

    // Update sentiment text for display
    switch(currentSentiment)
    {
        case 1: currentSentimentText = "Bullish"; break;
        case -1: currentSentimentText = "Bearish"; break;
        case 2: currentSentimentText = "Risk-On"; break;
        case -2: currentSentimentText = "Risk-Off"; break;
        default: currentSentimentText = "Neutral"; break;
    }

    return currentSentiment;
}

// Strategy Selection Logic based on Market Sentiment
string SelectTradingStrategy()
{
    if(!UseSentimentFilter)
    {
        currentStrategy = "ALL"; // Use all strategies if no filter
        return currentStrategy;
    }

    int sentiment = CalculateMarketSentiment();

    // Strategy assignment based on market conditions
    switch(sentiment)
    {
        case 1:  // Strong Bullish
        case -1: // Strong Bearish
            currentStrategy = "BOS"; // Use Break of Structure in strong trends
            break;

        case 2:  // Risk-On (Bullish with breakout)
        case -2: // Risk-Off (Bearish with breakout)
            currentStrategy = "FVG"; // Use Fair Value Gaps during breakouts
            break;

        case 0:  // Neutral/Ranging
        default:
            currentStrategy = "OB"; // Use Order Blocks in ranging markets
            break;
    }

    return currentStrategy;
}
```

Here we calculate the market sentiment by blending multi-timeframe moving averages, structural swing analysis, and breakout detection into one unified bias score. The function begins by limiting recalculations to every 5 seconds for efficiency, then retrieves MA values from the higher and lower timeframe handles. It examines price relative to the MAs to determine higher-timeframe direction while simultaneously analyzing lower-timeframe structures using functions such as IsBullishStructure, IsBearishStructure, and HasBreakout. These layers of analysis—trend, structure, and breakouts—are passed to DetermineSentiment, which returns a sentiment classification ranging from strong bullish (1) to risk-off (-2). Once the numerical sentiment is determined, the EA assigns a descriptive label ("Bullish", "Bearish", etc.) for display on the panel.

The strategy selection logic then uses this sentiment output to dynamically choose which SMC methodology the EA should trade. If sentiment filtering is disabled, the EA simply enables all strategies. But when active, the EA intelligently adapts: strong bullish or bearish conditions activate BOS continuation trading, breakout-driven risk-on and risk-off environments prioritize FVG imbalance trading, and neutral or ranging markets favor Order Block reversals. This creates an adaptive system where the EA automatically rotates between BOS, FVG, and OB strategies based on real-time sentiment and price behavior, mimicking the decision-making of a professional trader who adjusts methods according to market regime.

```
// Execute Trade with Sentiment Filtering
bool ExecuteTradeWithFilter(ENUM_ORDER_TYPE type, string strategy)
{
    if(!EnableTrading) return false;

    // Check if selected strategy is allowed
    if((strategy == "BOS" && !AllowBOS) ||
       (strategy == "OB" && !AllowOB) ||
       (strategy == "FVG" && !AllowFVG))
        return false;

    // Check sentiment alignment before executing trade
    int sentiment = CalculateMarketSentiment();
    bool sentimentAligned = false;

    // Bullish trades aligned with bullish sentiment
    if((type == ORDER_TYPE_BUY && (sentiment == 1 || sentiment == 2)) ||
       // Bearish trades aligned with bearish sentiment
       (type == ORDER_TYPE_SELL && (sentiment == -1 || sentiment == -2)))
        sentimentAligned = true;

    // Allow neutral sentiment trades with caution
    if(sentiment == 0) sentimentAligned = true;

    if(!sentimentAligned)
    {
        Print("Trade rejected: Not aligned with market sentiment (", currentSentimentText, ")");
        return false;
    }

    // Execute the trade with strategy context
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double price = (type == ORDER_TYPE_BUY) ? Ask : Bid;
    double sl = (type == ORDER_TYPE_BUY) ? price - StopLoss * point : price + StopLoss * point;
    double tp = (type == ORDER_TYPE_BUY) ? price + TakeProfit * point : price - TakeProfit * point;

    sl = NormalizeDouble(sl, _Digits);
    tp = NormalizeDouble(tp, _Digits);

    trade.SetExpertMagicNumber(MagicNumber);
    bool result = trade.PositionOpen(_Symbol, type, LotSize, price, sl, tp,
       "SMC_" + strategy + "_Sent:" + currentSentimentText);

    if(result)
    {
        long ticket = trade.ResultOrder();
        Print("Trade executed: ", EnumToString(type),
              " | Strategy: ", strategy,
              " | Sentiment: ", currentSentimentText,
              " | Price: ", DoubleToString(price, _Digits),
              " | Ticket: ", ticket);

        return true;
    }

    return false;
}
```

This function ensures that every trade executed by the EA is both strategy-approved and sentiment-aligned, creating a safety layer that prevents trades from being placed against prevailing market conditions. It begins by checking whether trading is enabled and whether the selected SMC strategy (BOS, OB, or FVG) is currently allowed. It then retrieves the latest market sentiment and validates whether the intended order direction matches that sentiment—for example, buy trades are only permitted during bullish or risk-on conditions, while sell trades require bearish or risk-off sentiment.

Neutral markets allow trades but with added caution. If the sentiment and strategy align, the function calculates normalized SL and TP levels, assigns the EA’s magic number, and opens a new position with a comment describing the strategy and sentiment used. Successful trades are logged with detailed information, ensuring transparency and full traceability of how sentiment influenced the execution.

```
// Market Sentiment Helper Functions
int GetHigherTFBias(double price, double maValue)
{
    double deviation = MathAbs(price - maValue) / maValue;
    if(price > maValue && deviation > ATRThreshold) return 1;   // Bullish bias
    else if(price < maValue && deviation > ATRThreshold) return -1; // Bearish bias
    else return 0; // Neutral/no bias
}

bool IsBullishStructure(ENUM_TIMEFRAMES tf, int lookback)
{
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback*2, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback*2, 1);
    if(swingHighIndex == -1 || swingLowIndex == -1) return false;

    // Bullish structure: Higher highs AND higher lows
    return (iHigh(_Symbol, tf, swingHighIndex) > iHigh(_Symbol, tf, swingHighIndex + lookback) &&
            iLow(_Symbol, tf, swingLowIndex) > iLow(_Symbol, tf, swingLowIndex + lookback));
}

bool IsBearishStructure(ENUM_TIMEFRAMES tf, int lookback)
{
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback*2, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback*2, 1);
    if(swingHighIndex == -1 || swingLowIndex == -1) return false;

    // Bearish structure: Lower highs AND lower lows
    return (iHigh(_Symbol, tf, swingHighIndex) < iHigh(_Symbol, tf, swingHighIndex + lookback) &&
            iLow(_Symbol, tf, swingLowIndex) < iLow(_Symbol, tf, swingLowIndex + lookback));
}

bool HasBreakout(ENUM_TIMEFRAMES tf, int lookback, int higherTFBias)
{
    int swingHighIndex = iHighest(_Symbol, tf, MODE_HIGH, lookback, 1);
    int swingLowIndex = iLowest(_Symbol, tf, MODE_LOW, lookback, 1);
    if(swingHighIndex == -1 || swingLowIndex == -1) return false;

    double swingHigh = iHigh(_Symbol, tf, swingHighIndex);
    double swingLow = iLow(_Symbol, tf, swingLowIndex);
    double price = iClose(_Symbol, tf, 0);

    // Breakout in direction of higher timeframe bias
    if(higherTFBias == 1) return (price > swingHigh); // Bullish breakout
    if(higherTFBias == -1) return (price < swingLow); // Bearish breakout

    return false;
}
```

These helper functions form the analytical backbone of the EA’s sentiment engine by evaluating price behavior across multiple timeframes and translating it into directional biases. GetHigherTFBias() compares the current price to the higher-timeframe moving average, using both direction and the size of the deviation to determine whether the market is trending bullish, bearish, or showing no clear movement. The IsBullishStructure() and IsBearishStructure() functions then study market structure by comparing recent swing highs and lows: higher-high/higher-low formations confirm bullish structure, while lower-high/lower-low patterns confirm bearish structure. These functions help the EA detect true structural shifts rather than reacting to noise.

The HasBreakout() function adds a dynamic layer by checking whether price has broken out of recent swing levels in alignment with the higher-timeframe bias. For example, in a bullish environment, the EA only considers breakouts meaningful if price exceeds recent swing highs, signaling momentum continuation. When these three components—trend bias, structural direction, and breakout confirmation—are combined, they produce a robust multi-timeframe sentiment reading that the EA uses to guide strategy selection, filter trades, and intelligently adapt to changing market conditions.

```
int DetermineSentiment(int higherTFBias,
                      bool tf1Bullish, bool tf1Bearish, bool tf1Breakout,
                      bool tf2Bullish, bool tf2Bearish, bool tf2Breakout)
{
    // Strong Bullish: Higher TF bullish + both lower TFs bullish
    if(higherTFBias == 1 && tf1Bullish && tf2Bullish) return 1;

    // Strong Bearish: Higher TF bearish + both lower TFs bearish
    if(higherTFBias == -1 && tf1Bearish && tf2Bearish) return -1;

    // Risk-On: Higher TF bullish + breakout on either lower TF
    if(higherTFBias == 1 && (tf1Breakout || tf2Breakout)) return 2;

    // Risk-Off: Higher TF bearish + breakout on either lower TF
    if(higherTFBias == -1 && (tf1Breakout || tf2Breakout)) return -2;

    // Neutral: No clear bias or conflicting signals
    return 0;
}

// Main Execution Flow with Sentiment-Strategy Integration
void OnTick()
{
    Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    if(!IsNewBar()) return;

    // Clean up expired objects
    CleanupExpiredObjects();

    // Calculate current market sentiment
    CalculateMarketSentiment();

    // Select strategy based on sentiment
    string selectedStrategy = SelectTradingStrategy();

    // Execute selected strategies based on sentiment
    if(selectedStrategy == "ALL" || selectedStrategy == "BOS")
        DetectAndTradeBOS(); // Break of Structure strategy

    if(selectedStrategy == "ALL" || selectedStrategy == "OB")
        DetectAndTradeOrderBlocks(); // Order Blocks strategy

    if(selectedStrategy == "ALL" || selectedStrategy == "FVG")
        DetectAndTradeFVGs(); // Fair Value Gaps strategy

    // Display current status with sentiment info
    DisplayStatus();
}

// Display status with sentiment information
void DisplayStatus()
{
    int fvgCount = (detectedFVGs != NULL) ? detectedFVGs.Total() : 0;
    int bosCount = (detectedBOSZones != NULL) ? detectedBOSZones.Total() : 0;
    int obCount = (tradedOBs != NULL) ? tradedOBs.Total() : 0;
    if(currentOB != NULL) obCount++;

    string status = "SMC Strategy with Market Sentiment\n" +
                   "══════════════════════════════════\n" +
                   "Market Sentiment: " + currentSentimentText + "\n" +
                   "Active Strategy: " + currentStrategy + "\n" +
                   "══════════════════════════════════\n" +
                   "Patterns Detected:\n" +
                   "• Order Blocks: " + IntegerToString(obCount) + "\n" +
                   "• FVGs: " + IntegerToString(fvgCount) + "\n" +
                   "• BOS Zones: " + IntegerToString(bosCount) + "\n" +
                   "══════════════════════════════════\n" +
                   "Trading Status: " + (EnableTrading ? "ACTIVE" : "PAUSED");

    Comment(status);
}
```

The DetermineSentiment() function acts as the final decision layer of the sentiment engine, merging higher-timeframe trend direction with lower-timeframe structure and breakout confirmation. It categorizes sentiment into five distinct states—Strong Bullish, Strong Bearish, Risk-On, Risk-Off, and Neutral—based on how well the timeframes align with each other. When both lower timeframes agree with the higher-timeframe trend, the sentiment is marked as strong in that direction. When only breakout momentum aligns with the higher timeframe, it shifts to a risk-on or risk-off environment. Any conflicting or unclear conditions default to neutral, preventing the EA from making aggressive or misaligned decisions.

The OnTick() function orchestrates the EA’s main execution workflow, integrating market sentiment with strategy selection and trade execution. On every new bar, the EA performs object cleanup, recalculates sentiment, and chooses the most appropriate strategy—BOS, OB, FVG, or all—based on the current sentiment state. It then activates the corresponding detection and trading modules, enabling the system to fluidly adapt its behavior to market conditions in real time. This creates an engine that does not rely on a single trading technique but intelligently rotates strategies depending on whether the market is trending, consolidating, or breaking out.

Finally, the DisplayStatus() function enhances transparency by showing the trader a structured real-time summary of the EA’s internal state, including detected SMC patterns, the current sentiment level, the active strategy, and whether trading is enabled. This visual feedback ensures the user clearly understands why the EA is making certain decisions, building trust while also making the system easier to monitor and debug.

### **Back Test Results**

The following backtesting was evaluated on the H1 timeframe across roughly a 2-month testing window (01 October 2025 to 01 December 2025), with the following settings:

![](https://c.mql5.com/2/185/SMC_sent_Inps.png)

Now here is the equity curve and the backtest results:

![](https://c.mql5.com/2/185/SMC_sent__eq_curv.png)

![](https://c.mql5.com/2/185/SMC_sent_BT.png)

### Conclusion

In summary, we developed an intelligent SMC trading EA by integrating three core Smart Money Concept strategies—Order Blocks, Fair Value Gaps, and Break of Structure—with real-time multi-timeframe market sentiment analysis. We created a dynamic system that automatically selects the most appropriate strategy based on prevailing market conditions: Order Blocks for neutral/ranging markets, Break of Structure for strong trending environments, and Fair Value Gaps for transitional breakout scenarios. The EA features comprehensive visual drawing with trade-based object management, where graphical elements are drawn only from their origin to trade execution time and automatically cleaned up when trades close, ensuring a clutter-free chart. We also enhanced the Break of Structure to clearly show the direction of structural breaks, making market movements immediately visible to traders.

In conclusion, this sophisticated EA implementation provides traders with a context-aware, adaptive trading system that significantly enhances decision-making by aligning SMC strategies with real-time market conditions. By eliminating the guesswork of strategy selection and automating trade execution within proper market contexts, traders can maintain discipline while capitalizing on high-probability setups. The comprehensive visual feedback system helps traders understand market dynamics in real-time, while the automatic cleanup ensures optimal chart clarity for ongoing analysis. This integrated approach increases trading efficiency and provides an educational framework for understanding how different SMC strategies perform under various market conditions, ultimately helping traders develop better market intuition and improved risk management practices.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/20414.zip "Download all attachments in the single ZIP archive")

[SMC\_Sent.mq5](https://www.mql5.com/en/articles/download/20414/SMC_Sent.mq5 "Download SMC_Sent.mq5")(109.63 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/501643)**
(16)


![chhotu choudhary](https://c.mql5.com/avatar/2024/10/67168876-EA1D.jpg)

**[chhotu choudhary](https://www.mql5.com/en/users/chhotuchoudhary)**
\|
13 Dec 2025 at 17:40

i want this ea can you demo first .and what is the price of this ea


![Juvenille Emperor Limited](https://c.mql5.com/avatar/2019/4/5CB0FE21-E283.jpg)

**[Eleni Anna Branou](https://www.mql5.com/en/users/eleanna74)**
\|
13 Dec 2025 at 19:31

**chhotu choudhary [#](https://www.mql5.com/en/forum/501643/page2#comment_58722682):**

i want this ea can you demo first .and what is the price of this ea

Read the article in the first post of this topic please.


![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
17 Dec 2025 at 12:34

**YDILLC [#](https://www.mql5.com/en/forum/501643#comment_58700827):**

My friend, are you sure those are the same backtest inputs? Something might be off but I entered those same exact settings & it's blowing the account.

Results will differ, sometimes it's because of your broker or your account type.


![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
17 Dec 2025 at 12:35

**Ariston Hutauruk [#](https://www.mql5.com/en/forum/501643#comment_58704588):**

Thank you my friends, your code very helpfully for me

You are welcome.


![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
17 Dec 2025 at 12:37

**GL99K [#](https://www.mql5.com/en/forum/501643#comment_58706070):**

It worked perfectly though our profit differ a little. The [attached image](https://www.mql5.com/en/articles/24#insert-image "Article: MQL5.community - User Memo ") is a screenshot of my output.

Much respect to **Hlomohang John Borotho** for his insightful and comprehensive work. I believe he must have a good backgroung in math, physics and computer science. God bless.

Thank you, you're welcome.


![Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://c.mql5.com/2/185/20550-codex-pipelines-from-python-logo.png)[Codex Pipelines: From Python to MQL5 for Indicator Selection — A Multi-Quarter Analysis of the FXI ETF](https://www.mql5.com/en/articles/20550)

We continue our look at how MetaTrader can be used outside its forex trading ‘comfort-zone’ by looking at another tradable asset in the form of the FXI ETF. Unlike in the last article where we tried to do ‘too-much’ by delving into not just indicator selection, but also considering indicator pattern combinations, for this article we will swim slightly upstream by focusing more on indicator selection. Our end product for this is intended as a form of pipeline that can help recommend indicators for various assets, provided we have a reasonable amount of their price history.

![Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://c.mql5.com/2/185/20514-overcoming-the-limitation-of-logo.png)[Overcoming The Limitation of Machine Learning (Part 9): Correlation-Based Feature Learning in Self-Supervised Finance](https://www.mql5.com/en/articles/20514)

Self-supervised learning is a powerful paradigm of statistical learning that searches for supervisory signals generated from the observations themselves. This approach reframes challenging unsupervised learning problems into more familiar supervised ones. This technology has overlooked applications for our objective as a community of algorithmic traders. Our discussion, therefore, aims to give the reader an approachable bridge into the open research area of self-supervised learning and offers practical applications that provide robust and reliable statistical models of financial markets without overfitting to small datasets.

![Automated Risk Management for Passing Prop Firm Challenges](https://c.mql5.com/2/185/19655-automated-risk-management-for-logo.png)[Automated Risk Management for Passing Prop Firm Challenges](https://www.mql5.com/en/articles/19655)

This article explains the design of a prop-firm Expert Advisor for GOLD, featuring breakout filters, multi-timeframe analysis, robust risk management, and strict drawdown protection. The EA helps traders pass prop-firm challenges by avoiding rule breaches and stabilizing trade execution under volatile market conditions.

![Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://c.mql5.com/2/122/Developing_a_Multicurrency_Advisor_Part_24___LOGO.png)[Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)

In this article, we will look at how to connect a new strategy to the auto optimization system we have created. Let's see what kind of EAs we need to create and whether it will be possible to do without changing the EA library files or minimize the necessary changes.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=caqchgxhyrsedpielnhpqvksbbdvdpob&ssn=1769251463859879656&ssn_dr=0&ssn_sr=0&fv_date=1769251463&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F20414&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Adaptive%20Smart%20Money%20Architecture%20(ASMA)%3A%20Merging%20SMC%20Logic%20With%20Market%20Sentiment%20for%20Dynamic%20Strategy%20Switching%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925146335490409&fz_uniq=5083077379800175750&sv=2552)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).