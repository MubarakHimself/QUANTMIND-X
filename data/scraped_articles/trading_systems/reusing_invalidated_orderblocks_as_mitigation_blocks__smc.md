---
title: Reusing Invalidated Orderblocks As Mitigation Blocks (SMC)
url: https://www.mql5.com/en/articles/19619
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:45:33.110651
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/19619&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083089027751482571)

MetaTrader 5 / Examples


### Table of contents:

1. [Introduction](https://www.mql5.com/en/articles/19619#Introduction)
2. [Understanding mitigation orderblock](https://www.mql5.com/en/articles/19619#Understandingmitigationorderblock)
3. [Getting started](https://www.mql5.com/en/articles/19619#Gettingstarted)
4. [Backtest results](https://www.mql5.com/en/articles/19619#Backtestresults)
5. [Conclusion](https://www.mql5.com/en/articles/19619#Conclusion)

### Introduction

In Smart Money Concepts (SMC) trading, orderblocks represent key areas where institutional traders accumulate or distribute positions before driving price in a particular direction. However, not all orderblocks remain valid—some are violated as market conditions evolve. When an orderblock fails to hold, it doesn’t necessarily lose its significance; instead, it transforms into a potential mitigation block. This concept focuses on how price often retraces back to these invalidated zones to “mitigate” remaining orders before continuing in the direction of the new trend, offering traders deeper insight into institutional intent and market structure behavior.

The idea of reusing invalidated orderblocks as mitigation blocks allows traders to recognize where smart money might re-engage with the market after shifting bias. By studying how price interacts with these areas, traders can anticipate high-probability entries that align with the new directional flow. Understanding this transformation from orderblock to mitigation block enhances precision in timing entries, and managing risk.

### Understanding Mitigation Orderblock:

![](https://c.mql5.com/2/174/rally.png)

Normally, when the price rallies upward, it tends to leave behind a form of bullish orderblock near or at the swing low, representing the last point where buyers stepped in before driving the price higher. As the rally continues, it often approaches a perceived resistance level—which could be an old high, previous orderblock, or breaker structure. At this level, we look for clear signs that willing sellers are present, such as rejection wicks or bearish candles indicating a shift in order flow. If the market begins to reprice downward and later rallies back up into that same resistance zone, the next step is to observe whether the market shows willingness to break down from that level, signaling that selling pressure is being respected and that a potential bearish move could follow.

![](https://c.mql5.com/2/174/M_patt.png)

As the market begins to decline, it commonly forms an ‘M’ pattern, representing a failure swing that signals exhaustion of bullish momentum near a resistance level. The second peak of the ‘M’ typically fails to create a new high, showing that buyers are losing strength while sellers begin to step in. This price behavior is then confirmed by a break in market structure, often referred to as a market structure shift, when the market breaks below a key low. This shift provides confirmation that smart money or institutions indeed positioning to drive prices lower, transitioning the market from a bullish to a bearish environment.

![](https://c.mql5.com/2/174/MRet_2.png)

In this scenario, we focus on the range from the short-term low up to the short-term high, identifying areas where buying activity previously occurred. The short-term rally in price draws attention to a specific institutional reference point, known as the mitigation block. This is where price retraces back into or within a bearish mitigation block, providing an opportunity for smart money to re-enter in alignment with the prevailing bearish bias. The reason this zone becomes significant is that the original bullish orderblock failed to hold—it was not respected due to strong selling pressure, indicating that institutional order flow has shifted from accumulation to distribution, and that price is likely to continue lower after mitigating that block.

### Getting Started

```
//+------------------------------------------------------------------+
//|                                            OB_&_MitigationOB.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Trade/Trade.mqh>
#include <Arrays\ArrayObj.mqh>

CTrade trade;
CArrayObj obList;

#define BullOB clrLime
#define BearOB clrRed
#define ViolatedBullOB clrBlue
#define ViolatedBearOB clrMagenta

//+------------------------------------------------------------------+
//|                           Input parameters                       |
//+------------------------------------------------------------------+
input double Lots = 0.11;
input int takeProfit = 3000;
input double stopLoss = 2000;
input int Time1Hstrt = 1;
input int Time1Hend = 10;
```

We start off by defining the foundation and structure of the Expert Advisor (EA), which is designed to identify and manage orderblocks and their mitigation counterparts within the Smart Money Concepts framework. Then, two important library inclusions are made—Trade/Trade.mqh to handle trading operations like order execution, and Arrays/ArrayObj.mqh to manage dynamic collections of objects, which in this case will be used to store orderblock data efficiently. The declaration of CTrade trade initializes the trading handler, while CArrayObj obList creates a container for storing all detected orderblocks during runtime.

Next, we define color constants for visual clarity on the chart—green (clrLime) for bullish orderblocks, red (clrRed) for bearish ones, and distinct colors like blue and magenta to represent violated or mitigated orderblocks. These visual cues are crucial for distinguishing valid from invalidated zones. The input parameters that follow allow traders to customize the EA’s behavior: position sizing (Lots), profit and loss targets (takeProfit and stopLoss), and time-based filters (Time1Hstrt and Time1Hend) to limit trade execution to specific market sessions. This initial setup effectively establishes the visual, logical, and operational groundwork upon which the rest of the EA will detect, classify, and reuse orderblocks as mitigation zones.

```
//+------------------------------------------------------------------+
//|                         OrderBlock Class                         |
//+------------------------------------------------------------------+
class COrderBlock : public CObject
{
public:
   int direction;
   datetime time;
   double high;
   double low;
   bool violated;
   datetime violatedTime;
   bool traded;
   bool inTrade;
   string identifier;
   string tradeComment;

   COrderBlock()
   {
      traded = false;
      violated = false;
      inTrade = false;
      tradeComment = "";
   }

   void UpdateIdentifier()
   {
      identifier = IntegerToString(time) + IntegerToString(direction) + IntegerToString(violated);
   }

   void draw(datetime tmS, datetime tmE, color clr)
   {
      UpdateIdentifier();
      string objOB = "OB_REC_" + identifier;
      if(ObjectFind(0, objOB) == -1)
      {
         ObjectCreate(0, objOB, OBJ_RECTANGLE, 0, time, low, tmS, high);
         ObjectSetInteger(0, objOB, OBJPROP_FILL, true);
         ObjectSetInteger(0, objOB, OBJPROP_COLOR, clr);
         ObjectSetInteger(0, objOB, OBJPROP_BACK, true);
      }

      string objtrade = "OB_TRADE_" + identifier;
      if(ObjectFind(0, objtrade) == -1)
      {
         ObjectCreate(0, objtrade, OBJ_RECTANGLE, 0, tmS, high, tmE, low);
         ObjectSetInteger(0, objtrade, OBJPROP_FILL, true);
         ObjectSetInteger(0, objtrade, OBJPROP_COLOR, clr);
         ObjectSetInteger(0, objtrade, OBJPROP_BACK, true);
      }
   }

   void RemoveObjects()
   {
      UpdateIdentifier();
      string objOB = "OB_REC_" + identifier;
      string objtrade = "OB_TRADE_" + identifier;
      if(ObjectFind(0, objOB) >= 0) ObjectDelete(0, objOB);
      if(ObjectFind(0, objtrade) >= 0) ObjectDelete(0, objtrade);
   }
};

//+------------------------------------------------------------------+
//|                           Global variables                       |
//+------------------------------------------------------------------+
COrderBlock *activeMitigationOB = NULL;
int activeMitigationDirection = 0;
```

We then define the COrderBlock class, which serves as the core data structure for representing and managing orderblocks within the EA. This class inherits from CObject, allowing instances to be stored and manipulated within object arrays (CArrayObj). Each orderblock stores critical attributes such as its direction (bullish or bearish), time, high and low price levels, and states like violated, traded, and inTrade. Additional identifiers like identifier and tradeComment help uniquely reference each orderblock on the chart and track its related trade logic. The constructor initializes default states to ensure that newly detected orderblocks start as untraded and unviolated, establishing a clean baseline before chart or trade interactions occur.

The class also includes several powerful methods that handle both visualization and lifecycle management. The UpdateIdentifier() method generates a unique string based on time, direction, and violation state—essential for differentiating between multiple orderblocks. The draw() function is responsible for plotting the orderblock zones and their associated trade regions as colored rectangles on the chart, ensuring each block is visually distinct and non-overlapping. Meanwhile, RemoveObjects() cleans up these graphical objects when they are no longer valid, maintaining chart clarity and performance. Finally, two global variables—activeMitigationOB and activeMitigationDirection—are declared to track the currently active mitigation orderblock and its directional bias, forming a bridge between orderblock detection, violation, and trading logic later in the EA.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(76543);
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   for(int i = obList.Total()-1; i >= 0; i--)
   {
      COrderBlock* ob = obList.At(i);
      ob.RemoveObjects();
      delete ob;
   }
   obList.Clear();
}
```

In this section, the initialization and deinitialization processes of the Expert Advisor are defined to ensure proper management of trading and graphical resources. The OnInit() function assigns a unique magic number to distinguish trades placed by the system, allowing accurate tracking and management of open positions. Conversely, the OnDeinit() function ensures clean termination by removing all graphical order block objects from the chart and freeing allocated memory for each order block instance. This structure prevents data leaks or graphical clutter, maintaining a stable and efficient trading environment during restarts or shutdowns of the Expert Advisor.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(isNewBar())
   {
      CheckForClosedTradesAndCleanup();
      CheckForViolations();
      getOrderB();
      CheckForTradeEntries();
   }
}

//+------------------------------------------------------------------+
//| Helpers to find bars and swings safely                           |
//+------------------------------------------------------------------+
int BarIndexFromTimeSafely(datetime t)
{
   int idx = iBarShift(_Symbol, _Period, t, true);
   if(idx < 0) idx = Bars(_Symbol, _Period) - 1;
   return(idx);
}

int FindHighestAfter(datetime t, int lookbackBars)
{
   int obIdx = BarIndexFromTimeSafely(t);
   int start = MathMax(1, obIdx - lookbackBars);
   int count = obIdx - start;
   if(count <= 0) return(-1);
   int idx = iHighest(_Symbol, _Period, MODE_HIGH, count, start);
   return(idx);
}

int FindLowestAfter(datetime t, int lookbackBars)
{
   int obIdx = BarIndexFromTimeSafely(t);
   int start = obIdx + 1;
   int maxPossible = Bars(_Symbol, _Period) - start;
   int count = MathMin(lookbackBars, maxPossible);
   if(count <= 0) return(-1);
   int idx = iLowest(_Symbol, _Period, MODE_LOW, count, start);
   return(idx);
}
```

The OnTick() function serves as the main execution loop of the Expert Advisor, handling real-time decision-making on each market tick. It primarily executes its logic once a new bar is confirmed using the isNewBar() condition to prevent redundant operations within the same candle. On every new bar, it performs essential tasks such as cleaning up closed trades, checking for violated order blocks, identifying new potential order blocks, and scanning for valid trade entry opportunities. This structured approach ensures efficient resource use and maintains the strategy’s logical flow without overloading the system.

Supporting these main operations, the helper functions—BarIndexFromTimeSafely(), FindHighestAfter(), and FindLowestAfter()—are designed to manage data retrieval and swing analysis reliably. BarIndexFromTimeSafely() ensures the program can safely find the correct bar index from a given time, even under irregular market conditions or missing data. The FindHighestAfter() and FindLowestAfter() functions identify key swing highs and lows within a defined lookback period, critical for detecting market structure and validating order block behavior. Together, these functions form the foundation for precise and adaptive order block analysis.

```
//+------------------------------------------------------------------+
//| Check for violations of existing orderblocks                     |
//+------------------------------------------------------------------+
void CheckForViolations()
{
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i = 0; i < obList.Total(); i++)
   {
      COrderBlock* ob = obList.At(i);
      if(ob.violated || ob.traded) continue;

      if(ob.direction == 1 && currentBid < ob.low)
      {
         if(activeMitigationOB != NULL)
         {
            activeMitigationOB = NULL;
            activeMitigationDirection = 0;
         }

         ob.violated = true;
         ob.violatedTime = iTime(_Symbol, PERIOD_CURRENT, 0);
         ob.traded = false;
         ob.UpdateIdentifier();
         activeMitigationOB = ob;
         activeMitigationDirection = -1;
         Print("Bullish OB violated and becomes BEARISH mitigation: ", TimeToString(ob.time));
      }
      else if(ob.direction == -1 && currentAsk > ob.high)
      {
         if(activeMitigationOB != NULL)
         {
            activeMitigationOB = NULL;
            activeMitigationDirection = 0;
         }

         ob.violated = true;
         ob.violatedTime = iTime(_Symbol, PERIOD_CURRENT, 0);
         ob.traded = false;
         ob.UpdateIdentifier();
         activeMitigationOB = ob;
         activeMitigationDirection = 1;
         Print("Bearish OB violated and becomes BULLISH mitigation: ", TimeToString(ob.time));
      }
   }
}
```

The CheckForViolations() function is responsible for continuously monitoring all existing order blocks to determine whether any have been violated by current market price action. It retrieves the current bid and ask prices, then iterates through every stored order block while skipping those already marked as traded or violated.

For bullish order blocks, a violation occurs when the bid price drops below the block’s low, while for bearish order blocks, a violation happens when the ask price rises above the block’s high. Once a violation is detected, the system updates the order block’s status, marks the violation time, and resets any active mitigation block references before reassigning the violated block as the new active mitigation order block with an opposite directional bias.

This dynamic transition from one market state to another allows the EA to adapt to changing price behavior and treat previously strong zones as new mitigation areas, effectively refining the strategy’s responsiveness to order flow shifts.

```
//+------------------------------------------------------------------+
//| Check for trade entries in orderblocks                           |
//+------------------------------------------------------------------+
void CheckForTradeEntries()
{
   double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double currentAsk = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   for(int i = 0; i < obList.Total(); i++)
   {
      COrderBlock* ob = obList.At(i);
      if(ob.traded) continue;

      bool priceInZone = false;
      bool isBullishTrade = false;
      bool isMitigation = ob.violated;

      if(!ob.violated)
      {
         if(ob.direction == 1 && currentAsk < ob.high && currentAsk > ob.low)
         {
            priceInZone = true;
            isBullishTrade = true;
         }
         else if(ob.direction == -1 && currentBid > ob.low && currentBid < ob.high)
         {
            priceInZone = true;
            isBullishTrade = false;
         }
      }
      else
      {
         if(ob.direction == 1 && currentBid > ob.low && currentBid < ob.high)
         {
            priceInZone = true;
            isBullishTrade = false;
         }
         else if(ob.direction == -1 && currentAsk < ob.high && currentAsk > ob.low)
         {
            priceInZone = true;
            isBullishTrade = true;
         }
      }

      if(priceInZone)
      {
         if(isMitigation)
         {
            if(activeMitigationOB == NULL || activeMitigationOB != ob)
            {
               continue;
            }
         }

         if(isBullishTrade)
         {
            ExecuteBuyTrade(ob);
         }
         else
         {
            ExecuteSellTrade(ob);
         }
         ob.traded = true;
         break;
      }
   }
}
```

The CheckForTradeEntries() function handles the core logic that determines when and where the Expert Advisor should execute trades based on price interaction with active order blocks or mitigation blocks. It begins by fetching the current bid and ask prices and iterating through each stored order block while skipping those already used for trading. The logic then checks whether the current price is trading inside a valid order block zone and identifies the trade direction based on the block’s nature—bullish or bearish. For unviolated order blocks, bullish trades are triggered when the ask price dips into the bullish zone, while bearish trades occur when the bid price rallies into the bearish zone.

When dealing with violated order blocks (now acting as mitigation blocks), the trade logic is inverted to reflect the opposite reaction in market structure. The system ensures that only the currently active mitigation block can trigger trades, avoiding unnecessary reentries or conflicts. Once conditions are met, it executes either a buy or sell trade using the respective order block parameters and marks the block as “traded” to prevent duplicate positions. This two-layer logic—handling both valid and mitigated order blocks—provides a dynamic and adaptive structure for smart money trading, allowing the algorithm to respond intelligently as market conditions evolve.

```
//+------------------------------------------------------------------+
//| Execute a buy trade                                              |
//+------------------------------------------------------------------+
void ExecuteBuyTrade(COrderBlock* ob)
{
   double entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   int swingHighIdx = FindHighestAfter(ob.time, 20);
   double tp = (swingHighIdx>0) ? getHigh(swingHighIdx) : entry + takeProfit * _Point;
   double sl = NormalizeDouble(ob.low - stopLoss * _Point, _Digits);

   string comment = "Bull_OB_" + ob.identifier;
   ob.tradeComment = comment;

   if(trade.Buy(Lots, _Symbol, entry, sl, tp, comment))
   {
      color obColor = ob.violated ? ViolatedBullOB : BullOB;
      ob.draw(ob.time, iTime(_Symbol, PERIOD_CURRENT, 0), obColor);
      ob.inTrade = true;
      Print("Buy trade executed. Type: ", ob.violated ? "Mitigation" : "Regular",
            " OB Time: ", TimeToString(ob.time), " Entry: ", entry, " SL: ", sl, " TP: ", tp);
   }
}

//+------------------------------------------------------------------+
//| Execute a sell trade                                             |
//+------------------------------------------------------------------+
void ExecuteSellTrade(COrderBlock* ob)
{
   double entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int swingLowIdx = FindLowestAfter(ob.time, 20);
   double tp = (swingLowIdx>0) ? getLow(swingLowIdx) : entry - takeProfit * _Point;
   double sl = NormalizeDouble(ob.high + stopLoss * _Point, _Digits);

   string comment = "Bear_OB_" + ob.identifier;
   ob.tradeComment = comment;

   if(trade.Sell(Lots, _Symbol, entry, sl, tp, comment))
   {
      color obColor = ob.violated ? ViolatedBearOB : BearOB;
      ob.draw(ob.time, iTime(_Symbol, PERIOD_CURRENT, 0), obColor);
      ob.inTrade = true;
      Print("Sell trade executed. Type: ", ob.violated ? "Mitigation" : "Regular",
            " OB Time: ", TimeToString(ob.time), " Entry: ", entry, " SL: ", sl, " TP: ", tp);
   }
}
```

The ExecuteBuyTrade() and ExecuteSellTrade() functions handle the precise execution of trades once an order block or mitigation block meets the entry conditions. In a buy scenario, the system uses the current ask price as the entry level and calculates the take-profit target based on the next detected swing high. If no valid swing high is found, it defaults to a fixed pip distance defined by the takeProfit input. The stop-loss is positioned just below the order block’s low, ensuring a risk-managed setup. The function then labels the trade with a descriptive comment for tracking purposes and visually updates the chart with the correct color—green for a valid bullish block or blue for a mitigated one. This structure allows clear differentiation between regular and mitigation-based trades in both logic and chart visualization.

The sell trade logic follows a mirrored process, using the current bid price as the entry level, with the take-profit determined by the nearest swing low or a fixed fallback value. The stop-loss is placed just above the order block’s high, maintaining consistent risk-to-reward symmetry with buy setups. A clear trade comment is generated for identification, and the corresponding bearish or mitigated bearish color (red or magenta) is applied on the chart to highlight active trade zones. Both functions conclude by setting the inTrade flag to true, ensuring that each order block triggers only one position. This modular design effectively separates buy and sell execution logic, enabling structured, visually tracked, and rule-based trading behavior that aligns with smart money concepts.

```
//+------------------------------------------------------------------+
//| Detect new orderblocks                                           |
//+------------------------------------------------------------------+
void getOrderB()
{
   MqlDateTime structTime;
   TimeCurrent(structTime);
   static int prevDay = 0;

   if(structTime.hour >= Time1Hstrt && structTime.hour < Time1Hend)
   {
      if(prevDay != structTime.day)
      {
         prevDay = structTime.day;

         for(int i = 3; i < 100; i++)
         {
            if(i + 3 >= Bars(_Symbol, _Period)) continue;

            if(getOpen(i+2) > getClose(i+2) &&
               getClose(i+1) > getOpen(i+1) &&
               getClose(i) > getOpen(i))
            {
               COrderBlock* ob = new COrderBlock();
               ob.direction = 1;
               ob.time = getTime(i+2);
               ob.high = getHigh(i+2);
               ob.low = getLow(i+2);
               ob.violated = false;
               ob.traded = false;
               ob.UpdateIdentifier();
               obList.Add(ob);
               Print("Bullish Order Block detected at: ", TimeToString(ob.time));
            }
            else if(getOpen(i+2) < getClose(i+2) &&
                    getClose(i+1) < getOpen(i+1) &&
                    getClose(i) < getOpen(i))
            {
               COrderBlock* ob = new COrderBlock();
               ob.direction = -1;
               ob.time = getTime(i+2);
               ob.high = getHigh(i+2);
               ob.low = getLow(i+2);
               ob.violated = false;
               ob.traded = false;
               ob.UpdateIdentifier();
               obList.Add(ob);
               Print("Bearish Order Block detected at: ", TimeToString(ob.time));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check for closed trades and cleanup objects after closure        |
//+------------------------------------------------------------------+
void CheckForClosedTradesAndCleanup()
{
   for(int i = obList.Total()-1; i >= 0; i--)
   {
      COrderBlock* ob = obList.At(i);
      if(ob.inTrade)
      {
         if(!PositionWithCommentExists(ob.tradeComment))
         {
            Print("Trade closed for OB: ", ob.identifier, " -> cleaning up drawings");
            ob.RemoveObjects();
            ob.inTrade = false;
            ob.traded = false;
            ob.tradeComment = "";

            if(activeMitigationOB == ob)
            {
               activeMitigationOB = NULL;
               activeMitigationDirection = 0;
               Print("Active mitigation OB cleared after trade closure");
            }

            obList.Delete(i);
            delete ob;
         }
      }
   }
}
```

The getOrderB() function is responsible for detecting fresh order blocks based on a simple three-candle pattern logic. It only scans within a specific time window defined by the user inputs Time1Hstrt and Time1Hend, ensuring that detection happens during controlled trading hours. The function first checks for bullish order block conditions—where a bearish candle is followed by two consecutive bullish candles—indicating potential accumulation and a shift in market sentiment. When this pattern is identified, a new COrderBlock object is created, its attributes (such as direction, time, high, and low) are initialized, and it is added to the global list.

Similarly, a bearish order block is detected when a bullish candle is followed by two consecutive bearish candles, signaling possible distribution and upcoming price declines. This modular and time-sensitive detection ensures only significant blocks are registered per day, reducing clutter and focusing on actionable zones.

The CheckForClosedTradesAndCleanup() function complements the detection process by maintaining chart cleanliness and system integrity. Once a trade associated with an order block is closed, the function verifies its status by checking for the existence of a position with the block’s trade comment. If the trade no longer exists, all related visual elements (rectangles and highlights) are removed from the chart to prevent confusion with active setups. It also resets the order block’s internal state, clears the mitigation reference if applicable, and deletes the order block object from the global list to free memory. This cleanup cycle keeps the system dynamic and efficient, ensuring that only relevant and active order blocks remain tracked, while outdated ones are safely removed from both memory and the chart environment.

```
//+------------------------------------------------------------------+
//| Check if position with specific comment exists                   |
//+------------------------------------------------------------------+
bool PositionWithCommentExists(string comment)
{
   if(StringLen(comment) == 0) return(false);
   for(int i = PositionsTotal()-1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0 && PositionSelectByTicket(ticket))
      {
         string c = PositionGetString(POSITION_COMMENT);
         if(c == comment) return(true);
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| Helper functions                                                 |
//+------------------------------------------------------------------+
bool isNewBar()
{
   static datetime lastBar;
   datetime currentBar = iTime(_Symbol, _Period, 0);
   if(lastBar != currentBar)
   {
      lastBar = currentBar;
      return true;
   }
   return false;
}

double getHigh(int index) { return iHigh(_Symbol, _Period, index); }
double getLow(int index) { return iLow(_Symbol, _Period, index); }
double getOpen(int index) { return iOpen(_Symbol, _Period, index); }
double getClose(int index) { return iClose(_Symbol, _Period, index); }
datetime getTime(int index) { return iTime(_Symbol, _Period, index); }
```

The PositionWithCommentExists() function plays a crucial role in maintaining trade state integrity by checking whether a position with a specific identifying comment still exists. Since each order block trade is tagged with a unique comment, this function loops through all open positions, retrieves their comments, and compares them to the one provided as input. If a match is found, it returns true, indicating that the trade is still active; otherwise, it returns false. This verification process is essential for functions such as trade cleanup and mitigation logic, ensuring that actions like object removal or state resets only occur after the associated trade has truly closed. By using this method, the EA avoids double processing and maintains an accurate reflection of live versus completed trades.

The helper functions that follow serve as supportive utilities for data retrieval and bar management. The isNewBar() function identifies when a new bar has formed, allowing key operations like order block detection or violation checks to execute only once per bar, thus preventing redundant processing during every tick. Meanwhile, the other helper functions—getHigh(), getLow(), getOpen(), getClose(), and getTime()—act as convenient abstractions for fetching specific candle attributes from the chart. This simplifies code readability and ensures consistency in data handling throughout the Expert Advisor. Collectively, these small but essential tools provide the foundation that enables more complex routines, like order block identification and trade execution, to function smoothly and efficiently.

### **Backtest Results**

The back-testing was evaluated on the 1H timeframe across roughly a 2-month testing window (02 June 2025 to 29 July 2025), with the default settings.

![](https://c.mql5.com/2/174/mitt.png)

![](https://c.mql5.com/2/174/mitBT1.png)

### Conclusion

In summary, this Expert Advisor (EA) integrates smart money concepts (SMC) such as order blocks and mitigation order blocks into a structured, rule-based trading system. It begins by identifying valid bullish and bearish order blocks from price structure and candle formations, then monitors for violations to transform invalidated order blocks into mitigation zones. Through its detection, validation, and execution mechanisms, the EA manages trades dynamically—executing buy or sell positions when price re-enters key levels. It also handles all visual representations of order blocks, drawing and removing rectangles as trades open and close, ensuring the chart always reflects the most relevant institutional zones.

In conclusion, this system represents a methodical approach to interpreting market structure and institutional price behavior within MQL5. By automating the recognition of order blocks, their transitions into mitigation zones, and subsequent trade management, the EA captures the essence of how smart money reuses invalidated zones for liquidity and continuation setups. The inclusion of cleanup routines, helper functions, and strict validation conditions enhances its reliability and realism. Overall, this project bridges technical execution with conceptual market structure trading, providing a complete, self-managing framework that mirrors how professional traders interpret and act on order flow dynamics.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19619.zip "Download all attachments in the single ZIP archive")

[OB\_t\_MitigationOB.mq5](https://www.mql5.com/en/articles/download/19619/OB_t_MitigationOB.mq5 "Download OB_t_MitigationOB.mq5")(14.75 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Developing Market Memory Zones Indicator: Where Price Is Likely To Return](https://www.mql5.com/en/articles/20973)
- [Adaptive Smart Money Architecture (ASMA): Merging SMC Logic With Market Sentiment for Dynamic Strategy Switching](https://www.mql5.com/en/articles/20414)
- [Fortified Profit Architecture: Multi-Layered Account Protection](https://www.mql5.com/en/articles/20449)
- [Analytical Volume Profile Trading (AVPT): Liquidity Architecture, Market Memory, and Algorithmic Execution](https://www.mql5.com/en/articles/20327)
- [Automating Black-Scholes Greeks: Advanced Scalping and Microstructure Trading](https://www.mql5.com/en/articles/20287)
- [Integrating MQL5 with Data Processing Packages (Part 6): Merging Market Feedback with Model Adaptation](https://www.mql5.com/en/articles/20235)
- [Formulating Dynamic Multi-Pair EA (Part 5): Scalping vs Swing Trading Approaches](https://www.mql5.com/en/articles/19989)

**[Go to discussion](https://www.mql5.com/en/forum/497226)**

![From Novice to Expert: Market Periods Synchronizer](https://c.mql5.com/2/174/19841-from-novice-to-expert-market-logo.png)[From Novice to Expert: Market Periods Synchronizer](https://www.mql5.com/en/articles/19841)

In this discussion, we introduce a Higher-to-Lower Timeframe Synchronizer tool designed to solve the problem of analyzing market patterns that span across higher timeframe periods. The built-in period markers in MetaTrader 5 are often limited, rigid, and not easily customizable for non-standard timeframes. Our solution leverages the MQL5 language to develop an indicator that provides a dynamic and visual way to align higher timeframe structures within lower timeframe charts. This tool can be highly valuable for detailed market analysis. To learn more about its features and implementation, I invite you to join the discussion.

![Market Simulation (Part 03): A Matter of Performance](https://c.mql5.com/2/110/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 03): A Matter of Performance](https://www.mql5.com/en/articles/12580)

Often we have to take a step back and then move forward. In this article, we will show all the changes necessary to ensure that the Mouse and Chart Trade indicators do not break. As a bonus, we'll also cover other changes that have occurred in other header files that will be widely used in the future.

![Introduction to MQL5 (Part 22): Building an Expert Advisor for the 5-0 Harmonic Pattern](https://c.mql5.com/2/174/19856-introduction-to-mql5-part-22-logo__1.png)[Introduction to MQL5 (Part 22): Building an Expert Advisor for the 5-0 Harmonic Pattern](https://www.mql5.com/en/articles/19856)

This article explains how to detect and trade the 5-0 harmonic pattern in MQL5, validate it using Fibonacci levels, and display it on the chart.

![How to publish code to CodeBase: A practical guide](https://c.mql5.com/2/173/19441-kak-opublikovat-kod-v-codebase-logo.png)[How to publish code to CodeBase: A practical guide](https://www.mql5.com/en/articles/19441)

In this article, we will use real-life examples to illustrate posting various types of terminal programs in the MQL5 source code base.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/19619&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083089027751482571)

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