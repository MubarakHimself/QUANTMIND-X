---
title: Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG
url: https://www.mql5.com/en/articles/16340
categories: Trading Systems, Expert Advisors
relevance_score: 0
scraped_at: 2026-01-24T13:46:31.064009
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/16340&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083099967033185541)

MetaTrader 5 / Examples


### Introduction

Sometimes traders face the challenge of inconsistency in their strategies, often hopping between indicators, setups, and methods without a solid framework to guide them. This lack of structure can result in confusion, missed opportunities, and mounting frustration as market conditions shift and traditional tools struggle to keep up. In this topic, we explore the benefits of a single EA that unifies multiple Smart Money Concept (SMC) strategies into one powerful solution.

Smart Money Concepts (SMC) provide a structured way to understand market behavior through Order Blocks (OB), Break of Structure (BOS), and Fair Value Gaps (FVG). By combining these into a single EA, traders can simplify their workflow, automate decision-making, and focus on the most powerful price-action signals. Whether using auto mode for seamless execution or selecting individual concepts, this approach reduces guesswork and makes trading more efficient, consistent, and adaptive to changing market conditions.

### Expert Logic

Order Block:

An order block (OB) represents the last bullish or bearish candle before a significant market move, often signaling where institutional traders have placed orders. When price revisits these zones, they tend to act as strong areas of support or resistance, providing high-probability trade setups. In this EA, order block detection is still enhanced by integrating the Fibonacci indicator for validation, ensuring that trades align with key retracement or extension levels. As discussed in the previous [article](https://www.mql5.com/en/articles/13396), combining OB with Fibonacci confirmation filters out weaker setups, giving traders a more reliable framework for entering and managing trades.

![](https://c.mql5.com/2/166/OB.png)

Fair Value Gap:

A Fair Value Gap (FVG) occurs when there is an imbalance in price movement, typically created by a strong impulsive move where one or more candles leave a gap between the wicks of preceding and following candles. This gap reflects inefficiency in the market where not all orders were matched, and price often returns to "fill" that space before continuing in its intended direction. By detecting and trading around FVGs, traders can anticipate potential retracements into these zones, offering precise entry opportunities with well-defined risk and reward.

![](https://c.mql5.com/2/166/FVG.png)

Break Of Structure:

A Break of Structure (BOS) happens when price breaks above a previous swing high or below a previous swing low, signaling a potential shift in market direction. In the previous [article](https://www.mql5.com/en/articles/15030), we focused on selling when a swing high was broken, capitalizing on bearish momentum. However, in this approach, the BOS logic is refined to align with the prevailing market trend meaning if a swing high is broken, we now look to buy, confirming bullish strength, while a break of a swing low signals selling opportunities. This shift ensures that trades follow the dominant market flow rather than countering it.

![](https://c.mql5.com/2/166/BOS.png)

System Architecture:

![](https://c.mql5.com/2/168/Sys_Arch-2.png)

### Getting Started

```
//+------------------------------------------------------------------+
//|                                                 SMC_ALL_IN_1.mq5 |
//|                        GIT under Copyright 2025, MetaQuotes Ltd. |
//|                     https://www.mql5.com/en/users/johnhlomohang/ |
//+------------------------------------------------------------------+
#property copyright "GIT under Copyright 2025, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/johnhlomohang/"
#property version   "1.01"
#property description "Unified SMC: FVG + Order Blocks + BOS. Detect + Draw + Trade."

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

CTrade         trade;
CPositionInfo  pos;

enum ENUM_STRATEGY
{
   STRAT_OB,         // Use Order Blocks Only
   STRAT_FVG,        // Use FVGs Only
   STRAT_BOS,        // Use Break of Structure Only
   STRAT_AUTO        // Auto (All SMC Concepts)
};
enum SWING_TYPE{
   SWING_OB,
   SWING_BOS,
};
```

The code includes MetaTrader’s trading libraries, specifically CTrade and CPositionInfo, which allow the EA to place and manage trades, as well as access information about open positions. By setting up these classes at the beginning, the EA ensures it has the core tools required for order execution and position tracking.

The next part defines strategic flexibility and swing classification using enumerations. ENUM\_STRATEGY gives traders the option to choose which method the EA should follow: strictly Order Blocks, strictly FVGs, strictly BOS, or a fully automated mode (STRAT\_AUTO) that uses all three concepts dynamically. Similarly, the SWING\_TYPE enumeration differentiates between swing points used for Order Blocks and those used for Break of Structure analysis. This structure makes the code modular and adaptable, allowing traders to experiment with different SMC approaches or let the EA decide automatically based on market conditions.

```
//----------------------------- Inputs ------------------------------//
input ENUM_STRATEGY TradeStrategy   = STRAT_AUTO;
input double        In_Lot          = 0.02;
input double        StopLoss        = 3500;   // points
input double        TakeProfit      = 7500;   // points
input long          MagicNumber     = 76543;

input int           SwingPeriod     = 5;      // bars each side to confirm swing
input int           SwingProbeBar   = 5;      // bar index we test for swings (>=SwingPeriod)
input double        Fib_Trade_lvls  = 61.8;   // OB retrace must reach this %
input bool          DrawBOSLines    = true;

input int           FVG_MinPoints   = 3;      // minimal gap in points
input int           FVG_ScanBars    = 20;     // how many bars to scan for FVGs
input bool          FVG_TradeAtEQ   = true;   // trade at 50% of the gap (EQ)
input bool          OneTradePerBar  = true;

//---------------------------- Colors -------------------------------//
#define BullOB   clrLime
#define BearOB   clrRed
#define BullFVG  clrPaleGreen
#define BearFVG  clrMistyRose
#define BOSBull  clrDodgerBlue
#define BOSBear  clrTomato
```

This section of the code defines the inputs that traders can customize to fit their trading style and risk management preferences. The TradeStrategy input allows the user to select whether the EA should trade using only Order Blocks, only Fair Value Gaps, only Break of Structure, or automatically combine all three. Risk and money management settings are controlled through In\_Lot, StopLoss, TakeProfit, and MagicNumber, which ensure proper lot sizing, protective stops, profit targets, and unique trade identification. Additionally, swing detection parameters like SwingPeriod and SwingProbeBar specify how many bars are considered when identifying swing highs and lows, while Fib\_Trade\_lvls ensures that Order Block trades align with Fibonacci retracement confirmation. The option DrawBOSLines gives flexibility for chart visualization, allowing traders to see Break of Structure levels directly on the chart.

The Fair Value Gap (FVG) settings follow, providing finer control over how gaps are detected and traded. FVG\_MinPoints ensures that only significant imbalances are considered, while FVG\_ScanBars determines how far back in history the EA will search for valid gaps. The FVG\_TradeAtEQ option gives the ability to trade at the equilibrium level (50% of the gap), which is often seen as a balanced entry point in SMC theory. Finally, color definitions such as BullOB, BearOB, BullFVG, BearFVG, BOSBull, and BOSBear make chart objects visually intuitive, allowing traders to quickly distinguish bullish vs bearish setups at a glance.

```
//---------------------------- Globals ------------------------------//
double   Bid, Ask;
datetime g_lastBarTime = 0;

// OB state
class COrderBlock : public CObject
{
public:
   int      direction;   // +1 bullish, -1 bearish
   datetime time;        // OB candle time
   double   high;        // OB candle high
   double   low;         // OB candle low

   string Key() const { return TimeToString(time, TIME_DATE|TIME_MINUTES); }

   void draw(datetime tmS, datetime tmE, color clr){
      string objOB = " OB REC" + TimeToString(time);
      ObjectCreate( 0, objOB, OBJ_RECTANGLE, 0, time, low, tmS, high);
      ObjectSetInteger( 0, objOB, OBJPROP_FILL, true);
      ObjectSetInteger( 0, objOB, OBJPROP_COLOR, clr);

      string objtrade = " OB trade" + TimeToString(time);
      ObjectCreate( 0, objtrade, OBJ_RECTANGLE, 0, tmS, high, tmE, low); // trnary operator
      ObjectSetInteger( 0, objtrade, OBJPROP_FILL, true);
      ObjectSetInteger( 0, objtrade, OBJPROP_COLOR, clr);
   }
};
COrderBlock* OB = NULL;

// OB fib state
// Track if an OB has already been traded
datetime lastTradedOBTime = 0;
bool tradedOB = false;
double fib_low, fib_high;
datetime fib_t1, fib_t2;
bool isBullishOB = false;
bool isBearishOB = false;
datetime T1;
datetime T2;
color OBClr;
#define FIB_OB_BULL "FIB_OB_BULL"
#define FIB_OB_BEAR "FIB_OB_BEAR"
#define FIBO_OBJ "Fibo Retracement"

// BOS state
datetime lastBOSTradeTime = 0;
bool Bull_BOS_traded, Bear_BOS_traded;
int lastBOSTradeDirection = 0; // 1 for buy, -1 for sell
double   swng_High = -1.0, swng_Low = -1.0;
datetime bos_tH = 0, bos_tL = 0;
```

Here, this global section of the code establishes the core variables and structures that keep track of market state, trading opportunities, and chart objects. It begins with simple global values like Bid, Ask, and g\_lastBarTime, which are essential for real-time price tracking and ensuring the Expert Advisor only executes logic once per new bar. From there, a specialized class, COrderBlock, is created to encapsulate the properties of an Order Block, including its direction (bullish or bearish), time, and price levels (high/low). The class also contains a draw() function, which automatically creates and colors rectangles on the chart, making it easy for traders to visually identify active order blocks and their trading zones. By centralizing these functions within a class, the code remains organized and reusable.

Beyond basic OB identification, additional variables are defined to handle Fibonacci-based validation of Order Blocks. For example, fib\_low, fib\_high, and the time markers fib\_t1 and fib\_t2 store the boundaries for Fibonacci retracement levels, while flags like isBullishOB, isBearishOB, and tradedOB ensure that each order block is validated and traded only once. These variables allow the EA to reference Fibonacci levels dynamically, ensuring that only valid retracements (such as the 61.8% level defined in inputs) are considered for trade entries. To make the visualization clearer, constants like FIB\_OB\_BULL, FIB\_OB\_BEAR, and FIBO\_OBJ define the names of chart objects, ensuring that Fibonacci retracements are distinguishable when drawn.

Finally, the Break of Structure (BOS) state management is introduced. The variables lastBOSTradeTime, Bull\_BOS\_traded, and Bear\_BOS\_traded prevent duplicate trades from being executed when a BOS signal occurs. The lastBOSTradeDirection keeps track of whether the last BOS trade was bullish (+1) or bearish (–1), while swng\_High and swng\_Low store the most recent swing levels for structure detection. Time markers bos\_tH and bos\_tL are included to identify the precise moments when swings were confirmed, ensuring alignment with market structure rules. By managing these states globally, the EA ensures consistency across different strategies (Order Blocks, BOS, FVG) and prevents overlapping trades, creating a structured foundation for Smart Money Concept trading automation.

```
//--------------------------- Helpers -------------------------------//
double  getHigh(int i)   { return iHigh(_Symbol, _Period, i);  }
double  getLow(int i)    { return iLow(_Symbol, _Period, i);   }
double  getOpen(int i)   { return iOpen(_Symbol, _Period, i);  }
double  getClose(int i)  { return iClose(_Symbol, _Period, i); }
datetime getTimeBar(int i){ return iTime(_Symbol, _Period, i); }

bool IsNewBar()
{
   datetime lastbar_time = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);
   if(g_lastBarTime == 0) { g_lastBarTime = lastbar_time; return false; }
   if(g_lastBarTime != lastbar_time) { g_lastBarTime = lastbar_time; return true; }
   return false;
}

void ExecuteTrade(ENUM_ORDER_TYPE type)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double price = (type==ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                         : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl = (type==ORDER_TYPE_BUY) ? price - StopLoss*point
                                      : price + StopLoss*point;
   double tp = (type==ORDER_TYPE_BUY) ? price + TakeProfit*point
                                      : price - TakeProfit*point;
   sl = NormalizeDouble(sl, _Digits);
   tp = NormalizeDouble(tp, _Digits);

   trade.SetExpertMagicNumber(MagicNumber);
   trade.PositionOpen(_Symbol, type, In_Lot, price, sl, tp, "SMC");
}
```

The helper functions here provide a clean and reusable way to interact with chart data. Instead of calling built-in functions like iHigh, iLow, or iClose directly throughout the code, wrappers such as getHigh(), getLow(), and getClose() make it easier to access bar information while improving readability. The IsNewBar() function plays a crucial role in ensuring the EA only processes logic once per candle, avoiding repeated executions within the same bar. It does this by storing the last known bar time in g\_lastBarTime and comparing it against the current bar’s timestamp. This efficient approach ensures signals and trades are only considered when a new bar opens, which is essential for strategies that rely on confirmed candle closes.

The ExecuteTrade() function encapsulates the entire trade execution logic. It automatically calculates the correct entry price, stop loss, and take profit levels based on the order type (buy or sell) while ensuring values are normalized to match the instrument’s decimal precision. By centralizing this logic, the EA avoids repetitive code and reduces errors when opening trades. The function also sets the EA’s unique MagicNumber for tracking and tags each trade with the comment "SMC," allowing traders to easily identify positions opened by this system.

```
//----------------------- Unified Swing Detection -------------------//
// Detects if barIndex is a swing high and/or swing low using len bars on each side.
// If swing high found -> updates fib_high/fib_tH (and swng_High/bos_tH if for BOS).
// If swing low  found -> updates fib_low/fib_tL (and swng_Low/bos_tL if for BOS).
// return: true if at least one swing found.
void DetectSwingForBar(int barIndex, SWING_TYPE type)
{
   const int len = 5;
   bool isSwingH = true, isSwingL = true;

   for(int i = 1; i <= len; i++){
      int right_bars = barIndex - i;
      int left_bars  = barIndex + i;

      if(right_bars < 0) {
         isSwingH = false;
         isSwingL = false;
         break;
      }

      if((getHigh(barIndex) <= getHigh(right_bars)) || (left_bars < Bars(_Symbol, _Period) && getHigh(barIndex) < getHigh(left_bars)))
         isSwingH = false;

      if((getLow(barIndex) >= getLow(right_bars)) || (left_bars < Bars(_Symbol, _Period) && getLow(barIndex) > getLow(left_bars)))
         isSwingL = false;
   }

   // Assign with ternary operator depending on swing type
   if(isSwingH){
      if(type == SWING_OB) {
         fib_high = getHigh(barIndex);
         fib_t1 = getTimeBar(barIndex);
      } else {
         swng_High = getHigh(barIndex);
         bos_tH = getTimeBar(barIndex);
      }
   }
   if(isSwingL){
      if(type == SWING_OB) {
         fib_low = getLow(barIndex);
         fib_t2 = getTimeBar(barIndex);
      } else {
         swng_Low = getLow(barIndex);
         bos_tL = getTimeBar(barIndex);
      }
   }
}
```

The unified swing detection function is designed to identify potential swing highs and swing lows on the chart by examining a given bar and comparing it to a set number of neighboring bars. Using a symmetrical range (len bars on each side), the function checks if the current bar forms a local extremum, meaning its high is greater than both the left and right bars (swing high), or its low is lower than both sides (swing low). If any bar to the left or right invalidates these conditions, the swing designation is dismissed. This method ensures that swings are not falsely triggered by minor fluctuations, instead focusing on meaningful pivot points in market structure.

Once a swing is confirmed, the function updates key variables depending on the swing type. For order block (SWING\_OB) detection, swing highs and lows are assigned to Fibonacci anchor points (fib\_high and fib\_low) along with their timestamps, which will later be used for retracement validation. For break of structure (BOS) logic, the same swing levels are assigned to swng\_High or swng\_Low, paired with their respective timestamps (bos\_tH or bos\_tL). By handling both OB and BOS in a single function, the EA keeps swing detection streamlined and avoids redundant logic, ensuring that both structure validation and Fibonacci retracement setups share the same consistent swing identification process.

```
void DetectAndDrawOrderBlocks()
{
   static datetime lastDetect = 0;
   datetime lastBar = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);

   // Reset OB detection on new bar
   if(lastDetect != lastBar)
   {
      if(OB != NULL)
      {
         delete OB;
         OB = NULL;
      }
      lastDetect = lastBar;
   }

   // Only detect new OB if we don't have one already
   if(OB == NULL)
   {
      for(int i = 1; i < 100; i++)
      {
         // Bullish OB candidate
         if(getOpen(i) < getClose(i) &&
            getOpen(i+2) < getClose(i+2) &&
            getOpen(i+3) > getClose(i+3) &&
            getOpen(i+3) < getClose(i+2))
         {
            OB = new COrderBlock();
            OB.direction = 1;
            OB.time = getTimeBar(i+3);
            OB.high = getHigh(i+3);
            OB.low = getLow(i+3);
            OBClr = BullOB;
            T1 = OB.time;
            Print("Bullish Order Block detected at: ", TimeToString(OB.time));
            break;
         }

         // Bearish OB candidate
         if(getOpen(i) > getClose(i) &&
            getOpen(i+2) > getClose(i+2) &&
            getOpen(i+3) < getClose(i+3) &&
            getOpen(i+3) > getClose(i+2)) // Fixed condition
         {
            OB = new COrderBlock();
            OB.direction = -1;
            OB.time = getTimeBar(i+3);
            OB.high = getHigh(i+3);
            OB.low = getLow(i+3);
            OBClr = BearOB;
            T1 = OB.time;
            Print("Bearish Order Block detected at: ", TimeToString(OB.time));
            break;
         }
      }
   }

   if(OB == NULL) return;

   // Check if we already traded this OB
   if(lastTradedOBTime == OB.time) return;

   // If price retraces inside OB zone
   Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   bool inBullZone = (OB.direction > 0 && Ask <= OB.high && Ask >= OB.low);
   bool inBearZone = (OB.direction < 0 && Bid >= OB.low && Bid <= OB.high);

   if(!inBullZone && !inBearZone) return;

   // Use your DetectSwing function to find swings
   // We need to call it multiple times to find the most recent swings
   double mostRecentSwingHigh = 0;
   double mostRecentSwingLow = EMPTY_VALUE;
   datetime mostRecentSwingHighTime = 0;
   datetime mostRecentSwingLowTime = 0;

   // Scan recent bars to find the most recent swings
   for(int i = 0; i < 20; i++) // Check the last 20 bars
   {
      // Reset swing variables
      fib_high = 0;
      fib_low = 0;
      fib_t1 = 0;
      fib_t2 = 0;

      DetectSwingForBar(i, SWING_OB);

      if(fib_high > 0 && (mostRecentSwingHighTime == 0 || fib_t1 > mostRecentSwingHighTime))
      {
         mostRecentSwingHigh = fib_high;
         mostRecentSwingHighTime = fib_t1;
      }

      if(fib_low < EMPTY_VALUE && (mostRecentSwingLowTime == 0 || fib_t2 > mostRecentSwingLowTime))
      {
         mostRecentSwingLow = fib_low;
         mostRecentSwingLowTime = fib_t2;
      }
   }

   // Ensure we found both swing points
   if(mostRecentSwingHighTime == 0 || mostRecentSwingLowTime == 0) return;

   // Draw Fibonacci before trading to validate
   if(OB.direction > 0 && inBullZone)
   {
      // Draw Fibonacci from recent swing low to recent swing high
      ObjectDelete(0, "FIB_OB_BULL");
      if(ObjectCreate(0, "FIB_OB_BULL", OBJ_FIBO, 0, mostRecentSwingLowTime, mostRecentSwingLow,
                     mostRecentSwingHighTime, mostRecentSwingHigh))
      {
         // Format Fibonacci
         ObjectSetInteger(0, "FIB_OB_BULL", OBJPROP_COLOR, clrBlack);
         for(int i = 0; i < ObjectGetInteger(0, "FIB_OB_BULL", OBJPROP_LEVELS); i++)
         {
            ObjectSetInteger(0, "FIB_OB_BULL", OBJPROP_LEVELCOLOR, i, clrBlack);
         }

         double entLvlBull = mostRecentSwingHigh - (mostRecentSwingHigh - mostRecentSwingLow) * (Fib_Trade_lvls / 100.0);

         if(Ask <= entLvlBull)
         {
            T2 = getTimeBar(0);
            OB.draw(T1, T2, BullOB);
            ExecuteTrade(ORDER_TYPE_BUY);
            lastTradedOBTime = OB.time; // Mark this OB as traded
            delete OB;
            OB = NULL;
         }
      }
   }
   else if(OB.direction < 0 && inBearZone)
   {
      // Draw Fibonacci from recent swing high to recent swing low
      ObjectDelete(0, "FIB_OB_BEAR");
      if(ObjectCreate(0, "FIB_OB_BEAR", OBJ_FIBO, 0, mostRecentSwingHighTime, mostRecentSwingHigh,
                     mostRecentSwingLowTime, mostRecentSwingLow))
      {
         // Format Fibonacci
         ObjectSetInteger(0, "FIB_OB_BEAR", OBJPROP_COLOR, clrBlack);
         for(int i = 0; i < ObjectGetInteger(0, "FIB_OB_BEAR", OBJPROP_LEVELS); i++)
         {
            ObjectSetInteger(0, "FIB_OB_BEAR", OBJPROP_LEVELCOLOR, i, clrBlack);
         }

         double entLvlBear = mostRecentSwingLow + (mostRecentSwingHigh - mostRecentSwingLow) * (Fib_Trade_lvls / 100.0);

         if(Bid >= entLvlBear)
         {
            T2 = getTimeBar(0);
            OB.draw(T1, T2, BearOB);
            ExecuteTrade(ORDER_TYPE_SELL);
            lastTradedOBTime = OB.time; // Mark this OB as traded
            delete OB;
            OB = NULL;
         }
      }
   }
}
```

The DetectAndDrawOrderBlocks() function is responsible for identifying, validating, and trading order blocks while integrating Fibonacci confluence into the decision-making process. It begins by resetting detection on each new bar and then scans recent candles for bullish or bearish order block patterns. Once a valid order block is found, it checks whether the current price retraces into the zone, signaling a potential trade opportunity. Before executing, the function calls swing detection to locate the most recent swing high and swing low, ensuring that a Fibonacci retracement can be drawn between them.

The Fibonacci tool is then used to validate the entry, requiring price alignment with a predefined retracement level before confirming the trade. This way, the system avoids premature entries and ensures trades are executed only when both the order block zone and Fibonacci retracement align, thereby reinforcing accuracy and consistency in trade validation.

```
//============================== FVG ================================//
// Definition (ICT-style):
// Let C=i, B=i+1, A=i+2.
// Bullish FVG if Low(A) > High(C) -> gap [High(C), Low(A)]
// Bearish FVG if High(A) < Low(C) -> gap [High(A), Low(C)]
struct SFVG
{
   int      dir;    // +1 bull, -1 bear
   datetime tLeft;  // left time anchor
   double   top;    // zone top price
   double   bot;    // zone bottom price

   string Name() const
   {
      string k = TimeToString(tLeft, TIME_DATE|TIME_MINUTES);
      return (dir>0 ? "FVG_B_" : "FVG_S_") + k + "_" + IntegerToString((int)(top*1000.0));
   }
};

bool FVGExistsAt(const string &name){ return ObjectFind(0, name) != -1; }

void DetectAndDrawFVGs()
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int counted = 0;

   for(int i=2; i<MathMin(FVG_ScanBars, Bars(_Symbol, _Period))-2; i++)
   {
      // Build A,B,C
      double lowA  = getLow(i+2);
      double highA = getHigh(i+2);
      double highC = getHigh(i);
      double lowC  = getLow(i);

      // Bullish FVG: Low of A > High of C
      if(lowA > highC && (lowA - highC >= FVG_MinPoints * point))
      {
         SFVG z;
         z.dir   = +1;
         z.tLeft = getTimeBar(i+2);  // Changed from getTimeBar to getTime
         z.top   = lowA;
         z.bot   = highC;
         DrawFVG(z);
         counted++;
      }
      // Bearish FVG: High of A < Low of C
      else if(highA < lowC && (lowC - highA >= FVG_MinPoints * point))
      {
         SFVG z;
         z.dir   = -1;
         z.tLeft = getTimeBar(i+2);  // Changed from getTimeBar to getTime
         z.top   = lowC;          // Fixed: should be lowC for bearish FVG top
         z.bot   = highA;         // Fixed: should be highA for bearish FVG bottom
         DrawFVG(z);
         counted++;
      }

      if(counted > 15) break; // avoid clutter
   }

   // --- Simplified trading for FVGs ---
   Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // scan drawn objects and trade on first valid touch of EQ (50%)
   int total = ObjectsTotal(0, 0, -1);
   static datetime lastTradeBar = 0;

   if(OneTradePerBar)
   {
      datetime barNow = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);
      if(lastTradeBar == barNow) return; // already traded this bar
   }

   for(int idx=0; idx<total; idx++)
   {
      string name = ObjectName(0, idx);
      if(StringFind(name, "FVG_", 0) != 0) continue; // only our FVGs

      // Get object coordinates
      datetime t1 = (datetime)ObjectGetInteger(0, name, OBJPROP_TIME, 0);
      double y1 = ObjectGetDouble(0, name, OBJPROP_PRICE, 0);
      datetime t2 = (datetime)ObjectGetInteger(0, name, OBJPROP_TIME, 1);
      double y2 = ObjectGetDouble(0, name, OBJPROP_PRICE, 1);

      double top = MathMax(y1, y2);
      double bot = MathMin(y1, y2);
      bool isBull = (StringFind(name, "FVG_B_", 0) == 0);
      double mid  = (top + bot) * 0.5;

      if(isBull)
      {
         // trade when Ask is inside the gap and at/under EQ
         if(Ask <= top && Ask >= bot && (!FVG_TradeAtEQ || Ask <= mid))
         {
            ExecuteTrade(ORDER_TYPE_BUY);
            lastTradeBar = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);
            break;
         }
      }
      else
      {
         // trade when Bid is inside the gap and at/over EQ
         if(Bid <= top && Bid >= bot && (!FVG_TradeAtEQ || Bid >= mid))
         {
            ExecuteTrade(ORDER_TYPE_SELL);
            lastTradeBar = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);
            break;
         }
      }
   }
}
```

This function detects, draws, and trades Fair Value Gaps (FVGs) based on ICT logic. It first scans recent bars to identify bullish gaps, where the low of candle A is higher than the high of candle C, and bearish gaps, where the high of candle A is lower than the low of candle C, provided the gap meets a minimum size in points. Once an FVG is detected, it is stored in a structure, visually drawn on the chart, and tracked for trading. The system then monitors active FVG zones, checking when price enters the gap and aligns with the equilibrium level (50% of the zone) if enabled. When conditions are met, the function executes a trade, buying from bullish FVGs and selling from bearish FVGs while ensuring only one trade per bar to prevent overtrading. This combination of detection, visualization, and execution makes the FVG tool both analytical and directly tradable.

```
//=============================== BOS ===============================//
// Use unified swings (no RSI). Trading logic mirrors your earlier code:
// - Sell when price breaks above last swing high (liquidity run idea)
// - Buy  when price breaks below last swing low
void DetectAndDrawBOS()
{
   // Use DetectSwingForBar to find the most recent swing points
   double mostRecentSwingHigh = 0;
   double mostRecentSwingLow = EMPTY_VALUE;
   datetime mostRecentSwingHighTime = 0;
   datetime mostRecentSwingLowTime = 0;

   // Scan recent bars to find the most recent swings for BOS
   for(int i = 0; i < 20; i++) // Check the last 20 bars
   {
      // Reset swing variables
      swng_High = 0;
      swng_Low = 0;
      bos_tH = 0;
      bos_tL = 0;

      // Detect swing at this bar for BOS
      DetectSwingForBar(i, SWING_BOS);

      if(swng_High > 0 && (mostRecentSwingHighTime == 0 || bos_tH > mostRecentSwingHighTime))
      {
         mostRecentSwingHigh = swng_High;
         mostRecentSwingHighTime = bos_tH;
      }

      if(swng_Low < EMPTY_VALUE && (mostRecentSwingLowTime == 0 || bos_tL > mostRecentSwingLowTime))
      {
         mostRecentSwingLow = swng_Low;
         mostRecentSwingLowTime = bos_tL;
      }
   }

   // Update the global BOS variables with the most recent swings
   if(mostRecentSwingHighTime > 0)
   {
      if(mostRecentSwingHighTime != bos_tH)
         Bull_BOS_traded = false;
      swng_High = mostRecentSwingHigh;
      bos_tH = mostRecentSwingHighTime;
   }

   if(mostRecentSwingLowTime > 0)
   {
      if(mostRecentSwingLowTime != bos_tL)
         Bear_BOS_traded = false;
      swng_Low = mostRecentSwingLow;
      bos_tL = mostRecentSwingLowTime;
   }

   // Now check for break of structure
   Bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   Ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   // Get current bar time to prevent multiple trades on same bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);

   // SELL on break above swing high
   if(swng_High > 0 && Ask > swng_High && Bull_BOS_traded == false)
   {
      // Check if we haven't already traded this breakout
      if(lastBOSTradeTime != currentBarTime || lastBOSTradeDirection != -1)
      {
         if(DrawBOSLines)
            DrawBOS("BOS_H_" + TimeToString(bos_tH), bos_tH, swng_High,
                    TimeCurrent(), swng_High, BOSBear, -1);

         ExecuteTrade(ORDER_TYPE_BUY);

         // Update trade tracking
         lastBOSTradeTime = currentBarTime;
         lastBOSTradeDirection = -1;
         Bull_BOS_traded = true;

         // Reset the swing high to prevent immediate re-trading
         swng_High = -1.0;
      }
   }

   // BUY on break below swing low
   if(swng_Low > 0 && Bid < swng_Low && Bear_BOS_traded == false)
   {
      // Check if we haven't already traded this breakout
      if(lastBOSTradeTime != currentBarTime || lastBOSTradeDirection != 1)
      {
         if(DrawBOSLines)
            DrawBOS("BOS_L_" + TimeToString(bos_tL), bos_tL, swng_Low,
                    TimeCurrent(), swng_Low, BOSBull, +1);

         ExecuteTrade(ORDER_TYPE_SELL);

         // Update trade tracking
         Bear_BOS_traded = true;
         lastBOSTradeTime = currentBarTime;
         lastBOSTradeDirection = 1;

         // Reset the swing low to prevent immediate re-trading
         swng_Low = -1.0;
      }
   }
}
```

The BOS function detects and trades Break of Structure (BOS) events using unified swing logic. It scans the last 20 bars to identify the most recent swing high and swing low, updates global BOS variables, and ensures duplicate trades on the same bar are avoided. If the price breaks above the last swing high, it executes a buy (liquidity run above highs), and if price breaks below the last swing low, it executes a sell. Optional BOS lines can be drawn for visualization, and internal flags prevent immediate re-trading on the same breakout.

```
//---------------------------- BOS UI -------------------------------//
void DrawBOS(const string name, datetime t1, double p1, datetime t2, double p2, color col, int dir)
{
   if(ObjectFind(0, name) == -1)
   {
      ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2);
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);

      string lbl = name + "_lbl";
      ObjectCreate(0, lbl, OBJ_TEXT, 0, t2, p2);
      ObjectSetInteger(0, lbl, OBJPROP_COLOR, col);
      ObjectSetInteger(0, lbl, OBJPROP_FONTSIZE, 10);
      ObjectSetString(0,  lbl, OBJPROP_TEXT, "Break");
      ObjectSetInteger(0, lbl, OBJPROP_ANCHOR, (dir>0)?ANCHOR_RIGHT_UPPER:ANCHOR_RIGHT_LOWER);
   }
}
```

The DrawBOS function is responsible for visually marking a Break of Structure (BOS) on a chart by creating a trend line between two points (t1, p1) and (t2, p2) with a specified color and width. If the object with the given name doesn’t already exist, it first creates the trend line, sets its color and thickness, and then adds a text label at the endpoint indicating "Break." The label’s position is anchored depending on the direction dir, placing it at the upper-right if bullish or lower-right if bearish, effectively providing a clear visual cue of market structure changes directly on the chart.

```
void DrawFVG(const SFVG &z)
{
   string name = z.Name();
   datetime tNow = (datetime)SeriesInfoInteger(_Symbol, _Period, SERIES_LASTBAR_DATE);

   // Delete existing object if it exists
   if(ObjectFind(0, name) != -1)
      ObjectDelete(0, name);

   // Create rectangle object for FVG
   if(!ObjectCreate(0, name, OBJ_RECTANGLE, 0, z.tLeft, z.bot, tNow, z.top))
   {
      Print("Error creating FVG object: ", GetLastError());
      return;
   }

   // Set object properties
   ObjectSetInteger(0, name, OBJPROP_COLOR, z.dir>0 ? BullFVG : BearFVG);
   ObjectSetInteger(0, name, OBJPROP_FILL, true);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);

   // Set Z-order to make sure it's visible
   ObjectSetInteger(0, name, OBJPROP_ZORDER, 0);
}
```

The DrawFVG function visualizes a Fair Value Gap (FVG) on the chart by first checking if an object with the FVG’s name already exists and deleting it to avoid duplicates. It then creates a rectangle spanning from the gap’s left time tLeft to the current bar and from the bottom bot to the top top of the gap. The rectangle is styled with a color based on the gap’s direction, filled, sent to the back of the chart, and given a solid border. By setting its Z-order to 0, the FVG remains visible behind other chart elements, providing traders with a clear, real-time visual reference of price inefficiencies.

```
void OnTick()
{
   if(!IsNewBar()) return;

   // Strategy switch
   if(TradeStrategy == STRAT_FVG || TradeStrategy == STRAT_AUTO)
      DetectAndDrawFVGs();

   if(TradeStrategy == STRAT_OB  || TradeStrategy == STRAT_AUTO)
      DetectAndDrawOrderBlocks();

   if(TradeStrategy == STRAT_BOS || TradeStrategy == STRAT_AUTO)
      DetectAndDrawBOS();
}
```

The OnTick function executes on each market tick but only proceeds when a new bar is formed, ensuring calculations are performed once per candle. It then checks the selected trading strategy: if Fair Value Gaps (FVG) or automatic mode is active, it calls DetectAndDrawFVGs; if Order Blocks (OB) or automatic mode is active, it calls DetectAndDrawOrderBlocks; and if Break of Structure (BOS) or automatic mode is active, it calls DetectAndDrawBOS. This structure allows the EA or indicator to dynamically detect and visualize different market structures in real-time based on the chosen strategy.

### **Backtest Results**

The back-testing was evaluated on the 1H timeframe across roughly a 2-month testing window (01 July 2025 to 01 September 2025), with the default settings using the BOS as the strategy.

![](https://c.mql5.com/2/167/EC.png)

![](https://c.mql5.com/2/167/BT.png)

### Conclusion

In summary, we developed a unified Smart Money Concepts (SMC) trading framework that integrates three key pillars: Order Blocks (OBs), Break of Structure (BOS), and Fair Value Gaps (FVGs). Each concept was coded with detection, drawing, and trading logic. OBs were identified as institutional footprints with Fibonacci retracement validation, BOS captured liquidity runs when price swept swing highs or lows, and FVGs highlighted inefficiencies in price delivery that often act as strong reaction zones. By combining these elements into a single system, the code ensures dynamic detection of high-probability setups directly on the chart, complete with visual aids and automated execution.

In conclusion, this unified approach empowers traders with a systematic way of applying SMC principles without the guesswork that often comes with manual chart analysis. The EA not only marks and tracks market structure shifts but also executes trades in real time when price aligns with predefined SMC conditions. This provides traders with a disciplined, rules-based method that reduces emotional bias, increases consistency, and offers a professional edge in capturing institutional-style opportunities across different market conditions.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16340.zip "Download all attachments in the single ZIP archive")

[SMC\_ALL\_IN\_1.mq5](https://www.mql5.com/en/articles/download/16340/SMC_ALL_IN_1.mq5 "Download SMC_ALL_IN_1.mq5")(21.41 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/495130)**
(13)


![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
16 Sep 2025 at 15:59

**Stanislav Korotky [#](https://www.mql5.com/en/forum/495130#comment_58041014):**

The Fair Value Gap is described and depicted incorrectly, please look at the correct picture in [this article](https://www.mql5.com/en/articles/16659).

You are mistaken, it's the same thing, I depicted the FVG with the price/candlesticks already retraced inside the FVG zone.


![Hlomohang John Borotho](https://c.mql5.com/avatar/2023/9/6505ca3e-1abb.jpg)

**[Hlomohang John Borotho](https://www.mql5.com/en/users/johnhlomohang)**
\|
16 Sep 2025 at 16:00

**Yata Tema Gea [#](https://www.mql5.com/en/forum/495130#comment_58045741):**

Wow, your article is great. I just know that there are indonesians in the article. Our concept is the same, but the difference is that I use Python for SMC+GPT.

I see that :)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
16 Sep 2025 at 17:37

**Hlomohang John Borotho [#](https://www.mql5.com/en/forum/495130#comment_58048813):**

You are mistaken, it's the same thing, I depicted the FVG with the price/candlesticks already retraced inside the FVG zone.

You're wrong or prepared your graphic poorly. As the 1-st and 3-rd candles (from the left) are shown in red, they are bearish and form 2 large gaps _before_ and _after_ the 2-nd candle (where presumably FVG occurred).

![](https://c.mql5.com/2/166/FVG.png)

If you search through Internet for FVG, you'll find that this formation is about large unidirectional jump/gap in the price, not a zig-zag of jumps, which would obscure the whole effect for possible future retracements, because you don't know which one (of the 3 gaps) is about to close first (in your picture it's the last gap along the 3-rd candle, not the 2-nd one - it's closed by the 4-th candle).

![Bao Thuan Thai](https://c.mql5.com/avatar/2025/4/680C8606-3BDF.png)

**[Bao Thuan Thai](https://www.mql5.com/en/users/thuanthai)**
\|
28 Sep 2025 at 11:06

I don’t need to know how it works, but thank you very much for your sharing and kindness


![ritik pathak](https://c.mql5.com/avatar/2023/10/651D0917-4E29.jpg)

**[ritik pathak](https://www.mql5.com/en/users/ritikpathak2425)**
\|
4 Oct 2025 at 11:01

great job, it can become more good if stop is below/ above FVG or Order block not fixed point [stop loss](https://www.metatrader5.com/en/terminal/help/trading/general_concept "User Guide: Types of orders").


![Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://c.mql5.com/2/168/19428-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

This article describes a simple but comprehensive statistical arbitrage pipeline for trading a basket of cointegrated stocks. It includes a fully functional Python script for data download and storage; correlation, cointegration, and stationarity tests, along with a sample Metatrader 5 Service implementation for database updating, and the respective Expert Advisor. Some design choices are documented here for reference and for helping in the experiment replication.

![Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://c.mql5.com/2/130/Moving_to_MQL5_Algo_Forge_Part_LOGO__3.png)[Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)

When working on projects in MetaEditor, developers often face the need to manage code versions. MetaQuotes recently announced migration to GIT and the launch of MQL5 Algo Forge with code versioning and collaboration capabilities. In this article, we will discuss how to use the new and previously existing tools more efficiently.

![Developing a Custom Market Sentiment Indicator](https://c.mql5.com/2/168/19422-developing-a-custom-market-logo.png)[Developing a Custom Market Sentiment Indicator](https://www.mql5.com/en/articles/19422)

In this article we are developing a custom market sentiment indicator to classify conditions into bullish, bearish, risk-on, risk-off, or neutral. Using multi-timeframe, the indicator can provide traders with a clearer perspective of overall market bias and short-term confirmations.

![Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/168/19442-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19442)

In this article, we develop an AB=CD Pattern EA in MQL5 that identifies bullish and bearish AB=CD harmonic patterns using pivot points and Fibonacci ratios, executing trades with precise entry, stop loss, and take-profit levels. We enhance trader insight with visual feedback through chart objects.

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/16340&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083099967033185541)

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