---
title: Automating Trading Strategies in MQL5 (Part 35): Creating a Breaker Block Trading System
url: https://www.mql5.com/en/articles/19638
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:25:28.134171
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/19638&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049122304660710742)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 34)](https://www.mql5.com/en/articles/19625), we developed a [Trendline Breakout](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/spotting-breakouts "https://www.babypips.com/learn/forex/spotting-breakouts") System in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that identified support and resistance trendlines using swing points, validated by R-squared goodness of fit, to execute breakout trades with dynamic chart visualizations. In Part 35, we create a [Breaker Block Trading System](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks") that detects consolidation ranges, validates breaker blocks with swing points, and trades retests with customizable risk parameters and visual feedback. We will cover the following topics:

1. [Understanding the Breaker Block Strategy Framework](https://www.mql5.com/en/articles/19638#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19638#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19638#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19638#para4)

By the end, you’ll have a functional MQL5 strategy for trading breaker block retests, ready for customization—let’s dive in!

### Understanding the Breaker Block Strategy Framework

The [breaker block strategy](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks") is a trading strategy that identifies consolidation ranges where price moves within a tight range, followed by a breakout and an impulsive move, forming order blocks that, when invalidated, become breaker blocks for potential retest trades. It capitalizes on price returning to these blocks after a significant move away, entering trades in the direction of the breakout with defined stop-loss and take-profit levels, enhanced by visual chart elements. Have a look at the different breaker blocks we could have below.

Bearish Breaker Block:

![BEARISH BREAKER BLOCK](https://c.mql5.com/2/171/Screenshot_2025-09-20_154605.png)

Bullish Breaker Block:

![BULLISH BREAKER BLOCK](https://c.mql5.com/2/171/Screenshot_2025-09-20_154621.png)

Our plan is to detect consolidation ranges by analyzing a set number of bars, identify breakouts when price exits the range, and confirm impulsive moves using a multiplier-based threshold. We will implement logic to validate breaker blocks with swing points, execute trades on retests with customizable parameters, and visualize blocks with dynamic labels and arrows, thereby creating a system for identifying and trading breaker block opportunities. In brief, here is a visual representation of our objectives.

![BREAKER BLOCK FRAMEWORK](https://c.mql5.com/2/171/Screenshot_2025-09-20_155325.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Experts folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                    Breaker Block Strategy EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property strict

#include <Trade/Trade.mqh>                         //--- Include Trade library for position management
CTrade obj_Trade;                                  //--- Instantiate trade object for order operations
```

We begin the implementation by including the trade library with "#include <Trade/Trade.mqh>", which provides built-in functions for managing trade operations. We then initialize the trade object "obj\_Trade" using the [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) class, allowing the Expert Advisor to execute buy and sell orders programmatically. This setup will ensure that trade execution is handled efficiently without requiring manual intervention. Then we can provide some inputs so the user can change and control the behavior from the [user interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI).

```
//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input double tradeLotSize = 0.01;                  // Trade size for each position in lots
input bool   enableTrading = true;                 // Toggle to enable or disable automated trading
input bool   enableTrailingStop = true;            // Toggle to enable or disable trailing stop
input double trailingStopPoints = 30;              // Distance in points for trailing stop adjustment
input double minProfitToTrail = 50;                // Minimum profit in points before trailing starts
input int    uniqueMagicNumber = 12345;            // Unique identifier for EA trades
input int    consolidationBars = 7;                // Number of bars to check for consolidation range
input double maxConsolidationSpread = 50;          // Maximum allowed spread in points for consolidation
input int    barsToWaitAfterBreakout = 3;          // Bars to wait after breakout before impulse check
input double impulseMultiplier = 1.0;              // Multiplier for detecting impulsive price moves
input double stopLossDistance = 1500;              // Stop loss distance in points from entry
input double takeProfitDistance = 1500;            // Take profit distance in points from entry
input double moveAwayDistance = 50;                // Distance in points for price to move away post-invalidation
input color  bullishColor = clrGreen;              // Base color for bullish order/breaker blocks
input color  bearishColor = clrRed;                // Base color for bearish order/breaker blocks
input color  labelTextColor = clrBlack;            // Color for text labels on blocks
input bool   enableSwingValidation = true;         // Enable validation of swing points for block invalidation
input bool   showSwingPoints = true;               // Show swing point labels if validation enabled
input color  swingLabelColor = clrWhite;           // Color for swing point labels
input int    swingFontSize = 10;                   // Font size for swing point labels
```

Here, we define [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables) to configure the program's behavior. "tradeLotSize" sets the position size, while "enableTrading" and "enableTrailingStop" control execution and trailing stops, with "trailingStopPoints" and "minProfitToTrail" refining stop logic. "uniqueMagicNumber" identifies trades, and consolidation is detected using "consolidationBars" and "maxConsolidationSpread". The rest of the inputs are self-explanatory. We have added comments to ease their understanding. Lastly, we need to define some global variables that we will use for the overall system control.

```
//+------------------------------------------------------------------+
//| Structure for price and index                                    |
//+------------------------------------------------------------------+
struct PriceAndIndex {                             //--- Define structure for price and index data
   double price;                                   //--- Store price value (high or low)
   int    index;                                   //--- Store bar index of price
};

//+------------------------------------------------------------------+
//| Global variables for market state tracking                       |
//+------------------------------------------------------------------+
PriceAndIndex rangeHighestHigh = {0, 0};           //--- Track highest high in consolidation range
PriceAndIndex rangeLowestLow = {0, 0};             //--- Track lowest low in consolidation range
bool   isBreakoutDetected = false;                 //--- Flag for breakout detection
double lastImpulseLow = 0.0;                       //--- Store low price after breakout for impulse
double lastImpulseHigh = 0.0;                      //--- Store high price after breakout for impulse
int    breakoutBarNumber = -1;                     //--- Store bar index of breakout
datetime breakoutTimestamp = 0;                    //--- Store timestamp of breakout
string   blockNames[];                             //--- Store names of block objects
datetime blockEndTimes[];                          //--- Store end times of blocks
bool     invalidatedStatus[];                      //--- Track invalidation status of blocks
string   blockTypes[];                             //--- Track block types (OB/BB, bullish/bearish)
bool     movedAwayStatus[];                        //--- Track if price moved away after invalidation
bool     retestedStatus[];                         //--- Track if block was retested
string   blockLabels[];                            //--- Store label object names for blocks
datetime creationTimes[];                          //--- Store creation times of blocks
datetime invalidationTimes[];                      //--- Store invalidation times of blocks
double   invalidationSwings[];                     //--- Store swing high/low at invalidation
bool     isBullishImpulse = false;                 //--- Flag for bullish impulsive move
bool     isBearishImpulse = false;                 //--- Flag for bearish impulsive move
#define OB_Prefix "OB REC "                        //--- Define prefix for order block names
```

Here, we establish the data [structures](https://www.mql5.com/en/docs/basis/types/classes) and state tracking for the system. First, we define the "PriceAndIndex" structure to hold a price value (high or low) and its corresponding bar index, enabling precise tracking of consolidation range boundaries. Then, we initialize [global variables](https://www.mql5.com/en/docs/basis/variables/global): "rangeHighestHigh" and "rangeLowestLow" as "PriceAndIndex" instances to store the highest high and lowest low of the consolidation range, "isBreakoutDetected" as false to flag breakout events, "lastImpulseLow" and "lastImpulseHigh" as 0.0 to record prices post-breakout for impulse checks, "breakoutBarNumber" as -1 and "breakoutTimestamp" as 0 to track breakout timing, and arrays "blockNames", "blockEndTimes", "invalidatedStatus", "blockTypes", "movedAwayStatus", "retestedStatus", "blockLabels", "creationTimes", "invalidationTimes", and "invalidationSwings" to manage block objects, their expiration, invalidation, types (order or breaker block, bullish or bearish), retest status, and associated swing points.

Finally, we [define](https://www.mql5.com/en/docs/basis/preprosessor/constant) "OB\_Prefix" as "OB REC " for consistent naming of order block objects. Let us now define some helper functions that we will need to darken the color of invalidated order blocks and handle event handlers.

```
//+------------------------------------------------------------------+
//| Darken color by factor                                           |
//+------------------------------------------------------------------+
color DarkenColor(color colorValue, double factor = 0.8) {
   int red = int((colorValue & 0xFF) * factor);          //--- Extract and darken red component
   int green = int(((colorValue >> 8) & 0xFF) * factor); //--- Extract and darken green component
   int blue = int(((colorValue >> 16) & 0xFF) * factor); //--- Extract and darken blue component
   return (color)(red | (green << 8) | (blue << 16));    //--- Combine components into color
}

//+------------------------------------------------------------------+
//| Price data accessors                                             |
//+------------------------------------------------------------------+
double high(int index) {
   return iHigh(_Symbol, _Period, index);         //--- Return high price for specified index
}
double low(int index) {
   return iLow(_Symbol, _Period, index);          //--- Return low price for specified index
}
double close(int index) {
   return iClose(_Symbol, _Period, index);        //--- Return close price for specified index
}
datetime time(int index) {
   return iTime(_Symbol, _Period, index);         //--- Return time for specified index
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   obj_Trade.SetExpertMagicNumber(uniqueMagicNumber); //--- Set magic number for trade identification
   return(INIT_SUCCEEDED);                            //--- Return initialization success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ObjectsDeleteAll(0, OB_Prefix);                //--- Remove all objects with OB prefix
   ChartRedraw(0);                                //--- Redraw chart to clear objects
}
```

We continue to implement utility and lifecycle management functions for our system. First, we develop the "DarkenColor" function, which takes a color value and an optional factor (default 0.8), extracts red, green, and blue components using bitwise operations ("colorValue & 0xFF", "(colorValue >> 8) & 0xFF", "(colorValue >> 16) & 0xFF"), darkens them by multiplying with the factor, and combines them with [bitwise shifts](https://www.mql5.com/en/docs/basis/operations/bit) ("red \| (green << 8) \| (blue << 16)") to return a darkened color for visual distinction of invalidated blocks.

Then, we create accessor functions "high", "low", "close", and "time", which return the high price ( [iHigh](https://www.mql5.com/en/docs/series/ihigh)), low price ("iLow"), close price ("iClose"), and timestamp ( [iTime](https://www.mql5.com/en/docs/series/itime)) for a given bar index on the current symbol and period, simplifying price data retrieval. Next, in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function, we call "SetExpertMagicNumber" on "obj\_Trade" with "uniqueMagicNumber" to tag trades for identification and return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) for successful initialization. Finally, in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, we use [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) to remove all chart objects with "OB\_Prefix" and call [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) to refresh the chart, ensuring clean resource cleanup. We can now graduate to the main [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to implement our main control logic.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   static bool isNewBar = false;                  //--- Track new bar status
   int currentBarCount = iBars(_Symbol, _Period); //--- Get current bar count
   static int previousBarCount = currentBarCount; //--- Store previous bar count
   if (previousBarCount == currentBarCount) {     //--- Check if no new bar
      isNewBar = false;                           //--- Set no new bar
   } else {                                       //--- New bar detected
      isNewBar = true;                            //--- Set new bar flag
      previousBarCount = currentBarCount;         //--- Update previous bar count
   }
   if (!isNewBar) return;                         //--- Exit if not new bar
   int startBarIndex = 1;                         //--- Set start index for analysis
   int chartScale = (int)ChartGetInteger(0, CHART_SCALE); //--- Get chart zoom scale
   int dynamicFontSize = 8 + (chartScale * 2);    //--- Calculate dynamic font size
   if (!isBreakoutDetected) {                     //--- Check if no breakout detected
      if (rangeHighestHigh.price == 0 && rangeLowestLow.price == 0) { //--- Check if range not set
         bool isConsolidated = true;              //--- Assume consolidation
         for (int i = startBarIndex; i < startBarIndex + consolidationBars - 1; i++) { //--- Iterate consolidation bars
            if (MathAbs(high(i) - high(i + 1)) > maxConsolidationSpread * Point() || MathAbs(low(i) - low(i + 1)) > maxConsolidationSpread * Point()) { //--- Check spread
               isConsolidated = false;            //--- Mark as not consolidated
               break;                             //--- Exit loop
            }
         }
         if (isConsolidated) {                    //--- Confirm consolidation
            rangeHighestHigh.price = high(startBarIndex); //--- Set initial high
            rangeHighestHigh.index = startBarIndex; //--- Set high index
            for (int i = startBarIndex + 1; i < startBarIndex + consolidationBars; i++) { //--- Find highest high
               if (high(i) > rangeHighestHigh.price) { //--- Check higher high
                  rangeHighestHigh.price = high(i); //--- Update highest high
                  rangeHighestHigh.index = i;    //--- Update high index
               }
            }
            rangeLowestLow.price = low(startBarIndex); //--- Set initial low
            rangeLowestLow.index = startBarIndex; //--- Set low index
            for (int i = startBarIndex + 1; i < startBarIndex + consolidationBars; i++) { //--- Find lowest low
               if (low(i) < rangeLowestLow.price) { //--- Check lower low
                  rangeLowestLow.price = low(i);  //--- Update lowest low
                  rangeLowestLow.index = i;       //--- Update low index
               }
            }
            Print("Consolidation range established: Highest High = ", rangeHighestHigh.price, " at index ", rangeHighestHigh.index, " and Lowest Low = ", rangeLowestLow.price, " at index ", rangeLowestLow.index); //--- Log range
         }
      } else {                                    //--- Range already set
         double currentHigh = high(1);            //--- Get current high
         double currentLow = low(1);              //--- Get current low
         if (currentHigh <= rangeHighestHigh.price && currentLow >= rangeLowestLow.price) { //--- Check within range
            Print("Range extended: High = ", currentHigh, ", Low = ", currentLow); //--- Log range extension
         } else {                                 //--- Outside range
            Print("No extension: Bar outside range."); //--- Log no extension
         }
      }
   }
}
```

We begin by defining a logic to first detect the consolidation ranges. In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, we track new bars by comparing the current bar count from [iBars](https://www.mql5.com/en/docs/series/ibars) with a static "previousBarCount", setting "isNewBar" to true and updating "previousBarCount" if a new bar is detected, or false to exit if not. We then retrieve the chart’s zoom scale with [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) using [CHART\_SCALE](https://www.mql5.com/en/docs/constants/chartconstants/charts_samples#chart_scale) and calculate a "dynamicFontSize" as 8 plus twice the scale for adaptive label sizing. If no breakout is detected ("isBreakoutDetected" is false) and no range is set ("rangeHighestHigh.price" and "rangeLowestLow.price" are 0), we check for consolidation by iterating through "consolidationBars" starting at "startBarIndex" 1, ensuring adjacent bars’ highs and lows differ by less than "maxConsolidationSpread \* [Point()](https://www.mql5.com/en/docs/check/point)" using [MathAbs](https://www.mql5.com/en/docs/math/mathabs), setting "isConsolidated" to false if exceeded.

If consolidated, we set "rangeHighestHigh.price" and "rangeLowestLow.price" to the high and low of "startBarIndex" using "high" and "low", then iterate through "consolidationBars" to update them to the highest high and lowest low with their indices, logging the range with the [Print](https://www.mql5.com/en/docs/common/print) function. If a range exists, we check if the current bar’s high and low ("high(1)", "low(1)") are within "rangeHighestHigh.price" and "rangeLowestLow.price", logging the extension if true, or no extension if outside. We can now use the prices to detect, visualize, and manage the order blocks before we use them for further analysis, because we need to invalidate them first before they become breaker blocks.

```
if (rangeHighestHigh.price > 0 && rangeLowestLow.price > 0) { //--- Check if range defined
   double currentClosePrice = close(1);        //--- Get current close price
   if (currentClosePrice > rangeHighestHigh.price) { //--- Check upward breakout
      Print("Upward breakout at ", currentClosePrice, " > ", rangeHighestHigh.price); //--- Log breakout
      isBreakoutDetected = true;               //--- Set breakout flag
   } else if (currentClosePrice < rangeLowestLow.price) { //--- Check downward breakout
      Print("Downward breakout at ", currentClosePrice, " < ", rangeLowestLow.price); //--- Log breakout
      isBreakoutDetected = true;               //--- Set breakout flag
   }
}
if (isBreakoutDetected) {                      //--- Process breakout
   Print("Breakout detected. Resetting for the next range."); //--- Log reset
   breakoutBarNumber = 1;                      //--- Set breakout bar index
   breakoutTimestamp = TimeCurrent();          //--- Set breakout timestamp
   lastImpulseHigh = rangeHighestHigh.price;   //--- Store high for impulse check
   lastImpulseLow = rangeLowestLow.price;      //--- Store low for impulse check
   isBreakoutDetected = false;                 //--- Reset breakout flag
   rangeHighestHigh.price = 0;                 //--- Clear highest high
   rangeHighestHigh.index = 0;                 //--- Clear high index
   rangeLowestLow.price = 0;                   //--- Clear lowest low
   rangeLowestLow.index = 0;                   //--- Clear low index
}
if (breakoutBarNumber >= 0 && TimeCurrent() > breakoutTimestamp + barsToWaitAfterBreakout * PeriodSeconds()) { //--- Check impulse window
   double impulseRange = lastImpulseHigh - lastImpulseLow; //--- Calculate impulse range
   double impulseThresholdPrice = impulseRange * impulseMultiplier; //--- Calculate impulse threshold
   isBullishImpulse = false;                   //--- Reset bullish impulse flag
   isBearishImpulse = false;                   //--- Reset bearish impulse flag
   for (int i = 1; i <= barsToWaitAfterBreakout; i++) { //--- Check bars for impulse
      double closePrice = close(i);            //--- Get close price
      if (closePrice >= lastImpulseHigh + impulseThresholdPrice) { //--- Check bullish impulse
         isBullishImpulse = true;              //--- Set bullish impulse flag
         Print("Impulsive upward move: ", closePrice, " >= ", lastImpulseHigh + impulseThresholdPrice); //--- Log bullish impulse
         break;                                //--- Exit loop
      } else if (closePrice <= lastImpulseLow - impulseThresholdPrice) { //--- Check bearish impulse
         isBearishImpulse = true;              //--- Set bearish impulse flag
         Print("Impulsive downward move: ", closePrice, " <= ", lastImpulseLow - impulseThresholdPrice); //--- Log bearish impulse
         break;                                //--- Exit loop
      }
   }
   if (!isBullishImpulse && !isBearishImpulse) { //--- Check no impulse
      Print("No impulsive movement detected."); //--- Log no impulse
   }
   bool isOrderBlockValid = isBearishImpulse || isBullishImpulse; //--- Validate order block
   if (isOrderBlockValid) {                    //--- Process valid order block
      datetime blockStartTime = iTime(_Symbol, _Period, consolidationBars + barsToWaitAfterBreakout + 1); //--- Set block start time
      double blockTopPrice = lastImpulseHigh;  //--- Set block top price
      int visibleBarsOnChart = (int)ChartGetInteger(0, CHART_VISIBLE_BARS); //--- Get visible bars
      datetime blockEndTime = blockStartTime + (visibleBarsOnChart / 1) * PeriodSeconds(); //--- Set block end time
      double blockBottomPrice = lastImpulseLow; //--- Set block bottom price
      string blockName = OB_Prefix + "(" + TimeToString(blockStartTime) + ")"; //--- Generate block name
      color blockColor = isBullishImpulse ? bullishColor : bearishColor; //--- Set block color
      string blockLabel = isBullishImpulse ? "Bullish Order Block" : "Bearish Order Block"; //--- Set block label
      string blockType = isBullishImpulse ? "OB-bullish" : "OB-bearish"; //--- Set block type
      if (ObjectFind(0, blockName) < 0) {      //--- Check if block exists
         ObjectCreate(0, blockName, OBJ_RECTANGLE, 0, blockStartTime, blockTopPrice, blockEndTime, blockBottomPrice); //--- Create block rectangle
         ObjectSetInteger(0, blockName, OBJPROP_TIME, 0, blockStartTime); //--- Set start time
         ObjectSetDouble(0, blockName, OBJPROP_PRICE, 0, blockTopPrice); //--- Set top price
         ObjectSetInteger(0, blockName, OBJPROP_TIME, 1, blockEndTime); //--- Set end time
         ObjectSetDouble(0, blockName, OBJPROP_PRICE, 1, blockBottomPrice); //--- Set bottom price
         ObjectSetInteger(0, blockName, OBJPROP_FILL, true); //--- Enable fill
         ObjectSetInteger(0, blockName, OBJPROP_COLOR, blockColor); //--- Set block color
         ObjectSetInteger(0, blockName, OBJPROP_BACK, false); //--- Set to foreground
         datetime labelTime = blockStartTime + (blockEndTime - blockStartTime) / 2; //--- Calculate label time
         double labelPrice = (blockTopPrice + blockBottomPrice) / 2; //--- Calculate label price
         string labelObjectName = blockName + " Label"; //--- Generate label name
         ObjectCreate(0, labelObjectName, OBJ_TEXT, 0, labelTime, labelPrice); //--- Create label
         ObjectSetString(0, labelObjectName, OBJPROP_TEXT, blockLabel); //--- Set label text
         ObjectSetInteger(0, labelObjectName, OBJPROP_COLOR, labelTextColor); //--- Set label color
         ObjectSetInteger(0, labelObjectName, OBJPROP_FONTSIZE, dynamicFontSize); //--- Set label font size
         ObjectSetInteger(0, labelObjectName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set label anchor
         ChartRedraw(0);                       //--- Redraw chart
         ArrayResize(blockNames, ArraySize(blockNames) + 1); //--- Resize block names array
         blockNames[ArraySize(blockNames) - 1] = blockName; //--- Add block name
         ArrayResize(blockEndTimes, ArraySize(blockEndTimes) + 1); //--- Resize end times array
         blockEndTimes[ArraySize(blockEndTimes) - 1] = blockEndTime; //--- Add end time
         ArrayResize(invalidatedStatus, ArraySize(invalidatedStatus) + 1); //--- Resize invalidated status
         invalidatedStatus[ArraySize(invalidatedStatus) - 1] = false; //--- Set invalidated status
         ArrayResize(blockTypes, ArraySize(blockTypes) + 1); //--- Resize block types array
         blockTypes[ArraySize(blockTypes) - 1] = blockType; //--- Add block type
         ArrayResize(movedAwayStatus, ArraySize(movedAwayStatus) + 1); //--- Resize moved away status
         movedAwayStatus[ArraySize(movedAwayStatus) - 1] = false; //--- Set moved away status
         ArrayResize(retestedStatus, ArraySize(retestedStatus) + 1); //--- Resize retested status
         retestedStatus[ArraySize(retestedStatus) - 1] = false; //--- Set retested status
         ArrayResize(blockLabels, ArraySize(blockLabels) + 1); //--- Resize block labels array
         blockLabels[ArraySize(blockLabels) - 1] = labelObjectName; //--- Add label name
         ArrayResize(creationTimes, ArraySize(creationTimes) + 1); //--- Resize creation times
         creationTimes[ArraySize(creationTimes) - 1] = time(1); //--- Set creation time
         ArrayResize(invalidationTimes, ArraySize(invalidationTimes) + 1); //--- Resize invalidation times
         invalidationTimes[ArraySize(invalidationTimes) - 1] = 0; //--- Set invalidation time
         ArrayResize(invalidationSwings, ArraySize(invalidationSwings) + 1); //--- Resize invalidation swings
         invalidationSwings[ArraySize(invalidationSwings) - 1] = 0.0; //--- Set invalidation swing
         Print("Order Block created: ", blockName); //--- Log block creation
      }
   }
   breakoutBarNumber = -1;                     //--- Reset breakout bar
   breakoutTimestamp = 0;                      //--- Reset breakout timestamp
   lastImpulseHigh = 0;                        //--- Reset impulse high
   lastImpulseLow = 0;                         //--- Reset impulse low
   isBullishImpulse = false;                   //--- Reset bullish impulse
   isBearishImpulse = false;                   //--- Reset bearish impulse
}
```

Here, we implement the breakout detection and order block creation logic. First, we check if a consolidation range is defined ("rangeHighestHigh.price" and "rangeLowestLow.price" > 0), retrieving the current bar’s close price with "close(1)"; if it exceeds "rangeHighestHigh.price", we log an upward breakout and set "isBreakoutDetected" to true, or if below "rangeLowestLow.price", we log a downward breakout and set the flag. Then, if "isBreakoutDetected" is true, we log the breakout, set "breakoutBarNumber" to 1 and "breakoutTimestamp" to [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent), store "lastImpulseHigh" and "lastImpulseLow", and reset range variables and the breakout flag.

Next, if "breakoutBarNumber" is non-negative and the current time exceeds "breakoutTimestamp + barsToWaitAfterBreakout \* [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds)", we calculate the "impulseRange" ("lastImpulseHigh - lastImpulseLow") and threshold ("impulseRange \* impulseMultiplier"), checking bars within "barsToWaitAfterBreakout" for a close price exceeding "lastImpulseHigh + impulseThresholdPrice" (setting "isBullishImpulse") or below "lastImpulseLow - impulseThresholdPrice" (setting "isBearishImpulse").

If no impulse is detected, we log it; if valid ("isBearishImpulse" or "isBullishImpulse"), we create an order block using [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) ( [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_rectangle)) with "blockStartTime" from [iTime](https://www.mql5.com/en/docs/series/itime), top/bottom prices from "lastImpulseHigh"/"lastImpulseLow", and end time based on " [ChartGetInteger(CHART\_VISIBLE\_BARS)](https://www.mql5.com/en/docs/chart_operations/chartgetinteger)", setting properties like "OBJPROP\_FILL" and "OBJPROP\_COLOR" ("bullishColor" or "bearishColor"), adding a centered label with "blockLabel" via "OBJ\_TEXT", and updating arrays "blockNames", "blockEndTimes", "invalidatedStatus", "blockTypes", "movedAwayStatus", "retestedStatus", "blockLabels", "creationTimes", and "invalidationSwings". Finally, we reset breakout variables. This creates a system for detecting breakouts and visualizing order blocks. Upon compilation, we get the following outcome.

![INITIAL ORDER BLOCKS DETECTION](https://c.mql5.com/2/171/Screenshot_2025-09-20_163253.png)

Now that we can detect the initial order blocks, let us define some logic so that when there is a respective price breakout, we mark them as invalidated, and confirm the invalidation via price action swing points. We will darken their color for distinction.

```
for (int j = ArraySize(blockNames) - 1; j >= 0; j--) { //--- Iterate through blocks
   string currentBlockName = blockNames[j];    //--- Get current block name
   bool doesBlockExist = false;                //--- Initialize block existence flag
   double blockHigh = ObjectGetDouble(0, currentBlockName, OBJPROP_PRICE, 0); //--- Get block high
   double blockLow = ObjectGetDouble(0, currentBlockName, OBJPROP_PRICE, 1); //--- Get block low
   datetime blockStartTime = (datetime)ObjectGetInteger(0, currentBlockName, OBJPROP_TIME, 0); //--- Get block start
   datetime blockEndTime = (datetime)ObjectGetInteger(0, currentBlockName, OBJPROP_TIME, 1); //--- Get block end
   color blockCurrentColor = (color)ObjectGetInteger(0, currentBlockName, OBJPROP_COLOR); //--- Get block color
   if (time(1) < blockEndTime) {               //--- Check if block still valid
      doesBlockExist = true;                   //--- Set block exists
   }
   if (StringFind(blockTypes[j], "OB-") == 0 && !invalidatedStatus[j]) { //--- Check valid order block
      bool invalidated = false;                //--- Initialize invalidation flag
      string newBlockType = "";                //--- Initialize new block type
      color invalidatedColor = clrNONE;        //--- Initialize invalidated color
      string newLabel = "";                    //--- Initialize new label
      bool isForBullishBB = false;             //--- Initialize bullish breaker block flag
      double breakPrice = 0.0;                 //--- Initialize break price
      int arrowCode = 0;                       //--- Initialize arrow code
      int anchor = 0;                          //--- Initialize anchor
      if (blockTypes[j] == "OB-bearish" && close(1) > blockHigh) { //--- Check bearish block invalidation
         isForBullishBB = true;                //--- Set bullish breaker block
         breakPrice = blockHigh;               //--- Set break price
         arrowCode = 233;                      //--- Set upward arrow
         anchor = ANCHOR_BOTTOM;               //--- Set bottom anchor
         newBlockType = "Invalidated-bearish"; //--- Set invalidated type
         invalidatedColor = DarkenColor(bearishColor); //--- Darken bearish color
         newLabel = "Invalidated Bearish Order Block"; //--- Set invalidated label
      } else if (blockTypes[j] == "OB-bullish" && close(1) < blockLow) { //--- Check bullish block invalidation
         isForBullishBB = false;               //--- Set bearish breaker block
         breakPrice = blockLow;                //--- Set break price
         arrowCode = 234;                      //--- Set downward arrow
         anchor = ANCHOR_TOP;                  //--- Set top anchor
         newBlockType = "Invalidated-bullish"; //--- Set invalidated type
         invalidatedColor = DarkenColor(bullishColor); //--- Darken bullish color
         newLabel = "Invalidated Bullish Order Block"; //--- Set invalidated label
      } else {                                 //--- No invalidation
         continue;                             //--- Skip to next block
      }
      bool validSwingForInvalidation = true;   //--- Assume valid swing
      int swingShift = -1;                     //--- Initialize swing shift
      double swingPrice = 0.0;                 //--- Initialize swing price
      if (enableSwingValidation) {             //--- Check swing validation
         int creationShift = iBarShift(_Symbol, _Period, creationTimes[j], false); //--- Get creation bar shift
         if (creationShift > 1) {              //--- Ensure enough bars
            double extreme = isForBullishBB ? blockLow : blockHigh; //--- Set extreme price
            bool isBearishOB = isForBullishBB; //--- Set bearish OB flag
            if (isBearishOB) {                 //--- Handle bearish OB
               double minLow = extreme;        //--- Initialize minimum low
               for (int k = creationShift - 1; k > 1; k--) { //--- Find lower low
                  if (low(k) < minLow) {       //--- Check lower low
                     minLow = low(k);          //--- Update minimum low
                     swingShift = k;           //--- Update swing shift
                  }
               }
               validSwingForInvalidation = minLow < extreme; //--- Validate swing
               swingPrice = minLow;            //--- Set swing price
            } else {                           //--- Handle bullish OB
               double maxHigh = extreme;       //--- Initialize maximum high
               for (int k = creationShift - 1; k > 1; k--) { //--- Find higher high
                  if (high(k) > maxHigh) {     //--- Check higher high
                     maxHigh = high(k);        //--- Update maximum high
                     swingShift = k;           //--- Update swing shift
                  }
               }
               validSwingForInvalidation = maxHigh > extreme; //--- Validate swing
               swingPrice = maxHigh;           //--- Set swing price
            }
         } else {                              //--- Insufficient bars
            validSwingForInvalidation = false; //--- Invalidate swing
         }
      }
      if (validSwingForInvalidation) {        //--- Confirm swing validation
         invalidated = true;                  //--- Set invalidated flag
      }
      if (invalidated) {                      //--- Process invalidation
         ObjectSetInteger(0, currentBlockName, OBJPROP_COLOR, invalidatedColor); //--- Update block color
         ObjectDelete(0, blockLabels[j]);     //--- Delete old label
         datetime labelTime = blockStartTime + (blockEndTime - blockStartTime) / 2; //--- Calculate new label time
         double labelPrice = (blockHigh + blockLow) / 2; //--- Calculate new label price
         string newLabelObjectName = currentBlockName + " Label"; //--- Generate new label name
         ObjectCreate(0, newLabelObjectName, OBJ_TEXT, 0, labelTime, labelPrice); //--- Create new label
         ObjectSetString(0, newLabelObjectName, OBJPROP_TEXT, newLabel); //--- Set label text
         ObjectSetInteger(0, newLabelObjectName, OBJPROP_COLOR, labelTextColor); //--- Set label color
         ObjectSetInteger(0, newLabelObjectName, OBJPROP_FONTSIZE, dynamicFontSize); //--- Set label font size
         ObjectSetInteger(0, newLabelObjectName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set label anchor
         string arrowName = currentBlockName + "_break_arrow"; //--- Generate arrow name
         if (ObjectFind(0, arrowName) < 0) {  //--- Check if arrow exists
            ObjectCreate(0, arrowName, OBJ_ARROW, 0, time(1), breakPrice); //--- Create break arrow
            ObjectSetInteger(0, arrowName, OBJPROP_ARROWCODE, arrowCode); //--- Set arrow code
            ObjectSetInteger(0, arrowName, OBJPROP_ANCHOR, anchor); //--- Set arrow anchor
            ObjectSetInteger(0, arrowName, OBJPROP_COLOR, invalidatedColor); //--- Set arrow color
         }
         if (enableSwingValidation && showSwingPoints && swingShift > 0) { //--- Check swing point display
            string swingLabelName = currentBlockName + "_invalid_swing"; //--- Generate swing label name
            if (ObjectFind(0, swingLabelName) < 0) { //--- Check if swing label exists
               datetime swingTime = time(swingShift); //--- Get swing time
               ObjectCreate(0, swingLabelName, OBJ_TEXT, 0, swingTime, swingPrice); //--- Create swing label
               string swingText = isForBullishBB ? "LL" : "HH"; //--- Set swing text
               ObjectSetString(0, swingLabelName, OBJPROP_TEXT, swingText); //--- Set swing label text
               ObjectSetInteger(0, swingLabelName, OBJPROP_COLOR, swingLabelColor); //--- Set swing label color
               ObjectSetInteger(0, swingLabelName, OBJPROP_FONTSIZE, swingFontSize); //--- Set swing label font size
               ObjectSetInteger(0, swingLabelName, OBJPROP_ANCHOR, isForBullishBB ? ANCHOR_LEFT_UPPER : ANCHOR_LEFT_LOWER); //--- Set swing label anchor
            }
         }
         ChartRedraw(0);                       //--- Redraw chart
         invalidatedStatus[j] = true;          //--- Set invalidated status
         blockTypes[j] = newBlockType;         //--- Update block type
         movedAwayStatus[j] = false;           //--- Reset moved away status
         retestedStatus[j] = false;            //--- Reset retested status
         blockLabels[j] = newLabelObjectName;  //--- Update label name
         invalidationTimes[j] = time(1);       //--- Set invalidation time
         invalidationSwings[j] = isForBullishBB ? high(1) : low(1); //--- Set invalidation swing
         Print("Order Block invalidated: ", currentBlockName); //--- Log invalidation
      }
   }
   if (!doesBlockExist) {                     //--- Check if block expired
      ArrayRemove(blockNames, j, 1);          //--- Remove block name
      ArrayRemove(blockEndTimes, j, 1);       //--- Remove end time
      ArrayRemove(invalidatedStatus, j, 1);   //--- Remove invalidated status
      ArrayRemove(blockTypes, j, 1);          //--- Remove block type
      ArrayRemove(movedAwayStatus, j, 1);     //--- Remove moved away status
      ArrayRemove(retestedStatus, j, 1);      //--- Remove retested status
      ArrayRemove(blockLabels, j, 1);         //--- Remove label name
      ArrayRemove(creationTimes, j, 1);       //--- Remove creation time
      ArrayRemove(invalidationTimes, j, 1);   //--- Remove invalidation time
      ArrayRemove(invalidationSwings, j, 1);  //--- Remove invalidation swing
      Print("Removed expired block at index ", j); //--- Log block removal
   }
}
```

To implement the logic for managing and invalidating order blocks, we iterate backward through "blockNames" to process each block, retrieving its high and low prices with [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) and start/end times with [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger), marking it as existing if the current bar’s time ("time(1)") is before its end time. For valid order blocks (type starting with "OB-" and not invalidated), we check invalidation: if "OB-bearish" and the close price ("close(1)") exceeds the block’s high, we set up a bullish breaker block with an upward arrow (code 233) using "ObjectCreate" ( [OBJ\_ARROW](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) and a darkened color from "DarkenColor"; if "OB-bullish" and the close is below the block’s low, we set a bearish breaker block with a downward arrow (code 234). MQL5 offers a vast list of [Wingdings codes](https://www.mql5.com/en/docs/constants/objectconstants/wingdings) as below, so you can use them to your liking.

![MQL5 WINGDINGS](https://c.mql5.com/2/171/C_MQL5_WINGDINGS.png)

If "enableSwingValidation" is true, we validate the block by checking for a lower low (for bearish) or higher high (for bullish) since creation using [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) and "low"/"high", updating the block’s color and label with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) and [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) if valid. If "showSwingPoints" is enabled, we add a swing label ("LL" or "HH") with "ObjectCreate" at the swing’s time and price. If invalidated, we update block states with "invalidatedStatus", "blockTypes", and "invalidationTimes", log the invalidation, and reset retest status. If a block doesn’t exist, we use [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove) to remove its states, like "invalidatedStatus", from storage arrays and log the removal, then redraw the chart with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function. Upon compilation, you should get something as follows.

![INVALIDATED ORDER BLOCKS](https://c.mql5.com/2/171/Screenshot_2025-09-20_164526.png)

Now that we have the second step for invalidation being complete, we can graduate to the next step, where we keep track and wait for the price to retest our invalidated order blocks and mark them as breaker blocks.

```
for (int j = ArraySize(blockNames) - 1; j >= 0; j--) { //--- Iterate invalidated blocks
   if (StringFind(blockTypes[j], "Invalidated-") != 0) continue; //--- Skip non-invalidated
   string currentBlockName = blockNames[j];    //--- Get current block name
   double blockHigh = ObjectGetDouble(0, currentBlockName, OBJPROP_PRICE, 0); //--- Get block high
   double blockLow = ObjectGetDouble(0, currentBlockName, OBJPROP_PRICE, 1); //--- Get block low
   datetime blockStartTime = (datetime)ObjectGetInteger(0, currentBlockName, OBJPROP_TIME, 0); //--- Get block start
   datetime blockEndTime = (datetime)ObjectGetInteger(0, currentBlockName, OBJPROP_TIME, 1); //--- Get block end
   bool isForBullishBB = (blockTypes[j] == "Invalidated-bearish"); //--- Check for bullish breaker block
   datetime currentBarTime = time(1);          //--- Get current bar time
   if (currentBarTime <= invalidationTimes[j]) continue; //--- Skip if same or earlier bar
   if (!movedAwayStatus[j]) {                  //--- Check if not moved away
      if (isForBullishBB && close(1) > blockHigh + moveAwayDistance * _Point) { //--- Check bullish move away
         movedAwayStatus[j] = true;           //--- Set moved away
         Print("Moved away for bullish BB setup: ", currentBlockName); //--- Log move away
      } else if (!isForBullishBB && close(1) < blockLow - moveAwayDistance * _Point) { //--- Check bearish move away
         movedAwayStatus[j] = true;           //--- Set moved away
         Print("Moved away for bearish BB setup: ", currentBlockName); //--- Log move away
      }
   }
   if (movedAwayStatus[j] && !retestedStatus[j]) { //--- Check for retest
      bool retestCondition = false;            //--- Initialize retest condition
      if (isForBullishBB && low(1) <= blockHigh && close(1) > blockHigh) { //--- Check bullish retest
         retestCondition = true;               //--- Set retest condition
      } else if (!isForBullishBB && high(1) >= blockLow && close(1) < blockLow) { //--- Check bearish retest
         retestCondition = true;               //--- Set retest condition
      }
      bool validSwingForRetest = true;         //--- Assume valid swing
      int swingShift = -1;                     //--- Initialize swing shift
      double swingPrice = 0.0;                 //--- Initialize swing price
      if (enableSwingValidation && retestCondition) { //--- Check swing validation
         int invalidShift = iBarShift(_Symbol, _Period, invalidationTimes[j], false); //--- Get invalidation shift
         if (invalidShift > 1) {               //--- Ensure enough bars
            double extreme = invalidationSwings[j]; //--- Get invalidation swing
            if (isForBullishBB) {              //--- Handle bullish breaker block
               double maxHigh = extreme;       //--- Initialize maximum high
               for (int k = invalidShift - 1; k > 1; k--) { //--- Find higher high
                  if (high(k) > maxHigh) {     //--- Check higher high
                     maxHigh = high(k);        //--- Update maximum high
                     swingShift = k;           //--- Update swing shift
                  }
               }
               validSwingForRetest = maxHigh > extreme; //--- Validate swing
               swingPrice = maxHigh;           //--- Set swing price
            } else {                           //--- Handle bearish breaker block
               double minLow = extreme;        //--- Initialize minimum low
               for (int k = invalidShift - 1; k > 1; k--) { //--- Find lower low
                  if (low(k) < minLow) {       //--- Check lower low
                     minLow = low(k);          //--- Update minimum low
                     swingShift = k;           //--- Update swing shift
                  }
               }
               validSwingForRetest = minLow < extreme; //--- Validate swing
               swingPrice = minLow;            //--- Set swing price
            }
         } else {                              //--- Insufficient bars
            validSwingForRetest = false;       //--- Invalidate swing
         }
      }
   }
}
```

To implement the retest detection logic for invalidated breaker blocks, we iterate backward through "blockNames" for blocks with "Invalidated-" in "blockTypes", retrieving high and low prices with [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) and start/end times with [ObjectGetInteger](https://www.mql5.com/en/docs/objects/ObjectGetInteger), checking if the block is a bullish breaker block ("Invalidated-bearish"). If the current bar’s time ("time(1)") is not after "invalidationTimes", we skip to the next block. For blocks not yet moved away ("movedAwayStatus" false), we check if the close price ("close(1)") exceeds "blockHigh + moveAwayDistance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" for bullish or falls below "blockLow - moveAwayDistance \* \_Point" for bearish, setting "movedAwayStatus" to true.

For blocks that have moved away but not retested ("retestedStatus" false), we set a bullish retest if "low(1)" reaches "blockHigh" and "close(1)" is above, or a bearish retest if "high(1)" reaches "blockLow" and "close(1)" is below. If "enableSwingValidation" and a retest condition are met, we validate swings using [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) to get the invalidation bar, checking for a higher high ("high") for bullish or lower low ("low") for bearish since invalidation, setting "validSwingForRetest" and "swingPrice" accordingly, creating a system for identifying valid retest opportunities after price movement. We can track the retests, mark the blocks as breaker blocks, change their color for distinction, and open positions. Here is the logic we use to achieve that.

```
if (retestCondition && validSwingForRetest) { //--- Confirm retest and swing
   if (enableTrading) {                  //--- Check trading enabled
      double entryPrice = 0.0;           //--- Initialize entry price
      double stopLossPrice = 0.0;        //--- Initialize stop loss
      double takeProfitPrice = 0.0;      //--- Initialize take profit
      if (isForBullishBB) {              //--- Handle bullish trade
         entryPrice = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Set entry at ask
         stopLossPrice = entryPrice - stopLossDistance * _Point; //--- Set stop loss
         takeProfitPrice = entryPrice + takeProfitDistance * _Point; //--- Set take profit
         obj_Trade.Buy(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice); //--- Execute buy trade
         Print("Buy trade on bullish BB retest: ", currentBlockName); //--- Log buy trade
      } else {                           //--- Handle bearish trade
         entryPrice = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Set entry at bid
         stopLossPrice = entryPrice + stopLossDistance * _Point; //--- Set stop loss
         takeProfitPrice = entryPrice - takeProfitDistance * _Point; //--- Set take profit
         obj_Trade.Sell(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice); //--- Execute sell trade
         Print("Sell trade on bearish BB retest: ", currentBlockName); //--- Log sell trade
      }
   }
   color bbColor = isForBullishBB ? clrBlueViolet : clrOrange; //--- Set breaker block color
   ObjectSetInteger(0, currentBlockName, OBJPROP_COLOR, bbColor); //--- Update block color
   ObjectDelete(0, blockLabels[j]);     //--- Delete old label
   string newLabel = isForBullishBB ? "Bullish Breaker Block" : "Bearish Breaker Block"; //--- Set new label
   datetime labelTime = blockStartTime + (blockEndTime - blockStartTime) / 2; //--- Calculate label time
   double labelPrice = (blockHigh + blockLow) / 2; //--- Calculate label price
   string newLabelObjectName = currentBlockName + " Label"; //--- Generate new label name
   ObjectCreate(0, newLabelObjectName, OBJ_TEXT, 0, labelTime, labelPrice); //--- Create new label
   ObjectSetString(0, newLabelObjectName, OBJPROP_TEXT, newLabel); //--- Set label text
   ObjectSetInteger(0, newLabelObjectName, OBJPROP_COLOR, labelTextColor); //--- Set label color
   ObjectSetInteger(0, newLabelObjectName, OBJPROP_FONTSIZE, dynamicFontSize); //--- Set label font size
   ObjectSetInteger(0, newLabelObjectName, OBJPROP_ANCHOR, ANCHOR_CENTER); //--- Set label anchor
   if (enableSwingValidation && showSwingPoints && swingShift > 0) { //--- Check swing point display
      string swingLabelName = currentBlockName + "_retest_swing"; //--- Generate swing label name
      if (ObjectFind(0, swingLabelName) < 0) { //--- Check if swing label exists
         datetime swingTime = time(swingShift); //--- Get swing time
         ObjectCreate(0, swingLabelName, OBJ_TEXT, 0, swingTime, swingPrice); //--- Create swing label
         string swingText = isForBullishBB ? "HH" : "LL"; //--- Set swing text
         ObjectSetString(0, swingLabelName, OBJPROP_TEXT, swingText); //--- Set swing label text
         ObjectSetInteger(0, swingLabelName, OBJPROP_COLOR, swingLabelColor); //--- Set swing label color
         ObjectSetInteger(0, swingLabelName, OBJPROP_FONTSIZE, swingFontSize); //--- Set swing label font size
         ObjectSetInteger(0, swingLabelName, OBJPROP_ANCHOR, isForBullishBB ? ANCHOR_LEFT_LOWER : ANCHOR_LEFT_UPPER); //--- Set swing label anchor
      }
   }
   ChartRedraw(0);                       //--- Redraw chart
   blockTypes[j] = isForBullishBB ? "BB-bullish" : "BB-bearish"; //--- Update block type
   retestedStatus[j] = true;             //--- Set retested status
   blockLabels[j] = newLabelObjectName;  //--- Update label name
   Print("Converted to ", newLabel, ": ", currentBlockName); //--- Log conversion
}
```

Finally, we implement the trading and visualization logic for retested breaker blocks. If a retest is confirmed ("retestCondition" and "validSwingForRetest" are true) and trading is enabled ("enableTrading"), we execute trades: for a bullish breaker block ("isForBullishBB"), we set the entry at the ask price using [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble), calculate stop loss ("stopLossDistance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" below entry) and take profit ("takeProfitDistance \* \_Point" above entry), and execute a buy with "obj\_Trade.Buy"; for bearish, we use the bid price, set stop loss above and take profit below, and execute a sell with "obj\_Trade.Sell", logging accordingly.

Then, we update the block’s appearance by setting its color to [clrBlueViolet](https://www.mql5.com/en/docs/constants/objectconstants/webcolors) for bullish or "clrOrange" for bearish with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), delete the old label with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete), and create a new [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text) label ("Bullish Breaker Block" or "Bearish Breaker Block") at the block’s midpoint using "ObjectCreate" with "labelTextColor" and "dynamicFontSize". If "enableSwingValidation" and "showSwingPoints" are true with a valid "swingShift", we add a swing label ("HH" for bullish, "LL" for bearish) at the swing’s time and price using [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) with "swingLabelColor" and "swingFontSize". Finally, we update "blockTypes" to "BB-bullish" or "BB-bearish", set "retestedStatus" to true, update "blockLabels", log the conversion, and redraw the chart. Upon compilation, we have the following outcome.

![CONFIRMED BREAKER BLOCKS](https://c.mql5.com/2/171/Screenshot_2025-09-20_171130.png)

From the image, we can see that we can detect, visualize, and trade the breaker blocks accordingly. What we now need to do is add a trailing stop logic to maximize profits, and all will be done. We define a function for that to modularize the code.

```
//+------------------------------------------------------------------+
//| Apply trailing stop to open positions                            |
//+------------------------------------------------------------------+
void applyTrailingStop(double trailingPoints, CTrade &trade_object, int magicNo = 0) {
   double buyStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - trailingPoints * _Point, _Digits); //--- Calculate buy stop loss
   double sellStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + trailingPoints * _Point, _Digits); //--- Calculate sell stop loss
   for (int i = PositionsTotal() - 1; i >= 0; i--) { //--- Iterate through open positions
      ulong ticket = PositionGetTicket(i);         //--- Get position ticket
      if (ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && (magicNo == 0 || PositionGetInteger(POSITION_MAGIC) == magicNo)) { //--- Verify position
         if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY && buyStopLoss > PositionGetDouble(POSITION_PRICE_OPEN) && (buyStopLoss > PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL) == 0)) { //--- Check buy trailing
            trade_object.PositionModify(ticket, buyStopLoss, PositionGetDouble(POSITION_TP)); //--- Update buy stop loss
         } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL && sellStopLoss < PositionGetDouble(POSITION_PRICE_OPEN) && (sellStopLoss < PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL) == 0)) { //--- Check sell trailing
            trade_object.PositionModify(ticket, sellStopLoss, PositionGetDouble(POSITION_TP)); //--- Update sell stop loss
         }
      }
   }
}

//--- Call the function on every tick

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   if (enableTrailingStop) {                      //--- Check if trailing stop enabled
      applyTrailingStop(trailingStopPoints, obj_Trade, uniqueMagicNumber); //--- Apply trailing stop
   }

   //---

}
```

Here, we implement the trailing stop functionality and integrate it into the event-driven logic. First, we develop the "applyTrailingStop" function, which calculates a buy stop loss as the current bid price ( [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double)) minus "trailingPoints \* \_Point" and a sell stop loss as the ask price ("SYMBOL\_ASK") plus "trailingPoints \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)", both normalized with [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) to the symbol’s digits. We iterate backward through open positions using [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal), retrieving each position’s ticket with [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket)" and verifying it matches the symbol and "magicNo" (if non-zero) with the [PositionGetString](https://www.mql5.com/en/docs/trading/positiongetstring) and "PositionGetInteger" functions.

For buy positions ( [POSITION\_TYPE\_BUY](https://www.mql5.com/en/docs/constants/tradingconstants/positionproperties#enum_position_type)), we check if "buyStopLoss" is above the open price (" [PositionGetDouble(POSITION\_PRICE\_OPEN)](https://www.mql5.com/en/docs/trading/positiongetdouble)") and higher than the current stop loss or unset, updating it with "trade\_object.PositionModify"; for sell positions, we ensure "sellStopLoss" is below the open price and lower than the current stop loss or unset, updating similarly. Then, in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, we check if "enableTrailingStop" is true and call "applyTrailingStop" with "trailingStopPoints", "obj\_Trade", and "uniqueMagicNumber" to manage open positions on every tick. Upon compilation, we get the following outcome.

![TRAILING STOP ENABLED](https://c.mql5.com/2/171/Screenshot_2025-09-20_172624.png)

From the image, we can see that the trailing stop is fully enabled when the price goes in our favour. Here is a unified test for both bearish and bullish breaker blocks.

![UNIFIED BREAKER BLOCKS GIF](https://c.mql5.com/2/171/BREAKER_BLOCKS_GIF.gif)

From the visualization, we can see that the program identifies and verifies all the entry conditions, and if validated, opens the respective position with the respective entry parameters, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/171/Screenshot_2025-09-20_175325.png)

Backtest report:

![REPORT](https://c.mql5.com/2/171/Screenshot_2025-09-20_175353.png)

### Conclusion

In conclusion, we’ve created a [breaker blocks](https://www.mql5.com/go?link=https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks "https://www.fluxcharts.com/articles/Trading-Concepts/Price-Action/breaker-blocks") trading system in MQL5 for identifying consolidation ranges, validating breaker blocks with swing points, and executing retest trades with customizable risk parameters and trailing stops. The system visualizes order and breaker blocks with dynamic labels and arrows, enhancing trade decision clarity.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

With this breaker block strategy, you’re equipped to capture price retest opportunities, ready for further refinement in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19638.zip "Download all attachments in the single ZIP archive")

[Breaker\_Block\_Strategy\_EA.mq5](https://www.mql5.com/en/articles/download/19638/Breaker_Block_Strategy_EA.mq5 "Download Breaker_Block_Strategy_EA.mq5")(40.33 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MQL5 Trading Tools (Part 12): Enhancing the Correlation Matrix Dashboard with Interactivity](https://www.mql5.com/en/articles/20962)
- [Creating Custom Indicators in MQL5 (Part 5): WaveTrend Crossover Evolution Using Canvas for Fog Gradients, Signal Bubbles, and Risk Management](https://www.mql5.com/en/articles/20815)
- [MQL5 Trading Tools (Part 11): Correlation Matrix Dashboard (Pearson, Spearman, Kendall) with Heatmap and Standard Modes](https://www.mql5.com/en/articles/20945)
- [Creating Custom Indicators in MQL5 (Part 4): Smart WaveTrend Crossover with Dual Oscillators](https://www.mql5.com/en/articles/20811)
- [Building AI-Powered Trading Systems in MQL5 (Part 8): UI Polish with Animations, Timing Metrics, and Response Management Tools](https://www.mql5.com/en/articles/20722)
- [Creating Custom Indicators in MQL5 (Part 3): Multi-Gauge Enhancements with Sector and Round Styles](https://www.mql5.com/en/articles/20719)
- [Creating Custom Indicators in MQL5 (Part 2): Building a Gauge-Style RSI Display with Canvas and Needle Mechanics](https://www.mql5.com/en/articles/20632)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/496512)**
(1)


![hiteshdaoya](https://c.mql5.com/avatar/avatar_na2.png)

**[hiteshdaoya](https://www.mql5.com/en/users/hiteshdaoya)**
\|
27 Nov 2025 at 08:06

Hello,

I tried adding the shared file but it is [not drawing](https://www.mql5.com/en/docs/constants/indicatorconstants/drawstyles#enum_plot_property_double "MQL5 documentation: Drawing Styles") or I can say nothing is happening on chart or no trades are taken. Kindly help with execution flow.

Thanks in advance


![Visual assessment and adjustment of trading in MetaTrader 5](https://c.mql5.com/2/113/Visual_assessment_and_adjustment_of_trading_in_MetaTrader_5____LOGO2.png)[Visual assessment and adjustment of trading in MetaTrader 5](https://www.mql5.com/en/articles/16952)

The strategy tester allows you to do more than just optimize your trading robot's parameters. I will show how to evaluate your account's trading history post-factum and make adjustments to your trading in the tester by changing the stop-losses of your open positions.

![From Novice to Expert: Backend Operations Monitor using MQL5](https://c.mql5.com/2/172/19649-from-novice-to-expert-backend-logo.png)[From Novice to Expert: Backend Operations Monitor using MQL5](https://www.mql5.com/en/articles/19649)

Using a ready-made solution in trading without concerning yourself with the internal workings of the system may sound comforting, but this is not always the case for developers. Eventually, an upgrade, misperformance, or unexpected error will arise, and it becomes essential to trace exactly where the issue originates to diagnose and resolve it quickly. Today’s discussion focuses on uncovering what normally happens behind the scenes of a trading Expert Advisor, and on developing a custom dedicated class for displaying and logging backend processes using MQL5. This gives both developers and traders the ability to quickly locate errors, monitor behavior, and access diagnostic information specific to each EA.

![Reimagining Classic Strategies (Part 16): Double Bollinger Band Breakouts](https://c.mql5.com/2/173/19418-reimagining-classic-strategies-logo__1.png)[Reimagining Classic Strategies (Part 16): Double Bollinger Band Breakouts](https://www.mql5.com/en/articles/19418)

This article walks the reader through a reimagined version of the classical Bollinger Band breakout strategy. It identifies key weaknesses in the original approach, such as its well-known susceptibility to false breakouts. The article aims to introduce a possible solution: the Double Bollinger Band trading strategy. This relatively lesser known approach supplements the weaknesses of the classical version and offers a more dynamic perspective on financial markets. It helps us overcome the old limitations defined by the original rules, providing traders with a stronger and more adaptive framework.

![MQL5 Trading Tools (Part 9): Developing a First Run User Setup Wizard for Expert Advisors with Scrollable Guide](https://c.mql5.com/2/172/19714-mql5-trading-tools-part-9-developing-logo__1.png)[MQL5 Trading Tools (Part 9): Developing a First Run User Setup Wizard for Expert Advisors with Scrollable Guide](https://www.mql5.com/en/articles/19714)

In this article, we develop an MQL5 First Run User Setup Wizard for Expert Advisors, featuring a scrollable guide with an interactive dashboard, dynamic text formatting, and visual controls like buttons and a checkbox allowing users to navigate instructions and configure trading parameters efficiently. Users of the program get to have insight of what the program is all about and what to do on the first run, more like an orientation model.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jgexacttwllqvapbyomszhwldycnlbcq&ssn=1769091926227207371&ssn_dr=0&ssn_sr=0&fv_date=1769091926&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19638&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2035)%3A%20Creating%20a%20Breaker%20Block%20Trading%20System%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909192667363457&fz_uniq=5049122304660710742&sv=2552)

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