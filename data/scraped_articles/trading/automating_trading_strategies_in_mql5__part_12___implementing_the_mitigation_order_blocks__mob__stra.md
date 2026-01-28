---
title: Automating Trading Strategies in MQL5 (Part 12): Implementing the Mitigation Order Blocks (MOB) Strategy
url: https://www.mql5.com/en/articles/17547
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-22T17:57:47.531490
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=cogwrgkctwxpdexkggwhvnxmeqecyiua&ssn=1769093866353424951&ssn_dr=0&ssn_sr=0&fv_date=1769093866&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17547&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2012)%3A%20Implementing%20the%20Mitigation%20Order%20Blocks%20(MOB)%20Strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909386634948949&fz_uniq=5049517879738608873&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 11)](https://www.mql5.com/en/articles/17350), we built a multi-level grid trading system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) to capitalize on market fluctuations. Now, in Part 12, we focus on implementing the [Mitigation Order Blocks (MOB)](https://www.mql5.com/go?link=https://theicttrader.com/2024/03/24/order-blocks-breaker-blocks-and-mitigation-blocks/ "https://theicttrader.com/2024/03/24/order-blocks-breaker-blocks-and-mitigation-blocks/") Strategy, a Smart Money concept that identifies key price zones where institutional orders are mitigated before significant market moves. We will cover the following topics:

1. [Strategy Blueprint](https://www.mql5.com/en/articles/17547#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17547#para2)
3. [Backtesting](https://www.mql5.com/en/articles/17547#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17547#para4)

By the end of this article, you will have a fully automated [mitigation order blocks](https://www.mql5.com/go?link=https://innercircletrader.net/tutorials/ict-mitigation-block-explained/ "https://innercircletrader.net/tutorials/ict-mitigation-block-explained/") trading system ready for trading. Let’s get started!

### Strategy Blueprint

To implement the [Mitigation Order Blocks Strategy](https://www.mql5.com/go?link=https://innercircletrader.net/tutorials/ict-mitigation-block-explained/ "https://innercircletrader.net/tutorials/ict-mitigation-block-explained/"), we will develop an automated system that detects, validates, and executes trades based on order block mitigation events. The strategy will focus on identifying institutional price zones where liquidity is absorbed before trend continuation. Our system will incorporate precise conditions for entry, stop-loss placement, and trade management to ensure efficiency and accuracy. We will structure the development as follows:

- **Order Block Identification** – The system will scan historical price action to detect bullish and bearish order blocks, filtering out weak zones based on volatility, liquidity grabs, and price imbalance.
- **Mitigation Validation** – We will program conditions that confirm a valid mitigation event, ensuring that the price revisits the order block and reacts with rejection signals such as wicks or momentum shifts.
- **Market Structure Confirmation** – The EA will analyze higher-timeframe trends and liquidity sweeps to ensure that the identified mitigation aligns with the broader market flow.
- **Trade Execution Rules** – Once mitigation is confirmed, the system will define precise entry points, dynamically calculate stop-loss levels based on order block structure, and set take-profit targets based on risk-reward parameters.
- **Risk and Money Management** – The strategy will integrate position sizing, drawdown protection, and exit strategies to manage trade risk effectively.

In a nutshell, here is a general visualization of the strategy.

![Mitigation Order Block](https://c.mql5.com/2/126/Screenshot_2025-03-20_121647.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades based on Mitigation Order Blocks Strategy"
#property strict

//--- Include the trade library for managing positions
#include <Trade/Trade.mqh>
CTrade obj_Trade;
```

We begin the implementation by including the trade library with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade/Trade.mqh>", which provides built-in functions for managing trade operations. We then initialize the trade object "obj\_Trade" using the "CTrade" class, allowing the Expert Advisor to execute buy and sell orders programmatically. This setup will ensure that trade execution is handled efficiently without requiring manual intervention. Then we can provide some inputs so the user can change and control the behavior from the [user interface](https://en.wikipedia.org/wiki/User_interface "https://en.wikipedia.org/wiki/User_interface") (UI).

```
//+------------------------------------------------------------------+
//| Input Parameters                                                 |
//+------------------------------------------------------------------+
input double tradeLotSize = 0.01;           // Trade size for each position
input bool enableTrading = true;            // Toggle to allow or disable trading
input bool enableTrailingStop = true;       // Toggle to enable or disable trailing stop
input double trailingStopPoints = 30;       // Distance in points for trailing stop
input double minProfitToTrail = 50;         // Minimum profit in points before trailing starts (not used yet)
input int uniqueMagicNumber = 12345;        // Unique identifier for EA trades
input int consolidationBars = 7;            // Number of bars to check for consolidation
input double maxConsolidationSpread = 50;   // Maximum allowed spread in points for consolidation
input int barsToWaitAfterBreakout = 3;      // Bars to wait after breakout before checking impulse
input double impulseMultiplier = 1.0;       // Multiplier for detecting impulsive moves
input double stopLossDistance = 1500;       // Stop loss distance in points
input double takeProfitDistance = 1500;     // Take profit distance in points
input color bullishOrderBlockColor = clrGreen;    // Color for bullish order blocks
input color bearishOrderBlockColor = clrRed;     // Color for bearish order blocks
input color mitigatedOrderBlockColor = clrGray;  // Color for mitigated order blocks
input color labelTextColor = clrBlack;           // Color for text labels
```

Here, we define [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameters to configure the program's behavior. "tradeLotSize" sets the position size, while "enableTrading" and "enableTrailingStop" control execution and trailing stops, with "trailingStopPoints" and "minProfitToTrail" refining stop logic. "uniqueMagicNumber" identifies trades, and consolidation is detected using "consolidationBars" and "maxConsolidationSpread". Breakouts are confirmed with "barsToWaitAfterBreakout" and "impulseMultiplier". "stopLossDistance" and "takeProfitDistance" manage risk, while "bullishOrderBlockColor", "bearishOrderBlockColor", "mitigatedOrderBlockColor", and "labelTextColor" handle chart visuals.

Lastly, we need to define some global variables that we will use for the overall system control.

```
//--- Struct to store price and index for highs and lows
struct PriceAndIndex {
   double price;  // Price value
   int    index;  // Bar index where this price occurs
};

//--- Global variables for tracking market state
PriceAndIndex rangeHighestHigh = {0, 0};    // Highest high in the consolidation range
PriceAndIndex rangeLowestLow = {0, 0};      // Lowest low in the consolidation range
bool isBreakoutDetected = false;            // Flag for when a breakout occurs
double lastImpulseLow = 0.0;                // Low price after breakout for impulse check
double lastImpulseHigh = 0.0;               // High price after breakout for impulse check
int breakoutBarNumber = -1;                 // Bar index where breakout happened
datetime breakoutTimestamp = 0;             // Time of the breakout
string orderBlockNames[];                   // Array of order block object names
datetime orderBlockEndTimes[];              // Array of order block end times
bool orderBlockMitigatedStatus[];           // Array tracking if order blocks are mitigated
bool isBullishImpulse = false;              // Flag for bullish impulsive move
bool isBearishImpulse = false;              // Flag for bearish impulsive move

#define OB_Prefix "OB REC "     // Prefix for order block object names
```

We first define the "PriceAndIndex" [struct](https://www.mql5.com/en/docs/basis/types/classes), which stores a "price" value and the "index" of the bar where this price occurs. This structure will be useful for tracking specific price points within a defined range. The global variables manage key aspects of market structure and breakout detection. "rangeHighestHigh" and "rangeLowestLow" will store the highest and lowest prices in the consolidation range, respectively, helping to define the boundaries of potential order blocks. "isBreakoutDetected" will act as a flag to indicate when a breakout has occurred, while "lastImpulseLow" and "lastImpulseHigh" will store the first low and high after a breakout, used to confirm impulsive moves.

"breakoutBarNumber" will record the bar index where the breakout happened, and "breakoutTimestamp" store the exact time of the breakout event. The arrays "orderBlockNames", "orderBlockEndTimes", and "orderBlockMitigatedStatus" will handle the identification, lifespan, and mitigation tracking of order blocks. The boolean flags "isBullishImpulse" and "isBearishImpulse" determine whether the breakout move qualifies as a bullish or bearish impulse. Finally, "OB\_Prefix" is a predefined string prefix defined by the macro [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) used when naming order block objects, ensuring consistency in graphical representation. With the variables, we are all set to begin the program logic.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   //--- Set the magic number for the trade object to identify EA trades
   obj_Trade.SetExpertMagicNumber(uniqueMagicNumber);
   return(INIT_SUCCEEDED);
}
```

Here, we initialize the Expert Advisor in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler. We set the expert magic number using the "SetExpertMagicNumber" method, ensuring that all trades executed by our EA are uniquely tagged, preventing conflicts with other trades. This step is crucial for tracking and managing only the trades opened by our strategy. Once the initialization is complete, we return [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode), confirming that our program is ready to operate. We can then graduate to the main [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler to our main control logic.

```
//+------------------------------------------------------------------+
//| Expert OnTick function                                           |
//+------------------------------------------------------------------+
void OnTick() {

   //--- Check for a new bar to process logic only once per bar
   static bool isNewBar = false;
   int currentBarCount = iBars(_Symbol, _Period);
   static int previousBarCount = currentBarCount;
   if (previousBarCount == currentBarCount) {
      isNewBar = false;
   } else if (previousBarCount != currentBarCount) {
      isNewBar = true;
      previousBarCount = currentBarCount;
   }

   //--- Exit if not a new bar to avoid redundant processing
   if (!isNewBar)
      return;
   //---
}
```

To ensure we process data on every bar and not on every tick, on the OnTick function, which executes on every new tick received, we use the [iBars](https://www.mql5.com/en/docs/series/ibars) function to get the total number of bars on the chart and store it in "currentBarCount". We then compare it with "previousBarCount", and if they are equal, "isNewBar" remains false, preventing redundant processing. If a new bar is detected, we update "previousBarCount" and set "isNewBar" to true, allowing the strategy logic to execute. Finally, if "isNewBar" is false, we return early, optimizing performance by skipping unnecessary computations. If it is a new bar, we continue looking for consolidations.

```
//--- Define the starting bar index for consolidation checks
int startBarIndex = 1;

//--- Check for consolidation or extend the existing range
if (!isBreakoutDetected) {
   if (rangeHighestHigh.price == 0 && rangeLowestLow.price == 0) {
      //--- Check if bars are in a tight consolidation range
      bool isConsolidated = true;
      for (int i = startBarIndex; i < startBarIndex + consolidationBars - 1; i++) {
         if (MathAbs(high(i) - high(i + 1)) > maxConsolidationSpread * Point()) {
            isConsolidated = false;
            break;
         }
         if (MathAbs(low(i) - low(i + 1)) > maxConsolidationSpread * Point()) {
            isConsolidated = false;
            break;
         }
      }
      if (isConsolidated) {
         //--- Find the highest high in the consolidation range
         rangeHighestHigh.price = high(startBarIndex);
         rangeHighestHigh.index = startBarIndex;
         for (int i = startBarIndex + 1; i < startBarIndex + consolidationBars; i++) {
            if (high(i) > rangeHighestHigh.price) {
               rangeHighestHigh.price = high(i);
               rangeHighestHigh.index = i;
            }
         }
         //--- Find the lowest low in the consolidation range
         rangeLowestLow.price = low(startBarIndex);
         rangeLowestLow.index = startBarIndex;
         for (int i = startBarIndex + 1; i < startBarIndex + consolidationBars; i++) {
            if (low(i) < rangeLowestLow.price) {
               rangeLowestLow.price = low(i);
               rangeLowestLow.index = i;
            }
         }
         //--- Log the established consolidation range
         Print("Consolidation range established: Highest High = ", rangeHighestHigh.price,
               " at index ", rangeHighestHigh.index,
               " and Lowest Low = ", rangeLowestLow.price,
               " at index ", rangeLowestLow.index);
      }
   } else {
      //--- Check if the current bar extends the existing range
      double currentHigh = high(1);
      double currentLow = low(1);
      if (currentHigh <= rangeHighestHigh.price && currentLow >= rangeLowestLow.price) {
         Print("Range extended: High = ", currentHigh, ", Low = ", currentLow);
      } else {
         Print("No extension: Bar outside range.");
      }
   }
}
```

Here, we define and establish the consolidation range by analyzing recent price movements. We begin by setting "startBarIndex" to 1, determining the starting point for our consolidation checks. If we have not yet detected a breakout, as indicated by "isBreakoutDetected", we proceed to assess whether the market is in a tight consolidation phase. We iterate through the last "consolidationBars" count of bars, using the [MathAbs](https://www.mql5.com/en/docs/math/mathabs) function to measure the absolute differences between consecutive highs and lows. If all differences remain within "maxConsolidationSpread", we confirm consolidation.

Once consolidation is detected, we determine the highest high and lowest low within the range. We initialize "rangeHighestHigh" and "rangeLowestLow" with the high and low of "startBarIndex", then iterate through the range to update these values whenever we encounter a new highest high or lowest low. These values define our consolidation boundaries.

If the consolidation range is already established, we check whether the current bar extends the existing range. We retrieve "currentHigh" and "currentLow" using the "high" and "low" functions and compare them to "rangeHighestHigh.price" and "rangeLowestLow.price". If the price remains within the range, we print a message indicating the range extension using the [Print](https://www.mql5.com/en/docs/common/print) function. Otherwise, we print that no extension occurred, signaling a potential breakout scenario. The custom price functions are as below.

```
//+------------------------------------------------------------------+
//| Price data accessors                                                 |
//+------------------------------------------------------------------+
double high(int index) { return iHigh(_Symbol, _Period, index); }   //--- Get high price of a bar
double low(int index) { return iLow(_Symbol, _Period, index); }     //--- Get low price of a bar
double open(int index) { return iOpen(_Symbol, _Period, index); }   //--- Get open price of a bar
double close(int index) { return iClose(_Symbol, _Period, index); } //--- Get close price of a bar
datetime time(int index) { return iTime(_Symbol, _Period, index); } //--- Get time of a bar
```

These custom functions help us retrieve price data. The "high" function uses [iHigh](https://www.mql5.com/en/docs/series/ihigh) to return the high price of a bar at a specified "index", while the "low" function calls [iLow](https://www.mql5.com/en/docs/series/ilow) to obtain the corresponding low price. The "open" function fetches the opening price using [iOpen](https://www.mql5.com/en/docs/series/iopen), and the "close" function retrieves the closing price via [iClose](https://www.mql5.com/en/docs/series/iclose). Additionally, the "time" function employs [iTime](https://www.mql5.com/en/docs/series/itime) to return the timestamp of the given bar. Upon running the program, we have the following result.

![Consolidation Confirmation](https://c.mql5.com/2/126/Screenshot_2025-03-20_002653.png)

From the image, we can see that once the price range is established and price revolves within the range, we extend it until we have a range breach. So now, we need to detect a breakout on the confirmed price lag range. We achieve this using the following logic.

```
//--- Detect a breakout from the consolidation range
if (rangeHighestHigh.price > 0 && rangeLowestLow.price > 0) {
   double currentClosePrice = close(1);
   if (currentClosePrice > rangeHighestHigh.price) {
      Print("Upward breakout at ", currentClosePrice, " > ", rangeHighestHigh.price);
      isBreakoutDetected = true;
   } else if (currentClosePrice < rangeLowestLow.price) {
      Print("Downward breakout at ", currentClosePrice, " < ", rangeLowestLow.price);
      isBreakoutDetected = true;
   }
}

//--- Reset state after a breakout is detected
if (isBreakoutDetected) {
   Print("Breakout detected. Resetting for the next range.");
   breakoutBarNumber = 1;
   breakoutTimestamp = TimeCurrent();
   lastImpulseHigh = rangeHighestHigh.price;
   lastImpulseLow = rangeLowestLow.price;

   isBreakoutDetected = false;
   rangeHighestHigh.price = 0;
   rangeHighestHigh.index = 0;
   rangeLowestLow.price = 0;
   rangeLowestLow.index = 0;
}
```

To detect and handle breakouts from a previously identified consolidation range, we first verify that the "rangeHighestHigh.price" and "rangeLowestLow.price" values are valid, ensuring a consolidation range has been established. We then compare the "currentClosePrice", obtained using the "close" function, against the range boundaries. If the closing price exceeds "rangeHighestHigh.price", we recognize an upward breakout, logging the event and setting "isBreakoutDetected" to true. Similarly, if the closing price falls below "rangeLowestLow.price", we identify a downward breakout and flag it accordingly.

Once a breakout is confirmed, we reset the necessary state variables to prepare for tracking a new consolidation phase. We log the breakout occurrence and store the "breakoutBarNumber" as 1, marking the first bar of the breakout sequence. The "breakoutTimestamp" is recorded using [TimeCurrent](https://www.mql5.com/en/docs/dateandtime/timecurrent) to note the exact time of the breakout. Additionally, we store "lastImpulseHigh" and "lastImpulseLow" to track post-breakout price behavior. Finally, we reset "isBreakoutDetected" to false and clear the previous consolidation range by setting "rangeHighestHigh.price" and "rangeLowestLow.price" to 0, ensuring the system is ready to detect the next trading opportunity.

If there are confirmed breakouts, we wait and verify them via impulsive movements, and then plot them on the chart.

```
//--- Check for impulsive movement after breakout and create order blocks
if (breakoutBarNumber >= 0 && TimeCurrent() > breakoutTimestamp + barsToWaitAfterBreakout * PeriodSeconds()) {
   double impulseRange = lastImpulseHigh - lastImpulseLow;
   double impulseThresholdPrice = impulseRange * impulseMultiplier;
   isBullishImpulse = false;
   isBearishImpulse = false;
   for (int i = 1; i <= barsToWaitAfterBreakout; i++) {
      double closePrice = close(i);
      if (closePrice >= lastImpulseHigh + impulseThresholdPrice) {
         isBullishImpulse = true;
         Print("Impulsive upward move: ", closePrice, " >= ", lastImpulseHigh + impulseThresholdPrice);
         break;
      } else if (closePrice <= lastImpulseLow - impulseThresholdPrice) {
         isBearishImpulse = true;
         Print("Impulsive downward move: ", closePrice, " <= ", lastImpulseLow - impulseThresholdPrice);
         break;
      }
   }
   if (!isBullishImpulse && !isBearishImpulse) {
      Print("No impulsive movement detected.");
   }
   //---
}
```

Here, we analyze price action after a breakout to determine whether an impulsive movement has occurred, which is critical for identifying valid order blocks. We first check if "breakoutBarNumber" is valid and if the current time, retrieved via TimeCurrent, has surpassed the "breakoutTimestamp" plus "barsToWaitAfterBreakout" multiplied by [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds), ensuring a sufficient waiting period has elapsed. We then calculate "impulseRange" as the difference between "lastImpulseHigh" and "lastImpulseLow", which represents the post-breakout price fluctuation. Using this, we compute "impulseThresholdPrice" by multiplying "impulseRange" by "impulseMultiplier" to define the minimum price extension required for an impulsive move.

Next, we initialize "isBullishImpulse" and "isBearishImpulse" as false, preparing to evaluate the price action over the last "barsToWaitAfterBreakout" bars. We iterate through these bars using a [for loop](https://www.mql5.com/en/docs/basis/operators/for), retrieving the closing price with the "close" function. If "closePrice" is greater than or equal to "lastImpulseHigh + impulseThresholdPrice", we detect an impulsive bullish move, set "isBullishImpulse" to true, and log the event. If "closePrice" is less than or equal to "lastImpulseLow - impulseThresholdPrice", we identify an impulsive bearish move, set "isBearishImpulse" to true, and log it. If neither condition is met, we print a message stating that no impulsive movement was detected. This logic ensures that only strong breakout continuations qualify as valid order blocks for further processing. To visualize them, we use the following logic.

```
bool isOrderBlockValid = isBearishImpulse || isBullishImpulse;

if (isOrderBlockValid) {
   datetime blockStartTime = iTime(_Symbol, _Period, consolidationBars + barsToWaitAfterBreakout + 1);
   double blockTopPrice = lastImpulseHigh;
   int visibleBarsOnChart = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);
   datetime blockEndTime = blockStartTime + (visibleBarsOnChart / 1) * PeriodSeconds();
   double blockBottomPrice = lastImpulseLow;
   string orderBlockName = OB_Prefix + "(" + TimeToString(blockStartTime) + ")";
   color orderBlockColor = isBullishImpulse ? bullishOrderBlockColor : bearishOrderBlockColor;
   string orderBlockLabel = isBullishImpulse ? "Bullish OB" : "Bearish OB";

   if (ObjectFind(0, orderBlockName) < 0) {
      //--- Create a rectangle for the order block
      ObjectCreate(0, orderBlockName, OBJ_RECTANGLE, 0, blockStartTime, blockTopPrice, blockEndTime, blockBottomPrice);
      ObjectSetInteger(0, orderBlockName, OBJPROP_TIME, 0, blockStartTime);
      ObjectSetDouble(0, orderBlockName, OBJPROP_PRICE, 0, blockTopPrice);
      ObjectSetInteger(0, orderBlockName, OBJPROP_TIME, 1, blockEndTime);
      ObjectSetDouble(0, orderBlockName, OBJPROP_PRICE, 1, blockBottomPrice);
      ObjectSetInteger(0, orderBlockName, OBJPROP_FILL, true);
      ObjectSetInteger(0, orderBlockName, OBJPROP_COLOR, orderBlockColor);
      ObjectSetInteger(0, orderBlockName, OBJPROP_BACK, false);

      //--- Add a text label in the middle of the order block with dynamic font size
      datetime labelTime = blockStartTime + (blockEndTime - blockStartTime) / 2;
      double labelPrice = (blockTopPrice + blockBottomPrice) / 2;
      string labelObjectName = orderBlockName + orderBlockLabel;
      if (ObjectFind(0, labelObjectName) < 0) {
         ObjectCreate(0, labelObjectName, OBJ_TEXT, 0, labelTime, labelPrice);
         ObjectSetString(0, labelObjectName, OBJPROP_TEXT, orderBlockLabel);
         ObjectSetInteger(0, labelObjectName, OBJPROP_COLOR, labelTextColor);
         ObjectSetInteger(0, labelObjectName, OBJPROP_FONTSIZE, dynamicFontSize);
         ObjectSetInteger(0, labelObjectName, OBJPROP_ANCHOR, ANCHOR_CENTER);
      }
      ChartRedraw(0);

      //--- Store the order block details in arrays
      ArrayResize(orderBlockNames, ArraySize(orderBlockNames) + 1);
      orderBlockNames[ArraySize(orderBlockNames) - 1] = orderBlockName;
      ArrayResize(orderBlockEndTimes, ArraySize(orderBlockEndTimes) + 1);
      orderBlockEndTimes[ArraySize(orderBlockEndTimes) - 1] = blockEndTime;
      ArrayResize(orderBlockMitigatedStatus, ArraySize(orderBlockMitigatedStatus) + 1);
      orderBlockMitigatedStatus[ArraySize(orderBlockMitigatedStatus) - 1] = false;

      Print("Order Block created: ", orderBlockName);
   }
}
```

Here, we determine whether an order block should be created based on the detection of an impulsive move. We first evaluate "isOrderBlockValid" by checking if either "isBearishImpulse" or "isBullishImpulse" is true. If valid, we define key parameters for the order block: "blockStartTime" is obtained using the [iTime](https://www.mql5.com/en/docs/series/itime) function to reference the bar at "consolidationBars + barsToWaitAfterBreakout + 1", ensuring it aligns with the identified structure. "blockTopPrice" is set to "lastImpulseHigh", and "blockBottomPrice" to "lastImpulseLow", marking the price range of the order block. We use the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function to determine "visibleBarsOnChart" and calculate "blockEndTime" dynamically based on [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds), ensuring the rectangle remains visible within the chart’s current scope.

The order block name is constructed using "OB\_Prefix" and the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function to include the timestamp for uniqueness. The color and label are determined based on whether the impulse is bullish or bearish, selecting "bullishOrderBlockColor" or "bearishOrderBlockColor", and assigning the respective label.

We then check for the existence of the order block using [ObjectFind](https://www.mql5.com/en/docs/objects/ObjectFind). If it does not exist, we use the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to draw a rectangle ( [OBJ\_RECTANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object)) representing the order block, setting its time and price boundaries with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) and [ObjectSetDouble](https://www.mql5.com/en/docs/objects/objectsetdouble). The rectangle is filled (OBJPROP\_FILL), color is applied (OBJPROP\_COLOR), and it is drawn in the foreground ( [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) = false).

Next, we create a label inside the order block for better visualization. The label's time ("labelTime") is set at the midpoint of "blockStartTime" and "blockEndTime", while "labelPrice" is calculated as the midpoint of "blockTopPrice" and "blockBottomPrice". We generate a unique label name by appending "orderBlockLabel" to "orderBlockName". If the label does not exist, we create a text object (OBJ\_TEXT) with "ObjectCreate", setting the text content (OBJPROP\_TEXT), color (OBJPROP\_COLOR), font size ( [OBJPROP\_FONTSIZE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer)), and centering it with (OBJPROP\_ANCHOR = ANCHOR\_CENTER). The [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function ensures that the newly created elements appear immediately. Since font size would significantly matter based on the chart scale, we calculate it dynamically as below.

```
//--- Calculate dynamic font size based on chart scale (0 = zoomed out, 5 = zoomed in)
int chartScale = (int)ChartGetInteger(0, CHART_SCALE); // Scale ranges from 0 to 5
int dynamicFontSize = 8 + (chartScale * 2);           // Font size: 8 (min) to 18 (max)
```

Finally, we store the order block details in arrays: "orderBlockNames" (stores object names), "orderBlockEndTimes" (stores expiration times), and "orderBlockMitigatedStatus" (tracks whether the order block has been mitigated). We resize each array dynamically using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function to accommodate new entries, ensuring our order block management remains flexible. A confirmation message is printed to indicate the successful creation of an order block. Finally, we just need to reset the breakout tracking variables.

```
//--- Reset breakout tracking variables
breakoutBarNumber = -1;
breakoutTimestamp = 0;
lastImpulseHigh = 0;
lastImpulseLow = 0;
isBullishImpulse = false;
isBearishImpulse = false;
```

Upon compilation and running the program, we have the following outcome.

![Confirmed OBs](https://c.mql5.com/2/126/Screenshot_2025-03-20_010100.png)

From the image, we can see that we have confirmed and labeled order blocks that result from the impulsive breakout movements. So now we just need to proceed to validate the mitigated order blocks via continued management of the setups within the chart boundaries.

```
//--- Process existing order blocks for mitigation and trading
for (int j = ArraySize(orderBlockNames) - 1; j >= 0; j--) {
   string currentOrderBlockName = orderBlockNames[j];
   bool doesOrderBlockExist = false;

   //--- Retrieve order block properties
   double orderBlockHigh = ObjectGetDouble(0, currentOrderBlockName, OBJPROP_PRICE, 0);
   double orderBlockLow = ObjectGetDouble(0, currentOrderBlockName, OBJPROP_PRICE, 1);
   datetime orderBlockStartTime = (datetime)ObjectGetInteger(0, currentOrderBlockName, OBJPROP_TIME, 0);
   datetime orderBlockEndTime = (datetime)ObjectGetInteger(0, currentOrderBlockName, OBJPROP_TIME, 1);
   color orderBlockCurrentColor = (color)ObjectGetInteger(0, currentOrderBlockName, OBJPROP_COLOR);

   //--- Check if the order block is still valid (not expired)
   if (time(1) < orderBlockEndTime) {
      doesOrderBlockExist = true;
   }
   //---
}
```

We iterate through "orderBlockNames" in reverse, processing each order block for mitigation and trading. "currentOrderBlockName" stores the name of the block being checked. We use [ObjectGetDouble](https://www.mql5.com/en/docs/objects/objectgetdouble) and [ObjectGetInteger](https://www.mql5.com/en/docs/objects/objectgetinteger) to retrieve "orderBlockHigh", "orderBlockLow", "orderBlockStartTime", "orderBlockEndTime", and "orderBlockCurrentColor", ensuring precise handling of each order block's properties.

To verify if the order block is still valid, we compare "time(1)" (retrieved using the "time" function) with "orderBlockEndTime". If the current time is within the order block’s lifespan, "doesOrderBlockExist" is set to true, confirming that the order block remains active for further processing. If it does, we proceed to process it and trade it.

```
//--- Get current market prices
double currentAskPrice = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits);
double currentBidPrice = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits);

//--- Check for mitigation and execute trades if trading is enabled
if (enableTrading && orderBlockCurrentColor == bullishOrderBlockColor && close(1) < orderBlockLow && !orderBlockMitigatedStatus[j]) {
   //--- Sell trade when price breaks below a bullish order block
   double entryPrice = currentBidPrice;
   double stopLossPrice = entryPrice + stopLossDistance * _Point;
   double takeProfitPrice = entryPrice - takeProfitDistance * _Point;
   obj_Trade.Sell(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice);
   orderBlockMitigatedStatus[j] = true;
   ObjectSetInteger(0, currentOrderBlockName, OBJPROP_COLOR, mitigatedOrderBlockColor);
   string blockDescription = "Bullish Order Block";
   string textObjectName = currentOrderBlockName + blockDescription;
   if (ObjectFind(0, textObjectName) >= 0) {
      ObjectSetString(0, textObjectName, OBJPROP_TEXT, "Mitigated " + blockDescription);
   }
   Print("Sell trade entered upon mitigation of bullish OB: ", currentOrderBlockName);
} else if (enableTrading && orderBlockCurrentColor == bearishOrderBlockColor && close(1) > orderBlockHigh && !orderBlockMitigatedStatus[j]) {
   //--- Buy trade when price breaks above a bearish order block
   double entryPrice = currentAskPrice;
   double stopLossPrice = entryPrice - stopLossDistance * _Point;
   double takeProfitPrice = entryPrice + takeProfitDistance * _Point;
   obj_Trade.Buy(tradeLotSize, _Symbol, entryPrice, stopLossPrice, takeProfitPrice);
   orderBlockMitigatedStatus[j] = true;
   ObjectSetInteger(0, currentOrderBlockName, OBJPROP_COLOR, mitigatedOrderBlockColor);
   string blockDescription = "Bearish Order Block";
   string textObjectName = currentOrderBlockName + blockDescription;
   if (ObjectFind(0, textObjectName) >= 0) {
      ObjectSetString(0, textObjectName, OBJPROP_TEXT, "Mitigated " + blockDescription);
   }
   Print("Buy trade entered upon mitigation of bearish OB: ", currentOrderBlockName);
}
```

We begin by retrieving the current market prices using the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function, ensuring both "currentAskPrice" and "currentBidPrice" are normalized to the appropriate number of decimal places using [\_Digits](https://www.mql5.com/en/docs/predefined/_Digits). This guarantees precision when placing trades. Next, we check if "enableTrading" is active and whether an order block mitigation condition has been met. Mitigation occurs when a price breaks through an order block, indicating a failure in its holding structure.

For bullish order blocks, we verify if the "close" price of the previous bar (obtained using the "close" function) has dropped below "orderBlockLow" and ensure that this order block has not already been mitigated ("orderBlockMitigatedStatus\[j\] == false"). If these conditions hold, we place a sell trade using the "Sell" function of the "obj\_Trade" object. The trade is executed at "currentBidPrice", with a stop-loss ("stopLossPrice") positioned above the entry price by "stopLossDistance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" and a take-profit ("takeProfitPrice") set below the entry price by "takeProfitDistance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)".

Once the trade is executed, the order block is marked as mitigated by updating "orderBlockMitigatedStatus\[j\]" to true, and its color is changed using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger) to indicate its mitigated state. If a text label exists for this order block (checked using [ObjectFind](https://www.mql5.com/en/docs/objects/objectfind)), we update it using [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) to display "Mitigated Bullish Order Block". A [Print](https://www.mql5.com/en/docs/common/print) statement logs the trade execution for tracking and debugging.

For bearish order blocks, the process is similar. We check if the "close" price has risen above "orderBlockHigh", indicating a break of the bearish order block. If the conditions are met, a buy trade is placed via the "Buy" function, using "currentAskPrice" as the entry price. The "stopLossPrice" is positioned below the entry price, and the "takeProfitPrice" is set above it, ensuring proper risk management. After placing the buy trade, we update "orderBlockMitigatedStatus\[j\]", change the order block’s color using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/ObjectSetInteger), and modify the text label (if found) to display "Mitigated Bearish Order Block". Finally, a "Print" statement logs the buy trade execution for monitoring purposes. Here is what we achieve.

![Mitigated & Traded OB ](https://c.mql5.com/2/126/Screenshot_2025-03-20_013215.png)

Finally, once the blocks are out of bounds, we remove them from the storage arrays.

```
//--- Remove expired order blocks from arrays
if (!doesOrderBlockExist) {
   bool removedName = ArrayRemove(orderBlockNames, j, 1);
   bool removedTime = ArrayRemove(orderBlockEndTimes, j, 1);
   bool removedStatus = ArrayRemove(orderBlockMitigatedStatus, j, 1);
   if (removedName && removedTime && removedStatus) {
      Print("Success removing OB DATA from arrays at index ", j);
   }
}
```

If the order block no longer exists, we remove its name, end time, and mitigation status from their respective arrays using the [ArrayRemove](https://www.mql5.com/en/docs/array/arrayremove) function. If all removals are successful, we log the action with a [Print](https://www.mql5.com/en/docs/common/print) statement to confirm the cleanup. Here is a sample cleanup confirmation.

![Blocks Cleanup](https://c.mql5.com/2/126/Screenshot_2025-03-20_013621.png)

From the image, we can see that we successfully do the blocks cleanup. Now we just need to add a trailing stop logic, and for this, we employ a function to encapsulate everything.

```
//+------------------------------------------------------------------+
//| Trailing stop function                                           |
//+------------------------------------------------------------------+
void applyTrailingStop(double trailingPoints, CTrade &trade_object, int magicNo = 0) {
   //--- Calculate trailing stop levels based on current market prices
   double buyStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID) - trailingPoints * _Point, _Digits);
   double sellStopLoss = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK) + trailingPoints * _Point, _Digits);

   //--- Loop through all open positions
   for (int i = PositionsTotal() - 1; i >= 0; i--) {
      ulong ticket = PositionGetTicket(i);
      if (ticket > 0) {
         if (PositionGetString(POSITION_SYMBOL) == _Symbol &&
             (magicNo == 0 || PositionGetInteger(POSITION_MAGIC) == magicNo)) {
            //--- Adjust stop loss for buy positions
            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY &&
                buyStopLoss > PositionGetDouble(POSITION_PRICE_OPEN) &&
                (buyStopLoss > PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL) == 0)) {
               trade_object.PositionModify(ticket, buyStopLoss, PositionGetDouble(POSITION_TP));
            }
            //--- Adjust stop loss for sell positions
            else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL &&
                       sellStopLoss < PositionGetDouble(POSITION_PRICE_OPEN) &&
                       (sellStopLoss < PositionGetDouble(POSITION_SL) || PositionGetDouble(POSITION_SL) == 0)) {
               trade_object.PositionModify(ticket, sellStopLoss, PositionGetDouble(POSITION_TP));
            }
         }
      }
   }
}
```

Here, we define the "applyTrailingStop" function to dynamically adjust stop-loss levels for active positions. We begin by calculating "buyStopLoss" and "sellStopLoss" using the current bid/ask prices and the specified "trailingPoints". Next, we loop through all open positions, filtering them by symbol and magic number (if provided). If a buy position has a valid stop-loss level above its entry price, and it either exceeds the current stop-loss or is unset, we update it. Similarly, for sell positions, we ensure the new stop-loss is below the entry price before modifying it.

We then call the function inside the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, to process on every tick and not on every bar this time for real-time price checks as follows.

```
//--- Apply trailing stop to open positions if enabled
if (enableTrailingStop) {
   applyTrailingStop(trailingStopPoints, obj_Trade, uniqueMagicNumber);
}
```

Upon compilation and running of the program, we have the following outcome.

![MOB GIF](https://c.mql5.com/2/126/MOB.gif)

From the visualization, we can see that the program identifies and verifies all the entry conditions and if validated, opens the respective position with the respective entry parameters, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/126/Screenshot_2025-03-20_021731.png)

Backtest report:

![REPORT](https://c.mql5.com/2/126/Screenshot_2025-03-20_021832.png)

### Conclusion

In conclusion, we have successfully implemented the [Mitigation Order Blocks](https://www.mql5.com/go?link=https://innercircletrader.net/tutorials/ict-mitigation-block-explained/ "https://innercircletrader.net/tutorials/ict-mitigation-block-explained/") (MOB) Strategy in MQL5, allowing for precise detection, visualization, and automated trading based on smart money concepts. By integrating breakout validation, impulsive move recognition, and mitigation-based trade execution, our system effectively identifies and processes order blocks while adapting to market dynamics. Additionally, we incorporated trailing stops and risk management mechanisms to optimize trade performance and enhance robustness.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market conditions can be unpredictable. Proper backtesting and risk management are essential before live deployment.

By leveraging these techniques, you can refine your algorithmic trading strategies and improve order block-based trading efficiency. Keep testing, optimizing, and adapting your approach for long-term success. Best of luck!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17547.zip "Download all attachments in the single ZIP archive")

[Mitigation\_Order\_Blocks.mq5](https://www.mql5.com/en/articles/download/17547/mitigation_order_blocks.mq5 "Download Mitigation_Order_Blocks.mq5")(19.27 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/483609)**
(6)


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
28 Mar 2025 at 10:56

**linfo2 [#](https://www.mql5.com/en/forum/483609#comment_56288172):**

Thank you Allan , nicely put together really tike the visuals and the change colour on mitigated and your handling of the arrays . Thanks for sharing

Thanks for the kind feedback. You're welcome.

![davesarge1](https://c.mql5.com/avatar/avatar_na2.png)

**[davesarge1](https://www.mql5.com/en/users/davesarge1)**
\|
12 Apr 2025 at 13:08

It's not taking trades in the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 "). Default settings on all pairs.  No error messages in the journal.  Journal messages present; "No extension: Bar outside range" and "No impulsive movement detected".

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
14 Apr 2025 at 13:47

**davesarge1 [#](https://www.mql5.com/en/forum/483609#comment_56437668):**

It's not taking trades in the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 "). Default settings on all pairs.  No error messages in the journal.  Journal messages present; "No extension: Bar outside range" and "No impulsive movement detected".

Did you even read the article? Because we're sure the article pretty much provides all your answers.

![Bao Thuan Thai](https://c.mql5.com/avatar/2025/4/680C8606-3BDF.png)

**[Bao Thuan Thai](https://www.mql5.com/en/users/thuanthai)**
\|
1 Aug 2025 at 23:35

Thank you so much. Appreciate it! I’ll give it a shot


![Chen Yurui](https://c.mql5.com/avatar/avatar_na2.png)

**[Chen Yurui](https://www.mql5.com/en/users/700802)**
\|
8 Jan 2026 at 03:10

Is it possible to make an EA that can both trade mitigation order blocks and retesting order blocks (i.e., it buys when price falls to a bullish block)?


![From Basic to Intermediate: WHILE and DO WHILE Statements](https://c.mql5.com/2/90/logo-image_15375_417_4038__2.png)[From Basic to Intermediate: WHILE and DO WHILE Statements](https://www.mql5.com/en/articles/15375)

In this article, we will take a practical and very visual look at the first loop statement. Although many beginners feel intimidated when faced with the task of creating loops, knowing how to do it correctly and safely can only come with experience and practice. But who knows, maybe I can reduce your troubles and suffering by showing you the main issues and precautions to take when using loops in your code.

![Developing a Replay System (Part 62): Playing the service (III)](https://c.mql5.com/2/90/logo-image_12231_394_3793__1.png)[Developing a Replay System (Part 62): Playing the service (III)](https://www.mql5.com/en/articles/12231)

In this article, we will begin to address the issue of tick excess that can impact application performance when using real data. This excess often interferes with the correct timing required to construct a one-minute bar in the appropriate window.

![Neural Networks in Trading: Hierarchical Vector Transformer (HiVT)](https://c.mql5.com/2/90/logo-15688.png)[Neural Networks in Trading: Hierarchical Vector Transformer (HiVT)](https://www.mql5.com/en/articles/15688)

We invite you to get acquainted with the Hierarchical Vector Transformer (HiVT) method, which was developed for fast and accurate forecasting of multimodal time series.

![Bacterial Chemotaxis Optimization (BCO)](https://c.mql5.com/2/92/Bacterial_Chemotaxis_Optimization___LOGO__2.png)[Bacterial Chemotaxis Optimization (BCO)](https://www.mql5.com/en/articles/15711)

The article presents the original version of the Bacterial Chemotaxis Optimization (BCO) algorithm and its modified version. We will take a closer look at all the differences, with a special focus on the new version of BCOm, which simplifies the bacterial movement mechanism, reduces the dependence on positional history, and uses simpler math than the computationally heavy original version. We will also conduct the tests and summarize the results.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nzeadoletjfxpdtjshfxsufeguykxgzk&ssn=1769093866353424951&ssn_dr=0&ssn_sr=0&fv_date=1769093866&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17547&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2012)%3A%20Implementing%20the%20Mitigation%20Order%20Blocks%20(MOB)%20Strategy%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17690938663486877&fz_uniq=5049517879738608873&sv=2552)

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