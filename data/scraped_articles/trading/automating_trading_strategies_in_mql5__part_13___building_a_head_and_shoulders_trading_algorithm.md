---
title: Automating Trading Strategies in MQL5 (Part 13): Building a Head and Shoulders Trading Algorithm
url: https://www.mql5.com/en/articles/17618
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 9
scraped_at: 2026-01-22T17:27:00.952757
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/17618&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049141571884000690)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 12)](https://www.mql5.com/en/articles/17547), we implemented the [Mitigation Order Blocks (MOB)](https://www.mql5.com/go?link=https://theicttrader.com/2024/03/24/order-blocks-breaker-blocks-and-mitigation-blocks/ "https://theicttrader.com/2024/03/24/order-blocks-breaker-blocks-and-mitigation-blocks/") Strategy in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) to leverage institutional price zones for trading. Now, in Part 13, we shift our focus to building a [Head and Shoulders trading algorithm](https://www.mql5.com/go?link=https://www.investopedia.com/terms/h/head-shoulders.asp "https://www.investopedia.com/terms/h/head-shoulders.asp"), automating a classic reversal pattern to capture market turns with precision. We will cover the following topics:

1. [Understanding the Head and Shoulders Pattern Architecture](https://www.mql5.com/en/articles/17618#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17618#para2)
3. [Backtesting](https://www.mql5.com/en/articles/17618#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17618#para4)

By the end of this article, you’ll have a fully functional Expert Advisor ready to trade the Head and Shoulders pattern—let’s dive in!

### Understanding the Head and Shoulders Pattern Architecture

The [Head and Shoulders pattern](https://www.mql5.com/go?link=https://www.investopedia.com/terms/h/head-shoulders.asp "https://www.investopedia.com/terms/h/head-shoulders.asp") is a classic chart formation widely recognized in technical analysis for predicting trend reversals, appearing in both standard (bearish) and inverse (bullish) variations, each defined by a unique sequence of price peaks or troughs. In the [standard pattern](https://www.mql5.com/go?link=https://thetradingbible.com/head-and-shoulders-pattern "https://thetradingbible.com/head-and-shoulders-pattern"), in our program, an uptrend will give way to three peaks: the left shoulder will establish a high, the head will tower distinctly higher as the trend’s climax (surpassing both shoulders significantly), and the right shoulder will form lower than the head yet close in height to the left, all tied together by a neckline linking the two troughs—once price breaks below this line, we’ll enter a bearish trade at the breakout, set a stop-loss above the right shoulder, and target a take-profit by projecting the head-to-neckline height downward as illustrated below.

![BEARISH HEAD & SHOULDERS PATTERN](https://c.mql5.com/2/127/Screenshot_2025-03-24_161251.png)

For the [inverse pattern](https://www.mql5.com/go?link=https://thetradingbible.com/head-and-shoulders-pattern "https://thetradingbible.com/head-and-shoulders-pattern"), a downtrend will yield three troughs: a left shoulder will mark a low, a head will plunge notably deeper (below both shoulders), and a right shoulder will align near the left’s level, with a neckline across the peaks—price breaking above it will trigger a bullish entry, with a stop-loss below the right shoulder and a take-profit extending upward by the neckline-to-head distance, all built on the head’s standout height and the shoulders’ near symmetry as our guiding rules. Here is its visualization.

![BULLISH HEAD & SHOULDERS PATTERN](https://c.mql5.com/2/127/Screenshot_2025-03-24_162215.png)

As for the risk management, we will integrate an optional trailing stop feature to lock in profits maximizing gains. Let's go.

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                  Head & Shoulders Pattern EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link      "https://youtube.com/@ForexAlgo-Trader?"
#property version   "1.00"

#include <Trade\Trade.mqh>                    //--- Include the Trade.mqh library for trading functions
CTrade obj_Trade;                            //--- Trade object for executing and managing trades

// Input Parameters
input int LookbackBars = 50;                   // Number of historical bars to analyze for pattern detection
input double ThresholdPoints = 70.0;           // Minimum price movement in points to identify a reversal
input double ShoulderTolerancePoints = 15.0;   // Maximum allowable price difference between left and right shoulders
input double TroughTolerancePoints = 30.0;     // Maximum allowable price difference between neckline troughs or peaks
input double BufferPoints = 10.0;              // Additional points added to stop-loss for safety buffer
input double LotSize = 0.1;                    // Volume of each trade in lots
input ulong MagicNumber = 123456;              // Unique identifier for trades opened by this EA
input int MaxBarRange = 30;                    // Maximum number of bars allowed between key pattern points
input int MinBarRange = 5;                     // Minimum number of bars required between key pattern points
input double BarRangeMultiplier = 2.0;         // Maximum multiple of the smallest bar range for pattern uniformity
input int ValidationBars = 3;                  // Number of bars after right shoulder to validate breakout
input double PriceTolerance = 5.0;             // Price tolerance in points for matching traded patterns
input double RightShoulderBreakoutMultiplier = 1.5; // Maximum multiple of pattern range for right shoulder to breakout distance
input int MaxTradedPatterns = 20;              // Maximum number of patterns stored in traded history
input bool UseTrailingStop = false;             // Toggle to enable or disable trailing stop functionality
input int MinTrailPoints = 50;                 // Minimum profit in points before trailing stop activates
input int TrailingPoints = 30;                 // Distance in points to maintain behind current price when trailing
```

Here, we start with " [#include](https://www.mql5.com/en/docs/basis/preprosessor/include) <Trade\\Trade.mqh>" and a "CTrade" object, "obj\_Trade", to [include](https://www.mql5.com/en/docs/basis/preprosessor/include) extra trading files for trade management. We set inputs like "LookbackBars" (default 50) for historical analysis, "ThresholdPoints" (default 70.0) for reversal confirmation, and "ShoulderTolerancePoints" (default 15.0) and "TroughTolerancePoints" (default 30.0) for symmetry. The rest of the inputs are self-explanatory. We have added detailed comments for ease of understanding. Next, we need to define some structures we will use to find the patterns and manage the considered trades.

```
// Structure to store peaks and troughs
struct Extremum {
   int bar;           //--- Bar index where extremum occurs
   datetime time;     //--- Timestamp of the bar
   double price;      //--- Price at extremum (high for peak, low for trough)
   bool isPeak;       //--- True if peak (high), false if trough (low)
};

// Structure to store traded patterns
struct TradedPattern {
   datetime leftShoulderTime;  //--- Timestamp of the left shoulder
   double leftShoulderPrice;   //--- Price of the left shoulder
};
```

We set up two key structures using [struct](https://www.mql5.com/en/docs/basis/types/classes) keyword to drive our Head and Shoulders trading algorithm: "Extremum" will store peaks and troughs with "bar" (index), "time" (timestamp), "price" (value), and "isPeak" (true for peaks, false for troughs) to pinpoint pattern components, while "TradedPattern" will track executed trades using "leftShoulderTime" and "leftShoulderPrice" to prevent duplicates. To ensure we trade once per bar and keep track of the running trades, we declare a [variable](https://www.mql5.com/en/docs/basis/variables) and an [array](https://www.mql5.com/en/book/basis/arrays/arrays_usage) as follows.

```
// Global Variables
static datetime lastBarTime = 0;         //--- Tracks the timestamp of the last processed bar to avoid reprocessing
TradedPattern tradedPatterns[];          //--- Array to store details of previously traded patterns
```

With that, we are all set. However, since we will need to display the pattern on the chart, we will need to get the chart architecture and bar components to ensure it adapts to the pattern requirements.

```
int chart_width         = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS);        //--- Width of the chart in pixels for visualization
int chart_height        = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS);       //--- Height of the chart in pixels for visualization
int chart_scale         = (int)ChartGetInteger(0, CHART_SCALE);                  //--- Zoom level of the chart (0-5)
int chart_first_vis_bar = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR);      //--- Index of the first visible bar on the chart
int chart_vis_bars      = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);           //--- Number of visible bars on the chart
double chart_prcmin     = ChartGetDouble(0, CHART_PRICE_MIN, 0);                 //--- Minimum price visible on the chart
double chart_prcmax     = ChartGetDouble(0, CHART_PRICE_MAX, 0);                 //--- Maximum price visible on the chart

//+------------------------------------------------------------------+
//| Converts the chart scale property to bar width/spacing           |
//+------------------------------------------------------------------+
int BarWidth(int scale) { return (int)pow(2, scale); }                           //--- Calculates bar width in pixels based on chart scale (zoom level)

//+------------------------------------------------------------------+
//| Converts the bar index (as series) to x in pixels                |
//+------------------------------------------------------------------+
int ShiftToX(int shift) { return (chart_first_vis_bar - shift) * BarWidth(chart_scale) - 1; } //--- Converts bar index to x-coordinate in pixels on the chart

//+------------------------------------------------------------------+
//| Converts the price to y in pixels                                |
//+------------------------------------------------------------------+
int PriceToY(double price) {                                                     //--- Function to convert price to y-coordinate in pixels
   if (chart_prcmax - chart_prcmin == 0.0) return 0;                             //--- Return 0 if price range is zero to avoid division by zero
   return (int)round(chart_height * (chart_prcmax - price) / (chart_prcmax - chart_prcmin) - 1); //--- Calculate y-pixel position based on price and chart dimensions
}
```

We prepare and equip the program with visualization by defining variables like "chart\_width" and "chart\_height" using the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function for chart dimensions, "chart\_scale" for zoom, "chart\_first\_vis\_bar" and "chart\_vis\_bars" for bar details, and "chart\_prcmin" and "chart\_prcmax" via [ChartGetDouble](https://www.mql5.com/en/docs/chart_operations/chartgetdouble) for price range. We use the "BarWidth" function with [pow](https://www.mql5.com/en/docs/math/mathpow) to calculate bar spacing from "chart\_scale", the "ShiftToX" function to convert bar indices to x-coordinates using "chart\_first\_vis\_bar" and "chart\_scale", and the "PriceToY" function with [round](https://www.mql5.com/en/docs/math/mathround) to map prices to y-coordinates based on "chart\_height", "chart\_prcmax", and "chart\_prcmin", enabling precise pattern display. We are now fully set. We can proceed to initialize the program in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {                                                           //--- Expert Advisor initialization function
   obj_Trade.SetExpertMagicNumber(MagicNumber);                          //--- Set the magic number for trades opened by this EA
   ArrayResize(tradedPatterns, 0);                                       //--- Initialize tradedPatterns array with zero size
   return(INIT_SUCCEEDED);                                               //--- Return success code to indicate successful initialization
}
```

In the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit), we use the "SetExpertMagicNumber" method on the "obj\_Trade" object to assign "MagicNumber" as the unique identifier for all trades, ensuring our program’s positions are distinguishable, and call the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function to set the "tradedPatterns" array to zero sizes, clearing any prior data for a fresh start. We then conclude by returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm the successful setup, preparing the Expert Advisor to detect and trade the pattern effectively. We can now move on to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler and make sure we do analysis once per bar.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {                                                          //--- Main tick function executed on each price update
   datetime currentBarTime = iTime(_Symbol, _Period, 0);                 //--- Get the timestamp of the current bar
   if (currentBarTime == lastBarTime) return;                            //--- Exit if the current bar has already been processed

   lastBarTime = currentBarTime;                                         //--- Update the last processed bar time

   // Update chart properties
   chart_width         = (int)ChartGetInteger(0, CHART_WIDTH_IN_PIXELS); //--- Update chart width in pixels
   chart_height        = (int)ChartGetInteger(0, CHART_HEIGHT_IN_PIXELS); //--- Update chart height in pixels
   chart_scale         = (int)ChartGetInteger(0, CHART_SCALE);           //--- Update chart zoom level
   chart_first_vis_bar = (int)ChartGetInteger(0, CHART_FIRST_VISIBLE_BAR); //--- Update index of the first visible bar
   chart_vis_bars      = (int)ChartGetInteger(0, CHART_VISIBLE_BARS);    //--- Update number of visible bars
   chart_prcmin        = ChartGetDouble(0, CHART_PRICE_MIN, 0);          //--- Update minimum visible price on chart
   chart_prcmax        = ChartGetDouble(0, CHART_PRICE_MAX, 0);          //--- Update maximum visible price on chart

   // Skip pattern detection if a position is already open
   if (PositionsTotal() > 0) return;                                     //--- Exit function if there are open positions to avoid multiple trades
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, which activates on each price update to monitor and respond to market changes, we use the [iTime](https://www.mql5.com/en/docs/series/itime) function to fetch "currentBarTime" for the latest bar and compare it with "lastBarTime" to avoid reprocessing, updating "lastBarTime" only for new bars; then, we refresh chart visuals by calling [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) to update "chart\_width", "chart\_height", "chart\_scale", "chart\_first\_vis\_bar", and "chart\_vis\_bars", and [ChartGetDouble](https://www.mql5.com/en/docs/chart_operations/chartgetdouble) for "chart\_prcmin" and "chart\_prcmax". We also employ the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function to check for open trades, exiting early if any exist to prevent overlapping positions, setting the stage for pattern detection and trading. We can then define a function to find the extremum points or the key pattern points.

```
//+------------------------------------------------------------------+
//| Find extrema in the last N bars                                  |
//+------------------------------------------------------------------+
void FindExtrema(Extremum &extrema[], int lookback) {                    //--- Function to identify peaks and troughs in price history
   ArrayFree(extrema);                                                   //--- Clear the extrema array to start fresh
   int bars = Bars(_Symbol, _Period);                                    //--- Get total number of bars available
   if (lookback >= bars) lookback = bars - 1;                            //--- Adjust lookback if it exceeds available bars

   double highs[], lows[];                                               //--- Arrays to store high and low prices
   ArraySetAsSeries(highs, true);                                        //--- Set highs array as time series (newest first)
   ArraySetAsSeries(lows, true);                                         //--- Set lows array as time series (newest first)
   CopyHigh(_Symbol, _Period, 0, lookback + 1, highs);                   //--- Copy high prices for lookback period
   CopyLow(_Symbol, _Period, 0, lookback + 1, lows);                     //--- Copy low prices for lookback period

   bool isUpTrend = highs[lookback] < highs[lookback - 1];               //--- Determine initial trend based on first two bars
   double lastHigh = highs[lookback];                                    //--- Initialize last high price
   double lastLow = lows[lookback];                                      //--- Initialize last low price
   int lastExtremumBar = lookback;                                       //--- Initialize last extremum bar index

   for (int i = lookback - 1; i >= 0; i--) {                             //--- Loop through bars from oldest to newest
      if (isUpTrend) {                                                   //--- If currently in an uptrend
         if (highs[i] > lastHigh) {                                      //--- Check if current high exceeds last high
            lastHigh = highs[i];                                         //--- Update last high price
            lastExtremumBar = i;                                         //--- Update last extremum bar index
         } else if (lows[i] < lastHigh - ThresholdPoints * _Point) {     //--- Check if current low indicates a reversal (trough)
            int size = ArraySize(extrema);                               //--- Get current size of extrema array
            ArrayResize(extrema, size + 1);                              //--- Resize array to add new extremum
            extrema[size].bar = lastExtremumBar;                         //--- Store bar index of the peak
            extrema[size].time = iTime(_Symbol, _Period, lastExtremumBar); //--- Store timestamp of the peak
            extrema[size].price = lastHigh;                              //--- Store price of the peak
            extrema[size].isPeak = true;                                 //--- Mark as a peak
            //Print("Extrema added: Bar ", lastExtremumBar, ", Time ", TimeToString(extrema[size].time), ", Price ", DoubleToString(lastHigh, _Digits), ", IsPeak true"); //--- Log new peak
            isUpTrend = false;                                           //--- Switch trend to downtrend
            lastLow = lows[i];                                           //--- Update last low price
            lastExtremumBar = i;                                         //--- Update last extremum bar index
         }
      } else {                                                        //--- If currently in a downtrend
         if (lows[i] < lastLow) {                                     //--- Check if current low is below last low
            lastLow = lows[i];                                        //--- Update last low price
            lastExtremumBar = i;                                      //--- Update last extremum bar index
         } else if (highs[i] > lastLow + ThresholdPoints * _Point) {  //--- Check if current high indicates a reversal (peak)
            int size = ArraySize(extrema);                            //--- Get current size of extrema array
            ArrayResize(extrema, size + 1);                           //--- Resize array to add new extremum
            extrema[size].bar = lastExtremumBar;                      //--- Store bar index of the trough
            extrema[size].time = iTime(_Symbol, _Period, lastExtremumBar); //--- Store timestamp of the trough
            extrema[size].price = lastLow;                            //--- Store price of the trough
            extrema[size].isPeak = false;                             //--- Mark as a trough
            //Print("Extrema added: Bar ", lastExtremumBar, ", Time ", TimeToString(extrema[size].time), ", Price ", DoubleToString(lastLow, _Digits), ", IsPeak false"); //--- Log new trough
            isUpTrend = true;                                         //--- Switch trend to uptrend
            lastHigh = highs[i];                                      //--- Update last high price
            lastExtremumBar = i;                                      //--- Update last extremum bar index
         }
      }
   }
}
```

Here, we pinpoint the peaks and troughs that define our head and shoulders pattern by implementing the "FindExtrema" function, which analyzes the last "lookback" bars to build an "extrema" array of critical price points. We begin by resetting the "extrema" array with the [ArrayFree](https://www.mql5.com/en/docs/array/arrayfree) function to ensure a clean slate, then use the "Bars" function to fetch the total available bars and cap "lookback" if it exceeds this limit, guaranteeing we stay within the chart’s data range. Next, we prepare "highs" and "lows" arrays to hold price data, setting them as time series with the [ArraySetAsSeries](https://www.mql5.com/en/docs/array/arraysetasseries) function (newest first), and populate them using [CopyHigh](https://www.mql5.com/en/docs/series/copyhigh) and [CopyLow](https://www.mql5.com/en/docs/series/copylow) to extract high and low prices over "lookback + 1" bars.

In a loop from oldest to newest bar, we determine the trend with "isUpTrend" based on initial price movement, then track "lastHigh" or "lastLow" and their "lastExtremumBar"; when a reversal exceeds "ThresholdPoints", we expand "extrema" with the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function, store details like "bar", "time" (via "iTime"), "price", and "isPeak" (true for peaks, false for troughs), and switch the trend, enabling precise pattern identification. We can now take the identified price levels and store them for further use.

```
Extremum extrema[];                                                   //--- Array to store identified peaks and troughs
FindExtrema(extrema, LookbackBars);                                   //--- Find extrema in the last LookbackBars bars
```

Here, we declare an "extrema" [array](https://www.mql5.com/en/book/basis/arrays/arrays_usage) of type "Extremum" to hold identified peaks and troughs, which will store the pattern’s shoulders and head. We then call the "FindExtrema" function, passing "extrema" and "LookbackBars" as arguments, to scan the last "LookbackBars" bars and populate the array with key extrema, laying the groundwork for pattern recognition and subsequent trading decisions. When we print the array values using the [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint) function, we have something that depicts the below structure.

![STORED PRICE DATA](https://c.mql5.com/2/127/Screenshot_2025-03-24_182902.png)

This confirms we have the necessary data points. So we can proceed to identifying the pattern components. To make the code modularized, we employ functions.

```
//+------------------------------------------------------------------+
//| Detect standard Head and Shoulders pattern                       |
//+------------------------------------------------------------------+
bool DetectHeadAndShoulders(Extremum &extrema[], int &leftShoulderIdx, int &headIdx, int &rightShoulderIdx, int &necklineStartIdx, int &necklineEndIdx) { //--- Function to detect standard H&S pattern
   int size = ArraySize(extrema);                                        //--- Get the size of the extrema array
   if (size < 6) return false;                                           //--- Return false if insufficient extrema for pattern (need at least 6 points)

   for (int i = size - 6; i >= 0; i--) {                                 //--- Loop through extrema to find H&S pattern (start at size-6 to ensure enough points)
      if (!extrema[i].isPeak && extrema[i+1].isPeak && !extrema[i+2].isPeak && //--- Check sequence: trough, peak (LS), trough
          extrema[i+3].isPeak && !extrema[i+4].isPeak && extrema[i+5].isPeak) { //--- Check sequence: peak (head), trough, peak (RS)
         double leftShoulder = extrema[i+1].price;                       //--- Get price of left shoulder
         double head = extrema[i+3].price;                               //--- Get price of head
         double rightShoulder = extrema[i+5].price;                      //--- Get price of right shoulder
         double trough1 = extrema[i+2].price;                            //--- Get price of first trough (neckline start)
         double trough2 = extrema[i+4].price;                            //--- Get price of second trough (neckline end)

         bool isHeadHighest = true;                                      //--- Flag to verify head is the highest peak in range
         for (int j = MathMax(0, i - 5); j < MathMin(size, i + 10); j++) { //--- Check surrounding bars (5 before, 10 after) for higher peaks
            if (extrema[j].isPeak && extrema[j].price > head && j != i + 3) { //--- If another peak is higher than head
               isHeadHighest = false;                                    //--- Set flag to false
               break;                                                    //--- Exit loop as head is not highest
            }
         }

         int lsBar = extrema[i+1].bar;                                   //--- Get bar index of left shoulder
         int headBar = extrema[i+3].bar;                                 //--- Get bar index of head
         int rsBar = extrema[i+5].bar;                                   //--- Get bar index of right shoulder
         int lsToHead = lsBar - headBar;                                 //--- Calculate bars from left shoulder to head
         int headToRs = headBar - rsBar;                                 //--- Calculate bars from head to right shoulder

         if (lsToHead < MinBarRange || lsToHead > MaxBarRange || headToRs < MinBarRange || headToRs > MaxBarRange) continue; //--- Skip if bar ranges are out of bounds

         int minRange = MathMin(lsToHead, headToRs);                     //--- Get the smaller of the two ranges for uniformity check
         if (lsToHead > minRange * BarRangeMultiplier || headToRs > minRange * BarRangeMultiplier) continue; //--- Skip if ranges exceed uniformity multiplier

         bool rsValid = false;                                           //--- Flag to validate right shoulder breakout
         int rsBarIndex = extrema[i+5].bar;                              //--- Get bar index of right shoulder for validation
         for (int j = rsBarIndex - 1; j >= MathMax(0, rsBarIndex - ValidationBars); j--) { //--- Check bars after right shoulder for breakout
            if (iLow(_Symbol, _Period, j) < rightShoulder - ThresholdPoints * _Point) { //--- Check if price drops below RS by threshold
               rsValid = true;                                           //--- Set flag to true if breakout confirmed
               break;                                                    //--- Exit loop once breakout is validated
            }
         }
         if (!rsValid) continue;                                         //--- Skip if right shoulder breakout not validated

         if (isHeadHighest && head > leftShoulder && head > rightShoulder && //--- Verify head is highest and above shoulders
             MathAbs(leftShoulder - rightShoulder) < ShoulderTolerancePoints * _Point && //--- Check shoulder price difference within tolerance
             MathAbs(trough1 - trough2) < TroughTolerancePoints * _Point) { //--- Check trough price difference within tolerance
            leftShoulderIdx = i + 1;                                     //--- Set index for left shoulder
            headIdx = i + 3;                                             //--- Set index for head
            rightShoulderIdx = i + 5;                                    //--- Set index for right shoulder
            necklineStartIdx = i + 2;                                    //--- Set index for neckline start (first trough)
            necklineEndIdx = i + 4;                                      //--- Set index for neckline end (second trough)
            Print("Bar Ranges: LS to Head = ", lsToHead, ", Head to RS = ", headToRs); //--- Log bar ranges for debugging
            return true;                                                 //--- Return true to indicate pattern found
         }
      }
   }
   return false;                                                         //--- Return false if no pattern detected
}
```

Here, we spot the standard pattern through the "DetectHeadAndShoulders" function, which examines the "extrema" array to find a valid sequence of six points: a trough, a peak (left shoulder), a trough, a peak (head), a trough, and a peak (right shoulder), requiring at least six entries as checked by the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function. We loop through "extrema" starting at "size - 6", verifying the pattern’s alternating peak-trough structure, then extract prices for "leftShoulder", "head", "rightShoulder", and neckline troughs ("trough1", "trough2"); a nested loop ensures the head is the highest peak within a range using the [MathMax](https://www.mql5.com/en/docs/math/mathmax) and [MathMin](https://www.mql5.com/en/docs/math/mathmin) functions, while bar distances between points are constrained by "MinBarRange" and "MaxBarRange" and uniformity by "BarRangeMultiplier".

We confirm the right shoulder breakout by checking the [iLow](https://www.mql5.com/en/docs/series/ilow) function against "ThresholdPoints" over "ValidationBars", and if the head exceeds both shoulders and tolerances ("ShoulderTolerancePoints", "TroughTolerancePoints") are met, we assign indices like "leftShoulderIdx", "headIdx", and "necklineStartIdx", log bar ranges with the [Print](https://www.mql5.com/en/docs/common/print) function for debugging, and return true to signal a detected pattern, otherwise returning false. We use the same logic to find the inversed pattern.

```
//+------------------------------------------------------------------+
//| Detect inverse Head and Shoulders pattern                        |
//+------------------------------------------------------------------+
bool DetectInverseHeadAndShoulders(Extremum &extrema[], int &leftShoulderIdx, int &headIdx, int &rightShoulderIdx, int &necklineStartIdx, int &necklineEndIdx) { //--- Function to detect inverse H&S pattern
   int size = ArraySize(extrema);                                        //--- Get the size of the extrema array
   if (size < 6) return false;                                           //--- Return false if insufficient extrema for pattern (need at least 6 points)

   for (int i = size - 6; i >= 0; i--) {                                 //--- Loop through extrema to find inverse H&S pattern
      if (extrema[i].isPeak && !extrema[i+1].isPeak && extrema[i+2].isPeak && //--- Check sequence: peak, trough (LS), peak
          !extrema[i+3].isPeak && extrema[i+4].isPeak && !extrema[i+5].isPeak) { //--- Check sequence: trough (head), peak, trough (RS)
         double leftShoulder = extrema[i+1].price;                       //--- Get price of left shoulder
         double head = extrema[i+3].price;                               //--- Get price of head
         double rightShoulder = extrema[i+5].price;                      //--- Get price of right shoulder
         double peak1 = extrema[i+2].price;                              //--- Get price of first peak (neckline start)
         double peak2 = extrema[i+4].price;                              //--- Get price of second peak (neckline end)

         bool isHeadLowest = true;                                       //--- Flag to verify head is the lowest trough in range
         int headBar = extrema[i+3].bar;                                 //--- Get bar index of head for range check
         for (int j = MathMax(0, headBar - 5); j <= MathMin(Bars(_Symbol, _Period) - 1, headBar + 5); j++) { //--- Check 5 bars before and after head
            if (iLow(_Symbol, _Period, j) < head) {                      //--- If any low is below head
               isHeadLowest = false;                                     //--- Set flag to false
               break;                                                    //--- Exit loop as head is not lowest
            }
         }

         int lsBar = extrema[i+1].bar;                                   //--- Get bar index of left shoulder
         int rsBar = extrema[i+5].bar;                                   //--- Get bar index of right shoulder
         int lsToHead = lsBar - headBar;                                 //--- Calculate bars from left shoulder to head
         int headToRs = headBar - rsBar;                                 //--- Calculate bars from head to right shoulder

         if (lsToHead < MinBarRange || lsToHead > MaxBarRange || headToRs < MinBarRange || headToRs > MaxBarRange) continue; //--- Skip if bar ranges are out of bounds

         int minRange = MathMin(lsToHead, headToRs);                     //--- Get the smaller of the two ranges for uniformity check
         if (lsToHead > minRange * BarRangeMultiplier || headToRs > minRange * BarRangeMultiplier) continue; //--- Skip if ranges exceed uniformity multiplier

         bool rsValid = false;                                           //--- Flag to validate right shoulder breakout
         int rsBarIndex = extrema[i+5].bar;                              //--- Get bar index of right shoulder for validation
         for (int j = rsBarIndex - 1; j >= MathMax(0, rsBarIndex - ValidationBars); j--) { //--- Check bars after right shoulder for breakout
            if (iHigh(_Symbol, _Period, j) > rightShoulder + ThresholdPoints * _Point) { //--- Check if price rises above RS by threshold
               rsValid = true;                                           //--- Set flag to true if breakout confirmed
               break;                                                    //--- Exit loop once breakout is validated
            }
         }
         if (!rsValid) continue;                                         //--- Skip if right shoulder breakout not validated

         if (isHeadLowest && head < leftShoulder && head < rightShoulder && //--- Verify head is lowest and below shoulders
             MathAbs(leftShoulder - rightShoulder) < ShoulderTolerancePoints * _Point && //--- Check shoulder price difference within tolerance
             MathAbs(peak1 - peak2) < TroughTolerancePoints * _Point) { //--- Check peak price difference within tolerance
            leftShoulderIdx = i + 1;                                     //--- Set index for left shoulder
            headIdx = i + 3;                                             //--- Set index for head
            rightShoulderIdx = i + 5;                                    //--- Set index for right shoulder
            necklineStartIdx = i + 2;                                    //--- Set index for neckline start (first peak)
            necklineEndIdx = i + 4;                                      //--- Set index for neckline end (second peak)
            Print("Bar Ranges: LS to Head = ", lsToHead, ", Head to RS = ", headToRs); //--- Log bar ranges for debugging
            return true;                                                 //--- Return true to indicate pattern found
         }
      }
   }
   return false;                                                         //--- Return false if no pattern detected
}
```

We define the "DetectInverseHeadAndShoulders" function to identify the inverse pattern, which sifts through the "extrema" array to locate a sequence of six points—peak, trough (left shoulder), peak, trough (head), peak, trough (right shoulder)—needing at least six entries as verified by the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function. We iterate from "size - 6" downward, confirming the pattern’s peak-trough alternation, then pull prices for "leftShoulder", "head", "rightShoulder", and neckline peaks ("peak1", "peak2"); a nested loop checks if the head is the lowest trough within a five-bar range around "headBar" using the [MathMax](https://www.mql5.com/en/docs/math/mathmax), [MathMin](https://www.mql5.com/en/docs/math/mathmin), and [iLow](https://www.mql5.com/en/docs/series/ilow) functions, while "Bars" ensures we stay within chart limits.

We enforce bar spacing with "MinBarRange" and "MaxBarRange", calculate uniformity with the MathMin function and "BarRangeMultiplier", and validate the right shoulder breakout using the [iHigh](https://www.mql5.com/en/docs/series/ihigh) function against "ThresholdPoints" over "ValidationBars"; if "head" is below both shoulders and tolerances ("ShoulderTolerancePoints", "TroughTolerancePoints") are satisfied, we set indices like "leftShoulderIdx" and "necklineStartIdx", log ranges, and return true, otherwise returning false. With these 2 functions now, we can proceed to identify the patterns as below.

```
int leftShoulderIdx, headIdx, rightShoulderIdx, necklineStartIdx, necklineEndIdx; //--- Indices for pattern components

// Standard Head and Shoulders (Sell)
if (DetectHeadAndShoulders(extrema, leftShoulderIdx, headIdx, rightShoulderIdx, necklineStartIdx, necklineEndIdx)) { //--- Check for standard H&S pattern
   double closePrice = iClose(_Symbol, _Period, 1);                   //--- Get the closing price of the previous bar
   double necklinePrice = extrema[necklineEndIdx].price;              //--- Get the price of the neckline end point

   if (closePrice < necklinePrice) {                                  //--- Check if price has broken below the neckline (sell signal)
      datetime lsTime = extrema[leftShoulderIdx].time;                //--- Get the timestamp of the left shoulder
      double lsPrice = extrema[leftShoulderIdx].price;                //--- Get the price of the left shoulder

      //---
   }
}
```

Here, we advance by declaring variables "leftShoulderIdx", "headIdx", "rightShoulderIdx", "necklineStartIdx", and "necklineEndIdx" to store indices of the pattern’s components, then use the "DetectHeadAndShoulders" function to check the "extrema" array for a standard pattern, passing these indices as references. If detected, we retrieve "closePrice" with the [iClose](https://www.mql5.com/en/docs/series/iclose) function for the previous bar and "necklinePrice" from "extrema\[necklineEndIdx\].price", triggering a sell signal if "closePrice" dips below "necklinePrice"; we then extract "lsTime" and "lsPrice" from "extrema\[leftShoulderIdx\]" to prepare for trade execution based on the left shoulder’s position. At this point, we need to make sure that the pattern is not traded. We define a function to do the check.

```
//+------------------------------------------------------------------+
//| Check if pattern has already been traded                         |
//+------------------------------------------------------------------+
bool IsPatternTraded(datetime lsTime, double lsPrice) {                  //--- Function to check if a pattern has already been traded
   int size = ArraySize(tradedPatterns);                                 //--- Get the current size of the tradedPatterns array
   for (int i = 0; i < size; i++) {                                      //--- Loop through all stored traded patterns
      if (tradedPatterns[i].leftShoulderTime == lsTime &&                //--- Check if left shoulder time matches
          MathAbs(tradedPatterns[i].leftShoulderPrice - lsPrice) < PriceTolerance * _Point) { //--- Check if left shoulder price is within tolerance
         Print("Pattern already traded: Left Shoulder Time ", TimeToString(lsTime), ", Price ", DoubleToString(lsPrice, _Digits)); //--- Log that pattern was previously traded
         return true;                                                    //--- Return true to indicate pattern has been traded
      }
   }
   return false;                                                         //--- Return false if no match found
}
```

Here, we ensure our program avoids duplicate trades by implementing the "IsPatternTraded" function, which checks if a pattern, identified by "lsTime" and "lsPrice", already exists in the "tradedPatterns" array. We use the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function to get the array’s "size", then loop through it, comparing each entry’s "leftShoulderTime" with "lsTime" and "leftShoulderPrice" with "lsPrice" within a "PriceTolerance" range via the [MathAbs](https://www.mql5.com/en/docs/math/mathabs) function; if a match is found, we log it with the [Print](https://www.mql5.com/en/docs/common/print) function, including [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) and [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) for readability, and return true, otherwise returning false to allow a new trade. We then call the function to do the check and proceed if none is found.

```
if (IsPatternTraded(lsTime, lsPrice)) return;                   //--- Exit if this pattern has already been traded

datetime breakoutTime = iTime(_Symbol, _Period, 1);             //--- Get the timestamp of the breakout bar (previous bar)
int lsBar = extrema[leftShoulderIdx].bar;                       //--- Get the bar index of the left shoulder
int headBar = extrema[headIdx].bar;                             //--- Get the bar index of the head
int rsBar = extrema[rightShoulderIdx].bar;                      //--- Get the bar index of the right shoulder
int necklineStartBar = extrema[necklineStartIdx].bar;           //--- Get the bar index of the neckline start
int necklineEndBar = extrema[necklineEndIdx].bar;               //--- Get the bar index of the neckline end
int breakoutBar = 1;                                            //--- Set breakout bar index (previous bar)

int lsToHead = lsBar - headBar;                                 //--- Calculate number of bars from left shoulder to head
int headToRs = headBar - rsBar;                                 //--- Calculate number of bars from head to right shoulder
int rsToBreakout = rsBar - breakoutBar;                         //--- Calculate number of bars from right shoulder to breakout
int lsToNeckStart = lsBar - necklineStartBar;                   //--- Calculate number of bars from left shoulder to neckline start
double avgPatternRange = (lsToHead + headToRs) / 2.0;           //--- Calculate average bar range of the pattern for uniformity check

if (rsToBreakout > avgPatternRange * RightShoulderBreakoutMultiplier) { //--- Check if breakout distance exceeds allowed range
   Print("Pattern rejected: Right Shoulder to Breakout (", rsToBreakout,
         ") exceeds ", RightShoulderBreakoutMultiplier, "x average range (", avgPatternRange, ")"); //--- Log rejection due to excessive breakout range
   return;                                                      //--- Exit function if pattern is invalid
}

double necklineStartPrice = extrema[necklineStartIdx].price;    //--- Get the price of the neckline start point
double necklineEndPrice = extrema[necklineEndIdx].price;        //--- Get the price of the neckline end point
datetime necklineStartTime = extrema[necklineStartIdx].time;    //--- Get the timestamp of the neckline start point
datetime necklineEndTime = extrema[necklineEndIdx].time;        //--- Get the timestamp of the neckline end point
int barDiff = necklineStartBar - necklineEndBar;                //--- Calculate bar difference between neckline points for slope
double slope = (necklineEndPrice - necklineStartPrice) / barDiff; //--- Calculate the slope of the neckline (price change per bar)
double breakoutNecklinePrice = necklineStartPrice + slope * (necklineStartBar - breakoutBar); //--- Calculate neckline price at breakout point

// Extend neckline backwards
int extendedBar = necklineStartBar;                             //--- Initialize extended bar index with neckline start
datetime extendedNecklineStartTime = necklineStartTime;         //--- Initialize extended neckline start time
double extendedNecklineStartPrice = necklineStartPrice;         //--- Initialize extended neckline start price
bool foundCrossing = false;                                     //--- Flag to track if neckline crosses a bar within range

for (int i = necklineStartBar + 1; i < Bars(_Symbol, _Period); i++) { //--- Loop through bars to extend neckline backwards
   double checkPrice = necklineStartPrice - slope * (i - necklineStartBar); //--- Calculate projected neckline price at bar i
   if (NecklineCrossesBar(checkPrice, i)) {                     //--- Check if neckline intersects the bar's high-low range
      int distance = i - necklineStartBar;                      //--- Calculate distance from neckline start to crossing bar
      if (distance <= avgPatternRange * RightShoulderBreakoutMultiplier) { //--- Check if crossing is within uniformity range
         extendedBar = i;                                       //--- Update extended bar index
         extendedNecklineStartTime = iTime(_Symbol, _Period, i); //--- Update extended neckline start time
         extendedNecklineStartPrice = checkPrice;              //--- Update extended neckline start price
         foundCrossing = true;                                  //--- Set flag to indicate crossing found
         Print("Neckline extended to first crossing bar within uniformity: Bar ", extendedBar); //--- Log successful extension
         break;                                                 //--- Exit loop after finding valid crossing
      } else {                                                  //--- If crossing exceeds uniformity range
         Print("Crossing bar ", i, " exceeds uniformity (", distance, " > ", avgPatternRange * RightShoulderBreakoutMultiplier, ")"); //--- Log rejection of crossing
         break;                                                 //--- Exit loop as crossing is too far
      }
   }
}

if (!foundCrossing) {                                           //--- If no valid crossing found within range
   int barsToExtend = 2 * lsToNeckStart;                        //--- Set fallback extension distance as twice LS to neckline start
   extendedBar = necklineStartBar + barsToExtend;               //--- Calculate extended bar index
   if (extendedBar >= Bars(_Symbol, _Period)) extendedBar = Bars(_Symbol, _Period) - 1; //--- Cap extended bar at total bars if exceeded
   extendedNecklineStartTime = iTime(_Symbol, _Period, extendedBar); //--- Update extended neckline start time
   extendedNecklineStartPrice = necklineStartPrice - slope * (extendedBar - necklineStartBar); //--- Update extended neckline start price
   Print("Neckline extended to fallback (2x LS to Neckline Start): Bar ", extendedBar, " (no crossing within uniformity)"); //--- Log fallback extension
}

Print("Standard Head and Shoulders Detected:");                 //--- Log detection of standard H&S pattern
Print("Left Shoulder: Bar ", lsBar, ", Time ", TimeToString(lsTime), ", Price ", DoubleToString(lsPrice, _Digits)); //--- Log left shoulder details
Print("Head: Bar ", headBar, ", Time ", TimeToString(extrema[headIdx].time), ", Price ", DoubleToString(extrema[headIdx].price, _Digits)); //--- Log head details
Print("Right Shoulder: Bar ", rsBar, ", Time ", TimeToString(extrema[rightShoulderIdx].time), ", Price ", DoubleToString(extrema[rightShoulderIdx].price, _Digits)); //--- Log right shoulder details
Print("Neckline Start: Bar ", necklineStartBar, ", Time ", TimeToString(necklineStartTime), ", Price ", DoubleToString(necklineStartPrice, _Digits)); //--- Log neckline start details
Print("Neckline End: Bar ", necklineEndBar, ", Time ", TimeToString(necklineEndTime), ", Price ", DoubleToString(necklineEndPrice, _Digits)); //--- Log neckline end details
Print("Close Price: ", DoubleToString(closePrice, _Digits));    //--- Log closing price at breakout
Print("Breakout Time: ", TimeToString(breakoutTime));           //--- Log breakout timestamp
Print("Neckline Price at Breakout: ", DoubleToString(breakoutNecklinePrice, _Digits)); //--- Log neckline price at breakout
Print("Extended Neckline Start: Bar ", extendedBar, ", Time ", TimeToString(extendedNecklineStartTime), ", Price ", DoubleToString(extendedNecklineStartPrice, _Digits)); //--- Log extended neckline start details
Print("Bar Ranges: LS to Head = ", lsToHead, ", Head to RS = ", headToRs, ", RS to Breakout = ", rsToBreakout, ", LS to Neckline Start = ", lsToNeckStart); //--- Log bar ranges for pattern analysis
```

Here, we enhance the pattern detection by validating a detected standard pattern and setting up a sell trade, beginning with the "IsPatternTraded" function to check if "lsTime" and "lsPrice" match a prior trade in "tradedPatterns", exiting if true to avoid duplicates. We then use the [iTime](https://www.mql5.com/en/docs/series/itime) function to assign "breakoutTime" as the previous bar’s timestamp and retrieve bar indices like "lsBar", "headBar", "rsBar", "necklineStartBar", and "necklineEndBar" from "extrema", calculating ranges such as "lsToHead", "headToRs", and "rsToBreakout"; if "rsToBreakout" exceeds "avgPatternRange" times "RightShoulderBreakoutMultiplier", we reject the pattern and log it with the [Print](https://www.mql5.com/en/docs/common/print) function.

Next, we determine the neckline’s "slope" using "necklineStartPrice" and "necklineEndPrice" over "barDiff", compute "breakoutNecklinePrice", and extend the neckline backward with a loop, using the "NecklineCrossesBar" function to find a crossing within "avgPatternRange \* RightShoulderBreakoutMultiplier", updating "extendedBar", "extendedNecklineStartTime" (via "iTime"), and "extendedNecklineStartPrice"; if no crossing fits, we fall back to "2 \* lsToNeckStart", capping at "Bars" total, and log all details—bar indices, prices, and ranges—with the Print, [TimeToString](https://www.mql5.com/en/docs/convert/timetostring), and [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) functions for thorough documentation. The custom function's code snippet is as below.

```
//+------------------------------------------------------------------+
//| Check if neckline crosses a bar's high-low range                 |
//+------------------------------------------------------------------+
bool NecklineCrossesBar(double necklinePrice, int barIndex) {            //--- Function to check if neckline price intersects a bar's range
   double high = iHigh(_Symbol, _Period, barIndex);                      //--- Get the high price of the specified bar
   double low = iLow(_Symbol, _Period, barIndex);                        //--- Get the low price of the specified bar
   return (necklinePrice >= low && necklinePrice <= high);               //--- Return true if neckline price is within bar's high-low range
}
```

The function checks if the "necklinePrice" intersects a bar’s price range at "barIndex" to ensure accurate neckline extension. We use the [iHigh](https://www.mql5.com/en/docs/series/ihigh) function to fetch the bar’s "high" price and the "iLow" function to get its "low" price, then return true if "necklinePrice" falls between "low" and "high", confirming the neckline crosses the bar’s range for pattern validation. If there is a validation of the pattern, we visualize it on the chart. We will need functions to draw and label it.

```
//+------------------------------------------------------------------+
//| Draw a trend line for visualization                              |
//+------------------------------------------------------------------+
void DrawTrendLine(string name, datetime timeStart, double priceStart, datetime timeEnd, double priceEnd, color lineColor, int width, int style) { //--- Function to draw a trend line on the chart
   if (ObjectCreate(0, name, OBJ_TREND, 0, timeStart, priceStart, timeEnd, priceEnd)) { //--- Create a trend line object if possible
      ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);               //--- Set the color of the trend line
      ObjectSetInteger(0, name, OBJPROP_STYLE, style);                   //--- Set the style (e.g., solid, dashed) of the trend line
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width);                   //--- Set the width of the trend line
      ObjectSetInteger(0, name, OBJPROP_BACK, true);                     //--- Set the line to draw behind chart elements
      ChartRedraw();                                                     //--- Redraw the chart to display the new line
   } else {                                                              //--- If line creation fails
      Print("Failed to create line: ", name, ". Error: ", GetLastError()); //--- Log the error with the object name and error code
   }
}

//+------------------------------------------------------------------+
//| Draw a filled triangle for visualization                         |
//+------------------------------------------------------------------+
void DrawTriangle(string name, datetime time1, double price1, datetime time2, double price2, datetime time3, double price3, color fillColor) { //--- Function to draw a filled triangle on the chart
   if (ObjectCreate(0, name, OBJ_TRIANGLE, 0, time1, price1, time2, price2, time3, price3)) { //--- Create a triangle object if possible
      ObjectSetInteger(0, name, OBJPROP_COLOR, fillColor);               //--- Set the fill color of the triangle
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);             //--- Set the border style to solid
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);                       //--- Set the border width to 1 pixel
      ObjectSetInteger(0, name, OBJPROP_FILL, true);                     //--- Enable filling of the triangle
      ObjectSetInteger(0, name, OBJPROP_BACK, true);                     //--- Set the triangle to draw behind chart elements
      ChartRedraw();                                                     //--- Redraw the chart to display the new triangle
   } else {                                                              //--- If triangle creation fails
      Print("Failed to create triangle: ", name, ". Error: ", GetLastError()); //--- Log the error with the object name and error code
   }
}

//+------------------------------------------------------------------+
//| Draw text label for visualization                                |
//+------------------------------------------------------------------+
void DrawText(string name, datetime time, double price, string text, color textColor, bool above, double angle = 0) { //--- Function to draw a text label on the chart
   int chartscale = (int)ChartGetInteger(0, CHART_SCALE);                //--- Get the current chart zoom level
   int dynamicFontSize = 5 + int(chartscale * 1.5);                      //--- Calculate font size based on zoom level for visibility
   double priceOffset = (above ? 10 : -10) * _Point;                     //--- Set price offset above or below the point for readability
   if (ObjectCreate(0, name, OBJ_TEXT, 0, time, price + priceOffset)) {  //--- Create a text object if possible
      ObjectSetString(0, name, OBJPROP_TEXT, text);                      //--- Set the text content of the label
      ObjectSetInteger(0, name, OBJPROP_COLOR, textColor);               //--- Set the color of the text
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, dynamicFontSize);      //--- Set the font size based on chart scale
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_CENTER);          //--- Center the text at the specified point
      ObjectSetDouble(0, name, OBJPROP_ANGLE, angle);                    //--- Set the rotation angle of the text in degrees
      ObjectSetInteger(0, name, OBJPROP_BACK, false);                    //--- Set the text to draw in front of chart elements
      ChartRedraw();                                                     //--- Redraw the chart to display the new text
      Print("Text created: ", name, ", Angle: ", DoubleToString(angle, 2)); //--- Log successful creation of the text with its angle
   } else {                                                              //--- If text creation fails
      Print("Failed to create text: ", name, ". Error: ", GetLastError()); //--- Log the error with the object name and error code
   }
}
```

Here, we enrich the program with visualization tools to highlight the pattern on the chart, starting with the "DrawTrendLine" function, which uses the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to plot a line from "timeStart" and "priceStart" to "timeEnd" and "priceEnd", setting properties like "lineColor", "style", and "width" via [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), drawing it behind bars with [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), and refreshing the display with the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function, logging failures with "Print" and [GetLastError](https://www.mql5.com/en/docs/check/getlasterror) if needed.

We then implement the "DrawTriangle" function to shade the pattern’s structure, calling the "ObjectCreate" function with three points ("time1", "price1", etc.), applying "fillColor" and a solid border using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger), filling it with [OBJPROP\_FILL](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), placing it behind the chart, and updating the view with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw), again logging errors with "Print" if creation fails.

Finally, we add the "DrawText" function to label key points, using the [ChartGetInteger](https://www.mql5.com/en/docs/chart_operations/chartgetinteger) function to adjust "dynamicFontSize" based on "chartscale", positioning text at "time" and "price" plus an offset via "ObjectCreate", customizing it with "ObjectSetString" for "text", "ObjectSetInteger" for "textColor" and "FONTSIZE", and "ObjectSetDouble" for "angle", drawing it in front with [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw), and confirming creation with "Print" and "DoubleToString" or noting errors. We now can call the functions to add the visibility feature, and the first thing we do is add the lines as follows.

```
string prefix = "HS_" + TimeToString(extrema[headIdx].time, TIME_MINUTES); //--- Create unique prefix for chart objects based on head time
// Lines
DrawTrendLine(prefix + "_LeftToNeckStart", lsTime, lsPrice, necklineStartTime, necklineStartPrice, clrRed, 3, STYLE_SOLID); //--- Draw line from left shoulder to neckline start
DrawTrendLine(prefix + "_NeckStartToHead", necklineStartTime, necklineStartPrice, extrema[headIdx].time, extrema[headIdx].price, clrRed, 3, STYLE_SOLID); //--- Draw line from neckline start to head
DrawTrendLine(prefix + "_HeadToNeckEnd", extrema[headIdx].time, extrema[headIdx].price, necklineEndTime, necklineEndPrice, clrRed, 3, STYLE_SOLID); //--- Draw line from head to neckline end
DrawTrendLine(prefix + "_NeckEndToRight", necklineEndTime, necklineEndPrice, extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, clrRed, 3, STYLE_SOLID); //--- Draw line from neckline end to right shoulder
DrawTrendLine(prefix + "_Neckline", extendedNecklineStartTime, extendedNecklineStartPrice, breakoutTime, breakoutNecklinePrice, clrBlue, 2, STYLE_SOLID); //--- Draw neckline from extended start to breakout
DrawTrendLine(prefix + "_RightToBreakout", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, breakoutTime, breakoutNecklinePrice, clrRed, 3, STYLE_SOLID); //--- Draw line from right shoulder to breakout
DrawTrendLine(prefix + "_ExtendedToLeftShoulder", extendedNecklineStartTime, extendedNecklineStartPrice, lsTime, lsPrice, clrRed, 3, STYLE_SOLID); //--- Draw line from extended neckline to left shoulder
```

Here, we visually map the standard pattern by creating a unique prefix based on the head’s timestamp using the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function and drawing trend lines with the "DrawTrendLine" function to connect the left shoulder to the neckline start, neckline start to the head, head to the neckline end, and neckline end to the right shoulder in red with a width of 3, while the neckline from its extended start to the breakout point uses blue with a width of 2, and additional lines link the right shoulder to the breakout and the extended neckline back to the left shoulder in red, all in solid style to display the pattern on the chart. Upon compilation, we have the following outcome.

![WITH LINES](https://c.mql5.com/2/127/Screenshot_2025-03-24_213619.png)

To add the triangles, we use the "DrawTriangle" function. Technically, we construct it within the shoulders and the head.

```
// Triangles
DrawTriangle(prefix + "_LeftShoulderTriangle", lsTime, lsPrice, necklineStartTime, necklineStartPrice, extendedNecklineStartTime, extendedNecklineStartPrice, clrLightCoral); //--- Draw triangle for left shoulder area
DrawTriangle(prefix + "_HeadTriangle", extrema[headIdx].time, extrema[headIdx].price, necklineStartTime, necklineStartPrice, necklineEndTime, necklineEndPrice, clrLightCoral); //--- Draw triangle for head area
DrawTriangle(prefix + "_RightShoulderTriangle", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, necklineEndTime, necklineEndPrice, breakoutTime, breakoutNecklinePrice, clrLightCoral); //--- Draw triangle for right shoulder area
```

Here, we enhance the visualization by using the "DrawTriangle" function to shade key areas in light coral, forming a triangle for the left shoulder from the left shoulder point to the neckline start and extended neckline start, another for the head from the head point to the neckline start and end, and a third for the right shoulder from the right shoulder point to the neckline end and breakout point, highlighting the pattern’s structure on the chart. Upon compilation, we have the following outcome.

![WITH TRIANGLES](https://c.mql5.com/2/127/Screenshot_2025-03-24_213827.png)

Finally, we need to add labels to the pattern to make it fully visually appealing and self-illustrative.

```
// Text Labels
DrawText(prefix + "_LS_Label", lsTime, lsPrice, "LS", clrRed, true); //--- Draw "LS" label above left shoulder
DrawText(prefix + "_Head_Label", extrema[headIdx].time, extrema[headIdx].price, "HEAD", clrRed, true); //--- Draw "HEAD" label above head
DrawText(prefix + "_RS_Label", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, "RS", clrRed, true); //--- Draw "RS" label above right shoulder
datetime necklineMidTime = extendedNecklineStartTime + (breakoutTime - extendedNecklineStartTime) / 2; //--- Calculate midpoint time of the neckline
double necklineMidPrice = extendedNecklineStartPrice + slope * (iBarShift(_Symbol, _Period, extendedNecklineStartTime) - iBarShift(_Symbol, _Period, necklineMidTime)); //--- Calculate midpoint price of the neckline
// Calculate angle in pixel space
int x1 = ShiftToX(iBarShift(_Symbol, _Period, extendedNecklineStartTime)); //--- Convert extended neckline start to x-pixel coordinate
int y1 = PriceToY(extendedNecklineStartPrice);                          //--- Convert extended neckline start price to y-pixel coordinate
int x2 = ShiftToX(iBarShift(_Symbol, _Period, breakoutTime));           //--- Convert breakout time to x-pixel coordinate
int y2 = PriceToY(breakoutNecklinePrice);                               //--- Convert breakout price to y-pixel coordinate
double pixelSlope = (y2 - y1) / (double)(x2 - x1);                     //--- Calculate slope in pixel space (rise over run)
double necklineAngle = -atan(pixelSlope) * 180 / M_PI;                  //--- Calculate neckline angle in degrees, negated for visual alignment
Print("Pixel X1: ", x1, ", Y1: ", y1, ", X2: ", x2, ", Y2: ", y2, ", Pixel Slope: ", DoubleToString(pixelSlope, 4), ", Neckline Angle: ", DoubleToString(necklineAngle, 2)); //--- Log pixel coordinates and angle
DrawText(prefix + "_Neckline_Label", necklineMidTime, necklineMidPrice, "NECKLINE", clrBlue, false, necklineAngle); //--- Draw "NECKLINE" label at midpoint with calculated angle
```

Finally, we annotate the pattern by using the "DrawText" function to place red "LS", "HEAD", and "RS" label above the left shoulder, head, and right shoulder points at their respective times and prices, enhancing chart readability. We then calculate the neckline’s midpoint by averaging "extendedNecklineStartTime" and "breakoutTime" for "necklineMidTime" and adjust "extendedNecklineStartPrice" with "slope" and bar differences via the [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) function for "necklineMidPrice"; to align the label, we convert times to x-pixels with the "ShiftToX" function and prices to y-pixels with the "PriceToY" function at the neckline’s start and breakout, compute a "pixelSlope", and derive "necklineAngle" in degrees using the [atan](https://www.mql5.com/en/docs/math/matharctan) function and "M\_PI", logging these with the "Print" function and the [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) function for verification.

We then draw a blue "NECKLINE" label at the midpoint with the "DrawText" function, positioned below and rotated to match "necklineAngle", ensuring the annotation follows the neckline’s tilt. Here is the outcome.

![FINAL PATTERN OUTCOME](https://c.mql5.com/2/127/Screenshot_2025-03-24_214035.png)

From the image, we can see that the pattern is fully visualized. We now need to detect its breakout, basically, the extended line, open a sell position, and modify the extension range to the breakout bar. Easy-peasy. We achieve that via the following logic.

```
double entryPrice = 0;                                                  //--- Set entry price to 0 for market order (uses current price)
double sl = extrema[rightShoulderIdx].price + BufferPoints * _Point;    //--- Calculate stop-loss above right shoulder with buffer
double patternHeight = extrema[headIdx].price - necklinePrice;          //--- Calculate pattern height from head to neckline
double tp = closePrice - patternHeight;                                 //--- Calculate take-profit below close by pattern height
if (sl > closePrice && tp < closePrice) {                               //--- Validate trade direction (SL above, TP below for sell)
   if (obj_Trade.Sell(LotSize, _Symbol, entryPrice, sl, tp, "Head and Shoulders")) { //--- Attempt to open a sell trade
      AddTradedPattern(lsTime, lsPrice);                                //--- Add pattern to traded list
      Print("Sell Trade Opened: SL ", DoubleToString(sl, _Digits), ", TP ", DoubleToString(tp, _Digits)); //--- Log successful trade opening
   }
}
```

Once the pattern is confirmed, we execute a sell trade by setting "entryPrice" to 0 for a market order, calculating "sl" above the right shoulder price with "BufferPoints", determining "patternHeight" as the difference between the head and neckline prices, and setting "tp" below "closePrice" by "patternHeight".

We validate the trade direction—ensuring "sl" is above and "tp" below "closePrice"—before using the "Sell" function on "obj\_Trade" to open the trade with "LotSize", "sl", "tp", and comment; if successful, we call the "AddTradedPattern" function with "lsTime" and "lsPrice" to log the pattern and use the [Print](https://www.mql5.com/en/docs/common/print) function with [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) to record "sl" and "tp" details. The custom function's code snippet to mark the pattern as traded is as below.

```
//+------------------------------------------------------------------+
//| Add pattern to traded list with size management                  |
//+------------------------------------------------------------------+
void AddTradedPattern(datetime lsTime, double lsPrice) {                 //--- Function to add a new traded pattern to the list
   int size = ArraySize(tradedPatterns);                                 //--- Get the current size of the tradedPatterns array
   if (size >= MaxTradedPatterns) {                                      //--- Check if array size exceeds maximum allowed
      for (int i = 0; i < size - 1; i++) {                               //--- Shift all elements left to remove the oldest
         tradedPatterns[i] = tradedPatterns[i + 1];                      //--- Copy next element to current position
      }
      ArrayResize(tradedPatterns, size - 1);                            //--- Reduce array size by 1
      size--;                                                           //--- Decrement size variable
      Print("Removed oldest traded pattern to maintain max size of ", MaxTradedPatterns); //--- Log removal of oldest pattern
   }
   ArrayResize(tradedPatterns, size + 1);                                //--- Increase array size to add new pattern
   tradedPatterns[size].leftShoulderTime = lsTime;                       //--- Store the left shoulder time of the new pattern
   tradedPatterns[size].leftShoulderPrice = lsPrice;                     //--- Store the left shoulder price of the new pattern
   Print("Added traded pattern: Left Shoulder Time ", TimeToString(lsTime), ", Price ", DoubleToString(lsPrice, _Digits)); //--- Log addition of new pattern
}
```

We define the "AddTradedPattern" function to keep track of traded setups. It uses "lsTime" and "lsPrice" to log left shoulder details since the left shoulder does not repaint. We check the "tradedPatterns" size with the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function. If it hits "MaxTradedPatterns", we shift elements left to drop the oldest. We resize "tradedPatterns" with the "ArrayResize" function to shrink it. We log this and then expand "tradedPatterns" using the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function for a new entry. We set "leftShoulderTime" to "lsTime" and "leftShoulderPrice" to "lsPrice". We log the addition with the [Print](https://www.mql5.com/en/docs/common/print) function, the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function, and the [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) function. Upon compilation, we have the following outcome.

![TRADED SETUP](https://c.mql5.com/2/127/Screenshot_2025-03-24_215025.png)

From the image, we can see that we not only visualize the setup but also trade it accordingly. The inverse head and shoulder pattern recognition, visualization, and trade operation employ the same logic, in an inversed manner. Here is its logic.

```
// Inverse Head and Shoulders (Buy)
if (DetectInverseHeadAndShoulders(extrema, leftShoulderIdx, headIdx, rightShoulderIdx, necklineStartIdx, necklineEndIdx)) { //--- Check for inverse H&S pattern
   double closePrice = iClose(_Symbol, _Period, 1);                   //--- Get the closing price of the previous bar
   double necklinePrice = extrema[necklineEndIdx].price;              //--- Get the price of the neckline end point

   if (closePrice > necklinePrice) {                                  //--- Check if price has broken above the neckline (buy signal)
      datetime lsTime = extrema[leftShoulderIdx].time;                //--- Get the timestamp of the left shoulder
      double lsPrice = extrema[leftShoulderIdx].price;                //--- Get the price of the left shoulder

      if (IsPatternTraded(lsTime, lsPrice)) return;                   //--- Exit if this pattern has already been traded

      datetime breakoutTime = iTime(_Symbol, _Period, 1);             //--- Get the timestamp of the breakout bar (previous bar)
      int lsBar = extrema[leftShoulderIdx].bar;                       //--- Get the bar index of the left shoulder
      int headBar = extrema[headIdx].bar;                             //--- Get the bar index of the head
      int rsBar = extrema[rightShoulderIdx].bar;                      //--- Get the bar index of the right shoulder
      int necklineStartBar = extrema[necklineStartIdx].bar;           //--- Get the bar index of the neckline start
      int necklineEndBar = extrema[necklineEndIdx].bar;               //--- Get the bar index of the neckline end
      int breakoutBar = 1;                                            //--- Set breakout bar index (previous bar)

      int lsToHead = lsBar - headBar;                                 //--- Calculate number of bars from left shoulder to head
      int headToRs = headBar - rsBar;                                 //--- Calculate number of bars from head to right shoulder
      int rsToBreakout = rsBar - breakoutBar;                         //--- Calculate number of bars from right shoulder to breakout
      int lsToNeckStart = lsBar - necklineStartBar;                   //--- Calculate number of bars from left shoulder to neckline start
      double avgPatternRange = (lsToHead + headToRs) / 2.0;           //--- Calculate average bar range of the pattern for uniformity check

      if (rsToBreakout > avgPatternRange * RightShoulderBreakoutMultiplier) { //--- Check if breakout distance exceeds allowed range
         Print("Pattern rejected: Right Shoulder to Breakout (", rsToBreakout,
               ") exceeds ", RightShoulderBreakoutMultiplier, "x average range (", avgPatternRange, ")"); //--- Log rejection due to excessive breakout range
         return;                                                      //--- Exit function if pattern is invalid
      }

      double necklineStartPrice = extrema[necklineStartIdx].price;    //--- Get the price of the neckline start point
      double necklineEndPrice = extrema[necklineEndIdx].price;        //--- Get the price of the neckline end point
      datetime necklineStartTime = extrema[necklineStartIdx].time;    //--- Get the timestamp of the neckline start point
      datetime necklineEndTime = extrema[necklineEndIdx].time;        //--- Get the timestamp of the neckline end point
      int barDiff = necklineStartBar - necklineEndBar;                //--- Calculate bar difference between neckline points for slope
      double slope = (necklineEndPrice - necklineStartPrice) / barDiff; //--- Calculate the slope of the neckline (price change per bar)
      double breakoutNecklinePrice = necklineStartPrice + slope * (necklineStartBar - breakoutBar); //--- Calculate neckline price at breakout point

      // Extend neckline backwards
      int extendedBar = necklineStartBar;                             //--- Initialize extended bar index with neckline start
      datetime extendedNecklineStartTime = necklineStartTime;         //--- Initialize extended neckline start time
      double extendedNecklineStartPrice = necklineStartPrice;         //--- Initialize extended neckline start price
      bool foundCrossing = false;                                     //--- Flag to track if neckline crosses a bar within range

      for (int i = necklineStartBar + 1; i < Bars(_Symbol, _Period); i++) { //--- Loop through bars to extend neckline backwards
         double checkPrice = necklineStartPrice - slope * (i - necklineStartBar); //--- Calculate projected neckline price at bar i
         if (NecklineCrossesBar(checkPrice, i)) {                     //--- Check if neckline intersects the bar's high-low range
            int distance = i - necklineStartBar;                      //--- Calculate distance from neckline start to crossing bar
            if (distance <= avgPatternRange * RightShoulderBreakoutMultiplier) { //--- Check if crossing is within uniformity range
               extendedBar = i;                                       //--- Update extended bar index
               extendedNecklineStartTime = iTime(_Symbol, _Period, i); //--- Update extended neckline start time
               extendedNecklineStartPrice = checkPrice;              //--- Update extended neckline start price
               foundCrossing = true;                                  //--- Set flag to indicate crossing found
               Print("Neckline extended to first crossing bar within uniformity: Bar ", extendedBar); //--- Log successful extension
               break;                                                 //--- Exit loop after finding valid crossing
            } else {                                                  //--- If crossing exceeds uniformity range
               Print("Crossing bar ", i, " exceeds uniformity (", distance, " > ", avgPatternRange * RightShoulderBreakoutMultiplier, ")"); //--- Log rejection of crossing
               break;                                                 //--- Exit loop as crossing is too far
            }
         }
      }

      if (!foundCrossing) {                                           //--- If no valid crossing found within range
         int barsToExtend = 2 * lsToNeckStart;                        //--- Set fallback extension distance as twice LS to neckline start
         extendedBar = necklineStartBar + barsToExtend;               //--- Calculate extended bar index
         if (extendedBar >= Bars(_Symbol, _Period)) extendedBar = Bars(_Symbol, _Period) - 1; //--- Cap extended bar at total bars if exceeded
         extendedNecklineStartTime = iTime(_Symbol, _Period, extendedBar); //--- Update extended neckline start time
         extendedNecklineStartPrice = necklineStartPrice - slope * (extendedBar - necklineStartBar); //--- Update extended neckline start price
         Print("Neckline extended to fallback (2x LS to Neckline Start): Bar ", extendedBar, " (no crossing within uniformity)"); //--- Log fallback extension
      }

      Print("Inverse Head and Shoulders Detected:");                  //--- Log detection of inverse H&S pattern
      Print("Left Shoulder: Bar ", lsBar, ", Time ", TimeToString(lsTime), ", Price ", DoubleToString(lsPrice, _Digits)); //--- Log left shoulder details
      Print("Head: Bar ", headBar, ", Time ", TimeToString(extrema[headIdx].time), ", Price ", DoubleToString(extrema[headIdx].price, _Digits)); //--- Log head details
      Print("Right Shoulder: Bar ", rsBar, ", Time ", TimeToString(extrema[rightShoulderIdx].time), ", Price ", DoubleToString(extrema[rightShoulderIdx].price, _Digits)); //--- Log right shoulder details
      Print("Neckline Start: Bar ", necklineStartBar, ", Time ", TimeToString(necklineStartTime), ", Price ", DoubleToString(necklineStartPrice, _Digits)); //--- Log neckline start details
      Print("Neckline End: Bar ", necklineEndBar, ", Time ", TimeToString(necklineEndTime), ", Price ", DoubleToString(necklineEndPrice, _Digits)); //--- Log neckline end details
      Print("Close Price: ", DoubleToString(closePrice, _Digits));    //--- Log closing price at breakout
      Print("Breakout Time: ", TimeToString(breakoutTime));           //--- Log breakout timestamp
      Print("Neckline Price at Breakout: ", DoubleToString(breakoutNecklinePrice, _Digits)); //--- Log neckline price at breakout
      Print("Extended Neckline Start: Bar ", extendedBar, ", Time ", TimeToString(extendedNecklineStartTime), ", Price ", DoubleToString(extendedNecklineStartPrice, _Digits)); //--- Log extended neckline start details
      Print("Bar Ranges: LS to Head = ", lsToHead, ", Head to RS = ", headToRs, ", RS to Breakout = ", rsToBreakout, ", LS to Neckline Start = ", lsToNeckStart); //--- Log bar ranges for pattern analysis

      string prefix = "IHS_" + TimeToString(extrema[headIdx].time, TIME_MINUTES); //--- Create unique prefix for chart objects based on head time
      // Lines
      DrawTrendLine(prefix + "_LeftToNeckStart", lsTime, lsPrice, necklineStartTime, necklineStartPrice, clrGreen, 2, STYLE_SOLID); //--- Draw line from left shoulder to neckline start
      DrawTrendLine(prefix + "_NeckStartToHead", necklineStartTime, necklineStartPrice, extrema[headIdx].time, extrema[headIdx].price, clrGreen, 2, STYLE_SOLID); //--- Draw line from neckline start to head
      DrawTrendLine(prefix + "_HeadToNeckEnd", extrema[headIdx].time, extrema[headIdx].price, necklineEndTime, necklineEndPrice, clrGreen, 2, STYLE_SOLID); //--- Draw line from head to neckline end
      DrawTrendLine(prefix + "_NeckEndToRight", necklineEndTime, necklineEndPrice, extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, clrGreen, 2, STYLE_SOLID); //--- Draw line from neckline end to right shoulder
      DrawTrendLine(prefix + "_Neckline", extendedNecklineStartTime, extendedNecklineStartPrice, breakoutTime, breakoutNecklinePrice, clrBlue, 2, STYLE_SOLID); //--- Draw neckline from extended start to breakout
      DrawTrendLine(prefix + "_RightToBreakout", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, breakoutTime, breakoutNecklinePrice, clrGreen, 2, STYLE_SOLID); //--- Draw line from right shoulder to breakout
      DrawTrendLine(prefix + "_ExtendedToLeftShoulder", extendedNecklineStartTime, extendedNecklineStartPrice, lsTime, lsPrice, clrGreen, 2, STYLE_SOLID); //--- Draw line from extended neckline to left shoulder
      // Triangles
      DrawTriangle(prefix + "_LeftShoulderTriangle", lsTime, lsPrice, necklineStartTime, necklineStartPrice, extendedNecklineStartTime, extendedNecklineStartPrice, clrLightGreen); //--- Draw triangle for left shoulder area
      DrawTriangle(prefix + "_HeadTriangle", extrema[headIdx].time, extrema[headIdx].price, necklineStartTime, necklineStartPrice, necklineEndTime, necklineEndPrice, clrLightGreen); //--- Draw triangle for head area
      DrawTriangle(prefix + "_RightShoulderTriangle", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, necklineEndTime, necklineEndPrice, breakoutTime, breakoutNecklinePrice, clrLightGreen); //--- Draw triangle for right shoulder area
      // Text Labels
      DrawText(prefix + "_LS_Label", lsTime, lsPrice, "LS", clrGreen, false); //--- Draw "LS" label below left shoulder
      DrawText(prefix + "_Head_Label", extrema[headIdx].time, extrema[headIdx].price, "HEAD", clrGreen, false); //--- Draw "HEAD" label below head
      DrawText(prefix + "_RS_Label", extrema[rightShoulderIdx].time, extrema[rightShoulderIdx].price, "RS", clrGreen, false); //--- Draw "RS" label below right shoulder
      datetime necklineMidTime = extendedNecklineStartTime + (breakoutTime - extendedNecklineStartTime) / 2; //--- Calculate midpoint time of the neckline
      double necklineMidPrice = extendedNecklineStartPrice + slope * (iBarShift(_Symbol, _Period, extendedNecklineStartTime) - iBarShift(_Symbol, _Period, necklineMidTime)); //--- Calculate midpoint price of the neckline
      // Calculate angle in pixel space
      int x1 = ShiftToX(iBarShift(_Symbol, _Period, extendedNecklineStartTime)); //--- Convert extended neckline start to x-pixel coordinate
      int y1 = PriceToY(extendedNecklineStartPrice);                          //--- Convert extended neckline start price to y-pixel coordinate
      int x2 = ShiftToX(iBarShift(_Symbol, _Period, breakoutTime));           //--- Convert breakout time to x-pixel coordinate
      int y2 = PriceToY(breakoutNecklinePrice);                               //--- Convert breakout price to y-pixel coordinate
      double pixelSlope = (y2 - y1) / (double)(x2 - x1);                     //--- Calculate slope in pixel space (rise over run)
      double necklineAngle = -atan(pixelSlope) * 180 / M_PI;                  //--- Calculate neckline angle in degrees, negated for visual alignment
      Print("Pixel X1: ", x1, ", Y1: ", y1, ", X2: ", x2, ", Y2: ", y2, ", Pixel Slope: ", DoubleToString(pixelSlope, 4), ", Neckline Angle: ", DoubleToString(necklineAngle, 2)); //--- Log pixel coordinates and angle
      DrawText(prefix + "_Neckline_Label", necklineMidTime, necklineMidPrice, "NECKLINE", clrBlue, true, necklineAngle); //--- Draw "NECKLINE" label at midpoint with calculated angle

      double entryPrice = 0;                                                  //--- Set entry price to 0 for market order (uses current price)
      double sl = extrema[rightShoulderIdx].price - BufferPoints * _Point;    //--- Calculate stop-loss below right shoulder with buffer
      double patternHeight = necklinePrice - extrema[headIdx].price;          //--- Calculate pattern height from neckline to head
      double tp = closePrice + patternHeight;                                 //--- Calculate take-profit above close by pattern height
      if (sl < closePrice && tp > closePrice) {                               //--- Validate trade direction (SL below, TP above for buy)
         if (obj_Trade.Buy(LotSize, _Symbol, entryPrice, sl, tp, "Inverse Head and Shoulders")) { //--- Attempt to open a buy trade
            AddTradedPattern(lsTime, lsPrice);                                //--- Add pattern to traded list
            Print("Buy Trade Opened: SL ", DoubleToString(sl, _Digits), ", TP ", DoubleToString(tp, _Digits)); //--- Log successful trade opening
         }
      }
   }
}
```

What now remains is managing the opened positions by applying a trailing stop logic to maximize the profits. We create a function to handle the trailing logic as below.

```
//+------------------------------------------------------------------+
//| Apply trailing stop with minimum profit threshold                |
//+------------------------------------------------------------------+
void ApplyTrailingStop(int minTrailPoints, int trailingPoints, CTrade &trade_object, ulong magicNo = 0) { //--- Function to apply trailing stop to open positions
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);                           //--- Get current bid price
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);                           //--- Get current ask price

   for (int i = PositionsTotal() - 1; i >= 0; i--) {                             //--- Loop through all open positions from last to first
      ulong ticket = PositionGetTicket(i);                                       //--- Retrieve position ticket number
      if (ticket > 0 && PositionSelectByTicket(ticket)) {                        //--- Check if ticket is valid and select the position
         if (PositionGetString(POSITION_SYMBOL) == _Symbol &&                    //--- Verify position is for the current symbol
             (magicNo == 0 || PositionGetInteger(POSITION_MAGIC) == magicNo)) {  //--- Check if magic number matches or no magic filter applied
            double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);           //--- Get position opening price
            double currentSL = PositionGetDouble(POSITION_SL);                   //--- Get current stop-loss price
            double currentProfit = PositionGetDouble(POSITION_PROFIT) / (LotSize * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE)); //--- Calculate profit in points

            if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {         //--- Check if position is a Buy
               double profitPoints = (bid - openPrice) / _Point;                 //--- Calculate profit in points for Buy position
               if (profitPoints >= minTrailPoints + trailingPoints) {            //--- Check if profit exceeds minimum threshold for trailing
                  double newSL = NormalizeDouble(bid - trailingPoints * _Point, _Digits); //--- Calculate new stop-loss price
                  if (newSL > openPrice && (newSL > currentSL || currentSL == 0)) { //--- Ensure new SL is above open price and better than current SL
                     if (trade_object.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP))) { //--- Attempt to modify position with new SL
                        Print("Trailing Stop Updated: Ticket ", ticket, ", New SL: ", DoubleToString(newSL, _Digits)); //--- Log successful SL update
                     }
                  }
               }
            } else if (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_SELL) { //--- Check if position is a Sell
               double profitPoints = (openPrice - ask) / _Point;                 //--- Calculate profit in points for Sell position
               if (profitPoints >= minTrailPoints + trailingPoints) {            //--- Check if profit exceeds minimum threshold for trailing
                  double newSL = NormalizeDouble(ask + trailingPoints * _Point, _Digits); //--- Calculate new stop-loss price
                  if (newSL < openPrice && (newSL < currentSL || currentSL == 0)) { //--- Ensure new SL is below open price and better than current SL
                     if (trade_object.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP))) { //--- Attempt to modify position with new SL
                        Print("Trailing Stop Updated: Ticket ", ticket, ", New SL: ", DoubleToString(newSL, _Digits)); //--- Log successful SL update
                     }
                  }
               }
            }
         }
      }
   }
}
```

Here, we add a trailing stop feature with the "ApplyTrailingStop" function. It uses "minTrailPoints" and "trailingPoints" to adjust open positions. We fetch "bid" and "ask" prices with the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function. We loop through positions using the [PositionsTotal](https://www.mql5.com/en/docs/trading/positionstotal) function. For each, we get the "ticket" with the [PositionGetTicket](https://www.mql5.com/en/docs/trading/positiongetticket) function and select it with the [PositionSelectByTicket](https://www.mql5.com/en/docs/trading/positionselectbyticket) function. We verify the symbol and "magicNo" using the "PositionGetString" and "PositionGetInteger" functions. We retrieve "openPrice", "currentSL", and "currentProfit" with the [PositionGetDouble](https://www.mql5.com/en/docs/trading/positiongetdouble) function.

For a Buy, we calculate profit with "bid" and check against "minTrailPoints" plus "trailingPoints". If met, we set a new "newSL" with the [NormalizeDouble](https://www.mql5.com/en/docs/convert/normalizedouble) function and update it via the "PositionModify" method on the "trade\_object" object. For a Sell, we use "ask" instead and adjust "newSL" below. Successful price modification is logged. We then can call this function in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler.

```
// Apply trailing stop if enabled and positions exist
if (UseTrailingStop && PositionsTotal() > 0) {                        //--- Check if trailing stop is enabled and there are open positions
   ApplyTrailingStop(MinTrailPoints, TrailingPoints, obj_Trade, MagicNumber); //--- Apply trailing stop to positions with specified parameters
}
```

Calling the function with the input parameters is all we need to ensure trailing stop is enabled. What remains now is freeing the storage arrays once the program is no longer in use and deleting the visual objects we mapped. We handle that in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {                                        //--- Expert Advisor deinitialization function
   ArrayFree(tradedPatterns);                                            //--- Free memory used by tradedPatterns array
   ObjectsDeleteAll(0, "HS_");                                           //--- Delete all chart objects with "HS_" prefix (standard H&S)
   ObjectsDeleteAll(0, "IHS_");                                          //--- Delete all chart objects with "IHS_" prefix (inverse H&S)
   ChartRedraw();                                                        //--- Redraw the chart to remove deleted objects
}
```

In the OnDeinit event handler, which runs when the Expert Advisor shuts down, we clean up the program and the chart it is attached to. We use the [ArrayFree](https://www.mql5.com/en/docs/array/ArrayFree) function to release memory from "tradedPatterns". Then, we remove all chart objects. The [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function clears items with the "HS\_" prefix for standard patterns. It also clears those with the "IHS\_" prefix for inverse patterns. Finally, we refresh the chart. The [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function updates the display to reflect these changes before total closure. Upon compilation, we have the following outcome.

![FINAL OUTCOME WITH TRAILING STOP](https://c.mql5.com/2/127/Screenshot_2025-03-24_225735.png)

From the image, we can see that we apply trailing stop to the traded setup, hence achieving our objective. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/127/Screenshot_2025-03-24_231656.png)

Backtest report:

![REPORT](https://c.mql5.com/2/127/Screenshot_2025-03-24_231743.png)

### Conclusion

In conclusion, we have successfully built a [Head and Shoulders](https://www.mql5.com/go?link=https://www.investopedia.com/terms/h/head-shoulders.asp "https://www.investopedia.com/terms/h/head-shoulders.asp") trading algorithm in MQL5. It features precise pattern detection, detailed visualization, and automated trade execution for the classic reversal signal. By using validation rules, neckline plotting, and trailing stops, our Expert Advisor adapts to market shifts effectively. You can use the illustrations made as a steppingstone to enhance it with extra steps like parameter tuning or advanced risk controls. Also, note that it is a rare pattern setup.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market conditions can be unpredictable. Proper backtesting and risk management are essential before live deployment.

With this foundation, you can refine your trading skills and improve this algorithm. Keep testing and optimizing for success. Best of luck!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17618.zip "Download all attachments in the single ZIP archive")

[Head\_f\_Shoulders\_Pattern.mq5](https://www.mql5.com/en/articles/download/17618/head_f_shoulders_pattern.mq5 "Download Head_f_Shoulders_Pattern.mq5")(133.64 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/483990)**

![Simple solutions for handling indicators conveniently](https://c.mql5.com/2/93/Simple_solutions_for_convenient_work_with_indicators__LOGO.png)[Simple solutions for handling indicators conveniently](https://www.mql5.com/en/articles/14672)

In this article, I will describe how to make a simple panel to change the indicator settings directly from the chart, and what changes need to be made to the indicator to connect the panel. This article is intended for novice MQL5 users.

![Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://c.mql5.com/2/129/Day_Trading_Larry_Connors_RSI2_Mean-Reversion_Strategies___LOGO.png)[Day Trading Larry Connors RSI2 Mean-Reversion Strategies](https://www.mql5.com/en/articles/17636)

Larry Connors is a renowned trader and author, best known for his work in quantitative trading and strategies like the 2-period RSI (RSI2), which helps identify short-term overbought and oversold market conditions. In this article, we’ll first explain the motivation behind our research, then recreate three of Connors’ most famous strategies in MQL5 and apply them to intraday trading of the S&P 500 index CFD.

![Archery Algorithm (AA)](https://c.mql5.com/2/93/Archery_Algorithm____LOGO.png)[Archery Algorithm (AA)](https://www.mql5.com/en/articles/15782)

The article takes a detailed look at the archery-inspired optimization algorithm, with an emphasis on using the roulette method as a mechanism for selecting promising areas for "arrows". The method allows evaluating the quality of solutions and selecting the most promising positions for further study.

![MQL5 Wizard Techniques you should know (Part 58): Reinforcement Learning (DDPG) with Moving Average and Stochastic Oscillator Patterns](https://c.mql5.com/2/130/MQL5_Wizard_Techniques_you_should_know_Part_58__LOGO__2.png)[MQL5 Wizard Techniques you should know (Part 58): Reinforcement Learning (DDPG) with Moving Average and Stochastic Oscillator Patterns](https://www.mql5.com/en/articles/17668)

Moving Average and Stochastic Oscillator are very common indicators whose collective patterns we explored in the prior article, via a supervised learning network, to see which “patterns-would-stick”. We take our analyses from that article, a step further by considering the effects' reinforcement learning, when used with this trained network, would have on performance. Readers should note our testing is over a very limited time window. Nonetheless, we continue to harness the minimal coding requirements afforded by the MQL5 wizard in showcasing this.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=asrdncvkzbxjdamwaalvpeneyckbcjom&ssn=1769092018401246482&ssn_dr=0&ssn_sr=0&fv_date=1769092018&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17618&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2013)%3A%20Building%20a%20Head%20and%20Shoulders%20Trading%20Algorithm%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909201899836346&fz_uniq=5049141571884000690&sv=2552)

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