---
title: Automating Trading Strategies in MQL5 (Part 34): Trendline Breakout System with R-Squared Goodness of Fit
url: https://www.mql5.com/en/articles/19625
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:31:35.234177
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/19625&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068338215976368197)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 33)](https://www.mql5.com/en/articles/19479), we developed a [Shark Pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/shark-pattern/ "https://harmonictrader.com/harmonic-patterns/shark-pattern/") system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that detected bullish and bearish Shark harmonic patterns using Fibonacci ratios, automating trades with customizable take-profit and stop-loss levels, visualized through chart objects like triangles and trendlines. In Part 34, we create a [Trendline Breakout](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/spotting-breakouts "https://www.babypips.com/learn/forex/spotting-breakouts") System that identifies support and resistance trendlines using swing points, validated by [R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination "https://en.wikipedia.org/wiki/Coefficient_of_determination") goodness of fit and angle constraints, to execute trades on breakouts with dynamic chart visualizations. We will cover the following topics:

1. [Understanding the Trendline Breakout Strategy Framework](https://www.mql5.com/en/articles/19625#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19625#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19625#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19625#para4)

By the end, you’ll have a robust MQL5 strategy for trendline breakout trading, ready for customization—let’s dive in!

### Understanding the Trendline Breakout Strategy Framework

The [trendline breakout strategy](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/spotting-breakouts "https://www.babypips.com/learn/forex/spotting-breakouts") involves drawing diagonal lines on price charts to connect swing highs (resistance) or swing lows (support), identifying key price levels where the market is likely to reverse or continue. When the price breaks through these trendlines—either closing above a resistance line or below a support line—it signals a potential shift in market momentum, prompting traders to enter trades in the direction of the breakout with defined risk and reward parameters. This approach capitalizes on strong price movements following the break, aiming to capture significant trends while managing risk through stop-loss and take-profit levels. Here is an illustration of a downward trendline breakout.

![TRENDLINE BREAKOUT SAMPLE](https://c.mql5.com/2/170/Screenshot_2025-09-19_134050.png)

Our plan is to detect swing highs and lows within a specified lookback period, construct trendlines with a minimum number of touch points, and validate them using [R-squared metrics](https://en.wikipedia.org/wiki/Coefficient_of_determination "https://en.wikipedia.org/wiki/Coefficient_of_determination") and angle constraints to ensure reliability. In case you need to know, R-squared, also called the coefficient of determination, is a statistical measure that indicates how well a regression model explains the variability of the dependent variable using the independent variables. It represents the proportion of the total variation in the outcome that is accounted for by the model, with values ranging from 0 to 1. Here is a quick visualization of the model.

![R-SQUARED MODEL](https://c.mql5.com/2/170/Screenshot_2025-09-19_133927.png)

We will implement trade execution logic for breakouts, triggered by candle closes or entire candles crossing the trendline, with visual feedback through trendlines, arrows, and labels, and manage the trendline lifecycle by removing expired or broken ones, creating a breakout trading system. Have a look at the result we aim for, and then we can proceed to the implementation.

![TRENDLINE FRAMEWORK](https://c.mql5.com/2/170/Screenshot_2025-09-19_134449.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will start by declaring some [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [structures](https://www.mql5.com/en/docs/basis/types/classes) that will make the program more dynamic.

```
//+------------------------------------------------------------------+
//|                                 Trendline Breakout Trader EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Allan Munene Mutiiria."
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property strict

#include <Trade\Trade.mqh> //--- Include Trade library for trading operations
CTrade obj_Trade;          //--- Instantiate trade object
//+------------------------------------------------------------------+
//| Breakout definition enumeration                                  |
//+------------------------------------------------------------------+
enum ENUM_BREAKOUT_TYPE {
   BREAKOUT_CLOSE = 0,      // Breakout on close above/below line
   BREAKOUT_CANDLE = 1      // Breakout on entire candle above/below line
};
//+------------------------------------------------------------------+
//| Swing point structure                                            |
//+------------------------------------------------------------------+
struct Swing {     //--- Define swing point structure
   datetime time;  //--- Store swing time
   double price;   //--- Store swing price
};
//+------------------------------------------------------------------+
//| Starting point structure                                         |
//+------------------------------------------------------------------+
struct StartingPoint { //--- Define starting point structure
   datetime time;      //--- Store starting point time
   double price;       //--- Store starting point price
   bool is_support;    //--- Indicate support/resistance flag
};
//+------------------------------------------------------------------+
//| Trendline storage structure                                      |
//+------------------------------------------------------------------+
struct TrendlineInfo {      //--- Define trendline info structure
   string name;             //--- Store trendline name
   datetime start_time;     //--- Store start time
   datetime end_time;       //--- Store end time
   double start_price;      //--- Store start price
   double end_price;        //--- Store end price
   double slope;            //--- Store slope
   bool is_support;         //--- Indicate support/resistance flag
   int touch_count;         //--- Store number of touches
   datetime creation_time;  //--- Store creation time
   int touch_indices[];     //--- Store touch indices array
   bool is_signaled;        //--- Indicate signal flag
};
//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
void DetectSwings(); //--- Declare swing detection function
void SortSwings(Swing &swings[], int count); //--- Declare swing sorting function
double CalculateAngle(datetime time1, double price1, datetime time2, double price2); //--- Declare angle calculation function
bool ValidateTrendline(bool isSupport, datetime start_time, datetime ref_time, double ref_price, double slope, double tolerance_pen); //--- Declare trendline validation function
void FindAndDrawTrendlines(bool isSupport); //--- Declare trendline finding/drawing function
void UpdateTrendlines(); //--- Declare trendline update function
void RemoveTrendlineFromStorage(int index); //--- Declare trendline removal function
bool IsStartingPointUsed(datetime time, double price, bool is_support); //--- Declare starting point usage check function
double CalculateRSquared(const datetime &times[], const double &prices[], int n, double slope, double intercept); //--- Declare R-squared calculation function
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input ENUM_BREAKOUT_TYPE BreakoutType = BREAKOUT_CLOSE; // Breakout Definition
input int LookbackBars = 200;                           // Set bars for swing detection lookback
input double TouchTolerance = 10.0;                     // Set tolerance for touch points (points)
input int MinTouches = 3;                               // Set minimum touch points for valid trendline
input double PenetrationTolerance = 5.0;                // Set allowance for bar penetration (points)
input int ExtensionBars = 100;                          // Set bars to extend trendline right
input int MinBarSpacing = 10;                           // Set minimum bar spacing between touches
input double inpLot = 0.01;                             // Set lot size
input double inpSLPoints = 100.0;                       // Set stop loss (points)
input double inpRRRatio = 1.1;                          // Set risk:reward ratio
input double MinAngle = 1.0;                            // Set minimum inclination angle (degrees)
input double MaxAngle = 89.0;                           // Set maximum inclination angle (degrees)
input double MinRSquared = 0.8;                         // Minimum R-squared for trendline acceptance
input bool DeleteExpiredObjects = false;                // Enable deletion of expired/broken objects
input bool EnableTradingSignals = true;                 // Enable buy/sell signals and trades
input bool DrawTouchArrows = true;                      // Enable drawing arrows at touch points
input bool DrawLabels = true;                           // Enable drawing trendline/point labels
input color SupportLineColor = clrGreen;                // Set color for support trendlines
input color ResistanceLineColor = clrRed;               // Set color for resistance trendlines
//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
Swing swingLows[];              //--- Store swing lows
int numLows = 0;                //--- Track number of swing lows
Swing swingHighs[];             //--- Store swing highs
int numHighs = 0;               //--- Track number of swing highs
TrendlineInfo trendlines[];     //--- Store trendlines
int numTrendlines = 0;          //--- Track number of trendlines
StartingPoint startingPoints[]; //--- Store used starting points
int numStartingPoints = 0;      //--- Track number of starting points
```

We start the implementation of our trendline breakout system by setting up the foundational components for detecting and trading trendline breakouts. First, we include the "Trade.mqh" library and instantiate a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object named "obj\_Trade" for trade operations. Then, we define the "ENUM\_BREAKOUT\_TYPE" [enumeration](https://www.mql5.com/en/docs/basis/types/integer/enumeration) with options "BREAKOUT\_CLOSE" (breakout on candle close) and "BREAKOUT\_CANDLE" (breakout on entire candle), allowing flexible breakout detection. Next, we create the "Swing" [structure](https://www.mql5.com/en/docs/basis/types/classes) to store swing point time and price, the "StartingPoint" structure for tracking used trendline starting points with a support/resistance flag, and the "TrendlineInfo" structure to store trendline details like name, start/end times and prices, slope, touch count, creation time, touch indices, and signal status.

We declare forward functions like "DetectSwings", "SortSwings", and "CalculateAngle" for core logic. Then, we set input parameters: "BreakoutType" as "BREAKOUT\_CLOSE", "LookbackBars" at 200, and the rest, which are self-explanatory. Finally, we initialize global arrays "swingLows", "swingHighs", "trendlines", and "startingPoints" with counters "numLows", "numHighs", "numTrendlines", and "numStartingPoints" to manage swing points and trendlines, forming the backbone for detecting and validating trendlines for breakout trading. Since we are all set, we can initialize the storage arrays in the initialization.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   ArrayResize(trendlines, 0);     //--- Resize trendlines array
   numTrendlines = 0;              //--- Reset trendlines count
   ArrayResize(startingPoints, 0); //--- Resize starting points array
   numStartingPoints = 0;          //--- Reset starting points count
   return(INIT_SUCCEEDED);         //--- Return success
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ArrayResize(trendlines, 0);     //--- Resize trendlines array
   numTrendlines = 0;              //--- Reset trendlines count
   ArrayResize(startingPoints, 0); //--- Resize starting points array
   numStartingPoints = 0;          //--- Reset starting points count
}
```

To ensure proper setup and cleanup of resources, we implement the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) function by calling [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to set the "trendlines" array to zero, resetting "numTrendlines" to 0, resizing the "startingPoints" array to zero, and resetting "numStartingPoints" to 0, then returning [INIT\_SUCCEEDED](https://www.mql5.com/en/docs/basis/function/events#enum_init_retcode) to confirm successful initialization. Then, in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, we perform identical cleanup, ensuring no memory leaks when the program terminates. With initialization complete, we can now proceed to defining the strategy logic. To help modularize the logic, we will use functions, and the first logic we will define is swing points detection, so we can have base trendline points.

```
//+------------------------------------------------------------------+
//| Check for new bar                                                |
//+------------------------------------------------------------------+
bool IsNewBar() {
   static datetime lastTime = 0;                      //--- Store last bar time
   datetime currentTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   if (lastTime != currentTime) {                     //--- Check for new bar
      lastTime = currentTime;                         //--- Update last time
      return true;                                    //--- Indicate new bar
   }
   return false;                                      //--- Indicate no new bar
}
//+------------------------------------------------------------------+
//| Sort swings by time (ascending, oldest first)                    |
//+------------------------------------------------------------------+
void SortSwings(Swing &swings[], int count) {
   for (int i = 0; i < count - 1; i++) {            //--- Iterate through swings
      for (int j = 0; j < count - i - 1; j++) {     //--- Compare adjacent swings
         if (swings[j].time > swings[j + 1].time) { //--- Check time order
            Swing temp = swings[j];                 //--- Store temporary swing
            swings[j] = swings[j + 1];              //--- Swap swings
            swings[j + 1] = temp;                   //--- Complete swap
         }
      }
   }
}
//+------------------------------------------------------------------+
//| Detect swing highs and lows                                      |
//+------------------------------------------------------------------+
void DetectSwings() {
   numLows = 0;                                              //--- Reset lows count
   ArrayResize(swingLows, 0);                                //--- Resize lows array
   numHighs = 0;                                             //--- Reset highs count
   ArrayResize(swingHighs, 0);                               //--- Resize highs array
   int totalBars = iBars(_Symbol, _Period);                  //--- Get total bars
   int effectiveLookback = MathMin(LookbackBars, totalBars); //--- Calculate effective lookback
   if (effectiveLookback < 5) {                              //--- Check sufficient bars
      Print("Not enough bars for swing detection.");         //--- Log insufficient bars
      return;                                                //--- Exit function
   }
   for (int i = 2; i < effectiveLookback - 2; i++) {         //--- Iterate through bars
      double low_i = iLow(_Symbol, _Period, i);              //--- Get current low
      double low_im1 = iLow(_Symbol, _Period, i - 1);        //--- Get previous low
      double low_im2 = iLow(_Symbol, _Period, i - 2);        //--- Get two bars prior low
      double low_ip1 = iLow(_Symbol, _Period, i + 1);        //--- Get next low
      double low_ip2 = iLow(_Symbol, _Period, i + 2);        //--- Get two bars next low
      if (low_i < low_im1 && low_i < low_im2 && low_i < low_ip1 && low_i < low_ip2) { //--- Check for swing low
         Swing s;                                            //--- Create swing struct
         s.time = iTime(_Symbol, _Period, i);                //--- Set swing time
         s.price = low_i;                                    //--- Set swing price
         ArrayResize(swingLows, numLows + 1);                //--- Resize lows array
         swingLows[numLows] = s;                             //--- Add swing low
         numLows++;                                          //--- Increment lows count
      }
      double high_i = iHigh(_Symbol, _Period, i);       //--- Get current high
      double high_im1 = iHigh(_Symbol, _Period, i - 1); //--- Get previous high
      double high_im2 = iHigh(_Symbol, _Period, i - 2); //--- Get two bars prior high
      double high_ip1 = iHigh(_Symbol, _Period, i + 1); //--- Get next high
      double high_ip2 = iHigh(_Symbol, _Period, i + 2); //--- Get two bars next high
      if (high_i > high_im1 && high_i > high_im2 && high_i > high_ip1 && high_i > high_ip2) { //--- Check for swing high
         Swing s;                                       //--- Create swing struct
         s.time = iTime(_Symbol, _Period, i);           //--- Set swing time
         s.price = high_i;                              //--- Set swing price
         ArrayResize(swingHighs, numHighs + 1);         //--- Resize highs array
         swingHighs[numHighs] = s;                      //--- Add swing high
         numHighs++;                                    //--- Increment highs count
      }
   }
   if (numLows > 0) SortSwings(swingLows, numLows);     //--- Sort swing lows
   if (numHighs > 0) SortSwings(swingHighs, numHighs);  //--- Sort swing highs
}
```

With the foundational setup complete, we now implement the core logic for detecting swing points and managing bar updates. First, we develop the "IsNewBar" function, which uses a static "lastTime" variable to store the previous bar’s time, compares it with the current bar’s time obtained via [iTime](https://www.mql5.com/en/docs/series/itime) for the symbol and period at shift 0, updates "lastTime" if different, and returns true to indicate a new bar, or false otherwise. Then, we implement the "SortSwings" function, which sorts a "Swing" array by time in ascending order using a bubble sort algorithm, iterating through the array with nested loops, comparing adjacent elements’ "time" fields, and swapping them using a temporary "Swing" [struct](https://www.mql5.com/en/docs/basis/types/classes) if out of order.

Next, we create the "DetectSwings" function, resetting "numLows" and "numHighs" to 0 and resizing "swingLows" and "swingHighs" arrays to zero, calculating an effective lookback with [MathMin](https://www.mql5.com/en/docs/math/mathmin) of "LookbackBars" and total bars from [iBars](https://www.mql5.com/en/docs/series/ibars), and exiting with a [Print](https://www.mql5.com/en/docs/common/print) error if fewer than 5 bars are available. We then iterate through bars from index 2 to "effectiveLookback - 2", checking for swing lows by comparing the current bar’s low ("iLow") against two prior and two subsequent bars, and for swing highs similarly using [iHigh](https://www.mql5.com/en/docs/series/ihigh); if a swing is detected, we create a "Swing" struct, set its "time" with "iTime" and "price" with the low or high, append it to "swingLows" or "swingHighs" using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and increment the respective counter. Finally, we call "SortSwings" on "swingLows" and "swingHighs" if they contain elements, ensuring chronological order for trendline construction. Let us now define functions to calculate the trendline inclination for restriction based on inclination and its validation.

```
//+------------------------------------------------------------------+
//| Calculate visual inclination angle                               |
//+------------------------------------------------------------------+
double CalculateAngle(datetime time1, double price1, datetime time2, double price2) {
   int x1, y1, x2, y2; //--- Declare coordinate variables
   if (!ChartTimePriceToXY(0, 0, time1, price1, x1, y1)) return 0.0; //--- Convert time1/price1 to XY
   if (!ChartTimePriceToXY(0, 0, time2, price2, x2, y2)) return 0.0; //--- Convert time2/price2 to XY
   double dx = (double)(x2 - x1);                                    //--- Calculate x difference
   double dy = (double)(y2 - y1);                                    //--- Calculate y difference
   if (dx == 0.0) return (dy > 0.0 ? -90.0 : 90.0);                  //--- Handle vertical line case
   double angle = MathArctan(-dy / dx) * 180.0 / M_PI;               //--- Calculate angle in degrees
   return angle;                                                     //--- Return angle
}
//+------------------------------------------------------------------+
//| Validate trendline                                               |
//+------------------------------------------------------------------+
bool ValidateTrendline(bool isSupport, datetime start_time, datetime ref_time, double ref_price, double slope, double tolerance_pen) {
   int bar_start = iBarShift(_Symbol, _Period, start_time); //--- Get start bar index
   if (bar_start < 0) return false;                         //--- Check invalid bar index
   for (int bar = bar_start; bar >= 0; bar--) {             //--- Iterate through bars
      datetime bar_time = iTime(_Symbol, _Period, bar);     //--- Get bar time
      double dk = (double)(bar_time - ref_time);            //--- Calculate time difference
      double line_price = ref_price + slope * dk;           //--- Calculate line price
      if (isSupport) {                                      //--- Check support case
         double low = iLow(_Symbol, _Period, bar);          //--- Get bar low
         if (low < line_price - tolerance_pen) return false;//--- Check if broken
      } else {                                              //--- Handle resistance case
         double high = iHigh(_Symbol, _Period, bar);        //--- Get bar high
         if (high > line_price + tolerance_pen) return false;//--- Check if broken
      }
   }
   return true; //--- Return valid
}
```

Here, we implement functions to calculate trendline angles and validate their integrity. First, we develop the "CalculateAngle" function, which converts two time-price points ("time1", "price1" and "time2", "price2") to chart coordinates ("x1", "y1" and "x2", "y2") using [ChartTimePriceToXY](https://www.mql5.com/en/docs/chart_operations/chartxytotimeprice), returning 0.0 if either conversion fails; we calculate the differences "dx" and "dy", handle vertical lines by returning -90.0 or 90.0 if "dx" is zero based on "dy", and compute the angle in degrees using [MathArctan](https://www.mql5.com/en/docs/math/matharctan) of "-dy / dx" multiplied by 180/ [M\_PI](https://www.mql5.com/en/docs/constants/namedconstants/mathsconstants) for visual inclination.

Then, we implement the "ValidateTrendline" function, which validates a trendline by obtaining the bar index of "start\_time" with [iBarShift](https://www.mql5.com/en/docs/series/ibarshift), returning false if invalid; we iterate from this index to the present, calculating the trendline price at each bar’s time ("iTime") using the formula "ref\_price + slope \* (bar\_time - ref\_time)"; for support trendlines ("isSupport" true), we check if the bar’s low ( [iLow](https://www.mql5.com/en/docs/series/ilow)) falls below "line\_price - tolerance\_pen", returning false if broken; for resistance, we check if the bar’s high ( [iHigh](https://www.mql5.com/en/docs/series/ihigh)) exceeds "line\_price + tolerance\_pen", returning false if broken, otherwise returning true, ensuring trendlines meet angle constraints and remain unbreached for reliable breakout detection. We can now define the function for the R-squared goodness-of-fit model.

```
//+------------------------------------------------------------------+
//| Calculate R-squared for goodness of fit                          |
//+------------------------------------------------------------------+
double CalculateRSquared(const datetime &times[], const double &prices[], int n, double slope, double intercept) {
   double sum_y = 0.0;                             //--- Initialize sum of y
   for (int k = 0; k < n; k++) {                   //--- Iterate through points
      sum_y += prices[k];                          //--- Accumulate y
   }
   double mean_y = sum_y / n;                      //--- Calculate mean y
   double ss_tot = 0.0, ss_res = 0.0;              //--- Initialize sums of squares
   for (int k = 0; k < n; k++) {                   //--- Iterate through points
      double x = (double)times[k];                 //--- Get x (time)
      double y_pred = intercept + slope * x;       //--- Calculate predicted y
      double y = prices[k];                        //--- Get actual y
      ss_res += (y - y_pred) * (y - y_pred);       //--- Accumulate residual sum
      ss_tot += (y - mean_y) * (y - mean_y);       //--- Accumulate total sum
   }
   if (ss_tot == 0.0) return 1.0;                  //--- Handle constant y case
   return 1.0 - ss_res / ss_tot;                   //--- Calculate and return R-squared
}
```

We develop the "CalculateRSquared" function, which takes arrays of times and prices, the number of points "n", and the trendline’s "slope" and "intercept" as inputs; we initialize "sum\_y" to 0 and iterate through "prices" to compute the sum, then calculate the mean "mean\_y" by dividing "sum\_y" by "n". Then, we initialize "ss\_tot" and "ss\_res" for total and residual sums of squares, iterate again to compute predicted prices ("y\_pred") using the formula "intercept + slope \* time", accumulate residuals ("y - y\_pred" squared) in "ss\_res" and deviations from the mean ("y - mean\_y" squared) in "ss\_tot", and return 1.0 if "ss\_tot" is zero (constant prices) or calculate R-squared as "1.0 - ss\_res / ss\_tot". We just use the R-squared formula for the calculation of the trendlines' validity. Let us now define a function to manage the trendlines.

```
//+------------------------------------------------------------------+
//| Check if starting point is already used                          |
//+------------------------------------------------------------------+
bool IsStartingPointUsed(datetime time, double price, bool is_support) {
   for (int i = 0; i < numStartingPoints; i++) { //--- Iterate through starting points
      if (startingPoints[i].time == time && MathAbs(startingPoints[i].price - price) < TouchTolerance * _Point && startingPoints[i].is_support == is_support) { //--- Check match
         return true; //--- Return used
      }
   }
   return false; //--- Return not used
}
//+------------------------------------------------------------------+
//| Remove trendline from storage and optionally chart objects       |
//+------------------------------------------------------------------+
void RemoveTrendlineFromStorage(int index) {
   if (index < 0 || index >= numTrendlines) return;             //--- Check valid index
   Print("Removing trendline from storage: ", trendlines[index].name); //--- Log removal
   if (DeleteExpiredObjects) {                                  //--- Check deletion flag
      ObjectDelete(0, trendlines[index].name);                  //--- Delete trendline object
      for (int m = 0; m < trendlines[index].touch_count; m++) { //--- Iterate touches
         string arrow_name = trendlines[index].name + "_touch" + IntegerToString(m); //--- Generate arrow name
         ObjectDelete(0, arrow_name);                           //--- Delete touch arrow
         string text_name = trendlines[index].name + "_point_label" + IntegerToString(m); //--- Generate text name
         ObjectDelete(0, text_name);                            //--- Delete point label
      }
      string label_name = trendlines[index].name + "_label";    //--- Generate label name
      ObjectDelete(0, label_name);                              //--- Delete trendline label
      string signal_arrow = trendlines[index].name + "_signal_arrow"; //--- Generate signal arrow name
      ObjectDelete(0, signal_arrow);                            //--- Delete signal arrow
      string signal_text = trendlines[index].name + "_signal_text"; //--- Generate signal text name
      ObjectDelete(0, signal_text);                             //--- Delete signal text
   }
   for (int i = index; i < numTrendlines - 1; i++) {            //--- Shift array
      trendlines[i] = trendlines[i + 1];                        //--- Copy next trendline
   }
   ArrayResize(trendlines, numTrendlines - 1);                  //--- Resize trendlines array
   numTrendlines--;                                             //--- Decrement trendlines count
}
```

Here, we implement functions to manage trendline starting points and their cleanup. First, we develop the "IsStartingPointUsed" function, which iterates through the "startingPoints" array, checking if a given "time", "price", and "is\_support" match an existing starting point within "TouchTolerance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" using [MathAbs](https://www.mql5.com/en/docs/math/mathabs), returning true if found, or false otherwise. This will help to ensure no more than 1 trendline comes from one point.

Then, we create the "RemoveTrendlineFromStorage" function, which validates the input "index" against "numTrendlines", logs the removal of the trendline’s "name" with "Print", and, if "DeleteExpiredObjects" is true, deletes chart objects using [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) for the trendline ("name"), touch arrows ("name + '\_touch' + index"), point labels ("name + '\_point\_label' + index"), trendline label ("name + '\_label'"), signal arrow ("name + '\_signal\_arrow'"), and signal text ("name + '\_signal\_text'"). Next, we shift elements in the "trendlines" array left from "index" using a loop, resize the array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to reduce its size by one, and decrement "numTrendlines", ensuring unique trendline starting points and proper cleanup of invalid trendlines and their chart visuals. Let us now define a function to find and draw the trendlines using the helper functions we have defined.

```
//+------------------------------------------------------------------+
//| Find and draw trendlines if no active one exists                 |
//+------------------------------------------------------------------+
void FindAndDrawTrendlines(bool isSupport) {
   bool has_active = false;                        //--- Initialize active flag
   for (int i = 0; i < numTrendlines; i++) {       //--- Iterate through trendlines
      if (trendlines[i].is_support == isSupport) { //--- Check type match
         has_active = true;                        //--- Set active flag
         break;                                    //--- Exit loop
      }
   }
   if (has_active) return;                          //--- Exit if active trendline exists
   Swing swings[];                                  //--- Initialize swings array
   int numSwings;                                   //--- Initialize swings count
   color lineColor;                                 //--- Initialize line color
   string prefix;                                   //--- Initialize prefix
   if (isSupport) {                                 //--- Handle support case
      numSwings = numLows;                          //--- Set number of lows
      ArrayResize(swings, numSwings);               //--- Resize swings array
      for (int i = 0; i < numSwings; i++) {         //--- Iterate through lows
         swings[i].time = swingLows[i].time;        //--- Copy low time
         swings[i].price = swingLows[i].price;      //--- Copy low price
      }
      lineColor = SupportLineColor;                 //--- Set support line color
      prefix = "Trendline_Support_";                //--- Set support prefix
   } else {                                         //--- Handle resistance case
      numSwings = numHighs;                         //--- Set number of highs
      ArrayResize(swings, numSwings);               //--- Resize swings array
      for (int i = 0; i < numSwings; i++) {         //--- Iterate through highs
         swings[i].time = swingHighs[i].time;       //--- Copy high time
         swings[i].price = swingHighs[i].price;     //--- Copy high price
      }
      lineColor = ResistanceLineColor;              //--- Set resistance line color
      prefix = "Trendline_Resistance_";             //--- Set resistance prefix
   }
   if (numSwings < 2) return;                       //--- Exit if insufficient swings
   double pointValue = _Point;                      //--- Get point value
   double touch_tolerance = TouchTolerance * pointValue; //--- Calculate touch tolerance
   double pen_tolerance = PenetrationTolerance * pointValue; //--- Calculate penetration tolerance
   int best_j = -1;                                 //--- Initialize best j index
   int max_touches = 0;                             //--- Initialize max touches
   double best_rsquared = -1.0;                     //--- Initialize best R-squared
   int best_touch_indices[];                        //--- Initialize best touch indices
   double best_slope = 0.0;                         //--- Initialize best slope
   double best_intercept = 0.0;                     //--- Initialize best intercept
   datetime best_min_time = 0;                      //--- Initialize best min time
   for (int i = 0; i < numSwings - 1; i++) {        //--- Iterate through first points
      for (int j = i + 1; j < numSwings; j++) {     //--- Iterate through second points
         datetime time1 = swings[i].time;           //--- Get first time
         double price1 = swings[i].price;           //--- Get first price
         datetime time2 = swings[j].time;           //--- Get second time
         double price2 = swings[j].price;           //--- Get second price
         double dt = (double)(time2 - time1);       //--- Calculate time difference
         if (dt <= 0) continue;                     //--- Skip invalid time difference
         double initial_slope = (price2 - price1) / dt; //--- Calculate initial slope
         int touch_indices[];                       //--- Initialize touch indices
         ArrayResize(touch_indices, 0);             //--- Resize touch indices
         int touches = 0;                           //--- Initialize touches count
         ArrayResize(touch_indices, touches + 1);   //--- Add first index
         touch_indices[touches] = i;                //--- Set first index
         touches++;                                 //--- Increment touches
         ArrayResize(touch_indices, touches + 1);   //--- Add second index
         touch_indices[touches] = j;                //--- Set second index
         touches++;                                 //--- Increment touches
         for (int k = 0; k < numSwings; k++) {      //--- Iterate through swings
            if (k == i || k == j) continue;         //--- Skip used indices
            datetime tk = swings[k].time;           //--- Get swing time
            double dk = (double)(tk - time1);       //--- Calculate time difference
            double expected = price1 + initial_slope * dk; //--- Calculate expected price
            double actual = swings[k].price;        //--- Get actual price
            if (MathAbs(expected - actual) <= touch_tolerance) { //--- Check touch within tolerance
               ArrayResize(touch_indices, touches + 1); //--- Add index
               touch_indices[touches] = k;          //--- Set index
               touches++;                           //--- Increment touches
            }
         }
         if (touches >= MinTouches) {               //--- Check minimum touches
            ArraySort(touch_indices);               //--- Sort touch indices
            bool valid_spacing = true;              //--- Initialize spacing flag
            for (int m = 0; m < touches - 1; m++) { //--- Iterate through touches
               int idx1 = touch_indices[m];         //--- Get first index
               int idx2 = touch_indices[m + 1];     //--- Get second index
               int bar1 = iBarShift(_Symbol, _Period, swings[idx1].time); //--- Get first bar
               int bar2 = iBarShift(_Symbol, _Period, swings[idx2].time); //--- Get second bar
               int diff = MathAbs(bar1 - bar2);     //--- Calculate bar difference
               if (diff < MinBarSpacing) {          //--- Check minimum spacing
                  valid_spacing = false;            //--- Mark invalid spacing
                  break;                            //--- Exit loop
               }
            }
            if (valid_spacing) {                        //--- Check valid spacing
               datetime touch_times[];                  //--- Initialize touch times
               double touch_prices[];                   //--- Initialize touch prices
               ArrayResize(touch_times, touches);       //--- Resize times array
               ArrayResize(touch_prices, touches);      //--- Resize prices array
               for (int m = 0; m < touches; m++) {      //--- Iterate through touches
                  int idx = touch_indices[m];           //--- Get index
                  touch_times[m] = swings[idx].time;    //--- Set time
                  touch_prices[m] = swings[idx].price;  //--- Set price
               }
               double slope = initial_slope;            //--- Use initial slope from two points
               double intercept = price1 - slope * (double)time1; //--- Calculate intercept
               double rsquared = CalculateRSquared(touch_times, touch_prices, touches, slope, intercept); //--- Calculate R-squared
               if (rsquared >= MinRSquared) {           //--- Check minimum R-squared
                  int adjusted_touch_indices[];         //--- Initialize adjusted indices
                  ArrayResize(adjusted_touch_indices, touches); //--- Resize to current touches
                  ArrayCopy(adjusted_touch_indices, touch_indices); //--- Copy indices
                  int adjusted_touches = touches;       //--- Set adjusted touches
                  if (adjusted_touches >= MinTouches) { //--- Check minimum adjusted touches
                     datetime temp_min_time = swings[adjusted_touch_indices[0]].time; //--- Get min time
                     double temp_ref_price = intercept + slope * (double)temp_min_time; //--- Calculate ref price
                     if (ValidateTrendline(isSupport, temp_min_time, temp_min_time, temp_ref_price, slope, pen_tolerance)) { //--- Validate trendline
                        datetime temp_max_time = swings[adjusted_touch_indices[adjusted_touches - 1]].time; //--- Get max time
                        double temp_max_price = intercept + slope * (double)temp_max_time; //--- Calculate max price
                        double angle = CalculateAngle(temp_min_time, temp_ref_price, temp_max_time, temp_max_price); //--- Calculate angle
                        double abs_angle = MathAbs(angle); //--- Get absolute angle
                        if (abs_angle >= MinAngle && abs_angle <= MaxAngle) { //--- Check angle range
                           if (adjusted_touches > max_touches || (adjusted_touches == max_touches && rsquared > best_rsquared)) { //--- Check better trendline
                              max_touches = adjusted_touches; //--- Update max touches
                              best_rsquared = rsquared;       //--- Update best R-squared
                              best_j = j;                     //--- Update best j
                              best_slope = slope;             //--- Update best slope
                              best_intercept = intercept;     //--- Update best intercept
                              best_min_time = temp_min_time;  //--- Update best min time
                              ArrayResize(best_touch_indices, adjusted_touches); //--- Resize best indices
                              ArrayCopy(best_touch_indices, adjusted_touch_indices); //--- Copy indices
                           }
                        }
                     }
                  }
               }
            }
         }
      }
   }
   if (max_touches < MinTouches) {                        //--- Check insufficient touches
      string type = isSupport ? "Support" : "Resistance"; //--- Set type string
      return;                                             //--- Exit function
   }
   int touch_indices[];                                   //--- Initialize touch indices
   ArrayResize(touch_indices, max_touches);               //--- Resize touch indices
   ArrayCopy(touch_indices, best_touch_indices);          //--- Copy best indices
   int touches = max_touches;                             //--- Set touches count
   datetime min_time = best_min_time;                     //--- Set min time
   double price_min = best_intercept + best_slope * (double)min_time; //--- Calculate min price
   datetime max_time = swings[touch_indices[touches - 1]].time; //--- Set max time
   double price_max = best_intercept + best_slope * (double)max_time; //--- Calculate max price
   datetime start_time_check = min_time;                  //--- Set start time check
   double start_price_check = price_min;                  //--- Set start price check (approximate if not exact)
   if (IsStartingPointUsed(start_time_check, start_price_check, isSupport)) { //--- Check used starting point
      return; //--- Skip if used
   }
   datetime time_end = iTime(_Symbol, _Period, 0) + PeriodSeconds(_Period) * ExtensionBars; //--- Calculate end time
   double dk_end = (double)(time_end - min_time);         //--- Calculate end time difference
   double price_end = price_min + best_slope * dk_end;    //--- Calculate end price
   string unique_name = prefix + TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES|TIME_SECONDS); //--- Generate unique name
   if (ObjectFind(0, unique_name) < 0) {                  //--- Check if trendline exists
      ObjectCreate(0, unique_name, OBJ_TREND, 0, min_time, price_min, time_end, price_end); //--- Create trendline
      ObjectSetInteger(0, unique_name, OBJPROP_COLOR, lineColor); //--- Set color
      ObjectSetInteger(0, unique_name, OBJPROP_STYLE, STYLE_SOLID); //--- Set style
      ObjectSetInteger(0, unique_name, OBJPROP_WIDTH, 1);  //--- Set width
      ObjectSetInteger(0, unique_name, OBJPROP_RAY_RIGHT, false); //--- Disable right ray
      ObjectSetInteger(0, unique_name, OBJPROP_RAY_LEFT, false); //--- Disable left ray
      ObjectSetInteger(0, unique_name, OBJPROP_BACK, false); //--- Set to foreground
   }
   ArrayResize(trendlines, numTrendlines + 1);               //--- Resize trendlines array
   trendlines[numTrendlines].name = unique_name;             //--- Set trendline name
   trendlines[numTrendlines].start_time = min_time;          //--- Set start time
   trendlines[numTrendlines].end_time = time_end;            //--- Set end time
   trendlines[numTrendlines].start_price = price_min;        //--- Set start price
   trendlines[numTrendlines].end_price = price_end;          //--- Set end price
   trendlines[numTrendlines].slope = best_slope;             //--- Set slope
   trendlines[numTrendlines].is_support = isSupport;         //--- Set type
   trendlines[numTrendlines].touch_count = touches;          //--- Set touch count
   trendlines[numTrendlines].creation_time = TimeCurrent();  //--- Set creation time
   trendlines[numTrendlines].is_signaled = false;            //--- Set signaled flag
   ArrayResize(trendlines[numTrendlines].touch_indices, touches); //--- Resize touch indices
   ArrayCopy(trendlines[numTrendlines].touch_indices, touch_indices); //--- Copy touch indices
   numTrendlines++;                                          //--- Increment trendlines count
   ArrayResize(startingPoints, numStartingPoints + 1);       //--- Resize starting points array
   startingPoints[numStartingPoints].time = start_time_check;//--- Set starting point time
   startingPoints[numStartingPoints].price = start_price_check; //--- Set starting point price
   startingPoints[numStartingPoints].is_support = isSupport; //--- Set starting point type
   numStartingPoints++;                                      //--- Increment starting points count
   if (DrawTouchArrows) {                                    //--- Check draw arrows
      for (int m = 0; m < touches; m++) {                    //--- Iterate through touches
         int idx = touch_indices[m];                         //--- Get touch index
         datetime tk_time = swings[idx].time;                //--- Get touch time
         double tk_price = swings[idx].price;                //--- Get touch price
         string arrow_name = unique_name + "_touch" + IntegerToString(m); //--- Generate arrow name
         if (ObjectFind(0, arrow_name) < 0) { //--- Check if arrow exists
            ObjectCreate(0, arrow_name, OBJ_ARROW, 0, tk_time, tk_price); //--- Create touch arrow
            ObjectSetInteger(0, arrow_name, OBJPROP_ARROWCODE, 159); //--- Set arrow code
            ObjectSetInteger(0, arrow_name, OBJPROP_ANCHOR, isSupport ? ANCHOR_TOP : ANCHOR_BOTTOM); //--- Set anchor
            ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, lineColor); //--- Set color
            ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 1); //--- Set width
            ObjectSetInteger(0, arrow_name, OBJPROP_BACK, false); //--- Set to foreground
         }
      }
   }
   double angle = CalculateAngle(min_time, price_min, max_time, price_max); //--- Calculate angle
   string type = isSupport ? "Support" : "Resistance"; //--- Set type string
   Print(type + " Trendline " + unique_name + " drawn with " + IntegerToString(touches) + " touches. Inclination angle: " + DoubleToString(angle, 2) + " degrees."); //--- Log trendline
   if (DrawLabels) {                                            //--- Check draw labels
      datetime mid_time = min_time + (max_time - min_time) / 2; //--- Calculate mid time
      double dk_mid = (double)(mid_time - min_time);            //--- Calculate mid time difference
      double mid_price = price_min + best_slope * dk_mid;       //--- Calculate mid price
      double label_offset = 20 * _Point * (isSupport ? -1 : 1); //--- Calculate label offset
      double label_price = mid_price + label_offset;            //--- Calculate label price
      int label_anchor = isSupport ? ANCHOR_TOP : ANCHOR_BOTTOM;//--- Set label anchor
      string label_text = type + " Trendline";                  //--- Set label text
      string label_name = unique_name + "_label";               //--- Generate label name
      if (ObjectFind(0, label_name) < 0) {                      //--- Check if label exists
         ObjectCreate(0, label_name, OBJ_TEXT, 0, mid_time, label_price); //--- Create label
         ObjectSetString(0, label_name, OBJPROP_TEXT, label_text); //--- Set text
         ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrBlack); //--- Set color
         ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 8);  //--- Set font size
         ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, label_anchor); //--- Set anchor
         ObjectSetDouble(0, label_name, OBJPROP_ANGLE, angle);  //--- Set angle
         ObjectSetInteger(0, label_name, OBJPROP_BACK, false);  //--- Set to foreground
      }
      color point_label_color = isSupport ? clrSaddleBrown : clrDarkGoldenrod; //--- Set point label color
      double point_text_offset = 20.0 * _Point;    //--- Set point text offset
      for (int m = 0; m < touches; m++) {          //--- Iterate through touches
         int idx = touch_indices[m];               //--- Get touch index
         datetime tk_time = swings[idx].time;      //--- Get touch time
         double tk_price = swings[idx].price;      //--- Get touch price
         double text_price;                        //--- Initialize text price
         int point_text_anchor;                    //--- Initialize text anchor
         if (isSupport) {                          //--- Handle support
            text_price = tk_price - point_text_offset; //--- Set text price below
            point_text_anchor = ANCHOR_LEFT;       //--- Set left anchor
         } else {                                  //--- Handle resistance
            text_price = tk_price + point_text_offset; //--- Set text price above
            point_text_anchor = ANCHOR_BOTTOM;     //--- Set bottom anchor
         }
         string text_name = unique_name + "_point_label" + IntegerToString(m); //--- Generate text name
         string point_text = "Pt " + IntegerToString(m + 1); //--- Set point text
         if (ObjectFind(0, text_name) < 0) { //--- Check if text exists
            ObjectCreate(0, text_name, OBJ_TEXT, 0, tk_time, text_price); //--- Create text
            ObjectSetString(0, text_name, OBJPROP_TEXT, point_text); //--- Set text
            ObjectSetInteger(0, text_name, OBJPROP_COLOR, point_label_color); //--- Set color
            ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, 8); //--- Set font size
            ObjectSetInteger(0, text_name, OBJPROP_ANCHOR, point_text_anchor); //--- Set anchor
            ObjectSetDouble(0, text_name, OBJPROP_ANGLE, 0); //--- Set angle
            ObjectSetInteger(0, text_name, OBJPROP_BACK, false); //--- Set to foreground
         }
      }
   }
}
```

Here, we implement the trendline detection and visualization logic. First, in the "FindAndDrawTrendlines" function, we check for existing trendlines of type "isSupport" in "trendlines", setting "has\_active" to true and exiting if found. Then, we initialize a "swings" array, copying "swingLows" or "swingHighs" based on "isSupport", setting "lineColor" to "SupportLineColor" or "ResistanceLineColor" and "prefix" to "Trendline\_Support\_" or "Trendline\_Resistance\_", and exit if fewer than two swings exist.

Next, we calculate tolerances ("TouchTolerance" and "PenetrationTolerance" scaled by [\_Point](https://www.mql5.com/en/docs/predefined/_point)) and iterate through pairs of swing points to compute "initial\_slope", collecting touch points within "touch\_tolerance" into "touch\_indices". We validate touches with "MinTouches" and "MinBarSpacing" using [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) and [ArraySort](https://www.mql5.com/en/docs/array/arraysort), compute "slope" and "intercept", and evaluate "CalculateRSquared" and "ValidateTrendline" to select the best trendline based on "max\_touches" and "best\_rsquared". If valid, we draw the trendline using "ObjectCreate" ( [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_trend)) with "unique\_name", set properties like [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer), "OBJPROP\_STYLE", and disable rays, then store it in "trendlines" with details like "start\_time", "end\_time" (extended by "ExtensionBars"), and "touch\_indices". We update "startingPoints" with "IsStartingPointUsed" to prevent duplicates, and if "DrawTouchArrows" is true, draw arrows ( [OBJ\_ARROW](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_arrow)) at touch points with "lineColor" and appropriate anchors.

If "DrawLabels" is true, we add a trendline label ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) with "type + ' Trendline'" at the midpoint, angled via "CalculateAngle", and point labels ("Pt 1", etc.) with colors " [clrSaddleBrown](https://www.mql5.com/en/docs/constants/objectconstants/webcolors)" or "clrDarkGoldenrod", logging the trendline details. What now remains is the management of the existing trendlines via continuous updates and checking for crosses for signals. We will incorporate all the logic in a single function for simplicity.

```
//+------------------------------------------------------------------+
//| Update trendlines and check for signals                          |
//+------------------------------------------------------------------+
void UpdateTrendlines() {
   datetime current_time = iTime(_Symbol, _Period, 0);     //--- Get current time
   double pointValue = _Point;                             //--- Get point value
   double pen_tolerance = PenetrationTolerance * pointValue; //--- Calculate penetration tolerance
   double touch_tolerance = TouchTolerance * pointValue;   //--- Calculate touch tolerance
   for (int i = numTrendlines - 1; i >= 0; i--) {          //--- Iterate trendlines backward
      string type = trendlines[i].is_support ? "Support" : "Resistance"; //--- Determine trendline type
      string name = trendlines[i].name;                    //--- Get trendline name
      if (current_time > trendlines[i].end_time) {         //--- Check if expired
         PrintFormat("%s trendline %s is no longer valid (expired). End time: %s, Current time: %s.", type, name, TimeToString(trendlines[i].end_time), TimeToString(current_time)); //--- Log expiration
         RemoveTrendlineFromStorage(i);                    //--- Remove trendline
         continue;                                         //--- Skip to next
      }
      datetime prev_bar_time = iTime(_Symbol, _Period, 1); //--- Get previous bar time
      double dk = (double)(prev_bar_time - trendlines[i].start_time); //--- Calculate time difference
      double line_price = trendlines[i].start_price + trendlines[i].slope * dk; //--- Calculate line price
      double prev_close = iClose(_Symbol, _Period, 1);     //--- Get previous bar close
      double prev_low = iLow(_Symbol, _Period, 1);         //--- Get previous bar low
      double prev_high = iHigh(_Symbol, _Period, 1);       //--- Get previous bar high
      bool broken = false;                                 //--- Initialize broken flag
      if (BreakoutType == BREAKOUT_CLOSE) {                //--- Check breakout on close
         if (trendlines[i].is_support && prev_close < line_price) { //--- Support break by close
            PrintFormat("%s trendline %s is no longer valid (broken by close). Line price: %.5f, Prev close: %.5f.", type, name, line_price, prev_close); //--- Log break
            broken = true;                                 //--- Set broken flag
         } else if (!trendlines[i].is_support && prev_close > line_price) { //--- Resistance break by close
            PrintFormat("%s trendline %s is no longer valid (broken by close). Line price: %.5f, Prev close: %.5f.", type, name, line_price, prev_close); //--- Log break
            broken = true;                                 //--- Set broken flag
         }
      } else if (BreakoutType == BREAKOUT_CANDLE) {        //--- Check breakout on entire candle
         if (trendlines[i].is_support && prev_high < line_price) { //--- Entire candle below support
            PrintFormat("%s trendline %s is no longer valid (entire candle below). Line price: %.5f, Prev high: %.5f.", type, name, line_price, prev_high); //--- Log break
            broken = true;                                 //--- Set broken flag
         } else if (!trendlines[i].is_support && prev_low > line_price) { //--- Entire candle above resistance
            PrintFormat("%s trendline %s is no longer valid (entire candle above). Line price: %.5f, Prev low: %.5f.", type, name, line_price, prev_low); //--- Log break
            broken = true;                                  //--- Set broken flag
         }
      }
      if (broken && EnableTradingSignals && !trendlines[i].is_signaled) { //--- Check for breakout signal
         bool signaled = false;                           //--- Initialize signaled flag
         string signal_type = "";                         //--- Initialize signal type
         color signal_color = clrNONE;                    //--- Initialize signal color
         int arrow_code = 0;                              //--- Initialize arrow code
         int anchor = 0;                                  //--- Initialize anchor
         double text_angle = 0.0;                         //--- Initialize text angle
         double text_offset = 0.0;                        //--- Initialize text offset
         double text_price = 0.0;                         //--- Initialize text price
         int text_anchor = 0;                             //--- Initialize text anchor
         if (trendlines[i].is_support) {                  //--- Support break: SELL
            signaled = true;                              //--- Set signaled flag
            signal_type = "SELL BREAK";                   //--- Set sell break signal
            signal_color = clrRed;                        //--- Set red color
            arrow_code = 218;                             //--- Set down arrow
            anchor = ANCHOR_BOTTOM;                       //--- Set bottom anchor
            text_angle = 90.0;                            //--- Set vertical downward
            text_offset = 20 * pointValue;                //--- Set text offset
            text_price = line_price + text_offset;        //--- Calculate text price
            text_anchor = ANCHOR_BOTTOM;                  //--- Set bottom anchor
            double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Get bid price
            double SL = NormalizeDouble(line_price + inpSLPoints * _Point, _Digits); //--- SL above the line
            double risk = SL - Bid;                       //--- Calculate risk
            double TP = NormalizeDouble(Bid - risk * inpRRRatio, _Digits); //--- Calculate take profit
            obj_Trade.Sell(inpLot, _Symbol, Bid, SL, TP); //--- Execute sell trade
         } else {                                         //--- Resistance break: BUY
            signaled = true;                              //--- Set signaled flag
            signal_type = "BUY BREAK";                    //--- Set buy break signal
            signal_color = clrBlue;                       //--- Set blue color
            arrow_code = 217;                             //--- Set up arrow
            anchor = ANCHOR_TOP;                          //--- Set top anchor
            text_angle = -90.0;                           //--- Set vertical upward
            text_offset = -20 * pointValue;               //--- Set text offset
            text_price = line_price + text_offset;        //--- Calculate text price
            text_anchor = ANCHOR_LEFT;                    //--- Set left anchor
            double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Get ask price
            double SL = NormalizeDouble(line_price - inpSLPoints * _Point, _Digits); //--- SL below the line
            double risk = Ask - SL;                       //--- Calculate risk
            double TP = NormalizeDouble(Ask + risk * inpRRRatio, _Digits); //--- Calculate take profit
            obj_Trade.Buy(inpLot, _Symbol, Ask, SL, TP);  //--- Execute buy trade
         }
         if (signaled) { //--- Check if signaled
            PrintFormat("Breakout signal generated for %s trendline %s: %s at price %.5f, time %s.", type, name, signal_type, line_price, TimeToString(current_time)); //--- Log signal
            string arrow_name = name + "_signal_arrow"; //--- Generate signal arrow name
            if (ObjectFind(0, arrow_name) < 0) { //--- Check if arrow exists
               ObjectCreate(0, arrow_name, OBJ_ARROW, 0, prev_bar_time, line_price); //--- Create signal arrow
               ObjectSetInteger(0, arrow_name, OBJPROP_ARROWCODE, arrow_code); //--- Set arrow code
               ObjectSetInteger(0, arrow_name, OBJPROP_ANCHOR, anchor); //--- Set anchor
               ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, signal_color); //--- Set color
               ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 1); //--- Set width
               ObjectSetInteger(0, arrow_name, OBJPROP_BACK, false); //--- Set to foreground
            }
            string text_name = name + "_signal_text"; //--- Generate signal text name
            if (ObjectFind(0, text_name) < 0) { //--- Check if text exists
               ObjectCreate(0, text_name, OBJ_TEXT, 0, prev_bar_time, text_price); //--- Create signal text
               ObjectSetString(0, text_name, OBJPROP_TEXT, " " + signal_type); //--- Set text content
               ObjectSetInteger(0, text_name, OBJPROP_COLOR, signal_color); //--- Set color
               ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, 10); //--- Set font size
               ObjectSetInteger(0, text_name, OBJPROP_ANCHOR, text_anchor); //--- Set anchor
               ObjectSetDouble(0, text_name, OBJPROP_ANGLE, text_angle); //--- Set angle
               ObjectSetInteger(0, text_name, OBJPROP_BACK, false); //--- Set to foreground
            }
            trendlines[i].is_signaled = true; //--- Set signaled flag
         }
      }
      if (broken) {                           //--- Remove if broken
         RemoveTrendlineFromStorage(i);       //--- Remove trendline
      }
   }
}
```

To implement the trendline update and breakout trading logic, in the "UpdateTrendlines" function, we retrieve the current bar’s time with [iTime](https://www.mql5.com/en/docs/series/itime) and calculate "pointValue", "pen\_tolerance" ("PenetrationTolerance \* pointValue"), and "touch\_tolerance" ("TouchTolerance \* pointValue"). Then, we iterate backward through "trendlines", determining the "type" (Support or Resistance) and "name", and check if the trendline has expired using "current\_time > end\_time", logging with [PrintFormat](https://www.mql5.com/en/docs/common/printformat) and removing it with "RemoveTrendlineFromStorage" if expired.

Next, we calculate the trendline’s price at the previous bar ("prev\_bar\_time" from "iTime") using "start\_price + slope \* (prev\_bar\_time - start\_time)", and check for breakouts: for "BreakoutType" as "BREAKOUT\_CLOSE", we verify if the support trendline’s "prev\_close" ( [iClose](https://www.mql5.com/en/docs/series/iclose)) is below "line\_price" or resistance’s is above, logging with "PrintFormat" and setting "broken" to true; for "BREAKOUT\_CANDLE", we check if the support’s "prev\_high" ("iHigh") is below or resistance’s "prev\_low" ( [iLow](https://www.mql5.com/en/docs/series/ilow)) is above "line\_price", logging and setting it as broken.

If broken and "EnableTradingSignals" is true with "is\_signaled" false, we set trade parameters: for support (sell), we use "signal\_type" as "SELL BREAK", red color, down arrow (218), and calculate bid ( [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble)), stop loss ("line\_price + inpSLPoints \* \_Point"), risk, and take profit using "inpRRRatio", executing with "obj\_Trade.Sell"; for resistance (buy), we use "BUY BREAK", blue color, up arrow (217), and calculate ask, stop loss, and take profit, executing with "obj\_Trade.Buy". We then draw a signal arrow ("OBJ\_ARROW") and text ("OBJ\_TEXT") with "ObjectCreate", setting properties like "OBJPROP\_ARROWCODE", "OBJPROP\_ANCHOR", and "OBJPROP\_COLOR", log the signal with "PrintFormat", set "is\_signaled" to true, and remove broken trendlines from storage. The choice of the arrow codes to use is dependent on you. Here is a list of codes you could use from the [MQL5-defined Wingdings](https://www.mql5.com/en/docs/constants/objectconstants/wingdings) codes.

![MQL5 WINGDINGS](https://c.mql5.com/2/170/C_MQL5_WINGDINGS.png)

We can now call these functions in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler for the system to give tick-based feedback.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   if (!IsNewBar()) return;      //--- Exit if not new bar
   DetectSwings();               //--- Detect swings
   UpdateTrendlines();           //--- Update trendlines
   FindAndDrawTrendlines(true);  //--- Find/draw support trendlines
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function, we first call "IsNewBar" to check for a new bar, exiting if none to optimize performance. If a new bar is detected, we invoke "DetectSwings" to identify swing highs and lows, followed by "UpdateTrendlines" to check for breakouts or expired trendlines and execute trades if applicable. Then, we call "FindAndDrawTrendlines" with "true" to detect and draw support trendlines, ensuring only valid trendlines are visualized. Upon compilation, we get the following outcome.

![CONFIRMED SUPPORT BREAKOUT SIGNAL](https://c.mql5.com/2/170/Screenshot_2025-09-19_150855.png)

From the image, we can see that we find, analyze, draw, and trade the trendline upon breakout. Expired lines are also removed from the storage array successfully. We can achieve the same thing for resistance trendlines as well by calling the same function as support, but having the input parameter as false.

```
//--- other ontick functions

FindAndDrawTrendlines(false);                   //--- Find/draw resistance trendlines

//---
```

Upon passing the function and compilation, we get the following outcome.

![CONFIRMED RESISTANCE BREAKOUT SIGNAL](https://c.mql5.com/2/170/Screenshot_2025-09-19_151500.png)

From the image, we can see that we detect and trade the resistance trendlines as well. When we test and combine everything, we get the following outcome.

![COMBINED OUTCOME](https://c.mql5.com/2/170/Screenshot_2025-09-19_151801.png)

From the image, we can see that we detect the trendlines, visualize them, and act upon them when the price breaks them, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/170/Screenshot_2025-09-19_153301.png)

Backtest report:

![REPORT](https://c.mql5.com/2/170/Screenshot_2025-09-19_153334.png)

### Conclusion

In conclusion, we’ve developed a [trendline breakout](https://www.mql5.com/go?link=https://www.babypips.com/learn/forex/spotting-breakouts "https://www.babypips.com/learn/forex/spotting-breakouts") system in MQL5 utilizing swing points to identify and validate support and resistance trendlines with R-squared goodness of fit, executing breakout trades with customizable risk parameters. The system enhances trading decisions with dynamic visualizations, including trendlines, touch point arrows, and labels, ensuring clear market analysis.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By implementing this trendline breakout strategy, you’re equipped to capture market movements, ready for further customization in your trading journey. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19625.zip "Download all attachments in the single ZIP archive")

[a.\_Trendline\_Breakout\_Trader\_EA.mq5](https://www.mql5.com/en/articles/download/19625/a._Trendline_Breakout_Trader_EA.mq5 "Download a._Trendline_Breakout_Trader_EA.mq5")(47.5 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/496297)**
(2)


![linfo2](https://c.mql5.com/avatar/2023/4/6438c14d-e2f0.png)

**[linfo2](https://www.mql5.com/en/users/neilhazelwood)**
\|
1 Oct 2025 at 16:27

Great thank you for sharing I really appreciate you sharing ( all of your codes) , Robust and well marked code , Excellent template to build from !. Something I have tried to create myself but definitely not as well put together as this


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
1 Oct 2025 at 22:05

**linfo2 [#](https://www.mql5.com/en/forum/496297#comment_58166587):**

Great thank you for sharing I really appreciate you sharing ( all of your codes) , Robust and well marked code , Excellent template to build from !. Something I have tried to create myself but definitely not as well put together as this

Thanks you find it helpful. Most welcome.


![Cyclic Parthenogenesis Algorithm (CPA)](https://c.mql5.com/2/113/Cyclic_Parthenogenesis_Algorithm____LOGO.png)[Cyclic Parthenogenesis Algorithm (CPA)](https://www.mql5.com/en/articles/16877)

The article considers a new population optimization algorithm - Cyclic Parthenogenesis Algorithm (CPA), inspired by the unique reproductive strategy of aphids. The algorithm combines two reproduction mechanisms — parthenogenesis and sexual reproduction — and also utilizes the colonial structure of the population with the possibility of migration between colonies. The key features of the algorithm are adaptive switching between different reproductive strategies and a system of information exchange between colonies through the flight mechanism.

![Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://c.mql5.com/2/171/19626-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 5): Screening](https://www.mql5.com/en/articles/19626)

This article proposes an asset screening process for a statistical arbitrage trading strategy through cointegrated stocks. The system starts with the regular filtering by economic factors, like asset sector and industry, and finishes with a list of criteria for a scoring system. For each statistical test used in the screening, a respective Python class was developed: Pearson correlation, Engle-Granger cointegration, Johansen cointegration, and ADF/KPSS stationarity. These Python classes are provided along with a personal note from the author about the use of AI assistants for software development.

![Price Action Analysis Toolkit Development (Part 42): Interactive Chart Testing with Button Logic and Statistical Levels](https://c.mql5.com/2/172/19697-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 42): Interactive Chart Testing with Button Logic and Statistical Levels](https://www.mql5.com/en/articles/19697)

In a world where speed and precision matter, analysis tools need to be as smart as the markets we trade. This article presents an EA built on button logic—an interactive system that instantly transforms raw price data into meaningful statistical levels. With a single click, it calculates and displays mean, deviation, percentiles, and more, turning advanced analytics into clear on-chart signals. It highlights the zones where price is most likely to bounce, retrace, or break, making analysis both faster and more practical.

![MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://c.mql5.com/2/172/19627-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://www.mql5.com/en/articles/19627)

This article follows up ‘Part-74’, where we examined the pairing of Ichimoku and the ADX under a Supervised Learning framework, by moving our focus to Reinforcement Learning. Ichimoku and ADX form a complementary combination of support/resistance mapping and trend strength spotting. In this installment, we indulge in how the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm can be used with this indicator set. As with earlier parts of the series, the implementation is carried out in a custom signal class designed for integration with the MQL5 Wizard, which facilitates seamless Expert Advisor assembly.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dljgdiimfvvpbrrseiybnuzsqhprgdvz&ssn=1769178693041422243&ssn_dr=0&ssn_sr=0&fv_date=1769178693&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19625&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2034)%3A%20Trendline%20Breakout%20System%20with%20R-Squared%20Goodness%20of%20Fit%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917869346222473&fz_uniq=5068338215976368197&sv=2552)

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