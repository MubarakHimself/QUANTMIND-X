---
title: Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation
url: https://www.mql5.com/en/articles/19077
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 5
scraped_at: 2026-01-23T17:31:45.158184
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fqblycfjsundtcfudieistlicsgytnsn&ssn=1769178703195111693&ssn_dr=0&ssn_sr=0&fv_date=1769178703&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19077&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2025)%3A%20Trendline%20Trader%20with%20Least%20Squares%20Fit%20and%20Dynamic%20Signal%20Generation%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917870352280976&fz_uniq=5068341347007526991&sv=2552)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 24)](https://www.mql5.com/en/articles/18867), we developed a [London Session Breakout](https://www.mql5.com/go?link=https://tradingstrategyguides.com/london-breakout-strategy/ "https://tradingstrategyguides.com/london-breakout-strategy/") System in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that utilized pre-London ranges to place pending orders with risk management and trailing stops, enabling effective session-based trading. In Part 25, we create a trendline trading program that employs a least squares fit to detect support and resistance trendlines, generating automated buy and sell signals when prices touch these lines, enhanced by visual indicators such as arrows and customizable trade parameters. We will cover the following topics:

1. [Designing the Trendline Trading Framework](https://www.mql5.com/en/articles/19077#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19077#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19077#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19077#para4)

By the end, you’ll have a powerful MQL5 strategy for trend-based trading, ready for customization—let’s dive in!

### Designing the Trendline Trading Framework

The [trendline trading](https://www.mql5.com/go?link=https://howtotrade.com/trading-strategies/trendline-strategy/ "https://howtotrade.com/trading-strategies/trendline-strategy/") is a strategy that uses diagonal lines drawn on price charts to connect swing highs (resistance) or swing lows (support), helping traders identify the prevailing trend. Traders buy near upward-sloping trendlines (support) in an uptrend or sell near downward-sloping trendlines (resistance) in a downtrend, expecting the price to bounce. A break of the trendline often signals a potential reversal or trend weakening, prompting traders to exit or reverse their positions. Here is an illustration of a downtrend trendline.

![DOWNWARD TRENDLINE](https://c.mql5.com/2/161/Screenshot_2025-08-06_001304.png)

We will now be developing a trendline trader program to automate trading by detecting support and resistance trendlines using a least squares fit method, enabling precise buy and sell signals when prices touch these lines.

In case you need to know, the [least squares fit method](https://en.wikipedia.org/wiki/Least_squares "https://en.wikipedia.org/wiki/Least_squares") is a statistical technique used to determine a line (or curve) that best fits a set of data points by minimizing the sum of the squares of the vertical deviations (errors) between the data points and the fitted line. It will be important to us because it will provide the most accurate linear approximation of the relationship between swing points, which will be essential for prediction, trend analysis, and data modeling in our discipline. Have a look below at the statistical logic.

![LEAST SQUARES OF FIT METHOD](https://c.mql5.com/2/161/Screenshot_2025-08-06_002619.png)

We plan to combine the mathematical trendline detection with visual feedback and configurable trading parameters, allowing us to capitalize on trend bounces efficiently in dynamic markets. We intend to identify swing points, fit trendlines with sufficient touches (a minimum of 3 touches, validate their integrity, and trigger trades with risk management, all while displaying trendlines and touch points on the chart for clarity. Have a look at the result we aim for, and then we can proceed to the implementation.

![TRENDLINE FRAMEWORK](https://c.mql5.com/2/161/Screenshot_2025-08-06_003724.png)

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will start by declaring some [inputs](https://www.mql5.com/en/docs/basis/variables/inputvariables) and [structures](https://www.mql5.com/en/docs/basis/types/classes) that will make the program more dynamic.

```
//+------------------------------------------------------------------+
//|                                       a. Trendline Trader EA.mq5 |
//|                           Copyright 2025, Allan Munene Mutiiria. |
//|                                   https://t.me/Forex_Algo_Trader |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2025, Allan Munene Mutiiria."
#property link        "https://t.me/Forex_Algo_Trader"
#property description "Trendline Trader using mean Least Squares Fit"
#property version     "1.00"
#property strict

#include <Trade\Trade.mqh>                         //--- Include Trade library for trading operations
CTrade obj_Trade;                                  //--- Instantiate trade object

//+------------------------------------------------------------------+
//| Swing point structure                                            |
//+------------------------------------------------------------------+
struct Swing {                                     //--- Define swing point structure
   datetime time;                                  //--- Store swing time
   double   price;                                 //--- Store swing price
};

//+------------------------------------------------------------------+
//| Starting point structure                                         |
//+------------------------------------------------------------------+
struct StartingPoint {                             //--- Define starting point structure
   datetime time;                                  //--- Store starting point time
   double   price;                                 //--- Store starting point price
   bool     is_support;                            //--- Indicate support/resistance flag
};

//+------------------------------------------------------------------+
//| Trendline storage structure                                      |
//+------------------------------------------------------------------+
struct TrendlineInfo {                             //--- Define trendline info structure
   string   name;                                  //--- Store trendline name
   datetime start_time;                            //--- Store start time
   datetime end_time;                              //--- Store end time
   double   start_price;                           //--- Store start price
   double   end_price;                             //--- Store end price
   double   slope;                                 //--- Store slope
   bool     is_support;                            //--- Indicate support/resistance flag
   int      touch_count;                           //--- Store number of touches
   datetime creation_time;                         //--- Store creation time
   int      touch_indices[];                       //--- Store touch indices array
   bool     is_signaled;                           //--- Indicate signal flag
};

//+------------------------------------------------------------------+
//| Forward declarations                                             |
//+------------------------------------------------------------------+
void DetectSwings();                               //--- Declare swing detection function
void SortSwings(Swing &swings[], int count);       //--- Declare swing sorting function
double CalculateAngle(datetime time1, double price1, datetime time2, double price2); //--- Declare angle calculation function
bool ValidateTrendline(bool isSupport, datetime start_time, datetime ref_time, double ref_price, double slope, double tolerance_pen); //--- Declare trendline validation function
void FindAndDrawTrendlines(bool isSupport);        //--- Declare trendline finding/drawing function
void UpdateTrendlines();                           //--- Declare trendline update function
void RemoveTrendlineFromStorage(int index);        //--- Declare trendline removal function
bool IsStartingPointUsed(datetime time, double price, bool is_support); //--- Declare starting point usage check function
void LeastSquaresFit(const datetime &times[], const double &prices[], int n, double &slope, double &intercept); //--- Declare least squares fit function

//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
input int    LookbackBars = 200;                   // Set bars for swing detection lookback
input double TouchTolerance = 10.0;                // Set tolerance for touch points (points)
input int    MinTouches = 3;                       // Set minimum touch points for valid trendline
input double PenetrationTolerance = 5.0;           // Set allowance for bar penetration (points)
input int    ExtensionBars = 100;                  // Set bars to extend trendline right
input int    MinBarSpacing = 10;                   // Set minimum bar spacing between touches
input double inpLot = 0.01;                        // Set lot size
input double inpSLPoints = 100.0;                  // Set stop loss (points)
input double inpRRRatio = 1.1;                     // Set risk:reward ratio
input double MinAngle = 1.0;                       // Set minimum inclination angle (degrees)
input double MaxAngle = 89.0;                      // Set maximum inclination angle (degrees)
input bool   DeleteExpiredObjects = false;         // Enable deletion of expired/broken objects
input bool   EnableTradingSignals = true;          // Enable buy/sell signals and trades
input bool   DrawTouchArrows = true;               // Enable drawing arrows at touch points
input bool   DrawLabels = true;                    // Enable drawing trendline/point labels
input color  SupportLineColor = clrGreen;          // Set color for support trendlines
input color  ResistanceLineColor = clrRed;         // Set color for resistance trendlines

//+------------------------------------------------------------------+
//| Global variables                                                 |
//+------------------------------------------------------------------+
Swing swingLows[];                                 //--- Store swing lows
int numLows = 0;                                   //--- Track number of swing lows
Swing swingHighs[];                                //--- Store swing highs
int numHighs = 0;                                  //--- Track number of swing highs
TrendlineInfo trendlines[];                        //--- Store trendlines
int numTrendlines = 0;                             //--- Track number of trendlines
StartingPoint startingPoints[];                    //--- Store used starting points
int numStartingPoints = 0;                         //--- Track number of starting points
```

We begin by setting up the core components for the program to automate trading based on trendline touches. First, we include the "<Trade\\Trade.mqh>" library and instantiate "obj\_Trade" as a " [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade)" object to manage trade operations like executing buy and sell orders. Then, we proceed to define three [structures](https://www.mql5.com/en/docs/basis/types/classes): "Swing" with "time" (datetime) and "price" (double) to capture swing points; "StartingPoint" with "time" ( [datetime](https://www.mql5.com/en/docs/basis/types/integer/datetime)), "price" (double), and "is\_support" (bool) to track used starting points for support or resistance; and "TrendlineInfo" with "name" (string), "start\_time" and "end\_time" (datetimes), "start\_price" and "end\_price" (doubles), "slope" ( [double](https://www.mql5.com/en/docs/basis/types/double)), "is\_support" (bool), "touch\_count" (int), "creation\_time" (datetime), "touch\_indices" (int array), and "is\_signaled" (bool) to store trendline details.

Next, we forward-declare functions to handle key tasks: "DetectSwings" for identifying swing points, "SortSwings" for ordering swings, "CalculateAngle" for computing trendline inclination, "ValidateTrendline" for ensuring trendline validity, "FindAndDrawTrendlines" for creating and drawing trendlines, "UpdateTrendlines" for maintaining them, "RemoveTrendlineFromStorage" for cleanup, "IsStartingPointUsed" for checking point usage, and "LeastSquaresFit" for calculating slope and intercept using the [least squares](https://en.wikipedia.org/wiki/Least_squares "https://en.wikipedia.org/wiki/Least_squares") method.

Last, we configure [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameters and [global variables](https://www.mql5.com/en/docs/basis/variables/global): inputs like "LookbackBars" (200) for swing detection range, "TouchTolerance" (10.0 points) for touch precision, "MinTouches" (3) for validity, and the rest which are self explanatory; and globals like "swingLows" and "swingHighs" arrays with "numLows" and "numHighs" (0) for swing points, and "trendlines" and "startingPoints" arrays with "numTrendlines" and "numStartingPoints" (0) for trendline and point storage. This structured setup establishes the EA’s foundation for detecting and trading trendlines effectively. Since we are all set, we can initialize the storage arrays in the initialization.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   ArrayResize(trendlines, 0);                     //--- Resize trendlines array
   numTrendlines = 0;                              //--- Reset trendlines count
   ArrayResize(startingPoints, 0);                 //--- Resize starting points array
   numStartingPoints = 0;                          //--- Reset starting points count
   return(INIT_SUCCEEDED);                         //--- Return success
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   ArrayResize(trendlines, 0);                     //--- Resize trendlines array
   numTrendlines = 0;                              //--- Reset trendlines count
   ArrayResize(startingPoints, 0);                 //--- Resize starting points array
   numStartingPoints = 0;                          //--- Reset starting points count
}
```

To ensure proper setup and cleanup of resources, in the [OnInit](https://www.mql5.com/en/docs/event_handlers/oninit) event handler, we prepare the EA by resizing the "trendlines" array to 0 with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and setting "numTrendlines" to 0 to clear any existing trendline data, then resize the "startingPoints" array to 0 and set "numStartingPoints" to 0 to reset starting point records, and finally return "INIT\_SUCCEEDED" to confirm successful initialization.

Then, in the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) function, we do the same thing, ensuring no memory leaks when the program is removed, establishing a clean slate for the EA’s operation and proper resource management. With the initialization done, we can now proceed to defining the strategy logic. To help modularize the logic, we will use functions, and the first logic we will define is swing points detection, so we can have base trendline points.

```
//+------------------------------------------------------------------+
//| Check for new bar                                                |
//+------------------------------------------------------------------+
bool IsNewBar() {
   static datetime lastTime = 0;                      //--- Store last bar time
   datetime currentTime = iTime(_Symbol, _Period, 0); //--- Get current bar time
   if (lastTime != currentTime) {                     //--- Check for new bar
      lastTime = currentTime;                         //--- Update last time
      return true;                                    //--- Indicate new bar
   }
   return false;                                      //--- Indicate no new bar
}

//+------------------------------------------------------------------+
//| Sort swings by time (ascending, oldest first)                    |
//+------------------------------------------------------------------+
void SortSwings(Swing &swings[], int count) {
   for (int i = 0; i < count - 1; i++) {               //--- Iterate through swings
      for (int j = 0; j < count - i - 1; j++) {        //--- Compare adjacent swings
         if (swings[j].time > swings[j + 1].time) {    //--- Check time order
            Swing temp = swings[j];                    //--- Store temporary swing
            swings[j] = swings[j + 1];                 //--- Swap swings
            swings[j + 1] = temp;                      //--- Complete swap
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Detect swing highs and lows                                      |
//+------------------------------------------------------------------+
void DetectSwings() {
   numLows = 0;                                         //--- Reset lows count
   ArrayResize(swingLows, 0);                           //--- Resize lows array
   numHighs = 0;                                        //--- Reset highs count
   ArrayResize(swingHighs, 0);                          //--- Resize highs array
   int totalBars = iBars(_Symbol, _Period);             //--- Get total bars
   int effectiveLookback = MathMin(LookbackBars, totalBars); //--- Calculate effective lookback
   if (effectiveLookback < 5) {                         //--- Check sufficient bars
      Print("Not enough bars for swing detection.");    //--- Log insufficient bars
      return;                                           //--- Exit function
   }
   for (int i = 2; i < effectiveLookback - 2; i++) {    //--- Iterate through bars
      double low_i = iLow(_Symbol, _Period, i);         //--- Get current low
      double low_im1 = iLow(_Symbol, _Period, i - 1);   //--- Get previous low
      double low_im2 = iLow(_Symbol, _Period, i - 2);   //--- Get two bars prior low
      double low_ip1 = iLow(_Symbol, _Period, i + 1);   //--- Get next low
      double low_ip2 = iLow(_Symbol, _Period, i + 2);   //--- Get two bars next low
      if (low_i < low_im1 && low_i < low_im2 && low_i < low_ip1 && low_i < low_ip2) { //--- Check for swing low
         Swing s;                                       //--- Create swing struct
         s.time = iTime(_Symbol, _Period, i);           //--- Set swing time
         s.price = low_i;                               //--- Set swing price
         ArrayResize(swingLows, numLows + 1);           //--- Resize lows array
         swingLows[numLows] = s;                        //--- Add swing low
         numLows++;                                     //--- Increment lows count
      }
      double high_i = iHigh(_Symbol, _Period, i);       //--- Get current high
      double high_im1 = iHigh(_Symbol, _Period, i - 1); //--- Get previous high
      double high_im2 = iHigh(_Symbol, _Period, i - 2); //--- Get two bars prior high
      double high_ip1 = iHigh(_Symbol, _Period, i + 1); //--- Get next high
      double high_ip2 = iHigh(_Symbol, _Period, i + 2); //--- Get two bars next high
      if (high_i > high_im1 && high_i > high_im2 && high_i > high_ip1 && high_i > high_ip2) { //--- Check for swing high
         Swing s;                                       //--- Create swing struct
         s.time = iTime(_Symbol, _Period, i);           //--- Set swing time
         s.price = high_i;                              //--- Set swing price
         ArrayResize(swingHighs, numHighs + 1);         //--- Resize highs array
         swingHighs[numHighs] = s;                      //--- Add swing high
         numHighs++;                                    //--- Increment highs count
      }
   }
   if (numLows > 0) SortSwings(swingLows, numLows);     //--- Sort swing lows
   if (numHighs > 0) SortSwings(swingHighs, numHighs);  //--- Sort swing highs
}
```

Here, we implement key functions to manage bar detection and swing point identification, laying the groundwork for trendline analysis. First, we create the "IsNewBar" function, which checks for a new bar by storing "lastTime" statically as 0, comparing it with "currentTime" from [iTime](https://www.mql5.com/en/docs/series/itime) for the current symbol and period at shift 0 for the current bar, updating "lastTime" if different, and returning true for a new bar or false otherwise. Then, we proceed to implement the "SortSwings" function, which sorts the "swings" array by "time" in ascending order (oldest first) using a bubble sort algorithm, iterating through "count - 1" elements and swapping adjacent "Swing" [structs](https://www.mql5.com/en/docs/basis/types/classes) with a temporary "temp" if their times are out of order.

Last, we implement the "DetectSwings" function, resetting "numLows" and "numHighs" to 0 and resizing "swingLows" and "swingHighs" arrays to 0, calculating "effectiveLookback" as the minimum of "LookbackBars" and total bars from [iBars](https://www.mql5.com/en/docs/series/ibars), exiting with a [Print](https://www.mql5.com/en/docs/common/print) log if fewer than 5 bars are available, and iterating through bars from 2 to "effectiveLookback - 2" to identify swing lows and highs by comparing "iLow" and "iHigh" values against two prior and subsequent bars, creating "Swing" structs with "time" from "iTime" and "price" from "iLow" or [iHigh](https://www.mql5.com/en/docs/series/ihigh), adding them to "swingLows" or "swingHighs" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), incrementing counters, and sorting arrays with "SortSwings" if non-empty. These will ensure timely swing detection for accurate trendline construction. Let us now define functions to calculate the trendline inclination for restriction based on inclination and its validation.

```
//+------------------------------------------------------------------+
//| Calculate visual inclination angle                               |
//+------------------------------------------------------------------+
double CalculateAngle(datetime time1, double price1, datetime time2, double price2) {
   int x1, y1, x2, y2;                                               //--- Declare coordinate variables
   if (!ChartTimePriceToXY(0, 0, time1, price1, x1, y1)) return 0.0; //--- Convert time1/price1 to XY
   if (!ChartTimePriceToXY(0, 0, time2, price2, x2, y2)) return 0.0; //--- Convert time2/price2 to XY
   double dx = (double)(x2 - x1);                                    //--- Calculate x difference
   double dy = (double)(y2 - y1);                                    //--- Calculate y difference
   if (dx == 0.0) return (dy > 0.0 ? -90.0 : 90.0);                  //--- Handle vertical line case
   double angle = MathArctan(-dy / dx) * 180.0 / M_PI;               //--- Calculate angle in degrees
   return angle;                                                     //--- Return angle
}

//+------------------------------------------------------------------+
//| Validate trendline                                               |
//+------------------------------------------------------------------+
bool ValidateTrendline(bool isSupport, datetime start_time, datetime ref_time, double ref_price, double slope, double tolerance_pen) {
   int bar_start = iBarShift(_Symbol, _Period, start_time);          //--- Get start bar index
   if (bar_start < 0) return false;                                  //--- Check invalid bar index
   for (int bar = bar_start; bar >= 0; bar--) {                      //--- Iterate through bars
      datetime bar_time = iTime(_Symbol, _Period, bar);              //--- Get bar time
      double dk = (double)(bar_time - ref_time);                     //--- Calculate time difference
      double line_price = ref_price + slope * dk;                    //--- Calculate line price
      if (isSupport) {                                               //--- Check support case
         double low = iLow(_Symbol, _Period, bar);                   //--- Get bar low
         if (low < line_price - tolerance_pen) return false;         //--- Check if broken
      } else {                                                       //--- Handle resistance case
         double high = iHigh(_Symbol, _Period, bar);                 //--- Get bar high
         if (high > line_price + tolerance_pen) return false;        //--- Check if broken
      }
   }
   return true;                                                      //--- Return valid
}
```

We proceed to implement critical functions to calculate trendline angles and validate their integrity, ensuring robust trendline detection. First, we create the "CalculateAngle" function, which converts two points ("time1", "price1" and "time2", "price2") to chart coordinates using the [ChartTimePriceToXY](https://www.mql5.com/en/docs/chart_operations/chartxytotimeprice) function into "x1", "y1", "x2", "y2", returning 0.0 if conversion fails, then computes the x-difference "dx" and y-difference "dy", handling vertical lines by returning -90.0 or 90.0 if "dx" is zero, and calculates the angle in degrees using " [MathArctan](https://www.mql5.com/en/docs/math/matharctan)(-dy / dx) \* 180.0 / [M\_PI](https://www.mql5.com/en/docs/constants/namedconstants/mathsconstants)" for visual inclination.

Then, we proceed to implement the "ValidateTrendline" function, which validates a trendline by getting the start bar index with [iBarShift](https://www.mql5.com/en/docs/series/ibarshift) for "start\_time", returning false if invalid, and iterating from "bar\_start" to 0, calculating the trendline price at each "bar\_time" using "ref\_price + slope \* dk" where "dk" is the time difference from reference time. For support trendlines ("isSupport" true), we check if the bar’s [iLow](https://www.mql5.com/en/docs/series/ilow) falls below "line\_price - tolerance\_pen", returning false if broken; for resistance, we check if [iHigh](https://www.mql5.com/en/docs/series/ihigh) exceeds "line\_price + tolerance\_pen", returning false if broken, and return true if the trendline holds. We can now concentrate on the function for the [least squares fit](https://en.wikipedia.org/wiki/Least_squares "https://en.wikipedia.org/wiki/Least_squares") calculation logic. We will keep it straightforward.

```
//+------------------------------------------------------------------+
//| Perform least-squares fit for slope and intercept                |
//+------------------------------------------------------------------+
void LeastSquaresFit(const datetime &times[], const double &prices[], int n, double &slope, double &intercept) {
   double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0; //--- Initialize sums
   for (int k = 0; k < n; k++) {                        //--- Iterate through points
      double x = (double)times[k];                      //--- Convert time to x
      double y = prices[k];                             //--- Set price as y
      sum_x += x;                                       //--- Accumulate x
      sum_y += y;                                       //--- Accumulate y
      sum_xy += x * y;                                  //--- Accumulate x*y
      sum_x2 += x * x;                                  //--- Accumulate x^2
   }
   slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x); //--- Calculate slope
   intercept = (sum_y - slope * sum_x) / n;             //--- Calculate intercept
}
```

We implement the "LeastSquaresFit" function to compute the optimal slope and intercept for trendlines, enabling precise trendline fitting. First, we initialize variables "sum\_x", "sum\_y", "sum\_xy", and "sum\_x2" to 0 to accumulate values for the least squares calculation. Then, we proceed to iterate through "n" points in the "times" and "prices" arrays, converting each "times\[k\]" to a double as "x" and setting "prices\[k\]" as "y", adding "x" to "sum\_x", "y" to "sum\_y", "x \* y" to "sum\_xy", and "x \* x" to "sum\_x2". Last, we calculate the "slope" using the formula "(n \* sum\_xy - sum\_x \* sum\_y) / (n \* sum\_x2 - sum\_x \* sum\_x)" and the "intercept" as "(sum\_y - slope \* sum\_x) / n", providing the best-fit line for the trendline based on the input points. In case you are wondering about the formula, have a look below.

![LEAST SQUARES FIT METHOD](https://c.mql5.com/2/161/Screenshot_2025-08-06_115700.png)

This will ensure mathematically accurate trendline placement for reliable trading signals. Let us now define a function to manage the trendlines.

```
//+------------------------------------------------------------------+
//| Check if starting point is already used                          |
//+------------------------------------------------------------------+
bool IsStartingPointUsed(datetime time, double price, bool is_support) {
   for (int i = 0; i < numStartingPoints; i++) {  //--- Iterate through starting points
      if (startingPoints[i].time == time && MathAbs(startingPoints[i].price - price) < TouchTolerance * _Point && startingPoints[i].is_support == is_support) { //--- Check match
         return true;                             //--- Return used
      }
   }
   return false;                                   //--- Return not used
}

//+------------------------------------------------------------------+
//| Remove trendline from storage and optionally chart objects       |
//+------------------------------------------------------------------+
void RemoveTrendlineFromStorage(int index) {
   if (index < 0 || index >= numTrendlines) return;                    //--- Check valid index
   Print("Removing trendline from storage: ", trendlines[index].name); //--- Log removal
   if (DeleteExpiredObjects) {                                         //--- Check deletion flag
      ObjectDelete(0, trendlines[index].name);                         //--- Delete trendline object
      for (int m = 0; m < trendlines[index].touch_count; m++) {        //--- Iterate touches
         string arrow_name = trendlines[index].name + "_touch" + IntegerToString(m); //--- Generate arrow name
         ObjectDelete(0, arrow_name);                                  //--- Delete touch arrow
         string text_name = trendlines[index].name + "_point_label" + IntegerToString(m); //--- Generate text name
         ObjectDelete(0, text_name);                                   //--- Delete point label
      }
      string label_name = trendlines[index].name + "_label";           //--- Generate label name
      ObjectDelete(0, label_name);                                     //--- Delete trendline label
      string signal_arrow = trendlines[index].name + "_signal_arrow";  //--- Generate signal arrow name
      ObjectDelete(0, signal_arrow);                                   //--- Delete signal arrow
      string signal_text = trendlines[index].name + "_signal_text";    //--- Generate signal text name
      ObjectDelete(0, signal_text);                                    //--- Delete signal text
   }
   for (int i = index; i < numTrendlines - 1; i++) {                   //--- Shift array
      trendlines[i] = trendlines[i + 1];                               //--- Copy next trendline
   }
   ArrayResize(trendlines, numTrendlines - 1);                         //--- Resize trendlines array
   numTrendlines--;                                                    //--- Decrement trendlines count
}
```

We proceed to implement utility functions to manage trendline starting points and cleanup, ensuring efficient trendline tracking and chart management. First, we create the "IsStartingPointUsed" function, which iterates through "numStartingPoints" in the "startingPoints" array, checking if a given "time", "price", and "is\_support" match any existing starting point by comparing "time" exactly, "price" within "TouchTolerance \* [\_Point](https://www.mql5.com/en/docs/predefined/_point)" using [MathAbs](https://www.mql5.com/en/docs/math/mathabs), and "is\_support", returning true if found or false if not. Then, we proceed to implement the "RemoveTrendlineFromStorage" function, which validates the input "index" against "numTrendlines", exiting if invalid, and logs removal.

If "DeleteExpiredObjects" is true, we delete the trendline object with [ObjectDelete](https://www.mql5.com/en/docs/objects/objectdelete) for "trendlines\[index\].name", loop through "touch\_count" to delete touch arrows and labels with names like "trendlines\[index\].name + '\_touch' + [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString)(m)" and "trendlines\[index\].name + '\_point\_label' + IntegerToString(m)", and remove the trendline label, signal arrow, and signal text using "label\_name", "signal\_arrow", and "signal\_text". Last, we shift the "trendlines" array from "index" to "numTrendlines - 1" to remove the entry, resize "trendlines" with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and decrement their number, helping prevent duplicate trendlines and clean up expired or broken trendlines effectively. Let us now define a function to find and draw the trendlines using the helper functions we have defined.

```
//+------------------------------------------------------------------+
//| Find and draw trendlines if no active one exists                 |
//+------------------------------------------------------------------+
void FindAndDrawTrendlines(bool isSupport) {
   bool has_active = false;                       //--- Initialize active flag
   for (int i = 0; i < numTrendlines; i++) {      //--- Iterate through trendlines
      if (trendlines[i].is_support == isSupport) { //--- Check type match
         has_active = true;                       //--- Set active flag
         break;                                   //--- Exit loop
      }
   }
   if (has_active) return;                        //--- Exit if active trendline exists
   Swing swings[];                                //--- Initialize swings array
   int numSwings;                                 //--- Initialize swings count
   color lineColor;                               //--- Initialize line color
   string prefix;                                 //--- Initialize prefix
   if (isSupport) {                               //--- Handle support case
      numSwings = numLows;                        //--- Set number of lows
      ArrayResize(swings, numSwings);             //--- Resize swings array
      for (int i = 0; i < numSwings; i++) {       //--- Iterate through lows
         swings[i].time = swingLows[i].time;      //--- Copy low time
         swings[i].price = swingLows[i].price;    //--- Copy low price
      }
      lineColor = SupportLineColor;               //--- Set support line color
      prefix = "Trendline_Support_";              //--- Set support prefix
   } else {                                       //--- Handle resistance case
      numSwings = numHighs;                       //--- Set number of highs
      ArrayResize(swings, numSwings);             //--- Resize swings array
      for (int i = 0; i < numSwings; i++) {       //--- Iterate through highs
         swings[i].time = swingHighs[i].time;     //--- Copy high time
         swings[i].price = swingHighs[i].price;   //--- Copy high price
      }
      lineColor = ResistanceLineColor;            //--- Set resistance line color
      prefix = "Trendline_Resistance_";           //--- Set resistance prefix
   }
   if (numSwings < 2) return;                     //--- Exit if insufficient swings
   double pointValue = _Point;                    //--- Get point value
   double touch_tolerance = TouchTolerance * pointValue; //--- Calculate touch tolerance
   double pen_tolerance = PenetrationTolerance * pointValue; //--- Calculate penetration tolerance
   int best_j = -1;                               //--- Initialize best j index
   int max_touches = 0;                           //--- Initialize max touches
   int best_touch_indices[];                      //--- Initialize best touch indices
   double best_slope = 0.0;                       //--- Initialize best slope
   double best_intercept = 0.0;                   //--- Initialize best intercept
   datetime best_min_time = 0;                    //--- Initialize best min time
   for (int i = 0; i < numSwings - 1; i++) {      //--- Iterate through first points
      for (int j = i + 1; j < numSwings; j++) {   //--- Iterate through second points
         datetime time1 = swings[i].time;         //--- Get first time
         double price1 = swings[i].price;         //--- Get first price
         datetime time2 = swings[j].time;         //--- Get second time
         double price2 = swings[j].price;         //--- Get second price
         double dt = (double)(time2 - time1);     //--- Calculate time difference
         if (dt <= 0) continue;                   //--- Skip invalid time difference
         double initial_slope = (price2 - price1) / dt; //--- Calculate initial slope
         int touch_indices[];                     //--- Initialize touch indices
         ArrayResize(touch_indices, 0);           //--- Resize touch indices
         int touches = 0;                         //--- Initialize touches count
         ArrayResize(touch_indices, touches + 1); //--- Add first index
         touch_indices[touches] = i;              //--- Set first index
         touches++;                               //--- Increment touches
         ArrayResize(touch_indices, touches + 1); //--- Add second index
         touch_indices[touches] = j;              //--- Set second index
         touches++;                               //--- Increment touches
         for (int k = 0; k < numSwings; k++) {    //--- Iterate through swings
            if (k == i || k == j) continue;       //--- Skip used indices
            datetime tk = swings[k].time;         //--- Get swing time
            double dk = (double)(tk - time1);     //--- Calculate time difference
            double expected = price1 + initial_slope * dk; //--- Calculate expected price
            double actual = swings[k].price;      //--- Get actual price
            if (MathAbs(expected - actual) <= touch_tolerance) { //--- Check touch within tolerance
               ArrayResize(touch_indices, touches + 1); //--- Add index
               touch_indices[touches] = k;        //--- Set index
               touches++;                         //--- Increment touches
            }
         }
         if (touches >= MinTouches) {             //--- Check minimum touches
            ArraySort(touch_indices);             //--- Sort touch indices
            bool valid_spacing = true;            //--- Initialize spacing flag
            for (int m = 0; m < touches - 1; m++) { //--- Iterate through touches
               int idx1 = touch_indices[m];       //--- Get first index
               int idx2 = touch_indices[m + 1];   //--- Get second index
               int bar1 = iBarShift(_Symbol, _Period, swings[idx1].time); //--- Get first bar
               int bar2 = iBarShift(_Symbol, _Period, swings[idx2].time); //--- Get second bar
               int diff = MathAbs(bar1 - bar2);   //--- Calculate bar difference
               if (diff < MinBarSpacing) {        //--- Check minimum spacing
                  valid_spacing = false;          //--- Mark invalid spacing
                  break;                          //--- Exit loop
               }
            }
            if (valid_spacing) {                  //--- Check valid spacing
               datetime touch_times[];            //--- Initialize touch times
               double touch_prices[];             //--- Initialize touch prices
               ArrayResize(touch_times, touches); //--- Resize times array
               ArrayResize(touch_prices, touches); //--- Resize prices array
               for (int m = 0; m < touches; m++) { //--- Iterate through touches
                  int idx = touch_indices[m];      //--- Get index
                  touch_times[m] = swings[idx].time;   //--- Set time
                  touch_prices[m] = swings[idx].price; //--- Set price
               }
               double slope, intercept;                //--- Declare slope and intercept
               LeastSquaresFit(touch_times, touch_prices, touches, slope, intercept); //--- Perform least squares fit
               int adjusted_touch_indices[];           //--- Initialize adjusted indices
               ArrayResize(adjusted_touch_indices, 0); //--- Resize adjusted indices
               int adjusted_touches = 0;               //--- Initialize adjusted touches count
               for (int k = 0; k < numSwings; k++) {   //--- Iterate through swings
                  double expected = intercept + slope * (double)swings[k].time; //--- Calculate expected price
                  double actual = swings[k].price;     //--- Get actual price
                  if (MathAbs(expected - actual) <= touch_tolerance) { //--- Check touch
                     ArrayResize(adjusted_touch_indices, adjusted_touches + 1); //--- Add index
                     adjusted_touch_indices[adjusted_touches] = k; //--- Set index
                     adjusted_touches++;               //--- Increment adjusted touches
                  }
               }
               if (adjusted_touches >= MinTouches) { //--- Check minimum adjusted touches
                  datetime temp_min_time = swings[adjusted_touch_indices[0]].time; //--- Get min time
                  double temp_ref_price = intercept + slope * (double)temp_min_time; //--- Calculate ref price
                  if (ValidateTrendline(isSupport, temp_min_time, temp_min_time, temp_ref_price, slope, pen_tolerance)) { //--- Validate trendline
                     datetime temp_max_time = swings[adjusted_touch_indices[adjusted_touches - 1]].time; //--- Get max time
                     double temp_max_price = intercept + slope * (double)temp_max_time; //--- Calculate max price
                     double angle = CalculateAngle(temp_min_time, temp_ref_price, temp_max_time, temp_max_price); //--- Calculate angle
                     double abs_angle = MathAbs(angle); //--- Get absolute angle
                     if (abs_angle >= MinAngle && abs_angle <= MaxAngle) { //--- Check angle range
                        if (adjusted_touches > max_touches || (adjusted_touches == max_touches && j > best_j)) { //--- Check better trendline
                           max_touches = adjusted_touches; //--- Update max touches
                           best_j = j;                     //--- Update best j
                           best_slope = slope;             //--- Update best slope
                           best_intercept = intercept;     //--- Update best intercept
                           best_min_time = temp_min_time;  //--- Update best min time
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
   if (max_touches < MinTouches) {                //--- Check insufficient touches
      string type = isSupport ? "Support" : "Resistance"; //--- Set type string
      return;                                     //--- Exit function
   }
   int touch_indices[];                           //--- Initialize touch indices
   ArrayResize(touch_indices, max_touches);       //--- Resize touch indices
   ArrayCopy(touch_indices, best_touch_indices);  //--- Copy best indices
   int touches = max_touches;                     //--- Set touches count
   datetime min_time = best_min_time;             //--- Set min time
   double price_min = best_intercept + best_slope * (double)min_time; //--- Calculate min price
   datetime max_time = swings[touch_indices[touches - 1]].time; //--- Set max time
   double price_max = best_intercept + best_slope * (double)max_time; //--- Calculate max price
   datetime start_time_check = min_time;          //--- Set start time check
   double start_price_check = swings[touch_indices[0]].price; //--- Set start price check
   if (IsStartingPointUsed(start_time_check, start_price_check, isSupport)) { //--- Check used starting point
      return;                                     //--- Skip if used
   }
   datetime time_end = iTime(_Symbol, _Period, 0) + PeriodSeconds(_Period) * ExtensionBars; //--- Calculate end time
   double dk_end = (double)(time_end - min_time);      //--- Calculate end time difference
   double price_end = price_min + best_slope * dk_end; //--- Calculate end price
   string unique_name = prefix + TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES|TIME_SECONDS); //--- Generate unique name
   if (ObjectFind(0, unique_name) < 0) {               //--- Check if trendline exists
      ObjectCreate(0, unique_name, OBJ_TREND, 0, min_time, price_min, time_end, price_end); //--- Create trendline
      ObjectSetInteger(0, unique_name, OBJPROP_COLOR, lineColor);   //--- Set color
      ObjectSetInteger(0, unique_name, OBJPROP_STYLE, STYLE_SOLID); //--- Set style
      ObjectSetInteger(0, unique_name, OBJPROP_WIDTH, 1);           //--- Set width
      ObjectSetInteger(0, unique_name, OBJPROP_RAY_RIGHT, false);   //--- Disable right ray
      ObjectSetInteger(0, unique_name, OBJPROP_RAY_LEFT, false);    //--- Disable left ray
      ObjectSetInteger(0, unique_name, OBJPROP_BACK, false);        //--- Set to foreground
   }
   ArrayResize(trendlines, numTrendlines + 1);                      //--- Resize trendlines array
   trendlines[numTrendlines].name = unique_name;                    //--- Set trendline name
   trendlines[numTrendlines].start_time = min_time;                 //--- Set start time
   trendlines[numTrendlines].end_time = time_end;                   //--- Set end time
   trendlines[numTrendlines].start_price = price_min;               //--- Set start price
   trendlines[numTrendlines].end_price = price_end;                 //--- Set end price
   trendlines[numTrendlines].slope = best_slope;                    //--- Set slope
   trendlines[numTrendlines].is_support = isSupport;                //--- Set type
   trendlines[numTrendlines].touch_count = touches;                 //--- Set touch count
   trendlines[numTrendlines].creation_time = TimeCurrent();         //--- Set creation time
   trendlines[numTrendlines].is_signaled = false;                   //--- Set signaled flag
   ArrayResize(trendlines[numTrendlines].touch_indices, touches);   //--- Resize touch indices
   ArrayCopy(trendlines[numTrendlines].touch_indices, touch_indices); //--- Copy touch indices
   numTrendlines++;                                                 //--- Increment trendlines count
   ArrayResize(startingPoints, numStartingPoints + 1);              //--- Resize starting points array
   startingPoints[numStartingPoints].time = start_time_check; //--- Set starting point time
   startingPoints[numStartingPoints].price = start_price_check; //--- Set starting point price
   startingPoints[numStartingPoints].is_support = isSupport; //--- Set starting point type
   numStartingPoints++;                           //--- Increment starting points count
   if (DrawTouchArrows) {                         //--- Check draw arrows
      for (int m = 0; m < touches; m++) {         //--- Iterate through touches
         int idx = touch_indices[m];              //--- Get touch index
         datetime tk_time = swings[idx].time;     //--- Get touch time
         double tk_price = swings[idx].price;     //--- Get touch price
         string arrow_name = unique_name + "_touch" + IntegerToString(m); //--- Generate arrow name
         if (ObjectFind(0, arrow_name) < 0) {                             //--- Check if arrow exists
            ObjectCreate(0, arrow_name, OBJ_ARROW, 0, tk_time, tk_price); //--- Create touch arrow
            ObjectSetInteger(0, arrow_name, OBJPROP_ARROWCODE, 159);      //--- Set arrow code
            ObjectSetInteger(0, arrow_name, OBJPROP_ANCHOR, isSupport ? ANCHOR_TOP : ANCHOR_BOTTOM); //--- Set anchor
            ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, lineColor);    //--- Set color
            ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 1);            //--- Set width
            ObjectSetInteger(0, arrow_name, OBJPROP_BACK, false);         //--- Set to foreground
         }
      }
   }
   double angle = CalculateAngle(min_time, price_min, max_time, price_max); //--- Calculate angle
   string type = isSupport ? "Support" : "Resistance"; //--- Set type string
   Print(type + " Trendline " + unique_name + " drawn with " + IntegerToString(touches) + " touches. Inclination angle: " + DoubleToString(angle, 2) + " degrees."); //--- Log trendline
   if (DrawLabels) {                              //--- Check draw labels
      datetime mid_time = min_time + (max_time - min_time) / 2; //--- Calculate mid time
      double dk_mid = (double)(mid_time - min_time); //--- Calculate mid time difference
      double mid_price = price_min + best_slope * dk_mid; //--- Calculate mid price
      double label_offset = 20 * _Point * (isSupport ? -1 : 1); //--- Calculate label offset
      double label_price = mid_price + label_offset; //--- Calculate label price
      int label_anchor = isSupport ? ANCHOR_TOP : ANCHOR_BOTTOM; //--- Set label anchor
      string label_text = type + " Trendline";    //--- Set label text
      string label_name = unique_name + "_label"; //--- Generate label name
      if (ObjectFind(0, label_name) < 0) {        //--- Check if label exists
         ObjectCreate(0, label_name, OBJ_TEXT, 0, mid_time, label_price); //--- Create label
         ObjectSetString(0, label_name, OBJPROP_TEXT, label_text); //--- Set text
         ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrBlack); //--- Set color
         ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 8); //--- Set font size
         ObjectSetInteger(0, label_name, OBJPROP_ANCHOR, label_anchor); //--- Set anchor
         ObjectSetDouble(0, label_name, OBJPROP_ANGLE, angle); //--- Set angle
         ObjectSetInteger(0, label_name, OBJPROP_BACK, false); //--- Set to foreground
      }
      color point_label_color = isSupport ? clrSaddleBrown : clrDarkGoldenrod; //--- Set point label color
      double point_text_offset = 20.0 * _Point;   //--- Set point text offset
      for (int m = 0; m < touches; m++) {         //--- Iterate through touches
         int idx = touch_indices[m];              //--- Get touch index
         datetime tk_time = swings[idx].time;     //--- Get touch time
         double tk_price = swings[idx].price;     //--- Get touch price
         double text_price;                       //--- Initialize text price
         int point_text_anchor;                   //--- Initialize text anchor
         if (isSupport) {                         //--- Handle support
            text_price = tk_price - point_text_offset; //--- Set text price below
            point_text_anchor = ANCHOR_LEFT;      //--- Set left anchor
         } else {                                 //--- Handle resistance
            text_price = tk_price + point_text_offset; //--- Set text price above
            point_text_anchor = ANCHOR_BOTTOM;    //--- Set bottom anchor
         }
         string text_name = unique_name + "_point_label" + IntegerToString(m); //--- Generate text name
         string point_text = "Pt " + IntegerToString(m + 1); //--- Set point text
         if (ObjectFind(0, text_name) < 0) {     //--- Check if text exists
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

Here, we implement the "FindAndDrawTrendlines" function to identify and draw trendlines, ensuring only one active trendline per type with optimal touch points. First, we check for an existing trendline by iterating through "numTrendlines" in "trendlines", setting "has\_active" to true if "is\_support" matches the input, and exiting if found. Then, we proceed to set up for support or resistance based on "isSupport": for support, we copy "numLows" to "numSwings", populate "swings" from "swingLows", set "lineColor" to "SupportLineColor", and "prefix" to "Trendline\_Support\_"; for resistance, we use "numHighs", "swingHighs", "ResistanceLineColor", and "Trendline\_Resistance\_", exiting if "numSwings" is less than 2. Next, we calculate "touch\_tolerance" and "pen\_tolerance" using "TouchTolerance" and "PenetrationTolerance" with [\_Point](https://www.mql5.com/en/docs/predefined/_point), and iterate through "numSwings" pairs to compute an initial "initial\_slope", collecting "touch\_indices" for points within touch tolerance.

If touches meet "MinTouches" and pass "MinBarSpacing" via [iBarShift](https://www.mql5.com/en/docs/series/ibarshift), we use "LeastSquaresFit" to get "slope" and "intercept", recheck touches, and validate with "ValidateTrendline" and "CalculateAngle" against "MinAngle" and "MaxAngle", updating "best\_j", "max\_touches", "best\_slope", "best\_intercept", "best\_min\_time", and "best\_touch\_indices" for the best trendline. Last, if "max\_touches" meets "MinTouches" and the starting point is unused via "IsStartingPointUsed", we create a trendline with the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function as [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) using "unique\_name", draw touch arrows and labels if "DrawTouchArrows" and "DrawLabels" are true, store details in "trendlines", add the starting point to starting points and log, ensuring precise trendline creation. What now remains is the management of the existing trendlines via continuous updates and checking for crosses for signals. We will incorporate all the logic in a single function for simplicity.

```
//+------------------------------------------------------------------+
//| Update trendlines and check for signals                          |
//+------------------------------------------------------------------+
void UpdateTrendlines() {
   datetime current_time = iTime(_Symbol, _Period, 0);       //--- Get current time
   double pointValue = _Point;                               //--- Get point value
   double pen_tolerance = PenetrationTolerance * pointValue; //--- Calculate penetration tolerance
   double touch_tolerance = TouchTolerance * pointValue;     //--- Calculate touch tolerance
   for (int i = numTrendlines - 1; i >= 0; i--) {            //--- Iterate trendlines backward
      string type = trendlines[i].is_support ? "Support" : "Resistance"; //--- Determine trendline type
      string name = trendlines[i].name;                      //--- Get trendline name
      if (current_time > trendlines[i].end_time) {           //--- Check if expired
         PrintFormat("%s trendline %s is no longer valid (expired). End time: %s, Current time: %s.", type, name, TimeToString(trendlines[i].end_time), TimeToString(current_time)); //--- Log expiration
         RemoveTrendlineFromStorage(i);                      //--- Remove trendline
         continue;                                           //--- Skip to next
      }
      datetime prev_bar_time = iTime(_Symbol, _Period, 1);   //--- Get previous bar time
      double dk = (double)(prev_bar_time - trendlines[i].start_time); //--- Calculate time difference
      double line_price = trendlines[i].start_price + trendlines[i].slope * dk; //--- Calculate line price
      double prev_low = iLow(_Symbol, _Period, 1);           //--- Get previous bar low
      double prev_high = iHigh(_Symbol, _Period, 1);         //--- Get previous bar high
      bool broken = false;                                   //--- Initialize broken flag
      if (trendlines[i].is_support && prev_low < line_price - pen_tolerance) { //--- Check support break
         PrintFormat("%s trendline %s is no longer valid (broken by price). Line price: %.5f, Prev low: %.5f, Penetration: %.5f points.", type, name, line_price, prev_low, PenetrationTolerance); //--- Log break
         RemoveTrendlineFromStorage(i);           //--- Remove trendline
         broken = true;                           //--- Set broken flag
      } else if (!trendlines[i].is_support && prev_high > line_price + pen_tolerance) { //--- Check resistance break
         PrintFormat("%s trendline %s is no longer valid (broken by price). Line price: %.5f, Prev high: %.5f, Penetration: %.5f points.", type, name, line_price, prev_high, PenetrationTolerance); //--- Log break
         RemoveTrendlineFromStorage(i);           //--- Remove trendline
         broken = true;                           //--- Set broken flag
      }
      if (!broken && !trendlines[i].is_signaled && EnableTradingSignals) { //--- Check for trading signal
         bool touched = false;                    //--- Initialize touched flag
         string signal_type = "";                 //--- Initialize signal type
         color signal_color = clrNONE;            //--- Initialize signal color
         int arrow_code = 0;                      //--- Initialize arrow code
         int anchor = 0;                          //--- Initialize anchor
         double text_angle = 0.0;                 //--- Initialize text angle
         double text_offset = 0.0;                //--- Initialize text offset
         double text_price = 0.0;                 //--- Initialize text price
         int text_anchor = 0;                     //--- Initialize text anchor
         if (trendlines[i].is_support && MathAbs(prev_low - line_price) <= touch_tolerance) { //--- Check support touch
            touched = true;                       //--- Set touched flag
            signal_type = "BUY";                  //--- Set buy signal
            signal_color = clrBlue;               //--- Set blue color
            arrow_code = 217;                     //--- Set up arrow for support (BUY)
            anchor = ANCHOR_TOP;                  //--- Set top anchor
            text_angle = -90.0;                   //--- Set vertical upward for BUY
            text_offset = -20 * pointValue;        //--- Set text offset
            text_price = line_price + text_offset; //--- Calculate text price
            text_anchor = ANCHOR_LEFT;            //--- Set left anchor
            double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK), _Digits); //--- Get ask price
            double SL = NormalizeDouble(Ask - inpSLPoints * _Point, _Digits); //--- Calculate stop loss
            double TP = NormalizeDouble(Ask + (inpSLPoints * inpRRRatio) * _Point, _Digits); //--- Calculate take profit
            obj_Trade.Buy(inpLot, _Symbol, Ask, SL, TP); //--- Execute buy trade
         } else if (!trendlines[i].is_support && MathAbs(prev_high - line_price) <= touch_tolerance) { //--- Check resistance touch
            touched = true;                       //--- Set touched flag
            signal_type = "SELL";                 //--- Set sell signal
            signal_color = clrRed;                //--- Set red color
            arrow_code = 218;                     //--- Set down arrow for resistance (SELL)
            anchor = ANCHOR_BOTTOM;               //--- Set bottom anchor
            text_angle = 90.0;                    //--- Set vertical downward for SELL
            text_offset = 20 * pointValue;       //--- Set text offset
            text_price = line_price + text_offset; //--- Calculate text price
            text_anchor = ANCHOR_BOTTOM;          //--- Set bottom anchor
            double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID), _Digits); //--- Get bid price
            double SL = NormalizeDouble(Bid + inpSLPoints * _Point, _Digits); //--- Calculate stop loss
            double TP = NormalizeDouble(Bid - (inpSLPoints * inpRRRatio) * _Point, _Digits); //--- Calculate take profit
            obj_Trade.Sell(inpLot, _Symbol, Bid, SL, TP); //--- Execute sell trade
         }
         if (touched) {                           //--- Check if touched
            PrintFormat("Signal generated for %s trendline %s: %s at price %.5f, time %s.", type, name, signal_type, line_price, TimeToString(current_time)); //--- Log signal
            string arrow_name = name + "_signal_arrow"; //--- Generate signal arrow name
            if (ObjectFind(0, arrow_name) < 0) {  //--- Check if arrow exists
               ObjectCreate(0, arrow_name, OBJ_ARROW, 0, prev_bar_time, line_price); //--- Create signal arrow
               ObjectSetInteger(0, arrow_name, OBJPROP_ARROWCODE, arrow_code); //--- Set arrow code
               ObjectSetInteger(0, arrow_name, OBJPROP_ANCHOR, anchor); //--- Set anchor
               ObjectSetInteger(0, arrow_name, OBJPROP_COLOR, signal_color); //--- Set color
               ObjectSetInteger(0, arrow_name, OBJPROP_WIDTH, 1); //--- Set width
               ObjectSetInteger(0, arrow_name, OBJPROP_BACK, false); //--- Set to foreground
            }
            string text_name = name + "_signal_text"; //--- Generate signal text name
            if (ObjectFind(0, text_name) < 0) {   //--- Check if text exists
               ObjectCreate(0, text_name, OBJ_TEXT, 0, prev_bar_time, text_price); //--- Create signal text
               ObjectSetString(0, text_name, OBJPROP_TEXT, " " + signal_type); //--- Set text content
               ObjectSetInteger(0, text_name, OBJPROP_COLOR, signal_color); //--- Set color
               ObjectSetInteger(0, text_name, OBJPROP_FONTSIZE, 10); //--- Set font size
               ObjectSetInteger(0, text_name, OBJPROP_ANCHOR, text_anchor); //--- Set anchor
               ObjectSetDouble(0, text_name, OBJPROP_ANGLE, text_angle); //--- Set angle
               ObjectSetInteger(0, text_name, OBJPROP_BACK, false); //--- Set to foreground
            }
            trendlines[i].is_signaled = true;     //--- Set signaled flag
         }
      }
   }
}
```

To ensure active trendlines are monitored and acted upon, we create the "UpdateTrendlines" function, a void one since we don't need to return anything. First, we obtain the "current\_time" using [iTime](https://www.mql5.com/en/docs/series/itime) for the current bar and calculate "pointValue" as [\_Point](https://www.mql5.com/en/docs/predefined/_point), "pen\_tolerance" as "PenetrationTolerance \* pointValue", and "touch\_tolerance" as "TouchTolerance \* pointValue". Then, we proceed to iterate backward through "numTrendlines" in the "trendlines" array, determining the trendline "type" as "Support" or "Resistance" based on "is\_support", and checking if "current\_time" exceeds "end\_time", logging expiration with [PrintFormat](https://www.mql5.com/en/docs/common/printformat) and removing the trendline with "RemoveTrendlineFromStorage" if expired.

Next, for non-expired trendlines, we calculate the trendline price at "prev\_bar\_time" (from [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1) using "start\_price + slope \* dk", check if the trendline is broken by comparing "prev\_low" or "prev\_high" against "line\_price" with "pen\_tolerance", logging breaks with "PrintFormat" and removing with "RemoveTrendlineFromStorage" if broken.

Last, if not broken and "is\_signaled" is false with "EnableTradingSignals" true, we check for touches: for support, if "prev\_low" is within "touch\_tolerance" of "line\_price", we set a "BUY" signal, execute a buy trade with "obj\_Trade.Buy" using "inpLot", "Ask", "SL", and "TP" calculated with "inpSLPoints" and "inpRRRatio", and draw a blue up arrow (code 217) and text; for resistance, if "prev\_high" is within tolerance, we set a "SELL" signal, execute a sell trade with "obj\_Trade.Sell", and draw a red down arrow (code 218) and text, logging with "PrintFormat", creating objects with [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate), setting properties with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) and [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring), and marking "is\_signaled" true, ensuring trendlines are updated and generate accurate trading signals. The choice of the arrow codes to use is dependent on you. Here is a list of codes you could use from the [MQL5-defined Wingdings](https://www.mql5.com/en/docs/constants/objectconstants/wingdings) codes.

![MQL5 WINGDINGS](https://c.mql5.com/2/161/C_MQL5_WINGDINGS.png)

We can now call these functions in the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler for the system to give tick-based feedback.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   if (!IsNewBar()) return;                        //--- Exit if not new bar
   DetectSwings();                                 //--- Detect swings
   UpdateTrendlines();                             //--- Update trendlines
   FindAndDrawTrendlines(true);                    //--- Find/draw support trendlines
}
```

In the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, we orchestrate trendline detection and trading on each new bar logic. First, we check if a new bar has formed by calling "IsNewBar", exiting immediately if false to avoid redundant processing. Then, we proceed to call "DetectSwings" to identify and update swing highs and lows stored in "swingHighs" and "swingLows". Next, we invoke "UpdateTrendlines" to validate existing trendlines, remove expired or broken ones, and generate trading signals if price touches are detected within a defined touch tolerance. Last, we call the "FindAndDrawTrendlines" function with a true parameter to create support trendlines, ensuring new trendlines are drawn only if no active trendline of the support type exists. Upon compilation, we get the following outcome.

![CONFIRMED SUPPORT TRENDLINE](https://c.mql5.com/2/161/Screenshot_2025-08-06_131220.png)

From the image, we can see that we find, analyze, draw, and trade the trendline upon touch. Expired lines are also removed from the storage array successfully. We can achieve the same thing for resistance trendlines as well by calling the same function as support, but having the input parameter as false.

```
//--- other ontick functions

FindAndDrawTrendlines(false);                   //--- Find/draw resistance trendlines

//---
```

Upon passing the function and compilation, we get the following outcome.

![RESISTANCE TRENDLINE](https://c.mql5.com/2/161/Screenshot_2025-08-06_134552.png)

From the image, we can see that we detect and trade the resistance trendlines as well. When we test and combine everything, we get the following outcome.

![COMBINED OUTCOME](https://c.mql5.com/2/161/Screenshot_2025-08-06_134954.png)

From the image, we can see that we detect the trendlines, visualize them, and act upon them when the price touches them, hence achieving our objectives. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/162/Screenshot_2025-08-06_164554.png)

Backtest report:

![REPORT](https://c.mql5.com/2/162/Screenshot_2025-08-06_164641.png)

### Conclusion

In conclusion, we’ve developed a [trendline trading strategy](https://www.mql5.com/go?link=https://howtotrade.com/trading-strategies/trendline-strategy/ "https://howtotrade.com/trading-strategies/trendline-strategy/") program in MQL5 utilizing the [least squares fit method](https://en.wikipedia.org/wiki/Least_squares "https://en.wikipedia.org/wiki/Least_squares") to detect robust support and resistance trendlines, generating automated buy and sell signals with visual aids like arrows and labels. Through modular components like the "TrendlineInfo" [structure](https://www.mql5.com/en/docs/basis/types/classes) and functions such as "FindAndDrawTrendlines", this offers a disciplined approach to trend-based trading that you can refine by adjusting its parameters.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this trendline system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19077.zip "Download all attachments in the single ZIP archive")

[a.\_Trendline\_Trader\_EA.mq5](https://www.mql5.com/en/articles/download/19077/a._trendline_trader_ea.mq5 "Download a._Trendline_Trader_EA.mq5")(45.62 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/493131)**

![Neural Networks in Trading: A Parameter-Efficient Transformer with Segmented Attention (PSformer)](https://c.mql5.com/2/102/Parameter-efficient_Transformer_with_segmented_attention_PSformer____LOGO.png)[Neural Networks in Trading: A Parameter-Efficient Transformer with Segmented Attention (PSformer)](https://www.mql5.com/en/articles/16439)

This article introduces the new PSformer framework, which adapts the architecture of the vanilla Transformer to solving problems related to multivariate time series forecasting. The framework is based on two key innovations: the Parameter Sharing (PS) mechanism and the Segment Attention (SegAtt).

![Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://c.mql5.com/2/162/18761-integrating-mql5-with-data-logo.png)[Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://www.mql5.com/en/articles/18761)

This part focuses on building a flexible, adaptive trading model trained on historical XAUUSD data, preparing it for ONNX export and potential integration into live trading systems.

![Developing a Replay System (Part 76): New Chart Trade (III)](https://c.mql5.com/2/103/Desenvolvendo_um_sistema_de_Replay_Parte_76___LOGO.png)[Developing a Replay System (Part 76): New Chart Trade (III)](https://www.mql5.com/en/articles/12443)

In this article, we'll look at how the code of DispatchMessage, missing from the previous article, works. We will laso introduce the topic of the next article. For this reason, it is important to understand how this code works before moving on to the next topic. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Neural Networks in Trading: Enhancing Transformer Efficiency by Reducing Sharpness (Final Part)](https://c.mql5.com/2/102/Neural_Networks_in_Trading__Improving_Transformer_Efficiency_by_Reducing_Sharpness___Final__LOGO.png)[Neural Networks in Trading: Enhancing Transformer Efficiency by Reducing Sharpness (Final Part)](https://www.mql5.com/en/articles/16403)

SAMformer offers a solution to the key drawbacks of Transformer models in long-term time series forecasting, such as training complexity and poor generalization on small datasets. Its shallow architecture and sharpness-aware optimization help avoid suboptimal local minima. In this article, we will continue to implement approaches using MQL5 and evaluate their practical value.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/19077&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068341347007526991)

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