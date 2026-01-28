---
title: Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization
url: https://www.mql5.com/en/articles/17865
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 7
scraped_at: 2026-01-22T17:47:36.994896
---

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/17865&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049383309823289990)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 14)](https://www.mql5.com/en/articles/17741), we developed a trade layering strategy using [Moving Average Convergence Divergence](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/macd") (MACD) and [Relative Strength Indicator](https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi "https://www.metatrader5.com/en/terminal/help/indicators/oscillators/rsi") (RSI) with statistical methods to scale positions in trending markets dynamically. Now, in Part 15, we focus on automating the [Cypher harmonic pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/ "https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/"), a Fibonacci-based reversal pattern, with an Expert Advisor (EA) that detects, visualizes and trades this structure in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5). We will cover the following topics:

1. [Understanding the Cypher Pattern Architecture](https://www.mql5.com/en/articles/17865#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/17865#para2)
3. [Backtesting and Optimization](https://www.mql5.com/en/articles/17865#para3)
4. [Conclusion](https://www.mql5.com/en/articles/17865#para4)

By the end of this article, you’ll have a fully functional program that identifies Cypher patterns, annotates charts with clear visuals, and executes trades with precision—let’s dive in!

### Understanding the Cypher Pattern Architecture

The Cypher pattern is a harmonic trading formation defined by five key swing points—X, A, B, C, and D—and exists in two forms: a bullish pattern and a bearish pattern. In a bullish Cypher, the structure forms a low-high-low-high-low sequence where point X is a swing low, point A a swing high, point B a swing low, point C a swing high, and point D a swing low (with D positioned below X). Conversely, a bearish Cypher forms a high-low-high-low-high sequence, with point X as a swing high and point D positioned above X. Below are the visualized pattern types.

Bullish Cypher Harmonic Pattern:

![BULLISH CYPHER](https://c.mql5.com/2/136/Screenshot_2025-04-20_164430.png)

Bearish Cypher Harmonic Pattern:

![BEARISH CYPHER](https://c.mql5.com/2/136/Screenshot_2025-04-20_164335.png)

To identify the patterns, below is our structured approach:

- Defining the XA Leg: The initial move from point X to point A establishes the reference distance for the pattern, setting the direction (upward for bearish, downward for bullish).
- Establishing the AB Leg: For both pattern types, point B should retrace between 38.2% and 61.8% of the XA move, confirming a moderate correction of the initial movement.
- Analyzing the BC Leg: This leg should extend between 127.2% and 141.4% of the AB leg, ensuring a strong counter-movement before the final leg.
- Setting the CD Leg: The final leg should retrace approximately 78.6% of the XC move (from X to C), marking the potential reversal zone.

[By applying these geometric and](https://www.mql5.com/en/articles/17865#para1) [Fibonacci-based](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement") criteria, our trading system will systematically detect valid Cypher patterns in historical price data. Once a pattern is confirmed, the program will visualize the formation on the chart with annotated triangles, trend lines, and labels for points X, A, B, C, and D, as well as trade levels. This setup enables automated trade execution based on the calculated entry, stop-loss, and take-profit levels, leveraging the pattern’s predictive power for market reversals.

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link      "https://t.me/Forex_Algo_Trader"
#property version   "1.00"
#property description "This EA trades based on Cypher Strategy with visualization"
#property strict //--- Forces strict coding rules to catch errors early

//--- Include the Trade library from MQL5 to handle trading operations like buying and selling
#include <Trade\Trade.mqh>
//--- Create an instance (object) of the CTrade class to use for placing trades
CTrade obj_Trade;

//--- Input parameters let the user customize the EA without editing the code
input int    SwingHighCount    = 5;      // How many bars to check on the left to find a swing point (high or low)
input int    SwingLowCount     = 5;      // How many bars to check on the right to confirm a swing point
input double FibonacciTolerance = 0.10;  // Allowed error margin (10%) for Fibonacci ratios in the pattern
input double TradeVolume       = 0.01;   // Size of the trade (e.g., 0.01 lots is small for testing)
input bool   TradingEnabled    = true;   // True = EA can trade; False = only visualize patterns

//--- Define the Cypher pattern rules as a comment for reference
//--- Bullish Cypher: X (low), A (high), B (low), C (high), D (low)
//---   XA > 0; AB = 0.382-0.618 XA; BC = 1.272-1.414 AB; CD = 0.786 XC; D < X
//--- Bearish Cypher: X (high), A (low), B (high), C (low), D (high)
//---   XA > 0; AB = 0.382-0.618 XA; BC = 1.272-1.414 AB; CD = 0.786 XC; D > X

//--- Define a structure (like a custom data type) to store swing point info
struct SwingPoint {
   datetime TimeOfSwing;    //--- When the swing happened (date and time of the bar)
   double   PriceAtSwing;   //--- Price at the swing (high or low)
   bool     IsSwingHigh;    //--- True = swing high; False = swing low
};

//--- Create a dynamic array to hold all detected swing points
SwingPoint SwingPoints[];
```

Here, we initiate the [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") implementation of the Cypher pattern trading system by including the "Trade.mqh" library to enable trading operations and creating an instance of the "CTrade" class, named "obj\_Trade", for trade execution.

We define [input](https://www.mql5.com/en/docs/basis/variables/inputvariables) parameters—"SwingHighCount" and "SwingLowCount" (both 5) for swing point detection, "FibonacciTolerance" (0.10) for [Fibonacci](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement") ratio flexibility, "TradeVolume" (0.01 lots) for trade size, and "TradingEnabled" (true) to toggle trading—allowing user customization.

The "SwingPoint" [structure](https://www.mql5.com/en/docs/basis/types/classes) is defined with "TimeOfSwing" (datetime), "PriceAtSwing" (double), and "IsSwingHigh" (boolean) to store swing point details, and a dynamic "SwingPoints" array holds all detected swing points for pattern analysis.

Next, we can define functions that will help us to visualize the patterns in the chart.

```
//+------------------------------------------------------------------+
//| Helper: Draw a filled triangle                                   |
//+------------------------------------------------------------------+
//--- Function to draw a triangle on the chart to highlight pattern segments
void DrawTriangle(string TriangleName, datetime Time1, double Price1, datetime Time2, double Price2, datetime Time3, double Price3, color LineColor, int LineWidth, bool FillTriangle, bool DrawBehind) {
   //--- Create a triangle object using three points (time, price) on the chart
   if(ObjectCreate(0, TriangleName, OBJ_TRIANGLE, 0, Time1, Price1, Time2, Price2, Time3, Price3)) {
      ObjectSetInteger(0, TriangleName, OBJPROP_COLOR, LineColor);      //--- Set the triangle’s color (e.g., blue or red)
      ObjectSetInteger(0, TriangleName, OBJPROP_STYLE, STYLE_SOLID);    //--- Use a solid line style
      ObjectSetInteger(0, TriangleName, OBJPROP_WIDTH, LineWidth);      //--- Set the line thickness
      ObjectSetInteger(0, TriangleName, OBJPROP_FILL, FillTriangle);    //--- Fill the triangle with color if true
      ObjectSetInteger(0, TriangleName, OBJPROP_BACK, DrawBehind);      //--- Draw behind candles if true
   }
}
```

Here, we implement the "DrawTriangle" function to enhance the visualization of the Cypher pattern by drawing a filled triangle on the [MetaTrader 5](https://www.metatrader5.com/ "https://www.metatrader5.com/") chart, highlighting specific segments of the pattern for better trader comprehension. The function accepts multiple parameters to define the triangle’s appearance and position: "TriangleName" ( [string](https://www.mql5.com/en/docs/basis/types/stringconst)) provides a unique identifier for the object, "Time1", "Time2", and "Time3" (datetime) specify the time coordinates of the triangle’s three vertices, while "Price1", "Price2", and "Price3" (double) set the corresponding price levels.

Additional parameters include "LineColor" (color) to determine the outline color (e.g., blue for bullish patterns, red for bearish), "LineWidth" (int) to set the thickness of the triangle’s borders, "FillTriangle" (bool) to decide whether the triangle is filled with color, and "DrawBehind" (bool) to control whether the triangle is rendered behind chart candles to avoid obstructing price data.

Within the function, we use the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to create a triangle object on the chart, specifying the object type as [OBJ\_TRIANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) and passing the provided time and price coordinates for the three points. The function checks if the object creation is successful before proceeding to configure its properties.

If creation succeeds, we call the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function multiple times to set the triangle’s attributes: [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) assigns the "LineColor" value, "OBJPROP\_STYLE" is set to "STYLE\_SOLID" for a solid outline, "OBJPROP\_WIDTH" applies the "LineWidth" value, "OBJPROP\_FILL" uses the "FillTriangle" boolean to enable or disable filling, and [OBJPROP\_BACK](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) uses the "DrawBehind" boolean to ensure the triangle appears behind candles when true.

We can now define the rest of the helper functions via the same logic.

```
//+------------------------------------------------------------------+
//| Helper: Draw a trend line                                        |
//+------------------------------------------------------------------+
//--- Function to draw a straight line between two points on the chart
void DrawTrendLine(string LineName, datetime StartTime, double StartPrice, datetime EndTime, double EndPrice, color LineColor, int LineWidth, int LineStyle) {
   //--- Create a trend line object connecting two points (start time/price to end time/price)
   if(ObjectCreate(0, LineName, OBJ_TREND, 0, StartTime, StartPrice, EndTime, EndPrice)) {
      ObjectSetInteger(0, LineName, OBJPROP_COLOR, LineColor);      //--- Set the line color
      ObjectSetInteger(0, LineName, OBJPROP_STYLE, LineStyle);      //--- Set line style (e.g., solid or dashed)
      ObjectSetInteger(0, LineName, OBJPROP_WIDTH, LineWidth);      //--- Set line thickness
      ObjectSetInteger(0, LineName, OBJPROP_BACK, true);
   }
}

//+------------------------------------------------------------------+
//| Helper: Draw a dotted trend line                                 |
//+------------------------------------------------------------------+
//--- Function to draw a horizontal dotted line (e.g., for entry or take-profit levels)
void DrawDottedLine(string LineName, datetime StartTime, double LinePrice, datetime EndTime, color LineColor) {
   //--- Create a horizontal line from start time to end time at a fixed price
   if(ObjectCreate(0, LineName, OBJ_TREND, 0, StartTime, LinePrice, EndTime, LinePrice)) {
      ObjectSetInteger(0, LineName, OBJPROP_COLOR, LineColor);      //--- Set the line color
      ObjectSetInteger(0, LineName, OBJPROP_STYLE, STYLE_DOT);      //--- Use dotted style
      ObjectSetInteger(0, LineName, OBJPROP_WIDTH, 1);              //--- Thin line
   }
}

//+------------------------------------------------------------------+
//| Helper: Draw anchored text label (for pivots and levels)         |
//+------------------------------------------------------------------+
//--- Function to place text labels (e.g., "X" or "TP1") on the chart
void DrawTextLabel(string LabelName, string LabelText, datetime LabelTime, double LabelPrice, color TextColor, int FontSize, bool IsAbove) {
   //--- Create a text object at a specific time and price
   if(ObjectCreate(0, LabelName, OBJ_TEXT, 0, LabelTime, LabelPrice)) {
      ObjectSetString(0, LabelName, OBJPROP_TEXT, LabelText);          //--- Set the text to display
      ObjectSetInteger(0, LabelName, OBJPROP_COLOR, TextColor);        //--- Set text color
      ObjectSetInteger(0, LabelName, OBJPROP_FONTSIZE, FontSize);      //--- Set text size
      ObjectSetString(0, LabelName, OBJPROP_FONT, "Arial Bold");       //--- Use bold Arial font
      //--- Position text below if it’s a high point, above if it’s a low point
      ObjectSetInteger(0, LabelName, OBJPROP_ANCHOR, IsAbove ? ANCHOR_BOTTOM : ANCHOR_TOP);
      ObjectSetInteger(0, LabelName, OBJPROP_ALIGN, ALIGN_CENTER);     //--- Center the text
   }
}
```

Here, we implement the "DrawTrendLine" function to draw a straight line connecting [Cypher pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/ "https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/") swing points on the chart, using parameters "LineName" (string), "StartTime", "EndTime" (datetime), "StartPrice", "EndPrice" (double), "LineColor" (color), "LineWidth" (int), and "LineStyle" (int). We use the ObjectCreate function to create an "OBJ\_TREND" line and, if successful, set "OBJPROP\_COLOR", "OBJPROP\_STYLE", "OBJPROP\_WIDTH", and "OBJPROP\_BACK" (true) with "ObjectSetInteger" for visibility behind candles.

The "DrawDottedLine" function draws a horizontal dotted line for trade levels, using "LineName" (string), "StartTime", "EndTime" (datetime), "LinePrice" (double), and "LineColor" (color). We create an [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object with "ObjectCreate" at a fixed price and set "OBJPROP\_COLOR", "OBJPROP\_STYLE" to "STYLE\_DOT", and "OBJPROP\_WIDTH" to 1 using [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) for a subtle marker.

The "DrawTextLabel" function places text labels for swing points or trade levels, taking "LabelName", "LabelText" (string), "LabelTime" (datetime), "LabelPrice" (double), "TextColor" (color), "FontSize" (int), and "IsAbove" (bool). We create an [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object) object with "ObjectCreate" and use "ObjectSetString" for "OBJPROP\_TEXT" and "OBJPROP\_FONT" ("Arial Bold"), and "ObjectSetInteger" for "OBJPROP\_COLOR", "OBJPROP\_FONTSIZE", "OBJPROP\_ANCHOR" ("ANCHOR\_BOTTOM" or "ANCHOR\_TOP"), and "OBJPROP\_ALIGN" ( [ALIGN\_CENTER](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer)) to ensure clear annotations.

Armed with these variables and functions, we can graduate to the OnTick event handler and begin the pattern recognition. However, since we won't need to process anything on every tick, we need to define a logic that we can use to process the identification once per bar.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
//--- Main function that runs every time a new price tick arrives
void OnTick() {
   //--- Use a static variable to track the last bar’s time so we only process new bars
   static datetime LastProcessedBarTime = 0;
   //--- Get the time of the second-to-last bar (latest complete bar)
   datetime CurrentBarTime = iTime(_Symbol, _Period, 1);
   //--- If no new bar has formed, exit to avoid over-processing
   if(CurrentBarTime == LastProcessedBarTime)
      return;
   LastProcessedBarTime = CurrentBarTime;  //--- Update to the current bar
}
```

On the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler, which serves as the main event handler that executes every time a new price tick arrives in the [MetaTrader 5 platform](https://www.metatrader5.com/ "https://www.metatrader5.com/"), we declare a static variable "LastProcessedBarTime" to track the timestamp of the last processed bar, ensuring the function processes only new bars to optimize performance.

Using the [iTime](https://www.mql5.com/en/docs/series/itime) function, we retrieve the time of the second-to-last bar (the latest complete bar) and store it in "CurrentBarTime". We then compare "CurrentBarTime" with "LastProcessedBarTime" to check if a new bar has formed; if they are equal, we exit the function with a return statement to avoid redundant processing.

If a new bar is detected, we update "LastProcessedBarTime" to "CurrentBarTime", allowing the function to proceed with subsequent logic for analyzing price data and detecting Cypher patterns. Next, we need to define variables that will help define swing point levels.

```
//--- Clear the SwingPoints array to start fresh each time
ArrayResize(SwingPoints, 0);
//--- Get the total number of bars on the chart
int TotalBars = Bars(_Symbol, _Period);
int StartBarIndex = SwingHighCount;         //--- Start checking swings after SwingHighCount bars
int EndBarIndex = TotalBars - SwingLowCount; //--- Stop before the last SwingLowCount bars
```

We use the [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) function to clear the "SwingPoints" array by setting its size to 0, ensuring a fresh start for storing new swing points on each new bar. We then retrieve the total number of bars on the chart using the [Bars](https://www.mql5.com/en/docs/series/bars) function, storing the result in "TotalBars", which defines the scope of historical data to analyze.

To focus on relevant bars for swing detection, we set "StartBarIndex" to the value of "SwingHighCount", marking the earliest bar to check for swings, and calculate "EndBarIndex" as "TotalBars" minus "SwingLowCount", ensuring we stop before the last few bars to allow sufficient data for confirming swing points.

With these, we can loop and gather swing point data.

```
//--- Loop through bars to find swing highs and lows (swing points)
for(int BarIndex = EndBarIndex - 1; BarIndex >= StartBarIndex; BarIndex--) {
   bool IsSwingHigh = true;   //--- Assume it’s a high until proven otherwise
   bool IsSwingLow = true;    //--- Assume it’s a low until proven otherwise
   double CurrentBarHigh = iHigh(_Symbol, _Period, BarIndex); //--- Get the high of this bar
   double CurrentBarLow = iLow(_Symbol, _Period, BarIndex);   //--- Get the low of this bar
   //--- Check bars to the left and right to confirm it’s a swing point
   for(int NeighborIndex = BarIndex - SwingHighCount; NeighborIndex <= BarIndex + SwingLowCount; NeighborIndex++) {
      if(NeighborIndex < 0 || NeighborIndex >= TotalBars || NeighborIndex == BarIndex) //--- Skip invalid bars or current bar
         continue;
      if(iHigh(_Symbol, _Period, NeighborIndex) > CurrentBarHigh) //--- If any bar is higher, not a high
         IsSwingHigh = false;
      if(iLow(_Symbol, _Period, NeighborIndex) < CurrentBarLow)   //--- If any bar is lower, not a low
         IsSwingLow = false;
   }
   //--- If it’s a high or low, store it in the SwingPoints array
   if(IsSwingHigh || IsSwingLow) {
      SwingPoint NewSwing;
      NewSwing.TimeOfSwing = iTime(_Symbol, _Period, BarIndex); //--- Store the bar’s time
      NewSwing.PriceAtSwing = IsSwingHigh ? CurrentBarHigh : CurrentBarLow; //--- Store high or low price
      NewSwing.IsSwingHigh = IsSwingHigh;              //--- Mark as high or low
      int CurrentArraySize = ArraySize(SwingPoints);   //--- Get current array size
      ArrayResize(SwingPoints, CurrentArraySize + 1);  //--- Add one more slot
      SwingPoints[CurrentArraySize] = NewSwing;        //--- Add the swing to the array
   }
}
```

Here, we implement the swing point detection logic to identify swing highs and lows for the Cypher pattern. We use a for loop to iterate through bars from "EndBarIndex - 1" to "StartBarIndex" in descending order, with "BarIndex" tracking the current bar. For each bar, we initialize "IsSwingHigh" and "IsSwingLow" as true, assuming the bar is a swing point until disproven, and retrieve the bar’s high and low prices using the [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow) functions, storing them in "CurrentBarHigh" and "CurrentBarLow". A nested for loop checks neighboring bars from "BarIndex - SwingHighCount" to "BarIndex + SwingLowCount", using "NeighborIndex" to skip invalid indices or the current bar itself with a continue statement.

If any neighboring bar’s high exceeds "CurrentBarHigh" or low falls below "CurrentBarLow" (via [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow)), we set "IsSwingHigh" or "IsSwingLow" to false, respectively. If either remains true, we create a "SwingPoint" instance named "NewSwing", assigning "TimeOfSwing" with the bar’s time from [iTime](https://www.mql5.com/en/docs/series/itime), "PriceAtSwing" as "CurrentBarHigh" or "CurrentBarLow" based on "IsSwingHigh", and "IsSwingHigh" accordingly.

We then use the "ArraySize" function to get the current size of "SwingPoints", expand it by one with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize), and store "NewSwing" in the "SwingPoints" array at the new index, building the collection of swing points for pattern analysis. When we print the data using the [ArrayPrint](https://www.mql5.com/en/docs/array/arrayprint)(SwingPoints) function, we have the following outcome.

![STRUCTURED DATA OUTCOME](https://c.mql5.com/2/136/Screenshot_2025-04-20_173223.png)

With the data, we can extract the pivot points and if we have enough pivots, we can analyze and detect the patterns. Here is the logic we implement to achieve that.

```
//--- Check if we have enough swing points (need 5 for Cypher: X, A, B, C, D)
int TotalSwingPoints = ArraySize(SwingPoints);
if(TotalSwingPoints < 5)
   return;  //--- Exit if not enough swing points

//--- Assign the last 5 swing points to X, A, B, C, D (most recent is D)
SwingPoint PointX = SwingPoints[TotalSwingPoints - 5];
SwingPoint PointA = SwingPoints[TotalSwingPoints - 4];
SwingPoint PointB = SwingPoints[TotalSwingPoints - 3];
SwingPoint PointC = SwingPoints[TotalSwingPoints - 2];
SwingPoint PointD = SwingPoints[TotalSwingPoints - 1];

//--- Variables to track if we found a pattern and its type
bool PatternFound = false;
string PatternDirection = "";

//--- Check for Bearish Cypher pattern
if(PointX.IsSwingHigh && !PointA.IsSwingHigh && PointB.IsSwingHigh && !PointC.IsSwingHigh && PointD.IsSwingHigh) {
   double LegXA = PointX.PriceAtSwing - PointA.PriceAtSwing;  //--- Calculate XA leg (should be positive)
   if(LegXA > 0) {
      double LegAB = PointB.PriceAtSwing - PointA.PriceAtSwing; //--- AB leg
      double LegBC = PointB.PriceAtSwing - PointC.PriceAtSwing; //--- BC leg
      double LegXC = PointX.PriceAtSwing - PointC.PriceAtSwing; //--- XC leg
      double LegCD = PointD.PriceAtSwing - PointC.PriceAtSwing; //--- CD leg
      //--- Check Fibonacci rules and D > X for bearish
      if(LegAB >= 0.382 * LegXA && LegAB <= 0.618 * LegXA &&
         LegBC >= 1.272 * LegAB && LegBC <= 1.414 * LegAB &&
         MathAbs(LegCD - 0.786 * LegXC) <= FibonacciTolerance * LegXC && PointD.PriceAtSwing > PointX.PriceAtSwing) {
         PatternFound = true;
         PatternDirection = "Bearish";
      }
   }
}
//--- Check for Bullish Cypher pattern
else if(!PointX.IsSwingHigh && PointA.IsSwingHigh && !PointB.IsSwingHigh && PointC.IsSwingHigh && !PointD.IsSwingHigh) {
   double LegXA = PointA.PriceAtSwing - PointX.PriceAtSwing;  //--- Calculate XA leg (should be positive)
   if(LegXA > 0) {
      double LegAB = PointA.PriceAtSwing - PointB.PriceAtSwing; //--- AB leg
      double LegBC = PointC.PriceAtSwing - PointB.PriceAtSwing; //--- BC leg
      double LegXC = PointC.PriceAtSwing - PointX.PriceAtSwing; //--- XC leg
      double LegCD = PointC.PriceAtSwing - PointD.PriceAtSwing; //--- CD leg
      //--- Check Fibonacci rules and D < X for bullish
      if(LegAB >= 0.382 * LegXA && LegAB <= 0.618 * LegXA &&
         LegBC >= 1.272 * LegAB && LegBC <= 1.414 * LegAB &&
         MathAbs(LegCD - 0.786 * LegXC) <= FibonacciTolerance * LegXC && PointD.PriceAtSwing < PointX.PriceAtSwing) {
         PatternFound = true;
         PatternDirection = "Bullish";
      }
   }
}
```

Here, we continue to validate the [Cypher pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/ "https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/") by checking for sufficient swing points and analyzing the last five for pattern formation. We use the [ArraySize](https://www.mql5.com/en/docs/array/arraysize) function to determine the number of elements in the "SwingPoints" array, storing it in "TotalSwingPoints", and exit with a return statement if "TotalSwingPoints" is less than 5, as the Cypher pattern requires five points (X, A, B, C, D). If enough points exist, we assign the last five swing points to "PointX", "PointA", "PointB", "PointC", and "PointD" from the "SwingPoints" array, with indices from "TotalSwingPoints - 5" to "TotalSwingPoints - 1", where "PointD" is the most recent.

We then initialize "PatternFound" as false to track whether a valid pattern is detected and "PatternDirection" as an empty string to store the pattern type. To check for a bearish Cypher, we verify that "PointX.IsSwingHigh" is true, "PointA.IsSwingHigh" is false, "PointB.IsSwingHigh" is true, "PointC.IsSwingHigh" is false, and "PointD.IsSwingHigh" is true, ensuring the high-low-high-low-high sequence.

If true, we calculate leg lengths: "LegXA" as "PointX.PriceAtSwing" minus "PointA.PriceAtSwing" (positive for bearish), "LegAB" as "PointB.PriceAtSwing" minus "PointA.PriceAtSwing", "LegBC" as "PointB.PriceAtSwing" minus "PointC.PriceAtSwing", "LegXC" as "PointX.PriceAtSwing" minus "PointC.PriceAtSwing", and "LegCD" as "PointD.PriceAtSwing" minus "PointC.PriceAtSwing".

We validate Fibonacci ratios—ensuring "LegAB" is 38.2% to 61.8% of "LegXA", "LegBC" is 127.2% to 141.4% of "LegAB", "LegCD" is within "FibonacciTolerance" of 78.6% of "LegXC" using the "MathAbs" function, and "PointD.PriceAtSwing" exceeds "PointX.PriceAtSwing"—setting "PatternFound" to true and "PatternDirection" to "Bearish" if all conditions are met.

For a bullish Cypher, we check the opposite sequence: "PointX.IsSwingHigh" false, "PointA.IsSwingHigh" true, "PointB.IsSwingHigh" false, "PointC.IsSwingHigh" true, and "PointD.IsSwingHigh" false.

We compute "LegXA" as "PointA.PriceAtSwing" minus "PointX.PriceAtSwing" (positive for bullish), "LegAB" as "PointA.PriceAtSwing" minus "PointB.PriceAtSwing", "LegBC" as "PointC.PriceAtSwing" minus "PointB.PriceAtSwing", "LegXC" as "PointC.PriceAtSwing" minus "PointX.PriceAtSwing", and "LegCD" as "PointC.PriceAtSwing" minus "PointD.PriceAtSwing".

The same Fibonacci checks apply, with "PointD.PriceAtSwing" less than "PointX.PriceAtSwing", updating "PatternFound" to true and "PatternDirection" to "Bullish" if valid, enabling subsequent visualization and trading logic. If the pattern is found, we can proceed to visualize it on the chart.

```
//--- If a pattern is found, visualize it and trade
if(PatternFound) {
   //--- Log the pattern detection in the Experts tab
   Print(PatternDirection, " Cypher pattern detected at ", TimeToString(PointD.TimeOfSwing, TIME_DATE|TIME_MINUTES));

   //--- Create a unique prefix for all chart objects using D’s time
   string ObjectPrefix = "CY_" + IntegerToString(PointD.TimeOfSwing);
   //--- Set triangle color: blue for bullish, red for bearish
   color TriangleColor = (PatternDirection == "Bullish") ? clrBlue : clrRed;

   //--- **Visualization Steps**
   //--- 1. Draw two filled triangles to highlight the pattern
   DrawTriangle(ObjectPrefix + "_Triangle1", PointX.TimeOfSwing, PointX.PriceAtSwing, PointA.TimeOfSwing, PointA.PriceAtSwing, PointB.TimeOfSwing, PointB.PriceAtSwing, TriangleColor, 2, true, true);
   DrawTriangle(ObjectPrefix + "_Triangle2", PointB.TimeOfSwing, PointB.PriceAtSwing, PointC.TimeOfSwing, PointC.PriceAtSwing, PointD.TimeOfSwing, PointD.PriceAtSwing, TriangleColor, 2, true, true);
}
```

We proceed to handle visualization of a detected Cypher pattern when "PatternFound" is true. We use the [Print](https://www.mql5.com/en/docs/common/print) function to log the pattern detection in the Experts tab, outputting "PatternDirection" followed by a message indicating a Cypher pattern was detected, with the time formatted by the [TimeToString](https://www.mql5.com/en/docs/convert/timetostring) function using "PointD.TimeOfSwing" and the "TIME\_DATE\|TIME\_MINUTES" flags for readability.

To organize chart objects, we create a unique prefix "ObjectPrefix" by concatenating "CY\_" with the string representation of "PointD.TimeOfSwing" obtained via the [IntegerToString](https://www.mql5.com/en/docs/convert/IntegerToString) function, ensuring each pattern’s objects are distinctly named. We then set "TriangleColor" using a ternary operator, assigning "clrBlue" for a "Bullish" pattern or "clrRed" for a "Bearish" pattern based on "PatternDirection".

For visualization, we call the "DrawTriangle" function twice: first to draw a triangle named "ObjectPrefix + '\_Triangle1'" connecting "PointX", "PointA", and "PointB" using their "TimeOfSwing" and "PriceAtSwing" values, and second for "ObjectPrefix + '\_Triangle2'" connecting "PointB", "PointC", and "PointD", both with "TriangleColor", a line width of 2, and "true" for filling and drawing behind candles, highlighting the pattern’s structure on the chart. Here is what we have achieved so far.

![MAPPING OF THE CYPHER HARMONIC TRIANGLES](https://c.mql5.com/2/136/Screenshot_2025-04-20_181349.png)

From the image, we can see that we can map and visualize the detected pattern correctly. We now need to continue mapping the trendlines to fully make it visible within boundaries and adding label to it for easier identification of the levels.

```
//--- 2. Draw six trend lines connecting the swing points
DrawTrendLine(ObjectPrefix + "_Line_XA", PointX.TimeOfSwing, PointX.PriceAtSwing, PointA.TimeOfSwing, PointA.PriceAtSwing, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(ObjectPrefix + "_Line_AB", PointA.TimeOfSwing, PointA.PriceAtSwing, PointB.TimeOfSwing, PointB.PriceAtSwing, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(ObjectPrefix + "_Line_BC", PointB.TimeOfSwing, PointB.PriceAtSwing, PointC.TimeOfSwing, PointC.PriceAtSwing, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(ObjectPrefix + "_Line_CD", PointC.TimeOfSwing, PointC.PriceAtSwing, PointD.TimeOfSwing, PointD.PriceAtSwing, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(ObjectPrefix + "_Line_XB", PointX.TimeOfSwing, PointX.PriceAtSwing, PointB.TimeOfSwing, PointB.PriceAtSwing, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(ObjectPrefix + "_Line_BD", PointB.TimeOfSwing, PointB.PriceAtSwing, PointD.TimeOfSwing, PointD.PriceAtSwing, clrBlack, 2, STYLE_SOLID);

//--- 3. Draw labels for each swing point (X, A, B, C, D)
double LabelOffset = 15 * SymbolInfoDouble(_Symbol, SYMBOL_POINT); //--- Offset in points for label placement
DrawTextLabel(ObjectPrefix + "_Label_X", "X", PointX.TimeOfSwing, PointX.PriceAtSwing + (PointX.IsSwingHigh ? LabelOffset : -LabelOffset), clrBlack, 11, PointX.IsSwingHigh);
DrawTextLabel(ObjectPrefix + "_Label_A", "A", PointA.TimeOfSwing, PointA.PriceAtSwing + (PointA.IsSwingHigh ? LabelOffset : -LabelOffset), clrBlack, 11, PointA.IsSwingHigh);
DrawTextLabel(ObjectPrefix + "_Label_B", "B", PointB.TimeOfSwing, PointB.PriceAtSwing + (PointB.IsSwingHigh ? LabelOffset : -LabelOffset), clrBlack, 11, PointB.IsSwingHigh);
DrawTextLabel(ObjectPrefix + "_Label_C", "C", PointC.TimeOfSwing, PointC.PriceAtSwing + (PointC.IsSwingHigh ? LabelOffset : -LabelOffset), clrBlack, 11, PointC.IsSwingHigh);
DrawTextLabel(ObjectPrefix + "_Label_D", "D", PointD.TimeOfSwing, PointD.PriceAtSwing + (PointD.IsSwingHigh ? LabelOffset : -LabelOffset), clrBlack, 11, PointD.IsSwingHigh);

//--- 4. Draw a central label to identify the pattern
datetime CenterTime = (PointX.TimeOfSwing + PointB.TimeOfSwing) / 2;  //--- Middle time between X and B
double CenterPrice = PointD.PriceAtSwing;                            //--- Place it at D’s price level
if(ObjectCreate(0, ObjectPrefix + "_Label_Center", OBJ_TEXT, 0, CenterTime, CenterPrice)) {
   ObjectSetString(0, ObjectPrefix + "_Label_Center", OBJPROP_TEXT, "Cypher"); //--- Label as "Cypher"
   ObjectSetInteger(0, ObjectPrefix + "_Label_Center", OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(0, ObjectPrefix + "_Label_Center", OBJPROP_FONTSIZE, 11);
   ObjectSetString(0, ObjectPrefix + "_Label_Center", OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, ObjectPrefix + "_Label_Center", OBJPROP_ALIGN, ALIGN_CENTER);
}
```

We continue the visualization process to further illustrate the Cypher pattern on the chart. We call the "DrawTrendLine" function six times to draw solid black lines connecting the swing points, each named with "ObjectPrefix" plus a unique suffix (e.g., "\_Line\_XA"). These lines link "PointX" to "PointA", "PointA" to "PointB", "PointB" to "PointC", "PointC" to "PointD", "PointX" to "PointB", and "PointB" to "PointD" using their respective "TimeOfSwing" and "PriceAtSwing" values, with a line width of 2 and [STYLE\_SOLID](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) for clear delineation of the pattern’s structure.

Next, we add text labels for each swing point by calculating "LabelOffset" as 15 times the symbol’s point size, retrieved via the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function with [SYMBOL\_POINT](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), to position labels appropriately. We call the "DrawTextLabel" function five times to label "PointX", "PointA", "PointB", "PointC", and "PointD" with names like "ObjectPrefix + '\_Label\_X'" and text "X", "A", "B", "C", "D". Each label uses the point’s "TimeOfSwing" and "PriceAtSwing" adjusted by "LabelOffset" (added if "IsSwingHigh" is true, subtracted if false), with "clrBlack" color, font size 11, and "IsSwingHigh" determining placement above or below the point.

Finally, we create a central label to identify the pattern by calculating "CenterTime" as the average of "PointX.TimeOfSwing" and "PointB.TimeOfSwing" and setting "CenterPrice" to "PointD.PriceAtSwing". We use the [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) function to create a text object named "ObjectPrefix + '\_Label\_Center'" of type "OBJ\_TEXT" at these coordinates. If successful, we configure it with "ObjectSetString" to set "OBJPROP\_TEXT" to "Cypher" and "OBJPROP\_FONT" to "Arial Bold", and with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to set "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_FONTSIZE" to 11, and "OBJPROP\_ALIGN" to "ALIGN\_CENTER", clearly marking the pattern on the chart. On compilation, we have the following outcome.

![PATTERN WITH EDGES AND LABELS](https://c.mql5.com/2/136/Screenshot_2025-04-20_182800.png)

From the image, we can see that we have added the edges and the labels to the pattern, making it more revealing and illustrative. What we need to do next is determine the trade levels for the pattern.

```
//--- 5. Draw trade levels (entry, take-profits) as dotted lines
datetime LineStartTime = PointD.TimeOfSwing;                    //--- Start at D’s time
datetime LineEndTime = PointD.TimeOfSwing + PeriodSeconds(_Period) * 2; //--- Extend 2 bars to the right
double EntryPrice, StopLossPrice, TakeProfitPrice, TakeProfit1Level, TakeProfit2Level, TakeProfit3Level, TradeDistance;
if(PatternDirection == "Bullish") {
   EntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK); //--- Buy at current ask price
   StopLossPrice = PointX.PriceAtSwing;                //--- Stop-loss at X (below entry for bullish)
   TakeProfitPrice = PointC.PriceAtSwing;              //--- Take-profit at C (target level)
   TakeProfit3Level = PointC.PriceAtSwing;             //--- Highest TP at C
   TradeDistance = TakeProfit3Level - EntryPrice;      //--- Distance to TP3
   TakeProfit1Level = EntryPrice + TradeDistance / 3;  //--- First TP at 1/3 of the distance
   TakeProfit2Level = EntryPrice + 2 * TradeDistance / 3; //--- Second TP at 2/3
} else {
   EntryPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID); //--- Sell at current bid price
   StopLossPrice = PointX.PriceAtSwing;                //--- Stop-loss at X (above entry for bearish)
   TakeProfitPrice = PointC.PriceAtSwing;              //--- Take-profit at C
   TakeProfit3Level = PointC.PriceAtSwing;             //--- Lowest TP at C
   TradeDistance = EntryPrice - TakeProfit3Level;      //--- Distance to TP3
   TakeProfit1Level = EntryPrice - TradeDistance / 3;  //--- First TP at 1/3
   TakeProfit2Level = EntryPrice - 2 * TradeDistance / 3; //--- Second TP at 2/3
}

DrawDottedLine(ObjectPrefix + "_EntryLine", LineStartTime, EntryPrice, LineEndTime, clrMagenta); //--- Entry line
DrawDottedLine(ObjectPrefix + "_TP1Line", LineStartTime, TakeProfit1Level, LineEndTime, clrForestGreen); //--- TP1 line
DrawDottedLine(ObjectPrefix + "_TP2Line", LineStartTime, TakeProfit2Level, LineEndTime, clrGreen);      //--- TP2 line
DrawDottedLine(ObjectPrefix + "_TP3Line", LineStartTime, TakeProfit3Level, LineEndTime, clrDarkGreen);  //--- TP3 line

//--- 6. Draw labels for trade levels
datetime LabelTime = LineEndTime + PeriodSeconds(_Period) / 2; //--- Place labels further right
string EntryLabelText = (PatternDirection == "Bullish") ? "BUY (" : "SELL (";
EntryLabelText += DoubleToString(EntryPrice, _Digits) + ")"; //--- Add price to label
DrawTextLabel(ObjectPrefix + "_EntryLabel", EntryLabelText, LabelTime, EntryPrice, clrMagenta, 11, true);

string TP1LabelText = "TP1 (" + DoubleToString(TakeProfit1Level, _Digits) + ")";
DrawTextLabel(ObjectPrefix + "_TP1Label", TP1LabelText, LabelTime, TakeProfit1Level, clrForestGreen, 11, true);

string TP2LabelText = "TP2 (" + DoubleToString(TakeProfit2Level, _Digits) + ")";
DrawTextLabel(ObjectPrefix + "_TP2Label", TP2LabelText, LabelTime, TakeProfit2Level, clrGreen, 11, true);

string TP3LabelText = "TP3 (" + DoubleToString(TakeProfit3Level, _Digits) + ")";
DrawTextLabel(ObjectPrefix + "_TP3Label", TP3LabelText, LabelTime, TakeProfit3Level, clrDarkGreen, 11, true);
```

Here, we continue to visualize trade levels for the Cypher pattern by drawing dotted lines and labels. We set "LineStartTime" to "PointD.TimeOfSwing" and "LineEndTime" to two bars beyond it using the [PeriodSeconds](https://www.mql5.com/en/docs/common/periodseconds) function multiplied by 2, defining the time range for horizontal lines.

For a "Bullish" pattern (when "PatternDirection" is "Bullish"), we set "EntryPrice" to the current ask price via [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) with [SYMBOL\_ASK](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double), "StopLossPrice" to "PointX.PriceAtSwing", "TakeProfitPrice" and "TakeProfit3Level" to "PointC.PriceAtSwing", calculate "TradeDistance" as "TakeProfit3Level" minus "EntryPrice", and compute "TakeProfit1Level" and "TakeProfit2Level" as one-third and two-thirds of "TradeDistance" added to "EntryPrice".

For a bearish pattern, we use [SYMBOL\_BID](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_info_double) for "EntryPrice", set "StopLossPrice" and "TakeProfit3Level" similarly, calculate "TradeDistance" as "EntryPrice" minus "TakeProfit3Level", and subtract one-third and two-thirds of "TradeDistance" from "EntryPrice" for "TakeProfit1Level" and "TakeProfit2Level".

We then call the "DrawDottedLine" function four times to draw horizontal lines: "ObjectPrefix + '\_EntryLine'" at "EntryPrice" in "clrMagenta", and "ObjectPrefix + '\_TP1Line'", "ObjectPrefix + '\_TP2Line'", "ObjectPrefix + '\_TP3Line'" at "TakeProfit1Level", "TakeProfit2Level", and "TakeProfit3Level" in "clrForestGreen", "clrGreen", and "clrDarkGreen", respectively, from "LineStartTime" to "LineEndTime".

For labeling, we set "LabelTime" to "LineEndTime" plus half a bar’s duration using "PeriodSeconds". We create "EntryLabelText" as "BUY " or ("SELL") based on "PatternDirection", appending "EntryPrice" formatted by "DoubleToString" with "\_Digits", and call "DrawTextLabel" for "ObjectPrefix + '\_EntryLabel'" at "EntryPrice" in "clrMagenta".

Similarly, we define "TP1LabelText", "TP2LabelText", and "TP3LabelText" with "TakeProfit1Level", "TakeProfit2Level", and "TakeProfit3Level" formatted prices, calling "DrawTextLabel" for each at their respective levels in "clrForestGreen", "clrGreen", and "clrDarkGreen", all with font size 11 and placed above the price, enhancing trade level clarity. Here is the outcome.

Bearish pattern:

![BEARISH](https://c.mql5.com/2/136/Screenshot_2025-04-20_184438.png)

Bullish pattern:

![BULLISH](https://c.mql5.com/2/136/Screenshot_2025-04-20_185001.png)

From the images, we can see that we have correctly mapped the trade levels. What we need to do now is initiate the actual trade positions and that is all.

```
//--- **Trading Logic**
//--- Check if trading is allowed and no position is already open
if(TradingEnabled && !PositionSelect(_Symbol)) {
   //--- Place a buy or sell order based on pattern type
   bool TradeSuccessful = (PatternDirection == "Bullish") ?
      obj_Trade.Buy(TradeVolume, _Symbol, EntryPrice, StopLossPrice, TakeProfitPrice, "Cypher Buy") :
      obj_Trade.Sell(TradeVolume, _Symbol, EntryPrice, StopLossPrice, TakeProfitPrice, "Cypher Sell");
   //--- Log the result of the trade attempt
   if(TradeSuccessful)
      Print(PatternDirection, " order opened successfully.");
   else
      Print(PatternDirection, " order failed: ", obj_Trade.ResultRetcodeDescription());
}

//--- Force the chart to update and show all drawn objects
ChartRedraw();
```

Here, we implement the trading logic to execute trades for the Cypher pattern when conditions are met. We check if "TradingEnabled" is true and no existing position is open for the current symbol using the [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect) function with [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), ensuring trades are only placed when allowed and no conflicting positions exist. If both conditions are satisfied, we use a ternary operator to place a trade based on "PatternDirection": for a "Bullish" pattern, we call the "obj\_Trade.Buy" function with parameters "TradeVolume", [\_Symbol](https://www.mql5.com/en/docs/predefined/_symbol), "EntryPrice", "StopLossPrice", "TakeProfitPrice", and a comment "Cypher Buy", while for a bearish pattern, we call "obj\_Trade.Sell" with the same parameters but a comment "Cypher Sell", storing the result in "TradeSuccessful".

We then log the outcome using the [Print](https://www.mql5.com/en/docs/common/print) function, outputting "PatternDirection" and "order opened successfully" if "TradeSuccessful" is true, or "order failed" with the error description from "obj\_Trade.ResultRetcodeDescription" if false. Finally, we call the [ChartRedraw](https://www.mql5.com/en/docs/chart_operations/ChartRedraw) function to force the MetaTrader 5 chart to update, ensuring all drawn objects, such as triangles, lines, and labels, are immediately visible to the user.

Lastly, we just need to delete the patterns from the chart when we remove the program.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
//--- Runs when the EA stops (e.g., removed from chart)
void OnDeinit(const int reason) {
   //--- Remove all objects starting with "CY_" (our Cypher pattern objects)
   ObjectsDeleteAll(0, "CY_");
}
```

Within the [OnDeinit](https://www.mql5.com/en/docs/event_handlers/ondeinit) event handler, we use the [ObjectsDeleteAll](https://www.mql5.com/en/docs/objects/ObjectDeleteAll) function to remove all chart objects with names starting with the prefix "CY\_", ensuring that all Cypher pattern-related visualizations, such as triangles, trend lines, and labels, are cleared from the chart, maintaining a clean workspace when the system is no longer active. Upon compilation, we have the following outcome.

![TRADE CONFIRMATION](https://c.mql5.com/2/136/Screenshot_2025-04-20_191114.png)

From the image, we can see that we plot the Cypher pattern and are still able to trade it accordingly once it is confirmed, hence achieving our objective of identifying, plotting, and trading the pattern. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting and Optimization

During initial backtesting, we identified a critical issue: the system was prone to repainting patterns. Repainting occurred when a Cypher pattern appeared valid on one bar but changed or disappeared as new price data arrived, leading to unreliable trade signals. This issue caused false positives, where trades were executed based on patterns that later proved invalid, negatively impacting performance. Here is an example of what we mean.

![REPAINTED PATTERN](https://c.mql5.com/2/136/Screenshot_2025-04-20_193101.png)

To address this, we implemented a pattern-locking mechanism, using global variables "g\_patternFormationBar" and "g\_lockedPatternX" to lock the pattern on detection and confirm it on the next bar, ensuring the X swing point remains consistent. This fix significantly reduced repainting, as confirmed by subsequent tests showing more stable pattern detection and fewer invalid trades. Here is a sample code snippet to lock the pattern to ensure we wait until the pattern is stable before trading it.

```
//--- If the pattern has changed, update the lock
g_patternFormationBar = CurrentBarIndex;
g_lockedPatternX = PointX.TimeOfSwing;
Print("Cypher pattern has changed; updating lock on bar ", CurrentBarIndex, ". Waiting for confirmation.");
return;
```

We add a confirmation logic to always wait until the pattern is confirmed and stable for an extra bar so that we don't enter the position early only to realize it is the start of the pattern formation. After adding the lock pattern, we can see the issue is now settled.

![ENHANCED PATTERN LOCK](https://c.mql5.com/2/136/Screenshot_2025-04-20_194936.png)

After the correction and thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/136/Screenshot_2025-04-21_182535.png)

Backtest report:

![REPORT](https://c.mql5.com/2/136/Screenshot_2025-04-21_182506.png)

### Conclusion

In conclusion, we have successfully developed a [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") Expert Advisor that detects and trades the [Cypher Harmonic Pattern](https://www.mql5.com/go?link=https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/ "https://howtotrade.com/chart-patterns/cypher-harmonic-pattern/") with precision. By integrating swing point detection, Fibonacci-based validation, comprehensive visualization, and a pattern-locking mechanism to prevent repainting, we created a robust system that dynamically adapts to market conditions.

Disclaimer: This article is for educational purposes only. Trading involves significant financial risk, and market conditions can be unpredictable. While the strategy outlined provides a structured approach to harmonic trading, it does not guarantee profitability. Comprehensive backtesting and proper risk management are essential before deploying this program in a live environment.

By implementing these techniques, you can refine your harmonic pattern trading skills, enhance your technical analysis, and advance your algorithmic trading strategies. Best of luck on your trading journey!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17865.zip "Download all attachments in the single ZIP archive")

[Cypher\_Harmonic\_Pattern\_EA.mq5](https://www.mql5.com/en/articles/download/17865/cypher_harmonic_pattern_ea.mq5 "Download Cypher_Harmonic_Pattern_EA.mq5")(46.26 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/485535)**
(4)


![Kyle Young Sangster](https://c.mql5.com/avatar/2024/11/6736F47E-D362.png)

**[Kyle Young Sangster](https://www.mql5.com/en/users/ksngstr)**
\|
3 May 2025 at 19:30

Hey there, great article. Thank you for your hard work. Could you share the completed code file? Attach it at the end of the article?

Thanks very much

![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
5 May 2025 at 07:04

**Kyle Young Sangster [#](https://www.mql5.com/en/forum/485535#comment_56610937):**

Hey there, great article. Thank you for your hard work. Could you share the completed code file? Attach it at the end of the article?

Thanks very much

Welcome. Did you even check? Thanks.

![Muhammad Syamil Bin Abdullah](https://c.mql5.com/avatar/2025/8/6898ae71-56b0.jpg)

**[Muhammad Syamil Bin Abdullah](https://www.mql5.com/en/users/matfx)**
\|
7 Jun 2025 at 11:08

Thanks for sharing this article. Useful code to implement other harmonics patterns.


![Allan Munene Mutiiria](https://c.mql5.com/avatar/2022/11/637df59b-9551.jpg)

**[Allan Munene Mutiiria](https://www.mql5.com/en/users/29210372)**
\|
7 Jun 2025 at 17:15

**Muhammad Syamil Bin Abdullah [#](https://www.mql5.com/en/forum/485535#comment_56889974):**

Thanks for sharing this article. Useful code to implement other harmonics patterns.

Sure. Welcome.

![Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://c.mql5.com/2/137/websockets.png)[Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)

This article details the development of a custom dynamically linked library designed to facilitate asynchronous websocket client connections for MetaTrader programs.

![Atmosphere Clouds Model Optimization (ACMO): Practice](https://c.mql5.com/2/95/Atmosphere_Clouds_Model_Optimization__LOGO___1.png)[Atmosphere Clouds Model Optimization (ACMO): Practice](https://www.mql5.com/en/articles/15921)

In this article, we will continue diving into the implementation of the ACMO (Atmospheric Cloud Model Optimization) algorithm. In particular, we will discuss two key aspects: the movement of clouds into low-pressure regions and the rain simulation, including the initialization of droplets and their distribution among clouds. We will also look at other methods that play an important role in managing the state of clouds and ensuring their interaction with the environment.

![Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://c.mql5.com/2/137/Building_a_Custom_Market_Regime_Detection_System_in_MQL5_Part_1.png)[Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)

This article details building an adaptive Expert Advisor (MarketRegimeEA) using the regime detector from Part 1. It automatically switches trading strategies and risk parameters for trending, ranging, or volatile markets. Practical optimization, transition handling, and a multi-timeframe indicator are included.

![Neural Networks in Trading: Exploring the Local Structure of Data](https://c.mql5.com/2/94/Neural_Networks_in_Trading__Studying_Local_Data_Structure____LOGO__1.png)[Neural Networks in Trading: Exploring the Local Structure of Data](https://www.mql5.com/en/articles/15882)

Effective identification and preservation of the local structure of market data in noisy conditions is a critical task in trading. The use of the Self-Attention mechanism has shown promising results in processing such data; however, the classical approach does not account for the local characteristics of the underlying structure. In this article, I introduce an algorithm capable of incorporating these structural dependencies.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=jmharrfyimzegcbtjroegsjjvosoufut&ssn=1769093255761876046&ssn_dr=0&ssn_sr=0&fv_date=1769093255&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17865&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2015)%3A%20Price%20Action%20Harmonic%20Cypher%20Pattern%20with%20Visualization%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909325536378202&fz_uniq=5049383309823289990&sv=2552)

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