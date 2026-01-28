---
title: Automating Trading Strategies in MQL5 (Part 33): Creating a Price Action Shark Harmonic Pattern System
url: https://www.mql5.com/en/articles/19479
categories: Trading, Trading Systems, Expert Advisors
relevance_score: 10
scraped_at: 2026-01-22T17:17:44.050732
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/19479&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049025594882106245)

MetaTrader 5 / Trading


### Introduction

In our [previous article (Part 32)](https://www.mql5.com/en/articles/19463), we developed a [5 Drives (5-0) pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/5-0/ "https://harmonictrader.com/harmonic-patterns/5-0/") system in [MetaQuotes Language 5](https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5 "https://www.metaquotes.net/en/metatrader5/algorithmic-trading/mql5") (MQL5) that detected bullish and bearish 5 Drives harmonic patterns using Fibonacci ratios, automating trades with customizable stop loss and take-profit levels, visualized through chart objects like triangles and trendlines. In Part 33, we develop a [Shark Pattern system](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/shark-pattern/ "https://harmonictrader.com/harmonic-patterns/shark-pattern/") that identifies bullish and bearish Shark harmonic patterns using pivot points and specific Fibonacci retracements and extensions. This system executes trades with flexible entry, stop-loss, and multi-level take-profit options, enhanced by visual triangles, trendlines, and labels for clear pattern representation. We will cover the following topics:

1. [Understanding the Shark Harmonic Pattern Framework](https://www.mql5.com/en/articles/19479#para1)
2. [Implementation in MQL5](https://www.mql5.com/en/articles/19479#para2)
3. [Backtesting](https://www.mql5.com/en/articles/19479#para3)
4. [Conclusion](https://www.mql5.com/en/articles/19479#para4)

By the end, you’ll have a robust MQL5 strategy for Shark harmonic pattern trading, ready for customization—let’s dive in!

### Understanding the Shark Harmonic Pattern Framework

The [Shark pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/shark-pattern/ "https://harmonictrader.com/harmonic-patterns/shark-pattern/") is a harmonic trading formation defined by five key swing points—X, A, B, C, and D—existing in bullish and bearish forms, designed to identify high-probability reversal zones through specific [Fibonacci retracements](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement") and extensions. In a bullish Shark pattern, the structure forms a low-high-low-high-low sequence where X is a swing low, A a swing high, B a swing low (retracing 0.32 to 0.50 of XA), C a swing high (extending 1.13 to 1.618 of AB), and D a swing low (extending 1.618 to 2.24 of BC, below B); a bearish Shark reverses this sequence with D above B. Here is a visualization of the patterns:

Bearish Harmonic Shark pattern:

![BEARISH HARMONIC SHARK PATTERN](https://c.mql5.com/2/168/Screenshot_2025-09-08_125815.png)

Bullish Harmonic Shark pattern:

![BULLISH HARMONIC SHARK PATTERN](https://c.mql5.com/2/168/Screenshot_2025-09-08_125842.png)

Our approach involves detecting these swing pivots within a specified bar range, validating the pattern’s legs against user-defined Fibonacci criteria, visualizing the X-A-B-C-D structure with chart objects like triangles and trendlines, and executing trades at the D point with customizable stop loss (Fibonacci-based or fixed) and take-profit levels (one-third, two-thirds, or C pivot) to capitalize on anticipated reversals. Let’s proceed to the implementation!

### Implementation in MQL5

To create the program in MQL5, open the [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), go to the Navigator, locate the Indicators folder, click on the "New" tab, and follow the prompts to create the file. Once it is made, in the coding environment, we will need to declare some [global variables](https://www.mql5.com/en/docs/basis/variables/global) that we will use throughout the program.

```
//+------------------------------------------------------------------+
//|                                             Shark Pattern EA.mq5 |
//|                        Copyright 2025, Forex Algo-Trader, Allan. |
//|                                 "https://t.me/Forex_Algo_Trader" |
//+------------------------------------------------------------------+
#property copyright "Forex Algo-Trader, Allan"
#property link "https://t.me/Forex_Algo_Trader"
#property version "1.00"
#property description "This EA trades based on Shark Strategy"
#property strict

//--- Include the trading library for order functions
#include <Trade\Trade.mqh>    //--- Include Trade library
CTrade obj_Trade;             //--- Instantiate a obj_Trade object
//--- Enumeration for TP levels
enum ENUM_TAKE_PROFIT_LEVEL {
   TP1 = 1, // One-third of the move to C
   TP2 = 2, // Two-thirds of the move to C
   TP3 = 3  // Pivot C Price
};
//--- Enumeration for SL types
enum ENUM_STOP_LOSS_TYPE {
   SL_FIBO = 1, // Fibonacci Extension
   SL_FIXED = 2 // Fixed Points
};
//--- Input parameters for user configuration
input int PivotLeft = 5;                             // Number of bars to the left for pivot check
input int PivotRight = 5;                            // Number of bars to the right for pivot check
input double Tolerance = 0.10;                       // Allowed deviation (10% of XA move)
input double MinRetrace = 0.32;                      // Minimum retracement for AB (strictness control)
input double MaxRetrace = 0.50;                      // Maximum retracement for AB (strictness control)
input double MinExt1 = 1.13;                         // Minimum extension for BC (strictness control)
input double MaxExt1 = 1.618;                        // Maximum extension for BC (strictness control)
input double MinExt2 = 1.618;                        // Minimum extension for CD (strictness control)
input double MaxExt2 = 2.24;                         // Maximum extension for CD (strictness control)
input double LotSize = 0.01;                         // Lot size for new orders
input bool AllowTrading = true;                      // Enable or disable trading
input ENUM_TAKE_PROFIT_LEVEL TakeProfitLevel = TP2;  // Take Profit Level
input ENUM_STOP_LOSS_TYPE StopLossType = SL_FIBO;    // Stop Loss Type
input double SL_FiboExtension = 1.618;               // Fibonacci Extension for SL
input double SL_FixedPoints = 50;                    // Fixed Points for SL (in points)
//---------------------------------------------------------------------------

//--- Structure for a pivot point
struct Pivot {
   datetime time; //--- Bar time of the pivot
   double price;  //--- Pivot price (High for swing high, low for swing low)
   bool isHigh;   //--- True if swing high; false if swing low
};
//--- Global dynamic array for storing pivots in chronological order
Pivot pivots[]; //--- Declare a dynamic array to hold identified pivot points
//--- Global variables to lock in a pattern (avoid trading on repaint)
int g_patternFormationBar = -1; //--- Bar index where the pattern was formed (-1 means none)
datetime g_lockedPatternX = 0; //--- The key X pivot time for the locked pattern
//--- Global array to track traded patterns (using X.time as identifier)
datetime tradedPatterns[];
```

To establish the foundation for the [Shark Pattern system](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/shark-pattern/ "https://harmonictrader.com/harmonic-patterns/shark-pattern/"), we first include the "<Trade\\Trade.mqh>" library and instantiate "obj\_Trade" as a [CTrade](https://www.mql5.com/en/docs/standardlibrary/tradeclasses/ctrade) object to manage trade operations, such as executing buy and sell orders. Then, we define [enumerations](https://www.mql5.com/en/docs/basis/types/integer/enumeration) "ENUM\_TAKE\_PROFIT\_LEVEL" (TP1 for one-third, TP2 for two-thirds, TP3 for pivot C price) and "ENUM\_STOP\_LOSS\_TYPE" (SL\_FIBO for Fibonacci extension, SL\_FIXED for fixed points) for flexible trade settings, and set [input parameters](https://www.mql5.com/en/docs/basis/variables/inputvariables): "PivotLeft" and "PivotRight" at 5 bars for pivot detection, "Tolerance" at 0.10 for Fibonacci deviation, "MinRetrace" at 0.32 and "MaxRetrace" at 0.50 for AB leg, "MinExt1" at 1.13 and "MaxExt1" at 1.618 for BC leg, "MinExt2" at 1.618 and "MaxExt2" at 2.24 for CD leg, "LotSize" at 0.01, "AllowTrading" as true, "TakeProfitLevel" as TP2, "StopLossType" as SL\_FIBO, "SL\_FiboExtension" at 1.618, and "SL\_FixedPoints" at 50.

Next, we define the "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes) with "time", "price", and "isHigh" to store swing points, declare "pivots" as a dynamic array, and initialize globals "g\_patternFormationBar" to -1 for tracking pattern formation, "g\_lockedPatternX" to 0 for locking the X pivot time, and "tradedPatterns" as an array to track traded patterns using X’s time. This setup provides the core framework for detecting and trading Shark patterns. For visualization, we can have functions to draw lines, labels, and triangles.

```
//+------------------------------------------------------------------+
//| Helper: Draw a filled triangle                                   |
//+------------------------------------------------------------------+
void DrawTriangle(string name, datetime t1, double p1, datetime t2, double p2, datetime t3, double p3, color cl, int width, bool fill, bool back) {
   //--- Attempt to create a triangle object with three coordinate points
   if(ObjectCreate(0, name, OBJ_TRIANGLE, 0, t1, p1, t2, p2, t3, p3)) {
      //--- Set the triangle's color
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl);
      //--- Set the triangle's line style to solid
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
      //--- Set the line width of the triangle
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
      //--- Determine if the triangle should be filled
      ObjectSetInteger(0, name, OBJPROP_FILL, fill);
      //--- Set whether the object is drawn in the background
      ObjectSetInteger(0, name, OBJPROP_BACK, back);
   }
}
//+------------------------------------------------------------------+
//| Helper: Draw a trend line                                        |
//+------------------------------------------------------------------+
void DrawTrendLine(string name, datetime t1, double p1, datetime t2, double p2, color cl, int width, int style) {
   //--- Create a trend line object connecting two points
   if(ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2)) {
      //--- Set the trend line's color
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl);
      //--- Set the trend line's style (solid, dotted, etc.)
      ObjectSetInteger(0, name, OBJPROP_STYLE, style);
      //--- Set the width of the trend line
      ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   }
}
//+------------------------------------------------------------------+
//| Helper: Draw a dotted trend line                                 |
//+------------------------------------------------------------------+
void DrawDottedLine(string name, datetime t1, double p, datetime t2, color lineColor) {
   //--- Create a horizontal trend line at a fixed price level with dotted style
   if(ObjectCreate(0, name, OBJ_TREND, 0, t1, p, t2, p)) {
      //--- Set the dotted line's color
      ObjectSetInteger(0, name, OBJPROP_COLOR, lineColor);
      //--- Set the line style to dotted
      ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DOT);
      //--- Set the line width to 1
      ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   }
}
//+------------------------------------------------------------------+
//| Helper: Draw anchored text label (for pivots)                    |
//| If isHigh is true, anchor at the bottom (label appears above);   |
//| if false, anchor at the top (label appears below).               |
//+------------------------------------------------------------------+
void DrawTextEx(string name, string text, datetime t, double p, color cl, int fontsize, bool isHigh) {
   //--- Create a text label object at the specified time and price
   if(ObjectCreate(0, name, OBJ_TEXT, 0, t, p)) {
      //--- Set the text of the label
      ObjectSetString(0, name, OBJPROP_TEXT, text);
      //--- Set the color of the text
      ObjectSetInteger(0, name, OBJPROP_COLOR, cl);
      //--- Set the font size for the text
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontsize);
      //--- Set the font type and style
      ObjectSetString(0, name, OBJPROP_FONT, "Arial Bold");
      //--- Anchor the text depending on whether it's a swing high or low
      if(isHigh)
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_BOTTOM);
      else
         ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_TOP);
      //--- Center-align the text
      ObjectSetInteger(0, name, OBJPROP_ALIGN, ALIGN_CENTER);
   }
}
```

We proceed to implement visualization functions to create clear chart representations of the Shark harmonic pattern and its trade levels. First, we develop the "DrawTriangle" function, which uses [ObjectCreate](https://www.mql5.com/en/docs/objects/objectcreate) to draw a filled triangle ( [OBJ\_TRIANGLE](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_triangle)) defined by three points with times ("t1", "t2", "t3") and prices ("p1", "p2", "p3"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to the specified color, "OBJPROP\_STYLE" to "STYLE\_SOLID", "OBJPROP\_WIDTH" to the given width, "OBJPROP\_FILL" to enable or disable filling, and "OBJPROP\_BACK" to set background or foreground placement using the [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) function. Then, we proceed to create the "DrawTrendLine" function, which draws a trend line ("OBJ\_TREND") between two points.

Next, we implement the "DrawDottedLine" function, which creates a horizontal dotted line ( [OBJ\_TREND](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_trend)) at a specified price from "t1" to "t2". Last, we develop the "DrawTextEx" function, which creates a text label ( [OBJ\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_text)) at coordinates ("t", "p") with "ObjectCreate", setting "OBJPROP\_TEXT" to the specified text, "OBJPROP\_COLOR", "OBJPROP\_FONTSIZE", and "OBJPROP\_FONT" to "Arial Bold" using [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger", anchoring above for swing highs or below for lows based on "isHigh" with "OBJPROP\_ANCHOR", and centering with "OBJPROP\_ALIGN". We can now proceed to the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler and attempt to identify pivot points that we can use later for pattern recognition. Here is the logic we use to achieve that.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   //--- Declare a static variable to store the time of the last processed bar
   static datetime lastBarTime = 0;
   //--- Get the time of the current confirmed bar
   datetime currentBarTime = iTime(_Symbol, _Period, 1);
   //--- If the current bar time is the same as the last processed, exit
   if(currentBarTime == lastBarTime)
      return;
   //--- Update the last processed bar time
   lastBarTime = currentBarTime;

   //--- Clear the pivot array for fresh analysis
   ArrayResize(pivots, 0);
   //--- Get the total number of bars available on the chart
   int barsCount = Bars(_Symbol, _Period);
   //--- Define the starting index for pivot detection (ensuring enough left bars)
   int start = PivotLeft;
   //--- Define the ending index for pivot detection (ensuring enough right bars)
   int end = barsCount - PivotRight;

   //--- Loop through bars from 'end-1' down to 'start' to find pivot points
   for(int i = end - 1; i >= start; i--) {
      //--- Assume current bar is both a potential swing high and swing low
      bool isPivotHigh = true;
      bool isPivotLow = true;
      //--- Get the high and low of the current bar
      double currentHigh = iHigh(_Symbol, _Period, i);
      double currentLow = iLow(_Symbol, _Period, i);
      //--- Loop through the window of bars around the current bar
      for(int j = i - PivotLeft; j <= i + PivotRight; j++) {
         //--- Skip if the index is out of bounds
         if(j < 0 || j >= barsCount)
            continue;
         //--- Skip comparing the bar with itself
         if(j == i)
            continue;
         //--- If any bar in the window has a higher high, it's not a swing high
         if(iHigh(_Symbol, _Period, j) > currentHigh)
            isPivotHigh = false;
         //--- If any bar in the window has a lower low, it's not a swing low
         if(iLow(_Symbol, _Period, j) < currentLow)
            isPivotLow = false;
      }
      //--- If the current bar qualifies as either a swing high or swing low
      if(isPivotHigh || isPivotLow) {
         //--- Create a new pivot structure
         Pivot p;
         //--- Set the pivot's time
         p.time = iTime(_Symbol, _Period, i);
         //--- Set the pivot's price depending on whether it is a high or low
         p.price = isPivotHigh ? currentHigh : currentLow;
         //--- Set the pivot type (true for swing high, false for swing low)
         p.isHigh = isPivotHigh;
         //--- Get the current size of the pivots array
         int size = ArraySize(pivots);
         //--- Increase the size of the pivots array by one
         ArrayResize(pivots, size + 1);
         //--- Add the new pivot to the array
         pivots[size] = p;
      }
   }
}
```

Here, we implement the initial logic of the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) function to detect swing pivots, which are essential for identifying the Shark harmonic pattern. First, we declare a static "lastBarTime" initialized to 0 to track the last processed bar and compare it with "currentBarTime" obtained from [iTime](https://www.mql5.com/en/docs/series/itime) at shift 1 for the current symbol and period, exiting if unchanged to avoid redundant processing, and updating "lastBarTime" when a new bar is detected. Then, we proceed to clear the "pivots" array with [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) to ensure a fresh analysis. Next, we retrieve the total number of bars with [Bars](https://www.mql5.com/en/docs/series/bars), set the pivot detection range with "start" as "PivotLeft" and "end" as total bars minus "PivotRight", and iterate through bars selected bars.

For each bar, we assume it’s a swing high ("isPivotHigh" true) and low ("isPivotLow" true), obtain its high and low prices using [iHigh](https://www.mql5.com/en/docs/series/ihigh) and [iLow](https://www.mql5.com/en/docs/series/ilow), and validate the pivot by checking surrounding bars within "PivotLeft" and "PivotRight" with "iHigh" and "iLow", invalidating the pivot if any neighboring bar has a higher high or lower low. Last, if the bar qualifies as a pivot, we create a "Pivot" [structure](https://www.mql5.com/en/docs/basis/types/classes), set its "time" with "iTime", "price" to the high or low based on "isPivotHigh", and "isHigh" flag, then append it to the "pivots" array using [ArrayResize](https://www.mql5.com/en/docs/array/arrayresize) and store it. We get the following array of data when we print the pivot structure.

![ANALYSED PIVOTS DATA](https://c.mql5.com/2/168/Screenshot_2025-09-06_105517.png)

With the data, we can extract the pivot points, and if we have enough pivots, we can analyze and detect the patterns. Here is the logic we implement to achieve that.

```
//--- Determine the total number of pivots found
int pivotCount = ArraySize(pivots);
//--- If fewer than five pivots are found, the pattern cannot be formed
if(pivotCount < 5) {
   //--- Reset pattern lock variables
   g_patternFormationBar = -1;
   g_lockedPatternX = 0;
   //--- Exit the OnTick function
   return;
}

//--- Extract the last five pivots as X, A, B, C, and D
Pivot X = pivots[pivotCount - 5];
Pivot A = pivots[pivotCount - 4];
Pivot B = pivots[pivotCount - 3];
Pivot C = pivots[pivotCount - 2];
Pivot D = pivots[pivotCount - 1];

//--- Initialize a flag to indicate if a valid Shark pattern is found
bool patternFound = false;
//--- Initialize pattern type
string patternType = "";
//--- Check for the low-high-low-high-low (Bullish reversal) structure
if((!X.isHigh) && A.isHigh && (!B.isHigh) && C.isHigh && (!D.isHigh)) {
   //--- Calculate the difference between pivot A and X
   double diff = A.price - X.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the retracement from A to B
      double retrace = A.price - B.price;
      //--- Verify retracement is within user-defined range
      if((retrace >= MinRetrace * diff) && (retrace <= MaxRetrace * diff)) {
         //--- Calculate the extension from B to C
         double extension1 = C.price - B.price;
         //--- Verify extension is within user-defined range of retrace
         if((extension1 >= MinExt1 * retrace) && (extension1 <= MaxExt1 * retrace)) {
            //--- Calculate the extension from C to D
            double extension2 = C.price - D.price;
            //--- Verify extension is within user-defined range of previous extension
            if((extension2 >= MinExt2 * extension1) && (extension2 <= MaxExt2 * extension1) && (D.price < B.price)) {
               patternFound = true;
               patternType = "Bullish";
            }
         }
      }
   }
}
//--- Check for the high-low-high-low-high (Bearish reversal) structure
if(X.isHigh && (!A.isHigh) && B.isHigh && (!C.isHigh) && D.isHigh) {
   //--- Calculate the difference between pivot X and A
   double diff = X.price - A.price;
   //--- Ensure the difference is positive
   if(diff > 0) {
      //--- Calculate the retracement from A to B
      double retrace = B.price - A.price;
      //--- Verify retracement is within user-defined range
      if((retrace >= MinRetrace * diff) && (retrace <= MaxRetrace * diff)) {
         //--- Calculate the extension from B to C
         double extension1 = B.price - C.price;
         //--- Verify extension is within user-defined range of retrace
         if((extension1 >= MinExt1 * retrace) && (extension1 <= MaxExt1 * retrace)) {
            //--- Calculate the extension from C to D
            double extension2 = D.price - C.price;
            //--- Verify extension is within user-defined range of previous extension
            if((extension2 >= MinExt2 * extension1) && (extension2 <= MaxExt2 * extension1) && (D.price > B.price)) {
               patternFound = true;
               patternType = "Bearish";
            }
         }
      }
   }
}
```

First, we determine the total number of pivots with " [ArraySize(pivots)](https://www.mql5.com/en/docs/array/arraysize)" stored in "pivotCount" and exit if fewer than 5 pivots are found, resetting "g\_patternFormationBar" and "g\_lockedPatternX" to -1 and 0, as the Shark pattern requires X, A, B, C, and D points. Then, we proceed to extract the last five pivots from the "pivots" array, assigning "X" (earliest), "A", "B", "C", and "D" (latest).

Next, for a bullish pattern (X low, A high, B low, C high, D low), we calculate the XA leg difference ("A.price - X.price"), ensure it’s positive, verify the AB retracement ("A.price - B.price") is within "MinRetrace" (0.32) to "MaxRetrace" (0.50) of XA, check the BC extension ("C.price - B.price") is within "MinExt1" (1.13) to "MaxExt1" (1.618) of AB, and confirm the CD extension ("C.price - D.price") is within "MinExt2" (1.618) to "MaxExt2" (2.24) of BC with "D.price < B.price", setting "patternFound" to true and "patternType" to "Bullish" if valid. Last, for a bearish pattern (X high, A low, B high, C low, D high), we apply similar validations for XA ("X.price - A.price"), AB retracement ("B.price - A.price"), BC extension ("B.price - C.price"), and CD extension ("D.price - C.price") with "D.price > B.price", setting "patternFound" to true and "patternType" to "Bearish" if valid. If the pattern is found, we can proceed to visualize it on the chart.

```
//--- If a valid Shark pattern is detected
if(patternFound) {
   //--- Print a message indicating the pattern type and detection time
   Print(patternType, " Shark pattern detected at ", TimeToString(D.time, TIME_DATE|TIME_MINUTES|TIME_SECONDS));

   //--- Create a unique prefix for all graphical objects related to this pattern
   string signalPrefix = "SH_" + IntegerToString(X.time);

   //--- Choose triangle color based on the pattern type
   color triangleColor = (patternType=="Bullish") ? clrBlue : clrRed;

   //--- Draw the first triangle connecting pivots X, A, and B
   DrawTriangle(signalPrefix+"_Triangle1", X.time, X.price, A.time, A.price, B.time, B.price,
                triangleColor, 2, true, true);
   //--- Draw the second triangle connecting pivots B, C, and D
   DrawTriangle(signalPrefix+"_Triangle2", B.time, B.price, C.time, C.price, D.time, D.price,
                triangleColor, 2, true, true);
}
```

Here, we initiate visualization of detected patterns on the chart. First, if a valid pattern is detected ("patternFound" is true), we log the detection with [Print](https://www.mql5.com/en/docs/common/print), outputting the "patternType" ("Bullish" or "Bearish") and the D pivot’s time formatted with [TimeToString](https://www.mql5.com/en/docs/convert/timetostring), including date, minutes, and seconds. Then, we proceed to create a unique identifier "signalPrefix" by concatenating "SH\_" with "X.time" converted to a string using [IntegerToString](https://www.mql5.com/en/docs/convert/integertostring) to ensure distinct naming for chart objects.

Next, we set "triangleColor" to blue for bullish patterns or red for bearish patterns to differentiate them visually. Last, we call "DrawTriangle" twice to visualize the pattern: first to draw the XAB triangle connecting pivots X, A, and B, and then to draw the BCD triangle connecting pivots B, C, and D, using "signalPrefix" with suffixes "\_Triangle1" and "\_Triangle2", respective pivot times and prices, "triangleColor", a width of 2, and enabling fill and background display with true flags. We get the following outcome.

![TRIANGLES SET](https://c.mql5.com/2/168/Screenshot_2025-09-08_154236.png)

From the image, we can see that we can map and visualize the detected pattern correctly. We now need to continue mapping the trendlines to fully make it visible within boundaries and add a label to it for easier identification of the levels.

```
//--- Draw boundary trend lines connecting the pivots for clarity
DrawTrendLine(signalPrefix+"_TL_XA", X.time, X.price, A.time, A.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_AB", A.time, A.price, B.time, B.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_BC", B.time, B.price, C.time, C.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_CD", C.time, C.price, D.time, D.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_XB", X.time, X.price, B.time, B.price, clrBlack, 2, STYLE_SOLID);
DrawTrendLine(signalPrefix+"_TL_BD", B.time, B.price, D.time, D.price, clrBlack, 2, STYLE_SOLID);

//--- Retrieve the symbol's point size to calculate offsets for text positioning
double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
//--- Calculate an offset (15 points) for positioning text above or below pivots
double offset = 15 * point;

//--- Determine the Y coordinate for each pivot label based on its type
double textY_X = (X.isHigh ? X.price + offset : X.price - offset);
double textY_A = (A.isHigh ? A.price + offset : A.price - offset);
double textY_B = (B.isHigh ? B.price + offset : B.price - offset);
double textY_C = (C.isHigh ? C.price + offset : C.price - offset);
double textY_D = (D.isHigh ? D.price + offset : D.price - offset);

//--- Draw text labels for each pivot with appropriate anchoring
DrawTextEx(signalPrefix+"_Text_X", "X", X.time, textY_X, clrBlack, 11, X.isHigh);
DrawTextEx(signalPrefix+"_Text_A", "A", A.time, textY_A, clrBlack, 11, A.isHigh);
DrawTextEx(signalPrefix+"_Text_B", "B", B.time, textY_B, clrBlack, 11, B.isHigh);
DrawTextEx(signalPrefix+"_Text_C", "C", C.time, textY_C, clrBlack, 11, C.isHigh);
DrawTextEx(signalPrefix+"_Text_D", "D", D.time, textY_D, clrBlack, 11, D.isHigh);

//--- Calculate the central label's time as the midpoint between pivots X and B
datetime centralTime = (X.time + B.time) / 2;
//--- Set the central label's price at pivot D's price
double centralPrice = D.price;
//--- Create the central text label indicating the pattern type
if(ObjectCreate(0, signalPrefix+"_Text_Center", OBJ_TEXT, 0, centralTime, centralPrice)) {
   ObjectSetString(0, signalPrefix+"_Text_Center", OBJPROP_TEXT,
      (patternType=="Bullish") ? "Bullish Shark" : "Bearish Shark");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_COLOR, clrBlack);
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_FONTSIZE, 11);
   ObjectSetString(0, signalPrefix+"_Text_Center", OBJPROP_FONT, "Arial Bold");
   ObjectSetInteger(0, signalPrefix+"_Text_Center", OBJPROP_ALIGN, ALIGN_CENTER);
}
```

We further enhance the visualization of detected patterns by adding detailed chart objects to clearly depict the pattern structure. First, we draw six solid trend lines using "DrawTrendLine" with the unique "signalPrefix" to connect key pivot points: XA, AB, BC, CD, XB, and BD, using pivot times and prices (e.g., "X.time", "X.price"), setting [OBJPROP\_COLOR](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_integer) to "clrBlack", "OBJPROP\_WIDTH" to 2, and "OBJPROP\_STYLE" to "STYLE\_SOLID" with [ObjectSetInteger](https://www.mql5.com/en/docs/objects/objectsetinteger) to outline the pattern’s legs. Then, we retrieve the symbol’s point size with " [SymbolInfoDouble(\_Symbol, SYMBOL\_POINT)](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble)" and calculate a 15-point offset for label positioning, determining Y-coordinates ("textY\_X", "textY\_A", "textY\_B", "textY\_C", "textY\_D") by adding or subtracting the offset based on whether each pivot is a swing high ("isHigh" true) or low to place labels above highs or below lows.

Next, we use "DrawTextEx" to create text labels for pivots X, A, B, C, and D with "signalPrefix" and suffixes like "\_Text\_X", displaying the respective letter, positioned at the pivot time and adjusted Y-coordinate, using "clrBlack", font size 11, and the pivot’s "isHigh" status for anchoring. Last, we calculate the central label’s position at "centralTime" as the midpoint of "X.time" and "B.time" and "centralPrice" at "D.price", creating a text object with "ObjectCreate" named "signalPrefix + '\_Text\_Center'", setting [OBJPROP\_TEXT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Bullish Shark" or "Bearish Shark" based on "patternType", and configuring "OBJPROP\_COLOR" to "clrBlack", "OBJPROP\_FONTSIZE" to 11, [OBJPROP\_FONT](https://www.mql5.com/en/docs/constants/objectconstants/enum_object_property#enum_object_property_string) to "Arial Bold", and "OBJPROP\_ALIGN" to "ALIGN\_CENTER" with the [ObjectSetString](https://www.mql5.com/en/docs/objects/objectsetstring) and "ObjectSetInteger" functions. When we run the program, here is a visualization of the output we receive.

![PATTERN WITH LABELS AND EDGES](https://c.mql5.com/2/168/Screenshot_2025-09-08_154422.png)

From the image, we can see that we have added the edges and the labels to the pattern, making it more revealing and illustrative. What we need to do next is determine the trade levels for the pattern.

```
//--- Define start and end times for drawing horizontal dotted lines for trade levels
datetime lineStart = D.time;
datetime lineEnd = D.time + PeriodSeconds(_Period)*2;

//--- Declare variables for entry price and take profit levels
double entryPriceLevel, TP1Level, TP2Level, TP3Level, tradeDiff;
//--- Calculate trade levels based on whether the pattern is Bullish or Bearish
if(patternType=="Bullish") { //--- Bullish → BUY signal
   //--- Use the current ASK price as the entry
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   //--- Set TP3 at pivot C's price
   TP3Level = C.price;
   //--- Calculate the total distance to be covered by the trade
   tradeDiff = TP3Level - entryPriceLevel;
   //--- Set TP1 at one-third of the total move
   TP1Level = entryPriceLevel + tradeDiff/3;
   //--- Set TP2 at two-thirds of the total move
   TP2Level = entryPriceLevel + 2*tradeDiff/3;
} else { //--- Bearish → SELL signal
   //--- Use the current BID price as the entry
   entryPriceLevel = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   //--- Set TP3 at pivot C's price
   TP3Level = C.price;
   //--- Calculate the total distance to be covered by the trade
   tradeDiff = entryPriceLevel - TP3Level;
   //--- Set TP1 at one-third of the total move
   TP1Level = entryPriceLevel - tradeDiff/3;
   //--- Set TP2 at two-thirds of the total move
   TP2Level = entryPriceLevel - 2*tradeDiff/3;
}

//--- Draw dotted horizontal lines to represent the entry and TP levels
DrawDottedLine(signalPrefix+"_EntryLine", lineStart, entryPriceLevel, lineEnd, clrMagenta);
DrawDottedLine(signalPrefix+"_TP1Line", lineStart, TP1Level, lineEnd, clrForestGreen);
DrawDottedLine(signalPrefix+"_TP2Line", lineStart, TP2Level, lineEnd, clrGreen);
DrawDottedLine(signalPrefix+"_TP3Line", lineStart, TP3Level, lineEnd, clrDarkGreen);

//--- Define a label time coordinate positioned just to the right of the dotted lines
datetime labelTime = lineEnd + PeriodSeconds(_Period)/2;

//--- Construct the entry label text with the price
string entryLabel = (patternType=="Bullish") ? "BUY (" : "SELL (";
entryLabel += DoubleToString(entryPriceLevel, _Digits) + ")";
//--- Draw the entry label on the chart
DrawTextEx(signalPrefix+"_EntryLabel", entryLabel, labelTime, entryPriceLevel, clrMagenta, 11, true);

//--- Construct and draw the TP1 label
string tp1Label = "TP1 (" + DoubleToString(TP1Level, _Digits) + ")";
DrawTextEx(signalPrefix+"_TP1Label", tp1Label, labelTime, TP1Level, clrForestGreen, 11, true);

//--- Construct and draw the TP2 label
string tp2Label = "TP2 (" + DoubleToString(TP2Level, _Digits) + ")";
DrawTextEx(signalPrefix+"_TP2Label", tp2Label, labelTime, TP2Level, clrGreen, 11, true);

//--- Construct and draw the TP3 label
string tp3Label = "TP3 (" + DoubleToString(TP3Level, _Digits) + ")";
DrawTextEx(signalPrefix+"_TP3Label", tp3Label, labelTime, TP3Level, clrDarkGreen, 11, true);
```

To define and visualize the trade levels for the detected pattern, we set "lineStart" to the D pivot’s time ("D.time") and "lineEnd" to two periods ahead using " [PeriodSeconds(\_Period)](https://www.mql5.com/en/docs/common/periodseconds) \\* 2", and declare variables "entryPriceLevel", "TP1Level", "TP2Level", "TP3Level", and "tradeDiff" for trade calculations. Then, for a bullish pattern ("patternType == 'Bullish'"), we set "entryPriceLevel" to the current ask price with the [SymbolInfoDouble](https://www.mql5.com/en/docs/marketinformation/symbolinfodouble) function, "TP3Level" to the C pivot’s price, calculate "tradeDiff" as "TP3Level - entryPriceLevel", and compute "TP1Level" and "TP2Level" as one-third and two-thirds of "tradeDiff" added to "entryPriceLevel"; for a bearish pattern, we use the bid price, set "TP3Level" to C’s price, calculate "tradeDiff" as "entryPriceLevel - TP3Level", and compute "TP1Level" and "TP2Level" by subtracting one-third and two-thirds of the trade difference.

Next, we draw four dotted horizontal lines using "DrawDottedLine": an entry line at "entryPriceLevel" in magenta, and take-profit lines at "TP1Level" (forest green), "TP2Level" (green), and "TP3Level" (dark green), spanning from "lineStart" to "lineEnd". Last, we set "labelTime" to "lineEnd" plus half a period, create label texts with prices formatted via the [DoubleToString](https://www.mql5.com/en/docs/convert/doubletostring) function (e.g., "BUY (price)" or "SELL (price)" for entry, "TP1 (price)", etc.), and use "DrawTextEx" to draw these labels at "labelTime" with corresponding colors, font size 11, and anchored above the price levels. Upon compilation, we have the following outcome.

Bearish pattern:

![BEARISH PATTERN](https://c.mql5.com/2/168/Screenshot_2025-09-08_155211.png)

Bullish pattern:

![BULLISH PATTERN](https://c.mql5.com/2/168/Screenshot_2025-09-08_155535.png)

From the images, we can see that we have correctly mapped the trade levels. What we need to do now is initiate the actual trade positions, and that is all.

```
//--- Retrieve the index of the current bar
int currentBarIndex = Bars(_Symbol, _Period) - 1;
//--- If no pattern has been previously locked, lock the current pattern formation
if(g_patternFormationBar == -1) {
   g_patternFormationBar = currentBarIndex;
   g_lockedPatternX = X.time;
   //--- Print a message that the pattern is detected and waiting for confirmation
   Print("Pattern detected on bar ", currentBarIndex, ". Waiting for confirmation on next bar.");
   return;
}
//--- If still on the same formation bar, the pattern is considered to be repainting
if(currentBarIndex == g_patternFormationBar) {
   Print("Pattern is repainting; still on locked formation bar ", currentBarIndex, ". No trade yet.");
   return;
}
//--- If we are on a new bar compared to the locked formation
if(currentBarIndex > g_patternFormationBar) {
   //--- Check if the locked pattern still corresponds to the same X pivot
   if(g_lockedPatternX == X.time) {
      Print("Confirmed pattern (locked on bar ", g_patternFormationBar, "). Opening trade on bar ", currentBarIndex, ".");
      //--- Update the pattern formation bar to the current bar
      g_patternFormationBar = currentBarIndex;
      //--- Only proceed with trading if allowed and if there is no existing position
      if(AllowTrading && !PositionSelect(_Symbol)) {
         //--- Check if this pattern has already been traded
         bool alreadyTraded = false;
         for(int k = 0; k < ArraySize(tradedPatterns); k++) {
            if(tradedPatterns[k] == X.time) {
               alreadyTraded = true;
               break;
            }
         }
         if(alreadyTraded) {
            Print("This pattern has already been traded. No new trade executed.");
            return;
         }
         double entryPriceTrade = 0, stopLoss = 0, takeProfit = 0;
         point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         bool tradeResult = false;
         //--- Select TP level based on user input
         switch(TakeProfitLevel) {
            case TP1:
               takeProfit = TP1Level;
               break;
            case TP2:
               takeProfit = TP2Level;
               break;
            case TP3:
               takeProfit = TP3Level;
               break;
            default:
               takeProfit = TP2Level; // Fallback to TP2
         }
         //--- Calculate SL based on user-selected method
         if(patternType=="Bullish") { //--- BUY signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            if(StopLossType == SL_FIBO) {
               double second_drive = C.price - D.price;
               stopLoss = D.price - (SL_FiboExtension - 1.0) * second_drive;
            } else { // SL_FIXED
               stopLoss = entryPriceTrade - SL_FixedPoints * point;
            }
            // Ensure SL is below entry for BUY
            if(stopLoss >= entryPriceTrade) {
               stopLoss = entryPriceTrade - 10 * point;
            }
            tradeResult = obj_Trade.Buy(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Shark Signal");
            if(tradeResult)
               Print("Buy order opened successfully.");
            else
               Print("Buy order failed: ", obj_Trade.ResultRetcodeDescription());
         }
         //--- For a Bearish pattern, execute a SELL trade
         else if(patternType=="Bearish") { //--- SELL signal
            entryPriceTrade = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            if(StopLossType == SL_FIBO) {
               double second_drive = D.price - C.price;
               stopLoss = D.price + (SL_FiboExtension - 1.0) * second_drive;
            } else { // SL_FIXED
               stopLoss = entryPriceTrade + SL_FixedPoints * point;
            }
            // Ensure SL is above entry for SELL
            if(stopLoss <= entryPriceTrade) {
               stopLoss = entryPriceTrade + 10 * point;
            }
            tradeResult = obj_Trade.Sell(LotSize, _Symbol, entryPriceTrade, stopLoss, takeProfit, "Shark Signal");
            if(tradeResult)
               Print("Sell order opened successfully.");
            else
               Print("Sell order failed: ", obj_Trade.ResultRetcodeDescription());
         }
         //--- If trade was successful, mark the pattern as traded
         if(tradeResult) {
            int size = ArraySize(tradedPatterns);
            ArrayResize(tradedPatterns, size + 1);
            tradedPatterns[size] = X.time;
         }
      }
      else {
         //--- If a position is already open, do not execute a new trade
         Print("A position is already open for ", _Symbol, ". No new trade executed.");
      }
   }
   else {
      //--- If the pattern has changed, update the lock with the new formation bar and X pivot
      g_patternFormationBar = currentBarIndex;
      g_lockedPatternX = X.time;
      Print("Pattern has changed; updating lock on bar ", currentBarIndex, ". Waiting for confirmation.");
      return;
   }

}
else {
   //--- If no valid Shark pattern is detected, reset the pattern lock variables
   g_patternFormationBar = -1;
   g_lockedPatternX = 0;
}
```

Here, we finalize the [OnTick](https://www.mql5.com/en/docs/event_handlers/ontick) event handler implementation by managing trade execution and pattern confirmation for the detected pattern. First, we retrieve the current bar index with " [Bars(\_Symbol, \_Period)](https://www.mql5.com/en/docs/series/bars) \- 1" and store it in "currentBarIndex". Then, if no pattern is locked ("g\_patternFormationBar == -1"), we set "g\_patternFormationBar" to "currentBarIndex", lock the X pivot time in "g\_lockedPatternX" with "X.time", log the detection with [Print](https://www.mql5.com/en/docs/common/print) indicating a wait for confirmation, and exit.

Next, if still on the formation bar ("currentBarIndex == g\_patternFormationBar"), we log repainting and exit to prevent premature trading. Last, if a new bar has formed ("currentBarIndex > g\_patternFormationBar") and the X pivot matches "g\_lockedPatternX", we confirm the pattern, update "g\_patternFormationBar", and check if trading is permitted and no open positions exist via the [PositionSelect](https://www.mql5.com/en/docs/trading/positionselect) function; we verify the pattern hasn’t been traded by checking "tradedPatterns", select the take-profit level ("TP1Level", "TP2Level", or "TP3Level") based on your selection, calculate stop loss again based on user selection, ensure stop loss is valid (below entry for buy, above for sell, adjusted by 10 points if needed), execute a buy or sell with "obj\_Trade.Buy" or "obj\_Trade.Sell" using "LotSize" and "Shark Signal", log success or failure, and mark the pattern as traded in "tradedPatterns"; if trading is disallowed, a position exists, or the pattern was traded, we log no trade; if the pattern changes, we update the lock and wait; if no pattern is found, we reset the global variables. Upon compilation, we have the following outcome.

Bearish signal:

![CONFIRMED BEARISH SIGNAL](https://c.mql5.com/2/168/Screenshot_2025-09-08_155743.png)

Bullish signal:

![CONFIRMED BULLISH SIGNAL](https://c.mql5.com/2/168/Screenshot_2025-09-08_160339.png)

From the image, we can see that we plot the harmonic pattern and are still able to trade it accordingly once it is confirmed, hence achieving our objective of identifying, plotting, and trading the pattern. The thing that remains is backtesting the program, and that is handled in the next section.

### Backtesting

After thorough backtesting, we have the following results.

Backtest graph:

![GRAPH](https://c.mql5.com/2/168/Screenshot_2025-09-08_134206.png)

Backtest report:

![REPORT](https://c.mql5.com/2/168/Screenshot_2025-09-08_134237.png)

### Conclusion

In conclusion, we’ve developed a [Shark Pattern](https://www.mql5.com/go?link=https://harmonictrader.com/harmonic-patterns/shark-pattern/ "https://harmonictrader.com/harmonic-patterns/shark-pattern/") system in MQL5, leveraging price action to detect bullish and bearish Shark harmonic patterns with [Fibonacci ratios](https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement "https://www.metatrader5.com/en/terminal/help/objects/fibo/fibo_retracement"), automating trades with customizable entry, stop loss, and multi-level take-profit points, and visualizing patterns with chart objects like [triangles](https://www.mql5.com/en/docs/constants/objectconstants/enum_object/obj_triangle) and trendlines.

Disclaimer: This article is for educational purposes only. Trading carries significant financial risks, and market volatility may result in losses. Thorough backtesting and careful risk management are crucial before deploying this program in live markets.

By leveraging the concepts and implementation presented, you can adapt this Shark pattern system to your trading style, enhancing your algorithmic strategies. Happy trading!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/19479.zip "Download all attachments in the single ZIP archive")

[Shark\_Pattern\_EA.mq5](https://www.mql5.com/en/articles/download/19479/Shark_Pattern_EA.mq5 "Download Shark_Pattern_EA.mq5")(53.71 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/496159)**

![How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://c.mql5.com/2/171/19547-how-to-build-and-optimize-a-logo.png)[How to build and optimize a cycle-based trading system (Detrended Price Oscillator - DPO)](https://www.mql5.com/en/articles/19547)

This article explains how to design and optimise a trading system using the Detrended Price Oscillator (DPO) in MQL5. It outlines the indicator's core logic, demonstrating how it identifies short-term cycles by filtering out long-term trends. Through a series of step-by-step examples and simple strategies, readers will learn how to code it, define entry and exit signals, and conduct backtesting. Finally, the article presents practical optimization methods to enhance performance and adapt the system to changing market conditions.

![The MQL5 Standard Library Explorer (Part 1): Introduction with CTrade, CiMA, and CiATR](https://c.mql5.com/2/171/19341-the-mql5-standard-library-explorer-logo.png)[The MQL5 Standard Library Explorer (Part 1): Introduction with CTrade, CiMA, and CiATR](https://www.mql5.com/en/articles/19341)

The MQL5 Standard Library plays a vital role in developing trading algorithms for MetaTrader 5. In this discussion series, our goal is to master its application to simplify the creation of efficient trading tools for MetaTrader 5. These tools include custom Expert Advisors, indicators, and other utilities. We begin today by developing a trend-following Expert Advisor using the CTrade, CiMA, and CiATR classes. This is an especially important topic for everyone—whether you are a beginner or an experienced developer. Join this discussion to discover more.

![MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://c.mql5.com/2/172/19627-mql5-wizard-techniques-you-logo.png)[MQL5 Wizard Techniques you should know (Part 80): Using Patterns of Ichimoku and the ADX-Wilder with TD3 Reinforcement Learning](https://www.mql5.com/en/articles/19627)

This article follows up ‘Part-74’, where we examined the pairing of Ichimoku and the ADX under a Supervised Learning framework, by moving our focus to Reinforcement Learning. Ichimoku and ADX form a complementary combination of support/resistance mapping and trend strength spotting. In this installment, we indulge in how the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm can be used with this indicator set. As with earlier parts of the series, the implementation is carried out in a custom signal class designed for integration with the MQL5 Wizard, which facilitates seamless Expert Advisor assembly.

![Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://c.mql5.com/2/170/19436-perehodim-na-mql5-algo-forge-logo.png)[Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)

Let's explore how you can start integrating external code from any repository in the MQL5 Algo Forge storage into your own project. In this article, we finally turn to this promising, yet more complex, task: how to practically connect and use libraries from third-party repositories within MQL5 Algo Forge.

[![](https://www.mql5.com/ff/si/dwquj7nmuxsb297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F994%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.use.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=enhudadyvnrfwcvutcjazdvrxjyrzhyf&s=8f8a773cbff7e7ca26346dfb885f4f329a8b1f2c99472f858f32c0b06b662998&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pnuvklrscoirbaplnybjwojrhbdpgjtp&ssn=1769091461553852521&ssn_dr=1&ssn_sr=0&fv_date=1769091461&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19479&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Automating%20Trading%20Strategies%20in%20MQL5%20(Part%2033)%3A%20Creating%20a%20Price%20Action%20Shark%20Harmonic%20Pattern%20System%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909146203892818&fz_uniq=5049025594882106245&sv=2552)

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